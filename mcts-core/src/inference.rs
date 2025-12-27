use std::sync::atomic::{AtomicU64, Ordering};
use anyhow::{Result, anyhow};

use crate::comms::{RequestScratchPad, ResponseScratchPad};
use crate::enums::{GRID_MAX, MAX_BATCH, MAX_OBJ0, MAX_OBJ1, NUM_ACTIONS};
use crate::ipc_core::{now_ns, Arena, Slot, SLOT_DONE, SLOT_READY};

// ---------------------------------------------------------------------------------------------
// Generic inference boundary trait
// ---------------------------------------------------------------------------------------------
/// Pure inference boundary implementing different inference agents (local, IPC, python, etc).
pub trait InferenceClient {
    fn infer_into(
        &self,
        to_agent: &RequestScratchPad,
        b: usize,
        from_agent: &mut ResponseScratchPad,
    ) -> Result<()>;
}

// ---------------------------------------------------------------------------------------------
// An IPC capable inference client communicating via shared memory.
// ---------------------------------------------------------------------------------------------
/// Producer-side inference client that communicates with inference workers via shared memory.
pub struct IpcClient {
    arena: Arena,
    owner_id: u32,
    next_req_id: AtomicU64,
    print_timing: bool,
}

impl IpcClient {
    pub fn new(arena: Arena, owner_id: u32) -> Self {
        Self {
            arena,
            owner_id,
            next_req_id: AtomicU64::new(0),
            print_timing: true,
        }
    }

    pub fn set_handler_start_time(&self, slot_idx: u32) {
        let sm = self.arena.slot_mut(slot_idx);
        sm.slot.handler_start_time_ns.store(now_ns(), Ordering::Release);
    }

    #[inline]
    fn copy_req_into_slot(req: &RequestScratchPad, b: usize, slot: &mut Slot) -> Result<()> {
        if b == 0 || b > MAX_BATCH {
            return Err(anyhow!("batch size {} out of range (1..={})", b, MAX_BATCH));
        }
        if b > req.max_batch {
            return Err(anyhow!(
                "batch size {} exceeds RequestScratchPad.max_batch={}",
                b,
                req.max_batch
            ));
        }

        // Per-item arrays (length max_batch; copy prefix b)
        slot.h[..b].copy_from_slice(&req.h[..b]);
        slot.w[..b].copy_from_slice(&req.w[..b]);
        slot.obj0_len[..b].copy_from_slice(&req.obj0_len[..b]);
        slot.obj1_len[..b].copy_from_slice(&req.obj1_len[..b]);

        // Flat packed arrays
        let p_n = b * GRID_MAX;
        let o0_n = b * MAX_OBJ0;
        let o1_n = b * MAX_OBJ1;
        let m_n = b * NUM_ACTIONS;

        // Note: TokenId is u16 in your RequestScratchPad; Slot uses u16 too.
        slot.placement[..p_n].copy_from_slice(&req.placement[..p_n]);
        slot.obj0[..o0_n].copy_from_slice(&req.obj0[..o0_n]);
        slot.obj1[..o1_n].copy_from_slice(&req.obj1[..o1_n]);
        slot.action_mask[..m_n].copy_from_slice(&req.action_mask[..m_n]);

        Ok(())
    }

    #[inline]
    fn copy_slot_into_resp(slot: &Slot, b: usize, resp: &mut ResponseScratchPad) -> Result<()> {
        if b == 0 || b > MAX_BATCH {
            return Err(anyhow!("batch size {} out of range (1..={})", b, MAX_BATCH));
        }
        if b > resp.priors.len() || b > resp.values.len() {
            return Err(anyhow!(
                "ResponseScratchPad too small: b={}, priors.len()={}, values.len()={}",
                b,
                resp.priors.len(),
                resp.values.len()
            ));
        }

        // Use your helper to get the output slices.
        let (priors_out, values_out) = resp.out_mut(b);

        // Slot priors are flat [MAX_BATCH * NUM_ACTIONS].
        for i in 0..b {
            let start = i * NUM_ACTIONS;
            let end = start + NUM_ACTIONS;
            priors_out[i].copy_from_slice(&slot.priors[start..end]);
            values_out[i] = slot.values[i];
        }

        Ok(())
    }
}

impl InferenceClient for IpcClient {
    fn infer_into(
        &self,
        req: &RequestScratchPad,
        b: usize,
        resp: &mut ResponseScratchPad,
    ) -> Result<()> {
        let slot_idx = self.arena.acquire_slot();
        let req_id = self.next_req_id.fetch_add(1, Ordering::Relaxed);

        // Write request
        {
            let sm = self.arena.slot_mut(slot_idx);
            sm.slot.request_time_ns.store(now_ns(), Ordering::Release);
            sm.slot.b = b as u32;
            sm.slot.owner_id = self.owner_id;
            sm.slot.req_id = req_id;
            Self::copy_req_into_slot(req, b, sm.slot)?;
            // Publish READY after all writes.
            sm.slot.state.store(SLOT_READY, Ordering::Release);
        }

        // Submit + wait
        self.arena.submit_to_handler(slot_idx);
        self.arena.wait_done(slot_idx);

        // Read response
        {
            let sr = self.arena.slot(slot_idx);

            // Optional (but nice) sanity check.
            if sr.slot.owner_id != self.owner_id || sr.slot.req_id != req_id {
                self.arena.release_slot(slot_idx);
                return Err(anyhow!(
                    "mismatched response: expected owner_id={} req_id={}, got owner_id={} req_id={}",
                    self.owner_id,
                    req_id,
                    sr.slot.owner_id,
                    sr.slot.req_id
                ));
            }

            debug_assert_eq!(sr.slot.state.load(Ordering::Acquire), SLOT_DONE);

            Self::copy_slot_into_resp(sr.slot, b, resp)?;
        }

        // Record response time
        {
            let sm = self.arena.slot_mut(slot_idx);
            sm.slot.response_time_ns.store(now_ns(), Ordering::Release);
        }

        // Optional: print timing info for instrumentation.
        if self.print_timing {
            let sr = self.arena.slot(slot_idx);
            let handler_start = sr.slot.handler_start_time_ns.load(Ordering::Acquire);
            let request_time = sr.slot.request_time_ns.load(Ordering::Acquire);
            let response_time = sr.slot.response_time_ns.load(Ordering::Acquire);
            println!(
                "request time: {}    response time: {}",
                (handler_start - request_time) as f64 / 1e6,
                (response_time - handler_start) as f64 / 1e6,
            );
        };
        self.arena.release_slot(slot_idx);
        Ok(())
    }
}
