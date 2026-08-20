use std::sync::atomic::Ordering;

use anyhow::{anyhow, Result};

use mcts_core::ipc_core::{now_ns, Arena, IpcClient, SLOT_DONE, SLOT_READY};
use mcts_core::inference::InferenceClient;

use crate::constants::*;
use crate::environment::{TilersEnv, TilersObs};
use crate::slot::TilersSlot;

//--------------------------------------------------------------------------------
/// For neural network inference.
pub struct TilersIpcClient {
    inner: IpcClient<TilersSlot>,
}

impl TilersIpcClient {
    pub fn new(arena: Arena<TilersSlot>, owner_id: u32) -> Self {
        Self {
            inner: IpcClient::new(arena, owner_id),
        }
    }

    pub fn with_timing(mut self, enabled: bool) -> Self {
        self.inner.print_timing = enabled;
        self
    }
}

impl InferenceClient<TilersEnv> for TilersIpcClient {
    fn infer(&self, observations: &[TilersObs]) -> Result<(Vec<Prior>, Vec<Value>)> {
        let b = observations.len();
        if b == 0 || b > MAX_BATCH {
            return Err(anyhow!("batch size {} out of range (1..={})", b, MAX_BATCH));
        }

        let arena = &self.inner.arena;
        let slot_idx = arena.acquire_slot();
        // eprintln!("[rust client {}] acquired slot {}", self.inner.owner_id, slot_idx);
        let req_id = self.inner.next_req_id.fetch_add(1, Ordering::Relaxed);
        // Write request
        {
            let sm = arena.slot_mut(slot_idx);
            sm.slot.request_time_ns.store(now_ns(), Ordering::Release);
            sm.slot.b = b as u32;
            sm.slot.owner_id = self.inner.owner_id;
            sm.slot.req_id = req_id;
            // Release the slot before propagating: `pack_observations` gained a
            // real error path (layer count past LOOKAHEAD_MAX), and bailing with
            // `?` here would strand an acquired slot — never returned to
            // free_q, never marked READY. Leak one per inference and the pool
            // drains until `acquire_slot` blocks forever, deadlocking the
            // gather with no error message.
            if let Err(e) = sm.slot.pack_observations(observations) {
                drop(sm);
                arena.release_slot(slot_idx);
                return Err(e);
            }
            sm.slot.state.store(SLOT_READY, Ordering::Release);
        }
        // eprintln!("[rust client {}] slot {} marked READY, submitting to handler...", self.inner.owner_id, slot_idx);

        // Submit and wait
        arena.submit_to_handler(slot_idx);
        // eprintln!("[rust client {}] slot {} submitted, waiting for DONE...", self.inner.owner_id, slot_idx);
        arena.wait_done(slot_idx);
        // eprintln!("[rust client {}] slot {} DONE", self.inner.owner_id, slot_idx);

        // Read response
        let result = {
            let sr = arena.slot(slot_idx);

            if sr.slot.owner_id != self.inner.owner_id || sr.slot.req_id != req_id {
                arena.release_slot(slot_idx);
                return Err(anyhow!(
                    "mismatched response: expected owner_id={} req_id={}, got owner_id={} req_id={}",
                    self.inner.owner_id,
                    req_id,
                    sr.slot.owner_id,
                    sr.slot.req_id
                ));
            }

            debug_assert_eq!(sr.slot.state.load(Ordering::Acquire), SLOT_DONE);

            // Read flat priors and convert to HashMap<Action, f32>
            let priors: Vec<Prior> = (0..b)
                .map(|i| {
                    let offset = i * NUM_ACTIONS;
                    observations[i]
                        .action_mask
                        .iter()
                        .enumerate()
                        .filter(|(_, valid)| **valid)
                        .map(|(a, _)| (a as Action, sr.slot.priors[offset + a]))
                        .collect()
                })
                .collect();

            let values: Vec<Value> = sr.slot.values[..b].to_vec();

            (priors, values)
        };

        // Timing
        if self.inner.print_timing {
            let sr = arena.slot(slot_idx);
            let response_time = now_ns();
            let request_time = sr.slot.request_time_ns.load(Ordering::Acquire);
            let handler_start = sr.slot.handler_start_time_ns.load(Ordering::Acquire);
            println!(
                "request time: {:.3}ms    response time: {:.3}ms",
                (handler_start.wrapping_sub(request_time)) as f64 / 1e6,
                (response_time.wrapping_sub(handler_start)) as f64 / 1e6,
            );
        }

        arena.release_slot(slot_idx);
        Ok(result)
    }
}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
/// For fast non-neural uniform priors.
pub struct TrivialTilersIpcClient { }


impl InferenceClient<TilersEnv> for TrivialTilersIpcClient {
    fn infer(&self, observations: &[TilersObs]) -> Result<(Vec<Prior>, Vec<Value>)> {
        let priors = observations
            .iter()
            .map(|obs| {
                obs.action_mask
                    .iter()
                    .enumerate()
                    .filter(|(_, valid)| **valid)
                    .map(|(a, _)| (a as Action, 1.0))
                    .collect()
            })
            .collect();
        let values = vec![0.0; observations.len()];
        Ok((priors, values))
    }
}
//--------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_client_claim_release() {
        use mcts_core::ipc_core::Arena;
        use mcts_core::ipc_core::SLOT_FREE;

        let arena: Arena<TilersSlot> = Arena::create_or_open("test_claim_release", 1, 1).unwrap();
        let slot = arena.slot(0);

        // Slot should start free
        assert_eq!(slot.slot.state.load(Ordering::Relaxed), SLOT_FREE);

        // Simulate claiming
        slot.slot.state.store(SLOT_READY, Ordering::Release);
        assert_eq!(slot.slot.state.load(Ordering::Relaxed), SLOT_READY);

        // Simulate handler completing
        slot.slot.state.store(SLOT_DONE, Ordering::Release);
        assert_eq!(slot.slot.state.load(Ordering::Relaxed), SLOT_DONE);

        // Release back to free
        slot.slot.state.store(SLOT_FREE, Ordering::Release);
        assert_eq!(slot.slot.state.load(Ordering::Relaxed), SLOT_FREE);
    }
}
