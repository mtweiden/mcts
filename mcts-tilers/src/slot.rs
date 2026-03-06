use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::{anyhow, Result};
use mcts_core::ipc_core::{SlotInit, SLOT_FREE};

use tilers::qubit::Qubit;
use tilers::objective::Objective;
use tilers::enums::{Direction, Operation, Orientation, QubitId};

use crate::constants::*;
use crate::environment::TilersObs;

// ─────────────────────────────────────────────────────────────────────────────
// Slot states
// ─────────────────────────────────────────────────────────────────────────────
pub use mcts_core::ipc_core::{SLOT_DONE, SLOT_READY, SLOT_WAITING};

// ─────────────────────────────────────────────────────────────────────────────
// Tilers-specific Slot
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub struct TilersSlot {
    // Header
    pub state: AtomicU32,
    pub b: u32,
    pub owner_id: u32,
    pub req_id: u64,

    // Inputs
    pub h: [u8; MAX_BATCH],
    pub w: [u8; MAX_BATCH],
    pub num_ancillas: [u8; MAX_BATCH],
    pub num_qubits: [u16; MAX_BATCH],
    pub num_layers: [u8; MAX_BATCH],
    pub num_objectives: [[u16; LOOKAHEAD_MAX]; MAX_BATCH],

    pub placement: [[u8; PLACEMENT_MAX]; MAX_BATCH],
    pub objectives: [[u8; OBJECTIVES_MAX]; MAX_BATCH],

    pub action_mask: [u8; MAX_BATCH * NUM_ACTIONS],

    // Outputs
    pub priors: [f32; MAX_BATCH * NUM_ACTIONS],
    pub values: [f32; MAX_BATCH],

    // Timing
    pub request_time_ns: AtomicU64,
    pub handler_start_time_ns: AtomicU64,
    pub response_time_ns: AtomicU64,
}

impl SlotInit for TilersSlot {
    fn init_free(&self) {
        self.state.store(SLOT_FREE, Ordering::Relaxed);
    }

    fn state(&self) -> &AtomicU32 {
        &self.state
    }
}

impl TilersSlot {
    pub fn pack_observations(&mut self, observations: &[TilersObs]) -> Result<()> {
        let b = observations.len();
        if b == 0 || b > MAX_BATCH {
            return Err(anyhow!("batch size {} out of range (1..={})", b, MAX_BATCH));
        }
        self.b = b as u32;

        for (i, obs) in observations.iter().enumerate() {
            self.h[i] = obs.height as u8;
            self.w[i] = obs.width as u8;
            self.num_ancillas[i] = obs.num_ancillas as u8;

            // Pack placement
            let nq = obs.placement.len().min(GRID_MAX);
            self.num_qubits[i] = nq as u16;
            self.placement[i].fill(0);
            for (j, qubit) in obs.placement.iter().take(nq).enumerate() {
                let offset = j * QUBIT_SIZE;
                let id_bytes = qubit.id.as_i32().to_le_bytes();
                self.placement[i][offset..offset + 4].copy_from_slice(&id_bytes);
                self.placement[i][offset + 4] = qubit.orientation as u8;
            }

            // Pack objectives
            let nl = obs.objectives.len().min(LOOKAHEAD_MAX);
            self.num_layers[i] = nl as u8;
            self.objectives[i].fill(0);
            for (l, layer) in obs.objectives.iter().take(nl).enumerate() {
                let no = layer.len().min(GRID_MAX);
                self.num_objectives[i][l] = no as u16;
                let layer_offset = l * OBJECTIVES_LAYER_MAX;
                for (k, obj) in layer.iter().take(no).enumerate() {
                    let offset = layer_offset + k * OBJECTIVE_SIZE;
                    self.objectives[i][offset] = obj.opcode as u8;
                    let a0 = obj.arg_0.as_i32().to_le_bytes();
                    self.objectives[i][offset + 1..offset + 5].copy_from_slice(&a0);
                    let a1 = obj.arg_1.as_i32().to_le_bytes();
                    self.objectives[i][offset + 5..offset + 9].copy_from_slice(&a1);
                    self.objectives[i][offset + 9] = obj.duration as u8;
                    self.objectives[i][offset + 10] = obj.direction as u8;
                }
            }

            // Pack action mask
            let mask_offset = i * NUM_ACTIONS;
            let mask_slice = &mut self.action_mask[mask_offset..mask_offset + NUM_ACTIONS];
            mask_slice.fill(0);
            for (a, &m) in obs.action_mask.iter().enumerate() {
                if a < NUM_ACTIONS && m {
                    mask_slice[a] = 1;
                }
            }
        }

        Ok(())
    }

    pub fn unpack_observations(&self) -> Vec<TilersObs> {
        let b = self.b as usize;
        let mut out = Vec::with_capacity(b);

        for i in 0..b {
            let height = self.h[i] as usize;
            let width = self.w[i] as usize;
            let num_ancillas = self.num_ancillas[i] as usize;
            let nq = self.num_qubits[i] as usize;
            let nl = (self.num_layers[i] as usize).min(LOOKAHEAD_MAX);

            // Unpack placement
            let mut placement = Vec::with_capacity(nq);
            for j in 0..nq {
                let offset = j * QUBIT_SIZE;
                let id = i32::from_le_bytes([
                    self.placement[i][offset],
                    self.placement[i][offset + 1],
                    self.placement[i][offset + 2],
                    self.placement[i][offset + 3],
                ]);
                let orientation = self.placement[i][offset + 4];
                placement.push(Qubit {
                    id: QubitId(id),
                    orientation: Orientation::from_u8(orientation).unwrap(),
                });
            }

            // Unpack objectives
            let mut objectives = Vec::with_capacity(nl);
            for l in 0..nl {
                let no = self.num_objectives[i][l] as usize;
                let layer_offset = l * OBJECTIVES_LAYER_MAX;
                let mut layer = Vec::with_capacity(no);
                for k in 0..no {
                    let offset = layer_offset + k * OBJECTIVE_SIZE;
                    let opcode = self.objectives[i][offset];
                    let arg_0 = i32::from_le_bytes([
                        self.objectives[i][offset + 1],
                        self.objectives[i][offset + 2],
                        self.objectives[i][offset + 3],
                        self.objectives[i][offset + 4],
                    ]);
                    let arg_1 = i32::from_le_bytes([
                        self.objectives[i][offset + 5],
                        self.objectives[i][offset + 6],
                        self.objectives[i][offset + 7],
                        self.objectives[i][offset + 8],
                    ]);
                    let duration = self.objectives[i][offset + 9] as usize;
                    let direction = self.objectives[i][offset + 10];
                    layer.push(Objective {
                        opcode: Operation::from_u8(opcode).unwrap(),
                        arg_0: QubitId(arg_0),
                        // WARNING: ancilla list is not serialized through IPC. This means we
                        // cannot send concrete objectives through IPC.
                        ancilla: Vec::new(),
                        arg_1: QubitId(arg_1),
                        duration,
                        direction: Direction::from_u8(direction).unwrap(),
                    });
                }
                objectives.push(layer);
            }

            // Unpack action mask
            let mask_offset = i * NUM_ACTIONS;
            let action_mask: Vec<bool> = self.action_mask[mask_offset..mask_offset + NUM_ACTIONS]
                .iter()
                .map(|&m| m != 0)
                .collect();

            out.push(TilersObs {
                placement,
                objectives,
                height,
                width,
                num_ancillas,
                action_mask,
            });
        }

        out
    }
}