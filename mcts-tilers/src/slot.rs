use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::{anyhow, Result};
use mcts_core::ipc_core::{SlotInit, SLOT_FREE};

use tilers::qubit::Qubit;
use tilers::objective::Objective;
use tilers::enums::{Operation, Orientation, QubitId};

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

impl Default for TilersSlot {
    fn default() -> Self {
        Self {
            state: AtomicU32::new(SLOT_FREE),
            b: 0,
            owner_id: 0,
            req_id: 0,
            h: [0; MAX_BATCH],
            w: [0; MAX_BATCH],
            num_ancillas: [0; MAX_BATCH],
            num_qubits: [0; MAX_BATCH],
            num_layers: [0; MAX_BATCH],
            num_objectives: [[0; LOOKAHEAD_MAX]; MAX_BATCH],
            placement: [[0; PLACEMENT_MAX]; MAX_BATCH],
            objectives: [[0; OBJECTIVES_MAX]; MAX_BATCH],
            action_mask: [0; MAX_BATCH * NUM_ACTIONS],
            priors: [0.0; MAX_BATCH * NUM_ACTIONS],
            values: [0.0; MAX_BATCH],
            request_time_ns: AtomicU64::new(0),
            handler_start_time_ns: AtomicU64::new(0),
            response_time_ns: AtomicU64::new(0),
        }
    }
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
                    layer.push(
                        Objective::new(
                            Operation::from_u8(opcode).unwrap(),
                            QubitId(arg_0),
                            vec![],
                            QubitId(arg_1),
                        )
                    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::TilersObs;
    use crate::environment::TilersEnv;
    use mcts_core::Environment;
    use tilers::qubit::{Qubit};
    use tilers::objective::Objective;
    use tilers::enums::{Orientation, QubitId};
    use tilers::env::Environment as TilersEnvInner;

    #[test]
    fn test_slot_roundtrip() {
        let obs = TilersObs {
            placement: vec![
                Qubit { id: QubitId(0), orientation: Orientation::Vertical },
                Qubit { id: QubitId(1), orientation: Orientation::Horizontal },
            ],
            objectives: vec![vec![Objective::cx(QubitId(0), QubitId(1))]],
            height: 3,
            width: 4,
            num_ancillas: 1,
            action_mask: vec![true, false, true, false],
        };

        let mut slot = TilersSlot::default();
        slot.pack_observations(&[obs]).unwrap();

        // Read back and verify fields match
        assert_eq!(slot.b, 1);
        assert_eq!(slot.h[0], 3);
        assert_eq!(slot.w[0], 4);
        assert_eq!(slot.num_ancillas[0], 1);
        assert_eq!(slot.num_qubits[0], 2);
        assert_eq!(slot.num_layers[0], 1);
        assert_eq!(slot.num_objectives[0][0], 1);

        // Test full roundtrip through unpack
        let unpacked = slot.unpack_observations();
        assert_eq!(unpacked.len(), 1);
        let u = &unpacked[0];
        assert_eq!(u.height, 3);
        assert_eq!(u.width, 4);
        assert_eq!(u.num_ancillas, 1);
        assert_eq!(u.placement.len(), 2);
        assert_eq!(u.placement[0].id, QubitId(0));
        assert_eq!(u.placement[1].id, QubitId(1));
        assert_eq!(u.objectives.len(), 1);
        assert_eq!(u.objectives[0].len(), 1);
        assert_eq!(u.objectives[0][0].opcode, Operation::CX);
        assert_eq!(u.objectives[0][0].arg_0, QubitId(0));
        assert_eq!(u.objectives[0][0].arg_1, QubitId(1));
        assert!(u.action_mask[0]);
        assert!(!u.action_mask[1]);
        assert!(u.action_mask[2]);
        assert!(!u.action_mask[3]);
    }

    #[test]
    fn test_slot_roundtrip_larger_grid() {
        let mut placement = Vec::new();
        for i in 0..25 {
            placement.push(Qubit {
                id: QubitId(i),
                orientation: if i % 2 == 0 { Orientation::Vertical } else { Orientation::Horizontal },
            });
        }
        // Add some ancillas with negative IDs
        placement.push(Qubit { id: QubitId(-1), orientation: Orientation::Vertical });
        placement.push(Qubit { id: QubitId(-2), orientation: Orientation::Horizontal });

        let obs = TilersObs {
            placement,
            objectives: vec![],
            height: 5,
            width: 5,
            num_ancillas: 2,
            action_mask: vec![false; NUM_ACTIONS],
        };

        let mut slot = TilersSlot::default();
        slot.pack_observations(&[obs]).unwrap();
        let unpacked = slot.unpack_observations();
        let u = &unpacked[0];

        assert_eq!(u.placement.len(), 27);
        assert_eq!(u.placement[25].id, QubitId(-1));
        assert_eq!(u.placement[26].id, QubitId(-2));
        assert_eq!(u.placement[0].orientation, Orientation::Vertical);
        assert_eq!(u.placement[1].orientation, Orientation::Horizontal);
    }

    #[test]
    fn test_slot_roundtrip_multi_layer() {
        let obs = TilersObs {
            placement: vec![
                Qubit { id: QubitId(0), orientation: Orientation::Vertical },
                Qubit { id: QubitId(1), orientation: Orientation::Horizontal },
                Qubit { id: QubitId(2), orientation: Orientation::Vertical },
            ],
            objectives: vec![
                vec![
                    Objective::cx(QubitId(0), QubitId(1)),
                ],
                vec![
                    Objective::cz(QubitId(1), QubitId(2)),
                    Objective::x(QubitId(0)),
                ],
            ],
            height: 3,
            width: 3,
            num_ancillas: 0,
            action_mask: vec![true; NUM_ACTIONS],
        };

        let mut slot = TilersSlot::default();
        slot.pack_observations(&[obs]).unwrap();
        let unpacked = slot.unpack_observations();
        let u = &unpacked[0];

        assert_eq!(u.objectives.len(), 2);
        assert_eq!(u.objectives[0].len(), 1);
        assert_eq!(u.objectives[0][0].opcode, Operation::CX);
        assert_eq!(u.objectives[1].len(), 2);
        assert_eq!(u.objectives[1][0].opcode, Operation::CZ);
        assert_eq!(u.objectives[1][1].opcode, Operation::X);
    }

    #[test]
    fn test_slot_roundtrip_batch() {
        let obs1 = TilersObs {
            placement: vec![Qubit { id: QubitId(0), orientation: Orientation::Vertical }],
            objectives: vec![vec![
                Objective::cx(QubitId(0), QubitId(1)),
            ]],
            height: 3,
            width: 3,
            num_ancillas: 0,
            action_mask: vec![true; NUM_ACTIONS],
        };

        let obs2 = TilersObs {
            placement: vec![
                Qubit { id: QubitId(10), orientation: Orientation::Horizontal },
                Qubit { id: QubitId(11), orientation: Orientation::Vertical },
            ],
            objectives: vec![vec![
                Objective::cz(QubitId(10), QubitId(11)),
            ]],
            height: 5,
            width: 4,
            num_ancillas: 1,
            action_mask: vec![false; NUM_ACTIONS],
        };

        let mut slot = TilersSlot::default();
        slot.pack_observations(&[obs1, obs2]).unwrap();
        let unpacked = slot.unpack_observations();

        assert_eq!(unpacked.len(), 2);

        assert_eq!(unpacked[0].height, 3);
        assert_eq!(unpacked[0].placement[0].id, QubitId(0));
        assert_eq!(unpacked[0].objectives[0][0].opcode, Operation::CX);
        assert!(unpacked[0].action_mask[0]);

        assert_eq!(unpacked[1].height, 5);
        assert_eq!(unpacked[1].width, 4);
        assert_eq!(unpacked[1].num_ancillas, 1);
        assert_eq!(unpacked[1].placement[0].id, QubitId(10));
        assert_eq!(unpacked[1].objectives[0][0].opcode, Operation::CZ);
        assert!(!unpacked[1].action_mask[0]);
    }

    #[test]
    fn test_action_mask_roundtrip() {
        let mut mask = vec![false; NUM_ACTIONS];
        mask[0] = true;
        mask[42] = true;
        mask[NUM_ACTIONS - 1] = true;

        let obs = TilersObs {
            placement: vec![Qubit { id: QubitId(0), orientation: Orientation::Vertical }],
            objectives: vec![],
            height: 2,
            width: 2,
            num_ancillas: 0,
            action_mask: mask.clone(),
        };

        let mut slot = TilersSlot::default();
        slot.pack_observations(&[obs]).unwrap();
        let unpacked = slot.unpack_observations();

        assert!(unpacked[0].action_mask[0]);
        assert!(unpacked[0].action_mask[42]);
        assert!(unpacked[0].action_mask[NUM_ACTIONS - 1]);
        assert!(!unpacked[0].action_mask[1]);
        assert!(!unpacked[0].action_mask[43]);
    }    

    #[test]
    fn test_env_to_slot_roundtrip() {
        let env = TilersEnvInner::new(3, 3, 1);
        // set up objectives however your Environment API requires
        let tilers_env = TilersEnv::new(env, 2);
        let obs = tilers_env.observation();

        let mut slot = TilersSlot::default();
        slot.pack_observations(&[obs.clone()]).unwrap();
        let unpacked = slot.unpack_observations();

        assert_eq!(unpacked[0].height, obs.height);
        assert_eq!(unpacked[0].width, obs.width);
        assert_eq!(unpacked[0].placement.len(), obs.placement.len());
        assert_eq!(unpacked[0].objectives.len(), obs.objectives.len());
    }
}