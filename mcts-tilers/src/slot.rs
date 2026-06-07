use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::{anyhow, Result};
use mcts_core::ipc_core::{SlotInit, SLOT_FREE};

use tilers::rl::board::BoardCell;

use crate::constants::*;
use crate::environment::TilersObs;

// Note: the pre-PauliProduct env exposed `Orientation::from_u8` and
// `Operation::from_u8` for the placement/objective wire format used by
// the old slot layout.  Both helpers + that wire format are gone — the
// observation is now `Vec<Vec<BoardCell>>` (the 10-channel board) and
// the slot serializes those cells directly as i16 channels.

// ─────────────────────────────────────────────────────────────────────────────
// Slot states
// ─────────────────────────────────────────────────────────────────────────────
pub use mcts_core::ipc_core::{SLOT_DONE, SLOT_READY, SLOT_WAITING};

// ─────────────────────────────────────────────────────────────────────────────
// Tilers-specific Slot
//
// The observation is the 10-channel `tilers::rl::board` (see
// TILERS_MIGRATION.md §4): `num_layers` layers, each `h*w` cells in
// row-major order, each cell `CELL_FIELDS` i16 channels.  The flat
// placement/objective byte packing of the pre-PauliProduct env is gone.
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub struct TilersSlot {
    // Header
    pub state: AtomicU32,
    pub b: u32,
    pub owner_id: u32,
    pub req_id: u64,

    // Inputs — per-batch metadata
    pub h: [u8; MAX_BATCH],
    pub w: [u8; MAX_BATCH],
    pub num_ancillas: [u8; MAX_BATCH],
    pub num_layers: [u8; MAX_BATCH],

    // Observation board, laid out [layer][cell][channel] within BOARD_MAX i16.
    pub board: [[i16; BOARD_MAX]; MAX_BATCH],

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
            num_layers: [0; MAX_BATCH],
            board: [[0; BOARD_MAX]; MAX_BATCH],
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

#[inline]
fn cell_to_array(c: &BoardCell) -> [i16; CELL_FIELDS] {
    [
        c.qubit_role,
        c.factor_kind,
        c.resource_kind,
        c.pp_group_row,
        c.pp_group_col,
        c.ancilla_idx,
        c.last_move_dir,
        c.weight_in_pp,
        c.is_hub_for_pp,
        c.is_y_ready,
    ]
}

#[inline]
fn cell_from_slice(s: &[i16]) -> BoardCell {
    BoardCell {
        qubit_role: s[0],
        factor_kind: s[1],
        resource_kind: s[2],
        pp_group_row: s[3],
        pp_group_col: s[4],
        ancilla_idx: s[5],
        last_move_dir: s[6],
        weight_in_pp: s[7],
        is_hub_for_pp: s[8],
        is_y_ready: s[9],
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

            let nl = obs.board.len().min(LOOKAHEAD_MAX);
            self.num_layers[i] = nl as u8;

            self.board[i].fill(0);
            for (l, layer) in obs.board.iter().take(nl).enumerate() {
                let layer_off = l * BOARD_LAYER_MAX;
                for (c, cell) in layer.iter().take(GRID_MAX).enumerate() {
                    let off = layer_off + c * CELL_FIELDS;
                    self.board[i][off..off + CELL_FIELDS].copy_from_slice(&cell_to_array(cell));
                }
            }

            // Pack action mask (length NUM_ACTIONS, already encoded ids).
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
            let nl = (self.num_layers[i] as usize).min(LOOKAHEAD_MAX);
            let hw = (height * width).min(GRID_MAX);

            let mut board = Vec::with_capacity(nl);
            for l in 0..nl {
                let layer_off = l * BOARD_LAYER_MAX;
                let mut layer = Vec::with_capacity(hw);
                for c in 0..hw {
                    let off = layer_off + c * CELL_FIELDS;
                    layer.push(cell_from_slice(&self.board[i][off..off + CELL_FIELDS]));
                }
                board.push(layer);
            }

            let mask_offset = i * NUM_ACTIONS;
            let action_mask: Vec<bool> = self.action_mask[mask_offset..mask_offset + NUM_ACTIONS]
                .iter()
                .map(|&m| m != 0)
                .collect();

            out.push(TilersObs {
                height,
                width,
                num_ancillas,
                num_layers: nl,
                board,
                action_mask,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::TilersEnv;
    use mcts_core::Environment;
    use tilers::env::Environment as TilersEnvInner;

    fn sample_obs(h: usize, w: usize, na: usize) -> TilersObs {
        let env = TilersEnvInner::new(h, w, na);
        TilersEnv::new(env, DEFAULT_LOOKAHEAD).observation()
    }

    #[test]
    fn test_slot_roundtrip_board() {
        let obs = sample_obs(3, 3, 1);
        let mut slot = TilersSlot::default();
        slot.pack_observations(std::slice::from_ref(&obs)).unwrap();

        assert_eq!(slot.b, 1);
        assert_eq!(slot.h[0] as usize, obs.height);
        assert_eq!(slot.w[0] as usize, obs.width);
        assert_eq!(slot.num_ancillas[0] as usize, obs.num_ancillas);
        assert_eq!(slot.num_layers[0] as usize, obs.num_layers);

        let unpacked = slot.unpack_observations();
        assert_eq!(unpacked.len(), 1);
        let u = &unpacked[0];
        assert_eq!(u.height, obs.height);
        assert_eq!(u.width, obs.width);
        assert_eq!(u.num_layers, obs.num_layers);
        // Exact board + mask round-trip.
        assert_eq!(u.board, obs.board);
        assert_eq!(u.action_mask, obs.action_mask);
    }

    #[test]
    fn test_slot_roundtrip_batch() {
        let o1 = sample_obs(3, 3, 1);
        let o2 = sample_obs(4, 4, 2);
        let mut slot = TilersSlot::default();
        slot.pack_observations(&[o1.clone(), o2.clone()]).unwrap();
        assert_eq!(slot.b, 2);

        let u = slot.unpack_observations();
        assert_eq!(u[0].board, o1.board);
        assert_eq!(u[1].board, o2.board);
    }
}
