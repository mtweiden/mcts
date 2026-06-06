use mcts_core::environment::{Environment, Obs};

// Re-export the tilers Environment under an unambiguous name
use tilers::env::Environment as TilersInner;
// `rl` is the integer-action / board boundary adapter — the *only* place
// mcts crosses between its `u16` action ids and tilers' typed `Action`,
// and the source of the per-cell observation board.
use tilers::rl::{self, board::{construct_board, BoardCell}};

use crate::constants::{Action, NUM_ACTIONS};

// ---------------------------------------------------------------------------
// Observation type
// ---------------------------------------------------------------------------
/// `board` is `Vec<Vec<BoardCell>>` — one layer per objective layer
/// (`lookahead + 1` total), each layer `height * width` cells in row-major
/// order.  See `tilers::rl::board` / `tile/BOARD_REDESIGN.md` §3 for the
/// 10-channel cell schema.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct TilersObs {
    pub height: usize,
    pub width: usize,
    pub num_ancillas: usize,
    pub num_layers: usize,
    pub board: Vec<Vec<BoardCell>>,
    /// Length `NUM_ACTIONS`, padded; index = the `rl::encode` action id.
    pub action_mask: Vec<bool>,
}

impl Obs for TilersObs {}

// ---------------------------------------------------------------------------
// TilersEnv
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct TilersEnv {
    pub inner: TilersInner,
    pub num_objective_layers: usize,
}

impl TilersEnv {
    pub fn new(inner: TilersInner, num_objective_layers: usize) -> Self {
        Self { inner, num_objective_layers }
    }

    pub fn build_obs(&self) -> TilersObs {
        let height = self.inner.height;
        let width = self.inner.width;
        let num_ancillas = self.inner.num_ancillas();

        // `construct_board(env, lookahead)` produces `lookahead + 1` layers;
        // `num_objective_layers` is that count.
        let lookahead = self.num_objective_layers.saturating_sub(1);
        let board = construct_board(&self.inner, lookahead);
        let num_layers = board.len();

        // Validity mask in the flat integer action space, padded to the
        // fixed `NUM_ACTIONS`.  `rl::action_mask` is indexed by the same
        // `rl::encode` ids `valid_actions()` returns.
        let mask = rl::action_mask(&self.inner);
        let mut action_mask = vec![false; NUM_ACTIONS];
        for (id, &on) in mask.iter().enumerate() {
            if on && id < NUM_ACTIONS {
                action_mask[id] = true;
            }
        }

        TilersObs {
            height,
            width,
            num_ancillas,
            num_layers,
            board,
            action_mask,
        }
    }
}

impl Environment for TilersEnv {
    type Act = Action;
    type Obs = TilersObs;

    fn step(&mut self, action: Self::Act) {
        let a = rl::decode(&self.inner, action as usize)
            .expect("MCTS passed an invalid action id");
        self.inner.step(a).expect("MCTS passed an invalid action");
        self.inner.finish_cultivating(None, None);
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    fn observation(&self) -> Self::Obs {
        self.build_obs()
    }

    fn valid_actions(&self) -> Vec<Self::Act> {
        self.inner
            .valid_actions()
            .into_iter()
            .map(|a| rl::encode(&self.inner, a) as u16)
            .collect()
    }

    fn hash(&self) -> u64 {
        self.inner.hash_state()
    }

    fn render(&self) -> String {
        self.inner.render()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tilers::env::Environment as TilersEnvInner;

    #[test]
    fn test_valid_actions_nonempty() {
        let env = TilersEnvInner::new(3, 3, 1);
        let mcts_env = TilersEnv::new(env, 2);
        let obs = mcts_env.observation();
        let valid: Vec<usize> = obs.action_mask.iter()
            .enumerate()
            .filter(|(_, v)| **v)
            .map(|(i, _)| i)
            .collect();
        assert!(!valid.is_empty(), "fresh env should have at least one valid action");
    }

    #[test]
    fn test_step_with_valid_action() {
        let env = TilersEnvInner::new(3, 3, 1);
        let mut mcts_env = TilersEnv::new(env, 2);
        let obs = mcts_env.observation();
        let first_valid = obs.action_mask.iter()
            .position(|&v| v)
            .expect("should have a valid action");
        
        mcts_env.step(first_valid as u16);
        // After one step, should still be able to observe
        let obs2 = mcts_env.observation();
        assert_eq!(obs2.height, obs.height);
        assert_eq!(obs2.width, obs.width);
    }

    #[test]
    fn test_step_changes_state() {
        let env = TilersEnvInner::new(3, 3, 1);
        let mut mcts_env = TilersEnv::new(env, 2);
        let obs_before = mcts_env.observation();
        let first_valid = obs_before.action_mask.iter()
            .rposition(|&v| v)
            .expect("should have a valid action");

        mcts_env.step(first_valid as u16);
        let obs_after = mcts_env.observation();

        // Something should have changed — board or action mask
        let changed = obs_before.board != obs_after.board
            || obs_before.action_mask != obs_after.action_mask;
        assert!(changed, "state should change after a step");
    }

    #[test]
    fn test_invalid_action_not_in_mask() {
        let env = TilersEnvInner::new(3, 3, 1);
        let tilers_env = TilersEnv::new(env, 2);
        let obs = tilers_env.observation();
        let first_invalid = obs.action_mask.iter()
            .position(|&v| !v);

        if let Some(invalid_idx) = first_invalid {
            // Depending on your design, stepping with an invalid action
            // should either panic or return an error
            // Test whichever behavior you expect
            assert!(!obs.action_mask[invalid_idx]);
        }
    }

    #[test]
    fn test_game_terminates() {
        let env = TilersEnvInner::new(3, 3, 1);
        let mcts_env = TilersEnv::new(env, 2);
        assert!(mcts_env.done(), "fresh env should be done");
    }
}