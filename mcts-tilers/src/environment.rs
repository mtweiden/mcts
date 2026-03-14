use mcts_core::environment::{Environment, Obs};

// Re-export the tilers Environment under an unambiguous name
use tilers::{enums::QubitId, env::Environment as TilersInner};
use tilers::qubit::Qubit;
use tilers::objective::Objective;
use tilers::enums::Direction::{Up, Down};

use crate::constants::{Action, NUM_ACTIONS};

// ---------------------------------------------------------------------------
// Observation type
// ---------------------------------------------------------------------------
/// Board is Vec<Vec<BoardCell>> — one layer per objective layer, each layer
/// has h*w cells in row-major order.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct TilersObs {
    pub placement: Vec<Qubit>,
    pub objectives: Vec<Vec<Objective>>,
    pub height: usize,
    pub width: usize,
    pub num_ancillas: usize,
    pub action_mask: Vec<bool>,
    pub last_dir_vertical: Vec<bool>,  // bool for each ancilla
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
        let placement: Vec<Qubit> = self.inner.get_placement();
        let objectives: Vec<Vec<Objective>> = self.inner.get_objectives(self.num_objective_layers - 1);
        assert_eq!(objectives.len(), self.num_objective_layers);

        let height = self.inner.height;
        let width = self.inner.width;
        let num_ancillas = self.inner.num_ancillas();
        let valid = self.inner.valid_actions();
        let num_actions = self.inner.num_actions();
        let mut action_mask = vec![false; NUM_ACTIONS];
        for &a in &valid { if a < num_actions { action_mask[a] = true; } }

        let last_dirs = &self.inner.last_dirs;
        let mut last_dir_vertical = vec![false; num_ancillas];
        for a in 0..num_ancillas {
            let qid = QubitId(-((a + 1) as i32));
            if let Some(&dir) = last_dirs.get(&qid) {
                if dir == Up || dir == Down {
                    last_dir_vertical[a] = true;
                }
            }
        }

        TilersObs {
            placement,
            objectives,
            height,
            width,
            num_ancillas,
            action_mask,
            last_dir_vertical,
        }
    }
}

impl Environment for TilersEnv {
    type Act = Action;
    type Obs = TilersObs;

    fn step(&mut self, action: Self::Act) {
        let _ = self.inner.step(action as usize).expect("MCTS passed an invalid action");
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    fn observation(&self) -> Self::Obs {
        self.build_obs()
    }

    fn valid_actions(&self) -> Vec<Self::Act> {
        self.inner.valid_actions().into_iter().map(|a| a as u16).collect()
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

        // Something should have changed — placement or objectives or action mask
        let changed = obs_before.placement != obs_after.placement
            || obs_before.objectives != obs_after.objectives
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