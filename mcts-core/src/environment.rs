use crate::enums::Action;
use crate::enums::Observation;
use crate::enums::TokenId;

/// Abstract Environment trait used by MCTS (single-threaded).
/// Implement this trait for any concrete environment you want to run MCTS on.
pub trait Environment: Clone {
    /// Apply the action to the environment (mutates self).
    fn step(&mut self, action: Action);

    /// Is the environment in a terminal state?
    fn done(&self) -> bool;

    /// Return the observation vector for the current state.
    fn observation(&self) -> Observation;

    /// Return the list of valid actions in the current state.
    fn valid_actions(&self) -> Vec<Action>;

    /// Return a compact hash / id for the current state.
    fn hash_state(&self) -> u64;

    /// Render a string representation (used for debugging / repr).
    fn render(&self) -> String;
}

// Provide an implementation for tilers_core::env::Environment so existing code works.
impl Environment for tilers_core::env::Environment {
    fn step(&mut self, action: Action) {
        tilers_core::env::Environment::step(self, action as usize);
    }

    fn done(&self) -> bool {
        tilers_core::env::Environment::done(self)
    }

    fn observation(&self) -> Observation {
        let (placement, obj_0) = tilers_core::env::Environment::get_tokens(self);
        let obj_1 = tilers_core::env::Environment::get_objective_tokens(self, 1);
        let valid_actions = tilers_core::env::Environment::valid_actions(self);
        let p: Vec<TokenId> = placement.iter().map(|&x| x as TokenId).collect();
        let o0: Vec<TokenId> = obj_0.iter().map(|&x| x as TokenId).collect();
        let o1: Vec<TokenId> = obj_1.iter().map(|&x| x as TokenId).collect();
        let va: Vec<Action> = valid_actions.iter().map(|&x| x as Action).collect();
        Observation::from((p, o0, o1, self.height, self.width, va))
    }

    fn valid_actions(&self) -> Vec<Action> {
        tilers_core::env::Environment::valid_actions(self)
            .into_iter()
            .map(|a| a as u16)
            .collect()
    }

    fn hash_state(&self) -> u64 {
        tilers_core::env::Environment::hash_state(self)
    }

    fn render(&self) -> String {
        tilers_core::env::Environment::render(self)
    }
}