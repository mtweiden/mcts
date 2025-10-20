use crate::enums::Action;
use crate::enums::Observation;

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
        tilers_core::env::Environment::step(self, action);
    }

    fn done(&self) -> bool {
        tilers_core::env::Environment::done(self)
    }

    fn observation(&self) -> Observation {
        // tilers_core::env::Environment::observation(self)
        vec![0]
    }

    fn valid_actions(&self) -> Vec<Action> {
        tilers_core::env::Environment::valid_actions(self)
    }

    fn hash_state(&self) -> u64 {
        tilers_core::env::Environment::hash_state(self)
    }

    fn render(&self) -> String {
        tilers_core::env::Environment::render(self)
    }
}