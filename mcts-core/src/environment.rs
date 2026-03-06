use std::fmt::Debug;
use std::hash::Hash;

/// A generic observation type
pub trait Obs: Clone + Debug + Send + Hash + Eq {}

/// A generic action type
pub trait Act: Copy + Debug + Send + Hash + Eq {}
impl<T: Copy + Debug + Send + Hash + Eq> Act for T {}

/// Implement this trait for any concrete environment you want to run MCTS on.
pub trait Environment: Clone {
    type Act: Act;
    type Obs: Obs;

    /// Apply the action to the environment (mutates self).
    fn step(&mut self, action: Self::Act);
    /// Is the environment in a terminal state?
    fn done(&self) -> bool;
    /// Return the observation vector for the current state.
    fn observation(&self) -> Self::Obs;
    /// Return the list of valid actions in the current state.
    fn valid_actions(&self) -> Vec<Self::Act>;
    /// Return a compact hash / id for the current state.
    fn hash(&self) -> u64;
    /// Render a string representation (used for debugging / repr).
    fn render(&self) -> String;
}