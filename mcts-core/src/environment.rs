use std::fmt::Debug;
use std::hash::Hash;

/// A generic observation type
pub trait Obs: Clone + Debug + Send + Hash + Eq {}

/// A generic action type.
///
/// `to_action_index` is required so per-edge data (priors, visit counts,
/// children) can be stored in dense `Vec<T>` indexed by action id rather
/// than `HashMap<Self, T>`. The intended pattern: every action implements
/// the method by an `as usize` cast (or equivalent), and `Node` /
/// `MCTS` use it to translate `Self::Act` values into Vec indices.
///
/// `Into<usize>` would be the natural bound but stdlib withholds the
/// `From<u32> for usize` impl because of 16-bit-usize platforms — none
/// of which we target — so we define a small dedicated method instead.
/// Anyone adding a new env with an integer action type just adds the
/// one-line impl.
pub trait Act: Copy + Debug + Send + Hash + Eq {
    fn to_action_index(self) -> usize;
}

impl Act for u8 { fn to_action_index(self) -> usize { self as usize } }
impl Act for u16 { fn to_action_index(self) -> usize { self as usize } }
impl Act for u32 { fn to_action_index(self) -> usize { self as usize } }
impl Act for u64 { fn to_action_index(self) -> usize { self as usize } }
impl Act for usize { fn to_action_index(self) -> usize { self } }

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
    /// Return the size of the action space at the current state. Per-state
    /// dense storage in `Node` is sized to this value, so the contract
    /// is: every valid action's `to_action_index()` must be in
    /// `0..num_actions(&state)`. Action ids are scoped to a single
    /// state — the same id may mean different things in different
    /// states, but the env must not return an action whose index would
    /// land outside its own state's bound.
    fn num_actions(&self) -> usize;
    /// Return a compact hash / id for the current state.
    fn hash(&self) -> u64;
    /// Render a string representation (used for debugging / repr).
    fn render(&self) -> String;
}
