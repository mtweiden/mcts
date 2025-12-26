use std::collections::HashMap;

use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------------------------
// Types and constants
// ---------------------------------------------------------------------------------------------
/// There is a global "execute" action and 5 actions per ancilla (max 20 ancillas)
pub const NUM_ACTIONS: usize = 101;
/// Right now we limit to 20x20 grids
pub const GRID_MAX: usize = 400;
/// Max number of objective tokens per layer
pub const MAX_OBJ0: usize = 800;
pub const MAX_OBJ1: usize = 800;
/// The zero token is for padding
pub const PAD_U16: u16 = 0x0000;
/// Maximum batch size for inference requests
pub const MAX_BATCH: usize = 8;
/// NodeId is a unique identifier for each node in the MCTS tree
pub type NodeId = u64;
/// Prior probabilities over actions implemented as a fixed size array
pub type Prior = HashMap<Action, f32>;
pub type StoredPrior = [f32; NUM_ACTIONS];
/// Value estimate of a state in the MCTS tree
pub type Value = f32;
/// Actions are represented as much denser u16 values
pub type Action = u16;
/// TokenIs are u16 ids representing different tokens in the environment
pub type TokenId = u16;

// ---------------------------------------------------------------------------------------------
// Environment observation
// ---------------------------------------------------------------------------------------------
/// Observation as a concrete struct
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub placement: Vec<TokenId>,
    pub objectives_0: Vec<TokenId>,
    pub objectives_1: Vec<TokenId>,
    pub height: usize,
    pub width: usize,
    pub valid_actions: Vec<Action>,
}

/// Convenience conversions to/from the old tuple shape:
impl From<(Vec<TokenId>, Vec<TokenId>, Vec<TokenId>, usize, usize, Vec<Action>)> for Observation {
    fn from(t: (Vec<TokenId>, Vec<TokenId>, Vec<TokenId>, usize, usize, Vec<Action>)) -> Self {
        Self {
            placement: t.0,
            objectives_0: t.1,
            objectives_1: t.2,
            height: t.3,
            width: t.4,
            valid_actions: t.5,
        }
    }
}

impl From<Observation> for (Vec<TokenId>, Vec<TokenId>, Vec<TokenId>, usize, usize, Vec<Action>) {
    fn from(o: Observation) -> Self {
        (o.placement, o.objectives_0, o.objectives_1, o.height, o.width, o.valid_actions)
    }
}

// ---------------------------------------------------------------------------------------------
// Agent Type Enum
// ---------------------------------------------------------------------------------------------
pub enum AgentType {
    Dummy,
    Python,
    Shm,
}