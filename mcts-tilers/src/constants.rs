use std::collections::HashMap;

/// Maximum number of ancillas that can be handled (limited by action space and packing)
pub const MAX_ANCILLAS: usize = 50;
/// There is a global "execute" action and 5 actions per ancilla (max 50 ancillas)
pub const NUM_ACTIONS: usize = 251;
/// Right now we limit to 20x20 grids
pub const GRID_MAX: usize = 400;
/// Maximum number of layers that can be looked ahead for
pub const LOOKAHEAD_MAX: usize = 3;
/// Default number of layers to look ahead for when creating the environment
pub const DEFAULT_LOOKAHEAD: usize = 2;
/// Maximum batch size for inference requests
pub const MAX_BATCH: usize = 8;

/// Packed qubit size: 4 bytes i32 id + 1 byte orientation
pub const QUBIT_SIZE: usize = 5;
/// Packed objective size: 1 opcode + 4 arg0 + 4 arg1
pub const OBJECTIVE_SIZE: usize = 9;
/// Max bytes for placement per observation (GRID_MAX * QUBIT_SIZE)
pub const PLACEMENT_MAX: usize = 2000;
/// Max bytes for one layer of objectives (GRID_MAX * OBJECTIVE_SIZE)
pub const OBJECTIVES_LAYER_MAX: usize = 3600;
/// Max bytes for all objective layers per observation (LOOKAHEAD_MAX * OBJECTIVES_LAYER_MAX)
pub const OBJECTIVES_MAX: usize = 10800;

/// Actions are represented as dense u16 values
pub type Action = u16;
/// Prior probabilities over actions
pub type Prior = HashMap<Action, f32>;
/// Value estimate of a state
pub type Value = f32;