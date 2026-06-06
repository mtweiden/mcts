use std::collections::HashMap;

/// Maximum number of ancillas that can be handled (limited by action space and packing)
pub const MAX_ANCILLAS: usize = 50;
/// Flat action space: 1 auto-execute + 6 actions per ancilla (cultivate,
/// wire, and 4 moves) = `1 + 6·MAX_ANCILLAS`.  Matches tilers' `rl`
/// encoding; was `1 + 5N` before the `pauli`-branch wire band (see
/// TILERS_MIGRATION.md §2).  **Literal** (not an expression) because the
/// maturin cffi header step only accepts integer `#define`s.
pub const NUM_ACTIONS: usize = 301;
const _: () = assert!(NUM_ACTIONS == 1 + 6 * MAX_ANCILLAS);
/// Right now we limit to 20x20 grids
pub const GRID_MAX: usize = 400;
/// Maximum number of layers that can be looked ahead for
pub const LOOKAHEAD_MAX: usize = 3;
/// Default number of layers to look ahead for when creating the environment
pub const DEFAULT_LOOKAHEAD: usize = 2;
/// Maximum batch size for inference requests
pub const MAX_BATCH: usize = 8;

// --- Observation board packing (10-channel `tilers::rl::board::BoardCell`) ---
// These are literals (not expressions) so the maturin cffi header step
// emits plain-integer `#define`s; the const asserts keep them honest.
/// Channels per cell (i16 each) — must equal `tilers::rl::board::CELL_FIELDS`.
pub const CELL_FIELDS: usize = 10;
/// i16 values for one board layer = `GRID_MAX * CELL_FIELDS`.
pub const BOARD_LAYER_MAX: usize = 4000;
const _: () = assert!(BOARD_LAYER_MAX == GRID_MAX * CELL_FIELDS);
/// i16 values for the full board per observation = `LOOKAHEAD_MAX * BOARD_LAYER_MAX`.
pub const BOARD_MAX: usize = 12000;
const _: () = assert!(BOARD_MAX == LOOKAHEAD_MAX * BOARD_LAYER_MAX);

/// Actions are represented as dense u16 values
pub type Action = u16;
/// Prior probabilities over actions
pub type Prior = HashMap<Action, f32>;
/// Value estimate of a state
pub type Value = f32;