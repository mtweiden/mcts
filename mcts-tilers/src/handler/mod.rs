// Handler-side preprocessing in Rust.
//
// Mirrors the Python+numpy hot path in
// mcts-tilers/python/mcts_tilers/handler.py:
//
//   unpack_placement_batch(raw, num_qubits)             -> boards.rs path
//   unpack_objectives_batch(raw, num_layers, num_objs)  -> boards.rs path
//   build_boards(...)                                   -> boards.rs path
//
// Each of these takes raw byte slabs out of the shared-memory arena and
// produces the int32 board tensor the model consumes. They live here as
// the natural home for "Rust replacements for handler.py preprocessing".
//
// PyO3 exports are wired in lib.rs (cfg = "python"), under the same
// `mcts_tilers` extension module the Python side already imports.
//
// Correctness gate: tests/test_build_boards.py — the Python
// `_reference_build_boards` is the source of truth. Both the existing
// Python `build_boards` and the new Rust `build_boards_rs` must match
// it bit-for-bit on every test case.

#[cfg(feature = "python")]
pub mod boards;
#[cfg(feature = "python")]
pub mod unpack;
