pub mod constants;
pub mod environment;
pub mod slot;
pub mod client;
pub mod evaluator;
pub mod gatherer;

#[cfg(feature = "python")]
pub mod py_ipc;
#[cfg(feature = "python")]
pub mod py_mcts;
#[cfg(feature = "python")]
use pyo3::prelude::*;


// ─────────────────────────────────────────────────────────────────────────────
// Python module
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "python")]
#[pymodule]
fn mcts_tilers(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Constants
    m.add("MAX_ANCILLAS", crate::constants::MAX_ANCILLAS)?;
    m.add("MAX_BATCH", crate::constants::MAX_BATCH)?;
    m.add("NUM_ACTIONS", crate::constants::NUM_ACTIONS)?;
    m.add("GRID_MAX", crate::constants::GRID_MAX)?;
    m.add("LOOKAHEAD_MAX", crate::constants::LOOKAHEAD_MAX)?;
    m.add("CELL_FIELDS", crate::constants::CELL_FIELDS)?;
    m.add("BOARD_LAYER_MAX", crate::constants::BOARD_LAYER_MAX)?;
    m.add("BOARD_MAX", crate::constants::BOARD_MAX)?;

    // IPC Classes
    m.add_class::<py_ipc::PyArena>()?;
    m.add_class::<py_ipc::PySlotView>()?;

    // MCTS Classes
    m.add_class::<py_mcts::PyMcts>()?;
    m.add_class::<py_mcts::MctsAgent>()?;
    m.add_class::<py_mcts::MctsNode>()?;

    // tilers classes.  These come from the tilers crate statically linked
    // into *this* extension, so they are distinct Python types from the
    // standalone `tilers` package — always use `mcts_tilers.{Environment,
    // Action, ...}` together (a `tilers.Environment` won't interop here).
    m.add_class::<tilers::enums::PyOrientation>()?;
    m.add_class::<tilers::enums::PyDirection>()?;
    m.add_class::<tilers::enums::PyAction>()?;
    m.add_class::<tilers::enums::PyOperation>()?;
    m.add_class::<tilers::enums::PyPosition>()?;
    m.add_class::<tilers::qubit::PyQubit>()?;
    m.add_class::<tilers::objective::PyObjective>()?;
    m.add_class::<tilers::env::PyEnvironment>()?;

    // `mcts_tilers.rl` — the action/board adapter over *this* extension's
    // tilers types, so a Python MCTS loop can `decode` an action id and
    // `step` the env without crossing module boundaries.
    let rl_mod = PyModule::new(m.py(), "rl")?;
    rl_mod.add_function(wrap_pyfunction!(tilers::rl::py::observe,     &rl_mod)?)?;
    rl_mod.add_function(wrap_pyfunction!(tilers::rl::py::action_mask, &rl_mod)?)?;
    rl_mod.add_function(wrap_pyfunction!(tilers::rl::py::encode,      &rl_mod)?)?;
    rl_mod.add_function(wrap_pyfunction!(tilers::rl::py::decode,      &rl_mod)?)?;
    rl_mod.add_function(wrap_pyfunction!(tilers::rl::py::opposite,    &rl_mod)?)?;
    m.add_submodule(&rl_mod)?;

    // for gathering and evaluating
    m.add_function(wrap_pyfunction!(gatherer::run_gatherer, m)?)?;
    m.add_function(wrap_pyfunction!(evaluator::run_evaluator, m)?)?;

    Ok(())
}

