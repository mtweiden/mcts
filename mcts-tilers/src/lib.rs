pub mod constants;
pub mod environment;
pub mod slot;
pub mod client;

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
    m.add("QUBIT_SIZE", crate::constants::QUBIT_SIZE)?;
    m.add("OBJECTIVE_SIZE", crate::constants::OBJECTIVE_SIZE)?;
    m.add("PLACEMENT_MAX", crate::constants::PLACEMENT_MAX)?;
    m.add("OBJECTIVES_LAYER_MAX", crate::constants::OBJECTIVES_LAYER_MAX)?;
    m.add("OBJECTIVES_MAX", crate::constants::OBJECTIVES_MAX)?;

    // IPC Classes
    m.add_class::<py_ipc::PyArena>()?;
    m.add_class::<py_ipc::PySlotView>()?;

    // MCTS Classes
    m.add_class::<py_mcts::PyMcts>()?;
    m.add_class::<py_mcts::MctsAgent>()?;
    m.add_class::<py_mcts::MctsNode>()?;

    // tilers classes
    m.add_class::<tilers::enums::PyOrientation>()?;
    m.add_class::<tilers::enums::PyDirection>()?;
    m.add_class::<tilers::enums::PyOperation>()?;
    m.add_class::<tilers::enums::PyPosition>()?;
    m.add_class::<tilers::qubit::PyQubit>()?;
    m.add_class::<tilers::objective::PyObjective>()?;
    m.add_class::<tilers::env::PyEnvironment>()?;

    Ok(())
}

