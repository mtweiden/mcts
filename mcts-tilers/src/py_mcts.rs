use anyhow::Result;
use std::collections::HashMap;

use mcts_core::InferenceClient;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use mcts_core::mcts::MCTS;
use mcts_core::node::Node;

use tilers::env::PyEnvironment;

use crate::constants::*;
use crate::environment::{TilersObs, TilersEnv};

// ─────────────────────────────────────────────────────────────────────────────
// Observation <-> Python conversion
// ─────────────────────────────────────────────────────────────────────────────
fn obs_to_pydict<'py>(py: Python<'py>, obs: &TilersObs) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    // placement: list of (i32, u8) tuples
    let placement: Vec<(i32, u8)> = obs
        .placement
        .iter()
        .map(|q| (q.id.as_i32(), q.orientation as u8))
        .collect();
    dict.set_item("placement", &placement)?;

    // objectives: list of list of (u8, i32, i32, usize, u8) tuples
    let objectives: Vec<Vec<(u8, i32, i32, usize, u8)>> = obs
        .objectives
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|o| {
                    (
                        o.opcode as u8,
                        o.arg_0.as_i32(),
                        o.arg_1.as_i32(),
                        o.duration,
                        o.direction as u8,
                    )
                })
                .collect()
        })
        .collect();
    dict.set_item("objectives", &objectives)?;

    dict.set_item("height", obs.height)?;
    dict.set_item("width", obs.width)?;
    dict.set_item("num_ancillas", obs.num_ancillas)?;

    let valid_actions: Vec<Action> = obs
        .action_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i as Action) } else { None })
        .collect();
    dict.set_item("valid_actions", &valid_actions)?;
    Ok(dict)
}

// ─────────────────────────────────────────────────────────────────────────────
// MctsNode
// ─────────────────────────────────────────────────────────────────────────────
#[pyclass(module = "mcts_tilers", name = "MctsNode")]
pub struct MctsNode {
    pub inner: Node<Action>,
}

#[pymethods]
impl MctsNode {
    fn id(&self) -> u64 {
        self.inner.id
    }

    fn prior_probs(&self) -> HashMap<Action, f32> {
        self.inner.prior_probs.clone()
    }

    fn value(&self) -> f32 {
        self.inner.value
    }

    fn terminal_state(&self) -> bool {
        self.inner.terminal_state
    }

    fn repr(&self) -> Option<String> {
        self.inner.repr.clone()
    }

    fn edge_visits(&self) -> HashMap<Action, usize> {
        self.inner.edge_visits.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MctsAgent — wraps a Python inference callable
// ─────────────────────────────────────────────────────────────────────────────
#[pyclass(module = "mcts_tilers", name = "MctsAgent")]
pub struct MctsAgent {
    agent: Py<PyAny>,
}

#[pymethods]
impl MctsAgent {
    #[new]
    fn new(agent: Py<PyAny>) -> Self {
        MctsAgent { agent }
    }
}

impl InferenceClient<TilersEnv> for MctsAgent {
    fn infer(&self, observations: &[TilersObs]) -> Result<(Vec<Prior>, Vec<Value>)> {
        Python::attach(|py| {
            let obs_dicts: Vec<Bound<PyDict>> = observations
                .iter()
                .map(|obs| obs_to_pydict(py, obs))
                .collect::<std::result::Result<_, PyErr>>()
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            let out = self
                .agent
                .as_ref()
                .call_method1(py, "infer", (obs_dicts,))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let tup = out.as_ref().cast_bound::<PyTuple>(py)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            let prior: Vec<Prior> = tup.get_item(0)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .extract()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let value: Vec<Value> = tup.get_item(1)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .extract()
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            Ok((prior, value))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyMcts
// ─────────────────────────────────────────────────────────────────────────────
#[pyclass(module = "mcts_tilers", name = "PyMcts", unsendable)]
pub struct PyMcts {
    pub inner: MCTS<TilersEnv>,
}

#[pymethods]
impl PyMcts {
    #[new]
    #[pyo3(signature = (terminal_value = 1.0, batch_size = 8))]
    fn new(terminal_value: f32, batch_size: usize) -> Self {
        let mcts = MCTS::new(terminal_value, batch_size);
        PyMcts { inner: mcts }
    }

    #[pyo3(signature = (env, agent, num_steps = 1000))]
    fn run(
        &mut self,
        env: &PyEnvironment,
        agent: &MctsAgent,
        num_steps: usize,
    ) -> PyResult<MctsNode> {
        // Note that because of this copy, we probably don't want to run MCTS on huge environments.
        // We should chunk them before sending them off to MCTS.
        let env = TilersEnv::new(env.to_inner(), DEFAULT_LOOKAHEAD);
        let node = self.inner.run(&env, agent, num_steps);
        Ok(MctsNode { inner: node })
    }

    fn advance_root(&mut self, action: Action) -> PyResult<()> {
        self.inner.advance_root(action);
        Ok(())
    }
}