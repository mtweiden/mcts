use anyhow::Result;
use std::collections::HashMap;

use mcts_core::InferenceClient;
use pyo3::prelude::*;
use pyo3::Py;
use pyo3::types::{PyAny, PyDict, PyTuple};

use mcts_core::mcts::MCTS;
use mcts_core::node::Node;

use tilers::env::PyEnvironment;
use tilers::env::Environment as TilersEnvInner;
use tilers::solver::Solver as TilersSolver;

use crate::constants::*;
use crate::environment::{TilersObs, TilersEnv};

// ─────────────────────────────────────────────────────────────────────────────
// Observation <-> Python conversion
// ─────────────────────────────────────────────────────────────────────────────
fn obs_to_pydict<'py>(py: Python<'py>, obs: &TilersObs) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    // board: list[layer] of list[cell] of [CELL_FIELDS i16] — the 10-channel
    // observation (see TILERS_MIGRATION.md §4 / tilers::rl::board).
    let board: Vec<Vec<[i16; CELL_FIELDS]>> = obs
        .board
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|c| {
                    [
                        c.qubit_role,
                        c.factor_kind,
                        c.resource_kind,
                        c.pp_group_row,
                        c.pp_group_col,
                        c.ancilla_idx,
                        c.last_move_dir,
                        c.weight_in_pp,
                        c.is_hub_for_pp,
                        c.is_y_ready,
                        c.pp_needs_t,
                    ]
                })
                .collect()
        })
        .collect();
    dict.set_item("board", &board)?;

    dict.set_item("height", obs.height)?;
    dict.set_item("width", obs.width)?;
    dict.set_item("num_ancillas", obs.num_ancillas)?;
    dict.set_item("num_layers", obs.num_layers)?;

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
        // Inner storage is a dense Vec<f32> indexed by action; the
        // Python contract is still HashMap<Action, f32>, so re-emit
        // entries only for actions that are valid at this node.
        use mcts_core::environment::Act;
        let mut out = HashMap::with_capacity(self.inner.valid_actions.len());
        for &a in &self.inner.valid_actions {
            out.insert(a, self.inner.prior_probs[a.to_action_index()]);
        }
        out
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
        // Inner storage is a dense Vec<usize>; rebuild HashMap for the
        // Python contract.
        use mcts_core::environment::Act;
        let mut out = HashMap::with_capacity(self.inner.valid_actions.len());
        for &a in &self.inner.valid_actions {
            out.insert(a, self.inner.edge_visits[a.to_action_index()]);
        }
        out
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

/// MctsAgent needs to implement infer_from_obs on the Python side.
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
                .call_method1(py, "infer_from_obs", (obs_dicts,))
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
    #[pyo3(signature = (batch_size = 8, root_softmax_temp = 1.03))]
    fn new(batch_size: usize, root_softmax_temp: f32) -> Self {
        let mut mcts = MCTS::new(batch_size);
        // T < 1 SHARPENS the root prior (P^(1/T), renormalised); T > 1
        // flattens. Root-only by design: advance_root makes every decision
        // point the root, so choices are sharpened while in-tree lookahead
        // keeps honest priors.
        mcts.root_softmax_temp = root_softmax_temp;
        PyMcts { inner: mcts }
    }

    #[pyo3(signature = (env, agent, num_steps = 1000, c_puct = 1.4, forced_playouts = false,
                        margin_terminal = false, reward_saturation_temperature = 1.0,
                        band_terminal = false))]
    fn run(
        &mut self,
        env: &PyEnvironment,
        agent: &MctsAgent,
        num_steps: usize,
        c_puct: f32,
        forced_playouts: bool,
        margin_terminal: bool,
        reward_saturation_temperature: f32,
        band_terminal: bool,
    ) -> PyResult<MctsNode> {
        let inner: TilersEnvInner = env.to_inner();
        // Terminal evaluator vs the heuristic solver. Default: the historical
        // ternary {beat, tie, lose/unfinished} -> {1, 0, -1}, kept bit-
        // identical so probe scripts stay comparable across eras. CAUTION
        // (2026-07-12): production eval/gather search the MARGIN currency
        // (done_score tanh, bare completion ~ 0) — the ternary is
        // completion-greedy, so absolute completion rates measured with it
        // overstate production. Pass margin_terminal=true to search the
        // production game.
        let ref_depth = {
            let mut e = inner.clone();
            let solver = TilersSolver::new();
            let _ = solver.solve(&mut e, true);
            e.depth(true, true) as f32
        };
        let terminal_evaluator = move |e: &TilersEnv| -> f32 {
            if !e.inner.done() {
                crate::reward::NOT_DONE_SCORE
            } else {
                let d = e.inner.depth(true, true) as f32;
                if band_terminal {
                    // E3b: completion floor + margin gradient (0.5 + 0.5*margin)
                    0.5 + 0.5 * crate::reward::done_score(ref_depth, d, reward_saturation_temperature)
                } else if margin_terminal {
                    crate::reward::done_score(ref_depth, d, reward_saturation_temperature)
                } else if d < ref_depth { 1.0 }
                else if d == ref_depth { 0.0 }
                else { -1.0 }
            }
        };
        let env = TilersEnv::new(inner, DEFAULT_LOOKAHEAD);
        let node = self.inner.run(
            &env, agent, num_steps, c_puct, &terminal_evaluator, forced_playouts
        );
        Ok(MctsNode { inner: node })
    }

    fn advance_root(&mut self, action: Action) -> PyResult<()> {
        self.inner.advance_root(action);
        Ok(())
    }
}