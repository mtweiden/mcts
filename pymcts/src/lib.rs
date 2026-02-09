use std::collections::HashMap;

use mcts_core::enums::NodeId;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use mcts_core::mcts::MCTS;
use mcts_core::node::Node;
use mcts_core::environment::Environment as GenericRustEnvironment;
use mcts_core::enums::{Action, Observation, Prior, TokenId, Value};


#[pymodule]
fn pymcts(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMcts>()?;
    m.add_class::<MctsAgent>()?;
    m.add_class::<MctsNode>()?;
    m.add_class::<MctsEnvironment>()?;
    Ok(())
}


/// Convert an Observation to a Python dictionary.
pub fn obs_to_pydict<'py>(py: Python<'py>, obs: &Observation) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("placement", &obs.placement)?;
    dict.set_item("objectives_0", &obs.objectives_0)?;
    dict.set_item("objectives_1", &obs.objectives_1)?;
    dict.set_item("height", obs.height)?;
    dict.set_item("width", obs.width)?;
    dict.set_item("num_ancillas", obs.num_ancillas)?;
    dict.set_item("valid_actions", &obs.valid_actions)?;
    Ok(dict)
}

// ----------------------------------------------------------------------------
// MctsNode definition
// ----------------------------------------------------------------------------
#[pyclass(module = "pymcts", name="MctsNode")]
pub struct MctsNode {
    pub inner: Node,
}

#[pymethods]
impl MctsNode {
    fn id(&self) -> u64 { self.inner.id }
    fn prior_probs(&self) -> &Prior { &self.inner.prior_probs }
    fn value(&self) -> Value { self.inner.value }
    fn terminal_state(&self) -> bool { self.inner.terminal_state }
    fn repr(&self) -> Option<String> { self.inner.repr.as_ref().cloned() }
    fn edge_visits(&self) -> HashMap<Action, usize> { self.inner.edge_visits.clone() }
}

// ----------------------------------------------------------------------------
// MctsAgent definition
// ----------------------------------------------------------------------------
#[pyclass(module = "pymcts", name="MctsAgent")]
pub struct MctsAgent {
    agent: Py<PyAny>,
}

#[pymethods]
impl MctsAgent {
    #[new]
    fn new(agent: Py<PyAny>) -> Self { MctsAgent { agent } }

}

impl MctsAgent {
    pub fn infer(&self, observations: &Vec<Observation>) -> (Vec<Prior>, Vec<Value>) {
        let (prior, value) = Python::attach(|py| {
            // Convert observations to Python dictionaries
            let obs_dicts: Vec<Bound<PyDict>> = observations
                .iter()
                .map(|obs| obs_to_pydict(py, obs))
                .collect::<Result<_, _>>().unwrap();
            // Call inference on Python agent
            let out = self.agent.as_ref().call_method1(py, "infer", (obs_dicts,)).unwrap();
            let tup = out.as_ref().cast_bound::<PyTuple>(py).unwrap();
            // item0: list[dict[int (Action), float (Probability)]]
            // item1: list[float (Value)]
            let prior_list = tup.get_item(0).unwrap();
            let prior: Vec<Prior> = prior_list.extract().unwrap();
            let value: Vec<Value> = tup.get_item(1).unwrap().extract().unwrap();
            (prior, value)
        });
        (prior, value)
    }
}

// ----------------------------------------------------------------------------
// MctsEnvironment definition
// ----------------------------------------------------------------------------
#[pyclass(module = "pymcts", name="MctsEnvironment")]
pub struct MctsEnvironment {
    obj: Py<PyAny>,
}

#[pymethods]
impl MctsEnvironment {
    #[new]
    fn new(obj: Py<PyAny>) -> Self { MctsEnvironment { obj } }
}

/// Need a custom Clone implementation because Py<PyAny> does not implement Clone.
impl Clone for MctsEnvironment {
    fn clone(&self) -> Self {
        Python::attach(|py| {
            let obj = self.obj.as_ref();
            let copy_mod = py.import("copy").unwrap();
            let new_obj = copy_mod.call_method1("deepcopy", (obj,)).unwrap();
            MctsEnvironment { obj: new_obj.into() }
        })
    }
}

impl GenericRustEnvironment for MctsEnvironment {
    fn step(&mut self, action: Action) {
        Python::attach(|py| {
            let env = self.obj.as_ref();
            env.call_method1(py, "step", (action,)).unwrap();
        });
    }

    fn done(&self) -> bool {
        Python::attach(|py| {
            let env = self.obj.as_ref();
            let result: bool =  env.call_method0(py, "done").unwrap().extract(py).unwrap();
            result
        })
    }

    fn observation(&self) -> Observation {
        Python::attach(|py| {
            let env = self.obj.as_ref();
            let placement: Vec<TokenId> = env.call_method0(py, "get_placement_tokens").unwrap().extract(py).unwrap();
            let objectives_0: Vec<TokenId> = env.call_method1(py, "get_objective_tokens", (0,)).unwrap().extract(py).unwrap();
            let objectives_1: Vec<TokenId> = env.call_method1(py, "get_objective_tokens", (1,)).unwrap().extract(py).unwrap();
            let height: usize = env.getattr(py, "height").unwrap().extract(py).unwrap();
            let width: usize = env.getattr(py, "width").unwrap().extract(py).unwrap();
            let num_ancillas: usize = env.getattr(py, "num_ancillas").unwrap().extract(py).unwrap();
            let valid_actions: Vec<Action> = env.call_method0(py, "valid_actions").unwrap().extract(py).unwrap();
            Observation::from((placement, objectives_0, objectives_1, height, width, num_ancillas, valid_actions))
        })
    }

    fn valid_actions(&self) -> Vec<Action> {
        Python::attach(|py| {
            let env = self.obj.as_ref();
            let va_obj = env.call_method0(py, "valid_actions").unwrap();
            let va_vec: Vec<Action> = va_obj.extract(py).unwrap();
            va_vec
        })
    }

    fn hash_state(&self) -> u64 {
        Python::attach(|py| {
            let env = self.obj.as_ref();
            let h = env.call_method0(py, "hash_state").unwrap();
            let hash: u64 = h.extract(py).unwrap();
            hash
        })
    }

    fn render(&self) -> String {
        Python::attach(|py| {
            let env = self.obj.as_ref();
            let r = env.call_method0(py, "render").unwrap();
            let render_str: String = r.extract(py).unwrap();
            render_str
        })
    }
}


// ----------------------------------------------------------------------------
// PyMcts definition
// ----------------------------------------------------------------------------
#[pyclass(module = "pymcts", name="PyMcts", unsendable)]
pub struct PyMcts {
    pub inner: MCTS<MctsEnvironment>,
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
        env: &MctsEnvironment,
        agent: &MctsAgent,
        num_steps: usize,
    ) -> PyResult<MctsNode> {
        // Ensure root node exists
        let root_hash = self.inner.get_hash(env);
        if !self.inner.node_exists(root_hash) {
            let obs = env.observation();
            let (p, v) = agent.infer(&vec![obs]);
            let value = v[0];
            let mut priors = p[0].clone();
            priors = self.inner.normalize_prior(priors, env.valid_actions());
            self.inner.create_node(env, priors, value);
        }

        if env.done() {
            let node = self.inner.get_node_mut(root_hash).unwrap();
            let mcts_node = MctsNode { inner: node.clone() };
            return Ok(mcts_node);
        };
        let num_batches = num_steps / self.inner.batch_size.max(1);
        for _ in 0..num_batches {
            // Selection: collect a batch of leaf observations / metadata
            let mut leaf_batch: Vec<Observation> = Vec::with_capacity(self.inner.batch_size);
            let mut path_batch: Vec<Vec<(NodeId, Action)>> = Vec::with_capacity(self.inner.batch_size);
            let mut parent_batch: Vec<(Option<NodeId>, Action, MctsEnvironment)> =Vec::with_capacity(self.inner.batch_size);
            let mut repeat_batch: Vec<bool> = Vec::with_capacity(self.inner.batch_size);

            for _ in 0..self.inner.batch_size {
                // select_leaf clones the environment internally and returns the reached env
                let (path, parent, action, final_env, repeat) = self.inner.select_leaf(root_hash, env);
                let obs = final_env.observation();
                leaf_batch.push(obs);
                path_batch.push(path);
                parent_batch.push((parent, action, final_env));
                repeat_batch.push(repeat);
            }

            if leaf_batch.is_empty() { continue; }

            // --- Batched Inference ---
            let (prior_batch, value_batch) = agent.infer(&leaf_batch);

            // Expansion & Backpropagation
            let n = prior_batch.len().min(value_batch.len()).min(parent_batch.len());
            for i in 0..n {
                let (parent_opt, action, game) = &parent_batch[i];
                if let Some(parent_id) = parent_opt {
                    let priors = prior_batch[i].clone();
                    let value = value_batch[i];
                    // expand and then backpropagate
                    let _leaf = self.inner.expand(*parent_id, *action, game.clone(), priors, value);
                    self.inner.backpropagate(&path_batch[i], repeat_batch[i]);
                }
            }
        }
        let node = self.inner.get_node_mut(root_hash).unwrap();
        let mcts_node = MctsNode { inner: node.clone() };
        Ok(mcts_node)
    }
}