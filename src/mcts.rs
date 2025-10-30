use crate::node::Node;
use crate::enums::{Action, NodeId, Observation, Prior, Value};
use crate::agent::Agent;
use crate::environment::Environment as EnvTrait;
use std::collections::HashMap;
use std::marker::PhantomData;
use rmp_serde::{to_vec_named, from_slice};
use serde::{Serialize, Deserialize};
use std::sync::Arc;

/// ----------------------------------------------------------------------------
/// Communication data formats
/// ----------------------------------------------------------------------------
#[derive(Serialize)]
pub struct InferenceRequest {
    pub observation_batch: Vec<Observation>,
}

#[derive(Deserialize)]
pub struct InferenceResponse {
    pub prior_batch: Vec<HashMap<usize, f32>>,
    pub value_batch: Vec<f32>,
}


/// ----------------------------------------------------------------------------
/// Monte Carlo Tree Search
/// ----------------------------------------------------------------------------
/// Generic over an Environment type `E` that implements the `EnvTrait` trait.
pub struct MCTS<E: EnvTrait> {
    // Arena style storage for all nodes.
    pub transposition_table: HashMap<NodeId, usize>,
    pub nodes: Vec<Node>,
    pub terminal_value: Value,
    pub batch_size: usize,
    pub server_url: Option<String>,
    pub client: Option<Arc<reqwest::Client>>,

    // Keep the generic type around (no runtime data).
    _env_marker: PhantomData<E>,
}

impl<E: EnvTrait> MCTS<E> {
    pub fn new(
        terminal_value: Value,
        batch_size: usize,
        server_url: Option<String>,
        client: Option<Arc<reqwest::Client>>,
    ) -> Self {
        Self {
            transposition_table: HashMap::new(),
            nodes: Vec::new(),
            terminal_value,
            batch_size,
            server_url,
            client,
            _env_marker: PhantomData,
        }
    }

    /// Run MCTS for a given number of steps from the current environment state.
    /// `env` is borrowed immutably; select_leaf clones it internally as needed.
    pub async fn run<T: Agent + Sync>(&mut self, env: &E, agent: &T, num_steps: usize) -> Node {
        // Ensure root node exists
        let root_hash = self.get_hash(env);
        if !self.node_exists(root_hash) {
            let obs = env.observation();
            let (mut priors, value) = agent.infer(&obs);
            priors = self.normalize_prior(priors, env.valid_actions());
            self.create_node(env, priors, value);
        }

        if env.done() {
            return self.get_node_mut(root_hash).unwrap().clone();
        }

        let batches = num_steps / self.batch_size.max(1);
        for _ in 0..batches {
            // Selection: collect a batch of leaf observations / metadata
            let mut leaf_batch: Vec<Observation> = Vec::with_capacity(self.batch_size);
            let mut path_batch: Vec<Vec<(NodeId, Action)>> = Vec::with_capacity(self.batch_size);
            let mut parent_batch: Vec<(Option<NodeId>, Action, E)> =
                Vec::with_capacity(self.batch_size);
            let mut repeat_batch: Vec<bool> = Vec::with_capacity(self.batch_size);

            for _ in 0..self.batch_size {
                // select_leaf clones the environment internally and returns the reached env
                let (path, parent, action, final_env, repeat) = self.select_leaf(root_hash, env);
                let obs = final_env.observation();
                leaf_batch.push(obs);
                path_batch.push(path);
                parent_batch.push((parent, action, final_env));
                repeat_batch.push(repeat);
            }

            if leaf_batch.is_empty() {
                continue;
            }

            // --- Batched Inference ---
            let (prior_batch, value_batch) = if let Some(_) = self.server_url {
                // Remote inference
                match self.remote_infer(&leaf_batch).await {
                    Ok((priors, values)) => (priors, values),
                    Err(e) => {
                        eprintln!("Remote inference error: {}", e);
                        continue;
                    }
                }
            } else {
                // Local inference
                agent.batch_infer(&leaf_batch)
            };

            // Expansion & Backpropagation
            let n = prior_batch.len().min(value_batch.len()).min(parent_batch.len());
            for i in 0..n {
                let (parent_opt, action, game) = &parent_batch[i];
                if let Some(parent_id) = parent_opt {
                    let priors = prior_batch[i].clone();
                    let value = value_batch[i];
                    // expand and then backpropagate
                    let _leaf = self.expand(*parent_id, *action, game.clone(), priors, value);
                    self.backpropagate(&path_batch[i], repeat_batch[i]);
                }
            }
        }
        self.get_node_mut(root_hash).unwrap().clone()
    }

    /// ------------------------------------------------------------------------
    /// Node functions
    /// ------------------------------------------------------------------------
    /// Create a node from a reference to an Environment `E`.
    pub fn create_node(
        &mut self,
        env: &E,
        priors: Prior,
        value: Value,
    ) -> NodeId {
        let node_id = self.get_hash(env);
        let repr = env.render();
        let node = if env.done() {
            Node::new_terminal(node_id, self.terminal_value, Some(repr))
        } else {
            Node::new(priors, value, node_id, Some(repr))
        };
        self.insert_node(node_id, node);
        node_id
    }

    /// Look up a node by its ID.
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut Node> {
        if let Some(entry) = self.transposition_table.get(&node_id) {
            return self.nodes.get_mut(*entry);
        } else {
            None
        }
    }

    pub fn get_node_immut(&self, node_id: NodeId) -> Option<&Node> {
        if let Some(entry) = self.transposition_table.get(&node_id) {
            return self.nodes.get(*entry);
        } else {
            None
        }
    }

    /// Insert a new node into the transposition table and node arena.
    pub fn insert_node(&mut self, node_id: NodeId, node: Node) {
        let index = self.nodes.len();
        self.nodes.push(node);
        self.transposition_table.insert(node_id, index);
    }

    /// Recompute the cached value of a node based on its children's values.
    pub fn recompute_value(&mut self, node_id: NodeId) {
        // Snapshot parent data immutably to avoid overlapping mutable borrows.
        let (virtual_loss_counts, edge_visits, children, node_value_estimate) = {
            let parent = match self.get_node_immut(node_id) {
                Some(p) => p,
                None => return,
            };
            let vl: usize = parent.virtual_losses.values().copied().sum();
            let ev = parent.edge_visits.clone();
            let ch = parent.children.clone();
            (vl, ev, ch, parent.value_estimate)
        };
        let edge_visit_count: usize = edge_visits.values().copied().sum::<usize>();
        let total_edge_visits = edge_visit_count + virtual_loss_counts;
        if total_edge_visits == 0 {
            if let Some(node) = self.get_node_mut(node_id) {
                node.node_visits = 1;
                node.value = node_value_estimate;
            }
            return;
        }

        // Accumulate weighted child values using immutable borrows for children.
        let mut acc: f32 = 0.0;
        for (a, child_id) in children {
            if let Some(child) = self.get_node_immut(child_id) {
                let ev = edge_visits.get(&a).copied().unwrap_or(0);
                if ev == 0 {
                    continue;
                }
                acc += (ev as f32) * child.value;
            }
        }

        // Single mutable borrow to update the node.
        if let Some(node) = self.get_node_mut(node_id) {
            node.node_visits = 1 + total_edge_visits;
            node.value = (node.value_estimate + acc) / (node.node_visits as f32);
        }
    }

    /// Compute PUCT scores for all actions from this node.
    pub fn puct_scores(&self, node_id: NodeId, c_puct: f32) -> HashMap<Action, f32> {
        let node = match self.get_node_immut(node_id) {
            Some(n) => n,
            None => return HashMap::new(),
        };

        let total_visits: usize = node.edge_visits.values().copied().sum::<usize>();
        let sqrt_total = (total_visits as f32).sqrt() + 1e-8;

        let mut scores = HashMap::new();

        for (&action, &prior) in &node.prior_probs {
            let this_edge_visits = node.edge_visits.get(&action).copied().unwrap_or(0);
            let num_virtual_losses = node.virtual_losses.get(&action).copied().unwrap_or(0);
            let adjusted_visits = this_edge_visits + num_virtual_losses;

            let penalty = node.edge_penalties.get(&action).copied().unwrap_or(0.0);

            // Determine Q value: use child's value if present in table, otherwise use parent_value.
            let q_value = match node.children.get(&action).copied() {
                None => node.value + penalty,
                Some(child_id) => {
                    let child = self.get_node_immut(child_id);
                    child.map(|c| c.value + penalty).unwrap_or(node.value + penalty)
                }
            };

            let u_value = c_puct * prior * (sqrt_total / (1.0 + adjusted_visits as f32));
            scores.insert(action, q_value + u_value);
        }
        scores
    }

    pub fn select_action_puct(&self, node_id: NodeId, c_puct: f32) -> Option<Action> {
        let scores = self.puct_scores(node_id, c_puct);
        scores.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(action, _)| action)
    }

    pub fn select_action(&self, node: Node) -> Option<Action> {
        node.select_action()
    }

    pub fn add_child(&mut self, parent_id: NodeId, action: Action, child_id: NodeId) {
        if let Some(parent) = self.get_node_mut(parent_id) {
            parent.children.insert(action, child_id);
        }
    }

    pub fn node_exists(&self, node_id: NodeId) -> bool {
        self.transposition_table.contains_key(&node_id)
    }

    /// Get a hash of the current environment state.
    pub fn get_hash(&self, env: &E) -> NodeId {
        env.hash_state() as NodeId
    }

    pub fn normalize_prior(&self, priors: Prior, actions: Vec<Action>) -> Prior {
        let mut normalized = HashMap::new();
        let mut total: f32 = 0.0;
        for &a in &actions {
            if let Some(&p) = priors.get(&a) {
                total += p;
            }
        }
        if total == 0.0 {
            let uniform_prob = 1.0 / (actions.len() as f32);
            for &a in &actions {
                normalized.insert(a, uniform_prob);
            }
        } else {
            for &a in &actions {
                if let Some(&p) = priors.get(&a) {
                    normalized.insert(a, p / total);
                }
            }
        }
        normalized
    }

    /// Traverse from root to a leaf. Returns:
    /// (path, parent, action_taken, final_env, repeat_detected)
    /// `final_env` is the environment state after taking actions along the path.
    pub fn select_leaf(
        &mut self,
        root_id: NodeId,
        env: &E,
    ) -> (
        Vec<(NodeId, Action)>,
        Option<NodeId>,
        Action,
        E,
        bool,
    ) {
        let mut path: Vec<(NodeId, Action)> = Vec::new();
        let mut node_id = root_id;
        let mut parent: Option<NodeId> = None;
        let mut action: Action = usize::MAX;
        let mut repeat_detected = false;

        // work on a cloned environment so caller's env is not mutated
        let mut game = env.clone();

        while !game.done() && !repeat_detected {
            if !self.node_exists(node_id) {
                break;
            }

            // detect repeats based on path
            repeat_detected = self.check_state_repeat(game.hash_state() as NodeId, &path);
            if repeat_detected {
                break;
            }

            // choose action via PUCT
            let chosen = match self.select_action_puct(node_id, 1.4) {
                Some(a) => a,
                None => break,
            };
            action = chosen;

            // mark virtual loss quickly (single-threaded counter)
            if let Some(node) = self.get_node_mut(node_id) {
                node.add_virtual_loss(action);
            }

            // advance environment
            game.step(action);

            // record step in path and advance
            path.push((node_id, action));
            parent = Some(node_id);

            // lookup child id from the parent snapshot
            if let Some(parent_node) = self.get_node_immut(node_id) {
                if let Some(cid) = parent_node.children.get(&action).copied() {
                    node_id = cid;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        (path, parent, action, game, repeat_detected)
    }

    pub fn check_state_repeat(&self, state_hash: NodeId, path: &[(NodeId, Action)]) -> bool {
        path.iter().any(|(h, _)| *h == state_hash)
    }

    pub fn expand(
        &mut self,
        parent_id: NodeId,
        action: Action,
        env: E,
        priors: Prior,
        value: Value,
    ) -> NodeId {
        let leaf_id = self.get_hash(&env);
        if !self.node_exists(leaf_id) {
            let normalized_priors = self.normalize_prior(priors, env.valid_actions());
            self.create_node(&env, normalized_priors, value);
        }

        if let Some(parent) = self.get_node_mut(parent_id) {
            if parent.children.contains_key(&action) {
                let existing_id = parent.children.get(&action).copied().unwrap();
                if existing_id != leaf_id {
                    eprintln!("[Mismatched IDs] existing {} != leaf {}", existing_id, leaf_id);
                }
            } else {
                parent.children.insert(action, leaf_id);
                *parent.edge_visits.entry(action).or_insert(0) += 1;
                for (&a, &cid) in &parent.children {
                    if cid == leaf_id && a != action {
                        eprintln!("[Alias detected] actions {} - {}", a, action);
                    }
                }
            }
        }

        leaf_id
    }

    pub fn backpropagate(&mut self, search_path: &[(NodeId, Action)], repeat_detected: bool) {
        if search_path.is_empty() {
            return;
        }

        if repeat_detected {
            if let Some((last_parent_hash, last_action)) = search_path.last() {
                if let Some(last_parent) = self.get_node_mut(*last_parent_hash) {
                    last_parent.apply_penalty(*last_action);
                }
            }
        }

        for &(node_hash, action) in search_path.iter().rev() {
            if let Some(node) = self.get_node_mut(node_hash) {
                node.revert_virtual_loss(action);
                *node.edge_visits.entry(action).or_insert(0) += 1;
            }
            self.recompute_value(node_hash);
        }
    }

    pub async fn remote_infer(
        &self,
        obs_batch: &[Observation],
    ) -> anyhow::Result<(Vec<Prior>, Vec<Value>)> {
        let client = self.client.as_ref().expect("HTTP client not initialized");
        let client = Arc::clone(client);
        let server_url = self.server_url.as_ref().unwrap();

        let req = InferenceRequest { observation_batch: obs_batch.to_vec() };
        let body = to_vec_named(&req)?;

        let t0 = std::time::Instant::now();
        let resp = client
            .post(format!("{}/infer", server_url))
            .header("Content-Type", "application/msgpack")
            .header("Accept", "application/msgpack")
            .body(body)
            .send()
            .await?
            .error_for_status()?;
        let bytes = resp.bytes().await?;
        let parsed: InferenceResponse = from_slice(&bytes)?;

        let dt = t0.elapsed().as_micros() as f64 / 1000.0;
        println!("remote_infer: {:.3}ms (batch={})", dt, obs_batch.len());
        // response.prior_batch already contains Priors keyed by Action (usize),
        // so we can use it directly.
        Ok((parsed.prior_batch, parsed.value_batch))
    }
}

// Note: tests must now construct the MCTS with the concrete environment type:
// let mut mcts: MCTS<Environment> = MCTS::<tilers_core::env::Environment>::new(0.0, 4, None);

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::collections::HashMap;
//     use tilers_core::env::Environment;
// 
//     fn make_priors(pairs: &[(Action, f32)]) -> Prior {
//         let mut m = HashMap::new();
//         for &(a, p) in pairs {
//             m.insert(a, p);
//         }
//         m
//     }
// 
//     fn make_priors_from_vec(actions: Vec<Action>) -> Prior {
//         let mut m = HashMap::new();
//         let prob = 1.0 / (actions.len() as f32);
//         for &a in &actions {
//             m.insert(a, prob);
//         }
//         m
//     }
// 
//     #[test]
//     fn test_mcts_creation() {
//         let mcts: MCTS<Environment> = MCTS::new(1.0, 16, None);
//         assert_eq!(mcts.terminal_value, 1.0);
//         assert_eq!(mcts.batch_size, 16);
//     }
// 
//     #[test]
//     fn test_insert_and_get_node() {
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let priors = make_priors(&[]);
//         let node = Node::new(priors.clone(), 0.42, 1, None);
//         mcts.insert_node(1, node);
//         let node = mcts.get_node_immut(1);
//         assert!(node.is_some());
//         let val = node.unwrap().value;
//         assert!((val - 0.42).abs() < 1e-6);
//     }
// 
//     #[test]
//     fn test_with_node_write_updates() {
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let priors = make_priors(&[]);
//         let node = Node::new(priors, 0.1, 2, None);
//         mcts.insert_node(2, node);
//         // mutate under write helper
//         let node = mcts.get_node_mut(2).unwrap();
//         node.value = 0.5;
//         node.node_visits = 3;
//         let node = mcts.get_node_immut(2).unwrap();
//         assert_eq!(node.node_visits, 3);
//         assert!((node.value - 0.5).abs() < 1e-6);
//     }
// 
//     #[test]
//     fn test_recompute_value_leaf_sets_prior() {
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let priors = make_priors(&[]);
//         let node = Node::new(priors, 0.33, 3, None);
//         mcts.insert_node(3, node);
//         mcts.recompute_value(3);
//         let node = mcts.get_node_immut(3).unwrap();
//         assert!((node.value - 0.33).abs() < 1e-6);
//         assert_eq!(node.node_visits, 1);
//     }
// 
//     #[test]
//     fn test_puct_scores_and_select_puct() {
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         // one action with prior 1.0
//         let priors = make_priors(&[(0usize, 1.0f32)]);
//         let node = Node::new(priors.clone(), 0.5, 4, None);
//         mcts.insert_node(4, node);
// 
//         let scores = mcts.puct_scores(4, 1.0);
//         assert!(scores.contains_key(&0));
//         let score = scores.get(&0).copied().unwrap();
//         // score should be at least parent value (plus tiny exploration term)
//         assert!(score >= 0.5);
// 
//         // select_action_puct should pick the only action
//         let chosen = mcts.select_action_puct(4, 1.0).unwrap();
//         assert_eq!(chosen, 0);
//     }
// 
//     #[test]
//     fn test_select_action() {
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let priors = make_priors(&[(0usize, 0.5f32), (1usize, 0.5f32)]);
//         let mut node = Node::new(priors.clone(), 0.0, 5, None);
//         node.edge_visits.insert(0, 10);
//         node.edge_visits.insert(1, 20);
//         mcts.insert_node(5, node);
//         let node = mcts.get_node_mut(5).unwrap().clone();
//         let chosen = mcts.select_action(node).unwrap();
//         assert_eq!(chosen, 1); // action 1 has more visits, prefer higher prior
//     }
// 
//     #[test]
//     fn test_select_action_is_none_if_no_visits() {
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let priors = make_priors(&[(0usize, 0.2f32), (1usize, 0.8f32)]);
//         let node = Node::new(priors.clone(), 0.0, 6, None);
//         mcts.insert_node(6, node);
//         let node = mcts.get_node_mut(6).unwrap().clone();
//         let chosen = mcts.select_action(node);
//         assert_eq!(chosen, None);
//     }
// 
//     #[test]
//     fn test_add_child_inserts_mapping() {
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let parent_priors = make_priors(&[(0usize, 1.0f32)]);
//         let parent = Node::new(parent_priors, 0.0, 10, None);
//         let child = Node::new(make_priors(&[]), 0.0, 11, None);
//         mcts.insert_node(10, parent);
//         mcts.insert_node(11, child);
// 
//         // add child under action 0
//         mcts.add_child(10, 0, 11);
// 
//         // verify parent's children map contains the mapping action -> child_id
//         let node = mcts.get_node_immut(10).unwrap();
//         let child = mcts.get_node_immut(11).unwrap();
//         assert_eq!(node.children.get(&0), Some(&child.id));
//         assert_eq!(child.id, 11);
//     }
// 
//     #[test]
//     fn test_add_and_revert_virtual_losses() {
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let priors = make_priors(&[]);
//         let node = Node::new(priors, 0.0, 20, None);
//         mcts.insert_node(20, node);
// 
//         // add virtual loss using helper (inserts counter if missing)
//         let node_mut = mcts.get_node_mut(20).unwrap();
//         node_mut.add_virtual_loss(0);
// 
//         let val1 = node_mut.virtual_losses.get(&0).unwrap();
//         assert_eq!(*val1, 1);
// 
//         // revert virtual loss using helper
//         node_mut.revert_virtual_loss(0);
//         let val2 = node_mut.virtual_losses.get(&0).unwrap();
//         assert_eq!(*val2, 0);
//     }
// 
//     #[test]
//     fn test_add_and_revert_penalties() {
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let priors = make_priors(&[]);
//         let node = Node::new(priors, 0.0, 21, None);
//         mcts.insert_node(21, node);
// 
//         // apply penalty (should create entry and decrease by 1.0)
//         let node_mut = mcts.get_node_mut(21).unwrap();
//         node_mut.apply_penalty(0);
// 
//         let p = *node_mut.edge_penalties.get(&0).unwrap();
//         assert!((p + 1.0).abs() < 1e-6);
// 
//         // revert penalty (should add back 1.0)
//         node_mut.revert_penalty(0);
//         let p2 = *node_mut.edge_penalties.get(&0).unwrap();
//         assert!((p2 - 0.0).abs() < 1e-6);
//     }
// 
//     #[test]
//     fn test_check_state_repeat() {
//         let mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let path = vec![(1, 0), (2, 1), (3, 0)];
//         assert!(mcts.check_state_repeat(2, &path));
//         assert!(!mcts.check_state_repeat(4, &path));
//     }
// 
//     #[test]
//     fn test_expand_inserts_node_and_updates_parent() {
//         let mut env = Environment::new(4, 4, 2);
//         let root_hash = env.hash_state() as NodeId;
//         let valid_actions = env.valid_actions();
//         let priors = make_priors_from_vec(valid_actions.clone());
//         let value = -1.0f32;
//         let action = valid_actions[0];
//         env.step(action);
//         let leaf_hash = env.hash_state() as NodeId;
// 
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let parent = Node::new(priors.clone(), value, root_hash, None);
//         mcts.insert_node(root_hash, parent);
// 
//         // first expand should insert the leaf and update parent mapping and visits
//         let returned = mcts.expand(root_hash, action, env, priors.clone(), value);
//         assert_eq!(returned, leaf_hash);
//         assert!(mcts.node_exists(leaf_hash));
//         let root = mcts.get_node_immut(root_hash).unwrap();
//         assert_eq!(root.children.get(&action), Some(&leaf_hash));
//         assert_eq!(*root.edge_visits.get(&action).unwrap(), 1);
//     }
// 
//     #[test]
//     fn test_select_leaf_expected_path() {
//         let qasm = "OPENQASM 2.0;
//             include \"qelib1.inc\";
//             qreg q[10];
//             h q[0];
//             cx q[0],q[1];";
//         // Build a high value path
//         let mut mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let mut env = Environment::from_qasm(qasm, 2, Some(4), Some(4));
//         let mut env_clone = env.clone();
//         // First node
//         let hash_1 = env.hash_state() as NodeId;
//         let valid_actions_1 = env.valid_actions();
//         let priors_1 = make_priors_from_vec(valid_actions_1.clone());
//         let value_1 = 1.0f32;
//         let action_1 = valid_actions_1[0];
//         let node_1 = Node::new(priors_1.clone(), value_1, hash_1, None);
//         mcts.insert_node(hash_1, node_1);
//         // Second node
//         env.step(action_1);
//         let hash_2 = env.hash_state() as NodeId;
//         let valid_actions_2 = env.valid_actions();
//         let priors_2 = make_priors_from_vec(valid_actions_2.clone());
//         let value_2 = 2.0f32;
//         let action_2 = valid_actions_2[0];
//         mcts.expand(hash_1, action_1, env.clone(), priors_2.clone(), value_2);
//         // Third node
//         env.step(action_2);
//         let hash_3 = env.hash_state() as NodeId;
//         let valid_actions_3 = env.valid_actions();
//         let priors_3 = make_priors_from_vec(valid_actions_3.clone());
//         let value_3 = 3.0f32;
//         mcts.expand(hash_2, action_2, env.clone(), priors_3.clone(), value_3);
// 
//         let (path, _parent_opt, _action_taken, _obs, _) = mcts.select_leaf(hash_1, &mut env_clone);
//         assert_eq!(path.len(), 3);
//         let (hash_1_ret, action_1_ret) = path[0];
//         let (hash_2_ret, action_2_ret) = path[1];
//         let (hash_3_ret, _) = path[2];
// 
//         assert_eq!(hash_1_ret, hash_1);
//         assert_eq!(action_1_ret, action_1);
//         assert_eq!(hash_2_ret, hash_2);
//         assert_eq!(action_2_ret, action_2);
//         assert_eq!(hash_3_ret, hash_3);
//     }
// 
//     #[test]
//     fn test_normalize_prior() {
//         let mcts: MCTS<Environment> = MCTS::new(0.0, 4, None);
//         let priors = make_priors(&[(0usize, 0.2f32), (1usize, 0.3f32), (2usize, 0.5f32)]);
//         let actions = vec![0, 1];
//         let normalized = mcts.normalize_prior(priors, actions.clone());
//         let total: f32 = normalized.values().sum();
//         assert!((total - 1.0).abs() < 1e-6);
//         assert!((normalized.get(&0).unwrap() - 0.4).abs() < 1e-6);
//         assert!((normalized.get(&1).unwrap() - 0.6).abs() < 1e-6);
//     }
// 
//     // #[test]
//     // fn simple_mcts_run_with_dummy_agent() {
//     //     use crate::agent::DummyAgent;
// 
//     //     // A tiny QASM-like program (adapt if your Environment expects a different format)
//     //     let qasm = "
//     //         OPENQASM 2.0;
//     //         include \"qelib1.inc\";
//     //         qreg q[14];
//     //         cx q[3],q[6];
//     //         t q[2];
//     //     ";
// 
//     //     // Build the environment. from_qasm takes Option<usize> for height/width.
//     //     let env = Environment::from_qasm(qasm, 2, Some(4), Some(4));
// 
//     //     // Create MCTS and a trivial agent. Adjust terminal value / batch size to taste.
//     //     let mut mcts: MCTS<Environment> = MCTS::new(0.0_f32, 4usize, None);
//     //     let agent = DummyAgent::new(env.num_actions()); // 4 actions
// 
//     //     // Run MCTS for a small number of steps.
//     //     let root_node = mcts.run(&env, &agent, 10000usize);
// 
//     //     // Ensure the root node exists in the transposition table after running.
//     //     assert!(mcts.node_exists(root_node.id), "root node should be present");
// 
//     //     let num_nodes = mcts.nodes.len();
//     //     assert!(num_nodes > 1, "should have expanded some nodes");
//     // }
// }
// 