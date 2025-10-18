use crate::node::Node;
use crate::enums::{Action, NodeId};
use crate::agent::Agent;
use tilers_core::env::Environment;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::sync::atomic::Ordering;


/// Core Monte Carlo Tree Search engine.
/// TODO: Remove repr from Node for faster performance
pub struct MCTS {
    // Arena style storage for all nodes.
    // TODO: Introduce sharding for nodes storage
    pub transposition_table: DashMap<NodeId, usize>,
    pub nodes: Mutex<Vec<Arc<RwLock<Node>>>>,
    pub terminal_value: f32,
    pub batch_size: usize,
}

impl MCTS {
    pub fn new(terminal_value: f32, batch_size: usize) -> Self {
        Self {
            transposition_table: DashMap::new(),
            nodes: Mutex::new(Vec::new()),
            terminal_value,
            batch_size,
        }
    }
    /// ------------------------------------------------------------------------
    /// MCTS run function
    /// ------------------------------------------------------------------------
    /// Run MCTS for a given number of steps from the current environment state.
    /// Returns the NodeId of the root node after search. An action can be selected
    /// from this node using select_action.
    /// This function runs single-threaded MCTS but batches inference requests.
    pub fn run<T: Agent>(
        &self,
        env: &Environment,
        agent: &T,
        num_steps: usize,
    ) -> NodeId {
        // Ensure root node exists
        let root_hash = self.get_hash(env);
        if !self.node_exists(root_hash) {
            // build observation and ask agent for priors/value to create the root
            // TODO: Consistent observation representation
            //let obs = env.observation();
            let obs = vec![0.0];
            let (mut priors, value) = agent.infer(&obs);
            priors = self.normalize_prior(priors, env.valid_actions());
            self.create_node(env, priors, value);
        }

        if env.done() {
            return root_hash;
        }

        let batches = num_steps / self.batch_size.max(1);
        for _ in 0..batches {
            // Selection: collect a batch of leaf observations / metadata
            let mut leaf_batch: Vec<Vec<f32>> = Vec::with_capacity(self.batch_size);
            let mut path_batch: Vec<Vec<(NodeId, Action)>> = Vec::with_capacity(self.batch_size);
            let mut parent_batch: Vec<(Option<NodeId>, Action, Environment)> =
                Vec::with_capacity(self.batch_size);
            let mut repeat_batch: Vec<bool> = Vec::with_capacity(self.batch_size);

            for _ in 0..self.batch_size {
                let mut game = env.clone();
                let (path, parent, action, obs, repeat) = self.select_leaf(root_hash, &mut game);
                leaf_batch.push(obs);
                path_batch.push(path);
                parent_batch.push((parent, action, game));
                repeat_batch.push(repeat);
            }

            if leaf_batch.is_empty() {
                continue;
            }

            // Batched inference via agent
            let (prior_batch, value_batch) = agent.batch_infer(&leaf_batch);

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

        root_hash
    }

    /// ------------------------------------------------------------------------
    /// Node functions
    /// ------------------------------------------------------------------------
    /// Create a node
    pub fn create_node(
        &self,
        env: &Environment,
        priors: HashMap<Action, f32>,
        value: f32,
    ) -> NodeId {
        let node_id = self.get_hash(env);
        let repr = env.render();
        let node = if env.done() {
            Node::new_terminal(node_id, self.terminal_value, Some(repr))
        } else  {
            Node::new(priors, value, node_id, Some(repr))
        };
        self.insert_node(node);
        node_id
    }

    /// Look up a node by its ID.
    pub fn get_node_arc(&self, node_id: NodeId) -> Option<Arc<RwLock<Node>>> {
        let node_arc = if let Some(entry) = self.transposition_table.get(&node_id) {
            let nodes = self.nodes.lock().unwrap();
            let index = *entry.value();
            nodes.get(index).cloned()
        } else {
            None
        };
        node_arc
    }

    /// Insert a new node into the transposition table and node arena.
    pub fn insert_node(&self, node: Node) -> Arc<RwLock<Node>> {
        let mut nodes = self.nodes.lock().unwrap();
        let node_id = node.id;
        let arc_node = Arc::new(RwLock::new(node));
        let index = nodes.len();
        nodes.push(arc_node.clone());
        self.transposition_table.insert(node_id, index);
        arc_node
    }

    /// Run a closure with a read guard to the node.
    /// Allows safe concurrent read access.
    pub fn with_node_read<F, R>(&self, node_id: NodeId, f: F) -> Option<R>
    where
        F: FnOnce(&Node) -> R,
    {
        let arc = self.get_node_arc(node_id)?;
        let guard = arc.read().unwrap();
        Some(f(&*guard))
    }

    /// Run a closure with a write guard to the node.
    /// Allows safe exclusive write access.
    pub fn with_node_write<F, R>(&self, node_id: NodeId, f: F) -> Option<R>
    where
        F: FnOnce(&mut Node) -> R,
    {
        let arc = self.get_node_arc(node_id)?;
        let mut guard = arc.write().unwrap();
        Some(f(&mut *guard))
    }

    pub fn node_snapshot(&self, node_id: NodeId) -> Option<(
        HashMap<Action, f32>,
        HashMap<Action, usize>,
        HashMap<Action, usize>,
        HashMap<Action, f32>,
        HashMap<Action, NodeId>,
        f32,
    )> {
        let snapshot = self.with_node_read(node_id, |n| {
            (
                n.prior_probs.clone(),
                n.edge_visits.clone(),
                n.virtual_loss_copy(),
                n.edge_penalties.clone(),
                n.children.clone(),
                n.value,
            )
        });
        snapshot
    }

    /// Recompute the cached value of a node based on its children's values.
    pub fn recompute_value(&self, node_id: NodeId) -> f32 {

        // Get read only copies of edge_visits and virtual losses
        let virtual_losses: HashMap<Action, usize> = self
            .with_node_read(node_id, |n| n.virtual_loss_copy())
            .unwrap_or_default();
        let edge_visits = self
            .with_node_read(node_id, |n| n.edge_visits.clone())
            .unwrap_or_default();
        let virtual_loss_counts: usize = virtual_losses.values().copied().sum();
        let edge_visit_count = edge_visits.values().copied().sum::<usize>();

        // Read-only snapshot of total edge visits
        let total_edge_visits = edge_visit_count + virtual_loss_counts;
        if total_edge_visits == 0 {
            return self
                .with_node_write(node_id, |node| {
                    node.node_visits = 1;
                    node.value = node.value_estimate;
                    node.value
                })
                .unwrap_or(0.0);
        }

        // Get read only copies of children and their values
        let children: Vec<(Action, NodeId)> = self
            .with_node_read(node_id, |n| n.children.iter().map(|(&a, &cid)| (a, cid)).collect())
            .unwrap_or_default();
        let child_values: HashMap<NodeId, f32> = children
            .iter()
            .filter_map(|&(_, cid)| {
                self.with_node_read(cid, |c| c.value).map(|v| (cid, v))
            })
            .collect();

        // accumulate weighted child values (read-only on children)
        let mut acc: f32 = 0.0;
        for (a, child_id) in children {
            let edge_visit_count = edge_visits.get(&a).copied().unwrap_or(0);
            if edge_visit_count == 0 {
                continue;
            }
            let child_value = child_values.get(&child_id).copied().unwrap_or(0.0);
            acc += (edge_visit_count as f32) * child_value;
        }

        // final update under write lock
        self.with_node_write(node_id, |node| {
            node.node_visits = 1 + total_edge_visits;
            node.value = (node.value_estimate + acc) / (node.node_visits as f32);
            node.value
        })
        .unwrap_or(0.0)
    }

    /// Compute PUCT scores for all actions from this node.
    pub fn puct_scores(&self, node_id: NodeId, c_puct: f32) -> HashMap<Action, f32> {

        // Snapshot parent fields under a read lock.
        let snapshot_opt = self.node_snapshot(node_id);
        let (prior_probs, edge_visits, virtual_losses, edge_penalties, children, parent_value) =
            match snapshot_opt {
                Some(s) => s,
                None => return HashMap::new(),
            };

        let total_visits: usize = edge_visits.values().copied().sum::<usize>();
        let sqrt_total = (total_visits as f32).sqrt() + 1e-8;

        let mut scores = HashMap::new();

        for (&action, &prior) in &prior_probs {
            let this_edge_visits = edge_visits.get(&action).copied().unwrap_or(0);
            let num_virtual_losses = virtual_losses.get(&action).copied().unwrap_or(0);
            let adjusted_visits = this_edge_visits + num_virtual_losses;

            let penalty = edge_penalties.get(&action).copied().unwrap_or(0.0);

            // Determine Q value: use child's value if present in table, otherwise use parent_value.
            let q_value = match children.get(&action).copied() {
                None => parent_value + penalty,
                Some(child_id) => {
                    // read child's value with helper (returns Option<f32>)
                    self.with_node_read(child_id, |c| c.value).unwrap_or(parent_value) + penalty
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

    pub fn select_action(&self, node_id: NodeId) -> Option<Action> {
        self.with_node_read(node_id, |n| n.select_action()).flatten()
    }

    pub fn add_child(&self, parent_id: NodeId, action: Action, child_id: NodeId) {
        self.with_node_write(parent_id, |parent| {
            parent.children.insert(action, child_id);
        });
    }

    pub fn node_exists(&self, node_id: NodeId) -> bool {
        self.transposition_table.contains_key(&node_id)
    }

    // ------------------------------------------------------------------------
    // Environment interaction functions
    // ------------------------------------------------------------------------
    /// Get a hash of the current environment state.
    pub fn get_hash(&self, env: &Environment) -> NodeId {
        // For simplicity, we assume the environment provides a method to get a unique hash.
        env.hash_state() as NodeId
    }

    /// Normalize the prior probabilities for a set of actions.
    pub fn normalize_prior(&self, priors: HashMap<Action, f32>, actions: Vec<Action>) -> HashMap<Action, f32> {
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

    // ------------------------------------------------------------------------
    // MCTS search functions
    // ------------------------------------------------------------------------
    /// Traverse from root to a leaf. Returns the following:
    ///   - path: Vec of (parent, action_to_child) pairs taken during traversal
    ///   - parent: Option<NodeId> of the leaf's parent (None if root is leaf)
    ///   - action_taken: Action taken to reach the leaf from its parent
    ///   - observation: observation vector at the leaf
    ///   - repeat_detected: whether a repeat state was detected during traversal
    pub fn select_leaf(
        &self,
        root_id: NodeId,
        env: &mut Environment,
    ) -> (
        Vec<(NodeId, Action)>,
        Option<NodeId>,
        Action,
        Vec<f32>,
        bool,
    ) {
        let mut path: Vec<(NodeId, Action)> = Vec::new();
        let mut node_id = root_id;
        let mut parent: Option<NodeId> = None;
        let mut action: Action = usize::MAX;
        let mut repeat_detected = false;

        while !env.done() && !repeat_detected {
            // ensure node still exists
            if self.get_node_arc(node_id).is_none() {
                break;
            }

            // current state hash from environment (assumed to be NodeId-compatible)
            let node_hash = self.get_hash(env);

            // detect repeats based on path
            repeat_detected = self.check_state_repeat(node_hash, &path);
            if repeat_detected {
                break;
            }

            // choose action via PUCT
            let chosen = match self.select_action_puct(node_id, 1.4) {
                Some(a) => a,
                None => break,
            };
            action = chosen;

            // mark virtual loss quickly using atomic counter if present (no write lock needed)
            let a_copy = action;
            let _ = self.with_node_read(node_id, |n| {
                if let Some(counter) = n.virtual_losses.get(&a_copy) {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            });

            // advance environment
            env.step(action);

            // record step in path and advance
            path.push((node_hash, action));
            parent = Some(node_id);

            // lookup child id (snapshot under read lock)
            let child_opt = self
                .with_node_read(node_id, |n| n.children.get(&action).copied())
                .unwrap_or(None);

            if let Some(cid) = child_opt {
                node_id = cid;
            } else {
                // no child allocated yet -> we've reached a leaf
                break;
            }
        }

        // get observation (even if terminal or repeat) so return type is consistent
        // TODO: Need consistent representation of observations
        // let obs = env.observation();
        let obs = vec![0.0];

        (path, parent, action, obs, repeat_detected)
    }

    /// Check if the given state hash has already been encountered in the path.
    pub fn check_state_repeat(&self, state_hash: NodeId, path: &[(NodeId, Action)]) -> bool {
        path.iter().any(|(h, _)| *h == state_hash)
    }

    // ------------------------------------------------------------------------
    // MCTS Expansion and Backpropagation
    // ------------------------------------------------------------------------
    /// Expand the tree by adding a new node for the given environment state. This function
    /// requires:
    ///   - parent_id: the node id of the parent from which we are expanding
    ///   - action: the action taken from the parent to reach this state
    ///   - env: the environment in the new state
    ///   - priors: prior probabilities for actions of the new node
    ///   - value: the value estimate for the new node
    pub fn expand(
        &self,
        parent_id: NodeId,
        action: Action,
        env: Environment,
        priors: HashMap<Action, f32>,
        value: f32,
    ) -> NodeId {
        // compute leaf id from environment state
        let leaf_id = self.get_hash(&env);
        // Reuse existing node if present, otherwise create and insert a new node
        if !self.node_exists(leaf_id) {
            let normalized_priors = self.normalize_prior(priors, env.valid_actions());
            self.create_node(&env, normalized_priors, value);
        }

        // Ensure parent's children map contains the mapping action -> leaf_id.
        // Also increment the parent's edge visit counter for this action.
        let leaf_id_copy = leaf_id;
        let _ = self.with_node_write(parent_id, |parent| {
            // insert returns the previous value (if any)
            match parent.children.insert(action, leaf_id_copy) {
                None => {
                    for (&a, &cid) in &parent.children {
                        if cid == leaf_id_copy && a != action {
                            eprintln!("[Alias detected] actions {} - {}", a, action);
                            eprintln!("{}-{} {}-{}", a, cid, action, leaf_id_copy);
                            eprintln!("actions {:?}", parent.children.keys().cloned().collect::<Vec<_>>());
                        }
                    }
                    // was not present -> increment visits
                    *parent.edge_visits.entry(action).or_insert(0) += 1;
                }
                Some(existing_id) => {
                    // DEBUG: Check for action mapping to different node
                    if existing_id != leaf_id_copy {
                        eprintln!("[Mismatched IDs] existing {} != leaf {}", existing_id, leaf_id_copy);
                    }
                    // else: same mapping already present — no-op
                }
            }
        });
        leaf_id
    }

    /// Backpropagate a leaf value up the search path.
    /// Takes as input:
    ///  - search_path: Vec of (parent_node_id, action_taken) pairs from root to a node
    ///  - repeat_detected: whether a repeat state was detected during traversal
    pub fn backpropagate(&self, search_path: &[(NodeId, Action)], repeat_detected: bool) {
        if search_path.is_empty() {
            return;
        }

        // If a repeat was detected, apply a penalty to the last parent/action.
        if repeat_detected {
            if let Some((last_parent_hash, last_action)) = search_path.last() {
                let _ = self.with_node_write(*last_parent_hash, |n| {
                    n.apply_penalty(*last_action);
                });
            }
        }

        // Walk the path in reverse (from leaf's parent back to root).
        for &(node_hash, action) in search_path.iter().rev() {
            if !self.node_exists(node_hash) {
                eprintln!("Node {} not found during backpropagation", node_hash);
            }
            // Revert the virtual loss and increment edge visits under a write lock,
            // then recompute the cached value without holding that write lock.
            let _ = self.with_node_write(node_hash, |node| {
                // Revert virtual loss
                node.revert_virtual_loss(action);
                // increment edge visits
                *node.edge_visits.entry(action).or_insert(0) += 1;
            });
            // Recompute node value via the MCTS helper (acquires its own locks).
            let _ = self.recompute_value(node_hash);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;

    fn make_priors(pairs: &[(Action, f32)]) -> HashMap<Action, f32> {
        let mut m = HashMap::new();
        for &(a, p) in pairs {
            m.insert(a, p);
        }
        m
    }

    fn make_priors_from_vec(actions: Vec<Action>) -> HashMap<Action, f32> {
        let mut m = HashMap::new();
        let prob = 1.0 / (actions.len() as f32);
        for &a in &actions {
            m.insert(a, prob);
        }
        m
    }

    #[test]
    fn test_mcts_creation() {
        let mcts = MCTS::new(1.0, 16);
        assert_eq!(mcts.terminal_value, 1.0);
        assert_eq!(mcts.batch_size, 16);
    }

    #[test]
    fn test_insert_and_get_node() {
        let mcts = MCTS::new(0.0, 4);
        let priors = make_priors(&[]);
        let node = Node::new(priors.clone(), 0.42, 1, None);
        let _arc = mcts.insert_node(node);
        let arc_opt = mcts.get_node_arc(1);
        assert!(arc_opt.is_some());
        let val = mcts.with_node_read(1, |n| n.value).unwrap();
        assert!((val - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_with_node_write_updates() {
        let mcts = MCTS::new(0.0, 4);
        let priors = make_priors(&[]);
        let node = Node::new(priors, 0.1, 2, None);
        mcts.insert_node(node);
        // mutate under write helper
        mcts.with_node_write(2, |n| {
            n.value = 0.9;
            n.node_visits = 5;
        })
        .expect("write should succeed");
        let (visits, val) = mcts.with_node_read(2, |n| (n.node_visits, n.value)).unwrap();
        assert_eq!(visits, 5);
        assert!((val - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_recompute_value_leaf_sets_prior() {
        let mcts = MCTS::new(0.0, 4);
        let priors = make_priors(&[]);
        let node = Node::new(priors, 0.33, 3, None);
        mcts.insert_node(node);
        let v = mcts.recompute_value(3);
        assert!((v - 0.33).abs() < 1e-6);
        mcts.with_node_read(3, |n| {
            assert_eq!(n.node_visits, 1);
            assert!((n.value - 0.33).abs() < 1e-6);
        });
    }

    #[test]
    fn test_puct_scores_and_select_puct() {
        let mcts = MCTS::new(0.0, 4);
        // one action with prior 1.0
        let priors = make_priors(&[(0usize, 1.0f32)]);
        let node = Node::new(priors.clone(), 0.5, 4, None);
        mcts.insert_node(node);

        let scores = mcts.puct_scores(4, 1.0);
        assert!(scores.contains_key(&0));
        let score = scores.get(&0).copied().unwrap();
        // score should be at least parent value (plus tiny exploration term)
        assert!(score >= 0.5);

        // select_action_puct should pick the only action
        let chosen = mcts.select_action_puct(4, 1.0).unwrap();
        assert_eq!(chosen, 0);
    }

    #[test]
    fn test_select_action() {
        let mcts = MCTS::new(0.0, 4);
        let priors = make_priors(&[(0usize, 0.5f32), (1usize, 0.5f32)]);
        let mut node = Node::new(priors.clone(), 0.0, 5, None);
        node.edge_visits.insert(0, 10);
        node.edge_visits.insert(1, 20);
        mcts.insert_node(node);
        let chosen = mcts.select_action(5).unwrap();
        assert_eq!(chosen, 1); // action 1 has more visits, prefer higher prior
    }

    #[test]
    fn test_select_action_is_none_if_no_visits() {
        let mcts = MCTS::new(0.0, 4);
        let priors = make_priors(&[(0usize, 0.2f32), (1usize, 0.8f32)]);
        let node = Node::new(priors.clone(), 0.0, 6, None);
        mcts.insert_node(node);
        let chosen = mcts.select_action(6);
        assert_eq!(chosen, None);
    }

    #[test]
    fn test_add_child_inserts_mapping() {
        let mcts = MCTS::new(0.0, 4);
        let parent_priors = make_priors(&[(0usize, 1.0f32)]);
        let parent = Node::new(parent_priors, 0.0, 10, None);
        let child = Node::new(make_priors(&[]), 0.0, 11, None);
        mcts.insert_node(parent);
        mcts.insert_node(child);

        // add child under action 0
        mcts.add_child(10, 0, 11);

        // verify parent's children map contains the mapping action -> child_id
        mcts.with_node_read(10, |n| {
            assert_eq!(n.children.get(&0), Some(&11));
        });
    }

    #[test]
    fn test_add_and_revert_virtual_losses() {
        let mcts = MCTS::new(0.0, 4);
        let priors = make_priors(&[]);
        let node = Node::new(priors, 0.0, 20, None);
        mcts.insert_node(node);

        // add virtual loss using helper (inserts counter if missing)
        mcts.with_node_write(20, |n| {
            n.add_virtual_loss(0);
        })
        .expect("add virtual loss");

        let val = mcts
            .with_node_read(20, |n| n.virtual_losses.get(&0).unwrap().load(Ordering::SeqCst))
            .unwrap();
        assert_eq!(val, 1);

        // revert virtual loss using helper
        mcts.with_node_write(20, |n| {
            n.revert_virtual_loss(0);
        })
        .expect("revert virtual loss");

        let val2 = mcts
            .with_node_read(20, |n| n.virtual_losses.get(&0).unwrap().load(Ordering::SeqCst))
            .unwrap();
        assert_eq!(val2, 0);
    }

    #[test]
    fn test_add_and_revert_penalties() {
        let mcts = MCTS::new(0.0, 4);
        let priors = make_priors(&[]);
        let node = Node::new(priors, 0.0, 21, None);
        mcts.insert_node(node);

        // apply penalty (should create entry and decrease by 1.0)
        mcts.with_node_write(21, |n| {
            n.apply_penalty(0);
        })
        .expect("apply penalty");

        let p = mcts.with_node_read(21, |n| *n.edge_penalties.get(&0).unwrap()).unwrap();
        assert!((p + 1.0).abs() < 1e-6);

        // revert penalty (should add back 1.0)
        mcts.with_node_write(21, |n| {
            n.revert_penalty(0);
        })
        .expect("revert penalty");

        let p2 = mcts.with_node_read(21, |n| *n.edge_penalties.get(&0).unwrap()).unwrap();
        assert!((p2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_check_state_repeat() {
        let mcts = MCTS::new(0.0, 4);
        let path = vec![(1, 0), (2, 1), (3, 0)];
        assert!(mcts.check_state_repeat(2, &path));
        assert!(!mcts.check_state_repeat(4, &path));
    }

    #[test]
    fn test_expand_inserts_node_and_updates_parent() {
        let mut env = Environment::new(4, 4, 2);
        let root_hash = env.hash_state() as NodeId;
        let valid_actions = env.valid_actions();
        let priors = make_priors_from_vec(valid_actions.clone());
        let value = -1.0f32;
        let action = valid_actions[0];
        env.step(action);
        let leaf_hash = env.hash_state() as NodeId;

        let mcts = MCTS::new(0.0, 4);
        let parent = Node::new(priors.clone(), value, root_hash, None);
        mcts.insert_node(parent);

        // first expand should insert the leaf and update parent mapping and visits
        let returned = mcts.expand(root_hash, action, env, priors.clone(), value);
        assert_eq!(returned, leaf_hash);
        assert!(mcts.node_exists(leaf_hash));
        mcts.with_node_read(root_hash, |n| {
            assert_eq!(n.children.get(&action), Some(&leaf_hash));
            assert_eq!(*n.edge_visits.get(&action).unwrap(), 1);
        });
    }

    #[test]
    fn test_select_leaf_expected_path() {
        let qasm = "OPENQASM 2.0;
            include \"qelib1.inc\";
            qreg q[10];
            h q[0];
            cx q[0],q[1];";
        // Build a high value path
        let mcts = MCTS::new(0.0, 4);
        let mut env = Environment::from_qasm(qasm, 2, Some(4), Some(4));
        let mut env_clone = env.clone();
        // First node
        let hash_1 = env.hash_state() as NodeId;
        let valid_actions_1 = env.valid_actions();
        let priors_1 = make_priors_from_vec(valid_actions_1.clone());
        let value_1 = 1.0f32;
        let action_1 = valid_actions_1[0];
        let node_1 = Node::new(priors_1.clone(), value_1, hash_1, None);
        mcts.insert_node(node_1);
        // Second node
        env.step(action_1);
        let hash_2 = env.hash_state() as NodeId;
        let valid_actions_2 = env.valid_actions();
        let priors_2 = make_priors_from_vec(valid_actions_2.clone());
        let value_2 = 2.0f32;
        let action_2 = valid_actions_2[0];
        mcts.expand(hash_1, action_1, env.clone(), priors_2.clone(), value_2);
        // Third node
        env.step(action_2);
        let hash_3 = env.hash_state() as NodeId;
        let valid_actions_3 = env.valid_actions();
        let priors_3 = make_priors_from_vec(valid_actions_3.clone());
        let value_3 = 3.0f32;
        mcts.expand(hash_2, action_2, env.clone(), priors_3.clone(), value_3);

        let (path, _parent_opt, _action_taken, _obs, _) = mcts.select_leaf(hash_1, &mut env_clone);
        assert_eq!(path.len(), 3);
        let (hash_1_ret, action_1_ret) = path[0];
        let (hash_2_ret, action_2_ret) = path[1];
        let (hash_3_ret, _) = path[2];

        assert_eq!(hash_1_ret, hash_1);
        assert_eq!(action_1_ret, action_1);
        assert_eq!(hash_2_ret, hash_2);
        assert_eq!(action_2_ret, action_2);
        assert_eq!(hash_3_ret, hash_3);
    }

    #[test]
    fn test_normalize_prior() {
        let mcts = MCTS::new(0.0, 4);
        let priors = make_priors(&[(0usize, 0.2f32), (1usize, 0.3f32), (2usize, 0.5f32)]);
        let actions = vec![0, 1];
        let normalized = mcts.normalize_prior(priors, actions.clone());
        let total: f32 = normalized.values().sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!((normalized.get(&0).unwrap() - 0.4).abs() < 1e-6);
        assert!((normalized.get(&1).unwrap() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn simple_mcts_run_with_dummy_agent() {
        use crate::agent::DummyAgent;

        // A tiny QASM-like program (adapt if your Environment expects a different format)
        let qasm = "
            OPENQASM 2.0;
            include \"qelib1.inc\";
            qreg q[14];
            cx q[3],q[6];
            t q[2];
        ";

        // Build the environment. from_qasm takes Option<usize> for height/width.
        let env = Environment::from_qasm(qasm, 2, Some(4), Some(4));

        // Create MCTS and a trivial agent. Adjust terminal value / batch size to taste.
        let mcts = MCTS::new(0.0_f32, 4usize);
        let agent = DummyAgent::new(env.num_actions()); // 4 actions

        // Run MCTS for a small number of steps.
        let root_id = mcts.run(&env, &agent, 10000usize);

        // Ensure the root node exists in the transposition table after running.
        assert!(mcts.node_exists(root_id), "root node should be present");

        let num_nodes = {
            let nodes = mcts.nodes.lock().unwrap();
            nodes.len()
        };
        assert!(num_nodes > 1, "should have expanded some nodes");
    }
}
