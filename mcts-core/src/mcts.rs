use crate::node::{Node, NodeId};
use crate::environment::Environment;
use crate::inference::InferenceClient;
use std::collections::HashMap;


/// ----------------------------------------------------------------------------
/// Monte Carlo Tree Search
/// ----------------------------------------------------------------------------
pub struct MCTS<E: Environment> {
    // Track the current root of the search tree
    pub root_id: Option<NodeId>,
    // Arena style storage for all nodes.
    pub transposition_table: HashMap<NodeId, usize>,
    pub nodes: Vec<Node<E::Act>>,
    pub terminal_value: f32,
    pub batch_size: usize,
}

impl<E: Environment> MCTS<E> {
    pub fn new(terminal_value: f32, batch_size: usize) -> Self {
        Self {
            root_id: None,
            transposition_table: HashMap::new(),
            nodes: Vec::new(),
            terminal_value,
            batch_size,
        }
    }

    pub fn default() -> Self { Self::new(1.0, 8) }

    /// Run MCTS for a given number of steps from the current environment state. `env` is borrowed
    /// immutably; select_leaf clones it internally as needed. The `client` is an inference client
    /// that provides value and prior estimates for leaf nodes.
    pub fn run(
        &mut self,
        env: &E,
        client: &dyn InferenceClient<E>,
        num_steps: usize
    ) -> Node<E::Act> {
        // Set the current root for the search session
        let root_hash = self.get_hash(env);
        self.root_id = Some(root_hash);
        // Check if the current root node exists
        if !self.node_exists(root_hash) {
            let obs = env.observation();
            let (p, v) = self.blocking_infer(&vec![obs], client);
            let value = v[0];
            let mut priors = p[0].clone();
            priors = self.normalize_prior(priors, &env.valid_actions());
            self.create_node(env, priors, value);
        }

        if env.done() { return self.get_node_mut(root_hash).unwrap().clone(); }

        let num_batches = num_steps / self.batch_size.max(1);
        for _ in 0..num_batches {
            // Selection: collect a batch of leaf observations / metadata
            let mut leaf_batch: Vec<E::Obs> = Vec::with_capacity(self.batch_size);
            let mut path_batch: Vec<Vec<(NodeId, E::Act)>> = Vec::with_capacity(self.batch_size);
            let mut parent_batch: Vec<(Option<NodeId>, E::Act, E)> =Vec::with_capacity(self.batch_size);
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

            if leaf_batch.is_empty() { continue; }

            // --- Batched Inference ---
            let (prior_batch, value_batch) = self.blocking_infer(&leaf_batch, client);

            // Expansion & Backpropagation
            let n = prior_batch.len().min(value_batch.len()).min(parent_batch.len());
            for i in 0..n {
                let (parent_opt, action, game) = &parent_batch[i];
                if let Some(parent_id) = parent_opt {
                    if repeat_batch[i] {
                        // Just backpropagate with a penalty for repeats, no expansion
                        self.backpropagate(&path_batch[i], true);
                    } else {
                        let priors = prior_batch[i].clone();
                        let value = value_batch[i];
                        // Expand and then backpropagate
                        let _leaf = self.expand(*parent_id, *action, game.clone(), priors, value);
                        self.backpropagate(&path_batch[i], false);
                    }
                }
            }
        }
        self.get_node_mut(root_hash).unwrap().clone()
    }

    /// Advances the root of the tree to the child corresponding to the given action.
    /// This preserves the entire subtree of that child for the next search, allowing for
    /// MCTS to think more deeply.
    /// 
    /// If the child node does not exist in the tree, the tree is effectively reset by
    /// setting the root to `None`.
    pub fn advance_root(&mut self, action: E::Act) {
        let old_root_id = match self.root_id {
            Some(id) => id,
            None => return, // no root to advance from
        };
        // Immutable borrow to get the children map
        let old_root_node = match self.get_node_immut(old_root_id) {
            Some(node) => node,
            None => {
                // A bug or inconsistent state. The root ID should always be valid.
                eprintln!(
                    "[MCTS] Root ID {} not found in transposition table while advancing root.",
                    old_root_id
                );
                self.root_id = None;
                return;
            }
        };
        if let Some(&new_root_id) = old_root_node.children.get(&action) {
            self.root_id = Some(new_root_id);
            // TODO: Implement pruning here to conserve memory for long games.
            // The goal is to remove all nodes that are no longer reachable from the `new_root_id`.
            // This is a non-trivial garbage collection process because of the arena storage.
            //
            // HIGH-LEVEL ALGORITHM:
            // 1. Perform a graph traversal (like DFS or BFS) starting from `new_root_id`.
            //    Collect all reachable `NodeId`s into a `HashSet` for fast lookups.
            //
            // 2. Create a new `nodes_after_pruning: Vec<Node>` and a new
            //    `table_after_pruning: HashMap<NodeId, usize>`.
            //
            // 3. Iterate through `self.nodes`. If a node's `id` is in the reachable set,
            //    clone it and push it into `nodes_after_pruning`.
            //
            // 4. As you add a node, populate `table_after_pruning`, mapping the `NodeId`
            //    to its new index in the `nodes_after_pruning` vector.
            //
            // 5. Finally, replace the old data structures with the pruned ones:
            //    `self.nodes = nodes_after_pruning;`
            //    `self.transposition_table = table_after_pruning;`
            //
            // This process rebuilds the arena with only the necessary nodes, keeping all
            // indices in the transposition table valid relative to the new `nodes` vector.
        } else {
            // The action does not lead to a known child. This might happen if the search is
            // shallow and the node was never fully expanded.
            self.root_id = None; 
        }
    }

    /// ------------------------------------------------------------------------
    /// Node functions
    /// ------------------------------------------------------------------------
    /// Create a node from a reference to an Environment `E`.
    pub fn create_node(
        &mut self,
        env: &E,
        priors: HashMap<E::Act, f32>,
        value: f32,
    ) -> NodeId {
        let node_id = self.get_hash(env);
        // let repr = Some(env.render());
        let repr = None;
        let node = if env.done() {
            Node::new_terminal(node_id, self.terminal_value, repr)
        } else {
            Node::new(priors, value.clamp(-1.0, 1.0), node_id, repr)
        };
        self.insert_node(node_id, node);
        node_id
    }

    /// Look up a node by its ID.
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut Node<E::Act>> {
        if let Some(entry) = self.transposition_table.get(&node_id) {
            return self.nodes.get_mut(*entry);
        } else {
            None
        }
    }

    pub fn get_node_immut(&self, node_id: NodeId) -> Option<&Node<E::Act>> {
        if let Some(entry) = self.transposition_table.get(&node_id) {
            return self.nodes.get(*entry);
        } else {
            None
        }
    }

    /// Insert a new node into the transposition table and node arena.
    pub fn insert_node(&mut self, node_id: NodeId, node: Node<E::Act>) {
        let index = self.nodes.len();
        self.nodes.push(node);
        self.transposition_table.insert(node_id, index);
    }

    /// Recompute the cached value of a node based on its children's values.
    pub fn recompute_value(&mut self, node_id: NodeId) {
        // Snapshot parent data immutably to avoid overlapping mutable borrows.
        let (virtual_loss_counts, edge_visits, children, edge_penalties, node_value_estimate) = {
            let parent = match self.get_node_immut(node_id) {
                Some(p) => p,
                None => return,
            };
            let vl: usize = parent.virtual_losses.values().copied().sum();
            let ev = parent.edge_visits.clone();
            let ch = parent.children.clone();
            let ep = parent.edge_penalties.clone();
            (vl, ev, ch, ep, parent.value_estimate)
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
                let penalty = edge_penalties.get(&a).copied().unwrap_or(0.0);
                acc += (ev as f32) * (child.value + penalty);
            }
        }

        // Single mutable borrow to update the node.
        if let Some(node) = self.get_node_mut(node_id) {
            node.node_visits = 1 + total_edge_visits;
            node.value = (node.value_estimate + acc) / (node.node_visits as f32);
        }
    }

    /// Compute PUCT scores for all actions from this node.
    pub fn puct_scores(&self, node_id: NodeId, c_puct: f32) -> HashMap<E::Act, f32> {
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

    pub fn select_action_puct(&self, node_id: NodeId, c_puct: f32) -> Option<E::Act> {
        let scores = self.puct_scores(node_id, c_puct);
        scores.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(action, _)| action)
    }

    pub fn select_action(&self, node: &Node<E::Act>) -> Option<E::Act> {
        node.select_action()
    }

    pub fn add_child(&mut self, parent_id: NodeId, action: E::Act, child_id: NodeId) {
        if let Some(parent) = self.get_node_mut(parent_id) {
            parent.children.insert(action, child_id);
        }
    }

    pub fn node_exists(&self, node_id: NodeId) -> bool {
        self.transposition_table.contains_key(&node_id)
    }

    pub fn get_hash(&self, env: &E) -> NodeId { env.hash() as NodeId }

    pub fn normalize_prior(&self, priors: HashMap<E::Act, f32>, actions: &[E::Act]) -> HashMap<E::Act, f32> {
        let mut normalized = HashMap::new();
        let mut total: f32 = 0.0;
        for &a in actions {
            if let Some(&p) = priors.get(&a) {
                total += p;
            }
        }
        if total == 0.0 {
            let uniform_prob = 1.0 / (actions.len() as f32);
            for &a in actions {
                normalized.insert(a, uniform_prob);
            }
        } else {
            for &a in actions {
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
        Vec<(NodeId, E::Act)>,
        Option<NodeId>,
        E::Act,
        E,
        bool,
    ) {
        let mut path: Vec<(NodeId, E::Act)> = Vec::new();
        let mut node_id = root_id;
        let mut parent: Option<NodeId> = None;
        let mut action: Option<E::Act> = None;
        let mut repeat_detected = false;

        // work on a cloned environment so caller's env is not mutated
        let mut game = env.clone();

        while !game.done() && !repeat_detected {
            if !self.node_exists(node_id) {
                break;
            }

            // detect repeats based on path
            repeat_detected = self.check_state_repeat(game.hash() as NodeId, &path);
            if repeat_detected {
                break;
            }

            // choose action via PUCT
            let chosen = match self.select_action_puct(node_id, 1.4) {
                Some(a) => a,
                None => break,
            };
            action = Some(chosen);

            // mark virtual loss quickly (single-threaded counter)
            if let Some(node) = self.get_node_mut(node_id) {
                node.add_virtual_loss(action.unwrap());
            }

            // advance environment
            game.step(action.unwrap());

            // record step in path and advance
            path.push((node_id, action.unwrap()));
            parent = Some(node_id);

            // lookup child id from the parent snapshot
            if let Some(parent_node) = self.get_node_immut(node_id) {
                if let Some(cid) = parent_node.children.get(&action.unwrap()).copied() {
                    node_id = cid;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        (path, parent, action.unwrap(), game, repeat_detected)
    }

    pub fn check_state_repeat(&self, state_hash: NodeId, path: &[(NodeId, E::Act)]) -> bool {
        path.iter().any(|(h, _)| *h == state_hash)
    }

    pub fn expand(
        &mut self,
        parent_id: NodeId,
        action: E::Act,
        env: E,
        priors: HashMap<E::Act, f32>,
        value: f32,
    ) -> NodeId {
        let leaf_id = self.get_hash(&env);
        if !self.node_exists(leaf_id) {
            let normalized_priors = self.normalize_prior(priors, &env.valid_actions());
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
                for (&a, &cid) in &parent.children {
                    if cid == leaf_id && a != action {
                        eprintln!("[Alias detected] actions {:?} - {:?}", a, action);
                    }
                }
            }
        }

        leaf_id
    }

    pub fn backpropagate(&mut self, search_path: &[(NodeId, E::Act)], repeat_detected: bool) {
        if search_path.is_empty() {
            return;
        }
        // First revert virtual losses along the search path.
        for &(node_hash, action) in search_path.iter().rev() {
            if let Some(node) = self.get_node_mut(node_hash) {
                node.revert_virtual_loss(action);
            }
        }

        if repeat_detected {
            if let Some(&(last_parent_hash, last_action)) = search_path.last() {
                if let Some(parent) = self.get_node_mut(last_parent_hash) {
                    parent.apply_penalty(last_action);
                }
            }
        }

        // Backpropagate visits and recompute values in both cases
        for &(node_hash, action) in search_path.iter().rev() {
            if let Some(node) = self.get_node_mut(node_hash) {
                *node.edge_visits.entry(action).or_insert(0) += 1;
            }
            self.recompute_value(node_hash);
        }
    }

    /// Do inference on a batch of observations
    pub fn blocking_infer(
        &mut self,
        batch: &[E::Obs],
        client: &dyn InferenceClient<E>,
    ) -> (Vec<HashMap<E::Act, f32>>, Vec<f32>) {
        let (priors, values) = client.infer(batch).expect("Inference failed");
        (priors, values)
    }
}


#[cfg(test)]
pub mod test_env {
    use crate::environment::{Obs, Environment};
    use crate::inference::InferenceClient;
    use std::collections::HashMap;
    use anyhow::Result;

    /// Action: 0 = go right, 1 = go left
    pub type TestAction = u8;

    #[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
    pub struct TestObs {
        pub position: i32,
        pub valid: Vec<TestAction>,
    }
    impl Obs for TestObs {}

    #[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
    pub struct NumberLineEnv {
        pub position: i32,
        pub target: i32,
    }

    impl NumberLineEnv {
        pub fn new(target: i32) -> Self {
            Self { position: 0, target }
        }
    }

    impl Environment for NumberLineEnv {
        type Act = TestAction;
        type Obs = TestObs;

        fn step(&mut self, action: TestAction) {
            match action {
                0 => self.position += 1,
                1 => self.position -= 1,
                _ => {}
            }
        }

        fn observation(&self) -> TestObs {
            TestObs {
                position: self.position,
                valid: self.valid_actions(),
            }
        }

        fn valid_actions(&self) -> Vec<TestAction> {
            if self.done() {
                vec![]
            } else if self.position <= 0 {
                vec![0]
            } else {
                vec![0, 1]
            }
        }

        fn done(&self) -> bool {
            self.position == self.target
        }

        fn hash(&self) -> u64 {
            self.position as u64
        }

        fn render(&self) -> String {
            format!("Position: {}, Target: {}", self.position, self.target)
        }
    }

    /// Returns uniform priors and a simple heuristic value.
    pub struct UniformClient;

    impl InferenceClient<NumberLineEnv> for UniformClient {
        fn infer(
            &self,
            observations: &[TestObs],
        ) -> Result<(Vec<HashMap<TestAction, f32>>, Vec<f32>)> {
            let mut priors = Vec::new();
            let mut values = Vec::new();

            for obs in observations {
                let n = obs.valid.len() as f32;
                let mut prior = HashMap::new();
                for &a in &obs.valid {
                    prior.insert(a, 1.0 / n);
                }
                priors.push(prior);
                // Simple heuristic: closer to target = higher value
                values.push(-obs.position.abs() as f32 * 0.1);
            }

            Ok((priors, values))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use super::test_env::*;

    #[test]
    fn test_mcts_creation() {
        let mcts: MCTS<NumberLineEnv> = MCTS::new(1.0, 4);
        assert_eq!(mcts.terminal_value, 1.0);
        assert_eq!(mcts.batch_size, 4);
    }

    #[test]
    fn test_insert_and_get_node() {
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(0.0, 4);
        let priors = HashMap::from([(0u8, 0.5), (1u8, 0.5)]);
        let node = Node::new(priors, 0.42, 1, None);
        mcts.insert_node(1, node);
        let node = mcts.get_node_immut(1).unwrap();
        assert!((node.value - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_run_expands_tree() {
        let env = NumberLineEnv::new(3);
        let client = UniformClient;
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(1.0, 4);
        let root = mcts.run(&env, &client, 100);
        assert!(mcts.node_exists(root.id));
        assert!(mcts.nodes.len() > 1);
    }

    #[test]
    fn test_normalize_prior() {
        let mcts: MCTS<NumberLineEnv> = MCTS::new(0.0, 4);
        let priors = HashMap::from([(0u8, 0.2), (1u8, 0.3), (2u8, 0.5)]);
        let normalized = mcts.normalize_prior(priors, &[0, 1]);
        let total: f32 = normalized.values().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}