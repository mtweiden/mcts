use crate::node::Node;
use std::collections::HashMap;
use std::cell::RefCell;
// Single-threaded transposition table and node refs.

/// Trait defining the interface for an environment.
pub trait Environment: Send + Sync + Clone {
    fn hash_state(&self) -> u64;
    fn done(&self) -> bool;
    fn valid_actions(&self) -> Vec<i32>;
    fn step(&mut self, action: i32);
    fn copy(&self) -> Self;
    fn observation(&self) -> Vec<f32>;
}

/// Trait defining the interface for an agent.
pub trait Agent: Send + Sync {
    fn infer(&self, obs: &[f32]) -> (Vec<f32>, f32);
}

/// Monte Carlo Tree Search (with graph-style transpositions)
pub struct MCTS {
    pub terminal_value: f32,
    pub transposition_table: RefCell<HashMap<u64, usize>>,
    // single-threaded arena of nodes
    pub arena: RefCell<Vec<Node>>,
}

impl MCTS {
    pub fn new() -> Self {
        Self {
            terminal_value: 1.0,
            transposition_table: RefCell::new(HashMap::new()),
            arena: RefCell::new(Vec::new()),
        }
    }

    pub fn run<E: Environment, A: Agent>(
        &self,
        env: &E,
        agent: &A,
        num_steps: usize,
    ) -> usize {
        let root_hash = self.get_hash(env);
        let root_idx = self
            .get_node(root_hash)
            .unwrap_or_else(|| {
                let idx = self.create_node(env, agent);
                self.transposition_table.borrow_mut().insert(root_hash, idx);
                idx
            });

        let root = root_idx;

        if env.done() {
            return root;
        }

        for _ in 0..num_steps {
            let mut game = env.copy();
            let mut parent: Option<usize> = None;
            let mut node = Some(root);
            let mut search_path: Vec<(u64, i32)> = vec![];
            let mut repeat_detected = false;

            while let Some(n_idx) = node {
                let arena_ref = self.arena.borrow();
                let act = arena_ref[n_idx].select_action_puct(&*arena_ref);
                let node_hash = self.get_hash(&game);
                repeat_detected = self.check_state_repeat(node_hash, &search_path);
                if repeat_detected {
                    break;
                }
                search_path.push((node_hash, act));
                parent = node;
                node = self.get_child_node(parent.unwrap(), act);
                game.step(act);
                if game.done() {
                    break;
                }
            }

            if !repeat_detected && !game.done() {
                if let Some(p_idx) = parent {
                    self.expand(p_idx, search_path.last().unwrap().1, &game, agent);
                }
            }

            self.backpropagate(&search_path, repeat_detected);
        }

        // return Rc-like index wrapped in Node by reference: but keep API simple and return root index
        // (caller can call choose_action with this index)
        // For compatibility, create a dummy Node wrapper is not necessary; return the index as usize
        // Changing signature would be breaking; instead return a Node reference by index.
        // For now we return the node index wrapped in the arena by returning a usize as placeholder.
        // NOTE: Keep original return type signature changed earlier; return index as Rc/RefCell replacement.
        root_idx
    }

    pub fn run_parallel<E: Environment, A: Agent>(
        &self,
        env: &E,
        agent: &A,
        num_simulations: usize,
    ) {
        // Execute simulations sequentially on the current thread.
        for _ in 0..num_simulations {
            let mut local_env = env.clone();
            self.simulate(&mut local_env, agent);
        }
    }

    fn simulate<E: Environment, A: Agent>(&self, env: &mut E, agent: &A) {
        let mut path: Vec<(u64, i32)> = vec![];
        let mut current_hash = env.hash_state();

        let mut node_idx = self
            .get_node(current_hash)
            .unwrap_or_else(|| {
                let idx = self.create_node(env, agent);
                self.transposition_table.borrow_mut().insert(current_hash, idx);
                idx
            });

        while !env.done() {
            let arena_ref = self.arena.borrow();
            let act = arena_ref[node_idx].select_action_puct(&*arena_ref);
            drop(arena_ref);
            path.push((current_hash, act));
            self.arena.borrow_mut()[node_idx].add_virtual_loss(act, 1);
            env.step(act);
            current_hash = env.hash_state();

            node_idx = match self.get_node(current_hash) {
                Some(n) => n,
                None => break,
            };
        }

        if !env.done() {
            let new_node = self.create_node(env, agent);
            self.transposition_table.borrow_mut().insert(current_hash, new_node);
            if let Some((parent_hash, act)) = path.last() {
                if let Some(p_idx) = self.get_node(*parent_hash) {
                    let mut arena = self.arena.borrow_mut();
                    if let Some(edge) = arena[p_idx].edges.iter_mut().find(|e| e.action == *act) {
                        edge.child = Some(new_node);
                    }
                }
            }
        }

        self.backpropagate(&path, false);
    }

    pub fn get_hash<E: Environment>(&self, env: &E) -> u64 {
        env.hash_state()
    }

    pub fn get_node(&self, node_hash: u64) -> Option<usize> {
        self.transposition_table.borrow().get(&node_hash).copied()
    }

    pub fn get_child_node(&self, parent: usize, action: i32) -> Option<usize> {
        self.arena.borrow()[parent].edges.iter().find(|e| e.action == action).and_then(|e| e.child)
    }

    pub fn check_state_repeat(&self, state_hash: u64, path: &[(u64, i32)]) -> bool {
        path.iter().any(|(h, _)| *h == state_hash)
    }

    pub fn is_expanded(&self, node: &Option<usize>) -> bool {
        node.is_some()
    }

    pub fn expand<E: Environment, A: Agent>(
        &self,
        parent_idx: usize,
        action: i32,
        env: &E,
        agent: &A,
    ) -> usize {
        let leaf_hash = env.hash_state();
        let leaf_idx = self
            .get_node(leaf_hash)
            .unwrap_or_else(|| {
                let n = self.create_node(env, agent);
                self.transposition_table.borrow_mut().insert(leaf_hash, n);
                n
            });

        let mut arena = self.arena.borrow_mut();
        let p = &mut arena[parent_idx];
        if let Some(edge) = p.edges.iter_mut().find(|e| e.action == action) {
            if edge.child.is_none() {
                edge.child = Some(leaf_idx);
            }
            edge.visits = edge.visits.saturating_add(1);
        }

        leaf_idx
    }

    pub fn backpropagate(&self, search_path: &[(u64, i32)], repeat_detected: bool) {
        if repeat_detected {
            if let Some((last_hash, act)) = search_path.last() {
                if let Some(p_idx) = self.get_node(*last_hash) {
                    self.arena.borrow_mut()[p_idx].apply_penalty(*act, -1.0);
                }
            }
        }

        for (h, act) in search_path.iter().rev() {
            if let Some(n_idx) = self.get_node(*h) {
                // compute new value using an immutable borrow
                let arena_read = self.arena.borrow();
                let (new_value, new_visits) = arena_read[n_idx].compute_recomputed_value(&*arena_read);
                drop(arena_read);

                // now mutate the node
                let mut arena = self.arena.borrow_mut();
                let node = &mut arena[n_idx];
                if let Some(edge) = node.edges.iter_mut().find(|e| e.action == *act) {
                    edge.visits = edge.visits.saturating_add(1);
                }
                node.revert_virtual_loss(*act, 1);
                node.value = new_value;
                node.node_visits = new_visits;
            }
        }
    }

    pub fn create_node<E: Environment, A: Agent>(
        &self,
        env: &E,
        agent: &A,
    ) -> usize {
        if env.done() {
            let mut n = Node::new(Vec::new(), self.terminal_value);
            n.terminal_state = true;
            let mut arena = self.arena.borrow_mut();
            arena.push(n);
            return arena.len() - 1;
        }
        let obs = env.observation();
        let (priors, value) = agent.infer(&obs);
        let actions = env.valid_actions();
        let normed = self.normalize_prior(&priors, &actions);
        let mut edges = Vec::new();
        for (i, &a) in actions.iter().enumerate() {
            let p = normed.get(i).copied().unwrap_or(0.0);
            edges.push(crate::node::Edge { action: a, child: None, visits: 0, virtual_losses: 0, penalty: 0.0, prior: p });
        }
        let mut arena = self.arena.borrow_mut();
        arena.push(Node::new(edges, value));
        arena.len() - 1
    }

    pub fn normalize_prior(&self, logits: &[f32], actions: &[i32]) -> Vec<f32> {
        let exp: Vec<f32> = logits.iter().map(|x| x.exp()).collect();
        let sum: f32 = exp.iter().sum::<f32>().max(1e-8);
        let mut priors = Vec::with_capacity(actions.len());
        for &a in actions {
            let p = exp.get(a as usize).copied().unwrap_or(0.0) / sum;
            priors.push(p);
        }
        priors
    }

    /// Choose the action from root with the highest visit count.
    pub fn choose_action(&self, root: usize) -> i32 {
        self.arena.borrow()[root].select_action()
    }
}
