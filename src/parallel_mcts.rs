use crate::parallel_node::Node;
use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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
    pub transposition_table: DashMap<u64, Arc<RwLock<Node>>>,
}

impl MCTS {
    pub fn new() -> Self {
        Self {
            terminal_value: 1.0,
            transposition_table: DashMap::new(),
        }
    }

    pub fn run<E: Environment, A: Agent>(
        &self,
        env: &E,
        agent: &A,
        num_steps: usize,
    ) -> Arc<RwLock<Node>> {
        let root_hash = self.get_hash(env);
        let root = self
            .get_node(root_hash)
            .unwrap_or_else(|| {
                let node = self.create_node(env, agent);
                self.transposition_table.insert(root_hash, node.clone());
                node
            });

        if env.done() {
            return root;
        }

        for _ in 0..num_steps {
            let mut game = env.copy();
            let mut parent: Option<Arc<RwLock<Node>>> = None;
            let mut node = Some(root.clone());
            let mut search_path: Vec<(u64, i32)> = vec![];
            let mut repeat_detected = false;

            while let Some(n) = node.clone() {
                let act = n.read().unwrap().select_action_puct();
                let node_hash = self.get_hash(&game);
                repeat_detected = self.check_state_repeat(node_hash, &search_path);
                if repeat_detected {
                    break;
                }
                search_path.push((node_hash, act));
                parent = node;
                node = self.get_child_node(&parent.clone().unwrap(), act);
                game.step(act);
                if game.done() {
                    break;
                }
            }

            if !repeat_detected && !game.done() {
                if let Some(p) = parent {
                    self.expand(&p, search_path.last().unwrap().1, &game, agent);
                }
            }

            self.backpropagate(&search_path, repeat_detected);
        }

        root
    }

    pub fn run_parallel<E: Environment, A: Agent>(
        &self,
        env: &E,
        agent: &A,
        num_simulations: usize,
    ) {
        (0..num_simulations).into_par_iter().for_each(|_| {
            let mut local_env = env.clone();
            self.simulate(&mut local_env, agent);
        });
    }

    fn simulate<E: Environment, A: Agent>(&self, env: &mut E, agent: &A) {
        let mut path: Vec<(u64, i32)> = vec![];
        let mut current_hash = env.hash_state();

        let mut node_arc = self
            .get_node(current_hash)
            .unwrap_or_else(|| {
                let node = self.create_node(env, agent);
                self.transposition_table.insert(current_hash, node.clone());
                node
            });

        while !env.done() {
            let act = node_arc.read().unwrap().select_action_puct();
            path.push((current_hash, act));
            node_arc.write().unwrap().add_virtual_loss(act, 1);
            env.step(act);
            current_hash = env.hash_state();

            node_arc = match self.get_node(current_hash) {
                Some(n) => n,
                None => break,
            };
        }

        if !env.done() {
            let new_node = self.create_node(env, agent);
            self.transposition_table.insert(current_hash, new_node.clone());
            if let Some((parent_hash, act)) = path.last() {
                if let Some(p) = self.get_node(*parent_hash) {
                    p.write().unwrap().children.insert(*act, new_node.clone());
                }
            }
        }

        self.backpropagate(&path, false);
    }

    pub fn get_hash<E: Environment>(&self, env: &E) -> u64 {
        env.hash_state()
    }

    pub fn get_node(&self, node_hash: u64) -> Option<Arc<RwLock<Node>>> {
        self.transposition_table.get(&node_hash).map(|r| r.clone())
    }

    pub fn get_child_node(&self, parent: &Arc<RwLock<Node>>, action: i32) -> Option<Arc<RwLock<Node>>> {
        parent.read().unwrap().children.get(&action).cloned()
    }

    pub fn check_state_repeat(&self, state_hash: u64, path: &[(u64, i32)]) -> bool {
        path.iter().any(|(h, _)| *h == state_hash)
    }

    pub fn is_expanded(&self, node: &Option<Arc<RwLock<Node>>>) -> bool {
        node.is_some()
    }

    pub fn expand<E: Environment, A: Agent>(
        &self,
        parent: &Arc<RwLock<Node>>,
        action: i32,
        env: &E,
        agent: &A,
    ) -> Arc<RwLock<Node>> {
        let leaf_hash = env.hash_state();
        let leaf = self
            .get_node(leaf_hash)
            .unwrap_or_else(|| {
                let n = self.create_node(env, agent);
                self.transposition_table.insert(leaf_hash, n.clone());
                n
            });

        let mut p = parent.write().unwrap();
        if !p.children.contains_key(&action) {
            p.children.insert(action, leaf.clone());
            *p.edge_visits.entry(action).or_insert(0) += 1;
        }

        leaf
    }

    pub fn backpropagate(&self, search_path: &[(u64, i32)], repeat_detected: bool) {
        if repeat_detected {
            if let Some((last_hash, act)) = search_path.last() {
                if let Some(p) = self.get_node(*last_hash) {
                    p.write().unwrap().apply_penalty(*act, -1.0);
                }
            }
        }

        for (h, act) in search_path.iter().rev() {
            if let Some(n) = self.get_node(*h) {
                let mut node = n.write().unwrap();
                *node.edge_visits.entry(*act).or_insert(0) += 1;
                node.revert_virtual_loss(*act, 1);
                node.recompute_value();
            }
        }
    }

    pub fn create_node<E: Environment, A: Agent>(
        &self,
        env: &E,
        agent: &A,
    ) -> Arc<RwLock<Node>> {
        if env.done() {
            let mut n = Node::new(HashMap::new(), self.terminal_value);
            n.terminal_state = true;
            return Arc::new(RwLock::new(n));
        }
        let obs = env.observation();
        let (priors, value) = agent.infer(&obs);
        let actions = env.valid_actions();
        let normed = self.normalize_prior(&priors, &actions);
        Arc::new(RwLock::new(Node::new(normed, value)))
    }

    pub fn normalize_prior(&self, logits: &[f32], actions: &[i32]) -> HashMap<i32, f32> {
        let exp: Vec<f32> = logits.iter().map(|x| x.exp()).collect();
        let sum: f32 = exp.iter().sum::<f32>().max(1e-8);
        let mut priors = HashMap::new();
        for &a in actions {
            let p = exp.get(a as usize).copied().unwrap_or(0.0) / sum;
            priors.insert(a, p);
        }
        priors
    }

    /// Choose the action from root with the highest visit count.
    pub fn choose_action(&self, root: &Arc<RwLock<Node>>) -> i32 {
        root.read().unwrap().select_action()
    }
}
