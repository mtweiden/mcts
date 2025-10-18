use crate::node::{Action, Node, NodeId};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};


/// Core Monte Carlo Tree Search engine.
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
    /// Node functions
    /// ------------------------------------------------------------------------
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
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_priors(pairs: &[(Action, f32)]) -> HashMap<Action, f32> {
        let mut m = HashMap::new();
        for &(a, p) in pairs {
            m.insert(a, p);
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
        let node = Node::new(priors.clone(), 0.42, 1);
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
        let node = Node::new(priors, 0.1, 2);
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
        let node = Node::new(priors, 0.33, 3);
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
        let node = Node::new(priors.clone(), 0.5, 4);
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
        let mut node = Node::new(priors.clone(), 0.0, 5);
        node.edge_visits.insert(0, 10);
        node.edge_visits.insert(1, 20);
        mcts.insert_node(node);
        let chosen = mcts.select_action(5).unwrap();
        assert_eq!(chosen, 1); // action 1 has more visits, prefer higher prior
    }

    #[test]
    fn test_select_action_all_zero_visits_prefers_higher_prior() {
        let mcts = MCTS::new(0.0, 4);
        let priors = make_priors(&[(0usize, 0.2f32), (1usize, 0.8f32)]);
        let node = Node::new(priors.clone(), 0.0, 6);
        mcts.insert_node(node);
        let chosen = mcts.select_action(6);
        assert_eq!(chosen, None);
    }

    #[test]
    fn test_add_child_inserts_mapping() {
        let mcts = MCTS::new(0.0, 4);
        let parent_priors = make_priors(&[(0usize, 1.0f32)]);
        let parent = Node::new(parent_priors, 0.0, 10);
        let child = Node::new(make_priors(&[]), 0.0, 11);
        mcts.insert_node(parent);
        mcts.insert_node(child);

        // add child under action 0
        mcts.add_child(10, 0, 11);

        // verify parent's children map contains the mapping action -> child_id
        mcts.with_node_read(10, |n| {
            assert_eq!(n.children.get(&0), Some(&11));
        });
    }
}
