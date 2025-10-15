use std::collections::HashMap;
use rand::seq::IteratorRandom;
use rand::rng;

/// Represents a search node in MCTS.
#[derive(Clone)]
pub struct Node {
    // priors are stored per-edge now
    pub value_estimate: f32,
    pub node_visits: u32,
    // compact per-node edge vector
    pub edges: Vec<Edge>,
    pub value: f32,
    pub terminal_state: bool,
}

#[derive(Clone)]
pub struct Edge {
    pub action: i32,
    pub child: Option<usize>,
    pub visits: u32,
    pub virtual_losses: u32,
    pub penalty: f32,
    pub prior: f32,
}

impl Node {
    pub fn new(edges: Vec<Edge>, value: f32) -> Self {
        Self {
            value_estimate: value,
            node_visits: 0,
            edges,
            value,
            terminal_state: false,
        }
    }

    // Recompute value using the arena to access child node values.
    pub fn recompute_value(&mut self, arena: &mut Vec<Node>) -> f32 {
        let total_edge_visits = self.edges.iter().map(|e| e.visits + e.virtual_losses).sum::<u32>();
        self.node_visits = 1 + total_edge_visits;

        if self.edges.is_empty() || total_edge_visits == 0 {
            self.value = self.value_estimate;
            return self.value;
        }

        let mut acc = 0.0;
        for edge in &self.edges {
            let n = edge.visits;
            if n > 0 {
                if let Some(child_idx) = edge.child {
                    acc += (n as f32) * arena[child_idx].value;
                }
            }
        }

        self.value = (self.value_estimate + acc) / (self.node_visits as f32);
        self.value
    }

    /// Compute the recomputed value and node_visits without mutating self.
    /// Returns (value, node_visits).
    pub fn compute_recomputed_value(&self, arena: &Vec<Node>) -> (f32, u32) {
        let total_edge_visits = self.edges.iter().map(|e| e.visits + e.virtual_losses).sum::<u32>();
        let node_visits = 1 + total_edge_visits;

        if self.edges.is_empty() || total_edge_visits == 0 {
            return (self.value_estimate, node_visits);
        }

        let mut acc = 0.0;
        for edge in &self.edges {
            let n = edge.visits;
            if n > 0 {
                if let Some(child_idx) = edge.child {
                    acc += (n as f32) * arena[child_idx].value;
                }
            }
        }

        ((self.value_estimate + acc) / (node_visits as f32), node_visits)
    }

    pub fn puct_scores(&self, c_puct: f32, arena: &Vec<Node>) -> HashMap<i32, f32> {
        let total_visits = (self.edges.iter().map(|e| e.visits + e.virtual_losses).sum::<u32>()) as f32;
        let sqrt_total = (total_visits + 1e-8).sqrt();

        let mut scores = HashMap::new();
        // iterate edges vector (matches priors)
        for edge in &self.edges {
            let action = edge.action;
            let prior = edge.prior;
            let n_edge = edge.visits as f32;
            let n_eff = n_edge + edge.virtual_losses as f32;
            let child_value = edge.child.map(|idx| arena[idx].value).unwrap_or(self.value);
            let penalty = edge.penalty;
            let q = child_value + penalty;
            let u = c_puct * prior * (sqrt_total / (1.0 + n_eff));
            scores.insert(action, q + u);
        }
        scores
    }

    pub fn select_action_puct(&self, arena: &Vec<Node>) -> i32 {
        let scores = self.puct_scores(1.1, arena);
        let max_score = scores.values().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let best: Vec<i32> = scores
            .iter()
            .filter(|(_, &s)| (s - max_score).abs() < 1e-8)
            .map(|(&a, _)| a)
            .collect();
        *best.iter().choose(&mut rng()).unwrap()
    }

    pub fn select_action(&self) -> i32 {
        if self.edges.is_empty() {
            panic!("Cannot select action from unexpanded node.");
        }
        let max_visits = self.edges.iter().map(|e| e.visits).max().unwrap_or(0);
        let best: Vec<i32> = self
            .edges
            .iter()
            .filter(|e| e.visits == max_visits)
            .map(|e| e.action)
            .collect();
        *best.iter().choose(&mut rng()).unwrap()
    }

    pub fn add_virtual_loss(&mut self, action: i32, loss: u32) {
        if let Some(e) = self.edges.iter_mut().find(|e| e.action == action) {
            e.virtual_losses = e.virtual_losses.saturating_add(loss);
        }
    }

    pub fn revert_virtual_loss(&mut self, action: i32, loss: u32) {
        if let Some(e) = self.edges.iter_mut().find(|e| e.action == action) {
            e.virtual_losses = e.virtual_losses.saturating_sub(loss);
        }
    }

    pub fn apply_penalty(&mut self, action: i32, penalty: f32) {
        if let Some(e) = self.edges.iter_mut().find(|e| e.action == action) {
            e.penalty += penalty;
        }
    }

    pub fn revert_penalty(&mut self, action: i32, penalty: f32) {
        if let Some(e) = self.edges.iter_mut().find(|e| e.action == action) {
            e.penalty -= penalty;
            if e.penalty.abs() < 1e-12 {
                e.penalty = 0.0;
            }
        }
    }
}
