use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use rand::seq::IteratorRandom;
use rand::rng;

/// Represents a search node in MCTS.
#[derive(Clone)]
pub struct Node {
    pub priors: HashMap<i32, f32>,
    pub value_estimate: f32,
    pub node_visits: u32,
    pub children: HashMap<i32, Arc<RwLock<Node>>>,
    pub edge_visits: HashMap<i32, u32>,
    pub virtual_losses: HashMap<i32, u32>,
    pub edge_penalties: HashMap<i32, f32>,
    pub value: f32,
    pub terminal_state: bool,
}

impl Node {
    pub fn new(priors: HashMap<i32, f32>, value: f32) -> Self {
        let edge_visits = priors.keys().map(|&a| (a, 0)).collect();
        let virtual_losses = priors.keys().map(|&a| (a, 0)).collect();
        let edge_penalties = priors.keys().map(|&a| (a, 0.0)).collect();
        Self {
            priors,
            value_estimate: value,
            node_visits: 0,
            children: HashMap::new(),
            edge_visits,
            virtual_losses,
            edge_penalties,
            value,
            terminal_state: false,
        }
    }

    pub fn recompute_value(&mut self) -> f32 {
        let total_edge_visits =
            self.edge_visits.values().sum::<u32>() + self.virtual_losses.values().sum::<u32>();
        self.node_visits = 1 + total_edge_visits;

        if self.children.is_empty() || total_edge_visits == 0 {
            self.value = self.value_estimate;
            return self.value;
        }

        let mut acc = 0.0;
        for (a, child) in &self.children {
            let n = *self.edge_visits.get(a).unwrap_or(&0);
            if n > 0 {
                acc += (n as f32) * child.read().unwrap().value;
            }
        }

        self.value = (self.value_estimate + acc) / (self.node_visits as f32);
        self.value
    }

    pub fn puct_scores(&self, c_puct: f32) -> HashMap<i32, f32> {
        let total_visits =
            (self.edge_visits.values().sum::<u32>() + self.virtual_losses.values().sum::<u32>())
                as f32;
        let sqrt_total = (total_visits + 1e-8).sqrt();

        let mut scores = HashMap::new();
        for (&action, &prior) in &self.priors {
            let n_edge = *self.edge_visits.get(&action).unwrap_or(&0) as f32;
            let n_eff = n_edge + *self.virtual_losses.get(&action).unwrap_or(&0) as f32;
            let child_value = self
                .children
                .get(&action)
                .map(|c| c.read().unwrap().value)
                .unwrap_or(self.value);
            let penalty = *self.edge_penalties.get(&action).unwrap_or(&0.0);
            let q = child_value + penalty;
            let u = c_puct * prior * (sqrt_total / (1.0 + n_eff));
            scores.insert(action, q + u);
        }
        scores
    }

    pub fn select_action_puct(&self) -> i32 {
        let scores = self.puct_scores(1.1);
        let max_score = scores.values().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let best: Vec<i32> = scores
            .iter()
            .filter(|(_, &s)| (s - max_score).abs() < 1e-8)
            .map(|(&a, _)| a)
            .collect();
        *best.iter().choose(&mut rng()).unwrap()
    }

    pub fn select_action(&self) -> i32 {
        if self.children.is_empty() {
            panic!("Cannot select action from unexpanded node.");
        }
        let max_visits = self.edge_visits.values().copied().max().unwrap_or(0);
        let best: Vec<i32> = self
            .edge_visits
            .iter()
            .filter(|(_, &v)| v == max_visits)
            .map(|(&a, _)| a)
            .collect();
        *best.iter().choose(&mut rng()).unwrap()
    }

    pub fn add_virtual_loss(&mut self, action: i32, loss: u32) {
        *self.virtual_losses.entry(action).or_insert(0) += loss;
    }

    pub fn revert_virtual_loss(&mut self, action: i32, loss: u32) {
        let entry = self.virtual_losses.entry(action).or_insert(0);
        *entry = entry.saturating_sub(loss);
    }

    pub fn apply_penalty(&mut self, action: i32, penalty: f32) {
        let val = *self.edge_penalties.get(&action).unwrap_or(&0.0) + penalty;
        self.edge_penalties.insert(action, val);
    }

    pub fn revert_penalty(&mut self, action: i32, penalty: f32) {
        let val = *self.edge_penalties.get(&action).unwrap_or(&0.0) - penalty;
        self.edge_penalties
            .insert(action, if val.abs() < 1e-12 { 0.0 } else { val });
    }
}
