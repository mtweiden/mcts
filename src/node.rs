use crate::enums::Action;
use crate::enums::NodeId;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};


/// A single node in the MCTS graph.
pub struct Node {
    pub id: NodeId,
    pub prior_probs: HashMap<Action, f32>,
    pub value_estimate: f32,
    pub node_visits: usize,
    pub children: HashMap<Action, NodeId>,
    pub edge_visits: HashMap<Action, usize>,
    pub virtual_losses: HashMap<Action, AtomicUsize>,
    pub edge_penalties: HashMap<Action, f32>,
    pub value: f32,
    pub terminal_state: bool,
    pub repr: Option<String>,
}

impl Node {
    pub fn new(
        prior_probs: HashMap<Action, f32>,
        value: f32,
        id: NodeId,
        repr: Option<String>
    ) -> Self {
        let mut edge_visits: HashMap<Action, usize> = HashMap::new();
        let mut virtual_losses: HashMap<Action, AtomicUsize> = HashMap::new();
        let mut edge_penalties: HashMap<Action, f32> = HashMap::new();

        for &action in prior_probs.keys() {
            edge_visits.insert(action, 0);
            virtual_losses.insert(action, AtomicUsize::new(0));
            edge_penalties.insert(action, 0.0);
        }
        Self {
            id,
            prior_probs,
            value_estimate: value,
            node_visits: 0,
            children: HashMap::new(),
            edge_visits: edge_visits,
            virtual_losses: virtual_losses,
            edge_penalties: edge_penalties,
            value: value,
            terminal_state: false,
            repr: repr,
        }
    }

    pub fn new_terminal(id: NodeId, value: f32, repr: Option<String>) -> Self {
        Self {
            id,
            prior_probs: HashMap::new(),
            value_estimate: value,
            node_visits: 0,
            children: HashMap::new(),
            edge_visits: HashMap::new(),
            virtual_losses: HashMap::new(),
            edge_penalties: HashMap::new(),
            value,
            terminal_state: true,
            repr: repr,
        }
    }

    pub fn virtual_loss_copy(&self) -> HashMap<Action, usize> {
        self.virtual_losses
            .iter()
            .map(|(&action, v)| (action, v.load(Ordering::Relaxed)))
            .collect()
    }

    pub fn add_virtual_loss(&mut self, action: Action) {
        if let Some(count) = self.virtual_losses.get_mut(&action) {
            count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.virtual_losses.insert(action, AtomicUsize::new(1));
        }
    }

    pub fn revert_virtual_loss(&mut self, action: Action) {
        if let Some(count) = self.virtual_losses.get_mut(&action) {
            count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub fn apply_penalty(&mut self, action: Action) {
        *self.edge_penalties.entry(action).or_insert(0.0) -= 1.0;
    }

    pub fn revert_penalty(&mut self, action: Action) {
        if let Some(penalty) = self.edge_penalties.get_mut(&action) {
            *penalty += 1.0;
        }
    }

    pub fn select_action(&self) -> Option<Action> {
        self.edge_visits
            .iter()
            .max_by_key(|(_, &visits)| visits)
            .and_then(|(&action, &visits)| if visits == 0 { None } else { Some(action) })
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node {{ id: {}, value: {:.3}, edges: {} }}",
            self.id,
            self.value,
            self.children.len()
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let mut priors = HashMap::new();
        priors.insert(0, 0.5);
        priors.insert(1, 0.5);
        let node = Node::new(priors.clone(), 0.0, 1, None);
        assert_eq!(node.id, 1);
        assert_eq!(node.prior_probs, priors);
        assert_eq!(node.value_estimate, 0.0);
        assert_eq!(node.node_visits, 0);
        assert!(node.children.is_empty());
        assert_eq!(node.edge_visits, HashMap::from([(0, 0), (1, 0)]));
        assert_eq!(node.virtual_loss_copy(), HashMap::from([(0, 0), (1, 0)]));
        assert_eq!(node.edge_penalties, HashMap::from([(0, 0.0), (1, 0.0)]));
        assert_eq!(node.value, 0.0);
        assert!(!node.terminal_state);
    }
}