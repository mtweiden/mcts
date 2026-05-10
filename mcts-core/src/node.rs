use std::collections::HashMap;

use crate::environment::Act;

pub type NodeId = u64;

/// A single node in the MCTS graph.
///
/// `num_actions` is the size of the action space at this node's state.
/// `valid_actions` is the dense list of actions actually playable from
/// this node (subset of `0..num_actions`). All "for each action" loops
/// iterate `valid_actions` — never the full prior_probs Vec — so
/// invalid slots in the dense storage are never read.
///
/// `prior_probs` is now a dense `Vec<f32>` of length `num_actions`,
/// indexed by `action.to_action_index()`. Slots for invalid actions
/// remain at the default 0.0 and are never read.
///
/// `edge_visits` / `virtual_losses` / `edge_penalties` / `children`
/// are still HashMap-backed in this step. Steps 4 and 5 of the refactor
/// will convert them to dense Vecs the same way.
///
/// Action ids are scoped to a single node's state. The same numeric id
/// may mean different things in another node's state — the contract is
/// that within this node, every entry of `valid_actions` is a valid
/// action whose `to_action_index()` is in `0..num_actions`.
#[derive(Clone)]
pub struct Node<A: Act> {
    pub id: NodeId,
    pub num_actions: usize,
    pub valid_actions: Vec<A>,
    pub prior_probs: Vec<f32>,
    pub value_estimate: f32,
    pub node_visits: usize,
    pub children: HashMap<A, NodeId>,
    pub edge_visits: HashMap<A, usize>,
    pub virtual_losses: HashMap<A, usize>,
    pub edge_penalties: HashMap<A, f32>,
    pub value: f32,
    pub terminal_state: bool,
    pub repr: Option<String>,
}

impl<A: Act> Node<A> {
    pub fn new(
        num_actions: usize,
        prior_probs_map: HashMap<A, f32>,
        value: f32,
        id: NodeId,
        repr: Option<String>
    ) -> Self {
        let mut prior_probs = vec![0.0_f32; num_actions];
        let mut valid_actions: Vec<A> = Vec::with_capacity(prior_probs_map.len());
        let mut edge_visits: HashMap<A, usize> = HashMap::new();
        let mut virtual_losses: HashMap<A, usize> = HashMap::new();
        let mut edge_penalties: HashMap<A, f32> = HashMap::new();

        for (action, prob) in prior_probs_map.iter() {
            let idx = action.to_action_index();
            debug_assert!(
                idx < num_actions,
                "Node::new: action index {} out of bounds for num_actions={}",
                idx, num_actions
            );
            if idx < num_actions {
                prior_probs[idx] = *prob;
            }
            valid_actions.push(*action);
            edge_visits.insert(*action, 0);
            virtual_losses.insert(*action, 0);
            edge_penalties.insert(*action, 0.0);
        }

        Self {
            id,
            num_actions,
            valid_actions,
            prior_probs,
            value_estimate: value,
            node_visits: 0,
            children: HashMap::new(),
            edge_visits,
            virtual_losses,
            edge_penalties,
            value,
            terminal_state: false,
            repr,
        }
    }

    pub fn new_terminal(id: NodeId, value: f32, repr: Option<String>) -> Self {
        Self {
            id,
            // Terminal nodes have no outgoing actions; size 0 is a
            // sentinel that prevents any accidental dense indexing.
            num_actions: 0,
            valid_actions: Vec::new(),
            prior_probs: Vec::new(),
            value_estimate: value,
            node_visits: 0,
            children: HashMap::new(),
            edge_visits: HashMap::new(),
            virtual_losses: HashMap::new(),
            edge_penalties: HashMap::new(),
            value,
            terminal_state: true,
            repr,
        }
    }

    pub fn add_virtual_loss(&mut self, action: A) {
        if let Some(count) = self.virtual_losses.get_mut(&action) {
            *count += 1;
        } else {
            self.virtual_losses.insert(action, 1);
        }
    }

    pub fn revert_virtual_loss(&mut self, action: A) {
        if let Some(count) = self.virtual_losses.get_mut(&action) {
            *count = (*count).saturating_sub(1);
        }
    }

    pub fn apply_penalty(&mut self, action: A) {
        // Scale the penalty amount by the number of valid actions at
        // this state. Pre-refactor this used prior_probs.len() (a
        // HashMap whose size equalled the number of valid actions);
        // valid_actions.len() is the dense-Vec equivalent.
        let n = self.valid_actions.len() as f32;
        let penalty_amount = -1.0 / n;
        *self.edge_penalties.entry(action).or_insert(0.0) += penalty_amount;
    }

    pub fn revert_penalty(&mut self, action: A) {
        if let Some(penalty) = self.edge_penalties.get_mut(&action) {
            *penalty = (*penalty + 1.0).min(0.0);
        }
    }

    pub fn select_action(&self) -> Option<A> {
        self.edge_visits
            .iter()
            .max_by_key(|(_, visits)| *visits)
            .and_then(|(&action, &visits)| if visits == 0 { None } else { Some(action) })
    }
}
    
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let mut priors: HashMap<u16, f32> = HashMap::new();
        priors.insert(0 as u16, 0.5);
        priors.insert(1 as u16, 0.5);
        let node = Node::new(2, priors.clone(), 0.0, 1, None);
        assert_eq!(node.id, 1);
        assert_eq!(node.num_actions, 2);
        assert_eq!(node.prior_probs.len(), 2);
        assert!((node.prior_probs[0] - 0.5).abs() < 1e-6);
        assert!((node.prior_probs[1] - 0.5).abs() < 1e-6);
        assert_eq!(node.valid_actions.len(), 2);
        assert!(node.valid_actions.contains(&0));
        assert!(node.valid_actions.contains(&1));
        assert_eq!(node.value_estimate, 0.0);
        assert_eq!(node.node_visits, 0);
        assert!(node.children.is_empty());
        assert_eq!(node.edge_visits, HashMap::from([(0 as u16, 0), (1 as u16, 0)]));
        assert_eq!(node.virtual_losses, HashMap::from([(0 as u16, 0), (1 as u16, 0)]));
        assert_eq!(node.edge_penalties, HashMap::from([(0 as u16, 0.0), (1 as u16, 0.0)]));
        assert_eq!(node.value, 0.0);
        assert!(!node.terminal_state);
    }
}