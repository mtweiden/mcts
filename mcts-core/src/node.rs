use std::collections::HashMap;

use crate::environment::Act;

// All per-edge data on Node (priors, visits, virtual losses, penalties)
// is now stored as dense `Vec<T>` of length `num_actions`, indexed by
// `action.to_action_index()`. `valid_actions` is the iteration list —
// per-edge loops walk it and look up dense slots, so invalid action
// slots are never read.

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
/// `edge_visits` / `virtual_losses` / `edge_penalties` / `children` are
/// dense `Vec<T>` of length `num_actions`. Slots for invalid actions
/// stay at the default zero / `None` and are never read (loops walk
/// `valid_actions`). `children[i] = None` means "action `i` has no
/// expanded child yet" (the FPU branch in PUCT applies); `Some(node_id)`
/// means the child is in the transposition table.
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
    pub children: Vec<Option<NodeId>>,
    pub edge_visits: Vec<usize>,
    pub virtual_losses: Vec<usize>,
    pub edge_penalties: Vec<f32>,
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
        }

        Self {
            id,
            num_actions,
            valid_actions,
            prior_probs,
            value_estimate: value,
            node_visits: 0,
            children: vec![None; num_actions],
            edge_visits: vec![0; num_actions],
            virtual_losses: vec![0; num_actions],
            edge_penalties: vec![0.0; num_actions],
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
            children: Vec::new(),
            edge_visits: Vec::new(),
            virtual_losses: Vec::new(),
            edge_penalties: Vec::new(),
            value,
            terminal_state: true,
            repr,
        }
    }

    pub fn add_virtual_loss(&mut self, action: A) {
        let idx = action.to_action_index();
        if idx < self.virtual_losses.len() {
            self.virtual_losses[idx] += 1;
        }
    }

    pub fn revert_virtual_loss(&mut self, action: A) {
        let idx = action.to_action_index();
        if idx < self.virtual_losses.len() {
            self.virtual_losses[idx] = self.virtual_losses[idx].saturating_sub(1);
        }
    }

    pub fn apply_penalty(&mut self, action: A) {
        // Scale the penalty amount by the number of valid actions at
        // this state.
        let n = self.valid_actions.len() as f32;
        let penalty_amount = -1.0 / n;
        let idx = action.to_action_index();
        if idx < self.edge_penalties.len() {
            self.edge_penalties[idx] += penalty_amount;
        }
    }

    pub fn revert_penalty(&mut self, action: A) {
        let idx = action.to_action_index();
        if idx < self.edge_penalties.len() {
            let p = self.edge_penalties[idx];
            self.edge_penalties[idx] = (p + 1.0).min(0.0);
        }
    }

    pub fn select_action(&self) -> Option<A> {
        // Iterate valid_actions and pick the one with the highest
        // visit count (skipping zero-visit actions, matching the
        // pre-refactor HashMap semantics).
        let mut best: Option<(A, usize)> = None;
        for &a in &self.valid_actions {
            let v = self.edge_visits[a.to_action_index()];
            if v == 0 { continue; }
            match best {
                Some((_, bv)) if bv >= v => {}
                _ => best = Some((a, v)),
            }
        }
        best.map(|(a, _)| a)
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
        // Children Vec is sized to num_actions and starts with all None.
        assert_eq!(node.children, vec![None, None]);
        assert_eq!(node.edge_visits, vec![0_usize, 0]);
        assert_eq!(node.virtual_losses, vec![0_usize, 0]);
        assert_eq!(node.edge_penalties, vec![0.0_f32, 0.0]);
        assert_eq!(node.value, 0.0);
        assert!(!node.terminal_state);
    }
}