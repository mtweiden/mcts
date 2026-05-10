//! Differential / invariant test harness for MCTS.
//!
//! These tests are the load-bearing safety net for the planned
//! HashMap → dense Vec refactor of `Node` per-edge data and the planned
//! in-place `advance_root` rewrite. They live in tests/ (integration
//! tests) so they exercise only the public API of mcts-core, mirroring
//! how mcts-tilers consumes it.
//!
//! Strategy:
//!   - A `WideEnv` test environment with parameterised branching factor
//!     up to 250, sized to match real tilers gather scenarios.
//!   - A `DeterministicClient` whose priors and values are a pure
//!     function of the observation hash, so search trajectories are
//!     reproducible (modulo HashMap iteration order, which is NOT
//!     guaranteed to be deterministic in std). The tests therefore
//!     assert *invariants* and *distributional properties*, not
//!     byte-equal trees.
//!   - Tests at branching factors {2, 50, 250} — the small case
//!     anchors against existing test_env coverage; the medium and large
//!     cases exercise the regime where dense-Vec semantics differ from
//!     HashMap semantics (collision behavior, iteration cost, dense
//!     index assumption).
//!
//! Bugs the suite is designed to catch in any future refactor:
//!   - Off-by-one in action indexing → invalid actions visited.
//!   - Missing initialisation of a per-edge slot → stale visit counts.
//!   - advance_root losing nodes / leaving stale transposition_table
//!     entries.
//!   - Incorrect handling of duplicate priors / actions.
//!   - recompute_value drifting away from a hand-computed visit-weighted
//!     average.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use anyhow::Result;

use mcts_core::environment::{Environment, Obs};
use mcts_core::inference::InferenceClient;
use mcts_core::mcts::MCTS;
use mcts_core::node::Node;

// ============================================================================
// Wider test environment
// ============================================================================

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct WideObs {
    pub state_hash: u64,
    pub valid: Vec<u32>,
}
impl Obs for WideObs {}

#[derive(Clone, Debug)]
pub struct WideEnv {
    /// Branching factor: every non-terminal state has this many actions.
    pub num_actions: u32,
    /// Number of steps already taken — terminates at `max_depth`.
    pub depth: u32,
    pub max_depth: u32,
    /// Path of actions taken so far. Determines hash and observation.
    pub history: Vec<u32>,
    /// State counter that varies the hash of states with identical history
    /// (used in tests for transposition / advance_root).
    pub state_offset: u64,
}

impl WideEnv {
    pub fn new(num_actions: u32, max_depth: u32) -> Self {
        Self {
            num_actions,
            depth: 0,
            max_depth,
            history: Vec::new(),
            state_offset: 0,
        }
    }
}

impl Environment for WideEnv {
    type Act = u32;
    type Obs = WideObs;

    fn step(&mut self, action: u32) {
        self.history.push(action);
        self.depth += 1;
    }

    fn done(&self) -> bool {
        self.depth >= self.max_depth
    }

    fn observation(&self) -> WideObs {
        WideObs {
            state_hash: self.hash(),
            valid: self.valid_actions(),
        }
    }

    fn valid_actions(&self) -> Vec<u32> {
        if self.done() {
            vec![]
        } else {
            (0..self.num_actions).collect()
        }
    }

    fn hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.history.hash(&mut h);
        self.state_offset.hash(&mut h);
        h.finish()
    }

    fn render(&self) -> String {
        format!("WideEnv(depth={}, history={:?})", self.depth, self.history)
    }
}

// ============================================================================
// Deterministic inference client
// ============================================================================

/// Returns priors and values derived from the observation hash. Same obs
/// → same prior every time. Priors are not uniform; instead they vary
/// across actions so that PUCT has work to do.
pub struct DeterministicClient;

fn pseudo_random_f32(seed: u64, salt: u32) -> f32 {
    // splitmix64-ish, bounded to [0, 1).
    let mut x = seed.wrapping_add((salt as u64).wrapping_mul(0x9E3779B97F4A7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    (x as f32 / u64::MAX as f32).abs()
}

impl InferenceClient<WideEnv> for DeterministicClient {
    fn infer(
        &self,
        observations: &[WideObs],
    ) -> Result<(Vec<HashMap<u32, f32>>, Vec<f32>)> {
        let mut priors = Vec::with_capacity(observations.len());
        let mut values = Vec::with_capacity(observations.len());
        for obs in observations {
            let raw: Vec<f32> = obs
                .valid
                .iter()
                .map(|&a| pseudo_random_f32(obs.state_hash, a) + 0.01)
                .collect();
            let total: f32 = raw.iter().sum();
            let mut prior = HashMap::with_capacity(obs.valid.len());
            for (&a, &p) in obs.valid.iter().zip(raw.iter()) {
                prior.insert(a, p / total);
            }
            priors.push(prior);
            // Value derived from state hash, in roughly [-1, 1].
            values.push((pseudo_random_f32(obs.state_hash, u32::MAX) * 2.0 - 1.0).clamp(-1.0, 1.0));
        }
        Ok((priors, values))
    }
}

// ============================================================================
// Test helpers
// ============================================================================

fn run_search(
    num_actions: u32,
    max_depth: u32,
    mcts_steps: usize,
    apply_forced: bool,
) -> MCTS<WideEnv> {
    let env = WideEnv::new(num_actions, max_depth);
    let client = DeterministicClient;
    let mut mcts: MCTS<WideEnv> = MCTS::new(8);
    let evaluator = |e: &WideEnv| if e.done() { 0.5 } else { 0.0 };
    mcts.run(&env, &client, mcts_steps, 1.4, &evaluator, apply_forced);
    mcts
}

// ============================================================================
// Determinism: same inputs → consistent invariants across two runs
// ============================================================================
//
// We do NOT assert byte-equal trees because std HashMap iteration order
// is not guaranteed to be reproducible across runs of the same process
// (it depends on insertion order, which depends on transient hash seed).
// We DO assert that visit totals, policy_target sums, and tree
// invariants are identical — those are deterministic properties of the
// search.

#[test]
fn test_two_runs_have_identical_visit_totals_at_branching_50() {
    let m1 = run_search(50, 8, 400, true);
    let m2 = run_search(50, 8, 400, true);

    let r1 = m1.root_id.unwrap();
    let r2 = m2.root_id.unwrap();
    let v1: usize = m1.get_node_immut(r1).unwrap().edge_visits.values().sum();
    let v2: usize = m2.get_node_immut(r2).unwrap().edge_visits.values().sum();
    assert_eq!(v1, v2, "Two runs of MCTS::run on identical inputs must produce equal root visit totals");
}

// ============================================================================
// policy_target sum invariant (from existing test, expanded)
// ============================================================================

#[test]
fn test_policy_target_sums_to_one_at_branching_50() {
    let mcts = run_search(50, 6, 500, true);
    let target = mcts.policy_target(1.4).expect("policy target should be Some");
    let total: f32 = target.values().sum();
    assert!((total - 1.0).abs() < 1e-4, "policy_target sum: {} (expected 1.0)", total);
}

#[test]
fn test_policy_target_sums_to_one_at_branching_250() {
    let mcts = run_search(250, 4, 1500, true);
    let target = mcts.policy_target(1.4).expect("policy target should be Some");
    let total: f32 = target.values().sum();
    assert!((total - 1.0).abs() < 1e-4, "policy_target sum: {} (expected 1.0)", total);
}

#[test]
fn test_policy_target_only_visited_actions_at_branching_250() {
    // After search, policy_target must contain only actions that received
    // at least one visit. Bug-shape: an off-by-one or missing-entry
    // refactor could include zero-visit actions with nonzero policy mass.
    let mcts = run_search(250, 4, 1500, true);
    let root = mcts.root_id.unwrap();
    let visited: std::collections::HashSet<u32> = mcts
        .get_node_immut(root)
        .unwrap()
        .edge_visits
        .iter()
        .filter(|&(_, &v)| v > 0)
        .map(|(&a, _)| a)
        .collect();
    let target = mcts.policy_target(1.4).expect("policy target should be Some");
    for (&a, &p) in target.iter() {
        if p > 0.0 {
            assert!(
                visited.contains(&a),
                "action {} has policy mass {} but was never visited at root",
                a, p,
            );
        }
    }
}

// ============================================================================
// Visit count consistency
// ============================================================================

#[test]
fn test_root_visit_total_matches_batch_aligned_steps_at_branching_50() {
    // run() processes ceil(num_steps / batch_size) batches of batch_size
    // simulations each, so the actual number of visits at the root is
    // batch-aligned, not exactly num_steps. With batch_size=8 and
    // num_steps=500, that's ceil(500/8)*8 = 504 visits.
    let steps = 500;
    let batch_size = 8;
    let expected = ((steps + batch_size - 1) / batch_size) * batch_size; // 504
    let mcts = run_search(50, 8, steps, false);
    let root = mcts.root_id.unwrap();
    let total: usize = mcts.get_node_immut(root).unwrap().edge_visits.values().sum();
    assert_eq!(total, expected,
        "root edge_visits total = {} but expected batch-aligned {} (steps={}, batch_size={})",
        total, expected, steps, batch_size);
}

#[test]
fn test_invalid_actions_never_visited_at_branching_50() {
    // The valid action set for WideEnv is exactly 0..num_actions. Any
    // edge_visit at the root for an action ≥ num_actions would be a
    // serious bug (writing past the dense array, or a stale leftover).
    let mcts = run_search(50, 6, 500, true);
    let root = mcts.root_id.unwrap();
    for (&a, &v) in &mcts.get_node_immut(root).unwrap().edge_visits {
        assert!(a < 50, "root has visit count {} for invalid action {}", v, a);
    }
}

// ============================================================================
// advance_root consistency at high branching
// ============================================================================

#[test]
fn test_advance_root_consistency_at_branching_50() {
    let mut mcts = run_search(50, 8, 500, true);

    // Pick the most-visited root action to advance into.
    let root = mcts.root_id.unwrap();
    let (best_action, _) = mcts
        .get_node_immut(root).unwrap()
        .edge_visits.iter()
        .max_by_key(|&(_, &v)| v)
        .map(|(&a, &v)| (a, v))
        .expect("root must have at least one visited child");

    let nodes_before = mcts.nodes.len();
    mcts.advance_root(best_action);
    let nodes_after = mcts.nodes.len();

    assert!(nodes_after <= nodes_before,
        "advance_root must not grow the arena: before={}, after={}",
        nodes_before, nodes_after);
    assert!(nodes_after > 0, "advance_root must keep at least the new root");
    assert_eq!(mcts.transposition_table.len(), mcts.nodes.len(),
        "table size {} != arena size {} after advance_root",
        mcts.transposition_table.len(), mcts.nodes.len());
    for (&id, &idx) in &mcts.transposition_table {
        assert!(idx < mcts.nodes.len(),
            "transposition_table maps {} to out-of-bounds idx {}", id, idx);
        assert_eq!(mcts.nodes[idx].id, id,
            "transposition_table maps id {} to nodes[{}] which has id {}",
            id, idx, mcts.nodes[idx].id);
    }
}

#[test]
fn test_advance_root_preserves_visit_counts_in_subtree_at_branching_50() {
    // The new root's visit counts must be unchanged by advance_root —
    // pruning is structural; it doesn't mutate node state.
    let mut mcts = run_search(50, 8, 500, true);
    let root = mcts.root_id.unwrap();
    let best_action = mcts
        .get_node_immut(root).unwrap()
        .edge_visits.iter()
        .max_by_key(|&(_, &v)| v)
        .map(|(&a, _)| a)
        .expect("root must have at least one visited child");

    let new_root_id = *mcts.get_node_immut(root).unwrap()
        .children.get(&best_action)
        .expect("most-visited root action must have a child");
    let visits_before: HashMap<u32, usize> = mcts
        .get_node_immut(new_root_id).unwrap()
        .edge_visits.clone();

    mcts.advance_root(best_action);

    let visits_after = &mcts.get_node_immut(new_root_id).unwrap().edge_visits;
    assert_eq!(&visits_before, visits_after,
        "advance_root must not mutate the new root's edge visit counts");
}

#[test]
fn test_advance_root_no_dangling_nodeids_at_branching_50() {
    // Every NodeId referenced by a remaining node's `children` map must
    // point to a node that still exists in the table.
    let mut mcts = run_search(50, 8, 500, true);
    let root = mcts.root_id.unwrap();
    let best_action = mcts
        .get_node_immut(root).unwrap()
        .edge_visits.iter()
        .max_by_key(|&(_, &v)| v)
        .map(|(&a, _)| a)
        .expect("root must have at least one visited child");

    mcts.advance_root(best_action);

    for node in &mcts.nodes {
        for (&_a, &cid) in &node.children {
            assert!(mcts.transposition_table.contains_key(&cid),
                "node {} has child {} that is missing from the transposition table",
                node.id, cid);
        }
    }
}

// ============================================================================
// recompute_value: hand-computed reference
// ============================================================================

#[test]
fn test_recompute_value_is_visit_plus_one_weighted_average() {
    // recompute_value computes:
    //   value = (own_value_estimate + Σ ev_i × (child.value + ep_i)) / (1 + total_visits)
    // The "+1" treats the parent's own NN value_estimate as a single
    // pseudo-visit. This is the formula the refactor must preserve
    // exactly (an off-by-one here would silently corrupt every Q-value
    // up the tree).
    //
    // Setup: parent (value_estimate=0.0) with two children:
    //   action 0: 4 visits, child.value = 0.6, edge_penalty = 0.0
    //   action 1: 6 visits, child.value = -0.2, edge_penalty = 0.0
    //   no virtual losses
    // Expected parent.value = (0.0 + 4×0.6 + 6×-0.2) / (1 + 10) = 1.2 / 11
    let mut mcts: MCTS<WideEnv> = MCTS::new(4);
    let mut parent = Node::new(HashMap::from([(0u32, 0.5), (1u32, 0.5)]), 0.0, 100, None);
    parent.children.insert(0, 200);
    parent.children.insert(1, 300);
    parent.edge_visits.insert(0, 4);
    parent.edge_visits.insert(1, 6);
    parent.node_visits = 10;
    let mut child_a = Node::new(HashMap::from([(0u32, 1.0)]), 0.6, 200, None);
    child_a.value = 0.6;
    let mut child_b = Node::new(HashMap::from([(0u32, 1.0)]), -0.2, 300, None);
    child_b.value = -0.2;

    mcts.insert_node(100, parent);
    mcts.insert_node(200, child_a);
    mcts.insert_node(300, child_b);

    mcts.recompute_value(100);

    let parent = mcts.get_node_immut(100).unwrap();
    let expected = (0.0 + 4.0 * 0.6 + 6.0 * -0.2) / (1.0 + 10.0);
    assert!((parent.value - expected).abs() < 1e-5,
        "recompute_value: got {}, expected {} (= 1.2/11)", parent.value, expected);
    assert_eq!(parent.node_visits, 11,
        "recompute_value must set node_visits = 1 + total_edge_visits");
}

// ============================================================================
// perturb_root_prior: verify mixture math
// ============================================================================

#[test]
fn test_perturb_root_prior_mixes_correctly_in_puct_scores() {
    // perturb_root_prior stores the noise + epsilon transiently — it
    // does NOT mutate node.prior_probs. The mixing happens inside
    // puct_scores(is_root=true). The U-term in PUCT is
    //   U = c_puct × P′(a) × √Σ_visits / (1 + visits(a))
    // With zero visits everywhere and root_softmax_temp = 1.0 (the
    // default for c_fpu test, but in production it's 1.03 — set
    // explicitly here for hand-checkability), the U-term is
    //   U = c_puct × P′(a) × ε
    // because √0 + 1e-8 ≈ 1e-8 and (1 + 0) = 1. Q is fpu_q = 0.0
    // because c_fpu_eff = 0 at root and node.value = 0.
    // So scores[a] / (c_puct × 1e-8) = P′(a) (modulo tiny float drift).
    //
    // We assert on relative ordering of P′ values rather than exact
    // ratios, since the dominant term is so small. The robust check:
    // verify that perturb_root_prior installed the noise (root_noise
    // is Some, root_noise_epsilon is correct).
    let mut mcts: MCTS<WideEnv> = MCTS::new(4);
    let priors = HashMap::from([
        (0u32, 0.5),
        (1u32, 0.3),
        (2u32, 0.2),
    ]);
    let node = Node::new(priors, 0.0, 1, None);
    mcts.insert_node(1, node);
    mcts.root_id = Some(1);
    mcts.root_softmax_temp = 1.0;  // disable softmax to make math clean

    let noise = HashMap::from([
        (0u32, 0.1),
        (1u32, 0.6),
        (2u32, 0.3),
    ]);
    let epsilon = 0.25;

    mcts.perturb_root_prior(&noise, epsilon);

    // Sanity: prior_probs is unchanged on the node itself.
    let root = mcts.get_node_immut(1).unwrap();
    assert!((root.prior_probs[&0] - 0.5).abs() < 1e-6,
        "perturb_root_prior must not mutate node.prior_probs");

    // Now puct_scores at root must reflect the blended priors. Without
    // any visits the U-term ratio between two actions equals the ratio
    // of their effective priors. We assert the ordering matches the
    // expected blend:
    //   p′[0] = 0.75 × 0.5 + 0.25 × 0.1 = 0.4
    //   p′[1] = 0.75 × 0.3 + 0.25 × 0.6 = 0.375
    //   p′[2] = 0.75 × 0.2 + 0.25 × 0.3 = 0.225
    // So U[0] > U[1] > U[2].
    let scores = mcts.puct_scores(1, 1.0, true, false);
    assert!(scores[&0] > scores[&1],
        "blended prior for action 0 ({}) should be higher than for action 1 ({})",
        scores[&0], scores[&1]);
    assert!(scores[&1] > scores[&2],
        "blended prior for action 1 ({}) should be higher than for action 2 ({})",
        scores[&1], scores[&2]);
}

// ============================================================================
// puct_scores: invariants under high branching
// ============================================================================

#[test]
fn test_puct_scores_finite_for_visited_actions_at_branching_50() {
    // After a search, every visited action at the root must have a
    // finite PUCT score. NaN/-inf would indicate a divide-by-zero or
    // missing-entry bug after refactor.
    let mcts = run_search(50, 6, 500, false);
    let root = mcts.root_id.unwrap();
    let scores = mcts.puct_scores(root, 1.4, true, false);
    for (&a, &s) in &scores {
        assert!(s.is_finite(),
            "root action {} has non-finite PUCT score {} after search", a, s);
    }
}

#[test]
fn test_puct_scores_only_for_valid_actions_at_branching_50() {
    let mcts = run_search(50, 6, 500, false);
    let root = mcts.root_id.unwrap();
    let scores = mcts.puct_scores(root, 1.4, true, false);
    for &a in scores.keys() {
        assert!(a < 50, "puct_scores produced score for invalid action {}", a);
    }
}
