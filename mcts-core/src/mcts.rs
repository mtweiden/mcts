use crate::node::{Node, NodeId};
use crate::environment::{Act, Environment};
use crate::inference::InferenceClient;
use std::collections::HashMap;
use rustc_hash::FxHashMap;


/// ----------------------------------------------------------------------------
/// A pending inference payload
/// ----------------------------------------------------------------------------
struct PendingInference<E: Environment> {
    path: Vec<(NodeId, E::Act)>,
    parent_id: NodeId,
    action: E::Act,
    env: E,
}


/// ----------------------------------------------------------------------------
/// Monte Carlo Tree Search
/// ----------------------------------------------------------------------------
///
/// Several improvements from the KataGo paper are implemented here:
///   Wu, D.J. (2020). "Accelerating Self-Play Learning in Go."
///   arXiv:1902.10565. Referred to as [Wu 2020] throughout this file.
pub struct MCTS<E: Environment> {
    /// The NodeId of the current root of the search tree.
    pub root_id: Option<NodeId>,
    /// Arena-style storage for all nodes.
    /// FxHashMap (linear-probing, FxHash) instead of std HashMap (SipHash)
    /// because NodeId is already a hash (random 64-bit u64) — running it
    /// through SipHash a second time is wasted work. Profiling showed
    /// transposition_table lookups in `puct_scores` and `select_leaf` were
    /// a meaningful fraction of MCTS step time at branching factor ≥ 100.
    pub transposition_table: FxHashMap<NodeId, usize>,
    pub nodes: Vec<Node<E::Act>>,
    pub batch_size: usize,

    /// First-play urgency (FPU) reduction coefficient for non-root nodes.
    ///
    /// When a child `c` of a non-root node `n` has never been visited, its
    /// Q-value fallback is:
    ///
    ///   Q_fpu = V(n) − c_fpu × √P_explored
    ///
    /// where `P_explored` is the sum of policy priors over already-visited
    /// children. As more of `n`'s children are explored the urgency of
    /// revisiting the remaining unexplored ones decays, focusing search.
    ///
    /// At the root `c_fpu` is always overridden to `0.0` because Dirichlet
    /// noise already ensures adequate exploration there.
    ///
    /// Default: `0.2`. Reference: [Wu 2020, §2, footnote 3].
    pub c_fpu: f32,

    /// Softmax temperature applied to the policy prior **at the root only**.
    ///
    /// Before computing PUCT scores at the root the prior is re-weighted:
    ///
    ///   P′(c) ∝ P(c)^(1/T),  then renormalised.
    ///
    /// With `T = 1.03` the effect is a very mild flattening of the prior
    /// distribution, which [Wu 2020] reports improves policy convergence
    /// stability during self-play training. Set to `1.0` to disable.
    ///
    /// Default: `1.03`. Reference: [Wu 2020, §2].
    pub root_softmax_temp: f32,

    /// Coefficient controlling the forced-playout visit floor at the root.
    ///
    /// For each root child `c` the minimum number of forced visits is:
    ///
    ///   n_forced(c) = √(k_forced × P(c) × Σ_{c′} N(c′))
    ///
    /// Any child below this threshold is assigned `PUCT = ∞`, guaranteeing
    /// it will be selected on the next playout. The exponent of `1/2` ensures
    /// forced visits decay to a zero proportion as the total visit budget
    /// grows, so they never dominate a large search.
    ///
    /// This only applies during full (forced_playouts = true) searches; fast searches
    /// (`forced_playouts= false` in [`run`]) skip forced playouts to maximise strength.
    ///
    /// Default: `2.0`. Reference: [Wu 2020, §3.2].
    pub k_forced: f32,

    /// Transient Dirichlet noise blended into root priors during PUCT scoring.
    ///
    /// Stored on the struct rather than in the node so it never permanently
    /// contaminates the transposition table. Set via [`perturb_root_prior`]
    /// before [`run`]; automatically cleared at the end of each [`run`] call.
    root_noise: Option<HashMap<E::Act, f32>>,
    root_noise_epsilon: f32,
}

impl<E: Environment> MCTS<E> {
    pub fn new(batch_size: usize) -> Self {
        Self {
            root_id: None,
            transposition_table: FxHashMap::default(),
            nodes: Vec::new(),
            batch_size,
            c_fpu: 0.2,
            root_softmax_temp: 1.03,
            k_forced: 2.0,
            root_noise: None,
            root_noise_epsilon: 0.0,
        }
    }

    pub fn default() -> Self { Self::new(8) }

    /// Run MCTS for a given number of steps from the current environment state.
    ///
    /// # Arguments
    /// * `env`                — Reference to the current state. Not mutated; all
    ///                          simulations run on internal clones.
    /// * `client`             — Inference client supplying policy priors and values.
    /// * `num_steps`          — Number of MCTS iterations. More → stronger search.
    /// * `c_puct`             — Exploration constant in the PUCT formula.
    /// * `terminal_evaluator` — Callback returning a scalar value for terminal states.
    /// * `forced_playouts`       — Whether to apply forced playouts during this search.
    ///                          Enable for full training searches; disable for fast inference.
    ///                          See *Playout Cap Randomization* below.
    ///
    /// # Playout Cap Randomization \[Wu 2020, §3.1\]
    ///
    /// Value training benefits from many short games (each supplies one independent
    /// outcome signal), while policy training requires deep search to produce
    /// non-trivial visit distributions.  These goals conflict when using a fixed
    /// playout budget.
    ///
    /// The solution is to randomly vary the budget per turn:
    /// * On a fraction `p` of turns run a **full search** (`num_steps = N`,
    ///   `forced_playouts = true`).  The inference client **should** inject Dirichlet noise.
    ///   Forced playouts are active.  Call [`policy_target`] after the search
    ///   to obtain a training-ready policy distribution.
    /// * On the remaining turns run a **fast search** (`num_steps = n ≪ N`,
    ///   `forced_playouts = false`).  The inference client **should not** inject noise.
    ///   Forced playouts are disabled, maximising move strength.  Do **not** use the
    ///   returned node's `edge_visits` as a training target.
    ///
    /// KataGo used `p = 0.25` and `(N, n) = (600, 100)` initially.
    pub fn run<F>(
        &mut self,
        env: &E,
        client: &dyn InferenceClient<E>,
        num_steps: usize,
        c_puct: f32,
        terminal_evaluator: &F,
        forced_playouts: bool,
    ) -> Node<E::Act>
        where F: Fn(&E) -> f32
    {
        // Set the current root for the search session
        let root_hash = self.get_hash(env);
        self.root_id = Some(root_hash);
        // Check if the current root node exists
        if !self.node_exists(root_hash) {
            let (priors, value) = if env.done() {
                (HashMap::new(), terminal_evaluator(env))
            } else {
                let obs = env.observation();
                let (p, v) = self.blocking_infer(&vec![obs], client);
                (self.normalize_prior(p[0].clone(), &env.valid_actions()), v[0])
            };
            self.create_node(env, priors, value);
        }

        if env.done() { return self.get_node_mut(root_hash).unwrap().clone(); }

        // Do ceiling division to determine the number of batches
        let num_batches = (num_steps + self.batch_size - 1) / self.batch_size.max(1);
        for _ in 0..num_batches {
            // Selection: collect a batch of leaf observations / metadata
            let mut leaf_batch: Vec<E::Obs> = Vec::with_capacity(self.batch_size);
            let mut pending_inferences = Vec::with_capacity(self.batch_size);

            for _ in 0..self.batch_size {
                // select_leaf clones the environment internally and returns the reached env
                let (path, parent, action, final_env, repeat) = self.select_leaf(root_hash, env, c_puct, forced_playouts);
                let obs = final_env.observation();
                // Continue so we don't add leaf nodes to the batch if no action was selected
                // (e.g. terminal state or no valid actions)
                let action = match action {
                    Some(a) => a,
                    None => continue,
                };
                // Continue if the parent is missing, which can happen if the root is not fully expanded
                let parent_id = match parent {
                    Some(p) => p,
                    None => continue,
                };

                // Handle repeats immediately
                if repeat || final_env.done() {
                    // A terminal state, expand with terminal value and empty priors
                    if !repeat {
                        let terminal_value = terminal_evaluator(&final_env);
                        self.expand(parent_id, action, final_env, HashMap::new(), terminal_value);
                    }
                    // Do backprop immediately
                    self.backpropagate(&path, repeat);
                    continue;
                }
                leaf_batch.push(obs);
                pending_inferences.push(PendingInference { path, parent_id, action, env: final_env });
            }

            if leaf_batch.is_empty() { continue; }

            // --- Batched Inference ---
            let (prior_batch, value_batch) = self.blocking_infer(&leaf_batch, client);

            // Expansion & Backpropagation
            let n = prior_batch.len().min(value_batch.len()).min(pending_inferences.len());
            for i in 0..n {
                let pending = &pending_inferences[i];
                let priors = prior_batch[i].clone();
                let value = value_batch[i];

                self.expand(pending.parent_id, pending.action, pending.env.clone(), priors, value);
                self.backpropagate(&pending.path, false);
            }
        }
        // Clear transient noise so it doesn't leak into future searches or
        // contaminate policy_target() calculations.
        self.root_noise = None;
        self.root_noise_epsilon = 0.0;

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
        let new_root_id_opt = old_root_node.children
            .get(action.to_action_index())
            .and_then(|opt| *opt);
        if let Some(new_root_id) = new_root_id_opt {
            self.root_id = Some(new_root_id);

            // Collect all NodeIds reachable from new_root_id via a BFS.
            // A visited HashSet is required to terminate on cycles (transposition table
            // graphs can have cycles when states repeat, e.g. sliding-tile puzzles).
            let mut reachable: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
            let mut queue: std::collections::VecDeque<NodeId> = std::collections::VecDeque::new();
            queue.push_back(new_root_id);
            while let Some(id) = queue.pop_front() {
                if !reachable.insert(id) {
                    continue; // already visited
                }
                if let Some(idx) = self.transposition_table.get(&id) {
                    if let Some(node) = self.nodes.get(*idx) {
                        // Walk dense children: only Some slots are real.
                        for slot in &node.children {
                            if let Some(child_id) = *slot {
                                if !reachable.contains(&child_id) {
                                    queue.push_back(child_id);
                                }
                            }
                        }
                    }
                }
            }

            // In-place compaction: drop unreachable Nodes from self.nodes
            // without allocating a new Vec, then rebuild the transposition
            // table over the surviving indices. After dense-Vec refactor
            // each Node is ~10 KB (5 dense Vecs of length num_actions);
            // pre-rewrite this path drained every Node into a freshly
            // allocated Vec, which means the survivors' Node headers were
            // moved twice (drain → push). retain_mut moves them at most
            // once and avoids the new-Vec allocation entirely. The dropped
            // Nodes' inner dense Vecs are deallocated by Drop the same as
            // before, so memory behavior is unchanged.
            self.nodes.retain(|n| reachable.contains(&n.id));

            // Rebuild transposition_table fresh over the new (possibly
            // compacted) indices. Same big-O as before; the
            // with_capacity hint preserves the previous allocation
            // shape so rehashing during repopulation is avoided.
            self.transposition_table.clear();
            self.transposition_table.reserve(self.nodes.len());
            for (i, n) in self.nodes.iter().enumerate() {
                self.transposition_table.insert(n.id, i);
            }
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
            Node::new_terminal(node_id, value, repr)
        } else {
            Node::new(env.num_actions(), priors, value.clamp(-1.0, 1.0), node_id, repr)
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
        // 1. Snapshot parent data immutably and extract the data we need
        let (virtual_loss_counts, node_value_estimate, child_data) = {
            let parent = match self.get_node_immut(node_id) {
                Some(p) => p,
                None => return,
            };
            // Sum the entire dense Vec — invalid slots are zero so they
            // contribute nothing to the total.
            let vl: usize = parent.virtual_losses.iter().sum();
            // Walk valid_actions and collect (child_id, edge_visits, penalty)
            // for each slot whose child is expanded. Iterating the dense
            // children Vec works too, but valid_actions short-circuits the
            // None slots without bounds checks.
            let data: Vec<(NodeId, usize, f32)> = parent.valid_actions.iter()
                .filter_map(|&a| {
                    let idx = a.to_action_index();
                    parent.children.get(idx).and_then(|c| *c).map(|child_id| {
                        let ev = parent.edge_visits.get(idx).copied().unwrap_or(0);
                        let ep = parent.edge_penalties.get(idx).copied().unwrap_or(0.0);
                        (child_id, ev, ep)
                    })
                })
                .collect();
            (vl, parent.value_estimate, data)
        };  // Immutable borrows end here

        let edge_visit_count: usize = child_data.iter().map(|(_, ev, _)| *ev).sum();
        let total_edge_visits = edge_visit_count + virtual_loss_counts;

        if total_edge_visits == 0 {
            // No visits yet, keep the parent's value estimate
            if let Some(node) = self.get_node_mut(node_id) {
                node.node_visits = 1;
                node.value = node_value_estimate;
            }
            return;
        }

        // 2. Accumulate weighted child values
        let mut acc: f32 = 0.0;
        for (child_id, ev, ep) in child_data {
            if ev == 0 { continue; }

            if let Some(child) = self.get_node_immut(child_id) {
                acc += (ev as f32) * (child.value + ep);
            }
        }
        // 3. Single mutable borrow to update the parent node
        if let Some(node) = self.get_node_mut(node_id) {
            node.node_visits = 1 + total_edge_visits;
            node.value = (node_value_estimate + acc) / (node.node_visits as f32);
        }

    }

    /// Compute PUCT scores for all actions from a given node.
    ///
    /// Three improvements from [Wu 2020] are applied here:
    ///
    /// ## First-Play Urgency (FPU) \[Wu 2020, §2, footnote 3\]
    ///
    /// For an action whose child is not yet in the transposition table, the
    /// Q-value fallback is:
    ///
    ///   Q_fpu = V(parent) − c_fpu_eff × √P_explored
    ///
    /// where `P_explored` is the total prior mass of already-visited children.
    /// As more children are explored the penalty grows, making it progressively
    /// less attractive to keep revisiting unexplored corners of the tree.
    ///
    /// At the root `c_fpu_eff = 0.0` because Dirichlet noise already provides
    /// exploration; at all other nodes `c_fpu_eff = self.c_fpu` (default `0.2`).
    ///
    /// ## Root Softmax Temperature \[Wu 2020, §2\]
    ///
    /// When `is_root` is `true` the policy prior is re-weighted before scoring:
    ///
    ///   P′(c) ∝ P(c)^(1/T),  renormalised  (T = `self.root_softmax_temp`, default 1.03)
    ///
    /// This mildly flattens the distribution, improving policy convergence
    /// stability. The effect is negligible at T ≈ 1 but accumulates over
    /// millions of self-play games.
    ///
    /// ## Forced Playouts \[Wu 2020, §3.2\]
    ///
    /// When both `is_root` and `apply_forced` are `true`, any root child with
    /// fewer actual visits than
    ///
    ///   n_forced(c) = √(k_forced × P(c) × N_total)
    ///
    /// is assigned a score of `f32::INFINITY`, forcing the next playout to
    /// visit it. This prevents a Dirichlet-noise-suggested move from being
    /// abandoned after an initially poor evaluation before it has been given a
    /// fair chance.  The `√` exponent ensures forced visits shrink to a zero
    /// *proportion* of all visits as the budget grows.
    pub fn puct_scores(
        &self,
        node_id: NodeId,
        c_puct: f32,
        is_root: bool,
        apply_forced: bool,
    ) -> Vec<f32> {
        let node = match self.get_node_immut(node_id) {
            Some(n) => n,
            None => return Vec::new(),
        };

        let total_visits: usize = node.edge_visits.iter().sum();
        let sqrt_total = (total_visits as f32).sqrt() + 1e-8;

        // --- FPU ---
        // Sum the prior mass of children that have received at least one visit.
        // At the root c_fpu is 0: Dirichlet noise handles exploration there.
        // Reference: [Wu 2020, §2, footnote 3].
        let c_fpu_eff: f32 = if is_root { 0.0 } else { self.c_fpu };
        let p_explored: f32 = node.valid_actions
            .iter()
            .filter(|&&a| node.edge_visits[a.to_action_index()] > 0)
            .map(|&a| node.prior_probs[a.to_action_index()])
            .sum();
        let fpu_q = node.value - c_fpu_eff * p_explored.sqrt();

        // --- Root softmax temperature + transient Dirichlet noise ---
        // For root nodes that need either Dirichlet noise blending or a
        // softmax temperature != 1, allocate ONE owned buffer indexed by
        // action.to_action_index() and apply both transforms in place.
        // For all other nodes, read priors directly from
        // `node.prior_probs` — no allocation needed.
        // Reference: [Wu 2020, §2].
        let needs_noise = is_root && self.root_noise.is_some();
        let needs_softmax = is_root && (self.root_softmax_temp - 1.0).abs() > 1e-6;
        let effective_buf: Option<Vec<f32>> = if needs_noise || needs_softmax {
            let mut buf = node.prior_probs.clone();

            if let Some(noise) = &self.root_noise {
                let eps = self.root_noise_epsilon;
                for &a in &node.valid_actions {
                    let idx = a.to_action_index();
                    let n = noise.get(&a).copied().unwrap_or(0.0);
                    buf[idx] = (1.0 - eps) * node.prior_probs[idx] + eps * n;
                }
                let total: f32 = node.valid_actions.iter()
                    .map(|&a| buf[a.to_action_index()])
                    .sum();
                if total > 0.0 {
                    for &a in &node.valid_actions {
                        buf[a.to_action_index()] /= total;
                    }
                }
            }

            if needs_softmax {
                let inv_temp = 1.0 / self.root_softmax_temp;
                for &a in &node.valid_actions {
                    let idx = a.to_action_index();
                    buf[idx] = buf[idx].max(1e-30).powf(inv_temp);
                }
                let sum: f32 = node.valid_actions.iter()
                    .map(|&a| buf[a.to_action_index()])
                    .sum();
                if sum > 0.0 {
                    for &a in &node.valid_actions {
                        buf[a.to_action_index()] /= sum;
                    }
                }
            }
            Some(buf)
        } else {
            None
        };

        // Score Vec indexed by action.to_action_index(). Invalid action
        // slots stay at NEG_INFINITY so a downstream argmax over the
        // dense Vec can never pick an invalid action.
        let mut scores = vec![f32::NEG_INFINITY; node.num_actions];

        for &action in &node.valid_actions {
            let idx = action.to_action_index();
            let this_edge_visits = node.edge_visits[idx];
            let num_virtual_losses = node.virtual_losses[idx];
            let adjusted_visits = this_edge_visits + num_virtual_losses;
            let penalty = node.edge_penalties[idx];
            let prior = match &effective_buf {
                Some(buf) => buf[idx],
                None => node.prior_probs[idx],
            };

            // --- Forced playouts ---
            // If this root child is under-visited, force selection by returning ∞.
            // Only applies during full (forced_playouts = true) searches to avoid
            // wasting fast search playouts on exploratory moves.
            // Reference: [Wu 2020, §3.2].
            if is_root && apply_forced && total_visits > 0 {
                let n_forced = (self.k_forced * prior * total_visits as f32).sqrt();
                if (this_edge_visits as f32) < n_forced {
                    scores[idx] = f32::INFINITY;
                    continue;
                }
            }

            // --- Q-value ---
            // Use the child's backed-up value when it is in the transposition table.
            // If the child has never been reached at all, apply the FPU fallback.
            // Reference: [Wu 2020, §2, footnote 3].
            let q_value = match node.children.get(idx).and_then(|c| *c) {
                Some(child_id) => self.get_node_immut(child_id)
                    .map(|c| c.value + penalty)
                    .unwrap_or(fpu_q + penalty),
                None => fpu_q + penalty,
            };

            let u_value = c_puct * prior * (sqrt_total / (1.0 + adjusted_visits as f32));
            scores[idx] = q_value + u_value;
        }
        scores
    }

    pub fn select_action_puct(
        &self,
        node_id: NodeId,
        c_puct: f32,
        is_root: bool,
        apply_forced: bool,
    ) -> Option<E::Act> {
        let scores = self.puct_scores(node_id, c_puct, is_root, apply_forced);
        if scores.is_empty() {
            return None;
        }
        // Iterate valid_actions and look up dense scores. Invalid action
        // slots are NEG_INFINITY, so a global argmax over the Vec would
        // also work — but iterating valid_actions is cheaper at branching
        // factor 251 with typically all-valid actions.
        let node = self.get_node_immut(node_id)?;
        node.valid_actions.iter()
            .copied()
            .max_by(|&a, &b| {
                let sa = scores[a.to_action_index()];
                let sb = scores[b.to_action_index()];
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    pub fn select_action(&self, node: &Node<E::Act>) -> Option<E::Act> {
        node.select_action()
    }

    pub fn add_child(&mut self, parent_id: NodeId, action: E::Act, child_id: NodeId) {
        if let Some(parent) = self.get_node_mut(parent_id) {
            let idx = action.to_action_index();
            if idx < parent.children.len() {
                parent.children[idx] = Some(child_id);
            }
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
    /// `(path, parent, action_taken, final_env, repeat_detected)`
    ///
    /// `final_env` is the environment state after taking all actions along the
    /// path. `forced_playouts` is forwarded to [`select_action_puct`] to enable or
    /// disable forced playouts at the root (see [`run`] for details).
    pub fn select_leaf(
        &mut self,
        root_id: NodeId,
        env: &E,
        c_puct: f32,
        forced_playouts: bool,
    ) -> (Vec<(NodeId, E::Act)>, Option<NodeId>, Option<E::Act>, E, bool) {
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

            // Root-specific PUCT behaviours (FPU override, softmax temperature,
            // forced playouts) are only applied on the first step of each playout.
            let is_root = node_id == root_id;

            // choose action via PUCT
            let chosen = match self.select_action_puct(node_id, c_puct, is_root, forced_playouts) {
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
                let idx = action.unwrap().to_action_index();
                match parent_node.children.get(idx).and_then(|c| *c) {
                    Some(cid) => { node_id = cid; }
                    None => break,
                }
            } else {
                break;
            }
        }

        (path, parent, action, game, repeat_detected)
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
            let idx = action.to_action_index();
            if idx < parent.children.len() {
                match parent.children[idx] {
                    Some(existing_id) => {
                        if existing_id != leaf_id {
                            eprintln!("[Mismatched IDs] existing {} != leaf {}", existing_id, leaf_id);
                        }
                    }
                    None => {
                        parent.children[idx] = Some(leaf_id);
                        // Alias detection: warn if any other slot already
                        // points at the same leaf_id (would mean two
                        // actions in the same parent transposed to a
                        // single state).
                        for (i, slot) in parent.children.iter().enumerate() {
                            if i != idx {
                                if let Some(cid) = *slot {
                                    if cid == leaf_id {
                                        eprintln!("[Alias detected] action index {} - {}", i, idx);
                                    }
                                }
                            }
                        }
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
                let idx = action.to_action_index();
                if idx < node.edge_visits.len() {
                    node.edge_visits[idx] += 1;
                }
            }
            self.recompute_value(node_hash);
        }
    }

    /// Compute the policy training target from the root's visit distribution,
    /// with forced-playout visits subtracted.
    ///
    /// # Policy Target Pruning \[Wu 2020, §3.2\]
    ///
    /// Using raw visit counts as the training target would teach the network to
    /// imitate forced exploratory visits — visits that were added deliberately
    /// to evaluate noise-suggested moves, not because those moves were good.
    /// This method removes those spurious visits before producing a training
    /// distribution:
    ///
    /// 1. Identify `c*`, the root child with the most visits.
    /// 2. For every other child `c`, compute the forced-visit floor:
    ///    `n_forced(c) = √(k_forced × P(c) × N_total)`
    ///    Determine the largest number of visits `s` that can be subtracted
    ///    without making `PUCT(c, N(c) − s) ≥ PUCT(c*)` (i.e. without making
    ///    `c` look as strong as the best move once the forced visits are gone).
    ///    Subtract `min(n_forced, s)` visits from `c`.
    /// 3. Children pruned down to exactly **1** visit are removed entirely.
    /// 4. Renormalise the remaining visit counts to a probability distribution.
    ///
    /// This decouples the policy target from the MCTS exploration dynamics,
    /// allowing the two to be optimised independently.
    ///
    /// Returns `None` if the root does not exist or has received no visits.
    /// **Only meaningful after a full search** (`forced_playouts = true` in [`run`]).
    ///
    /// Reference: [Wu 2020, §3.2].
    pub fn policy_target(&self, c_puct: f32) -> Option<HashMap<E::Act, f32>> {
        let root_id = self.root_id?;
        let node = self.get_node_immut(root_id)?;

        let n_total: usize = node.edge_visits.iter().sum();
        if n_total == 0 {
            return None;
        }
        let sqrt_n_total = (n_total as f32).sqrt();

        // FPU at root is 0 (Dirichlet noise provides exploration).
        let fpu_q = node.value;

        // Pre-compute Q(c) for each action using the transposition table.
        // Q(c) = backed-up child value if available, otherwise the FPU fallback.
        let q_values: HashMap<E::Act, f32> = node.valid_actions.iter().map(|&action| {
            let idx = action.to_action_index();
            let penalty = node.edge_penalties[idx];
            let q = match node.children.get(idx).and_then(|c| *c) {
                Some(child_id) => self.get_node_immut(child_id)
                    .map(|c| c.value + penalty)
                    .unwrap_or(fpu_q + penalty),
                None => fpu_q + penalty,
            };
            (action, q)
        }).collect();

        // PUCT scores at current visit counts.
        // apply_forced=false so we get clean finite scores for the comparison.
        let puct_scores = self.puct_scores(root_id, c_puct, true, false);

        // Step 1: find c* — the most-visited root child. Iterate
        // valid_actions and look up dense visit counts.
        let best_action = {
            let mut best: Option<(E::Act, usize)> = None;
            for &a in &node.valid_actions {
                let v = node.edge_visits[a.to_action_index()];
                match best {
                    Some((_, bv)) if bv >= v => {}
                    _ => best = Some((a, v)),
                }
            }
            best.map(|(a, _)| a)?
        };
        let puct_best = puct_scores
            .get(best_action.to_action_index())
            .copied()
            .unwrap_or(f32::NEG_INFINITY);

        // Step 2: build the pruned visit map, starting from all visited children.
        let mut pruned: HashMap<E::Act, usize> = node.valid_actions.iter()
            .filter_map(|&a| {
                let v = node.edge_visits[a.to_action_index()];
                if v > 0 { Some((a, v)) } else { None }
            })
            .collect();

        for (&action, visits) in pruned.iter_mut() {
            if action == best_action {
                continue; // never modify the best child's count
            }
            let prior = node.prior_probs
                .get(action.to_action_index())
                .copied()
                .unwrap_or(0.0);
            if prior == 0.0 {
                continue;
            }

            let n_forced = (self.k_forced * prior * n_total as f32).sqrt();
            let n_forced_int = n_forced.floor() as usize;
            if n_forced_int == 0 {
                continue;
            }

            // Maximum visits we can subtract while keeping PUCT(c, pruned) < PUCT(c*).
            //
            // PUCT(c, n) = Q(c) + c_puct × P(c) × √N_total / (1 + n)
            // Setting this equal to PUCT(c*) and solving for n:
            //   n_boundary = c_puct × P(c) × √N_total / (PUCT(c*) − Q(c)) − 1
            //
            // We may subtract at most floor(N(c) − n_boundary) visits.
            // If Q(c) ≥ PUCT(c*), c is already dominant on Q-value alone and
            // we must not subtract anything.
            let q_c = q_values.get(&action).copied().unwrap_or(fpu_q);
            let puct_diff = puct_best - q_c;

            let max_subtract = if puct_diff <= 0.0 {
                0
            } else {
                let n_boundary = c_puct * prior * sqrt_n_total / puct_diff - 1.0;
                (*visits as f32 - n_boundary).floor().max(0.0) as usize
            };

            let subtract = n_forced_int.min(max_subtract);
            *visits = visits.saturating_sub(subtract);
        }

        // Step 3: remove children that have been pruned to exactly 1 visit.
        pruned.retain(|_, &mut v| v > 1);

        if pruned.is_empty() {
            // Edge case: every child was pruned away. Keep the best action only.
            return Some(HashMap::from([(best_action, 1.0_f32)]));
        }

        // Step 4: normalise to a probability distribution.
        let total: f32 = pruned.values().copied().sum::<usize>() as f32;
        Some(pruned.into_iter().map(|(a, v)| (a, v as f32 / total)).collect())
    }

    /// Register a pre-computed noise distribution to be blended into the root
    /// prior **during the next [`run`] call**.
    ///
    /// `P′(c) = (1 − epsilon) × P(c) + epsilon × noise(c)`
    ///
    /// The noise is stored transiently on the MCTS struct and applied only
    /// inside [`puct_scores`] when scoring the root node, so it never
    /// permanently modifies `node.prior_probs` in the transposition table.
    /// This prevents noise from one search turn contaminating future searches
    /// or corrupting [`policy_target`] calculations (which rely on the original
    /// network priors to compute `n_forced`).
    ///
    /// Call this **before** [`run`] on full-search (training) turns. The noise
    /// is automatically cleared at the end of [`run`].  The `noise` map should
    /// already be normalised (e.g. sampled from a Dirichlet distribution over
    /// the legal actions).
    ///
    /// Reference: [Wu 2020, §2].
    pub fn perturb_root_prior(&mut self, noise: &HashMap<E::Act, f32>, epsilon: f32) {
        self.root_noise = Some(noise.clone());
        self.root_noise_epsilon = epsilon;
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

        fn num_actions(&self) -> usize { 2 }

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
        let mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        assert_eq!(mcts.batch_size, 4);
    }

    #[test]
    fn test_insert_and_get_node() {
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        let priors = HashMap::from([(0u8, 0.5), (1u8, 0.5)]);
        let node = Node::new(2, priors, 0.42, 1, None);
        mcts.insert_node(1, node);
        let node = mcts.get_node_immut(1).unwrap();
        assert!((node.value - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_run_expands_tree() {
        let env = NumberLineEnv::new(3);
        let client = UniformClient;
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        let evaluator = |e: &NumberLineEnv| if e.done() { 1.0 } else { 0.0 };
        let root = mcts.run(&env, &client, 100, 1.4, &evaluator, true);
        assert!(mcts.node_exists(root.id));
        assert!(mcts.nodes.len() > 1);
    }

    // -------------------------------------------------------------------------
    // Tests for FPU, forced playouts, softmax temperature, and policy_target
    // -------------------------------------------------------------------------

    /// Helper: build a two-action root node with one visited and one unvisited child.
    ///
    /// Action 0 has prior 0.6 and has been visited once.
    /// Action 1 has prior 0.4 and has never been visited.
    /// Parent value = 0.5.
    fn build_fpu_node() -> MCTS<NumberLineEnv> {
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        let node = Node::new(2, HashMap::from([(0u8, 0.6), (1u8, 0.4)]), 0.5, 1, None);
        mcts.insert_node(1, node);
        if let Some(n) = mcts.get_node_mut(1) {
            n.edge_visits[0] = 1;
        }
        mcts.root_id = Some(1);
        mcts
    }

    #[test]
    fn test_fpu_reduces_unvisited_q_at_non_root() {
        // With c_fpu = 0.2 and P_explored = 0.6 (action 0 visited):
        //   fpu_q = 0.5 − 0.2 × √0.6 ≈ 0.345
        // Disabling FPU (c_fpu = 0) gives the plain parent value 0.5.
        // The PUCT score for the unvisited action 1 should be lower with FPU.
        let mut mcts = build_fpu_node();
        let scores_fpu = mcts.puct_scores(1, 1.0, false, false);

        mcts.c_fpu = 0.0;
        let scores_no_fpu = mcts.puct_scores(1, 1.0, false, false);

        assert!(
            scores_fpu[1] < scores_no_fpu[1],
            "FPU should lower the Q-fallback for unvisited children: {} vs {}",
            scores_fpu[1],
            scores_no_fpu[1],
        );
    }

    #[test]
    fn test_fpu_is_zero_at_root() {
        // At the root c_fpu_eff is always 0 regardless of self.c_fpu.
        // So the score for the unvisited child should equal the no-FPU score.
        let mut mcts = build_fpu_node();
        let scores_root = mcts.puct_scores(1, 1.0, true, false);

        mcts.c_fpu = 0.0;
        let scores_no_fpu = mcts.puct_scores(1, 1.0, true, false);

        // Scores should be identical (within float precision).
        let diff = (scores_root[1] - scores_no_fpu[1]).abs();
        assert!(diff < 1e-5, "FPU should be inactive at root: diff = {}", diff);
    }

    #[test]
    fn test_forced_playouts_assign_infinity() {
        // n_forced(action 1) = √(2.0 × 0.4 × 11) ≈ 2.97
        // Action 1 has only 1 visit, so it should receive INFINITY.
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        let node = Node::new(2, HashMap::from([(0u8, 0.6), (1u8, 0.4)]), 0.0, 1, None);
        mcts.insert_node(1, node);
        if let Some(n) = mcts.get_node_mut(1) {
            n.edge_visits[0] = 10;
            n.edge_visits[1] = 1;
        }
        mcts.root_id = Some(1);

        let scores = mcts.puct_scores(1, 1.0, true, true);
        assert_eq!(scores[1], f32::INFINITY, "under-visited root child should get ∞");

        // With apply_forced = false, no infinity should appear.
        let scores_no_forced = mcts.puct_scores(1, 1.0, true, false);
        assert!(scores_no_forced[1].is_finite(), "forced playouts disabled — should be finite");
    }

    #[test]
    fn test_softmax_temp_flattens_prior_at_root() {
        // With T = 2.0 the high-prior action should get a relatively lower PUCT
        // U-term (flatter prior), and the low-prior action a higher one, compared
        // to T = 1.0 (identity).
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        // Strongly skewed prior: action 0 = 0.9, action 1 = 0.1.
        let node = Node::new(2, HashMap::from([(0u8, 0.9), (1u8, 0.1)]), 0.0, 1, None);
        mcts.insert_node(1, node);
        mcts.root_id = Some(1);

        mcts.root_softmax_temp = 1.0; // identity
        let scores_t1 = mcts.puct_scores(1, 1.0, true, false);

        mcts.root_softmax_temp = 2.0; // flatten
        let scores_t2 = mcts.puct_scores(1, 1.0, true, false);

        // Flatter prior → lower score for action 0 (dominant prior shaved down).
        assert!(
            scores_t2[0] < scores_t1[0],
            "higher temperature should reduce score for dominant action: {} vs {}",
            scores_t2[0], scores_t1[0]
        );
        // Flatter prior → higher score for action 1 (minority prior boosted).
        assert!(
            scores_t2[1] > scores_t1[1],
            "higher temperature should increase score for minority action: {} vs {}",
            scores_t2[1], scores_t1[1]
        );
    }

    #[test]
    fn test_policy_target_sums_to_one() {
        let env = NumberLineEnv::new(5);
        let client = UniformClient;
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        let evaluator = |e: &NumberLineEnv| if e.done() { 1.0 } else { 0.0 };
        mcts.run(&env, &client, 200, 1.4, &evaluator, true);

        let target = mcts.policy_target(1.4).expect("policy target should be Some after search");
        let total: f32 = target.values().sum();
        assert!((total - 1.0).abs() < 1e-5, "policy target must sum to 1.0, got {}", total);
    }

    #[test]
    fn test_policy_target_no_more_actions_than_raw() {
        let env = NumberLineEnv::new(5);
        let client = UniformClient;
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        let evaluator = |e: &NumberLineEnv| if e.done() { 1.0 } else { 0.0 };
        mcts.run(&env, &client, 200, 1.4, &evaluator, true);

        let root_id = mcts.root_id.unwrap();
        let raw_actions = mcts.get_node_immut(root_id).unwrap()
            .edge_visits.iter().filter(|&&v| v > 0).count();
        let pruned_actions = mcts.policy_target(1.4).unwrap().len();
        assert!(pruned_actions <= raw_actions,
            "pruning should not add actions: {} > {}", pruned_actions, raw_actions);
    }

    #[test]
    fn test_normalize_prior() {
        let mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        let priors = HashMap::from([(0u8, 0.2), (1u8, 0.3), (2u8, 0.5)]);
        let normalized = mcts.normalize_prior(priors, &[0, 1]);
        let total: f32 = normalized.values().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    // Helper: build a small tree manually.
    //
    //   root (id=10) --0--> child_a (id=20) --0--> grandchild (id=40)
    //   root (id=10) --1--> child_b (id=30)
    //
    // child_b and root become unreachable after advancing via action 0.
    fn build_simple_tree() -> MCTS<NumberLineEnv> {
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        let (root_id, child_a_id, child_b_id, grandchild_id): (NodeId, NodeId, NodeId, NodeId) =
            (10, 20, 30, 40);

        let mut root = Node::new(2, HashMap::from([(0u8, 0.5), (1u8, 0.5)]), 0.0, root_id, None);
        root.children[0] = Some(child_a_id);
        root.children[1] = Some(child_b_id);

        let mut child_a = Node::new(1, HashMap::from([(0u8, 1.0)]), 0.0, child_a_id, None);
        child_a.children[0] = Some(grandchild_id);

        let child_b = Node::new(1, HashMap::from([(0u8, 1.0)]), 0.0, child_b_id, None);
        let grandchild = Node::new(1, HashMap::from([(0u8, 1.0)]), 0.0, grandchild_id, None);

        mcts.insert_node(root_id, root);
        mcts.insert_node(child_a_id, child_a);
        mcts.insert_node(child_b_id, child_b);
        mcts.insert_node(grandchild_id, grandchild);
        mcts.root_id = Some(root_id);
        mcts
    }

    #[test]
    fn test_advance_root_prunes_unreachable_nodes() {
        let mut mcts = build_simple_tree();
        assert_eq!(mcts.nodes.len(), 4);

        mcts.advance_root(0u8); // move to child_a

        assert_eq!(mcts.root_id, Some(20));
        // root and child_b are no longer reachable from child_a
        assert_eq!(mcts.nodes.len(), 2);
        assert!(mcts.node_exists(20));
        assert!(mcts.node_exists(40));
        assert!(!mcts.node_exists(10));
        assert!(!mcts.node_exists(30));
    }

    #[test]
    fn test_advance_root_table_consistent_after_pruning() {
        // Every entry in transposition_table must point to the node with the matching id.
        let mut mcts = build_simple_tree();
        mcts.advance_root(0u8);

        assert_eq!(mcts.transposition_table.len(), mcts.nodes.len());
        for (&id, &idx) in &mcts.transposition_table {
            assert_eq!(mcts.nodes[idx].id, id);
        }
    }

    #[test]
    fn test_advance_root_preserves_transposition() {
        // A node reachable from two different children of the new root must be kept.
        //
        //   root (id=10) --0--> child_a (id=20) --0--> shared (id=40)
        //   root (id=10) --1--> child_b (id=30) --0--> shared (id=40)  [unreachable after advance]
        //
        // After advancing to child_a, shared is still reachable via child_a.
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        let (root_id, child_a_id, child_b_id, shared_id): (NodeId, NodeId, NodeId, NodeId) =
            (10, 20, 30, 40);

        let mut root = Node::new(2, HashMap::from([(0u8, 0.5), (1u8, 0.5)]), 0.0, root_id, None);
        root.children[0] = Some(child_a_id);
        root.children[1] = Some(child_b_id);

        let mut child_a = Node::new(1, HashMap::from([(0u8, 1.0)]), 0.0, child_a_id, None);
        child_a.children[0] = Some(shared_id);

        let mut child_b = Node::new(1, HashMap::from([(0u8, 1.0)]), 0.0, child_b_id, None);
        child_b.children[0] = Some(shared_id);

        let shared = Node::new(1, HashMap::from([(0u8, 1.0)]), 0.0, shared_id, None);

        mcts.insert_node(root_id, root);
        mcts.insert_node(child_a_id, child_a);
        mcts.insert_node(child_b_id, child_b);
        mcts.insert_node(shared_id, shared);
        mcts.root_id = Some(root_id);

        mcts.advance_root(0u8);

        assert_eq!(mcts.root_id, Some(child_a_id));
        assert_eq!(mcts.nodes.len(), 2); // child_a and shared
        assert!(mcts.node_exists(child_a_id));
        assert!(mcts.node_exists(shared_id));
        assert!(!mcts.node_exists(root_id));
        assert!(!mcts.node_exists(child_b_id));
    }

    #[test]
    fn test_advance_root_handles_cycle() {
        // Ensure BFS terminates when child_a and child_b point back at each other.
        //
        //   root (id=10) --0--> child_a (id=20) --1--> child_b (id=30)
        //                       child_b (id=30) --0--> child_a (id=20)  [cycle]
        let mut mcts: MCTS<NumberLineEnv> = MCTS::new(4);
        let (root_id, child_a_id, child_b_id): (NodeId, NodeId, NodeId) = (10, 20, 30);

        let mut root = Node::new(1, HashMap::from([(0u8, 1.0)]), 0.0, root_id, None);
        root.children[0] = Some(child_a_id);

        let mut child_a = Node::new(2, HashMap::from([(0u8, 0.5), (1u8, 0.5)]), 0.0, child_a_id, None);
        child_a.children[1] = Some(child_b_id);

        let mut child_b = Node::new(1, HashMap::from([(0u8, 1.0)]), 0.0, child_b_id, None);
        child_b.children[0] = Some(child_a_id); // back-edge

        mcts.insert_node(root_id, root);
        mcts.insert_node(child_a_id, child_a);
        mcts.insert_node(child_b_id, child_b);
        mcts.root_id = Some(root_id);

        mcts.advance_root(0u8); // does not hang

        assert_eq!(mcts.root_id, Some(child_a_id));
        assert_eq!(mcts.nodes.len(), 2);
        assert!(mcts.node_exists(child_a_id));
        assert!(mcts.node_exists(child_b_id));
        assert!(!mcts.node_exists(root_id));
    }
}