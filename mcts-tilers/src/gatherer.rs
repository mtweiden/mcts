use std::collections::HashMap;
use std::io::{BufWriter, Write};

use serde_json::{Value, json, Map};
use rand_distr::{Gamma, Distribution};
use rand_distr::weighted::WeightedIndex;
use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use mcts_core::ipc_core::Arena;
use mcts_core::inference::InferenceClient;
use mcts_core::mcts::MCTS;
use mcts_core::node::Node;

use tilers::env::Environment;
use tilers::rl;
use tilers::rl::board::BoardCell;
use tilers::solver::Solver;
use tilers::core::enums::Action as PlanAction; // raw typed heuristic action, for replay

use crate::constants::*;
use crate::environment::TilersEnv;
use crate::slot::TilersSlot;
use crate::client::TilersIpcClient;

/// (total, max) PauliProduct weight over an env's objective queue.
///
/// `total` is the aggregate merge work in the circuit; `max` is the single
/// widest merge, which is the measured difficulty axis -- max Pauli factors in
/// one lattice-surgery merge stays predictive of completion even after
/// controlling for episode length. Logged per episode so gather difficulty can
/// be analysed on the merge-weight axis, not just ancillas/objectives.
fn pp_weights(env: &Environment) -> (usize, usize) {
    let mut total = 0usize;
    let mut mx = 0usize;
    for o in env.objective_queue.objectives_iter() {
        if let tilers::objective::Objective::PauliProduct(pp) = o {
            let w = pp.weight();
            total += w;
            if w > mx {
                mx = w;
            }
        }
    }
    (total, mx)
}



/// ----------------------------------------------------------------------------
/// Gatherer
/// ----------------------------------------------------------------------------
/// A struct to gather data from MCTS simulations.
/// ----------------------------------------------------------------------------
pub struct Gatherer {
    batch_size: usize,
    /// Playout budget for full (recorded) searches.
    mcts_steps: usize,
    /// Playout budget for fast (unrecorded) searches. Should be much smaller
    /// than `mcts_steps`. Reference: [Wu 2020, §3.1].
    fast_steps: usize,
    /// Fraction of turns that run a full search and are recorded for training.
    /// The remaining turns run a fast search used only for move selection.
    /// Reference: [Wu 2020, §3.1].
    p_full_search: f32,
    max_actions: Option<usize>,
    /// Per-episode action budget as a multiple of the heuristic solution
    /// length: `max_actions = reference_actions.len() * max_action_multiplier`
    /// (used only when `max_actions` above is `None`). 1.2 is the historical
    /// default. Lowering it shortens the post-progress flailing tail of a stuck
    /// agent — faster gather, and fewer HER episodes whose reward saturates to
    /// -1 from that wasted tail.
    max_action_multiplier: f32,
    output_path: String,
    noise_strength: f64,
    /// Mixing weight for Dirichlet noise injected into the MCTS root prior on
    /// full-search turns. 0.0 disables root noise.
    /// Reference: [Wu 2020, §2].
    dirichlet_epsilon: f32,
    /// Future objective layers beyond the current one; board has
    /// `lookahead + 1` layers.  See `tilers::rl::board::construct_board`.
    lookahead: usize,
    gather_id: usize,
    /// Saturation temperature for the terminal reward. The terminal value
    /// target is `tanh((reference_depth - actual_depth) / (reference_depth + ε)
    /// / reward_saturation_temperature)`. Smaller temperature → sharper
    /// saturation toward ±1 (closer to AlphaZero's categorical signal); larger
    /// temperature → near-linear in depth-delta with broad dynamic range. The
    /// previous clip-and-normalize scheme corresponds to the limit of the
    /// linear region with slope 1/temperature at the origin, but without the
    /// hard discontinuity at the saturation boundary.
    reward_saturation_temperature: f32,
    /// Resignation threshold: end the episode early when the agent's
    /// MCTS-backed Q estimate is at or below this value for
    /// `resign_consecutive_moves` consecutive moves. (An earlier variant
    /// also required the agent to have overshot the solver's reference
    /// depth; that gate suppressed ~96% of resignations and was removed —
    /// see the resignation-check comment in `gather`.) Set
    /// `resign_consecutive_moves = 0` to disable resignation entirely;
    /// values of `resign_value_threshold` above 1.0 also have that effect
    /// (Q is bounded above by 1).
    resign_value_threshold: f32,
    resign_consecutive_moves: usize,
    /// Step floor for resignation: no episode may resign before step
    /// max(20, ceil(mult * reference_depth)). 0.0 disables the floor (legacy:
    /// resign as early as the streak allows). 2026-07-20 calibration on 118k
    /// logged episodes (iters 45-53 forced-play counterfactual): NO (threshold,
    /// streak) setting reaches the AGZ ~5% false-positive target — every cell
    /// of the 6x7 grid has FP >= 19%, because Q pins near -1 from ~step 5 on
    /// hard envs whether or not the episode eventually finishes, so the streak
    /// carries no outcome information. The elapsed-step dimension does:
    /// mult=1.25 gives FP 8.0% overall / 6.1% on hard singles (vs 31.4% /
    /// 24.9% bare) and raises hard-single done-records per gather ~34% rel.
    /// Cost: resignation's step savings drop 49.5% -> 6.1% of episode mass.
    resign_min_step_ref_mult: f32,
    /// See set_clustered_reverse_prob. Negative sentinel = use the generic prob.
    clustered_reverse_prob: f32,
    /// See set_env_meta: generated-env max PP factor weight / cluster count,
    /// stamped on records for dashboard slicing. 0/0 when unset.
    env_mw: usize,
    /// Sum of the weights of ALL objectives in the env. This is the difficulty
    /// metric: `env_mw` (the single widest product) says nothing about how many
    /// others sit behind it, and `num_ancillas` is actively misleading here
    /// because full weight is `100 - num_ancillas`, so FEWER ancillas means a
    /// HARDER env, not an easier one.
    env_tw: usize,
    env_k: usize,
    /// Fraction of episodes in which resignation is *disabled* and the
    /// game is played out to its natural end. These serve as a sanity
    /// check: if too many of the would-have-been-resigned positions
    /// actually flip, the threshold is too aggressive. Set to 0.0 to
    /// always allow resignation (no sanity sample) or 1.0 to never
    /// resign (effectively disables the feature, equivalent to
    /// `resign_consecutive_moves = 0`).
    no_resign_rate: f32,
    /// Probability of KEEPING a zero-progress ("floor", reward = -1) episode's
    /// records. 1.0 writes every floor episode (historical behavior); lower
    /// values randomly drop that fraction so the corpus isn't dominated by
    /// uninformative -1s. Only "floor" episodes are subsampled — "done" and
    /// graded "her" episodes are always written. This rebalances the training
    /// distribution toward the rare non-floored signal without changing the
    /// reward of any kept record.
    floor_keep_fraction: f32,
    /// HER reward baseline margin (fraction of the heuristic reference depth).
    /// Applies ONLY to "her" (partial, goal-relabeled) episodes — never to
    /// "done" or to the evaluator, so the value head's done-reward calibration
    /// stays consistent between gather and eval. The HER terminal reward is
    /// `tanh(((1 + her_reward_margin) * her_ref_depth - actual_depth)
    ///       / (her_ref_depth + ε) / temperature)`. With margin = 0.0 (default)
    /// the zero-reward point is "match the heuristic on the achieved sub-goal"
    /// (historical behavior). A positive margin moves the zero point to
    /// `(1 + margin)×` the heuristic depth, so a partial completion that lands
    /// within `margin` of the heuristic scores slightly positive instead of
    /// negative. This lifts the large achieved-but-negative HER mass into a
    /// usable "make progress toward completion" gradient and counteracts the
    /// full-trajectory-depth penalty that otherwise makes more-objectives
    /// episodes score *worse* (the observed non-monotonicity in achieved count).
    her_reward_margin: f32,
    /// Master switch for the vanilla-HER fallback (default true = historical
    /// behaviour). When false, an episode that reaches neither `done` nor the
    /// cusp branch scores -1/"floor" instead of being relabelled to the goal it
    /// happened to achieve. Turning HER off is only sane alongside the
    /// outcome-only value contract: a non-done record trains toward -1 anyway,
    /// so the `achieved_goal_env` + `Solver::solve` HER pays per failed episode
    /// buys a reward the value head discards. It also removes the relabel-down
    /// ("hard env scored as an efficient success") that the cusp reward exists
    /// to prevent — which matters most when `cusp_reward` is OFF, since then
    /// every hard env falls through to this branch.
    her_reward: bool,
    /// If set, winning trajectories are written here as pretraining data.
    trajectory_dir: Option<String>,
    /// If set, the gatherer writes one JSON line per episode summarising
    /// the per-step root-Q trace plus the final outcome. This is the data
    /// needed to calibrate `resign_value_threshold` AGZ-style: pick the
    /// threshold T such that on the no-resign sanity sample (where
    /// `resign_allowed = false`), no more than 5% of episodes that would
    /// have resigned at T actually went on to win.
    resignation_log_dir: Option<String>,
    reverse_curriculum: bool,
    reverse_curriculum_min_len: usize,
    reverse_curriculum_max_probes: usize,
    /// Prefix length (# heuristic actions replayed from the START) for the FIRST
    /// cusp probe. Prior: the roadblock lives near the opening, so seed the
    /// binary search close to the full unscaffolded env (~10) rather than near
    /// the goal. Subsequent probes bisect. See gather_reverse_curriculum.
    reverse_curriculum_prefix_start: usize,
    /// Fraction of eligible (plan >= min_len) instances that actually run the
    /// reverse-curriculum climb; the rest gather the full env normally (so the
    /// objective-axis cusp reward applies to them). 1.0 = always (old behavior).
    reverse_curriculum_prob: f32,
    /// Expert-iteration "gold" banking (Phase 2). When Some, genuine HARD wins
    /// (done, score > 0, reference_action_count >= gold_min_len) are ALSO
    /// appended to `<gold_shard_dir>/gold-<gather_id>.jsonl` with a `"gold": true`
    /// tag, so they can be pinned into every training set regardless of the
    /// K-window and compound instead of aging out. None = banking OFF.
    gold_shard_dir: Option<String>,
    gold_min_len: usize,
    /// Minimum episode score (margin over the heuristic) to bank as gold.
    /// Tightened from >0 so only REAL beats compound, not tie-level wins.
    gold_min_reward: f32,
    /// Cusp reward (Axis A2): when true, HARD envs (num_objectives >
    /// cusp_frontier) grade partials against the cusp goal
    /// min(num_obj, cusp_frontier + cusp_margin) instead of HER relabeling
    /// down to the achieved subset. See the reward block in run_episode_from.
    cusp_reward: bool,
    cusp_frontier: usize,
    cusp_margin: usize,
    /// Q-filtered behavior-cloning demos (Phase 4, HORIZON_EXTENSION_DESIGN.md).
    /// For a fraction of HARD envs (num_objectives > demo_min_objectives) the
    /// gatherer replays the heuristic solver's plan and writes one demo training
    /// record per plan step (one-hot policy = the heuristic action, reward 0.0,
    /// reward_kind "demo", is_demo=true) IN ADDITION to the agent's own failed
    /// episode records (failure-triggered; see gather()).
    /// The trainer applies these under a Q-filter so the agent only imitates the
    /// heuristic where it currently underestimates the demo state (never capped
    /// at the heuristic's level). 0.0 = OFF (no demos; default).
    demo_fraction: f32,
    demo_min_objectives: usize,
    /// Keep-probability per demo STATE. Each demo emits one record per plan
    /// step, and hard-env plans run 35 to hundreds of actions — unthrottled,
    /// demos flooded 64% of the corpus (measured 2026-07-07), drowning the
    /// agent's own MCTS-derived policy targets. Each state-action pair is
    /// independent supervision, so random subsampling is legitimate and keeps
    /// coverage of deep states (unlike truncation). 1.0 = keep all.
    demo_subsample: f32,
    /// Envs with env_mw >= this are IMITATION-ONLY: write the (subsampled)
    /// heuristic demo and skip the agent episode + reverse-curriculum entirely.
    /// Full-weight PPs are far beyond the agent's frontier — playing them only
    /// flails ~1000+ steps and floods the corpus with failures. 0 = disabled.
    demo_only_min_weight: usize,
    /// Play temperature used DEEP in an episode (step >= deep_play_after).
    /// 0 = disabled (keep the legacy decay).
    ///
    /// The legacy schedule is `0.1 + 0.9*exp(-0.5*step)`: a 1.4-step half-life,
    /// so from step ~10 onward the agent samples from visits^10, i.e. argmax.
    /// On a 900-step full-weight episode that is ~1% exploration, against
    /// AlphaZero's ~15%-of-game exploration phase -- and it explores at the
    /// START, where the position is easy, then plays the hard deep states
    /// greedily. Measured 2026-07-29 on the 20-env full-weight probe (same
    /// agent, same seeds, only play policy differs): argmax 30.8% vs
    /// visit-proportional sampling after step 20 at 44.2% (15W/3L), and 27.8%
    /// vs 63.8% on the subset a T=1.0 arm finished. Under Path B the gap holds
    /// (81.2% -> 98.4% on the four hardest envs), so the two fixes stack.
    deep_play_temp: f32,
    /// Step from which `deep_play_temp` takes over. Keeps the opening greedy so
    /// only genuinely deep states get the extra diversity.
    deep_play_after: usize,
    /// Probability that a FULL-WEIGHT (demo-only) env ALSO gets an agent
    /// reverse-curriculum episode instead of just the heuristic demo. Without
    /// this the agent never ACTS on the target distribution at all — it only
    /// behavior-clones (the demo-only branch skips play AND RC), so it gets no
    /// reward signal where we actually want competence. 0 = disabled (old
    /// behaviour). See full_weight_rc_tail for why the seed must be goal-side.
    full_weight_rc_prob: f32,
    /// Agent-played SUFFIX length for full-weight RC episodes: the heuristic
    /// scaffolds `plan.len() - tail` actions and the agent finishes the last
    /// `tail`. The normal RC seed (reverse_curriculum_prefix_start = 10) seeds
    /// near the FULL env, which on a ~1600-action full-weight plan would have
    /// the agent flail ~1590 steps per probe. Measured 2026-07-29: on full-weight
    /// envs the agent stalls after ~120-170 productive steps, so the tail is
    /// sized to what it can actually sustain. Bisection is floored at
    /// `len - 2*tail` so a probe can never blow up into a full-length flail.
    full_weight_rc_tail: usize,
    /// Length-normalized demo subsampling: keep ~this many states per demo
    /// episode REGARDLESS of plan length (keep-prob = min(1, target/plan_len)),
    /// so a long high-weight demo (refn ~2600) and a short one (~900) contribute
    /// EQUAL records. Fixes the weight-mass imbalance a flat demo_subsample
    /// fraction causes (long demos over-represent high weights, cannibalizing the
    /// easier regime — measured cand-69 2026-07-27). 0 = disabled (use fraction).
    demo_target_states: usize,
}

/// One self-play episode's outcome, decoupled from record-writing so the
/// reverse-curriculum climb can run several probes and choose which to keep.
struct EpisodeOutcome {
    reference_depth: f32, solution_depth: f32, done: bool,
    score: f32, reward_kind: &'static str, achieved_objectives: usize,
    /// Heuristic action-budget count for this episode: the full heuristic plan
    /// length for a normal episode, or `k` (the tail length) for a
    /// reverse-curriculum probe. Used as the "hardness" gate for gold banking.
    reference_action_count: usize,
    temp_data: Vec<(Vec<Vec<BoardCell>>, usize, Vec<Action>, HashMap<Action, usize>)>,
    all_steps_data: Vec<(Vec<Vec<BoardCell>>, usize, Vec<Action>, Action, usize, usize)>,
    q_trace: Vec<f32>, over_solver_trace: Vec<bool>,
    resigned: bool, resign_allowed: bool, resignation_enabled: bool,
    height: usize, width: usize, num_ancillas: usize, num_objectives: usize,
}

impl Gatherer {
    pub fn new(
        batch_size: usize,
        mcts_steps: usize,
        fast_steps: usize,
        p_full_search: f32,
        output_path: String,
        noise_strength: f64,
        dirichlet_epsilon: f32,
        lookahead: usize,
        gather_id: usize,
        trajectory_dir: Option<String>,
        reward_saturation_temperature: Option<f32>,
        max_actions: Option<usize>,
        max_action_multiplier: Option<f32>,
        resign_value_threshold: Option<f32>,
        resign_consecutive_moves: Option<usize>,
        resign_min_step_ref_mult: Option<f32>,
        no_resign_rate: Option<f32>,
        resignation_log_dir: Option<String>,
        floor_keep_fraction: Option<f32>,
        her_reward_margin: Option<f32>,
    ) -> Self {
        Self {
            batch_size,
            mcts_steps,
            fast_steps,
            p_full_search,
            max_actions,
            max_action_multiplier: max_action_multiplier.unwrap_or(1.2),
            output_path,
            noise_strength,
            dirichlet_epsilon,
            lookahead,
            gather_id,
            // Default of 0.3 preserves the slope-at-origin (1/0.3 ≈ 3.33) of
            // the previous clip-and-normalize default with reward_ratio_limit
            // = 0.3, so behavior in the unsaturated region is unchanged on
            // first switch. Tune up (more linear) or down (more categorical)
            // explicitly via the constructor / Python kwarg.
            reward_saturation_temperature: reward_saturation_temperature.unwrap_or(0.3),
            // Resignation defaults: AZ-paper-style conservative thresholds.
            // Q ≤ -0.9 for 5 consecutive moves while already over solver
            // depth → resign. 10% of episodes never resign and serve as the
            // false-positive sanity check.
            resign_value_threshold: resign_value_threshold.unwrap_or(-0.9),
            resign_consecutive_moves: resign_consecutive_moves.unwrap_or(5),
            // 0.0 = no step floor (legacy). See the field doc for the
            // 2026-07-20 calibration behind the recommended 1.25.
            resign_min_step_ref_mult: resign_min_step_ref_mult.unwrap_or(0.0),
            // Set via setters, not constructor args (per-env metadata + the
            // clustered-RC override; sentinel -1.0 = generic prob).
            clustered_reverse_prob: -1.0,
            env_mw: 0,
            env_tw: 0,
            env_k: 0,
            no_resign_rate: no_resign_rate.unwrap_or(0.1),
            // 1.0 = keep every floor episode (unchanged default).
            floor_keep_fraction: floor_keep_fraction.unwrap_or(1.0),
            // 0.0 = zero-reward at exactly the heuristic depth (unchanged default).
            her_reward_margin: her_reward_margin.unwrap_or(0.0),
            her_reward: true,
            trajectory_dir,
            resignation_log_dir,
            reverse_curriculum: false,
            reverse_curriculum_min_len: usize::MAX,
            reverse_curriculum_max_probes: 5,
            reverse_curriculum_prefix_start: 10,
            reverse_curriculum_prob: 1.0,
            // Gold banking OFF by default (no dir → never writes; the sentinel
            // gold_min_len = usize::MAX means nothing qualifies even if a dir slips in).
            gold_shard_dir: None,
            gold_min_len: usize::MAX,
            gold_min_reward: 0.0,
            cusp_reward: false,
            cusp_frontier: 2,
            cusp_margin: 2,
            // Q-filtered BC demos OFF by default (fraction 0.0 → never replays).
            demo_fraction: 0.0,
            demo_subsample: 1.0,
            demo_min_objectives: 2,
            demo_only_min_weight: 0,
            demo_target_states: 0,
            full_weight_rc_prob: 0.0,
            full_weight_rc_tail: 0,
            deep_play_temp: 0.0,
            deep_play_after: 20,
        }
    }

    pub fn set_demo(&mut self, demo_fraction: f32, demo_min_objectives: usize,
                    demo_subsample: f32, demo_only_min_weight: usize,
                    demo_target_states: usize) {
        self.demo_fraction = demo_fraction.clamp(0.0, 1.0);
        self.demo_subsample = demo_subsample.clamp(0.0, 1.0);
        self.demo_min_objectives = demo_min_objectives;
        self.demo_only_min_weight = demo_only_min_weight;
        self.demo_target_states = demo_target_states;
    }

    pub fn set_deep_play(&mut self, temp: f32, after: usize) {
        self.deep_play_temp = temp.max(0.0);
        self.deep_play_after = after;
    }

    pub fn set_full_weight_rc(&mut self, prob: f32, tail: usize) {
        self.full_weight_rc_prob = prob.clamp(0.0, 1.0);
        self.full_weight_rc_tail = tail;
    }

    pub fn set_her_reward(&mut self, enabled: bool) {
        self.her_reward = enabled;
    }

    pub fn set_cusp_reward(&mut self, enabled: bool, frontier: usize, margin: usize) {
        self.cusp_reward = enabled;
        self.cusp_frontier = frontier;
        self.cusp_margin = margin.max(1);
    }

    pub fn set_gold_banking(&mut self, gold_shard_dir: Option<String>, gold_min_len: usize, gold_min_reward: f32) {
        self.gold_shard_dir = gold_shard_dir;
        self.gold_min_reward = gold_min_reward;
        self.gold_min_len = gold_min_len;
    }

    pub fn set_reverse_curriculum(&mut self, enabled: bool, min_len: usize,
                                  max_probes: usize, prefix_start: usize, prob: f32) {
        self.reverse_curriculum = enabled;
        self.reverse_curriculum_prob = prob.clamp(0.0, 1.0);
        self.reverse_curriculum_min_len = min_len;
        self.reverse_curriculum_max_probes = max_probes.max(1);
        self.reverse_curriculum_prefix_start = prefix_start;
    }

    /// Separate reverse-curriculum probability for CLUSTERED wide-PP envs.
    /// Negative = use the generic `reverse_curriculum_prob` (legacy).
    ///
    /// 2026-07-21 trace analysis: wide envs complete at 0.92 under reverse
    /// curriculum vs 0.43 from scratch, but RC ran on only ~4% of wide
    /// episodes (no mw/cluster gating existed). A global prob raise would
    /// flood narrow long-plan envs with endgame-tilted data; this targets
    /// the arm that needs it. Guarded abort metric: from-scratch wide
    /// done-rate must not fall (RC stealing opening data).
    pub fn set_clustered_reverse_prob(&mut self, prob: f32) {
        self.clustered_reverse_prob = if prob < 0.0 { -1.0 } else { prob.clamp(0.0, 1.0) };
    }

    /// Per-env generation metadata, stamped onto every record this episode
    /// writes (env_mw / env_k) so gather-side dashboards can slice done-rate
    /// by merge weight and cluster count without decoding boards. k=0 means
    /// the env did not come from the clustered wide-PP arm.
    pub fn set_env_meta(&mut self, mw: usize, k: usize) {
        self.env_mw = mw;
        self.env_k = k;
    }

    /// Total objective weight for this env — the difficulty metric. Separate
    /// setter so existing `set_env_meta` callers keep working unchanged.
    pub fn set_env_total_weight(&mut self, tw: usize) {
        self.env_tw = tw;
    }

    /// Solve the environment using a heuristic solver and return the depth of the solution.
    pub fn solve_with_heuristic(&self, env: &Environment) -> (f32, Vec<Action>) {
        let mut solved_env = env.clone();
        let solver = Solver::new();
        let actions = solver.solve(&mut solved_env, true)
            .unwrap()
            .into_iter()
            .map(|a| rl::encode(&solved_env, a).expect("valid_actions ids always encode") as Action)
            .collect();
        let depth = solved_env.depth(true, true);
        (depth, actions)
    }

    /// Directly sampling from Dirichlet distribution requires num_actions to be known at
    /// compile time, so we sample using Gamma distributions instead.
    ///
    /// alpha is constant for all actions in a single call (10 / num_actions, capped at
    /// 0.5), so the Gamma distribution is constructed once and shared across all
    /// num_actions samples — pre-refactor this allocated num_actions × Gamma
    /// distributions per call.
    fn _dirichlet_noise(&self, num_actions: usize, rng: &mut impl Rng) -> Vec<f64> {
        let alpha = (10f64 / (num_actions as f64)).min(0.5);
        let gamma = Gamma::new(alpha, 1.0).unwrap();
        let mut xs: Vec<f64> = (0..num_actions).map(|_| gamma.sample(rng)).collect();
        let sum_xs: f64 = xs.iter().sum();
        if sum_xs > 0.0 {
            for x in xs.iter_mut() {
                *x /= sum_xs;
            }
        }
        xs
    }

    fn _action_probabilities(&self, visit_counts: &[usize], temperature: f64) -> Vec<f64> {
        let total_visits: usize = visit_counts.iter().sum();
        if total_visits == 0 {
            return vec![1.0 / (visit_counts.len() as f64); visit_counts.len()];
        }

        if temperature < 1e-8 {
            // Greedy: put all weight on the most-visited action (break ties uniformly)
            let max_count = *visit_counts.iter().max().unwrap();
            let num_max = visit_counts.iter().filter(|&&c| c == max_count).count();
            return visit_counts
                .iter()
                .map(|&c| if c == max_count { 1.0 / num_max as f64 } else { 0.0 })
                .collect();
        }

        let inv_temp = 1.0 / temperature;
        let log_counts: Vec<f64> = visit_counts
            .iter()
            .map(|&c| if c > 0 { (c as f64).ln() * inv_temp } else { f64::NEG_INFINITY })
            .collect();

        // Subtract max for numerical stability before exponentiating
        let max_log = log_counts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = log_counts.iter().map(|&l| (l - max_log).exp()).collect();
        let sum_exps: f64 = exps.iter().sum();

        exps.iter().map(|&e| e / sum_exps).collect()
    }

    pub fn select_action(
        &self,
        node: &Node<Action>,
        env: &Environment,
        step: usize,
        rng: &mut impl Rng,
    ) -> Action {
        // Flat `u16` action ids the tree/visits are keyed by.
        let ids: Vec<Action> = env
            .valid_actions()
            .iter()
            .map(|&a| rl::encode(env, a).expect("valid_actions ids always encode") as Action)
            .collect();
        let num_actions = ids.len();
        if num_actions == 0 {
            panic!("No valid actions available");
        }

        // High temperature early (exploration), low temperature later
        // (exploitation) -- UNLESS deep_play_temp is set, in which case deep
        // states keep sampling instead of collapsing to argmax. See the
        // deep_play_temp field docs for the measurement.
        let temperature = if self.deep_play_temp > 0.0 && step >= self.deep_play_after {
            self.deep_play_temp as f64
        } else {
            0.1 + 0.9 * (-0.5 * step as f64).exp()
        };
        let noise_strength = self.noise_strength * (-0.5 * step as f64).exp();

        let probs = self._action_probabilities(
            &ids
                .iter()
                .map(|&a| node.edge_visits.get(a as usize).copied().unwrap_or(0))
                .collect::<Vec<_>>(),
            temperature,
        );

        let noise = if noise_strength > 0.0 {
            self._dirichlet_noise(num_actions, rng)
        } else {
            vec![0.0; num_actions]
        };

        let mixed_probs: Vec<f64> = probs
            .iter()
            .zip(noise.iter())
            .map(|(&p, &n)| (1.0 - noise_strength) * p + noise_strength * n)
            .map(|x| x.max(0.0))
            .collect();

        let dist = WeightedIndex::new(&mixed_probs).unwrap();
        ids[dist.sample(rng)]
    }

    /// Serialize the 10-channel observation board as nested JSON:
    /// `[layer][cell] -> [10 ints]` (channel order = `BoardCell` field
    /// order).  Replaces the old placement+objectives serialization.
    fn serialize_board(board: &[Vec<BoardCell>]) -> Value {
        Value::Array(
            board
                .iter()
                .map(|layer| {
                    Value::Array(
                        layer
                            .iter()
                            .map(|c| {
                                json!([
                                    c.qubit_role,
                                    c.factor_kind,
                                    c.resource_kind,
                                    c.pp_group_row,
                                    c.pp_group_col,
                                    c.ancilla_idx,
                                    c.last_move_dir,
                                    c.weight_in_pp,
                                    c.is_hub_for_pp,
                                    c.is_y_ready,
                                    c.pp_needs_t,
                                ])
                            })
                            .collect::<Vec<Value>>(),
                    )
                })
                .collect::<Vec<Value>>(),
        )
    }

    /// Raw typed heuristic plan for an environment (for reverse-curriculum
    /// replay). Empty if the solver fails.
    /// Solve `env` with the heuristic once, returning the full-solution depth
    /// (D_full) AND the typed plan. D_full is reused as the reverse-curriculum
    /// sub-problem reference so intermediate states are NEVER re-solved.
    fn heuristic_typed_plan(&self, env: &Environment) -> (f32, Vec<PlanAction>) {
        let mut e = env.clone();
        let plan = Solver::new().solve(&mut e, true).unwrap_or_default();
        (e.depth(true, true), plan)
    }

    /// Build a reverse-curriculum start state: replay the first `prefix_len`
    /// heuristic actions from `game`, leaving the remaining `len - prefix_len`
    /// objectives for the agent. `prefix_len == 0` yields the true start S0;
    /// `prefix_len == plan.len()` yields the (near-)solved terminal.
    /// Returns the scaffolded start AND the depth the scaffold itself consumed.
    ///
    /// The prefix is applied with real `step`s, so `S_k`'s depth already includes
    /// it — and so does the agent's final `solution_depth`. The scoring ratio must
    /// therefore subtract it from BOTH sides, else the denominator is the full
    /// reference while the agent only ever controls the suffix (see the call in
    /// `run_episode_from`).
    fn make_reverse_start(
        &self,
        game: &Environment,
        plan: &[PlanAction],
        prefix_len: usize,
    ) -> (Environment, f32) {
        let mut sk = game.clone();
        sk.set_cultivation_time(10);
        for a in plan.iter().take(prefix_len) {
            let _ = sk.step(a.clone());
            sk.finish_cultivating(None, None);
        }
        let scaffold_depth = sk.depth(true, true) as f32;
        (sk, scaffold_depth)
    }

    pub fn gather(
        &self,
        env: &Environment,
        client: &dyn InferenceClient<TilersEnv>,
        c_puct: f32,
        rng: &mut impl Rng,
    ) -> (f32, f32, bool) {
        let mut game = env.clone();
        game.set_cultivation_time(10);
        // IMITATION-ONLY for full-weight envs: env_mw >= demo_only_min_weight are
        // far past the agent's frontier, so we skip the agent episode AND the
        // reverse-curriculum and write only the (subsampled) heuristic demo. This
        // gives full-weight imitation cheaply without flailing / corpus flooding.
        if self.demo_only_min_weight > 0 && self.env_mw >= self.demo_only_min_weight {
            // ...UNLESS full-weight RC practice is enabled. Pure imitation gives
            // the agent no reward signal on the target distribution; a goal-side
            // reverse-curriculum episode lets it actually FINISH a full-weight
            // merge (positive reward) from a deep scaffold, and the cusp search
            // walks the scaffold back as it improves.
            if self.full_weight_rc_prob > 0.0
                && self.full_weight_rc_tail > 0
                && self.reverse_curriculum
                && rng.random::<f32>() < self.full_weight_rc_prob
            {
                let (d_full, plan) = self.heuristic_typed_plan(&game);
                if plan.len() > self.full_weight_rc_tail {
                    // Seed GOAL-side (agent plays the last `tail`), and floor the
                    // bisection so no probe ever exceeds a 2*tail agent suffix.
                    let seed = plan.len() - self.full_weight_rc_tail;
                    let floor = plan.len().saturating_sub(2 * self.full_weight_rc_tail);
                    return self.gather_reverse_curriculum(
                        &game, d_full, &plan, client, c_puct, rng, Some((seed, floor)),
                    );
                }
            }
            return self.write_demo_episode(&game, rng);
        }
        // Clustered wide-PP envs (env_k > 0) may use a separate, higher RC
        // probability — see set_clustered_reverse_prob.
        let rc_prob = if self.env_k > 0 && self.clustered_reverse_prob >= 0.0 {
            self.clustered_reverse_prob
        } else {
            self.reverse_curriculum_prob
        };
        if self.reverse_curriculum && rng.random::<f32>() < rc_prob {
            let (d_full, plan) = self.heuristic_typed_plan(&game);
            if plan.len() >= self.reverse_curriculum_min_len {
                return self.gather_reverse_curriculum(&game, d_full, &plan, client, c_puct, rng, None);
            }
        }
        let o = self.run_episode_from(&game, client, c_puct, rng, None);
        let ret = self.write_episode(&o, false, rng);
        // Q-filtered BC demos (Phase 4), FAILURE-TRIGGERED. The agent always
        // plays its own episode above (its cusp/floor records are written
        // unchanged). Only when it FAILS a HARD env do we ALSO append the
        // heuristic solver's corrective demo for that SAME env. Rationale:
        //   - targets the agent's actual failure distribution (not a random
        //     pre-selected slice), so BC signal lands where the agent is weak;
        //   - gives paired signal on the same env (the agent's graded partial
        //     AND the solver's demo), which the trainer's per-state Q-filter
        //     then gates so only the weak states get imitated (never capped);
        //   - reuses the reference solve already done in run_episode_from (see
        //     the duplicate-solve note in write_demo_episode).
        // Demos apply ONLY to this normal full-env path — the reverse-curriculum
        // path returns early above and is never demo-augmented.
        if self.demo_fraction > 0.0
            && !o.done
            && game.num_objectives() > self.demo_min_objectives
            && rng.random::<f32>() < self.demo_fraction
        {
            self.write_demo_episode(&game, rng);
        }
        ret
    }

    /// Replay the heuristic solver's plan on `game` and write ONE demo training
    /// record per plan step to the normal shard output (NOT the gold dir). Each
    /// record has the same shape as a normal full-search record EXCEPT:
    ///   - `edge_visits` is a ONE-HOT { <encoded heuristic action id>: 1.0 },
    ///     keyed exactly like normal records (via `rl::encode`), so the dataset
    ///     decodes it into a single policy target at prob 1.0;
    ///   - `reward` = 0.0 (the heuristic-completion reference the Q-filter reads);
    ///   - `reward_kind` = "demo", `is_demo` = true.
    /// The board recorded for each step is the state BEFORE the action (the state
    /// in which the heuristic chose that action). Returns the same (solution_depth,
    /// reference_depth, done) tuple shape `gather()` expects; only aggregate stats
    /// consume it, so (0.0, 0.0, false) is fine.
    fn write_demo_episode(&self, game: &Environment, rng: &mut impl Rng) -> (f32, f32, bool) {
        // NOTE: this re-solves `game` with the heuristic. The caller
        // (run_episode_from via gather) already solved `game` for the episode
        // reference, so this is a DUPLICATE solve. It only runs on the small
        // fraction of FAILED hard envs that pass the demo probe, so the extra
        // cost is negligible; the typed plan could be threaded out of
        // run_episode_from later to eliminate it entirely.
        let (_d_full, plan) = self.heuristic_typed_plan(game);
        if plan.is_empty() {
            return (0.0, 0.0, false);
        }

        // Evolving env, stepped one heuristic action at a time. The wrapper's
        // build_obs() gives the same windowed 10-channel board write_episode
        // records emit; `cultivation_time = 10` mirrors run_episode_from.
        let mut tenv = TilersEnv::new(game.clone(), self.lookahead);
        tenv.inner.set_cultivation_time(10);
        let height = tenv.inner.height;
        let width = tenv.inner.width;

        let file_raw = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)
            .expect("Unable to open output file");
        let mut file = BufWriter::new(file_raw);

        for a in plan.iter() {
            // Encode the heuristic action against the CURRENT (pre-step) state —
            // the same mapping normal records key edge_visits by. If the action
            // can't encode or isn't among the currently-valid actions, skip the
            // record (its one-hot target would be masked out by the legal mask)
            // but still step so we stay on the heuristic plan.
            let encoded = match rl::encode(&tenv.inner, a.clone()) {
                Ok(id) => id as Action,
                Err(_) => {
                    let _ = tenv.inner.step(a.clone());
                    tenv.inner.finish_cultivating(None, None);
                    continue;
                }
            };

            // Board + valid actions BEFORE stepping (state the heuristic chose in).
            let board = tenv.build_obs().board;
            let num_ancillas = tenv.inner.num_ancillas();
            let valid_actions: Vec<Action> = tenv
                .inner
                .valid_actions()
                .iter()
                .map(|&va| rl::encode(&tenv.inner, va).expect("valid_actions ids always encode") as Action)
                .collect();

            // Subsample: keep each demo state with prob keep_prob. Every
            // state-action pair is independent supervision, so a random subset
            // is unbiased and covers deep states (truncation wouldn't).
            // Length-normalized when demo_target_states>0: keep-prob scales as
            // target/plan_len so every demo yields ~target states regardless of
            // length -> equal record weight across PP weights (no long-demo bias).
            let keep_prob = if self.demo_target_states > 0 {
                (self.demo_target_states as f32 / (plan.len().max(1) as f32)).min(1.0)
            } else {
                self.demo_subsample
            };
            if valid_actions.contains(&encoded) && rng.random::<f32>() < keep_prob {
                let board_json = Self::serialize_board(&board);
                let valid_actions_json =
                    Value::Array(valid_actions.iter().map(|&x| Value::from(x)).collect());
                // ONE-HOT policy target, keyed like normal edge_visits.
                let mut visits_map = Map::with_capacity(1);
                visits_map.insert(encoded.to_string(), Value::from(1.0f64));
                let visits_json = Value::Object(visits_map);

                let record = json!({
                    "height": height,
                    "width": width,
                    "num_ancillas": num_ancillas,
                    "board": board_json,
                    "valid_actions": valid_actions_json,
                    "edge_visits": visits_json,
                    "reward": 0.0,
                    "reward_kind": "demo",
                    "achieved_objectives": 0,
                    "reverse_curriculum": false,
                    "is_demo": true,
                    // env context so demos are sliceable by weight (full-weight
                    // demos = env_mw >= demo_only_min_weight) for analysis.
                    "env_mw": self.env_mw,
                    "env_tw": self.env_tw,
                    "env_k": self.env_k,
                });
                writeln!(file, "{}", record).expect("Failed to write demo record");
            }

            let _ = tenv.inner.step(a.clone());
            tenv.inner.finish_cultivating(None, None);
        }
        file.flush().expect("Failed to flush demo file");
        (0.0, 0.0, false)
    }

    /// Reverse-curriculum bounded climb: start near the heuristic terminal
    /// Cusp partial reward (Axis A2): grade `achieved` objectives against the
    /// cusp goal `min(num_objectives, frontier + margin)` (a target just past
    /// the agent's reach), in [-1, 0]. Capping at the cusp — NOT the full goal —
    /// keeps far-past-cusp envs informative (1/20 would floor-collapse to -0.95,
    /// but 1/4 = -0.75); reaching the cusp scores 0. Pure → unit-tested.
    fn cusp_partial_reward(achieved: usize, num_objectives: usize, frontier: usize, margin: usize) -> f32 {
        let cusp = num_objectives.min(frontier + margin).max(1);
        (achieved.min(cusp) as f32) / (cusp as f32) - 1.0
    }

    /// Factor-level cusp reward (Axis A2): grade `merged` factors against the cusp
    /// goal's proportional factor budget = total * min(num_obj, frontier+margin) /
    /// num_obj. The cusp stays in OBJECTIVE units (env selection unchanged) but the
    /// grade is in FACTOR units (dense partial credit). Proportional budget assumes
    /// ~uniform objective weight (true here: objectives are almost all weight-2);
    /// a wide objective only shifts the zero-point slightly, never the gradient
    /// direction. Capping at the budget keeps far-past-cusp envs informative.
    /// Returns a value in [-1, 0]; reaching the cusp budget → 0. Pure → unit-tested.
    fn cusp_factor_reward(merged: usize, total: usize, num_objectives: usize,
                          frontier: usize, margin: usize) -> f32 {
        if total == 0 || num_objectives == 0 {
            return -1.0;
        }
        let cusp_objs = num_objectives.min(frontier + margin).max(1);
        let budget = (total as f32 * cusp_objs as f32 / num_objectives as f32).max(1.0);
        (merged as f32).min(budget) / budget - 1.0
    }

    /// Next scaffold-prefix to probe in the reverse-curriculum cusp search, by
    /// bisecting the open bracket (`lo`, `hi`) where `lo` = largest prefix that
    /// FAILED and `hi` = smallest prefix that SOLVED. The minimal solving prefix
    /// (the cusp) lies in (`lo`, `hi`]. Returns `None` once the edge is pinned
    /// (`hi <= lo + 1`). The FIRST probe is seeded separately (prefix_start), so
    /// this only ever bisects. Pure → unit-tested.
    fn next_prefix_probe(lo: usize, hi: usize) -> Option<usize> {
        if hi <= lo + 1 { return None; }
        let mid = lo + (hi - lo) / 2;
        if mid == lo { None } else { Some(mid) }
    }

    /// Find the agent's per-instance CUSP — the minimal scaffold prefix `p` (the
    /// heuristic replays the first `p` actions from the START; the agent solves
    /// the remaining `len - p` tail) at which the agent still finishes — via
    /// binary search, keeping the barely-solvable probe (`best`, POSITIVE signal:
    /// the path that works from the cusp) and the barely-too-hard probe (`failed`,
    /// NEGATIVE signal) — the sharpest contrastive signal at the ability boundary
    /// on the SAME env. Bounded by `reverse_curriculum_max_probes` (the precision
    /// dial). Prior: the roadblock lives near the opening, so the FIRST probe is
    /// seeded at `reverse_curriculum_prefix_start` (~10, right next to the full
    /// unscaffolded env) instead of near the goal; subsequent probes bisect.
    /// Bracket invariant: `lo` = largest prefix that FAILED (0 = the full env),
    /// `hi` = smallest prefix that SOLVED (`len` = fully scaffolded, trivially at
    /// the goal); the cusp lies in (`lo`, `hi`].
    fn gather_reverse_curriculum(
        &self,
        game: &Environment,
        d_full: f32,
        plan: &[PlanAction],
        client: &dyn InferenceClient<TilersEnv>,
        c_puct: f32,
        rng: &mut impl Rng,
        // (prefix_seed, lo_floor) override. None = normal band behaviour: seed
        // near the FULL env and bisect over the whole range. Some(..) is used by
        // the full-weight path to seed GOAL-side and floor the bisection so a
        // probe can't degenerate into a full-length flail.
        seed_override: Option<(usize, usize)>,
    ) -> (f32, f32, bool) {
        let len = plan.len();
        let mut lo = seed_override.map(|(_, f)| f).unwrap_or(0);  // largest prefix FAILED
        let mut hi = len;             // smallest prefix SOLVED (len = fully scaffolded)
        let mut best: Option<EpisodeOutcome> = None;
        let mut failed: Option<EpisodeOutcome> = None;
        let mut ret = (0.0, 0.0, false);
        let mut p = match seed_override {
            Some((s, _)) => s.min(len),
            None => self.reverse_curriculum_prefix_start.min(len),  // seed near full env
        };
        // Per-env cusp-search telemetry. RC_CUSP_LOG holds the target FILE PATH
        // (unset = off). Records the seed, full probe sequence, pinned cusp prefix,
        // and best/failed scores so we can confirm the search converges and cusps
        // cluster low. Written via atomic O_APPEND (below) so concurrent gatherer
        // workers never interleave — stderr/tee can't be used (they double + garble).
        let cusp_log_path = std::env::var("RC_CUSP_LOG").ok();
        let log_cusp = cusp_log_path.is_some();
        let seed = p;
        let mut probe_log: Vec<(usize, bool)> = Vec::new();
        for _ in 0..self.reverse_curriculum_max_probes {
            let (sk, scaffold_depth) = self.make_reverse_start(game, plan, p);
            // Reference for S_p = D_full (heuristic finishes S_p via its
            // remaining actions to the same terminal), with `len - p` remaining
            // heuristic actions for the action budget. No S_p re-solve → no
            // `[stuck] no_ready_pp` panics from the greedy solver on mid-states.
            let tail = len - p;
            let o =
                self.run_episode_from(&sk, client, c_puct, rng, Some((d_full, tail, scaffold_depth)));
            ret = (o.solution_depth, o.reference_depth, o.done);
            if log_cusp { probe_log.push((p, o.done)); }
            if o.done {
                hi = p;
                best = Some(o);
                if p == 0 { break; }          // solves the FULL unscaffolded env
            } else {
                lo = p;
                failed = Some(o);
            }
            match Self::next_prefix_probe(lo, hi) {
                Some(next) => p = next,
                None => break,                // cusp pinned / nothing new
            }
        }
        if log_cusp {
            // cusp_prefix = smallest solved prefix (hi) = the pinned cusp; -1 = never solved.
            // failed_prefix = largest failed prefix (lo); -1 = nothing failed.
            let cusp_prefix: i64 = if best.is_some() { hi as i64 } else { -1 };
            let failed_prefix: i64 = if failed.is_some() { lo as i64 } else { -1 };
            let bscore = best.as_ref().map_or(f32::NAN, |o| o.score);
            let fscore = failed.as_ref().map_or(f32::NAN, |o| o.score);
            let line = format!(
                "[RC_CUSP] mw={} k={} len={} seed={} probes={:?} cusp_prefix={} \
                 best_score={:.3} failed_prefix={} failed_score={:.3}\n",
                self.env_mw, self.env_k, len, seed, probe_log, cusp_prefix,
                bscore, failed_prefix, fscore
            );
            // Atomic O_APPEND: a single write < PIPE_BUF is atomic even across the
            // 40 concurrent gatherer workers, so lines never interleave.
            if let Some(path) = cusp_log_path.as_deref() {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = f.write_all(line.as_bytes());
                }
            }
        }
        if let Some(o) = &best   { self.write_episode(o, true, rng); }
        if let Some(o) = &failed { self.write_episode(o, true, rng); }
        ret
    }

    /// Run one self-play MCTS episode from `start_env`, returning its outcome
    /// (reward, HER provenance, traces, and captured training records) WITHOUT
    /// writing anything. Record-writing is done separately by `write_episode`
    /// so the reverse-curriculum climb can run several probes first.
    fn run_episode_from(
        &self,
        start_env: &Environment,
        client: &dyn InferenceClient<TilersEnv>,
        c_puct: f32,
        rng: &mut impl Rng,
        reference_override: Option<(f32, usize, f32)>,
    ) -> EpisodeOutcome {
        let mut mcts: MCTS<TilersEnv> = MCTS::new(self.batch_size);

        let mut game = start_env.clone();
        // The played game keeps its FULL objective set — the agent must solve
        // the entire problem, and the heuristic reference below is the
        // full-problem depth. The observation board is windowed to
        // `lookahead + 1` layers by `construct_board` via
        // `TilersEnv::new(.., self.lookahead)` below, so the agent sees a
        // sliding window without the future objectives being deleted.
        game.set_cultivation_time(10);
        // Reference depth + action budget. For reverse-curriculum probes the
        // caller passes Some((D_full, k)) — the heuristic's finishing depth from
        // S_k IS the full-solution depth (it completes S_k via its remaining
        // actions to the same terminal), with `k` remaining actions for the
        // budget — so we do NOT re-solve S_k (the greedy solver panics on
        // `[stuck] no_ready_pp` from arbitrary mid-solution states). None →
        // solve S0 normally.
        let (reference_depth, reference_action_count) = match reference_override {
            Some((d, n, _)) => (d, n),
            None => {
                let (d, acts) = self.solve_with_heuristic(&game);
                (d, acts.len())
            }
        };

        let h = game.height;
        let w = game.width;
        let nb = game.num_ancillas();
        let no = game.num_objectives();
        let (tw, mw) = pp_weights(&game);
        println!(
            "[Gatherer {}] Starting Env(h={}, w={}, nb={}, no={}, tw={}, mw={})",
            self.gather_id, h, w, nb, no, tw, mw,
        );

        // Training data for full-search turns only (written to output_path).
        // (board, num_ancillas, valid_action_ids, edge_visits)
        let mut temp_data: Vec<(
            Vec<Vec<BoardCell>>,
            usize,
            Vec<Action>,
            HashMap<Action, usize>,
        )> = Vec::new();

        // Full trajectory data for every step (written to trajectory_dir on a win).
        // (board, num_ancillas, valid_action_ids, action_taken, height, width)
        let mut all_steps_data: Vec<(
            Vec<Vec<BoardCell>>,
            usize,
            Vec<Action>,
            Action,
            usize,
            usize,
        )> = Vec::new();

        let mut tilers_env = TilersEnv::new(game.clone(), self.lookahead);

        let max_actions = if let Some(max) = self.max_actions {
            max
        } else {
            // ceil + 2 slack actions: a fixed multiplier alone rounds to zero
            // margin on short-reference envs (ref=2 at 1.0x-1.2x -> exactly 2
            // actions), which forbids ever completing sub-optimally. Resignation
            // bounds the flail cost on genuinely stuck episodes.
            (reference_action_count as f32 * self.max_action_multiplier).ceil() as usize + 2
        };

        // Resignation bookkeeping. resign_allowed is decided once per
        // episode so the no-resign sanity sample is a uniform 1 - rate
        // fraction. Episodes that fall into the sanity sample play out
        // to natural termination; the rest may resign once the
        // low-Q-streak condition is met.
        let resignation_enabled =
            self.resign_consecutive_moves > 0 && self.resign_value_threshold < 1.0;
        let resign_allowed =
            resignation_enabled && rng.random::<f32>() >= self.no_resign_rate;
        let mut low_value_streak: usize = 0;

        // Per-step traces for resignation-threshold calibration. Only
        // collected when `resignation_log_dir` is set; size is bounded by
        // max_actions which is O(reference_actions × 1.2). Cleared at the
        // start of every episode.
        let mut q_trace: Vec<f32> = Vec::new();
        let mut over_solver_trace: Vec<bool> = Vec::new();
        let mut resigned: bool = false;

        // Shared done-scorer (reward.rs). tanh(ratio / temperature) replaces
        // the previous clip-then-normalize — smooth gradient at all input
        // scales; saturation is governed entirely by the temperature knob.
        let tau = self.reward_saturation_temperature;
        // SCAFFOLD-RELATIVE SCORING. For a reverse-curriculum probe the prefix was
        // applied with real steps, so it sits in BOTH `reference_depth` (= D_full)
        // and the episode's depth. Subtracting it from both leaves the numerator
        // unchanged (heuristic_suffix - agent_suffix) while fixing the denominator,
        // which was D_full instead of the suffix the agent actually controls.
        // Without this, RC rewards are scaled by suffix/D_full < 1 -- compressed
        // toward zero, so a deeply scaffolded episode looks better than an
        // equally-performing from-scratch one. Measured on iter-72 shards:
        // rc mean -0.231 vs from-scratch -0.364, a ratio of 0.63 that the
        // compression alone accounts for.
        let scaffold = reference_override.map_or(0.0, |(_, _, s)| s);
        let ref_suffix = (reference_depth - scaffold).max(1.0);
        let terminal_evaluator = |e: &TilersEnv| -> f32 {
            if !e.inner.done() { crate::reward::NOT_DONE_SCORE } else {
                let d = (e.inner.depth(true, true) as f32 - scaffold).max(0.0);
                crate::reward::done_score(ref_suffix, d, tau)
            }
        };

        // In-episode progress telemetry. The gatherer only ever printed at
        // "Starting Env" and at termination, which was fine when episodes ran
        // tens of steps -- but a full-weight episode runs 2400-3900 steps and can
        // occupy a worker for the better part of an hour, during which the log is
        // silent and there is no way to tell a slow episode from a hung one or to
        // know how far along a wave is. Interval is set by GATHER_PROGRESS_EVERY
        // (unset/0 = off, matching the RC_CUSP_LOG env-gated telemetry pattern).
        // Self-limiting: an episode shorter than the interval never prints, so
        // the small-PP band stays quiet and only the long merges report.
        let progress_every: usize = std::env::var("GATHER_PROGRESS_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        for step in 0..max_actions {
            if progress_every > 0 && step > 0 && step % progress_every == 0 {
                let (fm, ft) = game.factor_progress(&tilers_env.inner);
                println!(
                    "[Gatherer {}] progress step={}/{} mw={} factors={}/{} ({:.0}%) depth={:.0}",
                    self.gather_id,
                    step,
                    max_actions,
                    pp_weights(&game).1,
                    fm,
                    ft,
                    if ft > 0 { 100.0 * fm as f32 / ft as f32 } else { 0.0 },
                    tilers_env.inner.depth(true, true),
                );
            }
            // Playout cap randomization [Wu 2020, §3.1]:
            // Full searches are run on a random fraction of turns and recorded
            // for training. Fast searches use a smaller budget and are used
            // only for move selection.
            let is_full_search = rng.random::<f32>() < self.p_full_search;
            let steps = if is_full_search { self.mcts_steps } else { self.fast_steps };

            // Inject Dirichlet noise into the MCTS root prior before full
            // searches to encourage exploration of low-prior moves during data
            // generation. Has no effect if the root doesn't exist yet.
            // Reference: [Wu 2020, §2].
            if is_full_search && self.dirichlet_epsilon > 0.0 {
                let valid = tilers_env.inner.valid_actions();
                let raw_noise = self._dirichlet_noise(valid.len(), rng);
                let noise_map: HashMap<Action, f32> = valid.iter()
                    .zip(raw_noise.iter())
                    .map(|(&a, &n)| (rl::encode(&tilers_env.inner, a).expect("valid_actions ids always encode") as Action, n as f32))
                    .collect();
                mcts.perturb_root_prior(&noise_map, self.dirichlet_epsilon);
            }

            let root = mcts.run(&tilers_env, client, steps, c_puct, &terminal_evaluator, is_full_search);

            // Resignation check (after the search, before recording or
            // stepping). root.value is the visit-weighted Q estimate at
            // the current state — the agent's best estimate of "how is
            // this position going". We resign on a sustained low-Q streak
            // alone: Q ≤ threshold for `resign_consecutive_moves`
            // consecutive moves. A violating move resets the streak.
            // Always capture root.value + over-solver state (cheap), even
            // when resignation_enabled is false, so the calibration log
            // can compute counterfactuals across any threshold.
            let q = root.value;
            let current_depth = tilers_env.inner.depth(true, true) as f32;
            let over_solver = current_depth > reference_depth;
            if self.resignation_log_dir.is_some() {
                q_trace.push(q);
                over_solver_trace.push(over_solver);
            }

            if resignation_enabled {
                // Resign on sustained low Q alone. The former
                // `&& over_solver` gate (resign only once past the solver's
                // depth) suppressed ~96% of resignations — on the iter-6
                // no-resign sample it fired on just 3.3% of episodes.
                // Dropping it and resigning at Q ≤ -0.9 for K consecutive
                // moves cuts ~44% of zero-progress episodes ~29 steps early
                // (≈1.7x gather throughput) while wrongly resigning <3% of
                // eventual wins — under the AGZ 5% guideline. `over_solver`
                // is still traced above for the calibration log.
                if q <= self.resign_value_threshold {
                    low_value_streak += 1;
                } else {
                    low_value_streak = 0;
                }

                // Step floor: on hard envs Q is pinned near -1 from the
                // initial state, so the streak alone fires at ~step 5 with a
                // 31% false-positive rate (2026-07-20 calibration). Requiring
                // the episode to first run max(20, mult*ref_depth) steps
                // restores outcome information (FP 8.0% at mult=1.25).
                let resign_min_step = if self.resign_min_step_ref_mult > 0.0 {
                    (self.resign_min_step_ref_mult * reference_depth)
                        .ceil()
                        .max(20.0) as usize
                } else {
                    0
                };
                if resign_allowed
                    && low_value_streak >= self.resign_consecutive_moves
                    && step >= resign_min_step
                {
                    println!(
                        "[Gatherer {}] Resigning at step {}: Q={:.3} \
                         depth={} ref_depth={} streak={} min_step={}",
                        self.gather_id, step, q, current_depth as i32,
                        reference_depth as i32, low_value_streak, resign_min_step,
                    );
                    resigned = true;
                    break;
                }
            }

            // Snapshot state before stepping (shared between temp_data and all_steps_data).
            // The observation is the 10-channel board; `valid_actions()` (the
            // wrapper) already returns encoded `u16` ids.
            let board = tilers_env.build_obs().board;
            let num_ancillas = tilers_env.inner.num_ancillas();
            let valid_actions: Vec<Action> = tilers_env
                .inner
                .valid_actions()
                .iter()
                .map(|&a| rl::encode(&tilers_env.inner, a).expect("valid_actions ids always encode") as Action)
                .collect();
            let step_height = tilers_env.inner.height;
            let step_width = tilers_env.inner.width;

            // Only record training data for full searches.
            if is_full_search {
                let n_total: usize = root.edge_visits.iter().sum();
                let edge_visits: HashMap<Action, usize> = mcts
                    .policy_target(c_puct)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(a, p)| (a, (p * n_total as f32).round() as usize))
                    .collect();
                temp_data.push((
                    board.clone(),
                    num_ancillas,
                    valid_actions.clone(),
                    edge_visits,
                ));
            }

            let action = self.select_action(&root, &tilers_env.inner, step, rng);

            // Record every step for trajectory saving.
            all_steps_data.push((
                board,
                num_ancillas,
                valid_actions,
                action,
                step_height,
                step_width,
            ));

            let a = rl::decode(&tilers_env.inner, action as usize)
                .expect("gatherer produced an invalid action id");
            let _ = tilers_env.inner.step(a);
            tilers_env.inner.finish_cultivating(None, None);

            if tilers_env.inner.done() {
                break;
            }

            mcts.advance_root(action);
        }

        tilers_env.inner.set_cultivation_time(10);
        let solution_depth = tilers_env.inner.depth(true, true);

        // Terminal value target. A finished episode scores against the full
        // heuristic reference. An UNFINISHED episode is relabeled via Hindsight
        // Experience Replay: we rebuild the goal as exactly what the agent did
        // execute (achieved_goal_env, from the pristine S0 `game`), solve THAT
        // with the heuristic, and score the agent's actual depth against it — so
        // every episode that made any progress yields a graded, non-floor
        // signal instead of a flat -1. Zero-progress (or an unsolvable hindsight
        // goal) still floors to -1.
        // `reward_kind` records the provenance of `score` so the trainer/analysis
        // can tell natural completions from HER-relabeled partials from the
        // residual floor (and reweight or filter on it): "done" = finished the
        // full goal, "her" = partial, scored against the achieved sub-goal,
        // "floor" = zero progress or a hindsight goal the heuristic couldn't
        // solve. `achieved_objectives` is how many objectives the agent actually
        // completed (the HER goal size), for the flywheel metric.
        let (score, reward_kind, achieved_objectives): (f32, &'static str, usize) =
            if tilers_env.inner.done() {
                // Same scaffold-relative basis as `terminal_evaluator` above, or
                // the value head's targets disagree with what MCTS backed up.
                let d = (solution_depth as f32 - scaffold).max(0.0);
                let ref_d = ref_suffix;
                // Shared done-scorer (reward.rs) — must match the search
                // terminal_evaluator above so the value head's training
                // targets agree with the scalars MCTS backed up.
                (
                    crate::reward::done_score(ref_d, d, self.reward_saturation_temperature),
                    "done",
                    game.num_objectives(),
                )
            } else if reference_override.is_some() {
                // REVERSE-CURRICULUM partial. The agent stopped short from a
                // mid-solution S_k. HER's achieved_goal_env would rebuild the
                // goal on S_k's mid-transport layout (set_layout wipes runtime,
                // keeps packed positions) — a layout the greedy solver cannot
                // re-plan (every factor has 0 valid edge cells → no_ready_pp
                // give-up; see hindsight.rs:107 / solver/transport.rs). That
                // both spams `[stuck]` and floors recoverable partials. So for
                // reverse-curriculum episodes we do NOT re-solve: grade by
                // objective-completion fraction against the S_k sub-problem.
                // No solver call → no spam, and partial progress is credited.
                let start_objs = game.num_objectives();
                let remaining = tilers_env.inner.num_objectives();
                let achieved = start_objs.saturating_sub(remaining);
                if achieved == 0 || start_objs == 0 {
                    (-1.0, "floor", 0)
                } else {
                    // frac ∈ (0,1): completed none → -1, all-but-one → ~0.
                    // Cusp-cap the reverse partial too (Axis A2): grade against
                    // min(S_k objectives, frontier+margin) so a large-objective
                    // S_k doesn't floor-collapse. For the usual small near-goal
                    // S_k this equals the plain fraction.
                    let r = Self::cusp_partial_reward(
                        achieved, start_objs, self.cusp_frontier, self.cusp_margin);
                    (r, "her", achieved)
                }
            } else if self.cusp_reward
                && (game.num_objectives() > self.cusp_frontier
                    // ...OR the env is ONE wide PauliProduct. The gate used to
                    // count OBJECTIVES only, but full-weight envs are a single
                    // objective holding up to ~98 factors, so `1 > frontier` was
                    // false and every full-weight episode fell through to vanilla
                    // HER -- precisely the pathology the cusp reward exists to
                    // fix ("relabels a HARD env DOWN to the objectives already
                    // executed and scores it as an efficient success ... NO
                    // gradient to complete more"). It also paid a full
                    // achieved_goal_env + Solver::solve per failed episode for a
                    // score the outcome-only contract then overwrites with -1.
                    // Difficulty now lives in FACTORS within one objective, so
                    // fire whenever the widest merge exceeds the cusp budget
                    // (frontier + margin) in factor terms. For a single PP the
                    // budget resolves to `total`, i.e. dense merged/total - 1.
                    || pp_weights(&game).1 > self.cusp_frontier + self.cusp_margin)
            {
                // CUSP reward (Axis A2). Vanilla HER relabels a HARD env DOWN to
                // exactly the (usually 1-2) objectives the agent already executed
                // and scores it as an efficient success — so ~95% of effective
                // training targets collapse to trivial goals and the agent gets
                // NO gradient to complete more (the measured plateau root cause:
                // 0% done on >=4-objective envs, mean effective target 1.4). For
                // hard envs (num_objectives > cusp_frontier) we grade the partial
                // against the CUSP goal = min(num_obj, frontier + margin), a target
                // just past the agent's current reach, capped so far-past-cusp envs
                // stay informative instead of floor-collapsing. Reaching the cusp
                // → 0. HER is kept for the easy (<= frontier) band.
                //
                // Grade at the FACTOR level, not the objective level: an objective
                // (PauliProduct) is a bag of factors, and merging some-but-not-all
                // is real progress that objective-count throws away (it floors an
                // episode that merged 15 factors but finished 0 objectives). Factor
                // grading is far denser and gives partial credit WITHIN an
                // objective. Completed objectives vanish from the live queue, so —
                // like HER — factor_progress diffs S0 (`game`) against the final env
                // by exec.id. `achieved` (objective count) is still the record tag
                // for metric continuity; only the reward VALUE is factor-based.
                let start_objs = game.num_objectives();
                let achieved = start_objs.saturating_sub(tilers_env.inner.num_objectives());
                let (merged, total) = game.factor_progress(&tilers_env.inner);
                if merged == 0 {
                    (-1.0, "floor", 0)
                } else {
                    let r = Self::cusp_factor_reward(
                        merged, total, start_objs, self.cusp_frontier, self.cusp_margin);
                    (r, "cusp", achieved)
                }
            } else if !self.her_reward {
                // HER disabled: no relabel-down, and no achieved_goal_env +
                // Solver::solve for a reward the outcome-only contract discards.
                // A non-done episode is simply a loss.
                (-1.0, "floor", 0)
            } else {
                match game.achieved_goal_env(&tilers_env.inner) {
                    Ok(mut her_env) => {
                        // Capture the achieved-goal size before solving consumes it.
                        let k = her_env.num_objectives();
                        let her_ref = Solver::new()
                            .solve(&mut her_env, true)
                            .ok()
                            .map(|_| her_env.depth(true, true) as f32);
                        match her_ref {
                            Some(ref_d) => {
                                let d = solution_depth as f32;
                                // Baseline margin: shift the zero-reward point from
                                // "match the heuristic" to "(1 + margin)× the heuristic
                                // depth" so a partial completion close to the heuristic
                                // scores slightly positive instead of negative. HER-only
                                // (eval/done untouched → calibration preserved).
                                let baseline = (1.0 + self.her_reward_margin) * ref_d;
                                let ratio = (baseline - d) / (ref_d + 1e-6);
                                ((ratio / self.reward_saturation_temperature).tanh(), "her", k)
                            }
                            None => (-1.0, "floor", k),
                        }
                    }
                    Err(_) => (-1.0, "floor", 0),
                }
            };

        // Finish printout: log every episode that reaches done() (the agent
        // fully tiled the env), mirroring the "Starting Env" line. Distinct from
        // the resignation log — this is the positive event (a completion), with
        // the agent's depth vs the heuristic reference and whether it beat it.
        if tilers_env.inner.done() {
            println!(
                "[Gatherer {}] FINISHED Env(h={}, w={}, nb={}, no={}, tw={}, mw={}) | depth={} vs ref={} refacts={} | score={:.3} (beat_solver={})",
                self.gather_id,
                game.height,
                game.width,
                game.num_ancillas(),
                game.num_objectives(),
                pp_weights(&game).0,
                pp_weights(&game).1,
                solution_depth,
                reference_depth,
                reference_action_count,
                score,
                solution_depth < reference_depth,
            );
        } else {
            // Unfinished printout: mirror the evaluator's failure line — show the
            // factor-level completion fraction (the quantity the cusp reward
            // grades) so the gather log reveals HOW far each failed episode got,
            // not just that it fell short. `game` is the episode start (S0, or
            // S_k for reverse-curriculum episodes), so the fraction is relative
            // to the actual sub-problem the agent was handed.
            let (fm, ft) = game.factor_progress(&tilers_env.inner);
            println!(
                // ref/refacts added 2026-07-31: the solver's depth and action
                // count are the DISTANCE-TO-DONE for this env. Previously only
                // FINISHED lines carried them, i.e. they were available for
                // exactly the episodes that did not need stratifying. Logging
                // them here lets every episode be bucketed by how far from
                // terminal it started, which is what the value-ablation
                // experiment stratifies on (can the search reach a terminal
                // inside its budget, or is a constant leaf value blind?).
                "[Gatherer {}] UNFINISHED Env(h={}, w={}, nb={}, no={}, tw={}, mw={}) | factors {}/{} ({:.0}%) | depth={} vs ref={} refacts={} | score={:.3} kind={}{}",
                self.gather_id,
                game.height,
                game.width,
                game.num_ancillas(),
                game.num_objectives(),
                pp_weights(&game).0,
                pp_weights(&game).1,
                fm,
                ft,
                if ft > 0 { 100.0 * fm as f32 / ft as f32 } else { 0.0 },
                solution_depth,
                reference_depth,
                reference_action_count,
                score,
                reward_kind,
                if resigned { " (resigned)" } else { "" },
            );
        }

        EpisodeOutcome {
            reference_depth,
            solution_depth: solution_depth as f32,
            done: tilers_env.inner.done(),
            score,
            reward_kind,
            achieved_objectives,
            reference_action_count,
            temp_data,
            all_steps_data,
            q_trace,
            over_solver_trace,
            resigned,
            resign_allowed,
            resignation_enabled,
            height: game.height,
            width: game.width,
            num_ancillas: game.num_ancillas(),
            num_objectives: game.num_objectives(),
        }
    }

    /// Write one episode's records: resignation calibration log, winning
    /// trajectory (if score > 0), then the full-search training records —
    /// preserving the HER-saturation-floor and floor-keep-fraction skips.
    /// `is_reverse` tags training records with `reverse_curriculum`.
    fn write_episode(&self, o: &EpisodeOutcome, is_reverse: bool, rng: &mut impl Rng) -> (f32, f32, bool) {
        // Resignation calibration log (one JSON line per episode).
        // The no-resign sample (resign_allowed=false) provides the
        // ground truth for false-positive analysis: any episode whose
        // q_per_step would have triggered resignation at threshold T
        // but went on to win at the natural terminal is a false
        // positive at T. Sweep T post-hoc from this data to pick the
        // strictest threshold whose FP rate stays below 5%.
        if let Some(dir) = &self.resignation_log_dir {
            if let Err(e) = std::fs::create_dir_all(dir) {
                eprintln!(
                    "[Gatherer {}] failed to create resignation_log_dir {}: {}",
                    self.gather_id, dir, e,
                );
            } else {
                let filename = format!("{}/resignation_{}.jsonl", dir, self.gather_id);
                match std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&filename)
                {
                    Ok(mut f) => {
                        let record = json!({
                            "gather_id": self.gather_id,
                            "h": o.height,
                            "w": o.width,
                            "num_ancillas": o.num_ancillas,
                            "num_objectives": o.num_objectives,
                            "achieved_objectives": o.achieved_objectives,
                            "reference_depth": o.reference_depth,
                            "final_depth": o.solution_depth,
                            "final_score": o.score,
                            "done": o.done,
                            "resign_allowed": o.resign_allowed,
                            "resignation_enabled": o.resignation_enabled,
                            "resigned": o.resigned,
                            "resign_threshold_in_use": self.resign_value_threshold,
                            "resign_consecutive_moves_in_use": self.resign_consecutive_moves,
                            "num_steps": o.q_trace.len(),
                            "q_per_step": o.q_trace,
                            "over_solver_per_step": o.over_solver_trace,
                        });
                        if let Err(e) = writeln!(f, "{}", record) {
                            eprintln!(
                                "[Gatherer {}] failed to write resignation log line: {}",
                                self.gather_id, e,
                            );
                        }
                    }
                    Err(e) => eprintln!(
                        "[Gatherer {}] failed to open resignation log {}: {}",
                        self.gather_id, filename, e,
                    ),
                }
            }
        }

        // Write the full trajectory to trajectory_dir if we found a win.
        if o.score > 0.0 {
            if let Some(dir) = &self.trajectory_dir {
                std::fs::create_dir_all(dir).expect("Failed to create trajectory directory");

                let filename = format!("{}/traj_{}.json", dir, self.gather_id);

                // Buffer writes (~64 KB by default) so each writeln! is a
                // memcpy into the buffer rather than a syscall. On Lustre
                // small-record syncs are expensive; the implicit flush at
                // BufWriter drop (or the explicit one below) is one syscall
                // per episode instead of one per record.
                let traj_file_raw = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&filename)
                    .expect("Unable to open trajectory file");
                let mut traj_file = BufWriter::new(traj_file_raw);

                let depth = o.all_steps_data.len();
                let gamma = 0.80f32;

                for (step, (board, num_ancillas, valid_actions, action, s_height, s_width)) in o.all_steps_data.iter().enumerate() {
                    let board_json = Self::serialize_board(board);
                    let valid_actions_json = Value::Array(valid_actions.iter().map(|&a| Value::from(a)).collect());

                    // Near-one-hot edge visits: 99% weight to the action taken.
                    let n_others = valid_actions.len().saturating_sub(1);
                    let winner_weight: usize = if n_others > 0 { n_others * 99 } else { 1 };
                    let mut visits_map = Map::with_capacity(valid_actions.len());
                    for &va in valid_actions {
                        let w = if va == *action { winner_weight } else { 1 };
                        visits_map.insert(va.to_string(), Value::from(w));
                    }

                    // Exponential value target:
                    //   target_value = min(2.0 * gamma^(depth - step), 2.0) - 1.0
                    let steps_remaining = (depth - step) as i32;
                    let value = if steps_remaining > 2 {
                        (2.0 * gamma.powi(steps_remaining)).min(2.0) - 1.0
                    } else {
                        1.0
                    };

                    let record = json!({
                        "height": s_height,
                        "width": s_width,
                        "num_ancillas": num_ancillas,
                        "board": board_json,
                        "valid_actions": valid_actions_json,
                        "edge_visits": Value::Object(visits_map),
                        "reward": value,
                    });

                    writeln!(traj_file, "{}", record).expect("Failed to write trajectory record");
                }

                traj_file.flush().expect("Failed to flush trajectory file");
            }
        }

        // Drop HER episodes whose relabeled reward still saturates to the -1
        // floor: the agent did make partial progress, but compared against the
        // heuristic's depth on the achieved goal it was so inefficient that
        // tanh pinned to -1. Such a record would mislabel partial progress as
        // zero-progress and just dilutes the corpus with another uninformative
        // -1, so we don't write it. We keep the honest heuristic-achieved-depth
        // comparison (the true efficiency signal) — only the saturated examples
        // are filtered. "floor" (genuine zero-progress) and "done" are unaffected.
        // Saturated HER episodes are the same "uninformative -1" category as
        // floor, so they obey the same knob instead of being dropped
        // unconditionally: at floor_keep_fraction 1.0 they are KEPT. The drop
        // happens before temp_data is written, so it discarded that episode's
        // full-search states -- the only ones that become policy targets. Cheap
        // to throw away when episodes were short; at ~0.7M network evaluations
        // per full-weight episode it is the most expensive data we produce.
        const HER_SATURATION_FLOOR: f32 = -0.999;
        if o.reward_kind == "her"
            && o.score <= HER_SATURATION_FLOOR
            && self.floor_keep_fraction < 1.0
            && rng.random::<f32>() >= self.floor_keep_fraction
        {
            return (o.solution_depth, o.reference_depth, o.done);
        }

        // Subsample zero-progress "floor" episodes so the corpus isn't dominated
        // by uninformative -1s. With floor_keep_fraction < 1.0 we keep only that
        // fraction of floor episodes (chosen at random), skewing the written
        // distribution toward the rare "done"/"her" signal the trainer can learn
        // from. "done" and graded "her" episodes are never dropped here.
        if o.reward_kind == "floor"
            && self.floor_keep_fraction < 1.0
            && rng.random::<f32>() >= self.floor_keep_fraction
        {
            return (o.solution_depth, o.reference_depth, o.done);
        }

        // Write full-search data to output_path as normal.
        // BufWriter batches the per-record writes into one syscall per
        // ~64 KB on flush — important on Lustre where every individual
        // write is a small sync.
        let file_raw = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)
            .expect("Unable to open output file");
        let mut file = BufWriter::new(file_raw);

        for (board, num_ancillas, va, ev) in &o.temp_data {
            let board_json = Self::serialize_board(board);
            let valid_actions_json = Value::Array(va.iter().map(|&a| Value::from(a)).collect());
            let mut visits_map = Map::with_capacity(ev.len());
            for (action, count) in ev {
                visits_map.insert(action.to_string(), Value::from(*count));
            }
            let visits_json = Value::Object(visits_map);

            let record = json!({
                "height": o.height,
                "width": o.width,
                "num_ancillas": num_ancillas,
                "board": board_json,
                "valid_actions": valid_actions_json,
                "edge_visits": visits_json,
                "reward": o.score,
                "reward_kind": o.reward_kind,
                "achieved_objectives": o.achieved_objectives,
                "reverse_curriculum": is_reverse,
                // Generation metadata for gather-side dashboards (done-rate by
                // merge weight / cluster count). env_k=0 => non-clustered arm.
                "env_mw": self.env_mw,
                "env_tw": self.env_tw,
                "env_k": self.env_k,
            });
            writeln!(file, "{}", record).expect("Failed to write record");
        }
        file.flush().expect("Failed to flush file");

        // Expert-iteration gold banking (Phase 2). Bank an episode only if it is a
        // genuine HIGH-QUALITY hard win, then pin it into every training set (with
        // a `"gold": true` tag) so it never ages out of the K-window. Criteria
        // (tightened 2026-07-07 after finding the bank was 83% reverse, tie-level
        // junk that compounded the wrong behavior):
        //   - `done`: finished the full goal;
        //   - `!is_reverse`: FULL-env win only — reverse-curriculum wins are on
        //     scaffolded near-goal sub-problems (they replay the heuristic prefix,
        //     so they only ~tie it: measured mean margin +0.055 vs +0.315 for
        //     non-reverse) and must not pollute the compounding bank;
        //   - `score >= gold_min_reward`: beat the heuristic by a REAL margin, not
        //     a tie or a sliver (was `> 0.0`, which banked tie-level wins);
        //   - `reference_action_count >= gold_min_len`: hardness floor.
        // IN ADDITION to the normal output_path write above; the normal corpus is
        // unchanged.
        if let Some(gold_dir) = &self.gold_shard_dir {
            if o.done
                && !is_reverse
                && o.score >= self.gold_min_reward
                && o.reference_action_count >= self.gold_min_len
            {
                if let Err(e) = std::fs::create_dir_all(gold_dir) {
                    eprintln!(
                        "[Gatherer {}] failed to create gold_shard_dir {}: {}",
                        self.gather_id, gold_dir, e,
                    );
                } else {
                    let gold_path = format!("{}/gold-{}.jsonl", gold_dir, self.gather_id);
                    match std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&gold_path)
                    {
                        Ok(gold_raw) => {
                            let mut gold_file = BufWriter::new(gold_raw);
                            for (board, num_ancillas, va, ev) in &o.temp_data {
                                let board_json = Self::serialize_board(board);
                                let valid_actions_json =
                                    Value::Array(va.iter().map(|&a| Value::from(a)).collect());
                                let mut visits_map = Map::with_capacity(ev.len());
                                for (action, count) in ev {
                                    visits_map.insert(action.to_string(), Value::from(*count));
                                }
                                let visits_json = Value::Object(visits_map);

                                let record = json!({
                                    "height": o.height,
                                    "width": o.width,
                                    "num_ancillas": num_ancillas,
                                    "board": board_json,
                                    "valid_actions": valid_actions_json,
                                    "edge_visits": visits_json,
                                    "reward": o.score,
                                    "reward_kind": o.reward_kind,
                                    "achieved_objectives": o.achieved_objectives,
                                    "reverse_curriculum": is_reverse,
                                    "gold": true,
                                });
                                if let Err(e) = writeln!(gold_file, "{}", record) {
                                    eprintln!(
                                        "[Gatherer {}] failed to write gold record: {}",
                                        self.gather_id, e,
                                    );
                                    break;
                                }
                            }
                            if let Err(e) = gold_file.flush() {
                                eprintln!(
                                    "[Gatherer {}] failed to flush gold file {}: {}",
                                    self.gather_id, gold_path, e,
                                );
                            }
                        }
                        Err(e) => eprintln!(
                            "[Gatherer {}] failed to open gold file {}: {}",
                            self.gather_id, gold_path, e,
                        ),
                    }
                }
            }
        }

        (o.solution_depth, o.reference_depth, o.done)
    }
}

/// ----------------------------------------------------------------------------
/// Python bindings
/// ----------------------------------------------------------------------------
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;

/// Run a single gather iteration. Generates one environment, runs MCTS on it,
/// writes the resulting data to output_dir/output-{worker_id}.jsonl, and
/// returns (solution_depth, reference_depth, done).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    worker_id,
    num_handlers,
    output_dir,
    arena_tag = String::new(),
    height = 4,
    width = 4,
    num_objectives = 4,
    min_num_objectives = 2,
    num_blanks = 2,
    mcts_steps = 800,
    fast_steps = 140,
    p_full_search = 0.25,
    dirichlet_epsilon = 0.25,
    reward_saturation_temperature = 0.3,
    c_puct = 1.4,
    max_generated_depth = 10_000,
    num_shuffles = 0,
    trajectory_dir = None,
    seed = None,
    resign_value_threshold = -0.9,
    resign_consecutive_moves = 5,
    resign_min_step_ref_mult = 0.0,
    no_resign_rate = 0.1,
    resignation_log_dir = None,
    max_action_multiplier = 1.2,
    max_pp_weight = None,
    max_pp_cost = None,
    clustered_wide_pp_fraction = 0.0,
    clustered_wide_pp_weight = 15,
    clustered_wide_pp_block_side = 0,
    clustered_wide_pp_num_clusters = 1,
    clustered_wide_pp_weight_min = 0,
    clustered_learnable_prob = 0.0,
    clustered_learnable_min = 0,
    clustered_learnable_max = 0,
    clustered_max_generated_depth = None,
    clustered_reverse_prob = -1.0,
    floor_keep_fraction = 1.0,
    her_reward_margin = 0.0,
    reverse_curriculum = false,
    reverse_curriculum_min_len = 50,
    reverse_curriculum_max_probes = 5,
    reverse_curriculum_prefix_start = 10,
    reverse_curriculum_prob = 1.0,
    gold_shard_dir = None,
    gold_min_len = usize::MAX,
    gold_min_reward = 0.0,
    her_reward = true,
    cusp_reward = false,
    cusp_frontier = 2,
    cusp_margin = 2,
    demo_fraction = 0.0,
    demo_min_objectives = 2,
    demo_subsample = 1.0,
    demo_only_min_weight = 0,
    demo_target_states = 0,
    full_weight_rc_prob = 0.0,
    full_weight_rc_tail = 0,
    deep_play_temp = 0.0,
    deep_play_after = 20,
    gather_min_ref_actions = 0,
    gather_easy_keep_fraction = 0.15,
))]
pub fn run_gatherer(
    worker_id: u32,
    num_handlers: usize,
    output_dir: String,
    arena_tag: String,
    height: usize,
    width: usize,
    num_objectives: usize,
    min_num_objectives: usize,
    num_blanks: usize,
    mcts_steps: usize,
    fast_steps: usize,
    p_full_search: f32,
    dirichlet_epsilon: f32,
    reward_saturation_temperature: f32,
    c_puct: f32,
    max_generated_depth: usize,
    num_shuffles: usize,
    trajectory_dir: Option<String>,
    seed: Option<i32>,
    resign_value_threshold: f32,
    resign_consecutive_moves: usize,
    resign_min_step_ref_mult: f32,
    no_resign_rate: f32,
    resignation_log_dir: Option<String>,
    max_action_multiplier: f32,
    max_pp_weight: Option<usize>,
    max_pp_cost: Option<usize>,
    clustered_wide_pp_fraction: f32,
    clustered_wide_pp_weight: usize,
    clustered_wide_pp_block_side: usize,
    clustered_wide_pp_num_clusters: usize,
    clustered_wide_pp_weight_min: usize,
    clustered_learnable_prob: f32,
    clustered_learnable_min: usize,
    clustered_learnable_max: usize,
    clustered_max_generated_depth: Option<usize>,
    clustered_reverse_prob: f32,
    floor_keep_fraction: f32,
    her_reward_margin: f32,
    reverse_curriculum: bool,
    reverse_curriculum_min_len: usize,
    reverse_curriculum_max_probes: usize,
    reverse_curriculum_prefix_start: usize,
    reverse_curriculum_prob: f32,
    gold_shard_dir: Option<String>,
    gold_min_len: usize,
    gold_min_reward: f32,
    her_reward: bool,
    cusp_reward: bool,
    cusp_frontier: usize,
    cusp_margin: usize,
    demo_fraction: f32,
    demo_min_objectives: usize,
    demo_subsample: f32,
    demo_only_min_weight: usize,
    demo_target_states: usize,
    full_weight_rc_prob: f32,
    full_weight_rc_tail: usize,
    deep_play_temp: f32,
    deep_play_after: usize,
    gather_min_ref_actions: usize,
    gather_easy_keep_fraction: f32,
) -> PyResult<Option<(f32, f32, bool)>> {
    let num_slots = 2048;
    let lookahead = DEFAULT_LOOKAHEAD;

    let arena_name = if arena_tag.is_empty() {
        format!("mcts_{}_{}", num_slots, num_handlers)
    } else {
        format!("mcts_{}_{}_{}", arena_tag, num_slots, num_handlers)
    };

    let arena: Arena<TilersSlot> =
        Arena::create_or_open(&arena_name, num_slots, num_handlers)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open arena: {e}")))?;
    let client = TilersIpcClient::new(arena, worker_id);

    let output_path = format!("{}/output-{}.jsonl", output_dir, worker_id);
    let mut gatherer = Gatherer::new(
        8,
        mcts_steps,
        fast_steps,
        p_full_search,
        output_path,
        0.20,
        dirichlet_epsilon,
        lookahead,
        worker_id as usize,
        trajectory_dir,
        Some(reward_saturation_temperature),
        None,
        Some(max_action_multiplier),
        Some(resign_value_threshold),
        Some(resign_consecutive_moves),
        Some(resign_min_step_ref_mult),
        Some(no_resign_rate),
        resignation_log_dir,
        Some(floor_keep_fraction),
        Some(her_reward_margin),
    );
    gatherer.set_reverse_curriculum(
        reverse_curriculum,
        reverse_curriculum_min_len,
        reverse_curriculum_max_probes,
        reverse_curriculum_prefix_start,
        reverse_curriculum_prob,
    );
    gatherer.set_gold_banking(gold_shard_dir, gold_min_len, gold_min_reward);
    gatherer.set_her_reward(her_reward);
    gatherer.set_cusp_reward(cusp_reward, cusp_frontier, cusp_margin);
    gatherer.set_demo(demo_fraction, demo_min_objectives, demo_subsample, demo_only_min_weight, demo_target_states);
    gatherer.set_full_weight_rc(full_weight_rc_prob, full_weight_rc_tail);
    gatherer.set_deep_play(deep_play_temp, deep_play_after);

    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s as u64)
    } else {
        StdRng::from_rng(&mut rand::rng())
    };

    // Validate and skip degenerate configurations, returning None so the
    // orchestrator knows to just submit another job.
    let no = rng.random_range(min_num_objectives..=num_objectives);
    if num_blanks >= (height * width) - 1 || (height <= 2 && width <= 2) {
        return Ok(None);
    }

    let mut env = Environment::new(height, width, num_blanks);
    if let Some(s) = seed {
        env.set_seed(Some(s as u64));
    }
    // Cap PauliProduct weight/cost (easy-env curriculum) BEFORE random_start so
    // the generated objectives respect it. None = unbounded (historical). When
    // max_pp_cost is set it takes over as a Y-aware cost budget (X/Z=1, Y=2).
    env.set_max_pp_weight(max_pp_weight);
    env.set_max_pp_cost(max_pp_cost);
    let mut is_clustered = false;
    if clustered_wide_pp_fraction > 0.0
        && rng.random_range(0.0f32..1.0) < clustered_wide_pp_fraction
    {
        is_clustered = true;
        // Clustered wide PP: factors co-located in K compact blocks so the wide
        // merge solves under the action cap. Do NOT re-shuffle after — that would
        // scatter the cluster and defeat the purpose. Draw K uniformly in
        // [1, num_clusters] per env so the corpus is a DIFFICULTY MIX: K=1 is the
        // easy tight blob (bootstrap), higher K adds inter-cluster routing (hard,
        // more transferable) — ensuring a good amount of hard examples.
        let k = rng.random_range(1..=clustered_wide_pp_num_clusters.max(1));
        // Weight mixture (2026-07-21): the advance mechanism pins the generated
        // weight at the current effective cap, so the corpus is a single-weight
        // spike (18.3k done records at w19, 1.7k across w13-18) and each
        // weight's mass ages out of the K=5 replay window as the cap moves.
        // Drawing w ~ uniform[min, cap] keeps the whole band populated.
        // weight_min = 0 disables (legacy always-cap).
        // TRUE FULL-WEIGHT: the PP touches ALL non-ancilla qubits, i.e. weight =
        // (h*w - num_blanks) (num_blanks IS the ancilla budget). A small
        // clustered_learnable_prob fraction is instead a narrow LEARNABLE
        // near-frontier band [learnable_min, learnable_max] (agent plays +
        // reverse-curriculum, to keep climbing its wide-merge frontier); the rest
        // are full-weight (fraction 1.0), which the demo-only path imitates.
        let full_w = (height * width).saturating_sub(num_blanks).max(1);
        let w = if clustered_learnable_prob > 0.0
            && clustered_learnable_max >= clustered_learnable_min
            && clustered_learnable_min > 0
            && rng.random_range(0.0f32..1.0) < clustered_learnable_prob
        {
            let hi = clustered_learnable_max.min(full_w).max(clustered_learnable_min);
            rng.random_range(clustered_learnable_min..=hi)
        } else {
            full_w
        };
        env.random_start_clustered_pp(w, clustered_wide_pp_block_side, k);
        let (tw, mw) = pp_weights(&env);
        gatherer.set_env_meta(mw, k);
        gatherer.set_env_total_weight(tw);
    } else {
        env.random_start(no, false);
        env.shuffle(num_shuffles);
        let (tw, mw) = pp_weights(&env);
        gatherer.set_env_meta(mw, 0);
        gatherer.set_env_total_weight(tw);
    }
    gatherer.set_clustered_reverse_prob(clustered_reverse_prob);

    // Solve the POST-shuffle env (the one we actually gather on) once: it
    // gates both the max-depth cap and the trivial-env reject. Checking the
    // pre-shuffle env was a bug — shuffle advances the state, so a non-trivial
    // start can become auto-execute-only after shuffling and slip through.
    let solver = Solver::new();
    let mut temp_env = env.clone();
    if let Ok(solution) = solver.solve(&mut temp_env, false) {
        let n = solution.len();
        // Per-arm cap: the clustered wide-PP arm may carry its own action-count
        // cap. Rationale (2026-07-20 placement study): with spread anchors the
        // hard clustered geometry (K>=4 corners/spread) needs 185-225 solver
        // actions at w19 — the global tier cap of 220 rejects >50% of exactly
        // those draws (reject-and-regenerate below), silently censoring the
        // hard tail the arm exists to supply. A clustered cap of ~260-280
        // admits 80-95% of it without loosening the generic arm's tier gate.
        //
        // ACTION-BASED GATING (2026-08-02, user). The cap above is applied at
        // GENERATION, before the demo-only decision is taken at play time
        // (`demo_only_min_weight`, ~line 654), so a single threshold was gating
        // two populations with opposite needs:
        //   - clustered envs the agent PLAYS: these must respect the tier's
        //     action-count curriculum. Measured over 345 episodes in iter 82,
        //     heuristic action count is what actually separates a win from a
        //     floor (+1.02 sd; win median 50 actions, floor median 156), while
        //     PauliProduct weight barely separates at all (+0.66 sd, and the win
        //     rate is flat 17-24% across mw 4-12). Left uncapped at 3000 the
        //     played band ran to 450 actions -- harder along an axis the agent
        //     does not fail on, and episodes twice as long, halving the number
        //     of independent episodes per gather.
        //   - full-weight DEMOS: heuristic reference ~650-2000 actions by
        //     construction, so the tier cap (135) would reject every one of them
        //     at generation and the full-weight slice would vanish.
        // So: played clustered envs take the tier cap, demo-only ones keep the
        // permissive per-arm cap.
        let (_, gen_mw) = pp_weights(&env);
        let is_demo_only = demo_only_min_weight > 0 && gen_mw >= demo_only_min_weight;
        let eff_cap = if is_clustered && !is_demo_only {
            max_generated_depth
        } else if is_clustered {
            clustered_max_generated_depth.unwrap_or(max_generated_depth)
        } else {
            max_generated_depth
        };
        if n > eff_cap {
            return Ok(None);
        }
        // Reject trivial envs — ones the heuristic clears with the global
        // AutoExecute action alone (no real placement/movement decision). The
        // agent "solves" these just by selecting auto-execute, so they carry no
        // training signal. Mirrors the init_db.py holdout filter.
        if n == 0
            || solution
                .into_iter()
                .all(|a| matches!(a, tilers::core::enums::Action::AutoExecute))
        {
            return Ok(None);
        }
        // Frontier floor (2026-07-11): reject envs whose heuristic plan is
        // shorter than gather_min_ref_actions, so gather concentrates on
        // frontier-length episodes — their LATE-GAME states cover the easy
        // band for free (superset argument), while easy episodes teach
        // nothing new. gather_easy_keep_fraction of episodes bypass the
        // filter as insurance against easy-band forgetting (the eval window
        // still tests F-1 and the gate will veto regression). This is a
        // FLOOR inside the curriculum's cap — not a license to gather
        // beyond the frontier (the June "harder-envs backfire" regime:
        // done-rate ~0 = no outcome signal). 0 disables.
        if gather_min_ref_actions > 0
            && n < gather_min_ref_actions
            && rng.random::<f32>() >= gather_easy_keep_fraction
        {
            return Ok(None);
        }
    }

    let mut game_rng = StdRng::from_rng(&mut rand::rng());
    let result = gatherer.gather(&env, &client as &dyn InferenceClient<TilersEnv>, c_puct, &mut game_rng);

    Ok(Some(result))
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use mcts_core::node::Node;
    use tilers::env::Environment as TilersEnvInner;
    use crate::client::TrivialTilersIpcClient;

    fn default_gatherer(output_path: &str) -> Gatherer {
        Gatherer::new(
            1,                    // batch_size
            5,                    // mcts_steps
            2,                    // fast_steps
            1.0,                  // p_full_search (all turns are full searches)
            output_path.to_string(),
            0.0,                  // noise_strength
            0.0,                  // dirichlet_epsilon
            1,                    // lookahead
            0,                    // gather_id
            None,                 // trajectory_dir
            Some(1.0),            // reward_saturation_temperature
            None,                 // max_actions
            None,                 // max_action_multiplier (default 1.2)
            Some(2.0),            // resign_value_threshold (>1.0 disables)
            Some(0),              // resign_consecutive_moves (0 disables)
            None,                 // resign_min_step_ref_mult (moot: resignation off)
            Some(0.0),            // no_resign_rate
            None,                 // resignation_log_dir
            None,                 // floor_keep_fraction (default 1.0)
            None,                 // her_reward_margin (default 0.0)
        )
    }

    // ─── Gatherer::new ──────────────────────────────────────────────────────

    #[test]
    fn test_new_sets_explicit_reward_saturation_temperature() {
        let g = Gatherer::new(
            8, 10, 2, 0.5, "/dev/null".into(), 0.1, 0.25, 2, 1,
            None, Some(0.3), Some(100), None, None, None, None, None, None, None, None,
        );
        assert!((g.reward_saturation_temperature - 0.3).abs() < 1e-6);
        assert_eq!(g.mcts_steps, 10);
        assert_eq!(g.fast_steps, 2);
    }

    #[test]
    fn test_new_defaults_reward_saturation_temperature() {
        // Default mirrors the slope-at-origin of the previous clip+normalize
        // default (reward_ratio_limit = 0.3 → slope 1/0.3); see Gatherer::new.
        let g = Gatherer::new(
            8, 10, 2, 0.5, "/dev/null".into(), 0.0, 0.0, 2, 0,
            None, None, None, None, None, None, None, None, None, None, None,
        );
        assert!((g.reward_saturation_temperature - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_new_defaults_resignation_params() {
        let g = Gatherer::new(
            8, 10, 2, 0.5, "/dev/null".into(), 0.0, 0.0, 2, 0,
            None, None, None, None, None, None, None, None, None, None, None,
        );
        assert!((g.resign_value_threshold - (-0.9)).abs() < 1e-6);
        assert_eq!(g.resign_consecutive_moves, 5);
        assert!((g.no_resign_rate - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_new_sets_explicit_resignation_params() {
        let g = Gatherer::new(
            8, 10, 2, 0.5, "/dev/null".into(), 0.0, 0.0, 2, 0,
            None, None, None, None, Some(-0.5), Some(8), None, Some(0.2), None, None, None,
        );
        assert!((g.resign_value_threshold - (-0.5)).abs() < 1e-6);
        assert_eq!(g.resign_consecutive_moves, 8);
        assert!((g.no_resign_rate - 0.2).abs() < 1e-6);
    }

    // ─── _dirichlet_noise ───────────────────────────────────────────────────

    #[test]
    fn test_dirichlet_noise_correct_length() {
        let g = default_gatherer("/dev/null");
        let mut rng = StdRng::seed_from_u64(42);
        assert_eq!(g._dirichlet_noise(7, &mut rng).len(), 7);
    }

    #[test]
    fn test_dirichlet_noise_non_negative() {
        let g = default_gatherer("/dev/null");
        let mut rng = StdRng::seed_from_u64(42);
        assert!(g._dirichlet_noise(10, &mut rng).iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_dirichlet_noise_sums_to_one() {
        let g = default_gatherer("/dev/null");
        let mut rng = StdRng::seed_from_u64(42);
        let sum: f64 = g._dirichlet_noise(10, &mut rng).iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn test_dirichlet_noise_single_action_is_one() {
        let g = default_gatherer("/dev/null");
        let mut rng = StdRng::seed_from_u64(0);
        let noise = g._dirichlet_noise(1, &mut rng);
        assert_eq!(noise.len(), 1);
        assert!((noise[0] - 1.0).abs() < 1e-6);
    }

    // ─── _action_probabilities ───────────────────────────────────────────────

    #[test]
    fn test_action_probs_uniform_when_all_zero_visits() {
        let g = default_gatherer("/dev/null");
        let probs = g._action_probabilities(&[0, 0, 0, 0], 1.0);
        assert_eq!(probs.len(), 4);
        for p in &probs {
            assert!((p - 0.25).abs() < 1e-9, "expected 0.25 got {p}");
        }
    }

    #[test]
    fn test_action_probs_greedy_picks_max_visit() {
        let g = default_gatherer("/dev/null");
        let probs = g._action_probabilities(&[1, 5, 2], 0.0);
        assert!((probs[1] - 1.0).abs() < 1e-9, "probs[1]={}", probs[1]);
        assert!(probs[0] < 1e-9);
        assert!(probs[2] < 1e-9);
    }

    #[test]
    fn test_action_probs_greedy_splits_ties_equally() {
        let g = default_gatherer("/dev/null");
        let probs = g._action_probabilities(&[5, 5, 1], 0.0);
        assert!((probs[0] - 0.5).abs() < 1e-9);
        assert!((probs[1] - 0.5).abs() < 1e-9);
        assert!(probs[2] < 1e-9);
    }

    #[test]
    fn test_action_probs_proportional_at_temperature_one() {
        let g = default_gatherer("/dev/null");
        let probs = g._action_probabilities(&[1, 3], 1.0);
        assert!((probs[0] - 0.25).abs() < 1e-6, "probs[0]={}", probs[0]);
        assert!((probs[1] - 0.75).abs() < 1e-6, "probs[1]={}", probs[1]);
    }

    #[test]
    fn test_action_probs_high_temp_smoother_than_low_temp() {
        let g = default_gatherer("/dev/null");
        let visits = [1usize, 10];
        let diff_high = {
            let p = g._action_probabilities(&visits, 5.0);
            (p[0] - p[1]).abs()
        };
        let diff_low = {
            let p = g._action_probabilities(&visits, 0.5);
            (p[0] - p[1]).abs()
        };
        assert!(diff_high < diff_low, "high_temp diff={diff_high}, low_temp diff={diff_low}");
    }

    // ─── select_action ───────────────────────────────────────────────────────

    #[test]
    fn test_select_action_returns_valid_action() {
        let g = default_gatherer("/dev/null");
        let env = TilersEnvInner::new(3, 3, 1);
        let ids: Vec<Action> = env.valid_actions().iter()
            .map(|&a| tilers::rl::encode(&env, a).expect("valid_actions ids always encode") as Action)
            .collect();
        assert!(!ids.is_empty());

        let priors: HashMap<Action, f32> = ids.iter()
            .map(|&id| (id, 1.0 / ids.len() as f32))
            .collect();
        let node = Node::new(env.num_actions(), priors, 0.0, 0, None);

        let mut rng = StdRng::seed_from_u64(42);
        let action = g.select_action(&node, &env, 50, &mut rng);
        assert!(ids.contains(&action), "action {action} not in valid ids");
    }

    #[test]
    fn test_select_action_picks_dominant_at_late_step() {
        let g = default_gatherer("/dev/null");
        let env = TilersEnvInner::new(3, 3, 1);
        let ids: Vec<Action> = env.valid_actions().iter()
            .map(|&a| tilers::rl::encode(&env, a).expect("valid_actions ids always encode") as Action)
            .collect();
        assert!(ids.len() >= 2, "need ≥2 valid actions");

        let priors: HashMap<Action, f32> = ids.iter()
            .map(|&id| (id, 1.0 / ids.len() as f32))
            .collect();
        let mut node = Node::new(env.num_actions(), priors, 0.0, 0, None);
        let dominant = ids[0];
        node.edge_visits[dominant as usize] = 100_000;

        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..20 {
            assert_eq!(g.select_action(&node, &env, 1_000, &mut rng), dominant);
        }
    }

    // ─── serialize_board ─────────────────────────────────────────────────────

    #[test]
    fn test_serialize_board_shape() {
        // A board from a real env serializes to [layer][cell][10 ints].
        let env = TilersEnvInner::new(3, 3, 1);
        let board = TilersEnv::new(env, DEFAULT_LOOKAHEAD).build_obs().board;
        let v = Gatherer::serialize_board(&board);
        let layers = v.as_array().unwrap();
        assert_eq!(layers.len(), board.len());
        let cell = layers[0].as_array().unwrap()[0].as_array().unwrap();
        assert_eq!(cell.len(), CELL_FIELDS, "each cell carries CELL_FIELDS channels");
    }

    // ─── solve_with_heuristic ────────────────────────────────────────────────

    #[test]
    fn test_solve_with_heuristic_nonnegative_depth() {
        let g = default_gatherer("/dev/null");
        let mut env = TilersEnvInner::new(3, 3, 1);
        env.set_seed(Some(1));
        env.random_start(2, false);
        let (depth, actions) = g.solve_with_heuristic(&env);
        assert!(depth >= 0.0, "depth={depth}");
        assert!(!actions.is_empty(), "heuristic solution should be non-empty");
    }

    // ─── reverse curriculum ──────────────────────────────────────────────────

    #[test]
    fn test_make_reverse_start_objectives_monotone() {
        let g = default_gatherer("/dev/null");
        let mut env = TilersEnvInner::new(4, 4, 1);
        env.set_seed(Some(7));
        env.random_start(3, false);
        env.set_cultivation_time(10);

        let (_d_full, plan) = g.heuristic_typed_plan(&env);
        assert!(!plan.is_empty(), "heuristic plan should be non-empty");

        // prefix_len = 0 reproduces the true start S0 (all objectives remain).
        let (s0, _) = g.make_reverse_start(&env, &plan, 0);
        assert_eq!(
            s0.num_objectives(),
            env.num_objectives(),
            "prefix_len=0 must equal the true start's objective count",
        );

        // num_objectives() is monotonically non-increasing as prefix_len grows:
        // each replayed heuristic action can only complete objectives, never add.
        let mut prev = s0.num_objectives();
        for prefix_len in 1..=plan.len() {
            let (sk, _) = g.make_reverse_start(&env, &plan, prefix_len);
            let cur = sk.num_objectives();
            assert!(
                cur <= prev,
                "num_objectives increased at prefix_len={prefix_len}: {cur} > {prev}",
            );
            prev = cur;
        }

        // prefix_len = plan.len() replays the entire heuristic solution ⇒
        // done / near-done: no more remaining objectives than the true start.
        let (full, _) = g.make_reverse_start(&env, &plan, plan.len());
        assert!(
            full.num_objectives() <= s0.num_objectives(),
            "full replay should not have more objectives than S0",
        );
    }

    #[test]
    fn test_cusp_partial_reward() {
        let f = |a, n| Gatherer::cusp_partial_reward(a, n, 2, 2); // cusp = min(n, 4)
        // hard env (6 obj): graded against cusp=4, NOT floor-collapsed.
        assert!((f(1, 6) - (-0.75)).abs() < 1e-6);
        assert!((f(2, 6) - (-0.50)).abs() < 1e-6);
        assert!((f(4, 6) - 0.0).abs() < 1e-6);      // reached the cusp → 0
        assert!((f(5, 6) - 0.0).abs() < 1e-6);      // beyond cusp, capped → 0
        // very hard env (20 obj): still informative, not ~-0.95 floor.
        assert!((f(1, 20) - (-0.75)).abs() < 1e-6);
        // small env (3 obj): cusp shrinks to the env size.
        assert!((f(2, 3) - (-1.0 / 3.0)).abs() < 1e-6);
        assert!((f(3, 3) - 0.0).abs() < 1e-6);
        // reward is always in [-1, 0].
        for n in 1..25 { for a in 0..=n { let r = f(a, n); assert!((-1.0..=0.0).contains(&r)); } }
    }

    #[test]
    fn test_cusp_factor_reward() {
        let f = |m, t, n| Gatherer::cusp_factor_reward(m, t, n, 2, 2); // cusp = min(n, 4) objs
        // 6-obj / 12-factor env: budget = 12 * 4/6 = 8 factors.
        assert!((f(4, 12, 6) - (-0.5)).abs() < 1e-6);   // 4/8
        assert!((f(2, 12, 6) - (-0.75)).abs() < 1e-6);  // 2/8
        assert!((f(8, 12, 6) - 0.0).abs() < 1e-6);      // reached budget → 0
        assert!((f(10, 12, 6) - 0.0).abs() < 1e-6);     // beyond budget, capped → 0
        // very hard 20-obj / 40-factor env: budget = 40 * 4/20 = 8, still informative.
        assert!((f(4, 40, 20) - (-0.5)).abs() < 1e-6);  // NOT ~-0.9 floor
        // small env (num_obj <= cusp): budget = full total, grade whole env.
        assert!((f(3, 6, 3) - (-0.5)).abs() < 1e-6);    // budget = 6*3/3 = 6, 3/6
        // degenerate: no factors → floor.
        assert!((f(0, 0, 5) - (-1.0)).abs() < 1e-6);
        // reward always in [-1, 0].
        for n in 1..12 { for t in 1..30 { for m in 0..=t {
            let r = f(m, t, n); assert!((-1.0..=0.0).contains(&r));
        }}}
    }

    #[test]
    fn test_next_prefix_probe_pins_cusp() {
        // Simulate the reverse-curriculum cusp search against a known cusp P*
        // (the "agent" solves from scaffold prefix p iff p >= P*). The first
        // probe is seeded at `seed` (the prior, ~10); the rest bisect. Invariants:
        // lo (largest failed prefix) stays < P* and hi (smallest solved prefix)
        // stays >= P*; with enough budget the cusp is pinned (hi == P*).
        for &(len, pstar, seed, budget, expect_pinned) in &[
            (203usize, 10usize, 10usize, 8usize, true),  // cusp exactly at the prior
            (203, 60, 10, 9, true),                       // cusp ABOVE prior → search up
            (203, 3, 10, 8, true),                        // cusp BELOW prior → bisect down
            (50, 0, 10, 8, false),                        // solvable fully unscaffolded
            (203, 203, 10, 9, false),                     // needs full scaffold
        ] {
            let (mut lo, mut hi) = (0usize, len);
            let mut p = seed.min(len);
            for _ in 0..budget {
                if p >= pstar { hi = p; if p == 0 { break; } } else { lo = p; }
                match Gatherer::next_prefix_probe(lo, hi) {
                    Some(n) => p = n,
                    None => break,
                }
            }
            assert!(pstar == 0 || lo < pstar, "len={len} P*={pstar}: lo={lo} not below cusp");
            assert!(hi >= pstar, "len={len} P*={pstar}: hi={hi} below cusp");
            if expect_pinned {
                assert_eq!(hi, pstar, "len={len} P*={pstar}: cusp not pinned (lo={lo} hi={hi})");
            }
        }
    }

    /// Minimal reproduction: does replaying the heuristic plan reconstruct the
    /// solver's real intermediate state, and can the solver finish from S_k?
    /// Run with: cargo test --features python --lib diagnose_reverse_start -- --nocapture
    #[test]
    fn diagnose_reverse_start_replay() {
        let g = default_gatherer("/dev/null");
        let mut env = TilersEnvInner::new(10, 10, 8);
        env.set_seed(Some(3));
        env.random_start(4, false);
        env.set_cultivation_time(10);

        let (d_full, plan) = g.heuristic_typed_plan(&env);
        eprintln!("PLAN len={} d_full={} objectives={}", plan.len(), d_full, env.num_objectives());
        assert!(!plan.is_empty(), "need a non-empty plan");

        // (1) Replay the FULL plan exactly as make_reverse_start does, but CHECK
        // each step's Result and whether the replay actually reaches done().
        // If step_errors>0 or done=false, the replay diverges from the solver's
        // real state — that (not the solver) is the breakdown.
        let mut sk = env.clone();
        sk.set_cultivation_time(10);
        let mut step_errs = 0;
        for (i, a) in plan.iter().enumerate() {
            if let Err(e) = sk.step(a.clone()) {
                step_errs += 1;
                if step_errs <= 5 { eprintln!("  REPLAY step {i} REJECTED: {e:?}"); }
            }
            sk.finish_cultivating(None, None);
        }
        eprintln!(
            "FULL REPLAY: step_errors={} done={} remaining_objectives={}",
            step_errs, sk.done(), sk.num_objectives(),
        );

        // (2) Ask the solver to finish from mid-prefix intermediate states.
        for frac in [0.25f32, 0.5, 0.75] {
            let prefix = ((plan.len() as f32) * frac) as usize;
            let (mut s_k, _) = g.make_reverse_start(&env, &plan, prefix);
            let before = s_k.num_objectives();
            match Solver::new().solve(&mut s_k, true) {
                Ok(p) => eprintln!("  RE-SOLVE prefix={prefix} obj_remaining={before}: OK tail_len={}", p.len()),
                Err(e) => eprintln!("  RE-SOLVE prefix={prefix} obj_remaining={before}: STUCK/ERR {e:?}"),
            }
        }
    }

    /// Sweep many gather-like envs to find one where the reverse-start breaks
    /// down: either the replay diverges (step_errs>0 / !done) OR the solver
    /// gets stuck re-solving an intermediate state.
    /// RESULT (2026-07-06): 461 envs, ~1844 re-solves → 0 diverged, 0 stuck.
    /// The solver re-solves clean heuristic-prefix states fine; the gather
    /// `[stuck] no_ready_pp` is NOT from this path. #[ignore]d (slow, ~465s).
    /// cargo test --features python --lib sweep_reverse_start -- --ignored --nocapture
    #[test]
    #[ignore]
    fn sweep_reverse_start_breakdown() {
        let g = default_gatherer("/dev/null");
        let mut diverged = 0;
        let mut stuck = 0;
        let mut total = 0;
        let mut first_stuck_reported = false;
        for seed in 0u64..40 {
            for &no in &[2usize, 3, 4, 6] {
                for &nb in &[6usize, 12, 18] {
                    let mut env = TilersEnvInner::new(10, 10, nb);
                    env.set_seed(Some(seed));
                    env.random_start(no, false);
                    env.set_cultivation_time(10);
                    let (_d, plan) = g.heuristic_typed_plan(&env);
                    if plan.len() < 35 { continue; }
                    total += 1;

                    // replay fidelity
                    let mut sk = env.clone(); sk.set_cultivation_time(10);
                    let mut errs = 0;
                    for a in plan.iter() {
                        if sk.step(a.clone()).is_err() { errs += 1; }
                        sk.finish_cultivating(None, None);
                    }
                    if errs > 0 || !sk.done() { diverged += 1; }

                    // re-solve intermediate states
                    for frac in [0.2f32, 0.4, 0.6, 0.8] {
                        let prefix = ((plan.len() as f32) * frac) as usize;
                        let (mut s_k, _) = g.make_reverse_start(&env, &plan, prefix);
                        if let Err(e) = Solver::new().solve(&mut s_k, true) {
                            stuck += 1;
                            if !first_stuck_reported {
                                first_stuck_reported = true;
                                eprintln!("FIRST STUCK: seed={seed} no={no} nb={nb} plan_len={} prefix={prefix} obj_remaining={} err={e:?}",
                                    plan.len(), s_k.num_objectives());
                            }
                        }
                    }
                }
            }
        }
        eprintln!("SWEEP: {total} qualifying envs | replay diverged in {diverged} | re-solve STUCK events {stuck}");
    }

    /// Reproduce the REAL [stuck] source: the HER relabel-solve on a state the
    /// AGENT reaches (random, non-heuristic play) from a MID-SOLUTION S_k
    /// (partial-circuit layout) vs from a clean S0. If S_k stuck-rate >> S0
    /// stuck-rate, the HER path on mid-solution layouts is the culprit.
    /// RESULT (2026-07-06): S_k(mid-solution) 24/87 stuck vs S0(clean) 0/7 —
    /// confirms the HER relabel-solve on S_k's mid-transport layout is the
    /// no_ready_pp source (fixed by grading reverse partials without re-solving).
    /// #[ignore]d (slow, ~32s). cargo test ... diagnose_her_stuck -- --ignored --nocapture
    #[test]
    #[ignore]
    fn diagnose_her_stuck_midstate_vs_s0() {
        let g = default_gatherer("/dev/null");
        let (mut sk_stuck, mut s0_stuck, mut sk_n, mut s0_n) = (0, 0, 0, 0);

        // Random (non-heuristic) play from `start`, then HER-relabel + solve.
        // Returns Some(is_stuck) when HER produced a goal, None if zero-achieved.
        let try_her = |start: &TilersEnvInner, seed: u64| -> Option<bool> {
            let mut fin = start.clone();
            fin.set_cultivation_time(10);
            let mut rng = StdRng::seed_from_u64(seed);
            for _ in 0..40 {
                if fin.done() { break; }
                let va = fin.valid_actions();
                if va.is_empty() { break; }
                let a = va[rng.random_range(0..va.len())];
                let _ = fin.step(a);
                fin.finish_cultivating(None, None);
            }
            match start.achieved_goal_env(&fin) {
                Ok(mut her_env) => Some(Solver::new().solve(&mut her_env, true).is_err()),
                Err(_) => None,
            }
        };

        for seed in 0u64..30 {
            for &no in &[3usize, 4, 6] {
                let mut env = TilersEnvInner::new(10, 10, 12);
                env.set_seed(Some(seed));
                env.random_start(no, false);
                env.set_cultivation_time(10);
                let (_d, plan) = g.heuristic_typed_plan(&env);
                if plan.len() < 50 { continue; }

                let (s_k, _) = g.make_reverse_start(&env, &plan, plan.len() / 2);
                let pseed = seed * 7 + 1;
                if let Some(st) = try_her(&s_k, pseed) { sk_n += 1; if st { sk_stuck += 1;
                    if sk_stuck <= 3 { eprintln!("S_k HER STUCK: seed={seed} no={no}"); } } }
                if let Some(st) = try_her(&env, pseed) { s0_n += 1; if st { s0_stuck += 1; } }
            }
        }
        eprintln!("HER-solve STUCK rate — S_k(mid-solution): {sk_stuck}/{sk_n} | S0(clean): {s0_stuck}/{s0_n}");
    }

    // ─── gather integration ──────────────────────────────────────────────────

    #[test]
    fn test_gather_creates_jsonl_with_expected_keys() {
        use std::io::BufRead;
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out.jsonl").to_str().unwrap().to_string();

        let g = Gatherer::new(
            1, 5, 2, 1.0, out.clone(), 0.0, 0.0, 2, 0,
            None, Some(1.0), None,
            None,                            // max_action_multiplier (default 1.2)
            Some(2.0), Some(0), None, Some(0.0),  // resignation disabled in this test
            None,                            // resignation_log_dir
            None,                            // floor_keep_fraction
            None,                            // her_reward_margin
        );

        let mut env = TilersEnvInner::new(3, 3, 1);
        env.set_seed(Some(99));
        env.random_start(1, false);

        let client = TrivialTilersIpcClient {};
        let mut rng = StdRng::seed_from_u64(0);
        g.gather(&env, &client, 1.4, &mut rng);

        let file = std::fs::File::open(&out).expect("output file not created");
        let lines: Vec<String> = std::io::BufReader::new(file)
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .collect();

        assert!(!lines.is_empty(), "no data lines written");
        for line in &lines {
            let v: serde_json::Value = serde_json::from_str(line).expect("invalid JSON");
            for key in &["height", "width", "num_ancillas", "board",
                         "valid_actions", "edge_visits", "reward"] {
                assert!(v.get(key).is_some(), "missing key '{key}' in: {line}");
            }
        }
    }

    // ─── gold banking (Phase 2) ──────────────────────────────────────────────

    #[test]
    fn test_gold_banking_tags_records_and_does_not_panic() {
        use std::io::BufRead;
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out.jsonl").to_str().unwrap().to_string();
        let gold_dir = tmp.path().join("gold");
        let gold_dir_str = gold_dir.to_str().unwrap().to_string();

        let mut g = Gatherer::new(
            1, 5, 2, 1.0, out.clone(), 0.0, 0.0, 2, 0,
            None, Some(1.0), None,
            None,                            // max_action_multiplier (default 1.2)
            Some(2.0), Some(0), None, Some(0.0),  // resignation disabled in this test
            None,                            // resignation_log_dir
            None,                            // floor_keep_fraction
            None,                            // her_reward_margin
        );
        // gold_min_len = 1 → any win with score>0 qualifies for banking.
        g.set_gold_banking(Some(gold_dir_str.clone()), 1, 0.0);

        let mut env = TilersEnvInner::new(3, 3, 1);
        env.set_seed(Some(99));
        env.random_start(1, false);

        let client = TrivialTilersIpcClient {};
        let mut rng = StdRng::seed_from_u64(0);
        // Must not panic regardless of whether the trivial client happens to win.
        g.gather(&env, &client, 1.4, &mut rng);

        // If any gold file/records were written, every record MUST carry gold:true
        // and the normal training keys. (Trivial client may not win → no gold; the
        // assertion is only over records that were actually banked.)
        let gold_files: Vec<_> = std::fs::read_dir(&gold_dir)
            .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
            .unwrap_or_default();
        for path in &gold_files {
            let file = std::fs::File::open(path).unwrap();
            for line in std::io::BufReader::new(file).lines().filter_map(|l| l.ok()) {
                if line.trim().is_empty() { continue; }
                let v: serde_json::Value = serde_json::from_str(&line).expect("invalid gold JSON");
                assert_eq!(v.get("gold").and_then(|x| x.as_bool()), Some(true),
                           "gold record missing gold:true → {line}");
                for key in &["height", "width", "num_ancillas", "board",
                             "valid_actions", "edge_visits", "reward"] {
                    assert!(v.get(key).is_some(), "gold record missing key '{key}' in: {line}");
                }
            }
        }
    }

    // ─── Q-filtered BC demos (Phase 4) ────────────────────────────────────────

    #[test]
    fn test_demo_episode_records_shape_and_onehot() {
        use std::io::BufRead;
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out.jsonl").to_str().unwrap().to_string();

        let g = default_gatherer(&out);

        let mut env = TilersEnvInner::new(4, 4, 2);
        env.set_seed(Some(11));
        env.random_start(2, false);

        // Demos are failure-triggered in gather(); test the write path directly
        // so the assertions don't depend on whether the trivial client fails.
        let mut rng = StdRng::seed_from_u64(0);
        let (score, _ref, done) = g.write_demo_episode(&env, &mut rng);
        // Demo path returns the sentinel tuple (0.0, 0.0, false).
        assert_eq!(score, 0.0);
        assert!(!done);

        let file = std::fs::File::open(&out).expect("demo output file not created");
        let lines: Vec<String> = std::io::BufReader::new(file)
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert!(!lines.is_empty(), "no demo records written");

        for line in &lines {
            let v: serde_json::Value = serde_json::from_str(line).expect("invalid demo JSON");
            for key in &["height", "width", "num_ancillas", "board",
                         "valid_actions", "edge_visits", "reward"] {
                assert!(v.get(key).is_some(), "demo record missing key '{key}' in: {line}");
            }
            // Demo-specific invariants.
            assert_eq!(v.get("reward").and_then(|x| x.as_f64()), Some(0.0), "demo reward must be 0.0");
            assert_eq!(v.get("reward_kind").and_then(|x| x.as_str()), Some("demo"));
            assert_eq!(v.get("is_demo").and_then(|x| x.as_bool()), Some(true));
            // edge_visits is a ONE-HOT: exactly one key at weight 1.0, and that
            // key must be one of the record's valid_actions (so the legal mask
            // won't zero it out in the trainer).
            let ev = v.get("edge_visits").and_then(|x| x.as_object()).expect("edge_visits object");
            assert_eq!(ev.len(), 1, "demo edge_visits must be one-hot");
            let (k, val) = ev.iter().next().unwrap();
            assert_eq!(val.as_f64(), Some(1.0), "demo one-hot weight must be 1.0");
            let onehot_id: u64 = k.parse().expect("edge_visits key is an integer action id");
            let valid: Vec<u64> = v.get("valid_actions").and_then(|x| x.as_array()).unwrap()
                .iter().map(|x| x.as_u64().unwrap()).collect();
            assert!(valid.contains(&onehot_id), "demo one-hot action not in valid_actions");
        }
    }
}