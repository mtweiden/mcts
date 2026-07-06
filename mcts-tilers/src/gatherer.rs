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
    reverse_curriculum_k_start_frac: f32,
}

/// One self-play episode's outcome, decoupled from record-writing so the
/// reverse-curriculum climb can run several probes and choose which to keep.
struct EpisodeOutcome {
    reference_depth: f32, solution_depth: f32, done: bool,
    score: f32, reward_kind: &'static str, achieved_objectives: usize,
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
            no_resign_rate: no_resign_rate.unwrap_or(0.1),
            // 1.0 = keep every floor episode (unchanged default).
            floor_keep_fraction: floor_keep_fraction.unwrap_or(1.0),
            // 0.0 = zero-reward at exactly the heuristic depth (unchanged default).
            her_reward_margin: her_reward_margin.unwrap_or(0.0),
            trajectory_dir,
            resignation_log_dir,
            reverse_curriculum: false,
            reverse_curriculum_min_len: usize::MAX,
            reverse_curriculum_max_probes: 3,
            reverse_curriculum_k_start_frac: 0.25,
        }
    }

    pub fn set_reverse_curriculum(&mut self, enabled: bool, min_len: usize,
                                  max_probes: usize, k_start_frac: f32) {
        self.reverse_curriculum = enabled;
        self.reverse_curriculum_min_len = min_len;
        self.reverse_curriculum_max_probes = max_probes.max(1);
        self.reverse_curriculum_k_start_frac = k_start_frac.clamp(0.05, 1.0);
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

        // High temperature early (exploration), low temperature later (exploitation)
        let temperature = 0.1 + 0.9 * (-0.5 * step as f64).exp();
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
    fn make_reverse_start(&self, game: &Environment, plan: &[PlanAction], prefix_len: usize) -> Environment {
        let mut sk = game.clone();
        sk.set_cultivation_time(10);
        for a in plan.iter().take(prefix_len) {
            let _ = sk.step(a.clone());
            sk.finish_cultivating(None, None);
        }
        sk
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
        if self.reverse_curriculum {
            let (d_full, plan) = self.heuristic_typed_plan(&game);
            if plan.len() >= self.reverse_curriculum_min_len {
                return self.gather_reverse_curriculum(&game, d_full, &plan, client, c_puct, rng);
            }
        }
        let o = self.run_episode_from(&game, client, c_puct, rng, None);
        self.write_episode(&o, false, rng)
    }

    /// Reverse-curriculum bounded climb: start near the heuristic terminal
    /// (small remaining suffix `k`) and, whenever the agent solves the reduced
    /// problem, back off toward the true start S0 by increasing `k`. Keeps the
    /// hardest solved probe (`best`) and the first failed probe (`failed`), and
    /// writes both. Bounded by `reverse_curriculum_max_probes`.
    fn gather_reverse_curriculum(
        &self,
        game: &Environment,
        d_full: f32,
        plan: &[PlanAction],
        client: &dyn InferenceClient<TilersEnv>,
        c_puct: f32,
        rng: &mut impl Rng,
    ) -> (f32, f32, bool) {
        let len = plan.len();
        let mut k = (((len as f32) * self.reverse_curriculum_k_start_frac).ceil() as usize).clamp(1, len);
        let (mut best, mut failed) = (None, None);
        let mut ret = (0.0, 0.0, false);
        for _ in 0..self.reverse_curriculum_max_probes {
            let sk = self.make_reverse_start(game, plan, len - k);
            // Reference for S_k = D_full (heuristic finishes S_k via its
            // remaining actions to the same terminal), with `k` remaining
            // heuristic actions for the action budget. No S_k re-solve → no
            // `[stuck] no_ready_pp` panics from the greedy solver on mid-states.
            let o = self.run_episode_from(&sk, client, c_puct, rng, Some((d_full, k)));
            ret = (o.solution_depth, o.reference_depth, o.done);
            if o.done { let full = k >= len; best = Some(o); if full { break; }
                        k = (k + ((len - k)/2).max(1)).min(len); }
            else { failed = Some(o); break; }
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
        reference_override: Option<(f32, usize)>,
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
            Some((d, n)) => (d, n),
            None => {
                let (d, acts) = self.solve_with_heuristic(&game);
                (d, acts.len())
            }
        };

        let h = game.height;
        let w = game.width;
        let nb = game.num_ancillas();
        let no = game.num_objectives();
        println!(
            "[Gatherer {}] Starting Env(h={}, w={}, nb={}, no={})", self.gather_id, h, w, nb, no,
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
            (reference_action_count as f32 * self.max_action_multiplier) as usize
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

        let temperature = self.reward_saturation_temperature;
        let terminal_evaluator = |e: &TilersEnv| -> f32 {
            if !e.inner.done() { -1.0 } else {
                let d = e.inner.depth(true, true) as f32;
                let ratio = (reference_depth - d) / (reference_depth + 1e-6);
                // tanh(ratio / temperature) replaces the previous
                // clip-then-normalize. Smooth gradient at all input scales —
                // a 50% improvement still contributes signal instead of being
                // flattened to +1 the same as a 30% improvement was under
                // the clip. Saturation behavior is governed entirely by the
                // temperature knob.
                (ratio / temperature).tanh()
            }
        };

        for step in 0..max_actions {
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

                if resign_allowed && low_value_streak >= self.resign_consecutive_moves {
                    println!(
                        "[Gatherer {}] Resigning at step {}: Q={:.3} \
                         depth={} ref_depth={} streak={}",
                        self.gather_id, step, q, current_depth as i32,
                        reference_depth as i32, low_value_streak,
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
                let d = solution_depth as f32;
                let ref_d = reference_depth as f32;
                let ratio = (ref_d - d) / (ref_d + 1e-6);
                (
                    (ratio / self.reward_saturation_temperature).tanh(),
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
                    let frac = achieved as f32 / start_objs as f32;
                    (frac - 1.0, "her", achieved)
                }
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
                "[Gatherer {}] FINISHED Env(h={}, w={}, nb={}, no={}) | depth={} vs ref={} | score={:.3} (beat_solver={})",
                self.gather_id,
                game.height,
                game.width,
                game.num_ancillas(),
                game.num_objectives(),
                solution_depth,
                reference_depth,
                score,
                solution_depth < reference_depth,
            );
        }

        EpisodeOutcome {
            reference_depth,
            solution_depth: solution_depth as f32,
            done: tilers_env.inner.done(),
            score,
            reward_kind,
            achieved_objectives,
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
        const HER_SATURATION_FLOOR: f32 = -0.999;
        if o.reward_kind == "her" && o.score <= HER_SATURATION_FLOOR {
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
            });

            writeln!(file, "{}", record).expect("Failed to write record");
        }
        file.flush().expect("Failed to flush file");

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
    no_resign_rate = 0.1,
    resignation_log_dir = None,
    max_action_multiplier = 1.2,
    max_pp_weight = None,
    floor_keep_fraction = 1.0,
    her_reward_margin = 0.0,
    reverse_curriculum = false,
    reverse_curriculum_min_len = 50,
    reverse_curriculum_max_probes = 3,
    reverse_curriculum_k_start_frac = 0.25,
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
    no_resign_rate: f32,
    resignation_log_dir: Option<String>,
    max_action_multiplier: f32,
    max_pp_weight: Option<usize>,
    floor_keep_fraction: f32,
    her_reward_margin: f32,
    reverse_curriculum: bool,
    reverse_curriculum_min_len: usize,
    reverse_curriculum_max_probes: usize,
    reverse_curriculum_k_start_frac: f32,
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
        Some(no_resign_rate),
        resignation_log_dir,
        Some(floor_keep_fraction),
        Some(her_reward_margin),
    );
    gatherer.set_reverse_curriculum(
        reverse_curriculum,
        reverse_curriculum_min_len,
        reverse_curriculum_max_probes,
        reverse_curriculum_k_start_frac,
    );

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
    // Cap PauliProduct weight (easy-env curriculum) BEFORE random_start so the
    // generated objectives respect it. None = unbounded (historical behavior).
    env.set_max_pp_weight(max_pp_weight);
    env.random_start(no, false);

    env.shuffle(num_shuffles);

    // Solve the POST-shuffle env (the one we actually gather on) once: it
    // gates both the max-depth cap and the trivial-env reject. Checking the
    // pre-shuffle env was a bug — shuffle advances the state, so a non-trivial
    // start can become auto-execute-only after shuffling and slip through.
    let solver = Solver::new();
    let mut temp_env = env.clone();
    if let Ok(solution) = solver.solve(&mut temp_env, false) {
        let n = solution.len();
        if n > max_generated_depth {
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
            None, Some(0.3), Some(100), None, None, None, None, None, None, None,
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
            None, None, None, None, None, None, None, None, None, None,
        );
        assert!((g.reward_saturation_temperature - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_new_defaults_resignation_params() {
        let g = Gatherer::new(
            8, 10, 2, 0.5, "/dev/null".into(), 0.0, 0.0, 2, 0,
            None, None, None, None, None, None, None, None, None, None,
        );
        assert!((g.resign_value_threshold - (-0.9)).abs() < 1e-6);
        assert_eq!(g.resign_consecutive_moves, 5);
        assert!((g.no_resign_rate - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_new_sets_explicit_resignation_params() {
        let g = Gatherer::new(
            8, 10, 2, 0.5, "/dev/null".into(), 0.0, 0.0, 2, 0,
            None, None, None, None, Some(-0.5), Some(8), Some(0.2), None, None, None,
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
        let s0 = g.make_reverse_start(&env, &plan, 0);
        assert_eq!(
            s0.num_objectives(),
            env.num_objectives(),
            "prefix_len=0 must equal the true start's objective count",
        );

        // num_objectives() is monotonically non-increasing as prefix_len grows:
        // each replayed heuristic action can only complete objectives, never add.
        let mut prev = s0.num_objectives();
        for prefix_len in 1..=plan.len() {
            let sk = g.make_reverse_start(&env, &plan, prefix_len);
            let cur = sk.num_objectives();
            assert!(
                cur <= prev,
                "num_objectives increased at prefix_len={prefix_len}: {cur} > {prev}",
            );
            prev = cur;
        }

        // prefix_len = plan.len() replays the entire heuristic solution ⇒
        // done / near-done: no more remaining objectives than the true start.
        let full = g.make_reverse_start(&env, &plan, plan.len());
        assert!(
            full.num_objectives() <= s0.num_objectives(),
            "full replay should not have more objectives than S0",
        );
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
            let mut s_k = g.make_reverse_start(&env, &plan, prefix);
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
                        let mut s_k = g.make_reverse_start(&env, &plan, prefix);
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

                let s_k = g.make_reverse_start(&env, &plan, plan.len() / 2);
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
            Some(2.0), Some(0), Some(0.0),  // resignation disabled in this test
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
}