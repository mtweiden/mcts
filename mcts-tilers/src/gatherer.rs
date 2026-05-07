use std::collections::HashMap;
use std::io::Write;

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
use tilers::objective::Objective;
use tilers::qubit::Qubit;
use tilers::solver::Solver;
use tilers::enums::{Direction, QubitId};

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
    output_path: String,
    noise_strength: f64,
    /// Mixing weight for Dirichlet noise injected into the MCTS root prior on
    /// full-search turns. 0.0 disables root noise.
    /// Reference: [Wu 2020, §2].
    dirichlet_epsilon: f32,
    num_objective_layers: usize,
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
    /// `resign_consecutive_moves` consecutive moves AND the agent's
    /// current depth already exceeds the solver's reference depth. Both
    /// conditions are required so we only resign when the agent is
    /// confidently losing AND has already overshot — this avoids early
    /// resignation on positions that look bad but are still recoverable.
    /// Set `resign_consecutive_moves = 0` to disable resignation
    /// entirely; values of `resign_value_threshold` above 1.0 also have
    /// that effect (Q is bounded above by 1).
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
    /// If set, winning trajectories are written here as pretraining data.
    trajectory_dir: Option<String>,
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
        num_objective_layers: usize,
        gather_id: usize,
        trajectory_dir: Option<String>,
        reward_saturation_temperature: Option<f32>,
        max_actions: Option<usize>,
        resign_value_threshold: Option<f32>,
        resign_consecutive_moves: Option<usize>,
        no_resign_rate: Option<f32>,
    ) -> Self {
        Self {
            batch_size,
            mcts_steps,
            fast_steps,
            p_full_search,
            max_actions,
            output_path,
            noise_strength,
            dirichlet_epsilon,
            num_objective_layers,
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
            trajectory_dir,
        }
    }

    /// Solve the environment using a heuristic solver and return the depth of the solution.
    pub fn solve_with_heuristic(&self, env: &Environment) -> (f32, Vec<Action>) {
        let mut solved_env = env.clone();
        let solver = Solver::new();
        let actions = solver.solve(&mut solved_env, true)
            .unwrap()
            .into_iter()
            .map(|a| a as Action)
            .collect();
        let depth = solved_env.depth(true, true);
        (depth, actions)
    }

    /// Directly sampling from Dirichlet distribution requires num_actions to be known at
    /// compile time, so we sample using Gamma distributions instead.
    fn _dirichlet_noise(&self, num_actions: usize, rng: &mut impl Rng) -> Vec<f64> {
        let alpha = 10f64 / (num_actions as f64);  // Rule of thumb for Dirichlet noise
        let alphas = vec![alpha.min(0.5); num_actions];
        let mut xs: Vec<f64> = alphas
            .iter()
            .map(|&a| {
                let gamma = Gamma::new(a, 1.0).unwrap();
                gamma.sample(rng)
            })
            .collect();
        let sum_xs: f64 = xs.iter().sum();
        for x in xs.iter_mut() {
            *x /= sum_xs;
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
        let valid_actions = env.valid_actions();
        let num_actions = valid_actions.len();
        if num_actions == 0 {
            panic!("No valid actions available");
        }

        // High temperature early (exploration), low temperature later (exploitation)
        let temperature = 0.1 + 0.9 * (-0.5 * step as f64).exp();
        let noise_strength = self.noise_strength * (-0.5 * step as f64).exp();

        let probs = self._action_probabilities(
            &valid_actions
                .iter()
                .map(|&a| *node.edge_visits.get(&(a as Action)).unwrap_or(&0))
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
        valid_actions[dist.sample(rng)] as Action
    }

    fn serialize_placement(placement: &[Qubit]) -> Value {
        Value::Array(
            placement
                .iter()
                .map(|q| {
                    json!([q.id.as_i32(), q.orientation as u8])
                })
                .collect(),
        )
    }

    fn serialize_objectives(objectives: &[Vec<Objective>]) -> Value {
        Value::Array(
            objectives
                .iter()
                .map(|layer| {
                    Value::Array(
                        layer
                            .iter()
                            .map(|o| {
                                if o.opcode.is_single_qubit() {
                                    json!([o.opcode as u8, o.arg_0.as_i32()])
                                } else {
                                    json!([o.opcode as u8, o.arg_0.as_i32(), o.arg_1.as_i32()])
                                }
                            })
                            .collect::<Vec<Value>>(),
                    )
                })
                .collect::<Vec<Value>>(),
        )
    }

    pub fn gather(
        &self,
        env: &Environment,
        client: &dyn InferenceClient<TilersEnv>,
        c_puct: f32,
        rng: &mut impl Rng,
    ) -> (f32, f32, bool) {
        let mut mcts: MCTS<TilersEnv> = MCTS::new(self.batch_size);

        let mut game = env.clone();
        game.drop_objectives_beyond_nth_layer(self.num_objective_layers - 1);
        game.set_cultivation_time(10);
        let (reference_depth, reference_actions) = self.solve_with_heuristic(&game);

        let h = env.height;
        let w = env.width;
        let nb = env.num_ancillas();
        let no = env.num_objectives();
        println!(
            "[Gatherer {}] Starting Env(h={}, w={}, nb={}, no={})", self.gather_id, h, w, nb, no,
        );

        // Training data for full-search turns only (written to output_path).
        let mut temp_data: Vec<(
            (Vec<Qubit>, Vec<Vec<Objective>>),
            Vec<Action>,
            HashMap<Action, usize>,
            HashMap<QubitId, Direction>,
        )> = Vec::new();

        // Full trajectory data for every step (written to trajectory_dir on a win).
        let mut all_steps_data: Vec<(
            (Vec<Qubit>, Vec<Vec<Objective>>),
            Vec<Action>,        // valid actions
            Action,             // action taken
            HashMap<QubitId, Direction>,
            usize,              // height
            usize,              // width
        )> = Vec::new();

        let mut tilers_env = TilersEnv::new(game.clone(), self.num_objective_layers);

        let max_actions = if let Some(max) = self.max_actions {
            max
        } else {
            (reference_actions.len() as f32 * 1.2) as usize
        };

        // Resignation bookkeeping. resign_allowed is decided once per
        // episode so the no-resign sanity sample is a uniform 1 - rate
        // fraction. Episodes that fall into the sanity sample play out
        // to natural termination; the rest may resign once the
        // (low-Q-streak ∧ over-solver-depth) condition is met.
        let resignation_enabled =
            self.resign_consecutive_moves > 0 && self.resign_value_threshold < 1.0;
        let resign_allowed =
            resignation_enabled && rng.random::<f32>() >= self.no_resign_rate;
        let mut low_value_streak: usize = 0;

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
                    .map(|(&a, &n)| (a as Action, n as f32))
                    .collect();
                mcts.perturb_root_prior(&noise_map, self.dirichlet_epsilon);
            }

            let root = mcts.run(&tilers_env, client, steps, c_puct, &terminal_evaluator, is_full_search);

            // Resignation check (after the search, before recording or
            // stepping). root.value is the visit-weighted Q estimate at
            // the current state — the agent's best estimate of "how is
            // this position going". We require Q ≤ threshold AND that
            // we've already overshot the solver's depth, so the agent
            // is both confident it's losing and has exhausted its
            // budget. Both conditions reset the streak when violated;
            // resignation only fires after the streak hits the
            // configured length.
            if resignation_enabled {
                let q = root.value;
                let current_depth = tilers_env.inner.depth(true, true) as f32;
                let over_solver = current_depth > reference_depth;
                if q <= self.resign_value_threshold && over_solver {
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
                    break;
                }
            }

            // Snapshot state before stepping (shared between temp_data and all_steps_data).
            let placement = tilers_env.inner.get_placement();
            let objectives = tilers_env.inner.get_objectives(self.num_objective_layers);
            let last_dirs = tilers_env.inner.last_dirs.clone();
            let valid_actions: Vec<Action> = tilers_env
                .inner
                .valid_actions()
                .iter()
                .map(|&a| a as Action)
                .collect();
            let step_height = tilers_env.inner.height;
            let step_width = tilers_env.inner.width;

            // Only record training data for full searches.
            if is_full_search {
                let n_total: usize = root.edge_visits.values().sum();
                let edge_visits: HashMap<Action, usize> = mcts
                    .policy_target(c_puct)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(a, p)| (a, (p * n_total as f32).round() as usize))
                    .collect();
                temp_data.push((
                    (placement.clone(), objectives.clone()),
                    valid_actions.clone(),
                    edge_visits,
                    last_dirs.clone(),
                ));
            }

            let action = self.select_action(&root, &tilers_env.inner, step, rng);

            // Record every step for trajectory saving.
            all_steps_data.push((
                (placement, objectives),
                valid_actions,
                action,
                last_dirs,
                step_height,
                step_width,
            ));

            let _ = tilers_env.inner.step(action as usize);
            tilers_env.inner.finish_cultivating(None, None);

            if tilers_env.inner.done() {
                break;
            }

            mcts.advance_root(action);
        }

        tilers_env.inner.set_cultivation_time(10);
        let solution_depth = tilers_env.inner.depth(true, true);

        let score: f32 = if tilers_env.inner.done() && reference_depth > solution_depth {
            1.0
        } else if tilers_env.inner.done() && (reference_depth - solution_depth).abs() < 1e-3 {
            0.0
        } else {
            -1.0
        };

        // Write the full trajectory to trajectory_dir if we found a win.
        if score > 0.0 {
            if let Some(dir) = &self.trajectory_dir {
                std::fs::create_dir_all(dir).expect("Failed to create trajectory directory");

                let filename = format!("{}/traj_{}.json", dir, self.gather_id);

                let mut traj_file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&filename)
                    .expect("Unable to open trajectory file");

                let depth = all_steps_data.len();
                let gamma = 0.80f32;

                for (step, ((placement, objectives), valid_actions, action, last_dirs, s_height, s_width)) in all_steps_data.iter().enumerate() {
                    let num_ancillas = placement.iter().filter(|q| q.id.as_i32() < 0).count();
                    let placement_json = Self::serialize_placement(placement);
                    let objectives_json = Self::serialize_objectives(objectives);
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

                    let last_dirs_json = Value::Array(
                        last_dirs
                            .iter()
                            .map(|(qid, dir)| json!([qid.as_i32(), dir.to_string()]))
                            .collect(),
                    );

                    let record = json!({
                        "height": s_height,
                        "width": s_width,
                        "num_ancillas": num_ancillas,
                        "placement": placement_json,
                        "objectives": objectives_json,
                        "valid_actions": valid_actions_json,
                        "edge_visits": Value::Object(visits_map),
                        "reward": value,
                        "last_dirs": last_dirs_json,
                    });

                    writeln!(traj_file, "{}", record).expect("Failed to write trajectory record");
                }

                traj_file.flush().expect("Failed to flush trajectory file");
            }
        }

        // Write full-search data to output_path as normal.
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)
            .expect("Unable to open output file");

        for ((p, o), va, ev, last_dirs) in temp_data {
            let num_ancillas = p.iter().filter(|q| q.id.as_i32() < 0).count();
            let placement_json = Self::serialize_placement(&p);
            let objectives_json = Self::serialize_objectives(&o);
            let valid_actions_json = Value::Array(va.into_iter().map(Value::from).collect());
            let mut visits_map = Map::with_capacity(ev.len());
            for (action, count) in ev {
                visits_map.insert(action.to_string(), Value::from(count));
            }
            let visits_json = Value::Object(visits_map);

            let last_dirs_json = Value::Array(
                last_dirs
                    .iter()
                    .map(|(qid, dir)| json!([qid.as_i32(), dir.to_string()]))
                    .collect(),
            );

            let record = json!({
                "height": tilers_env.inner.height,
                "width": tilers_env.inner.width,
                "num_ancillas": num_ancillas,
                "placement": placement_json,
                "objectives": objectives_json,
                "valid_actions": valid_actions_json,
                "edge_visits": visits_json,
                "reward": score,
                "last_dirs": last_dirs_json,
            });

            writeln!(file, "{}", record).expect("Failed to write record");
        }
        file.flush().expect("Failed to flush file");

        (solution_depth as f32, reference_depth as f32, tilers_env.inner.done())
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
) -> PyResult<Option<(f32, f32, bool)>> {
    let num_slots = 2048;
    let num_objective_layers = DEFAULT_LOOKAHEAD;

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
    let gatherer = Gatherer::new(
        8,
        mcts_steps,
        fast_steps,
        p_full_search,
        output_path,
        0.20,
        dirichlet_epsilon,
        num_objective_layers,
        worker_id as usize,
        trajectory_dir,
        Some(reward_saturation_temperature),
        None,
        Some(resign_value_threshold),
        Some(resign_consecutive_moves),
        Some(no_resign_rate),
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
    env.random_objectives(no, false);

    if env.valid_actions().contains(&0) {
        let mut tmp_env = env.clone();
        let _ = tmp_env.step(0);
        if tmp_env.done() {
            return Ok(None);
        }
    }

    env.shuffle(num_shuffles);

    let solver = Solver::new();
    let mut temp_env = env.clone();
    if let Ok(solution) = solver.solve(&mut temp_env, false) {
        if solution.len() > max_generated_depth {
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
            2,                    // num_objective_layers
            0,                    // gather_id
            None,                 // trajectory_dir
            Some(1.0),            // reward_saturation_temperature
            None,                 // max_actions
            Some(2.0),            // resign_value_threshold (>1.0 disables)
            Some(0),              // resign_consecutive_moves (0 disables)
            Some(0.0),            // no_resign_rate
        )
    }

    // ─── Gatherer::new ──────────────────────────────────────────────────────

    #[test]
    fn test_new_sets_explicit_reward_saturation_temperature() {
        let g = Gatherer::new(
            8, 10, 2, 0.5, "/dev/null".into(), 0.1, 0.25, 2, 1,
            None, Some(0.3), Some(100), None, None, None,
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
            None, None, None, None, None, None,
        );
        assert!((g.reward_saturation_temperature - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_new_defaults_resignation_params() {
        let g = Gatherer::new(
            8, 10, 2, 0.5, "/dev/null".into(), 0.0, 0.0, 2, 0,
            None, None, None, None, None, None,
        );
        assert!((g.resign_value_threshold - (-0.9)).abs() < 1e-6);
        assert_eq!(g.resign_consecutive_moves, 5);
        assert!((g.no_resign_rate - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_new_sets_explicit_resignation_params() {
        let g = Gatherer::new(
            8, 10, 2, 0.5, "/dev/null".into(), 0.0, 0.0, 2, 0,
            None, None, None, Some(-0.5), Some(8), Some(0.2),
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
        let valid = env.valid_actions();
        assert!(!valid.is_empty());

        let priors: HashMap<Action, f32> = valid.iter()
            .map(|&a| (a as Action, 1.0 / valid.len() as f32))
            .collect();
        let node = Node::new(priors, 0.0, 0, None);

        let mut rng = StdRng::seed_from_u64(42);
        let action = g.select_action(&node, &env, 50, &mut rng);
        assert!(valid.contains(&(action as usize)), "action {action} not in valid_actions");
    }

    #[test]
    fn test_select_action_picks_dominant_at_late_step() {
        let g = default_gatherer("/dev/null");
        let env = TilersEnvInner::new(3, 3, 1);
        let valid = env.valid_actions();
        assert!(valid.len() >= 2, "need ≥2 valid actions");

        let priors: HashMap<Action, f32> = valid.iter()
            .map(|&a| (a as Action, 1.0 / valid.len() as f32))
            .collect();
        let mut node = Node::new(priors, 0.0, 0, None);
        let dominant = valid[0] as Action;
        node.edge_visits.insert(dominant, 100_000);

        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..20 {
            assert_eq!(g.select_action(&node, &env, 1_000, &mut rng), dominant);
        }
    }

    // ─── serialize_placement ────────────────────────────────────────────────

    #[test]
    fn test_serialize_placement_empty() {
        let v = Gatherer::serialize_placement(&[]);
        assert!(matches!(v, serde_json::Value::Array(ref a) if a.is_empty()));
    }

    #[test]
    fn test_serialize_placement_single_qubit() {
        use tilers::qubit::Qubit;
        use tilers::enums::{Orientation, QubitId};
        let q = Qubit::new(QubitId(0), Orientation::Vertical);
        let v = Gatherer::serialize_placement(&[q]);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let pair = arr[0].as_array().unwrap();
        assert_eq!(pair.len(), 2);
        assert_eq!(pair[0].as_i64().unwrap(), 0);  // id = 0
        assert_eq!(pair[1].as_u64().unwrap(), 0);  // Vertical = 0
    }

    // ─── serialize_objectives ────────────────────────────────────────────────

    #[test]
    fn test_serialize_objectives_empty() {
        let v = Gatherer::serialize_objectives(&[]);
        assert!(matches!(v, serde_json::Value::Array(ref a) if a.is_empty()));
    }

    #[test]
    fn test_serialize_objectives_single_qubit_op_has_two_elements() {
        use tilers::objective::Objective;
        use tilers::enums::{Operation, QubitId};
        let obj = Objective::new(Operation::H, QubitId(0), vec![], QubitId(-1));
        let v = Gatherer::serialize_objectives(&[vec![obj]]);
        let layers = v.as_array().unwrap();
        let entry = layers[0].as_array().unwrap()[0].as_array().unwrap();
        assert_eq!(entry.len(), 2, "single-qubit op should produce [opcode, arg0]");
    }

    #[test]
    fn test_serialize_objectives_two_qubit_op_has_three_elements() {
        use tilers::objective::Objective;
        use tilers::enums::{Operation, QubitId};
        let obj = Objective::new(Operation::CX, QubitId(0), vec![], QubitId(1));
        let v = Gatherer::serialize_objectives(&[vec![obj]]);
        let layers = v.as_array().unwrap();
        let entry = layers[0].as_array().unwrap()[0].as_array().unwrap();
        assert_eq!(entry.len(), 3, "two-qubit op should produce [opcode, arg0, arg1]");
    }

    // ─── solve_with_heuristic ────────────────────────────────────────────────

    #[test]
    fn test_solve_with_heuristic_nonnegative_depth() {
        let g = default_gatherer("/dev/null");
        let mut env = TilersEnvInner::new(3, 3, 1);
        env.set_seed(Some(1));
        env.random_objectives(2, false);
        let (depth, actions) = g.solve_with_heuristic(&env);
        assert!(depth >= 0.0, "depth={depth}");
        assert!(!actions.is_empty(), "heuristic solution should be non-empty");
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
            Some(2.0), Some(0), Some(0.0),  // resignation disabled in this test
        );

        let mut env = TilersEnvInner::new(3, 3, 1);
        env.set_seed(Some(99));
        env.random_objectives(1, false);

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
            for key in &["height", "width", "placement", "objectives",
                         "valid_actions", "edge_visits", "reward", "last_dirs"] {
                assert!(v.get(key).is_some(), "missing key '{key}' in: {line}");
            }
        }
    }
}