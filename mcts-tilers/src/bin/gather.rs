use std::env;
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
use mcts_core::mcts::MCTS;
use mcts_core::node::Node;

use mcts_tilers::constants::*;
use mcts_tilers::environment::TilersEnv;
use mcts_tilers::slot::TilersSlot;
use mcts_tilers::client::TilersIpcClient;

use tilers::env::Environment;
use tilers::objective::Objective;
use tilers::qubit::Qubit;
use tilers::solver::Solver;
use tilers::enums::{Direction, QubitId};

/// ----------------------------------------------------------------------------
/// Gatherer
/// ----------------------------------------------------------------------------
/// A struct to gather data from MCTS simulations.
/// ----------------------------------------------------------------------------
struct Gatherer {
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
        max_actions: Option<usize>,
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
        client: &TilersIpcClient,
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

        let terminal_evaluator = |e: &TilersEnv| -> f32 {
            if !e.inner.done() { -1.0 } else {
                let d = e.inner.depth(true, true) as f32;
                if d < reference_depth {
                    1.0
                } else if (d - reference_depth).abs() < 1e-3 {
                    0.0
                } else {
                    -1.0
                }
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


fn main() {
    // Environment parameters
    let mut height = 4;
    let mut width = 4;
    let mut num_objectives = 4;
    let mut min_num_objectives = 2;
    let mut num_blanks = 2;
    let mut seed: Option<i32> = None;
    // IPC parameters
    let mut worker_id: u32 = 0;
    let mut num_handlers = 1;
    let mut num_shuffles = 0;
    let num_slots = 2048;
    let num_objective_layers = DEFAULT_LOOKAHEAD;
    let mut max_generated_depth = 10_000;
    let mut c_puct = 1.4;
    let mut mcts_steps: usize = 10_000;
    let mut fast_steps: usize = 1_600;
    let mut p_full_search: f32 = 0.25;
    let mut dirichlet_epsilon: f32 = 0.25;
    let mut trajectory_dir: Option<String> = None;

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--height" && i + 1 < args.len() {
            height = args[i + 1].parse().unwrap_or(4);
        }
        if args[i] == "--width" && i + 1 < args.len() {
            width = args[i + 1].parse().unwrap_or(4);
        }
        if args[i] == "--num_objectives" && i + 1 < args.len() {
            num_objectives = args[i + 1].parse().unwrap_or(4);
        }
        if args[i] == "--min_num_objectives" && i + 1 < args.len() {
            min_num_objectives = args[i + 1].parse().unwrap_or(2);
        }
        if args[i] == "--num_blanks" && i + 1 < args.len() {
            num_blanks = args[i + 1].parse().unwrap_or(2);
        }
        if args[i] == "--worker_id" && i + 1 < args.len() {
            worker_id = args[i + 1].parse().unwrap_or(0);
        }
        if args[i] == "--num_handlers" && i + 1 < args.len() {
            num_handlers = args[i + 1].parse().unwrap_or(1);
        }
        if args[i] == "--num_shuffles" && i + 1 < args.len() {
            num_shuffles = args[i + 1].parse().unwrap_or(0);
        }
        if args[i] == "--seed" && i + 1 < args.len() {
            seed = args[i + 1].parse().ok();
        }
        if args[i] == "--max_generated_depth" && i + 1 < args.len() {
            max_generated_depth = args[i + 1].parse().unwrap_or(10_000);
        }
        if args[i] == "--c_puct" && i + 1 < args.len() {
            c_puct = args[i + 1].parse().unwrap_or(1.4);
        }
        if args[i] == "--mcts_steps" && i + 1 < args.len() {
            mcts_steps = args[i + 1].parse().unwrap_or(10_000);
        }
        if args[i] == "--fast_steps" && i + 1 < args.len() {
            fast_steps = args[i + 1].parse().unwrap_or(1_600);
        }
        if args[i] == "--p_full_search" && i + 1 < args.len() {
            p_full_search = args[i + 1].parse().unwrap_or(0.25);
        }
        if args[i] == "--dirichlet_epsilon" && i + 1 < args.len() {
            dirichlet_epsilon = args[i + 1].parse().unwrap_or(0.25);
        }
        if args[i] == "--trajectory_dir" && i + 1 < args.len() {
            trajectory_dir = Some(args[i + 1].clone());
        }
    }

    let arena_name = format!("mcts_{}_{}", num_slots, num_handlers);
    let arena: Arena<TilersSlot> =
        Arena::create_or_open(&arena_name, num_slots, num_handlers).unwrap();
    let client = TilersIpcClient::new(arena, worker_id);

    let output_path = format!("/home/mtweiden/data/output-{}.json", worker_id);
    let gatherer = Gatherer::new(
        8,                  // inference batch size
        mcts_steps,         // full-search MCTS steps
        fast_steps,         // fast-search MCTS steps
        p_full_search,      // fraction of turns that are full searches
        output_path,
        0.20,               // action-selection noise strength
        dirichlet_epsilon,  // Dirichlet epsilon for MCTS root noise
        num_objective_layers,
        worker_id as usize,
        trajectory_dir,
        None,                 // max actions
    );

    loop {
        let mut rng = if let Some(s) = seed {
            let rng = StdRng::seed_from_u64(s as u64);
            seed = Some(s + 1);
            rng
        } else {
            StdRng::from_rng(&mut rand::rng())
        };

        let h = height;
        let w = width;
        let nb = num_blanks;
        let no = rng.random_range(min_num_objectives..=num_objectives);
        if nb >= (h * w) - 1 || (h <= 2 && w <= 2) {
            continue;
        }
        let mut env = Environment::new(h, w, nb);

        if seed.is_some() {
            env.set_seed(Some(seed.unwrap() as u64));
        }

        env.random_objectives(no, false);
        if env.valid_actions().contains(&0) {
            let mut tmp_env = env.clone();
            let _ = tmp_env.step(0);
            if tmp_env.done() {
                continue;
            }
        }
        env.shuffle(num_shuffles);

        let solver = Solver::new();
        let mut temp_env = env.clone();
        if let Ok(solution) = solver.solve(&mut temp_env, false) {
            if solution.len() > max_generated_depth { continue; }
        }

        let mut game_rng = StdRng::from_rng(&mut rand::rng());
        let (sol_depth, ref_depth, done) = gatherer.gather(&env, &client, c_puct, &mut game_rng);
        if done {
            println!(
                "[Gatherer {}] Env(h={}, w={}, nb={}, no={}): solution depth = {}, reference depth = {}",
                worker_id, h, w, nb, no, sol_depth, ref_depth
            );
        } else {
            println!(
                "[Gatherer {}] Env(h={}, w={}, nb={}, no={}): (unfinished) depth = {}, reference depth = {}",
                worker_id, h, w, nb, no, sol_depth, ref_depth
            );
        }
    }
}