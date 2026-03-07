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

/// ----------------------------------------------------------------------------
/// Gatherer
/// ----------------------------------------------------------------------------
/// A struct to gather data from MCTS simulations.
/// ----------------------------------------------------------------------------
struct Gatherer {
    batch_size: usize,
    mcts_steps: usize,
    max_actions: usize,
    output_path: String,
    terminal_value: f32,
    noise_strength: f64,
    num_objective_layers: usize,
}

impl Gatherer {
    pub fn new(
        batch_size: usize,
        mcts_steps: usize,
        max_actions: usize,
        output_path: String,
        noise_strength: f64,
        num_objective_layers: usize,
    ) -> Self {
        let terminal_value: f32 = 1.0;
        Self {
            batch_size,
            mcts_steps,
            max_actions,
            output_path,
            terminal_value,
            noise_strength,
            num_objective_layers,
        }
    }

    /// Solve the environment using a heuristic solver and return the depth of the solution.
    pub fn solve_with_heuristic(&self, env: &Environment) -> f32 {
        let mut solved_env = env.clone();
        let solver = Solver::new();
        let _ = solver.solve(&mut solved_env, true).unwrap();
        println!("[SOLVING ENVIRONMENT]");
        let depth = solved_env.depth(true, true);
        println!("[ENVIRONMENT SOLVED] depth = {}", depth);
        depth
    }

    /// Directly sampling from Dirichlet distribution requires num_actions to be known at
    /// compile time, so we sample using Gamma distributions instead.
    fn _dirichlet_noise(&self, num_actions: usize, rng: &mut impl Rng) -> Vec<f64> {
        let alpha = 10f64 / (num_actions as f64);  // Rule of thumb for Dirichlet noise
        let alphas = vec![alpha; num_actions];
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

    fn _action_probabilities(&self, visit_counts: &[usize]) -> Vec<f64> {
        let total_visits: usize = visit_counts.iter().sum();
        if total_visits == 0 {
            return vec![1.0 / (visit_counts.len() as f64); visit_counts.len()];
        }
        visit_counts
            .iter()
            .map(|&count| count as f64 / total_visits as f64)
            .collect()
    }

    pub fn select_action(
        &self,
        node: &Node<Action>,
        env: &Environment,
        noiseless: bool,
        rng: &mut impl Rng,
    ) -> Action {
        let valid_actions = env.valid_actions();
        let num_actions = valid_actions.len();
        if num_actions == 0 {
            panic!("No valid actions available");
        }
        let probs = self._action_probabilities(
            &valid_actions
                .iter()
                .map(|&a| *node.edge_visits.get(&(a as Action)).unwrap_or(&0))
                .collect::<Vec<_>>(),
        );
        let noise = if !noiseless {
            self._dirichlet_noise(num_actions, rng)
        } else {
            vec![0.0; num_actions]
        };
        let mixed_probs: Vec<f64> = probs
            .iter()
            .zip(noise.iter())
            .map(|(&p, &n)| (1.0 - self.noise_strength) * p + self.noise_strength * n)
            .map(|x| x.max(0.0)) // prevent tiny negatives
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
        rng: &mut impl Rng,
    ) -> (f32, f32, bool) {
        // Set up MCTS and Agent and copy the Environment
        let mut mcts: MCTS<TilersEnv> = MCTS::new(self.terminal_value, self.batch_size);

        // Only consider the first N layers of gates
        let mut game = env.clone();
        game.drop_objectives_beyond_nth_layer(self.num_objective_layers);
        game.set_cultivation_time(10);
        let reference_depth = self.solve_with_heuristic(&game);

        // Set up data storage
        // Format: ((placement, objectives), valid_actions, visit_counts)
        let mut temp_data: Vec<(
            (Vec<Qubit>, Vec<Vec<Objective>>),
            Vec<Action>,
            HashMap<Action, usize>,
        )> = Vec::new();

        let mut taken_actions = vec![];

        // Wrap in TilersEnv for MCTS
        let mut tilers_env = TilersEnv::new(game.clone(), self.num_objective_layers);

        for step in 0..self.max_actions {
            // Run MCTS
            let root = mcts.run(&tilers_env, client, self.mcts_steps);

            // Store the data
            let placement = tilers_env.inner.get_placement();
            let objectives = tilers_env.inner.get_objectives(self.num_objective_layers);
            let edge_visits = root.edge_visits.clone();
            let valid_actions: Vec<Action> = tilers_env
                .inner
                .valid_actions()
                .iter()
                .map(|&a| a as Action)
                .collect();
            temp_data.push(((placement, objectives), valid_actions, edge_visits));

            // Select action and step the environment
            // Add noise if we're very close to the root to encourage exploration
            let noiseless = step > 2;
            let action = self.select_action(&root, &tilers_env.inner, noiseless, rng);
            let _ = tilers_env.inner.step(action as usize);
            tilers_env.inner.finish_cultivating(None, None); // Cultivate resources in a single step
            taken_actions.push(action as usize);
            if tilers_env.inner.done() {
                break;
            }

            // Advance the root
            mcts.advance_root(action);
        }

        tilers_env.inner.set_cultivation_time(10);
        let solution_depth = tilers_env.inner.depth(true, true);

        // Save data to output_path in NDJSON format
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)
            .expect("Unable to open output file");

        let score = if tilers_env.inner.done() && reference_depth >= solution_depth {
            1.0
        } else {
            -1.0
        };
        for ((p, o), va, ev) in temp_data {
            // Convert to JSON-serializable format
            let placement_json = Self::serialize_placement(&p);
            let objectives_json = Self::serialize_objectives(&o);
            let valid_actions_json = Value::Array(va.into_iter().map(Value::from).collect());
            let mut visits_map = Map::with_capacity(ev.len());
            for (action, count) in ev {
                visits_map.insert(action.to_string(), Value::from(count));
            }
            let visits_json = Value::Object(visits_map);

            let record = json!({
                "height": tilers_env.inner.height,
                "width": tilers_env.inner.width,
                "num_ancillas": tilers_env.inner.num_ancillas(),
                "placement": placement_json,
                "objectives": objectives_json,
                "valid_actions": valid_actions_json,
                "edge_visits": visits_json,
                "reward": score
            });

            let line = record.to_string(); // compact JSON
            writeln!(file, "{}", line).expect("Failed to write record");
        }
        writeln!(file, "").expect("Failed to write newline");
        file.flush().expect("Failed to flush file");
        (solution_depth as f32, reference_depth as f32, tilers_env.inner.done())
    }
}


fn main() {
    // Environment parameters
    let mut height = 4;
    let mut width = 4;
    let mut num_objectives = 2;
    let mut num_blanks = 2;
    let mut seed: Option<i32> = None;
    // IPC parameters
    let mut worker_id: u32 = 0;
    let mut num_handlers = 1;
    let mut num_shuffles = 0;
    let num_slots = 2048;
    let num_objective_layers = DEFAULT_LOOKAHEAD;

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
            num_objectives = args[i + 1].parse().unwrap_or(2);
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
    }

    let arena_name = format!("mcts_{}_{}", num_slots, num_handlers);
    let arena: Arena<TilersSlot> =
        Arena::create_or_open(&arena_name, num_slots, num_handlers).unwrap();
    let client = TilersIpcClient::new(arena, worker_id);

    // Spawn all gatherers as independent tasks
    let output_path = format!("output-{}.json", worker_id);
    let gatherer = Gatherer::new(
        8,         // inference batch size
        10_000,    // MCTS steps
        80,        // max actions
        output_path,
        0.20,      // noise strength
        num_objective_layers,
    );

    loop {
        // Prepare the RNG
        let mut rng = if let Some(s) = seed {
            let rng = StdRng::seed_from_u64(s as u64);
            seed = Some(s + 1); // Increment seed for next iteration
            rng
        } else {
            StdRng::from_rng(&mut rand::rng())
        };

        let h = rng.random_range(2..=height);
        let w = rng.random_range(2..=width);
        let dim_max = h.max(w);
        let dim_min = h.min(w);
        let h = dim_min;
        let w = dim_max;
        let nb = rng.random_range(1..=num_blanks);
        let no = rng.random_range(1..=num_objectives);
        if nb >= (h * w) - 1 || (h <= 2 && w <= 2) {
            continue;
        }
        let mut env = Environment::new(h, w, nb);

        // Seed the environment
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
        let mut game_rng = StdRng::from_rng(&mut rand::rng());
        let (sol_depth, ref_depth, done) = gatherer.gather(&env, &client, &mut game_rng);
        if done {
            println!(
                "[Gatherer {}] Env(h={}, w={}, nb={}, no={}): solution depth = {}, reference depth = {}",
                worker_id, h, w, nb, no, sol_depth, ref_depth
            );
        } else {
            println!(
                "[Gatherer {}] Env(h={}, w={}, nb={}, no={}): (bootstrapped) depth = {}, reference depth = {}",
                worker_id, h, w, nb, no, sol_depth, ref_depth
            );
        }
    }
}