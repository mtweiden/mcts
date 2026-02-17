use std::env;
use std::collections::HashMap;
use std::io::Write;
use std::iter::zip;
use json::JsonValue;
use rand_distr::{Gamma, Distribution};
use rand_distr::weighted::WeightedIndex;
use rand::Rng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

use mcts_core::enums::Action;
use mcts_core::{Arena, InferenceClient, IpcClient, MCTS};
use mcts_core::node::Node;
use tilers_core::env::Environment;

/// ----------------------------------------------------------------------------
/// Gatherer
/// ----------------------------------------------------------------------------
/// A struct to gather data from MCTS simulations.
/// Args:
///   mcts: An instance of the MCTS struct.
///   agent: An instance of an Agent.
///   mcts_steps: Number of MCTS simulations per move.
///   max_actions: Maximum number of actions to consider.
///   url: The URL of the remote inference server.
///   output_path: Path to save the gathered data.
/// ----------------------------------------------------------------------------
struct Gatherer {
    batch_size: usize,
    mcts_steps: usize,
    max_actions: usize,
    output_path: String,
    terminal_value: f32,
    noise_strength: f64,
}


impl Gatherer {
    pub fn new(
        batch_size: usize,
        mcts_steps: usize,
        max_actions: usize,
        output_path: String,
        noise_strength: f64,
    ) -> Self {
        let terminal_value: f32 = 1.0;
        Self {
            batch_size,
            mcts_steps,
            max_actions,
            output_path,
            terminal_value,
            noise_strength,
        }
    }

    /// Solve the environment using a heuristic solver and return the depth of the solution.
    pub fn solve_with_heuristic(&self, env: &Environment) -> f32 {
        let mut solved_env = env.clone();
        solved_env.solve(true);
        solved_env.depth(true)
    }

    /// Directly sampling from Dirichlet distribution requires num_actions to be known at
    /// compile time, so we sample using Gamma distributions instead.
    fn _dirichlet_noise(&self, num_actions: usize, rng: &mut impl Rng) -> Vec<f64> {
        let alpha = 10f64 / (num_actions as f64);  // Rule of thumb for Dirichlet noise
        let alphas = vec![alpha; num_actions];
        let mut xs: Vec<f64> = alphas.iter()
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

    fn _action_probabilities(&self, visit_counts: &Vec<usize>) -> Vec<f64> {
        let total_visits: usize = visit_counts.iter().sum();
        if total_visits == 0 {
            return vec![1.0 / (visit_counts.len() as f64); visit_counts.len()];
        }
        visit_counts.iter()
            .map(|&count| count as f64 / total_visits as f64)
            .collect()
    }

    pub fn select_action(&self, node: &Node, env: &Environment, noiseless: bool, rng: &mut impl Rng) -> Action {
        let valid_actions = env.valid_actions();
        let num_actions = valid_actions.len();
        if num_actions == 0 { panic!("No valid actions available"); }
        let probs = self._action_probabilities(
            &valid_actions.iter()
                .map(|&a| *node.edge_visits.get(&(a as Action)).unwrap_or(&0)).collect()
        );
        let noise = if !noiseless { self._dirichlet_noise(num_actions, rng) } else { vec![0.0; num_actions] };
        let mixed_probs: Vec<f64> = probs.iter().zip(noise.iter())
            .map(|(&p, &n)| (1.0 - self.noise_strength) * p + self.noise_strength * n)
            .map(|x| x.max(0.0)) // prevent tiny negatives
            .collect();
        let mut rng = rand::rng();
        let dist = WeightedIndex::new(&mixed_probs).unwrap();
        valid_actions[dist.sample(&mut rng)] as Action
    }

    /// For each action taken, compare the agent's score to the heuristic's along every step of
    /// the trajectory.
    pub fn score_transitions(
        &self,
        base_env: &Environment,
        agent_actions: &Vec<usize>,
    ) -> Vec<f32> {
        // Plus 1 for the terminal state at the end
        let mut scores: Vec<f32> = Vec::with_capacity(agent_actions.len() + 1);

        let mut env = base_env.clone();

        // If the agent is better than the heuristic from the start, reward immediately.
        let initial_ref_depth = self.solve_with_heuristic(&env);
        let initial_agent_depth = {
            let mut temp_env = env.clone();
            for ac in agent_actions {
                let _ = temp_env.step(*ac);
                temp_env.finish_cultivating();
            }
            temp_env.depth(true)
        };
        if initial_agent_depth < initial_ref_depth {
            return vec![1.0; agent_actions.len()];
        }

        // For each action in the agent trajectory, produce a value target for the
        // current state (before taking that action).
        for i in 0..agent_actions.len() {
            // Starting at environment state after action i-1
            env.executed_objectives.clear();

            // Heuristic reference depth from this state (clear cultivated resources
            // when necessary so heuristic can move).
            let valid_actions = env.valid_actions();
            let ref_depth = if !valid_actions.contains(&0) && !valid_actions.iter().any(|&a| a > env.num_ancillas) {
                let mut tmp = env.clone();
                tmp.clear_cultivated_resources();
                self.solve_with_heuristic(&tmp)
            } else {
                self.solve_with_heuristic(&env)
            };

            // Depth if the agent follows its remaining actions from this state
            let mut temp_env = env.clone();
            let remaining_actions = &agent_actions[i..];
            for ac in remaining_actions {
                let _ = temp_env.step(*ac);
                temp_env.finish_cultivating();
            }
            let agent_depth = temp_env.depth(true);

            // Check whether the immediate action finishes the game (score = +1.0).
            let ac = agent_actions[i];
            let mut next_env = env.clone();
            let _ = next_env.step(ac);
            next_env.finish_cultivating();

            let score = if next_env.done() {
                1.0f32
            } else {
                // Small bias so recreating the heuristic's actions is not neutral but
                // slightly rewarded.
                ((0.1 + ref_depth - agent_depth) / 2.0).tanh()
            };
            scores.push(score);

            // Advance the working environment by the chosen action
            let _ = env.step(ac);
            env.finish_cultivating();
        }
        scores
    }

    pub fn supervised_edge_visits(
        &self,
        selected_action: Action,
        valid_actions: &Vec<usize>,
        weight_on_selected: f32,
    ) -> HashMap<Action, usize> {
        assert!(weight_on_selected >= 0.0 && weight_on_selected <= 1.0, "weight_on_selected must be in [0, 1]");
        let num_actions = valid_actions.len();
        let weight_on_unselected = if num_actions > 1 {
            (1.0 - weight_on_selected) / (num_actions as f32 - 1.0)
        } else {
            0.0
        };
        let mut edge_visits: HashMap<Action, usize> = HashMap::new();
        for &action in valid_actions {
            let visits = if action == (selected_action as usize) {
                (weight_on_selected * 1000.0) as usize
            } else {
                (weight_on_unselected * 1000.0) as usize
            };
            edge_visits.insert(action as Action, visits);
        }
        edge_visits
    }

    pub fn gather(&self, env: &Environment, client: &dyn InferenceClient, rng: &mut impl Rng) -> (f32, f32, bool) {
        // Set up MCTS and Agent and copy the Environment
        let mut mcts: MCTS<Environment> = MCTS::new(self.terminal_value, self.batch_size);

        // Only consider the first two layers of gates
        let mut game = env.clone();
        game.drop_objectives_beyond_nth_layer(2);
        game.set_cultivation_time(10);
        let reference_depth = self.solve_with_heuristic(&mut game);

        // Set up data storage
        // Format: ((placement, objectives_0, objectives_1), valid_actions, visit_counts)
        let mut temp_data: Vec<((Vec<usize>, Vec<usize>, Vec<usize>), Vec<usize>, HashMap<Action, usize>)> = Vec::new();

        let mut taken_actions = vec![];
        for step in 0..self.max_actions {
            // Run MCTS
            let root = mcts.run(&game , client, self.mcts_steps);

            // Store the data
            let placement = game.get_placement_tokens().unwrap();
            let objectives_0 = game.get_objective_tokens(0).unwrap();
            let objectives_1 = game.get_objective_tokens(1).unwrap();
            let edge_visits = root.edge_visits.clone();
            let valid_actions = game.valid_actions();
            temp_data.push(((placement, objectives_0, objectives_1), valid_actions, edge_visits));

            // Select action and step the environment
            // Add noise if we're very close to the root to encourage exploration
            let noiseless = step > 2;
            let action = self.select_action(&root, &game, noiseless, rng);
            let _ = game.step(action as usize);
            game.finish_cultivating();  // Cultivate resources in a single step
            taken_actions.push(action as usize);
            if game.done() { break; }

            // Advance the root
            mcts.advance_root(action);
        }

        // If not solved, bootstrap using a heuristic solution from the final state so that
        // some supervised learning can be done.
        let solution_depth = if !game.done() {
            let bootstrap_actions = game.solve(false);
            for ac in bootstrap_actions {
                let placement = game.get_placement_tokens().unwrap();
                let objectives_0 = game.get_objective_tokens(0).unwrap();
                let objectives_1 = game.get_objective_tokens(1).unwrap();
                let valid_actions = game.valid_actions();
                // Construct a plausable visit count distribution that heavily favors the next
                // action in the heuristic solution, but still has some mass on other valid actions.
                let edge_visits = self.supervised_edge_visits(
                    ac as Action,
                    &valid_actions, 
                    0.7,
                );
                temp_data.push(((placement, objectives_0, objectives_1), valid_actions, edge_visits));
                // Advance to next state
                let _ = game.step(ac);
                game.finish_cultivating();
                // Record heuristic action
                taken_actions.push(ac);
            }
            game.depth(true)
        } else {
            // If solved add the terminal state with a value of +1.0
            let solution_depth = game.depth(true);
            solution_depth
        };
        assert!(game.done(), "Heuristic failed to solve the environment");
        // Append final terminal state so we train on the done state itself.
        // Build a terminal record matching temp_data shape with empty visits.
        let final_placement = game.get_placement_tokens().unwrap();
        let final_o0 = game.get_objective_tokens(0).unwrap();
        let final_o1 = game.get_objective_tokens(1).unwrap();
        let final_valid: Vec<usize> = vec![];
        let final_visits: HashMap<Action, usize> = HashMap::new();
        temp_data.push(((final_placement, final_o0, final_o1), final_valid, final_visits));

        // Determine scores for all transitions
        let mut scoring_game = env.copy();
        scoring_game.set_cultivation_time(10);
        let mut scores = self.score_transitions(&scoring_game, &taken_actions);
        scores.push(self.terminal_value);


        // Save data to output_path in NDJSON format
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)
            .expect("Unable to open output file");

        for (((p, o0, o1), va, ev), score) in zip(temp_data, scores) {
            // Build JSON using `json` crate (avoids serde_json)
            let mut placement_tokens_json = JsonValue::new_array();
            for t in p {
                placement_tokens_json.push(t).expect("failed to push token");
            }

            let mut valid_actions_json = JsonValue::new_array();
            for a in va {
                valid_actions_json.push(a).expect("failed to push token");
            }

            let mut objectives_0_tokens_json = JsonValue::new_array();
            for t in o0 {
                objectives_0_tokens_json.push(t).expect("failed to push token");
            }

            let mut objectives_1_tokens_json = JsonValue::new_array();
            for t in o1 {
                objectives_1_tokens_json.push(t).expect("failed to push token");
            }

            let mut visits_json = JsonValue::new_object();
            for (action, count) in ev {
                visits_json[action.to_string()] = count.into();
            }

            let mut record = JsonValue::new_object();
            record["height"] = game.height.into();
            record["width"] = game.width.into();
            record["num_ancillas"] = game.num_ancillas.into();
            record["placement"] = placement_tokens_json;
            record["objectives_0"] = objectives_0_tokens_json;
            record["objectives_1"] = objectives_1_tokens_json;
            record["valid_actions"] = valid_actions_json;
            record["edge_visits"] = visits_json;
            record["reward"] = score.into();

            let line = record.dump(); // compact JSON string
            writeln!(file, "{}", line).expect("Failed to write record");
        }
        writeln!(file, "").expect("Failed to write newline");
        file.flush().expect("Failed to flush file");
        (solution_depth as f32, reference_depth as f32, game.done())
    }
}


pub fn shuffle_ancilla<'a>(env: &'a mut Environment, rng: &mut impl Rng) -> &'a Environment {
    let qubits = env.placement.qubits.clone();
    let ancilla_indices: Vec<usize> = (0..qubits.len())
        .filter(|&i| qubits[i].is_ancilla())
        .collect();
    let mut shuffled_ancilla_indices = ancilla_indices.clone();
    shuffled_ancilla_indices.shuffle(rng);
    let mut new_qubits = qubits.clone();

    for (orig_idx, shuffled_idx) in zip(ancilla_indices.iter(), shuffled_ancilla_indices.iter()) {
        new_qubits[*orig_idx] = qubits[*shuffled_idx].clone();
    }
    env.set_layout(new_qubits).unwrap();
    env
}


fn main() {
    // Environment parameters
    let mut height = 4;
    let mut width = 4;
    let mut num_objectives = 2;
    let mut num_blanks = 2;
    let mut seed: Option<i32> = None;
    // IPC parameters
    let mut worker_id = 0;
    let mut num_handlers = 1;
    let mut num_shuffles = 0;
    let num_slots = 2048;
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

    // let arena_name = format!("mcts_{}_{}", num_slots, num_handlers);
    let arena_name = "example_mcts".to_string();
    let arena = Arena::create_or_open(&arena_name, num_slots, num_handlers).unwrap();
    let client = IpcClient::new(arena, worker_id);
    // Spawn all gatherers as independent tasks
    // let output_path = format!("/pscratch/sd/m/mtweiden/tile_mcts/data/output-{}.json", worker_id);
    let output_path = format!("output-{}.json", worker_id);
    let gatherer = Gatherer::new(
        8,         // inference batch size
        10_000,    // MCTS steps
        80,       // max actions
        output_path,
        0.10,  // noise strength
    );

    loop {
        // Prepare the RNG
        let mut rng = if let Some(s) = seed {
            let rng = StdRng::seed_from_u64(s as u64);
            seed = Some(s + 1); // Increment seed for next iteration
            rng
        } else {
            StdRng::from_os_rng()
        };

        let h = rng.random_range(2..=height);
        let w = rng.random_range(2..=width);
        let dim_max = h.max(w);
        let dim_min = h.min(w);
        let h = dim_min;
        let w = dim_max;
        let nb = rng.random_range(1..=num_blanks);
        let no = rng.random_range(1..=num_objectives);
        if nb >= (h * w) - 1 || (h <= 2 && w <= 2) { continue; }
        let mut env = Environment::new(h, w, nb);

        // Seed the environment
        if !seed.is_none() { env.set_seed(Some(seed.unwrap() as u64)) }

        env.random_objectives(no, false);
        if env.valid_actions().contains(&0) {
            let mut tmp_env = env.clone();
            let _ = tmp_env.step(0);
            if tmp_env.done() { continue; }
        }
        env.shuffle(num_shuffles);
        shuffle_ancilla(&mut env, &mut rng);
        let mut game_rng = StdRng::from_os_rng();
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
