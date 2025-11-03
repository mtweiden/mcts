use std::env;
use std::collections::HashMap;
use std::io::Write;
use json::JsonValue;
use rand_distr::{Gamma, Distribution};
use futures::future::join_all;
use tokio::task;

use mcts::enums::Action;
use tilers_core::env::Environment;

/// ----------------------------------------------------------------------------
/// HeuristicGatherer
/// ----------------------------------------------------------------------------
/// A struct to gather data from MCTS simulations.
/// Args:
///   mcts: An instance of the MCTS struct.
///   mcts_steps: Number of MCTS simulations per move.
///   max_actions: Maximum number of actions to consider.
///   output_path: Path to save the gathered data.
/// ----------------------------------------------------------------------------
struct HeuristicGatherer {
    output_path: String,
}


impl HeuristicGatherer {
    pub fn new(
        output_path: String,
    ) -> Self {
        Self { output_path }
    }

    /// Solve the environment using a heuristic solver and return the depth of the solution.
    pub fn solve_with_heuristic(&self, env: &mut Environment) -> (Vec<usize>, usize) {
        let mut solved_env = env.clone();
        let actions = solved_env.solve_and_take_actions();
        let depth = solved_env.depth(true);
        (actions, depth)
    }

    /// For each legal action, apply it and solve the resulting state with the heuristic solver.
    /// Return a vector of action indices ranked by the depth of the solution.
    pub fn ranked_actions(&self, env: &Environment) -> HashMap<usize, usize> {
        if env.done() {
            return HashMap::new();
        }
        let base_line_depth = self.solve_with_heuristic(&mut env.clone()).1;
        let legal_actions = env.valid_actions();
        let mut rankings = HashMap::new();
        for ac in legal_actions.iter() {
            let mut game = env.clone();
            let (_, depth) = self.solve_with_heuristic(&mut game);
            if depth < base_line_depth {
                rankings.insert(*ac as usize, 10);
            } else if depth == base_line_depth {
                rankings.insert(*ac as usize, 5);
            } else {
                rankings.insert(*ac as usize, 1);
            }
            game.step(*ac);
            game.cultivator.finish_cultivating();
        }
        rankings
    }

    pub async fn run(&self, env: &Environment) {
        // Only consider the first two layers of gates
        let mut game = env.clone();
        game.drop_objectives_beyond_nth_layer(2);
        let (actions, _) = self.solve_with_heuristic(&mut game);

        // Set up data storage
        // Format: ((placement, objectives_0, objectives_1), valid_actions, visit_counts)
        let mut temp_data: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, HashMap<Action, usize>)> = Vec::new();

        for ac in actions {
            // Run MCTS
            let (placement, objectives_0) = game.get_tokens();
            let objectives_1 = game.get_objective_tokens(1);
            let valid_actions = game.valid_actions();
            let action_weights = self.ranked_actions(&game);
            temp_data.push((placement, objectives_0, objectives_1, valid_actions, action_weights));
            // Select action and step the environment
            game.step(ac);
            game.cultivator.finish_cultivating();
        }

        // Loss condition
        let reward = 0.0;
        // Save data to output_path in NDJSON format
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)
            .expect("Unable to open output file");

        for (placement, objectives_0, objectives_1, valid_actions, edge_visits) in temp_data {
            // Build JSON using `json` crate (avoids serde_json)
            let mut placement_tokens_json = JsonValue::new_array();
            for t in placement {
                placement_tokens_json.push(t).expect("failed to push token");
            }

            let mut valid_actions_json = JsonValue::new_array();
            for ac in valid_actions {
                valid_actions_json.push(ac).expect("failed to push token");
            }

            let mut objectives_0_tokens_json = JsonValue::new_array();
            for t in objectives_0 {
                objectives_0_tokens_json.push(t).expect("failed to push token");
            }

            let mut objectives_1_tokens_json = JsonValue::new_array();
            for t in objectives_1 {
                objectives_1_tokens_json.push(t).expect("failed to push token");
            }

            let mut visits_json = JsonValue::new_object();
            for (action, count) in edge_visits {
                visits_json[action.to_string()] = count.into();
            }

            let mut record = JsonValue::new_object();
            record["height"] = game.height.into();
            record["width"] = game.width.into();
            record["placement"] = placement_tokens_json;
            record["objectives_0"] = objectives_0_tokens_json;
            record["objectives_1"] = objectives_1_tokens_json;
            record["valid_actions"] = valid_actions_json;
            record["edge_visits"] = visits_json;
            record["reward"] = reward.into();

            let line = record.dump(); // compact JSON string
            writeln!(file, "{}", line).expect("Failed to write record");
         }
    }
}


#[tokio::main]
async fn main() {
    let mut height = 4;
    let mut width = 4;
    let mut num_objectives = 2;
    let mut num_blanks = 2;
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
    }

    // How many concurrent gatherers to run
    // let num_gatherers = num_cpus::get();
    let num_gatherers = 256;
    println!("Launching {num_gatherers} gatherers...");

    // Spawn all gatherers as independent tasks
    let mut handles = Vec::new();
    for i in 0..num_gatherers {
        let output_path = format!("/pscratch/sd/m/mtweiden/tile_mcts/data/output-{}.json", i);
        let handle = task::spawn(async move {
            let gatherer = HeuristicGatherer::new(output_path);

            loop {
                let mut env = Environment::new(height, width, num_blanks);
                env.random_start(num_objectives, false);
                gatherer.run(&env).await;
            }
        });
        handles.push(handle);
    }

    // Wait for all gatherers to finish
    join_all(handles).await;
    println!("All gatherers completed.");
}
