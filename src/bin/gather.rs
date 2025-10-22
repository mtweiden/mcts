use std::env;
use std::collections::HashMap;
use std::io::Write;
use json::JsonValue;
use rand_distr::{Gamma, Distribution};
use rand_distr::weighted::WeightedIndex;

use mcts::enums::Action;
use mcts::MCTS;
use mcts::node::Node;
use mcts::agent::DummyAgent;
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
    inference_batch_size: usize,
    mcts_steps: usize,
    max_actions: usize,
    url: String,
    output_path: String,
    terminal_value: f32,
    noise_strength: f64,
}


impl Gatherer {
    pub fn new(
        inference_batch_size: usize,
        mcts_steps: usize,
        max_actions: usize,
        url: String,
        output_path: String,
        noise_strength: f64,
    ) -> Self {
        let terminal_value: f32 = 1.0;
        Self {
            inference_batch_size,
            mcts_steps,
            max_actions,
            url,
            output_path,
            terminal_value,
            noise_strength,
        }
    }

    /// Solve the environment using a heuristic solver and return the depth of the solution.
    pub fn solve_with_heuristic(&self, env: &mut Environment) -> usize {
        let mut solved_env = env.clone();
        solved_env.solve_and_take_actions();
        solved_env.depth(true)
    }

    /// Directly sampling from Dirichlet distribution requires num_actions to be known at
    /// compile time, so we sample using Gamma distributions instead.
    fn _dirichlet_noise(&self, num_actions: usize) -> Vec<f64> {
        let alpha = 10f64 / (num_actions as f64);  // Rule of thumb for Dirichlet noise
        let mut rng = rand::rng();
        let alphas = vec![alpha; num_actions];
        let mut xs: Vec<f64> = alphas.iter()
            .map(|&a| {
                let gamma = Gamma::new(a, 1.0).unwrap();
                gamma.sample(&mut rng)
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

    pub fn select_action(&self, node: &Node, env: &Environment) -> Action {
        let valid_actions = env.valid_actions();
        let num_actions = valid_actions.len();
        if num_actions == 0 { panic!("No valid actions available"); }
        let noise = self._dirichlet_noise(num_actions);
        let probs = self._action_probabilities(
            &valid_actions.iter().map(|&a| *node.edge_visits.get(&a).unwrap_or(&0)).collect()
        );
        let mixed_probs: Vec<f64> = probs.iter().zip(noise.iter())
            .map(|(&p, &n)| (1.0 - self.noise_strength) * p + self.noise_strength * n)
            .map(|x| x.max(0.0)) // prevent tiny negatives
            .collect();
        let mut rng = rand::rng();
        let dist = WeightedIndex::new(&mixed_probs).unwrap();
        valid_actions[dist.sample(&mut rng)]
    }

    pub async fn run(&self, env: &Environment) {
        // Set up MCTS and Agent and copy the Environment
        let mut mcts: MCTS<Environment> = MCTS::new(self.terminal_value, self.inference_batch_size, Some(self.url.clone()));
        let agent = DummyAgent::new(env.num_actions());

        // Only consider the first two layers of gates
        let mut game = env.clone();
        game.drop_objectives_beyond_nth_layer(2);
        let reference_depth = self.solve_with_heuristic(&mut game);

        // Set up data storage
        // Format: (tokens, visit_counts)
        let mut temp_data: Vec<(Vec<usize>, HashMap<Action, usize>)> = Vec::new();

        for _ in 0..self.max_actions {
            // Run MCTS
            let root = mcts.run(&game , &agent, self.mcts_steps).await;

            // Store the data
            let tokens = game.get_tokens();
            let edge_visits = root.edge_visits.clone();
            temp_data.push((tokens, edge_visits));

            // Select action and step the environment
            let action = self.select_action(&root, &game);
            game.step(action);
            if game.done() { break; }
        }

        // Loss condition
        let solution_depth = game.depth(true);
        let reward =  if !game.done() || solution_depth > reference_depth {
            -1.0
        } else if solution_depth == reference_depth{
            0.0
        } else {
            1.0
        };
        // Save data to output_path in NDJSON format
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)
            .expect("Unable to open output file");

        for (tokens, edge_visits) in temp_data {
            // Build JSON using `json` crate (avoids serde_json)
            let mut tokens_json = JsonValue::new_array();
            for t in tokens {
                tokens_json.push(t).expect("failed to push token");
            }

            let mut visits_json = JsonValue::new_object();
            for (action, count) in edge_visits {
                visits_json[action.to_string()] = count.into();
            }

            let mut record = JsonValue::new_object();
            record["tokens"] = tokens_json;
            record["edge_visits"] = visits_json;
            record["reward"] = reward.into();

            let line = record.dump(); // compact JSON string
            writeln!(file, "{}", line).expect("Failed to write record");
         }
    }
}


#[tokio::main]
async fn main() {
    // Default URL if not provided
    let mut server_url = String::from("http://localhost:8000");

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--server" && i + 1 < args.len() {
            server_url = args[i + 1].clone();
        }
    }

    println!("Using inference server at: {}", server_url);

    let gatherer = Gatherer::new(
        128,
        10000,
        100,
        server_url,
        String::from("/pscratch/sd/m/mtweiden/tile/data/output.json"),
        0.25,
    );

    let mut count = 0;
    loop {
        let mut env = Environment::new(4, 4, 2);
        env.random_start(2, false);
        gatherer.run(&env).await;
        println!("Finished gather {}", count + 1);
        count += 1;
    }
}
