use std::env;

use mcts_tilers::evaluator::Evaluator;

// =============================================================================
// Defaults
// =============================================================================
const DEFAULT_MCTS_STEPS: usize = 10_000;
const DEFAULT_C_PUCT: f32 = 1.4;
const DEFAULT_REWARD_SATURATION_TEMPERATURE: f32 = 0.3;

// =============================================================================
// main
// =============================================================================
// Example usage:
// ./target/release/evaluate \
//    --agent_id 2 \
//    --db pipeline.db \
//    --arena_tag eval \
//    --num_handlers 1 \
//    --mcts_steps 10000
fn main() {
    let args: Vec<String> = env::args().collect();

    let mut agent_id: i64 = 0;
    let mut db_path = String::from("pipeline.db");
    let mut mcts_steps = DEFAULT_MCTS_STEPS;
    let mut c_puct = DEFAULT_C_PUCT;
    let mut reward_saturation_temperature = DEFAULT_REWARD_SATURATION_TEMPERATURE;
    let mut arena_tag = String::from("eval");
    let mut num_handlers: usize = 1;
    let mut max_num_objectives: Option<i64> = None;
    let mut node_idx: i64 = 0;
    let mut num_nodes: i64 = 1;
    let mut max_action_multiplier: f32 = 1.2;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--agent_id"            => { agent_id = args[i+1].parse().unwrap(); i += 2; }
            "--db"                  => { db_path = args[i+1].clone(); i += 2; }
            "--mcts_steps"          => { mcts_steps = args[i+1].parse().unwrap(); i += 2; }
            "--c_puct"              => { c_puct = args[i+1].parse().unwrap(); i += 2; }
            "--reward_saturation_temperature" => { reward_saturation_temperature = args[i+1].parse().unwrap(); i += 2; }
            "--arena_tag"           => { arena_tag = args[i+1].clone(); i += 2; }
            "--num_handlers"        => { num_handlers = args[i+1].parse().unwrap(); i += 2; }
            "--max_num_objectives"  => { max_num_objectives = Some(args[i+1].parse().unwrap()); i += 2; }
            "--node_idx"            => { node_idx = args[i+1].parse().unwrap(); i += 2; }
            "--num_nodes"           => { num_nodes = args[i+1].parse().unwrap(); i += 2; }
            "--max_action_multiplier" => { max_action_multiplier = args[i+1].parse().unwrap(); i += 2; }
            _                       => { i += 1; }
        }
    }

    assert!(agent_id > 0, "--agent_id is required and must be positive");

    let evaluator = Evaluator::new(
        mcts_steps,
        c_puct,
        reward_saturation_temperature,
        max_action_multiplier,
    );

    evaluator
        .evaluate_agent(
            agent_id, &db_path, &arena_tag, num_handlers,
            max_num_objectives, node_idx, num_nodes,
        )
        .expect("evaluation failed");
}