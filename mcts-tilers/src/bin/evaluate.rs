use std::env;

use mcts_tilers::evaluator::Evaluator;

// =============================================================================
// Defaults
// =============================================================================
const DEFAULT_MCTS_STEPS: usize = 10_000;
const DEFAULT_C_PUCT: f32 = 1.4;
const DEFAULT_REWARD_RATIO_LIMIT: f32 = 0.3;

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
    let mut reward_ratio_limit = DEFAULT_REWARD_RATIO_LIMIT;
    let mut arena_tag = String::from("eval");
    let mut num_handlers: usize = 1;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--agent_id"           => { agent_id = args[i+1].parse().unwrap(); i += 2; }
            "--db"                 => { db_path = args[i+1].clone(); i += 2; }
            "--mcts_steps"         => { mcts_steps = args[i+1].parse().unwrap(); i += 2; }
            "--c_puct"             => { c_puct = args[i+1].parse().unwrap(); i += 2; }
            "--reward_ratio_limit" => { reward_ratio_limit = args[i+1].parse().unwrap(); i += 2; }
            "--arena_tag"          => { arena_tag = args[i+1].clone(); i += 2; }
            "--num_handlers"       => { num_handlers = args[i+1].parse().unwrap(); i += 2; }
            _                      => { i += 1; }
        }
    }

    assert!(agent_id > 0, "--agent_id is required and must be positive");

    let evaluator = Evaluator::new(
        mcts_steps,
        c_puct,
        reward_ratio_limit,
    );

    evaluator
        .evaluate_agent(agent_id, &db_path, &arena_tag, num_handlers)
        .expect("evaluation failed");
}