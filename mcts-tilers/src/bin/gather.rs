use std::env;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use mcts_core::ipc_core::Arena;

use mcts_tilers::constants::*;
use mcts_tilers::slot::TilersSlot;
use mcts_tilers::client::TilersIpcClient;
use mcts_tilers::gatherer::Gatherer;

use tilers::env::Environment;
use tilers::solver::Solver;


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
    let lookahead = DEFAULT_LOOKAHEAD;
    let mut max_generated_depth = 10_000;
    let mut c_puct = 1.4;
    let mut mcts_steps: usize = 800;
    let mut fast_steps: usize = 140;
    let mut p_full_search: f32 = 0.25;
    let mut dirichlet_epsilon: f32 = 0.25;
    let mut trajectory_dir: Option<String> = None;
    let mut arena_tag = String::new();
    let mut output_dir = String::from("/shared/staging");

    // Reward ratio limit for value target scaling/clamping. 1.0 means no scaling.
    let mut reward_ratio_limit = 0.3f32;

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
        if args[i] == "--arena_tag" && i + 1 < args.len() {
            arena_tag = args[i + 1].clone();
        }
        if args[i] == "--output_dir" && i + 1 < args.len() {
            output_dir = args[i + 1].clone();
        }
        if args[i] == "--reward_ratio_limit" && i + 1 < args.len() {
            reward_ratio_limit = args[i + 1].parse().unwrap_or(0.3);
        }
    }

    let arena_name = if arena_tag.is_empty() {
        format!("mcts_{}_{}", num_slots, num_handlers)
    } else {
        format!("mcts_{}_{}_{}", arena_tag, num_slots, num_handlers)
    };
    let arena: Arena<TilersSlot> =
        Arena::create_or_open(&arena_name, num_slots, num_handlers).unwrap();
    let client = TilersIpcClient::new(arena, worker_id);

    let output_path = format!("{}/output-{}.jsonl", output_dir, worker_id);
    let gatherer = Gatherer::new(
        8,                  // inference batch size
        mcts_steps,         // full-search MCTS steps
        fast_steps,         // fast-search MCTS steps
        p_full_search,      // fraction of turns that are full searches
        output_path,
        0.20,               // action-selection noise strength
        dirichlet_epsilon,  // Dirichlet epsilon for MCTS root noise
        lookahead,
        worker_id as usize,
        trajectory_dir,
        Some(reward_ratio_limit),
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

        env.random_start(no, false);
        if env.valid_actions().contains(&tilers::enums::Action::AutoExecute) {
            let mut tmp_env = env.clone();
            let _ = tmp_env.step(tilers::enums::Action::AutoExecute);
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