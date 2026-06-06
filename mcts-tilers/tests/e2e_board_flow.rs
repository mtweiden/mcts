//! End-to-end smoke test of the migrated flow with the trivial (non-neural)
//! agent: real env → MCTS search → 10-channel board observation → `rl`
//! action encode/decode → step → board-format training JSON.
//!
//! This exercises the whole Rust pipeline except the GPU NN (which the
//! `TrivialTilersIpcClient` replaces with uniform priors).

use mcts_tilers::client::TrivialTilersIpcClient;
use mcts_tilers::constants::{CELL_FIELDS, NUM_ACTIONS};
use mcts_tilers::environment::TilersEnv;
use mcts_tilers::gatherer::Gatherer;

use mcts_core::Environment as MctsEnvironment;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tilers::env::Environment as TilersEnvInner;

#[test]
fn e2e_observation_is_a_well_formed_board() {
    // The migrated observation must be the 10-channel board, one layer per
    // objective layer, h*w cells in row-major order.
    let mut inner = TilersEnvInner::new(4, 4, 3);
    inner.set_seed(Some(7));
    inner.random_start(2, false);
    let env = TilersEnv::new(inner, 2);

    let obs = env.observation();
    assert_eq!(obs.num_layers, obs.board.len());
    assert_eq!(obs.action_mask.len(), NUM_ACTIONS, "mask padded to 1 + 6N");
    for layer in &obs.board {
        assert_eq!(layer.len(), obs.height * obs.width, "h*w cells per layer");
    }
    // BoardCell is a 10-channel record.
    assert_eq!(CELL_FIELDS, 10);
    eprintln!(
        "[e2e] obs: {}x{} na={} layers={} cells/layer={} mask_len={}",
        obs.height,
        obs.width,
        obs.num_ancillas,
        obs.num_layers,
        obs.board[0].len(),
        obs.action_mask.len(),
    );
}

#[test]
fn e2e_gather_full_game_writes_board_records() {
    // A real env with a few random gates, solved start-to-finish by the
    // MCTS gatherer driven by the trivial (uniform-prior) agent.
    let mut inner = TilersEnvInner::new(4, 4, 3);
    inner.set_seed(Some(11));
    inner.random_start(2, false);

    let tmp = std::env::temp_dir().join("mcts_e2e_gather.jsonl");
    let _ = std::fs::remove_file(&tmp);
    let out = tmp.to_str().unwrap().to_string();

    // batch_size, mcts_steps, fast_steps, p_full_search, output, noise,
    // dirichlet, lookahead, gather_id, traj_dir, reward_limit, max_actions
    let gatherer = Gatherer::new(
        1, 16, 8, 1.0, out.clone(), 0.0, 0.0, 1, 0, None, Some(1.0), None,
    );
    let client = TrivialTilersIpcClient {};
    let mut rng = StdRng::seed_from_u64(0);

    let (score, _r, win) = gatherer.gather(&inner, &client, 1.4, &mut rng);
    eprintln!("[e2e] gather finished: score={score:.3} win={win}");

    // The full-search records were appended to `out` in the new board schema.
    let data = std::fs::read_to_string(&out).expect("gatherer wrote output");
    let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(!lines.is_empty(), "gatherer produced no training records");

    let rec: serde_json::Value = serde_json::from_str(lines[0]).expect("record is valid JSON");
    // New schema: board instead of placement/objectives/last_dirs.
    assert!(rec.get("placement").is_none(), "old placement key must be gone");
    assert!(rec.get("board").is_some(), "record must carry the board");
    let board = rec["board"].as_array().expect("board is an array of layers");
    let layer0 = board[0].as_array().expect("layer is an array of cells");
    let cell0 = layer0[0].as_array().expect("cell is an array of channels");
    assert_eq!(cell0.len(), CELL_FIELDS, "each cell has 10 channels");

    // Action ids in the record are within the 1 + 6N action space.
    for va in rec["valid_actions"].as_array().unwrap() {
        let id = va.as_u64().unwrap() as usize;
        assert!(id < NUM_ACTIONS, "action id {id} out of range");
    }

    eprintln!(
        "[e2e] {} board records; layers={} cells/layer={} channels/cell={}",
        lines.len(),
        board.len(),
        layer0.len(),
        cell0.len(),
    );
    eprintln!("[e2e] sample cell[layer0][cell0] = {cell0:?}");

    let _ = std::fs::remove_file(&tmp);
}
