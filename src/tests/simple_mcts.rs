use std::collections::HashMap;
use crate::{MCTS};
use crate::agent::DummyAgent;
use tilers_core::env::Environment;

/// A small integration-style unit test that runs MCTS with a DummyAgent
/// on a tiny Environment. This is intentionally minimal — it checks that
/// run() returns a root id and that the node was inserted into the table.
#[test]
fn simple_mcts_run_with_dummy_agent() {
    // A tiny QASM-like program (adapt if your Environment expects a different format)
    let qasm = "
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[14];
        cx q[3],q[6];
        t q[2];
    ";

    // Build the environment. from_qasm takes Option<usize> for height/width.
    let mut env = Environment::from_qasm(qasm, 2, Some(4), Some(4));

    // Create MCTS and a trivial agent. Adjust terminal value / batch size to taste.
    let mcts = MCTS::new(0.0_f32, 4usize);
    let agent = DummyAgent::new(env.num_actions()); // 4 actions

    // Run MCTS for a small number of steps.
    let root_id = mcts.run(&env, agent, 64usize);

    // Ensure the root node exists in the transposition table after running.
    assert!(mcts.get_node(root_id).is_some(), "root node should be present");
}