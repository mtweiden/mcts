use mcts::MCTS;
use mcts::runner::MCTSRunner;
use mcts::agent::DummyAgent;
use tilers_core::env::Environment;

fn run_trial() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[14];
        cx q[3],q[6];
        t q[2];
    "#;

    let mut env = Environment::from_qasm(qasm, 2, Some(4), Some(4));
    let mcts = MCTS::new(0.0_f32, 4usize);
    let agent = DummyAgent::new(env.num_actions());

    let runner = MCTSRunner::new(
        mcts,
        1,
        8,
        1024,
    );

    println!("{}", env.render());

    for _ in 0..100 {
        let root_id = runner.run(env.clone(), agent.clone(), 100000usize);
        let best_action = runner.mcts.select_action(root_id);
        env.step(best_action.unwrap());
        println!("\n\nTook action: {:?}", best_action);
        println!("{}", env.render());
        if env.done() {
            break;
        }
    }
}

fn main() {
    // let num_trials = 100;
    // for i in 0..num_trials {
    //     println!("----------------------------------------------------------------------");
    //     println!("Running trial {}...", i + 1);
    //     println!("----------------------------------------------------------------------");
    //     run_trial();
    // }
    run_trial();
}