use mcts::MCTS;
use mcts::agent::DummyAgent;
use tilers_core::env::Environment;

fn main() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[14];
        cx q[3],q[6];
        t q[2];
    "#;

    let mut env = Environment::from_qasm(qasm, 2, Some(4), Some(4));
    let mut mcts = MCTS::new(0.0_f32, 4usize, None);
    let agent = DummyAgent::new(env.num_actions());

    println!("{}", env.render());

    for _ in 0..100 {
        let root_id = mcts.run(&env, &agent, 100000usize);
        let best_action = mcts.select_action(root_id);
        env.step(best_action.unwrap());
        if env.done() {
            break;
        }
        println!("{}", env.render());
    }
}