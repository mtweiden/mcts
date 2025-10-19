use mcts::MCTS;
use mcts::agent::DummyAgent;
use tilers_core::env::Environment;

fn run_trial() {
    let mut env = Environment::new(4, 4, 2);
    env.random_start(2, false);
    let mut mcts = MCTS::new(1.0_f32, 4usize);
    let agent = DummyAgent::new(env.num_actions());

    println!("{}", env.render());

    for _ in 0..100 {
        let root_id = mcts.run(&env, &agent, 20000usize);
        let best_action = mcts.select_action(root_id);
        env.step(best_action.unwrap());
        if env.done() {
            break;
        }
        println!("{}", env.render());
    }
}

fn main() {
    let num_trials = 100;

    for i in 0..num_trials {
        println!("----------------------------------------------------------------------");
        println!("Running trial {}...", i + 1);
        println!("----------------------------------------------------------------------");
        run_trial();
    }
}