use mcts_core::ipc_core::Arena;
use mcts_core::mcts::MCTS;
use mcts_core::inference::IpcClient;
use tilers_core::env::Environment;


pub fn run_producer(producer_id: usize, arena_name: &str, num_slots: usize, num_handlers: usize) {
    loop {
        let arena = Arena::create_or_open(arena_name, num_slots, num_handlers).unwrap();
        let mut env = Environment::new(4, 4, 1);
        env.random_start(2, false);
        let mut mcts = MCTS::<Environment>::default();
        let client = IpcClient::new(arena, producer_id as u32);
        // acquire a free slot
        println!("{}", env.render());
        while !env.done() {
            let node = mcts.run(&env, &client, 10000);
            let action = node.select_action().unwrap();
            let _ = env.step(action as usize);
            println!("{}", env.render());
        }
    }
}

fn main() {
    let producer_id = std::env::args().nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let arena_name = "example_mcts";
    let num_slots = 1024usize;
    let num_handlers = 1usize;
    run_producer(producer_id, arena_name, num_slots, num_handlers);
}
