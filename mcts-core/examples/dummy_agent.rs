// Loop pop_ready, read inputs, write dummy priors and values, mark_done
use mcts_core::ipc_core::Arena;
use mcts_core::agent::DummyAgent;

pub fn run_handler(arena: Arena, handler_id: usize) {
    let agent = DummyAgent::new();
    loop {
        let slot = arena.pop_ready(handler_id);

        // read inputs (read-only view)
        let obs = {
            let sr = arena.slot(slot);
            sr.slot.unpack_observations()

        };

        // simulate work
        let mut priors = vec![];
        let mut values = vec![];
        for i in 0..obs.len() {
            let (p, v) = agent.infer(&obs[i]);
            priors.push(p);
            values.push(v);
        }

        // write outputs
        {
            let sm = arena.slot_mut(slot);
            // fill priors uniformly for first action set
            sm.slot.pack_priors_values(&priors, &values).unwrap();
        }

        // mark done
        arena.mark_done(slot);
        println!("Agent {} saw {} actions", handler_id, priors.len());
    }
}

fn main() {
    let arena_name = "example_mcts";
    let num_slots = 1024usize;
    let num_handlers = 2usize;
    let arena = Arena::create_or_open(arena_name, num_slots, num_handlers).unwrap();
    let handler_id = std::env::args().nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    run_handler(arena, handler_id);
}