use mcts::{Agent, Environment, MCTS};

#[derive(Clone)]
struct MockEnv {
    state: i32,
    limit: i32,
}

impl MockEnv {
    pub fn num_actions(&self) -> usize {
        2 // two actions: left or right
    }
}

impl Environment for MockEnv {
    fn hash_state(&self) -> u64 {
        self.state as u64
    }
    fn done(&self) -> bool {
        self.state.abs() > self.limit
    }
    fn valid_actions(&self) -> Vec<i32> {
        vec![0, 1] // move left or right
    }
    fn step(&mut self, action: i32) {
        self.state += if action == 1 { 1 } else { -1 };
    }
    fn copy(&self) -> Self {
        self.clone()
    }
    fn observation(&self) -> Vec<f32> {
        vec![self.state as f32]
    }
}

struct MockAgent {
    num_outputs: usize,
}

impl MockAgent {
    fn new(num_outputs: usize) -> Self {
        Self { num_outputs }
    }
}

impl Agent for MockAgent {

    fn infer(&self, _obs: &[f32]) -> (Vec<f32>, f32) {
        // Random priors + heuristic value
        let val: f32 = -1.0;
        let priors = vec![1.0 / self.num_outputs as f32; self.num_outputs];
        (priors, val)
    }
}

fn main() {
    let env = MockEnv { state: 0, limit: 50 };
    let agent = MockAgent::new(env.num_actions());
    let mcts = MCTS::new();
    mcts.run_parallel(&env, &agent, 100_000);
    println!("Transposition table size: {}", mcts.transposition_table.borrow().len());
}
