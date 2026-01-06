# `mcts-core`
## An implementation of MCTS that uses shared memory IPC for inference
Running `cargo run --bin gather --release` will launch Gather processes that run MCTS and query an external process for prior weights and values.

A `dummy` version of an inference agent can also be run with `cargo run --bin dummy --release`.

## Porting to custom environments
Any class can be used as an environment for MCTS as long as it implements the `Environment` trait defined in `mcts-core/src/environment.rs`. Here is an example for a custom environment from an example `tilers_core` project.
```rust
/// Provided implementation for tilers_core::env::Environment
impl Environment for tilers_core::env::Environment {
    fn step(&mut self, action: Action) {
        tilers_core::env::Environment::step(self, action);
    }

    fn done(&self) -> bool {
        tilers_core::env::Environment::done(self)
    }

    fn observation(&self) -> Observation {
        let (state_0, state_1) = tilers_core::env::Environment::get_tokens(self);
        let state_2 = tilers_core::env::Environment::get_objective_tokens(self, 1);
        let valid_actions = tilers_core::env::Environment::valid_actions(self);
        (state_0, state_1, state_2, self.height, self.width, valid_actions)
    }

    fn valid_actions(&self) -> Vec<Action> {
        tilers_core::env::Environment::valid_actions(self)
    }

    fn hash_state(&self) -> u64 {
        tilers_core::env::Environment::hash_state(self)
    }

    fn render(&self) -> String {
        tilers_core::env::Environment::render(self)
    }
}
```

# `mcts-ipc`
This directory contains shim code so that Python processes can communicate over shared memory IPC. This lets neural networks defined in, say, PyTorch communicate with the rust MCTS code.

# `pymcts`
Enables MCTS to be run with Python environments and Agents.

```python
from pymcts import PyMcts, MctsAgent, MctsEnvironment

agent = Agent()  # implements infer(obs: list[Observations]) -> tuple[list[Prior], list[Value]]
env = Environment()  # needs step, done, observation, valid_actions, hash_state and render methods
mcts_agent = MctsAgent(agent)
mcts_env = MctsEnv(env)
mcts = PyMcts()

while not env.done():
    node = mcts.run(mcts_env, mcts_agent, num_steps=100)
    most_visits = -1
    for action, visits in node.edge_visits().items():
        if visits > most_visits:
            most_visits = visits
            best_action = action
        env.step(best_action)
```

# TODO
- Make things completely independent from `tilers_core`.