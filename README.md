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
This directory contains shim code so that Python processes can handle inference requests when MCTS is running in rust. This lets neural networks defined in, say, PyTorch communicate with the rust MCTS code.

```python
# Example handler code
import numpy as np
from mcts_ipc import PyArena

name = 'example_mcts'  # Needs to match name that gatherers use
# num_slots should almost always just be the default
# num_handlers should be close to the number of GPUs used for inference
num_slots = 2048
num_handlers = 2
arena = PyArena(name, num_slots=num_slots, num_handlers=num_handlers)

# Assume this is handler 0. Launch another similar process with id = 1.
handler_id = 0
while True:
    # Get next ready slot of data
    sv = arena.pop_ready_view(handler=handler_id)
    # input information
    action_mask = np.asarray(sv.action_mask())  # shape (b, NUM_ACTIONS)
    done = np.sum(np.asarray(sv.obj0())) == 0   # shape (b, MAX_OBJ0)
    # output information
    priors = np.asarray(sv.priors())  # shape (b, NUM_ACTIONS)
    values = np.asarray(sv.values())  # shape (b,)
    priors[:] = 0.0
    # Doing "inference"...
    # Otherwise put PyTorch inference code here
    b, n = priors.shape
    for j in range(b):
        norm = np.sum(action_mask[j])
        for k in range(n):
            if action_mask[j, k]:
                priors[j, k] = 1.0 / norm
        if done:
            values[j] = 1.0
        else:
            values[j] = -1.0
    # Let gatherers know that inference is done
    sv.mark_done()
```


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
