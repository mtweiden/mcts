# `mcts`

A Monte Carlo Tree Search (MCTS) implementation designed to support **batched neural inference** via:

- a pure Rust boundary (`mcts-core`)
- a **shared-memory IPC** layer (Rust <-> Python) for high-throughput inference (`mcts-core` + Rust/Python shims)
- a concrete example integration with the `tilers` environment (`mcts-tilers`)
- optional **Python bindings** for running MCTS directly from Python (`mcts-tilers` with `python` feature)

This repo is structured around a generic MCTS engine that can be ported to custom environments, plus an IPC mechanism that lets you keep your model in Python (e.g. PyTorch) while MCTS runs in Rust.

---

## Repository layout

- `mcts-core/`  
  Core MCTS implementation + environment/inference traits + shared-memory IPC primitives.

- `mcts-tilers/`  
  A concrete environment + slot definition + IPC client for the `tilers` project, plus optional Python bindings. Note that this requires `tilers` as a dependency, which is easiest to get by just cloning from my repo.

- `mcts-tilers/python/`  
  Python-side utilities (example handler loop, `.pyi` type hints, package init).

---

## `mcts-core`

### What it provides

- `Environment` trait (`mcts-core/src/environment.rs`)
- `InferenceClient<E>` trait (`mcts-core/src/inference.rs`)
- `MCTS<E>` implementation (`mcts-core/src/mcts.rs`)
- Shared-memory IPC arena + slot queues (`mcts-core/src/ipc_core.rs`)

### Environment trait

To run MCTS on a new domain, implement:

```rust
pub trait Environment: Clone {
    type Act: Act;
    type Obs: Obs;

    fn step(&mut self, action: Self::Act);
    fn done(&self) -> bool;
    fn observation(&self) -> Self::Obs;
    fn valid_actions(&self) -> Vec<Self::Act>;
    fn hash(&self) -> u64;
    fn render(&self) -> String;
}
```

### Inference boundary
- MCTS is model-agnostic. It queries an inference client for:
    - action priors: `HashMap<Act, f32>`
    - state values: `f32`

```rust
pub trait InferenceClient<E: Environment> {
    fn infer(&self, observations: &[E::Obs])
        -> anyhow::Result<(Vec<HashMap<E::Act, f32>>, Vec<f32>)>;
}
```

### Basic usage (Rust)
At a high level:
```rust
let mut mcts: MCTS<MyEnv> = MCTS::new(1.0, 8);
let root = mcts.run(&env, &client, 10_000);

let mut best_action = None;
let mut most_visits = 0usize;
for (a, v) in root.edge_visits.iter() {
    if *v > most_visits {
        most_visits = *v;
        best_action = Some(*a);
    }
}
```
If you’re running a multi-step episode, you can preserve search state between steps:

```rust
mcts.advance_root(action_taken);
```
This advances the tree root to the selected child (subtree reuse). If the child is unknown, the tree root resets.

### Shared-memory IPC design (Rust ↔ Python)

The IPC layer is built around:

- `Arena<S>`: a file-backed mmap region containing a header + an array of slots (S) ring queues in shared memory
- `SlotInit`: a trait implemented by slot structs so the arena can manage slot lifecycle

Slots have a simple state machine:

`SLOT_FREE`: slot available
`SLOT_READY`: inputs written; ready for handler
`SLOT_WAITING`: (reserved; optional usage)
`SLOT_DONE`: outputs written; ready to be read
A typical flow:

- Rust MCTS acquires a free slot
- Rust writes a batch of observations into the slot
- Rust pushes the slot index into a handler queue (round-robin)
- Python handler pops ready slots, runs inference, writes priors/values, marks done
- Rust waits for done, reads outputs, releases slot back to free queue

## `mcts-tilers`

This crate provides:

- a concrete `TilersEnv` implementing `mcts-core::Environment`
- a concrete shared-memory slot type `TilersSlot` that packs `TilersObs` efficiently
- a Rust `InferenceClient` implementation `TilersIpcClient` that speaks via shared memory
- optional Python bindings (--features python) that expose:
    - `PyArena`, `PySlotView` (IPC)
    - `PyMcts`, `MctsAgent`, `MctsNode` (MCTS from Python)
    - Running the Rust `gatherer` (tilers integration)

`mcts-tilers/src/bin/gather.rs` runs self-play / data gathering using:
- Rust `MCTS`
- shared-memory inference
- the `tilers` environment
- It expects one or more inference handlers to be running (typically in Python).

### Installation
```bash
# In top level directory
maturin develop --release --features python
```

#### Example:

```bash
cargo run --bin gather --release
```
The arena name used by the gatherer is derived from:

```rust
let arena_name = format!("mcts_{}_{}", num_slots, num_handlers);
```
__So handlers must connect to the same name.__

A reference Python handler implementation exists at: `mcts-tilers/python/mcts_tilers/handler.py`
It works by:
- popping ready slots from PyArena
- batching multiple slots together (up to MAX_SLOTS_PER_BATCH or BATCH_TIMEOUT)
- unpacking observations
- runing model inference (example uses PyTorch)
- writing priors/values back and marks the slots done

#### Example (typical pattern):

```python
from mcts_tilers import PyArena

arena_name = "mcts_2048_1"
arena = PyArena(arena_name, num_slots=2048, num_handlers=1)

handler_id = 0
while True:
    sv = arena.pop_ready_view(handler=handler_id, clear_outputs=True)
    # read inputs from sv.*
    # write outputs via sv.write_priors_values(...)
    sv.mark_done()
```

### Python: running MCTS directly (bindings)

If you build mcts-tilers with the python feature, it exposes:

PyMcts: runs Rust MCTS
MctsAgent: wraps a Python object that provides infer(obs_list) -> (priors, values)
MctsNode: the returned root node with visits/priors/value
The expected Python agent interface is:

```python
class Agent:
    def infer(self, obs: list[dict]) -> tuple[list[dict[int, float]], list[float]]:
        ...
```
#### Example:

```python
from mcts_tilers import PyMcts, MctsAgent
from tilers.env import PyEnvironment

agent = Agent()
mcts = PyMcts(terminal_value=1.0, batch_size=8)
wrapped = MctsAgent(agent)

env = PyEnvironment(...)  # from tilers
node = mcts.run(env, wrapped, num_steps=10_000)

best_action = max(node.edge_visits().items(), key=lambda kv: kv[1])[0]
mcts.advance_root(best_action)
```
