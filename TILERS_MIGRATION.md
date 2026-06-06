# `mcts` ↔ `tilers` API migration

*Companion to `~/Documents/research/tile/BOARD_REDESIGN.md` (the Python
agent's board migration).  This doc covers the **Rust** consumer:
`mcts-tilers` depends on `tilers` as a path crate
(`mcts-tilers/Cargo.toml: tilers = { path = "../../tilers" }`), so the
`pauli`-branch tilers rewrite breaks it at compile time.*

*Written 2026-06-06 against tilers HEAD `352a2e1` (QubitId enum).
Implementation on branch `tilers-action-board-migration`.*

## Progress (2026-06-06)

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Action-API bridge (`rl`) | **done** | `environment.rs`, `evaluator.rs`, `gatherer.rs`, `gather.rs` step/valid_actions/select_action route through `rl::encode`/`decode`; `QubitId::from`. mcts stays integer-native (`Act = u16`). |
| Phase 2 — width `1+5N → 1+6N` | **done** | `NUM_ACTIONS = 1 + 6*MAX_ANCILLAS` (251→301); IPC slot/stride pick it up automatically. **Python NN output head still 251 — see below.** |
| Phase 3 — observation → `rl::board` | **done (Rust)** | `TilersObs` carries `Vec<Vec<BoardCell>>` from `construct_board`; `slot.rs`/`py_ipc.rs` pack/expose the 10×i16 board (replacing placement/objectives/last_dir); `gatherer.rs` records + serializes the board JSON; `py_mcts.rs` `obs_to_pydict` emits `board`. |
| One tilers change | **done** | `BoardCell` derives `Hash` (so `TilersObs: Hash` for the mcts `Obs` bound). |

**Rust: green.** Workspace `cargo test` = 18 (mcts-core) + 28 (mcts-tilers) pass; `cargo check --features python` clean. (`cargo build --features python` link-fails standalone — that's the normal pyo3 extension-module limitation; build via `maturin`.)

### Remaining — Python NN side (coupled ML rewrite, NOT done)

The Rust IPC now exposes `sv.board()` (shape `(b, BOARD_MAX)` i16, reshape
to `(b, LOOKAHEAD_MAX, GRID_MAX, CELL_FIELDS)`), `sv.num_layers()`,
`sv.h/w/num_ancillas()`, `sv.action_mask()` — and **no longer**
`placement()`/`objectives()`/`num_qubits()`/`num_objectives()`/
`last_dir_vertical()`. So:

1. **`handler.py`**: delete `build_boards` (it reconstructs the *old*
   4-channel board from placement/objectives bytes that no longer exist);
   read `sv.board()` directly and reshape — it is already the 10-channel
   board.
2. **`Agent` model**: replace the 4-channel embedding + `1+5N` head with
   the 10-channel, 84-dim, `1+6N` board agent — converge on
   `tile/agent.py` (`BOARD_REDESIGN.md §4`). Old checkpoints invalidate
   (new wire-band logits + new tables).
3. **Dataset reader** (pretraining): parse the gatherer's new record JSON
   — `board` (nested `[layer][cell][10]`) instead of
   `placement`/`objectives`/`last_dirs`; mask length `1+6N`.

This is `BOARD_REDESIGN.md` Phases 1–3 on the mcts Python side; it depends
on tile's agent and can't be validated without the training loop, so it's
left as the next workstream.

## 0. TL;DR — three changes

1. **Action API (hard break, compile error).** tilers is now
   *Action-native*: `valid_actions() -> Vec<Action>` and `step(Action)`.
   The flat `usize` encoding moved out of `Environment` into the
   `tilers::rl` adapter (`rl::encode` / `rl::decode` / `rl::action_mask`;
   `num_actions()` stays on the env). `QubitId` is now a sum-type enum,
   not an `i32` newtype. mcts hand-rolls the integer action space and
   calls `step(usize)` — every one of those sites breaks.
   **Resolution: keep mcts integer-native; bridge at the `TilersEnv`
   boundary via `tilers::rl`** (see §3 — the conversion is *intrinsic*
   to a fixed-head NN + integer IPC, so it can't be removed, only placed
   well; "going Action-native" is a net loss here).

2. **Action-space width (latent correctness bug, same as tile's) — must
   change.** mcts assumes **`1 + 5N`** actions (`NUM_ACTIONS = 251 = 1 +
   5·50`). tilers has been **`1 + 6N`** since the `pauli` branch added
   the **wire** band (`1 auto + N cultivate + N wire + 4N move`). mcts is
   under-sized by the wire band — the same `5N+1 → 6N+1` bump tile did.

3. **Observation → adopt `tilers::rl::board` — should update.** tile
   migrated its per-cell observation to the 10-channel `BoardCell`
   (`tilers::rl::board`). mcts still ships its own flat `TilersObs`
   (raw placement + objectives + 1-bit `last_dir_vertical`). Migrate
   mcts onto `rl::board::construct_board` so the two projects share one
   observation contract (and a model/dataset can be shared). See §4.

## 1. What changed in `tilers` (the surface mcts touches)

| concern | old (mcts assumes) | new (tilers HEAD) |
|---|---|---|
| `Environment::valid_actions()` | `Vec<usize>` (raw action ids) | `Vec<Action>` (the `enums::Action` sum type) |
| `Environment::step(_)` | `step(usize)` | `step(Action)` |
| `encode/decode/num_actions/action_mask` | methods on `Environment` (impl/implicit) | `num_actions()` kept on env; `encode`/`decode`/`action_mask`/`opposite` moved to `tilers::rl` free fns over `&Environment` |
| `Action` type | n/a (mcts had only its own `u16`) | `enums::Action = AutoExecute \| Cultivate(QubitId) \| Wire(QubitId) \| Move(QubitId, Direction)`; N-independent, plan-stable |
| `QubitId` | `struct QubitId(pub i32)` — `QubitId(x)`, `.0` | `enum QubitId { Data(u32), Ancilla(u32), Placeholder(u32), AnyAncilla }`; construct via `QubitId::from(i32)` / variants, read via `.as_i32()` |
| action-space width | `1 + 5N` | `1 + 6N` (wire band `[N+1 ..= 2N]` is new) |

The integer↔Action mapping is now owned **solely** by `tilers::rl` —
mcts should call it rather than re-deriving the layout (which is what
makes mcts robust to any future layout change).

## 2. What breaks in `mcts-tilers` (file:line inventory)

`mcts-tilers/src/environment.rs`:
- **L51 `let valid = self.inner.valid_actions();`** → `valid: Vec<Action>`.
- **L54 `for &a in &valid { if a < num_actions { action_mask[a] = true; } }`** —
  `a` is an `Action`, not a `usize`: can't compare or index. Replace
  with `rl::encode` per action (or just `rl::action_mask`).
- **L59 `let qid = QubitId(-((a + 1) as i32));`** — tuple construction on
  an enum. → `QubitId::from(-((a + 1) as i32))` or `QubitId::Ancilla((a+1) as u32)`.
- **L84 `self.inner.step(action as usize)`** — `step` takes `Action`. →
  `self.inner.step(rl::decode(&self.inner, action as usize).expect(...))`.
- **L96-97 `valid_actions().into_iter().map(|a| a as u16)`** — `a` is
  `Action`. → `.map(|a| rl::encode(&self.inner, a) as u16)`.

`mcts-tilers/src/evaluator.rs`:
- **L102 `let valid_actions = tilers_env.inner.valid_actions();`** →
  `Vec<Action>` now.
- **L117-120** `max_by_key(|&&a| root.edge_visits.get(&(a as Action)) …) … as Action`
  — `a` is a tilers `Action`, but `crate::constants::Action` is `u16`.
  Encode: `rl::encode(&tilers_env.inner, a) as Action`.
- **L123 `tilers_env.inner.step(action as usize)`** → decode as above.

`mcts-tilers/src/constants.rs`:
- **`NUM_ACTIONS = 251` (= `1 + 5·MAX_ANCILLAS`)** → **`1 + 6·MAX_ANCILLAS = 301`**.
- Comment "5 actions per ancilla" → 6 (auto + cultivate + wire + 4 move).

Ripple from `NUM_ACTIONS` (it's baked into the shared-memory IPC layout):
- `mcts-tilers/src/lib.rs` — exports `NUM_ACTIONS` to Python (module const).
- `mcts-tilers/src/client.rs` — `offset = i * NUM_ACTIONS` (per-slot mask stride).
- `mcts-tilers/src/py_ipc.rs` — action-mask array shape `(b, NUM_ACTIONS)`.
- **Python NN** (`handler.py:305/350`, the model's output head) reads a
  `(b, NUM_ACTIONS)` mask and emits `NUM_ACTIONS` logits — the head must
  grow `251 → 301`, and any saved checkpoint is invalidated (random init
  for the 50 new wire-band logits, exactly like tile's output-head bump).

## 3. Proposed changes (Action API)

### Can we avoid the `rl` dependency? No — and it shouldn't be avoided.

The integer↔`Action` conversion is **intrinsic** to mcts's design, not
an artifact we can engineer away:

- The NN policy head is a **fixed-width logit vector**; the IPC shares a
  **fixed-width integer action mask**; priors come back as
  `HashMap<E::Act, f32>` keyed by mask index
  (`mcts-tilers/src/client.rs:86`, `a as Action`). All three require a
  stable *integer* per action.
- So an `Action ↔ usize` map must exist somewhere. The only choices are
  *where* it lives and *whether the layout is duplicated*.

`tilers::Action` *does* satisfy mcts-core's `Act: Copy + Debug + Send +
Hash + Eq` bound (no `Ord` needed — the tree uses `HashMap`), so a fully
**Action-native** mcts compiles. But it's the wrong move here: the
inference client (`client.rs`) holds only a shared-memory **slot**, not
a live `Environment`, so it could not call `rl::decode(&env, …)` to key
priors by `Action` — it would need `num_ancillas` plumbed into every IPC
slot plus a `decode_from_n` helper tilers doesn't expose. Action-native
**relocates** the conversion from 3 clean env-side calls into the IPC
hot path. Net loss.

**Decision:** keep mcts integer-native (`Act = u16`) and treat
`tilers::rl` as the **intended boundary API**, not a shim to remove. The
`TilersEnv` always has the live `Environment` in hand, so the three
conversions sit naturally there. The *only* alternative that drops the
`tilers::rl` import is re-hardcoding the `6N+1` layout inside mcts — the
exact drift the env-side refactor centralized away (and the cause of the
`5N` vs `6N` bug in §2). Don't.

### The bridge (3 call sites)

Add `use tilers::rl;` and:

```rust
// environment.rs — build_obs action_mask (replaces L51-54)
let mask = rl::action_mask(&self.inner);            // Vec<bool>, len = env.num_actions()
let mut action_mask = vec![false; NUM_ACTIONS];
for (id, &on) in mask.iter().enumerate() {
    if on && id < NUM_ACTIONS { action_mask[id] = true; }
}

// environment.rs — step (replaces L84)
fn step(&mut self, action: Self::Act) {
    let a = rl::decode(&self.inner, action as usize)
        .expect("MCTS passed an invalid action id");
    self.inner.step(a).expect("MCTS passed an invalid action");
    self.inner.finish_cultivating(None, None);
}

// environment.rs — valid_actions (replaces L96-97)
fn valid_actions(&self) -> Vec<Self::Act> {
    self.inner.valid_actions().into_iter()
        .map(|a| rl::encode(&self.inner, a) as u16)
        .collect()
}

// environment.rs L59 — ancilla id
let qid = QubitId::from(-((a + 1) as i32));  // or QubitId::Ancilla((a + 1) as u32)

// evaluator.rs L117-123
let action = valid_actions.iter()
    .map(|&a| rl::encode(&tilers_env.inner, a))
    .max_by_key(|&id| *root.edge_visits.get(&(id as Action)).unwrap_or(&0))
    .unwrap() as Action;
let a = rl::decode(&tilers_env.inner, action as usize).unwrap();
let _ = tilers_env.inner.step(a);
```

Note `rl::action_mask(&env)` already does the encode-each-valid-action
loop internally, so prefer it over re-deriving. (`rl` is `pub` in the
tilers crate; no FFI needed for the Rust path.)

**Then the width bump** (`NUM_ACTIONS 251 → 301`) + the Python head
`251 → 301` + checkpoint re-init. This is the same change as
`BOARD_REDESIGN.md §4 "Action space"`; coordinate the IPC arena
re-sizing (client.rs/py_ipc.rs) with the Python handler in one go so the
shared-memory stride matches on both sides.

## 4. Observation → adopt `tilers::rl::board`

mcts's `TilersObs` (raw `placement` + `objectives` + flat `action_mask`
+ 1-bit `last_dir_vertical`) is the **pre-redesign** flat encoding tile
already replaced with the 10-channel `BoardCell`. It has the same defects
catalogued in `BOARD_REDESIGN.md §2` (orphaned opcode/orientation tokens
now that gates are PauliProducts and lifecycle state left `Orientation`),
so it should be migrated, not preserved.

**Plan: replace `build_obs`'s placement/objective packing with the shared
Rust board core.** tilers exposes `tilers::rl::board::{construct_board,
build_layer, BoardCell}` (10× i16 per cell; schema = `BOARD_REDESIGN.md
§3`) — reachable directly from the Rust path-dep, no FFI needed.

- `TilersObs` carries `board: Vec<BoardCell>` (or `Vec<Vec<BoardCell>>`
  per lookahead layer) from `construct_board(&self.inner, lookahead)`
  instead of `placement` + `objectives`.
- The IPC slot widens to `CELL_FIELDS (=10) × i16 = 20 bytes/cell`
  (matching tile's wire format), replacing the `QUBIT_SIZE`/
  `OBJECTIVE_SIZE` packings in `constants.rs` + `py_ipc.rs`.
- The Python model grows to the 84-dim/cell, 10-embedding-table stack
  from `BOARD_REDESIGN.md §4`. Reusing tile's `agent.py` board embedding
  is the natural convergence point — mcts and tile then share one
  observation contract and can share a model / dataset.
- `last_dir_vertical` (1-bit) is subsumed by the board's 4-way
  `last_move_dir` channel — drop it from `TilersObs`.

This is larger than §2/§3 and couples to the Python model, so it's the
last phase (§6 Phase 3) — but it is **planned work, not optional**:
keeping the flat `TilersObs` leaves mcts permanently diverged from tile's
agent and unable to share datasets/checkpoints.

## 5. What needs to change in `tilers` (this repo)

For the **Rust** mcts path: **nothing is strictly required** — `rl`
(`encode`/`decode`/`num_actions`/`action_mask`) and `rl::board` are all
`pub` and reachable from the path-dep crate.

Loose ends (none block mcts, listed for completeness):
- **Board PyO3 wrappers + Phase-0 predicates** are still pending for the
  **Python** (tile) path — `BOARD_REDESIGN.md §7 Phase 0`: `PyEnvironment`
  has no `is_wire` / `is_resource` / `is_cultivating` / `is_doubling_ancilla`
  / `doubled_target_owning`, and `PyObjective` has no `pauli_hub_ancilla`.
  Irrelevant to mcts (Rust) but needed before tile's agent runs end-to-end.
- **Dangling doc refs.** `ACTION_ENCODING.md` and
  `NUM_ACTIONS_ACTION_MASK_BUG.md` were deleted in the Action refactor but
  are still cited by `tile/BOARD_REDESIGN.md` and `tilers/src/rl/board.rs`
  comments. Repoint to `tilers::rl` / this doc.

## 6. Phasing

- **Phase 1 — compile against new tilers (~1-2 h).** §3 Action-API
  bridge in `environment.rs` + `evaluator.rs` + `QubitId::from`. Keep
  `NUM_ACTIONS = 251` for the moment (works for `N ≤ 41`, where
  `6N+1 ≤ 251`). `cargo test -p mcts-tilers`. *Compiles + correct for
  small grids; wire-band actions silently dropped for large N.*
- **Phase 2 — action-space width (coordinated Rust+Python).** Bump
  `NUM_ACTIONS → 301`, re-size the IPC arena (client.rs / py_ipc.rs),
  grow the Python output head `251 → 301`, version-stamp + invalidate
  checkpoints (mirror `BOARD_REDESIGN.md §4`). Now wire actions are
  representable up to `MAX_ANCILLAS = 50`.
- **Phase 3 — observation alignment (couples to the Python model).**
  §4: replace flat `TilersObs` with `rl::board::construct_board`; widen
  the IPC slot to `10×i16`/cell; grow the Python model to the
  10-table/84-dim board embedding (converge on tile's `agent.py`).
  Larger and gated on the model work, but planned — not optional.

## 7. Out of scope

- MCTS search itself (tree, rollouts, PUCT) — unaffected; only the
  env-boundary action plumbing and the action-space width change.
- The `new_env_api_design.md` 7N mode-indexed action redesign — a later,
  separate workstream (would bump the width again, `6N+1 → 7N`).
