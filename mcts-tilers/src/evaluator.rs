use rusqlite::{params, Connection};
use serde_json::json;

use mcts_core::ipc_core::Arena;
use mcts_core::inference::InferenceClient;
use mcts_core::mcts::MCTS;

use tilers::env::Environment;
use tilers::rl;
use tilers::solver::Solver;

use crate::constants::*;
use crate::environment::TilersEnv;
use crate::slot::TilersSlot;
use crate::client::TilersIpcClient;

// =============================================================================
// Configuration
// =============================================================================
const NUM_SLOTS: usize = 2048;
const LOOKAHEAD: usize = DEFAULT_LOOKAHEAD;

// =============================================================================
// HoldoutEnvironment
// =============================================================================
pub struct HoldoutEnvironment {
    environment_id: i64,
    json: String,
    num_objectives: i64,
    difficulty_bin: i64,
    action_count: i64,
}

/// Load holdout environments from the database, optionally filtered to those
/// in difficulty bin `<= max_difficulty_bin` (the solver-action-count bin set
/// by init_db.py; see ACTION_BIN_EDGES), and round-robin sliced across nodes
/// by (node_idx, num_nodes). Single-node callers pass (0, 1) and get every
/// row. Multi-node callers pass distinct node_idx values and ROW_NUMBER over
/// (difficulty_bin ASC, environment_id ASC) distributes envs evenly across
/// difficulty bins — so each node gets a balanced mix of easy and hard envs
/// rather than a contiguous block.
///
/// Difficulty is keyed on the heuristic solver's action count (difficulty_bin),
/// not num_objectives: two envs with the same objective count can need wildly
/// different action budgets, and action count is what actually predicts eval
/// cost (max_actions in evaluate_single scales with the solver solution length).
fn load_holdout_environments(
    conn: &Connection,
    min_difficulty_bin: Option<i64>,
    max_difficulty_bin: Option<i64>,
    node_idx: i64,
    num_nodes: i64,
) -> Result<Vec<HoldoutEnvironment>, rusqlite::Error> {
    // Optional [min, max] difficulty window: the orchestrator's frontier
    // window (frontier bin +/- 1) skips both the mastered easy bins (all
    // draws) and the deep bins (slow episodes, both-fail) for eval speed.
    let mut clauses: Vec<&str> = Vec::new();
    if min_difficulty_bin.is_some() {
        clauses.push("difficulty_bin >= ?");
    }
    if max_difficulty_bin.is_some() {
        clauses.push("difficulty_bin <= ?");
    }
    let max_filter = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };
    // Slice via ROW_NUMBER so every node gets a balanced mix across the
    // difficulty bins. (rn - 1) % num_nodes = node_idx is the round-robin.
    let sql = format!(
        "WITH ordered AS (
            SELECT environment_id, json, num_objectives, difficulty_bin, action_count,
                   ROW_NUMBER() OVER (ORDER BY difficulty_bin ASC, environment_id ASC) AS rn
            FROM environments
            {}
         )
         SELECT environment_id, json, num_objectives, difficulty_bin, action_count
         FROM ordered
         WHERE (rn - 1) % ? = ?
         ORDER BY difficulty_bin ASC",
        max_filter
    );

    let mut bound: Vec<i64> = Vec::new();
    if let Some(m) = min_difficulty_bin {
        bound.push(m);
    }
    if let Some(m) = max_difficulty_bin {
        bound.push(m);
    }
    bound.push(num_nodes);
    bound.push(node_idx);

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(bound.iter()), |row| {
        Ok(HoldoutEnvironment {
            environment_id: row.get(0)?,
            json: row.get(1)?,
            num_objectives: row.get(2)?,
            difficulty_bin: row.get(3)?,
            action_count: row.get(4)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

// =============================================================================
// Evaluator
// =============================================================================
pub struct Evaluator {
    mcts_steps: usize,
    c_puct: f32,
    /// Saturation temperature for the terminal reward; see Gatherer for the
    /// full description. Kept identical to the gatherer's value during a
    /// given run so the evaluator's terminal scalars are comparable to the
    /// ones the trained value head saw at gather time.
    reward_saturation_temperature: f32,
    /// Per-episode action budget as a multiple of the heuristic solution
    /// length (`max_actions = ceil(reference_actions * multiplier)`). 1.2 is
    /// the historical default; raising it lets a sub-heuristic policy finish
    /// (earning a graded reward) instead of being truncated to the -1 floor.
    max_action_multiplier: f32,
    /// Root prior temperature for eval search: P'(c) ∝ P(c)^(1/T) at the
    /// root only (mcts-core root_softmax_temp). T < 1 SHARPENS. 0.6 rescued
    /// 14/24 -> 22/24 of a flat-prior candidate's eval failures (2026-07-09
    /// probe) — candidates trained on honest post-fix targets carry flatter
    /// priors at contested states than the stale-sharp incumbent, and greedy
    /// PUCT needs concentration at decision points. Eval-only: gather keeps
    /// Wu's 1.03 flattening so training targets stay exploratory.
    root_softmax_temp: f32,
}

impl Evaluator {
    pub fn new(
        mcts_steps: usize,
        c_puct: f32,
        reward_saturation_temperature: f32,
        max_action_multiplier: f32,
        root_softmax_temp: f32,
    ) -> Self {
        Self {
            mcts_steps,
            c_puct,
            reward_saturation_temperature,
            max_action_multiplier,
            root_softmax_temp,
        }
    }

    /// Solve the environment using the heuristic solver to obtain the
    /// reference depth used in the terminal evaluator. Returns `None` if
    /// the heuristic planner exhausts its iteration budget — these envs
    /// are skipped by `evaluate_single` so eval continues instead of
    /// panicking.
    fn solve_with_heuristic(&self, env: &Environment) -> Option<f32> {
        let mut solved_env = env.clone();
        let solver = Solver::new();
        match solver.solve(&mut solved_env, true) {
            Ok(_) => Some(solved_env.depth(true, true) as f32),
            Err(_) => None,
        }
    }

    /// Run MCTS on a single environment and return the action sequence taken,
    /// the final solution depth, and a flag indicating whether the heuristic
    /// baseline could be computed. The third element is `true` when the env
    /// was skipped because the heuristic planner exhausted — in that case the
    /// first two elements are empty / `None` and the caller should record the
    /// env as unsolvable rather than as "agent did not finish".
    fn evaluate_single(
        &self,
        env: &Environment,
        client: &dyn InferenceClient<TilersEnv>,
    ) -> (Vec<Action>, Option<f32>, i64, Option<f32>, bool, Option<(usize, usize)>) {
        let mut mcts: MCTS<TilersEnv> = MCTS::new(8);
        mcts.root_softmax_temp = self.root_softmax_temp;
        let mut tilers_env = TilersEnv::new(env.clone(), LOOKAHEAD);
        tilers_env.inner.set_cultivation_time(10);

        let reference_depth = match self.solve_with_heuristic(&tilers_env.inner) {
            Some(d) => d,
            None => return (Vec::new(), None, 0, None, true, None),
        };
        let temperature = self.reward_saturation_temperature;

        // Shared done-scorer (reward.rs); same formulation as the Gatherer's
        // terminal evaluator. Eval and gather must agree here so the value
        // head's terminal targets at training time match the scalars MCTS
        // sees at eval time.
        let terminal_evaluator = |e: &TilersEnv| -> f32 {
            if !e.inner.done() {
                crate::reward::NOT_DONE_SCORE
            } else {
                let d = e.inner.depth(true, true) as f32;
                crate::reward::done_score(reference_depth, d, temperature)
            }
        };

        let solver = Solver::new();
        let mut tmp_env = env.clone();
        let max_actions = match solver.solve(&mut tmp_env, false) {
            Ok(sol) => (sol.len() as f32 * self.max_action_multiplier).ceil() as usize,
            Err(_) => 1000,
        };

        let mut actions_taken: Vec<Action> = Vec::new();

        for _ in 0..max_actions {
            if tilers_env.inner.done() {
                break;
            }

            let valid_actions = tilers_env.inner.valid_actions();
            if valid_actions.is_empty() {
                break;
            }

            let root = mcts.run(
                &tilers_env,
                client,
                self.mcts_steps,
                self.c_puct,
                &terminal_evaluator,
                // forced_playouts=false. Eval wants maximum move strength —
                // forced playouts (Wu 2020 §3.2) deliberately routes some
                // playouts to low-prior root actions for training-data
                // exploration, which is correct during gather but skews
                // edge_visits during eval and weakens the agent's argmax
                // action selection below. See mcts-core/src/mcts.rs:121–140
                // for the contract.
                false,
            );

            // Greedy: pick the action with the most visits.  `valid_actions`
            // is `Vec<tilers::Action>`; encode to the flat `u16` id space the
            // tree/visits are keyed by.
            // `valid_actions` is `Vec<tilers::Action>` (new tilers API);
            // encode each to the u16 id space that the tree/visits are
            // keyed by, then look up against main's dense-Vec
            // `edge_visits`.
            let action = valid_actions
                .iter()
                .map(|&a| rl::encode(&tilers_env.inner, a).expect("valid_actions ids always encode") as Action)
                .max_by_key(|&id| root.edge_visits.get(id as usize).copied().unwrap_or(0))
                .unwrap();

            actions_taken.push(action);
            let a = rl::decode(&tilers_env.inner, action as usize)
                .expect("evaluator produced an invalid action id");
            let _ = tilers_env.inner.step(a);
            tilers_env.inner.finish_cultivating(None, None);
            mcts.advance_root(action);
        }

        let done = tilers_env.inner.done();
        let agent_depth = tilers_env.inner.depth(true, true) as f32;
        let solution_depth = if done { Some(agent_depth) } else { None };

        // Continuous eval reward + objectives satisfied, mirroring the
        // gatherer's terminal scoring EXACTLY so eval and gather agree on the
        // scalar (gatherer.rs:529-559): done -> tanh of depth-margin vs the
        // heuristic; not-done -> HER partial credit by solving the achieved
        // sub-goal; floor -> -1. The continuous reward is the promotion signal:
        // dense and magnitude-aware, so a paired z-test over a large holdout
        // detects the small per-iteration gains that binary done()/depth and
        // integer achieved-count comparisons are blind to.
        // The recorded eval_reward is the promotion-test currency: margin
        // scale, always. (A banded variant was measured to make the reward
        // z-test a noisy duplicate of the frontier gate — see
        // DONE_REWARD_REMAP.md amendment 1 — and was removed.)
        let temperature = self.reward_saturation_temperature;
        let (achieved_objectives, eval_reward, factor_prog): (i64, Option<f32>, Option<(usize, usize)>) = if done {
            let ratio = (reference_depth - agent_depth) / (reference_depth + 1e-6);
            (env.num_objectives() as i64, Some((ratio / temperature).tanh()), None)
        } else {
            // Unfinished: grade by FACTOR-level progress against the full goal
            // (merged / total factors), NOT HER's objective-collapse relabel. HER
            // scored both agents ~equally on hard envs (it relabels to the achieved
            // sub-goal, ~1-2 objectives regardless of true progress), so the paired
            // candidate−incumbent diff vanished and the promotion z-test was BLIND
            // to the frontier — the same collapse we fixed in gather. Factor
            // granularity is dense and monotonic, so a candidate that merges more
            // factors than the incumbent yields a real positive diff. No cusp cap
            // here (unlike gather's shaping reward): eval wants discrimination
            // across the WHOLE progress range, not a curriculum target. temperature
            // is unused on this path (linear fraction, already in [-1, 0]).
            let _ = temperature;
            let (merged, total) = env.factor_progress(&tilers_env.inner);
            let achieved = (env.num_objectives() as i64
                - tilers_env.inner.num_objectives() as i64)
                .max(0);
            if merged == 0 || total == 0 {
                (achieved, Some(-1.0), Some((merged, total)))
            } else {
                (achieved, Some(merged as f32 / total as f32 - 1.0), Some((merged, total)))
            }
        };

        (actions_taken, solution_depth, achieved_objectives, eval_reward, false, factor_prog)
    }

    /// Open the arena (which Python has already populated with live handlers)
    /// and evaluate a single agent against all holdout environments, writing
    /// results to the solutions table. Skips any environment the agent has
    /// already attempted so evaluation is safe to resume after a crash.
    pub fn evaluate_agent_with_client(
        &self,
        agent_id: i64,
        environments: &[HoldoutEnvironment],
        client: &dyn InferenceClient<TilersEnv>,
        conn: &Connection,
    ) {
        for holdout in environments {
            let already_attempted: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM solutions \
                     WHERE agent_id = ?1 AND environment_id = ?2",
                    params![agent_id, holdout.environment_id],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;

            if already_attempted {
                println!(
                    "[Evaluator] Skipping environment {} for agent {} (already attempted).",
                    holdout.environment_id, agent_id
                );
                continue;
            }

            let env = match Environment::from_json(&holdout.json) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!(
                        "[Evaluator] Failed to deserialise environment {}: {:?}",
                        holdout.environment_id, e
                    );
                    continue;
                }
            };

            let (actions, solution_depth, achieved_objectives, eval_reward, heuristic_unsolvable, factor_prog) =
                self.evaluate_single(&env, client);

            // Unsolvable envs (heuristic baseline can't be computed) get a
            // sentinel string in the actions column instead of a JSON array,
            // so the Python side can pull them out by SQL: solutions WHERE
            // actions = '"heuristic_unsolvable"'. The row still records the
            // attempt so resume-after-crash skips it.
            let actions_json = if heuristic_unsolvable {
                json!("heuristic_unsolvable").to_string()
            } else {
                json!(actions).to_string()
            };
            let attempted_at = chrono::Utc::now().to_rfc3339();

            conn.execute(
                "INSERT OR IGNORE INTO solutions
                    (agent_id, environment_id, actions, solution_depth, achieved_objectives, eval_reward, attempted_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    agent_id,
                    holdout.environment_id,
                    actions_json,
                    solution_depth,
                    achieved_objectives,
                    eval_reward,
                    attempted_at,
                ],
            )
            .expect("Failed to insert solution");

            if heuristic_unsolvable {
                println!(
                    "[Evaluator] Agent {} | Env {} (bin {}, {} actions, {} obj) | UNSOLVABLE (heuristic exhausted).",
                    agent_id, holdout.environment_id, holdout.difficulty_bin,
                    holdout.action_count, holdout.num_objectives
                );
            } else {
                match solution_depth {
                    Some(d) => println!(
                        "[Evaluator] Agent {} | Env {} (bin {}, {} actions, {} obj) | depth = {:.1}",
                        agent_id, holdout.environment_id, holdout.difficulty_bin,
                        holdout.action_count, holdout.num_objectives, d
                    ),
                    None => {
                        // Failure case: show factor-level progress — the quantity
                        // the new eval_reward grades — so eval logs reveal HOW far
                        // the agent got, not just that it failed.
                        let (m, t) = factor_prog.unwrap_or((0, 0));
                        println!(
                            "[Evaluator] Agent {} | Env {} (bin {}, {} actions, {} obj) | did not finish; factors merged {}/{} ({:.0}%).",
                            agent_id, holdout.environment_id, holdout.difficulty_bin,
                            holdout.action_count, holdout.num_objectives,
                            m, t, if t > 0 { 100.0 * m as f32 / t as f32 } else { 0.0 }
                        )
                    }
                }
            }
        }
    }

    pub fn evaluate_agent(
        &self,
        agent_id: i64,
        db_path: &str,
        arena_tag: &str,
        num_handlers: usize,
        min_difficulty_bin: Option<i64>,
        max_difficulty_bin: Option<i64>,
        node_idx: i64,
        num_nodes: i64,
    ) -> Result<(), String> {
        let conn = Connection::open(db_path).map_err(|e| format!("Failed to open database: {e}"))?;
        conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")
            .map_err(|e| format!("Failed to set PRAGMAs: {e}"))?;

        let environments = load_holdout_environments(
            &conn, min_difficulty_bin, max_difficulty_bin, node_idx, num_nodes,
        ).map_err(|e| format!("Failed to query environments: {e}"))?;

        let arena_name = format!("mcts_{}_{}_{}", arena_tag, NUM_SLOTS, num_handlers);
        let arena: Arena<TilersSlot> =
            Arena::create_or_open(&arena_name, NUM_SLOTS, num_handlers)
                .map_err(|e| format!("Failed to open arena: {e}"))?;
        let client = TilersIpcClient::new(arena, 0);

        self.evaluate_agent_with_client(agent_id, &environments, &client as &dyn InferenceClient<TilersEnv>, &conn);
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tilers::env::Environment as TilersEnvInner;
    use crate::client::TrivialTilersIpcClient;

    fn make_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE solutions (
                agent_id            INTEGER NOT NULL,
                environment_id      INTEGER NOT NULL,
                actions             TEXT,
                solution_depth      REAL,
                achieved_objectives INTEGER,
                eval_reward         REAL,
                attempted_at        TEXT,
                PRIMARY KEY (agent_id, environment_id)
            );",
        ).unwrap();
        conn
    }

    fn make_holdout(id: i64, env: &TilersEnvInner) -> HoldoutEnvironment {
        HoldoutEnvironment {
            environment_id: id,
            // `to_json` takes a layer count, not a lookahead.
            json: env.to_json(LOOKAHEAD + 1),
            num_objectives: 1,
            difficulty_bin: 0,
            action_count: 0,
        }
    }

    fn small_env() -> TilersEnvInner {
        let mut env = TilersEnvInner::new(3, 3, 1);
        env.set_seed(Some(7));
        env.random_start(1, false);
        env
    }

    // ─── Evaluator::new ──────────────────────────────────────────────────────

    #[test]
    fn test_new_stores_params() {
        let e = Evaluator::new(50, 1.5, 0.4, 1.2, 1.0);
        assert_eq!(e.mcts_steps, 50);
        assert!((e.c_puct - 1.5).abs() < 1e-6);
        assert!((e.reward_saturation_temperature - 0.4).abs() < 1e-6);
    }

    // ─── solve_with_heuristic ────────────────────────────────────────────────

    #[test]
    fn test_solve_with_heuristic_nonnegative() {
        let e = Evaluator::new(5, 1.4, 1.0, 1.2, 1.0);
        let env = small_env();
        let depth = e.solve_with_heuristic(&env)
            .expect("small_env should be solvable by the heuristic");
        assert!(depth >= 0.0, "depth={depth}");
    }

    // ─── skip already-attempted environments ─────────────────────────────────

    #[test]
    fn test_skips_already_attempted_environment() {
        let conn = make_db();
        let env = small_env();
        let holdout = make_holdout(1, &env);

        // Pre-insert a solution so this environment is marked attempted.
        conn.execute(
            "INSERT INTO solutions (agent_id, environment_id, actions, solution_depth, attempted_at)
             VALUES (1, 1, '[]', NULL, '2024-01-01T00:00:00Z')",
            [],
        ).unwrap();

        let evaluator = Evaluator::new(5, 1.4, 1.0, 1.2, 1.0);
        let client = TrivialTilersIpcClient {};
        evaluator.evaluate_agent_with_client(1, &[holdout], &client, &conn);

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM solutions", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1, "should not insert a second row for an already-attempted env");
    }

    // ─── insert solution for new environment ─────────────────────────────────

    #[test]
    fn test_inserts_solution_for_new_environment() {
        let conn = make_db();
        let env = small_env();
        let holdout = make_holdout(42, &env);

        let evaluator = Evaluator::new(5, 1.4, 1.0, 1.2, 1.0);
        let client = TrivialTilersIpcClient {};
        evaluator.evaluate_agent_with_client(99, &[holdout], &client, &conn);

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM solutions WHERE agent_id = 99 AND environment_id = 42",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "should have inserted exactly one solution row");
    }
}

// =============================================================================
// Python bindings
// =============================================================================
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;

/// Evaluate a single agent against all holdout environments. Python is
/// responsible for starting and stopping the inference server handlers before
/// and after calling this function. Results are written to the solutions table
/// in the database. Already-attempted environments are skipped so this call
/// is safe to retry after a crash.
///
/// Args:
///     agent_id:           ID of the agent to evaluate.
///     db_path:            Path to the SQLite database.
///     arena_tag:          Arena tag matching the running handlers.
///     num_handlers:       Number of handler processes Python started.
///     mcts_steps:         Number of MCTS iterations per action.
///     c_puct:             Exploration constant.
///     reward_saturation_temperature: Tanh saturation knob for the terminal
///                         reward. Smaller → more categorical (±1); larger
///                         → more linear in depth-delta. Must match the
///                         value used by the gatherer that produced this
///                         agent's training data, otherwise the value head's
///                         outputs are calibrated to a different scale.
///     max_difficulty_bin: If set, skip holdout environments whose
///                         difficulty_bin (solver-action-count bin; see
///                         init_db.py ACTION_BIN_EDGES) exceeds this value.
///                         Defaults to None (evaluate every environment).
///     node_idx:           This node's 0-based index in the gather/eval
///                         allocation. Defaults to 0.
///     num_nodes:          Total number of nodes that will collectively
///                         evaluate this agent. Each node sees a
///                         round-robin slice of the holdout, balanced by
///                         difficulty tier. Defaults to 1 (this node
///                         evaluates everything).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    agent_id,
    db_path,
    arena_tag = String::from("eval"),
    num_handlers = 1,
    mcts_steps = 10_000,
    c_puct = 1.4,
    reward_saturation_temperature = 0.3,
    max_difficulty_bin = None,
    node_idx = 0,
    num_nodes = 1,
    max_action_multiplier = 1.2,
    min_difficulty_bin = None,
    root_softmax_temp = 1.0,
))]
pub fn run_evaluator(
    agent_id: i64,
    db_path: String,
    arena_tag: String,
    num_handlers: usize,
    mcts_steps: usize,
    c_puct: f32,
    reward_saturation_temperature: f32,
    max_difficulty_bin: Option<i64>,
    node_idx: i64,
    num_nodes: i64,
    max_action_multiplier: f32,
    min_difficulty_bin: Option<i64>,
    root_softmax_temp: f32,
) -> PyResult<()> {
    let evaluator = Evaluator::new(
        mcts_steps,
        c_puct,
        reward_saturation_temperature,
        max_action_multiplier,
        root_softmax_temp,
    );

    let conn = Connection::open(&db_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open database: {e}")))?;
    conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to set PRAGMAs: {e}")))?;

    let environments = load_holdout_environments(
        &conn, min_difficulty_bin, max_difficulty_bin, node_idx, num_nodes,
    ).map_err(|e| PyRuntimeError::new_err(format!("Failed to query environments: {e}")))?;

    let max_filter_msg = match (min_difficulty_bin, max_difficulty_bin) {
        (Some(lo), Some(hi)) => format!(" (difficulty_bin in [{}, {}])", lo, hi),
        (Some(lo), None) => format!(" (difficulty_bin >= {})", lo),
        (None, Some(hi)) => format!(" (difficulty_bin <= {})", hi),
        (None, None) => String::new(),
    };
    let slice_msg = if num_nodes > 1 {
        format!(" [node {}/{} round-robin slice]", node_idx, num_nodes)
    } else {
        String::new()
    };
    println!(
        "[Evaluator] Loaded {} holdout environments{}{} for agent {}.",
        environments.len(), max_filter_msg, slice_msg, agent_id,
    );

    // Handlers are already running. Open the arena and connect a client.
    // Arena name must match the convention used by handler.py:
    //   f"mcts_{arena_tag}_{num_slots}_{num_handlers}"
    let arena_name = format!("mcts_{}_{}_{}", arena_tag, NUM_SLOTS, num_handlers);
    let arena: Arena<TilersSlot> =
        Arena::create_or_open(&arena_name, NUM_SLOTS, num_handlers)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open arena: {e}")))?;
    let client = TilersIpcClient::new(arena, 0);

    evaluator.evaluate_agent_with_client(agent_id, &environments, &client as &dyn InferenceClient<TilersEnv>, &conn);

    println!("[Evaluator] Evaluation complete for agent {}.", agent_id);
    Ok(())
}