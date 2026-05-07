use rusqlite::{params, Connection};
use serde_json::json;

use mcts_core::ipc_core::Arena;
use mcts_core::inference::InferenceClient;
use mcts_core::mcts::MCTS;

use tilers::env::Environment;
use tilers::solver::Solver;

use crate::constants::*;
use crate::environment::TilersEnv;
use crate::slot::TilersSlot;
use crate::client::TilersIpcClient;

// =============================================================================
// Configuration
// =============================================================================
const NUM_SLOTS: usize = 2048;
const NUM_OBJECTIVE_LAYERS: usize = DEFAULT_LOOKAHEAD;

// =============================================================================
// HoldoutEnvironment
// =============================================================================
pub struct HoldoutEnvironment {
    environment_id: i64,
    json: String,
    num_objectives: i64,
}

/// Load holdout environments from the database, optionally filtered to those
/// with at most `max_num_objectives` objectives, and round-robin sliced
/// across nodes by (node_idx, num_nodes). Single-node callers pass
/// (0, 1) and get every row. Multi-node callers pass distinct node_idx
/// values and ROW_NUMBER over (num_objectives ASC, environment_id ASC)
/// distributes envs evenly across difficulty tiers — so each node gets a
/// balanced mix of easy and hard envs rather than a contiguous block.
fn load_holdout_environments(
    conn: &Connection,
    max_num_objectives: Option<i64>,
    node_idx: i64,
    num_nodes: i64,
) -> Result<Vec<HoldoutEnvironment>, rusqlite::Error> {
    let max_filter = if max_num_objectives.is_some() {
        "WHERE num_objectives <= ?"
    } else {
        ""
    };
    // Slice via ROW_NUMBER so every node gets a balanced mix across the
    // difficulty tiers. (rn - 1) % num_nodes = node_idx is the round-robin.
    let sql = format!(
        "WITH ordered AS (
            SELECT environment_id, json, num_objectives,
                   ROW_NUMBER() OVER (ORDER BY num_objectives ASC, environment_id ASC) AS rn
            FROM environments
            {}
         )
         SELECT environment_id, json, num_objectives
         FROM ordered
         WHERE (rn - 1) % ? = ?
         ORDER BY num_objectives ASC",
        max_filter
    );

    let mut bound: Vec<i64> = Vec::new();
    if let Some(m) = max_num_objectives {
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
}

impl Evaluator {
    pub fn new(
        mcts_steps: usize,
        c_puct: f32,
        reward_saturation_temperature: f32,
    ) -> Self {
        Self {
            mcts_steps,
            c_puct,
            reward_saturation_temperature,
        }
    }

    /// Solve the environment using the heuristic solver to obtain the
    /// reference depth used in the terminal evaluator.
    fn solve_with_heuristic(&self, env: &Environment) -> f32 {
        let mut solved_env = env.clone();
        let solver = Solver::new();
        solver.solve(&mut solved_env, true).unwrap();
        solved_env.depth(true, true) as f32
    }

    /// Run MCTS on a single environment and return the action sequence taken
    /// and the final solution depth. Returns None for depth if the agent did
    /// not finish.
    fn evaluate_single(
        &self,
        env: &Environment,
        client: &dyn InferenceClient<TilersEnv>,
    ) -> (Vec<Action>, Option<f32>) {
        let mut mcts: MCTS<TilersEnv> = MCTS::new(8);
        let mut tilers_env = TilersEnv::new(env.clone(), NUM_OBJECTIVE_LAYERS);
        tilers_env.inner.set_cultivation_time(10);

        let reference_depth = self.solve_with_heuristic(&tilers_env.inner);
        let temperature = self.reward_saturation_temperature;

        let terminal_evaluator = |e: &TilersEnv| -> f32 {
            if !e.inner.done() {
                -1.0
            } else {
                let d = e.inner.depth(true, true) as f32;
                let ratio = (reference_depth - d) / (reference_depth + 1e-6);
                // tanh saturation; same formulation as Gatherer's terminal
                // evaluator. Eval and gather must agree on temperature so
                // the value-head's terminal targets at training time match
                // the scalars MCTS sees at eval time.
                (ratio / temperature).tanh()
            }
        };

        let solver = Solver::new();
        let mut tmp_env = env.clone();
        let max_actions = match solver.solve(&mut tmp_env, false) {
            Ok(sol) => (sol.len() as f32 * 1.2) as usize,
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
                true,
            );

            // Greedy: pick the action with the most visits.
            let action = *valid_actions
                .iter()
                .max_by_key(|&&a| *root.edge_visits.get(&(a as Action)).unwrap_or(&0))
                .unwrap() as Action;

            actions_taken.push(action);
            let _ = tilers_env.inner.step(action as usize);
            tilers_env.inner.finish_cultivating(None, None);
            mcts.advance_root(action);
        }

        let done = tilers_env.inner.done();
        let solution_depth = if done {
            Some(tilers_env.inner.depth(true, true) as f32)
        } else {
            None
        };

        (actions_taken, solution_depth)
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

            let (actions, solution_depth) = self.evaluate_single(&env, client);

            let actions_json = json!(actions).to_string();
            let attempted_at = chrono::Utc::now().to_rfc3339();

            conn.execute(
                "INSERT OR IGNORE INTO solutions
                    (agent_id, environment_id, actions, solution_depth, attempted_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    agent_id,
                    holdout.environment_id,
                    actions_json,
                    solution_depth,
                    attempted_at,
                ],
            )
            .expect("Failed to insert solution");

            match solution_depth {
                Some(d) => println!(
                    "[Evaluator] Agent {} | Env {} ({} obj) | depth = {:.1}",
                    agent_id, holdout.environment_id, holdout.num_objectives, d
                ),
                None => println!(
                    "[Evaluator] Agent {} | Env {} ({} obj) | did not finish.",
                    agent_id, holdout.environment_id, holdout.num_objectives
                ),
            }
        }
    }

    pub fn evaluate_agent(
        &self,
        agent_id: i64,
        db_path: &str,
        arena_tag: &str,
        num_handlers: usize,
        max_num_objectives: Option<i64>,
        node_idx: i64,
        num_nodes: i64,
    ) -> Result<(), String> {
        let conn = Connection::open(db_path).map_err(|e| format!("Failed to open database: {e}"))?;
        conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")
            .map_err(|e| format!("Failed to set PRAGMAs: {e}"))?;

        let environments = load_holdout_environments(
            &conn, max_num_objectives, node_idx, num_nodes,
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
                agent_id       INTEGER NOT NULL,
                environment_id INTEGER NOT NULL,
                actions        TEXT,
                solution_depth REAL,
                attempted_at   TEXT,
                PRIMARY KEY (agent_id, environment_id)
            );",
        ).unwrap();
        conn
    }

    fn make_holdout(id: i64, env: &TilersEnvInner) -> HoldoutEnvironment {
        HoldoutEnvironment {
            environment_id: id,
            json: env.to_json(NUM_OBJECTIVE_LAYERS),
            num_objectives: 1,
        }
    }

    fn small_env() -> TilersEnvInner {
        let mut env = TilersEnvInner::new(3, 3, 1);
        env.set_seed(Some(7));
        env.random_objectives(1, false);
        env
    }

    // ─── Evaluator::new ──────────────────────────────────────────────────────

    #[test]
    fn test_new_stores_params() {
        let e = Evaluator::new(50, 1.5, 0.4);
        assert_eq!(e.mcts_steps, 50);
        assert!((e.c_puct - 1.5).abs() < 1e-6);
        assert!((e.reward_saturation_temperature - 0.4).abs() < 1e-6);
    }

    // ─── solve_with_heuristic ────────────────────────────────────────────────

    #[test]
    fn test_solve_with_heuristic_nonnegative() {
        let e = Evaluator::new(5, 1.4, 1.0);
        let env = small_env();
        let depth = e.solve_with_heuristic(&env);
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

        let evaluator = Evaluator::new(5, 1.4, 1.0);
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

        let evaluator = Evaluator::new(5, 1.4, 1.0);
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
///     max_num_objectives: If set, skip holdout environments whose
///                         num_objectives exceeds this value. Defaults to
///                         None (evaluate every environment).
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
    max_num_objectives = None,
    node_idx = 0,
    num_nodes = 1,
))]
pub fn run_evaluator(
    agent_id: i64,
    db_path: String,
    arena_tag: String,
    num_handlers: usize,
    mcts_steps: usize,
    c_puct: f32,
    reward_saturation_temperature: f32,
    max_num_objectives: Option<i64>,
    node_idx: i64,
    num_nodes: i64,
) -> PyResult<()> {
    let evaluator = Evaluator::new(
        mcts_steps,
        c_puct,
        reward_saturation_temperature,
    );

    let conn = Connection::open(&db_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open database: {e}")))?;
    conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to set PRAGMAs: {e}")))?;

    let environments = load_holdout_environments(
        &conn, max_num_objectives, node_idx, num_nodes,
    ).map_err(|e| PyRuntimeError::new_err(format!("Failed to query environments: {e}")))?;

    let max_filter_msg = match max_num_objectives {
        Some(m) => format!(" (num_objectives <= {})", m),
        None => String::new(),
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