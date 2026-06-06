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
}

// =============================================================================
// Evaluator
// =============================================================================
pub struct Evaluator {
    mcts_steps: usize,
    c_puct: f32,
    reward_ratio_limit: f32,
}

impl Evaluator {
    pub fn new(
        mcts_steps: usize,
        c_puct: f32,
        reward_ratio_limit: f32,
    ) -> Self {
        Self {
            mcts_steps,
            c_puct,
            reward_ratio_limit,
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
        let mut tilers_env = TilersEnv::new(env.clone(), LOOKAHEAD);
        tilers_env.inner.set_cultivation_time(10);

        let reference_depth = self.solve_with_heuristic(&tilers_env.inner);
        let reward_ratio_limit = self.reward_ratio_limit;

        let terminal_evaluator = |e: &TilersEnv| -> f32 {
            if !e.inner.done() {
                -1.0
            } else {
                let d = e.inner.depth(true, true) as f32;
                let mut v = (reference_depth - d) / (reference_depth + 1e-6);
                v = v.max(-reward_ratio_limit).min(reward_ratio_limit);
                v / reward_ratio_limit
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

            // Greedy: pick the action with the most visits.  `valid_actions`
            // is `Vec<tilers::Action>`; encode to the flat `u16` id space the
            // tree/visits are keyed by.
            let action = valid_actions
                .iter()
                .map(|&a| rl::encode(&tilers_env.inner, a) as Action)
                .max_by_key(|&id| *root.edge_visits.get(&id).unwrap_or(&0))
                .unwrap();

            actions_taken.push(action);
            let a = rl::decode(&tilers_env.inner, action as usize)
                .expect("evaluator produced an invalid action id");
            let _ = tilers_env.inner.step(a);
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
    ) -> Result<(), String> {
        let conn = Connection::open(db_path).map_err(|e| format!("Failed to open database: {e}"))?;
        conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")
            .map_err(|e| format!("Failed to set PRAGMAs: {e}"))?;

        let environments: Vec<HoldoutEnvironment> = {
            let mut stmt = conn
                .prepare(
                    "SELECT environment_id, json, num_objectives \
                     FROM environments ORDER BY num_objectives ASC",
                )
                .map_err(|e| format!("Failed to prepare query: {e}"))?;

            stmt.query_map([], |row| {
                Ok(HoldoutEnvironment {
                    environment_id: row.get(0)?,
                    json: row.get(1)?,
                    num_objectives: row.get(2)?,
                })
            })
            .map_err(|e| format!("Failed to query environments: {e}"))?
            .filter_map(|r| r.ok())
            .collect()
        };

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
            // `to_json` takes a layer count, not a lookahead.
            json: env.to_json(LOOKAHEAD + 1),
            num_objectives: 1,
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
        let e = Evaluator::new(50, 1.5, 0.4);
        assert_eq!(e.mcts_steps, 50);
        assert!((e.c_puct - 1.5).abs() < 1e-6);
        assert!((e.reward_ratio_limit - 0.4).abs() < 1e-6);
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
///     reward_ratio_limit: Clips the terminal reward to [-limit, limit].
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
    agent_id,
    db_path,
    arena_tag = String::from("eval"),
    num_handlers = 1,
    mcts_steps = 10_000,
    c_puct = 1.4,
    reward_ratio_limit = 0.3,
))]
pub fn run_evaluator(
    agent_id: i64,
    db_path: String,
    arena_tag: String,
    num_handlers: usize,
    mcts_steps: usize,
    c_puct: f32,
    reward_ratio_limit: f32,
) -> PyResult<()> {
    let evaluator = Evaluator::new(
        mcts_steps,
        c_puct,
        reward_ratio_limit,
    );

    let conn = Connection::open(&db_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open database: {e}")))?;
    conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to set PRAGMAs: {e}")))?;

    // Load all holdout environments ordered by difficulty.
    let environments: Vec<HoldoutEnvironment> = {
        let mut stmt = conn
            .prepare(
                "SELECT environment_id, json, num_objectives \
                 FROM environments ORDER BY num_objectives ASC",
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to prepare query: {e}")))?;

        stmt.query_map([], |row| {
            Ok(HoldoutEnvironment {
                environment_id: row.get(0)?,
                json: row.get(1)?,
                num_objectives: row.get(2)?,
            })
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to query environments: {e}")))?
        .filter_map(|r| r.ok())
        .collect()
    };

    println!(
        "[Evaluator] Loaded {} holdout environments for agent {}.",
        environments.len(),
        agent_id,
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