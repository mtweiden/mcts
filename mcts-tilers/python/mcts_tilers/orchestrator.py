"""
orchestrate.py

Self-play pipeline orchestrator. Drives the following loop:
    1. GATHER  - run self-play workers until enough data has been collected.
    2. TRAIN   - launch the trainer and wait for a new checkpoint.
    3. EVALUATE - compare the new checkpoint against the incumbent.
    4. Repeat.

Resumption: a small JSON state file records the current phase so that if the
orchestrator is killed it picks up where it left off on restart.
"""

import os
import sys
import json
import time
import shutil
import sqlite3
import logging
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mcts_tilers import run_gatherer, run_evaluator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")


# ==============================================================================
# State helpers
# ==============================================================================

def load_state(state_path: str) -> dict:
    """
    Load orchestrator state from disk. If no file exists this is a fresh run
    and we start in the gathering phase at iteration 0.
    """
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        log.info(f"Resuming from state: {state}")
        return state
    return {
        "phase": "gathering",
        "iteration": 0,
        "candidate_id": None,
        "candidate_path": None,
    }


def save_state(state_path: str, state: dict) -> None:
    """
    Atomically write the state file via a temp file so a kill mid-write never
    corrupts it.
    """
    tmp = state_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)


# ==============================================================================
# Database helpers
# ==============================================================================

def get_incumbent(db_path: str) -> tuple[int, str]:
    """
    Return (agent_id, checkpoint_path) for the most recently promoted agent.
    Raises RuntimeError if no promoted agent exists — the pretrained checkpoint
    must be seeded into the database manually before the first run.
    """
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT agent_id, checkpoint_path
            FROM agents
            WHERE promoted_at IS NOT NULL
            ORDER BY promoted_at DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError(
            "No promoted agent found. Please seed the database with the "
            "pretrained checkpoint before starting the pipeline."
        )
    return int(row[0]), str(row[1])


def get_candidate(db_path: str, checkpoint_dir: str) -> tuple[int, str]:
    """
    Find the highest agent_id checkpoint file in checkpoint_dir that has not
    yet been promoted. Checkpoint files are named agent_{id}.ckpt.
    Returns (agent_id, checkpoint_path).
    """
    conn = sqlite3.connect(db_path)
    try:
        promoted_ids = {
            row[0] for row in conn.execute(
                "SELECT agent_id FROM agents WHERE promoted_at IS NOT NULL"
            ).fetchall()
        }
    finally:
        conn.close()

    candidates = []
    for ckpt in Path(checkpoint_dir).glob("agent_*.ckpt"):
        try:
            agent_id = int(ckpt.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if agent_id not in promoted_ids:
            candidates.append((agent_id, str(ckpt)))

    if not candidates:
        raise RuntimeError(
            f"No unpromoted checkpoint found in {checkpoint_dir}. "
            "Has the trainer run yet?"
        )
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def register_agent(db_path: str, agent_id: int, checkpoint_path: str) -> None:
    """
    Insert a new agent row into the database so the evaluator can reference
    it by ID.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO agents (agent_id, checkpoint_path, created_at)
            VALUES (?, ?, datetime('now'))
            """,
            (agent_id, checkpoint_path),
        )
        conn.commit()
    finally:
        conn.close()


# ==============================================================================
# Shard helpers
# ==============================================================================

def count_lines(directory: str) -> int:
    """
    Count total lines across all .jsonl files in directory. Each line is one
    training sample (one MCTS step) so this is the metric we use to decide
    when gathering is done.
    """
    total = 0
    for path in Path(directory).glob("*.jsonl"):
        with open(path, "rb") as f:
            total += f.read().count(b"\n")
    return total


def retire_shards(shard_dir: str, retired_dir: str) -> None:
    """
    Move all active .jsonl shards to the retired directory after training so
    the next gather iteration starts fresh. Appends a timestamp suffix if a
    filename collision occurs.
    """
    Path(retired_dir).mkdir(parents=True, exist_ok=True)
    moved = 0
    for shard in Path(shard_dir).glob("*.jsonl"):
        dest = Path(retired_dir) / shard.name
        if dest.exists():
            dest = Path(retired_dir) / f"{shard.stem}_{int(time.time())}.jsonl"
        shutil.move(str(shard), str(dest))
        moved += 1
    log.info(f"Retired {moved} shard(s) from {shard_dir} to {retired_dir}.")


# ==============================================================================
# Handler lifecycle
# ==============================================================================

def start_handlers(
    handle_script: str,
    checkpoint_path: str,
    arena_tag: str,
    num_slots: int,
    num_handlers: int,
    warmup_secs: float = 5.0,
) -> list[subprocess.Popen]:
    """
    Launch num_handlers inference server processes. Each gets a handler_id so
    it can select the correct GPU. Waits warmup_secs for them to register with
    the arena before returning.
    """
    procs = []
    for handler_id in range(num_handlers):
        proc = subprocess.Popen([
            sys.executable, handle_script,
            "--arena_name", "mcts",
            "--arena_tag", arena_tag,
            "--weights", checkpoint_path,
            "--num_slots", str(num_slots),
            "--num_handlers", str(num_handlers),
            "--handler_id", str(handler_id),
        ])
        procs.append(proc)
        log.info(f"Started handler {handler_id} (pid={proc.pid}).")
    log.info(f"Waiting {warmup_secs}s for handlers to initialise...")
    time.sleep(warmup_secs)
    return procs


def stop_handlers(procs: list[subprocess.Popen]) -> None:
    """
    Terminate all handler processes. Any handler that does not exit within
    10 seconds is killed.
    """
    for proc in procs:
        proc.terminate()
    for i, proc in enumerate(procs):
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning(f"Handler {i} did not exit cleanly, killing.")
            proc.kill()
            proc.wait()
    log.info("All handlers stopped.")


# ==============================================================================
# Evaluate phase helpers
# ==============================================================================

def promote_candidate(
    db_path: str,
    candidate_id: int,
    incumbent_id: int,
    promotion_threshold: float,
) -> tuple[bool, int, int, int, int]:
    """
    Read solution rows for both agents from the database, compare them
    environment by environment, and promote the candidate if its win rate
    exceeds promotion_threshold.

    Scoring per environment:
        candidate depth < incumbent depth                  -> candidate win
        incumbent depth < candidate depth                  -> incumbent win
        candidate finished, incumbent did not              -> candidate win
        incumbent finished, candidate did not              -> incumbent win
        both finished with equal depth, or both failed     -> draw

    Returns (promoted, candidate_wins, incumbent_wins, draws, total).
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                s_c.environment_id,
                s_c.solution_depth,
                s_i.solution_depth
            FROM solutions s_c
            JOIN solutions s_i
                ON s_c.environment_id = s_i.environment_id
            WHERE s_c.agent_id = ?
              AND s_i.agent_id = ?
            """,
            (candidate_id, incumbent_id),
        ).fetchall()

        candidate_wins = 0
        incumbent_wins = 0
        draws = 0

        for env_id, c_depth, i_depth in rows:
            if c_depth is not None and i_depth is not None:
                if c_depth < i_depth:
                    candidate_wins += 1
                elif i_depth < c_depth:
                    incumbent_wins += 1
                else:
                    draws += 1
            elif c_depth is not None:
                candidate_wins += 1
            elif i_depth is not None:
                incumbent_wins += 1
            else:
                draws += 1

        total = len(rows)
        win_rate = candidate_wins / total if total > 0 else 0.0
        promoted = win_rate > promotion_threshold

        log.info(
            f"[Evaluate] candidate_wins={candidate_wins} "
            f"incumbent_wins={incumbent_wins} "
            f"draws={draws} total={total} "
            f"win_rate={win_rate:.3f} threshold={promotion_threshold}"
        )

        evaluated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        conn.execute(
            """
            INSERT INTO evaluations
                (candidate_agent_id, incumbent_agent_id,
                 candidate_wins, incumbent_wins, draws, promoted, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (candidate_id, incumbent_id,
             candidate_wins, incumbent_wins, draws,
             int(promoted), evaluated_at),
        )

        if promoted:
            conn.execute(
                "UPDATE agents SET promoted_at = ? WHERE agent_id = ?",
                (evaluated_at, candidate_id),
            )

        conn.commit()
    finally:
        conn.close()

    return promoted, candidate_wins, incumbent_wins, draws, total

# ==============================================================================
# Gather phase
# ==============================================================================

def run_gather_phase(
    args: argparse.Namespace,
    state: dict,
    state_path: str,
    incumbent_checkpoint: str,
) -> None:
    """
    Start inference servers then submit run_gatherer jobs to a
    ProcessPoolExecutor. Each future is one environment: generate, search,
    write. When a future completes the orchestrator resubmits the same
    worker_id immediately unless the line count threshold has been reached,
    in which case it drains in-flight futures without submitting new ones.
    """
    log.info(
        f"[Gather] Starting iteration {state['iteration']}. "
        f"Target: {args.gather_threshold} lines in {args.shard_dir}."
    )

    handlers = start_handlers(
        handle_script=args.handle_script,
        checkpoint_path=incumbent_checkpoint,
        arena_tag=args.gather_arena_tag,
        num_slots=args.num_slots,
        num_handlers=args.num_handlers,
        warmup_secs=args.handler_warmup_secs,
    )

    gather_kwargs = dict(
        num_handlers=args.num_handlers,
        output_dir=args.shard_dir,
        arena_tag=args.gather_arena_tag,
        height=args.height,
        width=args.width,
        num_objectives=args.num_objectives,
        min_num_objectives=args.min_num_objectives,
        num_blanks=args.num_blanks,
        mcts_steps=args.mcts_steps,
        fast_steps=args.fast_steps,
        p_full_search=args.p_full_search,
        dirichlet_epsilon=args.dirichlet_epsilon,
        reward_saturation_temperature=args.reward_saturation_temperature,
        c_puct=args.c_puct,
        max_generated_depth=args.max_generated_depth,
        num_shuffles=args.num_shuffles,
        trajectory_dir=args.trajectory_dir,
    )

    try:
        with ProcessPoolExecutor(max_workers=args.num_gatherers) as executor:

            # Seed the pool with one future per worker. Each worker gets a
            # fixed worker_id so it always writes to the same output file and
            # uses the same arena client slot.
            future_to_worker = {
                executor.submit(run_gatherer, worker_id=i, **gather_kwargs): i
                for i in range(args.num_gatherers)
            }
            threshold_reached = False

            while future_to_worker:
                for future in as_completed(future_to_worker):
                    worker_id = future_to_worker.pop(future)

                    try:
                        result = future.result()
                        if result is not None:
                            sol_depth, ref_depth, done = result
                            log.debug(
                                f"[Gather] worker={worker_id} "
                                f"sol={sol_depth:.1f} ref={ref_depth:.1f} "
                                f"done={done}"
                            )
                    except Exception as e:
                        log.error(f"[Gather] worker={worker_id} raised: {e}")

                    if not threshold_reached:
                        lines = count_lines(args.shard_dir)
                        log.info(
                            f"[Gather] {lines}/{args.gather_threshold} lines."
                        )
                        if lines >= args.gather_threshold:
                            threshold_reached = True
                            log.info("[Gather] Threshold reached. Draining.")

                    # Resubmit the same worker_id if we still need data.
                    if not threshold_reached:
                        future_to_worker[
                            executor.submit(run_gatherer, worker_id=worker_id, **gather_kwargs)
                        ] = worker_id

                    # Break out so the while condition is re-evaluated with
                    # the updated future_to_worker dict.
                    break

    finally:
        stop_handlers(handlers)

    log.info("[Gather] Phase complete.")
    state["phase"] = "training"
    save_state(state_path, state)


# ==============================================================================
# Train phase
# ==============================================================================

def run_train_phase(
    args: argparse.Namespace,
    state: dict,
    state_path: str,
) -> None:
    """
    Launch train.py as a subprocess and block until it finishes. On success
    retire the active shards, identify the new checkpoint, register it in the
    database, and advance state to evaluating.
    """
    log.info("[Train] Launching trainer.")

    result = subprocess.run([
        sys.executable, args.train_script,
        "--manifest_path", args.manifest_path,
        "--checkpoint_dir", args.checkpoint_dir,
        "--global_state_dir", args.global_state_dir,
        "--embedding_dim", str(args.embedding_dim),
        "--num_layers", str(args.num_layers),
        "--lookahead", str(args.lookahead),
        "--num_steps", str(args.num_steps),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--max_size", str(args.max_size),
        "--kl_beta", str(args.kl_beta),
        "--value_coeff", str(args.value_coeff),
        "--entropy_coeff", str(args.entropy_coeff),
        "--sharpness_tau", str(args.sharpness_tau),
        "--log_interval", str(args.log_interval),
    ])
    if result.returncode != 0:
        raise RuntimeError(f"Trainer exited with code {result.returncode}.")

    log.info("[Train] Complete. Retiring shards.")
    retire_shards(args.shard_dir, args.retired_shard_dir)

    candidate_id, candidate_path = get_candidate(args.db_path, args.checkpoint_dir)
    register_agent(args.db_path, candidate_id, candidate_path)
    log.info(f"[Train] Registered agent {candidate_id} at {candidate_path}.")

    state["phase"] = "evaluating"
    state["candidate_id"] = candidate_id
    state["candidate_path"] = candidate_path
    save_state(state_path, state)


# ==============================================================================
# Evaluate phase
# ==============================================================================

def run_evaluate_phase(
    args: argparse.Namespace,
    state: dict,
    state_path: str,
) -> None:
    """
    Evaluate the candidate and incumbent agents sequentially against the
    holdout set. Handlers are started and stopped here around each agent's
    evaluation. Promotion logic runs in Python after both evaluations are
    complete.

    The two agents are evaluated sequentially with the same arena_tag so only
    one set of handler processes is live at a time, keeping GPU memory usage
    predictable.
    """
    candidate_id = state["candidate_id"]
    candidate_path = state["candidate_path"]
    incumbent_id, incumbent_path = get_incumbent(args.db_path)

    log.info(
        f"[Evaluate] Candidate agent {candidate_id} vs "
        f"incumbent agent {incumbent_id}."
    )

    # Shared kwargs for both run_evaluation calls.
    eval_kwargs = dict(
        db_path=args.db_path,
        arena_tag=args.eval_arena_tag,
        num_handlers=args.num_eval_handlers,
        mcts_steps=args.mcts_steps,
        c_puct=args.c_puct,
        reward_saturation_temperature=args.reward_saturation_temperature,
    )

    # --- Evaluate candidate --------------------------------------------------
    log.info(f"[Evaluate] Evaluating candidate agent {candidate_id}.")
    candidate_handlers = start_handlers(
        handle_script=args.handle_script,
        checkpoint_path=candidate_path,
        arena_tag=args.eval_arena_tag,
        num_slots=args.num_slots,
        num_handlers=args.num_eval_handlers,
        warmup_secs=args.handler_warmup_secs,
    )
    try:
        run_evaluator(agent_id=candidate_id, **eval_kwargs)
    finally:
        stop_handlers(candidate_handlers)

    # --- Evaluate incumbent --------------------------------------------------
    log.info(f"[Evaluate] Evaluating incumbent agent {incumbent_id}.")
    incumbent_handlers = start_handlers(
        handle_script=args.handle_script,
        checkpoint_path=incumbent_path,
        arena_tag=args.eval_arena_tag,
        num_slots=args.num_slots,
        num_handlers=args.num_eval_handlers,
        warmup_secs=args.handler_warmup_secs,
    )
    try:
        run_evaluator(agent_id=incumbent_id, **eval_kwargs)
    finally:
        stop_handlers(incumbent_handlers)

    # --- Compare and promote -------------------------------------------------
    promoted, candidate_wins, incumbent_wins, draws, total = promote_candidate(
        db_path=args.db_path,
        candidate_id=candidate_id,
        incumbent_id=incumbent_id,
        promotion_threshold=args.promotion_threshold,
    )

    if promoted:
        log.info(
            f"[Evaluate] Candidate agent {candidate_id} promoted "
            f"(win rate {candidate_wins}/{total})."
        )
    else:
        log.info(
            f"[Evaluate] Candidate agent {candidate_id} not promoted "
            f"(win rate {candidate_wins}/{total}). Incumbent retained."
        )

    # Advance to the next gather iteration regardless of outcome.
    state["phase"] = "gathering"
    state["iteration"] += 1
    state["candidate_id"] = None
    state["candidate_path"] = None
    save_state(state_path, state)


# ==============================================================================
# Main loop
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play pipeline orchestrator.")

    # --- Paths ----------------------------------------------------------------
    parser.add_argument("--state_path", type=str, default="orchestrator_state.json")
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--shard_dir", type=str, required=True)
    parser.add_argument("--retired_shard_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--global_state_dir", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--handle_script", type=str, default="handler.py")
    parser.add_argument("--train_script", type=str, default="train.py")
    parser.add_argument("--trajectory_dir", type=str, default=None)

    # --- Arena ----------------------------------------------------------------
    parser.add_argument("--num_slots", type=int, default=2048)
    parser.add_argument("--num_handlers", type=int, default=1)
    parser.add_argument("--num_eval_handlers", type=int, default=1)
    parser.add_argument("--gather_arena_tag", type=str, default="gather")
    parser.add_argument("--eval_arena_tag", type=str, default="eval")
    parser.add_argument("--handler_warmup_secs", type=float, default=5.0)

    # --- Gather ---------------------------------------------------------------
    parser.add_argument("--num_gatherers", type=int, default=64)
    parser.add_argument("--gather_threshold", type=int, default=100_000)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--num_objectives", type=int, default=4)
    parser.add_argument("--min_num_objectives", type=int, default=2)
    parser.add_argument("--num_blanks", type=int, default=2)
    parser.add_argument("--mcts_steps", type=int, default=10_000)
    parser.add_argument("--fast_steps", type=int, default=1_600)
    parser.add_argument("--p_full_search", type=float, default=0.25)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    parser.add_argument("--reward_saturation_temperature", type=float, default=0.3,
        help="Tanh saturation temperature for the terminal reward target. "
             "Replaces the old --reward_ratio_limit clip+normalize.")
    parser.add_argument("--c_puct", type=float, default=1.4)
    parser.add_argument("--max_generated_depth", type=int, default=10_000)
    parser.add_argument("--num_shuffles", type=int, default=0)

    # --- Train ----------------------------------------------------------------
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_size", type=int, default=500_000)
    parser.add_argument("--kl_beta", type=float, default=1.0)
    parser.add_argument("--value_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--sharpness_tau", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)

    # --- Evaluate -------------------------------------------------------------
    parser.add_argument("--promotion_threshold", type=float, default=0.55)

    args = parser.parse_args()

    # Make sure key directories exist.
    Path(args.shard_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    state = load_state(args.state_path)

    while True:
        phase = state["phase"]
        log.info(f"=== Iteration {state['iteration']} | Phase: {phase} ===")

        if phase == "gathering":
            _, incumbent_checkpoint = get_incumbent(args.db_path)
            run_gather_phase(args, state, args.state_path, incumbent_checkpoint)

        elif phase == "training":
            run_train_phase(args, state, args.state_path)

        elif phase == "evaluating":
            run_evaluate_phase(args, state, args.state_path)

        else:
            raise RuntimeError(f"Unknown phase '{phase}' in state file.")


if __name__ == "__main__":
    main()