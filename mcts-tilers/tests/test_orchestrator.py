"""
Tests for mcts_tilers.orchestrator.

Unit tests cover pure helper functions using real in-memory SQLite and temp
directories.  Phase-level tests mock all external I/O (subprocesses, the Rust
extension, file I/O) so they complete instantly and deterministically.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import tempfile
import time
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from mcts_tilers.orchestrator import (
    count_lines,
    get_candidate,
    get_incumbent,
    load_state,
    promote_candidate,
    register_agent,
    retire_shards,
    run_evaluate_phase,
    run_gather_phase,
    run_train_phase,
    save_state,
    start_handlers,
    stop_handlers,
)


# ==============================================================================
# Fixtures / helpers
# ==============================================================================

def make_agents_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE agents (
            agent_id        INTEGER PRIMARY KEY,
            checkpoint_path TEXT NOT NULL,
            created_at      TEXT,
            promoted_at     TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def make_full_db(path: str) -> None:
    """Create the full schema needed for promote_candidate."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE agents (
            agent_id        INTEGER PRIMARY KEY,
            checkpoint_path TEXT NOT NULL,
            created_at      TEXT,
            promoted_at     TEXT
        );
        CREATE TABLE solutions (
            agent_id        INTEGER NOT NULL,
            environment_id  INTEGER NOT NULL,
            actions         TEXT,
            solution_depth  REAL,
            attempted_at    TEXT,
            PRIMARY KEY (agent_id, environment_id)
        );
        CREATE TABLE evaluations (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_agent_id  INTEGER,
            incumbent_agent_id  INTEGER,
            candidate_wins      INTEGER,
            incumbent_wins      INTEGER,
            draws               INTEGER,
            promoted            INTEGER,
            evaluated_at        TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def make_args(**overrides) -> argparse.Namespace:
    """Return a minimal Namespace suitable for phase functions."""
    defaults = dict(
        handle_script="handler.py",
        train_script="train.py",
        db_path="pipeline.db",
        shard_dir="/tmp/shards",
        retired_shard_dir="/tmp/retired",
        checkpoint_dir="/tmp/ckpts",
        global_state_dir="/tmp/state",
        manifest_path="/tmp/manifest.json",
        trajectory_dir=None,
        # Arena
        num_slots=2048,
        num_handlers=1,
        num_eval_handlers=1,
        gather_arena_tag="gather",
        eval_arena_tag="eval",
        handler_warmup_secs=0.0,
        # Gather
        num_gatherers=2,
        gather_threshold=10,
        height=4,
        width=4,
        num_objectives=4,
        min_num_objectives=2,
        num_blanks=2,
        mcts_steps=100,
        fast_steps=20,
        p_full_search=0.25,
        dirichlet_epsilon=0.25,
        reward_ratio_limit=0.3,
        c_puct=1.4,
        max_generated_depth=1000,
        num_shuffles=0,
        # Train
        embedding_dim=64,
        num_layers=4,
        lookahead=1,
        num_steps=100,
        batch_size=64,
        learning_rate=1e-4,
        max_size=10000,
        kl_beta=1.0,
        value_coeff=1.0,
        entropy_coeff=0.01,
        sharpness_tau=1.0,
        log_interval=10,
        # Evaluate
        promotion_threshold=0.55,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ==============================================================================
# TestStateHelpers
# ==============================================================================

class TestStateHelpers:

    def test_load_state_fresh_run(self, tmp_path):
        state_path = str(tmp_path / "state.json")
        state = load_state(state_path)
        assert state["phase"] == "gathering"
        assert state["iteration"] == 0
        assert state["candidate_id"] is None
        assert state["candidate_path"] is None

    def test_load_state_existing_file(self, tmp_path):
        state_path = str(tmp_path / "state.json")
        saved = {"phase": "training", "iteration": 3, "candidate_id": 5, "candidate_path": "/p"}
        with open(state_path, "w") as f:
            json.dump(saved, f)
        state = load_state(state_path)
        assert state == saved

    def test_save_state_creates_file(self, tmp_path):
        state_path = str(tmp_path / "state.json")
        payload = {"phase": "evaluating", "iteration": 2, "candidate_id": 7, "candidate_path": "/q"}
        save_state(state_path, payload)
        assert os.path.exists(state_path)
        with open(state_path) as f:
            assert json.load(f) == payload

    def test_save_state_no_tmp_leftover(self, tmp_path):
        state_path = str(tmp_path / "state.json")
        save_state(state_path, {"phase": "gathering", "iteration": 0,
                                "candidate_id": None, "candidate_path": None})
        assert not os.path.exists(state_path + ".tmp")


# ==============================================================================
# TestDatabaseHelpers
# ==============================================================================

class TestDatabaseHelpers:

    def test_get_incumbent_raises_when_empty(self, tmp_path):
        db = str(tmp_path / "p.db")
        make_agents_db(db)
        with pytest.raises(RuntimeError, match="No promoted agent"):
            get_incumbent(db)

    def test_get_incumbent_returns_most_recent(self, tmp_path):
        db = str(tmp_path / "p.db")
        make_agents_db(db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO agents VALUES (1, 'ckpt1.ckpt', '2024-01-01', '2024-01-10T00:00:00Z')")
        conn.execute("INSERT INTO agents VALUES (2, 'ckpt2.ckpt', '2024-01-05', '2024-01-15T00:00:00Z')")
        conn.commit()
        conn.close()
        agent_id, path = get_incumbent(db)
        assert agent_id == 2
        assert path == "ckpt2.ckpt"

    def test_get_incumbent_ignores_unpromoted(self, tmp_path):
        db = str(tmp_path / "p.db")
        make_agents_db(db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO agents VALUES (1, 'ckpt1.ckpt', '2024-01-01', '2024-01-10T00:00:00Z')")
        conn.execute("INSERT INTO agents VALUES (2, 'ckpt2.ckpt', '2024-01-05', NULL)")
        conn.commit()
        conn.close()
        agent_id, _ = get_incumbent(db)
        assert agent_id == 1

    def test_get_candidate_raises_when_no_checkpoints(self, tmp_path):
        db = str(tmp_path / "p.db")
        make_agents_db(db)
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        with pytest.raises(RuntimeError, match="No unpromoted checkpoint"):
            get_candidate(db, str(ckpt_dir))

    def test_get_candidate_raises_when_all_promoted(self, tmp_path):
        db = str(tmp_path / "p.db")
        make_agents_db(db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO agents VALUES (3, 'agent_3.ckpt', NULL, '2024-01-01T00:00:00Z')")
        conn.commit()
        conn.close()
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        (ckpt_dir / "agent_3.ckpt").touch()
        with pytest.raises(RuntimeError, match="No unpromoted checkpoint"):
            get_candidate(db, str(ckpt_dir))

    def test_get_candidate_returns_highest_unpromoted_id(self, tmp_path):
        db = str(tmp_path / "p.db")
        make_agents_db(db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO agents VALUES (1, 'agent_1.ckpt', NULL, '2024-01-01T00:00:00Z')")
        conn.commit()
        conn.close()
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        (ckpt_dir / "agent_1.ckpt").touch()
        (ckpt_dir / "agent_5.ckpt").touch()
        (ckpt_dir / "agent_3.ckpt").touch()
        agent_id, path = get_candidate(db, str(ckpt_dir))
        assert agent_id == 5
        assert "agent_5.ckpt" in path

    def test_register_agent_inserts_row(self, tmp_path):
        db = str(tmp_path / "p.db")
        make_agents_db(db)
        register_agent(db, 10, "/ckpts/agent_10.ckpt")
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT agent_id, checkpoint_path FROM agents WHERE agent_id=10").fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 10
        assert row[1] == "/ckpts/agent_10.ckpt"

    def test_register_agent_idempotent(self, tmp_path):
        db = str(tmp_path / "p.db")
        make_agents_db(db)
        register_agent(db, 10, "/ckpts/agent_10.ckpt")
        register_agent(db, 10, "/ckpts/agent_10.ckpt")  # should not raise
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM agents WHERE agent_id=10").fetchone()[0]
        conn.close()
        assert count == 1


# ==============================================================================
# TestShardHelpers
# ==============================================================================

class TestShardHelpers:

    def test_count_lines_empty_dir(self, tmp_path):
        assert count_lines(str(tmp_path)) == 0

    def test_count_lines_single_file(self, tmp_path):
        shard = tmp_path / "out-0.jsonl"
        shard.write_text("line1\nline2\nline3\nline4\nline5\n")
        assert count_lines(str(tmp_path)) == 5

    def test_count_lines_multiple_files(self, tmp_path):
        (tmp_path / "out-0.jsonl").write_text("a\nb\nc\n")
        (tmp_path / "out-1.jsonl").write_text("x\ny\nz\nw\n")
        assert count_lines(str(tmp_path)) == 7

    def test_count_lines_ignores_non_jsonl(self, tmp_path):
        (tmp_path / "out-0.jsonl").write_text("line\n")
        (tmp_path / "notes.txt").write_text("a\nb\nc\n")
        assert count_lines(str(tmp_path)) == 1

    def test_retire_shards_moves_files(self, tmp_path):
        src = tmp_path / "shards"
        dst = tmp_path / "retired"
        src.mkdir()
        dst.mkdir()
        (src / "out-0.jsonl").write_text("data\n")
        (src / "out-1.jsonl").write_text("data\n")
        retire_shards(str(src), str(dst))
        assert len(list(src.glob("*.jsonl"))) == 0
        assert len(list(dst.glob("*.jsonl"))) == 2

    def test_retire_shards_creates_dest_dir(self, tmp_path):
        src = tmp_path / "shards"
        dst = tmp_path / "retired" / "nested"
        src.mkdir()
        (src / "out-0.jsonl").write_text("data\n")
        retire_shards(str(src), str(dst))
        assert dst.exists()
        assert len(list(dst.glob("*.jsonl"))) == 1

    def test_retire_shards_collision_gets_unique_name(self, tmp_path):
        src = tmp_path / "shards"
        dst = tmp_path / "retired"
        src.mkdir()
        dst.mkdir()
        # Pre-populate retired dir with a file of the same name.
        (dst / "out-0.jsonl").write_text("old\n")
        (src / "out-0.jsonl").write_text("new\n")
        retire_shards(str(src), str(dst))
        jsonl_files = list(dst.glob("*.jsonl"))
        assert len(jsonl_files) == 2, "both old and renamed new file should exist"


# ==============================================================================
# TestHandlerLifecycle
# ==============================================================================

class TestHandlerLifecycle:

    @patch("mcts_tilers.orchestrator.time.sleep")
    @patch("mcts_tilers.orchestrator.subprocess.Popen")
    def test_start_handlers_launches_correct_number(self, mock_popen, mock_sleep):
        mock_popen.return_value = MagicMock(pid=1234)
        procs = start_handlers("handler.py", "/ckpt.pt", "gather", 2048, 3, warmup_secs=0.0)
        assert len(procs) == 3
        assert mock_popen.call_count == 3

    @patch("mcts_tilers.orchestrator.time.sleep")
    @patch("mcts_tilers.orchestrator.subprocess.Popen")
    def test_start_handlers_passes_correct_args(self, mock_popen, mock_sleep):
        mock_popen.return_value = MagicMock(pid=1234)
        start_handlers("handler.py", "/weights.pt", "eval", 1024, 1, warmup_secs=0.0)
        args_used = mock_popen.call_args[0][0]
        assert "--arena_tag" in args_used
        assert "eval" in args_used
        assert "--weights" in args_used
        assert "/weights.pt" in args_used
        assert "--handler_id" in args_used

    def test_stop_handlers_terminates_all(self):
        procs = [MagicMock(), MagicMock()]
        stop_handlers(procs)
        for p in procs:
            p.terminate.assert_called_once()

    def test_stop_handlers_kills_on_timeout(self):
        proc = MagicMock()
        # First wait (with timeout) raises; second wait (after kill) returns normally.
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="handler.py", timeout=10),
            None,
        ]
        stop_handlers([proc])
        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2  # first (raises) + after kill


# ==============================================================================
# TestPromoteCandidate
# ==============================================================================

class TestPromoteCandidate:

    def _setup_db(self, tmp_path, solutions):
        """
        solutions: list of (env_id, c_depth, i_depth) where depth is float or None.
        Returns db path with agents 1 (candidate) and 2 (incumbent) having the
        given solution rows.
        """
        db = str(tmp_path / "p.db")
        make_full_db(db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO agents VALUES (1, 'c.ckpt', NULL, NULL)")
        conn.execute("INSERT INTO agents VALUES (2, 'i.ckpt', NULL, '2024-01-01T00:00:00Z')")
        for env_id, c_depth, i_depth in solutions:
            conn.execute(
                "INSERT OR IGNORE INTO solutions VALUES (1, ?, '[]', ?, '2024-01-01T00:00:00Z')",
                (env_id, c_depth),
            )
            conn.execute(
                "INSERT OR IGNORE INTO solutions VALUES (2, ?, '[]', ?, '2024-01-01T00:00:00Z')",
                (env_id, i_depth),
            )
        conn.commit()
        conn.close()
        return db

    def test_candidate_wins_majority_is_promoted(self, tmp_path):
        # 4 wins, 1 loss → win rate 0.8 > 0.55 threshold
        db = self._setup_db(tmp_path, [
            (1, 1.0, 2.0),  # c wins (lower depth)
            (2, 1.0, 2.0),
            (3, 1.0, 2.0),
            (4, 1.0, 2.0),
            (5, 3.0, 2.0),  # i wins
        ])
        promoted, c_wins, i_wins, draws, total = promote_candidate(db, 1, 2, 0.55)
        assert promoted is True
        assert c_wins == 4
        assert i_wins == 1
        assert draws == 0
        assert total == 5

    def test_candidate_loses_is_not_promoted(self, tmp_path):
        # 1 win, 4 losses → win rate 0.2 < 0.55
        db = self._setup_db(tmp_path, [
            (1, 1.0, 2.0),
            (2, 3.0, 2.0),
            (3, 3.0, 2.0),
            (4, 3.0, 2.0),
            (5, 3.0, 2.0),
        ])
        promoted, _, _, _, _ = promote_candidate(db, 1, 2, 0.55)
        assert promoted is False

    def test_equal_depths_counted_as_draw(self, tmp_path):
        db = self._setup_db(tmp_path, [(1, 2.0, 2.0)])
        promoted, c_wins, i_wins, draws, total = promote_candidate(db, 1, 2, 0.55)
        assert c_wins == 0
        assert i_wins == 0
        assert draws == 1

    def test_candidate_finished_incumbent_did_not(self, tmp_path):
        db = self._setup_db(tmp_path, [(1, 2.0, None)])
        promoted, c_wins, i_wins, draws, total = promote_candidate(db, 1, 2, 0.0)
        assert c_wins == 1
        assert i_wins == 0

    def test_incumbent_finished_candidate_did_not(self, tmp_path):
        db = self._setup_db(tmp_path, [(1, None, 2.0)])
        _, c_wins, i_wins, draws, total = promote_candidate(db, 1, 2, 0.55)
        assert c_wins == 0
        assert i_wins == 1

    def test_both_failed_counted_as_draw(self, tmp_path):
        db = self._setup_db(tmp_path, [(1, None, None)])
        _, c_wins, i_wins, draws, total = promote_candidate(db, 1, 2, 0.55)
        assert draws == 1

    def test_writes_evaluations_row(self, tmp_path):
        db = self._setup_db(tmp_path, [(1, 1.0, 2.0)])
        promote_candidate(db, 1, 2, 0.0)  # any threshold
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT candidate_agent_id, incumbent_agent_id FROM evaluations").fetchone()
        conn.close()
        assert row == (1, 2)

    def test_promoted_candidate_gets_promoted_at_set(self, tmp_path):
        db = self._setup_db(tmp_path, [(1, 1.0, 2.0)])  # candidate wins
        promote_candidate(db, 1, 2, 0.0)
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT promoted_at FROM agents WHERE agent_id=1").fetchone()
        conn.close()
        assert row[0] is not None

    def test_not_promoted_candidate_promoted_at_remains_null(self, tmp_path):
        db = self._setup_db(tmp_path, [(1, 3.0, 2.0)])  # incumbent wins
        promote_candidate(db, 1, 2, 0.55)
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT promoted_at FROM agents WHERE agent_id=1").fetchone()
        conn.close()
        assert row[0] is None


# ==============================================================================
# TestGatherPhase
# ==============================================================================

class MockExecutor:
    """Synchronous executor that calls the submitted function immediately."""

    def __init__(self, max_workers: int):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, **kwargs):
        f: Future = Future()
        try:
            f.set_result(fn(**kwargs))
        except Exception as exc:
            f.set_exception(exc)
        return f


class TestGatherPhase:

    def _run_gather(self, tmp_path, gatherer_retval=(1.0, 2.0, True)):
        shard_dir = str(tmp_path / "shards")
        os.makedirs(shard_dir, exist_ok=True)
        args = make_args(shard_dir=shard_dir, num_gatherers=1, gather_threshold=1)
        state = {"phase": "gathering", "iteration": 0, "candidate_id": None, "candidate_path": None}
        state_path = str(tmp_path / "state.json")

        with (
            patch("mcts_tilers.orchestrator.ProcessPoolExecutor", MockExecutor),
            patch("mcts_tilers.orchestrator.start_handlers", return_value=[MagicMock()]) as mock_start,
            patch("mcts_tilers.orchestrator.stop_handlers") as mock_stop,
            patch("mcts_tilers.orchestrator.run_gatherer", return_value=gatherer_retval),
            patch("mcts_tilers.orchestrator.count_lines", return_value=999),
            patch("mcts_tilers.orchestrator.save_state"),
        ):
            run_gather_phase(args, state, state_path, "/ckpt.pt")
            return mock_start, mock_stop, state

    def test_gather_phase_starts_handlers(self, tmp_path):
        mock_start, _, _ = self._run_gather(tmp_path)
        mock_start.assert_called_once()

    def test_gather_phase_stops_handlers(self, tmp_path):
        _, mock_stop, _ = self._run_gather(tmp_path)
        mock_stop.assert_called_once()

    def test_gather_phase_advances_state_to_training(self, tmp_path):
        _, _, state = self._run_gather(tmp_path)
        assert state["phase"] == "training"

    def test_gather_phase_stops_handlers_on_exception(self, tmp_path):
        shard_dir = str(tmp_path / "shards")
        os.makedirs(shard_dir, exist_ok=True)
        args = make_args(shard_dir=shard_dir, num_gatherers=1, gather_threshold=1)
        state = {"phase": "gathering", "iteration": 0, "candidate_id": None, "candidate_path": None}

        with (
            patch("mcts_tilers.orchestrator.ProcessPoolExecutor", MockExecutor),
            patch("mcts_tilers.orchestrator.start_handlers", return_value=[MagicMock()]),
            patch("mcts_tilers.orchestrator.stop_handlers") as mock_stop,
            patch("mcts_tilers.orchestrator.run_gatherer", side_effect=RuntimeError("boom")),
            patch("mcts_tilers.orchestrator.count_lines", return_value=999),
            patch("mcts_tilers.orchestrator.save_state"),
        ):
            # The future's exception is caught inside the loop, so gather_phase still completes.
            run_gather_phase(args, state, str(tmp_path / "state.json"), "/ckpt.pt")
            mock_stop.assert_called_once()


# ==============================================================================
# TestTrainPhase
# ==============================================================================

class TestTrainPhase:

    def _run_train(self, tmp_path, returncode=0):
        db = str(tmp_path / "p.db")
        make_agents_db(db)
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        (ckpt_dir / "agent_1.ckpt").touch()

        args = make_args(db_path=db, checkpoint_dir=str(ckpt_dir),
                         shard_dir=str(tmp_path / "shards"),
                         retired_shard_dir=str(tmp_path / "retired"))
        state = {"phase": "training", "iteration": 1, "candidate_id": None, "candidate_path": None}
        state_path = str(tmp_path / "state.json")

        mock_result = MagicMock()
        mock_result.returncode = returncode

        with (
            patch("mcts_tilers.orchestrator.subprocess.run", return_value=mock_result),
            patch("mcts_tilers.orchestrator.retire_shards"),
            patch("mcts_tilers.orchestrator.save_state"),
        ):
            run_train_phase(args, state, state_path)
        return state

    def test_train_phase_advances_state_to_evaluating(self, tmp_path):
        state = self._run_train(tmp_path)
        assert state["phase"] == "evaluating"
        assert state["candidate_id"] is not None

    def test_train_phase_raises_on_nonzero_exit(self, tmp_path):
        with pytest.raises(RuntimeError, match="Trainer exited"):
            self._run_train(tmp_path, returncode=1)


# ==============================================================================
# TestEvaluatePhase
# ==============================================================================

class TestEvaluatePhase:

    def _run_evaluate(self, tmp_path, promote_return=None):
        db = str(tmp_path / "p.db")
        make_full_db(db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO agents VALUES (1, 'inc.ckpt', NULL, '2024-01-01T00:00:00Z')")
        conn.execute("INSERT INTO agents VALUES (2, 'cand.ckpt', NULL, NULL)")
        conn.commit()
        conn.close()

        if promote_return is None:
            promote_return = (False, 2, 3, 0, 5)

        args = make_args(db_path=db)
        state = {"phase": "evaluating", "iteration": 1, "candidate_id": 2, "candidate_path": "cand.ckpt"}
        state_path = str(tmp_path / "state.json")

        mock_handlers = [MagicMock()]
        with (
            patch("mcts_tilers.orchestrator.start_handlers", return_value=mock_handlers),
            patch("mcts_tilers.orchestrator.stop_handlers") as mock_stop,
            patch("mcts_tilers.orchestrator.run_evaluator") as mock_eval,
            patch("mcts_tilers.orchestrator.promote_candidate", return_value=promote_return),
            patch("mcts_tilers.orchestrator.save_state"),
        ):
            run_evaluate_phase(args, state, state_path)
            return state, mock_eval, mock_stop

    def test_evaluate_phase_calls_run_evaluator_twice(self, tmp_path):
        _, mock_eval, _ = self._run_evaluate(tmp_path)
        assert mock_eval.call_count == 2

    def test_evaluate_phase_evaluates_candidate_and_incumbent(self, tmp_path):
        _, mock_eval, _ = self._run_evaluate(tmp_path)
        agent_ids = {c.kwargs["agent_id"] for c in mock_eval.call_args_list}
        assert agent_ids == {1, 2}

    def test_evaluate_phase_advances_state_to_gathering(self, tmp_path):
        state, _, _ = self._run_evaluate(tmp_path)
        assert state["phase"] == "gathering"
        assert state["iteration"] == 2
        assert state["candidate_id"] is None
        assert state["candidate_path"] is None

    def test_evaluate_phase_stops_handlers_on_exception(self, tmp_path):
        db = str(tmp_path / "p.db")
        make_full_db(db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO agents VALUES (1, 'inc.ckpt', NULL, '2024-01-01T00:00:00Z')")
        conn.execute("INSERT INTO agents VALUES (2, 'cand.ckpt', NULL, NULL)")
        conn.commit()
        conn.close()

        args = make_args(db_path=db)
        state = {"phase": "evaluating", "iteration": 1, "candidate_id": 2, "candidate_path": "cand.ckpt"}

        with (
            patch("mcts_tilers.orchestrator.start_handlers", return_value=[MagicMock()]),
            patch("mcts_tilers.orchestrator.stop_handlers") as mock_stop,
            patch("mcts_tilers.orchestrator.run_evaluator", side_effect=RuntimeError("gpu error")),
            patch("mcts_tilers.orchestrator.save_state"),
        ):
            with pytest.raises(RuntimeError, match="gpu error"):
                run_evaluate_phase(args, state, str(tmp_path / "state.json"))
            mock_stop.assert_called()
