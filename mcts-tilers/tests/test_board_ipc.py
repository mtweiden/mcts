"""
End-to-end Rust↔Python IPC test over the **10-channel board**.

A handler subprocess runs the refactored `handler.do_work` driven by a
lightweight numpy `DummyAgent` (no torch / GPU / checkpoint).  The main
process calls the Rust `submit_and_collect` helper, which packs a real
env's board into a slot, submits it, and waits for the handler to write
priors back — exercising:

    Rust gatherer/client  →  slot.board (10×i16)  →  PySlotView.board()
        →  handler.build_board_batch (reshape)  →  DummyAgent.infer
        →  write_priors_values  →  Rust collect.

Requires the extension built with `--features python,test-helpers`.
"""

import multiprocessing
import time

import numpy as np
import pytest


ARENA_NAME = "test_board_ipc"
NUM_SLOTS = 4
NUM_HANDLERS = 1
LOOKAHEAD = 1


def handler_process(arena_name: str, num_slots: int, num_handlers: int):
    """Standalone handler: opens the arena and pumps it with a dummy agent."""
    from mcts_tilers import PyArena, CELL_FIELDS
    from mcts_tilers.handler import do_work

    class DummyAgent:
        """Mimics `tile.Agent`'s infer contract; returns uniform masked
        priors + a constant value.  Asserts the board shape the Rust side
        delivered so a malformed board fails loudly."""
        lookahead = LOOKAHEAD

        def infer(self, *, boards, heights, widths, num_ancillas, action_masks, device):
            assert boards.ndim == 4, f"board must be 4D, got {boards.shape}"
            assert boards.shape[1] == self.lookahead + 1, \
                f"expected {self.lookahead + 1} layers, got {boards.shape[1]}"
            assert boards.shape[-1] == CELL_FIELDS, \
                f"expected {CELL_FIELDS} channels, got {boards.shape[-1]}"

            b, _n = action_masks.shape
            priors = action_masks.astype(np.float32)
            sums = priors.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            priors = priors / sums
            values = np.full(b, 0.42, dtype=np.float32)
            return priors, values

    arena = PyArena(arena_name, num_slots, num_handlers)
    do_work(arena, DummyAgent(), "cpu", lookahead=LOOKAHEAD)


class TestBoardIpc:
    @pytest.fixture(autouse=True)
    def setup_arena(self):
        from mcts_tilers import PyArena

        full_name = f"{ARENA_NAME}_{NUM_SLOTS}_{NUM_HANDLERS}"
        self.arena = PyArena(full_name, NUM_SLOTS, NUM_HANDLERS)

        self.handler_proc = multiprocessing.Process(
            target=handler_process,
            args=(full_name, NUM_SLOTS, NUM_HANDLERS),
            daemon=True,
        )
        self.handler_proc.start()
        time.sleep(0.5)  # let the handler open the arena

        yield

        self.handler_proc.terminate()
        self.handler_proc.join(timeout=3.0)

    def test_board_roundtrip_uniform_priors(self):
        priors_vecs, values = self.arena.submit_and_collect(
            h=3, w=3, num_blanks=1, num_objectives=1, seed=42,
        )

        assert len(values) == 1
        assert len(priors_vecs) == 1
        np.testing.assert_allclose(values[0], 0.42, atol=1e-5)

        priors = np.array(priors_vecs[0])
        valid = priors > 0
        assert valid.any(), "expected at least one valid action from the env"
        np.testing.assert_allclose(priors.sum(), 1.0, atol=1e-4)
        # Dummy agent puts uniform mass on the masked-valid actions.
        np.testing.assert_allclose(priors[valid], 1.0 / valid.sum(), atol=1e-5)

    def test_board_roundtrip_various_grids(self):
        for h, w, nb, no in [(3, 3, 1, 1), (4, 4, 2, 2), (5, 5, 3, 2)]:
            priors_vecs, values = self.arena.submit_and_collect(
                h=h, w=w, num_blanks=nb, num_objectives=no, seed=h * 100 + w,
            )
            assert len(values) == 1
            np.testing.assert_allclose(values[0], 0.42, atol=1e-5)
            priors = np.array(priors_vecs[0])
            assert (priors > 0).any()
            np.testing.assert_allclose(priors.sum(), 1.0, atol=1e-4)


# ------------------------------------------------------------------------------
# Real tile.Agent over the same IPC board path
# ------------------------------------------------------------------------------
def real_agent_handler_process(arena_name, num_slots, num_handlers):
    """Handler driven by the real (untrained) tile.Agent via TorchAgentRunner."""
    from mcts_tilers import PyArena
    from mcts_tilers.handler import do_work, TorchAgentRunner
    from tile.agent import Agent

    model = Agent(embedding_dim=32, num_layers=2, lookahead=LOOKAHEAD)
    model.to("cpu")
    arena = PyArena(arena_name, num_slots, num_handlers)
    do_work(arena, TorchAgentRunner(model), "cpu", lookahead=LOOKAHEAD)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("tile") is None,
    reason="tile package (the real agent) not importable",
)
class TestRealAgentBoardIpc:
    """Wires the real tile.Agent through the board IPC end to end."""

    @pytest.fixture(autouse=True)
    def setup_arena(self):
        from mcts_tilers import PyArena

        full_name = f"{ARENA_NAME}_real_{NUM_SLOTS}_{NUM_HANDLERS}"
        self.arena = PyArena(full_name, NUM_SLOTS, NUM_HANDLERS)
        self.handler_proc = multiprocessing.Process(
            target=real_agent_handler_process,
            args=(full_name, NUM_SLOTS, NUM_HANDLERS),
            daemon=True,
        )
        self.handler_proc.start()
        time.sleep(2.0)  # the real model takes longer to construct
        yield
        self.handler_proc.terminate()
        self.handler_proc.join(timeout=5.0)

    def test_real_agent_produces_valid_distribution(self):
        priors_vecs, values = self.arena.submit_and_collect(
            h=4, w=4, num_blanks=2, num_objectives=2, seed=123,
        )
        assert len(values) == 1
        assert len(priors_vecs) == 1
        # The (untrained) model still returns a proper softmax distribution.
        priors = np.array(priors_vecs[0])
        assert np.isfinite(priors).all()
        assert (priors >= 0).all()
        np.testing.assert_allclose(priors.sum(), 1.0, atol=1e-3)
        assert (priors > 0).any(), "model put zero mass on every action"
        # Value is a finite scalar in a sane range.
        assert np.isfinite(values[0])
