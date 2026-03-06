"""
Integration test: Rust gather (TilersIpcClient) ↔ Python handler round-trip.

Runs the handler in a subprocess so the Rust-side wait_done can block
the main process without deadlocking.
"""

import multiprocessing
import sys
import time

import numpy as np
import pytest


ARENA_NAME = "test_integ"
NUM_SLOTS = 4
NUM_HANDLERS = 1
HANDLER_ID = 0
TIMEOUT = 15.0


# ------------------------------------------------------------------------------
# Handler subprocess
# ------------------------------------------------------------------------------
def handler_process(arena_name: str, num_slots: int, num_handlers: int, handler_id: int):
    """
    Standalone handler process.
    Opens its own handle to the shared memory arena and processes slots.
    Uses a fake model that returns uniform priors and value=0.42.
    """
    import numpy as np
    import torch

    from mcts_tilers import PyArena, NUM_ACTIONS
    from mcts_tilers.handler import (
        unpack_placement_batch,
        unpack_objectives_batch,
        build_boards,
    )

    class FakeModel:
        def to(self, device):
            return self

        def infer(self, *, boards, heights, widths, num_ancillas, action_masks):
            batch = boards.shape[0]
            priors = action_masks.float()
            sums = priors.sum(dim=1, keepdim=True).clamp(min=1.0)
            priors = priors / sums
            values = torch.full((batch, 1), 0.42, dtype=torch.float32)
            return priors, values

    arena = PyArena(arena_name, num_slots, num_handlers)
    print(f"[handler] Started handler {handler_id} for arena {arena_name}", flush=True)
    model = FakeModel()

    # Process up to 20 slots then exit
    processed = 0
    max_iters = 20
    idle_count = 0
    max_idle = 5000  # 5 seconds of no work → exit

    while processed < max_iters and idle_count < max_idle:
        sv = arena.try_pop_ready_view(handler=handler_id, clear_outputs=True)
        if sv is None:
            time.sleep(0.00001)
            idle_count += 1
            continue

        idle_count = 0

        try:
            sv.set_handler_start_time()
            b = sv.b()

            placement_raw = np.asarray(sv.placement())
            objectives_raw = np.asarray(sv.objectives())
            action_mask_raw = np.asarray(sv.action_mask())
            h_all = np.asarray(sv.h())
            w_all = np.asarray(sv.w())
            ancillas_all = np.asarray(sv.num_ancillas())
            nq_all = np.asarray(sv.num_qubits())
            nl_all = np.asarray(sv.num_layers())
            no_all = np.asarray(sv.num_objectives())

            qubit_ids, qubit_oris = unpack_placement_batch(placement_raw, nq_all)
            obj_layers = unpack_objectives_batch(objectives_raw, nl_all, no_all)
            boards_np = build_boards(
                qubit_ids, qubit_oris, nq_all, obj_layers, nl_all, no_all,
            )

            boards_t = torch.from_numpy(np.ascontiguousarray(boards_np))
            masks_t = torch.from_numpy(
                np.ascontiguousarray(action_mask_raw.astype(bool, copy=False))
            )
            heights_t = torch.from_numpy(h_all.astype(np.int32, copy=False))
            widths_t = torch.from_numpy(w_all.astype(np.int32, copy=False))
            ancillas_t = torch.from_numpy(ancillas_all.astype(np.int32, copy=False))

            with torch.no_grad():
                priors_t, values_t = model.infer(
                    boards=boards_t,
                    heights=heights_t,
                    widths=widths_t,
                    num_ancillas=ancillas_t,
                    action_masks=masks_t,
                )

            priors_np = priors_t.detach().cpu().numpy()
            values_np = values_t.squeeze(-1).detach().cpu().numpy()

            sv.write_priors_values(
                np.ascontiguousarray(priors_np.astype(np.float32)),
                np.ascontiguousarray(values_np.astype(np.float32)),
            )
            sv.mark_done()
            processed += 1

        except Exception as e:
            print(f"Handler error: {e}", file=sys.stderr, flush=True)
            try:
                sv.mark_done()
            except Exception:
                pass


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

class TestGatherHandlerIntegration:

    @pytest.fixture(autouse=True)
    def setup_arena(self):
        from mcts_tilers import PyArena

        full_name = f"{ARENA_NAME}_{NUM_SLOTS}_{NUM_HANDLERS}"
        self.full_name = full_name

        # Create the arena in the main process first
        self.arena = PyArena(full_name, NUM_SLOTS, NUM_HANDLERS)

        # Start handler as a separate process
        self.handler_proc = multiprocessing.Process(
            target=handler_process,
            args=(full_name, NUM_SLOTS, NUM_HANDLERS, HANDLER_ID),
            daemon=True,
        )
        self.handler_proc.start()

        # Give the handler a moment to open the arena
        time.sleep(0.5)

        yield

        self.handler_proc.terminate()
        self.handler_proc.join(timeout=3.0)

    def test_submit_and_collect(self):
        from mcts_tilers import NUM_ACTIONS

        priors_vecs, values = self.arena.submit_and_collect(
            h=3, w=3, num_blanks=1, num_objectives=1, seed=42,
        )

        assert len(values) == 1
        assert len(priors_vecs) == 1

        np.testing.assert_allclose(values[0], 0.42, atol=1e-5)

        priors = np.array(priors_vecs[0])
        valid_mask = priors > 0
        assert valid_mask.any(), "should have at least one valid action"
        np.testing.assert_allclose(priors.sum(), 1.0, atol=1e-4)

        valid_priors = priors[valid_mask]
        expected = 1.0 / len(valid_priors)
        np.testing.assert_allclose(valid_priors, expected, atol=1e-5)

    def test_multiple_submissions(self):
        configs = [
            (3, 3, 1, 1, 42),
            (4, 4, 2, 2, 100),
            (3, 4, 1, 2, 7),
        ]
        for h, w, nb, no, seed in configs:
            priors_vecs, values = self.arena.submit_and_collect(
                h=h, w=w, num_blanks=nb, num_objectives=no, seed=seed,
            )
            assert len(values) == 1
            np.testing.assert_allclose(values[0], 0.42, atol=1e-5)

            priors = np.array(priors_vecs[0])
            valid_mask = priors > 0
            assert valid_mask.any()
            np.testing.assert_allclose(priors.sum(), 1.0, atol=1e-4)

    def test_various_grid_sizes(self):
        for h, w in [(2, 3), (3, 3), (4, 4), (2, 5), (5, 5)]:
            seed = h * 100 + w
            try:
                priors_vecs, values = self.arena.submit_and_collect(
                    h=h, w=w, num_blanks=1, num_objectives=1, seed=seed,
                )
            except Exception as e:
                pytest.fail(f"failed on ({h},{w}): {e}")

            assert len(values) == 1
            assert len(priors_vecs) == 1