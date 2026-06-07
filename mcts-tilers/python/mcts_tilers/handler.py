import json
import logging
import os
from pathlib import Path
from argparse import ArgumentParser

import time
import numpy as np

from mcts_tilers import PyArena
from mcts_tilers import LOOKAHEAD_MAX, GRID_MAX, CELL_FIELDS

# `torch` + the real `tile.Agent` are imported lazily in `main()` so this
# module (and `do_work`, which takes the agent as a parameter) can be
# driven by a lightweight dummy agent in tests without a GPU / checkpoint.

# Rust opt-in for the build_boards / unpack_* hot path. The actual
# rebinding happens at the bottom of this file once the Python defs
# exist; this constant just controls which branch wins. Set
# MCTS_HANDLER_RUST=1 to flip the handler over to the Rust
# implementations exposed by the mcts_tilers PyO3 module.
_USE_RUST_HANDLER = os.environ.get("MCTS_HANDLER_RUST", "0") == "1"

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
BATCH_TIMEOUT = 0.002
MAX_SLOTS_PER_BATCH = 64  # max slots to gather before one GPU call

# Note: main carried `_bucket_bs` / `_pad_to_bs` helpers for
# torch.compile(mode='reduce-overhead') static-shape bucketing on the
# GPU inference path.  Those belong with the perf machinery that the
# board-migration merge intentionally dropped — they pair with
# bf16 autocast + h_int/w_int kwargs + handler metrics, all of which
# need re-applying as a follow-up.

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)


def latest_checkpoint() -> str | None:
    ckpt_path = "/pscratch/sd/m/mtweiden/tile_mcts/checkpoints"
    files = sorted(
        [str(x) for x in Path(ckpt_path).glob("*.ckpt")],
        key= lambda x: int(x.split("_")[-1].split(".")[0])
    )
    if len(files) == 0:
        return None
    return files[-1]


# ------------------------------------------------------------------------------
# Observation board batching
#
# The Rust side now packs the 10-channel `tilers::rl::board` directly into
# the slot (`sv.board()` -> shape `(b, BOARD_MAX)` int16, where
# `BOARD_MAX = LOOKAHEAD_MAX * GRID_MAX * CELL_FIELDS`).  There is no more
# placement/objective reconstruction (`build_boards` is gone) — we just
# reshape and slice to the agent's expected
# `(batch, lookahead + 1, GRID_MAX, CELL_FIELDS)`.
# ------------------------------------------------------------------------------
def build_board_batch(slot_views, lookahead: int):
    """Collect every valid slot view's board + metadata into one batch.

    Returns ``(boards, heights, widths, num_ancillas, action_masks,
    slot_batch_sizes, valid_slot_views)`` where ``boards`` has shape
    ``(total, lookahead + 1, GRID_MAX, CELL_FIELDS)`` (int32) and
    ``action_masks`` shape ``(total, NUM_ACTIONS)`` (bool).
    """
    num_layers = lookahead + 1
    assert num_layers <= LOOKAHEAD_MAX, (
        f"lookahead+1={num_layers} exceeds LOOKAHEAD_MAX={LOOKAHEAD_MAX}"
    )

    raw_boards, mask_list = [], []
    h_list, w_list, anc_list = [], [], []
    slot_batch_sizes, valid_slot_views = [], []

    for sv in slot_views:
        b_i = sv.b()
        if b_i <= 0:
            sv.mark_done()
            continue
        slot_batch_sizes.append(b_i)
        valid_slot_views.append(sv)

        # (b_i, BOARD_MAX) int16 -> (b_i, LOOKAHEAD_MAX, GRID_MAX, CELL_FIELDS)
        # -> slice to the agent's num_layers.  Cells past each env's h*w are
        # PAD (zeros); we trim to the batch's max h*w below.
        raw = np.asarray(sv.board()[:b_i], copy=True)
        board = raw.reshape(b_i, LOOKAHEAD_MAX, GRID_MAX, CELL_FIELDS)
        raw_boards.append(board[:, :num_layers].astype(np.int32, copy=False))

        mask_list.append(np.asarray(sv.action_mask()[:b_i], copy=True))
        h_list.append(np.asarray(sv.h()[:b_i], copy=True))
        w_list.append(np.asarray(sv.w()[:b_i], copy=True))
        anc_list.append(np.asarray(sv.num_ancillas()[:b_i], copy=True))

    if not valid_slot_views:
        return None

    heights = np.concatenate(h_list).astype(np.int32, copy=False)
    widths = np.concatenate(w_list).astype(np.int32, copy=False)
    num_ancillas = np.concatenate(anc_list).astype(np.int32, copy=False)
    action_masks = np.concatenate(mask_list, axis=0).astype(bool, copy=False)

    # The agent expects (b, num_layers, h*w, CELL_FIELDS) with h*w =
    # the batch-max grid area; GRID_MAX padding beyond that is dropped.
    max_hw = max(1, int((heights.astype(np.int64) * widths.astype(np.int64)).max()))
    boards = np.concatenate([b[:, :, :max_hw, :] for b in raw_boards], axis=0)

    return (
        boards,
        heights,
        widths,
        num_ancillas,
        action_masks,
        slot_batch_sizes,
        valid_slot_views,
    )


# ------------------------------------------------------------------------------
# Inference loop
# ------------------------------------------------------------------------------
def do_work(arena: PyArena, agent, device: str, lookahead: int) -> None:
    """Pump the arena: batch ready slots, run ``agent.infer`` over the board,
    write priors/values back.

    ``agent`` must expose
    ``infer(boards, heights, widths, num_ancillas, action_masks) ->
    (priors, values)`` (tensors), matching ``tile.Agent`` — a dummy agent
    with the same signature works for tests.
    """
    while True:
        try:
            first_sv = arena.pop_ready_view(clear_outputs=True)
            first_sv.set_handler_start_time()
        except KeyboardInterrupt:
            break

        slot_views = [first_sv]

        fill_start = time.monotonic()
        while len(slot_views) < MAX_SLOTS_PER_BATCH and (time.monotonic() - fill_start) < BATCH_TIMEOUT:
            sv = arena.try_pop_ready_view(clear_outputs=True)
            if sv is None:
                time.sleep(0.0005)
                continue
            sv.set_handler_start_time()
            slot_views.append(sv)

        try:
            batched = build_board_batch(slot_views, lookahead)
            if batched is None:
                continue
            (boards, heights, widths, num_ancillas,
             action_masks, slot_batch_sizes, valid_slot_views) = batched

            priors_np, values_np = agent.infer(
                boards=boards,
                heights=heights,
                widths=widths,
                num_ancillas=num_ancillas,
                action_masks=action_masks,
                device=device,
            )

            # Write back per-slot.
            idx = 0
            for slot_len, sv in zip(slot_batch_sizes, valid_slot_views):
                pri_slice = np.ascontiguousarray(priors_np[idx: idx + slot_len])
                val_slice = np.ascontiguousarray(values_np[idx: idx + slot_len])
                idx += slot_len
                sv.write_priors_values(
                    pri_slice.astype(np.float32, copy=False),
                    val_slice.astype(np.float32, copy=False),
                )
                sv.mark_done()

        except KeyboardInterrupt:
            for sv in slot_views:
                try:
                    sv.mark_done()
                except Exception:
                    pass
            break

        except Exception as e:
            # Never leave a slot un-marked: a requester blocking on this
            # slot would hang forever.  Log and release every slot.
            logging.error("handler inference failed: %s", e, exc_info=True)
            for sv in slot_views:
                try:
                    sv.mark_done()
                except Exception:
                    pass


# ------------------------------------------------------------------------------
# Real-model adapter: wraps tile.Agent.infer to the (numpy in, numpy out)
# contract `do_work` uses.
# ------------------------------------------------------------------------------
class TorchAgentRunner:
    def __init__(self, model):
        import torch  # noqa: F401
        self.model = model
        self.lookahead = model.lookahead

    def infer(self, boards, heights, widths, num_ancillas, action_masks, device):
        import torch
        from torch import no_grad
        boards_t = torch.from_numpy(np.ascontiguousarray(boards)).to(device)
        masks_t = torch.from_numpy(np.ascontiguousarray(action_masks)).to(device)
        h_t = torch.from_numpy(heights).to(device)
        w_t = torch.from_numpy(widths).to(device)
        na_t = torch.from_numpy(num_ancillas).to(device)
        with no_grad():
            priors_t, values_t = self.model.infer(
                boards=boards_t,
                heights=h_t,
                widths=w_t,
                num_ancillas=na_t,
                action_masks=masks_t,
            )
        return (
            priors_t.detach().cpu().numpy(),
            values_t.squeeze(-1).detach().cpu().numpy(),
        )


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from tile.agent import Agent

    parser = ArgumentParser()
    parser.add_argument("--arena_name", type=str, default="mcts")
    parser.add_argument("--arena_tag", type=str, default="")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--num_slots", type=int, default=2048)
    parser.add_argument("--num_handlers", type=int, default=1)
    parser.add_argument("--handler_id", type=int, default=0)
    parser.add_argument("--metrics_dir", type=str, default=None,
        help="If set, the handler writes one JSON line per inference batch "
             "to {metrics_dir}/handler_{arena_tag}_{handler_id}.jsonl with "
             "fill / build / gpu wallclock breakdown. Used to calibrate "
             "MAX_BATCH and BATCH_TIMEOUT from real workload distributions.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"CUDA is available. {torch.cuda.device_count()} devices found.")
        ngpu = torch.cuda.device_count()
        device_idx = args.handler_id % ngpu
        torch.cuda.set_device(device_idx)
        device = f"cuda:{device_idx}"
    else:
        print("CUDA is not available. Using CPU.")
        device = "cpu"

    model = Agent(embedding_dim=128, num_layers=10, lookahead=1)
    if args.weights is not None:
        print(f"[handler {args.handler_id}] Loading weights from: {args.weights}")
        model.load_state(args.weights)
        print(f"[handler {args.handler_id}] Weights loaded.")
    else:
        ckpt = latest_checkpoint()
        if ckpt is not None:
            print(f"[handler {args.handler_id}] No --weights given; loading latest checkpoint: {ckpt}")
            model.load_state(ckpt)
        else:
            print(f"[handler {args.handler_id}] No --weights given and no checkpoint found; using random weights.")
    model.to(device)

    # torch.compile the model in CUDA-graph mode. Inputs are padded to one
    # of _BS_BUCKETS before inference, so TorchInductor sees only a small
    # fixed set of shapes — one CUDA graph per bucket. `reduce-overhead`
    # captures the kernel launch sequence per shape and replays it as a
    # single submission, eliminating the ~50us-per-launch overhead that
    # dominated gpu_ms in the profile (15ms of 17ms wallclock was launch
    # overhead, only 2.5ms was actual GPU compute).
    #
    # We compile MODEL.forward via attribute assignment instead of wrapping
    # MODEL in OptimizedModule. Reason: do_work calls MODEL.infer(...), and
    # OptimizedModule.__getattr__ returns the underlying module's `infer`
    # method whose `self.forward(...)` resolves back to the *original*
    # forward — bypassing the compiled wrapper entirely. By replacing the
    # bound method directly, every `self.forward(...)` lookup (including
    # the one inside infer) picks up the compiled version.
    #
    # First call per bucket pays a compile cost (10-30s). With ~6 buckets,
    # the first ~minute of inference is compile-heavy; after that, every
    # call replays its bucket's graph. Handler warmup should absorb this.
    MODEL.forward = torch.compile(
        MODEL.forward, mode="reduce-overhead", dynamic=False)

    # Pre-compile every bucket eagerly so a previously-unseen bs never
    # arrives during real inference (a cold compile takes 10-30s and
    # would block one inference call entirely — observed as a ~15s
    # gpu_ms outlier in production). Synthetic zero inputs are fine
    # because the compile cache is shape-keyed, not value-keyed.
    print(f"[handler {args.handler_id}] pre-warming compile cache for "
          f"buckets {_BS_BUCKETS}...", flush=True)
    for _bs in _BS_BUCKETS:
        _t0 = time.monotonic()
        _boards = torch.zeros((_bs, MODEL.lookahead + 1, 100, 4),
                              dtype=torch.int32, device=device)
        _heights = torch.full((_bs,), 10, dtype=torch.int32, device=device)
        _widths = torch.full((_bs,), 10, dtype=torch.int32, device=device)
        _ancillas = torch.full((_bs,), 1, dtype=torch.int32, device=device)
        _mask = torch.ones((_bs, MODEL.num_outputs), dtype=torch.bool,
                           device=device)
        with no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = MODEL.infer(
                boards=_boards, heights=_heights, widths=_widths,
                num_ancillas=_ancillas, action_masks=_mask,
                h_int=10, w_int=10,
                max_actions_int=MODEL.num_outputs,
            )
        torch.cuda.synchronize()
        print(f"[handler {args.handler_id}]   bs={_bs:>4} compiled in "
              f"{time.monotonic() - _t0:.1f}s", flush=True)
    print(f"[handler {args.handler_id}] pre-warm done.", flush=True)

    tag = f"_{args.arena_tag}" if args.arena_tag else ""
    arena_name = f"{args.arena_name}{tag}_{args.num_slots}_{args.num_handlers}"
    arena = PyArena(arena_name, args.num_slots, args.num_handlers)
    do_work(arena, TorchAgentRunner(model), device, model.lookahead)
