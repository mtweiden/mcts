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

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
BATCH_TIMEOUT = 0.002
MAX_SLOTS_PER_BATCH = 64  # max slots to gather before one GPU call

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
def do_work(
    arena: PyArena,
    agent,
    device: str,
    lookahead: int,
    *,
    metrics_path: Path | None = None,
    const_value: float | None = None,
) -> None:
    """Pump the arena: batch ready slots, run ``agent.infer`` over the board,
    write priors/values back.

    ``agent`` must expose
    ``infer(boards, heights, widths, num_ancillas, action_masks) ->
    (priors, values)`` (tensors), matching ``tile.Agent`` — a dummy agent
    with the same signature works for tests.

    Args:
        metrics_path: If set, append one JSON line per inference batch with
            ``{ts, num_slots, num_obs, fill_ms, build_ms, gpu_ms}`` for
            calibrating ``MAX_BATCH`` / ``BATCH_TIMEOUT`` against real
            workloads.  Parent dir is created if missing.
    """
    metrics_fh = None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered so a kill mid-loop still leaves complete lines.
        metrics_fh = open(metrics_path, "a", buffering=1)

    try:
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
            fill_ms = (time.monotonic() - fill_start) * 1000.0

            try:
                build_start = time.monotonic()
                batched = build_board_batch(slot_views, lookahead)
                if batched is None:
                    continue
                (boards, heights, widths, num_ancillas,
                 action_masks, slot_batch_sizes, valid_slot_views) = batched
                build_ms = (time.monotonic() - build_start) * 1000.0

                gpu_start = time.monotonic()
                priors_np, values_np = agent.infer(
                    boards=boards,
                    heights=heights,
                    widths=widths,
                    num_ancillas=num_ancillas,
                    action_masks=action_masks,
                    device=device,
                )
                gpu_ms = (time.monotonic() - gpu_start) * 1000.0

                # EXPERIMENT (2026-07-31): replace the value head's output at
                # every NON-TERMINAL leaf with a constant. Terminal states never
                # reach this path -- mcts.rs:159 and :195-199 route them to
                # `terminal_evaluator` -- so this ablates the learned value only,
                # leaving real outcomes intact. With a constant leaf value, Q is
                # identical everywhere the search has not reached a terminal, so
                # PUCT degenerates to prior + visit-count exploration; ALL value
                # discrimination then comes from terminals found inside the tree.
                # Motivation: on tw>=80 the head scores R^2 = -0.20, i.e. worse
                # than the best constant predictor. Priors are untouched.
                if const_value is not None:
                    values_np = np.full_like(values_np, const_value, dtype=values_np.dtype)

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

                if metrics_fh is not None:
                    metrics_fh.write(json.dumps({
                        "ts":        time.time(),
                        "num_slots": len(valid_slot_views),
                        "num_obs":   int(boards.shape[0]),
                        "fill_ms":   round(fill_ms,  3),
                        "build_ms":  round(build_ms, 3),
                        "gpu_ms":    round(gpu_ms,   3),
                    }) + "\n")

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
    finally:
        if metrics_fh is not None:
            metrics_fh.close()


# ------------------------------------------------------------------------------
# Real-model adapter: wraps tile.Agent.infer to the (numpy in, numpy out)
# contract `do_work` uses.
# ------------------------------------------------------------------------------
class TorchAgentRunner:
    # Static-shape buckets for the model input batch dim.  With these,
    # `torch.compile(mode='reduce-overhead')` builds one CUDA graph per
    # bucket; every subsequent call rounds bs up to its bucket and
    # replays that bucket's graph.  Eliminates per-kernel-launch
    # overhead, which dominated `gpu_ms` in production profiling.
    BS_BUCKETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512)

    def __init__(
        self,
        model,
        *,
        board_height: int = 10,
        board_width: int = 10,
        compile_model: bool = True,
    ):
        """Adapt `tile.Agent` to the numpy-in / numpy-out contract `do_work` uses.

        Args:
            model: A `tile.Agent` (or compatible).  Must expose
                ``infer(boards, heights, widths, num_ancillas, action_masks,
                h_int, w_int, max_actions_int)``.
            board_height / board_width: Static env grid dims.  Passed as
                Python ints into ``model.infer`` so ``forward`` skips the
                ``.item()`` calls that break the ``torch.compile`` graph
                (see `tile/agent.py::forward`'s ``h_int`` / ``w_int``
                kwargs).  Defaults match the 10x10 production envs; pass
                different values for other shapes.
            compile_model: If True (the default), replace
                ``model.forward`` with a ``torch.compile``'d version in
                ``reduce-overhead`` mode.  Set False to bypass — useful
                for tests + debugging compile failures.
        """
        import torch
        self.model = model
        self.lookahead = model.lookahead
        self.board_height = board_height
        self.board_width = board_width
        self.max_actions_int = model.num_outputs

        if compile_model:
            # Compile `model.forward` via attribute assignment instead of
            # wrapping `model` in `OptimizedModule`.  Reason: `do_work`
            # calls `model.infer(...)`, and `OptimizedModule.__getattr__`
            # returns the underlying module's `infer` method whose
            # `self.forward(...)` resolves back to the *original* forward —
            # bypassing the compiled wrapper entirely.  Replacing the
            # bound method directly makes every `self.forward(...)` lookup
            # (including the one inside `infer`) pick up the compiled
            # version.
            self.model.forward = torch.compile(
                self.model.forward, mode="reduce-overhead", dynamic=False,
            )

    @classmethod
    def _bucket_bs(cls, n: int) -> int:
        """Smallest bucket ≥ n.  Anything above the largest is clamped
        (rare in practice; we'd rather pay one extra graph hit than let
        one giant outlier blow the compile cache)."""
        for b in cls.BS_BUCKETS:
            if n <= b:
                return b
        return cls.BS_BUCKETS[-1]

    @staticmethod
    def _pad_to_bs(t, bs_padded: int, fill_value):
        """Pad tensor along dim 0 (batch) to length bs_padded with fill_value."""
        import torch
        bs = t.shape[0]
        if bs >= bs_padded:
            return t
        pad_shape = (bs_padded - bs,) + tuple(t.shape[1:])
        pad_tensor = torch.full(pad_shape, fill_value, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad_tensor], dim=0)

    def prewarm(self, device, *, log_prefix: str = "[handler]") -> None:
        """Compile every bucket eagerly so a previously-unseen bs never
        arrives during real inference (a cold compile takes 10–30s per
        bucket and would block one inference call entirely).  Synthetic
        zero inputs are fine — the compile cache is shape-keyed, not
        value-keyed.  Each iteration also exercises the bf16 autocast +
        static-shape kwargs path so the cache entries match what
        production calls produce.
        """
        import torch
        from contextlib import nullcontext
        from torch import no_grad

        autocast_ctx_factory = (
            (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
            if str(device).startswith("cuda") else (lambda: nullcontext())
        )
        hw = self.board_height * self.board_width
        n_layers = self.lookahead + 1
        print(f"{log_prefix} pre-warming compile cache for buckets {self.BS_BUCKETS}...", flush=True)
        for bs in self.BS_BUCKETS:
            t0 = time.monotonic()
            boards = torch.zeros((bs, n_layers, hw, CELL_FIELDS), dtype=torch.int32, device=device)
            heights = torch.full((bs,), self.board_height, dtype=torch.int32, device=device)
            widths = torch.full((bs,), self.board_width, dtype=torch.int32, device=device)
            ancillas = torch.full((bs,), 1, dtype=torch.int32, device=device)
            mask = torch.ones((bs, self.max_actions_int), dtype=torch.bool, device=device)
            with no_grad(), autocast_ctx_factory():
                _ = self.model.infer(
                    boards=boards, heights=heights, widths=widths,
                    num_ancillas=ancillas, action_masks=mask,
                    h_int=self.board_height,
                    w_int=self.board_width,
                    max_actions_int=self.max_actions_int,
                )
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            print(f"{log_prefix}   bs={bs:>4} compiled in {time.monotonic() - t0:.1f}s", flush=True)
        print(f"{log_prefix} pre-warm done.", flush=True)

    def infer(self, boards, heights, widths, num_ancillas, action_masks, device):
        import torch
        from contextlib import nullcontext
        from torch import no_grad

        # non_blocking H2D lets each copy overlap with the next Python op
        # (the model's own setup); pinned source memory isn't used since
        # the numpy arrays are backed by the shared-memory arena, but the
        # flag still skips an unnecessary cudaStreamSynchronize.
        boards_t = torch.from_numpy(np.ascontiguousarray(boards)).to(device, non_blocking=True)
        masks_t = torch.from_numpy(np.ascontiguousarray(action_masks)).to(device, non_blocking=True)
        h_t = torch.from_numpy(heights).to(device, non_blocking=True)
        w_t = torch.from_numpy(widths).to(device, non_blocking=True)
        na_t = torch.from_numpy(num_ancillas).to(device, non_blocking=True)

        # Pad the batch dim to a fixed bucket so the model sees one of
        # only ~6 distinct shapes.  Padded entries get safe defaults
        # (height=1, width=1, num_ancillas=1, action_mask=all-False); we
        # slice them off after inference so they don't leak into outputs.
        # action_mask=all-False is safe for softmax — the legal entries
        # on real rows are unaffected.
        real_bs = boards_t.shape[0]
        bs_padded = self._bucket_bs(real_bs)
        if bs_padded > real_bs:
            boards_t = self._pad_to_bs(boards_t, bs_padded, 0)
            h_t      = self._pad_to_bs(h_t,      bs_padded, 1)
            w_t      = self._pad_to_bs(w_t,      bs_padded, 1)
            na_t     = self._pad_to_bs(na_t,     bs_padded, 1)
            masks_t  = self._pad_to_bs(masks_t,  bs_padded, False)

        # bf16 autocast for forward — trainer trains in bf16
        # (tile/trainer.py), so weights + activations have been exposed
        # to this precision throughout training.  Outputs may come back
        # in bf16; cast to fp32 before numpy (numpy has no bf16 support).
        autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if str(device).startswith("cuda") else nullcontext())
        with no_grad(), autocast_ctx:
            # Static-shape kwargs (h_int / w_int / max_actions_int) skip
            # the .item() graph-breaks under torch.compile.  The model's
            # ``action_mask`` is already full-size (num_outputs), so
            # passing num_outputs as max_actions_int makes the trim a
            # no-op and avoids the .item() on `num_ancillas`.
            priors_t, values_t = self.model.infer(
                boards=boards_t,
                heights=h_t,
                widths=w_t,
                num_ancillas=na_t,
                action_masks=masks_t,
                h_int=self.board_height,
                w_int=self.board_width,
                max_actions_int=self.max_actions_int,
            )

        # Slice off padded rows BEFORE the D2H copy — saves transferring
        # the dead pad entries.
        return (
            priors_t[:real_bs].detach().to(torch.float32).cpu().numpy(),
            values_t[:real_bs].squeeze(-1).detach().to(torch.float32).cpu().numpy(),
        )


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from tile.agent import Agent

    parser = ArgumentParser()
    parser.add_argument("--const_value", type=float, default=None,
        help="EXPERIMENT: override the value head at every non-terminal leaf "
             "with this constant. Terminal states are unaffected (mcts.rs "
             "routes them to terminal_evaluator). Priors untouched.")
    parser.add_argument("--arena_name", type=str, default="mcts")
    parser.add_argument("--arena_tag", type=str, default="")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--num_slots", type=int, default=2048)
    parser.add_argument("--num_handlers", type=int, default=1)
    parser.add_argument("--handler_id", type=int, default=0)
    parser.add_argument("--metrics_dir", type=str, default=None,
        help="If set, the handler appends one JSON line per inference batch "
             "to {metrics_dir}/handler_{arena_tag}_{handler_id}.jsonl with "
             "{ts, num_slots, num_obs, fill_ms, build_ms, gpu_ms}.  Used "
             "to calibrate MAX_BATCH and BATCH_TIMEOUT from real workload "
             "distributions.")
    # Model architecture — must match the checkpoint in --weights. Defaults
    # match the d=128 production checkpoint; orchestrator.start_handlers
    # forwards args.embedding_dim/num_layers/lookahead/num_heads on every
    # launch.
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
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

    model = Agent(
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        lookahead=args.lookahead,
        num_heads=args.num_heads,
    )
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

    runner = TorchAgentRunner(model)
    runner.prewarm(device, log_prefix=f"[handler {args.handler_id}]")

    tag = f"_{args.arena_tag}" if args.arena_tag else ""
    arena_name = f"{args.arena_name}{tag}_{args.num_slots}_{args.num_handlers}"
    arena = PyArena(arena_name, args.num_slots, args.num_handlers)

    metrics_path = None
    if args.metrics_dir:
        atag = args.arena_tag or "default"
        metrics_path = Path(args.metrics_dir) / f"handler_{atag}_{args.handler_id}.jsonl"

    if args.const_value is not None:
        print(f"[handler {args.handler_id}] VALUE ABLATION: every non-terminal leaf "
              f"returns {args.const_value} (terminals unaffected)")
    do_work(arena, runner, device, model.lookahead, metrics_path=metrics_path,
            const_value=args.const_value)
