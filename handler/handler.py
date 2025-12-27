import logging
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from torch import int32
from torch import no_grad
from torch import tensor
from torch.cuda import is_available

from tile import Agent

from mcts_ipc import PyArena

# ------------------------------------------------------------------------------
# Some type definitions and constants
# ------------------------------------------------------------------------------
ObsType = list[float]
PriorType = dict[int, float]
ValueType = float
BATCH_TIMEOUT = 0.005
DEVICE = "cuda" if is_available() else "cpu"
num_slots = 2048
num_handlers = 2

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
    files = sorted([str(x) for x in Path(ckpt_path).glob("*")])
    if len(files) == 0:
        return None
    return files[-1]

MODEL = Agent()
ckpt = latest_checkpoint()
if ckpt is not None:
    MODEL.load_state(ckpt)
MODEL.to(DEVICE)

# ------------------------------------------------------------------------------
# Inference endpoint
# ------------------------------------------------------------------------------

def do_work(handler_id: int, arena_name: str, num_handlers: int) -> None:
    arena = PyArena(arena_name, num_slots, num_handlers)
    while True:
        try:
            sv = arena.pop_ready_view(handler=handler_id, clear_outputs=True)
        except KeyboardInterrupt:
            break

        # Read inputs from the shared slot view
        placement_np = np.asarray(sv.placement())   # (b, GRID_MAX)
        obj0_np = np.asarray(sv.obj0())             # (b, MAX_OBJ0)
        obj1_np = np.asarray(sv.obj1())             # (b, MAX_OBJ1)
        h_np = np.asarray(sv.h())                   # (b,)
        w_np = np.asarray(sv.w())                   # (b,)
        action_mask_np = np.asarray(sv.action_mask())  # (b, NUM_ACTIONS)

        # compute per-column/trailing-nonzero extents and slice once
        placement_lens = np.count_nonzero(placement_np, axis=1)
        max_p_len = int(max(1, placement_lens.max()))
        placements_np = placement_np[:, :max_p_len].astype(np.int32, copy=False)
        placements = tensor(placements_np, device=DEVICE, dtype=int32)  # shape (b, max_p_len)

        # objectives
        obj0_lens = np.count_nonzero(obj0_np, axis=1)
        max_o0 = int(max(1, obj0_lens.max()))
        obj0_np2 = obj0_np[:, :max_o0].astype(np.int32, copy=False)
        objectives_0 = tensor(obj0_np2, device=DEVICE, dtype=int32)

        obj1_lens = np.count_nonzero(obj1_np, axis=1)
        max_o1 = int(max(1, obj1_lens.max()))
        obj1_np2 = obj1_np[:, :max_o1].astype(np.int32, copy=False)
        objectives_1 = tensor(obj1_np2, device=DEVICE, dtype=int32)

        action_masks = tensor(action_mask_np).to(DEVICE)

        # heights/widths
        heights_t = tensor(h_np.astype(np.int32), device=DEVICE, dtype=int32)
        widths_t = tensor(w_np.astype(np.int32), device=DEVICE, dtype=int32)

        sv.set_handler_start_time()

        with no_grad():
            priors_tensor, values_tensor = MODEL.infer(
                placement=placements,
                objectives=objectives_0,
                lookahead_objectives=objectives_1,
                heights=heights_t,
                widths=widths_t,
                action_mask=action_masks,
            )

        priors_np = priors_tensor.detach().cpu().numpy()
        values_np = values_tensor.squeeze(-1).detach().cpu().numpy()

        priors_arr = np.asarray(sv.priors())
        values_arr = np.asarray(sv.values())

        n = priors_np.shape[1]
        priors_arr[:, :n] = np.where(priors_np[:, :] > 1e-6, priors_np[:, :], 0.0)
        priors_arr[:, n:] = 0.0  # zero out any excess actions

        values_arr[:] = values_np[:]

        # mark slot done so producer can consume results
        sv.mark_done()

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--arena_name", type=str, default='mcts')
    parser.add_argument("--num_slots", type=int, default=num_slots)
    parser.add_argument("--num_handlers", type=int, default=1)
    parser.add_argument("--handler_id", type=int, default=0)
    args = parser.parse_args()
    arena_name = f"{args.arena_name}_{args.num_slots}_{args.num_handlers}"
    do_work(args.handler_id, arena_name, args.num_handlers)
