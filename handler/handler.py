import logging
from pathlib import Path
from argparse import ArgumentParser

import time
import numpy as np
import torch
from torch import int32, no_grad, tensor
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
MAX_BATCH = 64  # tune to your system

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

def do_work(arena: PyArena, handler_id: int) -> None:
    while True:
        try:
            # block for first request
            first_sv = arena.pop_ready_view(handler=handler_id, clear_outputs=True)
        except KeyboardInterrupt:
            break

        slot_views = [first_sv]
        start = time.monotonic()
        # try to gather additional ready slots until timeout or MAX_BATCH
        while len(slot_views) < MAX_BATCH and (time.monotonic() - start) < BATCH_TIMEOUT:
            # try_pop_ready_view is an assumed non-blocking API.
            # If your PyArena doesn't expose one, implement a small helper on the Rust side
            # that returns None immediately when no ready slot exists.
            sv = arena.try_pop_ready_view(handler=handler_id, clear_outputs=True)
            if sv is None:
                # small sleep to avoid busy spin
                time.sleep(0.0005)
                continue
            slot_views.append(sv)

        # Now convert collected slot_views into batched tensors
        b = len(slot_views)
        print(b)
        placement_np = np.asarray(slot_views[0].placement())[:b, :]  # will slice below per row
        # gather numpy arrays for all slots
        placements_list = []
        obj0_list = []
        obj1_list = []
        masks_list = []
        hs = []
        ws = []

        for sv in slot_views:
            p = np.asarray(sv.placement())
            # trim trailing zeros per-row by slicing later after computing max len
            placements_list.append(p)
            obj0_list.append(np.asarray(sv.obj0()))
            obj1_list.append(np.asarray(sv.obj1()))
            masks_list.append(np.asarray(sv.action_mask()))
            hs.append(int(np.asarray(sv.h())[0]))
            ws.append(int(np.asarray(sv.w())[0]))

        # compute max lengths and stack (vectorized, minimal Python loop)
        placements_arr = np.stack(placements_list, axis=0)
        p_lens = np.count_nonzero(placements_arr, axis=1)
        max_p = max(1, int(p_lens.max()))
        placements_np = placements_arr[:, :max_p].astype(np.int32, copy=False)
        placements = torch.from_numpy(placements_np).to(DEVICE)

        obj0_arr = np.stack(obj0_list, axis=0)
        o0_lens = np.count_nonzero(obj0_arr, axis=1)
        max_o0 = max(1, int(o0_lens.max()))
        obj0_np2 = obj0_arr[:, :max_o0].astype(np.int32, copy=False)
        objectives_0 = torch.from_numpy(obj0_np2).to(DEVICE)

        obj1_arr = np.stack(obj1_list, axis=0)
        o1_lens = np.count_nonzero(obj1_arr, axis=1)
        max_o1 = max(1, int(o1_lens.max()))
        obj1_np2 = obj1_arr[:, :max_o1].astype(np.int32, copy=False)
        objectives_1 = torch.from_numpy(obj1_np2).to(DEVICE)

        action_mask_np = np.stack(masks_list, axis=0)  # shape (b, NUM_ACTIONS)
        action_masks = torch.from_numpy(action_mask_np.astype(bool, copy=False)).to(DEVICE)

        heights_t = tensor(hs, device=DEVICE, dtype=int32)
        widths_t = tensor(ws, device=DEVICE, dtype=int32)

        import pdb; pdb.set_trace()

        # model inference
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

        # write results back to each slot and mark done
        for i, sv in enumerate(slot_views):
            priors_arr = np.asarray(sv.priors())
            values_arr = np.asarray(sv.values())
            n = priors_np.shape[1]
            priors_arr[:, :n] = np.where(priors_np[i:i+1, :n] > 1e-6, priors_np[i:i+1, :n], 0.0)
            if n < priors_arr.shape[1]:
                priors_arr[:, n:] = 0.0
            values_arr[:] = values_np[i]
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
    arena = PyArena(arena_name, num_slots, args.num_handlers)
    do_work(arena, args.handler_id)
