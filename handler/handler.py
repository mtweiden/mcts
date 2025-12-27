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
MAX_BATCH = 32  # tune to your system

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
            first_sv.set_handler_start_time()
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
            sv.set_handler_start_time()
            slot_views.append(sv)

        # Now convert collected slot_views into batched tensors
        placements_list = []
        obj0_list = []
        obj1_list = []
        masks_list = []
        hs = []
        ws = []
        slot_batch_sizes = []

        for sv in slot_views:
            p = np.asarray(sv.placement())        # shape (b_i, GRID_MAX)
            placements_list.append(p)
            obj0_list.append(np.asarray(sv.obj0()))        # (b_i, MAX_OBJ0)
            obj1_list.append(np.asarray(sv.obj1()))        # (b_i, MAX_OBJ1)
            masks_list.append(np.asarray(sv.action_mask()))# (b_i, NUM_ACTIONS)
            h_arr = np.asarray(sv.h()).tolist()             # (b_i,)
            w_arr = np.asarray(sv.w()).tolist()
            hs.extend(h_arr)
            ws.extend(w_arr)
            slot_batch_sizes.append(p.shape[0])

        # concatenate along the batch dimension to form a single large batch
        placements_cat = np.concatenate(placements_list, axis=0)   # (total_obs, max_p)
        p_lens = np.count_nonzero(placements_cat, axis=1)
        max_p = max(1, int(p_lens.max()))
        placements_np = placements_cat[:, :max_p].astype(np.int32, copy=False)
        placements = torch.from_numpy(placements_np).to(DEVICE)

        obj0_cat = np.concatenate(obj0_list, axis=0)
        o0_lens = np.count_nonzero(obj0_cat, axis=1)
        max_o0 = max(1, int(o0_lens.max()))
        obj0_np2 = obj0_cat[:, :max_o0].astype(np.int32, copy=False)
        objectives_0 = torch.from_numpy(obj0_np2).to(DEVICE)

        obj1_cat = np.concatenate(obj1_list, axis=0)
        o1_lens = np.count_nonzero(obj1_cat, axis=1)
        max_o1 = max(1, int(o1_lens.max()))
        obj1_np2 = obj1_cat[:, :max_o1].astype(np.int32, copy=False)
        objectives_1 = torch.from_numpy(obj1_np2).to(DEVICE)

        action_mask_cat = np.concatenate(masks_list, axis=0)  # (total_obs, NUM_ACTIONS)
        action_masks = torch.from_numpy(action_mask_cat.astype(bool, copy=False)).to(DEVICE)

        heights_t = tensor(hs, device=DEVICE, dtype=int32)
        widths_t = tensor(ws, device=DEVICE, dtype=int32)

        # model inference on the big concatenated batch
        with no_grad():
            priors_tensor, values_tensor = MODEL.infer(
                placement=placements,
                objectives=objectives_0,
                lookahead_objectives=objectives_1,
                heights=heights_t,
                widths=widths_t,
                action_mask=action_masks,
            )

        priors_np = priors_tensor.detach().cpu().numpy()   # (total_obs, NUM_ACTIONS)
        values_np = values_tensor.squeeze(-1).detach().cpu().numpy()  # (total_obs,)

        # reseparate outputs per-slot and write back
        idx = 0
        for slot_len, sv in zip(slot_batch_sizes, slot_views):
            pri_slice = priors_np[idx: idx + slot_len]    # (slot_len, NUM_ACTIONS)
            val_slice = values_np[idx: idx + slot_len]    # (slot_len,)
            idx += slot_len

            priors_arr = np.asarray(sv.priors())   # shape (b_slot, NUM_ACTIONS)
            values_arr = np.asarray(sv.values())   # shape (b_slot,)

            n_cols = pri_slice.shape[1]
            priors_arr[:slot_len, :n_cols] = np.where(pri_slice > 1e-6, pri_slice, 0.0)
            if n_cols < priors_arr.shape[1]:
                priors_arr[:slot_len, n_cols:] = 0.0

            values_arr[:slot_len] = val_slice
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
    arena = PyArena(arena_name, args.num_slots, args.num_handlers)
    do_work(arena, args.handler_id)
