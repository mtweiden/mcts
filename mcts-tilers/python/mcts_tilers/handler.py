import logging
from pathlib import Path
from argparse import ArgumentParser

import time
import numpy as np
import torch
from torch import no_grad

from tile import Agent

from mcts_tilers import PyArena
from mcts_tilers import QUBIT_SIZE, OBJECTIVE_SIZE, OBJECTIVES_LAYER_MAX

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
BATCH_TIMEOUT = 0.005
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
    files = sorted([str(x) for x in Path(ckpt_path).glob("*")])
    if len(files) == 0:
        return None
    return files[-1]


MODEL = Agent()
ckpt = latest_checkpoint()
if ckpt is not None:
    MODEL.load_state(ckpt)


# ------------------------------------------------------------------------------
# Unpacking helpers
# ------------------------------------------------------------------------------
def unpack_placement_batch(
    raw: np.ndarray,       # (total, PLACEMENT_MAX) u8
    num_qubits: np.ndarray # (total,) u16
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        ids:          (total, max_nq) i32
        orientations: (total, max_nq) u8
    """
    total = raw.shape[0]
    max_nq = max(1, int(num_qubits.max()))

    ids = np.zeros((total, max_nq), dtype=np.int32)
    orientations = np.zeros((total, max_nq), dtype=np.uint8)

    for i in range(total):
        nq = int(num_qubits[i])
        if nq == 0:
            continue
        buf = raw[i, : nq * QUBIT_SIZE].reshape(nq, QUBIT_SIZE)
        ids[i, :nq] = buf[:, :4].copy().view(np.int32).reshape(nq)
        orientations[i, :nq] = buf[:, 4]

    return ids, orientations


def unpack_objectives_batch(
    raw: np.ndarray,              # (total, OBJECTIVES_MAX) u8
    num_layers: np.ndarray,       # (total,) u8
    num_objectives: np.ndarray,   # (total, LOOKAHEAD_MAX) u16
) -> list[dict[str, np.ndarray]]:
    """
    Returns a list (one per layer index) of dicts with keys:
        opcodes:    (total, max_no) u8
        arg0s:      (total, max_no) i32
        arg1s:      (total, max_no) i32
    Only layers 0..max_nl are returned.
    """
    total = raw.shape[0]
    max_nl = max(1, int(num_layers.max()))

    layers = []
    for l in range(max_nl):
        max_no = max(1, int(num_objectives[:, l].max())) if l < num_objectives.shape[1] else 1

        opcodes = np.zeros((total, max_no), dtype=np.uint8)
        arg0s = np.zeros((total, max_no), dtype=np.int32)
        arg1s = np.zeros((total, max_no), dtype=np.int32)

        layer_offset = l * OBJECTIVES_LAYER_MAX

        for i in range(total):
            if l >= int(num_layers[i]):
                continue
            no = int(num_objectives[i, l])
            if no == 0:
                continue
            buf = raw[i, layer_offset: layer_offset + no * OBJECTIVE_SIZE].reshape(no, OBJECTIVE_SIZE)
            opcodes[i, :no] = buf[:, 0]
            arg0s[i, :no] = buf[:, 1:5].copy().view(np.int32).reshape(no)
            arg1s[i, :no] = buf[:, 5:9].copy().view(np.int32).reshape(no)

        layers.append({
            "opcodes": opcodes,
            "arg0s": arg0s,
            "arg1s": arg1s,
        })

    return layers


_RAW_OPCODE_TO_TOKEN: dict[int, int] = {
    0:  2,   # X
    1:  2,   # Y
    2:  2,   # Z
    3:  3,   # H
    4:  4,   # S
    5:  4,   # Sdg
    6:  5,   # SX
    7:  5,   # SXdg
    8:  6,   # T
    9:  6,   # Tdg
    12: 2,   # RZ
    16: 7,   # MEASURE
    17: 10,  # RESET
}

_CX_OPCODE = 8
_CZ_OPCODE = 9

_RAW_ORI_TO_TOKEN: dict[int, int] = {
    0: 2,   # Vertical
    1: 3,   # Horizontal
    2: 4,   # Ancilla
    3: 5,   # Cultivating
    4: 6,   # Resource
}


def build_boards(
    qubit_ids: np.ndarray,      # (total, max_nq) i32
    qubit_oris: np.ndarray,     # (total, max_nq) u8
    num_qubits: np.ndarray,     # (total,) u16
    obj_layers: list[dict[str, np.ndarray]],
    num_layers: np.ndarray,     # (total,) u8
    num_objectives: np.ndarray, # (total, LOOKAHEAD_MAX) u16
) -> np.ndarray:
    """
    Transform placements and objectives into board tensors.

    Replicates construct_board() from tile/board.py on batched numpy arrays.

    Returns:
        boards: (total, max_nl * max_nq * 3) int32, zero-padded
    """
    total = qubit_ids.shape[0]
    max_nq = qubit_ids.shape[1]
    max_nl = len(obj_layers)

    boards = np.zeros((total, max_nl * max_nq * 3), dtype=np.int32)

    for i in range(total):
        nq = int(num_qubits[i])
        nl = int(num_layers[i])
        tag = 2

        for l in range(min(nl, max_nl)):
            no = int(num_objectives[i, l])
            layer_data = obj_layers[l]

            board_map: dict[int, tuple[int, int]] = {}
            for k in range(no):
                opcode = int(layer_data["opcodes"][i, k])
                arg0 = int(layer_data["arg0s"][i, k])
                arg1 = int(layer_data["arg1s"][i, k])

                if opcode == _CZ_OPCODE:
                    board_map[arg0] = (8, tag)
                    board_map[arg1] = (8, tag)
                elif opcode == _CX_OPCODE:
                    board_map[arg0] = (8, tag)
                    board_map[arg1] = (9, tag)
                else:
                    token = _RAW_OPCODE_TO_TOKEN.get(opcode, 1)
                    board_map[arg0] = (token, tag)

                tag += 1

            layer_offset = l * max_nq * 3
            for j in range(nq):
                qid = int(qubit_ids[i, j])
                ori_token = _RAW_ORI_TO_TOKEN.get(int(qubit_oris[i, j]), 1)
                cell_offset = layer_offset + j * 3

                if qid in board_map:
                    op, t = board_map[qid]
                    boards[i, cell_offset] = op
                    boards[i, cell_offset + 1] = t
                    boards[i, cell_offset + 2] = ori_token
                elif qid < 0:
                    boards[i, cell_offset] = 11
                    boards[i, cell_offset + 1] = -(qid + 1) + 2
                    boards[i, cell_offset + 2] = ori_token
                else:
                    boards[i, cell_offset] = 1
                    boards[i, cell_offset + 1] = 1
                    boards[i, cell_offset + 2] = 1

    return boards


# ------------------------------------------------------------------------------
# Inference loop
# ------------------------------------------------------------------------------
def do_work(arena: PyArena, handler_id: int, device: str) -> None:
    while True:
        try:
            first_sv = arena.pop_ready_view(handler=handler_id, clear_outputs=True)
            first_sv.set_handler_start_time()
        except KeyboardInterrupt:
            break

        slot_views = [first_sv]
        start = time.monotonic()
        while len(slot_views) < MAX_SLOTS_PER_BATCH and (time.monotonic() - start) < BATCH_TIMEOUT:
            sv = arena.try_pop_ready_view(handler=handler_id, clear_outputs=True)
            if sv is None:
                time.sleep(0.0005)
                continue
            sv.set_handler_start_time()
            slot_views.append(sv)

        try:
            # Gather raw arrays from all slots
            placement_list = []
            objectives_list = []
            mask_list = []
            h_list = []
            w_list = []
            ancillas_list = []
            nq_list = []
            nl_list = []
            no_list = []
            slot_batch_sizes = []

            for sv in slot_views:
                b_i = sv.b()
                if b_i <= 0:
                    continue  # Skip uninitialized/invalid slots
                slot_batch_sizes.append(b_i)

                placement_list.append(np.asarray(sv.placement())[:b_i])
                objectives_list.append(np.asarray(sv.objectives())[:b_i])
                mask_list.append(np.asarray(sv.action_mask())[:b_i])
                h_list.append(np.asarray(sv.h())[:b_i])
                w_list.append(np.asarray(sv.w())[:b_i])
                ancillas_list.append(np.asarray(sv.num_ancillas())[:b_i])
                nq_list.append(np.asarray(sv.num_qubits())[:b_i])
                nl_list.append(np.asarray(sv.num_layers())[:b_i])
                no_list.append(np.asarray(sv.num_objectives())[:b_i])

            # Concatenate into single batch
            placement_raw = np.concatenate(placement_list, axis=0)
            objectives_raw = np.concatenate(objectives_list, axis=0)
            action_mask_raw = np.concatenate(mask_list, axis=0)
            h_all = np.concatenate(h_list)
            w_all = np.concatenate(w_list)
            ancillas_all = np.concatenate(ancillas_list)
            nq_all = np.concatenate(nq_list)
            nl_all = np.concatenate(nl_list)
            no_all = np.concatenate(no_list, axis=0)

            # Unpack structured data
            qubit_ids, qubit_oris = unpack_placement_batch(placement_raw, nq_all)
            obj_layers = unpack_objectives_batch(objectives_raw, nl_all, no_all)

            # Build board representation for the model
            boards_np = build_boards(
                qubit_ids, qubit_oris, nq_all, obj_layers, nl_all, no_all,
            )

            # Build tensors
            boards_t = torch.from_numpy(np.ascontiguousarray(boards_np)).to(device)
            action_masks_t = torch.from_numpy(
                np.ascontiguousarray(action_mask_raw.astype(bool, copy=False))
            ).to(device)
            heights_t = torch.from_numpy(h_all.astype(np.int32, copy=False)).to(device)
            widths_t = torch.from_numpy(w_all.astype(np.int32, copy=False)).to(device)
            num_ancillas_t = torch.from_numpy(ancillas_all.astype(np.int32, copy=False)).to(device)

            # Model inference
            with no_grad():
                priors_tensor, values_tensor = MODEL.infer(
                    boards=boards_t,
                    heights=heights_t,
                    widths=widths_t,
                    num_ancillas=num_ancillas_t,
                    action_masks=action_masks_t,
                )

            priors_np = priors_tensor.detach().cpu().numpy()
            values_np = values_tensor.squeeze(-1).detach().cpu().numpy()

            # Write back per-slot
            idx = 0
            for slot_len, sv in zip(slot_batch_sizes, slot_views):
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


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--arena_name", type=str, default="mcts")
    parser.add_argument("--num_slots", type=int, default=2048)
    parser.add_argument("--num_handlers", type=int, default=1)
    parser.add_argument("--handler_id", type=int, default=0)
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
    MODEL.to(device)

    arena_name = f"{args.arena_name}_{args.num_slots}_{args.num_handlers}"
    arena = PyArena(arena_name, args.num_slots, args.num_handlers)
    arena.force_reset()  # Clear any stale state from previous runs
    do_work(arena, args.handler_id, device)