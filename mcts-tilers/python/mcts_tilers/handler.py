import json
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


MODEL = Agent(embedding_dim=128, num_layers=10, lookahead=1)

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

        opcodes = np.zeros((total, max_no), dtype=np.int8)
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
    0: 2,    # X
    1: 2,    # Y
    2: 2,    # Z
    3: 3,    # H
    4: 4,    # S
    5: 4,    # Sdg
    6: 5,    # SX
    7: 5,    # SXdg
    8: 6,    # T
    9: 6,    # Tdg
    10: 7,   # TX
    11: 7,   # TXdg
    14: 2,   # RZ
    18: 12,  # MEASURE
    19: 13,  # RESET
}

# opcodes from tilers (verified via int(Operation.CX) / int(Operation.CZ))
_RAW_CX_OPCODE = 12
_RAW_CZ_OPCODE = 13

# board.py two-qubit tokens (matching board.py construct_board / _build_layer)
_TOKEN_CZ_CONTROL = 8
_TOKEN_CZ_TARGET = 9
_TOKEN_CX_CONTROL = 10
_TOKEN_CX_TARGET = 11

# raw orientation -> board.py orientation token
_RAW_ORI_TO_TOKEN: dict[int, int] = {
    0: 2,  # Vertical
    1: 3,  # Horizontal
    3: 4,  # Cultivating
    4: 5,  # Resource
    2: 6,  # Ancilla (default / last move horizontal)
    # 7 is derived for ancilla + last_move_vertical
}


# Orientation token lookup table; built once at module import. Maps the raw
# orientation byte (0..255) to its board-vocabulary token. Default for
# unrecognised values is 1 (matches the dict.get default in the original).
_ORI_LUT = np.full(256, 1, dtype=np.int32)
for _raw, _tok in _RAW_ORI_TO_TOKEN.items():
    _ORI_LUT[_raw] = _tok

# Single-qubit opcode → board-vocabulary token. Default 1 for unmapped.
_OPCODE_LUT_SIZE = max(
    max(_RAW_OPCODE_TO_TOKEN.keys()),
    _RAW_CX_OPCODE,
    _RAW_CZ_OPCODE,
) + 1
_OPCODE_LUT = np.full(_OPCODE_LUT_SIZE, 1, dtype=np.int32)
for _raw, _tok in _RAW_OPCODE_TO_TOKEN.items():
    _OPCODE_LUT[_raw] = _tok


def build_boards(
    qubit_ids: np.ndarray,         # (total, max_nq) i32
    qubit_oris: np.ndarray,        # (total, max_nq) u8
    num_qubits: np.ndarray,        # (total,) u16
    widths: np.ndarray,            # (total,) i32
    obj_layers: list[dict[str, np.ndarray]],
    num_layers: np.ndarray,        # (total,) u8
    num_objectives: np.ndarray,    # (total, LOOKAHEAD_MAX) u16
    last_dir_vertical: np.ndarray, # (total, MAX_ANCILLAS) bool
) -> np.ndarray:
    """
    Transform placements and objectives into board tensors.

    Returns:
        boards: (total, max_nl, max_nq, 4) int32
            per-cell = (op, ori, mate_row, mate_col)
            padded cells remain zeros.

    Behavior pinned by tests/test_build_boards.py — the reference there is
    the literal pre-vectorization implementation. Any change to this
    function must keep that test green on every random seed. The hot loops
    (per-qubit board writes, orientation lookup) are vectorised; the
    per-objective obj_map dict build remains scalar because of the
    last-write-wins collision semantics on duplicate arg0 keys.
    """
    total = qubit_ids.shape[0]
    max_nq = qubit_ids.shape[1]
    max_nl = len(obj_layers)
    ldv_max = last_dir_vertical.shape[1]

    boards = np.zeros((total, max_nl, max_nq, 4), dtype=np.int32)

    # Vectorised orientation token lookup for the whole batch at once.
    ori_tokens_all = _ORI_LUT[qubit_oris]  # (total, max_nq) i32

    for i in range(total):
        nq = int(num_qubits[i])
        nl = int(num_layers[i])
        w = int(widths[i]) if i < widths.shape[0] else 0
        if nq <= 0 or w <= 0:
            continue

        active_qids = qubit_ids[i, :nq]
        ori_active = ori_tokens_all[i, :nq]

        # qid → (row, col). Dict (not array) because qids can be negative
        # and unbounded; iteration order is preserved so collisions resolve
        # to the last position (matches reference).
        pos_map: dict[int, tuple[int, int]] = {}
        for j in range(nq):
            pos_map[int(active_qids[j])] = (j // w, j % w)

        # Precompute the ancilla mask + indices for this instance, used by
        # every layer below.
        neg_mask = active_qids < 0
        ancilla_idx_all = -(active_qids.astype(np.int64) + 1)

        for l in range(min(nl, max_nl)):
            no = int(num_objectives[i, l])

            # Build obj_map dict. This is the per-objective scalar loop;
            # it's small (no <= ~250) and last-write-wins semantics on
            # duplicate arg0 require ordered scalar writes.
            obj_map: dict[int, tuple[int, int, int]] = {}
            if no > 0:
                layer_data = obj_layers[l]
                op_arr = layer_data["opcodes"][i]
                a0_arr = layer_data["arg0s"][i]
                a1_arr = layer_data["arg1s"][i]
                for k in range(no):
                    opcode = int(op_arr[k])
                    arg0 = int(a0_arr[k])
                    arg1 = int(a1_arr[k])

                    if opcode == _RAW_CZ_OPCODE:
                        if arg0 in pos_map and arg1 in pos_map:
                            r0, c0 = pos_map[arg0]
                            r1, c1 = pos_map[arg1]
                            obj_map[arg0] = (_TOKEN_CZ_CONTROL, r1, c1)
                            obj_map[arg1] = (_TOKEN_CZ_TARGET, r0, c0)
                    elif opcode == _RAW_CX_OPCODE:
                        if arg0 in pos_map and arg1 in pos_map:
                            r0, c0 = pos_map[arg0]
                            r1, c1 = pos_map[arg1]
                            obj_map[arg0] = (_TOKEN_CX_CONTROL, r1, c1)
                            obj_map[arg1] = (_TOKEN_CX_TARGET, r0, c0)
                    else:
                        token = _OPCODE_LUT[opcode] if 0 <= opcode < _OPCODE_LUT_SIZE else 1
                        obj_map[arg0] = (int(token), -1, -1)

            # Vectorised per-qubit write. Three branches by priority:
            #   1. qid in obj_map → (op, looked-up-ori, mate_row, mate_col)
            #   2. qid < 0 (ancilla) and not in obj_map → (13-qid, ori w/ 6→7, -1, -1)
            #   3. else → (1, 1, -1, -1)  [hardcoded ori=1 in default branch]
            ops = np.ones(nq, dtype=np.int32)
            oris_out = np.ones(nq, dtype=np.int32)  # default branch ori=1
            mate_rows = np.full(nq, -1, dtype=np.int32)
            mate_cols = np.full(nq, -1, dtype=np.int32)
            obj_mask = np.zeros(nq, dtype=bool)

            if obj_map:
                for qid, (op_v, mr_v, mc_v) in obj_map.items():
                    js = np.where(active_qids == qid)[0]
                    if js.size > 0:
                        ops[js] = op_v
                        oris_out[js] = ori_active[js]  # looked-up ori for obj branch
                        mate_rows[js] = mr_v
                        mate_cols[js] = mc_v
                        obj_mask[js] = True

            anc_mask = neg_mask & ~obj_mask
            if anc_mask.any():
                anc_qids = active_qids[anc_mask]
                ops[anc_mask] = 13 - anc_qids  # -1→14, etc
                # Ancilla branch keeps the looked-up ori, with possible 6→7
                # transition when the corresponding last_dir_vertical bit is set.
                anc_ori = ori_active[anc_mask].copy()
                anc_idx = ancilla_idx_all[anc_mask]
                # The transition only fires when ori==6 AND ancilla_idx is in
                # range AND the ldv bit at that index is True.
                cand = (anc_ori == 6) & (anc_idx >= 0) & (anc_idx < ldv_max)
                if cand.any():
                    cand_idx = anc_idx[cand].astype(np.intp)
                    ldv_bits = last_dir_vertical[i, cand_idx]
                    fires = np.zeros_like(cand)
                    fires[cand] = ldv_bits
                    anc_ori[fires] = 7
                oris_out[anc_mask] = anc_ori

            boards[i, l, :nq, 0] = ops
            boards[i, l, :nq, 1] = oris_out
            boards[i, l, :nq, 2] = mate_rows
            boards[i, l, :nq, 3] = mate_cols

    return boards


# ------------------------------------------------------------------------------
# Inference loop
# ------------------------------------------------------------------------------
def do_work(
    arena: PyArena,
    device: str,
    handler_id: int = 0,
    metrics_path: Path | None = None,
) -> None:
    """
    Inference handler main loop.

    metrics_path: if set, writes one JSON line per inference batch with
    {ts, num_slots, num_obs, fill_ms, build_ms, gpu_ms}. Used to calibrate
    MAX_BATCH and BATCH_TIMEOUT — small/sparse batches mean GPU
    underutilisation; long fill times mean slots arrive faster than
    BATCH_TIMEOUT lets us aggregate; large gpu_ms variance suggests
    inference itself is the variable cost.
    """
    metrics_fh = None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = open(metrics_path, "a", buffering=1)

    iteration = 0
    while True:
        try:
            first_sv = arena.pop_ready_view(clear_outputs=True)
            first_sv.set_handler_start_time()
        except KeyboardInterrupt:
            break

        iteration += 1

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

        # print(f"[handler {handler_id}] loop iteration {iteration}, {len(slot_views)} slots pending", flush=True)

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
            ldirs_list = []

            slot_batch_sizes = []
            valid_slot_views = []
            for sv in slot_views:
                b_i = sv.b()
                if b_i <= 0:
                    sv.mark_done()
                    continue  # Skip uninitialized/invalid slots
                slot_batch_sizes.append(b_i)
                valid_slot_views.append(sv)

                placement_list.append(np.asarray(sv.placement()[:b_i], copy=True))
                objectives_list.append(np.asarray(sv.objectives()[:b_i], copy=True))
                mask_list.append(np.asarray(sv.action_mask()[:b_i], copy=True))
                h_list.append(np.asarray(sv.h()[:b_i], copy=True))
                w_list.append(np.asarray(sv.w()[:b_i], copy=True))
                ancillas_list.append(np.asarray(sv.num_ancillas()[:b_i], copy=True))
                nq_list.append(np.asarray(sv.num_qubits()[:b_i], copy=True))
                # num_layers must be the same for all instances
                nl_list.append(np.asarray(sv.num_layers()[:b_i], copy=True))
                no_list.append(np.asarray(sv.num_objectives()[:b_i], copy=True))
                ldirs_list.append(np.asarray(sv.last_dir_vertical()[:b_i], copy=True))

            if not valid_slot_views:
                continue  # No valid slots to process

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
            last_dir_vertical_all = np.concatenate(ldirs_list, axis=0)

            # Unpack structured data
            qubit_ids, qubit_oris = unpack_placement_batch(placement_raw, nq_all)
            obj_layers = unpack_objectives_batch(objectives_raw, nl_all, no_all)

            # Build board representation for the model
            boards_np = build_boards(
                qubit_ids,
                qubit_oris,
                nq_all,
                w_all,  # width needed for mate coordinates
                obj_layers,
                nl_all,
                no_all,
                last_dir_vertical_all,
            )
            assert boards_np.shape[1] == MODEL.lookahead + 1, \
                f"build_boards returned shape {boards_np.shape}, expected num_layers={MODEL.lookahead + 1}"

            build_ms = (time.monotonic() - fill_start - fill_ms / 1000.0) * 1000.0

            # Build tensors. non_blocking=True lets the H2D copy overlap
            # with the next Python op (model.infer's own setup); pinned
            # source memory is not used here because the numpy arrays
            # are backed by the shared-memory arena, but the flag still
            # avoids unnecessary cudaStreamSynchronize.
            boards_t = torch.from_numpy(np.ascontiguousarray(boards_np)).to(device, non_blocking=True)
            action_masks_t = torch.from_numpy(
                np.ascontiguousarray(action_mask_raw.astype(bool, copy=False))
            ).to(device, non_blocking=True)
            heights_t = torch.from_numpy(h_all.astype(np.int32, copy=False)).to(device, non_blocking=True)
            widths_t = torch.from_numpy(w_all.astype(np.int32, copy=False)).to(device, non_blocking=True)
            num_ancillas_t = torch.from_numpy(ancillas_all.astype(np.int32, copy=False)).to(device, non_blocking=True)

            # Model inference
            gpu_start = time.monotonic()
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
            gpu_ms = (time.monotonic() - gpu_start) * 1000.0

            # Write back per-slot
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
                num_obs = int(boards_np.shape[0])
                metrics_fh.write(json.dumps({
                    "ts": time.time(),
                    "num_slots": len(valid_slot_views),
                    "num_obs": num_obs,
                    "fill_ms": round(fill_ms, 3),
                    "build_ms": round(build_ms, 3),
                    "gpu_ms": round(gpu_ms, 3),
                }) + "\n")

        except KeyboardInterrupt:
            for sv in valid_slot_views:
                try:
                    sv.mark_done()
                except Exception:
                    pass
            break

    if metrics_fh is not None:
        metrics_fh.close()


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
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

    if args.weights is not None:
        print(f"[handler {args.handler_id}] Loading weights from: {args.weights}")
        MODEL.load_state(args.weights)
        print(f"[handler {args.handler_id}] Weights loaded.")
    else:
        ckpt = latest_checkpoint()
        if ckpt is not None:
            print(f"[handler {args.handler_id}] No --weights given; loading latest checkpoint: {ckpt}")
            MODEL.load_state(ckpt)
        else:
            print(f"[handler {args.handler_id}] No --weights given and no checkpoint found; using random weights.")
    MODEL.to(device)

    tag = f"_{args.arena_tag}" if args.arena_tag else ""
    arena_name = f"{args.arena_name}{tag}_{args.num_slots}_{args.num_handlers}"
    arena = PyArena(arena_name, args.num_slots, args.num_handlers)

    metrics_path = None
    if args.metrics_dir:
        atag = args.arena_tag or "default"
        metrics_path = Path(args.metrics_dir) / f"handler_{atag}_{args.handler_id}.jsonl"

    do_work(arena, device, handler_id=args.handler_id, metrics_path=metrics_path)
