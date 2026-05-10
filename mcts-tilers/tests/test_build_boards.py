"""
Tests for handler.build_boards. Pinned-behavior tests:

- The canonical reference (`_reference_build_boards`) is a literal copy of the
  implementation as of 2026-05-09, before any vectorization. It encodes the
  intended semantics on representative inputs.
- The production `build_boards` must match this reference output exactly on
  every test case, including edge cases (empty layers, no objectives, ancillas
  with the 6→7 orientation transition, CX/CZ pairs, mixed batches).

The randomized smoke test uses a fixed RNG seed so failures are reproducible.
Run with: pytest mcts/mcts-tilers/tests/test_build_boards.py
"""

import numpy as np
import pytest

from mcts_tilers.handler import (
    build_boards,
    _RAW_OPCODE_TO_TOKEN,
    _RAW_CX_OPCODE,
    _RAW_CZ_OPCODE,
    _RAW_ORI_TO_TOKEN,
    _TOKEN_CZ_CONTROL,
    _TOKEN_CZ_TARGET,
    _TOKEN_CX_CONTROL,
    _TOKEN_CX_TARGET,
)


# ----------------------------------------------------------------------------
# Reference implementation (frozen pre-vectorization snapshot).
# DO NOT change this — it encodes the expected behavior. Production
# `build_boards` must match it on every input.
# ----------------------------------------------------------------------------
def _reference_build_boards(
    qubit_ids,
    qubit_oris,
    num_qubits,
    widths,
    obj_layers,
    num_layers,
    num_objectives,
    last_dir_vertical,
):
    total = qubit_ids.shape[0]
    max_nq = qubit_ids.shape[1]
    max_nl = len(obj_layers)

    boards = np.zeros((total, max_nl, max_nq, 4), dtype=np.int32)

    for i in range(total):
        nq = int(num_qubits[i])
        nl = int(num_layers[i])
        w = int(widths[i]) if i < widths.shape[0] else 0
        if nq <= 0 or w <= 0:
            continue

        pos_map = {}
        for j in range(nq):
            qid = int(qubit_ids[i, j])
            row, col = divmod(j, w)
            pos_map[qid] = (row, col)

        for l in range(min(nl, max_nl)):
            no = int(num_objectives[i, l])
            layer_data = obj_layers[l]

            obj_map = {}

            for k in range(no):
                opcode = int(layer_data["opcodes"][i, k])
                arg0 = int(layer_data["arg0s"][i, k])
                arg1 = int(layer_data["arg1s"][i, k])

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
                    token = _RAW_OPCODE_TO_TOKEN.get(opcode, 1)
                    obj_map[arg0] = (token, -1, -1)

            for j in range(nq):
                qid = int(qubit_ids[i, j])
                ori_token = _RAW_ORI_TO_TOKEN.get(int(qubit_oris[i, j]), 1)

                if qid in obj_map:
                    op, mate_row, mate_col = obj_map[qid]
                    boards[i, l, j, 0] = op
                    boards[i, l, j, 1] = ori_token
                    boards[i, l, j, 2] = mate_row
                    boards[i, l, j, 3] = mate_col
                elif qid < 0:
                    ancilla_idx = -(qid + 1)
                    if (
                        ori_token == 6
                        and ancilla_idx < last_dir_vertical.shape[1]
                        and last_dir_vertical[i, ancilla_idx]
                    ):
                        ori_token = 7
                    boards[i, l, j, 0] = 13 - qid  # -1..-50 -> 14..63
                    boards[i, l, j, 1] = ori_token
                    boards[i, l, j, 2] = -1
                    boards[i, l, j, 3] = -1
                else:
                    boards[i, l, j, 0] = 1
                    boards[i, l, j, 1] = 1
                    boards[i, l, j, 2] = -1
                    boards[i, l, j, 3] = -1

    return boards


# ----------------------------------------------------------------------------
# Helpers for building inputs
# ----------------------------------------------------------------------------
def _make_layer_data(opcodes_2d, arg0s_2d, arg1s_2d):
    """Build a single layer dict in the format build_boards expects."""
    return {
        "opcodes": np.asarray(opcodes_2d, dtype=np.int32),
        "arg0s": np.asarray(arg0s_2d, dtype=np.int32),
        "arg1s": np.asarray(arg1s_2d, dtype=np.int32),
    }


def _empty_layer_data(total, max_no):
    return {
        "opcodes": np.zeros((total, max_no), dtype=np.int32),
        "arg0s": np.zeros((total, max_no), dtype=np.int32),
        "arg1s": np.zeros((total, max_no), dtype=np.int32),
    }


# ----------------------------------------------------------------------------
# Hand-crafted edge cases
# ----------------------------------------------------------------------------
class TestBasicShapes:
    def test_shape_is_total_nl_maxnq_4(self):
        qids = np.array([[0, 1, 2, 3]], dtype=np.int32)
        oris = np.zeros((1, 4), dtype=np.uint8)
        nq = np.array([4], dtype=np.uint16)
        widths = np.array([2], dtype=np.int32)
        nl = np.array([1], dtype=np.uint8)
        no = np.array([[0, 0]], dtype=np.uint16)
        ldv = np.zeros((1, 4), dtype=bool)
        layers = [_empty_layer_data(1, 5), _empty_layer_data(1, 5)]

        boards = build_boards(qids, oris, nq, widths, layers, nl, no, ldv)
        assert boards.shape == (1, 2, 4, 4)
        assert boards.dtype == np.int32

    def test_returns_zeros_when_nq_is_zero(self):
        qids = np.zeros((1, 4), dtype=np.int32)
        oris = np.zeros((1, 4), dtype=np.uint8)
        nq = np.array([0], dtype=np.uint16)
        widths = np.array([2], dtype=np.int32)
        nl = np.array([1], dtype=np.uint8)
        no = np.array([[0]], dtype=np.uint16)
        ldv = np.zeros((1, 4), dtype=bool)
        layers = [_empty_layer_data(1, 1)]

        boards = build_boards(qids, oris, nq, widths, layers, nl, no, ldv)
        assert np.all(boards == 0)


class TestQubitDefaults:
    def test_positive_qid_no_objective_gives_default(self):
        # nq=4 qubits with positive ids, no objectives → default (1,1,-1,-1)
        # except orientation which comes from the ori lookup.
        qids = np.array([[0, 1, 2, 3]], dtype=np.int32)
        oris = np.array([[0, 1, 3, 4]], dtype=np.uint8)  # mapped to 2,3,4,5
        nq = np.array([4], dtype=np.uint16)
        widths = np.array([2], dtype=np.int32)
        nl = np.array([1], dtype=np.uint8)
        no = np.array([[0]], dtype=np.uint16)
        ldv = np.zeros((1, 4), dtype=bool)
        layers = [_empty_layer_data(1, 1)]

        boards = build_boards(qids, oris, nq, widths, layers, nl, no, ldv)
        # All qubits get op=1, mate_row=mate_col=-1
        assert (boards[0, 0, :, 0] == 1).all()
        assert (boards[0, 0, :, 2] == -1).all()
        assert (boards[0, 0, :, 3] == -1).all()
        # Orientations come from _RAW_ORI_TO_TOKEN: 0→2, 1→3, 3→4, 4→5
        # but the production default for "qid in obj_map" path is 1.
        # However when qid is positive and not in obj_map, the code writes
        # ori_token=1 hardcoded. This is the actual behavior we lock down.
        assert (boards[0, 0, :, 1] == 1).all()


class TestAncillaQubits:
    def test_ancilla_qid_negative_writes_ancilla_tokens(self):
        # qid -1, -2 → op = 13 - qid = 14, 15
        # ori_token comes from _RAW_ORI_TO_TOKEN
        qids = np.array([[-1, -2]], dtype=np.int32)
        oris = np.array([[2, 0]], dtype=np.uint8)  # 2→6 (Ancilla), 0→2 (Vertical)
        nq = np.array([2], dtype=np.uint16)
        widths = np.array([2], dtype=np.int32)
        nl = np.array([1], dtype=np.uint8)
        no = np.array([[0]], dtype=np.uint16)
        ldv = np.zeros((1, 50), dtype=bool)
        layers = [_empty_layer_data(1, 1)]

        boards = build_boards(qids, oris, nq, widths, layers, nl, no, ldv)
        # qid=-1 → op=14, ori=6 (no last_dir_vertical, so stays 6)
        assert boards[0, 0, 0, 0] == 14
        assert boards[0, 0, 0, 1] == 6
        # qid=-2 → op=15, ori=2 (Vertical, not Ancilla, no transition)
        assert boards[0, 0, 1, 0] == 15
        assert boards[0, 0, 1, 1] == 2

    def test_ancilla_with_last_dir_vertical_transitions_6_to_7(self):
        qids = np.array([[-1, -3]], dtype=np.int32)  # ancilla_idx = 0 and 2
        oris = np.array([[2, 2]], dtype=np.uint8)  # both → ori_token 6 (Ancilla)
        nq = np.array([2], dtype=np.uint16)
        widths = np.array([2], dtype=np.int32)
        nl = np.array([1], dtype=np.uint8)
        no = np.array([[0]], dtype=np.uint16)
        # last_dir_vertical: index 0 is True, index 2 is False
        ldv = np.zeros((1, 50), dtype=bool)
        ldv[0, 0] = True
        layers = [_empty_layer_data(1, 1)]

        boards = build_boards(qids, oris, nq, widths, layers, nl, no, ldv)
        # qid=-1, ancilla_idx=0, ldv[0,0]=True → ori transitions 6 → 7
        assert boards[0, 0, 0, 1] == 7
        # qid=-3, ancilla_idx=2, ldv[0,2]=False → ori stays 6
        assert boards[0, 0, 1, 1] == 6


class TestObjectives:
    def test_cz_objective_writes_control_and_target(self):
        # Two qubits at positions (0,0) and (0,1) on a 2-wide grid.
        qids = np.array([[10, 20]], dtype=np.int32)
        oris = np.zeros((1, 2), dtype=np.uint8)
        nq = np.array([2], dtype=np.uint16)
        widths = np.array([2], dtype=np.int32)
        nl = np.array([1], dtype=np.uint8)
        no = np.array([[1]], dtype=np.uint16)
        ldv = np.zeros((1, 4), dtype=bool)
        # CZ(qid=10 → control, qid=20 → target)
        layer = _make_layer_data(
            opcodes_2d=[[_RAW_CZ_OPCODE]],
            arg0s_2d=[[10]],
            arg1s_2d=[[20]],
        )
        layers = [layer]

        boards = build_boards(qids, oris, nq, widths, layers, nl, no, ldv)
        # qid=10 (j=0, row=0, col=0) → CZ_CONTROL with mate at (0, 1)
        assert boards[0, 0, 0, 0] == _TOKEN_CZ_CONTROL  # 8
        assert boards[0, 0, 0, 2] == 0
        assert boards[0, 0, 0, 3] == 1
        # qid=20 (j=1, row=0, col=1) → CZ_TARGET with mate at (0, 0)
        assert boards[0, 0, 1, 0] == _TOKEN_CZ_TARGET  # 9
        assert boards[0, 0, 1, 2] == 0
        assert boards[0, 0, 1, 3] == 0

    def test_cx_objective_writes_control_and_target(self):
        qids = np.array([[10, 20]], dtype=np.int32)
        oris = np.zeros((1, 2), dtype=np.uint8)
        nq = np.array([2], dtype=np.uint16)
        widths = np.array([2], dtype=np.int32)
        nl = np.array([1], dtype=np.uint8)
        no = np.array([[1]], dtype=np.uint16)
        ldv = np.zeros((1, 4), dtype=bool)
        layer = _make_layer_data(
            opcodes_2d=[[_RAW_CX_OPCODE]],
            arg0s_2d=[[10]],
            arg1s_2d=[[20]],
        )
        layers = [layer]

        boards = build_boards(qids, oris, nq, widths, layers, nl, no, ldv)
        assert boards[0, 0, 0, 0] == _TOKEN_CX_CONTROL  # 10
        assert boards[0, 0, 1, 0] == _TOKEN_CX_TARGET  # 11

    def test_single_qubit_objective_uses_opcode_lookup(self):
        # opcode 3 → token 3 (H gate)
        qids = np.array([[10, 20]], dtype=np.int32)
        oris = np.zeros((1, 2), dtype=np.uint8)
        nq = np.array([2], dtype=np.uint16)
        widths = np.array([2], dtype=np.int32)
        nl = np.array([1], dtype=np.uint8)
        no = np.array([[1]], dtype=np.uint16)
        ldv = np.zeros((1, 4), dtype=bool)
        # H(qid=10), arg1 unused
        layer = _make_layer_data(
            opcodes_2d=[[3]],
            arg0s_2d=[[10]],
            arg1s_2d=[[0]],
        )
        layers = [layer]

        boards = build_boards(qids, oris, nq, widths, layers, nl, no, ldv)
        # qid=10 → token 3 (H), mate_row=-1, mate_col=-1
        assert boards[0, 0, 0, 0] == 3
        assert boards[0, 0, 0, 2] == -1
        assert boards[0, 0, 0, 3] == -1

    def test_unknown_opcode_falls_back_to_token_1(self):
        qids = np.array([[10, 20]], dtype=np.int32)
        oris = np.zeros((1, 2), dtype=np.uint8)
        nq = np.array([2], dtype=np.uint16)
        widths = np.array([2], dtype=np.int32)
        nl = np.array([1], dtype=np.uint8)
        no = np.array([[1]], dtype=np.uint16)
        ldv = np.zeros((1, 4), dtype=bool)
        # opcode 99 not in _RAW_OPCODE_TO_TOKEN → token 1
        layer = _make_layer_data(
            opcodes_2d=[[99]],
            arg0s_2d=[[10]],
            arg1s_2d=[[0]],
        )
        layers = [layer]

        boards = build_boards(qids, oris, nq, widths, layers, nl, no, ldv)
        assert boards[0, 0, 0, 0] == 1


class TestMatchesReference:
    """The bulk-correctness guard. All other test classes verify specific
    semantics; this one verifies that the production implementation matches
    the frozen reference on a broad random suite."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 17, 42, 100, 256, 9999])
    def test_random_inputs_match_reference(self, seed):
        rng = np.random.default_rng(seed)
        total = int(rng.integers(1, 16))
        max_nq = int(rng.integers(1, 30))
        max_nl = int(rng.integers(1, 4))
        max_no = int(rng.integers(1, 20))
        # Width small so positions wrap predictably.
        widths = rng.integers(1, 6, size=total).astype(np.int32)

        # Mix positive and negative qids; some duplicates to test pos_map
        # last-write-wins semantics.
        qids = rng.integers(-10, 50, size=(total, max_nq)).astype(np.int32)
        oris = rng.integers(0, 8, size=(total, max_nq)).astype(np.uint8)

        # nq, nl, no with some zero entries
        num_qubits = rng.integers(0, max_nq + 1, size=total).astype(np.uint16)
        num_layers = rng.integers(0, max_nl + 1, size=total).astype(np.uint8)
        num_objectives = rng.integers(
            0, max_no + 1, size=(total, max_nl)
        ).astype(np.uint16)

        last_dir_vertical = rng.integers(
            0, 2, size=(total, 50)
        ).astype(bool)

        # Build random layers. Each entry is opcodes/arg0/arg1.
        opcode_choices = (
            list(_RAW_OPCODE_TO_TOKEN.keys()) + [_RAW_CX_OPCODE, _RAW_CZ_OPCODE]
        )
        layers = []
        for _ in range(max_nl):
            opcodes = rng.choice(
                opcode_choices, size=(total, max_no)
            ).astype(np.int32)
            arg0s = rng.integers(-10, 50, size=(total, max_no)).astype(np.int32)
            arg1s = rng.integers(-10, 50, size=(total, max_no)).astype(np.int32)
            layers.append(_make_layer_data(opcodes, arg0s, arg1s))

        actual = build_boards(
            qids, oris, num_qubits, widths,
            layers, num_layers, num_objectives, last_dir_vertical,
        )
        expected = _reference_build_boards(
            qids, oris, num_qubits, widths,
            layers, num_layers, num_objectives, last_dir_vertical,
        )
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        # Print first divergence for diagnosis.
        if not np.array_equal(actual, expected):
            diff_idx = np.argwhere(actual != expected)
            first = diff_idx[0]
            i, l, j, c = first
            pytest.fail(
                f"Mismatch at (i={i}, l={l}, j={j}, c={c}): "
                f"actual={actual[i, l, j, c]} expected={expected[i, l, j, c]} "
                f"(qid={qids[i, j]}, ori={oris[i, j]}, "
                f"nq={num_qubits[i]}, nl={num_layers[i]}, "
                f"no_il={num_objectives[i, l]})"
            )
        assert np.array_equal(actual, expected)
