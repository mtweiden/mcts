// Rust replacement for handler.py's build_boards. Signature mirrors the
// Python implementation exactly so handler.py can swap in build_boards_rs
// behind the MCTS_HANDLER_RUST env var without further code changes.
//
// Correctness target: tests/test_build_boards.py — the `_reference_build_boards`
// in that file is the source of truth. Output must match it bit-for-bit on
// every input.
//
// Parallelism: none. The outer (total,) loop is embarrassingly parallel,
// but the gather pipeline runs many handler processes concurrently — each
// would over-spawn Rayon threads (default = num CPUs), causing severe
// thread oversubscription that erased the per-call speedup in production
// (measured 2026-05-11). Sequential iteration here is the right choice;
// process-level parallelism does the work.

use numpy::ndarray::{Array2, Array4};
use numpy::{IntoPyArray, PyArray4, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustc_hash::FxHashMap;

// ----------------------------------------------------------------------------
// Constants — must stay in lockstep with handler.py's _RAW_*, _TOKEN_*, _ORI_LUT,
// and _OPCODE_LUT. Any change here must be made in handler.py first (or vice
// versa) and the parametrized seed test guards against drift.
// ----------------------------------------------------------------------------
const RAW_CX_OPCODE: i32 = 12;
const RAW_CZ_OPCODE: i32 = 13;
const TOKEN_CZ_CONTROL: i32 = 8;
const TOKEN_CZ_TARGET: i32 = 9;
const TOKEN_CX_CONTROL: i32 = 10;
const TOKEN_CX_TARGET: i32 = 11;

const OPCODE_LUT_SIZE: usize = 64;

// Mirrors handler.py's _OPCODE_LUT: raw_opcode → board-vocabulary token.
// Default (unmapped) is 1. CX/CZ raw opcodes are NOT in this table — they
// take the dedicated control/target branch.
const OPCODE_LUT: [i32; OPCODE_LUT_SIZE] = {
    let mut lut = [1i32; OPCODE_LUT_SIZE];
    lut[0] = 2;   // X
    lut[1] = 2;   // Y
    lut[2] = 2;   // Z
    lut[3] = 3;   // H
    lut[4] = 4;   // S
    lut[5] = 4;   // Sdg
    lut[6] = 5;   // SX
    lut[7] = 5;   // SXdg
    lut[8] = 6;   // T
    lut[9] = 6;   // Tdg
    lut[10] = 7;  // TX
    lut[11] = 7;  // TXdg
    lut[14] = 2;  // RZ
    lut[18] = 12; // MEASURE
    lut[19] = 13; // RESET
    lut
};

// Mirrors handler.py's _ORI_LUT: raw u8 orientation → token. Default is 1.
const ORI_LUT: [i32; 256] = {
    let mut lut = [1i32; 256];
    lut[0] = 2; // Vertical
    lut[1] = 3; // Horizontal
    lut[2] = 6; // Ancilla default (last-move horizontal); the 6→7 transition
                // is applied later when the corresponding last_dir_vertical
                // bit is set.
    lut[3] = 4; // Cultivating
    lut[4] = 5; // Resource
    lut
};

#[pyfunction]
pub fn build_boards_rs<'py>(
    py: Python<'py>,
    qubit_ids: PyReadonlyArray2<'py, i32>,      // (total, max_nq)
    qubit_oris: PyReadonlyArray2<'py, u8>,      // (total, max_nq)
    num_qubits: PyReadonlyArray1<'py, u16>,     // (total,)
    widths: PyReadonlyArray1<'py, u8>,          // (total,)  — slot view returns u8
    obj_layers: Bound<'py, PyList>,             // list[dict[str, np.ndarray]]
    num_layers: PyReadonlyArray1<'py, u8>,      // (total,)
    num_objectives: PyReadonlyArray2<'py, u16>, // (total, LOOKAHEAD_MAX)
    last_dir_vertical: PyReadonlyArray2<'py, u8>, // (total, MAX_ANCILLAS) — slot view returns u8 (0/1)
) -> PyResult<Bound<'py, PyArray4<i32>>> {
    let qids = qubit_ids.as_array();
    let oris = qubit_oris.as_array();
    let nq_arr = num_qubits.as_array();
    let w_arr = widths.as_array();
    let nl_arr = num_layers.as_array();
    let no_arr = num_objectives.as_array();
    let ldv_arr = last_dir_vertical.as_array();

    let total = qids.shape()[0];
    let max_nq = qids.shape()[1];
    let max_nl = obj_layers.len();
    let ldv_max = ldv_arr.shape()[1];

    // Precompute ori_tokens_all = _ORI_LUT[oris] once, before the parallel
    // section. Each per-instance pass then indexes this view directly
    // instead of recomputing the lookup.
    let mut ori_tokens_all = Array2::<i32>::zeros((total, max_nq));
    for ((i, j), &raw) in oris.indexed_iter() {
        ori_tokens_all[[i, j]] = ORI_LUT[raw as usize];
    }
    let ori_view = ori_tokens_all.view();

    // Extract obj_layers' opcode/arg0/arg1 numpy arrays. PyReadonlyArray2
    // borrows the underlying numpy buffer — keep the holder Vec alive
    // throughout the parallel section so the views remain valid.
    let mut layer_holders: Vec<(
        PyReadonlyArray2<'py, i32>,
        PyReadonlyArray2<'py, i32>,
        PyReadonlyArray2<'py, i32>,
    )> = Vec::with_capacity(max_nl);
    for layer in obj_layers.iter() {
        let d = layer.cast::<PyDict>()?;
        let opcodes: PyReadonlyArray2<i32> = d
            .get_item("opcodes")?
            .ok_or_else(|| PyKeyError::new_err("opcodes"))?
            .extract()?;
        let arg0s: PyReadonlyArray2<i32> = d
            .get_item("arg0s")?
            .ok_or_else(|| PyKeyError::new_err("arg0s"))?
            .extract()?;
        let arg1s: PyReadonlyArray2<i32> = d
            .get_item("arg1s")?
            .ok_or_else(|| PyKeyError::new_err("arg1s"))?
            .extract()?;
        layer_holders.push((opcodes, arg0s, arg1s));
    }
    let layer_views: Vec<_> = layer_holders
        .iter()
        .map(|(op, a0, a1)| (op.as_array(), a0.as_array(), a1.as_array()))
        .collect();

    // Output buffer. Array4::zeros gives C-order contiguous storage; the
    // .as_slice_mut() unwrap is safe by construction.
    let mut boards = Array4::<i32>::zeros((total, max_nl, max_nq, 4));
    let chunk_size = max_nl * max_nq * 4;

    // Empty-input fast path. chunks_mut(0) panics, and there's nothing
    // to do when total == 0 either — the zero-initialised array is already
    // the correct answer. Reference implementation returns the same empty
    // shape on these inputs.
    if total == 0 || chunk_size == 0 {
        return Ok(boards.into_pyarray(py));
    }

    boards
        .as_slice_mut()
        .expect("Array4::zeros produces contiguous storage")
        .chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, boards_i)| {
            let nq = nq_arr[i] as usize;
            let nl = (nl_arr[i] as usize).min(max_nl);
            let w = w_arr[i] as i32;
            if nq == 0 || w <= 0 {
                // Default-zeroed slab is already correct; nothing to do.
                return;
            }

            // pos_map: qid → (row, col). Built once per instance from the
            // first nq active qubits.
            let mut pos_map: FxHashMap<i32, (i32, i32)> =
                FxHashMap::with_capacity_and_hasher(nq, Default::default());
            for j in 0..nq {
                let qid = qids[[i, j]];
                pos_map.insert(qid, ((j as i32) / w, (j as i32) % w));
            }

            for l in 0..nl {
                let no = no_arr[[i, l]] as usize;

                // obj_map: qid → (op_token, mate_row, mate_col). Last-write-
                // wins on duplicate arg0 keys matches the Python dict
                // semantics that the reference implementation relies on.
                let mut obj_map: FxHashMap<i32, (i32, i32, i32)> = FxHashMap::default();
                if no > 0 {
                    let (op_arr, a0_arr, a1_arr) = &layer_views[l];
                    for k in 0..no {
                        let opcode = op_arr[[i, k]];
                        let arg0 = a0_arr[[i, k]];
                        let arg1 = a1_arr[[i, k]];

                        if opcode == RAW_CZ_OPCODE {
                            if let (Some(&(r0, c0)), Some(&(r1, c1))) =
                                (pos_map.get(&arg0), pos_map.get(&arg1))
                            {
                                obj_map.insert(arg0, (TOKEN_CZ_CONTROL, r1, c1));
                                obj_map.insert(arg1, (TOKEN_CZ_TARGET, r0, c0));
                            }
                        } else if opcode == RAW_CX_OPCODE {
                            if let (Some(&(r0, c0)), Some(&(r1, c1))) =
                                (pos_map.get(&arg0), pos_map.get(&arg1))
                            {
                                obj_map.insert(arg0, (TOKEN_CX_CONTROL, r1, c1));
                                obj_map.insert(arg1, (TOKEN_CX_TARGET, r0, c0));
                            }
                        } else {
                            let token = if (0..OPCODE_LUT_SIZE as i32).contains(&opcode) {
                                OPCODE_LUT[opcode as usize]
                            } else {
                                1
                            };
                            obj_map.insert(arg0, (token, -1, -1));
                        }
                    }
                }

                // Per-qubit writes. boards_i is the flat (max_nl, max_nq, 4)
                // slab for this instance; index as layer_offset + j*4 + c.
                let layer_offset = l * max_nq * 4;
                for j in 0..nq {
                    let qid = qids[[i, j]];
                    let ori_lookup_val = ori_view[[i, j]];
                    let off = layer_offset + j * 4;

                    if let Some(&(op_v, mr_v, mc_v)) = obj_map.get(&qid) {
                        // 1. Objective branch: use looked-up op + ori, with
                        //    mate coordinates from obj_map.
                        boards_i[off] = op_v;
                        boards_i[off + 1] = ori_lookup_val;
                        boards_i[off + 2] = mr_v;
                        boards_i[off + 3] = mc_v;
                    } else if qid < 0 {
                        // 2. Ancilla branch: token = 13 - qid (so -1 → 14,
                        //    -2 → 15, etc). Ori usually inherits the
                        //    looked-up value, with a 6 → 7 transition when
                        //    the corresponding last_dir_vertical bit is set.
                        let op = 13 - qid;
                        let anc_idx = -(qid as i64 + 1);
                        let mut ori = ori_lookup_val;
                        if ori == 6 && anc_idx >= 0 && (anc_idx as usize) < ldv_max
                        {
                            if ldv_arr[[i, anc_idx as usize]] != 0 {
                                ori = 7;
                            }
                        }
                        boards_i[off] = op;
                        boards_i[off + 1] = ori;
                        boards_i[off + 2] = -1;
                        boards_i[off + 3] = -1;
                    } else {
                        // 3. Default branch: (op=1, ori=1, mate=-1, -1).
                        boards_i[off] = 1;
                        boards_i[off + 1] = 1;
                        boards_i[off + 2] = -1;
                        boards_i[off + 3] = -1;
                    }
                }
            }
        });

    Ok(boards.into_pyarray(py))
}
