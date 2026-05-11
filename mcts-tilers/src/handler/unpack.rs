// Rust replacements for handler.py's unpack_placement_batch and
// unpack_objectives_batch. Take the raw u8 slabs the shared-memory arena
// hands over and produce typed numpy arrays the model + build_boards
// consume.
//
// Output dtype note: handler.py's Python `unpack_objectives_batch` writes
// opcodes into an `np.int8` array. We promote to `i32` here so the result
// is a drop-in input to `build_boards_rs` (which expects `PyReadonlyArray2<i32>`
// for opcodes). The sign-extension matches Python's `int(op_arr[k])` semantics
// for the only opcodes that exist in practice (0..19, well below the i8
// negative range), so no behavior change vs the reference.
//
// Correctness gate: tests/test_build_boards.py is extended with
// TestUnpackMatchesPythonRust to verify the Rust outputs match the Python
// reference bit-for-bit on a random-seed sweep.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::constants::{OBJECTIVE_SIZE, OBJECTIVES_LAYER_MAX, QUBIT_SIZE};

// ----------------------------------------------------------------------------
// unpack_placement_batch_rs
// ----------------------------------------------------------------------------
// raw[i, : nq*QUBIT_SIZE] is a packed sequence of (i32 id, u8 orientation)
// pairs. Each qubit occupies QUBIT_SIZE=5 bytes: bytes 0..4 are the id (LE i32),
// byte 4 is the orientation. We unpack into two parallel arrays.

#[pyfunction]
pub fn unpack_placement_batch_rs<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray2<'py, u8>,         // (total, PLACEMENT_MAX)
    num_qubits: PyReadonlyArray1<'py, u16>, // (total,)
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<u8>>)> {
    let raw_view = raw.as_array();
    let nq_view = num_qubits.as_array();

    let total = raw_view.shape()[0];
    // Matches Python: max_nq = max(1, int(num_qubits.max())).
    let max_nq = (nq_view.iter().copied().max().unwrap_or(0) as usize).max(1);

    let mut ids = Array2::<i32>::zeros((total, max_nq));
    let mut oris = Array2::<u8>::zeros((total, max_nq));

    // Empty-input fast path. Same shape as Python returns for total=0.
    if total == 0 {
        return Ok((ids.into_pyarray(py), oris.into_pyarray(py)));
    }

    let ids_slice = ids
        .as_slice_mut()
        .expect("Array2::zeros produces contiguous storage");
    let oris_slice = oris
        .as_slice_mut()
        .expect("Array2::zeros produces contiguous storage");

    // Parallel per-instance. zip the two output slices so each rayon iter
    // owns exclusive write access to its row of both ids and oris.
    ids_slice
        .par_chunks_mut(max_nq)
        .zip(oris_slice.par_chunks_mut(max_nq))
        .enumerate()
        .for_each(|(i, (ids_row, oris_row))| {
            let nq = nq_view[i] as usize;
            if nq == 0 {
                return;
            }
            for j in 0..nq {
                let off = j * QUBIT_SIZE;
                let b0 = raw_view[[i, off]];
                let b1 = raw_view[[i, off + 1]];
                let b2 = raw_view[[i, off + 2]];
                let b3 = raw_view[[i, off + 3]];
                ids_row[j] = i32::from_le_bytes([b0, b1, b2, b3]);
                oris_row[j] = raw_view[[i, off + 4]];
            }
        });

    Ok((ids.into_pyarray(py), oris.into_pyarray(py)))
}

// ----------------------------------------------------------------------------
// unpack_objectives_batch_rs
// ----------------------------------------------------------------------------
// For each layer l ∈ [0, max_nl), build a dict with three (total, max_no) i32
// arrays: opcodes, arg0s, arg1s. Each objective in raw occupies OBJECTIVE_SIZE=9
// bytes at offset `l * OBJECTIVES_LAYER_MAX + k * OBJECTIVE_SIZE`:
//   byte 0:   opcode  (i8, sign-extended to i32 here)
//   bytes 1-4: arg0   (i32 LE)
//   bytes 5-8: arg1   (i32 LE)
//
// Returns a Python list of dicts in the format build_boards_rs (and the Python
// build_boards reference) consume.

#[pyfunction]
pub fn unpack_objectives_batch_rs<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray2<'py, u8>,             // (total, OBJECTIVES_MAX)
    num_layers: PyReadonlyArray1<'py, u8>,      // (total,)
    num_objectives: PyReadonlyArray2<'py, u16>, // (total, LOOKAHEAD_MAX)
) -> PyResult<Bound<'py, PyList>> {
    let raw_view = raw.as_array();
    let nl_view = num_layers.as_array();
    let no_view = num_objectives.as_array();

    let total = raw_view.shape()[0];
    // Matches Python: max_nl = max(1, int(num_layers.max())).
    let max_nl = (nl_view.iter().copied().max().unwrap_or(0) as usize).max(1);
    let lookahead_max = no_view.shape()[1];

    let result = PyList::empty(py);

    for l in 0..max_nl {
        // Matches Python: per-layer max_no = max(1, int(num_objectives[:, l].max()))
        // (or 1 if l is past the num_objectives axis).
        let max_no = if l < lookahead_max {
            let mut m: u16 = 0;
            for i in 0..total {
                let v = no_view[[i, l]];
                if v > m {
                    m = v;
                }
            }
            (m as usize).max(1)
        } else {
            1
        };

        let mut opcodes = Array2::<i32>::zeros((total, max_no));
        let mut arg0s = Array2::<i32>::zeros((total, max_no));
        let mut arg1s = Array2::<i32>::zeros((total, max_no));

        if total > 0 {
            let layer_start = l * OBJECTIVES_LAYER_MAX;
            let op_slice = opcodes
                .as_slice_mut()
                .expect("Array2::zeros produces contiguous storage");
            let a0_slice = arg0s
                .as_slice_mut()
                .expect("Array2::zeros produces contiguous storage");
            let a1_slice = arg1s
                .as_slice_mut()
                .expect("Array2::zeros produces contiguous storage");

            // Parallelize per-instance. Triple-zip via nested zips; unpack
            // the nested tuple in the closure. Each rayon iter owns its
            // row in all three output arrays.
            op_slice
                .par_chunks_mut(max_no)
                .zip(a0_slice.par_chunks_mut(max_no))
                .zip(a1_slice.par_chunks_mut(max_no))
                .enumerate()
                .for_each(|(i, ((op_row, a0_row), a1_row))| {
                    if l >= nl_view[i] as usize {
                        return;
                    }
                    let no = if l < lookahead_max {
                        no_view[[i, l]] as usize
                    } else {
                        0
                    };
                    if no == 0 {
                        return;
                    }
                    for k in 0..no {
                        let off = layer_start + k * OBJECTIVE_SIZE;
                        // opcode: u8 → i8 → i32 sign-extends; matches Python's
                        // int8 view of the raw byte, then int(op_arr[k]).
                        op_row[k] = raw_view[[i, off]] as i8 as i32;
                        // arg0 / arg1: LE i32 from bytes 1..4 / 5..8.
                        let a0 = [
                            raw_view[[i, off + 1]],
                            raw_view[[i, off + 2]],
                            raw_view[[i, off + 3]],
                            raw_view[[i, off + 4]],
                        ];
                        let a1 = [
                            raw_view[[i, off + 5]],
                            raw_view[[i, off + 6]],
                            raw_view[[i, off + 7]],
                            raw_view[[i, off + 8]],
                        ];
                        a0_row[k] = i32::from_le_bytes(a0);
                        a1_row[k] = i32::from_le_bytes(a1);
                    }
                });
        }

        let d = PyDict::new(py);
        d.set_item("opcodes", opcodes.into_pyarray(py))?;
        d.set_item("arg0s", arg0s.into_pyarray(py))?;
        d.set_item("arg1s", arg1s.into_pyarray(py))?;
        result.append(d)?;
    }

    Ok(result)
}
