use std::slice::from_raw_parts;
use std::sync::atomic::Ordering;

use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray1, PyArray2};
use numpy::PyArrayMethods;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Bound;

use mcts_core::ipc_core::{now_ns, Arena};

use crate::constants::*;
use crate::slot::TilersSlot;

#[cfg(feature = "test-helpers")]
use tilers::env::Environment as TilersEnvInner;
#[cfg(feature = "test-helpers")]
use mcts_core::{Environment, InferenceClient};
#[cfg(feature = "test-helpers")]
use crate::client::TilersIpcClient;
#[cfg(feature = "test-helpers")]
use crate::environment::TilersEnv;

// ─────────────────────────────────────────────────────────────────────────────
// PyArena
// ─────────────────────────────────────────────────────────────────────────────
#[pyclass(unsendable)]
pub struct PyArena {
    arena: Arena<TilersSlot>,
    arena_name: String,
    num_handlers: usize,
}

#[pymethods]
impl PyArena {
    #[new]
    pub fn new(name: String, num_slots: usize, num_handlers: usize) -> PyResult<Self> {
        let arena = Arena::create_or_open(&name, num_slots, num_handlers)
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
        Ok( Self { arena, arena_name: name, num_handlers } )
    }

    pub fn num_slots(&self) -> u32 {
        self.arena.num_slots()
    }

    pub fn num_handlers(&self) -> usize {
        self.num_handlers
    }

    pub fn arena_name(&self) -> &str {
        &self.arena_name
    }

    pub fn pop_ready(&self, handler: usize) -> u32 {
        self.arena.pop_ready(handler)
    }

    pub fn try_pop_ready(&self, handler: usize) -> Option<u32> {
        self.arena.try_pop_ready(handler)
    }

    pub fn mark_done(&self, slot: u32) {
        self.arena.mark_done(slot)
    }

    pub fn clear_outputs(&self, slot: u32) {
        let sm = self.arena.slot_mut(slot);
        sm.slot.priors.fill(0.0);
        sm.slot.values.fill(0.0);
    }

    #[pyo3(signature = (handler, clear_outputs=false))]
    pub fn pop_ready_view<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        handler: usize,
        clear_outputs: bool,
    ) -> PyResult<Py<PySlotView>> {
        let slot = slf.arena.pop_ready(handler);
        if clear_outputs {
            let sm = slf.arena.slot_mut(slot);
            sm.slot.priors.fill(0.0);
            sm.slot.values.fill(0.0);
        }
        let ptr = slf.arena.slot_ptr(slot);
        let arena_obj: Py<PyArena> = slf.into_pyobject(py)?.unbind();
        Py::new(py, PySlotView { arena: arena_obj, slot, ptr })
    }

    #[pyo3(signature = (handler, clear_outputs=false))]
    pub fn try_pop_ready_view<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        handler: usize,
        clear_outputs: bool,
    ) -> PyResult<Option<Py<PySlotView>>> {
        if let Some(slot) = slf.arena.try_pop_ready(handler) {
            if clear_outputs {
                let sm = slf.arena.slot_mut(slot);
                sm.slot.priors.fill(0.0);
                sm.slot.values.fill(0.0);
            }
            let ptr = slf.arena.slot_ptr(slot);
            let arena_obj: Py<PyArena> = slf.into_pyobject(py)?.unbind();
            let view = Py::new(py, PySlotView { arena: arena_obj, slot, ptr })?;
            Ok(Some(view))
        } else {
            Ok(None)
        }
    }

    /// Temp test helper: create an env, pack obs, submit to handler, wait, return results.
    #[cfg(feature = "test-helpers")]
    fn submit_and_collect(
        &self,
        h: usize,
        w: usize,
        num_blanks: usize,
        num_objectives: usize,
        seed: u64,
    ) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        let mut env = TilersEnvInner::new(h, w, num_blanks);
        env.set_seed(Some(seed));
        env.random_objectives(num_objectives, false);
        let tilers_env = TilersEnv::new(env, 2);

        let obs = tilers_env.observation();
        let observations = vec![obs];

        // Open a second handle to the same shared memory arena
        let arena2 = Arena::<TilersSlot>::create_or_open(
            &self.arena_name,
            self.num_slots() as usize,
            self.num_handlers,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let client = TilersIpcClient::new(arena2, 9999);
        println!("[rust] about to call infer");
        let (priors, values) = client
            .infer(&observations)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        println!("[rust] finished calling infer");

        let priors_vecs: Vec<Vec<f32>> = priors
            .iter()
            .map(|a| {
                (0..NUM_ACTIONS)
                    .map(|a_idx| *a.get(&(a_idx as Action)).unwrap_or(&0.0))
                    .collect()
            })
            .collect();

        Ok((priors_vecs, values))
    }

    /// Reset the arena
    pub fn force_reset(&self) { self.arena.force_reset(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Numpy helpers
// ─────────────────────────────────────────────────────────────────────────────
unsafe fn copy_to_array1<'a, T: Element + Copy>(
    py: Python<'a>,
    data_ptr: *const T,
    len: usize,
) -> Bound<'a, PyArray1<T>> {
    let slice = unsafe { from_raw_parts(data_ptr, len) };
    PyArray1::from_slice(py, slice)
}

unsafe fn copy_to_array2<'a, T: Element + Copy>(
    py: Python<'a>,
    data_ptr: *const T,
    rows: usize,
    cols: usize,
) -> Bound<'a, PyArray2<T>> {
    let len = rows.checked_mul(cols).expect("overflow");
    let slice = unsafe { from_raw_parts(data_ptr, len) };
    let vec = slice.to_vec();
    let arr: Array2<T> = Array2::from_shape_vec((rows, cols), vec).expect("shape/len mismatch");
    arr.into_pyarray(py)
}

// ─────────────────────────────────────────────────────────────────────────────
// PySlotView
// ─────────────────────────────────────────────────────────────────────────────
#[pyclass(unsendable)]
pub struct PySlotView {
    #[allow(dead_code)]
    arena: Py<PyArena>,
    #[pyo3(get)]
    slot: u32,
    ptr: *mut TilersSlot,
}

impl PySlotView {
    fn slot_ref(&self) -> PyResult<&TilersSlot> {
        if self.ptr.is_null() {
            return Err(PyRuntimeError::new_err("null slot pointer"));
        }
        Ok(unsafe { &*self.ptr })
    }

    fn batch_len(&self) -> PyResult<usize> {
        let s = self.slot_ref()?;
        Ok((s.b as usize).min(MAX_BATCH))
    }
}

#[pymethods]
impl PySlotView {
    pub fn b(&self) -> PyResult<usize> {
        self.batch_len()
    }

    pub fn set_handler_start_time(&self) -> PyResult<()> {
        let s = self.slot_ref()?;
        s.handler_start_time_ns.store(now_ns(), Ordering::Release);
        Ok(())
    }

    // ── scalar metadata (per batch element) ─────────────────────────────

    pub fn h<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array1(py, s.h.as_ptr(), b) })
    }

    pub fn w<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array1(py, s.w.as_ptr(), b) })
    }

    pub fn num_ancillas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array1(py, s.num_ancillas.as_ptr(), b) })
    }

    pub fn num_qubits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array1(py, s.num_qubits.as_ptr(), b) })
    }

    pub fn num_layers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array1(py, s.num_layers.as_ptr(), b) })
    }

    /// Returns shape (b, LOOKAHEAD_MAX)
    pub fn num_objectives<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u16>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array2(py, s.num_objectives.as_ptr() as *const u16, b, LOOKAHEAD_MAX) })
    }

    // ── raw packed data ─────────────────────────────────────────────────

    /// Returns shape (b, PLACEMENT_MAX) as raw bytes.
    /// Python unpacks using QUBIT_SIZE stride: [i32_le id, u8 orientation].
    pub fn placement<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array2(py, s.placement.as_ptr() as *const u8, b, PLACEMENT_MAX) })
    }

    /// Returns shape (b, OBJECTIVES_MAX) as raw bytes.
    /// Python unpacks using OBJECTIVE_SIZE stride per objective,
    /// OBJECTIVES_LAYER_MAX stride per layer.
    pub fn objectives<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array2(py, s.objectives.as_ptr() as *const u8, b, OBJECTIVES_MAX) })
    }

    /// Returns shape (b, NUM_ACTIONS)
    pub fn action_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array2(py, s.action_mask.as_ptr(), b, NUM_ACTIONS) })
    }

    // ── outputs ──────────────────────────────────────────────────────────

    pub fn priors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array2(py, s.priors.as_ptr(), b, NUM_ACTIONS) })
    }

    pub fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let s = self.slot_ref()?;
        let b = self.batch_len()?;
        Ok(unsafe { copy_to_array1(py, s.values.as_ptr(), b) })
    }

    /// Write priors (b, NUM_ACTIONS) and values (b,) back into the slot.
    pub fn write_priors_values<'py>(
        &self,
        _py: Python<'py>,
        priors: &Bound<'py, PyArray2<f32>>,
        values: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<()> {
        let b = self.batch_len()?;
        let s = unsafe { &mut *self.ptr };

        let priors_ro = priors.readonly();
        let priors_slice = priors_ro.as_slice()
            .map_err(|e| PyRuntimeError::new_err(format!("priors not contiguous: {e}")))?;

        if priors_slice.len() != b * NUM_ACTIONS {
            return Err(PyRuntimeError::new_err(format!(
                "expected priors of len {}, got {}",
                b * NUM_ACTIONS,
                priors_slice.len()
            )));
        }

        let values_ro = values.readonly();
        let values_slice = values_ro.as_slice()
            .map_err(|e| PyRuntimeError::new_err(format!("values not contiguous: {e}")))?;

        if values_slice.len() != b {
            return Err(PyRuntimeError::new_err(format!(
                "expected values of len {}, got {}",
                b,
                values_slice.len()
            )));
        }

        s.priors[..b * NUM_ACTIONS].copy_from_slice(priors_slice);
        s.values[..b].copy_from_slice(values_slice);

        Ok(())
    }

    pub fn mark_done(&self, py: Python<'_>) -> PyResult<()> {
        let arena_ref = self.arena.borrow(py);
        arena_ref.arena.mark_done(self.slot);
        Ok(())
    }
}