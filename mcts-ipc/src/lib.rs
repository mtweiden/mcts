use std::slice::from_raw_parts;
use numpy::{Element, PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use mcts_core::enums::{GRID_MAX, MAX_BATCH, MAX_OBJ0, MAX_OBJ1, NUM_ACTIONS};
use mcts_core::ipc_core::{Arena, Slot};

use numpy::ndarray::Array2;
use numpy::IntoPyArray;

 
#[pymodule]
fn mcts_ipc(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArena>()?;
    m.add_class::<PySlotView>()?;

    // Expose constants (handy on the Python side)
    m.add("MAX_BATCH", MAX_BATCH)?;
    m.add("GRID_MAX", GRID_MAX)?;
    m.add("MAX_OBJ0", MAX_OBJ0)?;
    m.add("MAX_OBJ1", MAX_OBJ1)?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    Ok(())
}

#[pyclass(unsendable)]
pub struct PyArena {
    arena: Arena,
}

#[pymethods]
impl PyArena {
    #[new]
    pub fn new(name: String, num_slots: usize, num_handlers: usize) -> PyResult<Self> {
        let arena = Arena::create_or_open(&name, num_slots, num_handlers)
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
        Ok(Self { arena })
    }

    pub fn num_slots(&self) -> u32 {
        self.arena.num_slots()
    }

    pub fn pop_ready(&self, handler: usize) -> u32 {
        self.arena.pop_ready(handler)
    }

    pub fn mark_done(&self, slot: u32) {
        self.arena.mark_done(slot)
    }

    pub fn clear_outputs(&self, slot: u32) {
        self.arena.clear_outputs(slot)
    }

    /// Pop a ready slot and return a view into its shared-memory data.
    #[pyo3(signature = (handler, clear_outputs=false))]
    pub fn pop_ready_view<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        handler: usize,
        clear_outputs: bool,
    ) -> PyResult<Py<PySlotView>> {
        // pop slot id
        let slot = slf.arena.pop_ready(handler);
        if clear_outputs { slf.arena.clear_outputs(slot); }

        // get slot pointer from arena (you made slot_ptr public)
        let ptr = slf.arena.slot_ptr(slot);

        // keep arena alive by storing a Py<PyArena> inside the view
        let arena_obj: Py<PyArena> = slf.into_pyobject(py)?.unbind();
        Py::new(py, PySlotView { arena: arena_obj, slot, ptr })
    }
}

#[pyclass(unsendable)]
pub struct PySlotView {
    // keep mmap alive
    arena: Py<PyArena>,
    #[pyo3(get)]
    slot: u32,
    ptr: *mut Slot,
}

// ---- helpers for numpy 0.27 zero-copy views ----

unsafe fn view1<'a, T: Element>(
    py: Python<'a>,
    data_ptr: *mut T,
    len: usize,
) -> Bound<'a, PyArray1<T>> {
    // copy the underlying memory into a Rust slice -> new ndarray -> Python array
    let slice = unsafe { from_raw_parts(data_ptr as *const T, len) };
    PyArray1::from_slice(py, slice)
}

unsafe fn view2<'a, T: Element + Copy>(
    py: Python<'a>,
    data_ptr: *mut T,
    rows: usize,
    cols: usize,
) -> Bound<'a, PyArray2<T>> {
    let len = rows.checked_mul(cols).expect("overflow");
    let slice = unsafe { from_raw_parts(data_ptr as *const T, len) };
    let vec = slice.to_vec();
    let arr: Array2<T> = Array2::from_shape_vec((rows, cols), vec).expect("shape/len mismatch");
    arr.into_pyarray(py)
}

fn batch_len_from_slot(slot: &Slot) -> usize {
    let b = slot.b as usize;
    b.min(MAX_BATCH)
}

#[pymethods]
impl PySlotView {
    pub fn b(&self) -> PyResult<usize> {
        unsafe {
            if self.ptr.is_null() { return Err(PyRuntimeError::new_err("null slot pointer")); }
            Ok(batch_len_from_slot(&*self.ptr))
        }
    }

    // ---- inputs (views) ----
    pub fn h<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        unsafe {
            let s = &mut *self.ptr;
            let b = batch_len_from_slot(s);
            Ok(view1(py, s.h.as_mut_ptr(), b))
        }
    }

    pub fn w<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        unsafe {
            let s = &mut *self.ptr;
            let b = batch_len_from_slot(s);
            Ok(view1(py, s.w.as_mut_ptr(), b))
        }
    }

    pub fn obj0_len<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        unsafe {
            let s = &mut *self.ptr;
            let b = batch_len_from_slot(s);
            Ok(view1(py, s.obj0_len.as_mut_ptr(), b))
        }
    }

    pub fn obj1_len<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        unsafe {
            let s = &mut *self.ptr;
            let b = batch_len_from_slot(s);
            Ok(view1(py, s.obj1_len.as_mut_ptr(), b))
        }
    }

    pub fn placement<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u16>>> {
        unsafe {
            let s = &mut *self.ptr;
            let b = batch_len_from_slot(s);
            Ok(view2(py, s.placement.as_mut_ptr(), b, GRID_MAX))
        }
    }

    pub fn obj0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u16>>> {
        unsafe {
            let s = &mut *self.ptr;
            let b = batch_len_from_slot(s);
            Ok(view2(py, s.obj0.as_mut_ptr(), b, MAX_OBJ0))
        }
    }

    pub fn obj1<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u16>>> {
        unsafe {
            let s = &mut *self.ptr;
            let b = batch_len_from_slot(s);
            Ok(view2(py, s.obj1.as_mut_ptr(), b, MAX_OBJ1))
        }
    }

    pub fn action_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        unsafe {
            let s = &mut *self.ptr;
            let b = batch_len_from_slot(s);
            Ok(view2(py, s.action_mask.as_mut_ptr(), b, NUM_ACTIONS))
        }
    }

    // ---- outputs (views) ----
    pub fn priors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        unsafe {
            let s = &mut *self.ptr;
            let b = batch_len_from_slot(s);
            Ok(view2(py, s.priors.as_mut_ptr(), b, NUM_ACTIONS))
        }
    }

    pub fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        unsafe {
            let s = &mut *self.ptr;
            let b = batch_len_from_slot(s);
            Ok(view1(py, s.values.as_mut_ptr(), b))
        }
    }

    /// Convenience: mark this slot done using the backing arena.
    pub fn mark_done(&self, py: Python<'_>) -> PyResult<()> {
        let arena_ref = self.arena.borrow(py);
        arena_ref.arena.mark_done(self.slot);
        Ok(())
    }
}
