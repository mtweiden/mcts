use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use anyhow::Result;

use crate::environment::Environment;
use crate::ipc_core::{SlotInit, Arena};

// ---------------------------------------------------------------------------------------------
// Generic inference boundary trait
// ---------------------------------------------------------------------------------------------
pub trait InferenceClient<E: Environment> {
    fn infer(
        &self,
        observations: &[E::Obs],
    ) -> Result<(Vec<HashMap<E::Act, f32>>, Vec<f32>)>;
}

// ---------------------------------------------------------------------------------------------
// An IPC capable inference client communicating via shared memory.
// ---------------------------------------------------------------------------------------------
#[allow(dead_code)]
pub struct IpcClient<S: SlotInit> {
    arena: Arena<S>,
    owner_id: u32,
    next_req_id: AtomicU64,
    print_timing: bool,
}

impl<S: SlotInit> IpcClient<S> {
    pub fn new(arena: Arena<S>, owner_id: u32) -> Self {
        Self {
            arena,
            owner_id,
            next_req_id: AtomicU64::new(0),
            print_timing: true,
        }
    }
}
