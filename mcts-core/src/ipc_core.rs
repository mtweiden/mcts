use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{anyhow, Result};
use memmap2::{MmapMut, MmapOptions};
use libc::{clock_gettime, timespec, CLOCK_MONOTONIC};

use crate::enums::{Action, Observation, Prior, TokenId, Value};
use crate::enums::{GRID_MAX, MAX_BATCH, MAX_OBJ0, MAX_OBJ1, NUM_ACTIONS};

pub fn now_ns() -> u64 {
    unsafe {
        let mut ts: timespec = std::mem::zeroed();
        clock_gettime(CLOCK_MONOTONIC, &mut ts);
        (ts.tv_sec as u64) * 1_000_000_000u64 + (ts.tv_nsec as u64)
    }
}

// ---------------------------------------------------------------------------------------------
// IPC Shared Memory Slot
// ---------------------------------------------------------------------------------------------
/// The Slot is unused
pub const SLOT_FREE: u32 = 0;
/// The Slot has inputs which are ready for processing
pub const SLOT_READY: u32 = 1;
/// The Slot is being processed
pub const SLOT_WAITING: u32 = 2;
/// The Slot has completed processing and outputs are ready
pub const SLOT_DONE: u32 = 3;

/// Slots are the units of work exchanged between the MCTS process and the inference process.
/// Each slot contains memory space of batches of inputs and outputs. The state of a slot is
/// managed via atomic variables and ring queues.
#[repr(C)]
pub struct Slot {
    // Header information
    pub state: AtomicU32,
    pub b: u32,
    pub owner_id: u32,
    pub req_id: u64,

    // ----- inputs -----
    pub h: [u8; MAX_BATCH],
    pub w: [u8; MAX_BATCH],
    pub obj0_len: [u16; MAX_BATCH],
    pub obj1_len: [u16; MAX_BATCH],
    pub placement: [u16; MAX_BATCH * GRID_MAX],
    pub obj0: [u16; MAX_BATCH * MAX_OBJ0],
    pub obj1: [u16; MAX_BATCH * MAX_OBJ1],
    pub action_mask: [u8; MAX_BATCH * NUM_ACTIONS],

    // ----- outputs -----
    pub priors: [f32; MAX_BATCH * NUM_ACTIONS],
    pub values: [f32; MAX_BATCH],

    // ----- timing -----
    pub request_time_ns: AtomicU64,
    pub handler_start_time_ns: AtomicU64,
    pub response_time_ns: AtomicU64,
}

impl Slot {
    pub fn init_free(&self) {
        self.state.store(SLOT_FREE, Ordering::Relaxed);
    }

    /// Convert all `b` entries in this Slot into a Vec<Observation>.
    pub fn unpack_observations(&self) -> Vec<Observation> {
        let b = self.b as usize;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let height = self.h[i] as usize;
            let width = self.w[i] as usize;

            // placement
            let placement_len = (height * width).min(GRID_MAX);
            let placement_offset = i * GRID_MAX;
            let mut placement = Vec::with_capacity(placement_len);
            for j in 0..placement_len {
                placement.push(self.placement[placement_offset + j] as TokenId);
            }

            // objectives
            let obj0_len = self.obj0_len[i] as usize;
            let obj1_len = self.obj1_len[i] as usize;
            let obj0_offset = i * MAX_OBJ0;
            let obj1_offset = i * MAX_OBJ1;
            let mut objectives_0 = Vec::with_capacity(obj0_len);
            for j in 0..obj0_len {
                objectives_0.push(self.obj0[obj0_offset + j] as TokenId);
            }
            let mut objectives_1 = Vec::with_capacity(obj1_len);
            for j in 0..obj1_len {
                objectives_1.push(self.obj1[obj1_offset + j] as TokenId);
            }

            // valid actions from action_mask
            let mask_offset = i * NUM_ACTIONS;
            let mut valid_actions = Vec::new();
            for a in 0..NUM_ACTIONS {
                if self.action_mask[mask_offset + a] != 0 {
                    valid_actions.push(a as Action);
                }
            }

            out.push(Observation {
                placement,
                objectives_0,
                objectives_1,
                height,
                width,
                valid_actions,
            });
        }
        out
    }

    pub fn pack_priors_values(&mut self, priors: &Vec<Prior>, values: &Vec<Value>) -> Result<()> {
        let b = self.b as usize;
        if priors.len() != b {
            return Err(anyhow!("priors.len()={} != slot.b={}", priors.len(), b));
        }
        if values.len() != b {
            return Err(anyhow!("values.len()={} != slot.b={}", values.len(), b));
        }

        for i in 0..b {
            self.priors.fill(0.0);
            self.values[i] = 0.0;
            let offset = i * NUM_ACTIONS;
            let prior = &priors[i];
            for (a, p) in prior.iter() {
                self.priors[offset + (*a as usize)] = *p;
            }
            self.values[i] = values[i];
        }
        Ok(())
    }
}

pub struct SlotRef<'a> { pub slot: &'a Slot }

pub struct SlotMut<'a> { pub slot: &'a mut Slot }

// ---------------------------------------------------------------------------------------------
// Ring Queue for Slot Management
// ---------------------------------------------------------------------------------------------
/// Simple cross-process spinlock that lives in shared memory
#[repr(C)]
pub struct SpinLock {
    flag: AtomicU32, // 0 = unlocked, 1 = locked
}

impl SpinLock {
    pub fn init(&self) {
        self.flag.store(0, Ordering::Relaxed);
    }

    #[inline]
    pub fn lock(&self) {
        // Adaptive backoff: spin a bit, then yield, then sleep.
        let mut spins: u32 = 0;
        while self.flag.swap(1, Ordering::Acquire) == 1 {
            spins += 1;
            if spins < 1_000 {
                std::hint::spin_loop();
            } else if spins < 10_000 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(10));
            }
        }
    }

    #[inline]
    pub fn unlock(&self) {
        self.flag.store(0, Ordering::Release);
    }
}

pub const QCAP: usize = 2048;

/// A simpled fixed-size ring queue that stores slot indices for IPC.
#[repr(C)]
pub struct RingQueue {
    pub lock: SpinLock,  // for multi-producer/multi-consumer safety
    pub head: std::sync::atomic::AtomicU32,
    pub tail: std::sync::atomic::AtomicU32,
    pub slots: [u32; QCAP], // fixed max capacity
}

impl RingQueue {
    pub fn init(&self) {
        self.lock.init();
        self.head.store(0, Ordering::Relaxed);
        self.tail.store(0, Ordering::Relaxed);
        // capacity of slots buffer is fixed at creation time
    }

    /// Push one element. Returns Err if full.
    pub fn try_push(&self, x: u32) -> std::result::Result<(), ()> {
        self.lock.lock();
        let res = (|| {
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.head.load(Ordering::Relaxed);
            let next = (tail + 1) % (QCAP as u32);
            if next == head {
                return Err(());
            }
            // Safe because we hold the lock.
            unsafe {
                let p = self.slots.as_ptr() as *mut u32;
                *p.add(tail as usize) = x;
            }
            self.tail.store(next, Ordering::Relaxed);
            Ok(())
        })();
        self.lock.unlock();
        res
    }

    /// Pop one element. Returns None if empty.
    pub fn try_pop(&self) -> Option<u32> {
        self.lock.lock();
        let res = (|| {
            let head = self.head.load(Ordering::Relaxed);
            let tail = self.tail.load(Ordering::Relaxed);
            if head == tail {
                return None;
            }
            let x = unsafe { *self.slots.as_ptr().add(head as usize) };
            let next = (head + 1) % (QCAP as u32);
            self.head.store(next, Ordering::Relaxed);
            Some(x)
        })();
        self.lock.unlock();
        res
    }

    pub fn pop_blocking(&self) -> u32 {
        let mut spins = 0u32;
        loop {
            if let Some(x) = self.try_pop() {
                return x;
            }
            spins += 1;
            if spins < 10_000 {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }

    pub fn push_blocking(&self, x: u32) {
        let mut spins = 0u32;
        loop {
            if self.try_push(x).is_ok() {
                return;
            }
            spins += 1;
            if spins < 10_000 {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// IPC Shared Memory Arena
// ---------------------------------------------------------------------------------------------
/// The maximum number of GPUs supported for handling inference requests. For a DGX this is 4.
pub const MAX_HANDLERS: usize = 4;
const ARENA_MAGIC: u64 = 0x4D_43_54_53_49_50_43; // "MCTSIPC"ish
const ARENA_VERSION: u32 = 1;

#[repr(C)]
pub struct ArenaHeader {
    magic: u64,
    version: u32,
    num_slots: u32,
    num_handlers: u32,
    free_q: RingQueue,
    ready_q: [RingQueue; MAX_HANDLERS],
    // atomic round-robin counter for handlers (cross-process)
    next_handler: AtomicU32,
    init_lock: AtomicU32, // 0 unlocked, 1 locked
    init_done: AtomicU32, // 0 not initialized, 1 initialized
    // slots follow immediately after header in the shm regioni
}

pub struct Arena {
    #[allow(dead_code)]
    mmap: MmapMut,
    hdr: *mut ArenaHeader,
    slots: *mut Slot,
    num_slots: u32,
}

fn arena_file_path(name: &str) -> PathBuf {
    // file-backed mmap in /tmp, should work for Linux and MacOS
    PathBuf::from(format!("/tmp/{}.mmap", name))
}

impl Arena {
    pub fn create_or_open(name: &str, num_slots: usize, num_handlers: usize) -> Result<Self> {
        if num_slots == 0 {
            return Err(anyhow!("num_slots must be > 0"));
        }
        if num_slots > QCAP {
            return Err(anyhow!(
                "num_slots={} exceeds queue capacity QCAP={}",
                num_slots,
                QCAP
            ));
        }
        if num_handlers == 0 || num_handlers > MAX_HANDLERS {
            return Err(anyhow!("num_handlers must be in [1, {}]", MAX_HANDLERS));
        }

        let path = arena_file_path(name);

        let file: File = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        let total_size =
            std::mem::size_of::<ArenaHeader>() + num_slots * std::mem::size_of::<Slot>();

        file.set_len(total_size as u64)?;

        let mut mmap = unsafe { MmapOptions::new().len(total_size).map_mut(&file)? };

        let hdr = mmap.as_mut_ptr() as *mut ArenaHeader;
        let slots = unsafe {
            (mmap.as_mut_ptr() as *mut u8).add(std::mem::size_of::<ArenaHeader>()) as *mut Slot
        };

        // Initialize only when first created (or when magic mismatched)
        unsafe {
            let init_done = (*hdr).init_done.load(Ordering::Acquire);
            let need_init = init_done == 0 || (*hdr).magic != ARENA_MAGIC || (*hdr).version != ARENA_VERSION;

            if need_init {
                // Try to become the initializer.
                let got_lock = (*hdr)
                    .init_lock
                    .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok();

                if got_lock {
                    // header
                    (*hdr).magic = ARENA_MAGIC;
                    (*hdr).version = ARENA_VERSION;
                    (*hdr).num_slots = num_slots as u32;
                    (*hdr).num_handlers = num_handlers as u32;

                    (*hdr).free_q.init();
                    for q in &(*hdr).ready_q {
                        q.init();
                    }
                    // initialize round-robin counter
                    (*hdr).next_handler.store(0, Ordering::Relaxed);

                    // init slots and free list
                    for i in 0..(num_slots - 1) {
                        let s = &*slots.add(i);
                        s.init_free();
                        (*hdr).free_q.try_push(i as u32).unwrap();
                    }

                    // publish init completion
                    (*hdr).init_done.store(1, Ordering::Release);
                    (*hdr).init_lock.store(0, Ordering::Release);
                } else {
                    // Someone else is initializing; wait.
                    while (*hdr).init_done.load(Ordering::Acquire) == 0 {
                        std::hint::spin_loop();
                    }
                }
            }

            // sanity checks
            if (*hdr).magic != ARENA_MAGIC {
                return Err(anyhow!("arena magic mismatch"));
            }
            if (*hdr).version != ARENA_VERSION {
                return Err(anyhow!(
                    "arena version mismatch: {} != {}",
                    (*hdr).version,
                    ARENA_VERSION
                ));
            }
            if (*hdr).num_slots != num_slots as u32 {
                return Err(anyhow!(
                    "arena exists with num_slots={}, requested {}",
                    (*hdr).num_slots,
                    num_slots
                ));
            }
            if (*hdr).num_handlers != num_handlers as u32 {
                return Err(anyhow!(
                    "arena exists with num_handlers={}, requested {}",
                    (*hdr).num_handlers,
                    num_handlers
                ));
            }
        }

        Ok(Self {
            mmap,
            hdr,
            slots,
            num_slots: num_slots as u32,
        })
    }

    #[inline]
    fn header(&self) -> &ArenaHeader {
        unsafe { &*self.hdr }
    }

    #[inline]
    pub fn slot_ptr(&self, slot: u32) -> *mut Slot {
        assert!(slot < self.num_slots);
        unsafe { self.slots.add(slot as usize) }
    }

    pub fn slot_mut(&self, slot: u32) -> SlotMut<'_> {
        SlotMut {
            slot: unsafe { &mut *self.slot_ptr(slot) },
        }
    }

    pub fn slot(&self, slot: u32) -> SlotRef<'_> {
        SlotRef {
            slot: unsafe { &*self.slot_ptr(slot) },
        }
    }

    pub fn acquire_slot(&self) -> u32 {
        self.header().free_q.pop_blocking()
    }

    pub fn submit_to_handler(&self, slot: u32) {
        // producer should have already set state=READY after writing inputs
        let n = self.header().num_handlers as usize;
        // fetch-and-increment provides a simple cross-process round-robin assignment
        let idx = (self.header().next_handler.fetch_add(1, Ordering::AcqRel) as usize) % n;
        self.header().ready_q[idx].push_blocking(slot);
    }

    pub fn wait_done(&self, slot: u32) {
        let s = unsafe { &*self.slot_ptr(slot) };
        let mut spins = 0u32;
        loop {
            let st = s.state.load(Ordering::Acquire);
            if st == SLOT_DONE {
                return;
            }
            spins += 1;
            if spins < 100_000 {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }

    pub fn release_slot(&self, slot: u32) {
        let s = unsafe { &*self.slot_ptr(slot) };
        s.state.store(SLOT_FREE, Ordering::Release);
        self.header().free_q.push_blocking(slot);
    }

    pub fn num_slots(&self) -> u32 {
        self.num_slots
    }

    // -----------------------------------------------------------------------------------------
    // Helpers for inference workers / tests
    // -----------------------------------------------------------------------------------------
    pub fn pop_ready(&self, handler: usize) -> u32 {
        let n = self.header().num_handlers as usize;
        let h = handler % n;
        self.header().ready_q[h].pop_blocking()
    }

    pub fn mark_done(&self, slot: u32) {
        let s = unsafe { &*self.slot_ptr(slot) };
        s.state.store(SLOT_DONE, Ordering::Release);
    }

    pub fn mark_ready(&self, slot: u32, b: usize, owner_id: u32, req_id: u64) {
        let sm = unsafe { &mut *self.slot_ptr(slot) };
        sm.b = b as u32;
        sm.owner_id = owner_id;
        sm.req_id = req_id;
        sm.state.store(SLOT_READY, Ordering::Release);
    }

    pub fn clear_outputs(&self, slot: u32) {
        let sm = unsafe { &mut *self.slot_ptr(slot) };
        sm.priors.fill(0.0);
        sm.values.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn ringqueue_fifo() {
        // Stack-allocated queue is fine for unit test (same layout).
        let q = RingQueue {
            lock: SpinLock {
                flag: AtomicU32::new(0),
            },
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            slots: [0u32; QCAP],
        };
        q.init();

        for i in 0..100u32 {
            q.try_push(i).unwrap();
        }
        for i in 0..100u32 {
            assert_eq!(q.try_pop(), Some(i));
        }
        assert_eq!(q.try_pop(), None);
    }

    #[test]
    fn ringqueue_full_empty() {
        let q = RingQueue {
            lock: SpinLock {
                flag: AtomicU32::new(0),
            },
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            slots: [0u32; QCAP],
        };
        q.init();

        // Max usable capacity is QCAP-1 for this ring scheme.
        for i in 0..(QCAP as u32 - 1) {
            q.try_push(i).unwrap();
        }
        assert!(q.try_push(999).is_err());

        for i in 0..(QCAP as u32 - 1) {
            assert_eq!(q.try_pop(), Some(i));
        }
        assert_eq!(q.try_pop(), None);
    }

    #[test]
    fn ringqueue_concurrent_no_duplicates() {
        let q = Arc::new(RingQueue {
            lock: SpinLock {
                flag: AtomicU32::new(0),
            },
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            slots: [0u32; QCAP],
        });
        q.init();

        let producers = 4;
        let items_per = 200;
        let total = producers * items_per;

        let seen = Arc::new(
            (0..total)
                .map(|_| AtomicU32::new(0))
                .collect::<Vec<_>>(),
        );
        let popped = Arc::new(AtomicUsize::new(0));

        // producers
        let mut handles = vec![];
        for p in 0..producers {
            let q = q.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..items_per {
                    let x = (p * items_per + i) as u32;
                    q.push_blocking(x);
                }
            }));
        }

        // consumers
        let consumers = 2;
        for _ in 0..consumers {
            let q = q.clone();
            let seen = seen.clone();
            let popped = popped.clone();
            handles.push(std::thread::spawn(move || {
                while popped.load(Ordering::Relaxed) < total {
                    if let Some(x) = q.try_pop() {
                        let idx = x as usize;
                        if idx < seen.len() {
                            // mark seen; detect duplicates
                            let prev = seen[idx].fetch_add(1, Ordering::Relaxed);
                            assert_eq!(prev, 0);
                            popped.fetch_add(1, Ordering::Relaxed);
                        }
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(popped.load(Ordering::Relaxed), total);
        for s in seen.iter() {
            assert_eq!(s.load(Ordering::Relaxed), 1);
        }
    }
}
