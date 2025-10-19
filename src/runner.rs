use std::collections::HashMap;
use std::sync::{Arc};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use crossbeam::channel::{unbounded, Receiver, Sender};

use crate::MCTS;
use crate::agent::Agent;
use crate::enums::{Action, NodeId};
use tilers_core::env::Environment;

/// Small types used by the runner
struct LeafMeta {
    parent: NodeId,
    action: Action,
    env: Environment,
    path: Vec<(NodeId, Action)>,
    repeat: bool,
    obs: Vec<f32>,
}

struct InferenceResult {
    meta: LeafMeta,
    priors: HashMap<Action, f32>,
    value: f32,
}

/// MCTS runner modeled after your Python MCTSRunner.
/// Generic over concrete Agent type A.
pub struct MCTSRunner {
    pub mcts: Arc<MCTS>,

    pub num_search_threads: usize,
    pub inference_batch_size: usize,
    pub max_queue_size: usize,
}

impl MCTSRunner {
    pub fn new(
        mcts: MCTS,
        num_search_threads: usize,
        inference_batch_size: usize,
        max_queue_size: usize,
    ) -> Self {
        Self {
            mcts: Arc::new(mcts),
            num_search_threads,
            inference_batch_size,
            max_queue_size,
        }
    }

    /// Run the multi-threaded MCTS loop until `num_steps` expansions have been processed.
    /// Returns the root node id.
    pub fn run<T: Agent + 'static>(&self, env: Environment, agent: T,  num_steps: usize) -> NodeId {
        // Root must exist
        let root_id = self.mcts.get_hash(&env);
        if !self.mcts.node_exists(root_id) {
            // create root node with a cheap placeholder observation -> agent inference could be used
            // let obs = self.env.observation();
            let obs = vec![0.0];
            let (priors, value) = agent.infer(&obs);
            let priors = self.mcts.normalize_prior(priors, env.valid_actions());
            self.mcts.create_node(&env, priors, value);
        }

        // Channels
        let (leaf_tx, leaf_rx): (Sender<LeafMeta>, Receiver<LeafMeta>) = unbounded();
        let (result_tx, result_rx): (Sender<InferenceResult>, Receiver<InferenceResult>) = unbounded();

        // Shared control flags / counters
        let stop_flag = Arc::new(AtomicBool::new(false));
        let step_counter = Arc::new(AtomicUsize::new(0));

        // Clone handles for threads
        let mcts_arc = Arc::clone(&self.mcts);
        let env_clone_for_search = env.clone();
        let agent_arc = Arc::new(agent);

        // Spawn search workers
        let mut handles = Vec::new();
        for wid in 0..self.num_search_threads {
            let leaf_tx = leaf_tx.clone();
            let mcts = Arc::clone(&mcts_arc);
            let stop = Arc::clone(&stop_flag);
            let worker_env = env_clone_for_search.clone();
            let root = root_id;

            let handle = thread::spawn(move || {
                //println!("[search_worker {}] started", wid);
                while !stop.load(Ordering::Relaxed) {
                    //println!("[search_worker {}] alive", wid);
                    // each iteration use a fresh copy of the environment
                    let mut game = worker_env.clone();
                    // select leaf (returns path, parent, action, obs, repeat)
                    // protect against panics in select_leaf so thread doesn't silently die
                    let sel = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| {
                            mcts.select_leaf(root, &mut game)
                        }
                    ));
                    match sel {
                        Ok((path, parent_opt, action, obs, repeat)) => {
                            //println!("[search_worker {}] select_leaf -> parent={:?} action={}", wid, parent_opt, action);
                            if parent_opt.is_none() {
                                // nothing to do this iteration
                                // small sleep to avoid hot-looping if select_leaf keeps returning None
                                std::thread::sleep(std::time::Duration::from_millis(1));
                                continue;
                            }
                            let parent = parent_opt.unwrap();
                            // build meta and send for inference
                            let meta = LeafMeta {
                                parent,
                                action,
                                env: game,
                                path,
                                repeat,
                                obs,
                            };
                            //println!("[search_worker {}] sending meta parent={} action={}", wid, parent, action);
                            if leaf_tx.send(meta).is_err() {
                                //println!("[search_worker {}] leaf_tx send failed, coordinator closed. exiting", wid);
                                break;
                            }
                        }
                        Err(_) => {
                            //println!("[search_worker {}] panic occurred in select_leaf; continuing", wid);
                            continue;
                        }
                    }
                }
                //println!("[search_worker {}] exiting", wid);
            });
            handles.push(handle);
            // tiny stagger optional:
            let _ = wid;
        }

        // Spawn inference worker
        {
            let leaf_rx = leaf_rx.clone();
            let result_tx = result_tx.clone();
            let agent = Arc::clone(&agent_arc);
            let stop = Arc::clone(&stop_flag);
            let batch_size = self.inference_batch_size;

            let handle = thread::spawn(move || {
                //println!("[inference_worker] started");
                let mut obs_batch: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
                let mut meta_batch: Vec<LeafMeta> = Vec::with_capacity(batch_size);

                // Run while not requested to stop, but keep running to flush any pending batch.
                while !stop.load(Ordering::Relaxed) || !obs_batch.is_empty() {
                    // Try to receive an item, with timeout so we can flush partial batches periodically.
                    match leaf_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        Ok(meta) => {
                            obs_batch.push(meta.obs.clone());
                            meta_batch.push(meta);
                            //println!("[inference_worker] received meta; batch size now {}", obs_batch.len());
                            if obs_batch.len() >= batch_size {
                                //println!("[inference_worker] calling batch_infer with {}", obs_batch.len());
                                let (priors_batch, values_batch) = agent.batch_infer(&obs_batch);
                                //println!("[inference_worker] batch_infer returned {} priors {} values", priors_batch.len(), values_batch.len());
                                for (meta, priors, value) in
                                    itertools::izip!(meta_batch.drain(..), priors_batch.into_iter(), values_batch.into_iter())
                                {
                                    let res = InferenceResult { meta, priors, value };
                                    if result_tx.send(res).is_err() {
                                        //println!("[inference_worker] result_tx send failed, coordinator likely closed");
                                        break;
                                    }
                                }
                                obs_batch.clear();
                            }
                        }
                        Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                            // Flush partial batch on timeout so work isn't stuck waiting for full batch.
                            if !obs_batch.is_empty() {
                                //println!("[inference_worker] timeout flush of partial batch size {}", obs_batch.len());
                                let (priors_batch, values_batch) = agent.batch_infer(&obs_batch);
                                for (meta, priors, value) in
                                    itertools::izip!(meta_batch.drain(..), priors_batch.into_iter(), values_batch.into_iter())
                                {
                                    let res = InferenceResult { meta, priors, value };
                                    if result_tx.send(res).is_err() {
                                        //println!("[inference_worker] result_tx send failed during flush");
                                        break;
                                    }
                                }
                                obs_batch.clear();
                            }
                            // If stop requested and no pending items, break loop.
                            if stop.load(Ordering::Relaxed) && obs_batch.is_empty() {
                                //println!("[inference_worker] stop flag set and no pending batch -> exiting");
                                break;
                            }
                            // otherwise continue waiting for more work
                        }
                        Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                            //println!("[inference_worker] leaf_rx disconnected -> exiting");
                            break;
                        }
                    }
                }

                // Final flush if anything remains
                if !obs_batch.is_empty() {
                    //println!("[inference_worker] final flush remaining {}", obs_batch.len());
                    let (priors_batch, values_batch) = agent.batch_infer(&obs_batch);
                    for (meta, priors, value) in
                        itertools::izip!(meta_batch.drain(..), priors_batch.into_iter(), values_batch.into_iter())
                    {
                        let res = InferenceResult { meta, priors, value };
                        let _ = result_tx.send(res);
                    }
                }
                //println!("[inference_worker] exiting");
            });
            handles.push(handle);
        }

        // Spawn coordinator
        {
            let result_rx = result_rx.clone();
            let mcts = Arc::clone(&mcts_arc);
            let stop = Arc::clone(&stop_flag);
            let counter = Arc::clone(&step_counter);

            let handle = thread::spawn(move || {
                //println!("[coordinator] started");
                while !stop.load(Ordering::Relaxed) {
                    //println!("[coordinator] alive");
                    match result_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(inf) => {
                            //println!("[coordinator] received inference result for parent={} action={}", inf.meta.parent, inf.meta.action);

                            // debug: measure expand time and catch panics
                            let meta = inf.meta;
                            let start = std::time::Instant::now();
                            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                //println!("[coordinator] calling expand for parent={} action={}", meta.parent, meta.action);
                                let _leaf = mcts.expand(meta.parent, meta.action, meta.env, inf.priors, inf.value);
                                //println!("[coordinator] expand returned leaf={}", leaf);
                                //println!("[coordinator] calling backpropagate for parent={} action={}", meta.parent, meta.action);
                                mcts.backpropagate(&meta.path, meta.repeat);
                            }));
                            let _elapsed = start.elapsed();
                            if let Err(e) = res {
                                println!("[coordinator] panic during expand/backpropagate: {:?}", e);
                            } else {
                                //println!("[coordinator] expand+backpropagate done in {:?}", elapsed);
                            }

                            let _new = counter.fetch_add(1, Ordering::Relaxed) + 1;
                            //println!("[coordinator] processed result; total steps = {}", new);
                        }
                        Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                            if stop.load(Ordering::Relaxed) {
                                //println!("[coordinator] stop flag set and no results -> exiting");
                                break;
                            }
                        }
                        Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                            // critical: producers are gone; request shutdown of all workers
                            //println!("[coordinator] result_rx disconnected -> exiting and setting stop flag");
                            stop.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                }
                //println!("[coordinator] exiting");
            });
            handles.push(handle);
        }

        // Wait until step_counter reaches num_steps or a fatal stop is requested.
        let mut tick: usize = 0;
        while step_counter.load(Ordering::Relaxed) < num_steps && !stop_flag.load(Ordering::Relaxed) {
            if tick % 50 == 0 {
                //println!("[main] progress {}/{}", step_counter.load(Ordering::Relaxed), num_steps);
            }
            tick = tick.wrapping_add(1);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // If coordinator (or other component) signaled stop early, log it.
        if stop_flag.load(Ordering::Relaxed) && step_counter.load(Ordering::Relaxed) < num_steps {
            //println!("[main] stopping early: stop_flag set; processed {}/{} steps", step_counter.load(Ordering::Relaxed), num_steps);
        }
 
         // signal threads to stop and close channels
         stop_flag.store(true, Ordering::Relaxed);
         drop(leaf_tx);
         drop(result_tx);
 
         // join all threads
         for h in handles {
            let _ = h.join();
        }
        root_id
    }
}