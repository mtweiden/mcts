use std::collections::HashMap;

use crate::enums::{NUM_ACTIONS, GRID_MAX, MAX_OBJ0, MAX_OBJ1, PAD_U16};
use crate::enums::{Action, TokenId, Observation, Prior, StoredPrior, Value};

// ---------------------------------------------------------------------------------------------
// Inference Response Scratch Pad
// ---------------------------------------------------------------------------------------------
/// Scratch pad for storing intermediate inference results
pub struct ResponseScratchPad {
    pub priors: Vec<StoredPrior>, // length = max_batch
    pub values: Vec<Value>, // length = max_batch
}

impl ResponseScratchPad {
    pub fn new(max_batch: usize) -> Self {
        Self {
            priors: vec![[0.0; NUM_ACTIONS]; max_batch],
            values: vec![0.0; max_batch],
        }
    }

    pub fn out_mut(&mut self, b: usize) -> (&mut [StoredPrior], &mut [Value]) {
        (&mut self.priors[..b], &mut self.values[..b])
    }

    pub fn pack(&mut self, priors: &[Prior], values: &[Value]) {
        assert!(priors.len() <= self.priors.len());
        assert!(values.len() <= self.values.len());
        for (i, p) in priors.iter().enumerate() {
            let stored_prior = &mut self.priors[i];
            stored_prior.fill(0.0);
            for (&a, &prob) in p.iter() {
                stored_prior[a as usize] = prob;
            }
        }
        for (i, &v) in values.iter().enumerate() {
            self.values[i] = v;
        }
    }

    pub fn unpack(&self) -> (Vec<Prior>, Vec<Value>) {
        let mut priors = Vec::with_capacity(self.priors.len());
        let mut values = Vec::with_capacity(self.values.len());
        for i in 0..self.values.len() {
            let mut prior: HashMap<Action, f32> = HashMap::new();
            for a in 0..NUM_ACTIONS {
                let p = self.priors[i][a];
                if p > 0.0 {
                    prior.insert(a as Action, p);
                }
            }
            priors.push(prior);
            values.push(self.values[i]);
        }
        (priors, values)
    }
}


// ---------------------------------------------------------------------------------------------
// Inference Request Scratch Pad
// ---------------------------------------------------------------------------------------------
pub struct RequestScratchPad {
    pub max_batch: usize,
    pub h: Vec<u8>,
    pub w: Vec<u8>,
    pub num_ancillas: Vec<u8>,
    pub obj0_len: Vec<u16>,
    pub obj1_len: Vec<u16>,
    pub placement: Vec<TokenId>,  // [B*GRID_MAX]
    pub obj0: Vec<TokenId>,       // [B*MAX_OBJ0]
    pub obj1: Vec<TokenId>,       // [B*MAX_OBJ1]
    pub action_mask: Vec<u8>,     // [B*NUM_ACTIONS]
}

impl RequestScratchPad {
    pub fn new(max_batch: usize) -> Self {
        Self {
            max_batch,
            h: vec![0; max_batch],
            w: vec![0; max_batch],
            num_ancillas: vec![0; max_batch],
            obj0_len: vec![0; max_batch],
            obj1_len: vec![0; max_batch],
            placement: vec![PAD_U16; max_batch * GRID_MAX],
            obj0: vec![PAD_U16; max_batch * MAX_OBJ0],
            obj1: vec![PAD_U16; max_batch * MAX_OBJ1],
            action_mask: vec![0; max_batch * NUM_ACTIONS],
        }
    }

    pub fn pack(&mut self, obs: &[Observation]) {
        assert!(obs.len() <= self.max_batch);
        for (i, o) in obs.iter().enumerate() {
            self.h[i] = o.height as u8;
            self.w[i] = o.width as u8;
            self.num_ancillas[i] = o.num_ancillas as u8;

            // placement
            let p = &mut self.placement[i * GRID_MAX..(i + 1) * GRID_MAX];
            p.fill(PAD_U16);
            for (j, &id) in o.placement.iter().enumerate() {
                p[j] = id as u16;
            }

            // obj0
            let b0 = &mut self.obj0[i * MAX_OBJ0..(i + 1) * MAX_OBJ0];
            b0.fill(PAD_U16);
            let n0 = o.objectives_0.len().min(MAX_OBJ0);
            self.obj0_len[i] = n0 as u16;
            for j in 0..n0 { b0[j] = o.objectives_0[j] as u16; }

            // obj1
            let b1 = &mut self.obj1[i * MAX_OBJ1..(i + 1) * MAX_OBJ1];
            b1.fill(PAD_U16);
            let n1 = o.objectives_1.len().min(MAX_OBJ1);
            self.obj1_len[i] = n1 as u16;
            for j in 0..n1 { b1[j] = o.objectives_1[j] as u16; }

            // action mask
            let m = &mut self.action_mask[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS];
            m.fill(0);
            for &a in &o.valid_actions { m[a as usize] = 1; }
        }
    }

    pub fn unpack(&self) -> Vec<Observation> {
        let mut observations = Vec::with_capacity(self.max_batch);

        for i in 0..self.max_batch {

            let height = self.h[i] as usize;
            let width = self.w[i] as usize;
            let num_ancillas = self.num_ancillas[i] as usize;

            let placement_start = i * GRID_MAX;
            let placement_end = (i + 1) * GRID_MAX;
            let placement = self.placement[placement_start..placement_end]
                .iter()
                .cloned()
                .filter(|&id| id != PAD_U16)
                .map(|id| id as TokenId)
                .collect::<Vec<_>>();

            let obj0_len = self.obj0_len[i] as usize;
            let obj0_start = i * MAX_OBJ0;
            let obj0_end = obj0_start + obj0_len;
            let objectives_0 = self.obj0[obj0_start..obj0_end]
                .iter()
                .cloned()
                .filter(|&id| id != PAD_U16)
                .map(|id| id as TokenId)
                .collect::<Vec<_>>();

            let obj1_len = self.obj1_len[i] as usize;
            let obj1_start = i * MAX_OBJ1;
            let obj1_end = obj1_start + obj1_len;
            let objectives_1 = self.obj1[obj1_start..obj1_end]
                .iter()
                .cloned()
                .filter(|&id| id != PAD_U16)
                .map(|id| id as TokenId)
                .collect::<Vec<_>>();

            let action_mask_start = i * NUM_ACTIONS;
            let action_mask_end = (i + 1) * NUM_ACTIONS;
            let valid_actions = self.action_mask[action_mask_start..action_mask_end]
                .iter()
                .enumerate()
                .filter_map(|(a, &m)| if m != 0 { Some(a as Action) } else { None })
                .collect::<Vec<_>>();

            let obs = Observation {
                height,
                width,
                num_ancillas,
                placement,
                objectives_0,
                objectives_1,
                valid_actions,
            };
            observations.push(obs);
        }
        observations
    }
}