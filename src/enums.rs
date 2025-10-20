use std::collections::HashMap;

pub type Action = usize;
pub type NodeId = u64;
pub type Observation = Vec<usize>;
pub type Prior = HashMap<Action, f32>;
pub type Value = f32;