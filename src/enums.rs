use std::collections::HashMap;

pub type Action = usize;
pub type NodeId = u64;
// pub type Observation = Vec<usize>;
pub type Prior = HashMap<Action, f32>;
pub type Value = f32;
// (placement, objectives_0, objectives_1, height, width, valid_actions)
pub type Observation = (Vec<usize>, Vec<usize>, Vec<usize>, usize, usize, Vec<Action>);