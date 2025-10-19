use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize)]
pub struct InferenceRequest {
    pub observations: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
pub struct InferenceResponse {
    pub prior_batch: Vec<HashMap<String, f32>>,
    pub value_batch: Vec<f32>,
}