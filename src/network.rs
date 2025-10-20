use serde::{Serialize, Deserialize};
use crate::enums::Observation;
use crate::enums::Prior;
use crate::enums::Value;

#[derive(Serialize)]
pub struct InferenceRequest {
    pub observation_batch: Vec<Observation>,
}

#[derive(Deserialize)]
pub struct InferenceResponse {
    pub prior_batch: Vec<Prior>,
    pub value_batch: Vec<Value>,
}