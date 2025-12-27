use std::collections::HashMap;

use crate::comms::{RequestScratchPad, ResponseScratchPad};
use crate::enums::Observation;
use crate::enums::Prior;
use crate::enums::Value;

use crate::inference::InferenceClient;

// ---------------------------------------------------------------------------------------------
// A simple dummy inference client for tests / fallback.
// ---------------------------------------------------------------------------------------------
pub struct DummyAgent;

impl DummyAgent {
    pub fn new() -> Self { Self {} }
}

/// Simple dummy inference that returns uniform priors and a constant value.
impl DummyAgent {
    pub fn infer(&self, _obs: &Observation) -> (Prior, Value) {
        let mut priors = HashMap::new();
        let num_actions = _obs.valid_actions.len() as f32;
        priors = priors.iter().map(|(&a, _)| (a, 1.0f32 / num_actions)).collect();
        (priors, -1.0)
    }
}

/// An InferenceClient trait implementation for DummyAgent so that it can be used to run MCTS.
impl InferenceClient for DummyAgent {
    fn infer_into(
        &self,
        to_agent: &RequestScratchPad,
        b: usize,
        from_agent: &mut ResponseScratchPad,
    ) -> anyhow::Result<()> {
        let observations = to_agent.unpack();
        let mut priors = vec![];
        let mut values = vec![];
        for i in 0..b {
            let obs = &observations[i];
            let (p, v) = self.infer(&obs);
            priors.push(p);
            values.push(v);
        }
        from_agent.pack(&priors, &values);
        Ok(())
    }
}