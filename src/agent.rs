use crate::enums::Action;
use std::collections::HashMap;

/// Agent trait: provides single and batched inference APIs.
pub trait Agent: Send + Sync {
    /// Infer priors and value for a single observation.
    /// Returns (priors_map, value).
    fn infer(&self, obs: &[f32]) -> (HashMap<Action, f32>, f32);

    /// Batched inference: takes a slice of observations and returns
    /// a vector of priors maps and a vector of values (same length).
    fn batch_infer(
        &self,
        batch: &[Vec<f32>],
    ) -> (Vec<HashMap<Action, f32>>, Vec<f32>) {
        // Default implementation forwards to single infer for each item.
        let mut priors = Vec::with_capacity(batch.len());
        let mut values = Vec::with_capacity(batch.len());
        for obs in batch {
            let (p, v) = self.infer(obs);
            priors.push(p);
            values.push(v);
        }
        (priors, values)
    }
}

/// DummyAgent: a trivial Agent implementation for tests / fallback.
/// It returns uniform priors over a fixed action count and value 0.0.
#[derive(Debug, Clone)]
pub struct DummyAgent {
    pub action_count: usize,
}

impl DummyAgent {
    pub fn new(action_count: usize) -> Self {
        Self { action_count }
    }
}

impl Agent for DummyAgent {
    fn infer(&self, _obs: &[f32]) -> (HashMap<Action, f32>, f32) {
        let mut priors = HashMap::new();
        if self.action_count > 0 {
            let p = 1.0f32 / (self.action_count as f32);
            for a in 0..self.action_count {
                priors.insert(a, p);
            }
        }
        (priors, -1.0)
    }

    // Uses default batch_infer; you can override if desired.
}