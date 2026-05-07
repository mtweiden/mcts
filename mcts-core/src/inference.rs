use std::collections::HashMap;
use anyhow::Result;

use crate::environment::Environment;

// ---------------------------------------------------------------------------------------------
// Generic inference boundary trait
// ---------------------------------------------------------------------------------------------
pub trait InferenceClient<E: Environment> {
    fn infer(
        &self,
        observations: &[E::Obs],
    ) -> Result<(Vec<HashMap<E::Act, f32>>, Vec<f32>)>;
}
