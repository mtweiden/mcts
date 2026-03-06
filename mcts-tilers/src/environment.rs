use mcts_core::environment::{Environment, Obs};

// Re-export the tilers Environment under an unambiguous name
use tilers::env::Environment as TilersInner;
use tilers::qubit::Qubit;
use tilers::objective::Objective;

use crate::constants::{Action, NUM_ACTIONS};

// ---------------------------------------------------------------------------
// Observation type
// ---------------------------------------------------------------------------
/// Board is Vec<Vec<BoardCell>> — one layer per objective layer, each layer
/// has h*w cells in row-major order.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct TilersObs {
    pub placement: Vec<Qubit>,
    pub objectives: Vec<Vec<Objective>>,
    pub height: usize,
    pub width: usize,
    pub num_ancillas: usize,
    pub action_mask: Vec<bool>,
}

impl Obs for TilersObs {}

// ---------------------------------------------------------------------------
// TilersEnv
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct TilersEnv {
    pub inner: TilersInner,
    pub num_objective_layers: usize,
}

impl TilersEnv {
    pub fn new(inner: TilersInner, num_objective_layers: usize) -> Self {
        Self { inner, num_objective_layers }
    }

    pub fn build_obs(&self) -> TilersObs {
        let placement: Vec<Qubit> = self.inner.get_placement();
        let objectives: Vec<Vec<Objective>> = self.inner.get_objectives(self.num_objective_layers);

        let height = self.inner.height;
        let width = self.inner.width;
        let num_ancillas = self.inner.num_ancillas();
        let valid = self.inner.valid_actions();
        let num_actions = self.inner.num_actions();
        let mut action_mask = vec![false; NUM_ACTIONS];
        for &a in &valid { if a < num_actions { action_mask[a] = true; } }

        TilersObs {
            placement,
            objectives,
            height,
            width,
            num_ancillas,
            action_mask,
        }
    }
}

impl Environment for TilersEnv {
    type Act = Action;
    type Obs = TilersObs;

    fn step(&mut self, action: Self::Act) {
        let _ = self.inner.step(action as usize).expect("MCTS passed an invalid action");
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    fn observation(&self) -> Self::Obs {
        self.build_obs()
    }

    fn valid_actions(&self) -> Vec<Self::Act> {
        self.inner.valid_actions().into_iter().map(|a| a as u16).collect()
    }

    fn hash(&self) -> u64 {
        self.inner.hash_state()
    }

    fn render(&self) -> String {
        self.inner.render()
    }
}