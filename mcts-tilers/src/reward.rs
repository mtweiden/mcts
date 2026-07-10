//! Single source of truth for the DONE terminal score.
//!
//! Three call sites previously hard-coded their own copies (gatherer search
//! terminal, gatherer episode score, evaluator search terminal) plus a ternary
//! variant in py_mcts — the same divergence pattern that produced the
//! AutoExecute mask bug. All done-scoring now flows through `done_score`.
//!
//! Currencies (DONE_REWARD_REMAP.md):
//! - margin (band = false, the historical default): tanh(ratio / tau) where
//!   ratio = (ref_depth - depth) / ref_depth. Tie with the heuristic = 0,
//!   worse-than-heuristic completions are NEGATIVE.
//! - band (band = true): 0.5 + 0.5 * tanh(ratio / tau) ∈ (0, 1). Every
//!   completion outranks every non-completion (floor -1 < cusp < her-stalls
//!   ≤ 0 < done). Tie = 0.5. Within-band grading keeps the depth-efficiency
//!   gradient; pass a steeper tau (τ_done ≈ 0.5) to restore the margin
//!   spread the 0.5x affine compression would otherwise halve.
//!
//! The recorded eval_reward (promotion scalar) must stay on the margin
//! currency regardless of the band flag — see DONE_REWARD_REMAP.md amendment
//! 1 (banding it makes promotion strictly harder for quality-led candidates).

/// Terminal score for a COMPLETED episode.
pub fn done_score(reference_depth: f32, depth: f32, tau: f32, band: bool) -> f32 {
    let ratio = (reference_depth - depth) / (reference_depth + 1e-6);
    let t = (ratio / tau).tanh();
    if band { 0.5 + 0.5 * t } else { t }
}

/// Terminal score for a NON-completed episode at the search horizon. The
/// floor does not move with the band (resignation thresholds calibrate
/// against it).
pub const NOT_DONE_SCORE: f32 = -1.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_currency_unchanged() {
        // band=false must reproduce the historical tanh(ratio/tau) exactly.
        for (r, d, tau) in [(20.0, 15.0, 1.0), (20.0, 25.0, 1.0), (35.0, 35.0, 0.3)] {
            let ratio = (r - d) / (r + 1e-6);
            assert_eq!(done_score(r, d, tau, false), (ratio / tau).tanh());
        }
    }

    #[test]
    fn band_dominates_every_failure() {
        // Even a catastrophic completion (3x heuristic depth) stays > 0,
        // i.e., above floor/cusp/clamped-her.
        let worst = done_score(10.0, 30.0, 0.5, true);
        assert!(worst > 0.0 && worst < 0.5);
        assert!(worst > NOT_DONE_SCORE);
    }

    #[test]
    fn band_tie_is_half_and_monotone() {
        let tie = done_score(20.0, 20.0, 0.5, true);
        assert!((tie - 0.5).abs() < 1e-6);
        let win = done_score(20.0, 15.0, 0.5, true);
        let big_win = done_score(20.0, 10.0, 0.5, true);
        assert!(win > tie && big_win > win && big_win < 1.0);
    }

    #[test]
    fn steeper_tau_widens_within_band_spread() {
        // τ_done = 0.5 must (more than) restore the spread the 0.5x affine
        // compression removes relative to the old τ = 1.0 margin currency.
        let spread_old = done_score(20.0, 15.0, 1.0, false) - done_score(20.0, 19.0, 1.0, false);
        let spread_band = done_score(20.0, 15.0, 0.5, true) - done_score(20.0, 19.0, 0.5, true);
        assert!(spread_band >= 0.9 * spread_old);
    }
}
