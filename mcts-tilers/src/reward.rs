//! Single source of truth for the DONE terminal score.
//!
//! Three call sites previously hard-coded their own copies (gatherer search
//! terminal, gatherer episode score, evaluator search terminal) plus a ternary
//! variant in py_mcts — the same divergence pattern that produced the
//! AutoExecute mask bug. All done-scoring flows through `done_score`.
//!
//! Currency: margin over the heuristic — tanh(ratio / tau) with
//! ratio = (ref_depth - depth) / ref_depth. Tie with the heuristic = 0,
//! worse-than-heuristic completions are NEGATIVE, floor for non-completion
//! is -1. This is also the VALUE-TRAINING currency: the trainer's dataset
//! maps every non-done record's value target to -1 (outcome-only contract,
//! 2026-07-10), so search backups and value targets speak the same language.
//!
//! History: a "banded" variant (0.5 + 0.5*tanh) was shipped and REVERTED on
//! 2026-07-10 — it amplified off-policy value optimism (agent 23's value head
//! rated its own failing lines +0.77). See DONE_REWARD_REMAP.md before
//! reintroducing anything like it.

/// Terminal score for a COMPLETED episode.
pub fn done_score(reference_depth: f32, depth: f32, tau: f32) -> f32 {
    let ratio = (reference_depth - depth) / (reference_depth + 1e-6);
    (ratio / tau).tanh()
}

/// Terminal score for a NON-completed episode at the search horizon.
/// (Resignation thresholds calibrate against this floor.)
pub const NOT_DONE_SCORE: f32 = -1.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_currency() {
        for (r, d, tau) in [(20.0, 15.0, 1.0), (20.0, 25.0, 1.0), (35.0, 35.0, 0.3)] {
            let ratio = (r - d) / (r + 1e-6);
            assert_eq!(done_score(r, d, tau), (ratio / tau).tanh());
        }
        // tie = 0, win positive, loss negative, all above the floor
        assert_eq!(done_score(20.0, 20.0, 1.0), 0.0);
        assert!(done_score(20.0, 15.0, 1.0) > 0.0);
        let worse = done_score(20.0, 30.0, 1.0);
        assert!(worse < 0.0 && worse > NOT_DONE_SCORE);
    }
}
