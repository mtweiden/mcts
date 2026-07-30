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

    /// Scaffold-relative scoring: a reverse-curriculum probe must be scored on
    /// the SUFFIX it actually controls, not on the full solution.
    ///
    /// The prefix is applied with real steps, so it sits in both `D_full` and the
    /// episode's depth. Subtracting it from both leaves the numerator unchanged
    /// and fixes the denominator. Before the fix the denominator was `D_full`, so
    /// an identical proportional suffix performance scored differently depending
    /// on how deep the scaffold was -- compressed toward zero, making deeply
    /// scaffolded episodes look better than they were.
    #[test]
    fn rc_scoring_is_scaffold_relative() {
        let d_full = 100.0f32;
        let tau = 1.0f32;

        // (a) matching the heuristic's pace scores 0 at every scaffold depth.
        for scaffold in [0.0f32, 40.0, 80.0] {
            let ref_suffix = (d_full - scaffold).max(1.0);
            let agent = (d_full - scaffold).max(0.0);
            assert!(
                done_score(ref_suffix, agent, tau).abs() < 1e-6,
                "parity should score 0 at scaffold {scaffold}"
            );
        }

        // (b) the SAME proportional suffix advantage scores the SAME at every
        //     scaffold depth. This is the property the old denominator broke.
        let mut fixed: Vec<f32> = Vec::new();
        let mut old_buggy: Vec<f32> = Vec::new();
        for scaffold in [0.0f32, 40.0, 80.0] {
            let ref_suffix = (d_full - scaffold).max(1.0);
            let agent_suffix = ref_suffix * 0.9; // 10% better than the heuristic
            fixed.push(done_score(ref_suffix, agent_suffix, tau));
            // what the old code computed: full reference, full depth
            old_buggy.push(done_score(d_full, scaffold + agent_suffix, tau));
        }
        for w in fixed.windows(2) {
            assert!(
                (w[0] - w[1]).abs() < 1e-6,
                "scaffold depth must not change the score: {fixed:?}"
            );
        }
        // and confirm the old form really did vary (else this test proves nothing)
        assert!(
            (old_buggy[0] - old_buggy[2]).abs() > 0.05,
            "old form should have been scaffold-dependent, got {old_buggy:?}"
        );
    }

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
