//! # Drift attribution — where did a trajectory break, and what moved it
//!
//! CCOS's post-mortem tools *show* that a node's causal score drifted (the `energy` / `missing`
//! views); this module says **when** the drift happened and, wired into
//! [`AgentSession::attribute_drift`](crate::agent_session::AgentSession::attribute_drift), **which
//! recorded operation** caused it. A stateless retriever has no per-item trajectory and no event
//! log, so it structurally cannot charge a state change to a cause — this is a property only a
//! deterministic, replayable memory can have.
//!
//! The estimator is the classic **CUSUM** change-point: the index that maximises the absolute
//! cumulative deviation of the series from its mean marks the single most pronounced level shift.
//! Pure fixed-order `f64` arithmetic — no RNG, no wall-clock — so it is bit-reproducible and
//! `replay == live` holds. Distilled from `scirust-seasonal`'s `cusum` (oracle only; nothing linked,
//! per the "distill, don't link" discipline).

/// A detected level shift in a numeric series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Changepoint {
    /// Index of the first sample **after** the shift — the series splits into `[..index)` (before)
    /// and `[index..)` (after).
    pub index: usize,
    /// Signed magnitude of the shift: `mean(after) − mean(before)`. Positive ⇒ the series jumped up.
    pub delta: f64,
    /// Peak `|CUSUM|` statistic — how pronounced the break is (larger = sharper).
    pub cusum: f64,
}

/// Locate the single most pronounced level shift in `series` by the **CUSUM** estimator: the index
/// maximising `|Σ (x_i − mean)|`. Returns `None` for a series too short to have an interior break
/// (fewer than 2 points) or a perfectly flat one (no deviation to explain).
///
/// Deterministic: a single left-to-right prefix sum with a strict `>` update, so the first index to
/// reach the peak wins (stable tie-break). Pure — no RNG, no allocation beyond the return.
pub fn changepoint(series: &[f64]) -> Option<Changepoint> {
    let n = series.len();
    if n < 2 {
        return None;
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let mut cusum = 0.0f64;
    let mut peak_abs = 0.0f64;
    let mut peak_at = 0usize;
    for (i, &x) in series.iter().enumerate() {
        cusum += x - mean;
        if cusum.abs() > peak_abs {
            peak_abs = cusum.abs();
            peak_at = i;
        }
    }
    if peak_abs == 0.0 {
        return None; // flat series: nothing to attribute
    }
    // The cumulative deviation peaks at the last sample *before* the level reverts toward the mean,
    // so the post-shift segment starts right after it. Clamp so both segments are non-empty.
    let index = (peak_at + 1).clamp(1, n - 1);
    let before = series[..index].iter().sum::<f64>() / index as f64;
    let after = series[index..].iter().sum::<f64>() / (n - index) as f64;
    Some(Changepoint {
        index,
        delta: after - before,
        cusum: peak_abs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locates_a_clean_step() {
        // A level shift up between index 2 and 3.
        let cp = changepoint(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(cp.index, 3, "break at the first post-shift sample");
        assert!(
            (cp.delta - 1.0).abs() < 1e-9,
            "mean jumps by +1: {}",
            cp.delta
        );
        assert!(cp.cusum > 0.0);
    }

    #[test]
    fn signs_the_delta_by_direction() {
        let up = changepoint(&[0.0, 0.0, 2.0, 2.0]).unwrap();
        assert!(up.delta > 0.0, "a rise is positive");
        let down = changepoint(&[2.0, 2.0, 0.0, 0.0]).unwrap();
        assert!(down.delta < 0.0, "a fall is negative");
    }

    #[test]
    fn flat_and_too_short_return_none() {
        assert!(changepoint(&[]).is_none());
        assert!(changepoint(&[0.5]).is_none());
        assert!(
            changepoint(&[0.3, 0.3, 0.3, 0.3]).is_none(),
            "no deviation ⇒ no break"
        );
    }

    #[test]
    fn is_deterministic() {
        let s = [0.1, 0.1, 0.15, 0.9, 0.85, 0.88, 0.9];
        assert_eq!(changepoint(&s), changepoint(&s));
    }
}
