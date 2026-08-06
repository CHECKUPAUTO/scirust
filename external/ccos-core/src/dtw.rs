//! # Dynamic time warping — align two trajectories, find where they diverged
//!
//! Two runs of an agent over the same task produce two belief/score trajectories. When one
//! regresses, the question is **where did they stop tracking each other** — but the runs rarely line
//! up step-for-step (one may take an extra recall, an earlier failure). **Dynamic time warping**
//! aligns two real-valued series by warping the time axis to the lowest-cost correspondence, so the
//! comparison survives that misalignment; the warping path then pinpoints the first step at which the
//! two histories separate. A stateless retriever has no per-item trajectory and no replay, so it
//! cannot compare two histories at all — this is a property only a deterministic, replayable memory
//! has.
//!
//! Pure fixed-order `f64`/`usize` DP with a deterministic tie-break (diagonal-first traceback), so
//! the distance and path are bit-reproducible — no RNG, no wall-clock. Distilled from
//! `scirust-sequential`'s `dynamic_time_warping_with_path` (oracle only; nothing linked).

/// The result of aligning two series: the warping distance, the warping path (pairs of indices into
/// `a` and `b`), and the **divergence onset** — the first aligned pair whose values differ by more
/// than the caller's threshold, or `None` if they track within it throughout.
#[derive(Debug, Clone, PartialEq)]
pub struct Alignment {
    /// Total DTW cost — how far apart the two series are overall (0 = identical).
    pub distance: f64,
    /// The warping path: `(i, j)` pairs meaning `a[i]` is aligned to `b[j]`, from `(0,0)` to the end.
    pub path: Vec<(usize, usize)>,
    /// Index into [`path`](Self::path) of the first aligned pair exceeding the divergence threshold.
    pub divergence: Option<usize>,
}

/// The DTW **distance** between `a` and `b` (absolute-difference local cost). Returns `0.0` for two
/// empty series and `+∞` when exactly one is empty (no alignment exists). Deterministic.
pub fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    align(a, b, f64::INFINITY).distance
}

/// Align `a` and `b` by dynamic time warping, reporting the distance, the warping path, and the first
/// aligned pair that differs by more than `divergence_threshold`. Deterministic: the DP fills in
/// fixed order and the traceback prefers the diagonal, then up, then left, so ties resolve the same
/// way every run.
pub fn align(a: &[f64], b: &[f64], divergence_threshold: f64) -> Alignment {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return Alignment {
            distance: if n == m { 0.0 } else { f64::INFINITY },
            path: Vec::new(),
            divergence: None,
        };
    }
    // cost[i][j] = DTW cost of aligning a[..=i] with b[..=j]. Row 0 / col 0 padded with +∞.
    let inf = f64::INFINITY;
    let mut cost = vec![vec![inf; m + 1]; n + 1];
    cost[0][0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let d = (a[i - 1] - b[j - 1]).abs();
            let best = cost[i - 1][j - 1].min(cost[i - 1][j]).min(cost[i][j - 1]);
            cost[i][j] = d + best;
        }
    }
    // Traceback from (n, m) to (1, 1), diagonal-first on ties (deterministic).
    let mut path = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let diag = cost[i - 1][j - 1];
        let up = cost[i - 1][j];
        let left = cost[i][j - 1];
        if diag <= up && diag <= left {
            i -= 1;
            j -= 1;
        } else if up <= left {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    path.reverse();
    // Divergence onset: the first aligned pair whose values differ by more than the threshold.
    let divergence = path
        .iter()
        .position(|&(i, j)| (a[i] - b[j]).abs() > divergence_threshold);
    Alignment {
        distance: cost[n][m],
        path,
        divergence,
    }
}

/// Length of the **longest common subsequence** of two discrete label sequences — a companion to
/// [`align`] for the *ordered op stream* (how much two histories still share), rather than the
/// real-valued trajectory. Deterministic `O(n·m)` DP.
pub fn lcs_len<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return 0;
    }
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }
    dp[n][m]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_series_have_zero_distance() {
        let s = [0.0, 0.5, 1.0, 1.0, 0.5];
        let al = align(&s, &s, 0.01);
        assert_eq!(al.distance, 0.0);
        assert!(al.divergence.is_none(), "identical series never diverge");
    }

    #[test]
    fn warps_a_time_shift_to_low_cost() {
        // b is a's ramp delayed by one step (a leading repeat). DTW should absorb the shift, so the
        // distance stays small — much smaller than a rigid step-for-step comparison would give.
        let a = [0.0, 1.0, 2.0, 3.0];
        let b = [0.0, 0.0, 1.0, 2.0, 3.0];
        let al = align(&a, &b, 0.5);
        assert!(
            al.distance < 1.0,
            "the time shift is warped away: {}",
            al.distance
        );
    }

    #[test]
    fn locates_the_divergence_onset() {
        // Two series that track, then split at index 3.
        let a = [0.1, 0.2, 0.3, 0.4, 0.5];
        let b = [0.1, 0.2, 0.3, 0.9, 1.0];
        let al = align(&a, &b, 0.2);
        let d = al.divergence.expect("they diverge");
        let (i, j) = al.path[d];
        assert!(
            i >= 3 && j >= 3,
            "divergence onset is at the split, not before: {:?}",
            (i, j)
        );
    }

    #[test]
    fn empty_and_mismatched_edge_cases() {
        assert_eq!(dtw_distance(&[], &[]), 0.0);
        assert_eq!(dtw_distance(&[1.0], &[]), f64::INFINITY);
        assert!(align(&[], &[1.0], 0.1).path.is_empty());
    }

    #[test]
    fn is_deterministic() {
        let a = [0.0, 0.3, 0.31, 0.9, 0.4];
        let b = [0.0, 0.3, 0.8, 0.85, 0.4];
        assert_eq!(align(&a, &b, 0.2), align(&a, &b, 0.2));
    }

    #[test]
    fn lcs_of_op_streams() {
        assert_eq!(
            lcs_len(
                &["ingest", "recall", "fail", "recall"],
                &["ingest", "fail", "recall"]
            ),
            3
        );
        assert_eq!(lcs_len::<u8>(&[], &[1, 2]), 0);
    }
}
