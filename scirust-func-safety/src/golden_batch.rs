//! Golden-batch process comparison (GMP / FDA 21 CFR Part 11).
//!
//! In regulated manufacturing — pharma, biotech, specialty chemicals — a
//! *golden batch* is a reference run whose process trajectories (temperature,
//! pH, dissolved oxygen, agitation …) define "ideal". Each new batch is judged
//! against it. Two batches rarely line up in time (a lag phase runs long, a feed
//! starts late), so a pointwise comparison spuriously fails; we align with
//! **dynamic time warping** first, then check each aligned sample against a
//! per-variable tolerance band.
//!
//! The verdict is recorded into the crate's hash-chained [`crate::AuditLog`],
//! giving the tamper-evident electronic record Part 11 requires.

use serde::{Deserialize, Serialize};

/// Result of a dynamic-time-warping alignment.
#[derive(Debug, Clone)]
pub struct DtwResult {
    /// Total warping cost along the optimal path.
    pub distance: f64,
    /// Warping path as `(index_in_a, index_in_b)` pairs, start to end.
    pub path: Vec<(usize, usize)>,
}

/// Dynamic time warping between two multivariate sequences under the Euclidean
/// per-sample distance. `a` and `b` are sequences of equal-width samples.
pub fn dtw(a: &[Vec<f64>], b: &[Vec<f64>]) -> DtwResult {
    let n = a.len();
    let m = b.len();
    assert!(n > 0 && m > 0, "dtw needs non-empty sequences");
    let d = |i: usize, j: usize| -> f64 {
        a[i].iter()
            .zip(&b[j])
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };
    // Cost matrix with a guard border at +inf.
    let inf = f64::INFINITY;
    let mut cost = vec![vec![inf; m + 1]; n + 1];
    cost[0][0] = 0.0;
    for i in 1..=n
    {
        for j in 1..=m
        {
            let c = d(i - 1, j - 1);
            let best = cost[i - 1][j].min(cost[i][j - 1]).min(cost[i - 1][j - 1]);
            cost[i][j] = c + best;
        }
    }
    // Backtrack the optimal path.
    let mut path = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0
    {
        path.push((i - 1, j - 1));
        let diag = cost[i - 1][j - 1];
        let up = cost[i - 1][j];
        let left = cost[i][j - 1];
        if diag <= up && diag <= left
        {
            i -= 1;
            j -= 1;
        }
        else if up <= left
        {
            i -= 1;
        }
        else
        {
            j -= 1;
        }
    }
    path.reverse();
    DtwResult {
        distance: cost[n][m],
        path,
    }
}

/// A golden reference batch and the per-variable acceptance tolerance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenBatch {
    /// Reference trajectory: one sample (variable vector) per time step.
    pub reference: Vec<Vec<f64>>,
    /// Absolute acceptance tolerance per variable (same width as a sample).
    pub tolerance: Vec<f64>,
}

/// Conformance report for a candidate batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchReport {
    /// Whether every aligned sample stayed within tolerance.
    pub conforming: bool,
    /// DTW alignment cost to the golden batch.
    pub dtw_distance: f64,
    /// Largest deviation seen, expressed as a multiple of that variable's
    /// tolerance (`> 1.0` means out of band).
    pub worst_ratio: f64,
    /// Variable index responsible for `worst_ratio`.
    pub worst_variable: usize,
    /// Candidate-batch step at which it occurred.
    pub worst_step: usize,
}

impl GoldenBatch {
    /// New golden batch from a reference trajectory and per-variable tolerance.
    pub fn new(reference: Vec<Vec<f64>>, tolerance: Vec<f64>) -> Self {
        assert!(!reference.is_empty(), "golden batch is empty");
        assert_eq!(
            reference[0].len(),
            tolerance.len(),
            "tolerance width must match a sample"
        );
        Self {
            reference,
            tolerance,
        }
    }

    /// Compare a candidate `batch` against the golden batch: align by DTW, then
    /// measure the worst per-variable deviation along the warping path.
    pub fn compare(&self, batch: &[Vec<f64>]) -> BatchReport {
        let al = dtw(batch, &self.reference);
        let mut worst_ratio = 0.0;
        let mut worst_variable = 0;
        let mut worst_step = 0;
        for &(bi, ri) in &al.path
        {
            for (v, tol) in self.tolerance.iter().enumerate()
            {
                let dev = (batch[bi][v] - self.reference[ri][v]).abs();
                let ratio = if *tol > 0.0 { dev / tol } else { dev };
                if ratio > worst_ratio
                {
                    worst_ratio = ratio;
                    worst_variable = v;
                    worst_step = bi;
                }
            }
        }
        BatchReport {
            conforming: worst_ratio <= 1.0,
            dtw_distance: al.distance,
            worst_ratio,
            worst_variable,
            worst_step,
        }
    }

    /// Record a [`BatchReport`] into a Part 11 hash-chained [`crate::AuditLog`].
    pub fn record_audit(
        &self,
        log: &mut crate::AuditLog,
        batch_id: &str,
        report: &BatchReport,
        timestamp: f64,
    ) {
        let decision = if report.conforming
        {
            "RELEASE"
        }
        else
        {
            "REJECT"
        };
        let desc = format!(
            "golden-batch comparison: worst var {} ratio {:.3} at step {} (DTW {:.3})",
            report.worst_variable, report.worst_ratio, report.worst_step, report.dtw_distance
        );
        log.add(
            "golden_batch",
            &desc,
            batch_id,
            decision,
            (1.0 - report.worst_ratio).clamp(0.0, 1.0) as f32,
            timestamp,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AuditLog;

    fn ramp_then_hold(n: usize) -> Vec<Vec<f64>> {
        // Two variables: a temperature ramp to 37 then hold, and a slow pH rise.
        (0..n)
            .map(|k| {
                let t = k as f64 / n as f64;
                let temp = if t < 0.5 { 20.0 + 34.0 * t } else { 37.0 };
                let ph = 6.8 + 0.4 * t;
                vec![temp, ph]
            })
            .collect()
    }

    #[test]
    fn identical_batch_is_conforming_with_zero_distance() {
        let golden = GoldenBatch::new(ramp_then_hold(60), vec![0.5, 0.05]);
        let report = golden.compare(&golden.reference);
        assert!(report.conforming);
        assert!(report.dtw_distance < 1e-9, "dist {}", report.dtw_distance);
        assert!(report.worst_ratio < 1e-9);
    }

    #[test]
    fn dtw_absorbs_a_long_lag_phase_a_pointwise_check_would_fail() {
        // The real GMP case: a batch whose lag phase runs long. Golden ramps
        // 20→37 °C over steps [0,30) then holds; the candidate holds 20 °C for
        // 10 extra steps, then runs the *same* ramp and reaches the *same*
        // plateau. Both share endpoints, so DTW aligns the delayed ramp with no
        // residual — but a pointwise comparison fails while the candidate lags.
        let n = 100;
        let golden_traj: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                let temp = if k < 30
                {
                    20.0 + 17.0 * (k as f64 / 30.0)
                }
                else
                {
                    37.0
                };
                vec![temp]
            })
            .collect();
        let candidate: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                let temp = if k < 10
                {
                    20.0
                }
                else if k < 40
                {
                    20.0 + 17.0 * ((k - 10) as f64 / 30.0)
                }
                else
                {
                    37.0
                };
                vec![temp]
            })
            .collect();
        let golden = GoldenBatch::new(golden_traj.clone(), vec![1.0]);
        let report = golden.compare(&candidate);
        assert!(report.conforming, "worst ratio {}", report.worst_ratio);

        // Sanity: a naive pointwise max-abs at the same indices exceeds tolerance.
        let pointwise_max = golden_traj
            .iter()
            .zip(&candidate)
            .map(|(g, s)| (g[0] - s[0]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            pointwise_max > 1.0,
            "pointwise should violate tol: {pointwise_max}"
        );
    }

    #[test]
    fn an_out_of_band_excursion_is_rejected_and_attributed() {
        let mut traj = ramp_then_hold(60);
        let golden = GoldenBatch::new(traj.clone(), vec![0.5, 0.05]);
        // Inject a pH spike (variable 1) at step 40, well beyond 0.05.
        traj[40][1] += 0.5;
        let report = golden.compare(&traj);
        assert!(
            !report.conforming,
            "should reject, ratio {}",
            report.worst_ratio
        );
        assert_eq!(report.worst_variable, 1);
        assert_eq!(report.worst_step, 40);
        assert!(report.worst_ratio > 1.0);
    }

    #[test]
    fn verdict_is_written_to_a_tamper_evident_audit_chain() {
        let golden = GoldenBatch::new(ramp_then_hold(40), vec![0.5, 0.05]);
        let report = golden.compare(&golden.reference);
        let mut log = AuditLog::new(16);
        golden.record_audit(&mut log, "BATCH-2026-001", &report, 1718900000.0);
        assert_eq!(log.len(), 1);
        assert!(log.verify_chain(), "audit chain must verify");
        assert_eq!(log.entries[0].decision, "RELEASE");
    }
}
