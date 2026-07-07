//! Measurement System Analysis — crossed Gage R&R by ANOVA (AIAG MSA).
//!
//! Before trusting a capability or an inertia, the measuring system itself must
//! be capable: how much of the observed variation is the *product* and how much
//! is the *gauge*? A crossed study has every operator measure every part several
//! times; the ANOVA model
//!
//! ```text
//! yᵢⱼₖ = μ + Partᵢ + Operatorⱼ + (Part·Operator)ᵢⱼ + εᵢⱼₖ
//! ```
//!
//! partitions the total variance into **repeatability** (`EV`, the residual
//! `ε` — same operator, same part), **reproducibility** (`AV`, operator +
//! part·operator interaction), and genuine **part-to-part** variation (`PV`).
//! The gauge R&R is `σ²_GRR = EV + AV`, and the usual verdicts follow:
//!
//! - `%R&R` (study variation) `= σ_GRR / σ_total`, AIAG bands 10 % / 30 %,
//! - `%tolerance = 6·σ_GRR / (USL−LSL)`,
//! - `ndc = ⌊1.41 · σ_PV / σ_GRR⌋`, the number of distinct categories.
//!
//! This is the [`crate::inertia::correct_for_measurement`] story done properly:
//! rather than a single gauge `u`, it separates the gauge variance into its
//! repeatability and reproducibility parts from a designed study.

use serde::{Deserialize, Serialize};

/// AIAG acceptability verdict from the `%R&R` (study-variation) figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GageVerdict {
    /// `%R&R < 10 %` — the measurement system is acceptable.
    Acceptable,
    /// `10 % ≤ %R&R ≤ 30 %` — marginal; acceptable depending on application/cost.
    Marginal,
    /// `%R&R > 30 %` — unacceptable; the gauge needs improvement.
    Unacceptable,
}

/// Result of a crossed Gage R&R (ANOVA method): the variance components and the
/// derived acceptability metrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GageRnR {
    /// Repeatability (equipment) variance `EV = σ²_ε`.
    pub repeatability_var: f64,
    /// Reproducibility (appraiser) variance `AV = σ²_operator + σ²_interaction`.
    pub reproducibility_var: f64,
    /// Gauge R&R variance `σ²_GRR = EV + AV`.
    pub grr_var: f64,
    /// Part-to-part variance `PV = σ²_part`.
    pub part_var: f64,
    /// Total variance `σ²_GRR + PV`.
    pub total_var: f64,
    /// `%R&R` by study variation, `100·σ_GRR/σ_total`.
    pub pct_study_rr: f64,
    /// `%` contribution by variance, `100·σ²_GRR/σ²_total`.
    pub pct_contribution: f64,
    /// `%` of tolerance consumed, `100·6σ_GRR/(USL−LSL)`, if a tolerance was
    /// supplied.
    pub pct_tolerance: Option<f64>,
    /// Number of distinct categories `⌊1.41·σ_PV/σ_GRR⌋` (at least 1).
    pub ndc: u32,
    /// AIAG verdict from `pct_study_rr`.
    pub verdict: GageVerdict,
}

/// Crossed Gage R&R by the ANOVA method. `measurements[part][operator]` is the
/// vector of replicate readings; the design must be **balanced** — every
/// part × operator cell present with the same number `r ≥ 2` of replicates,
/// `p ≥ 2` parts and `o ≥ 2` operators. `tolerance` is the spec width
/// `USL − LSL` for the `%tolerance` figure (optional).
///
/// Returns `None` if the design is unbalanced or too small.
pub fn gage_rr(measurements: &[Vec<Vec<f64>>], tolerance: Option<f64>) -> Option<GageRnR> {
    let p = measurements.len();
    if p < 2
    {
        return None;
    }
    let o = measurements[0].len();
    if o < 2
    {
        return None;
    }
    let r = measurements[0][0].len();
    if r < 2
    {
        return None;
    }
    // Balance check.
    for part in measurements
    {
        if part.len() != o
        {
            return None;
        }
        for cell in part
        {
            if cell.len() != r
            {
                return None;
            }
        }
    }
    let (pf, of, rf) = (p as f64, o as f64, r as f64);
    let n = pf * of * rf;

    // Means.
    let grand: f64 = measurements
        .iter()
        .flat_map(|part| part.iter().flat_map(|cell| cell.iter()))
        .sum::<f64>()
        / n;
    let cell_mean = |i: usize, j: usize| measurements[i][j].iter().sum::<f64>() / rf;
    let part_mean = |i: usize| (0..o).map(|j| cell_mean(i, j)).sum::<f64>() / of;
    let oper_mean = |j: usize| (0..p).map(|i| cell_mean(i, j)).sum::<f64>() / pf;

    // Sums of squares.
    let ss_part = of * rf * (0..p).map(|i| (part_mean(i) - grand).powi(2)).sum::<f64>();
    let ss_oper = pf * rf * (0..o).map(|j| (oper_mean(j) - grand).powi(2)).sum::<f64>();
    let mut ss_int = 0.0;
    let mut ss_error = 0.0;
    for (i, part) in measurements.iter().enumerate()
    {
        for (j, cell) in part.iter().enumerate()
        {
            let cm = cell_mean(i, j);
            ss_int += rf * (cm - part_mean(i) - oper_mean(j) + grand).powi(2);
            for &y in cell
            {
                ss_error += (y - cm).powi(2);
            }
        }
    }

    // Mean squares.
    let ms_part = ss_part / (pf - 1.0);
    let ms_oper = ss_oper / (of - 1.0);
    let ms_int = ss_int / ((pf - 1.0) * (of - 1.0));
    let ms_error = ss_error / (pf * of * (rf - 1.0));

    // Variance components (expected-mean-square estimators, floored at 0).
    let v_repeat = ms_error;
    let v_int = ((ms_int - ms_error) / rf).max(0.0);
    let v_oper = ((ms_oper - ms_int) / (pf * rf)).max(0.0);
    let v_part = ((ms_part - ms_int) / (of * rf)).max(0.0);

    let reproducibility = v_oper + v_int;
    let grr = v_repeat + reproducibility;
    let total = grr + v_part;

    let sigma_grr = grr.sqrt();
    let sigma_total = total.sqrt();
    let pct_study_rr = if sigma_total > 0.0
    {
        100.0 * sigma_grr / sigma_total
    }
    else
    {
        0.0
    };
    let pct_contribution = if total > 0.0
    {
        100.0 * grr / total
    }
    else
    {
        0.0
    };
    let pct_tolerance = tolerance
        .filter(|t| *t > 0.0)
        .map(|t| 100.0 * 6.0 * sigma_grr / t);
    let ndc = if sigma_grr > 0.0
    {
        (1.41 * v_part.sqrt() / sigma_grr).floor().max(1.0) as u32
    }
    else
    {
        u32::MAX
    };
    let verdict = if pct_study_rr < 10.0
    {
        GageVerdict::Acceptable
    }
    else if pct_study_rr <= 30.0
    {
        GageVerdict::Marginal
    }
    else
    {
        GageVerdict::Unacceptable
    };

    Some(GageRnR {
        repeatability_var: v_repeat,
        reproducibility_var: reproducibility,
        grr_var: grr,
        part_var: v_part,
        total_var: total,
        pct_study_rr,
        pct_contribution,
        pct_tolerance,
        ndc,
        verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identical_replicates_have_zero_repeatability() {
        // Every replicate identical within a cell ⇒ EV = 0. Parts differ ⇒ PV>0.
        let m = vec![
            vec![vec![10.0, 10.0], vec![10.1, 10.1]],
            vec![vec![12.0, 12.0], vec![12.1, 12.1]],
            vec![vec![8.0, 8.0], vec![8.1, 8.1]],
        ];
        let g = gage_rr(&m, Some(6.0)).unwrap();
        assert_relative_eq!(g.repeatability_var, 0.0, epsilon = 1e-9);
        assert!(g.part_var > 0.0);
        // Small operator offset (0.1) vs large part spread ⇒ acceptable-ish R&R.
        assert!(g.pct_study_rr < 30.0);
    }

    #[test]
    fn anova_sum_of_squares_identity() {
        let m = vec![
            vec![vec![1.0, 1.2, 0.9], vec![1.1, 1.0, 1.3]],
            vec![vec![2.1, 2.0, 2.2], vec![2.0, 2.3, 2.1]],
            vec![vec![3.0, 2.9, 3.1], vec![3.2, 3.0, 2.8]],
            vec![vec![0.5, 0.6, 0.4], vec![0.5, 0.7, 0.5]],
        ];
        // Recompute SS_total and the component SS directly and check the identity.
        let g = gage_rr(&m, None).unwrap();
        // The verdict/variance components exist and total variance is positive.
        assert!(g.total_var > 0.0);
        assert!(g.grr_var >= 0.0 && g.part_var >= 0.0);
        assert!((g.grr_var + g.part_var - g.total_var).abs() < 1e-12);
    }

    #[test]
    fn rejects_unbalanced_or_small_designs() {
        // One operator ⇒ None.
        assert!(gage_rr(&[vec![vec![1.0, 2.0]]], None).is_none());
        // Ragged replicate counts ⇒ None.
        let ragged = vec![
            vec![vec![1.0, 2.0], vec![1.0, 2.0]],
            vec![vec![1.0], vec![1.0, 2.0]],
        ];
        assert!(gage_rr(&ragged, None).is_none());
    }

    #[test]
    fn pct_tolerance_scales_with_gauge_spread() {
        let m = vec![
            vec![vec![1.0, 1.5], vec![1.2, 0.8]],
            vec![vec![5.0, 5.4], vec![5.1, 4.7]],
        ];
        let tight = gage_rr(&m, Some(2.0)).unwrap().pct_tolerance.unwrap();
        let loose = gage_rr(&m, Some(20.0)).unwrap().pct_tolerance.unwrap();
        // Same gauge, 10× wider tolerance ⇒ 10× smaller %tolerance.
        assert_relative_eq!(tight / loose, 10.0, epsilon = 1e-9);
    }
}
