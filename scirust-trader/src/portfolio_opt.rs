//! Portfolio construction — turning a set of assets into target weights.
//!
//! The [`crate::portfolio`] layer can *rebalance to* target weights; this module
//! *computes* them. From per-asset return series it builds the covariance and
//! correlation matrices, then allocates by one of several schemes:
//!
//! * **Equal weight** — `1/n` each; the hard-to-beat baseline.
//! * **Inverse volatility** — `w_i ∝ 1/σ_i` (naive risk parity): quieter assets
//!   get more capital so each contributes more evenly to risk.
//! * **Inverse variance** — `w_i ∝ 1/σ_i²`; a sharper version of the above.
//! * **Minimum variance** — the closed-form `w = Σ⁻¹𝟙 / (𝟙ᵀΣ⁻¹𝟙)`, optionally
//!   projected long-only (clamp negatives, renormalise).
//!
//! Every allocation is reported with the portfolio volatility `√(wᵀΣw)`, the
//! per-asset **risk contributions** (which sum to 1), and the **diversification
//! ratio** `Σwᵢσᵢ / σ_p`. Pure Rust, deterministic, no dependencies.

// Matrix maths reads most clearly with explicit `i`/`j`/`k` index loops here.
#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};

/// Sample covariance matrix (`n × n`) from `n` equal-length return series.
/// Returns an empty matrix if the input is ragged or too short.
pub fn covariance_matrix(returns: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = returns.len();
    if n == 0
    {
        return Vec::new();
    }
    let t = returns[0].len();
    if t < 2 || returns.iter().any(|r| r.len() != t)
    {
        return vec![vec![0.0; n]; n];
    }
    let means: Vec<f32> = returns
        .iter()
        .map(|r| r.iter().sum::<f32>() / t as f32)
        .collect();
    let mut cov = vec![vec![0.0f32; n]; n];
    for i in 0..n
    {
        for j in i..n
        {
            let mut acc = 0.0f32;
            for k in 0..t
            {
                acc += (returns[i][k] - means[i]) * (returns[j][k] - means[j]);
            }
            let c = acc / (t as f32 - 1.0);
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }
    cov
}

/// Per-asset volatilities (√diagonal of the covariance matrix).
pub fn volatilities(cov: &[Vec<f32>]) -> Vec<f32> {
    (0..cov.len()).map(|i| cov[i][i].max(0.0).sqrt()).collect()
}

/// Correlation matrix derived from a covariance matrix.
pub fn correlation_matrix(cov: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = cov.len();
    let vols = volatilities(cov);
    let mut corr = vec![vec![0.0f32; n]; n];
    for i in 0..n
    {
        for j in 0..n
        {
            let d = vols[i] * vols[j];
            corr[i][j] = if d > 1e-12
            {
                (cov[i][j] / d).clamp(-1.0, 1.0)
            }
            else if i == j
            {
                1.0
            }
            else
            {
                0.0
            };
        }
    }
    corr
}

/// Portfolio variance `wᵀΣw`.
pub fn portfolio_variance(weights: &[f32], cov: &[Vec<f32>]) -> f32 {
    let n = weights.len();
    let mut v = 0.0f32;
    for i in 0..n
    {
        for j in 0..n
        {
            v += weights[i] * weights[j] * cov[i][j];
        }
    }
    v.max(0.0)
}

/// Normalised risk contributions: `RC_i = w_i·(Σw)_i / (wᵀΣw)`, summing to 1.
pub fn risk_contributions(weights: &[f32], cov: &[Vec<f32>]) -> Vec<f32> {
    let n = weights.len();
    let var = portfolio_variance(weights, cov);
    if var < 1e-18
    {
        return vec![0.0; n];
    }
    (0..n)
        .map(|i| {
            let marginal: f32 = (0..n).map(|j| cov[i][j] * weights[j]).sum();
            weights[i] * marginal / var
        })
        .collect()
}

/// Diversification ratio `Σ wᵢσᵢ / σ_p` (≥ 1; higher = better diversified).
pub fn diversification_ratio(weights: &[f32], cov: &[Vec<f32>]) -> f32 {
    let vols = volatilities(cov);
    let weighted_vol: f32 = weights.iter().zip(vols.iter()).map(|(w, s)| w * s).sum();
    let port_vol = portfolio_variance(weights, cov).sqrt();
    if port_vol < 1e-12
    {
        0.0
    }
    else
    {
        weighted_vol / port_vol
    }
}

fn normalise(mut w: Vec<f32>) -> Vec<f32> {
    let s: f32 = w.iter().sum();
    if s.abs() > 1e-12
    {
        for x in &mut w
        {
            *x /= s;
        }
    }
    else if !w.is_empty()
    {
        let eq = 1.0 / w.len() as f32;
        w.iter_mut().for_each(|x| *x = eq);
    }
    w
}

/// Equal weights `1/n`.
pub fn equal_weights(n: usize) -> Vec<f32> {
    if n == 0
    {
        Vec::new()
    }
    else
    {
        vec![1.0 / n as f32; n]
    }
}

/// Inverse-volatility weights `w_i ∝ 1/σ_i`.
pub fn inverse_vol_weights(cov: &[Vec<f32>]) -> Vec<f32> {
    let vols = volatilities(cov);
    normalise(
        vols.iter()
            .map(|s| if *s > 1e-9 { 1.0 / s } else { 0.0 })
            .collect(),
    )
}

/// Inverse-variance weights `w_i ∝ 1/σ_i²`.
pub fn inverse_variance_weights(cov: &[Vec<f32>]) -> Vec<f32> {
    let n = cov.len();
    normalise(
        (0..n)
            .map(|i| {
                if cov[i][i] > 1e-12
                {
                    1.0 / cov[i][i]
                }
                else
                {
                    0.0
                }
            })
            .collect(),
    )
}

/// Invert a square matrix by Gauss–Jordan elimination with partial pivoting.
/// Returns `None` if the matrix is singular.
fn invert(src: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
    let n = src.len();
    let mut a: Vec<Vec<f32>> = src.to_vec();
    let mut inv = vec![vec![0.0f32; n]; n];
    for (i, row) in inv.iter_mut().enumerate()
    {
        row[i] = 1.0;
    }
    for col in 0..n
    {
        // Partial pivot.
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n
        {
            if a[r][col].abs() > best
            {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-12
        {
            return None;
        }
        a.swap(col, piv);
        inv.swap(col, piv);
        let d = a[col][col];
        for j in 0..n
        {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..n
        {
            if r != col
            {
                let f = a[r][col];
                if f != 0.0
                {
                    for j in 0..n
                    {
                        a[r][j] -= f * a[col][j];
                        inv[r][j] -= f * inv[col][j];
                    }
                }
            }
        }
    }
    Some(inv)
}

/// Minimum-variance weights `Σ⁻¹𝟙 / (𝟙ᵀΣ⁻¹𝟙)`. A small ridge is added to the
/// diagonal for numerical stability. If `long_only`, negative weights are
/// clamped to zero and the result renormalised; if inversion fails, falls back
/// to inverse-variance weights.
pub fn min_variance_weights(cov: &[Vec<f32>], long_only: bool) -> Vec<f32> {
    let n = cov.len();
    if n == 0
    {
        return Vec::new();
    }
    // Ridge-regularise: add a small fraction of the mean variance to the diagonal.
    let mean_var: f32 = (0..n).map(|i| cov[i][i]).sum::<f32>() / n as f32;
    let ridge = (mean_var.max(1e-12)) * 1e-6 + 1e-12;
    let mut reg = cov.to_vec();
    for (i, row) in reg.iter_mut().enumerate()
    {
        row[i] += ridge;
    }
    let inv = match invert(&reg)
    {
        Some(m) => m,
        None => return inverse_variance_weights(cov),
    };
    // w_raw = Σ⁻¹ 𝟙 (row sums of the inverse).
    let raw: Vec<f32> = (0..n).map(|i| inv[i].iter().sum()).collect();
    let mut w = normalise(raw);
    if long_only && w.iter().any(|x| *x < 0.0)
    {
        w = normalise(w.iter().map(|x| x.max(0.0)).collect());
    }
    w
}

/// The allocation method.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AllocationMethod {
    EqualWeight,
    InverseVol,
    InverseVariance,
    MinVariance,
}

impl AllocationMethod {
    pub fn parse(s: &str) -> Option<Self> {
        match s
        {
            "equal" | "equal_weight" => Some(Self::EqualWeight),
            "inverse_vol" | "inv_vol" | "risk_parity" => Some(Self::InverseVol),
            "inverse_variance" | "inv_var" => Some(Self::InverseVariance),
            "min_variance" | "min_var" => Some(Self::MinVariance),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self
        {
            Self::EqualWeight => "equal_weight",
            Self::InverseVol => "inverse_vol",
            Self::InverseVariance => "inverse_variance",
            Self::MinVariance => "min_variance",
        }
    }
}

/// A constructed portfolio: the target weights and their risk analytics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConstruction {
    pub method: String,
    pub num_assets: usize,
    pub weights: Vec<f32>,
    /// Per-asset volatilities (annualised if `periods_per_year` was applied).
    pub volatilities: Vec<f32>,
    /// Portfolio volatility `√(wᵀΣw)` (same annualisation as the inputs).
    pub portfolio_volatility: f32,
    /// Normalised per-asset risk contributions (sum to 1).
    pub risk_contributions: Vec<f32>,
    pub diversification_ratio: f32,
    /// Average off-diagonal correlation across the universe.
    pub avg_correlation: f32,
}

/// Build a portfolio from per-asset return series. `periods_per_year > 0`
/// annualises the reported volatilities (weights are scale-free, unaffected).
pub fn construct(
    returns: &[Vec<f32>],
    method: AllocationMethod,
    periods_per_year: f32,
) -> PortfolioConstruction {
    let n = returns.len();
    let cov = covariance_matrix(returns);
    let weights = match method
    {
        AllocationMethod::EqualWeight => equal_weights(n),
        AllocationMethod::InverseVol => inverse_vol_weights(&cov),
        AllocationMethod::InverseVariance => inverse_variance_weights(&cov),
        AllocationMethod::MinVariance => min_variance_weights(&cov, true),
    };
    let ann = if periods_per_year > 0.0
    {
        periods_per_year.sqrt()
    }
    else
    {
        1.0
    };
    let vols: Vec<f32> = volatilities(&cov).iter().map(|s| s * ann).collect();
    let port_vol = portfolio_variance(&weights, &cov).sqrt() * ann;
    let corr = correlation_matrix(&cov);
    let mut off_sum = 0.0f32;
    let mut off_cnt = 0usize;
    for i in 0..n
    {
        for j in (i + 1)..n
        {
            off_sum += corr[i][j];
            off_cnt += 1;
        }
    }
    let avg_correlation = if off_cnt > 0
    {
        off_sum / off_cnt as f32
    }
    else
    {
        0.0
    };
    PortfolioConstruction {
        method: method.label().to_string(),
        num_assets: n,
        weights: weights.clone(),
        volatilities: vols,
        portfolio_volatility: port_vol,
        risk_contributions: risk_contributions(&weights, &cov),
        diversification_ratio: diversification_ratio(&weights, &cov),
        avg_correlation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // Two anticorrelated assets + one noisy one.
    fn sample() -> Vec<Vec<f32>> {
        let a: Vec<f32> = (0..50).map(|i| ((i as f32) * 0.3).sin() * 0.02).collect();
        let b: Vec<f32> = a.iter().map(|x| -x).collect(); // perfectly anticorrelated
        let c: Vec<f32> = (0..50).map(|i| ((i as f32) * 0.7).cos() * 0.05).collect(); // higher vol
        vec![a, b, c]
    }

    #[test]
    fn covariance_and_correlation_shapes() {
        let cov = covariance_matrix(&sample());
        assert_eq!(cov.len(), 3);
        let corr = correlation_matrix(&cov);
        // Diagonal correlations are 1.
        for i in 0..3
        {
            assert!(approx(corr[i][i], 1.0, 1e-4));
        }
        // Assets a and b are perfectly anticorrelated.
        assert!(approx(corr[0][1], -1.0, 1e-3), "corr {}", corr[0][1]);
    }

    #[test]
    fn equal_weights_sum_to_one() {
        let w = equal_weights(4);
        assert!(approx(w.iter().sum::<f32>(), 1.0, 1e-6));
        assert!(w.iter().all(|x| approx(*x, 0.25, 1e-6)));
    }

    #[test]
    fn inverse_vol_favours_the_quiet_asset() {
        let cov = covariance_matrix(&sample());
        let w = inverse_vol_weights(&cov);
        assert!(approx(w.iter().sum::<f32>(), 1.0, 1e-5));
        // Asset c has the highest vol -> smallest inverse-vol weight.
        assert!(w[2] < w[0] && w[2] < w[1], "weights {:?}", w);
    }

    #[test]
    fn risk_contributions_sum_to_one() {
        let cov = covariance_matrix(&sample());
        let w = inverse_vol_weights(&cov);
        let rc = risk_contributions(&w, &cov);
        assert!(approx(rc.iter().sum::<f32>(), 1.0, 1e-3), "rc {:?}", rc);
    }

    #[test]
    fn min_variance_beats_equal_weight_variance() {
        let cov = covariance_matrix(&sample());
        let wmv = min_variance_weights(&cov, true);
        let weq = equal_weights(3);
        let vmv = portfolio_variance(&wmv, &cov);
        let veq = portfolio_variance(&weq, &cov);
        assert!(approx(wmv.iter().sum::<f32>(), 1.0, 1e-4));
        assert!(vmv <= veq + 1e-9, "min-var {vmv} should be <= equal {veq}");
        // Long-only: no negative weights.
        assert!(wmv.iter().all(|x| *x >= -1e-6));
    }

    #[test]
    fn matrix_inverse_roundtrips_identity() {
        let m = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let inv = invert(&m).unwrap();
        // m * inv ≈ I
        let mut prod = vec![vec![0.0f32; 2]; 2];
        for i in 0..2
        {
            for j in 0..2
            {
                prod[i][j] = (0..2).map(|k| m[i][k] * inv[k][j]).sum();
            }
        }
        assert!(approx(prod[0][0], 1.0, 1e-4) && approx(prod[1][1], 1.0, 1e-4));
        assert!(approx(prod[0][1], 0.0, 1e-4) && approx(prod[1][0], 0.0, 1e-4));
    }

    #[test]
    fn construct_full_report() {
        let rep = construct(&sample(), AllocationMethod::InverseVol, 365.0);
        assert_eq!(rep.num_assets, 3);
        assert!(approx(rep.weights.iter().sum::<f32>(), 1.0, 1e-5));
        assert!(rep.portfolio_volatility >= 0.0);
        assert!(
            rep.diversification_ratio >= 1.0 - 1e-3,
            "div {}",
            rep.diversification_ratio
        );
        assert!(approx(
            rep.risk_contributions.iter().sum::<f32>(),
            1.0,
            1e-3
        ));
    }

    #[test]
    fn method_parsing() {
        assert_eq!(
            AllocationMethod::parse("risk_parity"),
            Some(AllocationMethod::InverseVol)
        );
        assert_eq!(
            AllocationMethod::parse("min_var"),
            Some(AllocationMethod::MinVariance)
        );
        assert!(AllocationMethod::parse("nope").is_none());
    }
}
