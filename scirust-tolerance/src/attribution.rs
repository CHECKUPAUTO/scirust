//! Data-driven root-cause attribution — which *measured* component actually
//! drives the assembly variation.
//!
//! [`crate::sensitivity`] ranks contributors from their **nominal** sensitivity
//! `αᵢ` and budgeted inertia. Attribution asks the empirical, after-the-fact
//! question: given co-measured component readings `xⱼ` and the resulting
//! assembly characteristic `y` off the shop floor, how much of the *observed*
//! `Var(y)` does each component explain? It fits the linear model
//!
//! ```text
//! y ≈ β₀ + Σⱼ βⱼ xⱼ
//! ```
//!
//! by ordinary least squares and decomposes the explained variance through the
//! exact identity (OLS with intercept)
//!
//! ```text
//! Σⱼ βⱼ·Cov(xⱼ, y) = Var(ŷ) = R²·Var(y) ,
//! ```
//!
//! so each component's **contribution** `cⱼ = βⱼ·Cov(xⱼ, y)/Var(y)` is an
//! additive share of the model fit and the shares sum exactly to `R²`
//! (Pratt / product-measure relative importance). The fitted `βⱼ` are the
//! *empirical* sensitivities — compare them to the design `αᵢ` to catch a wrong
//! kinematic model — and the **unexplained** remainder `1 − R²` is the tell-tale
//! of a cause the measured components do not capture (a missing datum, a fixture
//! drift, an unmeasured feature).
//!
//! A contribution may come out **negative** — a suppressor variable that
//! improves the fit only in concert with the others; that is a genuine signal,
//! not a bug, so the shares are reported signed.

// Index-based matrix math (normal equations, Gaussian elimination) reads most
// clearly with explicit indices, as in `spatial`.
#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};

/// One component's empirical influence on the measured assembly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attribution {
    /// Component name.
    pub name: String,
    /// Fitted sensitivity `βⱼ = ∂y/∂xⱼ` (the OLS coefficient) — the *measured*
    /// counterpart of the design factor `αⱼ`.
    pub sensitivity: f64,
    /// Signed share of the assembly variance the component explains,
    /// `cⱼ = βⱼ·Cov(xⱼ, y)/Var(y)`. The shares sum to `r_squared`.
    pub contribution: f64,
}

/// Result of a variance-transmission attribution fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionReport {
    /// Fitted intercept `β₀`.
    pub intercept: f64,
    /// Coefficient of determination `R² = 1 − SS_res/SS_tot ∈ (−∞, 1]`, the
    /// fraction of `Var(y)` the measured components jointly explain.
    pub r_squared: f64,
    /// Unexplained fraction `1 − R²` — the share of assembly variation with a
    /// cause outside the measured set.
    pub unexplained: f64,
    /// Per-component attributions, in input order.
    pub components: Vec<Attribution>,
}

/// Attribute the measured assembly variation `assembly` to the co-measured
/// component series `columns` (each inner vector is one component's readings,
/// all of length `n = assembly.len()`), named by `names`.
///
/// Requires `names.len() == columns.len() = k`, every column of length `n`,
/// `n ≥ k + 2` observations, a non-degenerate `Var(y) > 0`, and a
/// well-conditioned (non-collinear) design. Returns `None` otherwise.
pub fn attribute(
    names: &[&str],
    columns: &[Vec<f64>],
    assembly: &[f64],
) -> Option<AttributionReport> {
    let k = columns.len();
    if k == 0 || names.len() != k
    {
        return None;
    }
    let n = assembly.len();
    if n < k + 2 || columns.iter().any(|c| c.len() != n)
    {
        return None;
    }
    let nf = n as f64;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / nf;
    let y_mean = mean(assembly);
    let var_y = assembly.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>() / nf;
    if var_y <= 0.0
    {
        return None;
    }

    // Normal equations for [intercept, β₁..β_k]: (XᵀX) β = Xᵀy.
    let p = k + 1;
    let col = |j: usize, i: usize| if j == 0 { 1.0 } else { columns[j - 1][i] };
    let mut ata = vec![vec![0.0_f64; p]; p];
    let mut aty = vec![0.0_f64; p];
    for a in 0..p
    {
        for b in a..p
        {
            let s: f64 = (0..n).map(|i| col(a, i) * col(b, i)).sum();
            ata[a][b] = s;
            ata[b][a] = s;
        }
        aty[a] = (0..n).map(|i| col(a, i) * assembly[i]).sum();
    }
    let beta = solve(ata, aty)?;

    // Fitted values, residual sum of squares, R².
    let fitted = |i: usize| beta[0] + (0..k).map(|j| beta[j + 1] * columns[j][i]).sum::<f64>();
    let ss_res: f64 = (0..n).map(|i| (assembly[i] - fitted(i)).powi(2)).sum();
    let ss_tot = var_y * nf;
    let r_squared = 1.0 - ss_res / ss_tot;

    // Signed contributions cⱼ = βⱼ·Cov(xⱼ,y)/Var(y); Σ cⱼ = R².
    let components = (0..k)
        .map(|j| {
            let xj_mean = mean(&columns[j]);
            let cov = (0..n)
                .map(|i| (columns[j][i] - xj_mean) * (assembly[i] - y_mean))
                .sum::<f64>()
                / nf;
            Attribution {
                name: names[j].to_string(),
                sensitivity: beta[j + 1],
                contribution: beta[j + 1] * cov / var_y,
            }
        })
        .collect();

    Some(AttributionReport {
        intercept: beta[0],
        r_squared,
        unexplained: 1.0 - r_squared,
        components,
    })
}

/// Solve `a·x = b` for a square system by Gaussian elimination with partial
/// pivoting. Returns `None` if the matrix is singular / ill-conditioned.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n
    {
        // Partial pivot.
        let mut pivot = col;
        for r in (col + 1)..n
        {
            if a[r][col].abs() > a[pivot][col].abs()
            {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-12
        {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        // Eliminate below.
        for r in (col + 1)..n
        {
            let f = a[r][col] / a[col][col];
            for c in col..n
            {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    // Back-substitute.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev()
    {
        let s: f64 = ((i + 1)..n).map(|c| a[i][c] * x[c]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // A tiny deterministic normal generator (Box–Muller on a xorshift stream).
    fn samples(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (u1, u2) = (next().max(1e-12), next());
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            })
            .collect()
    }

    #[test]
    fn recovers_known_sensitivities_and_contributions_sum_to_r2() {
        let n = 400;
        let x1 = samples(n, 1);
        let x2 = samples(n, 2);
        // y = 2·x1 − 1·x2 exactly (no noise) ⇒ R² = 1, β = (2, −1).
        let y: Vec<f64> = (0..n).map(|i| 2.0 * x1[i] - x2[i]).collect();
        let rep = attribute(&["x1", "x2"], &[x1, x2], &y).unwrap();
        assert_relative_eq!(rep.components[0].sensitivity, 2.0, epsilon = 1e-6);
        assert_relative_eq!(rep.components[1].sensitivity, -1.0, epsilon = 1e-6);
        assert!(rep.r_squared > 0.999_999);
        let sum: f64 = rep.components.iter().map(|c| c.contribution).sum();
        assert_relative_eq!(sum, rep.r_squared, epsilon = 1e-9);
    }

    #[test]
    fn noise_shows_up_as_unexplained_variance() {
        let n = 500;
        let x1 = samples(n, 3);
        let noise = samples(n, 99);
        // Half signal, half unmodelled noise.
        let y: Vec<f64> = (0..n).map(|i| x1[i] + noise[i]).collect();
        let rep = attribute(&["x1"], &[x1], &y).unwrap();
        // ~half the variance is unexplained (the noise is not a regressor).
        assert!(
            rep.unexplained > 0.3 && rep.unexplained < 0.7,
            "unexpl={}",
            rep.unexplained
        );
        assert_relative_eq!(rep.r_squared + rep.unexplained, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn single_regressor_contribution_equals_r_squared() {
        let n = 300;
        let x1 = samples(n, 7);
        let noise = samples(n, 8);
        let y: Vec<f64> = (0..n).map(|i| 1.5 * x1[i] + 0.5 * noise[i]).collect();
        let rep = attribute(&["x1"], &[x1], &y).unwrap();
        // One regressor: its contribution is the whole R².
        assert_relative_eq!(
            rep.components[0].contribution,
            rep.r_squared,
            epsilon = 1e-9
        );
    }

    #[test]
    fn rejects_ill_shaped_or_collinear_input() {
        // Too few observations for the parameters.
        assert!(attribute(&["a", "b"], &[vec![1.0, 2.0], vec![3.0, 4.0]], &[1.0, 2.0]).is_none());
        // Name/column mismatch.
        assert!(
            attribute(
                &["a"],
                &[vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]],
                &[1.0, 2.0, 3.0]
            )
            .is_none()
        );
        // Collinear columns (identical) ⇒ singular normal equations.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(
            attribute(
                &["a", "b"],
                &[x.clone(), x.clone()],
                &[2.0, 3.9, 6.1, 8.0, 9.9]
            )
            .is_none()
        );
    }
}
