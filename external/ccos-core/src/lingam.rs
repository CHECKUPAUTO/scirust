//! # Pairwise LiNGAM — which way does the causal arrow point? (offline diagnostic)
//!
//! CCOS's `Causes` edges are *asserted*; its structural edges are *resolved*. Neither is
//! *inferred*: nothing in the codebase can look at two activity series and estimate which drives
//! the other. This module adds the smallest honest slice of causal discovery: the **pairwise
//! LiNGAM** direction test (Hyvärinen & Smith, JMLR 2013). For two linearly-related series with
//! **non-Gaussian** disturbances, the likelihood ratio of the model `x → y` against `y → x` has a
//! closed form — no ICA, so none of the sign/permutation ambiguity a full LiNGAM inherits.
//!
//! ## Honest scope — a gated diagnostic, never a live signal
//!
//! The test is only valid under the LiNGAM assumptions (linear acyclic model, non-Gaussian
//! disturbances) — so [`pairwise_direction`] **validates the assumptions first and abstains**
//! (`None`) when they do not hold: too little correlation (no linear relation to orient), or
//! excess kurtosis too close to Gaussian (the statistic's sign is uninformative there). The
//! returned strength is a *relative log-likelihood*, not a calibrated confidence. Nothing here is
//! wired into `compute_node_score`, recall, or the snapshot — it is an **offline eval-harness
//! diagnostic** for exploring recorded trajectories (e.g. two nodes' [`score_trajectory`]
//! series), per the design-pass review that demanded exactly this scoping.
//!
//! Pure fixed-order `f64` arithmetic over the input slices — no RNG, no wall-clock — so the
//! verdict is bit-reproducible and `replay == live` is untouched.
//!
//! [`score_trajectory`]: crate::agent_session::AgentSession::score_trajectory

/// Which way the pairwise test says the causal arrow points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `x` drives `y` (`x → y`).
    XToY,
    /// `y` drives `x` (`y → x`).
    YToX,
}

/// The outcome of a [`pairwise_direction`] test that did not abstain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionVerdict {
    /// The inferred direction.
    pub direction: Direction,
    /// |R| — the magnitude of the pairwise likelihood-ratio statistic. A *relative* measure
    /// (bigger = more asymmetric evidence), **not** a calibrated probability.
    pub strength: f64,
    /// The sample correlation the verdict rests on.
    pub correlation: f64,
}

/// Mean of a slice (0 for empty).
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Standardize to zero mean / unit variance; `None` when the series is (near-)constant.
fn standardize(xs: &[f64]) -> Option<Vec<f64>> {
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    if var < 1e-12 {
        return None;
    }
    let sd = var.sqrt();
    Some(xs.iter().map(|x| (x - m) / sd).collect())
}

/// Excess kurtosis of a **standardized** series (`E[z⁴] − 3`; 0 for a Gaussian).
fn excess_kurtosis(z: &[f64]) -> f64 {
    mean(&z.iter().map(|v| v.powi(4)).collect::<Vec<_>>()) - 3.0
}

/// The **pairwise LiNGAM** direction test. Given two equally-long series, decide whether the
/// linear-non-Gaussian evidence favours `x → y` or `y → x` — or **abstain** (`None`) when the
/// assumptions do not hold:
///
/// - fewer than 8 samples, mismatched lengths, or a (near-)constant series — nothing to orient;
/// - `|corr(x, y)| < min_corr` — no linear relation to orient;
/// - `min(excess kurtosis)` of the standardized series `< min_kurtosis` — too close to Gaussian,
///   where the statistic's sign carries no information (the classic LiNGAM failure mode).
///
/// The statistic is Hyvärinen–Smith's nonlinear-correlation likelihood ratio for sparse
/// (supergaussian) disturbances: `R = ρ̂ · mean(x·tanh(y) − tanh(x)·y)` over the standardized
/// series; `R > 0 ⇒ x → y`, `R < 0 ⇒ y → x` (that is why the kurtosis gate requires the
/// supergaussian regime — for subgaussian data the sign flips, so we abstain rather than guess).
/// Deterministic, pure, and read-only.
pub fn pairwise_direction(
    x: &[f64],
    y: &[f64],
    min_corr: f64,
    min_kurtosis: f64,
) -> Option<DirectionVerdict> {
    if x.len() != y.len() || x.len() < 8 {
        return None;
    }
    let zx = standardize(x)?;
    let zy = standardize(y)?;
    let n = zx.len() as f64;
    let rho = zx.iter().zip(&zy).map(|(a, b)| a * b).sum::<f64>() / n;
    if rho.abs() < min_corr.max(0.0) {
        return None; // no linear relation to orient
    }
    // Assumption gate: the sign of R is only informative for supergaussian disturbances.
    let k = excess_kurtosis(&zx).min(excess_kurtosis(&zy));
    if k < min_kurtosis {
        return None; // too Gaussian — abstain rather than emit an arbitrary arrow
    }
    let r = rho
        * (zx
            .iter()
            .zip(&zy)
            .map(|(a, b)| a * b.tanh() - a.tanh() * b)
            .sum::<f64>()
            / n);
    if r == 0.0 {
        return None; // perfectly symmetric evidence
    }
    Some(DirectionVerdict {
        direction: if r > 0.0 {
            Direction::XToY
        } else {
            Direction::YToX
        },
        strength: r.abs(),
        correlation: rho,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform stream in `[-1, 1]` from **splitmix64** (seeds yield well-separated,
    /// uncorrelated streams — unlike consecutive-seed LCGs). Test-only; the module has no RNG.
    fn uniforms(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    /// Sparse (supergaussian) noise: cubing a uniform spikes the tails (excess kurtosis ≈ +0.8).
    fn sparse_noise(seed: u64, n: usize) -> Vec<f64> {
        uniforms(seed, n)
            .into_iter()
            .map(|u| u.powi(3) * 3.0)
            .collect()
    }

    /// A causal pair x → y (`y = 0.8·x + 0.6·e`), with sparse disturbances. Measured on this
    /// fixture: kurtosis (x, y) ≈ (0.79, 0.42), ρ ≈ 0.80, R ≈ +0.020 — so the gates below
    /// (min_corr 0.2, min_kurtosis 0.2) hold with margin.
    fn causal_pair(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x = sparse_noise(42, n);
        let e = sparse_noise(1337, n);
        let y: Vec<f64> = x
            .iter()
            .zip(&e)
            .map(|(xi, ei)| 0.8 * xi + 0.6 * ei)
            .collect();
        (x, y)
    }

    #[test]
    fn recovers_the_true_direction_and_its_reverse() {
        let (x, y) = causal_pair(4000);
        let v = pairwise_direction(&x, &y, 0.2, 0.2).expect("assumptions hold");
        assert_eq!(
            v.direction,
            Direction::XToY,
            "x drives y: R strength {}",
            v.strength
        );
        assert!(
            v.correlation.abs() > 0.5,
            "strong linear relation: {}",
            v.correlation
        );
        // Swap the arguments: the verdict flips — the asymmetry is real, not positional.
        let r = pairwise_direction(&y, &x, 0.2, 0.2).expect("assumptions hold");
        assert_eq!(r.direction, Direction::YToX);
    }

    #[test]
    fn abstains_when_assumptions_fail() {
        // Uncorrelated series: nothing to orient.
        let a = sparse_noise(1, 4000);
        let b = sparse_noise(2, 4000);
        assert!(
            pairwise_direction(&a, &b, 0.2, 0.2).is_none(),
            "no relation ⇒ abstain"
        );
        // Near-Gaussian data — an Irwin–Hall sum of 12 independent uniforms (excess kurtosis
        // ≈ −0.13) — fails the kurtosis gate even when perfectly correlated: the regime where
        // LiNGAM's sign is uninformative, so the test must abstain rather than guess.
        let srcs: Vec<Vec<f64>> = (0..12).map(|j| uniforms(100 + j, 2000)).collect();
        let g: Vec<f64> = (0..2000).map(|i| srcs.iter().map(|s| s[i]).sum()).collect();
        let g2: Vec<f64> = g.iter().map(|v| 0.9 * v).collect();
        assert!(
            pairwise_direction(&g, &g2, 0.2, 0.2).is_none(),
            "Gaussian-ish input ⇒ abstain rather than guess"
        );
        // Degenerate inputs.
        assert!(
            pairwise_direction(&[1.0; 20], &[2.0; 20], 0.2, 0.2).is_none(),
            "constant"
        );
        assert!(
            pairwise_direction(&[1.0, 2.0], &[1.0, 2.0], 0.2, 0.2).is_none(),
            "too short"
        );
        assert!(
            pairwise_direction(&a[..10], &b[..9], 0.2, 0.2).is_none(),
            "length mismatch"
        );
    }

    #[test]
    fn is_bit_deterministic() {
        let (x, y) = causal_pair(1000);
        let a = pairwise_direction(&x, &y, 0.2, 0.2).unwrap();
        let b = pairwise_direction(&x, &y, 0.2, 0.2).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            a.strength.to_bits(),
            b.strength.to_bits(),
            "bit-reproducible"
        );
    }
}
