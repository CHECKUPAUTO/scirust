//! Monte-Carlo tolerance simulation.
//!
//! The analytical chain methods ([`crate::chain`], [`crate::correlated`]) combine
//! component inertias through a **linear** model. When the assembly response is a
//! genuinely non-linear function of its components — a trigonometric linkage, a
//! gear ratio, a stacked contact — or the components follow non-normal laws, the
//! honest tool is direct simulation: draw each component from its distribution,
//! push the sample through the transfer function `Y = f(X₁, …, Xₙ)`, and read the
//! resulting distribution of `Y`.
//!
//! This module provides
//!
//! - [`Distribution`] — the per-component laws a manufacturing characteristic
//!   commonly follows (normal, uniform, symmetric/︁skew triangular), each with
//!   its exact mean and variance.
//! - [`Rng`] — a small seeded, reproducible generator (xorshift64\* + Box–Muller),
//!   so a simulation is deterministic given its seed (the crate's ethos).
//! - [`simulate`] — run `n` trials through an arbitrary transfer closure and
//!   return the [`SimResult`]: the response mean, dispersion, **inertia about
//!   target** `√(δ²+σ²)`, the out-of-spec fraction in ppm, the yield, and the
//!   `0.135 / 50 / 99.865 %` percentiles (an empirical, distribution-free
//!   capability read).
//!
//! For a linear transfer with independent components the simulated mean and
//! dispersion converge to the [`crate::chain`] combination `√(Σ αᵢ² σᵢ²)`, which
//! is exactly how the example `fuzz_crosscheck` validates it.

use serde::{Deserialize, Serialize};

/// A seeded, reproducible pseudo-random generator: xorshift64\* for the uniform
/// stream and Box–Muller for standard normals. Deterministic given its seed.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A generator seeded with `seed` (a zero seed is remapped to a fixed
    /// non-zero constant, since xorshift degenerates at state 0).
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0
            {
                0x9E37_79B9_7F4A_7C15
            }
            else
            {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A uniform draw in `[0, 1)`.
    pub fn uniform01(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniform draw in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform01()
    }

    /// A standard-normal draw `N(0, 1)` via Box–Muller.
    pub fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform(1e-12, 1.0);
        let u2 = self.uniform01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// A per-component probability law, with an exact mean and variance and a
/// reproducible sampler.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Distribution {
    /// Normal `N(mean, sd²)`.
    Normal {
        /// Mean.
        mean: f64,
        /// Standard deviation (`≥ 0`).
        sd: f64,
    },
    /// Uniform on `[lo, hi]`.
    Uniform {
        /// Lower bound.
        lo: f64,
        /// Upper bound.
        hi: f64,
    },
    /// Triangular on `[lo, hi]` with the given `mode` (peak); `lo ≤ mode ≤ hi`.
    Triangular {
        /// Lower bound.
        lo: f64,
        /// Mode (peak).
        mode: f64,
        /// Upper bound.
        hi: f64,
    },
}

impl Distribution {
    /// The exact mean of the law.
    pub fn mean(&self) -> f64 {
        match *self
        {
            Distribution::Normal { mean, .. } => mean,
            Distribution::Uniform { lo, hi } => 0.5 * (lo + hi),
            Distribution::Triangular { lo, mode, hi } => (lo + mode + hi) / 3.0,
        }
    }

    /// The exact variance of the law.
    pub fn variance(&self) -> f64 {
        match *self
        {
            Distribution::Normal { sd, .. } => sd * sd,
            Distribution::Uniform { lo, hi } => (hi - lo) * (hi - lo) / 12.0,
            Distribution::Triangular { lo, mode, hi } =>
            {
                (lo * lo + mode * mode + hi * hi - lo * mode - lo * hi - mode * hi) / 18.0
            },
        }
    }

    /// One reproducible draw from the law, advancing `rng`.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match *self
        {
            Distribution::Normal { mean, sd } => mean + sd.abs() * rng.standard_normal(),
            Distribution::Uniform { lo, hi } => rng.uniform(lo, hi),
            Distribution::Triangular { lo, mode, hi } =>
            {
                // Inverse-CDF sampling of the triangular law.
                let u = rng.uniform01();
                let span = hi - lo;
                if span <= 0.0
                {
                    return lo;
                }
                let fc = (mode - lo) / span;
                if u < fc
                {
                    lo + (u * span * (mode - lo)).sqrt()
                }
                else
                {
                    hi - ((1.0 - u) * span * (hi - mode)).sqrt()
                }
            },
        }
    }
}

/// The linear assembly response `Y = Σ αᵢ xᵢ`, the common case and a ready
/// transfer function for [`simulate`]. Extra `coeffs` or `xs` past the shorter
/// slice are ignored.
pub fn linear(coeffs: &[f64], xs: &[f64]) -> f64 {
    coeffs.iter().zip(xs).map(|(a, x)| a * x).sum()
}

/// The outcome of a Monte-Carlo tolerance simulation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimResult {
    /// Mean of the response `Y`.
    pub mean: f64,
    /// Dispersion (population standard deviation) of `Y`.
    pub sigma: f64,
    /// Inertia of `Y` about `target`, `√((mean − target)² + σ²)`.
    pub inertia: f64,
    /// Predicted non-conformity in parts per million (fraction of trials
    /// outside `[lsl, usl]`).
    pub ppm: f64,
    /// Yield: the fraction of trials **inside** the spec, in `[0, 1]`.
    pub yield_fraction: f64,
    /// Smallest simulated response.
    pub min: f64,
    /// Largest simulated response.
    pub max: f64,
    /// Empirical `0.135 %` percentile of `Y`.
    pub p_low: f64,
    /// Empirical median of `Y`.
    pub median: f64,
    /// Empirical `99.865 %` percentile of `Y`.
    pub p_high: f64,
    /// Number of trials.
    pub n: usize,
}

/// Empirical percentile `p ∈ [0, 1]` of a **sorted** slice by linear
/// interpolation between order statistics. Returns 0 for an empty slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty()
    {
        return 0.0;
    }
    if sorted.len() == 1
    {
        return sorted[0];
    }
    let rank = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let frac = rank - lo as f64;
    if lo + 1 < sorted.len()
    {
        sorted[lo] * (1.0 - frac) + sorted[lo + 1] * frac
    }
    else
    {
        sorted[lo]
    }
}

/// Run `n` Monte-Carlo trials of an assembly. Each trial draws every component
/// from its [`Distribution`], evaluates the `transfer` closure
/// `Y = f(x₁, …, xₘ)`, and the batch of responses is reduced to a [`SimResult`]:
/// mean, dispersion, inertia about `target`, non-conformity vs `[lsl, usl]`, and
/// empirical percentiles.
///
/// The simulation is deterministic in `seed`. Returns an all-zero result (with
/// `n = 0`) for an empty `components` list or `n = 0`.
pub fn simulate<F>(
    components: &[Distribution],
    transfer: F,
    target: f64,
    lsl: f64,
    usl: f64,
    n: usize,
    seed: u64,
) -> SimResult
where
    F: Fn(&[f64]) -> f64,
{
    if components.is_empty() || n == 0
    {
        return SimResult {
            mean: 0.0,
            sigma: 0.0,
            inertia: 0.0,
            ppm: 0.0,
            yield_fraction: 0.0,
            min: 0.0,
            max: 0.0,
            p_low: 0.0,
            median: 0.0,
            p_high: 0.0,
            n: 0,
        };
    }
    let mut rng = Rng::new(seed);
    let mut xs = vec![0.0; components.len()];
    let mut ys = Vec::with_capacity(n);
    let mut inside = 0usize;
    for _ in 0..n
    {
        for (slot, dist) in xs.iter_mut().zip(components)
        {
            *slot = dist.sample(&mut rng);
        }
        let y = transfer(&xs);
        if y >= lsl && y <= usl
        {
            inside += 1;
        }
        ys.push(y);
    }
    let nf = n as f64;
    let mean = ys.iter().sum::<f64>() / nf;
    // Population variance and the second moment about target (= inertia²).
    let var = ys.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / nf;
    let msd = ys.iter().map(|y| (y - target) * (y - target)).sum::<f64>() / nf;
    let mut sorted = ys;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    SimResult {
        mean,
        sigma: var.sqrt(),
        inertia: msd.sqrt(),
        ppm: (n - inside) as f64 / nf * 1e6,
        yield_fraction: inside as f64 / nf,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        p_low: percentile_sorted(&sorted, 0.001_35),
        median: percentile_sorted(&sorted, 0.5),
        p_high: percentile_sorted(&sorted, 0.998_65),
        n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::nonconformity_ppm;
    use approx::assert_relative_eq;

    #[test]
    fn distribution_moments_match_closed_forms() {
        let u = Distribution::Uniform { lo: -1.0, hi: 3.0 };
        assert_relative_eq!(u.mean(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(u.variance(), 16.0 / 12.0, epsilon = 1e-12);
        let t = Distribution::Triangular {
            lo: 0.0,
            mode: 1.0,
            hi: 4.0,
        };
        assert_relative_eq!(t.mean(), 5.0 / 3.0, epsilon = 1e-12);
        // (0+1+16 − 0 − 0 − 4)/18 = 13/18.
        assert_relative_eq!(t.variance(), 13.0 / 18.0, epsilon = 1e-12);
    }

    #[test]
    fn sampled_moments_approach_the_law() {
        let mut rng = Rng::new(42);
        let d = Distribution::Triangular {
            lo: -1.0,
            mode: 0.5,
            hi: 2.0,
        };
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|_| d.sample(&mut rng)).collect();
        let m = xs.iter().sum::<f64>() / n as f64;
        let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
        assert_relative_eq!(m, d.mean(), epsilon = 0.02);
        assert_relative_eq!(v, d.variance(), epsilon = 0.02);
    }

    #[test]
    fn linear_normal_assembly_matches_analytical() {
        // Y = X1 − X2 + 0.5·X3, all normal ⇒ Y normal with known mean/var.
        let comps = [
            Distribution::Normal {
                mean: 10.0,
                sd: 0.10,
            },
            Distribution::Normal {
                mean: 4.0,
                sd: 0.08,
            },
            Distribution::Normal {
                mean: 2.0,
                sd: 0.20,
            },
        ];
        let coeffs = [1.0, -1.0, 0.5];
        let want_mean = 10.0 - 4.0 + 0.5 * 2.0;
        let want_var = 0.10f64.powi(2) + 0.08f64.powi(2) + 0.25 * 0.20f64.powi(2);
        let res = simulate(
            &comps,
            |xs| linear(&coeffs, xs),
            want_mean,
            want_mean - 1.0,
            want_mean + 1.0,
            400_000,
            7,
        );
        assert_relative_eq!(res.mean, want_mean, epsilon = 0.01);
        assert_relative_eq!(res.sigma, want_var.sqrt(), epsilon = 0.01);
        // Centred on target ⇒ inertia ≈ σ.
        assert_relative_eq!(res.inertia, res.sigma, epsilon = 0.01);
    }

    #[test]
    fn ppm_matches_normal_tail_for_linear_normal() {
        // Single normal component, identity transfer, off-centre spec.
        let comps = [Distribution::Normal { mean: 0.0, sd: 1.0 }];
        let res = simulate(&comps, |xs| xs[0], 0.0, -2.0, 2.5, 600_000, 99);
        let want = nonconformity_ppm(0.0, 1.0, -2.0, 2.5);
        // MC ppm within a few SE of the analytical tail.
        assert!((res.ppm - want).abs() < 400.0, "MC {} vs {}", res.ppm, want);
    }

    #[test]
    fn simulation_is_deterministic_in_seed() {
        let comps = [Distribution::Uniform { lo: -1.0, hi: 1.0 }];
        let a = simulate(&comps, |xs| xs[0], 0.0, -1.0, 1.0, 10_000, 123);
        let b = simulate(&comps, |xs| xs[0], 0.0, -1.0, 1.0, 10_000, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_input_is_null_result() {
        let res = simulate(&[], |_| 0.0, 0.0, -1.0, 1.0, 100, 1);
        assert_eq!(res.n, 0);
        assert_eq!(res.inertia, 0.0);
    }
}
