//! # `scirust-special` — special functions for scientific & industrial computing
//!
//! The numeric bedrock under probability distributions, quadrature rules, and
//! reliability / tolerancing / metrology formulas: the gamma family, the error
//! function, the beta family, and their regularized incomplete forms.
//!
//! ## Why this crate exists
//!
//! Before it, `scirust-tolerance` and `scirust-spc` each re-implemented `erf`,
//! `ln_gamma`, and the χ² tail — duplicated, epsilon-laden code that is a
//! correctness- and audit-liability for a determinism-first platform. This
//! crate is the single, oracle-tested home for those primitives so every
//! consumer shares one validated implementation.
//!
//! ## Guarantees
//!
//! - **Pure Rust, zero dependencies, `#![forbid(unsafe_code)]`.**
//! - **Deterministic**: no global state, no RNG, no platform-dependent paths —
//!   the same inputs yield bit-identical outputs everywhere.
//! - **Validated**: every function is tested against published reference values
//!   (√π, Euler–Mascheroni, tabulated erf/χ² points) to ≤ 1e-9 relative error
//!   on its accurate domain.
//!
//! ## Example
//!
//! ```
//! use scirust_special::{erf, gamma, regularized_gamma_p};
//!
//! // Γ(1/2) = √π
//! assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-12);
//! // erf(1) ≈ 0.8427007929
//! assert!((erf(1.0) - 0.842_700_792_949_715).abs() < 1e-12);
//! // The χ²(k=2) CDF at x=2 is P(1, 1) = 1 − e⁻¹.
//! let cdf = regularized_gamma_p(1.0, 1.0);
//! assert!((cdf - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::f64::consts::PI;

/// Euler–Mascheroni constant γ.
pub const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

// Maximum iterations for the series / continued-fraction expansions. Reaching
// this bound means the argument is outside the well-converging domain; the
// functions return their best estimate rather than looping unboundedly.
const MAX_ITERS: usize = 300;
// Relative convergence tolerance for the iterative expansions.
const EPS: f64 = 1e-15;
// Smallest positive value used to avoid division by zero in the modified
// Lentz continued-fraction algorithm.
const TINY: f64 = 1e-300;

// ============================================================ //
//  Gamma family                                                //
// ============================================================ //

// Lanczos approximation coefficients (g = 7, n = 9) — the classic Godfrey set,
// accurate to ~15 significant digits for `ln_gamma` across the positive axis.
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural logarithm of the absolute value of the gamma function, `ln|Γ(x)|`.
///
/// Uses the Lanczos approximation, with Euler's reflection formula for the
/// left half-plane so negative non-integer arguments are handled. Poles at the
/// non-positive integers return `f64::INFINITY`.
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan()
    {
        return f64::NAN;
    }
    // Poles at 0, -1, -2, …
    if x <= 0.0 && x == x.floor()
    {
        return f64::INFINITY;
    }
    if x < 0.5
    {
        // Reflection: Γ(x)Γ(1−x) = π / sin(πx)  ⇒
        // ln|Γ(x)| = ln(π/|sin(πx)|) − ln|Γ(1−x)|.
        let sin_pix = (PI * x).sin().abs();
        return (PI / sin_pix).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = LANCZOS_COEFFS[0];
    let t = x + LANCZOS_G + 0.5;
    for (i, &c) in LANCZOS_COEFFS.iter().enumerate().skip(1)
    {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// The gamma function `Γ(x)`.
///
/// Real-valued for all non-pole arguments (negative non-integers included, via
/// reflection). Returns `±∞` at the poles (non-positive integers) with the sign
/// of the one-sided limit, and `NaN` for `NaN` input.
pub fn gamma(x: f64) -> f64 {
    if x.is_nan()
    {
        return f64::NAN;
    }
    if x <= 0.0 && x == x.floor()
    {
        // Pole. Sign alternates but the magnitude is infinite; return +∞.
        return f64::INFINITY;
    }
    if x < 0.5
    {
        // Reflection formula keeps the sign correct for negative arguments.
        PI / ((PI * x).sin() * gamma(1.0 - x))
    }
    else
    {
        ln_gamma(x).exp()
    }
}

/// The digamma function ψ(x) = d/dx ln Γ(x).
///
/// Uses the recurrence ψ(x+1) = ψ(x) + 1/x to push the argument into the
/// asymptotic regime, then a Bernoulli asymptotic series. Reflection handles
/// negative arguments; poles (non-positive integers) return `NaN`.
pub fn digamma(mut x: f64) -> f64 {
    if x.is_nan()
    {
        return f64::NAN;
    }
    if x <= 0.0 && x == x.floor()
    {
        return f64::NAN;
    }
    let mut result = 0.0;
    // Reflection for x < 0: ψ(1−x) − ψ(x) = π·cot(πx).
    if x < 0.0
    {
        result -= PI / (PI * x).tan();
        x = 1.0 - x;
    }
    // Recurrence up to x >= 12 so the truncated Bernoulli asymptotic series
    // below is accurate to ~1e-13 (its error scales like B₁₀/(10·x¹⁰)).
    while x < 12.0
    {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic: ψ(x) ≈ ln x − 1/(2x) − Σ B_{2n}/(2n x^{2n}).
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += x.ln()
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0)));
    result
}

/// The beta function `B(a, b) = Γ(a)Γ(b)/Γ(a+b)`.
pub fn beta(a: f64, b: f64) -> f64 {
    ln_beta(a, b).exp()
}

/// `ln B(a, b)` — numerically stable for large arguments.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

// ============================================================ //
//  Error function                                              //
// ============================================================ //

/// The error function `erf(x) = (2/√π) ∫₀ˣ e^{−t²} dt`.
///
/// Built on the regularized lower incomplete gamma `P(1/2, x²)`, tying erf to
/// the gamma family so both share one validated implementation. Odd in `x`.
pub fn erf(x: f64) -> f64 {
    if x == 0.0
    {
        return 0.0;
    }
    let p = regularized_gamma_p(0.5, x * x);
    if x >= 0.0 { p } else { -p }
}

/// The complementary error function `erfc(x) = 1 − erf(x)`, accurate in the
/// far tail (where `1 − erf(x)` would lose all significance) via the upper
/// incomplete gamma `Q(1/2, x²)`.
pub fn erfc(x: f64) -> f64 {
    if x == 0.0
    {
        return 1.0;
    }
    let q = regularized_gamma_q(0.5, x * x);
    if x >= 0.0 { q } else { 2.0 - q }
}

/// The inverse error function `erfinv(y)` for `y ∈ (−1, 1)`.
///
/// Giles' rational approximation followed by two Halley refinement steps, good
/// to full `f64` precision. Returns `±∞` at `±1` and `NaN` outside `[−1, 1]`.
pub fn erfinv(y: f64) -> f64 {
    if y <= -1.0
    {
        return if y == -1.0
        {
            f64::NEG_INFINITY
        }
        else
        {
            f64::NAN
        };
    }
    if y >= 1.0
    {
        return if y == 1.0 { f64::INFINITY } else { f64::NAN };
    }
    if y == 0.0
    {
        return 0.0;
    }
    // Initial guess (Giles, 2010).
    let w = -((1.0 - y) * (1.0 + y)).ln();
    let mut x = if w < 5.0
    {
        let w = w - 2.5;
        let mut p = 2.810_226_36e-08;
        p = 3.432_739_39e-07 + p * w;
        p = -3.523_387_7e-06 + p * w;
        p = -4.391_506_54e-06 + p * w;
        p = 2.185_706_07e-04 + p * w;
        p = -0.001_253_725_03 + p * w;
        p = -0.004_177_681_64 + p * w;
        p = 0.246_640_727 + p * w;
        1.501_405_53 + p * w
    }
    else
    {
        let w = w.sqrt() - 3.0;
        let mut p = -0.000_200_214_257;
        p = 0.000_100_950_558 + p * w;
        p = 0.001_349_343_22 + p * w;
        p = -0.003_673_428_44 + p * w;
        p = 0.005_739_507_73 + p * w;
        p = -0.007_622_461_3 + p * w;
        p = 0.009_438_870_47 + p * w;
        p = 1.001_674_06 + p * w;
        2.832_976_82 + p * w
    } * y;
    // Two Halley steps sharpen the rational seed to full precision.
    for _ in 0..2
    {
        let err = erf(x) - y;
        let deriv = 2.0 / PI.sqrt() * (-x * x).exp();
        x -= err / (deriv - x * err); // Halley (uses erf'' = -2x·erf').
    }
    x
}

// ============================================================ //
//  Incomplete gamma  (regularized P and Q)                     //
// ============================================================ //

/// Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`,
/// `x ≥ 0`.
///
/// This is the CDF of a Gamma(a, 1) distribution — and, with `a = k/2`,
/// `x = χ²/2`, the χ²(k) CDF used throughout SPC and reliability. Series for
/// `x < a + 1`, continued fraction (via `Q`) otherwise, for accuracy across the
/// whole range.
pub fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 || a.is_nan() || x.is_nan()
    {
        return f64::NAN;
    }
    if x == 0.0
    {
        return 0.0;
    }
    if x < a + 1.0
    {
        gamma_series_p(a, x)
    }
    else
    {
        1.0 - gamma_cf_q(a, x)
    }
}

/// Regularized upper incomplete gamma `Q(a, x) = 1 − P(a, x) = Γ(a, x)/Γ(a)`.
///
/// Accurate in the far tail (survival function), where `1 − P` would cancel.
pub fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 || a.is_nan() || x.is_nan()
    {
        return f64::NAN;
    }
    if x == 0.0
    {
        return 1.0;
    }
    if x < a + 1.0
    {
        1.0 - gamma_series_p(a, x)
    }
    else
    {
        gamma_cf_q(a, x)
    }
}

// Series expansion for P(a, x), converging quickly when x < a + 1 — except in
// the boundary layer x ≈ a, where the number of terms needed to converge
// grows like O(√a) (this is precisely the regime Temme's 1987 uniform
// asymptotic expansion targets). Rather than silently truncate at a fixed
// `MAX_ITERS` and return a wrong result for large `a`, the cap scales with
// `√a` (empirically ~8·√a terms suffice; 20·√a leaves comfortable margin),
// bounded to avoid an unbounded loop on pathological input, and a genuine
// non-convergence returns `NaN` instead of a truncated series value.
fn gamma_series_p(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    let iters = ((20.0 * a.sqrt()).ceil() as usize)
        .saturating_add(MAX_ITERS)
        .min(50_000_000);
    let mut converged = false;
    for _ in 0..iters
    {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS
        {
            converged = true;
            break;
        }
    }
    if !converged
    {
        return f64::NAN;
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

// Continued-fraction expansion for Q(a, x) (modified Lentz), for x >= a + 1.
fn gamma_cf_q(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..MAX_ITERS
    {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY
        {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY
        {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS
        {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

// ============================================================ //
//  Incomplete beta  (regularized I_x)                          //
// ============================================================ //

/// Regularized incomplete beta `I_x(a, b) = B(x; a, b) / B(a, b)`, for
/// `x ∈ [0, 1]`, `a > 0`, `b > 0`.
///
/// The CDF of a Beta(a, b) distribution, and the tail integral behind the
/// Student-t and F distributions. Lentz continued fraction with the standard
/// `x < (a+1)/(a+b+2)` symmetry swap for fast convergence on both sides.
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || a.is_nan() || b.is_nan() || x.is_nan()
    {
        return f64::NAN;
    }
    if x <= 0.0
    {
        return 0.0;
    }
    if x >= 1.0
    {
        return 1.0;
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0)
    {
        front * beta_cf(a, b, x) / a
    }
    else
    {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

// Lentz continued fraction for the incomplete beta.
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY
    {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX_ITERS
    {
        let m = m as f64;
        let m2 = 2.0 * m;
        // Even step.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY
        {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY
        {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY
        {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY
        {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS
        {
            break;
        }
    }
    h
}

// ============================================================ //
//  Riemann zeta                                                //
// ============================================================ //

/// `B_{2k} / (2k)!` for `k = 1..=10` — the Bernoulli coefficients of the
/// Euler–Maclaurin correction (B₂ = 1/6, B₄ = −1/30, … B₂₀ = −174611/330).
const BERNOULLI_OVER_FACT: [f64; 10] = [
    8.333_333_333_333_333e-2,
    -1.388_888_888_888_889e-3,
    3.306_878_306_878_307e-5,
    -8.267_195_767_195_768e-7,
    2.087_675_698_786_81e-8,
    -5.284_190_138_687_493e-10,
    1.338_253_653_068_468e-11,
    -3.389_680_296_322_583e-13,
    8.586_062_056_277_845e-15,
    -2.174_868_698_558_062e-16,
];

/// Euler–Maclaurin tail `Σ_{j=m}^{∞} j^(−s)` for `s > 1`, `m ≥ 10`.
///
/// `∫_m^∞ x^(−s) dx + f(m)/2 + Σ_k B_{2k}/(2k)! · s(s+1)…(s+2k−2) ·
/// m^(−s−2k+1)` with a fixed 10-term budget — deterministic, and free of the
/// `ζ(s) − partial-sum` cancellation, which is what a far-tail survival
/// function needs.
pub fn riemann_zeta_tail(s: f64, m: f64) -> f64 {
    if s <= 1.0 || m < 10.0 || s.is_nan() || m.is_nan()
    {
        return f64::NAN;
    }
    let mut acc = m.powf(1.0 - s) / (s - 1.0) + 0.5 * m.powf(-s);
    let mut poch = s;
    let mut mpow = m.powf(-s - 1.0);
    for (k, c) in BERNOULLI_OVER_FACT.iter().enumerate()
    {
        acc += c * poch * mpow;
        let i = 2.0 * (k as f64 + 1.0);
        poch *= (s + i - 1.0) * (s + i);
        mpow /= m * m;
    }
    acc
}

/// Riemann zeta `ζ(s) = Σ_{j≥1} j^(−s)` for real `s > 1` (`NaN` otherwise).
///
/// Direct sum of the first 19 terms (smallest first, fixed order) plus the
/// Euler–Maclaurin tail at `m = 20` — deterministic, ~1e-15 relative across
/// the domain (checked against `scipy.special.zeta`: ζ(2) = π²/6,
/// ζ(3) = 1.2020569031595942…, ζ(1.5) = 2.6123753486854882…).
pub fn riemann_zeta(s: f64) -> f64 {
    if s <= 1.0 || s.is_nan()
    {
        return f64::NAN;
    }
    let mut acc = 0.0;
    for j in (1..20u32).rev()
    {
        acc += f64::from(j).powf(-s);
    }
    acc + riemann_zeta_tail(s, 20.0)
}

// ============================================================ //
//  Loader saddle-point pmf (Catherine Loader, 2000)            //
// ============================================================ //

const LN_2PI: f64 = 1.837_877_066_409_345_5; // ln(2π)

/// Stirling series remainder `δ(x) = ln Γ(x+1) − [(x+½)ln x − x + ½ln(2π)]`.
///
/// The small correction that Stirling's approximation omits. Computing a
/// binomial/Poisson pmf as `exp(Σ ln Γ)` loses relative precision at large
/// arguments because it subtracts large `ln Γ` values; keeping `δ` separate
/// (Loader 2000, as used by R's `dbinom`/`dpois`) avoids that cancellation.
///
/// For `x ≥ 16` an asymptotic series in `1/x` gives full `f64` accuracy; for
/// smaller `x` the direct `ln Γ` form is used (its cancellation there is mild).
/// `δ(0)` diverges and is never evaluated — the pmf endpoints are closed forms.
pub fn stirling_error(x: f64) -> f64 {
    if x <= 0.0 || x.is_nan()
    {
        return f64::NAN;
    }
    if x >= 16.0
    {
        // δ(x) = 1/(12x) − 1/(360x³) + 1/(1260x⁵) − 1/(1680x⁷) + 1/(1188x⁹).
        let inv = 1.0 / x;
        let z = inv * inv;
        inv * (1.0 / 12.0
            + z * (-1.0 / 360.0 + z * (1.0 / 1260.0 + z * (-1.0 / 1680.0 + z / 1188.0))))
    }
    else
    {
        ln_gamma(x + 1.0) - (x + 0.5) * x.ln() + x - 0.5 * LN_2PI
    }
}

/// Binomial deviance `D₀(x, np) = x·ln(x/np) + np − x`, evaluated by a series
/// near `x ≈ np` to avoid the cancellation of the naive logarithm form
/// (Loader 2000). `x > 0`, `np > 0`.
pub fn binom_deviance(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np)
    {
        let v = (x - np) / (x + np);
        let mut s = (x - np) * v;
        let mut ej = 2.0 * x * v;
        let v2 = v * v;
        for j in 1..MAX_ITERS
        {
            ej *= v2;
            let s1 = s + ej / (2 * j + 1) as f64;
            if s1 == s
            {
                return s1;
            }
            s = s1;
        }
        s
    }
    else
    {
        x * (x / np).ln() + np - x
    }
}

/// Natural log of the Poisson pmf `ln P(X = k)` for rate `λ > 0`, via the
/// Loader saddle-point form `−δ(k) − D₀(k, λ) − ½ln(2πk)` (with the closed
/// endpoint `−λ` at `k = 0`). Higher relative accuracy at large `λ` than
/// `k·ln λ − λ − ln k!`.
pub fn ln_poisson_pmf(k: u64, lambda: f64) -> f64 {
    if lambda <= 0.0 || lambda.is_nan()
    {
        return f64::NAN;
    }
    if k == 0
    {
        return -lambda;
    }
    let x = k as f64;
    -stirling_error(x) - binom_deviance(x, lambda) - 0.5 * (2.0 * PI * x).ln()
}

/// Natural log of the binomial pmf `ln P(X = k)` for `n` trials, success
/// probability `p ∈ [0, 1]`, via the Loader saddle-point form. Closed-form
/// endpoints handle `k = 0`, `k = n`, `p = 0`, `p = 1`; `k > n` gives `−∞`.
pub fn ln_binomial_pmf(k: u64, n: u64, p: f64) -> f64 {
    if k > n || !(0.0..=1.0).contains(&p)
    {
        return if k > n { f64::NEG_INFINITY } else { f64::NAN };
    }
    if k == 0
    {
        return n as f64 * (-p).ln_1p();
    }
    if k == n
    {
        return n as f64 * p.ln();
    }
    // Interior: p ∈ (0, 1) here (p = 0 ⇒ only k = 0, p = 1 ⇒ only k = n).
    let (nn, x) = (n as f64, k as f64);
    let q = 1.0 - p;
    let lc = stirling_error(nn)
        - stirling_error(x)
        - stirling_error(nn - x)
        - binom_deviance(x, nn * p)
        - binom_deviance(nn - x, nn * q);
    // dbinom = exp(lc)·sqrt(n / (2π·x·(n−x))).
    lc + 0.5 * (nn.ln() - LN_2PI - x.ln() - (nn - x).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    // ---- Riemann zeta ----

    #[test]
    fn riemann_zeta_matches_reference() {
        // ζ(2) = π²/6 (Basel), ζ(4) = π⁴/90.
        let pi = std::f64::consts::PI;
        assert!(close(riemann_zeta(2.0), pi * pi / 6.0, 1e-14));
        assert!(close(riemann_zeta(4.0), pi.powi(4) / 90.0, 1e-14));
        // scipy.special.zeta oracles.
        assert!(close(riemann_zeta(1.5), 2.612_375_348_685_488, 1e-14));
        assert!(close(riemann_zeta(2.5), 1.341_487_257_250_917_3, 1e-14));
        assert!(close(riemann_zeta(3.0), 1.202_056_903_159_594_2, 1e-14));
        assert!(close(riemann_zeta(4.2), 1.069_751_477_233_809_5, 1e-14));
        assert!(close(riemann_zeta(10.0), 1.000_994_575_127_818, 1e-14));
        assert!(close(riemann_zeta(25.0), 1.000_000_029_803_503_4, 1e-14));
        // Divergence pole side and invalid domain.
        assert!(riemann_zeta(1.0).is_nan());
        assert!(riemann_zeta(0.5).is_nan());
        assert!(riemann_zeta(f64::NAN).is_nan());
        // Near the pole: ζ(s) ~ 1/(s−1) + γ.
        let s = 1.000_001;
        assert!(close(
            riemann_zeta(s),
            1.0 / (s - 1.0) + 0.577_215_664_901_532_9,
            1e-9
        ));
    }

    #[test]
    fn riemann_zeta_tail_consistent_with_partial_sums() {
        // ζ(s) = Σ_{j<m} j^(−s) + tail(s, m) for any split m ≥ 10.
        for &s in &[1.2, 2.0, 3.7, 8.0]
        {
            for &m in &[10.0, 25.0, 100.0]
            {
                let mut partial = 0.0;
                let mi = m as u32;
                for j in (1..mi).rev()
                {
                    partial += f64::from(j).powf(-s);
                }
                assert!(
                    close(partial + riemann_zeta_tail(s, m), riemann_zeta(s), 1e-13),
                    "s = {s}, m = {m}"
                );
            }
        }
        assert!(riemann_zeta_tail(2.0, 5.0).is_nan()); // m below the budgeted floor
    }

    // ---- gamma family ----

    #[test]
    fn gamma_matches_factorials_and_sqrt_pi() {
        // Γ(n) = (n−1)!
        assert!(close(gamma(1.0), 1.0, 1e-13));
        assert!(close(gamma(5.0), 24.0, 1e-12));
        assert!(close(gamma(10.0), 362_880.0, 1e-11));
        // Γ(1/2) = √π
        assert!(close(gamma(0.5), PI.sqrt(), 1e-13));
        // Reflection: Γ(−0.5) = −2√π
        assert!(close(gamma(-0.5), -2.0 * PI.sqrt(), 1e-12));
    }

    #[test]
    fn ln_gamma_is_accurate_and_has_poles() {
        assert!(close(ln_gamma(100.0), 359.134_205_369_575_36, 1e-11));
        assert!(ln_gamma(0.0).is_infinite());
        assert!(ln_gamma(-3.0).is_infinite());
    }

    #[test]
    fn digamma_hits_known_values() {
        // ψ(1) = −γ
        assert!(close(digamma(1.0), -EULER_MASCHERONI, 1e-12));
        // ψ(1/2) = −γ − 2ln2
        assert!(close(
            digamma(0.5),
            -EULER_MASCHERONI - 2.0 * 2.0_f64.ln(),
            1e-12
        ));
        // ψ(2) = 1 − γ
        assert!(close(digamma(2.0), 1.0 - EULER_MASCHERONI, 1e-12));
    }

    #[test]
    fn beta_matches_closed_form() {
        // B(1, 1) = 1 ; B(2, 3) = 1/12.
        assert!(close(beta(1.0, 1.0), 1.0, 1e-13));
        assert!(close(beta(2.0, 3.0), 1.0 / 12.0, 1e-12));
    }

    // ---- error function ----

    #[test]
    fn erf_matches_tabulated_points() {
        assert!(close(erf(0.0), 0.0, 1e-15));
        assert!(close(erf(0.5), 0.520_499_877_813_046_5, 1e-12));
        assert!(close(erf(1.0), 0.842_700_792_949_714_9, 1e-12));
        assert!(close(erf(2.0), 0.995_322_265_018_952_7, 1e-12));
        // Odd symmetry.
        assert!(close(erf(-1.3), -erf(1.3), 1e-14));
    }

    #[test]
    fn erfc_is_accurate_in_the_tail() {
        // erfc stays meaningful where 1 − erf would be all rounding error.
        assert!(close(erfc(3.0), 2.209_049_699_858_544e-5, 1e-11));
        assert!(close(erfc(5.0), 1.537_459_794_428_035e-12, 1e-9));
        assert!(close(erf(2.0) + erfc(2.0), 1.0, 1e-14));
    }

    #[test]
    fn erfinv_inverts_erf() {
        for &y in &[-0.9, -0.4, 0.0, 0.25, 0.7, 0.99]
        {
            assert!(close(erf(erfinv(y)), y, 1e-12), "y = {y}");
        }
        assert!(erfinv(1.0).is_infinite());
        assert!(erfinv(-1.0).is_infinite());
        assert!(erfinv(1.5).is_nan());
    }

    // ---- incomplete gamma / χ² ----

    #[test]
    fn regularized_gamma_p_is_the_chi2_cdf() {
        // χ²(k=2) CDF at x is 1 − e^{−x/2}; with a = 1, arg = x/2.
        let cdf_at_2 = regularized_gamma_p(1.0, 1.0); // x = 2 → arg 1
        assert!(close(cdf_at_2, 1.0 - (-1.0_f64).exp(), 1e-12));
        // P + Q = 1.
        assert!(close(
            regularized_gamma_p(2.5, 3.0) + regularized_gamma_q(2.5, 3.0),
            1.0,
            1e-14
        ));
        // Boundary behaviour.
        assert!(close(regularized_gamma_p(3.0, 0.0), 0.0, 1e-15));
        assert!(regularized_gamma_p(3.0, 1e6) > 1.0 - 1e-12);
    }

    #[test]
    fn regularized_gamma_matches_reference_point() {
        // P(5, 5) = 0.5595067149… (a well-tabulated value).
        assert!(close(
            regularized_gamma_p(5.0, 5.0),
            0.559_506_714_934_7,
            1e-9
        ));
    }

    #[test]
    fn regularized_gamma_p_accurate_for_large_a_near_boundary() {
        // Regression test for a P0 audit finding: with the series capped at a
        // fixed MAX_ITERS=300, P(a, a) for large a converged too early and
        // returned a silently wrong value (e.g. P(1e4, 1e4) = 0.4999… instead
        // of 0.50133; P(1e5, 1e5) = 0.3294 instead of 0.50042). Reference
        // values computed with mpmath (dps=30): mp.gammainc(a, 0, a,
        // regularized=True).
        assert!(close(
            regularized_gamma_p(1e4, 1e4),
            0.501_329_808_339_955_2,
            1e-9
        ));
        assert!(close(
            regularized_gamma_p(1e5, 1e5),
            0.500_420_522_110_365_1,
            1e-9
        ));
        // P(a, a) → 1/2 as a → ∞ (Stirling), consistent with both points above.
        assert!(close(regularized_gamma_p(1e6, 1e6), 0.5, 1e-3));
        // P + Q = 1 must still hold exactly in this regime.
        assert!(close(
            regularized_gamma_p(1e5, 1e5) + regularized_gamma_q(1e5, 1e5),
            1.0,
            1e-12
        ));
    }

    // ---- incomplete beta ----

    #[test]
    fn incomplete_beta_endpoints_and_symmetry() {
        assert!(close(
            regularized_incomplete_beta(2.0, 3.0, 0.0),
            0.0,
            1e-15
        ));
        assert!(close(
            regularized_incomplete_beta(2.0, 3.0, 1.0),
            1.0,
            1e-15
        ));
        // Symmetry: I_x(a,b) = 1 − I_{1−x}(b,a).
        let a = 2.5;
        let b = 4.0;
        let x = 0.3;
        assert!(close(
            regularized_incomplete_beta(a, b, x),
            1.0 - regularized_incomplete_beta(b, a, 1.0 - x),
            1e-13
        ));
    }

    #[test]
    fn incomplete_beta_matches_reference_point() {
        // I_0.5(2,2) = 0.5 (symmetric Beta), and a tabulated asymmetric point.
        assert!(close(
            regularized_incomplete_beta(2.0, 2.0, 0.5),
            0.5,
            1e-13
        ));
        assert!(close(
            regularized_incomplete_beta(2.0, 3.0, 0.4),
            0.524_8,
            1e-4
        ));
    }

    #[test]
    // Miri deliberately randomizes the last ULPs of transcendental float
    // intrinsics (exp/ln/...) on every call, precisely so code cannot rely on
    // their unspecified precision — which makes this bit-identity check fail
    // under the interpreter by design. On real hardware (one binary, one libm)
    // the property holds and stays enforced by the native Build & Test jobs.
    #[cfg_attr(miri, ignore)]
    fn deterministic_across_calls() {
        // No global state / RNG: identical inputs give identical bits.
        assert_eq!(erf(1.234_567).to_bits(), erf(1.234_567).to_bits());
        assert_eq!(
            regularized_gamma_p(3.3, 2.2).to_bits(),
            regularized_gamma_p(3.3, 2.2).to_bits()
        );
    }

    #[test]
    fn stirling_error_matches_high_precision() {
        // Reference δ(n) from mpmath (40-digit lnΓ).
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * (1.0 + b.abs());
        assert!(close(stirling_error(1.0), 0.081_061_466_795_327_26));
        assert!(close(stirling_error(2.0), 0.041_340_695_955_409_3));
        assert!(close(stirling_error(5.0), 0.016_644_691_189_821_193));
        assert!(close(stirling_error(10.0), 0.008_330_563_433_362_87));
        assert!(close(stirling_error(15.0), 0.005_554_733_551_962_801));
        // Series branch (x ≥ 16); reference δ(16) = 0.00520765591960964…
        assert!(close(stirling_error(16.0), 0.005_207_655_919_609_64));
        assert!(close(stirling_error(100.0), 0.000_833_330_555_634_914_7));
        assert!(close(stirling_error(1000.0), 8.333_333_055_555_635e-5));
    }

    #[test]
    fn loader_pmf_matches_scipy_at_large_n() {
        // SciPy (which itself uses Loader) — the regime where the naive
        // exp(Σ lnΓ) path loses relative precision.
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * (1.0 + b.abs());
        assert!(close(
            ln_binomial_pmf(30_000, 100_000, 0.3).exp(),
            0.002_752_954_648_397_429
        ));
        assert!(close(
            ln_binomial_pmf(31_000, 100_000, 0.3).exp(),
            1.444_411_387_472_601_2e-13
        ));
        assert!(close(
            ln_binomial_pmf(500, 1000, 0.5).exp(),
            0.025_225_018_178_360_804
        ));
        assert!(close(
            ln_poisson_pmf(10_000, 10_000.0).exp(),
            0.003_989_389_558_963_281
        ));
        assert!(close(
            ln_poisson_pmf(50, 50.0).exp(),
            0.056_325_006_325_191_66
        ));
        // Endpoints and degenerate parameters.
        assert!(close(ln_binomial_pmf(0, 10, 0.3).exp(), 0.7_f64.powi(10)));
        assert!(close(ln_binomial_pmf(10, 10, 0.3).exp(), 0.3_f64.powi(10)));
        assert_eq!(ln_binomial_pmf(11, 10, 0.3), f64::NEG_INFINITY);
        assert!(close(ln_poisson_pmf(0, 4.2).exp(), (-4.2_f64).exp()));
        // Small-n agreement with the exact factorial form.
        let exact = |k: u32, n: u32, p: f64| {
            let fact = |m: u32| (1..=u64::from(m)).product::<u64>() as f64;
            let c = fact(n) / (fact(k) * fact(n - k));
            c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
        };
        assert!(close(ln_binomial_pmf(3, 10, 0.4).exp(), exact(3, 10, 0.4)));
    }
}

/// Property-based tests: mathematical identities and recurrences checked
/// against hundreds of randomly generated inputs, rather than a handful of
/// hand-picked points — the LAPACK/SciPy-style complement to the point-value
/// tests above (audit finding: no property-based testing anywhere in the
/// repo). Each property is a genuine, independent mathematical fact about
/// the function (a recurrence, a symmetry, a complementarity relation), not
/// a restatement of the implementation, so it can catch bugs the point
/// tests miss.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn rel_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    proptest! {
        /// Γ(x+1) = x·Γ(x), checked in log space (ln Γ(x+1) = ln Γ(x) + ln x)
        /// so it holds far beyond where `gamma()` itself would overflow.
        #[test]
        fn gamma_recurrence(x in 0.01f64..200.0) {
            let lhs = ln_gamma(x + 1.0);
            let rhs = ln_gamma(x) + x.ln();
            prop_assert!(rel_close(lhs, rhs, 1e-9), "x={x} lhs={lhs} rhs={rhs}");
        }

        /// ψ(x+1) = ψ(x) + 1/x (digamma recurrence).
        #[test]
        fn digamma_recurrence(x in 0.1f64..200.0) {
            let lhs = digamma(x + 1.0);
            let rhs = digamma(x) + 1.0 / x;
            prop_assert!(rel_close(lhs, rhs, 1e-7), "x={x} lhs={lhs} rhs={rhs}");
        }

        /// B(a+1, b) = B(a, b)·a/(a+b), checked in log space.
        #[test]
        fn beta_recurrence(a in 0.05f64..500.0, b in 0.05f64..500.0) {
            let lhs = ln_beta(a + 1.0, b);
            let rhs = ln_beta(a, b) + a.ln() - (a + b).ln();
            prop_assert!(rel_close(lhs, rhs, 1e-8), "a={a} b={b} lhs={lhs} rhs={rhs}");
        }

        /// erf is an odd function: erf(-x) = -erf(x).
        #[test]
        fn erf_is_odd(x in 0.0f64..8.0) {
            prop_assert!((erf(-x) + erf(x)).abs() < 1e-12, "x={x}");
        }

        /// erf and erfc are complementary: erf(x) + erfc(x) = 1.
        #[test]
        fn erf_erfc_complementary(x in -8.0f64..8.0) {
            prop_assert!(rel_close(erf(x) + erfc(x), 1.0, 1e-12), "x={x}");
        }

        /// erfinv is a genuine inverse of erf on (-1, 1).
        #[test]
        fn erfinv_inverts_erf_round_trip(y in -0.999_999f64..0.999_999) {
            let x = erfinv(y);
            let back = erf(x);
            prop_assert!(rel_close(back, y, 1e-6), "y={y} x={x} back={back}");
        }

        /// The regularized lower/upper incomplete gamma split unity:
        /// P(a, x) + Q(a, x) = 1 for every valid (a, x). NOTE: in both
        /// branches of `regularized_gamma_p`/`_q`, one side is defined
        /// *literally* as `1 - ` the other (see their bodies), so this
        /// identity holds by construction and cannot detect a wrong value
        /// from `gamma_series_p`/`gamma_cf_q` — it only guards against NaN,
        /// panics, and domain errors across a wide (a, x) range. Genuine
        /// accuracy checks are the recurrence and asymptotic properties
        /// below.
        #[test]
        fn regularized_gamma_p_plus_q_is_one(a in 0.01f64..2000.0, x in 0.0f64..4000.0) {
            let p = regularized_gamma_p(a, x);
            let q = regularized_gamma_q(a, x);
            prop_assert!(!p.is_nan() && !q.is_nan(), "a={a} x={x} p={p} q={q}");
            prop_assert!(rel_close(p + q, 1.0, 1e-6), "a={a} x={x} p={p} q={q}");
        }

        /// Independent accuracy check via the integration-by-parts
        /// recurrence `γ(a+1, x) = a·γ(a, x) − x^a·e^{−x}`, i.e.
        /// `P(a+1, x) = P(a, x) − x^a·e^{−x} / Γ(a+1)`. Unlike the P+Q=1
        /// identity above, the correction term is computed independently
        /// (via `ln_gamma`, not via `P` or `Q` themselves), so this *can*
        /// catch a truncated/non-converged `gamma_series_p` — including in
        /// the `x ≈ a` boundary layer (Temme 1987) where a P0 audit finding
        /// lived: a fixed iteration cap silently returned a wrong value for
        /// large `a` there.
        #[test]
        fn regularized_gamma_p_upward_recurrence(a in 0.5f64..5000.0, x in 0.01f64..10000.0) {
            let p_a = regularized_gamma_p(a, x);
            let p_a1 = regularized_gamma_p(a + 1.0, x);
            prop_assert!(!p_a.is_nan() && !p_a1.is_nan(), "a={a} x={x} p_a={p_a} p_a1={p_a1}");
            let correction = (a * x.ln() - x - ln_gamma(a + 1.0)).exp();
            prop_assert!(
                rel_close(p_a1, p_a - correction, 1e-6),
                "a={a} x={x} p_a={p_a} p_a1={p_a1} correction={correction}"
            );
        }

        /// Same recurrence, deliberately stressed at `x ≈ a` — the
        /// boundary layer where the defining series needs O(√a) terms to
        /// converge (Temme 1987). Uniform sampling over a wide (a, x) box
        /// (as in the property above) rarely lands close enough to `x = a`
        /// to exercise this regime, so it is targeted explicitly here.
        #[test]
        fn regularized_gamma_p_upward_recurrence_near_the_boundary_layer(
            a in 1000.0f64..50000.0,
            frac in 0.9f64..1.1,
        ) {
            let x = a * frac;
            let p_a = regularized_gamma_p(a, x);
            let p_a1 = regularized_gamma_p(a + 1.0, x);
            prop_assert!(!p_a.is_nan() && !p_a1.is_nan(), "a={a} x={x} p_a={p_a} p_a1={p_a1}");
            let correction = (a * x.ln() - x - ln_gamma(a + 1.0)).exp();
            prop_assert!(
                rel_close(p_a1, p_a - correction, 1e-6),
                "a={a} x={x} p_a={p_a} p_a1={p_a1} correction={correction}"
            );
        }

        /// Independent accuracy check at the boundary layer `x = a` itself,
        /// via Temme's (1987) uniform asymptotic expansion
        /// `P(a, a) = 1/2 + 1/(3·√(2πa)) + O(a^{-3/2})`, computed from
        /// scratch (no call into `regularized_gamma_p`/`_q`'s own
        /// machinery) — the same regime and the same P0 bug as above, but
        /// via a completely different oracle.
        #[test]
        fn regularized_gamma_p_matches_asymptotic_at_the_boundary(a in 1000.0f64..1_000_000.0) {
            let p = regularized_gamma_p(a, a);
            let asymptotic = 0.5 + 1.0 / (3.0 * (2.0 * std::f64::consts::PI * a).sqrt());
            prop_assert!(!p.is_nan(), "a={a} p={p}");
            prop_assert!(rel_close(p, asymptotic, 1e-4), "a={a} p={p} asymptotic={asymptotic}");
        }

        /// The regularized incomplete beta is complementary in its
        /// arguments: I_x(a, b) + I_{1-x}(b, a) = 1.
        #[test]
        fn regularized_incomplete_beta_symmetry(a in 0.05f64..200.0, b in 0.05f64..200.0, x in 0.0f64..1.0) {
            let lhs = regularized_incomplete_beta(a, b, x);
            let rhs = regularized_incomplete_beta(b, a, 1.0 - x);
            prop_assert!(rel_close(lhs + rhs, 1.0, 1e-6), "a={a} b={b} x={x} lhs={lhs} rhs={rhs}");
        }
    }
}
