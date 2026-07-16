//! **Certified transform pairs and certificate-driven summation** — the Phase-A
//! engineering deliverable of the CANR study
//! (`docs/research/CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md`).
//!
//! Two related capabilities:
//!
//! 1. **[`CertifiedMonotone`]** — monotone scalar transform pairs (encode/decode)
//!    that ship with *machine-checkable accuracy certificates*. The key quantity
//!    is the round-trip condition number
//!    `κ_rt(x) = |φ(x) / (x·φ′(x))|`, which is exactly the relative condition
//!    number of the **decoder** `φ⁻¹` at `y = φ(x)` (equivalently `1/cond(φ)(x)`;
//!    CANR §3.1 — standard inverse-function conditioning, no novelty claimed).
//!    First-order round-trip certificate (CANR §3.2, validated at 123/123 grid
//!    points in the study):
//!
//!    ```text
//!    |x̂ − x| / |x|  ≤  ( κ_rt(x)·B_ENC + B_DEC ) · u,          u = ε/2
//!    ```
//!
//!    and the hard domain rule `κ_rt(x)·u ≥ ½ ⇒ no recoverable digits`
//!    ([`CertifiedMonotone::invalid_at`]).
//!
//! 2. **Certificate-driven summation selection** ([`select_sum`]) — chooses the
//!    cheapest reduction whose *analytic* error bound (Higham 2002, ch. 4, in
//!    terms of the summation condition number `C_sum = Σ|xᵢ|/|Σxᵢ|`) meets a
//!    declared relative tolerance, falling back to an exact Shewchuk-expansion
//!    sum. Prototype validated in CANR §8/Y6 (all held-out checks passed).
//!
//! ## The libm-budget contract (honesty precondition)
//!
//! Every certificate here is conditional on the accuracy of the underlying
//! `ln/exp/ln_1p/exp_m1/powf/sqrt/asinh/sinh` kernels: the budgets
//! [`B_ENC`]/[`B_DEC`] (in ulps) assume *faithful-or-better* rounding, which
//! Rust's `std` math does not formally guarantee but comfortably meets on
//! mainstream platforms (typically ≤ 1–2 ulps). The default budget of 4 ulps
//! per direction is deliberately conservative; audited kernels (CORE-MATH /
//! Gappa-verified, cf. CANR §2) may lower it. These are *accuracy* budgets
//! only — for bit-exact cross-platform reproducibility of the transforms
//! themselves, pair this module with the deterministic kernels of
//! `scirust-simd` (see `reproducible`/`portable_f32` for the philosophy).
//!
//! ## Storage-policy notes baked into the API
//!
//! * **Box–Cox is provided as the *unshifted* power [`Power`]** (`x ↦ x^λ`),
//!   whose κ_rt ≡ 1/λ is flat. The classical shifted form `(x^λ−1)/λ` has
//!   κ_rt = |x^λ−1|/(λ·x^λ) → ∞ as x → 0 (up to 5.6×10¹⁴ ulps measured, and
//!   outright non-decodable below `x ≈ ε^{1/λ}` — CANR §5/Y1). Apply the affine
//!   shift symbolically on aggregates; it is exact there.
//! * **μ-law decodes via `exp_m1`** ([`MuLaw`]): the naive `pow(1+μ,y)−1`
//!   inverse loses up to 1.85×10¹² ulps near 0; the `exp_m1` kernel achieves
//!   ~2 ulps (CANR §5/Y2).
//! * **[`Anscombe`] keeps its 3/8 shift** — the Poisson variance stabilization
//!   requires it — and honestly reports the price: κ_rt = 2 + 0.75/x diverges
//!   as x → 0. The certificate machinery exposes this instead of hiding it.
//!   (For the full VST pipeline with bias-corrected inverses, see
//!   `scirust-signal::denoise::vst`.)
//!
//! Not to be confused with `nn::certified` (certified robustness of neural
//! networks) — unrelated notions of "certificate".

/// Encode-direction libm accuracy budget, in ulps (see module docs).
pub const B_ENC: f64 = 4.0;
/// Decode-direction libm accuracy budget, in ulps (see module docs).
pub const B_DEC: f64 = 4.0;

/// Unit roundoff of `f64` (half the machine epsilon).
const UNIT: f64 = f64::EPSILON / 2.0;

/// A closed interval `[lo, hi]` of `f64` values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Lower endpoint (inclusive).
    pub lo: f64,
    /// Upper endpoint (inclusive).
    pub hi: f64,
}

impl Interval {
    /// New interval; panics if `lo > hi` or either endpoint is NaN.
    pub fn new(lo: f64, hi: f64) -> Self {
        assert!(lo <= hi, "Interval requires lo <= hi (got {lo} > {hi})");
        Self { lo, hi }
    }

    /// Does the interval contain `x`?
    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }
}

/// A relative-error bound expressed in ulps of the reconstructed value.
///
/// Valid only under the module's libm-budget contract ([`B_ENC`]/[`B_DEC`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UlpBound {
    /// Maximum round-trip error, in units in the last place.
    pub ulps: f64,
}

/// A monotone scalar transform pair with machine-checkable accuracy
/// certificates (CANR §7, revised API).
///
/// Implementors guarantee, on [`Self::domain`]:
/// * `encode` is strictly monotone and `decode(encode(x)) = x` in exact
///   arithmetic (single branch, no hidden branch choices);
/// * [`Self::kappa_rt`] returns the exact round-trip condition number
///   `|φ(x)/(x·φ′(x))|` (limits filled in at removable singularities);
/// * [`Self::kappa_rt_sup`] returns a **sound** upper bound of `κ_rt` over the
///   given interval (each implementation documents its monotonicity argument —
///   endpoint evaluation plus, where κ_rt has an interior critical point, an
///   explicit closed-form cap).
pub trait CertifiedMonotone {
    /// Domain of validity of the pair.
    fn domain(&self) -> Interval;

    /// Encode `x`; `None` outside [`Self::domain`].
    fn encode(&self, x: f64) -> Option<f64>;

    /// Decode an encoded value (total on the encode image).
    fn decode(&self, y: f64) -> f64;

    /// Round-trip condition number `κ_rt(x) = |φ(x)/(x·φ′(x))|`.
    fn kappa_rt(&self, x: f64) -> f64;

    /// Sound upper bound of `κ_rt` over `iv` (must contain the true sup).
    fn kappa_rt_sup(&self, iv: Interval) -> f64;

    /// First-order round-trip certificate over `iv`, in ulps:
    /// `(sup κ_rt)·B_ENC + B_DEC` (CANR §3.2).
    fn roundtrip_bound(&self, iv: Interval) -> UlpBound {
        UlpBound {
            ulps: self.kappa_rt_sup(iv) * B_ENC + B_DEC,
        }
    }

    /// Hard invalidity rule: `κ_rt(x)·u ≥ ½` means the encoded value retains
    /// less than one significant bit about `x` — decode may fail entirely
    /// (CANR §3.3; predicted exactly the observed failures in Y1).
    fn invalid_at(&self, x: f64) -> bool {
        self.kappa_rt(x) * UNIT >= 0.5
    }
}

// ---------------------------------------------------------------------------
// Transform pairs
// ---------------------------------------------------------------------------

/// Natural logarithm pair `x ↦ ln x` on `(0, ∞)`. κ_rt(x) = |ln x|.
///
/// `κ_rt` is |ln| — monotone on each side of 1 — so its sup over any interval
/// is attained at an endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct Log;

impl CertifiedMonotone for Log {
    fn domain(&self) -> Interval {
        Interval::new(f64::MIN_POSITIVE, f64::MAX)
    }

    fn encode(&self, x: f64) -> Option<f64> {
        self.domain().contains(x).then(|| x.ln())
    }

    fn decode(&self, y: f64) -> f64 {
        y.exp()
    }

    fn kappa_rt(&self, x: f64) -> f64 {
        x.ln().abs()
    }

    fn kappa_rt_sup(&self, iv: Interval) -> f64 {
        self.kappa_rt(iv.lo).max(self.kappa_rt(iv.hi))
    }
}

/// Shifted logarithm pair `x ↦ ln(1+x)` on `(−1, ∞)`, exact fixed point at 0.
///
/// κ_rt(x) = |ln(1+x)|·(1+x)/|x| (limit 1 at x = 0). It is increasing on
/// `(0, ∞)` and increasing towards 1 on `(−1, 0)`, so
/// `sup = max(κ(lo), κ(hi), 1)` (the constant covers the interior limit).
#[derive(Debug, Clone, Copy, Default)]
pub struct Log1p;

impl CertifiedMonotone for Log1p {
    fn domain(&self) -> Interval {
        Interval::new(-1.0 + f64::EPSILON, f64::MAX)
    }

    fn encode(&self, x: f64) -> Option<f64> {
        self.domain().contains(x).then(|| x.ln_1p())
    }

    fn decode(&self, y: f64) -> f64 {
        y.exp_m1()
    }

    fn kappa_rt(&self, x: f64) -> f64 {
        if x == 0.0
        {
            return 1.0;
        }
        (x.ln_1p() * (1.0 + x) / x).abs()
    }

    fn kappa_rt_sup(&self, iv: Interval) -> f64 {
        self.kappa_rt(iv.lo).max(self.kappa_rt(iv.hi)).max(1.0)
    }
}

/// Signed logarithm pair `x ↦ asinh x` on all of ℝ, exact fixed point at 0.
///
/// κ_rt(x) = |asinh x|·√(1+x²)/|x| (limit 1 at 0), even and increasing in
/// |x|, so `sup = max(κ(lo), κ(hi), 1)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SignedLog;

impl CertifiedMonotone for SignedLog {
    fn domain(&self) -> Interval {
        Interval::new(f64::MIN, f64::MAX)
    }

    fn encode(&self, x: f64) -> Option<f64> {
        Some(x.asinh())
    }

    fn decode(&self, y: f64) -> f64 {
        y.sinh()
    }

    fn kappa_rt(&self, x: f64) -> f64 {
        if x == 0.0
        {
            return 1.0;
        }
        (x.asinh() * (1.0 + x * x).sqrt() / x).abs()
    }

    fn kappa_rt_sup(&self, iv: Interval) -> f64 {
        self.kappa_rt(iv.lo).max(self.kappa_rt(iv.hi)).max(1.0)
    }
}

/// Unshifted power pair `x ↦ x^λ` on `[0, ∞)`, λ > 0 — the numerically sound
/// storage for the Box–Cox family (see module docs for the shift policy).
///
/// κ_rt ≡ 1/λ, constant — but the certificate is **not** purely flat: the
/// decoder evaluates `y^(1/λ)` with the *rounded* exponent `fl(1/λ)`, whose
/// relative error `≤ u/2` is amplified by `|ln y| = λ·|ln x|`, contributing up
/// to `|ln x|/2` extra ulps. [`Power::roundtrip_bound`] includes that term
/// (this is precisely the ~13-ulp floor observed at |ln x| ≈ 28 in CANR Y1).
#[derive(Debug, Clone, Copy)]
pub struct Power {
    /// Exponent λ > 0.
    pub lambda: f64,
}

impl Power {
    /// New power transform; panics unless `lambda > 0` and finite.
    pub fn new(lambda: f64) -> Self {
        assert!(
            lambda > 0.0 && lambda.is_finite(),
            "Power requires lambda > 0"
        );
        Self { lambda }
    }
}

impl CertifiedMonotone for Power {
    fn domain(&self) -> Interval {
        Interval::new(0.0, f64::MAX)
    }

    fn encode(&self, x: f64) -> Option<f64> {
        self.domain().contains(x).then(|| x.powf(self.lambda))
    }

    fn decode(&self, y: f64) -> f64 {
        y.powf(1.0 / self.lambda)
    }

    fn kappa_rt(&self, _x: f64) -> f64 {
        1.0 / self.lambda
    }

    fn kappa_rt_sup(&self, _iv: Interval) -> f64 {
        1.0 / self.lambda
    }

    fn roundtrip_bound(&self, iv: Interval) -> UlpBound {
        // Exponent-rounding term: |ln x|/2 ulps, worst case over the interval
        // (endpoints suffice: |ln| is monotone on each side of 1).
        let ln_sup = ln_abs_sup(iv);
        UlpBound {
            ulps: B_ENC / self.lambda + B_DEC + 0.5 * ln_sup,
        }
    }
}

/// Sup of |ln x| over a positive interval (endpoint evaluation; 0 maps to 0
/// exactly in [`Power`], so a zero endpoint contributes nothing).
fn ln_abs_sup(iv: Interval) -> f64 {
    let at = |x: f64| if x > 0.0 { x.ln().abs() } else { 0.0 };
    at(iv.lo).max(at(iv.hi))
}

/// μ-law companding pair on `[−1, 1]` with the certified `exp_m1` decoder.
///
/// κ_rt(x) = ln(1+μ|x|)·(1+μ|x|)/(μ|x|) (limit 1 at 0), increasing in |x| and
/// bounded by κ(1) ≈ 5.57 for μ = 255, so `sup = max(κ(lo), κ(hi), 1)`.
#[derive(Debug, Clone, Copy)]
pub struct MuLaw {
    /// Companding parameter μ > 0 (255 for ITU-T G.711).
    pub mu: f64,
}

impl MuLaw {
    /// New μ-law transform; panics unless `mu > 0` and finite.
    pub fn new(mu: f64) -> Self {
        assert!(mu > 0.0 && mu.is_finite(), "MuLaw requires mu > 0");
        Self { mu }
    }
}

impl CertifiedMonotone for MuLaw {
    fn domain(&self) -> Interval {
        Interval::new(-1.0, 1.0)
    }

    fn encode(&self, x: f64) -> Option<f64> {
        self.domain()
            .contains(x)
            .then(|| (self.mu * x.abs()).ln_1p() / self.mu.ln_1p() * x.signum())
    }

    fn decode(&self, y: f64) -> f64 {
        (y.abs() * self.mu.ln_1p()).exp_m1() / self.mu * y.signum()
    }

    fn kappa_rt(&self, x: f64) -> f64 {
        let t = self.mu * x.abs();
        if t == 0.0
        {
            return 1.0;
        }
        t.ln_1p() * (1.0 + t) / t
    }

    fn kappa_rt_sup(&self, iv: Interval) -> f64 {
        self.kappa_rt(iv.lo).max(self.kappa_rt(iv.hi)).max(1.0)
    }
}

/// Logit pair `x ↦ ln(x/(1−x))` on `(0, 1)`.
///
/// κ_rt(x) = |ln(x/(1−x))|·(1−x): decreasing on `(0, ½)`, and on `(½, 1)` it
/// has one interior maximum bounded in closed form by
/// `max_t t·ln((1−t)/t) < 0.2785` (t = 1−x), so
/// `sup = max(κ(lo), κ(hi), 0.2785)`.
///
/// **Caveat**: the certificate bounds the relative error *in x*. Near x = 1
/// the statistically meaningful quantity is usually the complement 1−x, whose
/// relative error is **not** bounded by this certificate — encode the
/// complement instead when tail probabilities matter.
#[derive(Debug, Clone, Copy, Default)]
pub struct Logit;

/// Interior bound of κ_rt for [`Logit`] on `(½, 1)`: `max_t t·ln((1−t)/t)`.
const LOGIT_INTERIOR_CAP: f64 = 0.2785;

impl CertifiedMonotone for Logit {
    fn domain(&self) -> Interval {
        Interval::new(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    }

    fn encode(&self, x: f64) -> Option<f64> {
        self.domain().contains(x).then(|| (x / (1.0 - x)).ln())
    }

    fn decode(&self, y: f64) -> f64 {
        1.0 / (1.0 + (-y).exp())
    }

    fn kappa_rt(&self, x: f64) -> f64 {
        ((x / (1.0 - x)).ln() * (1.0 - x)).abs()
    }

    fn kappa_rt_sup(&self, iv: Interval) -> f64 {
        self.kappa_rt(iv.lo)
            .max(self.kappa_rt(iv.hi))
            .max(LOGIT_INTERIOR_CAP)
    }
}

/// Anscombe pair `x ↦ 2√(x + 3/8)` on `[0, ∞)` (Poisson variance stabilizer).
///
/// κ_rt(x) = 2 + 0.75/x, strictly decreasing, so the sup over an interval is
/// κ(lo). The divergence as x → 0 is the honest price of the 3/8 shift the
/// statistics require — [`CertifiedMonotone::invalid_at`] flags the region.
#[derive(Debug, Clone, Copy, Default)]
pub struct Anscombe;

impl CertifiedMonotone for Anscombe {
    fn domain(&self) -> Interval {
        Interval::new(0.0, f64::MAX)
    }

    fn encode(&self, x: f64) -> Option<f64> {
        self.domain().contains(x).then(|| 2.0 * (x + 0.375).sqrt())
    }

    fn decode(&self, y: f64) -> f64 {
        let h = y / 2.0;
        h * h - 0.375
    }

    fn kappa_rt(&self, x: f64) -> f64 {
        if x == 0.0
        {
            return f64::INFINITY;
        }
        2.0 + 0.75 / x
    }

    fn kappa_rt_sup(&self, iv: Interval) -> f64 {
        self.kappa_rt(iv.lo)
    }
}

// ---------------------------------------------------------------------------
// Compensated reductions + certificate-driven selection
// ---------------------------------------------------------------------------

/// Neumaier's compensated sum (Kahan–Babuška): error ≤ (2u + O(n²u²))·Σ|xᵢ|,
/// robust to terms larger than the running sum (unlike plain Kahan).
pub fn sum_neumaier(xs: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut c = 0.0;
    for &x in xs
    {
        let t = s + x;
        if s.abs() >= x.abs()
        {
            c += (s - t) + x;
        }
        else
        {
            c += (x - t) + s;
        }
        s = t;
    }
    s + c
}

/// Klein's second-order compensated sum (Klein, *Computing* 76:279–293, 2006):
/// a doubly-compensated Kahan–Babuška variant, roughly one order more accurate
/// than [`sum_neumaier`] on adversarial cancellation at ~2× its cost.
pub fn sum_klein(xs: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut cs = 0.0;
    let mut ccs = 0.0;
    for &x in xs
    {
        let t = s + x;
        let c = if s.abs() >= x.abs()
        {
            (s - t) + x
        }
        else
        {
            (x - t) + s
        };
        s = t;
        let t2 = cs + c;
        let cc = if cs.abs() >= c.abs()
        {
            (cs - t2) + c
        }
        else
        {
            (c - t2) + cs
        };
        cs = t2;
        ccs += cc;
    }
    s + cs + ccs
}

/// Fixed-tree pairwise sum: error ≤ (⌈log₂n⌉·u + O(u²))·Σ|xᵢ|. Deterministic
/// for a given input order (the tree is a pure function of the length).
pub fn sum_pairwise(xs: &[f64]) -> f64 {
    if xs.len() <= 8
    {
        let mut s = 0.0;
        for &x in xs
        {
            s += x;
        }
        return s;
    }
    let mid = xs.len() / 2;
    sum_pairwise(&xs[..mid]) + sum_pairwise(&xs[mid..])
}

/// Exact-expansion sum (Shewchuk 1997): maintains a non-overlapping expansion
/// whose exact value is the running sum, then rounds once at the end. The
/// result is faithful (≤ 1 ulp of the exact sum); the expansion itself is
/// exact for any input order.
pub fn sum_expansion(xs: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for &xi in xs
    {
        let mut x = xi;
        let mut i = 0;
        for j in 0..partials.len()
        {
            let mut y = partials[j];
            if x.abs() < y.abs()
            {
                core::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0
            {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    let mut s = 0.0;
    for &p in &partials
    {
        s += p;
    }
    s
}

/// Summation condition number `C_sum = Σ|xᵢ| / |Σxᵢ|` (Higham 2002, §4.2) —
/// the certificate input for [`select_sum`]. Returns `f64::INFINITY` when the
/// exact sum is 0. Both aggregates are computed with compensated/exact methods
/// so the certificate itself is trustworthy.
pub fn sum_condition(xs: &[f64]) -> f64 {
    let abs: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
    let denom = sum_expansion(xs).abs();
    if denom == 0.0
    {
        return f64::INFINITY;
    }
    sum_neumaier(&abs) / denom
}

/// Reduction methods [`select_sum`] can choose from, in increasing cost order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumMethod {
    /// Plain sequential sum — 1 flop/element.
    Naive,
    /// [`sum_pairwise`] — 1 flop/element, log-depth error growth.
    Pairwise,
    /// [`sum_neumaier`] — ~7 flops/element.
    Neumaier,
    /// [`sum_expansion`] — exact fallback, data-dependent cost.
    Expansion,
}

/// Result of a certificate-driven summation selection.
#[derive(Debug, Clone, Copy)]
pub struct SumSelection {
    /// The chosen (cheapest certified) method.
    pub method: SumMethod,
    /// The computed sum, using the chosen method.
    pub value: f64,
    /// The analytic relative-error bound of the chosen method on this input.
    pub cert_rel_bound: f64,
    /// The measured summation condition number `C_sum`.
    pub condition: f64,
}

/// Certificate-driven summation (CANR §8, stage S1+S2): picks the **cheapest**
/// method whose analytic Higham bound `K(method, n)·C_sum ≤ tau`, falling back
/// to the exact expansion. Sound but conservative — bounds are worst-case, so
/// the selector may over-provision (never under-provision); pair with
/// empirical refinement when the selection cost matters (CANR §8, S3/S4).
pub fn select_sum(xs: &[f64], tau: f64) -> SumSelection {
    let n = xs.len().max(2) as f64;
    let condition = sum_condition(xs);
    let candidates: [(SumMethod, f64); 3] = [
        (SumMethod::Naive, (n - 1.0) * UNIT),
        (SumMethod::Pairwise, n.log2().ceil() * UNIT * 1.05),
        (SumMethod::Neumaier, 2.0 * UNIT + n * n * UNIT * UNIT),
    ];
    for (method, coeff) in candidates
    {
        let bound = coeff * condition;
        if bound <= tau
        {
            let value = match method
            {
                SumMethod::Naive => xs.iter().sum(),
                SumMethod::Pairwise => sum_pairwise(xs),
                SumMethod::Neumaier => sum_neumaier(xs),
                SumMethod::Expansion => unreachable!(),
            };
            return SumSelection {
                method,
                value,
                cert_rel_bound: bound,
                condition,
            };
        }
    }
    SumSelection {
        method: SumMethod::Expansion,
        value: sum_expansion(xs),
        cert_rel_bound: UNIT,
        condition,
    }
}

// ---------------------------------------------------------------------------
// Tests — oracles are the measured bounds of the CANR study (Y1/Y2/Y5/Y6)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// One ulp of `x` (spacing to the next representable magnitude).
    fn ulp(x: f64) -> f64 {
        let a = x.abs().max(f64::MIN_POSITIVE);
        a.next_up() - a
    }

    /// Log-spaced samples of [a, b] (both > 0).
    fn logspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        let (la, lb) = (a.log10(), b.log10());
        (0..n)
            .map(|i| 10f64.powf(la + (lb - la) * i as f64 / (n - 1) as f64))
            .collect()
    }

    /// Assert the round-trip certificate over sampled points of `iv`, and that
    /// the sampled κ_rt never exceeds the claimed sup.
    fn check_certificate<T: CertifiedMonotone>(t: &T, iv: Interval, samples: &[f64]) {
        let sup = t.kappa_rt_sup(iv);
        let bound = t.roundtrip_bound(iv).ulps;
        for &x in samples
        {
            assert!(iv.contains(x));
            assert!(
                t.kappa_rt(x) <= sup * (1.0 + 1e-12),
                "kappa_rt({x}) = {} exceeds claimed sup {sup}",
                t.kappa_rt(x)
            );
            if t.invalid_at(x)
            {
                continue;
            }
            let y = t.encode(x).expect("sample must lie in the domain");
            let r = t.decode(y);
            let err_ulps = (r - x).abs() / ulp(x);
            assert!(
                err_ulps <= bound,
                "round trip at x = {x}: {err_ulps} ulps exceeds certified {bound}"
            );
        }
    }

    #[test]
    fn log_certificate_holds_over_600_decades() {
        let iv = Interval::new(1e-300, 1e300);
        check_certificate(&Log, iv, &logspace(1e-300, 1e300, 121));
        // CANR Y-series measured ~254 ulps worst case; certificate ~2767.
        assert!(Log.roundtrip_bound(iv).ulps < 3000.0);
    }

    #[test]
    fn log1p_and_signed_log_certificates_hold() {
        check_certificate(
            &Log1p,
            Interval::new(1e-15, 1e15),
            &logspace(1e-15, 1e15, 61),
        );
        let samples: Vec<f64> = logspace(1e-300, 1e300, 61)
            .into_iter()
            .map(|x| -x)
            .collect();
        check_certificate(&SignedLog, Interval::new(-1e300, -1e-300), &samples);
    }

    #[test]
    fn power_certificate_is_flat_and_tight() {
        for lambda in [0.1, 0.25, 0.5, 0.75, 1.5]
        {
            let p = Power::new(lambda);
            let iv = Interval::new(1e-12, 1e12);
            check_certificate(&p, iv, &logspace(1e-12, 1e12, 49));
            // Flat kappa term plus the exponent-rounding term (|ln 1e-12|/2).
            let expected = B_ENC / lambda + B_DEC + 0.5 * 1e12f64.ln();
            assert!((p.roundtrip_bound(iv).ulps - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn mu_law_expm1_decoder_survives_tiny_inputs() {
        let m = MuLaw::new(255.0);
        // CANR Y2: naive pow-based inverse loses 1.85e12 ulps at |x| = 1e-15;
        // the exp_m1 decoder stays within the ~9.4-ulp certificate.
        let iv = Interval::new(-1.0, 1.0);
        let mut samples = logspace(1e-15, 1.0, 46);
        samples.extend(logspace(1e-15, 1.0, 46).into_iter().map(|x| -x));
        check_certificate(&m, iv, &samples);
        assert!(m.roundtrip_bound(iv).ulps < 30.0);
    }

    #[test]
    fn logit_certificate_holds_including_interior_max() {
        let iv = Interval::new(1e-16, 1.0 - 1e-10);
        let mut samples = logspace(1e-16, 0.5, 31);
        samples.extend([0.6, 0.7, 0.78, 0.8, 0.9, 0.99, 1.0 - 1e-6, 1.0 - 1e-10]);
        check_certificate(&Logit, iv, &samples);
        // Interior maximum on (1/2, 1) is covered by the closed-form cap.
        assert!(Logit.kappa_rt(0.78) < LOGIT_INTERIOR_CAP);
    }

    #[test]
    fn anscombe_reports_its_shift_cost_honestly() {
        check_certificate(
            &Anscombe,
            Interval::new(0.5, 1e12),
            &logspace(0.5, 1e12, 49),
        );
        // Divergence near 0 is exposed, not hidden: the invalid region starts
        // where kappa_rt * u >= 1/2, i.e. x <= 0.75 * u / 0.5 ~ 1.7e-16.
        assert!(!Anscombe.invalid_at(1e-15));
        assert!(Anscombe.invalid_at(1e-17));
        assert!(!Anscombe.invalid_at(1.0));
    }

    #[test]
    fn klein_and_neumaier_survive_the_kahan_killer() {
        // Canonical case where plain Kahan returns 0 (CANR §3.5 / phase-2 X2a).
        let xs = [1.0, 1e100, 1.0, -1e100];
        assert_eq!(sum_neumaier(&xs), 2.0);
        assert_eq!(sum_klein(&xs), 2.0);
        assert_eq!(sum_expansion(&xs), 2.0);
        let naive: f64 = xs.iter().sum();
        assert_eq!(naive, 0.0); // documents why the compensated forms exist
    }

    #[test]
    fn expansion_sum_is_exact_on_adversarial_cancellation() {
        // Pairs (b, -b) plus small residuals: exact sum is the residual sum.
        let mut rng = StdRng::seed_from_u64(7);
        let mut xs = Vec::new();
        let mut residuals = Vec::new();
        for _ in 0..20_000
        {
            let b = 10f64.powf(rng.gen_range(8.0..14.0));
            let d = rng.gen_range(-1.0..1.0);
            xs.extend([b, -b, d]);
            residuals.push(d);
        }
        let exact = sum_expansion(&residuals); // residuals are benign
        let got = sum_expansion(&xs);
        assert!(
            (got - exact).abs() <= ulp(exact),
            "expansion sum not faithful"
        );
    }

    #[test]
    fn certificate_driven_selection_passes_held_out_validation() {
        // The Y6 prototype, in Rust: three workload families, selection on one
        // seed, validation on fresh seeds. Reference = exact expansion sum.
        let make = |kind: usize, seed: u64| -> Vec<f64> {
            let mut rng = StdRng::seed_from_u64(seed);
            match kind
            {
                0 => (0..30_000).map(|_| rng.gen_range(0.5..1.5)).collect(),
                1 => (0..30_000)
                    .map(|_| 10f64.powf(rng.gen_range(-25.0..25.0)))
                    .collect(),
                _ =>
                {
                    let mut xs = Vec::new();
                    for _ in 0..10_000
                    {
                        let b = 10f64.powf(rng.gen_range(8.0..14.0));
                        xs.extend([b, -b, rng.gen_range(-1.0..1.0)]);
                    }
                    xs
                },
            }
        };
        let cases = [
            (0usize, 1e-9, SumMethod::Naive),
            (1, 1e-13, SumMethod::Pairwise),
            (2, 1e-8, SumMethod::Expansion),
        ];
        for (kind, tau, expect) in cases
        {
            let sel = select_sum(&make(kind, 1), tau);
            assert_eq!(sel.method, expect, "workload {kind}: unexpected selection");
            assert!(sel.cert_rel_bound <= tau);
            for seed in [11, 12, 13]
            {
                let test = make(kind, seed);
                let reference = sum_expansion(&test);
                let value = match sel.method
                {
                    SumMethod::Naive => test.iter().sum(),
                    SumMethod::Pairwise => sum_pairwise(&test),
                    SumMethod::Neumaier => sum_neumaier(&test),
                    SumMethod::Expansion => sum_expansion(&test),
                };
                let rel = ((value - reference) / reference).abs();
                assert!(rel <= tau, "workload {kind} seed {seed}: {rel} > tau {tau}");
            }
        }
    }

    #[test]
    fn selection_reports_infinite_condition_for_zero_sums() {
        let xs = [1.0, -1.0, 2.5, -2.5];
        assert_eq!(sum_condition(&xs), f64::INFINITY);
        // Exact fallback still returns the true zero.
        assert_eq!(select_sum(&xs, 1e-12).value, 0.0);
        assert_eq!(select_sum(&xs, 1e-12).method, SumMethod::Expansion);
    }
}
