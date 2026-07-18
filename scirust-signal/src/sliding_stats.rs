//! Fixed-window online mean and centered second moment.
//!
//! # Mathematical contract
//!
//! For a window of `N` samples `x_1, …, x_N`, define the **mean**
//!
//! ```text
//! μ = (1/N) · Σ xᵢ
//! ```
//!
//! and the **centered second moment**
//!
//! ```text
//! M2 = Σ (xᵢ − μ)²
//! ```
//!
//! `M2` is the sum of squared deviations, *not yet divided by a sample count* —
//! it is the quantity the online recurrences below update directly, because it
//! (unlike variance) has a simple additive update. From it:
//!
//! * the **population variance** `σ²_pop = M2 / N` treats the current window as
//!   the entire population of interest (divisor `N`);
//! * the **sample variance** `s² = M2 / (N − 1)` is Bessel's-corrected unbiased
//!   estimator of the variance of the distribution the window was drawn from
//!   (divisor `N − 1`, undefined for `N < 2`).
//!
//! [`SlidingMoments::population_variance`] and [`SlidingMoments::sample_variance`]
//! expose both explicitly — this module never uses the word "variance"
//! unqualified. Callers building a statistic on top (e.g. the Variability Index
//! in [`crate::radar::vi_cfar`]) must state which one they use.
//!
//! # Input domain
//!
//! Every pushed value must be finite (`f64::is_finite`); `NaN` and `±∞` are
//! rejected with [`SlidingMomentsError::NonFiniteSample`] rather than silently
//! corrupting the running moments. A [`SlidingMoments`] constructed with
//! [`SampleDomain::NonNegative`] additionally rejects negative values — the
//! domain a power/energy series (such as a radar range profile) must satisfy —
//! while [`SampleDomain::Real`] accepts any finite value. Neither domain is a
//! default guess: the constructor name states which one a caller gets
//! ([`SlidingMoments::new`] for `Real`, [`SlidingMoments::new_non_negative`] for
//! `NonNegative`).
//!
//! # Online recurrences
//!
//! ## Warm-up (window not yet full): Welford's algorithm
//!
//! Welford's method (Welford, 1962; see Knuth, *The Art of Computer
//! Programming* Vol. 2, §4.2.2) updates the mean and `M2` from one previous
//! sample count `n − 1` to `n` without ever forming `Σ xᵢ²` (whose subtraction
//! from `(Σxᵢ)²/n` is the classic catastrophic-cancellation trap for
//! nearly-constant data):
//!
//! ```text
//! δ      = xₙ − μₙ₋₁
//! μₙ     = μₙ₋₁ + δ/n
//! M2ₙ    = M2ₙ₋₁ + δ·(xₙ − μₙ)
//! ```
//!
//! Sanity check on `{2, 4}`: `n=1` gives `μ=2, M2=0`. `n=2`: `δ = 4−2 = 2`,
//! `μ₂ = 2 + 2/2 = 3`, `M2₂ = 0 + 2·(4−3) = 2` — matching the direct computation
//! `(2−3)² + (4−3)² = 2`.
//!
//! ## Full window: replacement recurrence
//!
//! Once `N` samples have been seen, each further push replaces the oldest
//! sample `x_out` with the incoming `x_in`, holding `N` fixed. Derivation: let
//! `Q = Σ xᵢ²` (used only as an algebraic device here, never computed at
//! runtime). Since `M2 = Q − N·μ²`,
//!
//! ```text
//! M2_new − M2_old = (Q_new − N·μ_new²) − (Q_old − N·μ_old²)
//!                 = (x_in² − x_out²) + N·(μ_old² − μ_new²)        [only one term of Q changed]
//! ```
//!
//! With `μ_new − μ_old = (x_in − x_out)/N` (immediate from `N·μ = Σxᵢ`),
//! `N·(μ_old − μ_new) = −(x_in − x_out)`, so
//!
//! ```text
//! M2_new = M2_old + (x_in − x_out)(x_in + x_out) − (x_in − x_out)(μ_new + μ_old)
//!        = M2_old + (x_in − x_out)(x_in + x_out − μ_new − μ_old)
//! ```
//!
//! which is the recurrence implemented below. The implementation never
//! materializes `Q`; the derivation uses it only to find the closed form, which
//! is then evaluated using only the centered quantities `M2`, `μ_old`, `μ_new`
//! — preserving the cancellation resistance Welford-style tracking exists for.
//! `μ_new` itself is updated the same way as in the classical CFAR literature's
//! cell-averaging recurrence: `μ_new = μ_old + (x_in − x_out)/N`.
//!
//! # Complexity
//!
//! [`SlidingMoments::push`] is **O(1)** after construction — both branches above
//! do a fixed number of arithmetic operations independent of `N`. Storage is
//! **O(N)**: the full circular buffer of samples is retained (not just the
//! summary statistics), because [`SlidingMoments::recompute`] and the robust
//! estimators in [`crate::radar::vi_cfar`] need the raw values, not only their
//! moments. This is *not* O(1) space, and this module does not claim it is.
//!
//! # Compile-time vs. runtime capacity
//!
//! [`SlidingMoments<N>`] fixes `N` at compile time (a stack-allocated
//! `[f64; N]`, zero heap allocation). [`SlidingMomentsDyn`] runs the
//! identical recurrences and numerical-health policy below over a
//! `Box<[f64]>` sized at construction instead, for callers whose window size
//! is only known at runtime (e.g. read from a configuration value rather
//! than fixed as a `const`) — at the cost of one heap allocation up front;
//! neither type's `push` allocates. Both delegate to the same private core,
//! so this module has exactly one implementation of the algorithm below, not
//! two that could silently drift apart.
//!
//! # Numerical-health policy
//!
//! `M2` is mathematically non-negative. In floating point, the O(1) recurrences
//! above can drift slightly negative under sustained round-off (most visibly
//! when the data has a large common baseline and a tiny variation on top — see
//! the `huge_baseline_tiny_variation` test). The policy, checked after every
//! push:
//!
//! 1. **Clamp** — if `M2 ∈ [−tol, 0)`, treat it as an exact zero. `tol` is
//!    *scale-aware*: `tol = 64 · f64::EPSILON · max(1, μ²·len)`, i.e. a small
//!    multiple of machine epsilon relative to the natural magnitude of `M2` for
//!    well-behaved data of this mean and window occupancy (the standard
//!    forward-error argument for summing `len` floating-point terms of
//!    magnitude `~μ²` bounds the accumulated error at `O(len · u · μ²)` for unit
//!    roundoff `u`; 64 is a documented, generous constant, not a tight bound).
//!    This is *not* an unconditional clamp-to-zero: a negative excursion
//!    outside this band is treated as real degradation (step 2), not roundoff.
//! 2. **Recompute** — if `M2` (or `μ`) is non-finite, or `M2` is negative beyond
//!    `tol`, the state is rebuilt exactly from the retained circular buffer
//!    (two passes: mean, then Σ(xᵢ − μ)² term-by-term, each term individually
//!    non-negative so the sum cannot go negative from cancellation).
//! 3. **Integrity error** — if the state is still non-finite or `M2` is still
//!    negative after that exact recomputation (only reachable through sustained
//!    values near `f64::MAX` that overflow the running sum), `push` /
//!    `recompute` return [`SlidingMomentsError::NumericalIntegrity`] rather than
//!    silently returning a wrong statistic.
//!
//! There is deliberately no periodic "recompute every K pushes" timer: nothing
//! in this policy is calendar-based, only degradation-triggered, per the design
//! brief for this module.

use thiserror::Error;

/// Whether a [`SlidingMoments`] accepts any finite value, or restricts samples
/// to the finite non-negative power/energy domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleDomain {
    /// Any finite `f64` (only `NaN` and `±∞` are rejected).
    Real,
    /// Finite and `≥ 0.0` (power, energy, magnitude-squared samples).
    NonNegative,
}

/// Errors from [`SlidingMoments`] construction and updates.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum SlidingMomentsError {
    /// The compile-time window size `N` was `0`; a window must hold at least
    /// one sample.
    #[error("sliding window size must be at least 1, got {0}")]
    InvalidWindowSize(usize),
    /// A pushed value was `NaN` or `±∞`.
    #[error("sample {0} is not finite")]
    NonFiniteSample(f64),
    /// A pushed value was negative under [`SampleDomain::NonNegative`].
    #[error("sample {0} is negative but this estimator's domain is non-negative")]
    NegativeSample(f64),
    /// Exact recomputation still produced a non-finite or negative `M2`;
    /// see the module-level numerical-health policy.
    #[error("numerical integrity check failed after exact recomputation (M2 = {0})")]
    NumericalIntegrity(f64),
}

/// The result of one [`SlidingMoments::push`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlidingUpdate {
    /// Number of valid samples currently held (`1..=N`, since a successful
    /// push always holds at least the sample just pushed).
    pub len: usize,
    /// The current mean after this push.
    pub mean: f64,
    /// The current centered second moment `M2` after this push.
    pub m2: f64,
    /// The sample evicted by this push (the window was already full), or
    /// `None` during warm-up.
    pub evicted: Option<f64>,
    /// Whether this push triggered an exact recomputation (see the
    /// module-level numerical-health policy). Always `false` in ordinary
    /// operation; exposed for testing and diagnostics.
    pub recomputed: bool,
}

/// A small multiple of machine epsilon used to size the scale-aware
/// numerical-health tolerance (see the module docs). Chosen as generous
/// headroom over the `O(len · u)` accumulated-rounding argument, not as a
/// tight bound.
const TOLERANCE_ULPS: f64 = 64.0;

/// Backing storage for a [`SlidingMoments`]/[`SlidingMomentsDyn`] circular
/// buffer: a compile-time-sized array (stack, zero allocation) or a
/// runtime-sized boxed slice (one heap allocation at construction, none
/// after). [`Core`] implements the update recurrences and numerical-health
/// policy exactly once, over either.
trait MomentBuffer: Clone + std::fmt::Debug {
    fn as_slice(&self) -> &[f64];
    fn as_mut_slice(&mut self) -> &mut [f64];
}

impl<const N: usize> MomentBuffer for [f64; N] {
    fn as_slice(&self) -> &[f64] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [f64] {
        self
    }
}

impl MomentBuffer for Box<[f64]> {
    fn as_slice(&self) -> &[f64] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [f64] {
        self
    }
}

/// The shared implementation behind both [`SlidingMoments<N>`] and
/// [`SlidingMomentsDyn`] — see the module documentation for the
/// mathematical contract, the two update recurrences (warm-up vs. full
/// window), and the numerical-health policy. Generic only over where the
/// circular buffer's `N` slots live ([`MomentBuffer`]); every method below
/// is exactly the same code regardless.
#[derive(Debug, Clone)]
struct Core<B: MomentBuffer> {
    buffer: B,
    len: usize,
    /// Index of the next slot to write — during warm-up this is also `len`;
    /// once full it is also the index of the oldest remaining sample.
    head: usize,
    mean: f64,
    m2: f64,
    domain: SampleDomain,
}

impl<B: MomentBuffer> Core<B> {
    /// `buffer` must already be zeroed and sized to the intended capacity.
    /// Fails if that capacity is `0`.
    fn new(buffer: B, domain: SampleDomain) -> Result<Self, SlidingMomentsError> {
        if buffer.as_slice().is_empty()
        {
            return Err(SlidingMomentsError::InvalidWindowSize(0));
        }
        Ok(Self {
            buffer,
            len: 0,
            head: 0,
            mean: 0.0,
            m2: 0.0,
            domain,
        })
    }

    fn domain(&self) -> SampleDomain {
        self.domain
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn capacity(&self) -> usize {
        self.buffer.as_slice().len()
    }

    fn is_full(&self) -> bool {
        self.len == self.capacity()
    }

    fn as_slice(&self) -> &[f64] {
        &self.buffer.as_slice()[..self.len]
    }

    fn mean(&self) -> Option<f64> {
        (self.len > 0).then_some(self.mean)
    }

    fn m2(&self) -> Option<f64> {
        (self.len > 0).then_some(self.m2)
    }

    fn population_variance(&self) -> Option<f64> {
        // `.then(||...)` (lazy) rather than `.then_some(...)`: the arm must
        // not evaluate when `len == 0`, both to avoid a wasted 0.0/0.0 and on
        // general principle (see `sample_variance`, where the analogous eager
        // form underflows `len - 1` in `usize`).
        (self.len > 0).then(|| self.m2 / self.len as f64)
    }

    fn sample_variance(&self) -> Option<f64> {
        (self.len > 1).then(|| self.m2 / (self.len - 1) as f64)
    }

    fn validate(&self, value: f64) -> Result<(), SlidingMomentsError> {
        if !value.is_finite()
        {
            return Err(SlidingMomentsError::NonFiniteSample(value));
        }
        if self.domain == SampleDomain::NonNegative && value < 0.0
        {
            return Err(SlidingMomentsError::NegativeSample(value));
        }
        Ok(())
    }

    /// Push one sample. `O(1)`. Rejects non-finite values always, and
    /// negative values under [`SampleDomain::NonNegative`], *before* touching
    /// any state (a rejected push leaves `self` completely unchanged).
    fn push(&mut self, value: f64) -> Result<SlidingUpdate, SlidingMomentsError> {
        self.validate(value)?;
        let capacity = self.buffer.as_slice().len();

        let evicted = if self.len < capacity
        {
            // Warm-up: Welford's online recurrence (see module docs).
            let delta = value - self.mean;
            self.len += 1;
            self.mean += delta / self.len as f64;
            self.m2 += delta * (value - self.mean);
            None
        }
        else
        {
            // Full window: derived replacement recurrence (see module docs).
            let old = self.buffer.as_slice()[self.head];
            let mean_old = self.mean;
            self.mean += (value - old) / capacity as f64;
            self.m2 += (value - old) * (value + old - self.mean - mean_old);
            Some(old)
        };

        self.buffer.as_mut_slice()[self.head] = value;
        self.head = (self.head + 1) % capacity;

        let recomputed = self.repair_if_degraded()?;

        Ok(SlidingUpdate {
            len: self.len,
            mean: self.mean,
            m2: self.m2,
            evicted,
            recomputed,
        })
    }

    /// Force an exact recomputation of `mean` and `M2` from the retained
    /// circular buffer (two passes; see module docs). `O(len)`. Returns
    /// [`SlidingMomentsError::NumericalIntegrity`] if the recomputed state is
    /// still non-finite or `M2` is still negative (only reachable from
    /// sustained magnitudes near `f64::MAX`).
    fn recompute(&mut self) -> Result<(), SlidingMomentsError> {
        self.recompute_exact();
        if !self.mean.is_finite() || !self.m2.is_finite() || self.m2 < 0.0
        {
            return Err(SlidingMomentsError::NumericalIntegrity(self.m2));
        }
        Ok(())
    }

    /// Two-pass recomputation: mean via summation, then `M2` as a sum of
    /// individually-non-negative squared deviations (so it cannot go negative
    /// from cancellation the way `Σx² − N·μ²` can).
    fn recompute_exact(&mut self) {
        if self.len == 0
        {
            self.mean = 0.0;
            self.m2 = 0.0;
            return;
        }
        let data = &self.buffer.as_slice()[..self.len];
        let n = self.len as f64;
        let mean = data.iter().sum::<f64>() / n;
        let m2 = data.iter().map(|&x| (x - mean) * (x - mean)).sum();
        self.mean = mean;
        self.m2 = m2;
    }

    fn tolerance(&self) -> f64 {
        // Cap the magnitude used to build the scale *before* squaring: for
        // `|mean|` beyond `~1.3e154`, `mean * mean` itself overflows to
        // `+inf`, which would make the tolerance infinite and silently
        // disable the degradation check below (any finite M2 satisfies
        // `m2 >= -inf`). Capping keeps the tolerance large-but-finite instead,
        // so extreme-but-technically-finite input still gets a real check.
        let mean_abs = self.mean.abs().min(1.0e150);
        let scale = (mean_abs * mean_abs * self.len as f64).max(1.0);
        TOLERANCE_ULPS * f64::EPSILON * scale
    }

    /// Applies the numerical-health policy after an update; returns whether an
    /// exact recomputation was triggered.
    fn repair_if_degraded(&mut self) -> Result<bool, SlidingMomentsError> {
        if self.mean.is_finite() && self.m2.is_finite() && self.m2 >= -self.tolerance()
        {
            if self.m2 < 0.0
            {
                self.m2 = 0.0; // roundoff-level excursion: clamp, not "degraded".
            }
            return Ok(false);
        }
        self.recompute()?;
        Ok(true)
    }

    /// Discard all samples, returning to the just-constructed state.
    fn clear(&mut self) {
        self.buffer.as_mut_slice().fill(0.0);
        self.len = 0;
        self.head = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

/// Fixed-capacity `N`, O(1)-update online mean and centered second moment.
///
/// See the module documentation for the mathematical contract, the two update
/// recurrences (warm-up vs. full window), and the numerical-health policy.
/// Backed by a stack-allocated `[f64; N]` — zero heap allocation. For a
/// window size only known at runtime, see [`SlidingMomentsDyn`], which runs
/// the identical algorithm over a heap-allocated buffer instead.
#[derive(Debug, Clone)]
pub struct SlidingMoments<const N: usize> {
    core: Core<[f64; N]>,
}

impl<const N: usize> SlidingMoments<N> {
    /// A window accepting any finite `f64`. Fails if `N == 0`.
    pub fn new() -> Result<Self, SlidingMomentsError> {
        Self::with_domain(SampleDomain::Real)
    }

    /// A window restricted to finite, non-negative samples (power/energy
    /// domain). Fails if `N == 0`.
    pub fn new_non_negative() -> Result<Self, SlidingMomentsError> {
        Self::with_domain(SampleDomain::NonNegative)
    }

    fn with_domain(domain: SampleDomain) -> Result<Self, SlidingMomentsError> {
        Ok(Self {
            core: Core::new([0.0; N], domain)?,
        })
    }

    /// This estimator's input-domain policy.
    pub fn domain(&self) -> SampleDomain {
        self.core.domain()
    }

    /// Number of valid samples currently held, `0..=N`.
    pub fn len(&self) -> usize {
        self.core.len()
    }

    /// `true` if no sample has been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.core.is_empty()
    }

    /// The fixed window capacity `N`.
    pub fn capacity(&self) -> usize {
        N
    }

    /// `true` once `N` samples have been pushed (further pushes replace the
    /// oldest sample rather than growing the window).
    pub fn is_full(&self) -> bool {
        self.core.is_full()
    }

    /// The current window contents, in a fixed but otherwise unspecified
    /// deterministic order (physical circular-buffer order). `O(1)`, no
    /// allocation — used by the two-pass oracle in tests and by robust
    /// estimators that need the raw values rather than just the moments.
    pub fn as_slice(&self) -> &[f64] {
        self.core.as_slice()
    }

    /// The current mean, or `None` if empty.
    pub fn mean(&self) -> Option<f64> {
        self.core.mean()
    }

    /// The current centered second moment `M2 = Σ(xᵢ − μ)²`, or `None` if
    /// empty. `M2` for a single sample is `0.0` (zero deviation from itself).
    pub fn m2(&self) -> Option<f64> {
        self.core.m2()
    }

    /// Population variance `M2 / len` — the current window treated as the
    /// entire population. `None` if empty. `Some(0.0)` for a single sample.
    pub fn population_variance(&self) -> Option<f64> {
        self.core.population_variance()
    }

    /// Unbiased sample variance `M2 / (len − 1)`. `None` for fewer than two
    /// samples (undefined).
    pub fn sample_variance(&self) -> Option<f64> {
        self.core.sample_variance()
    }

    /// Push one sample. `O(1)`. Rejects non-finite values always, and
    /// negative values under [`SampleDomain::NonNegative`], *before* touching
    /// any state (a rejected push leaves `self` completely unchanged).
    pub fn push(&mut self, value: f64) -> Result<SlidingUpdate, SlidingMomentsError> {
        self.core.push(value)
    }

    /// Force an exact recomputation of `mean` and `M2` from the retained
    /// circular buffer (two passes; see module docs). `O(len)`. Returns
    /// [`SlidingMomentsError::NumericalIntegrity`] if the recomputed state is
    /// still non-finite or `M2` is still negative (only reachable from
    /// sustained magnitudes near `f64::MAX`).
    pub fn recompute(&mut self) -> Result<(), SlidingMomentsError> {
        self.core.recompute()
    }

    /// Discard all samples, returning to the just-constructed state.
    pub fn clear(&mut self) {
        self.core.clear()
    }
}

/// Runtime-capacity, O(1)-update online mean and centered second moment —
/// the identical algorithm and numerical-health policy as [`SlidingMoments`]
/// (see the module docs), for callers whose window size is only known at
/// runtime rather than at compile time (e.g. a radar reference-cell count
/// read from a configuration value, as
/// [`crate::radar::vi_cfar::CfarStreamDetector`] needs). Backed by a
/// `Box<[f64]>` allocated once at construction — [`SlidingMomentsDyn::push`]
/// itself never allocates.
///
/// Prefer [`SlidingMoments<N>`] when `N` is known at compile time: it avoids
/// the heap allocation entirely.
#[derive(Debug, Clone)]
pub struct SlidingMomentsDyn {
    core: Core<Box<[f64]>>,
}

impl SlidingMomentsDyn {
    /// A window of runtime `capacity`, accepting any finite `f64`. Fails if
    /// `capacity == 0`.
    pub fn new(capacity: usize) -> Result<Self, SlidingMomentsError> {
        Self::with_domain(capacity, SampleDomain::Real)
    }

    /// A window of runtime `capacity`, restricted to finite, non-negative
    /// samples (power/energy domain). Fails if `capacity == 0`.
    pub fn new_non_negative(capacity: usize) -> Result<Self, SlidingMomentsError> {
        Self::with_domain(capacity, SampleDomain::NonNegative)
    }

    fn with_domain(capacity: usize, domain: SampleDomain) -> Result<Self, SlidingMomentsError> {
        Ok(Self {
            core: Core::new(vec![0.0; capacity].into_boxed_slice(), domain)?,
        })
    }

    /// This estimator's input-domain policy.
    pub fn domain(&self) -> SampleDomain {
        self.core.domain()
    }

    /// Number of valid samples currently held, `0..=capacity()`.
    pub fn len(&self) -> usize {
        self.core.len()
    }

    /// `true` if no sample has been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.core.is_empty()
    }

    /// The window capacity fixed at construction.
    pub fn capacity(&self) -> usize {
        self.core.capacity()
    }

    /// `true` once `capacity()` samples have been pushed (further pushes
    /// replace the oldest sample rather than growing the window).
    pub fn is_full(&self) -> bool {
        self.core.is_full()
    }

    /// The current window contents, in a fixed but otherwise unspecified
    /// deterministic order (physical circular-buffer order). `O(1)`, no
    /// allocation.
    pub fn as_slice(&self) -> &[f64] {
        self.core.as_slice()
    }

    /// The current mean, or `None` if empty.
    pub fn mean(&self) -> Option<f64> {
        self.core.mean()
    }

    /// The current centered second moment `M2 = Σ(xᵢ − μ)²`, or `None` if
    /// empty. `M2` for a single sample is `0.0` (zero deviation from itself).
    pub fn m2(&self) -> Option<f64> {
        self.core.m2()
    }

    /// Population variance `M2 / len`. `None` if empty. `Some(0.0)` for a
    /// single sample.
    pub fn population_variance(&self) -> Option<f64> {
        self.core.population_variance()
    }

    /// Unbiased sample variance `M2 / (len − 1)`. `None` for fewer than two
    /// samples (undefined).
    pub fn sample_variance(&self) -> Option<f64> {
        self.core.sample_variance()
    }

    /// Push one sample. `O(1)`, no allocation. Rejects non-finite values
    /// always, and negative values under [`SampleDomain::NonNegative`],
    /// *before* touching any state (a rejected push leaves `self` completely
    /// unchanged).
    pub fn push(&mut self, value: f64) -> Result<SlidingUpdate, SlidingMomentsError> {
        self.core.push(value)
    }

    /// Force an exact recomputation of `mean` and `M2` from the retained
    /// buffer (two passes; see module docs). `O(len)`. Returns
    /// [`SlidingMomentsError::NumericalIntegrity`] if the recomputed state is
    /// still non-finite or `M2` is still negative (only reachable from
    /// sustained magnitudes near `f64::MAX`).
    pub fn recompute(&mut self) -> Result<(), SlidingMomentsError> {
        self.core.recompute()
    }

    /// Discard all samples, returning to the just-constructed state.
    pub fn clear(&mut self) {
        self.core.clear()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent O(len) reference: recompute mean/M2 directly from a slice,
    /// using pairwise (divide-and-conquer) summation for *both* passes so this
    /// oracle's own rounding error stays O(log len) rather than O(len) — a
    /// materially independent check on the O(1) recurrence under test, not
    /// just a copy of [`SlidingMoments::recompute_exact`] (which deliberately
    /// uses plain summation, adequate for production window sizes but not the
    /// strongest reference available for a test oracle).
    fn pairwise_sum(xs: &[f64]) -> f64 {
        match xs.len()
        {
            0 => 0.0,
            1 => xs[0],
            n =>
            {
                let mid = n / 2;
                pairwise_sum(&xs[..mid]) + pairwise_sum(&xs[mid..])
            },
        }
    }

    struct Oracle {
        mean: f64,
        m2: f64,
        population_variance: f64,
        sample_variance: Option<f64>,
    }

    fn oracle(data: &[f64]) -> Oracle {
        assert!(!data.is_empty(), "oracle is undefined on an empty window");
        let n = data.len() as f64;
        let mean = pairwise_sum(data) / n;
        let sq_dev: Vec<f64> = data.iter().map(|&x| (x - mean) * (x - mean)).collect();
        let m2 = pairwise_sum(&sq_dev);
        Oracle {
            mean,
            m2,
            population_variance: m2 / n,
            sample_variance: (data.len() > 1).then_some(m2 / (n - 1.0)),
        }
    }

    /// Absolute+relative tolerance for comparing the O(1) recurrence against
    /// the oracle: `1e-9` absolute (well above `f64` roundoff for the modest
    /// magnitudes used here) plus `1e-9` relative to the oracle's own
    /// magnitude, so the bound scales sensibly for the large-baseline test.
    fn assert_close(actual: f64, expected: f64, msg: &str) {
        let tol = 1e-9 + 1e-9 * expected.abs();
        assert!(
            (actual - expected).abs() <= tol,
            "{msg}: actual {actual}, expected {expected}, tol {tol}"
        );
    }

    /// Lets the oracle cross-check below drive either backing store through
    /// the identical assertions, since [`SlidingMoments<N>`] and
    /// [`SlidingMomentsDyn`] share no public trait of their own (only the
    /// private [`Core`] they both delegate to).
    trait MomentsLike {
        fn is_empty(&self) -> bool;
        fn as_slice(&self) -> &[f64];
        fn mean(&self) -> Option<f64>;
        fn m2(&self) -> Option<f64>;
        fn population_variance(&self) -> Option<f64>;
        fn sample_variance(&self) -> Option<f64>;
    }

    impl<const N: usize> MomentsLike for SlidingMoments<N> {
        fn is_empty(&self) -> bool {
            SlidingMoments::is_empty(self)
        }
        fn as_slice(&self) -> &[f64] {
            SlidingMoments::as_slice(self)
        }
        fn mean(&self) -> Option<f64> {
            SlidingMoments::mean(self)
        }
        fn m2(&self) -> Option<f64> {
            SlidingMoments::m2(self)
        }
        fn population_variance(&self) -> Option<f64> {
            SlidingMoments::population_variance(self)
        }
        fn sample_variance(&self) -> Option<f64> {
            SlidingMoments::sample_variance(self)
        }
    }

    impl MomentsLike for SlidingMomentsDyn {
        fn is_empty(&self) -> bool {
            SlidingMomentsDyn::is_empty(self)
        }
        fn as_slice(&self) -> &[f64] {
            SlidingMomentsDyn::as_slice(self)
        }
        fn mean(&self) -> Option<f64> {
            SlidingMomentsDyn::mean(self)
        }
        fn m2(&self) -> Option<f64> {
            SlidingMomentsDyn::m2(self)
        }
        fn population_variance(&self) -> Option<f64> {
            SlidingMomentsDyn::population_variance(self)
        }
        fn sample_variance(&self) -> Option<f64> {
            SlidingMomentsDyn::sample_variance(self)
        }
    }

    fn assert_matches_oracle(sm: &impl MomentsLike) {
        if sm.is_empty()
        {
            assert_eq!(sm.mean(), None);
            assert_eq!(sm.m2(), None);
            return;
        }
        let o = oracle(sm.as_slice());
        assert_close(sm.mean().unwrap(), o.mean, "mean");
        assert_close(sm.m2().unwrap(), o.m2, "m2");
        assert_close(
            sm.population_variance().unwrap(),
            o.population_variance,
            "population_variance",
        );
        match (sm.sample_variance(), o.sample_variance)
        {
            (Some(a), Some(e)) => assert_close(a, e, "sample_variance"),
            (None, None) =>
            {},
            (a, e) => panic!("sample_variance presence mismatch: {a:?} vs {e:?}"),
        }
    }

    #[test]
    fn rejects_zero_window_size() {
        assert_eq!(
            SlidingMoments::<0>::new().unwrap_err(),
            SlidingMomentsError::InvalidWindowSize(0)
        );
    }

    #[test]
    fn empty_estimator_reports_none() {
        let sm = SlidingMoments::<4>::new().unwrap();
        assert_eq!(sm.len(), 0);
        assert!(sm.is_empty());
        assert!(!sm.is_full());
        assert_eq!(sm.capacity(), 4);
        assert_eq!(sm.mean(), None);
        assert_eq!(sm.m2(), None);
        assert_eq!(sm.population_variance(), None);
        assert_eq!(sm.sample_variance(), None);
        assert_eq!(sm.as_slice(), &[] as &[f64]);
    }

    #[test]
    fn one_sample() {
        let mut sm = SlidingMoments::<5>::new().unwrap();
        let u = sm.push(3.0).unwrap();
        assert_eq!(u.len, 1);
        assert_eq!(u.evicted, None);
        assert_eq!(sm.mean(), Some(3.0));
        assert_eq!(sm.m2(), Some(0.0));
        assert_eq!(sm.population_variance(), Some(0.0));
        assert_eq!(sm.sample_variance(), None); // undefined for n < 2
        assert_matches_oracle(&sm);
    }

    #[test]
    fn warm_up_from_one_sample_through_n() {
        const N: usize = 6;
        let mut sm = SlidingMoments::<N>::new().unwrap();
        for (i, &x) in [4.0, 1.0, 7.0, 2.0, 9.0, 3.0].iter().enumerate()
        {
            let u = sm.push(x).unwrap();
            assert_eq!(u.len, i + 1);
            assert_eq!(u.evicted, None, "no eviction before the window is full");
            assert!(!sm.is_full() || i + 1 == N);
            assert_matches_oracle(&sm);
        }
        assert!(sm.is_full());
    }

    #[test]
    fn numerical_integrity_error_on_sustained_extreme_magnitude() {
        // Push one sample near `f64::MAX`, then a modest one: pulling the
        // running mean from `~huge` down to `~huge/2` makes the *second*
        // push's Welford term `delta * (value − mean)` a product of two
        // ~huge-magnitude quantities (`huge² ≈ 2.6e616`), which overflows to
        // `+inf` and poisons `m2`. `repair_if_degraded` detects the
        // non-finite state and calls `recompute_exact`, whose own summation
        // over the (still ~huge-magnitude) buffer also overflows — the exact
        // recomputation genuinely cannot recover a finite state here, so
        // `push` must report that rather than return a silently wrong
        // statistic. (Verified empirically: overflow already occurs on the
        // second push, one earlier than a first-pass hand analysis of only
        // the replacement recurrence would suggest — the warm-up recurrence
        // is not immune either once the mean has to move by ~huge.)
        let huge = 0.9 * f64::MAX;
        let mut sm = SlidingMoments::<2>::new().unwrap();
        sm.push(huge).unwrap(); // warm-up, len 1, exact: mean = huge, m2 = 0.
        let err = sm.push(1.0).unwrap_err();
        assert!(matches!(err, SlidingMomentsError::NumericalIntegrity(_)));
    }

    #[test]
    fn recompute_repairs_a_genuine_soft_tolerance_breach() {
        // `recompute()` is exposed as a public escape hatch independent of
        // automatic triggering: even starting from a deliberately corrupted
        // in-memory state (not reachable through `push` alone), it must
        // rebuild an exact, non-negative M2 from the retained buffer.
        let mut sm = SlidingMoments::<4>::new().unwrap();
        for x in [3.0, 5.0, 2.0, 9.0]
        {
            sm.push(x).unwrap();
        }
        let good_m2 = sm.m2().unwrap();
        sm.core.m2 = -1.0; // simulate a corrupted running statistic
        sm.recompute().unwrap();
        assert_close(
            sm.m2().unwrap(),
            good_m2,
            "m2 after repairing a corrupted state",
        );
        assert!(sm.m2().unwrap() >= 0.0);
    }

    #[test]
    fn constant_sequence_has_zero_variance() {
        let mut sm = SlidingMoments::<10>::new().unwrap();
        for _ in 0..25
        {
            sm.push(7.0).unwrap();
            assert_eq!(sm.mean(), Some(7.0));
            assert_eq!(sm.population_variance(), Some(0.0));
            assert_matches_oracle(&sm);
        }
    }

    #[test]
    fn alternating_sequence() {
        let mut sm = SlidingMoments::<8>::new().unwrap();
        for i in 0..40
        {
            let x = if i % 2 == 0 { 1.0 } else { 5.0 };
            sm.push(x).unwrap();
            assert_matches_oracle(&sm);
        }
        // Full window of alternating {1,5}: mean 3, population variance 4.
        assert_close(sm.mean().unwrap(), 3.0, "alternating mean");
        assert_close(
            sm.population_variance().unwrap(),
            4.0,
            "alternating pop var",
        );
    }

    #[test]
    fn monotonic_ramp() {
        let mut sm = SlidingMoments::<20>::new().unwrap();
        for i in 0..100
        {
            sm.push(i as f64).unwrap();
            assert_matches_oracle(&sm);
        }
    }

    #[test]
    fn negative_input_rejected_in_non_negative_mode() {
        let mut sm = SlidingMoments::<4>::new_non_negative().unwrap();
        assert_eq!(sm.domain(), SampleDomain::NonNegative);
        assert_eq!(
            sm.push(-0.001),
            Err(SlidingMomentsError::NegativeSample(-0.001))
        );
        assert_eq!(sm.len(), 0, "a rejected push must not mutate state");
        assert!(sm.push(0.0).is_ok(), "zero is not negative");
    }

    #[test]
    fn negative_input_accepted_in_real_mode() {
        let mut sm = SlidingMoments::<4>::new().unwrap();
        assert!(sm.push(-3.5).is_ok());
        assert_eq!(sm.mean(), Some(-3.5));
    }

    #[test]
    fn nan_is_always_rejected() {
        let mut sm = SlidingMoments::<4>::new().unwrap();
        let err = sm.push(f64::NAN).unwrap_err();
        assert!(matches!(err, SlidingMomentsError::NonFiniteSample(x) if x.is_nan()));
        assert_eq!(sm.len(), 0);
    }

    #[test]
    fn positive_infinity_is_always_rejected() {
        let mut sm = SlidingMoments::<4>::new().unwrap();
        assert_eq!(
            sm.push(f64::INFINITY),
            Err(SlidingMomentsError::NonFiniteSample(f64::INFINITY))
        );
        assert_eq!(sm.len(), 0);
    }

    #[test]
    fn negative_infinity_is_always_rejected() {
        let mut sm = SlidingMoments::<4>::new().unwrap();
        assert_eq!(
            sm.push(f64::NEG_INFINITY),
            Err(SlidingMomentsError::NonFiniteSample(f64::NEG_INFINITY))
        );
        assert_eq!(sm.len(), 0);
    }

    #[test]
    fn huge_baseline_tiny_variation_stays_accurate() {
        // A large common baseline (1e9) with a tiny oscillation (~1e-3) on
        // top is the classic stress case for *any* mean-centered variance
        // recurrence, not just a naive Σx² − Nμ² formula: `M2`'s update term
        // `value + old − mean − mean_old` is built from differences of
        // ~1e9-magnitude quantities, so its absolute rounding error scales
        // with `mean² · EPSILON` in the worst case — an intrinsic limitation
        // of this family of algorithms (see the module docs' numerical-health
        // section), not a bug to "fix" away here.
        //
        // In practice the centering keeps the residual error far below that
        // crude worst case, because `value`/`old` stay close to `mean`, so
        // most of the ~1e9 magnitude cancels before rounding — this test
        // checks that empirically-grounded, still-meaningful bound (1% of
        // the oracle's own M2, plus a tiny absolute floor for near-zero
        // cases) rather than either pretending 1e-9-level precision holds at
        // this scale, or accepting an arbitrarily large error.
        const N: usize = 16;
        let mut sm = SlidingMoments::<N>::new().unwrap();
        for i in 0..200
        {
            let x = 1.0e9 + (i % 5) as f64 * 1.0e-3;
            sm.push(x).unwrap();

            let o = oracle(sm.as_slice());
            // The mean is a single division, not a squared/cancelled
            // quantity: it stays accurate to the default tight tolerance.
            assert_close(sm.mean().unwrap(), o.mean, "mean at huge baseline");

            let m2_tol = 1.0e-2 * o.m2.abs() + 1.0e-9;
            let m2 = sm.m2().unwrap();
            assert!(
                (m2 - o.m2).abs() <= m2_tol,
                "m2 at huge baseline: actual {m2}, expected {}, tol {m2_tol}",
                o.m2
            );
        }
    }

    #[test]
    fn values_close_to_zero() {
        let mut sm = SlidingMoments::<12>::new().unwrap();
        for i in 0..60
        {
            let x = (i as f64 - 30.0) * 1.0e-300;
            sm.push(x).unwrap();
            assert_matches_oracle(&sm);
        }
    }

    #[test]
    fn replacement_of_the_oldest_sample() {
        let mut sm = SlidingMoments::<3>::new().unwrap();
        sm.push(1.0).unwrap();
        sm.push(2.0).unwrap();
        sm.push(3.0).unwrap();
        assert!(sm.is_full());
        let u = sm.push(10.0).unwrap();
        assert_eq!(
            u.evicted,
            Some(1.0),
            "the oldest sample (1.0) must be evicted"
        );
        // Window is now {2, 3, 10}.
        assert_close(sm.mean().unwrap(), 5.0, "mean after replacement");
        assert_matches_oracle(&sm);
    }

    #[test]
    fn wraparound_of_the_circular_index() {
        const N: usize = 4;
        let mut sm = SlidingMoments::<N>::new().unwrap();
        // Push through several full wraps of the circular index and check
        // against the oracle after every single push, not just at the end.
        for i in 0..(4 * N + 3)
        {
            sm.push((i as f64).sin() * 100.0).unwrap();
            assert_matches_oracle(&sm);
        }
    }

    #[test]
    fn clear_returns_to_the_constructed_state() {
        let mut sm = SlidingMoments::<5>::new().unwrap();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        {
            sm.push(x).unwrap();
        }
        assert!(sm.is_full());
        sm.clear();
        assert_eq!(sm.len(), 0);
        assert!(!sm.is_full());
        assert_eq!(sm.mean(), None);
        assert_eq!(sm.as_slice(), &[] as &[f64]);
        // Fully usable afterwards.
        sm.push(9.0).unwrap();
        assert_eq!(sm.mean(), Some(9.0));
    }

    #[test]
    fn exact_recomputation_matches_the_incremental_state() {
        let mut sm = SlidingMoments::<9>::new().unwrap();
        for i in 0..30
        {
            sm.push((i as f64 * 1.7).cos() * 50.0 + 10.0).unwrap();
        }
        let (mean_before, m2_before) = (sm.mean().unwrap(), sm.m2().unwrap());
        sm.recompute().unwrap();
        assert_close(sm.mean().unwrap(), mean_before, "recompute mean");
        assert_close(sm.m2().unwrap(), m2_before, "recompute m2");
        assert_matches_oracle(&sm);
    }

    #[test]
    fn long_deterministic_stream_matches_oracle_after_every_update() {
        // A small deterministic LCG (no OS/clock entropy) driving 5,000
        // pushes through an awkward, non-power-of-two window size.
        const N: usize = 17;
        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> f64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 200.0 - 100.0
            }
        }
        let mut rng = Lcg(0xC0FFEE);
        let mut sm = SlidingMoments::<N>::new().unwrap();
        for _ in 0..5000
        {
            sm.push(rng.next()).unwrap();
            assert_matches_oracle(&sm);
        }
    }

    /// Several awkward (non-power-of-two) compile-time window sizes, each
    /// checked against the oracle throughout warm-up, full occupancy and
    /// several full wraps.
    #[test]
    fn awkward_window_sizes() {
        fn drive<const N: usize>() {
            let mut sm = SlidingMoments::<N>::new().unwrap();
            for i in 0..(6 * N + 5)
            {
                sm.push(((i * 37 + 11) % 97) as f64).unwrap();
                assert_matches_oracle(&sm);
            }
        }
        drive::<1>();
        drive::<3>();
        drive::<7>();
        drive::<13>();
        drive::<31>();
    }

    // ---- SlidingMomentsDyn --------------------------------------------------
    //
    // `SlidingMomentsDyn` shares `Core<B>` with `SlidingMoments<N>` (see the
    // module docs, "Compile-time vs. runtime capacity"), so these are a
    // lighter cross-check of the storage abstraction itself, not a second
    // full re-verification of the algorithm — that already happened above.

    #[test]
    fn dyn_rejects_zero_capacity() {
        assert_eq!(
            SlidingMomentsDyn::new(0).unwrap_err(),
            SlidingMomentsError::InvalidWindowSize(0)
        );
    }

    #[test]
    fn dyn_matches_oracle_through_warmup_and_wraparound() {
        let mut sm = SlidingMomentsDyn::new(7).unwrap();
        for i in 0..40
        {
            sm.push(((i * 31 + 5) % 89) as f64).unwrap();
            assert_matches_oracle(&sm);
        }
        assert!(sm.is_full());
        assert_eq!(sm.capacity(), 7);
    }

    #[test]
    fn dyn_numerical_integrity_error_on_sustained_extreme_magnitude() {
        // Same construction as the const-generic version's equivalent test.
        let huge = 0.9 * f64::MAX;
        let mut sm = SlidingMomentsDyn::new(2).unwrap();
        sm.push(huge).unwrap();
        let err = sm.push(1.0).unwrap_err();
        assert!(matches!(err, SlidingMomentsError::NumericalIntegrity(_)));
    }

    #[test]
    fn dyn_recompute_repairs_a_genuine_soft_tolerance_breach() {
        let mut sm = SlidingMomentsDyn::new(4).unwrap();
        for x in [3.0, 5.0, 2.0, 9.0]
        {
            sm.push(x).unwrap();
        }
        let good_m2 = sm.m2().unwrap();
        sm.core.m2 = -1.0;
        sm.recompute().unwrap();
        assert_close(
            sm.m2().unwrap(),
            good_m2,
            "m2 after repairing a corrupted state",
        );
        assert!(sm.m2().unwrap() >= 0.0);
    }

    #[test]
    fn dyn_negative_input_rejected_in_non_negative_mode() {
        let mut sm = SlidingMomentsDyn::new_non_negative(4).unwrap();
        assert_eq!(sm.domain(), SampleDomain::NonNegative);
        assert_eq!(
            sm.push(-0.001),
            Err(SlidingMomentsError::NegativeSample(-0.001))
        );
        assert_eq!(sm.len(), 0, "a rejected push must not mutate state");
        assert!(sm.push(0.0).is_ok(), "zero is not negative");
    }

    #[test]
    fn dyn_nan_is_always_rejected() {
        let mut sm = SlidingMomentsDyn::new(4).unwrap();
        let err = sm.push(f64::NAN).unwrap_err();
        assert!(matches!(err, SlidingMomentsError::NonFiniteSample(x) if x.is_nan()));
        assert_eq!(sm.len(), 0);
    }

    #[test]
    fn dyn_clear_returns_to_the_constructed_state() {
        let mut sm = SlidingMomentsDyn::new(5).unwrap();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        {
            sm.push(x).unwrap();
        }
        assert!(sm.is_full());
        sm.clear();
        assert_eq!(sm.len(), 0);
        assert!(!sm.is_full());
        assert_eq!(sm.mean(), None);
        assert_eq!(sm.as_slice(), &[] as &[f64]);
        sm.push(9.0).unwrap();
        assert_eq!(sm.mean(), Some(9.0));
    }

    #[test]
    fn dyn_matches_const_generic_bit_for_bit_on_the_same_stream() {
        // Same algorithm, different backing store (`Box<[f64]>` vs
        // `[f64; N]`) — drive both through an identical stream and check
        // every observable output agrees exactly, not just approximately:
        // since both funnel through the same `Core<B>` logic, any divergence
        // here would mean the storage abstraction itself broke something.
        const N: usize = 11;
        let mut fixed = SlidingMoments::<N>::new().unwrap();
        let mut dynamic = SlidingMomentsDyn::new(N).unwrap();
        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> f64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 50.0 - 25.0
            }
        }
        let mut rng = Lcg(0xD1CE);
        for _ in 0..2000
        {
            let x = rng.next();
            let uf = fixed.push(x).unwrap();
            let ud = dynamic.push(x).unwrap();
            assert_eq!(uf.len, ud.len);
            assert_eq!(uf.mean.to_bits(), ud.mean.to_bits());
            assert_eq!(uf.m2.to_bits(), ud.m2.to_bits());
            assert_eq!(uf.evicted, ud.evicted);
            assert_eq!(uf.recomputed, ud.recomputed);
        }
    }
}
