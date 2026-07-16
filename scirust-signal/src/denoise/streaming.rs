//! Streaming (sample-by-sample) denoisers for real-time and embedded use.
//!
//! Every other family in [`super`] is *batch*: it sees the whole record
//! (`&[f64] → Vec<f64>`), so it can center its window on each sample, mirror the
//! borders, or even run backward passes (RTS smoothing). None of that is available
//! on a live sample stream — an edge device receives one sample per tick and must
//! answer before the next one arrives. This module provides **causal, stateful
//! counterparts** of the batch denoisers behind one small interface,
//! [`StreamingDenoiser`]: `push` one sample in, get one output sample back, with
//! `O(window)` memory and fully deterministic arithmetic.
//!
//! Causality has exactly one price: **delay**. A filter whose batch counterpart
//! looks `d` samples into the future must hold its answer back by `d` samples
//! (reported by [`StreamingDenoiser::delay`]); an estimator that never looks ahead
//! (EMA, Kalman) reports zero delay. On the interior of a long signal the window
//! filters here are *bit-for-bit identical* to their batch counterparts — only the
//! borders differ, because a stream has no future samples to mirror. Parameter
//! guards also mirror the batch conventions: a configuration that a batch function
//! would answer with `signal.to_vec()` (window ≤ 1, α out of range, non-positive
//! variances) degrades to the identity filter instead of panicking.
//!
//! | Streaming | Batch counterpart | `delay()` |
//! |-----------|-------------------|-----------|
//! | [`StreamingMovingAverage`] | [`super::linear::moving_average`] | `window / 2` |
//! | [`StreamingMedian`] | [`super::rank::median_filter`] | `half_window` |
//! | [`StreamingHampel`] | [`super::rank::hampel_filter`] | `half_window` |
//! | [`StreamingEma`] | [`super::linear::exp_moving_average`] (exact) | 0 |
//! | [`StreamingKalman`] | forward pass of [`super::adaptive::kalman_smooth`] | 0 |
//! | [`StreamingVst`]`<D>` | [`super::vst::vst_denoise`] around a streaming `D` | `D::delay()` |
//!
//! [`StreamingVst`] is the causal counterpart of the variance-stabilizing pipeline
//! ([`super::vst`]): it wraps *any* of the streaming denoisers above, running it in
//! the transformed domain between a pointwise forward transform and a
//! bias-corrected inverse — so signal-dependent (Poisson / multiplicative) noise
//! can be removed on a live stream, on an edge device, with `O(D + W)` memory.
//!
//! ```
//! use scirust_signal::denoise::streaming::{StreamingDenoiser, StreamingMedian};
//!
//! let mut f = StreamingMedian::new(1); // window 3, delay 1
//! let out: Vec<f64> = [0.0, 9.0, 0.1, 0.2].iter().map(|&x| f.push(x)).collect();
//! // After the 1-sample warm-up, out[i] estimates x[i − 1]: the spike pushed at
//! // index 1 is judged against its full window at push 2 — and removed.
//! assert_eq!(f.delay(), 1);
//! assert_eq!(out[2], 0.1); // median of [0.0, 9.0, 0.1]
//! ```

use std::collections::VecDeque;

use super::mad;
use super::vst::{
    VstKind, forward_scalar, inverse_corrected_pointwise_scalar, inverse_naive_scalar,
};

/// A causal, stateful, sample-by-sample denoiser.
///
/// # The delay contract
///
/// Let `out[i]` be the value returned by the `i`-th call to [`push`] (0-based). An
/// implementation with `delay() = d` guarantees that for every `i ≥ d`, `out[i]`
/// is its estimate of the sample pushed at index `i − d` — the filter answers `d`
/// samples late, because a causal implementation of a centered window must wait for
/// the `d` future samples the batch version could simply read. During the warm-up
/// (`i < d`), and until its window fills, the filter returns its best
/// *partial-window* estimate: always a finite number, never NaN, but not covered by
/// the batch-equivalence guarantee. For the window filters here the window is full
/// from `i = 2·d` on, and from that point `out[i]` equals the batch counterpart's
/// output at `i − d` exactly, wherever the batch filter's window did not touch a
/// mirrored border.
///
/// [`push`]: StreamingDenoiser::push
pub trait StreamingDenoiser {
    /// Feed one input sample; returns one (possibly delayed) output sample.
    fn push(&mut self, x: f64) -> f64;
    /// Return to the exact just-constructed state, discarding all buffered samples.
    fn reset(&mut self);
    /// Group delay in samples of the causal implementation (see the trait docs).
    fn delay(&self) -> usize {
        0
    }
}

/// Forward `StreamingDenoiser` to the struct's inherent methods, so callers can use
/// either the concrete type (no trait import needed) or a `dyn StreamingDenoiser`.
macro_rules! impl_streaming_denoiser {
    ($ty:ident) => {
        impl StreamingDenoiser for $ty {
            fn push(&mut self, x: f64) -> f64 {
                $ty::push(self, x)
            }
            fn reset(&mut self) {
                $ty::reset(self);
            }
            fn delay(&self) -> usize {
                $ty::delay(self)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Sorted-window helpers shared by the rank filters.
// ---------------------------------------------------------------------------

/// Insert `x` into an ascending `sorted` vector at its position under
/// [`f64::total_cmp`] — a *total* order (unlike `partial_cmp`, which leaves NaN
/// incomparable). A total order is what keeps insert and [`sorted_remove`] in
/// agreement: a NaN gets one fixed slot, so a later removal finds and deletes that
/// exact sample instead of an unrelated one.
pub(crate) fn sorted_insert(sorted: &mut Vec<f64>, x: f64) {
    let idx = match sorted.binary_search_by(|p| p.total_cmp(&x))
    {
        Ok(i) | Err(i) => i,
    };
    sorted.insert(idx, x);
}

/// Remove one occurrence of `x` from an ascending `sorted` vector, matching the
/// total order of [`sorted_insert`]. Under `total_cmp` a hit is bit-identical to
/// `x`, so even with a NaN resident in the window the search never deletes the wrong
/// element (the bug a `partial_cmp`-based search has: every probe against a NaN
/// compares "equal", so it removes a live sample and strands the NaN, silently
/// shrinking the window). A bit-exact scan backstops the impossible miss.
pub(crate) fn sorted_remove(sorted: &mut Vec<f64>, x: f64) {
    if let Ok(i) = sorted.binary_search_by(|p| p.total_cmp(&x))
    {
        sorted.remove(i);
        return;
    }
    if let Some(i) = sorted.iter().position(|v| v.to_bits() == x.to_bits())
    {
        sorted.remove(i);
    }
}

/// Median of an already-sorted window — same odd/even convention as
/// [`super::median`] (middle element, or the mean of the two middles).
pub(crate) fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0
    {
        return 0.0;
    }
    if n % 2 == 1
    {
        sorted[n / 2]
    }
    else
    {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

// ---------------------------------------------------------------------------
// Moving average.
// ---------------------------------------------------------------------------

/// Causal streaming moving-average (boxcar) filter — the streaming counterpart of
/// [`super::linear::moving_average`] (the classic FIR smoother; see e.g. Smith,
/// *The Scientist and Engineer's Guide to DSP*, ch. 15).
///
/// A ring buffer holds the last `window` samples (rounded up to the nearest odd
/// width, exactly like the batch filter) and each push returns their mean, summed
/// oldest-to-newest in the same order as the batch loop so that, once the window is
/// full, `out[i]` is **bit-for-bit** `moving_average(x, window)[i − window/2]`
/// wherever the batch filter's window did not hit a mirrored border. During
/// warm-up the mean of the partial window is returned. `window ≤ 1` degrades to
/// the identity, mirroring the batch guard.
#[derive(Debug, Clone)]
pub struct StreamingMovingAverage {
    half: usize,
    width: usize,
    buf: VecDeque<f64>,
}

impl StreamingMovingAverage {
    /// `window` is the full width; it is rounded up to the nearest odd number so
    /// the (delayed) filter is symmetric, exactly like the batch counterpart.
    pub fn new(window: usize) -> Self {
        let half = if window <= 1 { 0 } else { window / 2 };
        let width = 2 * half + 1;
        Self {
            half,
            width,
            buf: VecDeque::with_capacity(width),
        }
    }

    /// Feed one sample; see [`StreamingDenoiser::push`].
    pub fn push(&mut self, x: f64) -> f64 {
        if self.buf.len() == self.width
        {
            self.buf.pop_front();
        }
        self.buf.push_back(x);
        // Sum oldest → newest: the same accumulation order as the batch filter, so
        // full-window outputs are bitwise identical to `moving_average`.
        let sum: f64 = self.buf.iter().sum();
        sum / self.buf.len() as f64
    }

    /// Return to the just-constructed state.
    pub fn reset(&mut self) {
        self.buf.clear();
    }

    /// Group delay: half the (odd-rounded) window.
    pub fn delay(&self) -> usize {
        self.half
    }
}

impl_streaming_denoiser!(StreamingMovingAverage);

// ---------------------------------------------------------------------------
// Median.
// ---------------------------------------------------------------------------

/// Causal streaming median filter — the streaming counterpart of
/// [`super::rank::median_filter`] (the running median of Tukey, *Exploratory Data
/// Analysis*, 1977; the canonical impulse remover).
///
/// A ring buffer holds the last `2·half_window + 1` samples and a parallel sorted
/// `Vec` is maintained by binary-search insert/remove. Each update is `O(w)`
/// because of the element shifts inside the sorted vector, which is entirely fine —
/// and cache-friendly — at the window sizes rank filters use (a few to a few dozen
/// samples); the heap-based `O(log w)` structures only pay off at far larger
/// windows. Once the window is full, `out[i]` equals
/// `median_filter(x, half_window)[i − half_window]` on the batch filter's interior
/// (no mirrored border). During warm-up the median of the partial window is
/// returned. `half_window = 0` degrades to the identity, like the batch guard.
#[derive(Debug, Clone)]
pub struct StreamingMedian {
    half: usize,
    width: usize,
    buf: VecDeque<f64>,
    sorted: Vec<f64>,
}

impl StreamingMedian {
    /// Half-width `h`; the full window is `2·h + 1` samples and `delay()` is `h`.
    pub fn new(half_window: usize) -> Self {
        let width = 2 * half_window + 1;
        Self {
            half: half_window,
            width,
            buf: VecDeque::with_capacity(width),
            sorted: Vec::with_capacity(width),
        }
    }

    /// Feed one sample; see [`StreamingDenoiser::push`].
    pub fn push(&mut self, x: f64) -> f64 {
        if self.buf.len() == self.width
        {
            if let Some(old) = self.buf.pop_front()
            {
                sorted_remove(&mut self.sorted, old);
            }
        }
        self.buf.push_back(x);
        sorted_insert(&mut self.sorted, x);
        median_of_sorted(&self.sorted)
    }

    /// Return to the just-constructed state.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.sorted.clear();
    }

    /// Group delay: the half-window `h`.
    pub fn delay(&self) -> usize {
        self.half
    }
}

impl_streaming_denoiser!(StreamingMedian);

// ---------------------------------------------------------------------------
// Hampel.
// ---------------------------------------------------------------------------

/// Causal streaming Hampel filter — the streaming counterpart of
/// [`super::rank::hampel_filter`] (the Hampel identifier: Hampel 1974; Pearson et
/// al., *EURASIP J. Adv. Signal Process.* 2016).
///
/// Same buffering as [`StreamingMedian`] (ring buffer plus sorted `Vec`, `O(w)` per
/// sample). The *decision* is applied to the **delayed center sample** — the one
/// pushed `half_window` steps earlier, which sits at the center of the full window:
/// if it deviates from the window median by more than
/// `n_sigma · 1.4826 · MAD`, the median is returned in its place, otherwise the
/// sample passes verbatim. A zero MAD (locally constant window) disables the
/// correction, exactly like the batch guard, so constants are never "corrected".
/// Once the window is full, `out[i]` equals
/// `hampel_filter(x, half_window, n_sigma)[i − half_window]` on the batch interior.
/// During warm-up the same rule is applied to the oldest buffered sample against
/// the partial window. `half_window = 0` degrades to the identity.
#[derive(Debug, Clone)]
pub struct StreamingHampel {
    half: usize,
    width: usize,
    n_sigma: f64,
    buf: VecDeque<f64>,
    sorted: Vec<f64>,
}

impl StreamingHampel {
    /// Half-width `h` (full window `2·h + 1`, `delay()` = `h`) and the outlier
    /// threshold `n_sigma` in robust standard deviations (`1.4826 · MAD`).
    pub fn new(half_window: usize, n_sigma: f64) -> Self {
        let width = 2 * half_window + 1;
        Self {
            half: half_window,
            width,
            n_sigma,
            buf: VecDeque::with_capacity(width),
            sorted: Vec::with_capacity(width),
        }
    }

    /// Feed one sample; see [`StreamingDenoiser::push`].
    pub fn push(&mut self, x: f64) -> f64 {
        if self.half == 0
        {
            // `hampel_filter(_, 0, _)` is the identity; so are we.
            return x;
        }
        if self.buf.len() == self.width
        {
            if let Some(old) = self.buf.pop_front()
            {
                sorted_remove(&mut self.sorted, old);
            }
        }
        self.buf.push_back(x);
        sorted_insert(&mut self.sorted, x);
        // The candidate under test: the sample pushed `delay()` steps ago — the
        // window center once the buffer is full, the oldest sample during warm-up.
        let pos = (self.buf.len() - 1).saturating_sub(self.half);
        let center = self.buf[pos];
        let med = median_of_sorted(&self.sorted);
        let scale = 1.4826 * mad(&self.sorted);
        if scale > 0.0 && (center - med).abs() > self.n_sigma * scale
        {
            med
        }
        else
        {
            center
        }
    }

    /// Return to the just-constructed state.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.sorted.clear();
    }

    /// Group delay: the half-window `h` (0 in the identity configuration).
    pub fn delay(&self) -> usize {
        self.half
    }
}

impl_streaming_denoiser!(StreamingHampel);

// ---------------------------------------------------------------------------
// Exponential moving average.
// ---------------------------------------------------------------------------

/// Streaming first-order exponential moving average (exponential smoothing, Brown
/// 1956; a single-pole IIR low-pass) — the streaming counterpart of
/// [`super::linear::exp_moving_average`].
///
/// The batch EMA is already causal with `O(1)` state, so this is the one filter
/// with **no compromise at all**: `push` replicates the batch initialization
/// (`out[0] = x[0]`) and recursion (`out[i] = α·x[i] + (1 − α)·out[i−1]`)
/// operation-for-operation, so the streamed outputs equal
/// `exp_moving_average(x, alpha)` **exactly, sample for sample**, and `delay()` is
/// 0. (Zero *reporting* delay does not mean zero phase lag — like every causal
/// low-pass the EMA responds late by roughly `(1 − α)/α` samples; the batch filter
/// has the identical lag.) An out-of-range `alpha` (outside `(0, 1]`) degrades to
/// the identity, mirroring the batch guard.
#[derive(Debug, Clone)]
pub struct StreamingEma {
    alpha: f64,
    prev: Option<f64>,
}

impl StreamingEma {
    /// Smoothing factor `alpha` in `(0, 1]`; smaller means heavier smoothing.
    pub fn new(alpha: f64) -> Self {
        Self { alpha, prev: None }
    }

    /// Feed one sample; see [`StreamingDenoiser::push`].
    pub fn push(&mut self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&self.alpha) || self.alpha == 0.0
        {
            // Same out-of-range guard as `exp_moving_average`: identity.
            return x;
        }
        let y = match self.prev
        {
            None => x,
            Some(p) => self.alpha * x + (1.0 - self.alpha) * p,
        };
        self.prev = Some(y);
        y
    }

    /// Return to the just-constructed state.
    pub fn reset(&mut self) {
        self.prev = None;
    }

    /// Group delay of the reporting convention: 0 (see the type docs on phase lag).
    pub fn delay(&self) -> usize {
        0
    }
}

impl_streaming_denoiser!(StreamingEma);

// ---------------------------------------------------------------------------
// Kalman.
// ---------------------------------------------------------------------------

/// Causal local-level **Kalman filter** (Kalman 1960) — the forward-only streaming
/// counterpart of [`super::adaptive::kalman_smooth`].
///
/// State model: `x_k = x_{k−1} + w_k` (`w ~ N(0, process_var)`), observation
/// `y_k = x_k + v_k` (`v ~ N(0, meas_var)`). The batch smoother follows the
/// forward filter with a Rauch-Tung-Striebel (1965) *backward* pass that
/// re-estimates every state from the whole record; that pass needs the future and
/// therefore **cannot exist in a streaming filter**. What remains is still the
/// optimal *causal* linear estimator: compared to the RTS-smoothed batch output it
/// lags the signal by roughly `1/k` samples (with `k` the steady-state gain) and
/// retains about twice the residual variance — the irreducible price of causality.
/// The recursion and its semi-diffuse initialization (`x_pred` = first sample,
/// `p_pred = meas_var + process_var`) replicate the batch forward pass
/// operation-for-operation. `process_var` sets agility: small values give heavy
/// smoothing, large values track fast changes. Non-positive variances degrade to
/// the identity, mirroring the batch guard; `delay()` is 0.
#[derive(Debug, Clone)]
pub struct StreamingKalman {
    q: f64,
    r: f64,
    /// `(filtered state, filtered variance)` after the last push.
    state: Option<(f64, f64)>,
}

impl StreamingKalman {
    /// Process (state random-walk) variance `process_var` and measurement-noise
    /// variance `meas_var` — the same `q` and `r` as `kalman_smooth`.
    pub fn new(process_var: f64, meas_var: f64) -> Self {
        Self {
            q: process_var,
            r: meas_var,
            state: None,
        }
    }

    /// Feed one sample; see [`StreamingDenoiser::push`].
    pub fn push(&mut self, x: f64) -> f64 {
        if self.q <= 0.0 || self.r <= 0.0
        {
            // Same parameter guard as `kalman_smooth`: identity.
            return x;
        }
        let (x_pred, p_pred) = match self.state
        {
            // Semi-diffuse start: trust the first sample, generous uncertainty.
            None => (x, self.r + self.q),
            Some((xf, pf)) => (xf, pf + self.q),
        };
        let innov = x - x_pred;
        let k = p_pred / (p_pred + self.r);
        let xf = x_pred + k * innov;
        let pf = (1.0 - k) * p_pred;
        self.state = Some((xf, pf));
        xf
    }

    /// Return to the just-constructed state.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Group delay of the reporting convention: 0 (causal filter, no look-ahead).
    pub fn delay(&self) -> usize {
        0
    }
}

impl_streaming_denoiser!(StreamingKalman);

// ---------------------------------------------------------------------------
// Variance-stabilizing transform (streaming).
// ---------------------------------------------------------------------------

/// Default length of the sliding residual window feeding the streaming Duan
/// smearing inverse (signed-log / signed-sqrt / Box-Cox kinds). Bounds both the
/// per-sample cost (`O(W)` inverse evaluations) and the memory (`W` `f64`s); large
/// enough to represent the — by construction homoscedastic — transformed-domain
/// residual distribution, small enough for an edge device. Override with
/// [`StreamingVst::with_residual_window`].
pub const DEFAULT_RESIDUAL_WINDOW: usize = 64;

/// Causal **variance-stabilizing-transform denoiser** — the streaming counterpart of
/// [`super::vst::vst_denoise`], for *signal-dependent* (Poisson / multiplicative)
/// noise on a live stream.
///
/// It wraps any inner [`StreamingDenoiser`] `D` and runs the same three-stage
/// pipeline as the batch VST, one sample at a time:
///
/// ```text
/// x → φ (pointwise forward VST) → D (streaming Gaussian denoiser) → φ⁻¹ (bias-corrected) → x̂
/// ```
///
/// The transform `kind` is a **calibration input**, not detected on the fly:
/// [`super::vst::detect_noise_model`] needs the level-vs-scale statistics of a whole
/// record, so identify the noise law once, offline, on a representative batch
/// capture, then stream with that fixed kind. (`Identity` makes this a transparent
/// wrapper around `D`.)
///
/// # Delay and batch equivalence
///
/// `delay()` is exactly the inner denoiser's `D::delay()`: the forward transform and
/// the bias-corrected inverse are both pointwise-or-local and add none of their own.
/// For the **pointwise-inverse kinds** — [`VstKind::Identity`], [`VstKind::Anscombe`]
/// (exact unbiased inverse) and [`VstKind::Gat`] (exact unbiased GAT inverse) — the
/// whole pipeline is pointwise around `D`, so once `D`'s window is full the output is
/// **bit-for-bit** equal to the batch [`super::vst::vst_denoise`] run with `D`'s batch
/// counterpart, delayed by `delay()`, on the batch filter's interior (pinned by test).
///
/// # The smearing kinds are causal, not batch-identical
///
/// [`VstKind::SignedLog`], [`VstKind::SignedSqrt`] and [`VstKind::BoxCox`] use the
/// Duan (1983) smearing inverse, which needs the *distribution* of transformed-domain
/// residuals. The batch version smears over the whole record; a stream has no future,
/// so this smears over a **sliding window of the most recent `residual_window`
/// residuals** (default [`DEFAULT_RESIDUAL_WINDOW`]). That makes the correction
/// causal and *locally* adaptive — an honest difference from batch, not an
/// approximation of it — at `O(residual_window)` inverse evaluations per sample.
/// Non-finite residuals are dropped (as in the batch `smearing_sample`); until the
/// first finite residual arrives the naive inverse is used.
///
/// # Warm-up and degradation
///
/// During `D`'s warm-up the output is `D`'s best partial-window estimate mapped back
/// through the inverse — always finite, not covered by the batch-equivalence
/// guarantee. A degenerate `kind` (non-finite Box-Cox `λ`, `gain ≤ 0`) makes the
/// forward transform the identity and the inverse the naive map, i.e. the wrapper
/// reduces to `D` acting on the raw stream. `reset()` returns the whole thing —
/// inner denoiser, transformed-value buffer and residual window — to the
/// just-constructed state.
///
/// ```
/// use scirust_signal::denoise::streaming::{StreamingDenoiser, StreamingVst};
/// use scirust_signal::denoise::vst::VstKind;
/// use scirust_signal::denoise::streaming::StreamingMovingAverage;
///
/// // Poisson-ish counts streamed through an Anscombe-stabilized moving average.
/// let mut f = StreamingVst::new(VstKind::Anscombe, StreamingMovingAverage::new(5));
/// let out: Vec<f64> = [4.0, 6.0, 3.0, 9.0, 5.0, 7.0].iter().map(|&x| f.push(x)).collect();
/// assert_eq!(out.len(), 6);
/// assert!(out.iter().all(|v| v.is_finite()));
/// assert_eq!(f.delay(), 2); // = the inner moving average's delay
/// ```
#[derive(Debug, Clone)]
pub struct StreamingVst<D: StreamingDenoiser> {
    kind: VstKind,
    inner: D,
    /// Whether `kind`'s corrected inverse needs the residual distribution (smearing
    /// kinds) rather than being pointwise (Identity / Anscombe / GAT).
    needs_residuals: bool,
    /// Transformed values buffered to recover `t[i − delay]` — the value `D`'s output
    /// estimates — for the residual `t[i − delay] − D.push(t[i])`. Capacity `delay + 1`.
    tbuf: VecDeque<f64>,
    /// Sliding window of the most recent finite transformed-domain residuals.
    resid: VecDeque<f64>,
    /// Cap on `resid` (the `residual_window`).
    resid_cap: usize,
}

impl<D: StreamingDenoiser> StreamingVst<D> {
    /// Wrap `inner` with the transform `kind`, using [`DEFAULT_RESIDUAL_WINDOW`] for
    /// the smearing kinds' residual window.
    pub fn new(kind: VstKind, inner: D) -> Self {
        Self::with_residual_window(kind, inner, DEFAULT_RESIDUAL_WINDOW)
    }

    /// Wrap `inner` with the transform `kind` and an explicit `residual_window` (the
    /// sliding-window size of the streaming Duan smearing inverse; ignored by the
    /// pointwise-inverse kinds). A `residual_window` of 0 is treated as 1.
    pub fn with_residual_window(kind: VstKind, inner: D, residual_window: usize) -> Self {
        let needs_residuals = matches!(
            kind,
            VstKind::SignedLog | VstKind::SignedSqrt | VstKind::BoxCox(_)
        );
        let cap = residual_window.max(1);
        let delay = inner.delay();
        Self {
            kind,
            inner,
            needs_residuals,
            tbuf: VecDeque::with_capacity(delay + 1),
            resid: VecDeque::with_capacity(cap),
            resid_cap: cap,
        }
    }

    /// Feed one sample; see [`StreamingDenoiser::push`]. Returns the (delayed by
    /// [`Self::delay`]) bias-corrected estimate in the *original* coordinates.
    pub fn push(&mut self, x: f64) -> f64 {
        let t = forward_scalar(self.kind, x);
        let f = self.inner.push(t);
        if !self.needs_residuals
        {
            // Identity / Anscombe / GAT: the corrected inverse is pointwise.
            return inverse_corrected_pointwise_scalar(self.kind, f);
        }
        // Smearing kinds: recover t[i − delay] (the value `f` estimates) to form the
        // transformed-domain residual, keep a sliding window of recent residuals, and
        // average the naive inverse over that window (the causal Duan estimate).
        let cap = self.inner.delay() + 1;
        if self.tbuf.len() == cap
        {
            self.tbuf.pop_front();
        }
        self.tbuf.push_back(t);
        // Once full, front() == t[i − delay]; during warm-up it is the oldest buffered
        // transformed value paired with `D`'s partial estimate (finite, best-effort).
        let t_delayed = *self.tbuf.front().expect("just pushed, non-empty");
        let r = t_delayed - f;
        if r.is_finite()
        {
            if self.resid.len() == self.resid_cap
            {
                self.resid.pop_front();
            }
            self.resid.push_back(r);
        }
        if self.resid.is_empty()
        {
            return inverse_naive_scalar(self.kind, f);
        }
        let mut sum = 0.0;
        for &rj in &self.resid
        {
            sum += inverse_naive_scalar(self.kind, f + rj);
        }
        sum / self.resid.len() as f64
    }

    /// Return to the just-constructed state (inner denoiser reset, buffers cleared).
    pub fn reset(&mut self) {
        self.inner.reset();
        self.tbuf.clear();
        self.resid.clear();
    }

    /// Group delay: exactly the inner denoiser's delay (the VST stages add none).
    pub fn delay(&self) -> usize {
        self.inner.delay()
    }

    /// The transform kind this wrapper applies.
    pub fn kind(&self) -> VstKind {
        self.kind
    }
}

impl<D: StreamingDenoiser> StreamingDenoiser for StreamingVst<D> {
    fn push(&mut self, x: f64) -> f64 {
        StreamingVst::push(self, x)
    }
    fn reset(&mut self) {
        StreamingVst::reset(self);
    }
    fn delay(&self) -> usize {
        StreamingVst::delay(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Lcg, snr_db};
    use super::super::{linear, median, rank};
    use super::*;
    use core::f64::consts::PI;

    fn noisy_sine(n: usize, noise: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Lcg::new(seed);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + noise * rng.gauss()).collect();
        (clean, obs)
    }

    /// Stream a whole signal through a filter via the trait, so the delegating
    /// trait impls are exercised alongside the inherent methods.
    fn run<D: StreamingDenoiser>(f: &mut D, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|&x| f.push(x)).collect()
    }

    #[test]
    fn moving_average_matches_batch_on_the_interior() {
        let (_, obs) = noisy_sine(256, 0.4, 5);
        // Odd windows, plus an even one to pin the odd-rounding convention.
        for window in [3usize, 5, 9, 8]
        {
            let mut f = StreamingMovingAverage::new(window);
            let d = f.delay();
            assert_eq!(d, window / 2);
            let out = run(&mut f, &obs);
            let batch = linear::moving_average(&obs, window);
            for i in (2 * d)..obs.len()
            {
                assert_eq!(out[i], batch[i - d], "window {window}, i {i}");
            }
        }
        // The window parameter is live: different widths give different outputs.
        let out3 = run(&mut StreamingMovingAverage::new(3), &obs);
        let out9 = run(&mut StreamingMovingAverage::new(9), &obs);
        assert_ne!(out3, out9);
    }

    #[test]
    fn median_matches_batch_on_the_interior() {
        let (_, mut obs) = noisy_sine(256, 0.3, 7);
        for &idx in &[20usize, 77, 141, 200]
        {
            obs[idx] += 6.0; // spikes make the median actually act
        }
        // Equality against the batch filter with the SAME half_window pins the
        // parameter plumbing (the liveness idiom of the module).
        for h in [1usize, 2, 4]
        {
            let mut f = StreamingMedian::new(h);
            assert_eq!(f.delay(), h);
            let out = run(&mut f, &obs);
            let batch = rank::median_filter(&obs, h);
            for i in (2 * h)..obs.len()
            {
                assert_eq!(out[i], batch[i - h], "h {h}, i {i}");
            }
        }
    }

    #[test]
    fn hampel_matches_batch_on_the_interior() {
        let (_, mut obs) = noisy_sine(256, 0.2, 11);
        for &idx in &[25usize, 90, 160, 230]
        {
            obs[idx] -= 7.0;
        }
        // Distinct (half_window, n_sigma) pairs each matching their own batch
        // output pin both parameters against transposition or being ignored.
        for (h, ns) in [(2usize, 3.0), (3, 2.5), (4, 4.0)]
        {
            let mut f = StreamingHampel::new(h, ns);
            assert_eq!(f.delay(), h);
            let out = run(&mut f, &obs);
            let batch = rank::hampel_filter(&obs, h, ns);
            for i in (2 * h)..obs.len()
            {
                assert_eq!(out[i], batch[i - h], "h {h}, n_sigma {ns}, i {i}");
            }
        }
    }

    #[test]
    fn hampel_corrects_the_spike_at_its_delayed_position() {
        let mut sig: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin()).collect();
        sig[30] += 10.0;
        let mut f = StreamingHampel::new(3, 3.0);
        let d = f.delay();
        let out = run(&mut f, &sig);
        // The output for the spike sample appears d pushes later and must be the
        // window median — not the spike.
        assert_eq!(out[30 + d], median(&sig[27..=33]));
        assert_ne!(out[30 + d], sig[30]);
        // Clean samples pass verbatim (decision filter, not a smoother).
        assert_eq!(out[10 + d], sig[10]);
        // n_sigma is live: an absurdly lax threshold lets the spike through.
        let mut lax = StreamingHampel::new(3, 1.0e9);
        let out_lax = run(&mut lax, &sig);
        assert_eq!(out_lax[30 + d], sig[30]);
    }

    #[test]
    fn ema_matches_batch_exactly() {
        let (_, obs) = noisy_sine(512, 0.4, 13);
        // Exact sample-for-sample equality for each alpha pins the parameter.
        for alpha in [0.05, 0.3, 1.0]
        {
            let mut f = StreamingEma::new(alpha);
            assert_eq!(f.delay(), 0);
            let out = run(&mut f, &obs);
            assert_eq!(
                out,
                linear::exp_moving_average(&obs, alpha),
                "alpha {alpha}"
            );
        }
        // Out-of-range alpha degrades to the identity, exactly like the batch guard.
        for alpha in [0.0, -0.5, 1.5]
        {
            let mut f = StreamingEma::new(alpha);
            assert_eq!(run(&mut f, &obs), obs, "alpha {alpha}");
        }
    }

    #[test]
    fn kalman_replicates_the_batch_forward_recursion() {
        let mut rng = Lcg::new(17);
        let obs: Vec<f64> = (0..512).map(|_| rng.gauss()).collect();
        let (q, r) = (0.01, 1.0);
        let mut f = StreamingKalman::new(q, r);
        assert_eq!(f.delay(), 0);
        let out = run(&mut f, &obs);
        // Reference: the exact forward recursion of `adaptive::kalman_forward_rts`
        // with its semi-diffuse start (x_pred = first sample, p_pred = r + q).
        let mut expect = Vec::with_capacity(obs.len());
        let mut xf = 0.0;
        let mut pf = 0.0;
        for (i, &y) in obs.iter().enumerate()
        {
            let (x_pred, p_pred) = if i == 0 { (y, r + q) } else { (xf, pf + q) };
            let k = p_pred / (p_pred + r);
            xf = x_pred + k * (y - x_pred);
            pf = (1.0 - k) * p_pred;
            expect.push(xf);
        }
        assert_eq!(out, expect);
        // q and r are live and not transposed: swapping them changes the output.
        let mut swapped = StreamingKalman::new(r, q);
        assert_ne!(run(&mut swapped, &obs), expect);
    }

    #[test]
    fn kalman_reduces_white_noise_variance() {
        let mut rng = Lcg::new(19);
        let obs: Vec<f64> = (0..2048).map(|_| rng.gauss()).collect();
        let mut f = StreamingKalman::new(0.01, 1.0); // q/r = 0.01
        let out = run(&mut f, &obs);
        let var = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64
        };
        assert!(
            var(&out) * 3.0 < var(&obs),
            "output variance {} vs input {}",
            var(&out),
            var(&obs)
        );
    }

    #[test]
    fn kalman_tracks_a_step_quickly() {
        let step: Vec<f64> = (0..120).map(|i| if i < 40 { 0.0 } else { 4.0 }).collect();
        let mut f = StreamingKalman::new(0.1, 1.0);
        let out = run(&mut f, &step);
        // Steady-state gain for q/r = 0.1 is ≈ 0.27, so the bias 4·(1 − k)^t is
        // far below 1% of the step within 25 samples.
        assert!(
            (out[64] - 4.0).abs() < 0.04,
            "25 samples after the step: {}",
            out[64]
        );
        assert!(
            (out[119] - 4.0).abs() < 1.0e-3,
            "end of record: {}",
            out[119]
        );
        // Invalid variances degrade to the identity, like `kalman_smooth`.
        let mut bad = StreamingKalman::new(0.0, 1.0);
        assert_eq!(run(&mut bad, &step), step);
        let mut bad = StreamingKalman::new(0.1, -1.0);
        assert_eq!(run(&mut bad, &step), step);
    }

    #[test]
    fn reset_reproduces_identical_outputs_for_every_struct() {
        fn check<D: StreamingDenoiser>(mut f: D, sig: &[f64], name: &str) {
            let first: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
            f.reset();
            let second: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
            assert_eq!(first, second, "{name} not reproducible after reset()");
        }
        let (_, obs) = noisy_sine(128, 0.3, 23);
        check(StreamingMovingAverage::new(7), &obs, "moving average");
        check(StreamingMedian::new(3), &obs, "median");
        check(StreamingHampel::new(3, 3.0), &obs, "hampel");
        check(StreamingEma::new(0.2), &obs, "ema");
        check(StreamingKalman::new(0.05, 1.0), &obs, "kalman");
    }

    #[test]
    fn warmup_and_short_inputs_stay_finite() {
        // The streaming analogue of the module's empty/len-1..3 edge cases: push
        // nothing, one, two or three samples — every output must be finite (the
        // warm-up contract), through the trait object to prove object safety too.
        for len in 0..4usize
        {
            let sig: Vec<f64> = (0..len).map(|i| i as f64 - 1.0).collect();
            let mut filters: Vec<Box<dyn StreamingDenoiser>> = vec![
                Box::new(StreamingMovingAverage::new(7)),
                Box::new(StreamingMedian::new(3)),
                Box::new(StreamingHampel::new(3, 3.0)),
                Box::new(StreamingEma::new(0.3)),
                Box::new(StreamingKalman::new(0.1, 1.0)),
            ];
            for f in filters.iter_mut()
            {
                for &x in sig.iter()
                {
                    let y = f.push(x);
                    assert!(y.is_finite(), "len {len}: non-finite warm-up output {y}");
                }
            }
        }
    }

    #[test]
    fn constant_signal_passes_through_every_struct() {
        // A constant is a fixed point of all five filters (for Hampel via the
        // MAD == 0 guard), during warm-up included.
        let sig = vec![3.5; 40];
        let mut filters: Vec<(Box<dyn StreamingDenoiser>, &str)> = vec![
            (Box::new(StreamingMovingAverage::new(7)), "moving average"),
            (Box::new(StreamingMedian::new(3)), "median"),
            (Box::new(StreamingHampel::new(3, 3.0)), "hampel"),
            (Box::new(StreamingEma::new(0.3)), "ema"),
            (Box::new(StreamingKalman::new(0.1, 1.0)), "kalman"),
        ];
        for (f, name) in filters.iter_mut()
        {
            for &x in sig.iter()
            {
                let y = f.push(x);
                assert!((y - 3.5).abs() < 1.0e-12, "{name}: {y}");
            }
        }
    }

    #[test]
    fn degenerate_parameters_are_the_identity() {
        let sig = [1.0, -2.0, 3.5, 0.25, -1.75];
        let mut ma = StreamingMovingAverage::new(1);
        assert_eq!(run(&mut ma, &sig), sig.to_vec());
        assert_eq!(ma.delay(), 0);
        let mut ma0 = StreamingMovingAverage::new(0);
        assert_eq!(run(&mut ma0, &sig), sig.to_vec());
        let mut med = StreamingMedian::new(0);
        assert_eq!(run(&mut med, &sig), sig.to_vec());
        assert_eq!(med.delay(), 0);
        let mut ham = StreamingHampel::new(0, 3.0);
        assert_eq!(run(&mut ham, &sig), sig.to_vec());
        assert_eq!(ham.delay(), 0);
    }

    #[test]
    fn streaming_filters_improve_snr_after_delay_alignment() {
        let (clean, obs) = noisy_sine(512, 0.4, 29);
        let cases: Vec<(Box<dyn StreamingDenoiser>, &str)> = vec![
            (Box::new(StreamingMovingAverage::new(9)), "moving average"),
            (Box::new(StreamingMedian::new(3)), "median"),
            (Box::new(StreamingEma::new(0.15)), "ema"),
            (Box::new(StreamingKalman::new(0.016, 0.16)), "kalman"),
        ];
        for (mut f, name) in cases
        {
            let d = f.delay();
            let out: Vec<f64> = obs.iter().map(|&x| f.push(x)).collect();
            // Per the delay contract, out[i] estimates x[i − d]: align before scoring.
            let est = &out[d..];
            let reference = &clean[..clean.len() - d];
            let raw = &obs[..obs.len() - d];
            let s_out = snr_db(reference, est);
            let s_raw = snr_db(reference, raw);
            assert!(
                s_out > s_raw + 1.0,
                "{name}: {s_out:.2} dB vs raw {s_raw:.2} dB"
            );
        }
    }

    // ── StreamingVst ────────────────────────────────────────────────────────

    /// Knuth's Poisson sampler on the shared LCG (λ ≲ 30) — the deterministic
    /// signal-dependent-noise fixture generator for the streaming VST tests.
    fn poisson(rng: &mut Lcg, lambda: f64) -> f64 {
        if lambda <= 0.0
        {
            return 0.0;
        }
        let l = (-lambda).exp();
        let mut k: u64 = 0;
        let mut p = 1.0;
        loop
        {
            k += 1;
            p *= rng.uniform();
            if p <= l
            {
                break;
            }
        }
        (k - 1) as f64
    }

    /// `(clean intensity, Poisson counts)`: a slow intensity λ ∈ [1, ~12].
    fn poisson_fixture(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let clean: Vec<f64> = (0..n)
            .map(|i| 6.0 + 5.0 * (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let mut rng = Lcg::new(seed);
        let obs: Vec<f64> = clean.iter().map(|&l| poisson(&mut rng, l)).collect();
        (clean, obs)
    }

    #[test]
    fn vst_pointwise_kinds_match_batch_on_the_interior() {
        // For the pointwise-inverse kinds the streaming VST is bit-for-bit the batch
        // vst_denoise around the inner filter's batch counterpart, delayed by delay().
        use crate::denoise::{VstKind, moving_average, vst_denoise};
        let (_, obs) = poisson_fixture(512, 7);
        // Anscombe (exact unbiased inverse), Identity, and a GAT calibration — all
        // pointwise. Inner: moving average, whose batch/stream interior is exact.
        let kinds = [
            VstKind::Identity,
            VstKind::Anscombe,
            VstKind::Gat {
                gain: 1.3,
                sigma: 1.5,
            },
        ];
        for kind in kinds
        {
            for window in [5usize, 9]
            {
                let mut f = StreamingVst::new(kind, StreamingMovingAverage::new(window));
                let d = f.delay();
                assert_eq!(d, window / 2);
                let out = run(&mut f, &obs);
                let batch = vst_denoise(&obs, kind, |x| moving_average(x, window));
                for i in (2 * d)..obs.len()
                {
                    assert_eq!(out[i], batch[i - d], "{kind:?}, window {window}, i {i}");
                }
            }
        }
    }

    #[test]
    fn vst_anscombe_denoises_streamed_poisson() {
        // The end-to-end embedded claim: the Anscombe-stabilized streaming pipeline
        // removes signal-dependent noise on a live stream, and the stabilization
        // materially changes the reconstruction. (Whether stabilization *out-scores*
        // no-transform depends on the inner denoiser — for a plain linear smoother
        // the gain is ≈ 0, exactly as the batch vst_protocol P1 measured; the VST
        // pays off with floor-tracking denoisers. That comparison is settled
        // rigorously by the batch protocol, not re-litigated fragilely here.)
        use crate::denoise::VstKind;
        let (clean, obs) = poisson_fixture(2048, 11);
        let window = 9;
        let mut vst = StreamingVst::new(VstKind::Anscombe, StreamingMovingAverage::new(window));
        let d = vst.delay();
        let out_vst = run(&mut vst, &obs);
        let reference = &clean[..clean.len() - d];
        let s_vst = snr_db(reference, &out_vst[d..]);
        let s_raw = snr_db(reference, &obs[..obs.len() - d]);
        assert!(
            s_vst > s_raw + 1.0,
            "VST stream must denoise the raw counts: {s_vst:.2} vs {s_raw:.2} dB"
        );
        // Stabilization is not a no-op: the reconstruction differs from the untransformed
        // smoother (the pointwise Anscombe/exact-unbiased sandwich is genuinely applied).
        let mut plain = StreamingMovingAverage::new(window);
        let out_plain = run(&mut plain, &obs);
        assert_ne!(out_vst, out_plain);
    }

    #[test]
    fn vst_signed_log_smearing_is_causal_and_improves_snr() {
        // Multiplicative noise, streamed through a signed-log-stabilized median with
        // the sliding-window Duan smearing inverse.
        use crate::denoise::VstKind;
        let n = 2048;
        let mut rng = Lcg::new(9);
        let clean: Vec<f64> = (0..n)
            .map(|i| 20.0 + 16.0 * (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean
            .iter()
            .map(|&s| s * (1.0 + 0.3 * rng.gauss()))
            .collect();
        let mut f = StreamingVst::new(VstKind::SignedLog, StreamingMedian::new(4));
        let d = f.delay();
        let out = run(&mut f, &obs);
        assert!(out.iter().all(|v| v.is_finite()));
        let reference = &clean[..clean.len() - d];
        let s_out = snr_db(reference, &out[d..]);
        let s_raw = snr_db(reference, &obs[..obs.len() - d]);
        assert!(
            s_out > s_raw + 1.0,
            "signed-log smearing stream: {s_out:.2} vs raw {s_raw:.2} dB"
        );
        // The residual window is live: a length-1 window (single-point smearing ≈
        // naive inverse) gives a different, less smooth reconstruction.
        let mut f1 =
            StreamingVst::with_residual_window(VstKind::SignedLog, StreamingMedian::new(4), 1);
        let out1 = run(&mut f1, &obs);
        assert_ne!(
            out, out1,
            "residual_window must change the smearing estimate"
        );
    }

    #[test]
    fn vst_identity_kind_is_a_transparent_wrapper() {
        // Identity forward + pointwise identity inverse ⇒ the wrapper is exactly D.
        use crate::denoise::VstKind;
        let (_, obs) = noisy_sine(256, 0.4, 5);
        let mut wrapped = StreamingVst::new(VstKind::Identity, StreamingMovingAverage::new(7));
        let mut bare = StreamingMovingAverage::new(7);
        assert_eq!(wrapped.delay(), bare.delay());
        assert_eq!(run(&mut wrapped, &obs), run(&mut bare, &obs));
    }

    #[test]
    fn vst_reset_reproduces_identical_outputs() {
        use crate::denoise::VstKind;
        let (_, obs) = poisson_fixture(300, 23);
        // One pointwise kind and one smearing kind (residual window must reset too).
        let mut a = StreamingVst::new(VstKind::Anscombe, StreamingMovingAverage::new(7));
        let first = run(&mut a, &obs);
        a.reset();
        let second = run(&mut a, &obs);
        assert_eq!(first, second, "Anscombe VST not reproducible after reset()");
        let mut s = StreamingVst::new(VstKind::SignedLog, StreamingMedian::new(3));
        let first = run(&mut s, &obs);
        s.reset();
        let second = run(&mut s, &obs);
        assert_eq!(
            first, second,
            "signed-log VST not reproducible after reset()"
        );
    }

    #[test]
    fn vst_kind_is_live_and_object_safe() {
        // Different kinds give different reconstructions (the kind is plumbed), and
        // the wrapper is object-safe behind the trait.
        use crate::denoise::VstKind;
        let (_, obs) = poisson_fixture(512, 31);
        let mut ans: Box<dyn StreamingDenoiser> = Box::new(StreamingVst::new(
            VstKind::Anscombe,
            StreamingMovingAverage::new(5),
        ));
        let mut sqrt: Box<dyn StreamingDenoiser> = Box::new(StreamingVst::new(
            VstKind::SignedSqrt,
            StreamingMovingAverage::new(5),
        ));
        let out_ans: Vec<f64> = obs.iter().map(|&x| ans.push(x)).collect();
        let out_sqrt: Vec<f64> = obs.iter().map(|&x| sqrt.push(x)).collect();
        assert_ne!(out_ans, out_sqrt);
        assert_eq!(ans.delay(), 2);
    }

    #[test]
    fn vst_degrades_gracefully_on_short_and_degenerate_input() {
        use crate::denoise::VstKind;
        // Warm-up / short inputs stay finite for both a pointwise and a smearing kind.
        for len in 0..4usize
        {
            let sig: Vec<f64> = (0..len).map(|i| (i + 1) as f64).collect();
            let mut a = StreamingVst::new(VstKind::Anscombe, StreamingMovingAverage::new(7));
            let mut s = StreamingVst::new(VstKind::SignedLog, StreamingMedian::new(3));
            for &x in &sig
            {
                assert!(a.push(x).is_finite(), "anscombe len {len}");
                assert!(s.push(x).is_finite(), "signed-log len {len}");
            }
        }
        // Degenerate GAT parameters ⇒ forward is identity, inverse is naive: the
        // wrapper reduces to D on the raw stream (pointwise-identity path).
        let (_, obs) = poisson_fixture(256, 3);
        let mut bad = StreamingVst::new(
            VstKind::Gat {
                gain: 0.0,
                sigma: 1.0,
            },
            StreamingMovingAverage::new(5),
        );
        let mut bare = StreamingMovingAverage::new(5);
        assert_eq!(run(&mut bad, &obs), run(&mut bare, &obs));
        // residual_window = 0 is clamped to 1, never a divide-by-zero.
        let mut z =
            StreamingVst::with_residual_window(VstKind::SignedLog, StreamingMedian::new(3), 0);
        assert!(run(&mut z, &obs).iter().all(|v| v.is_finite()));
    }
}
