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
}
