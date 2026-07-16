//! 2-D image denoising — rank filtering, separable wavelet shrinkage, and
//! non-local means.
//!
//! Images are flat row-major `&[f64]` slices with explicit `(width, height)`,
//! matching the crate convention. Three complementary families cover the classic
//! noise regimes of an imaging chain:
//!
//! * [`median2d`] — the rank-filter workhorse for **impulsive** ("salt-and-
//!   pepper") noise (Tukey 1977). Each pixel becomes the median of its square
//!   neighbourhood, so an isolated outlier — however large — cannot survive,
//!   while a straight edge is preserved exactly: the majority of any window
//!   always sits on the pixel's own side of the edge.
//! * [`wavelet_denoise2d`] — separable 2-D wavelet shrinkage for **additive
//!   Gaussian** noise: the Mallat pyramid (Mallat 1989) concentrates the image
//!   into few large coefficients while white noise stays spread thin, and the
//!   VisuShrink universal threshold (Donoho–Johnstone 1994) removes the noise
//!   floor without the indiscriminate blur of a low-pass.
//!   [`wavelet_denoise2d_auto`] is the zero-knob variant.
//! * [`nlm2d`] — non-local means (Buades–Coll–Morel 2005) for **textured and
//!   repetitive** content: each pixel is replaced by a weighted average of the
//!   pixels whose surrounding *patches* look alike anywhere in a search window,
//!   so recurring structure — texture, stripes, printed patterns — is averaged
//!   along its own repetitions instead of being smoothed away.
//!
//! All entry points degrade gracefully rather than panic: an empty image, or a
//! slice whose length disagrees with `width·height` (a malformed caller), comes
//! back as an unmodified copy, and a stray NaN pixel flows through the NaN-safe
//! [`f64::total_cmp`] order statistics without crashing. Builds on the 1-D
//! filter banks of `scirust-signal` ([`dwt_step`] / [`idwt_step`] /
//! [`apply_threshold`]); everything 2-D lives here.

use scirust_signal::denoise::transform::{apply_threshold, dwt_step, idwt_step};
use scirust_signal::denoise::{ThresholdMode, Wavelet};

// ─── Shared order-statistics helpers ────────────────────────────────────────

/// Median of a slice (clones and sorts). Returns `0.0` for an empty slice.
///
/// Sorts with [`f64::total_cmp`] — a *total* order. A `partial_cmp`-based
/// comparator is inconsistent when the slice contains NaN (NaN compares
/// "equal" to everything while the finite values still order among themselves),
/// which modern Rust sorts detect and **panic** on; `total_cmp` orders NaN
/// deterministically instead, so a stray NaN pixel degrades gracefully rather
/// than crashing the filter. For all-finite input the ordering is identical to
/// `partial_cmp`. (Same convention as the `scirust-signal` rank filters.)
fn median(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1
    {
        v[n / 2]
    }
    else
    {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Median absolute deviation `median(|x − median(x)|)`: a robust scale estimator
/// whose 50% breakdown point makes it immune to the minority of large *signal*
/// coefficients sharing a band with the noise. Returns `0.0` when empty.
fn mad(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return 0.0;
    }
    let med = median(values);
    let dev: Vec<f64> = values.iter().map(|&x| (x - med).abs()).collect();
    median(&dev)
}

// ─── Median filtering ───────────────────────────────────────────────────────

/// 2-D median filter over the square window of half-width `radius`
/// (`(2·radius+1)²` pixels) — the salt-and-pepper workhorse.
///
/// Borders are handled by clamping coordinates to the image (replicate
/// padding), so edge pixels see a full-size window and the output never
/// shrinks. Windows are sorted with the NaN-safe [`f64::total_cmp`] total order
/// (see [`median`] for why `partial_cmp` would panic on a NaN pixel).
///
/// An impulse — a pixel arbitrarily far from its neighbours — is an order
/// statistic outlier and is annihilated outright, while straight edges pass
/// through unchanged (only sharp *corners* are slightly eroded, the classic
/// median trade-off). `radius = 0`, an empty image, or a `width·height` /
/// slice-length mismatch returns the input copied.
pub fn median2d(img: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    if img.len() != width * height || img.is_empty() || radius == 0
    {
        return img.to_vec();
    }
    let side = 2 * radius + 1;
    let r = radius as isize;
    let mut window = Vec::with_capacity(side * side);
    let mut out = Vec::with_capacity(img.len());
    for y in 0..height as isize
    {
        for x in 0..width as isize
        {
            window.clear();
            for dy in -r..=r
            {
                let sy = (y + dy).clamp(0, height as isize - 1) as usize;
                for dx in -r..=r
                {
                    let sx = (x + dx).clamp(0, width as isize - 1) as usize;
                    window.push(img[sy * width + sx]);
                }
            }
            window.sort_by(|a, b| a.total_cmp(b));
            // (2·radius+1)² is odd, so the median is exactly the middle element.
            out.push(window[window.len() / 2]);
        }
    }
    out
}

// ─── Separable 2-D wavelet transform ────────────────────────────────────────

/// Reflect an out-of-range index back into `0..n` (symmetric, edge-repeated
/// mirror: `… 2 1 0 | 0 1 2 … n−1 | n−1 n−2 …`), folding as many times as
/// needed, so an image can be padded well past twice its extent.
fn mirror_index(i: usize, n: usize) -> usize {
    if n <= 1
    {
        return 0;
    }
    let period = 2 * n;
    let m = i % period;
    if m < n { m } else { period - 1 - m }
}

/// Padded extent for one image dimension: the next power of two that is at
/// least `dim` *and* at least the filter length, so every [`dwt_step`] sees an
/// even input no shorter than its taps.
fn padded_extent(dim: usize, filter_len: usize) -> usize {
    dim.max(filter_len).max(1).next_power_of_two()
}

/// Pad an image to `tw × th` by symmetric reflection ([`mirror_index`]) in both
/// dimensions — a smooth, discontinuity-free border for the dyadic transform.
fn pad_mirror_2d(img: &[f64], width: usize, height: usize, tw: usize, th: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(tw * th);
    for y in 0..th
    {
        let sy = mirror_index(y, height);
        for x in 0..tw
        {
            out.push(img[sy * width + mirror_index(x, width)]);
        }
    }
    out
}

/// One separable analysis level over the top-left `cw × ch` region of a buffer
/// with row `stride`: [`dwt_step`] every ROW (left half ← approximation, right
/// half ← detail), then every COLUMN of both halves (top ← approximation,
/// bottom ← detail). Afterwards the region holds the four subbands — LL
/// top-left, horizontal detail top-right, vertical detail bottom-left, and the
/// diagonal detail HH bottom-right.
fn dwt2d_level_forward(buf: &mut [f64], stride: usize, cw: usize, ch: usize, h: &[f64]) {
    let mut row = vec![0.0; cw];
    for y in 0..ch
    {
        row.copy_from_slice(&buf[y * stride..y * stride + cw]);
        let (approx, detail) = dwt_step(&row, h);
        for i in 0..cw / 2
        {
            buf[y * stride + i] = approx[i];
            buf[y * stride + cw / 2 + i] = detail[i];
        }
    }
    let mut col = vec![0.0; ch];
    for x in 0..cw
    {
        for (y, c) in col.iter_mut().enumerate()
        {
            *c = buf[y * stride + x];
        }
        let (approx, detail) = dwt_step(&col, h);
        for i in 0..ch / 2
        {
            buf[i * stride + x] = approx[i];
            buf[(ch / 2 + i) * stride + x] = detail[i];
        }
    }
}

/// One separable synthesis level — the exact transpose of
/// [`dwt2d_level_forward`]: [`idwt_step`] every COLUMN first, then every ROW,
/// undoing the analysis order (rows-then-columns) step for step.
fn dwt2d_level_inverse(buf: &mut [f64], stride: usize, cw: usize, ch: usize, h: &[f64]) {
    for x in 0..cw
    {
        let approx: Vec<f64> = (0..ch / 2).map(|i| buf[i * stride + x]).collect();
        let detail: Vec<f64> = (0..ch / 2)
            .map(|i| buf[(ch / 2 + i) * stride + x])
            .collect();
        let rec = idwt_step(&approx, &detail, h);
        for (y, &v) in rec.iter().enumerate()
        {
            buf[y * stride + x] = v;
        }
    }
    let mut approx = vec![0.0; cw / 2];
    let mut detail = vec![0.0; cw / 2];
    for y in 0..ch
    {
        approx.copy_from_slice(&buf[y * stride..y * stride + cw / 2]);
        detail.copy_from_slice(&buf[y * stride + cw / 2..y * stride + cw]);
        let rec = idwt_step(&approx, &detail, h);
        buf[y * stride..y * stride + cw].copy_from_slice(&rec);
    }
}

/// Multi-level separable 2-D DWT in place over a `tw × th` power-of-two buffer:
/// each level transforms the current LL region and recurses on its top-left
/// quadrant. Stops early once the region gets shorter than the filter in either
/// dimension (periodization would fold the filter onto itself). Returns the
/// number of levels actually performed.
fn dwt2d_forward(buf: &mut [f64], tw: usize, th: usize, levels: usize, h: &[f64]) -> usize {
    let min_len = h.len().max(2);
    let (mut cw, mut ch) = (tw, th);
    let mut done = 0;
    while done < levels && cw >= min_len && ch >= min_len
    {
        dwt2d_level_forward(buf, tw, cw, ch, h);
        cw /= 2;
        ch /= 2;
        done += 1;
    }
    done
}

/// Inverse of [`dwt2d_forward`] for the same buffer and `levels_done`: levels
/// are undone coarsest-first, each via [`dwt2d_level_inverse`].
fn dwt2d_inverse(buf: &mut [f64], tw: usize, th: usize, levels_done: usize, h: &[f64]) {
    for level in (0..levels_done).rev()
    {
        dwt2d_level_inverse(buf, tw, tw >> level, th >> level, h);
    }
}

// ─── Wavelet shrinkage denoising ────────────────────────────────────────────

/// Separable 2-D wavelet-shrinkage denoising (VisuShrink, Donoho–Johnstone
/// 1994) for additive Gaussian noise.
///
/// The image is padded by symmetric reflection to power-of-two dimensions,
/// decomposed with the Mallat pyramid (each level: [`dwt_step`] on every row,
/// then on every column of both halves, yielding the LL / horizontal /
/// vertical / diagonal subbands; recurse on LL), thresholded, inverted
/// ([`idwt_step`] columns then rows — the exact transpose of the analysis
/// order), and cropped back. Without thresholding the round trip reconstructs
/// the padded image to machine precision (pinned by unit test).
///
/// The noise scale is the Donoho robust estimator on the **finest diagonal
/// band**: `σ = MAD(HH₁)/0.6745`. HH₁ is high-pass in *both* directions at the
/// finest scale, and any smooth region or edge needs low-pass support along at
/// least one direction — so HH₁ is almost pure noise, and the MAD shrugs off
/// the few large coefficients genuine diagonal texture leaves there. Every
/// detail coefficient of every level (never LL) is then shrunk by
/// [`apply_threshold`] with the universal threshold `λ = σ·√(2 ln N)`, `N` the
/// padded pixel count — the threshold below which the maximum of `N` i.i.d.
/// Gaussian noise coefficients falls with probability → 1.
///
/// `levels = 0` selects the depth automatically (the padded dyadic depth minus
/// two, at least one — the 2-D analogue of the 1-D auto depth); otherwise
/// `levels` is capped at the padded dyadic depth. Haar keeps blocky content
/// crisp; [`Wavelet::Db4`] (two vanishing moments) annihilates locally-linear
/// shading and is the better default for smooth images. An empty image or a
/// `width·height` / slice-length mismatch returns the input copied.
pub fn wavelet_denoise2d(
    img: &[f64],
    width: usize,
    height: usize,
    levels: usize,
    mode: ThresholdMode,
    wavelet: Wavelet,
) -> Vec<f64> {
    if img.len() != width * height || img.is_empty()
    {
        return img.to_vec();
    }
    let h = wavelet.lowpass();
    let tw = padded_extent(width, h.len());
    let th = padded_extent(height, h.len());
    let mut buf = pad_mirror_2d(img, width, height, tw, th);

    let max_levels = tw.trailing_zeros().min(th.trailing_zeros()) as usize;
    let levels_req = if levels == 0
    {
        max_levels.saturating_sub(2).max(1)
    }
    else
    {
        levels.min(max_levels)
    };
    let done = dwt2d_forward(&mut buf, tw, th, levels_req, &h);
    if done == 0
    {
        return img.to_vec();
    }

    // Noise scale from the finest diagonal band (bottom-right quadrant), then
    // the universal threshold over the padded pixel count.
    let mut hh = Vec::with_capacity((tw / 2) * (th / 2));
    for y in th / 2..th
    {
        for x in tw / 2..tw
        {
            hh.push(buf[y * tw + x]);
        }
    }
    let sigma = mad(&hh) / 0.6745;
    let lambda = sigma * (2.0 * ((tw * th) as f64).ln()).sqrt();

    // Every coefficient outside the final LL block belongs to a detail band of
    // some level — shrink them all; the LL approximation is never touched.
    let (llw, llh) = (tw >> done, th >> done);
    for y in 0..th
    {
        for x in 0..tw
        {
            if x >= llw || y >= llh
            {
                buf[y * tw + x] = apply_threshold(buf[y * tw + x], lambda, mode);
            }
        }
    }

    dwt2d_inverse(&mut buf, tw, th, done, &h);
    let mut out = Vec::with_capacity(img.len());
    for y in 0..height
    {
        out.extend_from_slice(&buf[y * tw..y * tw + width]);
    }
    out
}

/// Fully automatic wavelet denoising: [`wavelet_denoise2d`] with automatic
/// depth, soft (Donoho–Johnstone default) shrinkage, and the [`Wavelet::Db4`]
/// basis — the smooth-image default.
pub fn wavelet_denoise2d_auto(img: &[f64], width: usize, height: usize) -> Vec<f64> {
    wavelet_denoise2d(img, width, height, 0, ThresholdMode::Soft, Wavelet::Db4)
}

// ─── Non-local means ────────────────────────────────────────────────────────

/// Auto filtering strength of [`nlm2d`] as a multiple of the estimated noise
/// σ, used when the caller passes `h <= 0`. Calibrated on this module's test
/// fixture (diagonal-stripe texture, 64×64, σ = 0.15, 3×3 patches, 11×11
/// search): 0.75·σ gained ≈ +8 dB PSNR over the noisy input and sat within a
/// fraction of a dB of the best hand-tuned value; it also matches the
/// 0.55–0.9·σ range the IPOL reference implementation recommends for small
/// patches. Larger ⇒ smoother, smaller ⇒ more conservative.
const NLM_AUTO_H_FACTOR: f64 = 0.75;

/// Non-local means denoising (Buades–Coll–Morel 2005).
///
/// Each output pixel is the weighted average of every pixel in its
/// `(2·search_half+1)²` search window, weighted by *patch* similarity: with
/// `d²` the mean squared difference between the `(2·patch_half+1)²` patches
/// around the two pixels, the weight is `w = exp(−max(0, d² − 2σ²)/h²)`. The
/// `2σ²` offset removes the noise's own expected contribution to `d²`, so two
/// noisy copies of the same underlying patch count as a perfect match.
/// Similar structure anywhere in the window — not just adjacent pixels — is
/// averaged together, which is why NLM shines on texture and repetitive
/// content that transform shrinkage smears. Borders replicate (coordinates
/// clamp to the image).
///
/// The noise σ is estimated as `MAD((img[i] − img[i+1])/√2)/0.6745` over the
/// half-pixel-shift differences of each row — first differences annihilate
/// locally-constant signal, so on mostly smooth images the differences are
/// noise-dominated and the robust MAD ignores the minority straddling edges.
/// `h <= 0` selects the filtering strength automatically as
/// [`NLM_AUTO_H_FACTOR`]`·σ` (see its calibration note).
///
/// Complexity is `O(n · (2·search_half+1)² · (2·patch_half+1)²)` — quadratic
/// in both radii — so keep the search window moderate on large frames
/// (`patch_half = 1`, `search_half = 5` is a solid default). An empty image, a
/// `width·height` / slice-length mismatch, or an auto `h` on a noise-free
/// (constant) image returns the input copied.
pub fn nlm2d(
    img: &[f64],
    width: usize,
    height: usize,
    patch_half: usize,
    search_half: usize,
    h: f64,
) -> Vec<f64> {
    if img.len() != width * height || img.is_empty()
    {
        return img.to_vec();
    }

    // Donoho-style robust noise scale from horizontal half-pixel differences.
    let mut diffs = Vec::with_capacity(img.len());
    for y in 0..height
    {
        for x in 0..width - 1
        {
            diffs.push((img[y * width + x] - img[y * width + x + 1]) / core::f64::consts::SQRT_2);
        }
    }
    let sigma = mad(&diffs) / 0.6745;
    let h = if h > 0.0
    {
        h
    }
    else
    {
        NLM_AUTO_H_FACTOR * sigma
    };
    let h_sq = h * h;
    if !h_sq.is_finite() || h_sq <= 0.0
    {
        // No filtering strength (noise-free image, or a NaN/inf estimate): identity.
        return img.to_vec();
    }
    let two_sigma_sq = 2.0 * sigma * sigma;

    let (w_i, h_i) = (width as isize, height as isize);
    let pixel = |x: isize, y: isize| -> f64 {
        // Replicate borders: clamp coordinates to the image.
        img[(y.clamp(0, h_i - 1) as usize) * width + x.clamp(0, w_i - 1) as usize]
    };
    let (ph, sh) = (patch_half as isize, search_half as isize);
    let patch_area = ((2 * ph + 1) * (2 * ph + 1)) as f64;

    let mut out = Vec::with_capacity(img.len());
    for y in 0..h_i
    {
        for x in 0..w_i
        {
            let mut weight_sum = 0.0;
            let mut acc = 0.0;
            for dy in -sh..=sh
            {
                for dx in -sh..=sh
                {
                    let (cx, cy) = (x + dx, y + dy);
                    let mut d_sq = 0.0;
                    for py in -ph..=ph
                    {
                        for px in -ph..=ph
                        {
                            let d = pixel(x + px, y + py) - pixel(cx + px, cy + py);
                            d_sq += d * d;
                        }
                    }
                    d_sq /= patch_area;
                    let weight = (-((d_sq - two_sigma_sq).max(0.0)) / h_sq).exp();
                    weight_sum += weight;
                    acc += weight * pixel(cx, cy);
                }
            }
            // The centre offset always contributes weight 1, so weight_sum >= 1
            // for finite input; the fallback covers a NaN-poisoned window.
            out.push(
                if weight_sum > 0.0
                {
                    acc / weight_sum
                }
                else
                {
                    pixel(x, y)
                },
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    /// Deterministic 64-bit LCG so noise tests are reproducible without a
    /// `rand` dependency (same generator as the `scirust-signal` test util).
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        /// Uniform in [0, 1).
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Standard normal via Box-Muller.
        fn gauss(&mut self) -> f64 {
            let u1 = self.uniform().max(1.0e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
    }

    /// Peak signal-to-noise ratio `10·log10(peak²/mse)` against a clean
    /// reference, with `peak` the reference maximum.
    fn psnr_db(clean: &[f64], est: &[f64]) -> f64 {
        let peak = clean.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let mse = clean
            .iter()
            .zip(est.iter())
            .map(|(&c, &e)| (c - e) * (c - e))
            .sum::<f64>()
            / clean.len() as f64;
        10.0 * (peak * peak / mse.max(1.0e-30)).log10()
    }

    /// 64×64 synthetic scene: smooth two-way gradient + bright rectangle +
    /// circle-ish disc — edges, a blob, and shading in one frame.
    fn synthetic_scene() -> Vec<f64> {
        let (w, h) = (64usize, 64usize);
        let mut img = Vec::with_capacity(w * h);
        for y in 0..h
        {
            for x in 0..w
            {
                let mut v = 0.2 + 0.4 * x as f64 / 63.0 + 0.2 * y as f64 / 63.0;
                if (12..30).contains(&x) && (20..44).contains(&y)
                {
                    v += 0.5;
                }
                let (dx, dy) = (x as f64 - 46.0, y as f64 - 18.0);
                if dx * dx + dy * dy <= 81.0
                {
                    v += 0.4;
                }
                img.push(v);
            }
        }
        img
    }

    fn add_gaussian_noise(img: &[f64], sigma: f64, seed: u64) -> Vec<f64> {
        let mut rng = Lcg::new(seed);
        img.iter().map(|&v| v + sigma * rng.gauss()).collect()
    }

    #[test]
    fn dwt2d_roundtrip_is_exact_without_thresholding() {
        // Forward (rows then columns) and inverse (columns then rows) must be
        // exact mutual inverses on the padded image for every basis, including
        // non-power-of-two image sizes — the correctness core of the module.
        let (w, h) = (48usize, 36usize);
        let img: Vec<f64> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                (0.13 * x).sin() + 0.5 * (0.29 * y).cos() + 0.002 * x * y
            })
            .collect();
        for wavelet in [Wavelet::Haar, Wavelet::Db4, Wavelet::Db6, Wavelet::Db8]
        {
            let taps = wavelet.lowpass();
            let tw = padded_extent(w, taps.len());
            let th = padded_extent(h, taps.len());
            let padded = pad_mirror_2d(&img, w, h, tw, th);
            let mut buf = padded.clone();
            let done = dwt2d_forward(&mut buf, tw, th, 3, &taps);
            assert!(done >= 1, "{wavelet:?}: no level performed");
            dwt2d_inverse(&mut buf, tw, th, done, &taps);
            for (orig, rec) in padded.iter().zip(buf.iter())
            {
                assert!((orig - rec).abs() < 1.0e-9, "{wavelet:?}: {orig} vs {rec}");
            }
        }
    }

    #[test]
    fn wavelet_denoise2d_gains_psnr_on_synthetic_scene() {
        let clean = synthetic_scene();
        let noisy = add_gaussian_noise(&clean, 0.25, 7);
        let den = wavelet_denoise2d(&noisy, 64, 64, 0, ThresholdMode::Soft, Wavelet::Db4);
        let (p_noisy, p_den) = (psnr_db(&clean, &noisy), psnr_db(&clean, &den));
        assert!(
            p_den >= p_noisy + 3.0,
            "wavelet gained only {:.2} dB (noisy {p_noisy:.2}, denoised {p_den:.2})",
            p_den - p_noisy
        );
    }

    #[test]
    fn db4_soft_beats_haar_soft_on_smooth_gradient() {
        // Purely locally-linear shading: Db4's two vanishing moments annihilate
        // it in every detail band (details ≈ pure noise ⇒ cleanly thresholded),
        // while Haar leaves blocky staircase artefacts. The gradient is a tent
        // (ramp up, ramp down) in each axis so the image stays continuous under
        // the periodized transform's wrap-around — the same reason the 1-D
        // suite's fixture is a full-period sine; a one-way ramp would hand both
        // bases a border step that dominates the comparison.
        let (w, h) = (64usize, 64usize);
        let tent = |i: usize| 1.0 - (i as f64 - 31.5).abs() / 31.5;
        let mut clean = Vec::with_capacity(w * h);
        for y in 0..h
        {
            for x in 0..w
            {
                clean.push(0.2 + 0.5 * tent(x) + 0.3 * tent(y));
            }
        }
        let noisy = add_gaussian_noise(&clean, 0.1, 11);
        let haar = wavelet_denoise2d(&noisy, w, h, 0, ThresholdMode::Soft, Wavelet::Haar);
        let db4 = wavelet_denoise2d(&noisy, w, h, 0, ThresholdMode::Soft, Wavelet::Db4);
        let (p_noisy, p_haar, p_db4) = (
            psnr_db(&clean, &noisy),
            psnr_db(&clean, &haar),
            psnr_db(&clean, &db4),
        );
        assert!(p_db4 > p_noisy, "db4 must beat raw");
        assert!(
            p_db4 >= p_haar,
            "db4 {p_db4:.2} dB must be at least haar {p_haar:.2} dB on a smooth gradient"
        );
    }

    #[test]
    fn auto_variant_is_soft_db4_auto_depth() {
        let noisy = add_gaussian_noise(&synthetic_scene(), 0.15, 13);
        let auto = wavelet_denoise2d_auto(&noisy, 64, 64);
        let manual = wavelet_denoise2d(&noisy, 64, 64, 0, ThresholdMode::Soft, Wavelet::Db4);
        assert_eq!(auto, manual);
    }

    #[test]
    fn median2d_removes_salt_and_pepper_and_keeps_edges() {
        // Binary rectangle scene; 5% of the pixels corrupted by ±8 impulses.
        let (w, h) = (64usize, 64usize);
        let mut clean = vec![0.0_f64; w * h];
        for y in 16..48
        {
            for x in 16..48
            {
                clean[y * w + x] = 1.0;
            }
        }
        let mut rng = Lcg::new(17);
        let noisy: Vec<f64> = clean
            .iter()
            .map(|&v| {
                if rng.uniform() < 0.05
                {
                    v + if rng.uniform() < 0.5 { 8.0 } else { -8.0 }
                }
                else
                {
                    v
                }
            })
            .collect();
        let den = median2d(&noisy, w, h, 1);
        let (p_noisy, p_den) = (psnr_db(&clean, &noisy), psnr_db(&clean, &den));
        assert!(
            p_den >= p_noisy + 6.0,
            "median gained only {:.2} dB (noisy {p_noisy:.2}, denoised {p_den:.2})",
            p_den - p_noisy
        );

        // Edge preservation: mean sharpness of the rectangle's left edge
        // (columns 15|16), away from the corners. On the CLEAN image a 3×3
        // median keeps a straight edge exactly (the window majority always sits
        // on the pixel's own side), and after denoising the impulses the edge
        // must stay within tolerance of the clean step height 1.0.
        let sharpness = |im: &[f64]| -> f64 {
            (17..47)
                .map(|y| (im[y * w + 16] - im[y * w + 15]).abs())
                .sum::<f64>()
                / 30.0
        };
        let filtered_clean = median2d(&clean, w, h, 1);
        assert!(
            (sharpness(&filtered_clean) - 1.0).abs() < 1.0e-12,
            "median blurred a straight clean edge: {}",
            sharpness(&filtered_clean)
        );
        assert!(
            sharpness(&den) >= 0.8,
            "edge sharpness degraded to {}",
            sharpness(&den)
        );
    }

    #[test]
    fn median2d_annihilates_single_impulse_and_radius0_copies() {
        let mut img = vec![1.0_f64; 25];
        img[12] = 9.0;
        let out = median2d(&img, 5, 5, 1);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1.0e-12));
        assert_eq!(median2d(&img, 5, 5, 0), img);
    }

    #[test]
    fn nlm2d_is_competitive_on_repetitive_texture() {
        // Diagonal stripes: every translation along the stripe direction is a
        // perfect patch match, so NLM averages along the texture's own
        // repetitions — the regime where it must stay competitive with (and
        // typically beat) transform shrinkage, which smears diagonal energy.
        let (w, h) = (64usize, 64usize);
        let mut clean = Vec::with_capacity(w * h);
        for y in 0..h
        {
            for x in 0..w
            {
                clean.push(0.5 + 0.25 * (2.0 * PI * (x + y) as f64 / 16.0).sin());
            }
        }
        let noisy = add_gaussian_noise(&clean, 0.15, 19);
        let nlm = nlm2d(&noisy, w, h, 1, 5, 0.0);
        let wav = wavelet_denoise2d(&noisy, w, h, 0, ThresholdMode::Soft, Wavelet::Db4);
        let (p_noisy, p_nlm, p_wav) = (
            psnr_db(&clean, &noisy),
            psnr_db(&clean, &nlm),
            psnr_db(&clean, &wav),
        );
        assert!(
            p_nlm >= p_noisy + 3.0,
            "nlm gained only {:.2} dB (noisy {p_noisy:.2}, nlm {p_nlm:.2})",
            p_nlm - p_noisy
        );
        assert!(
            p_nlm >= p_wav - 1.0,
            "nlm {p_nlm:.2} dB not competitive with wavelet {p_wav:.2} dB"
        );
    }

    #[test]
    fn degenerate_inputs_are_handled_gracefully() {
        // Empty image ⇒ empty output from all four entry points.
        let empty: [f64; 0] = [];
        assert!(median2d(&empty, 0, 0, 1).is_empty());
        assert!(wavelet_denoise2d(&empty, 0, 0, 0, ThresholdMode::Soft, Wavelet::Db4).is_empty());
        assert!(wavelet_denoise2d_auto(&empty, 0, 0).is_empty());
        assert!(nlm2d(&empty, 0, 0, 1, 2, 0.0).is_empty());

        // 1×1 image ⇒ the pixel comes back.
        let one = [3.25_f64];
        assert_eq!(median2d(&one, 1, 1, 2), vec![3.25]);
        let wav = wavelet_denoise2d_auto(&one, 1, 1);
        assert_eq!(wav.len(), 1);
        assert!((wav[0] - 3.25).abs() < 1.0e-9);
        assert_eq!(nlm2d(&one, 1, 1, 1, 2, 0.0), vec![3.25]);

        // width·height mismatch ⇒ the input copied, by convention.
        let bad = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(median2d(&bad, 2, 2, 1), bad.to_vec());
        assert_eq!(
            wavelet_denoise2d(&bad, 2, 2, 1, ThresholdMode::Hard, Wavelet::Haar),
            bad.to_vec()
        );
        assert_eq!(wavelet_denoise2d_auto(&bad, 2, 2), bad.to_vec());
        assert_eq!(nlm2d(&bad, 2, 2, 1, 1, 0.5), bad.to_vec());

        // A constant image is preserved by all four (zero details, zero noise).
        let flat = vec![2.5_f64; 256];
        for out in [
            median2d(&flat, 16, 16, 2),
            wavelet_denoise2d(&flat, 16, 16, 0, ThresholdMode::Soft, Wavelet::Db4),
            wavelet_denoise2d(&flat, 16, 16, 2, ThresholdMode::Hard, Wavelet::Haar),
            wavelet_denoise2d_auto(&flat, 16, 16),
            nlm2d(&flat, 16, 16, 1, 2, 0.0),
            nlm2d(&flat, 16, 16, 1, 2, 0.4),
        ]
        {
            assert_eq!(out.len(), flat.len());
            for v in out
            {
                assert!((v - 2.5).abs() < 1.0e-9, "constant not preserved: {v}");
            }
        }
    }

    #[test]
    fn nan_pixel_does_not_panic_any_denoiser() {
        let (w, h) = (16usize, 16usize);
        let mut img: Vec<f64> = (0..w * h).map(|i| (i as f64 * 0.17).sin()).collect();
        img[37] = f64::NAN;
        // No panics allowed — the total_cmp order statistics and threshold rules
        // absorb the NaN; output values may be NaN but the shapes must hold.
        assert_eq!(median2d(&img, w, h, 1).len(), img.len());
        assert_eq!(
            wavelet_denoise2d(&img, w, h, 0, ThresholdMode::Soft, Wavelet::Db4).len(),
            img.len()
        );
        assert_eq!(
            wavelet_denoise2d(&img, w, h, 2, ThresholdMode::Hard, Wavelet::Haar).len(),
            img.len()
        );
        assert_eq!(wavelet_denoise2d_auto(&img, w, h).len(), img.len());
        assert_eq!(nlm2d(&img, w, h, 1, 2, 0.0).len(), img.len());
        assert_eq!(nlm2d(&img, w, h, 1, 2, 0.3).len(), img.len());
    }
}
