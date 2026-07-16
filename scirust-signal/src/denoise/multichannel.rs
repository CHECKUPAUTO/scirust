//! Multichannel operators that **genuinely couple channels** — Phase 2 of the TSHF
//! research roadmap (`TSHF_RESEARCH_2026-07-16.md`, §12 rec. 4 and "Feuille de route").
//!
//! The report's Proposition 1 (E5a) showed that componentwise filters gain *nothing*
//! from hypercomplex embeddings — any ℝ-linear filter applied per component is
//! algebraically identical to per-channel filtering. The only operators worth
//! evaluating are those whose output on one channel depends on the *other* channels:
//!
//! * [`vector_median`] — the L2 **vector median** of Astola, Haavisto & Neuvo,
//!   *"Vector median filters"*, Proc. IEEE 78(4):678–689 (1990): each output sample
//!   is the input *vector* of the window minimizing the sum of Euclidean distances
//!   to the other window vectors, so all channels are replaced coherently by one
//!   actually-observed sample.
//! * [`wiener_spatial`] — a **static spatial (joint cross-covariance) Wiener** gain
//!   `W = S·(S+N)⁻¹` over the channel dimension.
//!
//! ## Equivalence with quaternion widely-linear filtering
//!
//! The report's "quaternion track" names the widely-linear quaternion filters of
//! Cheong Took & Mandic (*The quaternion LMS algorithm…*, IEEE TSP 57(4), 2009;
//! *A quaternion widely linear adaptive filter*, IEEE TSP 58(8), 2010). For **real
//! 4-channel data** embedded in ℍ, a strictly quaternion-linear filter spans only a
//! 4-parameter subset of the real-linear maps on ℝ⁴; widely-linear filtering
//! augments the input with its three quaternion involutions precisely so that the
//! filter reaches *every* real-linear map ℝ⁴ → ℝ⁴ — at which point the MMSE solution
//! is, by definition, the ordinary joint (cross-covariance) Wiener estimator on the
//! real 4-vector. [`wiener_spatial`] therefore *is* the real-vector equivalent of
//! quaternion widely-linear Wiener filtering, without the algebraic detour; there is
//! no additional linear-MMSE performance hiding in the quaternion representation.
//!
//! ## Honest scope of [`wiener_spatial`]
//!
//! It is a **static spatial** operator: one C×C gain matrix for the whole record.
//! It exploits cross-channel correlation only — no temporal modelling whatsoever —
//! so it is *not* a replacement for the per-channel temporal denoisers of this
//! crate (which exploit the spectrum and routinely score far higher on smooth
//! signals). The honest gate comparison is against its own **diagonal restriction**
//! (identical estimator with the off-diagonal covariances zeroed — i.e. the same
//! static Wiener denied cross-channel information); numbers against the per-channel
//! *spectral* [`super::wiener_white`] are reported for context only.
//!
//! ## Phase 2 acceptance gate (measured, deterministic)
//!
//! The report's gate: an operator passes iff it beats its per-channel counterpart by
//! ≥ 0.5 dB (the report's §10 materiality threshold — "tout gain < 0,5 dB est déclaré
//! nul") on **at least 2 realistic multichannel fixtures**. [`phase2_gate_report`]
//! reruns the exact fixtures the tests assert on and returns every number, so the
//! evidence below is reproducible from library code:
//!
//! | fixture | joint / vector op | per-channel counterpart | context |
//! |---|---|---|---|
//! | 1. correlated smooth sources, C=4 | `wiener_spatial` **9.53 dB** | diagonal 7.05 dB | `wiener_white` 12.59 dB, raw 5.95 dB |
//! | 2. rank-1 stereo-like, C=3 | `wiener_spatial` **9.32 dB** | diagonal 5.65 dB | `wiener_white` 10.20 dB, raw 4.19 dB |
//! | 3. desynchronized impulses, C=3 | `vector_median` 24.17 dB | `median_filter` **26.19 dB** | raw −0.23 dB |
//! | 4. synchronized impulses, C=3 | `vector_median` 24.13 dB | `median_filter` **25.94 dB** | raw −4.97 dB |
//!
//! **Verdicts.** [`wiener_spatial`] **passes**: it beats its diagonal restriction by
//! +2.48 dB and +3.67 dB on fixtures 1 and 2 (both ≥ the fixtures' preset margins of
//! 0.5 and 1 dB). [`vector_median`] **fails**: it loses the SNR comparison on *both*
//! impulse fixtures — the synchronized loss (−1.81 dB) reproduces the report's E5b
//! finding, and even on desynchronized impulses, the use case the report singles out
//! as legitimate, it loses by −2.02 dB. The mechanism is structural: a vector median
//! outputs an *observed* sample vector, so the full background noise of the chosen
//! sample survives, while the scalar median averages that noise down. The same
//! mechanism defeats the classically claimed **correlation-preservation** advantage
//! on these fixtures — residual noise is exactly what shrinks pairwise Pearson
//! correlations, so the per-channel median's cleaner output preserves the clean
//! correlation structure *better* (mean absolute corr error 2.8e-3 vs the vector
//! median's 4.4e-3 desynchronized; 3.0e-3 vs 4.5e-3 synchronized). Measured, not
//! hoped: on this evidence [`vector_median`] should not be promoted past Phase 2.
//! Its one remaining property is a structural guarantee no per-channel filter
//! offers — every output vector is a sample that was actually observed (no
//! fabricated channel combinations), which can matter for direction/ratio-preserving
//! color or IMU processing; that guarantee is not the gate's criterion.
//!
//! ## Example
//!
//! ```
//! use scirust_signal::denoise::multichannel::{vector_median, wiener_spatial};
//!
//! // Three correlated channels; an impulse hits channel 0 only.
//! let mut chans: Vec<Vec<f64>> = (0..3)
//!     .map(|c| (0..64).map(|i| (c as f64 + 1.0) * (i as f64 * 0.1).sin()).collect())
//!     .collect();
//! chans[0][20] += 10.0;
//! let vm = vector_median(&chans, 2);
//! assert!((vm[0][20] - chans[0][20]).abs() > 5.0); // the outlier vector was replaced
//! let joint = wiener_spatial(&chans);
//! assert_eq!(joint.len(), 3);
//! assert_eq!(joint[0].len(), 64);
//! ```

use super::{estimate_noise_std_helper, median_filter, mirror_index, wiener_white};
use core::f64::consts::PI;

/// The report's §10 materiality threshold: a gain below 0.5 dB "est déclaré nul",
/// so "beats" in the Phase 2 gate means winning by at least this margin.
const GATE_MARGIN_DB: f64 = 0.5;

/// Record length shared by all gate fixtures.
const GATE_N: usize = 2048;

/// Maximum cyclic Jacobi sweeps; C ≤ 16 symmetric matrices converge in far fewer.
const JACOBI_MAX_SWEEPS: usize = 64;

// ---------------------------------------------------------------------------
// Vector median (Astola, Haavisto & Neuvo 1990)
// ---------------------------------------------------------------------------

/// **L2 vector median filter** (Astola, Haavisto & Neuvo, Proc. IEEE 78(4), 1990).
///
/// `channels` is one `Vec<f64>` per channel, all of the same length `n`; sample `i`
/// is the vector `x_i = (channels[0][i], …, channels[C-1][i])`. For every `i` the
/// window of vectors `x_j`, `j ∈ [i−h, i+h]` (borders mirrored with
/// [`super::mirror_index`], so border windows contain duplicated samples exactly like
/// the scalar rank filters), is searched for the candidate minimizing the sum of
/// Euclidean distances to the other window vectors — the *vector median*, the
/// sample-restricted geometric median. The output at `i` is that entire input
/// vector, so all channels are replaced **coherently** by one actually-observed
/// sample; this is what distinguishes it from `C` independent scalar medians.
///
/// Ties are broken by the earliest window position (deterministic). For `C = 1` and
/// the always-odd window `2h+1`, the minimizer of `Σ|x − x_b|` over the samples is
/// the middle order statistic, so the output equals [`super::median_filter`] exactly.
///
/// ## Degradation & NaN policy
///
/// * No channels, an empty channel, or mismatched channel lengths: the input is
///   returned as-is (`channels.to_vec()`), never a panic.
/// * A window candidate with any non-finite coordinate is treated as infinitely
///   distant (its distance sum is +∞ and it is excluded from the finite candidates'
///   sums), so finite candidates always win over corrupted ones.
/// * If **every** candidate in a window is non-finite, the center sample is kept
///   unchanged.
pub fn vector_median(channels: &[Vec<f64>], half_window: usize) -> Vec<Vec<f64>> {
    let c = channels.len();
    if c == 0
    {
        return channels.to_vec();
    }
    let n = channels[0].len();
    if n == 0 || channels.iter().any(|ch| ch.len() != n)
    {
        return channels.to_vec();
    }
    let w = 2 * half_window + 1;
    let h = half_window as isize;
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; c];
    // Per-window scratch: source index, flattened candidate vectors, finiteness.
    let mut idx = vec![0usize; w];
    let mut vecs = vec![0.0; w * c];
    let mut finite = vec![false; w];
    for i in 0..n
    {
        for (k, off) in (-h..=h).enumerate()
        {
            let j = mirror_index(i as isize + off, n);
            idx[k] = j;
            let mut ok = true;
            for (ch, chan) in channels.iter().enumerate()
            {
                let v = chan[j];
                vecs[k * c + ch] = v;
                ok &= v.is_finite();
            }
            finite[k] = ok;
        }
        // Medoid search among the finite candidates; window order + strict `<`
        // makes ties resolve to the earliest position, deterministically.
        let mut best: Option<(f64, usize)> = None;
        for a in 0..w
        {
            if !finite[a]
            {
                // Non-finite candidate ⇒ distance sum +∞: never beats a finite one.
                continue;
            }
            let mut sum = 0.0;
            for b in 0..w
            {
                if b == a || !finite[b]
                {
                    continue;
                }
                let mut d2 = 0.0;
                for ch in 0..c
                {
                    let diff = vecs[a * c + ch] - vecs[b * c + ch];
                    d2 += diff * diff;
                }
                sum += d2.sqrt();
            }
            if best.is_none_or(|(s, _)| sum < s)
            {
                best = Some((sum, a));
            }
        }
        match best
        {
            Some((_, a)) =>
            {
                let j = idx[a];
                for (ch, chan) in channels.iter().enumerate()
                {
                    out[ch][i] = chan[j];
                }
            },
            // Every candidate was non-finite: keep the center sample unchanged.
            None =>
            {
                for (ch, chan) in channels.iter().enumerate()
                {
                    out[ch][i] = chan[i];
                }
            },
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Static spatial (joint cross-covariance) Wiener
// ---------------------------------------------------------------------------

/// **Static spatial Wiener filter** over the channel dimension — the joint
/// cross-covariance estimator that quaternion *widely-linear* filtering (Took &
/// Mandic 2009–2010) reduces to on real 4-channel data (see the module docs for the
/// equivalence argument).
///
/// ## Estimator
///
/// 1. Per-channel noise scale `σ_c` via the robust MAD rule of
///    [`super::estimate_noise_std_helper`] → noise covariance `N = diag(σ_c²)`
///    (independent noise across channels is the model, matching the fixtures).
/// 2. Total covariance `T` of the sample vectors around their per-channel means
///    (`n−1` denominator; symmetric by construction).
/// 3. Signal covariance `S = T − N`, projected onto the positive-semidefinite cone
///    by a cyclic Jacobi eigendecomposition with negative eigenvalues clamped to 0
///    (`T − N` need not be PSD in finite samples).
/// 4. Gain `W = S·(S+N)⁻¹`, computed by Gauss–Jordan elimination with partial
///    pivoting on the tiny symmetric positive-semidefinite `C×C` system
///    (`W = ((S+N)⁻¹·S)ᵀ` since both matrices are symmetric).
/// 5. Output `y_t = m + W·(x_t − m)` with `m` the per-channel mean vector.
///
/// ## Honest scope
///
/// This is a **static spatial** Wiener: one gain matrix for the whole record, no
/// temporal modelling. It exploits cross-channel correlation *only* — it is the
/// coupling operator the Phase 2 roadmap asked to evaluate, **not** a replacement
/// for per-channel temporal denoisers (the spectral [`super::wiener_white`] scores
/// far higher on smooth narrowband fixtures; see the module-level gate table). Its
/// honest per-channel counterpart is its own diagonal restriction — the identical
/// estimator with the off-diagonal entries of `T` zeroed.
///
/// ## Degradation
///
/// No channels, `n < 2`, mismatched channel lengths, any non-finite input sample,
/// or a numerically singular `S + N` (e.g. all-constant channels): the input is
/// returned as-is (`channels.to_vec()`), never a panic.
pub fn wiener_spatial(channels: &[Vec<f64>]) -> Vec<Vec<f64>> {
    wiener_spatial_impl(channels, false)
}

/// Shared machinery of [`wiener_spatial`] and its diagonal (per-channel)
/// restriction: `diagonal_only` zeroes the off-diagonal entries of the total
/// covariance `T` before the *same* PSD projection and gain computation, so the
/// gate comparison isolates exactly the value of the cross-channel terms.
fn wiener_spatial_impl(channels: &[Vec<f64>], diagonal_only: bool) -> Vec<Vec<f64>> {
    let c = channels.len();
    if c == 0
    {
        return channels.to_vec();
    }
    let n = channels[0].len();
    if n < 2
        || channels.iter().any(|ch| ch.len() != n)
        || channels.iter().any(|ch| ch.iter().any(|v| !v.is_finite()))
    {
        return channels.to_vec();
    }

    let nf = n as f64;
    let mean: Vec<f64> = channels
        .iter()
        .map(|ch| ch.iter().sum::<f64>() / nf)
        .collect();

    // Total covariance T (n−1 denominator), symmetric by construction.
    let mut t = vec![0.0; c * c];
    for p in 0..c
    {
        for q in p..c
        {
            if diagonal_only && q != p
            {
                continue;
            }
            let acc: f64 = channels[p]
                .iter()
                .zip(channels[q].iter())
                .map(|(&xp, &xq)| (xp - mean[p]) * (xq - mean[q]))
                .sum();
            let cov = acc / (nf - 1.0);
            t[p * c + q] = cov;
            t[q * c + p] = cov;
        }
    }

    // Noise covariance N = diag(σ_c²); signal covariance S = PSD projection of T − N.
    let sigma2: Vec<f64> = channels
        .iter()
        .map(|ch| {
            let s = estimate_noise_std_helper(ch);
            s * s
        })
        .collect();
    let mut s = t;
    for ci in 0..c
    {
        s[ci * c + ci] -= sigma2[ci];
    }
    let s = psd_project(&s, c);

    // Gain W = S(S+N)⁻¹ = ((S+N)⁻¹S)ᵀ (both symmetric). Singular S+N ⇒ pass-through.
    let mut a = s.clone();
    for ci in 0..c
    {
        a[ci * c + ci] += sigma2[ci];
    }
    let Some(y) = solve_matrix(&a, &s, c)
    else
    {
        return channels.to_vec();
    };

    // y_t = m + W(x_t − m), with W[r][k] = y[k][r].
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; c];
    for i in 0..n
    {
        for r in 0..c
        {
            let mut acc = mean[r];
            for (k, chan) in channels.iter().enumerate()
            {
                acc += y[k * c + r] * (chan[i] - mean[k]);
            }
            out[r][i] = acc;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tiny symmetric linear algebra (C ≤ 16: everything is O(C³) with small constants)
// ---------------------------------------------------------------------------

/// Cyclic Jacobi eigendecomposition of a symmetric `c×c` matrix (row-major).
/// Returns `(eigenvalues, eigenvectors)` with eigenvector `k` stored in column `k`
/// of the returned row-major matrix (`v[r*c + k]`).
fn jacobi_eigen(mat: &[f64], c: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = mat.to_vec();
    let mut v = vec![0.0; c * c];
    for i in 0..c
    {
        v[i * c + i] = 1.0;
    }
    for _ in 0..JACOBI_MAX_SWEEPS
    {
        let mut off = 0.0;
        for p in 0..c
        {
            for q in (p + 1)..c
            {
                off += a[p * c + q] * a[p * c + q];
            }
        }
        let scale: f64 = (0..c)
            .map(|i| a[i * c + i].abs())
            .sum::<f64>()
            .max(1.0e-300);
        if off.sqrt() <= 1.0e-14 * scale
        {
            break;
        }
        for p in 0..c
        {
            for q in (p + 1)..c
            {
                let apq = a[p * c + q];
                if apq == 0.0
                {
                    continue;
                }
                // Classic stable rotation choice (Golub & Van Loan): zeroes a[p][q].
                let theta = (a[q * c + q] - a[p * c + p]) / (2.0 * apq);
                let tan = if theta >= 0.0
                {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                }
                else
                {
                    1.0 / (theta - (theta * theta + 1.0).sqrt())
                };
                let cos = 1.0 / (tan * tan + 1.0).sqrt();
                let sin = tan * cos;
                // A ← GᵀAG (columns then rows), V ← VG.
                for k in 0..c
                {
                    let akp = a[k * c + p];
                    let akq = a[k * c + q];
                    a[k * c + p] = cos * akp - sin * akq;
                    a[k * c + q] = sin * akp + cos * akq;
                }
                for k in 0..c
                {
                    let apk = a[p * c + k];
                    let aqk = a[q * c + k];
                    a[p * c + k] = cos * apk - sin * aqk;
                    a[q * c + k] = sin * apk + cos * aqk;
                }
                for k in 0..c
                {
                    let vkp = v[k * c + p];
                    let vkq = v[k * c + q];
                    v[k * c + p] = cos * vkp - sin * vkq;
                    v[k * c + q] = sin * vkp + cos * vkq;
                }
            }
        }
    }
    let eig: Vec<f64> = (0..c).map(|i| a[i * c + i]).collect();
    (eig, v)
}

/// Nearest positive-semidefinite matrix in the eigenbasis: eigendecompose the
/// symmetric input with [`jacobi_eigen`] and rebuild with negative eigenvalues
/// clamped to zero.
fn psd_project(mat: &[f64], c: usize) -> Vec<f64> {
    let (eig, v) = jacobi_eigen(mat, c);
    let mut out = vec![0.0; c * c];
    for (k, &lam) in eig.iter().enumerate()
    {
        if lam <= 0.0
        {
            continue;
        }
        for r in 0..c
        {
            for col in 0..c
            {
                out[r * c + col] += lam * v[r * c + k] * v[col * c + k];
            }
        }
    }
    out
}

/// Solve `A·X = B` for the `c×c` matrix `X` by Gauss–Jordan elimination with
/// partial pivoting. Returns `None` when a pivot is numerically zero (singular
/// system) so callers can degrade gracefully instead of dividing by ~0.
fn solve_matrix(a: &[f64], b: &[f64], c: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    let scale: f64 = a
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1.0e-300);
    for col in 0..c
    {
        let mut piv = col;
        for r in (col + 1)..c
        {
            if m[r * c + col].abs() > m[piv * c + col].abs()
            {
                piv = r;
            }
        }
        if m[piv * c + col].abs() <= 1.0e-12 * scale
        {
            return None;
        }
        if piv != col
        {
            for k in 0..c
            {
                m.swap(piv * c + k, col * c + k);
                x.swap(piv * c + k, col * c + k);
            }
        }
        let d = m[col * c + col];
        for r in 0..c
        {
            if r == col
            {
                continue;
            }
            let f = m[r * c + col] / d;
            if f == 0.0
            {
                continue;
            }
            for k in 0..c
            {
                m[r * c + k] -= f * m[col * c + k];
                x[r * c + k] -= f * x[col * c + k];
            }
        }
    }
    for col in 0..c
    {
        let d = m[col * c + col];
        for k in 0..c
        {
            x[col * c + k] /= d;
        }
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// Phase 2 gate: deterministic fixtures, measured report
// ---------------------------------------------------------------------------

/// The measured Phase 2 gate evidence — every number the module-level table and the
/// tests rely on, recomputed on demand from the deterministic fixtures.
///
/// All `*_db` fields are aggregate multichannel SNRs (total clean energy over total
/// error energy, in dB, against the known clean channels). The `*_corr_err` fields
/// are the mean absolute difference between the pairwise Pearson correlations of
/// the *output* channels and those of the *clean* channels — the vector median's
/// claimed advantage (cross-channel coherence), measured rather than assumed.
#[derive(Debug, Clone)]
pub struct MultichannelGateReport {
    /// Fixture 1 (correlated smooth sources, C=4): SNR of the noisy input.
    pub smooth_raw_db: f64,
    /// Fixture 1: joint [`wiener_spatial`].
    pub smooth_joint_db: f64,
    /// Fixture 1: the identical estimator restricted to its diagonal (per-channel).
    pub smooth_diagonal_db: f64,
    /// Fixture 1 context: per-channel spectral [`super::wiener_white`].
    pub smooth_per_channel_wiener_db: f64,
    /// Fixture 2 (rank-1 stereo-like, C=3): SNR of the noisy input.
    pub rank1_raw_db: f64,
    /// Fixture 2: joint [`wiener_spatial`].
    pub rank1_joint_db: f64,
    /// Fixture 2: diagonal restriction (per-channel).
    pub rank1_diagonal_db: f64,
    /// Fixture 2 context: per-channel spectral [`super::wiener_white`].
    pub rank1_per_channel_wiener_db: f64,
    /// Fixture 3 (desynchronized impulses, C=3): SNR of the noisy input.
    pub desync_raw_db: f64,
    /// Fixture 3: [`vector_median`] with `h = 2`.
    pub desync_vector_median_db: f64,
    /// Fixture 3: per-channel [`super::median_filter`] with `h = 2`.
    pub desync_per_channel_median_db: f64,
    /// Fixture 3: pairwise-correlation error of the vector median output.
    pub desync_vector_median_corr_err: f64,
    /// Fixture 3: pairwise-correlation error of the per-channel median output.
    pub desync_per_channel_median_corr_err: f64,
    /// Fixture 4 (synchronized impulses — the report's E5b case): raw SNR.
    pub sync_raw_db: f64,
    /// Fixture 4: [`vector_median`] with `h = 2`.
    pub sync_vector_median_db: f64,
    /// Fixture 4: per-channel [`super::median_filter`] with `h = 2`.
    pub sync_per_channel_median_db: f64,
    /// Fixture 4: pairwise-correlation error of the vector median output.
    pub sync_vector_median_corr_err: f64,
    /// Fixture 4: pairwise-correlation error of the per-channel median output.
    pub sync_per_channel_median_corr_err: f64,
    /// Gate verdict: [`wiener_spatial`] beats its per-channel (diagonal) counterpart
    /// by ≥ 0.5 dB on ≥ 2 fixtures.
    pub wiener_spatial_passes: bool,
    /// Gate verdict: [`vector_median`] beats the per-channel median by ≥ 0.5 dB on
    /// ≥ 2 fixtures.
    pub vector_median_passes: bool,
}

/// Run the deterministic Phase 2 gate fixtures and package every measured number —
/// the same code path the unit tests assert on, so the module documentation's gate
/// table is reproducible by calling this function.
pub fn phase2_gate_report() -> MultichannelGateReport {
    // Fixture 1: correlated smooth sources, C = 4.
    let f1 = fixture_smooth_sources();
    let smooth_raw_db = snr_db_multi(&f1.clean, &f1.noisy);
    let smooth_joint_db = snr_db_multi(&f1.clean, &wiener_spatial(&f1.noisy));
    let smooth_diagonal_db = snr_db_multi(&f1.clean, &wiener_spatial_impl(&f1.noisy, true));
    let smooth_per_channel_wiener_db = snr_db_multi(&f1.clean, &per_channel_wiener(&f1.noisy));

    // Fixture 2: rank-1 strongly correlated, C = 3.
    let f2 = fixture_rank1();
    let rank1_raw_db = snr_db_multi(&f2.clean, &f2.noisy);
    let rank1_joint_db = snr_db_multi(&f2.clean, &wiener_spatial(&f2.noisy));
    let rank1_diagonal_db = snr_db_multi(&f2.clean, &wiener_spatial_impl(&f2.noisy, true));
    let rank1_per_channel_wiener_db = snr_db_multi(&f2.clean, &per_channel_wiener(&f2.noisy));

    // Fixtures 3 & 4: impulses, desynchronized (rotating channel) vs synchronized.
    let f3 = fixture_impulses(false);
    let desync_raw_db = snr_db_multi(&f3.clean, &f3.noisy);
    let vm3 = vector_median(&f3.noisy, 2);
    let pc3 = per_channel_median(&f3.noisy, 2);
    let desync_vector_median_db = snr_db_multi(&f3.clean, &vm3);
    let desync_per_channel_median_db = snr_db_multi(&f3.clean, &pc3);
    let desync_vector_median_corr_err = corr_preservation_error(&f3.clean, &vm3);
    let desync_per_channel_median_corr_err = corr_preservation_error(&f3.clean, &pc3);

    let f4 = fixture_impulses(true);
    let sync_raw_db = snr_db_multi(&f4.clean, &f4.noisy);
    let vm4 = vector_median(&f4.noisy, 2);
    let pc4 = per_channel_median(&f4.noisy, 2);
    let sync_vector_median_db = snr_db_multi(&f4.clean, &vm4);
    let sync_per_channel_median_db = snr_db_multi(&f4.clean, &pc4);
    let sync_vector_median_corr_err = corr_preservation_error(&f4.clean, &vm4);
    let sync_per_channel_median_corr_err = corr_preservation_error(&f4.clean, &pc4);

    // Gate: beat the per-channel counterpart by ≥ 0.5 dB on ≥ 2 fixtures.
    let wiener_wins = usize::from(smooth_joint_db >= smooth_diagonal_db + GATE_MARGIN_DB)
        + usize::from(rank1_joint_db >= rank1_diagonal_db + GATE_MARGIN_DB);
    let vm_wins =
        usize::from(desync_vector_median_db >= desync_per_channel_median_db + GATE_MARGIN_DB)
            + usize::from(sync_vector_median_db >= sync_per_channel_median_db + GATE_MARGIN_DB);

    MultichannelGateReport {
        smooth_raw_db,
        smooth_joint_db,
        smooth_diagonal_db,
        smooth_per_channel_wiener_db,
        rank1_raw_db,
        rank1_joint_db,
        rank1_diagonal_db,
        rank1_per_channel_wiener_db,
        desync_raw_db,
        desync_vector_median_db,
        desync_per_channel_median_db,
        desync_vector_median_corr_err,
        desync_per_channel_median_corr_err,
        sync_raw_db,
        sync_vector_median_db,
        sync_per_channel_median_db,
        sync_vector_median_corr_err,
        sync_per_channel_median_corr_err,
        wiener_spatial_passes: wiener_wins >= 2,
        vector_median_passes: vm_wins >= 2,
    }
}

/// A gate fixture: known clean channels and their deterministic noisy observation.
struct GateFixture {
    clean: Vec<Vec<f64>>,
    noisy: Vec<Vec<f64>>,
}

/// Deterministic 64-bit LCG (same constants as the test-only `testutil::Lcg`,
/// duplicated here because [`phase2_gate_report`] must run in non-test builds).
struct GateRng(u64);

impl GateRng {
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

/// Fixture 1 — **correlated smooth sources** (IMU / color-like), C = 4: two latent
/// smooth sources (sin, 3 cycles; cos, 7 cycles over n = 2048) mixed by a fixed 4×2
/// matrix, plus independent Gaussian noise σ = 0.35 per channel.
fn fixture_smooth_sources() -> GateFixture {
    let n = GATE_N;
    let mix: [[f64; 2]; 4] = [[1.0, 0.3], [0.8, -0.5], [0.2, 1.0], [-0.6, 0.7]];
    let clean: Vec<Vec<f64>> = mix
        .iter()
        .map(|row| {
            (0..n)
                .map(|i| {
                    let t = i as f64 / n as f64;
                    let s1 = (2.0 * PI * 3.0 * t).sin();
                    let s2 = (2.0 * PI * 7.0 * t).cos();
                    row[0] * s1 + row[1] * s2
                })
                .collect()
        })
        .collect();
    let noisy = add_gaussian(&clean, 0.35, 0xF1);
    GateFixture { clean, noisy }
}

/// Fixture 2 — **rank-1 strongly correlated** (stereo-like), C = 3: one latent
/// source (sin, 5 cycles) mixed by `[1, 0.9, −0.8]`, independent noise σ = 0.4.
/// Rank-1 signal covariance is the best case for spatial coupling.
fn fixture_rank1() -> GateFixture {
    let n = GATE_N;
    let mix = [1.0, 0.9, -0.8];
    let clean: Vec<Vec<f64>> = mix
        .iter()
        .map(|&a| {
            (0..n)
                .map(|i| a * (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
                .collect()
        })
        .collect();
    let noisy = add_gaussian(&clean, 0.4, 0xF2);
    GateFixture { clean, noisy }
}

/// Fixtures 3 & 4 — **impulsive**, C = 3: smooth correlated channels (one shared
/// source scaled per channel) plus small independent Gaussian noise (σ = 0.05) and
/// ±6 impulses every 37 samples. `synchronized = false` (fixture 3) hits **one
/// channel at a time**, rotating the channel every hit — the desynchronized case the
/// report singles out as the vector median's legitimate use case; `synchronized =
/// true` (fixture 4) hits **all channels at once** — the report's E5b case, where
/// the vector median is expected to lose.
fn fixture_impulses(synchronized: bool) -> GateFixture {
    let n = GATE_N;
    let amps = [1.0, 0.7, -0.6];
    let clean: Vec<Vec<f64>> = amps
        .iter()
        .map(|&a| {
            (0..n)
                .map(|i| a * (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
                .collect()
        })
        .collect();
    let mut noisy = add_gaussian(&clean, 0.05, 0xF3);
    let mut hit = 0usize;
    let mut i = 37;
    while i < n
    {
        let sign = if hit.is_multiple_of(2) { 6.0 } else { -6.0 };
        if synchronized
        {
            for ch in &mut noisy
            {
                ch[i] += sign;
            }
        }
        else
        {
            noisy[hit % 3][i] += sign;
        }
        hit += 1;
        i += 37;
    }
    GateFixture { clean, noisy }
}

/// Add independent Gaussian noise of standard deviation `sigma` to every channel
/// (one deterministic stream per fixture, consumed channel-major).
fn add_gaussian(clean: &[Vec<f64>], sigma: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = GateRng::new(seed);
    clean
        .iter()
        .map(|ch| ch.iter().map(|&v| v + sigma * rng.gauss()).collect())
        .collect()
}

/// Per-channel spectral Wiener baseline: [`super::wiener_white`] on each channel
/// with its own MAD noise estimate — the context column of the gate table.
fn per_channel_wiener(channels: &[Vec<f64>]) -> Vec<Vec<f64>> {
    channels
        .iter()
        .map(|ch| wiener_white(ch, estimate_noise_std_helper(ch)))
        .collect()
}

/// Per-channel median baseline: [`super::median_filter`] on each channel.
fn per_channel_median(channels: &[Vec<f64>], half_window: usize) -> Vec<Vec<f64>> {
    channels
        .iter()
        .map(|ch| median_filter(ch, half_window))
        .collect()
}

/// Aggregate multichannel SNR in dB: total clean energy over total error energy.
fn snr_db_multi(clean: &[Vec<f64>], est: &[Vec<f64>]) -> f64 {
    let mut sig = 0.0;
    let mut err = 0.0;
    for (cch, ech) in clean.iter().zip(est.iter())
    {
        for (&cv, &ev) in cch.iter().zip(ech.iter())
        {
            sig += cv * cv;
            err += (cv - ev) * (cv - ev);
        }
    }
    10.0 * (sig / err.max(1.0e-30)).log10()
}

/// Pearson correlation of two equal-length series; 0.0 when either is degenerate.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2
    {
        return 0.0;
    }
    let nf = n as f64;
    let ma = a[..n].iter().sum::<f64>() / nf;
    let mb = b[..n].iter().sum::<f64>() / nf;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for i in 0..n
    {
        let da = a[i] - ma;
        let db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0
    {
        return 0.0;
    }
    sab / (saa * sbb).sqrt()
}

/// Cross-channel correlation preservation: mean absolute difference between the
/// pairwise Pearson correlations of the estimate's channels and the clean channels'
/// — the vector median's claimed advantage, measured.
fn corr_preservation_error(clean: &[Vec<f64>], est: &[Vec<f64>]) -> f64 {
    let c = clean.len().min(est.len());
    if c < 2
    {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut pairs = 0usize;
    for p in 0..c
    {
        for q in (p + 1)..c
        {
            acc += (pearson(&est[p], &est[q]) - pearson(&clean[p], &clean[q])).abs();
            pairs += 1;
        }
    }
    acc / pairs as f64
}

#[cfg(test)]
mod tests {
    use super::super::testutil::Lcg;
    use super::*;

    #[test]
    fn gate_fixture_1_smooth_sources_joint_beats_diagonal() {
        // Two latent sources mixed into 4 channels: the cross-channel covariance
        // carries real information (the signal lives in a 2-dim subspace of R^4),
        // so the joint gain must beat the identical estimator denied it.
        let r = phase2_gate_report();
        assert!(
            r.smooth_joint_db >= r.smooth_diagonal_db + 0.5,
            "fixture 1: joint {:.2} dB must beat diagonal {:.2} dB by >= 0.5 dB",
            r.smooth_joint_db,
            r.smooth_diagonal_db
        );
        // Context numbers are reported, not asserted directionally (the module docs
        // explain why the temporal spectral Wiener wins on smooth narrowband
        // fixtures) — but they must be finite and the joint gain must denoise.
        assert!(r.smooth_per_channel_wiener_db.is_finite());
        assert!(
            r.smooth_joint_db > r.smooth_raw_db + 2.0,
            "fixture 1: joint {:.2} dB vs raw {:.2} dB",
            r.smooth_joint_db,
            r.smooth_raw_db
        );
    }

    #[test]
    fn gate_fixture_2_rank1_joint_beats_diagonal_by_1db() {
        // Rank-1 signal covariance is the best case for spatial coupling: the joint
        // gain projects onto the single signal direction, averaging noise across
        // all three channels.
        let r = phase2_gate_report();
        assert!(
            r.rank1_joint_db >= r.rank1_diagonal_db + 1.0,
            "fixture 2: joint {:.2} dB must beat diagonal {:.2} dB by >= 1 dB",
            r.rank1_joint_db,
            r.rank1_diagonal_db
        );
        assert!(r.rank1_per_channel_wiener_db.is_finite());
        assert!(
            r.rank1_joint_db > r.rank1_raw_db + 2.0,
            "fixture 2: joint {:.2} dB vs raw {:.2} dB",
            r.rank1_joint_db,
            r.rank1_raw_db
        );
    }

    #[test]
    fn gate_fixture_3_desynchronized_impulses_honest_direction() {
        // The report singles out desynchronized impulses as the vector median's
        // legitimate use case. MEASURED on this fixture, the per-channel median
        // still wins (26.19 dB vs 24.17 dB): the vector median outputs an observed
        // sample vector, so the chosen sample's background noise survives intact,
        // while the scalar median averages it down. This test asserts the direction
        // that is TRUE, not the one that was hoped for.
        let r = phase2_gate_report();
        assert!(
            r.desync_per_channel_median_db >= r.desync_vector_median_db + 1.0,
            "fixture 3 measured direction changed: per-channel {:.2} dB vs vector {:.2} dB",
            r.desync_per_channel_median_db,
            r.desync_vector_median_db
        );
        // Both filters must still remove the impulses convincingly.
        assert!(
            r.desync_vector_median_db > r.desync_raw_db + 15.0,
            "vector median failed to remove impulses: {:.2} dB vs raw {:.2} dB",
            r.desync_vector_median_db,
            r.desync_raw_db
        );
        // Correlation preservation — the vector median's classically claimed
        // advantage — also fails to materialize here, for the same reason (residual
        // noise is what shrinks pairwise correlations). Both stay near-clean.
        assert!(
            r.desync_per_channel_median_corr_err < r.desync_vector_median_corr_err,
            "corr preservation direction changed: per-channel {:.2e} vs vector {:.2e}",
            r.desync_per_channel_median_corr_err,
            r.desync_vector_median_corr_err
        );
        assert!(r.desync_vector_median_corr_err < 0.02);
        assert!(r.desync_per_channel_median_corr_err < 0.02);
    }

    #[test]
    fn gate_fixture_4_synchronized_impulses_vector_median_loses() {
        // The report's E5b case: correlated impulses hitting all channels at once.
        // E5b measured the vector median losing by ~1.9 dB; this fixture reproduces
        // the direction (−1.81 dB) — the documented honest negative.
        let r = phase2_gate_report();
        assert!(
            r.sync_per_channel_median_db >= r.sync_vector_median_db + 1.0,
            "fixture 4 (E5b) direction changed: per-channel {:.2} dB vs vector {:.2} dB",
            r.sync_per_channel_median_db,
            r.sync_vector_median_db
        );
        assert!(
            r.sync_vector_median_db > r.sync_raw_db + 15.0,
            "vector median failed to remove impulses: {:.2} dB vs raw {:.2} dB",
            r.sync_vector_median_db,
            r.sync_raw_db
        );
        assert!(r.sync_vector_median_corr_err < 0.02);
        assert!(r.sync_per_channel_median_corr_err < 0.02);
    }

    #[test]
    fn gate_verdicts_are_consistent_and_honest() {
        // The packaged booleans must agree with the raw dB fields under the report's
        // >= 0.5 dB materiality margin, and must state the measured outcome:
        // wiener_spatial passes Phase 2 (2/2 fixtures), vector_median fails (0/2).
        let r = phase2_gate_report();
        let wiener_wins = usize::from(r.smooth_joint_db >= r.smooth_diagonal_db + 0.5)
            + usize::from(r.rank1_joint_db >= r.rank1_diagonal_db + 0.5);
        let vm_wins =
            usize::from(r.desync_vector_median_db >= r.desync_per_channel_median_db + 0.5)
                + usize::from(r.sync_vector_median_db >= r.sync_per_channel_median_db + 0.5);
        assert_eq!(r.wiener_spatial_passes, wiener_wins >= 2);
        assert_eq!(r.vector_median_passes, vm_wins >= 2);
        assert!(r.wiener_spatial_passes, "wiener_spatial must pass the gate");
        assert!(
            !r.vector_median_passes,
            "vector_median passing would contradict the measured fixtures — update \
             the module docs before flipping this"
        );
    }

    #[test]
    fn vector_median_equals_scalar_median_filter_for_one_channel() {
        // For C = 1 the L2 vector median minimizes Σ|x − x_b| over the window
        // samples, i.e. it is the scalar medoid — which for the always-odd window
        // 2h+1 is the middle order statistic, exactly median_filter's output.
        let mut rng = Lcg::new(41);
        let signal: Vec<f64> = (0..64).map(|_| rng.gauss()).collect();
        for h in 1..=3_usize
        {
            let vm = vector_median(std::slice::from_ref(&signal), h);
            assert_eq!(
                vm[0],
                median_filter(&signal, h),
                "C=1 vector median != median_filter for h = {h}"
            );
        }
    }

    #[test]
    fn degenerate_inputs_return_input_clone() {
        // Module convention: degenerate inputs come back unchanged, never panic.
        let empty: Vec<Vec<f64>> = Vec::new();
        assert!(vector_median(&empty, 2).is_empty());
        assert!(wiener_spatial(&empty).is_empty());

        let zero_len = vec![Vec::<f64>::new(), Vec::<f64>::new()];
        assert_eq!(vector_median(&zero_len, 2), zero_len);
        assert_eq!(wiener_spatial(&zero_len), zero_len);

        let mismatched = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        assert_eq!(vector_median(&mismatched, 1), mismatched);
        assert_eq!(wiener_spatial(&mismatched), mismatched);

        // half_window = 0: the window is the sample itself — identity.
        let chans = vec![vec![1.0, -2.0, 3.0], vec![0.5, 0.25, -0.125]];
        assert_eq!(vector_median(&chans, 0), chans);

        // n < 2: no covariance to estimate — pass-through.
        let single = vec![vec![1.0], vec![2.0]];
        assert_eq!(wiener_spatial(&single), single);

        // Non-finite input: wiener_spatial degrades to a pass-through clone.
        let mut nanned = chans.clone();
        nanned[1][1] = f64::NAN;
        let out = wiener_spatial(&nanned);
        assert_eq!(out[0], nanned[0]);
        assert!(out[1][1].is_nan());

        // All-constant channels: T = N = 0 makes S + N singular — pass-through.
        let flat = vec![vec![3.5; 32], vec![-1.0; 32]];
        assert_eq!(wiener_spatial(&flat), flat);
    }

    #[test]
    fn vector_median_quarantines_nan_candidates() {
        let mut rng = Lcg::new(43);
        let n = 64;
        let mut chans: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..n).map(|_| rng.gauss()).collect())
            .collect();
        // A single NaN coordinate poisons only its own candidate vector: no output
        // sample may come from index 30, and everything stays finite.
        chans[1][30] = f64::NAN;
        let out = vector_median(&chans, 2);
        for (ch, chan) in out.iter().enumerate()
        {
            for (i, v) in chan.iter().enumerate()
            {
                assert!(v.is_finite(), "channel {ch} sample {i} is {v}");
            }
        }
        // A window whose candidates are ALL non-finite keeps the center sample:
        // with h = 1 and NaNs at 10..=12 on channel 0, index 11 sees only poisoned
        // candidates. Channel 0 keeps its NaN, channel 1 keeps its original value.
        chans[0][10] = f64::NAN;
        chans[0][11] = f64::NAN;
        chans[0][12] = f64::NAN;
        let out = vector_median(&chans, 1);
        assert!(out[0][11].is_nan());
        assert_eq!(out[1][11], chans[1][11]);
        assert_eq!(out[2][11], chans[2][11]);
    }

    #[test]
    fn deterministic_two_runs_bit_identical() {
        let f = fixture_impulses(false);
        assert_eq!(vector_median(&f.noisy, 2), vector_median(&f.noisy, 2));
        let g = fixture_rank1();
        assert_eq!(wiener_spatial(&g.noisy), wiener_spatial(&g.noisy));
    }

    #[test]
    fn wiener_spatial_shrinks_pure_noise() {
        // With no signal, S = psd(T − N) is a small sampling-fluctuation matrix, so
        // the gain must pull the output strongly toward the channel means: centered
        // output energy far below centered input energy, on C = 3 and on C = 1.
        let mut rng = Lcg::new(47);
        for c in [3_usize, 1]
        {
            let chans: Vec<Vec<f64>> = (0..c)
                .map(|_| (0..1024).map(|_| rng.gauss()).collect())
                .collect();
            let out = wiener_spatial(&chans);
            assert_eq!(out.len(), c);
            let centered_energy = |chs: &[Vec<f64>]| -> f64 {
                chs.iter()
                    .map(|ch| {
                        let m = ch.iter().sum::<f64>() / ch.len() as f64;
                        ch.iter().map(|&v| (v - m) * (v - m)).sum::<f64>()
                    })
                    .sum()
            };
            let e_in = centered_energy(&chans);
            let e_out = centered_energy(&out);
            assert!(out.iter().flatten().all(|v| v.is_finite()));
            assert!(
                e_out < 0.5 * e_in,
                "C = {c}: pure-noise energy must shrink: in {e_in:.1}, out {e_out:.1}"
            );
        }
    }
}
