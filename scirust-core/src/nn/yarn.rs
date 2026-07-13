//! **YaRN — Yet another RoPE extensioN** (Peng et al., 2023, arXiv:2309.00071).
//!
//! Rotary position embeddings (RoPE) rotate the `p`-th 2-D sub-vector of a query
//! /key by an angle `position · θ_p`, with `θ_p = base^(−2p/d)`. A model trained
//! at context length `L` only ever sees positions `0..L`, so the *low-frequency*
//! dimensions (long wavelength) never complete a rotation; pushing inference past
//! `L` drives those angles out of distribution and the model degrades.
//!
//! YaRN extends the usable context by a factor `s` with **"NTK-by-parts"**
//! interpolation: rather than stretching *every* wavelength by `s` (plain
//! positional interpolation, which crushes the high-frequency dimensions the model
//! relies on for local order), it stretches **only** the low-frequency dimensions.
//! Each dimension is classified by its number of rotations over the original
//! context, `r_p = L · θ_p / 2π`:
//! - `r_p > β` (high frequency): **keep** `θ_p` — local positional detail is
//!   preserved;
//! - `r_p < α` (low frequency): **fully interpolate**, `θ_p → θ_p / s` — the
//!   angle at the *extended* length `s·L` lands exactly where it was at `L`;
//! - in between: a linear ramp `γ` blends the two,
//!   `θ'_p = θ_p · ((1 − γ)/s + γ)`.
//!
//! Crucially the transform is still a per-dimension rotation linear in position,
//! so RoPE's defining **relative-position property** — `⟨RoPE(q, m), RoPE(k, n)⟩`
//! depends only on `m − n` — is preserved exactly (tested below), while the
//! out-of-distribution angle blow-up is removed (also tested). Everything is pure,
//! deterministic arithmetic. The convention (interleaved pairs `(2p, 2p+1)`, angle
//! `pos·θ_p`) matches scirust's existing [`crate::autodiff::nd`] RoPE op, so
//! `scale = 1` reproduces plain RoPE.

use core::f32::consts::PI;

/// Default low/high-frequency thresholds (rotations over the original context) for
/// the NTK-by-parts ramp — the values from the YaRN paper.
pub const YARN_ALPHA: f32 = 1.0;
/// See [`YARN_ALPHA`].
pub const YARN_BETA: f32 = 32.0;

/// The `d/2` per-pair RoPE frequencies after **YaRN NTK-by-parts** interpolation
/// for a context-extension factor `scale` (`s ≥ 1`) and original training context
/// `orig_ctx` (`L`). `alpha`/`beta` are the ramp thresholds on the rotation count
/// `r_p = L·θ_p/2π` (use [`YARN_ALPHA`]/[`YARN_BETA`]). With `scale == 1` this
/// returns the plain RoPE frequencies `θ_p = base^(−2p/d)`.
pub fn yarn_frequencies(
    d: usize,
    base: f32,
    scale: f32,
    orig_ctx: f32,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    assert!(d.is_multiple_of(2), "yarn: d must be even");
    assert!(scale >= 1.0 && orig_ctx > 0.0, "yarn: need scale>=1, L>0");
    (0..d / 2)
        .map(|p| {
            let theta = base.powf(-2.0 * p as f32 / d as f32);
            // Rotations of this dimension over the original context.
            let rotations = orig_ctx * theta / (2.0 * PI);
            // Ramp γ: 0 (low freq, fully interpolate) → 1 (high freq, keep).
            let gamma = ((rotations - alpha) / (beta - alpha)).clamp(0.0, 1.0);
            // Blend full interpolation (θ/s) with no interpolation (θ).
            theta * ((1.0 - gamma) / scale + gamma)
        })
        .collect()
}

/// **Attention temperature** `1/t` recommended by YaRN to counter the entropy
/// change from longer contexts: queries/keys (equivalently the logits) are scaled
/// by `m = 0.1·ln(s) + 1` for `s > 1` (`1` when `s == 1`). Optional; orthogonal to
/// the rotation and to the relative-position property.
pub fn yarn_attention_scale(scale: f32) -> f32 {
    if scale <= 1.0
    {
        1.0
    }
    else
    {
        0.1 * scale.ln() + 1.0
    }
}

/// Apply a RoPE rotation to a `(seq, d)` row-major buffer using **explicit**
/// per-pair frequencies `freqs` (length `d/2`, e.g. from [`yarn_frequencies`]).
/// Pair `p` rotates `(x[2p], x[2p+1])` by `pos · freqs[p]`. Returns the rotated
/// copy; pure and deterministic.
pub fn rope_apply_freqs(x: &[f32], seq: usize, d: usize, freqs: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), seq * d, "rope_apply_freqs: x must be seq*d");
    assert_eq!(freqs.len(), d / 2, "rope_apply_freqs: freqs must be d/2");
    let mut out = vec![0.0f32; x.len()];
    for s in 0..seq
    {
        let row = s * d;
        for p in 0..d / 2
        {
            let (sin, cos) = (s as f32 * freqs[p]).sin_cos();
            let (a, b) = (x[row + 2 * p], x[row + 2 * p + 1]);
            out[row + 2 * p] = a * cos - b * sin;
            out[row + 2 * p + 1] = a * sin + b * cos;
        }
    }
    out
}

/// Convenience: YaRN-rotate a `(seq, d)` buffer with extension factor `scale` and
/// original context `orig_ctx`, using the default [`YARN_ALPHA`]/[`YARN_BETA`]
/// ramp and `base = 10000`. `scale = 1` reproduces plain RoPE.
pub fn rope_yarn(x: &[f32], seq: usize, d: usize, scale: f32, orig_ctx: f32) -> Vec<f32> {
    let freqs = yarn_frequencies(d, 10000.0, scale, orig_ctx, YARN_ALPHA, YARN_BETA);
    rope_apply_freqs(x, seq, d, &freqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    /// `scale = 1` reduces to plain RoPE frequencies `θ_p = base^(−2p/d)`.
    #[test]
    fn scale_one_is_plain_rope() {
        let d = 16;
        let f = yarn_frequencies(d, 10000.0, 1.0, 2048.0, YARN_ALPHA, YARN_BETA);
        for (p, &fp) in f.iter().enumerate()
        {
            let theta = 10000f32.powf(-2.0 * p as f32 / d as f32);
            assert!((fp - theta).abs() < 1e-7, "pair {p}: {fp} ≠ {theta}");
        }
    }

    /// NTK-by-parts boundaries: the highest-frequency pair is left untouched, the
    /// lowest-frequency pair is fully interpolated (`θ/s`), and the schedule is
    /// monotone non-increasing in `θ'/θ` from high to low frequency.
    #[test]
    fn ntk_by_parts_keeps_high_freq_interpolates_low() {
        let (d, base, scale, l) = (64usize, 10000.0f32, 8.0f32, 2048.0f32);
        let f = yarn_frequencies(d, base, scale, l, YARN_ALPHA, YARN_BETA);
        let theta = |p: usize| base.powf(-2.0 * p as f32 / d as f32);
        // Highest frequency (p = 0): unchanged.
        assert!(
            (f[0] / theta(0) - 1.0).abs() < 1e-4,
            "high freq interpolated"
        );
        // Lowest frequency (last pair): fully interpolated by 1/s.
        let last = d / 2 - 1;
        assert!(
            (f[last] / theta(last) - 1.0 / scale).abs() < 1e-3,
            "low freq not fully interpolated"
        );
        // Ratio θ'/θ is monotone (decreasing) from high to low frequency.
        let ratios: Vec<f32> = (0..d / 2).map(|p| f[p] / theta(p)).collect();
        for w in ratios.windows(2)
        {
            assert!(w[1] <= w[0] + 1e-6, "ratio not monotone: {w:?}");
        }
    }

    /// **RoPE's relative-position property survives YaRN.** For fixed `q`, `k` the
    /// inner product `⟨rope(q, m), rope(k, n)⟩` depends *only* on the offset
    /// `m − n`: sliding both positions by the same amount leaves it unchanged.
    #[test]
    fn relative_position_invariance() {
        let d = 32;
        let freqs = yarn_frequencies(d, 10000.0, 4.0, 2048.0, YARN_ALPHA, YARN_BETA);
        let q: Vec<f32> = (0..d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
        let k: Vec<f32> = (0..d).map(|i| (i as f32 * 0.2 + 0.4).cos()).collect();
        let rotate = |v: &[f32], pos: usize| {
            // One-token buffer rotated to absolute position `pos`.
            let mut buf = vec![0.0f32; (pos + 1) * d];
            buf[pos * d..(pos + 1) * d].copy_from_slice(v);
            let r = rope_apply_freqs(&buf, pos + 1, d, &freqs);
            r[pos * d..(pos + 1) * d].to_vec()
        };
        // Same offset δ at different absolute positions ⇒ same score.
        for &delta in &[0usize, 1, 3, 7]
        {
            let mut ref_score: Option<f32> = None;
            for base_pos in 0..6
            {
                let m = base_pos + delta;
                let n = base_pos;
                let score = dot(&rotate(&q, m), &rotate(&k, n));
                match ref_score
                {
                    None => ref_score = Some(score),
                    Some(r) => assert!(
                        (score - r).abs() < 1e-4,
                        "offset {delta}: score {score} ≠ {r} at base {base_pos}"
                    ),
                }
            }
        }
    }

    /// **Context extension, the point of YaRN.** Under plain RoPE the angle of a
    /// low-frequency dimension at the *extended* length `s·L` is `s×` larger than
    /// anything seen in training (out of distribution); YaRN's interpolation maps
    /// it back to exactly the angle that dimension had at the original length `L`.
    #[test]
    fn extension_keeps_low_freq_angle_in_distribution() {
        let (d, base, scale, l) = (64usize, 10000.0f32, 8.0f32, 2048.0f32);
        let f = yarn_frequencies(d, base, scale, l, YARN_ALPHA, YARN_BETA);
        let last = d / 2 - 1;
        let theta_last = base.powf(-2.0 * last as f32 / d as f32);
        let ext_len = scale * l;
        // Plain RoPE at the extended length: way out of the trained range.
        let plain_angle = ext_len * theta_last;
        // YaRN at the extended length ≈ plain RoPE at the original length L.
        let yarn_angle = ext_len * f[last];
        let trained_angle = l * theta_last;
        assert!(
            (yarn_angle - trained_angle).abs() < 1e-2,
            "YaRN angle {yarn_angle} not back in-distribution ({trained_angle})"
        );
        assert!(
            plain_angle > trained_angle * (scale - 1.0),
            "sanity: plain RoPE should blow up the angle"
        );
    }

    /// The temperature is 1 at `scale = 1` and grows with the extension factor.
    #[test]
    fn attention_scale_grows_with_extension() {
        assert_eq!(yarn_attention_scale(1.0), 1.0);
        assert!(yarn_attention_scale(8.0) > 1.0);
        assert!(yarn_attention_scale(16.0) > yarn_attention_scale(8.0));
    }

    /// Deterministic: identical inputs give bit-identical rotations.
    #[test]
    fn deterministic() {
        let d = 16;
        let x: Vec<f32> = (0..4 * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let a = rope_yarn(&x, 4, d, 4.0, 2048.0);
        let b = rope_yarn(&x, 4, d, 4.0, 2048.0);
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
