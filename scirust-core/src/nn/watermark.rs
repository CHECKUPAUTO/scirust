//! **A Watermark for Large Language Models** (Kirchenbauer et al., ICML 2023).
//!
//! A statistical watermark makes machine-generated text **auditable** without any
//! access to the model. At each step the previous token seeds a partition of the
//! vocabulary into a **green** list (a fraction `gamma`) and a red list; generation
//! nudges the logits toward green tokens. A detector who only knows the seed and
//! `gamma` recomputes each position's green list and counts how many emitted tokens
//! are green: watermarked text contains far more green tokens than the `gamma`
//! expected by chance, which a **z-test** flags with a tiny p-value — while natural
//! text scores `z ≈ 0`. Everything is a deterministic hash of `(seed, prev, token)`.

/// Deterministic uniform-`[0,1)` hash of `(seed, prev_token, token)` (a SplitMix64
/// mix), used to assign the green/red partition.
fn green_hash(seed: u64, prev_token: usize, token: usize) -> f32 {
    let mut z = seed
        .wrapping_add((prev_token as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((token as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 24 bits → a float in [0, 1).
    (z >> 40) as f32 / (1u64 << 24) as f32
}

/// Whether `token` is on the **green list** for the position following
/// `prev_token` (a `gamma` fraction of the vocabulary, seeded by `prev_token`).
pub fn is_green(seed: u64, prev_token: usize, token: usize, gamma: f32) -> bool {
    green_hash(seed, prev_token, token) < gamma
}

/// **Apply** the watermark to a logit row: add `delta` to every green token's
/// logit (seeded by `prev_token`), so sampling/argmax favours green tokens. The
/// caller samples from the biased logits as usual — this is the only change a
/// watermarked generator makes.
pub fn apply_green_bias(logits: &mut [f32], seed: u64, prev_token: usize, gamma: f32, delta: f32) {
    for (token, l) in logits.iter_mut().enumerate()
    {
        if is_green(seed, prev_token, token, gamma)
        {
            *l += delta;
        }
    }
}

/// The watermark **detection z-statistic** of a token sequence: with `g` green
/// tokens out of the `n = tokens.len() − 1` transitions, under the no-watermark
/// null `g ~ Binomial(n, gamma)`, so `z = (g − γn) / √(nγ(1−γ))`. A large `z` (small
/// one-sided p-value `Φ(−z)`) is strong evidence the text was watermarked — **no
/// model access required**.
pub fn detect_z(seed: u64, tokens: &[usize], gamma: f32) -> f32 {
    if tokens.len() < 2
    {
        return 0.0;
    }
    let n = (tokens.len() - 1) as f32;
    let green = tokens
        .windows(2)
        .filter(|w| is_green(seed, w[0], w[1], gamma))
        .count() as f32;
    (green - gamma * n) / (n * gamma * (1.0 - gamma)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

    /// The green list is an unbiased `gamma` fraction of the vocabulary: over all
    /// `(prev, token)` pairs the green fraction ≈ `gamma`.
    #[test]
    fn green_fraction_matches_gamma() {
        let (seed, gamma, vocab) = (1u64, 0.3f32, 200usize);
        let (mut green, mut total) = (0usize, 0usize);
        for prev in 0..vocab
        {
            for token in 0..vocab
            {
                if is_green(seed, prev, token, gamma)
                {
                    green += 1;
                }
                total += 1;
            }
        }
        let frac = green as f32 / total as f32;
        assert!(
            (frac - gamma).abs() < 0.02,
            "green fraction {frac} ≠ γ {gamma}"
        );
    }

    /// `apply_green_bias` adds `delta` to exactly the green tokens' logits.
    #[test]
    fn apply_green_bias_lifts_only_green_tokens() {
        let (seed, gamma, prev, vocab) = (3u64, 0.25f32, 11usize, 64usize);
        let mut logits = vec![0.0f32; vocab];
        apply_green_bias(&mut logits, seed, prev, gamma, 2.0);
        for (token, &l) in logits.iter().enumerate()
        {
            if is_green(seed, prev, token, gamma)
            {
                assert!((l - 2.0).abs() < 1e-6, "green token {token} not lifted");
            }
            else
            {
                assert!(l.abs() < 1e-6, "red token {token} changed");
            }
        }
    }

    /// **The watermark, tested.** Natural (random) text scores `z ≈ 0` and is not
    /// flagged; **green-biased** (watermarked) text scores a large `z` and is
    /// detected with a tiny p-value — all without model access. A *wrong* seed does
    /// not detect it (no false provenance). Deterministic.
    #[test]
    fn detects_watermark_not_natural_text() {
        let (seed, vocab, gamma) = (42u64, 128usize, 0.25f32);

        // Natural text: uniform random tokens ⇒ z ≈ 0.
        let mut rng = PcgEngine::new(7);
        let natural: Vec<usize> = (0..600)
            .map(|_| (rng.float() * vocab as f32) as usize % vocab)
            .collect();
        let z_nat = detect_z(seed, &natural, gamma);
        assert!(z_nat.abs() < 3.0, "natural text flagged: z = {z_nat}");

        // Watermarked text: a generator that biases toward green strongly, modelled
        // by sampling a (diverse) green token at each step.
        let mut draw = PcgEngine::new(99);
        let mut wm = vec![5usize];
        for _ in 0..600
        {
            let prev = *wm.last().unwrap();
            let mut t = (draw.float() * vocab as f32) as usize % vocab;
            while !is_green(seed, prev, t, gamma)
            {
                t = (draw.float() * vocab as f32) as usize % vocab;
            }
            wm.push(t);
        }
        let z_wm = detect_z(seed, &wm, gamma);
        assert!(z_wm > 8.0, "watermark not detected: z = {z_wm}");

        // Determinism.
        assert_eq!(detect_z(seed, &wm, gamma).to_bits(), z_wm.to_bits());
        // A different seed does not detect this watermark (no false provenance).
        assert!(detect_z(seed + 1, &wm, gamma).abs() < 3.0);
    }
}
