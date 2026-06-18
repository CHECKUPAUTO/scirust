//! **DiFR — Inference verification despite nondeterminism** (2025, arXiv:2511.20621).
//!
//! The [`proof`](crate::proof) certificates verify an inference by **bit-exact**
//! re-execution — which works only when the verifier can reproduce the prover's
//! arithmetic exactly. Across *different hardware* (different SIMD widths, fused
//! multiply-adds, thread counts) floating-point summation is **non-deterministic**,
//! so a bit-exact check would reject perfectly honest outputs.
//!
//! DiFR verifies the output *despite* that. It recomputes a **canonical reference**
//! with the order-independent [`reproducible_dot`] (products and sum accumulated in
//! `f64`, near-exact) and accepts the claimed output iff it lies within a **sound
//! floating-point error envelope** of that reference. Any legitimate `f32`
//! computation — in *any* summation order — is provably within the envelope, so it
//! is accepted; an output **tampered** beyond the envelope is rejected. The envelope
//! is the standard dot-product rounding bound `γ·Σ|terms|` propagated through the
//! layers (ReLU is 1-Lipschitz, so it carries the bound through unchanged), and it is
//! **tiny** (a few ppm of the activation scale), so the check still catches any
//! meaningful tampering.

use scirust_core::reproducible::reproducible_dot;

/// `f32` unit roundoff `2⁻²⁴` (half an ULP at 1.0).
const F32_U: f32 = 1.0 / (1u32 << 24) as f32;

/// A default safety multiplier on the first-order rounding bound (covers the
/// higher-order `γₙ` terms, the bias add, and the reference's own `f64→f32` cast).
pub const DEFAULT_SLACK: f32 = 4.0;

/// One affine layer `y = xᵀW + b` (`W` row-major `(in, out)`, `b` length `out`).
#[derive(Clone)]
pub struct DifrLayer {
    /// Row-major `(in_f, out_f)` weight.
    pub w: Vec<f32>,
    /// Length-`out_f` bias.
    pub b: Vec<f32>,
    /// Input width.
    pub in_f: usize,
    /// Output width.
    pub out_f: usize,
}

/// The verifier's verdict: whether the claim was accepted, plus, per output, the
/// canonical reference value, the sound error envelope, and the observed deviation.
#[derive(Clone, Debug)]
pub struct DifrVerdict {
    /// True iff every output's deviation is within its envelope.
    pub accepted: bool,
    /// The canonical (reproducible) reference output.
    pub reference: Vec<f32>,
    /// The per-output sound error envelope.
    pub envelope: Vec<f32>,
    /// The per-output `|claimed − reference|`.
    pub deviation: Vec<f32>,
}

/// Verify a `claimed` MLP output against the canonical reference within the FP
/// envelope. ReLU is applied after every layer except the last; `slack` scales the
/// envelope (use [`DEFAULT_SLACK`]). Deterministic.
pub fn difr_verify(
    layers: &[DifrLayer],
    input: &[f32],
    claimed: &[f32],
    slack: f32,
) -> DifrVerdict {
    let nl = layers.len();
    let mut ref_x = input.to_vec();
    let mut delta = vec![0.0f32; input.len()]; // the input is known exactly
    for (li, l) in layers.iter().enumerate()
    {
        assert_eq!(ref_x.len(), l.in_f, "difr: layer input width mismatch");
        let mut ref_out = vec![0.0f32; l.out_f];
        let mut new_delta = vec![0.0f32; l.out_f];
        for o in 0..l.out_f
        {
            let col: Vec<f32> = (0..l.in_f).map(|i| l.w[i * l.out_f + o]).collect();
            // Canonical (order-independent, f64-accumulated) reference.
            ref_out[o] = reproducible_dot(&col, &ref_x) + l.b[o];
            // Sound envelope: this layer's rounding bound + propagated input error.
            let scale: f32 = (0..l.in_f)
                .map(|i| (l.w[i * l.out_f + o] * ref_x[i]).abs())
                .sum::<f32>()
                + l.b[o].abs();
            let round_err = slack * (l.in_f as f32 + 1.0) * F32_U * scale;
            let propagated: f32 = (0..l.in_f)
                .map(|i| l.w[i * l.out_f + o].abs() * delta[i])
                .sum();
            new_delta[o] = round_err + propagated;
        }
        if li + 1 < nl
        {
            // ReLU on the reference; the envelope passes through unchanged (1-Lipschitz).
            for r in ref_out.iter_mut()
            {
                *r = r.max(0.0);
            }
        }
        ref_x = ref_out;
        delta = new_delta;
    }
    let deviation: Vec<f32> = claimed
        .iter()
        .zip(&ref_x)
        .map(|(&c, &r)| (c - r).abs())
        .collect();
    let accepted = deviation.iter().zip(&delta).all(|(&d, &e)| d <= e);
    DifrVerdict {
        accepted,
        reference: ref_x,
        envelope: delta,
        deviation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::nn::PcgEngine;

    fn rand_layer(in_f: usize, out_f: usize, rng: &mut PcgEngine) -> DifrLayer {
        DifrLayer {
            w: (0..in_f * out_f).map(|_| rng.float_signed()).collect(),
            b: (0..out_f).map(|_| 0.3 * rng.float_signed()).collect(),
            in_f,
            out_f,
        }
    }

    /// An honest `f32` forward, summing each dot product in a **shuffled** order to
    /// emulate a different machine's non-deterministic accumulation.
    fn honest_forward_shuffled(
        layers: &[DifrLayer],
        input: &[f32],
        rng: &mut PcgEngine,
    ) -> Vec<f32> {
        let nl = layers.len();
        let mut x = input.to_vec();
        for (li, l) in layers.iter().enumerate()
        {
            let mut out = vec![0.0f32; l.out_f];
            for (o, slot) in out.iter_mut().enumerate()
            {
                let mut terms: Vec<f32> =
                    (0..l.in_f).map(|i| l.w[i * l.out_f + o] * x[i]).collect();
                for i in (1..terms.len()).rev()
                {
                    let j = ((rng.float() * (i as f32 + 1.0)) as usize).min(i);
                    terms.swap(i, j);
                }
                let mut acc = 0.0f32;
                for t in terms
                {
                    acc += t; // native f32, shuffled order
                }
                *slot = acc + l.b[o];
            }
            if li + 1 < nl
            {
                for v in out.iter_mut()
                {
                    *v = v.max(0.0);
                }
            }
            x = out;
        }
        x
    }

    fn net(seed: u64) -> (Vec<DifrLayer>, Vec<f32>) {
        let mut rng = PcgEngine::new(seed);
        let layers = vec![rand_layer(8, 6, &mut rng), rand_layer(6, 3, &mut rng)];
        let input: Vec<f32> = (0..8).map(|_| rng.float_signed()).collect();
        (layers, input)
    }

    /// A legitimate output computed in a **different summation order** is accepted.
    #[test]
    fn difr_accepts_legitimate_nondeterministic_output() {
        let (layers, input) = net(1);
        let mut rng = PcgEngine::new(42);
        let honest = honest_forward_shuffled(&layers, &input, &mut rng);
        let v = difr_verify(&layers, &input, &honest, DEFAULT_SLACK);
        assert!(
            v.accepted,
            "honest reorder rejected: dev {:?} env {:?}",
            v.deviation, v.envelope
        );
    }

    /// **The envelope is sound**: across 1000 different summation orders, *every*
    /// honest output is accepted (the bound is never violated). And it is **tight** —
    /// far smaller than the activation scale, so it is a meaningful check.
    #[test]
    fn difr_envelope_is_sound_and_tight() {
        let (layers, input) = net(2);
        let mut rng = PcgEngine::new(7);
        let mut max_dev = 0.0f32;
        let mut env = vec![];
        for _ in 0..1000
        {
            let honest = honest_forward_shuffled(&layers, &input, &mut rng);
            let v = difr_verify(&layers, &input, &honest, DEFAULT_SLACK);
            assert!(v.accepted, "sound envelope violated by an honest reorder");
            max_dev = max_dev.max(v.deviation.iter().cloned().fold(0.0, f32::max));
            env = v.envelope;
        }
        // The envelope must be tiny relative to the output magnitude (meaningful check).
        let out_scale = env.len() as f32; // ~O(1) activations; envelope must be << 0.01.
        let max_env = env.iter().cloned().fold(0.0, f32::max);
        assert!(max_env < 0.001 * out_scale, "envelope not tight: {max_env}");
        assert!(max_dev <= max_env, "observed deviation exceeded envelope");
    }

    /// **Tampering is rejected**: nudging one output past its envelope (here enough to
    /// flip the predicted class) makes verification fail — and it is deterministic.
    #[test]
    fn difr_rejects_tampering() {
        let (layers, input) = net(3);
        let mut rng = PcgEngine::new(9);
        let honest = honest_forward_shuffled(&layers, &input, &mut rng);
        // Reference for sizing the tamper.
        let base = difr_verify(&layers, &input, &honest, DEFAULT_SLACK);
        assert!(base.accepted);
        // Tamper output 0 by far more than its envelope (also a class-flip-scale change).
        let mut tampered = honest.clone();
        tampered[0] += base.envelope[0] + 0.05;
        let v = difr_verify(&layers, &input, &tampered, DEFAULT_SLACK);
        assert!(!v.accepted, "tampering accepted");
        // Determinism.
        let v2 = difr_verify(&layers, &input, &tampered, DEFAULT_SLACK);
        assert_eq!(v.accepted, v2.accepted);
        assert_eq!(
            v.reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            v2.reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}
