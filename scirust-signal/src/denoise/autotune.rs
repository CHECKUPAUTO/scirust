//! **Data-driven VST selection for denoising** — wires the generic dev/held-out
//! autotuner of [`scirust_core::transform_autotune`] (CANR stages S3/S4,
//! `docs/research/CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md` §8/§12)
//! onto this crate's real VST denoising pipeline ([`super::vst_denoise`]).
//!
//! [`super::vst_denoise_auto`] already selects a [`VstKind`] — but by a *heuristic*
//! ([`super::detect_noise_model`], a level-vs-scale regression) that needs no
//! ground truth and conservatively returns [`VstKind::Identity`] when unsure. That
//! is the right tool at deployment. This module is its **calibration-time**
//! complement: when a *clean reference* is available (a phantom, a high-SNR
//! capture, or a simulator), it picks the VST that empirically denoises best,
//! measured on a development record and **validated on a disjoint held-out
//! record** against the identity (direct-denoise) baseline. It answers, with
//! numbers rather than a heuristic, "which VST should I ship for this sensor?"
//! and honestly reports whether the choice generalized (`beats_baseline`).
//!
//! The inner Gaussian denoiser is supplied by the caller (as in
//! [`super::vst_denoise`]); the objective is SNR in dB against the clean
//! reference. Because the transform carries no fitted parameter, the harness's
//! "fit" set is unused — selection is still a genuine held-out test of *which
//! candidate* wins, guarding against a kind that only beats the baseline on the
//! dev noise realization.

use scirust_core::transform_autotune::{GenericAutotune, autotune_by};

use super::VstKind;
use super::vst::vst_denoise;

/// A paired denoising record: a `noisy` observation and its `clean` reference,
/// same length. Used as the autotuner's dataset (`dev` and `eval` are two such
/// disjoint records).
#[derive(Debug, Clone)]
pub struct DenoiseCase {
    /// The noisy observation.
    pub noisy: Vec<f64>,
    /// The clean reference (ground truth), same length as `noisy`.
    pub clean: Vec<f64>,
}

impl DenoiseCase {
    /// New case; panics if the lengths differ.
    pub fn new(noisy: Vec<f64>, clean: Vec<f64>) -> Self {
        assert_eq!(
            noisy.len(),
            clean.len(),
            "noisy and clean must have equal length"
        );
        Self { noisy, clean }
    }
}

/// Signal-to-noise ratio in dB of an estimate against a clean reference:
/// `10·log10(Σclean² / Σ(clean−est)²)`. `+∞` when the estimate is exact.
fn snr_db(clean: &[f64], estimate: &[f64]) -> f64 {
    let mut sig = 0.0;
    let mut err = 0.0;
    for (&c, &e) in clean.iter().zip(estimate)
    {
        sig += c * c;
        err += (c - e) * (c - e);
    }
    if err == 0.0
    {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// Result of a VST autotune run: the empirically-selected transform plus the
/// held-out evidence for the choice.
#[derive(Debug, Clone)]
pub struct VstAutotuneResult {
    /// The dev-winning transform, or `None` if the candidate set was empty.
    pub kind: Option<VstKind>,
    /// The winner's SNR (dB) on the **held-out** record.
    pub eval_snr_db: f64,
    /// The identity (direct-denoise) baseline's SNR (dB) on the held-out record.
    pub baseline_snr_db: f64,
    /// Whether the empirically-chosen VST beat direct denoising on held-out data.
    /// A `false` here is the module's pre-registered kill signal (CANR §13): the
    /// VST did not generalize, so ship [`VstKind::Identity`].
    pub beats_baseline: bool,
    /// Every candidate's dev SNR (dB), in input order.
    pub dev_snr_db: Vec<(VstKind, f64)>,
}

/// Autotune the variance-stabilizing transform for a denoiser over `candidates`,
/// selecting on `dev` and validating on the held-out `eval` against the identity
/// (direct-denoise) baseline.
///
/// `denoiser` is the inner Gaussian denoiser applied in the transformed domain
/// (same contract as [`super::vst_denoise`]); it must return a signal of the same
/// length. The objective is SNR (dB) against the clean reference.
///
/// The winner is the transform with the highest dev SNR; `beats_baseline` says
/// whether it also beat direct denoising on the held-out record — ship
/// [`VstKind::Identity`] when it is `false`.
pub fn autotune_vst(
    dev: &DenoiseCase,
    eval: &DenoiseCase,
    candidates: &[VstKind],
    denoiser: impl Fn(&[f64]) -> Vec<f64> + Copy,
) -> VstAutotuneResult {
    let score = |kind: VstKind, _fit: &DenoiseCase, scr: &DenoiseCase| -> Option<f64> {
        let out = vst_denoise(&scr.noisy, kind, denoiser);
        if out.len() != scr.clean.len()
        {
            return None;
        }
        let snr = snr_db(&scr.clean, &out);
        snr.is_finite().then_some(snr)
    };
    let baseline = |_fit: &DenoiseCase, scr: &DenoiseCase| -> f64 {
        snr_db(
            &scr.clean,
            &vst_denoise(&scr.noisy, VstKind::Identity, denoiser),
        )
    };

    let GenericAutotune {
        chosen,
        chosen_eval_score,
        baseline_eval_score,
        beats_baseline,
        dev_scores,
    } = autotune_by(dev, eval, candidates, score, baseline);

    VstAutotuneResult {
        kind: chosen,
        eval_snr_db: chosen_eval_score,
        baseline_snr_db: baseline_eval_score,
        beats_baseline,
        dev_snr_db: dev_scores
            .into_iter()
            .map(|(k, s)| (k, s.unwrap_or(f64::NEG_INFINITY)))
            .collect(),
    }
}

/// A default VST candidate set for autotuning: the identity baseline plus the
/// automatically-selectable transforms and a small Box–Cox λ sweep. Excludes the
/// calibration-only [`VstKind::Gat`] (needs gain/σ inputs).
pub fn default_vst_candidates() -> Vec<VstKind> {
    vec![
        VstKind::Identity,
        VstKind::Anscombe,
        VstKind::SignedLog,
        VstKind::SignedSqrt,
        VstKind::BoxCox(0.25),
        VstKind::BoxCox(0.5),
    ]
}

#[cfg(test)]
mod tests {
    use super::super::{gaussian_smooth, wiener_white};
    use super::*;
    use scirust_core::philox::Philox4x32;

    /// A Wiener denoiser calibrated with a **single global** noise σ (robust MAD
    /// of first differences). It assumes homoscedastic noise, so it is
    /// mis-calibrated on signal-dependent noise in the raw domain and correctly
    /// calibrated once a VST has equalized the variance — exactly the estimator a
    /// VST is meant to help (cf. `vst_denoise_auto`'s Wiener inner filter).
    fn wiener_global(s: &[f64]) -> Vec<f64> {
        let mut d: Vec<f64> = s.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if d.is_empty() { 1.0 } else { d[d.len() / 2] };
        let sigma = (mad / (0.6745 * std::f64::consts::SQRT_2)).max(1e-6);
        wiener_white(s, sigma)
    }

    /// A smooth positive clean signal: a slow sinusoid offset well above zero.
    fn clean_signal(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                20.0 + 15.0 * (2.0 * std::f64::consts::PI * 3.0 * t).sin()
            })
            .collect()
    }

    /// Add multiplicative noise (σ ∝ level) deterministically via Philox
    /// Box–Muller: `y = x·(1 + α·z)`, clamped positive.
    fn multiplicative_noise(clean: &[f64], alpha: f64, seed: u64, stream: u32) -> Vec<f64> {
        let rng = Philox4x32::new(seed);
        clean
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let u1 = (rng.f32_at(stream, 2 * i as u64) as f64).max(1e-12);
                let u2 = rng.f32_at(stream, 2 * i as u64 + 1) as f64;
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                (x * (1.0 + alpha * z)).max(1e-6)
            })
            .collect()
    }

    fn case(seed: u64, stream: u32) -> DenoiseCase {
        let clean = clean_signal(1024);
        let noisy = multiplicative_noise(&clean, 0.35, seed, stream);
        DenoiseCase::new(noisy, clean)
    }

    #[test]
    fn autotuned_vst_beats_direct_denoising_on_multiplicative_noise() {
        // Multiplicative noise (σ ∝ level) with a globally-calibrated Wiener
        // filter: in the raw domain a single global σ is wrong everywhere (too
        // small where the signal is loud, too large where it is quiet); a
        // stabilizing VST equalizes the variance so the global calibration fits,
        // and the autotuner discovers this from data.
        let dev = case(1, 0);
        let eval = case(2, 1);
        let r = autotune_vst(&dev, &eval, &default_vst_candidates(), wiener_global);

        let kind = r.kind.expect("a candidate should be chosen");
        // The empirical winner is a stabilizing transform, not identity.
        assert_ne!(
            kind,
            VstKind::Identity,
            "a VST should win over identity here"
        );
        // And it generalizes: it beats direct denoising on the held-out record.
        assert!(
            r.beats_baseline,
            "chosen {:?}: eval SNR {:.2} dB vs baseline {:.2} dB",
            kind, r.eval_snr_db, r.baseline_snr_db
        );
        assert_eq!(r.dev_snr_db.len(), default_vst_candidates().len());
    }

    #[test]
    fn identity_is_chosen_when_no_vst_helps() {
        // Clean data with tiny homoscedastic (level-independent) noise: no VST
        // should beat identity, so the honest outcome is Identity / no gain.
        let clean = clean_signal(1024);
        let rng = Philox4x32::new(9);
        let add_noise = |seed_stream: u32| {
            let noisy: Vec<f64> = clean
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let u1 = (rng.f32_at(seed_stream, 2 * i as u64) as f64).max(1e-12);
                    let u2 = rng.f32_at(seed_stream, 2 * i as u64 + 1) as f64;
                    let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                    x + 0.05 * z // additive, level-independent
                })
                .collect();
            DenoiseCase::new(noisy, clean.clone())
        };
        let dev = add_noise(0);
        let eval = add_noise(1);
        let denoiser = |s: &[f64]| gaussian_smooth(s, 4.0);
        let r = autotune_vst(&dev, &eval, &default_vst_candidates(), denoiser);
        // Either identity is chosen, or whatever won does not beat the identity
        // baseline on held-out (the pre-registered kill signal fires).
        assert!(
            r.kind == Some(VstKind::Identity) || !r.beats_baseline,
            "no VST should generalize on homoscedastic noise (chose {:?}, beats={})",
            r.kind,
            r.beats_baseline
        );
    }

    #[test]
    fn empty_candidate_set_yields_no_choice() {
        let dev = case(1, 0);
        let eval = case(2, 1);
        let r = autotune_vst(&dev, &eval, &[], |s: &[f64]| gaussian_smooth(s, 4.0));
        assert_eq!(r.kind, None);
        assert!(!r.beats_baseline);
    }
}
