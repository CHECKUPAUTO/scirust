//! **Empirical representation autotuning** — stages S3/S4 of the CANR autotuner
//! (`docs/research/CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md`, §8;
//! the quantization experiment is §5/X3), layered on the certificate gate (S1)
//! of [`crate::transform_search`].
//!
//! S1 is *sound but conservative*: it accepts a representation only when its
//! worst-case round-trip certificate holds, and it cannot rank representations
//! by a **downstream** objective (quantization distortion, low-precision
//! reconstruction error) that the certificate does not see. This module adds the
//! measured half:
//!
//! * **S3 — dev measurement**: for every representation that clears the S1
//!   safety gate (in-domain, outside the `κ_rt·u ≥ ½` invalid region), measure
//!   the actual objective on a **development** set, sweeping any parameter knob
//!   (e.g. the Box–Cox exponent λ) exhaustively over the supplied dictionary.
//! * **S4 — held-out check**: re-evaluate the dev-winner on a disjoint
//!   **evaluation** set *without refitting*, and compare it against the
//!   identity/direct baseline on that same held-out set. The winner is reported
//!   together with whether it actually generalized (`beats_baseline`) — a
//!   negative outcome is surfaced, never hidden.
//!
//! ## The objective: transformed-domain uniform quantization (CANR X3)
//!
//! The bundled objective encodes the data with `φ`, quantizes the encoded values
//! with an `L`-level **uniform** quantizer whose range is fit on the dev set, and
//! decodes. For heavy-tailed / wide-range data a companding transform (log,
//! Box–Cox) concentrates the levels where the mass is, beating direct uniform
//! quantization by ~10 dB SQNR (CANR X3: direct 34.8 dB, log 38.0, Box–Cox 44.9).
//! The quantizer *range* is the fitted parameter, so fitting on dev and scoring
//! on held-out is a genuine generalization test (CANR §13).
//!
//! Exhaustive search over the dictionary is used (the report calls for it at
//! ≤ ~50 candidates; successive-halving / Bayesian search is only needed for
//! larger parameter spaces).

use crate::certified_numerics::CertifiedMonotone;
use crate::transform_search::{RejectReason, Representation};

/// Unit roundoff of `f64` (the invalid-region threshold uses `κ_rt·u ≥ ½`).
const UNIT: f64 = f64::EPSILON / 2.0;

/// An `L`-level uniform quantizer over a fixed encoded range `[lo, hi]`.
#[derive(Debug, Clone, Copy)]
struct UniformQuantizer {
    lo: f64,
    step: f64,
    levels: usize,
}

impl UniformQuantizer {
    /// Fit the range to the encoded values `enc` (min/max), with `levels` cells.
    fn fit(enc: &[f64], levels: usize) -> Self {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &e in enc
        {
            lo = lo.min(e);
            hi = hi.max(e);
        }
        let span = (hi - lo).max(f64::MIN_POSITIVE);
        Self {
            lo,
            step: span / levels as f64,
            levels,
        }
    }

    /// Quantize-then-reconstruct one encoded value (mid-tread cell centre).
    fn round_trip(&self, e: f64) -> f64 {
        let raw = ((e - self.lo) / self.step).floor();
        let idx = raw.clamp(0.0, (self.levels - 1) as f64);
        self.lo + (idx + 0.5) * self.step
    }
}

/// Signal-to-quantization-noise ratio in dB: `10·log10(Σx² / Σ(x−x̂)²)`.
/// `+∞` when the reconstruction is exact.
fn sqnr_db(reference: &[f64], estimate: &[f64]) -> f64 {
    let mut sig = 0.0;
    let mut err = 0.0;
    for (&r, &e) in reference.iter().zip(estimate)
    {
        sig += r * r;
        err += (r - e) * (r - e);
    }
    if err == 0.0
    {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// Score a representation on the quantization objective: fit the encoded-domain
/// quantizer on `dev`, then measure SQNR (dB) on `eval`. Returns `None` if any
/// `dev`/`eval` sample is outside the transform's domain (should not happen
/// after the S1 gate, but kept total for safety).
fn quantize_score(repr: Representation, dev: &[f64], eval: &[f64], levels: usize) -> Option<f64> {
    let enc_dev: Option<Vec<f64>> = dev.iter().map(|&x| repr.encode(x)).collect();
    let enc_dev = enc_dev?;
    let q = UniformQuantizer::fit(&enc_dev, levels);
    let mut recon = Vec::with_capacity(eval.len());
    for &x in eval
    {
        let e = repr.encode(x)?;
        recon.push(repr.decode(q.round_trip(e)));
    }
    Some(sqnr_db(eval, &recon))
}

/// Identity (direct) uniform quantization baseline: same protocol, no transform.
fn baseline_score(dev: &[f64], eval: &[f64], levels: usize) -> f64 {
    let q = UniformQuantizer::fit(dev, levels);
    let recon: Vec<f64> = eval.iter().map(|&x| q.round_trip(x)).collect();
    sqnr_db(eval, &recon)
}

/// S1 safety gate for the autotuner: in-domain over the data support and clear
/// of the invalid region. (Round-trip *tolerance* is intentionally not gated
/// here — quantization distortion dominates and is measured directly in S3.)
fn safety_gate(repr: Representation, dev: &[f64], eval: &[f64]) -> Result<(), RejectReason> {
    let domain = repr.domain();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in dev.iter().chain(eval)
    {
        if !domain.contains(x)
        {
            return Err(RejectReason::OutsideDomain { sample: x });
        }
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let ksup = repr.kappa_rt_sup(crate::certified_numerics::Interval::new(lo, hi));
    if ksup * UNIT >= 0.5
    {
        return Err(RejectReason::InvalidRegion { kappa_rt_sup: ksup });
    }
    Ok(())
}

/// Per-candidate outcome of an autotune run.
#[derive(Debug, Clone, Copy)]
pub struct AutotuneVerdict {
    /// The representation evaluated.
    pub repr: Representation,
    /// `Ok(dev_sqnr_db)` if it cleared S1 and was measured, else the S1 reason.
    pub dev_score: Result<f64, RejectReason>,
}

/// Result of an autotune run.
#[derive(Debug, Clone)]
pub struct AutotuneReport {
    /// The dev-winning representation, or `None` if none cleared the S1 gate.
    pub chosen: Option<Representation>,
    /// The winner's SQNR (dB) on the **held-out** evaluation set (S4).
    pub chosen_eval_sqnr_db: f64,
    /// The identity/direct baseline's SQNR (dB) on the same held-out set.
    pub baseline_eval_sqnr_db: f64,
    /// Whether the dev-winner actually beat the baseline on held-out data.
    pub beats_baseline: bool,
    /// Every candidate's dev outcome (measured score or S1 rejection), in order.
    pub verdicts: Vec<AutotuneVerdict>,
}

/// Autotune the transformed-domain uniform quantizer over `dictionary`: gate on
/// safety (S1), rank the survivors by measured dev SQNR (S3), then validate the
/// dev-winner on the held-out `eval` set against the direct baseline (S4).
///
/// `dictionary` may list the same family at several knob settings (e.g. several
/// [`Representation::Power`] λ values); the search is exhaustive.
pub fn autotune_quantizer(
    dev: &[f64],
    eval: &[f64],
    dictionary: &[Representation],
    levels: usize,
) -> AutotuneReport {
    let mut verdicts = Vec::with_capacity(dictionary.len());
    let mut best: Option<(Representation, f64)> = None;

    for &repr in dictionary
    {
        let dev_score = match safety_gate(repr, dev, eval)
        {
            Err(reason) => Err(reason),
            Ok(()) => match quantize_score(repr, dev, dev, levels)
            {
                // S3 measures on the dev set (fit and score both on dev).
                Some(s) => Ok(s),
                None => Err(RejectReason::OutsideDomain { sample: f64::NAN }),
            },
        };
        if let Ok(s) = dev_score
            && best.is_none_or(|(_, b)| s > b)
        {
            best = Some((repr, s));
        }
        verdicts.push(AutotuneVerdict { repr, dev_score });
    }

    let baseline_eval = baseline_score(dev, eval, levels);
    let (chosen, chosen_eval) = match best
    {
        // S4: score the dev-winner on held-out data (fit on dev, eval on eval).
        Some((repr, _)) =>
        {
            let s = quantize_score(repr, dev, eval, levels).unwrap_or(f64::NEG_INFINITY);
            (Some(repr), s)
        },
        None => (None, f64::NEG_INFINITY),
    };

    AutotuneReport {
        chosen,
        chosen_eval_sqnr_db: chosen_eval,
        baseline_eval_sqnr_db: baseline_eval,
        beats_baseline: chosen.is_some() && chosen_eval > baseline_eval,
        verdicts,
    }
}

/// Expand [`Representation::Power`] over a λ grid — a convenience for building a
/// knob-swept dictionary.
pub fn power_lambda_grid(lambdas: &[f64]) -> Vec<Representation> {
    lambdas.iter().map(|&l| Representation::Power(l)).collect()
}

/// A default knob-swept dictionary for positive, wide-range data: the log
/// family, a Box–Cox λ sweep, and the Poisson-matched Anscombe.
pub fn default_positive_quantizer_dictionary() -> Vec<Representation> {
    let mut d = vec![
        Representation::Log,
        Representation::Log1p,
        Representation::Anscombe,
    ];
    d.extend(power_lambda_grid(&[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]));
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Deterministic lognormal(0, σ) samples.
    fn lognormal(seed: u64, n: usize, sigma: f64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                // Box–Muller for one standard normal, then exp(σ·z).
                let u1: f64 = rng.gen_range(1e-12..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                (sigma * z).exp()
            })
            .collect()
    }

    #[test]
    fn transformed_quantizer_beats_direct_on_heavy_tailed_data() {
        let dev = lognormal(1, 8192, 1.3);
        let eval = lognormal(2, 8192, 1.3);
        let report = autotune_quantizer(&dev, &eval, &default_positive_quantizer_dictionary(), 64);
        let chosen = report
            .chosen
            .expect("a safe representation should be chosen");
        // The selection must generalize: transformed beats direct on held-out.
        assert!(
            report.beats_baseline,
            "chosen {:?} eval SQNR {:.2} dB did not beat baseline {:.2} dB",
            chosen, report.chosen_eval_sqnr_db, report.baseline_eval_sqnr_db
        );
        // The margin should be sizeable for lognormal data (CANR X3 ~+10 dB).
        assert!(
            report.chosen_eval_sqnr_db - report.baseline_eval_sqnr_db > 3.0,
            "margin only {:.2} dB",
            report.chosen_eval_sqnr_db - report.baseline_eval_sqnr_db
        );
        // A companding transform (log family or a small-λ power) should win over
        // the Poisson-matched Anscombe on this heavy tail.
        assert_ne!(chosen, Representation::Anscombe);
    }

    #[test]
    fn knob_sweep_picks_a_lambda_and_reports_all_scores() {
        let dev = lognormal(3, 4096, 1.5);
        let eval = lognormal(4, 4096, 1.5);
        let dict = power_lambda_grid(&[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]);
        let report = autotune_quantizer(&dev, &eval, &dict, 128);
        assert!(matches!(report.chosen, Some(Representation::Power(_))));
        // Every candidate cleared S1 and carries a measured dev score.
        assert_eq!(report.verdicts.len(), dict.len());
        assert!(report.verdicts.iter().all(|v| v.dev_score.is_ok()));
        // Monotone quantization theory: for lognormal the smaller-λ (stronger)
        // companding should out-score λ = 1 (near-linear) on dev.
        let score_of = |lam: f64| {
            report
                .verdicts
                .iter()
                .find(|v| v.repr == Representation::Power(lam))
                .and_then(|v| v.dev_score.ok())
                .unwrap()
        };
        assert!(
            score_of(0.2) > score_of(1.0),
            "companding should beat near-linear"
        );
    }

    #[test]
    fn out_of_domain_candidates_are_rejected_not_chosen() {
        // Logit needs (0,1); positive data > 1 is outside its domain.
        let dev = lognormal(5, 2048, 1.0);
        let eval = lognormal(6, 2048, 1.0);
        let dict = vec![Representation::Logit, Representation::Log];
        let report = autotune_quantizer(&dev, &eval, &dict, 64);
        let logit = report
            .verdicts
            .iter()
            .find(|v| v.repr == Representation::Logit)
            .unwrap();
        assert!(matches!(
            logit.dev_score,
            Err(RejectReason::OutsideDomain { .. })
        ));
        assert_eq!(report.chosen, Some(Representation::Log));
    }

    #[test]
    fn all_rejected_yields_no_choice() {
        // Data > 1 with a dictionary of only Logit ⇒ nothing clears S1.
        let dev = vec![2.0, 5.0, 10.0];
        let eval = vec![3.0, 7.0];
        let report = autotune_quantizer(&dev, &eval, &[Representation::Logit], 16);
        assert_eq!(report.chosen, None);
        assert!(!report.beats_baseline);
        assert!(matches!(
            report.verdicts[0].dev_score,
            Err(RejectReason::OutsideDomain { .. })
        ));
    }

    #[test]
    fn held_out_score_is_measured_on_eval_not_dev() {
        // Sanity: the reported eval score equals a direct recomputation on eval.
        let dev = lognormal(7, 4096, 1.2);
        let eval = lognormal(8, 4096, 1.2);
        let report = autotune_quantizer(&dev, &eval, &[Representation::Log], 64);
        let expect = quantize_score(Representation::Log, &dev, &eval, 64).unwrap();
        assert_eq!(report.chosen_eval_sqnr_db, expect);
    }
}
