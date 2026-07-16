//! **Certificate-gated representation selection** — the Phase-B core (stage S1
//! of the autotuner) of the CANR study
//! (`docs/research/CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md`, §8/§12).
//!
//! Given a dataset (as samples), a relative-error tolerance, and a **declared
//! finite family** of certified transform pairs, this selects the cheapest
//! representation whose *machine-checkable* round-trip certificate holds over the
//! data's support — and reports, for every candidate, either the certified bound
//! or the concrete reason it was rejected. It is the honest, sound-but-
//! conservative gate of the CANR autotuner: it never accepts an unsafe
//! representation, and it explains its refusals.
//!
//! This is deliberately **not** a full autotuner. It implements stage S1
//! (certificate gate) plus a static cost tie-break (a slice of S2); the
//! empirical dev/held-out refinement (S3/S4) and search over parameterized
//! knobs belong to a later increment (the report's `scirust-transform-search`
//! crate). Keeping S1 separate is the point: its output is *certified*, not
//! measured, so it composes under the crate's determinism guarantees.
//!
//! ## What "safe on the support" means
//!
//! For a strictly monotone pair `φ`, the round-trip relative error is bounded by
//! `(κ_rt_sup·B_ENC + B_DEC)` ulps over an interval (CANR §3.2, implemented by
//! [`CertifiedMonotone::roundtrip_bound`]). A representation is *accepted* for a
//! dataset iff (i) every sample lies in `φ`'s domain, (ii) the support does not
//! touch the invalid region `κ_rt·u ≥ ½` (CANR §3.3), and (iii) that certified
//! bound is `≤ tau_ulps`. Ranking is by ascending [`Representation::cost`], ties
//! broken by the tighter certificate.

use crate::certified_numerics::{
    Anscombe, B_DEC, B_ENC, CertifiedMonotone, Interval, Log, Log1p, Logit, MuLaw, Power, SignedLog,
};

/// Unit roundoff of `f64` (kept local; the constant is private in
/// `certified_numerics`).
const UNIT: f64 = f64::EPSILON / 2.0;

/// A member of the declared representation family. Wraps the concrete
/// [`CertifiedMonotone`] pairs so a dictionary can be a plain `&[Representation]`
/// (no trait objects, no allocation), while still dispatching to each pair's
/// exact certificate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Representation {
    /// `x ↦ ln x`.
    Log,
    /// `x ↦ ln(1+x)`.
    Log1p,
    /// `x ↦ asinh x` (signed log).
    SignedLog,
    /// `x ↦ x^λ` (unshifted Box–Cox storage); `λ > 0`.
    Power(f64),
    /// μ-law companding; `μ > 0`.
    MuLaw(f64),
    /// `x ↦ ln(x/(1−x))` on `(0, 1)`.
    Logit,
    /// `x ↦ 2√(x + 3/8)` (Poisson variance stabilizer).
    Anscombe,
}

impl Representation {
    /// Run `f` with the concrete transform this variant denotes.
    #[inline]
    fn with<R>(self, f: impl FnOnce(&dyn CertifiedMonotone) -> R) -> R {
        match self
        {
            Representation::Log => f(&Log),
            Representation::Log1p => f(&Log1p),
            Representation::SignedLog => f(&SignedLog),
            Representation::Power(l) => f(&Power::new(l)),
            Representation::MuLaw(m) => f(&MuLaw::new(m)),
            Representation::Logit => f(&Logit),
            Representation::Anscombe => f(&Anscombe),
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self
        {
            Representation::Log => "log",
            Representation::Log1p => "log1p",
            Representation::SignedLog => "signed-log",
            Representation::Power(_) => "power",
            Representation::MuLaw(_) => "mu-law",
            Representation::Logit => "logit",
            Representation::Anscombe => "anscombe",
        }
    }

    /// A relative encode+decode cost proxy (flops/element, order of magnitude).
    /// Used only to break ties among certified-safe candidates.
    pub fn cost(self) -> u32 {
        match self
        {
            Representation::Anscombe => 2, // sqrt + square
            Representation::Log | Representation::Log1p | Representation::Power(_) => 3,
            Representation::Logit | Representation::SignedLog => 4,
            Representation::MuLaw(_) => 5,
        }
    }
}

impl CertifiedMonotone for Representation {
    fn domain(&self) -> Interval {
        self.with(|t| t.domain())
    }
    fn encode(&self, x: f64) -> Option<f64> {
        self.with(|t| t.encode(x))
    }
    fn decode(&self, y: f64) -> f64 {
        self.with(|t| t.decode(y))
    }
    fn kappa_rt(&self, x: f64) -> f64 {
        self.with(|t| t.kappa_rt(x))
    }
    fn kappa_rt_sup(&self, iv: Interval) -> f64 {
        self.with(|t| t.kappa_rt_sup(iv))
    }
}

/// Why a candidate was rejected for a given dataset and tolerance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RejectReason {
    /// A sample lies outside the transform's domain.
    OutsideDomain {
        /// The offending sample.
        sample: f64,
    },
    /// The support enters the invalid region `κ_rt·u ≥ ½` (unrecoverable decode).
    InvalidRegion {
        /// The certified sup of κ_rt over the support.
        kappa_rt_sup: f64,
    },
    /// The certified round-trip bound over the support exceeds the tolerance.
    ToleranceExceeded {
        /// The certified bound, in ulps.
        bound_ulps: f64,
    },
}

/// Outcome for one candidate: either the certified round-trip bound (ulps) it
/// achieves on the data, or the reason it was rejected.
#[derive(Debug, Clone, Copy)]
pub struct CandidateVerdict {
    /// The representation evaluated.
    pub repr: Representation,
    /// `Ok(bound_ulps)` if accepted, else the rejection reason.
    pub verdict: Result<f64, RejectReason>,
}

/// The result of a selection: the chosen representation (cheapest certified-safe
/// one) and the full per-candidate verdict list.
#[derive(Debug, Clone)]
pub struct SelectionReport {
    /// The chosen representation, or `None` if none was certified safe at `tau`.
    pub chosen: Option<Representation>,
    /// The certified bound (ulps) of the chosen representation.
    pub chosen_bound_ulps: Option<f64>,
    /// The data support the decision was made over.
    pub support: Interval,
    /// Every candidate's verdict, in input order (rejections keep their reason).
    pub verdicts: Vec<CandidateVerdict>,
}

/// Select the cheapest certified-safe representation for `samples` at a
/// round-trip tolerance of `tau_ulps`, from the declared `dictionary`.
///
/// Returns a full [`SelectionReport`]: sound (never accepts a representation
/// whose certificate does not hold) and explanatory (each rejection carries its
/// reason). Empty `samples` yields a report with `chosen = None` and no
/// verdicts.
pub fn select_transform(
    samples: &[f64],
    tau_ulps: f64,
    dictionary: &[Representation],
) -> SelectionReport {
    if samples.is_empty()
    {
        return SelectionReport {
            chosen: None,
            chosen_bound_ulps: None,
            support: Interval::new(0.0, 0.0),
            verdicts: Vec::new(),
        };
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &s in samples
    {
        lo = lo.min(s);
        hi = hi.max(s);
    }
    let support = Interval::new(lo, hi);

    let mut verdicts = Vec::with_capacity(dictionary.len());
    for &repr in dictionary
    {
        verdicts.push(CandidateVerdict {
            repr,
            verdict: evaluate(repr, samples, support, tau_ulps),
        });
    }

    // Choose the accepted candidate with the least cost, ties broken by the
    // tighter certified bound.
    let mut chosen: Option<(Representation, f64)> = None;
    for v in &verdicts
    {
        if let Ok(bound) = v.verdict
        {
            let better = match chosen
            {
                None => true,
                Some((c, cb)) =>
                {
                    v.repr.cost() < c.cost() || (v.repr.cost() == c.cost() && bound < cb)
                },
            };
            if better
            {
                chosen = Some((v.repr, bound));
            }
        }
    }

    SelectionReport {
        chosen: chosen.map(|(r, _)| r),
        chosen_bound_ulps: chosen.map(|(_, b)| b),
        support,
        verdicts,
    }
}

/// Evaluate one representation against the data + tolerance.
fn evaluate(
    repr: Representation,
    samples: &[f64],
    support: Interval,
    tau_ulps: f64,
) -> Result<f64, RejectReason> {
    let domain = repr.domain();
    // (i) domain: report the first offending sample.
    for &s in samples
    {
        if !domain.contains(s)
        {
            return Err(RejectReason::OutsideDomain { sample: s });
        }
    }
    // (ii) invalid region: the certified sup of κ_rt over the support decides it.
    let ksup = repr.kappa_rt_sup(support);
    if ksup * UNIT >= 0.5
    {
        return Err(RejectReason::InvalidRegion { kappa_rt_sup: ksup });
    }
    // (iii) tolerance: the certified round-trip bound must fit.
    let bound = ksup * B_ENC + B_DEC;
    if bound > tau_ulps
    {
        return Err(RejectReason::ToleranceExceeded { bound_ulps: bound });
    }
    Ok(bound)
}

/// A reasonable default dictionary for **positive** data: the certified-safe,
/// identity-free monotone pairs that apply on `(0, ∞)` (log family, unshifted
/// square-root power, and the Poisson-matched Anscombe). Callers with signed or
/// bounded data should build their own slice including [`Representation::SignedLog`]
/// or [`Representation::Logit`] / [`Representation::MuLaw`].
pub fn default_positive_dictionary() -> Vec<Representation> {
    vec![
        Representation::Log,
        Representation::Log1p,
        Representation::Power(0.5),
        Representation::SignedLog,
        Representation::Anscombe,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        let (la, lb) = (a.log10(), b.log10());
        (0..n)
            .map(|i| 10f64.powf(la + (lb - la) * i as f64 / (n - 1) as f64))
            .collect()
    }

    /// Did `repr` receive an accepting verdict in `report`?
    fn accepted(report: &SelectionReport, repr: Representation) -> bool {
        report
            .verdicts
            .iter()
            .any(|v| v.repr == repr && v.verdict.is_ok())
    }

    #[test]
    fn wide_range_positive_data_selects_a_safe_cheap_transform() {
        let data = logspace(1e-3, 1e3, 200);
        let report = select_transform(&data, 5000.0, &default_positive_dictionary());
        // A safe representation exists here (log / power / signed-log all fit),
        // the chosen one is within tolerance, and it is genuinely certified.
        let chosen = report.chosen.expect("a safe representation should exist");
        assert!(report.chosen_bound_ulps.unwrap() <= 5000.0);
        assert!(accepted(&report, chosen));
    }

    #[test]
    fn data_crossing_zero_rejects_log_and_picks_signed_log() {
        let data = vec![-1000.0, -1.0, 0.5, 10.0, 500.0];
        let dict = vec![Representation::Log, Representation::SignedLog];
        let report = select_transform(&data, 5000.0, &dict);
        // Log must be rejected for a negative sample; signed-log accepted.
        let log_v = report
            .verdicts
            .iter()
            .find(|v| v.repr == Representation::Log)
            .unwrap();
        assert!(
            matches!(log_v.verdict, Err(RejectReason::OutsideDomain { .. })),
            "log should be rejected outside its domain"
        );
        assert_eq!(report.chosen, Some(Representation::SignedLog));
    }

    #[test]
    fn tiny_tolerance_rejects_everything_with_bounds() {
        let data = logspace(1e-6, 1e6, 100);
        let report = select_transform(&data, 1.0, &default_positive_dictionary());
        assert_eq!(report.chosen, None);
        // Every rejection at this tolerance is a ToleranceExceeded with a stated
        // bound (none is spuriously OutsideDomain / InvalidRegion here).
        for v in &report.verdicts
        {
            match v.verdict
            {
                Err(RejectReason::ToleranceExceeded { bound_ulps }) => assert!(bound_ulps > 1.0),
                other => panic!("{:?} expected ToleranceExceeded, got {other:?}", v.repr),
            }
        }
    }

    #[test]
    fn anscombe_near_zero_is_flagged_invalid_not_silently_accepted() {
        // A sample deep in the Anscombe invalid region (κ_rt·u ≥ ½ ⇒ x ≲ 1.7e-16).
        let data = vec![1e-18, 0.5, 4.0, 25.0];
        let dict = vec![Representation::Anscombe, Representation::Log];
        let report = select_transform(&data, 1e9, &dict);
        let ans = report
            .verdicts
            .iter()
            .find(|v| v.repr == Representation::Anscombe)
            .unwrap();
        assert!(
            matches!(ans.verdict, Err(RejectReason::InvalidRegion { .. })),
            "Anscombe must be flagged invalid near 0, not silently accepted"
        );
        // Log is fine on the same data and gets chosen.
        assert_eq!(report.chosen, Some(Representation::Log));
    }

    #[test]
    fn power_lambda_controls_the_certified_bound() {
        // κ_rt ≡ 1/λ for Power, so a larger λ gives a tighter (smaller) bound.
        let data = logspace(1.0, 1e3, 50);
        let loose = select_transform(&data, 1e12, &[Representation::Power(0.2)]);
        let tight = select_transform(&data, 1e12, &[Representation::Power(2.0)]);
        assert!(
            tight.chosen_bound_ulps.unwrap() < loose.chosen_bound_ulps.unwrap(),
            "larger λ (smaller 1/λ) must certify a tighter round trip"
        );
    }

    #[test]
    fn cost_breaks_ties_among_safe_candidates() {
        // On benign positive data both Anscombe (cost 2) and Log (cost 3) are
        // safe at a generous tolerance; the cheaper Anscombe wins.
        let data = logspace(1.0, 4.0, 40);
        let dict = vec![Representation::Log, Representation::Anscombe];
        let report = select_transform(&data, 1e6, &dict);
        assert_eq!(report.chosen, Some(Representation::Anscombe));
    }

    #[test]
    fn empty_input_is_handled() {
        let report = select_transform(&[], 100.0, &default_positive_dictionary());
        assert_eq!(report.chosen, None);
        assert!(report.verdicts.is_empty());
    }
}
