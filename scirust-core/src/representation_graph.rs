//! **Phase C prototype: representation graph + distribution-aware plan cache.**
//!
//! (`docs/research/ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md`,
//! §4/§6/§12/§13.) This is the *one* narrow, falsifiable prototype the ANEE
//! report recommended — not a general planner, not a new crate. It tests
//! whether *jointly* searching a representation (`R`) together with an
//! accumulation strategy (`A`), with a plan cache keyed by the data's
//! *distribution* (not just kernel identity and hardware, the key every
//! existing autotuning cache surveyed in the report uses), beats the
//! sequential, per-axis baseline every piece of this search already
//! independently supports: [`crate::transform_search`] (the certificate
//! gate), [`crate::transform_autotune`] (the generic dev/held-out harness
//! and, reused directly, its `UniformQuantizer`), and
//! [`crate::autotune_accumulate`] (the accumulation-strategy dictionary).
//!
//! ## The task
//!
//! A realistic "compress, store, later aggregate" pipeline for positive
//! sensor-style readings: encode each reading via a certified representation
//! `R`, quantize/dequantize it (fixed at 64 levels, matching [CANR]'s own
//! convention — the quantization *level count* is deliberately **not** a
//! searched axis here, to keep this prototype narrow per the report's own
//! recommendation), decode back to `f64`, narrow to `f32` (the
//! "compute wide, store narrow" pattern already documented in
//! [`crate::autotune_accumulate`]), then accumulate the reconstructed
//! readings via strategy `A`. The objective is the **held-out** relative
//! error of the accumulated total against the exact sum of the *original*
//! (pre-encode) readings ([`crate::certified_numerics::sum_expansion`]) — an
//! end-to-end, realistic objective that only the *pair* `(R, A)` jointly
//! determines, not either axis alone (the pipeline concretely generalizes
//! [CANR §1]'s H3 finding — "representations must be selected as pairs with
//! operators" — from `(representation, operator)` to
//! `(representation, accumulation)`).
//!
//! ## Pre-registered kill criterion
//!
//! Written into this module *before* the benchmark below was run, per this
//! research program's own established discipline ([TSA]/[ATRA]/[CANR]):
//!
//! > On held-out data, joint `(R, A)` search must reduce the relative error
//! > versus the sequential baseline ([`sequential_baseline`]: certificate-
//! > gated `R` selection first, then `A` selection) by **at least 20%
//! > relative** (`joint_error ≤ 0.8 × baseline_error`) on **at least 2 of
//! > the 3** tested workload families (benign / wide-range /
//! > stagnation-prone — see `examples/anee_phase_c_prototype.rs`). If this
//! > is not met, "distribution-aware joint (R,A) search beats sequential
//! > per-axis selection" is **falsified for this task**, and this line is
//! > closed exactly as [TSA]'s Γ-transform and [ATRA]'s
//! > hypercomplex-transform directions were closed.
//!
//! Results are reported honestly regardless of outcome — see the ANEE
//! document's Phase C addendum for the quoted numbers from this module's
//! companion example.

use crate::autotune_accumulate::{AccumMethod, accumulate};
use crate::certified_numerics::{CertifiedMonotone, Interval, UlpBound, sum_expansion};
use crate::transform_autotune::{GenericAutotune, UniformQuantizer, autotune_by};
use crate::transform_search::Representation;
use scirust_simd::dispatch::{BackendKind, detect_backend};
use std::collections::HashMap;

/// Unit roundoff of `f64` (the invalid-region threshold uses `κ_rt·u ≥ ½`,
/// [CANR §3.3]).
const UNIT: f64 = f64::EPSILON / 2.0;

/// Default quantizer level count for this prototype (deliberately not a
/// *searched* axis — see module docs). The dose-response experiment (ANEE
/// Addendum 3) varies it as an environmental parameter through the
/// `*_with_levels` entry points; the parameterless functions keep this
/// default so kernel 1's published results remain reproducible unchanged.
const QUANT_LEVELS: usize = 64;

// ---------------------------------------------------------------------------
// The representation graph's node type
// ---------------------------------------------------------------------------

/// A representation candidate for this prototype's search graph: no
/// transform at all, one of [`Representation`]'s certified pairs, or — the
/// two-hop graph experiment (ANEE Addendum 3) — a *composition* of two
/// certified pairs applied in sequence.
///
/// Modeled as its own [`CertifiedMonotone`] implementation — Identity's
/// `kappa_rt` is exactly `1.0` everywhere (the identity map's condition
/// number, `|x / (x·1)| = 1`) — so the existing certificate-gate *pattern*
/// from `transform_search.rs` applies unchanged to a dictionary that
/// includes "no transform" as a first-class, reachable member, without
/// modifying [`Representation`] itself. Single hops go through `f64` as the
/// hub exactly as [`crate::transform_search`] already assumes;
/// [`RepresentationChoice::Composed`] is the first member that actually
/// *exercises the graph structure* of ANEE §4 (a two-edge path) rather than
/// a flat dictionary — its condition number is the **exact product** of the
/// hops' condition numbers per Proposition ANEE-2 (the elasticity chain
/// rule, validated to Decimal prec-60 by experiment Z3), used here in real
/// code for the first time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RepresentationChoice {
    /// No transform: store/accumulate directly in `f64`/`f32`.
    Identity,
    /// One of [`Representation`]'s certified monotone pairs.
    Certified(Representation),
    /// Two certified pairs applied in sequence: `Composed(a, b)` encodes
    /// `x ↦ b(a(x))` and decodes `y ↦ a⁻¹(b⁻¹(y))`. Both hops are strictly
    /// monotone, so the composition is too. `domain()` reports the *first*
    /// hop's interval as an over-approximation — [`Self::encode`] is the
    /// authoritative membership test (it returns `None` whenever the
    /// intermediate value leaves the second hop's domain), and the pipeline
    /// gate uses `encode`, not `domain`, for exactly this reason. The
    /// default [`CertifiedMonotone::roundtrip_bound`] under-counts one
    /// intermediate rounding per extra hop (it charges `B_ENC` once);
    /// certificates are not the decision variable in these experiments
    /// (held-out error is), so this is documented rather than patched.
    Composed(Representation, Representation),
}

impl RepresentationChoice {
    /// Human-readable name (owned: composed names are built dynamically).
    pub fn name(self) -> String {
        match self
        {
            RepresentationChoice::Identity => "identity".to_string(),
            RepresentationChoice::Certified(r) => r.name().to_string(),
            RepresentationChoice::Composed(a, b) => format!("{}->{}", a.name(), b.name()),
        }
    }

    /// Relative encode+decode cost proxy, reusing [`Representation::cost`];
    /// `0` for the no-op identity, the sum of hops for a composition.
    pub fn cost(self) -> u32 {
        match self
        {
            RepresentationChoice::Identity => 0,
            RepresentationChoice::Certified(r) => r.cost(),
            RepresentationChoice::Composed(a, b) => a.cost() + b.cost(),
        }
    }
}

impl CertifiedMonotone for RepresentationChoice {
    fn domain(&self) -> Interval {
        match self
        {
            // Matches this prototype's task: positive sensor-style readings.
            RepresentationChoice::Identity => Interval::new(0.0, f64::MAX),
            RepresentationChoice::Certified(r) => r.domain(),
            // Over-approximation (see the variant's docs): encode() is the
            // authoritative membership test for compositions.
            RepresentationChoice::Composed(a, _) => a.domain(),
        }
    }

    fn encode(&self, x: f64) -> Option<f64> {
        match self
        {
            RepresentationChoice::Identity => self.domain().contains(x).then_some(x),
            RepresentationChoice::Certified(r) => r.encode(x),
            RepresentationChoice::Composed(a, b) => a.encode(x).and_then(|y| b.encode(y)),
        }
    }

    fn decode(&self, y: f64) -> f64 {
        match self
        {
            RepresentationChoice::Identity => y,
            RepresentationChoice::Certified(r) => r.decode(y),
            RepresentationChoice::Composed(a, b) => a.decode(b.decode(y)),
        }
    }

    fn kappa_rt(&self, x: f64) -> f64 {
        match self
        {
            RepresentationChoice::Identity => 1.0,
            RepresentationChoice::Certified(r) => r.kappa_rt(x),
            // Exact multiplicative composition (Proposition ANEE-2 / Z3):
            // kappa of the second hop is evaluated at the first hop's image.
            RepresentationChoice::Composed(a, b) => match a.encode(x)
            {
                Some(y) => a.kappa_rt(x) * b.kappa_rt(y),
                None => f64::INFINITY,
            },
        }
    }

    fn kappa_rt_sup(&self, iv: Interval) -> f64 {
        match self
        {
            RepresentationChoice::Identity => 1.0,
            RepresentationChoice::Certified(r) => r.kappa_rt_sup(iv),
            // Sound bound: product of per-hop sups >= sup of the pointwise
            // product. Every dictionary member is strictly increasing, so
            // the first hop maps the interval to an interval endpoint-wise.
            RepresentationChoice::Composed(a, b) => match (a.encode(iv.lo), a.encode(iv.hi))
            {
                (Some(ya), Some(yb)) =>
                {
                    a.kappa_rt_sup(iv) * b.kappa_rt_sup(Interval::new(ya.min(yb), ya.max(yb)))
                },
                _ => f64::INFINITY,
            },
        }
    }
}

/// The default representation dictionary for this prototype: positive-
/// domain-valid members only ([`Representation::SignedLog`]/`Logit`/`MuLaw`
/// are for signed/bounded data and would be domain-mismatched for the
/// sensor-reading task here — including them would just make every
/// candidate using them fail the certificate gate), plus
/// [`RepresentationChoice::Identity`] so "no transform is best" is a real,
/// reachable outcome, not assumed away.
pub fn default_representation_dictionary() -> Vec<RepresentationChoice> {
    vec![
        RepresentationChoice::Identity,
        RepresentationChoice::Certified(Representation::Log),
        RepresentationChoice::Certified(Representation::Log1p),
        RepresentationChoice::Certified(Representation::Power(0.5)),
        RepresentationChoice::Certified(Representation::Anscombe),
    ]
}

/// Every ordered two-hop composition this prototype's graph experiment
/// considers: first hop from the positive-domain singles, second hop from
/// those plus [`Representation::SignedLog`] (the one dictionary member whose
/// domain is all of ℝ — first-hop outputs can be negative, e.g. `log x < 0`
/// for `x < 1`, so it is only reachable as a *second* hop on this task).
///
/// Deliberately unfiltered: pairs whose intermediate values leave the second
/// hop's domain are rejected per-workload by the pipeline's encode-based
/// gate, and pairs that are affine-equivalent to a single hop (e.g.
/// `power(λ)` then `log` = `λ·log`, an affine image of `log`) are expected
/// to *tie* their single-hop equivalent under the affine-invariant uniform
/// quantizer — both facts are part of what the two-hop experiment is
/// honestly measuring, not noise to be pre-cleaned away.
pub fn two_hop_dictionary() -> Vec<RepresentationChoice> {
    let firsts = [
        Representation::Log,
        Representation::Log1p,
        Representation::Power(0.5),
        Representation::Anscombe,
    ];
    let seconds = [
        Representation::Log,
        Representation::Log1p,
        Representation::SignedLog,
        Representation::Power(0.5),
        Representation::Anscombe,
    ];
    let mut out = Vec::new();
    for &a in &firsts
    {
        for &b in &seconds
        {
            out.push(RepresentationChoice::Composed(a, b));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// A composed execution plan and its held-out score
// ---------------------------------------------------------------------------

/// A chosen `(representation, accumulation)` pair — the object this
/// prototype's plan cache stores, and the smallest useful instance of ANEE's
/// general `(R,O,A,T,Q,M,H)` execution plan (ANEE §1.2): `O` (the aggregate
/// operator is always "sum"), `T` (no separate transform stage), `Q` (fixed
/// at [`QUANT_LEVELS`]), and `M` are held fixed by the task; `H` is a
/// component of the cache key below, not a per-plan field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// Chosen representation.
    pub representation: RepresentationChoice,
    /// Chosen accumulation strategy.
    pub accumulation: AccumMethod,
}

/// Run the full encode → quantize → dequantize → decode → narrow-to-f32 →
/// accumulate pipeline for one `(representation, accumulation)` pair, fit on
/// `fit` and scored on `score_on`. Returns `None` if `representation` is not
/// certified-safe on the union of `fit`/`score_on` (out of domain, or in the
/// `kappa_rt · u ≥ ½` invalid region — the same S1 gate
/// `transform_search.rs` already enforces, checked here via
/// [`CertifiedMonotone`] directly rather than re-deriving it) — matching the
/// `dev.iter().chain(eval)` combined-support convention
/// `transform_autotune::safety_gate` already established.
pub fn pipeline_relative_error(plan: Plan, fit: &[f64], score_on: &[f64]) -> Option<f64> {
    pipeline_relative_error_with_levels(plan, fit, score_on, QUANT_LEVELS)
}

/// [`pipeline_relative_error`] with the quantizer level count as an explicit
/// parameter — the environmental knob the dose-response experiment (ANEE
/// Addendum 3) varies. The gate is *encode-based* (a sample is admissible
/// iff `encode` succeeds), which coincides with the previous interval-domain
/// check for every single-hop member and is the only correct test for
/// [`RepresentationChoice::Composed`] (whose true domain is the preimage of
/// the second hop's domain under the first hop).
pub fn pipeline_relative_error_with_levels(
    plan: Plan,
    fit: &[f64],
    score_on: &[f64],
    levels: usize,
) -> Option<f64> {
    let reconstructed = reconstruct_with_levels(plan.representation, fit, score_on, levels)?;
    let total = accumulate(plan.accumulation, &reconstructed) as f64;
    let exact = sum_expansion(score_on);
    if exact == 0.0
    {
        return None;
    }
    Some(((total - exact) / exact).abs())
}

/// The reconstruction stage of the pipeline — encode (gated) → fit quantizer
/// on `fit` → round-trip → decode → narrow to `f32` — **without** the final
/// accumulation. This is [`pipeline_relative_error_with_levels`] minus its
/// last step, extracted so an execution-regime experiment (Phase D's D6) can
/// compose the identical reconstruction with a *chunked* accumulation; the
/// split is a pure refactor (the pipeline functions delegate here), enforced
/// bitwise by a test below.
pub fn reconstruct_with_levels(
    representation: RepresentationChoice,
    fit: &[f64],
    score_on: &[f64],
    levels: usize,
) -> Option<Vec<f32>> {
    let r = representation;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut enc_fit = Vec::with_capacity(fit.len());
    for &x in fit
    {
        enc_fit.push(r.encode(x)?);
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let mut enc_score = Vec::with_capacity(score_on.len());
    for &x in score_on
    {
        enc_score.push(r.encode(x)?);
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let support = Interval::new(lo, hi);
    if r.kappa_rt_sup(support) * UNIT >= 0.5
    {
        return None; // invalid region, CANR Sec.3.3.
    }

    let q = UniformQuantizer::fit(&enc_fit, levels);
    Some(
        enc_score
            .iter()
            .map(|&e| r.decode(q.round_trip(e)) as f32)
            .collect(),
    )
}

/// Outcome of one search approach: the chosen plan, its held-out relative
/// error (lower is better), and the chosen representation's own certified
/// round-trip bound over the training support ([`CertifiedMonotone::roundtrip_bound`],
/// reused unchanged — not re-derived).
#[derive(Debug, Clone)]
pub struct PlanSearchReport {
    /// The chosen plan.
    pub plan: Plan,
    /// Held-out relative error of the chosen plan's accumulated total
    /// against the exact sum of the original readings.
    pub held_out_relative_error: f64,
    /// The chosen representation's certified round-trip bound (ulps) over
    /// `[min(dev ∪ eval), max(dev ∪ eval)]`.
    pub certificate: UlpBound,
}

/// The sequential ("CANR S1–S4 style") baseline: pick `R` first via the
/// existing certificate-gated selection rule (cheapest certified-safe
/// candidate on `dev`'s support — [`Representation`]'s own convention in
/// `transform_search::select_transform`, applied here to a dictionary that
/// additionally includes [`RepresentationChoice::Identity`]), **then** pick
/// `A` via [`autotune_by`] with `R` held fixed — never revisiting `R` once
/// `A` search starts. This is the honest baseline experiment Z2
/// (`docs/research/anee_experiments/anee_experiments.py`) argued for: what
/// SciRust's already-existing tools do today, called once per axis.
pub fn sequential_baseline(
    dev: &[f64],
    eval: &[f64],
    r_dict: &[RepresentationChoice],
    a_dict: &[AccumMethod],
) -> Option<PlanSearchReport> {
    sequential_baseline_with_levels(dev, eval, r_dict, a_dict, QUANT_LEVELS)
}

/// [`sequential_baseline`] with the quantizer level count as an explicit
/// parameter. The S1 gate checks encodability of the dev support's
/// endpoints — equivalent to the previous interval-domain check for single
/// hops (every member is strictly increasing with an interval domain, so
/// endpoint admissibility implies full-support admissibility), and correct
/// for compositions.
pub fn sequential_baseline_with_levels(
    dev: &[f64],
    eval: &[f64],
    r_dict: &[RepresentationChoice],
    a_dict: &[AccumMethod],
    levels: usize,
) -> Option<PlanSearchReport> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in dev
    {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let dev_support = Interval::new(lo, hi);

    // S1: cheapest certified-safe R on dev's support alone (this step
    // precedes any dev/held-out split for A, exactly mirroring
    // transform_search::select_transform's single-sample gate).
    let mut chosen_r: Option<RepresentationChoice> = None;
    for &r in r_dict
    {
        if r.encode(lo).is_none() || r.encode(hi).is_none()
        {
            continue;
        }
        if r.kappa_rt_sup(dev_support) * UNIT >= 0.5
        {
            continue;
        }
        let better = match chosen_r
        {
            None => true,
            Some(c) => r.cost() < c.cost(),
        };
        if better
        {
            chosen_r = Some(r);
        }
    }
    let r = chosen_r?;

    // S3/S4: A search with R held fixed, via the existing generic harness.
    let score = move |a: AccumMethod, fit: &[f64], scr: &[f64]| -> Option<f64> {
        pipeline_relative_error_with_levels(
            Plan {
                representation: r,
                accumulation: a,
            },
            fit,
            scr,
            levels,
        )
        .map(|e| -e)
    };
    let baseline = move |fit: &[f64], scr: &[f64]| -> f64 {
        pipeline_relative_error_with_levels(
            Plan {
                representation: r,
                accumulation: AccumMethod::NaiveF32,
            },
            fit,
            scr,
            levels,
        )
        .map(|e| -e)
        .unwrap_or(f64::NEG_INFINITY)
    };
    let out: GenericAutotune<AccumMethod> = autotune_by(dev, eval, a_dict, score, baseline);
    let a = out.chosen?;
    let held_out_relative_error = -out.chosen_eval_score;

    let mut lo_all = f64::INFINITY;
    let mut hi_all = f64::NEG_INFINITY;
    for &x in dev.iter().chain(eval)
    {
        lo_all = lo_all.min(x);
        hi_all = hi_all.max(x);
    }
    let full_support = Interval::new(lo_all, hi_all);
    Some(PlanSearchReport {
        plan: Plan {
            representation: r,
            accumulation: a,
        },
        held_out_relative_error,
        certificate: r.roundtrip_bound(full_support),
    })
}

/// Joint search: the **same** generic harness ([`autotune_by`]), fed the
/// Cartesian product of `r_dict × a_dict` as its candidate type — literally
/// the claim ANEE §6/§13 makes concrete: the missing piece was never the
/// harness, only the product-typed candidate list and the certificate gate
/// applied jointly (inside [`pipeline_relative_error`]) instead of per-axis.
pub fn joint_search(
    dev: &[f64],
    eval: &[f64],
    r_dict: &[RepresentationChoice],
    a_dict: &[AccumMethod],
) -> Option<PlanSearchReport> {
    joint_search_with_levels(dev, eval, r_dict, a_dict, QUANT_LEVELS)
}

/// [`joint_search`] with the quantizer level count as an explicit parameter.
pub fn joint_search_with_levels(
    dev: &[f64],
    eval: &[f64],
    r_dict: &[RepresentationChoice],
    a_dict: &[AccumMethod],
    levels: usize,
) -> Option<PlanSearchReport> {
    let candidates: Vec<Plan> = r_dict
        .iter()
        .flat_map(|&r| {
            a_dict.iter().map(move |&a| Plan {
                representation: r,
                accumulation: a,
            })
        })
        .collect();
    let score = move |plan: Plan, fit: &[f64], scr: &[f64]| {
        pipeline_relative_error_with_levels(plan, fit, scr, levels).map(|e| -e)
    };
    let baseline = move |fit: &[f64], scr: &[f64]| -> f64 {
        pipeline_relative_error_with_levels(
            Plan {
                representation: RepresentationChoice::Identity,
                accumulation: AccumMethod::NaiveF32,
            },
            fit,
            scr,
            levels,
        )
        .map(|e| -e)
        .unwrap_or(f64::NEG_INFINITY)
    };
    let out: GenericAutotune<Plan> = autotune_by(dev, eval, &candidates, score, baseline);
    let plan = out.chosen?;
    let held_out_relative_error = -out.chosen_eval_score;

    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in dev.iter().chain(eval)
    {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let support = Interval::new(lo, hi);
    Some(PlanSearchReport {
        plan,
        held_out_relative_error,
        certificate: plan.representation.roundtrip_bound(support),
    })
}

// ---------------------------------------------------------------------------
// The guarded ablation-first decision rule (Addendum 3, validated 13/15)
// ---------------------------------------------------------------------------

/// The Addendum-3 gap threshold: predict "joint search pays" when the cheap
/// R-axis ablation improves on the default by at least this fraction. The
/// 15-cell dose-response test validated this value prospectively (13/15,
/// with both misses at the noise floor — hence the guard below).
pub const ABLATION_GAP_THRESHOLD: f64 = 0.20;

/// Verdict of the guarded ablation-first predictor ([`ablation_first_advice`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AblationAdvice {
    /// The default plan's error is already at or below the caller's floor:
    /// relative gaps down there are noise (the dose-response run's only two
    /// mispredictions were exactly this case) — do not invest in joint
    /// search.
    AtErrorFloor {
        /// Held-out-free (dev-only) relative error of the default plan.
        default_error: f64,
    },
    /// The R-axis ablation gap meets the threshold: joint (R, A) search is
    /// predicted to pay.
    JointSearchPays {
        /// The ablation gap `1 − best_R / default`.
        gap: f64,
        /// Dev relative error of the default plan.
        default_error: f64,
    },
    /// The default representation is near-optimal on the R axis: joint
    /// search is predicted not to pay — run [`sequential_baseline`] (or
    /// keep the default) instead.
    DefaultAdequate {
        /// The ablation gap `1 − best_R / default`.
        gap: f64,
        /// Dev relative error of the default plan.
        default_error: f64,
    },
}

/// **The validated decision rule of this research line** (ANEE Addendum 3,
/// avenue 1): before paying for a joint `(R, A)` search, run a cheap
/// single-axis ablation on dev data only — score every representation in
/// `r_dict` with accumulation held fixed at [`AccumMethod::NeumaierF32`],
/// and compare the best against the [`RepresentationChoice::Identity`]
/// default. Joint search pays *exactly where the default representation is
/// objective-blind and wrong* (gap ≥ [`ABLATION_GAP_THRESHOLD`]).
///
/// `error_floor` is the **mandatory guard** the dose-response test added:
/// when the default's error already meets the caller's accuracy target,
/// relative gaps are noise-floor artifacts — both of the 15-cell run's
/// mispredictions were false "pays" of that type, and the guard removes
/// exactly that failure class. Pass the application's actual accuracy
/// target (a relative error); pass `0.0` to disable the guard and
/// reproduce the unguarded 13/15 predictor.
///
/// Returns `None` when the default plan itself cannot be scored on `dev`
/// (zero exact sum — relative error undefined).
pub fn ablation_first_advice(
    dev: &[f64],
    r_dict: &[RepresentationChoice],
    levels: usize,
    error_floor: f64,
) -> Option<AblationAdvice> {
    let err = |r: RepresentationChoice| {
        pipeline_relative_error_with_levels(
            Plan {
                representation: r,
                accumulation: AccumMethod::NeumaierF32,
            },
            dev,
            dev,
            levels,
        )
    };
    let default_error = err(RepresentationChoice::Identity)?;
    if default_error <= error_floor
    {
        return Some(AblationAdvice::AtErrorFloor { default_error });
    }
    if default_error == 0.0
    {
        // Exactly zero error with a zero floor: nothing to improve.
        return Some(AblationAdvice::DefaultAdequate {
            gap: 0.0,
            default_error,
        });
    }
    let best = r_dict
        .iter()
        .filter_map(|&r| err(r))
        .fold(default_error, f64::min);
    let gap = 1.0 - best / default_error;
    if gap >= ABLATION_GAP_THRESHOLD
    {
        Some(AblationAdvice::JointSearchPays { gap, default_error })
    }
    else
    {
        Some(AblationAdvice::DefaultAdequate { gap, default_error })
    }
}

// ---------------------------------------------------------------------------
// Distribution summary and the distribution-aware plan cache
// ---------------------------------------------------------------------------

/// A cheap, deterministic summary of a data batch's numeric distribution —
/// the extra plan-cache key component ANEE §12.2 argues is missing from
/// every existing autotuning cache examined in the report's literature
/// review (FFTW's wisdom, AutoTVM's TopHub, SPIRAL's learned cost model,
/// Kokkos+APEX's tuning YAML — all keyed on kernel shape and/or hardware
/// alone): a **schedule's** optimal choice doesn't depend on data *values*,
/// but a **representation/precision** choice's *correctness* does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DistributionSummary {
    /// `log10(max|x| / min|x|)` over nonzero samples, rounded to the nearest
    /// decade (so near-identical distributions collide in the cache; wildly
    /// different ones do not).
    pub log10_range_decades: i32,
    /// Whether at least 10% of samples are more than 3 decades smaller than
    /// the batch's max magnitude — the "stagnation risk" signal that
    /// motivates compensated accumulation ([`crate::autotune_accumulate`]'s
    /// own `wide_batch` test fixture uses exactly this shape of data).
    pub stagnation_risk: bool,
}

/// Summarize `xs`'s distribution for plan-cache keying.
pub fn summarize(xs: &[f64]) -> DistributionSummary {
    let mut min_abs = f64::INFINITY;
    let mut max_abs = 0.0_f64;
    for &x in xs
    {
        let a = x.abs();
        if a > 0.0
        {
            min_abs = min_abs.min(a);
        }
        max_abs = max_abs.max(a);
    }
    let ratio = if min_abs.is_finite() && min_abs > 0.0
    {
        (max_abs / min_abs).max(1.0)
    }
    else
    {
        1.0
    };
    let threshold = max_abs * 1e-3;
    let small_count = xs.iter().filter(|&&x| x.abs() < threshold).count();
    let stagnation_risk = !xs.is_empty() && (small_count as f64 / xs.len() as f64) >= 0.1;
    DistributionSummary {
        log10_range_decades: ratio.log10().round() as i32,
        stagnation_risk,
    }
}

/// A cached plan with the confidence/certificate/history fields the ANEE
/// mission's "Learning execution plans" section proposed.
#[derive(Debug, Clone)]
pub struct CachedPlan {
    /// The cached plan.
    pub plan: Plan,
    /// Held-out relative error achieved when this plan was chosen (lower is
    /// better) — the "confidence" field.
    pub held_out_relative_error: f64,
    /// The plan representation's own certified round-trip bound, reused
    /// unchanged from [`CertifiedMonotone::roundtrip_bound`].
    pub certificate: UlpBound,
    /// Every held-out score ever recorded for this key — the "benchmark
    /// history" field.
    pub history: Vec<f64>,
}

/// A plan cache keyed by `(kernel identity, data distribution, hardware)` —
/// ANEE §12.2's candidate extension of every existing `(kernel, hardware)`
/// cache examined in the report's literature review.
#[derive(Debug, Default)]
pub struct PlanCache {
    entries: HashMap<(String, DistributionSummary, BackendKind), CachedPlan>,
}

impl PlanCache {
    /// A new, empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a cached plan for this exact `(kernel, distribution,
    /// hardware)` key. Returns `None` on a miss — including when the
    /// distribution bucket differs from anything previously cached for the
    /// same kernel and hardware; that miss is the point: a `(kernel,
    /// hardware)`-only cache could not distinguish this case and would
    /// silently return a plan tuned for the wrong distribution instead.
    pub fn get(
        &self,
        kernel: &str,
        dist: DistributionSummary,
        hw: BackendKind,
    ) -> Option<&CachedPlan> {
        self.entries.get(&(kernel.to_string(), dist, hw))
    }

    /// Insert (or update, appending to history) a cache entry.
    pub fn insert(
        &mut self,
        kernel: &str,
        dist: DistributionSummary,
        hw: BackendKind,
        plan: Plan,
        held_out_relative_error: f64,
        certificate: UlpBound,
    ) {
        let key = (kernel.to_string(), dist, hw);
        self.entries
            .entry(key)
            .and_modify(|c| {
                c.plan = plan;
                c.held_out_relative_error = held_out_relative_error;
                c.certificate = certificate;
                c.history.push(held_out_relative_error);
            })
            .or_insert_with(|| CachedPlan {
                plan,
                held_out_relative_error,
                certificate,
                history: vec![held_out_relative_error],
            });
    }
}

/// The hardware component (ANEE's `H` axis) of [`PlanCache`] keys, as the
/// **single sanctioned source**. Decision, documented per the Phase D
/// pre-registration's E2: [`BackendKind`] stays the key *type* — it is
/// `Copy + Eq + Hash` and allocation-free, which a registry label string is
/// not — while `crate::compute_capability` remains the *reporting* view.
/// The two cannot drift: the registry's CPU entry is seeded from the same
/// [`detect_backend`] call this function returns, and a test in this module
/// pins `current_hardware_key().label()` to the registry's `cpu-simd` entry.
pub fn current_hardware_key() -> BackendKind {
    detect_backend()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autotune_accumulate::default_accumulators;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn benign(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(0.5..1.5)).collect()
    }

    fn wide_range(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| 10f64.powf(rng.gen_range(-6.0..6.0)))
            .collect()
    }

    #[test]
    fn identity_pipeline_has_bounded_quantization_error_only() {
        let dev = benign(1, 4096);
        let eval = benign(2, 4096);
        let plan = Plan {
            representation: RepresentationChoice::Identity,
            accumulation: AccumMethod::NeumaierF32,
        };
        let err =
            pipeline_relative_error(plan, &dev, &eval).expect("identity always certified here");
        // Benign, narrow-range positive data at 64 quantization levels: error
        // should be small (quantization noise only) and not blow up.
        assert!(err < 0.05, "unexpectedly large error: {err}");
    }

    #[test]
    fn representation_gate_rejects_anscombe_near_zero() {
        // Anscombe's invalid region starts around x ~ 1.7e-16 (CANR Sec.3.3);
        // a sample there must be gated out, not silently scored.
        let dev = vec![1e-18, 0.5, 4.0];
        let eval = vec![0.5, 4.0];
        let plan = Plan {
            representation: RepresentationChoice::Certified(Representation::Anscombe),
            accumulation: AccumMethod::NaiveF32,
        };
        assert!(pipeline_relative_error(plan, &dev, &eval).is_none());
    }

    #[test]
    fn joint_search_dev_score_is_never_worse_than_sequential_would_reach() {
        // Structural invariant (true by construction, not an empirical
        // finding): joint search evaluates a superset of the candidates
        // sequential search can ever reach (it includes every (R, A) pair,
        // in particular (R_seq, A) for every A), so on DEV data its best
        // score can never be worse. This is a regression guard, not the
        // held-out empirical claim under test (see the companion example).
        let dev = wide_range(3, 2048);
        let eval = wide_range(4, 2048);
        let r_dict = default_representation_dictionary();
        let a_dict = default_accumulators();

        let seq = sequential_baseline(&dev, &eval, &r_dict, &a_dict);
        let joint = joint_search(&dev, &eval, &r_dict, &a_dict);
        if let (Some(seq), Some(joint)) = (seq, joint)
        {
            // Re-score both chosen plans on dev alone for a same-footing
            // comparison (held-out scores are not directly comparable this
            // way, since eval is what varies the outcome).
            let seq_dev = pipeline_relative_error(seq.plan, &dev, &dev);
            let joint_dev = pipeline_relative_error(joint.plan, &dev, &dev);
            if let (Some(seq_dev), Some(joint_dev)) = (seq_dev, joint_dev)
            {
                assert!(
                    joint_dev <= seq_dev * 1.0 + 1e-12,
                    "joint dev error {joint_dev} worse than sequential {seq_dev}: \
                     joint search must dominate on dev by construction"
                );
            }
        }
    }

    #[test]
    fn plan_cache_hits_same_distribution_and_misses_different_one() {
        let mut cache = PlanCache::new();
        let hw = BackendKind::Scalar;
        let plan = Plan {
            representation: RepresentationChoice::Certified(Representation::Log),
            accumulation: AccumMethod::NeumaierF32,
        };
        let dist_a = summarize(&wide_range(5, 1024));
        cache.insert(
            "sensor_aggregate_sum",
            dist_a,
            hw,
            plan,
            0.01,
            UlpBound { ulps: 10.0 },
        );

        // Same distribution bucket: hit.
        assert!(cache.get("sensor_aggregate_sum", dist_a, hw).is_some());

        // A distribution with a very different range: expect a different
        // bucket (miss), demonstrating the cache does not silently conflate
        // unrelated data distributions the way a (kernel, hardware)-only
        // key would.
        let dist_b = summarize(&benign(6, 1024));
        assert_ne!(
            dist_a, dist_b,
            "fixture distributions must differ for this test to be meaningful"
        );
        assert!(cache.get("sensor_aggregate_sum", dist_b, hw).is_none());
    }

    #[test]
    fn cache_history_accumulates_across_reuses() {
        let mut cache = PlanCache::new();
        let hw = BackendKind::Scalar;
        let dist = summarize(&benign(7, 512));
        let plan = Plan {
            representation: RepresentationChoice::Identity,
            accumulation: AccumMethod::NaiveF32,
        };
        cache.insert("k", dist, hw, plan, 0.02, UlpBound { ulps: 4.0 });
        cache.insert("k", dist, hw, plan, 0.018, UlpBound { ulps: 4.0 });
        let entry = cache.get("k", dist, hw).expect("must be present");
        assert_eq!(entry.history, vec![0.02, 0.018]);
    }

    #[test]
    fn composed_kappa_is_the_exact_product_of_hops() {
        // Z3 in Rust: for power(0.5) then log, the composed map is
        // phi(x) = ln(x^0.5) = 0.5 ln x, whose kappa_rt is
        // |phi/(x phi')| = |0.5 ln x / (x * 0.5/x)| = |ln x| exactly.
        // Proposition ANEE-2 says the product kappa_A(x)*kappa_B(A(x)) must
        // equal this EXACTLY (elasticity chain rule), not to first order.
        let c = RepresentationChoice::Composed(Representation::Power(0.5), Representation::Log);
        for &x in &[1e-9f64, 0.3, 2.5, 1e4, 1e9]
        {
            let expected = x.ln().abs();
            let got = c.kappa_rt(x);
            assert!(
                (got - expected).abs() <= expected.abs() * 1e-12 + 1e-12,
                "kappa product at x={x}: got {got}, expected |ln x| = {expected}"
            );
        }
    }

    #[test]
    fn composed_encode_decode_round_trips() {
        let c = RepresentationChoice::Composed(Representation::Power(0.5), Representation::Log1p);
        for &x in &[1e-6, 0.5, 1.0, 42.0, 1e6]
        {
            let y = c.encode(x).expect("in domain");
            let back = c.decode(y);
            assert!(
                ((back - x) / x).abs() < 1e-12,
                "round trip at x={x}: got {back}"
            );
        }
    }

    #[test]
    fn composed_gate_rejects_out_of_domain_intermediates() {
        // log(0.5) < 0, and Anscombe's domain is [0, inf): the composition
        // log-then-anscombe must be rejected on data crossing 1 — by encode
        // (the authoritative test), and therefore by the pipeline.
        let c = RepresentationChoice::Composed(Representation::Log, Representation::Anscombe);
        assert!(c.encode(0.5).is_none(), "negative intermediate must reject");
        assert!(c.encode(2.0).is_some(), "positive intermediate is fine");
        let plan = Plan {
            representation: c,
            accumulation: AccumMethod::NaiveF32,
        };
        let data = vec![0.5, 2.0, 3.0];
        assert!(pipeline_relative_error(plan, &data, &data).is_none());
    }

    #[test]
    fn two_hop_dictionary_is_complete_and_all_composed() {
        let d = two_hop_dictionary();
        assert_eq!(d.len(), 20); // 4 firsts x 5 seconds
        assert!(
            d.iter()
                .all(|r| matches!(r, RepresentationChoice::Composed(_, _)))
        );
    }

    #[test]
    fn with_levels_at_default_matches_parameterless_entry_points() {
        // Guard: kernel 1's published numbers must remain reproducible — the
        // parameterless functions and _with_levels(.., 64) must agree bit-
        // for-bit on identical inputs.
        let dev = wide_range(3, 2048);
        let eval = wide_range(4, 2048);
        let plan = Plan {
            representation: RepresentationChoice::Certified(Representation::Power(0.5)),
            accumulation: AccumMethod::PairwiseF32,
        };
        assert_eq!(
            pipeline_relative_error(plan, &dev, &eval),
            pipeline_relative_error_with_levels(plan, &dev, &eval, 64)
        );
    }

    #[test]
    fn reconstruct_then_accumulate_is_bitwise_the_pipeline() {
        // The Phase D refactor contract: extracting the reconstruction stage
        // must not change the pipeline's output by a single bit, including
        // for composed representations. (D6 relies on composing the same
        // reconstruction with a different accumulation regime.)
        let dev = wide_range(5, 2048);
        let eval = wide_range(6, 2048);
        for &r in &[
            RepresentationChoice::Identity,
            RepresentationChoice::Certified(Representation::Log),
            RepresentationChoice::Composed(Representation::Power(0.5), Representation::Log),
        ]
        {
            let plan = Plan {
                representation: r,
                accumulation: AccumMethod::NeumaierF32,
            };
            let via_pipeline = pipeline_relative_error_with_levels(plan, &dev, &eval, 64);
            let via_parts = reconstruct_with_levels(r, &dev, &eval, 64).map(|rec| {
                let total = accumulate(plan.accumulation, &rec) as f64;
                let exact = crate::certified_numerics::sum_expansion(&eval);
                ((total - exact) / exact).abs()
            });
            assert_eq!(via_pipeline, via_parts, "split diverged for {r:?}");
        }
    }

    #[test]
    fn ablation_advice_reproduces_the_guarded_dose_response_predictor() {
        let r_dict = default_representation_dictionary();

        // Wide-range at L = 8: the dose-response run's clearest "pays" cell
        // (ablation gap ≈ 97%) — with a small floor the advice must be
        // JointSearchPays with a large gap.
        let wide = wide_range(1, 4096);
        match ablation_first_advice(&wide, &r_dict, 8, 1e-9).expect("scorable")
        {
            AblationAdvice::JointSearchPays { gap, .. } =>
            {
                assert!(gap >= ABLATION_GAP_THRESHOLD, "gap {gap} below threshold")
            },
            other => panic!("expected JointSearchPays, got {other:?}"),
        }

        // Same batch, but the caller's accuracy target is loose (10%): the
        // default already meets it, so the guard must fire and veto the
        // search regardless of the gap — this is exactly the failure class
        // the unguarded 15-cell run exhibited (its 2 misses were false
        // "pays" at the floor).
        let benign_batch = benign(1, 4096);
        let unguarded = ablation_first_advice(&benign_batch, &r_dict, 64, 0.0).expect("scorable");
        let guarded = ablation_first_advice(&benign_batch, &r_dict, 64, 1e-2).expect("scorable");
        assert!(
            matches!(guarded, AblationAdvice::AtErrorFloor { default_error } if default_error <= 1e-2),
            "guard must fire on benign data with a 1% target: {guarded:?}"
        );
        assert!(
            !matches!(unguarded, AblationAdvice::AtErrorFloor { .. }),
            "with a zero floor the guard must never fire: {unguarded:?}"
        );
    }

    #[test]
    fn current_hardware_key_matches_the_capability_registry_seed() {
        // E2's anti-divergence pin: the H-axis cache key and the unified
        // capability registry must report the same CPU tier, forever.
        let key = current_hardware_key();
        let caps = crate::compute_capability::compute_capabilities();
        assert!(
            caps.iter().any(|c| {
                c.domain == crate::compute_capability::ComputeDomain::CpuSimd
                    && c.label == key.label()
                    && c.available == Some(true)
            }),
            "registry CPU entry must match current_hardware_key() = {key:?}: {caps:?}"
        );
    }
}
