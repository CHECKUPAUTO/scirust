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
use scirust_simd::dispatch::BackendKind;
use std::collections::HashMap;

/// Unit roundoff of `f64` (the invalid-region threshold uses `κ_rt·u ≥ ½`,
/// [CANR §3.3]).
const UNIT: f64 = f64::EPSILON / 2.0;

/// Fixed quantizer level count for this prototype (deliberately not a
/// searched axis — see module docs).
const QUANT_LEVELS: usize = 64;

// ---------------------------------------------------------------------------
// The representation graph's node type
// ---------------------------------------------------------------------------

/// A representation candidate for this prototype's search graph: either no
/// transform at all, or one of [`Representation`]'s certified pairs.
///
/// Modeled as its own [`CertifiedMonotone`] implementation — Identity's
/// `kappa_rt` is exactly `1.0` everywhere (the identity map's condition
/// number, `|x / (x·1)| = 1`) — so the existing certificate-gate *pattern*
/// from `transform_search.rs` applies unchanged to a dictionary that
/// includes "no transform" as a first-class, reachable member, without
/// modifying [`Representation`] itself (ANEE §4's representation graph, at
/// the scope of a single hop from a hub node: every conversion here goes
/// through `f64` as the hub, exactly as [`crate::transform_search`]
/// already assumes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RepresentationChoice {
    /// No transform: store/accumulate directly in `f64`/`f32`.
    Identity,
    /// One of [`Representation`]'s certified monotone pairs.
    Certified(Representation),
}

impl RepresentationChoice {
    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self
        {
            RepresentationChoice::Identity => "identity",
            RepresentationChoice::Certified(r) => r.name(),
        }
    }

    /// Relative encode+decode cost proxy, reusing [`Representation::cost`];
    /// `0` for the no-op identity.
    pub fn cost(self) -> u32 {
        match self
        {
            RepresentationChoice::Identity => 0,
            RepresentationChoice::Certified(r) => r.cost(),
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
        }
    }

    fn encode(&self, x: f64) -> Option<f64> {
        match self
        {
            RepresentationChoice::Identity => self.domain().contains(x).then_some(x),
            RepresentationChoice::Certified(r) => r.encode(x),
        }
    }

    fn decode(&self, y: f64) -> f64 {
        match self
        {
            RepresentationChoice::Identity => y,
            RepresentationChoice::Certified(r) => r.decode(y),
        }
    }

    fn kappa_rt(&self, x: f64) -> f64 {
        match self
        {
            RepresentationChoice::Identity => 1.0,
            RepresentationChoice::Certified(r) => r.kappa_rt(x),
        }
    }

    fn kappa_rt_sup(&self, iv: Interval) -> f64 {
        match self
        {
            RepresentationChoice::Identity => 1.0,
            RepresentationChoice::Certified(r) => r.kappa_rt_sup(iv),
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
    let r = plan.representation;
    let domain = r.domain();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in fit.iter().chain(score_on)
    {
        if !domain.contains(x)
        {
            return None;
        }
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let support = Interval::new(lo, hi);
    if r.kappa_rt_sup(support) * UNIT >= 0.5
    {
        return None; // invalid region, CANR Sec.3.3.
    }

    let enc_fit: Vec<f64> = fit
        .iter()
        .map(|&x| r.encode(x).expect("gated above"))
        .collect();
    let q = UniformQuantizer::fit(&enc_fit, QUANT_LEVELS);

    let reconstructed: Vec<f32> = score_on
        .iter()
        .map(|&x| {
            let e = r.encode(x).expect("gated above");
            let qe = q.round_trip(e);
            r.decode(qe) as f32
        })
        .collect();

    let total = accumulate(plan.accumulation, &reconstructed) as f64;
    let exact = sum_expansion(score_on);
    if exact == 0.0
    {
        return None;
    }
    Some(((total - exact) / exact).abs())
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
        if !r.domain().contains(lo) || !r.domain().contains(hi)
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
        pipeline_relative_error(
            Plan {
                representation: r,
                accumulation: a,
            },
            fit,
            scr,
        )
        .map(|e| -e)
    };
    let baseline = move |fit: &[f64], scr: &[f64]| -> f64 {
        pipeline_relative_error(
            Plan {
                representation: r,
                accumulation: AccumMethod::NaiveF32,
            },
            fit,
            scr,
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
    let candidates: Vec<Plan> = r_dict
        .iter()
        .flat_map(|&r| {
            a_dict.iter().map(move |&a| Plan {
                representation: r,
                accumulation: a,
            })
        })
        .collect();
    let score =
        |plan: Plan, fit: &[f64], scr: &[f64]| pipeline_relative_error(plan, fit, scr).map(|e| -e);
    let baseline = |fit: &[f64], scr: &[f64]| -> f64 {
        pipeline_relative_error(
            Plan {
                representation: RepresentationChoice::Identity,
                accumulation: AccumMethod::NaiveF32,
            },
            fit,
            scr,
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
}
