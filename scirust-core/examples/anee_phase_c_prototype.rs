//! **ANEE Phase C prototype benchmark.**
//!
//! (`docs/research/ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md`
//! §13 Phase C; implementation in `scirust_core::representation_graph`.)
//!
//! Runs the pre-registered comparison — sequential (CANR S1-S4 style)
//! representation-then-accumulation selection vs. joint (R,A) search — on
//! three workload families, then evaluates the kill criterion that was
//! written down *before* this benchmark was run (see
//! `representation_graph`'s module docs, reproduced here verbatim):
//!
//! > On held-out data, joint (R, A) search must reduce the relative error
//! > versus the sequential baseline by at least 20% relative
//! > (`joint_error <= 0.8 * baseline_error`) on at least 2 of the 3 tested
//! > workload families. If this is not met, "distribution-aware joint (R,A)
//! > search beats sequential per-axis selection" is falsified for this task.
//!
//! Also demonstrates the distribution-aware plan cache: populates it once
//! per family, shows a fresh same-family sample hits the cache with a score
//! close to a fresh search, and shows that reusing a plan across a
//! *different* distribution (as a `(kernel, hardware)`-only cache would be
//! forced to) costs measurably more than that family's own properly-searched
//! plan.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scirust_core::autotune_accumulate::default_accumulators;
use scirust_core::representation_graph::{
    Plan, PlanCache, default_representation_dictionary, joint_search, pipeline_relative_error,
    sequential_baseline, summarize,
};
use scirust_simd::dispatch::detect_backend;

/// Family 1: benign, narrow-range positive data. Expected negative control —
/// representation/accumulation choice should barely matter.
fn benign(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(0.5..1.5)).collect()
}

/// Family 2: wide-range positive data, log-uniform over 12 decades (matches
/// the shape of CANR's own X2/Y5 wide-range workloads).
fn wide_range(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| 10f64.powf(rng.gen_range(-6.0..6.0)))
        .collect()
}

/// Family 3: stagnation-prone mixed-scale data (~80% tiny, ~20% huge, all
/// positive) — the exact shape `autotune_accumulate.rs`'s own `wide_batch`
/// fixture uses to demonstrate naive-f32 stagnation.
fn stagnation_prone(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let u: f64 = rng.gen_range(0.0..1.0);
            if u < 0.8
            {
                1e-3 * (1.0 + u)
            }
            else
            {
                1e3 * (1.0 + u)
            }
        })
        .collect()
}

struct FamilyResult {
    name: &'static str,
    baseline_error: f64,
    joint_error: f64,
    relative_reduction: f64,
    survives_20pct: bool,
    baseline_plan: String,
    joint_plan: String,
}

fn plan_label(plan: Plan) -> String {
    format!("{}+{:?}", plan.representation.name(), plan.accumulation)
}

fn main() {
    let r_dict = default_representation_dictionary();
    let a_dict = default_accumulators();
    let hw = detect_backend();

    println!("ANEE Phase C prototype: joint (R,A) search vs. sequential baseline");
    println!("Hardware backend: {}", hw.label());
    println!(
        "R dictionary ({}): {:?}",
        r_dict.len(),
        r_dict.iter().map(|r| r.name()).collect::<Vec<_>>()
    );
    println!("A dictionary ({}): {:?}", a_dict.len(), a_dict);
    println!(
        "Joint search space: {} combinations (vs. {} for sequential per-axis calls)",
        r_dict.len() * a_dict.len(),
        r_dict.len() + a_dict.len()
    );
    println!();

    type WorkloadGenerator = fn(u64, usize) -> Vec<f64>;
    let families: [(&str, WorkloadGenerator); 3] = [
        ("benign", benign),
        ("wide-range", wide_range),
        ("stagnation-prone", stagnation_prone),
    ];

    let mut results = Vec::new();
    let mut cache = PlanCache::new();

    for (name, make) in families
    {
        let dev = make(1, 8192);
        let eval = make(2, 8192);

        // Selection: one dev/eval split, as CANR's own S1-S4 convention uses.
        let baseline =
            sequential_baseline(&dev, &eval, &r_dict, &a_dict).expect("baseline must find a plan");
        let joint = joint_search(&dev, &eval, &r_dict, &a_dict).expect("joint must find a plan");

        // Robustness: re-score BOTH chosen (fixed) plans on 3 FRESH held-out
        // seeds never used for selection -- matching
        // certified_numerics.rs's own held-out convention ("selection on one
        // seed, validation on fresh seeds") exactly, rather than resting the
        // kill-criterion verdict on a single eval draw.
        let fresh_seeds = [13u64, 14, 15];
        let baseline_fresh_errors: Vec<f64> = fresh_seeds
            .iter()
            .map(|&s| {
                pipeline_relative_error(baseline.plan, &dev, &make(s, 8192)).unwrap_or(f64::NAN)
            })
            .collect();
        let joint_fresh_errors: Vec<f64> = fresh_seeds
            .iter()
            .map(|&s| pipeline_relative_error(joint.plan, &dev, &make(s, 8192)).unwrap_or(f64::NAN))
            .collect();
        let baseline_mean = baseline_fresh_errors.iter().sum::<f64>() / fresh_seeds.len() as f64;
        let joint_mean = joint_fresh_errors.iter().sum::<f64>() / fresh_seeds.len() as f64;

        let relative_reduction = 1.0 - joint_mean / baseline_mean;
        let survives_20pct = joint_mean <= 0.8 * baseline_mean;

        println!("=== family: {name} ===");
        println!(
            "  sequential: plan={} eval_rel_err={:.6e} 3-seed_fresh={:?} mean={:.6e} cert={:.2} ulps",
            plan_label(baseline.plan),
            baseline.held_out_relative_error,
            baseline_fresh_errors,
            baseline_mean,
            baseline.certificate.ulps
        );
        println!(
            "  joint:      plan={} eval_rel_err={:.6e} 3-seed_fresh={:?} mean={:.6e} cert={:.2} ulps",
            plan_label(joint.plan),
            joint.held_out_relative_error,
            joint_fresh_errors,
            joint_mean,
            joint.certificate.ulps
        );
        println!(
            "  relative reduction (3-seed mean): {:.1}%  (kill criterion >= 20%: {})",
            relative_reduction * 100.0,
            if survives_20pct { "MET" } else { "NOT MET" }
        );

        // Populate the distribution-aware cache with the joint result (the
        // 3-seed mean, not the single eval draw, as the recorded confidence).
        let dist = summarize(&dev);
        cache.insert(
            "sensor_aggregate_sum",
            dist,
            hw,
            joint.plan,
            joint_mean,
            joint.certificate,
        );

        // Cache-hit demo: a FRESH sample from the same family should hit the
        // cache (same distribution bucket) and the cached plan should score
        // close to what a fresh joint search on this new data would find.
        let fresh = make(3, 8192);
        let fresh_dist = summarize(&fresh);
        let cached = cache.get("sensor_aggregate_sum", fresh_dist, hw);
        match cached
        {
            Some(entry) =>
            {
                let cached_score_on_fresh =
                    pipeline_relative_error(entry.plan, &dev, &fresh).unwrap_or(f64::NAN);
                println!(
                    "  cache: HIT for fresh same-family sample -- cached plan scores \
                     {cached_score_on_fresh:.6e} on it (vs. {:.6e} originally)",
                    joint.held_out_relative_error
                );
            },
            None => println!("  cache: MISS (unexpected -- same family should hit)"),
        }

        results.push(FamilyResult {
            name,
            baseline_error: baseline_mean,
            joint_error: joint_mean,
            relative_reduction,
            survives_20pct,
            baseline_plan: plan_label(baseline.plan),
            joint_plan: plan_label(joint.plan),
        });
        println!();
    }

    // Distribution-mismatch demo: reuse the plan cached for "benign" on the
    // "stagnation-prone" family's fresh sample, as a (kernel, hardware)-only
    // cache would be forced to (it cannot see that the distribution
    // changed). Compare against that family's own properly-searched plan.
    println!("=== distribution-mismatch demo (why the cache key needs `distribution`) ===");
    let benign_dev = benign(1, 8192);
    let benign_dist = summarize(&benign_dev);
    let benign_cached_plan = cache
        .get("sensor_aggregate_sum", benign_dist, hw)
        .expect("benign was cached above")
        .plan;
    let stagnation_fresh = stagnation_prone(3, 8192);
    let wrong_bucket_score =
        pipeline_relative_error(benign_cached_plan, &benign_dev, &stagnation_fresh)
            .unwrap_or(f64::NAN);
    let stagnation_result = results
        .iter()
        .find(|r| r.name == "stagnation-prone")
        .unwrap();
    println!(
        "  plan cached for 'benign' ({}), misapplied to 'stagnation-prone' data: rel_err={wrong_bucket_score:.6e}",
        plan_label(benign_cached_plan)
    );
    println!(
        "  'stagnation-prone's own properly-searched joint plan ({}): rel_err={:.6e}",
        stagnation_result.joint_plan, stagnation_result.joint_error
    );
    println!(
        "  degradation from ignoring distribution in the cache key: {:.1}x worse",
        wrong_bucket_score / stagnation_result.joint_error
    );
    println!();

    // Kill-criterion verdict.
    println!("=== pre-registered kill criterion ===");
    println!(
        "  \"joint (R,A) search must reduce held-out relative error by >=20% vs. sequential\n\
         \\   on at least 2 of 3 workload families\""
    );
    let survivors: Vec<&str> = results
        .iter()
        .filter(|r| r.survives_20pct)
        .map(|r| r.name)
        .collect();
    for r in &results
    {
        println!(
            "  {:>16}: baseline={:.4e} joint={:.4e} reduction={:.1}% -> {}",
            r.name,
            r.baseline_error,
            r.joint_error,
            r.relative_reduction * 100.0,
            if r.survives_20pct { "MET" } else { "not met" }
        );
        println!(
            "  {:>16}  sequential plan: {}   joint plan: {}",
            "", r.baseline_plan, r.joint_plan
        );
    }
    println!(
        "  families meeting the criterion: {}/{} ({:?})",
        survivors.len(),
        results.len(),
        survivors
    );
    if survivors.len() >= 2
    {
        println!("  VERDICT: kill criterion MET -- the candidate survives this benchmark.");
    }
    else
    {
        println!(
            "  VERDICT: kill criterion NOT MET -- \"distribution-aware joint (R,A) search beats \
             sequential per-axis selection\" is FALSIFIED for this task."
        );
    }
}
