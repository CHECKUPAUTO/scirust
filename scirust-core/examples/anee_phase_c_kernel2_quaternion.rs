//! **ANEE Phase C, kernel 2 benchmark: quaternion orientation averaging.**
//!
//! (`docs/research/ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md`
//! §13/§14; implementation in `scirust_core::representation_graph_quaternion`.
//! Requires the `portable-simd` feature: `cargo run -p scirust-core
//! --features portable-simd --example anee_phase_c_kernel2_quaternion
//! --release`.)
//!
//! Replicates kernel 1's finding (joint `(R,A)` search beats sequential
//! per-axis selection) on a structurally different kernel — hypercomplex
//! orientation averaging, mirroring [ATRA]'s own X5 experiment — against the
//! **same** pre-registered kill criterion used for kernel 1 (>=20% relative
//! held-out error reduction on >=2 of 3 conditions), reproduced verbatim
//! from `representation_graph_quaternion`'s module docs.

use rand::SeedableRng;
use rand::rngs::StdRng;
use scirust_core::autotune_accumulate::default_accumulators;
use scirust_core::representation_graph_quaternion::{
    Chart, QuatPlan, default_chart_dictionary, joint_search, score_plan, sequential_baseline,
};
use scirust_simd::geometry::quaternion::Quaternion;

type Quat = Quaternion<f64>;

const N_TRIALS: usize = 20; // matches ATRA X5's "20 trials"
const N_PER_TRIAL: usize = 100; // matches ATRA X5's "100 observations/trial"

fn trials(truth: Quat, sigma: f64, seed: u64) -> Vec<Vec<Quat>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..N_TRIALS)
        .map(|_| {
            scirust_core::representation_graph_quaternion::noisy_trial(
                truth,
                sigma,
                N_PER_TRIAL,
                &mut rng,
            )
        })
        .collect()
}

fn plan_label(plan: QuatPlan) -> String {
    format!("{}+{:?}", plan.chart.name(), plan.accumulation)
}

struct NoiseResult {
    sigma: f64,
    baseline_error: f64,
    joint_error: f64,
    relative_reduction: f64,
    survives_20pct: bool,
    baseline_plan: String,
    joint_plan: String,
}

fn main() {
    let truth = Quaternion::from_axis_angle([1.0, 1.0, 1.0], 0.7).normalize();
    let chart_dict = default_chart_dictionary();
    let a_dict = default_accumulators();

    println!("ANEE Phase C, kernel 2: quaternion orientation averaging (replication)");
    println!(
        "Chart dictionary ({}): {:?}",
        chart_dict.len(),
        chart_dict.iter().map(|c| c.name()).collect::<Vec<_>>()
    );
    println!("Accumulation dictionary ({}): {:?}", a_dict.len(), a_dict);
    println!(
        "Joint search space: {} combinations (vs. {} for sequential per-axis calls)",
        chart_dict.len() * a_dict.len(),
        chart_dict.len() + a_dict.len()
    );
    println!("N_TRIALS={N_TRIALS} N_PER_TRIAL={N_PER_TRIAL} (matches [ATRA X5]'s own protocol)");
    println!();
    println!("Reference context -- [ATRA X5]'s own published numbers (angular error, degrees),");
    println!("NOT reproduced here (different RNG/implementation), sanity-check ballpark only:");
    println!("  sigma=0.2: componentwise=1.071  chordal=1.062  karcher=1.074");
    println!("  sigma=0.8: componentwise=4.651  chordal=3.815  karcher=4.455");
    println!("  sigma=1.5: componentwise=23.11  chordal=6.706  karcher=9.959");
    println!();

    let sigmas = [0.2, 0.8, 1.5];
    let mut results = Vec::new();

    for (i, &sigma) in sigmas.iter().enumerate()
    {
        let dev = trials(truth, sigma, 100 + i as u64 * 10 + 1);
        let eval = trials(truth, sigma, 100 + i as u64 * 10 + 2);

        let baseline =
            sequential_baseline(truth, &dev, &eval, &a_dict).expect("baseline must find a plan");
        let joint =
            joint_search(truth, &dev, &eval, &chart_dict, &a_dict).expect("joint must find a plan");

        // Robustness: 3 fresh held-out seeds beyond the dev/eval draw, same
        // convention as kernel 1.
        let fresh_seeds = [
            200u64 + i as u64 * 10,
            201 + i as u64 * 10,
            202 + i as u64 * 10,
        ];
        let baseline_fresh: Vec<f64> = fresh_seeds
            .iter()
            .map(|&s| -score_plan(baseline.plan, truth, &dev, &trials(truth, sigma, s)).unwrap())
            .collect();
        let joint_fresh: Vec<f64> = fresh_seeds
            .iter()
            .map(|&s| -score_plan(joint.plan, truth, &dev, &trials(truth, sigma, s)).unwrap())
            .collect();
        let baseline_mean = baseline_fresh.iter().sum::<f64>() / fresh_seeds.len() as f64;
        let joint_mean = joint_fresh.iter().sum::<f64>() / fresh_seeds.len() as f64;
        let relative_reduction = 1.0 - joint_mean / baseline_mean;
        let survives_20pct = joint_mean <= 0.8 * baseline_mean;

        println!("=== sigma = {sigma} rad ===");
        println!(
            "  sequential: plan={} eval_err={:.4} deg  3-seed_fresh={:?} mean={:.4} deg",
            plan_label(baseline.plan),
            baseline.eval_mean_error_degrees,
            baseline_fresh,
            baseline_mean
        );
        println!(
            "  joint:      plan={} eval_err={:.4} deg  3-seed_fresh={:?} mean={:.4} deg",
            plan_label(joint.plan),
            joint.eval_mean_error_degrees,
            joint_fresh,
            joint_mean
        );
        println!(
            "  relative reduction (3-seed mean): {:.1}%  (kill criterion >= 20%: {})",
            relative_reduction * 100.0,
            if survives_20pct { "MET" } else { "NOT MET" }
        );
        println!();

        results.push(NoiseResult {
            sigma,
            baseline_error: baseline_mean,
            joint_error: joint_mean,
            relative_reduction,
            survives_20pct,
            baseline_plan: plan_label(baseline.plan),
            joint_plan: plan_label(joint.plan),
        });
    }

    println!("=== pre-registered kill criterion (same bar as kernel 1) ===");
    println!(
        "  \"joint (Chart,A) search must reduce held-out mean angular error by >=20% vs.\n\
         sequential on at least 2 of 3 noise levels\""
    );
    let survivors: Vec<f64> = results
        .iter()
        .filter(|r| r.survives_20pct)
        .map(|r| r.sigma)
        .collect();
    for r in &results
    {
        println!(
            "  sigma={:>4}: baseline={:.4} joint={:.4} reduction={:.1}% -> {}",
            r.sigma,
            r.baseline_error,
            r.joint_error,
            r.relative_reduction * 100.0,
            if r.survives_20pct { "MET" } else { "not met" }
        );
        println!(
            "          sequential plan: {}   joint plan: {}",
            r.baseline_plan, r.joint_plan
        );
    }
    println!(
        "  noise levels meeting the criterion: {}/{} ({:?})",
        survivors.len(),
        results.len(),
        survivors
    );
    if survivors.len() >= 2
    {
        println!("  VERDICT: kill criterion MET -- the replication survives on kernel 2.");
    }
    else
    {
        println!(
            "  VERDICT: kill criterion NOT MET -- the joint-search finding does NOT replicate \
             on this kernel as tested."
        );
    }

    // Chart-only sanity check: does LogChart alone (any A) ever noticeably
    // beat Componentwise alone at high noise, matching [ATRA X5]'s own
    // qualitative finding that noise level changes which chart wins?
    println!();
    println!("=== chart-only comparison (context for the joint-search result above) ===");
    for &sigma in &sigmas
    {
        let eval = trials(truth, sigma, 900 + (sigma * 10.0) as u64);
        let dev = trials(truth, sigma, 800 + (sigma * 10.0) as u64);
        let comp_err = -score_plan(
            QuatPlan {
                chart: Chart::Componentwise,
                accumulation: scirust_core::autotune_accumulate::AccumMethod::NeumaierF32,
            },
            truth,
            &dev,
            &eval,
        )
        .unwrap();
        let log_err = -score_plan(
            QuatPlan {
                chart: Chart::LogChart { iterations: 2 },
                accumulation: scirust_core::autotune_accumulate::AccumMethod::NeumaierF32,
            },
            truth,
            &dev,
            &eval,
        )
        .unwrap();
        println!(
            "  sigma={sigma:>4}: componentwise={comp_err:.4} deg  log-chart={log_err:.4} deg  \
             (both with NeumaierF32 accumulation, chart effect in isolation)"
        );
    }
}
