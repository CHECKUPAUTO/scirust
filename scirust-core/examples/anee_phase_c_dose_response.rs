//! **ANEE Addendum 3, avenue 1: dose-response test of the boundary
//! heuristic.**
//!
//! (`docs/research/ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md`,
//! Addendum 2's closing claim.) Addendum 2 distilled kernels 1+2 into a
//! decision rule: *"joint (R,A) search pays off exactly where the cheap
//! default representation is objective-blind and empirically wrong; run a
//! cheap single-axis ablation first, and only invest in joint search where
//! the default is not already near-optimal."* That heuristic has never been
//! tested **prospectively**. This benchmark does so by turning kernel 1's
//! fixed quantizer level count (64) into an environmental dose knob: at high
//! level counts quantization noise collapses and the default (`identity`)
//! should become adequate; at low counts companding representations should
//! matter more.
//!
//! ## Pre-registered protocol and criteria (written before any run)
//!
//! Grid: kernel 1's three workload families (benign / wide-range /
//! stagnation-prone, generators identical to `anee_phase_c_prototype.rs`)
//! × levels `L ∈ {8, 16, 64, 256, 1024}` = **15 cells**. Per cell:
//!
//! 1. **Predictor first (dev data only, cheap):** the R-axis-only ablation
//!    gap `g = 1 − min_R err_dev(R, NeumaierF32) / err_dev(identity,
//!    NeumaierF32)` over the 5-member single-hop dictionary, with
//!    `err_dev = pipeline(fit=dev, score=dev)`. Predict "joint pays" iff
//!    `g ≥ 0.20`. The predictor sees **no held-out data** — structurally a
//!    genuine prospective prediction.
//! 2. **Outcome (kernel 1's exact protocol):** sequential vs. joint
//!    selection on (dev, eval), both chosen plans re-scored on 3 fresh
//!    held-out seeds; actual "joint pays" iff
//!    `joint_mean ≤ 0.8 × sequential_mean`.
//!
//! **P1 (decisive):** the predictor's classification must match the outcome
//! in **≥ 12 of 15 cells** (80%). If not, Addendum 2's heuristic is
//! **falsified as a usable decision rule** and must be reported as such.
//!
//! Secondary, descriptive (not decisive): **P2** stagnation-prone's
//! reduction decreases with L (at most one adjacent-pair inversion across
//! the 5 points); **P3** wide-range's reduction stays ≥ 20% at every L
//! (uniform-vs-companded quantization gaps are roughly level-independent in
//! dB, so the default stays wrong across this whole range); **P4** benign's
//! reduction stays < 20% for L ≥ 64. Declared hazard, known in advance:
//! cells where both approaches sit near the f32 accumulation noise floor
//! make the ratio unstable (kernel 1's benign family already showed this);
//! failures concentrated on such cells will be reported as a caveat, not
//! excused away.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scirust_core::autotune_accumulate::{AccumMethod, default_accumulators};
use scirust_core::representation_graph::{
    Plan, RepresentationChoice, default_representation_dictionary, joint_search_with_levels,
    pipeline_relative_error_with_levels, sequential_baseline_with_levels,
};

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

/// The cheap prospective predictor: best single-representation improvement
/// over identity on DEV data only, accumulation held fixed at Neumaier.
fn ablation_gap(dev: &[f64], r_dict: &[RepresentationChoice], levels: usize) -> f64 {
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
    let base = match err(RepresentationChoice::Identity)
    {
        Some(e) if e > 0.0 => e,
        _ => return 0.0,
    };
    let best = r_dict.iter().filter_map(|&r| err(r)).fold(base, f64::min);
    1.0 - best / base
}

struct Cell {
    family: &'static str,
    levels: usize,
    gap: f64,
    predicted_pays: bool,
    reduction: f64,
    actual_pays: bool,
    seq_plan: String,
    joint_plan: String,
}

fn plan_label(plan: Plan) -> String {
    format!("{}+{:?}", plan.representation.name(), plan.accumulation)
}

fn main() {
    type WorkloadGenerator = fn(u64, usize) -> Vec<f64>;
    let families: [(&str, WorkloadGenerator); 3] = [
        ("benign", benign),
        ("wide-range", wide_range),
        ("stagnation-prone", stagnation_prone),
    ];
    let level_grid = [8usize, 16, 64, 256, 1024];
    let r_dict = default_representation_dictionary();
    let a_dict = default_accumulators();

    println!("ANEE Addendum 3, avenue 1: dose-response test of the Addendum-2 heuristic");
    println!(
        "Grid: {} families x {} level counts = {} cells; predictor threshold 20%; outcome bar 20% on 3-fresh-seed means",
        families.len(),
        level_grid.len(),
        families.len() * level_grid.len()
    );
    println!("P1 (decisive): predictor/outcome agreement >= 12/15 cells\n");

    let mut cells: Vec<Cell> = Vec::new();
    let mut records: Vec<scirust_bench_schema::BenchRecord> = Vec::new();

    for (family, make) in families
    {
        for &levels in &level_grid
        {
            let dev = make(1, 8192);
            let eval = make(2, 8192);

            // 1. Predictor FIRST — dev only, R axis only, fixed Neumaier.
            let gap = ablation_gap(&dev, &r_dict, levels);
            let predicted_pays = gap >= 0.20;

            // 2. Outcome — kernel 1's exact protocol at this level count.
            let seq = sequential_baseline_with_levels(&dev, &eval, &r_dict, &a_dict, levels)
                .expect("sequential must find a plan");
            let joint = joint_search_with_levels(&dev, &eval, &r_dict, &a_dict, levels)
                .expect("joint must find a plan");
            let fresh_seeds = [13u64, 14, 15];
            // Per-seed held-out errors are both the mean's inputs and the
            // shared BenchRecords emitted at the end (one record per seed —
            // the schema's mandatory `seed` field is the actual seed that
            // generated that measurement, never a stand-in).
            let per_seed_of = |plan: Plan| -> Vec<(u64, f64)> {
                fresh_seeds
                    .iter()
                    .map(|&s| {
                        (
                            s,
                            pipeline_relative_error_with_levels(plan, &dev, &make(s, 8192), levels)
                                .unwrap_or(f64::NAN),
                        )
                    })
                    .collect()
            };
            let seq_per_seed = per_seed_of(seq.plan);
            let joint_per_seed = per_seed_of(joint.plan);
            let mean =
                |xs: &[(u64, f64)]| xs.iter().map(|&(_, e)| e).sum::<f64>() / xs.len() as f64;
            let seq_mean = mean(&seq_per_seed);
            let joint_mean = mean(&joint_per_seed);
            for (report, per_seed, approach) in [
                (&seq, &seq_per_seed, "sequential"),
                (&joint, &joint_per_seed, "joint"),
            ]
            {
                for &(s, e) in per_seed.iter()
                {
                    records.push(
                        scirust_bench_schema::BenchRecord::new(
                            "anee_phase_c_pipeline",
                            format!("{family}/L={levels}"),
                            format!("{approach}:{}", plan_label(report.plan)),
                            s,
                            "held_out_relative_error",
                            e,
                        )
                        .with_cert(scirust_bench_schema::Certificate {
                            description: "kappa_rt round-trip bound (CANR §3.2)".into(),
                            bound_ulps: Some(report.certificate.ulps),
                            determinism: None,
                        }),
                    );
                }
            }
            let reduction = 1.0 - joint_mean / seq_mean;
            let actual_pays = joint_mean <= 0.8 * seq_mean;

            println!(
                "[{family:>16} L={levels:>4}] ablation gap={:>6.1}% -> predict {}   |   seq={:.4e} ({}) joint={:.4e} ({}) reduction={:>6.1}% -> actual {}   [{}]",
                gap * 100.0,
                if predicted_pays { "PAYS" } else { "no  " },
                seq_mean,
                plan_label(seq.plan),
                joint_mean,
                plan_label(joint.plan),
                reduction * 100.0,
                if actual_pays { "PAYS" } else { "no  " },
                if predicted_pays == actual_pays
                {
                    "agree"
                }
                else
                {
                    "DISAGREE"
                },
            );

            cells.push(Cell {
                family,
                levels,
                gap,
                predicted_pays,
                reduction,
                actual_pays,
                seq_plan: plan_label(seq.plan),
                joint_plan: plan_label(joint.plan),
            });
        }
        println!();
    }

    // P1: predictor accuracy.
    let agreements = cells
        .iter()
        .filter(|c| c.predicted_pays == c.actual_pays)
        .count();
    println!("=== P1 (decisive): predictor/outcome agreement ===");
    println!(
        "  agreement: {}/{} cells (bar: >= 12/15)",
        agreements,
        cells.len()
    );
    for c in cells.iter().filter(|c| c.predicted_pays != c.actual_pays)
    {
        println!(
            "  DISAGREE at [{} L={}]: gap={:.1}% predicted {}, but reduction={:.1}% (seq {} vs joint {})",
            c.family,
            c.levels,
            c.gap * 100.0,
            if c.predicted_pays { "PAYS" } else { "no" },
            c.reduction * 100.0,
            c.seq_plan,
            c.joint_plan,
        );
    }
    if agreements >= 12
    {
        println!("  VERDICT: P1 MET — the Addendum-2 heuristic survives as a decision rule.");
    }
    else
    {
        println!(
            "  VERDICT: P1 NOT MET — the Addendum-2 heuristic is FALSIFIED as a decision rule."
        );
    }

    // P2–P4: descriptive secondary predictions.
    println!("\n=== secondary (descriptive) predictions ===");
    let series = |fam: &str| -> Vec<&Cell> { cells.iter().filter(|c| c.family == fam).collect() };
    let stag = series("stagnation-prone");
    let inversions = stag
        .windows(2)
        .filter(|w| w[1].reduction > w[0].reduction + 1e-12)
        .count();
    println!(
        "  P2 stagnation-prone reduction decreasing in L (<=1 adjacent inversion): inversions={inversions} -> {}",
        if inversions <= 1 { "held" } else { "FAILED" }
    );
    let wide_ok = series("wide-range").iter().all(|c| c.reduction >= 0.20);
    println!(
        "  P3 wide-range reduction >= 20% at every L: {}",
        if wide_ok { "held" } else { "FAILED" }
    );
    let benign_ok = series("benign")
        .iter()
        .filter(|c| c.levels >= 64)
        .all(|c| c.reduction < 0.20);
    println!(
        "  P4 benign reduction < 20% for L >= 64: {}",
        if benign_ok { "held" } else { "FAILED" }
    );

    // Machine-readable emission: the same measurements as shared CANR §9
    // records (scirust-bench-schema), one line per (cell, approach, fresh
    // seed) — pipe through `grep '^{'` to extract just the JSONL stream.
    println!(
        "\n=== bench-schema JSONL ({} records, scirust-bench-schema) ===",
        records.len()
    );
    print!("{}", scirust_bench_schema::to_jsonl(&records));
}
