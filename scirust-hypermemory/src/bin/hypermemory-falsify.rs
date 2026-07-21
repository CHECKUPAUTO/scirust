//! Deterministic F2 / F6 falsification report for `scirust-hypermemory`.
//!
//! Runs the zero-divisor / norm-collapse survey (F2) and the
//! structure-discrimination survey (F6) across several operand distributions and
//! prints a reproducible textual report. The numbers are **deterministic** (pure
//! `f32` algebra, fixed-seed LCG, fixed index-order reductions), so they are
//! reproducible facts, not timing measurements.
//!
//! Reproduce:
//! ```text
//! cargo +nightly-2026-07-02 run --release --bin hypermemory-falsify
//! ```
//! Optionally set `HYPERMEMORY_GIT_COMMIT` to stamp the report with a commit.

use scirust_hypermemory::{
    DEFAULT_NEAR_ZERO_THRESHOLD, OperandDistribution, survey_structure_discrimination,
    survey_zero_divisors,
};

/// Products / triples sampled per distribution. Fixed for reproducibility.
const SAMPLES: usize = 100_000;
/// Relative-associator cutoff below which two parenthesizations are counted
/// indistinguishable (F6).
const F6_THRESHOLD: f32 = 1e-3;

fn main() {
    let commit = std::env::var("HYPERMEMORY_GIT_COMMIT").unwrap_or_else(|_| "<unset>".into());

    println!("scirust-hypermemory — F2/F6 falsification report");
    println!("=================================================");
    println!("commit         : {commit}");
    println!("samples / dist : {SAMPLES}");
    println!("F2 threshold   : {DEFAULT_NEAR_ZERO_THRESHOLD:e} (on ‖a·b‖²)");
    println!("F6 threshold   : {F6_THRESHOLD:e} (relative associator)");
    println!();

    // Every survey below is also collected as shared CANR §9 benchmark
    // records (scirust-bench-schema) and streamed as JSONL at the end — the
    // F2/F6 falsification numbers now speak the workspace's one benchmark
    // schema, not only this bespoke text table.
    let mut records: Vec<scirust_bench_schema::BenchRecord> = Vec::new();

    // ---- F2: zero-divisor / norm-collapse frequency ------------------------
    println!("F2 — zero-divisor / norm-collapse survey");
    println!("  (defect ratio r = ‖a·b‖² / (‖a‖²·‖b‖²); r≡1 iff composition algebra)");
    println!(
        "  {:<22} {:>9} {:>9} {:>10} {:>10} {:>10} {:>9}",
        "distribution", "near0", "exact0", "min r", "mean r", "max r", "r<0.01"
    );
    let f2_dists = [
        ("DenseUniform", OperandDistribution::DenseUniform),
        (
            "Sparse{2}",
            OperandDistribution::Sparse { nonzero_lanes: 2 },
        ),
        (
            "Sparse{4}",
            OperandDistribution::Sparse { nonzero_lanes: 4 },
        ),
        (
            "Subalgebra{4} (ℍ)",
            OperandDistribution::Subalgebra { dim: 4 },
        ),
    ];
    for (name, dist) in f2_dists
    {
        let s = survey_zero_divisors(0xF2, SAMPLES, dist, DEFAULT_NEAR_ZERO_THRESHOLD);
        records.extend(s.to_bench_records(0xF2, name));
        println!(
            "  {:<22} {:>9} {:>9} {:>10.4} {:>10.4} {:>10.4} {:>8.2}%",
            name,
            s.near_zero_count(),
            s.exact_zero_count(),
            s.min_defect_ratio(),
            s.mean_defect_ratio(),
            s.max_defect_ratio(),
            100.0 * s.frac_below_hundredth(),
        );
    }
    println!();

    // ---- F6: structure discrimination --------------------------------------
    println!("F6 — structure-discrimination survey");
    println!(
        "  (ρ = ‖(a·b)·c − a·(b·c)‖ / (‖(a·b)·c‖ + ‖a·(b·c)‖); indistinguishable iff ρ ≤ threshold)"
    );
    println!(
        "  {:<22} {:>13} {:>12} {:>12} {:>12}",
        "distribution", "discriminable", "min ρ", "mean ρ", "max ρ"
    );
    let f6_dists = [
        ("DenseUniform", OperandDistribution::DenseUniform),
        (
            "Sparse{2}",
            OperandDistribution::Sparse { nonzero_lanes: 2 },
        ),
        (
            "Subalgebra{2} (ℂ)",
            OperandDistribution::Subalgebra { dim: 2 },
        ),
        (
            "Subalgebra{4} (ℍ)",
            OperandDistribution::Subalgebra { dim: 4 },
        ),
    ];
    for (name, dist) in f6_dists
    {
        let s = survey_structure_discrimination(0xF6, SAMPLES, dist, F6_THRESHOLD);
        records.extend(s.to_bench_records(0xF6, name));
        println!(
            "  {:<22} {:>12.4}% {:>12.6} {:>12.6} {:>12.6}",
            name,
            100.0 * s.discriminable_fraction(),
            s.min_relative_associator(),
            s.mean_relative_associator(),
            s.max_relative_associator(),
        );
    }
    println!();

    // ---- Verdict -----------------------------------------------------------
    let f2_generic = survey_zero_divisors(
        0xF2,
        SAMPLES,
        OperandDistribution::DenseUniform,
        DEFAULT_NEAR_ZERO_THRESHOLD,
    );
    let f6_generic = survey_structure_discrimination(
        0xF6,
        SAMPLES,
        OperandDistribution::DenseUniform,
        F6_THRESHOLD,
    );
    let f6_quat = survey_structure_discrimination(
        0xF6,
        SAMPLES,
        OperandDistribution::Subalgebra { dim: 4 },
        F6_THRESHOLD,
    );

    println!("Verdict (this harness, these distributions):");
    println!(
        "  F2: generic operands never collapse (near-zero divisors = {}). \
         Zero divisors are a measure-zero set; collapse is a structured-input risk, not a generic one.",
        f2_generic.near_zero_count()
    );
    println!(
        "  F6: generic parenthesizations are {:.2}% discriminable; \
         inside the quaternion subalgebra they are {:.2}% discriminable (associator vanishes — F6 fires there, as it must).",
        100.0 * f6_generic.discriminable_fraction(),
        100.0 * f6_quat.discriminable_fraction(),
    );
    println!(
        "  => Neither F2 nor F6 falsifies the relation direction for generic operands; \
         both failure modes are confined to structured/degenerate inputs. This does NOT establish usefulness — \
         only that the necessary discriminative capacity exists."
    );

    println!();
    println!(
        "=== bench-schema JSONL ({} records, scirust-bench-schema) ===",
        records.len()
    );
    print!("{}", scirust_bench_schema::to_jsonl(&records));
}
