//! **ANEE Phase D — D2 (determinism as a selection constraint) + D6 (plan
//! transfer across execution regimes).**
//!
//! Protocols, decisive criteria, and author priors are pre-registered in
//! `docs/research/ANEE_PHASE_D_PREREGISTRATION_2026-07-18.md` §2 (committed
//! before this file existed).
//!
//! * **D2** first *measures* which accumulation methods are chunk-invariant
//!   (bit-identical under P-way chunked reduction, P ∈ {2, 4, 16}) — the
//!   constraint set is what the measurement says, not what the author
//!   assumed — then compares constrained vs unconstrained joint search over
//!   the 15-cell grid. Bar: ρ ≤ 2.0 in ≥ 12/15 cells; author prior: MET.
//! * **D6** varies the execution model (chunked reduction, P ∈ {4, 16, 64})
//!   and asks whether the serially-tuned plan transfers (regret ≤ 1.2× vs
//!   the P-specific winner in every P). Bar: ≥ 12/15 cells transfer; author
//!   prior: MET, failures predicted in stagnation-prone cells where chunking
//!   breaks sequential compensation. **Scope, honestly:** one CPU, no second
//!   hardware — this probes ANEE's M axis (execution model), not H.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scirust_core::autotune_accumulate::{AccumMethod, accumulate, default_accumulators};
use scirust_core::certified_numerics::sum_expansion;
use scirust_core::representation_graph::{
    Plan, RepresentationChoice, default_representation_dictionary, joint_search_with_levels,
    pipeline_relative_error_with_levels, reconstruct_with_levels,
};
use scirust_core::transform_autotune::autotune_by;

const N: usize = 8192;
const FRESH: [u64; 3] = [13, 14, 15];
const LEVEL_GRID: [usize; 5] = [8, 16, 64, 256, 1024];

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

type WorkloadGenerator = fn(u64, usize) -> Vec<f64>;
const FAMILIES: [(&str, WorkloadGenerator); 3] = [
    ("benign", benign),
    ("wide-range", wide_range),
    ("stagnation-prone", stagnation_prone),
];

/// P-way chunked reduction: accumulate each chunk with `method`, then combine
/// the P partials with the same method — the standard shape of a data-parallel
/// deployment of a serial accumulator.
fn accumulate_chunked(method: AccumMethod, xs: &[f32], parts: usize) -> f32 {
    if parts <= 1 || xs.len() <= parts
    {
        return accumulate(method, xs);
    }
    let chunk = xs.len().div_ceil(parts);
    let partials: Vec<f32> = xs.chunks(chunk).map(|c| accumulate(method, c)).collect();
    accumulate(method, &partials)
}

fn plan_label(plan: Plan) -> String {
    format!("{}+{:?}", plan.representation.name(), plan.accumulation)
}

/// Pipeline error with the accumulation step replaced by the chunked regime —
/// reconstruction (representation + quantization) is the library's own,
/// bit-identical to the serial pipeline by the committed refactor test.
fn chunked_pipeline_error(
    plan: Plan,
    fit: &[f64],
    score_on: &[f64],
    levels: usize,
    parts: usize,
) -> Option<f64> {
    let reconstructed = reconstruct_with_levels(plan.representation, fit, score_on, levels)?;
    let total = accumulate_chunked(plan.accumulation, &reconstructed, parts) as f64;
    let exact = sum_expansion(score_on);
    if exact == 0.0
    {
        return None;
    }
    Some(((total - exact) / exact).abs())
}

// ---------------------------------------------------------------------------
// D2
// ---------------------------------------------------------------------------

/// Measure the chunk-invariant subset of the accumulator dictionary:
/// bit-identical serial vs chunked results for every P in `parts` on every
/// family's raw (f32-narrowed) batch.
fn measured_invariant_set(parts: &[usize]) -> Vec<(AccumMethod, bool)> {
    default_accumulators()
        .into_iter()
        .map(|m| {
            let invariant = FAMILIES.iter().all(|&(_, make)| {
                let xs: Vec<f32> = make(1, N).iter().map(|&x| x as f32).collect();
                let serial = accumulate(m, &xs);
                parts
                    .iter()
                    .all(|&p| accumulate_chunked(m, &xs, p).to_bits() == serial.to_bits())
            });
            (m, invariant)
        })
        .collect()
}

fn run_d2(records: &mut Vec<scirust_bench_schema::BenchRecord>) -> Vec<(String, usize, Plan)> {
    println!("=== D2: determinism (chunk-invariance) as a selection constraint ===");
    println!(
        "bar: constrained/unconstrained error ratio <= 2.0 in >= 12/15 cells; author prior: MET\n"
    );

    let probe_parts = [2usize, 4, 16];
    let matrix = measured_invariant_set(&probe_parts);
    println!("measured chunk-invariance (P in {probe_parts:?}, all families, n = {N}):");
    for (m, inv) in &matrix
    {
        println!(
            "  {m:?}: {}",
            if *inv { "invariant" } else { "NOT invariant" }
        );
    }
    let mut constrained_a: Vec<AccumMethod> =
        matrix.iter().filter(|(_, i)| *i).map(|&(m, _)| m).collect();
    if constrained_a.is_empty()
    {
        // Pre-registered escape: report the discrepancy, weaken to the
        // P = 2-invariant set, and label the deviation loudly.
        println!(
            "\nLABELED DEVIATION: no method is invariant for all P in {probe_parts:?}; falling back to the P = 2-invariant set."
        );
        constrained_a = default_accumulators()
            .into_iter()
            .filter(|&m| {
                FAMILIES.iter().all(|&(_, make)| {
                    let xs: Vec<f32> = make(1, N).iter().map(|&x| x as f32).collect();
                    accumulate_chunked(m, &xs, 2).to_bits() == accumulate(m, &xs).to_bits()
                })
            })
            .collect();
        println!("P = 2-invariant set: {constrained_a:?}");
        assert!(
            !constrained_a.is_empty(),
            "even the weakened constraint set is empty — D2 unsatisfiable as designed"
        );
    }
    else
    {
        println!("\nconstraint set: {constrained_a:?}");
    }

    let r_dict = default_representation_dictionary();
    let a_all = default_accumulators();
    let mut serial_winners: Vec<(String, usize, Plan)> = Vec::new();
    let mut cells_ok = 0usize;
    let mut cells = 0usize;
    let mut r_compensations = 0usize;

    for (family, make) in FAMILIES
    {
        for &levels in &LEVEL_GRID
        {
            let dev = make(1, N);
            let eval = make(2, N);
            let unc = joint_search_with_levels(&dev, &eval, &r_dict, &a_all, levels)
                .expect("unconstrained joint");
            let con = joint_search_with_levels(&dev, &eval, &r_dict, &constrained_a, levels)
                .expect("constrained joint");
            let mean = |plan: Plan| -> f64 {
                FRESH
                    .iter()
                    .map(|&s| {
                        pipeline_relative_error_with_levels(plan, &dev, &make(s, N), levels)
                            .expect("fresh-seed pipeline on positive data")
                    })
                    .sum::<f64>()
                    / FRESH.len() as f64
            };
            let (e_unc, e_con) = (mean(unc.plan), mean(con.plan));
            let rho = e_con / e_unc;
            let ok = rho <= 2.0;
            cells += 1;
            cells_ok += usize::from(ok);
            let r_differs = con.plan.representation != unc.plan.representation;
            r_compensations += usize::from(r_differs);
            println!(
                "[{family:>16} L={levels:>4}] unconstrained {} = {e_unc:.3e} | constrained {} = {e_con:.3e} | rho={rho:>7.2} {}{}",
                plan_label(unc.plan),
                plan_label(con.plan),
                if ok { "ok" } else { "OVER BAR" },
                if r_differs { "  [R compensates]" } else { "" },
            );
            for (arm, plan, e) in [
                ("unconstrained", unc.plan, e_unc),
                ("constrained", con.plan, e_con),
            ]
            {
                records.push(scirust_bench_schema::BenchRecord::new(
                    "anee_phase_d_determinism/D2",
                    format!("{family}/L={levels}"),
                    format!("{arm}:{}", plan_label(plan)),
                    1,
                    "mean_fresh_relative_error",
                    e,
                ));
            }
            serial_winners.push((family.to_string(), levels, unc.plan));
        }
        println!();
    }
    println!(
        "D2 VERDICT: {cells_ok}/{cells} cells within 2.0x (bar >= 12) -> {}; R-axis compensation observed in {r_compensations} cells\n",
        if cells_ok >= 12
        {
            "MET — determinism is cheap here (author prior confirmed)"
        }
        else
        {
            "NOT MET — the determinism constraint is expensive (author prior falsified)"
        }
    );
    serial_winners
}

// ---------------------------------------------------------------------------
// D6
// ---------------------------------------------------------------------------

fn run_d6(
    serial_winners: &[(String, usize, Plan)],
    records: &mut Vec<scirust_bench_schema::BenchRecord>,
) {
    println!(
        "=== D6: plan transfer across execution regimes (chunked reduction; M axis, not H) ==="
    );
    println!(
        "bar: serial plan within 1.2x of the P-specific winner for all P in {{4,16,64}} in >= 12/15 cells; author prior: MET\n"
    );

    let r_dict = default_representation_dictionary();
    let a_all = default_accumulators();
    let parts_grid = [4usize, 16, 64];
    let mut cells_ok = 0usize;

    let family_maker = |name: &str| -> WorkloadGenerator {
        FAMILIES
            .iter()
            .find(|&&(f, _)| f == name)
            .expect("known family")
            .1
    };

    for (family, levels, serial_plan) in serial_winners
    {
        let make = family_maker(family);
        let dev = make(1, N);
        let eval = make(2, N);
        let mut worst_regret = 0.0f64;
        let mut regret_details = Vec::new();
        for &p in &parts_grid
        {
            // P-specific joint winner through the same generic harness.
            let candidates: Vec<Plan> = r_dict
                .iter()
                .flat_map(|&r| {
                    a_all.iter().map(move |&a| Plan {
                        representation: r,
                        accumulation: a,
                    })
                })
                .collect();
            let score = move |plan: Plan, fit: &[f64], scr: &[f64]| {
                chunked_pipeline_error(plan, fit, scr, *levels, p).map(|e| -e)
            };
            let baseline = move |fit: &[f64], scr: &[f64]| -> f64 {
                chunked_pipeline_error(
                    Plan {
                        representation: RepresentationChoice::Identity,
                        accumulation: AccumMethod::NaiveF32,
                    },
                    fit,
                    scr,
                    *levels,
                    p,
                )
                .map(|e| -e)
                .unwrap_or(f64::NEG_INFINITY)
            };
            let out = autotune_by(&dev[..], &eval[..], &candidates, score, baseline);
            let winner_p = out.chosen.expect("P-specific joint winner");

            let mean_chunked = |plan: Plan| -> f64 {
                FRESH
                    .iter()
                    .map(|&s| {
                        chunked_pipeline_error(plan, &dev, &make(s, N), *levels, p)
                            .expect("fresh-seed chunked pipeline")
                    })
                    .sum::<f64>()
                    / FRESH.len() as f64
            };
            let (e_serial, e_winner) = (mean_chunked(*serial_plan), mean_chunked(winner_p));
            let regret = e_serial / e_winner;
            worst_regret = worst_regret.max(regret);
            regret_details.push((p, regret, winner_p));
            for (arm, plan, e) in [
                ("serial_transferred", *serial_plan, e_serial),
                ("p_specific", winner_p, e_winner),
            ]
            {
                records.push(scirust_bench_schema::BenchRecord::new(
                    "anee_phase_d_determinism/D6",
                    format!("{family}/L={levels}/P={p}"),
                    format!("{arm}:{}", plan_label(plan)),
                    1,
                    "mean_fresh_relative_error",
                    e,
                ));
            }
        }
        let transfers = worst_regret <= 1.2;
        cells_ok += usize::from(transfers);
        let detail: Vec<String> = regret_details
            .iter()
            .map(|(p, r, w)| format!("P={p}:{r:.2}x({})", plan_label(*w)))
            .collect();
        println!(
            "[{family:>16} L={levels:>4}] serial {} | {} | worst {worst_regret:.2}x -> {}",
            plan_label(*serial_plan),
            detail.join("  "),
            if transfers
            {
                "transfers"
            }
            else
            {
                "RE-TUNE NEEDED"
            },
        );
    }
    println!(
        "\nD6 VERDICT: {cells_ok}/15 cells transfer (bar >= 12) -> {}",
        if cells_ok >= 12
        {
            "MET — serial plans survive the execution-regime change (author prior confirmed)"
        }
        else
        {
            "NOT MET — plans do not transfer across execution regimes (author prior falsified)"
        }
    );
}

fn main() {
    println!(
        "ANEE Phase D — determinism + execution-regime transfer (pre-registered: docs/research/ANEE_PHASE_D_PREREGISTRATION_2026-07-18.md §2)\n"
    );
    let mut records: Vec<scirust_bench_schema::BenchRecord> = Vec::new();
    let serial_winners = run_d2(&mut records);
    run_d6(&serial_winners, &mut records);
    println!("\n=== bench-schema JSONL ({} records) ===", records.len());
    print!("{}", scirust_bench_schema::to_jsonl(&records));
}
