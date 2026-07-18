//! **ANEE Phase D — D1 (bucket-collision attack) + D4 (distribution-keyed
//! caching at scale).**
//!
//! Protocols, decisive criteria, and author priors are pre-registered in
//! `docs/research/ANEE_PHASE_D_PREREGISTRATION_2026-07-18.md` §2 (committed
//! before this file existed); nothing here may soften them.
//!
//! * **D1** constructs four workload pairs designed to **collide** in
//!   [`DistributionSummary`] (verified at runtime) while having different
//!   optimal plans, then measures the regret of serving one member's plan to
//!   the other. Attack succeeds if ≥ 1 verified-colliding pair reaches
//!   regret ≥ 2.0×. Author prior: succeeds.
//! * **D4** runs the pre-registered 240-batch drifting stream and compares
//!   three cache policies (kernel-only / distribution-keyed [`PlanCache`] /
//!   oracle re-search). Candidate **pays** if regret ≤ 1.25× oracle at
//!   ≤ 20% of oracle's searches AND kernel-only regret ≥ 2× its regret;
//!   candidate **dies** if kernel-only ≤ 1.1× its regret.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scirust_core::autotune_accumulate::{AccumMethod, default_accumulators};
use scirust_core::representation_graph::{
    Plan, PlanCache, RepresentationChoice, current_hardware_key, default_representation_dictionary,
    joint_search_with_levels, pipeline_relative_error_with_levels, summarize,
};

const LEVELS: usize = 64;
const N: usize = 8192;
const FRESH: [u64; 3] = [13, 14, 15];

// ---------------------------------------------------------------------------
// D1 — pair constructions (pre-registered designs P1–P4)
// ---------------------------------------------------------------------------

/// Log-uniform over 6 decades, `10^U(-3,3)`.
fn log_uniform_6(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| 10f64.powf(rng.gen_range(-3.0..3.0)))
        .collect()
}

/// Endpoint-bimodal over the same 6 decades: half the mass in the bottom
/// fifth-decade, half in the top fifth-decade.
fn endpoint_bimodal_6(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            if rng.gen_range(0.0..1.0) < 0.5
            {
                10f64.powf(rng.gen_range(-3.0..-2.8))
            }
            else
            {
                10f64.powf(rng.gen_range(2.8..3.0))
            }
        })
        .collect()
}

/// The classic stagnation-prone family (80% ≈ 1e-3, 20% ≈ 1e3).
fn stagnation_frac(seed: u64, n: usize, small_frac: f64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let u: f64 = rng.gen_range(0.0..1.0);
            if u < small_frac
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

/// 89.5% of mass in the top fifth-decade, 10.5% spread across the three
/// decades *below* the `max·1e-3` stagnation threshold — same summary bucket
/// as the classic stagnation family, very different mass structure.
fn thin_small_tail_6(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            if rng.gen_range(0.0..1.0) < 0.105
            {
                // [max·1e-6, max·1e-3) with max ≈ 2e3.
                2e3 * 10f64.powf(rng.gen_range(-6.0..-3.05))
            }
            else
            {
                1e3 * (1.0 + rng.gen_range(0.0..1.0))
            }
        })
        .collect()
}

/// Benign mass (0.5..1.5) plus one enormous outlier — six decades of range
/// produced by a single point.
fn benign_with_outlier(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<f64> = (0..n).map(|_| rng.gen_range(0.5..1.5)).collect();
    v[0] = 5e5;
    v
}

type Maker = fn(u64, usize) -> Vec<f64>;

fn plan_label(plan: Plan) -> String {
    format!("{}+{:?}", plan.representation.name(), plan.accumulation)
}

/// Per-fresh-seed held-out errors of `plan` on `make`'s data (quantizer refit
/// on `fit`, exactly the cache-serving situation). A gate failure falls back
/// to the ungated identity default and is counted — never silently dropped.
fn fresh_errors(
    plan: Plan,
    fit: &[f64],
    make: Maker,
    fallback_count: &mut usize,
) -> Vec<(u64, f64)> {
    FRESH
        .iter()
        .map(|&s| {
            let held = make(s, N);
            let e = pipeline_relative_error_with_levels(plan, fit, &held, LEVELS).unwrap_or_else(
                || {
                    *fallback_count += 1;
                    pipeline_relative_error_with_levels(
                        Plan {
                            representation: RepresentationChoice::Identity,
                            accumulation: AccumMethod::NaiveF32,
                        },
                        fit,
                        &held,
                        LEVELS,
                    )
                    .expect("identity pipeline cannot fail on positive data")
                },
            );
            (s, e)
        })
        .collect()
}

fn mean_of(per_seed: &[(u64, f64)]) -> f64 {
    per_seed.iter().map(|&(_, e)| e).sum::<f64>() / per_seed.len() as f64
}

fn run_d1(records: &mut Vec<scirust_bench_schema::BenchRecord>) -> bool {
    println!("=== D1: PlanCache bucket-collision attack (L = {LEVELS}) ===");
    println!("bar: >= 1 verified-colliding pair with regret >= 2.0x; author prior: SUCCEEDS\n");

    let pairs: [(&str, Maker, &str, Maker); 4] = [
        ("P1", log_uniform_6, "log-uniform-6dec", endpoint_bimodal_6),
        (
            "P2",
            stagnation_frac_80,
            "stagnation-80pc",
            thin_small_tail_6,
        ),
        ("P3", benign_with_outlier, "benign+outlier", log_uniform_6),
        (
            "P4",
            stagnation_frac_15,
            "stagnation-15pc",
            stagnation_frac_60,
        ),
    ];
    let pair_y_names = [
        "endpoint-bimodal-6dec",
        "thin-small-tail-6dec",
        "log-uniform-6dec",
        "stagnation-60pc",
    ];

    let r_dict = default_representation_dictionary();
    let a_dict = default_accumulators();
    let mut any_success = false;

    for (i, &(pair, make_x, name_x, make_y)) in pairs.iter().enumerate()
    {
        let name_y = pair_y_names[i];
        let dev_x = make_x(1, N);
        let dev_y = make_y(1, N);
        let (sx, sy) = (summarize(&dev_x), summarize(&dev_y));
        if sx != sy
        {
            println!(
                "[{pair}] DISCARDED: summaries do not collide ({sx:?} vs {sy:?}) — reported per pre-registration"
            );
            continue;
        }

        let plan_x = joint_search_with_levels(&dev_x, &make_x(2, N), &r_dict, &a_dict, LEVELS)
            .expect("joint search on x")
            .plan;
        let plan_y = joint_search_with_levels(&dev_y, &make_y(2, N), &r_dict, &a_dict, LEVELS)
            .expect("joint search on y")
            .plan;

        let mut fallbacks = 0usize;
        let served_per_seed = fresh_errors(plan_x, &dev_y, make_y, &mut fallbacks);
        let own_per_seed = fresh_errors(plan_y, &dev_y, make_y, &mut fallbacks);
        let served = mean_of(&served_per_seed);
        let own = mean_of(&own_per_seed);
        let regret = served / own;
        let success = regret >= 2.0;
        any_success |= success;

        println!(
            "[{pair}] collide {sx:?} | x={name_x} plan {} | y={name_y} plan {} | served={served:.3e} own={own:.3e} regret={regret:.2}x {}{}",
            plan_label(plan_x),
            plan_label(plan_y),
            if success { "-> ATTACK HIT" } else { "-> held" },
            if fallbacks > 0
            {
                format!("  [{fallbacks} gate fallbacks]")
            }
            else
            {
                String::new()
            },
        );
        if plan_x != plan_y
        {
            let r_differs = plan_x.representation != plan_y.representation;
            let a_differs = plan_x.accumulation != plan_y.accumulation;
            println!(
                "        divergent component(s): {}{}",
                if r_differs { "R " } else { "" },
                if a_differs { "A" } else { "" }
            );
        }
        for (role, plan, per_seed) in [
            ("served_cross", plan_x, &served_per_seed),
            ("own_optimum", plan_y, &own_per_seed),
        ]
        {
            for &(s, e) in per_seed.iter()
            {
                records.push(scirust_bench_schema::BenchRecord::new(
                    "anee_phase_d_cache/D1",
                    format!("{name_x}->{name_y}"),
                    format!("{role}:{}", plan_label(plan)),
                    s,
                    "fresh_relative_error",
                    e,
                ));
            }
        }
    }
    println!(
        "\nD1 VERDICT: attack {}
",
        if any_success
        {
            "SUCCEEDS — the 2-field summary is exploitably coarse (as the author prior predicted)"
        }
        else
        {
            "FAILS — no constructed collision produced >= 2x regret (author prior falsified)"
        }
    );
    any_success
}

fn stagnation_frac_80(seed: u64, n: usize) -> Vec<f64> {
    stagnation_frac(seed, n, 0.8)
}
fn stagnation_frac_15(seed: u64, n: usize) -> Vec<f64> {
    stagnation_frac(seed, n, 0.15)
}
fn stagnation_frac_60(seed: u64, n: usize) -> Vec<f64> {
    stagnation_frac(seed, n, 0.6)
}

// ---------------------------------------------------------------------------
// D4 — the 240-batch drifting stream
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Family {
    Benign,
    Wide,
    Stagnation,
}

/// Generate batch data for a family; `pos` in [0,1] drives the block drift,
/// and regime-noise batches use the family's fixed base parameters
/// (`pos = None`).
fn family_batch(family: Family, pos: Option<f64>, seed: u64, n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    match family
    {
        Family::Benign =>
        {
            let (lo0, hi0, lo1, hi1) = (0.5f64, 1.5f64, 0.1f64, 10.0f64);
            let (lo, hi) = match pos
            {
                Some(p) => (lo0 * (lo1 / lo0).powf(p), hi0 * (hi1 / hi0).powf(p)),
                None => (lo0, hi0),
            };
            (0..n).map(|_| rng.gen_range(lo..hi)).collect()
        },
        Family::Wide =>
        {
            let d = match pos
            {
                Some(p) => 4.0 + 4.0 * p,
                None => 6.0,
            };
            (0..n)
                .map(|_| 10f64.powf(rng.gen_range(-d / 2.0..d / 2.0)))
                .collect()
        },
        Family::Stagnation =>
        {
            let frac = match pos
            {
                Some(p) => 0.6 + 0.3 * p,
                None => 0.8,
            };
            (0..n)
                .map(|_| {
                    let u: f64 = rng.gen_range(0.0..1.0);
                    if u < frac
                    {
                        1e-3 * (1.0 + u)
                    }
                    else
                    {
                        1e3 * (1.0 + u)
                    }
                })
                .collect()
        },
    }
}

struct PolicyStats {
    errors: Vec<f64>,
    searches: usize,
    gate_fallbacks: usize,
}

impl PolicyStats {
    fn new() -> Self {
        Self {
            errors: Vec::new(),
            searches: 0,
            gate_fallbacks: 0,
        }
    }
    fn mean(&self) -> f64 {
        self.errors.iter().sum::<f64>() / self.errors.len() as f64
    }
}

fn served_error(plan: Plan, batch: &[f64], held: &[f64], stats: &mut PolicyStats) -> f64 {
    match pipeline_relative_error_with_levels(plan, batch, held, LEVELS)
    {
        Some(e) => e,
        None =>
        {
            // Pre-registered deviation handling: a served plan whose gate
            // fails on this batch falls back to the ungated default.
            stats.gate_fallbacks += 1;
            pipeline_relative_error_with_levels(
                Plan {
                    representation: RepresentationChoice::Identity,
                    accumulation: AccumMethod::NaiveF32,
                },
                batch,
                held,
                LEVELS,
            )
            .expect("identity pipeline cannot fail on positive data")
        },
    }
}

fn run_d4(records: &mut Vec<scirust_bench_schema::BenchRecord>) {
    const T: usize = 240;
    const BATCH_N: usize = 4096;
    println!(
        "=== D4: distribution-keyed caching over a {T}-batch drifting stream (L = {LEVELS}, n = {BATCH_N}) ==="
    );
    println!("pays: regret <= 1.25x oracle at <= 20% searches AND kernel-only >= 2x its regret");
    println!("dies: kernel-only regret <= 1.1x distribution-keyed regret; author prior: PAYS\n");

    let r_dict = default_representation_dictionary();
    let a_dict = default_accumulators();
    let hw = current_hardware_key();
    let mut stream_rng = StdRng::seed_from_u64(4242);

    let mut oracle = PolicyStats::new();
    let mut kernel_only = PolicyStats::new();
    let mut dist_keyed = PolicyStats::new();
    let mut cache = PlanCache::new();
    let mut frozen: Option<Plan> = None;

    for t in 0..T
    {
        let block_family = match t / 80
        {
            0 => Family::Benign,
            1 => Family::Wide,
            _ => Family::Stagnation,
        };
        let pos = (t % 80) as f64 / 79.0;
        let switch: f64 = stream_rng.gen_range(0.0..1.0);
        let (family, drift) = if switch < 0.2
        {
            let others: [Family; 2] = match block_family
            {
                Family::Benign => [Family::Wide, Family::Stagnation],
                Family::Wide => [Family::Benign, Family::Stagnation],
                Family::Stagnation => [Family::Benign, Family::Wide],
            };
            (others[stream_rng.gen_range(0..2usize)], None)
        }
        else
        {
            (block_family, Some(pos))
        };

        let batch = family_batch(family, drift, 20_000 + t as u64, BATCH_N);
        let held = family_batch(family, drift, 10_000 + t as u64, BATCH_N);
        let (dev, evl) = batch.split_at(BATCH_N / 2);

        // Oracle: search every batch.
        let plan_o = joint_search_with_levels(dev, evl, &r_dict, &a_dict, LEVELS)
            .expect("joint search must find a plan on positive data");
        oracle.searches += 1;
        let e_o = served_error(plan_o.plan, &batch, &held, &mut oracle);
        oracle.errors.push(e_o);

        // Kernel-only: one search on the first batch, frozen forever.
        let plan_k = *frozen.get_or_insert_with(|| {
            kernel_only.searches += 1;
            joint_search_with_levels(dev, evl, &r_dict, &a_dict, LEVELS)
                .expect("first-batch search")
                .plan
        });
        let e_k = served_error(plan_k, &batch, &held, &mut kernel_only);
        kernel_only.errors.push(e_k);

        // Distribution-keyed: the committed PlanCache.
        let key = summarize(&batch);
        let plan_d = match cache.get("phase_d_stream", key, hw)
        {
            Some(hit) => hit.plan,
            None =>
            {
                dist_keyed.searches += 1;
                let report = joint_search_with_levels(dev, evl, &r_dict, &a_dict, LEVELS)
                    .expect("miss search");
                cache.insert(
                    "phase_d_stream",
                    key,
                    hw,
                    report.plan,
                    report.held_out_relative_error,
                    report.certificate,
                );
                report.plan
            },
        };
        let e_d = served_error(plan_d, &batch, &held, &mut dist_keyed);
        dist_keyed.errors.push(e_d);

        for (policy, e, plan) in [
            ("oracle", e_o, plan_o.plan),
            ("kernel_only", e_k, plan_k),
            ("dist_keyed", e_d, plan_d),
        ]
        {
            records.push(scirust_bench_schema::BenchRecord::new(
                "anee_phase_d_cache/D4",
                format!("{family:?}/t={t}"),
                format!("{policy}:{}", plan_label(plan)),
                10_000 + t as u64,
                "held_out_relative_error",
                e,
            ));
        }
    }

    let (m_o, m_k, m_d) = (oracle.mean(), kernel_only.mean(), dist_keyed.mean());
    let (regret_k, regret_d) = (m_k / m_o, m_d / m_o);
    println!(
        "  oracle       : mean err {m_o:.4e}, searches {}",
        oracle.searches
    );
    println!(
        "  kernel-only  : mean err {m_k:.4e}, regret {regret_k:.2}x, searches {}, gate fallbacks {}",
        kernel_only.searches, kernel_only.gate_fallbacks
    );
    println!(
        "  dist-keyed   : mean err {m_d:.4e}, regret {regret_d:.2}x, searches {} ({:.1}% of oracle), gate fallbacks {}",
        dist_keyed.searches,
        100.0 * dist_keyed.searches as f64 / oracle.searches as f64,
        dist_keyed.gate_fallbacks
    );

    let pays = regret_d <= 1.25
        && dist_keyed.searches as f64 <= 0.20 * oracle.searches as f64
        && regret_k >= 2.0 * regret_d;
    let dies = regret_k <= 1.1 * regret_d;
    println!(
        "\nD4 VERDICT: the distribution-keyed candidate {}",
        if pays
        {
            "PAYS (both pre-registered conditions met — author prior confirmed)"
        }
        else if dies
        {
            "DIES — kernel-only matches it; distribution keying adds nothing here (author prior falsified)"
        }
        else
        {
            "is INCONCLUSIVE per the pre-registered bands — numbers above stand as reported"
        }
    );
}

fn main() {
    println!(
        "ANEE Phase D — cache experiments (pre-registered: docs/research/ANEE_PHASE_D_PREREGISTRATION_2026-07-18.md §2)\n"
    );
    let mut records: Vec<scirust_bench_schema::BenchRecord> = Vec::new();
    run_d1(&mut records);
    run_d4(&mut records);
    println!("\n=== bench-schema JSONL ({} records) ===", records.len());
    print!("{}", scirust_bench_schema::to_jsonl(&records));
}
