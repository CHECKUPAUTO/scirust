//! **ANEE Phase D — D3 (certificate conservatism) + D5 (composition slack).**
//!
//! Protocols, criteria, and author priors are pre-registered in
//! `docs/research/ANEE_PHASE_D_PREREGISTRATION_2026-07-18.md` §2 (committed
//! before this file existed).
//!
//! * **D3** measures, for every representation the searches select and every
//!   admissible dictionary member, the *slack* between the certified
//!   round-trip bound (`roundtrip_bound(support).ulps × u`, `u = ε/2` — the
//!   same unit `certified_numerics` uses internally) and the observed
//!   round-trip error, and — kept separate on purpose — the *coverage gap*
//!   between that round-trip bound and the full-pipeline error it never
//!   claimed to cover. Priors: median slack ∈ [2, 100]; pipeline error at
//!   L = 8 on wide-range exceeds the bound ≥ 10×. Any observed error above
//!   its certified bound is a **soundness bug** and overrides everything.
//! * **D5** asks whether two-hop composition slack is the product of the
//!   hops' slacks (prior: median ratio in [0.75, 1.33] — conservatism
//!   compounds multiplicatively, no better, no worse).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scirust_core::autotune_accumulate::default_accumulators;
use scirust_core::certified_numerics::{CertifiedMonotone, Interval};
use scirust_core::representation_graph::{
    RepresentationChoice, default_representation_dictionary, joint_search_with_levels,
    sequential_baseline_with_levels, two_hop_dictionary,
};

/// Mirror of `certified_numerics`'s internal unit roundoff (`u = ε/2`): the
/// certified bound `ulps × UNIT` is a relative error, comparable directly to
/// the observed relative round-trip error below.
const UNIT: f64 = f64::EPSILON / 2.0;

const N: usize = 8192;

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

/// Observed vs certified round-trip error of `r` over `points` on `support`.
/// Returns `None` when `r` is inadmissible there (gate or domain).
struct SlackRow {
    certified_rel: f64,
    observed_rel: f64,
    /// `certified / observed`; `f64::INFINITY` when observed is exactly 0.
    slack: f64,
    violation: bool,
}

fn slack_of(r: RepresentationChoice, support: Interval, points: &[f64]) -> Option<SlackRow> {
    r.encode(support.lo)?;
    r.encode(support.hi)?;
    if r.kappa_rt_sup(support) * UNIT >= 0.5
    {
        return None;
    }
    let certified_rel = r.roundtrip_bound(support).ulps * UNIT;
    let mut observed_rel = 0.0f64;
    for &x in points
    {
        let e = r.encode(x)?;
        let back = r.decode(e);
        observed_rel = observed_rel.max(((back - x) / x).abs());
    }
    Some(SlackRow {
        certified_rel,
        observed_rel,
        slack: if observed_rel > 0.0
        {
            certified_rel / observed_rel
        }
        else
        {
            f64::INFINITY
        },
        violation: observed_rel > certified_rel,
    })
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(f64::total_cmp);
    if xs.is_empty()
    {
        f64::NAN
    }
    else
    {
        xs[xs.len() / 2]
    }
}

fn main() {
    println!(
        "ANEE Phase D — certificate conservatism + composition slack (pre-registered: docs/research/ANEE_PHASE_D_PREREGISTRATION_2026-07-18.md §2)\n"
    );
    type WorkloadGenerator = fn(u64, usize) -> Vec<f64>;
    let families: [(&str, WorkloadGenerator); 3] = [
        ("benign", benign),
        ("wide-range", wide_range),
        ("stagnation-prone", stagnation_prone),
    ];
    let r_dict = default_representation_dictionary();
    let a_dict = default_accumulators();
    let two_hops = two_hop_dictionary();
    let mut records: Vec<scirust_bench_schema::BenchRecord> = Vec::new();

    // ------------------------------------------------------------------ D3
    println!("=== D3: certificate conservatism (round-trip scope) ===");
    println!("priors: median slack in [2, 100]; any observed > certified is a SOUNDNESS BUG\n");
    let mut slacks: Vec<f64> = Vec::new();
    let mut infinite_observed = 0usize;
    let mut violations = 0usize;
    let mut selected_slacks: Vec<(String, String, f64)> = Vec::new();

    for (family, make) in families
    {
        let dev = make(1, N);
        let eval = make(2, N);
        let mut points = dev.clone();
        points.extend_from_slice(&eval);
        let (lo, hi) = points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| {
                (l.min(x), h.max(x))
            });
        let support = Interval::new(lo, hi);

        // Representations the searches actually select at L = 64.
        let mut chosen: Vec<(&str, RepresentationChoice)> = Vec::new();
        if let Some(j) = joint_search_with_levels(&dev, &eval, &r_dict, &a_dict, 64)
        {
            chosen.push(("joint", j.plan.representation));
        }
        if let Some(s) = sequential_baseline_with_levels(&dev, &eval, &r_dict, &a_dict, 64)
        {
            chosen.push(("sequential", s.plan.representation));
        }
        for &(who, r) in &chosen
        {
            if matches!(r, RepresentationChoice::Identity)
            {
                println!(
                    "[{family:>16}] {who} selected identity (observed round-trip error 0; slack undefined, reported apart)"
                );
                continue;
            }
            if let Some(row) = slack_of(r, support, &points)
            {
                println!(
                    "[{family:>16}] {who} selected {:<24} certified {:.3e} observed {:.3e} slack {:>9.1}x{}",
                    r.name(),
                    row.certified_rel,
                    row.observed_rel,
                    row.slack,
                    if row.violation
                    {
                        "  ** SOUNDNESS VIOLATION **"
                    }
                    else
                    {
                        ""
                    },
                );
                selected_slacks.push((family.to_string(), r.name(), row.slack));
            }
        }

        // Every admissible dictionary member (singles beyond identity + all
        // two-hops) contributes to the slack distribution.
        for &r in r_dict.iter().chain(two_hops.iter())
        {
            if matches!(r, RepresentationChoice::Identity)
            {
                continue;
            }
            let Some(row) = slack_of(r, support, &points)
            else
            {
                continue;
            };
            violations += usize::from(row.violation);
            if row.slack.is_finite()
            {
                slacks.push(row.slack);
            }
            else
            {
                infinite_observed += 1;
            }
            records.push(scirust_bench_schema::BenchRecord::new(
                "anee_phase_d_certificates/D3",
                format!("{family}/dev1eval2"),
                r.name(),
                1,
                "certified_over_observed_slack",
                row.slack,
            ));
        }
    }
    let n_slacks = slacks.len();
    let med = median(&mut slacks);
    let (min_s, max_s) = (
        slacks.first().copied().unwrap_or(f64::NAN),
        slacks.last().copied().unwrap_or(f64::NAN),
    );
    println!(
        "\nslack distribution over {n_slacks} (family, member) samples: min {min_s:.1}x  median {med:.1}x  max {max_s:.1}x  ({infinite_observed} zero-observed excluded)"
    );
    println!(
        "D3 VERDICT (within scope): median slack {med:.1}x -> prior [2, 100] {}; soundness violations: {violations} {}",
        if (2.0..=100.0).contains(&med)
        {
            "CONFIRMED"
        }
        else
        {
            "FALSIFIED"
        },
        if violations == 0
        {
            "(bounds sound on all observed data)"
        }
        else
        {
            "** SOUNDNESS BUG — overrides everything in D3 **"
        },
    );

    // Coverage gap, kept apart: the certificate never covered the pipeline.
    let dev = wide_range(1, N);
    let eval = wide_range(2, N);
    let report = joint_search_with_levels(&dev, &eval, &r_dict, &a_dict, 8)
        .expect("wide-range joint at L=8");
    let certified_rel = report.certificate.ulps * UNIT;
    let ratio = report.held_out_relative_error / certified_rel;
    println!(
        "coverage gap (wide-range, L=8): pipeline error {:.3e} vs round-trip bound {:.3e} -> {ratio:.1e}x uncovered; prior (>= 10x) {}\n",
        report.held_out_relative_error,
        certified_rel,
        if ratio >= 10.0
        {
            "CONFIRMED"
        }
        else
        {
            "FALSIFIED"
        },
    );
    records.push(
        scirust_bench_schema::BenchRecord::new(
            "anee_phase_d_certificates/D3",
            "wide-range/L=8",
            format!("coverage:{}", report.plan.representation.name()),
            1,
            "pipeline_error_over_roundtrip_bound",
            ratio,
        )
        .with_cert(scirust_bench_schema::Certificate {
            description: "kappa_rt round-trip bound (CANR §3.2) — round-trip scope only".into(),
            bound_ulps: Some(report.certificate.ulps),
            determinism: None,
        }),
    );

    // ------------------------------------------------------------------ D5
    println!("=== D5: is composition slack the product of hop slacks? ===");
    println!("prior: median slack_composed / (slack_hop1 x slack_hop2) in [0.75, 1.33]\n");
    let mut ratios: Vec<f64> = Vec::new();
    for (family, make) in families
    {
        let dev = make(1, N);
        let eval = make(2, N);
        let mut points = dev.clone();
        points.extend_from_slice(&eval);
        let (lo, hi) = points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| {
                (l.min(x), h.max(x))
            });
        let support = Interval::new(lo, hi);

        for &c in &two_hops
        {
            let RepresentationChoice::Composed(a, b) = c
            else
            {
                continue;
            };
            let Some(row_c) = slack_of(c, support, &points)
            else
            {
                continue;
            };
            let hop1 = RepresentationChoice::Certified(a);
            let Some(row_1) = slack_of(hop1, support, &points)
            else
            {
                continue;
            };
            // Hop 2 lives on hop 1's image.
            let (Some(e_lo), Some(e_hi)) = (hop1.encode(support.lo), hop1.encode(support.hi))
            else
            {
                continue;
            };
            let image = Interval::new(e_lo.min(e_hi), e_lo.max(e_hi));
            let encoded: Vec<f64> = points
                .iter()
                .filter_map(|&x| hop1.encode(x))
                .filter(|&e| e != 0.0)
                .collect();
            let Some(row_2) = slack_of(RepresentationChoice::Certified(b), image, &encoded)
            else
            {
                continue;
            };
            if !(row_c.slack.is_finite() && row_1.slack.is_finite() && row_2.slack.is_finite())
            {
                continue;
            }
            let ratio = row_c.slack / (row_1.slack * row_2.slack);
            ratios.push(ratio);
            records.push(scirust_bench_schema::BenchRecord::new(
                "anee_phase_d_certificates/D5",
                format!("{family}/dev1eval2"),
                c.name(),
                1,
                "composed_slack_over_hop_product",
                ratio,
            ));
        }
    }
    let n_ratios = ratios.len();
    let med_r = median(&mut ratios);
    let (min_r, max_r) = (
        ratios.first().copied().unwrap_or(f64::NAN),
        ratios.last().copied().unwrap_or(f64::NAN),
    );
    println!(
        "ratio distribution over {n_ratios} (family, composition) samples: min {min_r:.2}  median {med_r:.2}  max {max_r:.2}"
    );
    let verdict = if !(0.75..=1.33).contains(&med_r) && med_r < 0.75
    {
        "BELOW band — composition certifies TIGHTER than naive slack compounding (small positive surprise; author prior falsified)"
    }
    else if !(0.75..=1.33).contains(&med_r)
    {
        "ABOVE band — composition adds conservatism beyond its parts (author prior falsified; the field-pessimism reproduced in miniature)"
    }
    else
    {
        "IN band — slack compounds multiplicatively, as the author prior predicted"
    };
    println!("D5 VERDICT: median ratio {med_r:.2} -> {verdict}");

    println!("\n=== bench-schema JSONL ({} records) ===", records.len());
    print!("{}", scirust_bench_schema::to_jsonl(&records));
}
