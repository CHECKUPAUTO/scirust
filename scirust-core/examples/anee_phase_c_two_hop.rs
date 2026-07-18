//! **ANEE Addendum 3, avenue 2: is the "representation graph" really a
//! graph?**
//!
//! (`docs/research/ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md`,
//! §4.) Honest gap in both Phase C kernels so far: every search ran over a
//! *flat dictionary* of single-hop representations — the graph structure of
//! ANEE §4 (multi-hop paths, the exact κ composition law of Proposition
//! ANEE-2/Z3) was formalized and validated but never *exercised*. This
//! benchmark adds every ordered two-hop composition of the dictionary
//! ([`two_hop_dictionary`], 20 candidates whose κ_rt is computed by the
//! exact product law — the first executable use of Proposition ANEE-2) and
//! asks whether any two-hop path ever beats the best single-hop plan.
//!
//! ## Pre-registered protocol and criterion (written before any run)
//!
//! Grid: kernel 1's three families × levels `L ∈ {8, 64}` = **6 cells**
//! (low L is where companding curvature matters most — the composed hops'
//! best shot; 64 is kernel 1's published operating point). Per cell, two
//! independent joint searches with kernel 1's exact protocol (selection on
//! (dev, eval), winner re-scored on 3 fresh held-out seeds):
//!
//! * **singles**: the 5-member single-hop dictionary;
//! * **pairs**: the 20-member two-hop dictionary alone.
//!
//! A cell is a **two-hop win** iff the pairs winner's 3-fresh-seed mean
//! error is `≤ 0.8 ×` the singles winner's (the same 20% bar as every
//! Phase C experiment).
//!
//! **Decisive criterion:** the graph structure earns its name iff **≥ 1 of
//! the 6 cells** is a two-hop win (the claim under test is existential —
//! "is the path structure ever useful?"). If **zero** cells qualify, then
//! on all evidence gathered so far the "representation graph" is a flat
//! dictionary wearing a graph's name, and Addendum 3 must recommend
//! renaming/simplifying accordingly.
//!
//! **Author's declared prior (stated in advance):** zero wins expected —
//! several compositions are affine-equivalent to a single hop (e.g.
//! `power(λ)` then `log` = `λ·log x`, which the affine-invariant uniform
//! quantizer cannot distinguish from `log`), others will be rejected
//! outright by the encode gate (negative intermediates into
//! positive-domain second hops), and the surviving genuinely-new companding
//! curves (e.g. `log1p(√x)`) have no obvious reason to beat the best single
//! compander by a further 20%. A zero-win outcome would *self-falsify our
//! own §4 naming*, which is precisely the kind of result this program
//! exists to surface.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scirust_core::autotune_accumulate::{AccumMethod, default_accumulators};
use scirust_core::representation_graph::{
    Plan, RepresentationChoice, default_representation_dictionary, joint_search_with_levels,
    pipeline_relative_error_with_levels, two_hop_dictionary,
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
    let level_grid = [8usize, 64];
    let singles = default_representation_dictionary();
    let pairs = two_hop_dictionary();
    let a_dict = default_accumulators();

    println!("ANEE Addendum 3, avenue 2: two-hop paths vs. flat single-hop dictionary");
    println!(
        "singles: {} members; pairs: {} members; grid: {} families x {} levels = 6 cells",
        singles.len(),
        pairs.len(),
        families.len(),
        level_grid.len()
    );
    println!("Criterion (existential): >= 1 of 6 cells where the pairs winner beats the");
    println!("singles winner by >= 20% on 3-fresh-seed mean error; zero => the 'graph' is a");
    println!("flat dictionary and the doc must say so.\n");

    let mut two_hop_wins: Vec<String> = Vec::new();

    for (family, make) in families
    {
        for &levels in &level_grid
        {
            let dev = make(1, 8192);
            let eval = make(2, 8192);

            // How many composed candidates even survive the encode gate here?
            let admitted = pairs
                .iter()
                .filter(|&&r| {
                    pipeline_relative_error_with_levels(
                        Plan {
                            representation: r,
                            accumulation: AccumMethod::NeumaierF32,
                        },
                        &dev,
                        &eval,
                        levels,
                    )
                    .is_some()
                })
                .count();

            let single = joint_search_with_levels(&dev, &eval, &singles, &a_dict, levels)
                .expect("singles search must find a plan");
            let pair = joint_search_with_levels(&dev, &eval, &pairs, &a_dict, levels);

            let fresh_seeds = [13u64, 14, 15];
            let mean_of = |plan: Plan| -> f64 {
                fresh_seeds
                    .iter()
                    .map(|&s| {
                        pipeline_relative_error_with_levels(plan, &dev, &make(s, 8192), levels)
                            .unwrap_or(f64::NAN)
                    })
                    .sum::<f64>()
                    / fresh_seeds.len() as f64
            };
            let single_mean = mean_of(single.plan);

            match pair
            {
                None =>
                {
                    println!(
                        "[{family:>16} L={levels:>2}] pairs admitted {admitted}/20; NO valid two-hop plan at all; singles winner {} = {:.4e}",
                        plan_label(single.plan),
                        single_mean
                    );
                },
                Some(pair) =>
                {
                    let pair_mean = mean_of(pair.plan);
                    let wins = pair_mean <= 0.8 * single_mean;
                    let margin = 1.0 - pair_mean / single_mean;
                    println!(
                        "[{family:>16} L={levels:>2}] pairs admitted {admitted}/20; singles {} = {:.4e} | pairs {} = {:.4e} | two-hop margin {:>6.1}% -> {}",
                        plan_label(single.plan),
                        single_mean,
                        plan_label(pair.plan),
                        pair_mean,
                        margin * 100.0,
                        if wins { "TWO-HOP WIN" } else { "no win" }
                    );
                    if wins
                    {
                        two_hop_wins.push(format!("{family} L={levels}"));
                    }
                },
            }
        }
        println!();
    }

    println!("=== pre-registered criterion ===");
    println!(
        "  two-hop wins: {}/6 cells ({:?})",
        two_hop_wins.len(),
        two_hop_wins
    );
    if two_hop_wins.is_empty()
    {
        println!(
            "  VERDICT: ZERO two-hop wins — on all evidence so far the 'representation graph'\n  reduces to a flat dictionary; Addendum 3 must recommend renaming/simplifying the\n  §4 framing accordingly."
        );
    }
    else
    {
        println!(
            "  VERDICT: the graph structure earned its name on at least one cell — multi-hop\n  paths are demonstrably useful for this task family."
        );
    }

    // ---------------------------------------------------------------------
    // POST-HOC DIAGNOSTIC — added AFTER the pre-registered run above was
    // executed and its verdict recorded (first run's winners: wide-range
    // L=8 anscombe->anscombe, L=64 power->power). Not part of the
    // criterion; kept in the committed record per program discipline.
    //
    // The observed two-hop winners look like *stronger companders* the
    // single dictionary simply lacks (power(0.5) twice = x^0.25 — literally
    // a single power hop at a lambda the singles never offered). Competing
    // explanations for the wins: (a) path structure per se is valuable, vs.
    // (b) composition merely DENSIFIES the dictionary with new curves, and
    // an enriched single-hop dictionary would do as well. This diagnostic
    // distinguishes them: rerun the singles search with Power(0.25) and
    // Power(0.125) added, on the two winning cells only.
    // ---------------------------------------------------------------------
    println!(
        "\n=== post-hoc diagnostic (not pre-registered): densified singles vs. two-hop winners ==="
    );
    let mut densified = default_representation_dictionary();
    densified.push(RepresentationChoice::Certified(
        scirust_core::transform_search::Representation::Power(0.25),
    ));
    densified.push(RepresentationChoice::Certified(
        scirust_core::transform_search::Representation::Power(0.125),
    ));
    for &levels in &level_grid
    {
        let dev = wide_range(1, 8192);
        let eval = wide_range(2, 8192);
        let fresh_seeds = [13u64, 14, 15];
        let mean_of = |plan: Plan| -> f64 {
            fresh_seeds
                .iter()
                .map(|&s| {
                    pipeline_relative_error_with_levels(plan, &dev, &wide_range(s, 8192), levels)
                        .unwrap_or(f64::NAN)
                })
                .sum::<f64>()
                / fresh_seeds.len() as f64
        };
        let dense = joint_search_with_levels(&dev, &eval, &densified, &a_dict, levels)
            .expect("densified search must find a plan");
        let pair = joint_search_with_levels(&dev, &eval, &pairs, &a_dict, levels)
            .expect("pairs search must find a plan");
        let dense_mean = mean_of(dense.plan);
        let pair_mean = mean_of(pair.plan);
        println!(
            "[wide-range L={levels:>2}] densified-singles {} = {:.4e} | two-hop {} = {:.4e} | densified within 20% of two-hop: {}",
            plan_label(dense.plan),
            dense_mean,
            plan_label(pair.plan),
            pair_mean,
            if dense_mean <= pair_mean * 1.25
            {
                "yes -> wins are dictionary densification"
            }
            else
            {
                "NO -> path structure adds value beyond densification"
            }
        );
    }
}
