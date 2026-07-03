//! Worked inertial-tolerancing example, reproducing the assembly of
//! arXiv:1002.0270 (`Y = X₁ − X₂ − X₃ − X₄ − X₅`, target gap 1 mm, tolerance
//! interval `R_Y = 1` mm), and demonstrating the capability + piloting layers.
//!
//! Run with: `cargo run -p scirust-tolerance --example stackup_comparison`

use scirust_tolerance::capability::{cpi, cpk, cpm};
use scirust_tolerance::chain::{
    Allocation, TraditionalMethod, allocate, allocate_traditional, max_dispersion,
};
use scirust_tolerance::chart::PilotingChart;
use scirust_tolerance::inertia::{Inertia, InertiaCone, i_max_from_tolerance, mix_lots};
use scirust_tolerance::sampling::design_plan;

fn main() {
    // ---- 1. Tolerance-chain allocation (top-down synthesis) ---------------
    let r_y = 1.0; // assembly tolerance interval
    let coeffs = [1.0, -1.0, -1.0, -1.0, -1.0]; // five ±1 links
    let i_y = i_max_from_tolerance(r_y, 1.0); // Cp=1 inertia budget = R_Y/6

    println!(
        "Assembly Y = X1 - X2 - X3 - X4 - X5,  R_Y = {r_y},  n = {}",
        coeffs.len()
    );
    println!("Cp=1 assembly inertia budget  I_Y = R_Y/6 = {i_y:.4}\n");

    let wc = allocate(i_y, &coeffs, &Allocation::WorstCase).unwrap();
    let st = allocate(i_y, &coeffs, &Allocation::Statistical).unwrap();
    let gk = allocate(i_y, &coeffs, &Allocation::GuaranteedCpk(1.0)).unwrap();

    // Traditional interval allocations, converted to a centred σ_max = R/6 so
    // they sit on the same scale as an inertia (for a centred batch σ_max=Iᵢ).
    let tw = allocate_traditional(r_y, &coeffs, TraditionalMethod::WorstCase);
    let ts = allocate_traditional(r_y, &coeffs, TraditionalMethod::Statistical);
    let ti = allocate_traditional(r_y, &coeffs, TraditionalMethod::Inflated(1.5));

    println!("Per-component budget on each Xᵢ (mm):");
    println!(
        "  traditional  worst-case      σ_max = {:.4}",
        max_dispersion(tw[0])
    );
    println!(
        "  traditional  statistical     σ_max = {:.4}",
        max_dispersion(ts[0])
    );
    println!(
        "  traditional  inflated f=1.5  σ_max = {:.4}",
        max_dispersion(ti[0])
    );
    println!("  inertial     worst-case      I_i   = {:.4}", wc[0]);
    println!("  inertial     statistical     I_i   = {:.4}", st[0]);
    println!("  inertial     guarantee Cpk=1 I_i   = {:.4}", gk[0]);
    println!("(matches paper Table 2: 0.033 / 0.075 / 0.050 / 0.033 / 0.075 / 0.060)\n");

    // ---- 2. Capability of a produced component ----------------------------
    // A link produced off-centre by 0.03 mm with σ = 0.05 mm, spec ±0.5.
    let (delta, sigma) = (0.03, 0.05);
    let inertia = Inertia::new(delta, sigma);
    let comp_it = 2.0 * 0.5; // ±0.5 ⇒ interval 1.0
    let i_max_comp = i_max_from_tolerance(comp_it, 1.0);
    let mean = 1.0 + delta; // target 1.0
    println!(
        "Component: δ = {delta}, σ = {sigma}  ⇒  I = {:.4}",
        inertia.value()
    );
    println!("  Cpk = {:.3}", cpk(mean, sigma, 0.5, 1.5));
    println!("  Cpm = {:.3}", cpm(mean, sigma, 1.0, 0.5, 1.5));
    println!(
        "  Cpi = I_max/I = {:.3}  (I_max = {i_max_comp:.4})",
        cpi(&inertia, i_max_comp)
    );
    let cone = InertiaCone::new(i_max_comp);
    println!("  inside inertia cone? {}\n", cone.accepts(&inertia));

    // ---- 3. Piloting a production run (carte de pilotage inertiel) --------
    let chart = PilotingChart::new(1.0, i_max_comp, 5);
    println!(
        "Piloting chart: target 1.0, I_max {i_max_comp:.4}, n=5, UPL(0.27%) = {:.4}",
        chart.upper_limit(0.0027)
    );
    let subgroups = [
        vec![1.00, 1.01, 0.99, 1.00, 1.01], // centred, tight   → let run
        vec![1.35, 1.36, 1.34, 1.35, 1.36], // off-centre high  → recentre
        vec![0.58, 1.42, 0.62, 1.38, 1.00], // centred, huge σ  → reduce σ
    ];
    for (k, g) in subgroups.iter().enumerate()
    {
        let s = chart.evaluate(g, 0.0027);
        println!(
            "  subgroup {}: Î = {:.4}  {:<11} → {:?} (shift {:+.3})",
            k + 1,
            s.inertia,
            if s.in_control { "in-control" } else { "OUT" },
            s.action,
            s.recommended_shift
        );
    }

    // ---- 4. Lot mixing (a headline advantage of inertial tolerancing) -----
    // Two sub-lots, each centred-ish but off-target in opposite directions.
    let lot_a = Inertia::new(0.08, 0.03); // I ≈ 0.0854
    let lot_b = Inertia::new(-0.08, 0.03);
    let mixed = mix_lots(&[(1.0, lot_a), (1.0, lot_b)]);
    println!(
        "\nLot mixing: I_A = {:.4}, I_B = {:.4}  ⇒  pooled I = {:.4} (δ = {:.3})",
        lot_a.value(),
        lot_b.value(),
        mixed.value(),
        mixed.off_centering
    );
    println!("  (I_c² = mean of the sub-lot I²; the mixed lot is on-target but wider)");

    // ---- 5. Acceptance sampling by inertia --------------------------------
    let (alpha, beta, ratio_bad) = (0.05, 0.10, 2.0);
    if let Some(plan) = design_plan(alpha, beta, ratio_bad, 500)
    {
        println!(
            "\nAcceptance plan (α={alpha}, β={beta}, bad at {ratio_bad}·I_max): \
             sample n = {}, accept if Î ≤ {:.3}·I_max",
            plan.n, plan.factor
        );
        println!(
            "  P(accept good @ I_max)   = {:.3}  (≥ {:.2})",
            plan.probability_of_acceptance_at(1.0, 1.0, 0.0),
            1.0 - alpha
        );
        println!(
            "  P(accept bad  @ {ratio_bad}·I_max) = {:.3}  (≤ {:.2})",
            plan.probability_of_acceptance_at(1.0, ratio_bad, 0.0),
            beta
        );
    }
}
