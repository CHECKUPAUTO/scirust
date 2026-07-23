//! Deterministic benchmark for phase 4E.6 — conditional and temporal conformal
//! prediction, each shown against the marginal split-conformal baseline it fixes.
//!
//! **Temporal drift.** A stream whose noise scale grows over time is *not*
//! exchangeable. A fixed split-conformal band, calibrated on the early (calm)
//! prefix, silently loses coverage as the stream drifts. Adaptive Conformal
//! Inference adjusts its working `αₜ` online and holds long-run coverage near the
//! nominal level. We calibrate the fixed band on the first `CALIB` steps and
//! report both methods' coverage on the remaining (drifting) steps.
//!
//! **Conditional (per-group).** Three sub-populations have very different spreads.
//! A single pooled band hits the nominal level *on average* while badly
//! over-covering the calm group and under-covering the volatile one. Mondrian
//! conformal calibrates one band per group and restores per-group coverage. We
//! calibrate on one half and report per-group coverage on a held-out half.
//!
//! No RNG: residuals come from fixed integer-hash sequences; the program is
//! byte-identical across runs.

use scirust_srcc_bench::{AdaptiveConformal, MondrianConformal, SplitConformal};

const LEVEL: f64 = 0.9;

/// A deterministic, roughly-uniform value in `[-0.5, 0.5]` from an index.
fn unit_noise(i: usize) -> f64 {
    ((i.wrapping_mul(2_654_435_761) % 1009) as f64) / 1009.0 - 0.5
}

fn temporal_section() {
    let total = 4000usize;
    let calib = 400usize;

    // Scale drifts from 1 up to 5 across the stream.
    let scale = |t: usize| 1.0 + 4.0 * (t as f64) / (total as f64);
    let score = |t: usize| (unit_noise(t) * scale(t)).abs();

    // Fixed marginal band from the calm prefix.
    let calibration: Vec<f64> = (0..calib).map(score).collect();
    let fixed = SplitConformal::fit(&calibration, LEVEL).expect("fixed band fits");

    // ACI runs online over the whole stream.
    let mut aci = AdaptiveConformal::new(LEVEL, 0.02, calib).expect("aci constructs");
    let mut fixed_covered = 0usize;
    let mut aci_covered = 0usize;
    let mut evaluated = 0usize;

    for t in 0..total
    {
        let s = score(t);
        let aci_radius = aci.radius();
        if t >= calib
        {
            evaluated += 1;
            if s <= fixed.half_width()
            {
                fixed_covered += 1;
            }
            if s <= aci_radius
            {
                aci_covered += 1;
            }
        }
        aci.observe(s).expect("finite score");
    }

    println!("# temporal drift (noise scale 1 -> 5 across {total} steps)");
    println!("#   nominal coverage        = {LEVEL:.3}");
    println!(
        "#   fixed marginal band     = {:.3}  (half-width {:.4}, calibrated on first {calib})",
        fixed_covered as f64 / evaluated as f64,
        fixed.half_width(),
    );
    println!(
        "#   adaptive (ACI)          = {:.3}  (final alpha {:.4}, final radius {:.4})",
        aci_covered as f64 / evaluated as f64,
        aci.current_alpha(),
        aci.radius(),
    );
}

fn conditional_section() {
    let per_group = 400usize; // half calibrate, half evaluate
    let scales = [1.0f64, 4.0, 12.0];

    let mut calib_keys = Vec::new();
    let mut calib_res = Vec::new();
    // Residual for group g, index i: a fixed spread scaled per group.
    let residual = |g: usize, i: usize| unit_noise(i.wrapping_mul(31).wrapping_add(g)) * scales[g];

    for (g, _) in scales.iter().enumerate()
    {
        for i in 0..(per_group / 2)
        {
            calib_keys.push(g as u64);
            calib_res.push(residual(g, i));
        }
    }

    let pooled = SplitConformal::fit(&calib_res, LEVEL).expect("pooled band fits");
    let mondrian = MondrianConformal::fit(&calib_keys, &calib_res, LEVEL).expect("mondrian fits");

    println!();
    println!("# conditional per-group coverage (held-out half; nominal {LEVEL:.3})");
    println!("#   group  scale   pooled-marginal   mondrian(per-group)");
    for (g, &scale) in scales.iter().enumerate()
    {
        let mut pooled_cov = 0usize;
        let mut mondrian_cov = 0usize;
        let held = per_group / 2;
        for i in (per_group / 2)..per_group
        {
            let r = residual(g, i);
            if pooled.covers(0.0, r)
            {
                pooled_cov += 1;
            }
            if mondrian.covers(g as u64, 0.0, r)
            {
                mondrian_cov += 1;
            }
        }
        println!(
            "#   {g:<6} {scale:<6.1} {:<17.3} {:.3}",
            pooled_cov as f64 / held as f64,
            mondrian_cov as f64 / held as f64,
        );
    }
    println!(
        "#   pooled half-width = {:.4}; per-group half-widths = [{}]",
        pooled.half_width(),
        mondrian
            .bands()
            .iter()
            .map(|b| format!("{:.4}", b.half_width))
            .collect::<Vec<_>>()
            .join(", "),
    );
}

fn main() {
    println!("# industrial_conditional_conformal — phase 4E.6");
    println!("# marginal split-conformal vs its conditional (Mondrian) and temporal (ACI) fixes.");
    println!();
    temporal_section();
    conditional_section();
}
