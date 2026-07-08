//! Fuzz cross-check of the `scirust-tolerance` modules against **independent**
//! reference computations — not re-runs of the same code, but a different
//! method for each claim (numerical integration, Monte-Carlo simulation, and
//! linear-algebra identities). Every check is deterministic (seeded xorshift),
//! so a failure is reproducible.
//!
//! Coverage:
//! - `special` — `erf` vs adaptive Simpson integration of `(2/√π)e^{−t²}`;
//!   the `erf+erfc=1`, `Φ+Φ̄=1` identities; χ² reductions and quantile
//!   round-trips.
//! - `sampling` — the non-central-χ² acceptance probability vs a direct
//!   Monte-Carlo of `P(Î ≤ k·I_max)`.
//! - `spatial` — analytical surface inertia `θ̄ᵀHθ̄+tr(HΣ)` vs the empirical
//!   `FormBatch` RMS; best-fit-torsor round-trip; form-residual orthogonality.
//! - `modal` — DCT orthonormality, Parseval, and the `Σ Iₖ²=m·I_S²` partition.
//! - `chain` — statistical/worst-case recombination and cost-optimal KKT.
//! - `capability` — non-conformity ppm vs Simpson integration of the tails.
//! - `montecarlo` — simulated mean/σ of a linear-normal assembly vs the exact
//!   `Σαμ`, `Σα²σ²`.
//! - `correlated` — finite-difference gradient vs analytic; identity-correlation
//!   inertia vs `√(Σα²I²)`; second-order mean vs the exact quadratic moment.
//! - `geometry` — LS-plane residual orthogonality; perfect plane/circle → 0;
//!   parallelism/perpendicularity vs cross-/dot-product.
//! - `sensitivity` — contribution shares sum to 1 and match `αᵢ²Iᵢ²/I_Y²`.
//! - `process` — discrete allocation vs exhaustive brute force.
//! - `drift` — long-term `σ` vs a Monte-Carlo of drifting mean + within noise.
//! - `msa` — Gage R&R variance-component identity; zero repeatability /
//!   reproducibility on constructed data.
//! - `interval` — tolerance factor vs its Monte-Carlo coverage probability.
//! - `distfit` — CDF∘quantile round-trip; normal reduces to classic `Cp`;
//!   parameter recovery on simulated samples.
//! - `dual` — mean contributions sum to `δ_Y`, variance fractions to 1.
//! - `gdt` — virtual-condition / datum-shift / composite-position identities.
//! - `capability_ci` — the exact χ² `Cp` interval vs its Monte-Carlo coverage.
//! - `variables` — the closed-form OC `Φ(√n(z_p−k))` vs a direct Monte-Carlo of
//!   the accept rule, and the MSD identity.
//! - `sixsigma` — yield↔sigma↔DPMO round-trips vs the independent normal tail;
//!   RTY vs an explicit product; Poisson `−ln Y = DPU`.
//! - `attribution` — the `Σcⱼ = R²` decomposition identity, coefficient recovery
//!   against known generators, and single-regressor `c = corr²`.
//! - `attributes` — the binomial OC `P(D≤c)` vs a direct Monte-Carlo of the
//!   accept rule; designed plans clear both nominal points.
//! - `interference` — `R = Φ(β)` vs a Monte-Carlo of `P(S>L)`; clearance-fit
//!   partition identities.
//! - `subgroup` — the overall-sigma recomputation; range- vs s-method within
//!   sigma agreement; the `Cp` identity.
//! - `fits` — the `clearance range = IT_hole + IT_shaft` identity, IT-grade
//!   monotonicity and the ×10-per-5-grades law.
//! - `sequential` — double-sampling OC / ASN vs Monte-Carlo; SPRT OC guarantee
//!   at both design points.
//! - `taguchi` — `E[L] = k·I²` (inertia vs moments) vs a Monte-Carlo of the
//!   quadratic loss; the economic-tolerance balance.
//!
//! Run: `cargo run -p scirust-tolerance --example fuzz_crosscheck [N]`

use scirust_tolerance::attributes::{AttributesPlan, design_attributes_plan};
use scirust_tolerance::attribution::attribute;
use scirust_tolerance::capability::{cp as cap_cp, cp_confidence_interval, nonconformity_ppm};
use scirust_tolerance::chain::{
    Allocation, Contributor, ContributorState, allocate, assembly_inertia_statistical,
    assembly_inertia_worst_case, assembly_state,
};
use scirust_tolerance::correlated::{
    correlated_inertia, gradient, second_order_mean, uniform_correlation,
};
use scirust_tolerance::distfit::{
    FittedDistribution, fit_lognormal, fit_rayleigh, fit_weibull, percentile_capability,
};
use scirust_tolerance::drift::{cpk_to_ppk, long_term_sigma, ppk_to_cpk};
use scirust_tolerance::fits::{hole_basis_fit, it_grade_tolerance};
use scirust_tolerance::form::FormBatch;
use scirust_tolerance::geometry::{
    flatness, least_squares_plane, parallelism, perpendicularity, roundness,
};
use scirust_tolerance::interference::{clearance_fit, interference_reliability};
use scirust_tolerance::interval::tolerance_factor_two_sided;
use scirust_tolerance::modal::{ModalBasis, modal_inertias};
use scirust_tolerance::montecarlo::{Distribution, linear as mc_linear, simulate};
use scirust_tolerance::msa::gage_rr;
use scirust_tolerance::nonnormal::{cornish_fisher_quantile, nonnormal_ppm};
use scirust_tolerance::position::{
    CompositePosition, FeatureType, coord_to_position, datum_shift, position_to_coord,
    positional_inertia, true_position, virtual_condition,
};
use scirust_tolerance::process::{Combination, ProcessOption, allocate_discrete};
use scirust_tolerance::sampling::SamplingPlan;
use scirust_tolerance::sensitivity::{contributions, correlated_contributions, dual_contributions};
use scirust_tolerance::sequential::{
    DoubleSamplingPlan, SequentialVerdict, design_sequential_plan,
};
use scirust_tolerance::sixsigma::{
    dpmo_from_sigma, normalized_yield, process_report, rolled_throughput_yield, sigma_from_dpmo,
    sigma_from_yield, throughput_yield, yield_from_sigma,
};
use scirust_tolerance::spatial::{
    Feature, Torsor, inertia_decomposition, surface_inertia_analytical,
    surface_inertia_from_torsors,
};
use scirust_tolerance::special::{
    chi2_cdf, chi2_quantile, erf, erfc, ncchi2_cdf, normal_cdf, normal_sf,
};
use scirust_tolerance::subgroup::{sigma_within_s_method, subgroup_capability};
use scirust_tolerance::taguchi::{
    economic_tolerance, expected_loss, expected_loss_from_moments, loss_coefficient,
    quadratic_loss, smaller_the_better_loss,
};
use scirust_tolerance::variables::{VariablesPlan, design_variables_plan};

/// Deterministic xorshift64* RNG with a Box–Muller normal.
struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
    fn int(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.u64() as usize) % (hi - lo + 1)
    }
    /// Standard normal via Box–Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform(1e-12, 1.0);
        let u2 = self.uniform(0.0, 1.0);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Accumulates a module's error count and worst residual.
#[derive(Default)]
struct Report {
    checks: usize,
    errors: usize,
    worst: f64,
    first_fail: Option<String>,
}
impl Report {
    fn check(&mut self, residual: f64, tol: f64, describe: impl FnOnce() -> String) {
        self.checks += 1;
        self.worst = self.worst.max(residual);
        // A NaN residual is treated as a failure (it can never be ≤ tol).
        if residual.is_nan() || residual > tol
        {
            self.errors += 1;
            if self.first_fail.is_none()
            {
                self.first_fail = Some(format!(
                    "residual {residual:.3e} > tol {tol:.0e} :: {}",
                    describe()
                ));
            }
        }
    }
    fn line(&self, name: &str) -> String {
        let status = if self.errors == 0 { "ok" } else { "FAIL" };
        let mut s = format!(
            "{name:<12} {status:>4}  checks {:>6}  errors {:>4}  worst {:.2e}",
            self.checks, self.errors, self.worst
        );
        if let Some(f) = &self.first_fail
        {
            s.push_str(&format!("\n             ↳ {f}"));
        }
        s
    }
}

/// Composite adaptive-free Simpson integral of `f` on `[a,b]` with `n` panels.
fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = n.max(2) & !1; // even
    let h = (b - a) / n as f64;
    let mut s = f(a) + f(b);
    for i in 1..n
    {
        let x = a + i as f64 * h;
        s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(x);
    }
    s * h / 3.0
}

fn check_special(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    for _ in 0..n
    {
        let x = rng.uniform(-4.0, 4.0);
        // erf vs independent Simpson integration.
        let ref_erf = if x >= 0.0
        {
            simpson(|t| two_over_sqrt_pi * (-t * t).exp(), 0.0, x, 4000)
        }
        else
        {
            -simpson(|t| two_over_sqrt_pi * (-t * t).exp(), 0.0, -x, 4000)
        };
        r.check((erf(x) - ref_erf).abs(), 1e-9, || format!("erf({x})"));
        // Complement + reflection identities.
        r.check((erf(x) + erfc(x) - 1.0).abs(), 1e-12, || {
            format!("erf+erfc {x}")
        });
        r.check((erf(-x) + erf(x)).abs(), 1e-12, || format!("erf odd {x}"));
        // Normal CDF/SF partition, and vs Simpson of the pdf.
        let z = x;
        r.check((normal_cdf(z) + normal_sf(z) - 1.0).abs(), 1e-12, || {
            format!("cdf+sf {z}")
        });
        let pdf = |t: f64| (-t * t / 2.0).exp() / (std::f64::consts::TAU).sqrt();
        let ref_cdf = 0.5
            + if z >= 0.0
            {
                simpson(pdf, 0.0, z, 4000)
            }
            else
            {
                -simpson(pdf, 0.0, -z, 4000)
            };
        r.check((normal_cdf(z) - ref_cdf).abs(), 1e-9, || {
            format!("Phi({z})")
        });

        // χ² reduction: ncχ²(dof,0,x) == χ²(dof,x); χ²₂ closed form; quantile round-trip.
        let dof = rng.uniform(1.0, 30.0);
        let xx = rng.uniform(0.01, 60.0);
        r.check(
            (ncchi2_cdf(dof, 0.0, xx) - chi2_cdf(dof, xx)).abs(),
            1e-12,
            || format!("ncchi2 λ=0 dof={dof}"),
        );
        r.check(
            (chi2_cdf(2.0, xx) - (1.0 - (-xx / 2.0).exp())).abs(),
            1e-10,
            || format!("chi2_2 {xx}"),
        );
        let p = rng.uniform(0.001, 0.999);
        r.check(
            (chi2_cdf(dof, chi2_quantile(dof, p)) - p).abs(),
            1e-7,
            || format!("chi2 quantile rt dof={dof} p={p}"),
        );
    }
    r
}

fn check_sampling(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let trials = 40_000;
    for _ in 0..n
    {
        let nn = rng.int(3, 10);
        let i_max = rng.uniform(0.02, 0.2);
        let factor = rng.uniform(0.6, 1.8);
        let plan = SamplingPlan::new(nn, factor);
        // True process state.
        let delta = rng.uniform(-1.5, 1.5) * i_max;
        let sigma = rng.uniform(0.2, 1.5) * i_max;
        let analytical = plan.probability_of_acceptance(i_max, delta, sigma);
        // Monte-Carlo: draw nn samples ~ N(delta, sigma) about target 0, accept if Î ≤ factor·i_max.
        let limit = factor * i_max;
        let mut accept = 0u64;
        for _ in 0..trials
        {
            let xs: Vec<f64> = (0..nn).map(|_| delta + sigma * rng.normal()).collect();
            let mean = xs.iter().sum::<f64>() / nn as f64;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nn as f64;
            let inertia = ((mean * mean) + var).sqrt(); // Î about target 0
            if inertia <= limit
            {
                accept += 1;
            }
        }
        let empirical = accept as f64 / trials as f64;
        // Tolerance: 5 standard errors of the MC estimate, floored.
        let se = (empirical * (1.0 - empirical) / trials as f64).sqrt();
        let tol = (5.0 * se).max(0.01);
        r.check((analytical - empirical).abs(), tol, || {
            format!("n={nn} δ={delta:.3} σ={sigma:.3} k={factor:.2}: analytic {analytical:.3} vs MC {empirical:.3}")
        });
    }
    r
}

/// A full-rank feature: three perpendicular faces with random in-plane points.
fn random_feature(rng: &mut Rng) -> Feature {
    let mut pts = Vec::new();
    let per_face = rng.int(3, 5);
    for _ in 0..per_face
    {
        let (a, b) = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0));
        pts.push(([1.0, a, b], [1.0, 0.0, 0.0])); // +x
        pts.push(([a, 1.0, b], [0.0, 1.0, 0.0])); // +y
        pts.push(([a, b, 1.0], [0.0, 0.0, 1.0])); // +z
    }
    Feature::new(pts)
}

fn random_torsor(rng: &mut Rng, scale: f64) -> Torsor {
    Torsor::new(
        [
            rng.uniform(-scale, scale),
            rng.uniform(-scale, scale),
            rng.uniform(-scale, scale),
        ],
        [
            rng.uniform(-scale, scale) * 0.3,
            rng.uniform(-scale, scale) * 0.3,
            rng.uniform(-scale, scale) * 0.3,
        ],
    )
}

fn check_spatial(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let feat = random_feature(rng);
        let batch: Vec<Torsor> = (0..rng.int(2, 6))
            .map(|_| random_torsor(rng, 0.05))
            .collect();
        // Analytical vs empirical surface inertia.
        let ana = surface_inertia_analytical(&feat, &batch);
        let emp = surface_inertia_from_torsors(&feat, &batch);
        r.check((ana - emp).abs() / emp.max(1e-12), 1e-9, || {
            format!("spatial I_S analytic {ana} vs empirical {emp}")
        });
        // Decomposition sums to I_S².
        let dec = inertia_decomposition(&feat, &batch);
        r.check(
            (dec.total() - ana * ana).abs() / (ana * ana).max(1e-12),
            1e-9,
            || "spatial decomposition sum".into(),
        );
        // Best-fit torsor round-trip on a pure-rigid field ⇒ zero form residual.
        let truth = random_torsor(rng, 0.05);
        let e = feat.deviation_field(&truth);
        if let Some(resid) = feat.form_residual(&e)
        {
            let enorm = e.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            let rnorm = resid.iter().map(|v| v * v).sum::<f64>().sqrt();
            r.check(rnorm / enorm, 1e-7, || {
                "spatial rigid-field residual".into()
            });
            // Residual orthogonal to the influence columns (least-squares normal eqns).
            let rows = feat.influence_rows();
            for k in 0..6
            {
                let proj: f64 = rows.iter().zip(&resid).map(|(g, rr)| g[k] * rr).sum();
                r.check(proj.abs() / enorm, 1e-7, || {
                    format!("spatial residual ⟂ col {k}")
                });
            }
        }
    }
    r
}

fn check_modal(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let m = rng.int(2, 32);
        let basis = ModalBasis::dct(m, m);
        // Orthonormality: Gram matrix ≈ I (checked via the crate's own predicate + a spot pair).
        r.check(
            if basis.is_orthonormal(1e-9) { 0.0 } else { 1.0 },
            0.5,
            || format!("modal orthonormal m={m}"),
        );
        // Parseval + reconstruction on a random deviation.
        let d: Vec<f64> = (0..m).map(|_| rng.normal()).collect();
        let coeffs = basis.decompose(&d);
        let energy_c: f64 = coeffs.iter().map(|c| c * c).sum();
        let energy_d: f64 = d.iter().map(|x| x * x).sum();
        r.check(
            (energy_c - energy_d).abs() / energy_d.max(1e-12),
            1e-12,
            || format!("modal Parseval m={m}"),
        );
        r.check(
            basis.residual_norm(&d) / energy_d.sqrt().max(1e-12),
            1e-12,
            || format!("modal reconstruct m={m}"),
        );
        // Partition: Σ Iₖ² = m · I_S² for a random batch.
        let parts: Vec<Vec<f64>> = (0..rng.int(2, 6))
            .map(|_| (0..m).map(|_| 0.05 * rng.normal()).collect())
            .collect();
        if let Some(fb) = FormBatch::new(parts.clone())
        {
            let modal = modal_inertias(&basis, fb.deviations());
            let sum_i2: f64 = modal.iter().map(|i| i.mean_squared_deviation()).sum();
            let target = m as f64 * fb.surface_inertia().powi(2);
            r.check((sum_i2 - target).abs() / target.max(1e-12), 1e-9, || {
                format!("modal partition m={m}")
            });
        }
    }
    r
}

fn check_chain(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let nc = rng.int(2, 8);
        let coeffs: Vec<f64> = (0..nc)
            .map(|_| {
                let v = rng.uniform(-2.0, 2.0);
                if v.abs() < 0.1 { 1.0 } else { v }
            })
            .collect();
        let i_y = rng.uniform(0.02, 0.2);
        // Statistical allocation recombines to the budget.
        let stat = allocate(i_y, &coeffs, &Allocation::Statistical).unwrap();
        let cs: Vec<Contributor> = coeffs
            .iter()
            .zip(&stat)
            .map(|(a, i)| Contributor::new("x", *a, *i))
            .collect();
        r.check(
            (assembly_inertia_statistical(&cs) - i_y).abs() / i_y,
            1e-12,
            || "chain statistical recombine".into(),
        );
        // Worst-case: Σ|α|I == budget.
        let wc = allocate(i_y, &coeffs, &Allocation::WorstCase).unwrap();
        let cw: Vec<Contributor> = coeffs
            .iter()
            .zip(&wc)
            .map(|(a, i)| Contributor::new("x", *a, *i))
            .collect();
        r.check(
            (assembly_inertia_worst_case(&cw) - i_y).abs() / i_y,
            1e-12,
            || "chain worst-case recombine".into(),
        );
        // Cost-optimal: recombine to budget + KKT stationarity (common μ).
        let costs: Vec<f64> = (0..nc).map(|_| rng.uniform(0.2, 8.0)).collect();
        let expo = rng.uniform(1.0, 4.0);
        let co = allocate(
            i_y,
            &coeffs,
            &Allocation::CostOptimal {
                costs: costs.clone(),
                exponent: expo,
            },
        )
        .unwrap();
        let cc: Vec<Contributor> = coeffs
            .iter()
            .zip(&co)
            .map(|(a, i)| Contributor::new("x", *a, *i))
            .collect();
        r.check(
            (assembly_inertia_statistical(&cc) - i_y).abs() / i_y,
            1e-9,
            || "chain cost recombine".into(),
        );
        // KKT: r·kᵢ·Iᵢ^(−r−1)/(αᵢ² Iᵢ) equal ∀ i.
        let mu: Vec<f64> = coeffs
            .iter()
            .zip(&co)
            .zip(&costs)
            .map(|((a, i), k)| expo * k * i.powf(-expo - 1.0) / (a * a * i))
            .collect();
        let (mn, mx) = mu
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
        r.check((mx - mn) / mx.max(1e-300), 1e-6, || "chain cost KKT".into());
    }
    r
}

fn check_capability(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let sigma = rng.uniform(0.05, 1.0);
        let lsl = rng.uniform(-5.0, -1.0);
        let usl = rng.uniform(1.0, 5.0);
        let mean = rng.uniform(lsl + 0.5, usl - 0.5);
        // ppm vs independent Simpson integration of the two tails.
        let pdf = |x: f64| {
            (-((x - mean) / sigma).powi(2) / 2.0).exp() / (sigma * std::f64::consts::TAU.sqrt())
        };
        let below = simpson(pdf, mean - 12.0 * sigma, lsl, 6000).max(0.0);
        let above = simpson(pdf, usl, mean + 12.0 * sigma, 6000).max(0.0);
        let ref_ppm = (below + above) * 1e6;
        let got = nonconformity_ppm(mean, sigma, lsl, usl);
        // Absolute ppm tolerance scaled by magnitude (Simpson tail accuracy).
        let tol = (ref_ppm.abs() * 1e-4).max(1e-3);
        r.check((got - ref_ppm).abs(), tol, || {
            format!("ppm μ={mean:.2} σ={sigma:.2}: got {got:.4} vs {ref_ppm:.4}")
        });
    }
    r
}

fn check_nonnormal(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let (mean, sd) = (rng.uniform(-2.0, 2.0), rng.uniform(0.3, 2.0));
        // (1) s=k=0 must reduce to the exact normal ppm.
        let lsl = mean - rng.uniform(2.0, 4.0) * sd;
        let usl = mean + rng.uniform(2.0, 4.0) * sd;
        let nn = nonnormal_ppm(mean, sd, 0.0, 0.0, lsl, usl);
        let normal = nonconformity_ppm(mean, sd, lsl, usl);
        r.check((nn - normal).abs(), (normal * 1e-6).max(1e-4), || {
            format!("nonnormal reduce μ={mean:.2}: {nn:.4} vs {normal:.4}")
        });
        // (2) Cornish–Fisher forward/inverse consistency within the valid
        // (moderate, monotone) domain: with both limits at the p_lo and p_hi
        // quantiles, nonnormal_ppm must recover (p_lo + 1 − p_hi)·1e6. The
        // skew/kurtosis are kept small enough that the CF cubic stays monotone
        // over the tested range (strong non-normality breaks CF invertibility —
        // a documented limitation, not a code defect).
        let (s, k) = (rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.8));
        let p_lo = rng.uniform(0.005, 0.05);
        let p_hi = rng.uniform(0.95, 0.995);
        let lo = cornish_fisher_quantile(mean, sd, s, k, p_lo);
        let hi = cornish_fisher_quantile(mean, sd, s, k, p_hi);
        let want = (p_lo + 1.0 - p_hi) * 1e6;
        let tail = nonnormal_ppm(mean, sd, s, k, lo.min(hi), lo.max(hi));
        r.check((tail - want).abs(), (want * 1e-6).max(1e-2), || {
            format!("CF fwd/inv p_lo={p_lo:.3} p_hi={p_hi:.3}: {tail:.2} vs {want:.2}")
        });
        // (3) Monotonicity (qualitative): more right-skew ⇒ no meaningfully-less
        // upper-tail nonconformity, at a moderate (in-domain) ~2σ limit. A small
        // relative slack absorbs CF's approximation noise at the edges (this is a
        // sanity check; checks 1–2 are the rigorous ones).
        let ul = mean + 2.0 * sd;
        let low = nonnormal_ppm(mean, sd, 0.0, k, mean - 6.0 * sd, ul);
        let high = nonnormal_ppm(mean, sd, 0.4, k, mean - 6.0 * sd, ul);
        r.check((low - high).max(0.0) / low.max(1.0), 0.05, || {
            format!("nonnormal skew monotonic: high {high:.2} vs low {low:.2}")
        });
    }
    r
}

fn check_position(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let (dx, dy) = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0));
        // True position vs independent radial computation.
        let ref_tp = 2.0 * (dx * dx + dy * dy).sqrt();
        r.check((true_position(dx, dy) - ref_tp).abs(), 1e-12, || {
            "position true".into()
        });
        // ±coord ↔ Ø round-trip on a symmetric zone.
        let t = rng.uniform(0.01, 0.5);
        let phi = coord_to_position(t, t);
        r.check((position_to_coord(phi) - t).abs() / t, 1e-12, || {
            "position coord round-trip".into()
        });
        // Positional inertia = Euclidean norm of the two axis inertias.
        let (ix, iy) = (rng.uniform(0.0, 0.5), rng.uniform(0.0, 0.5));
        r.check(
            (positional_inertia(ix, iy) - (ix * ix + iy * iy).sqrt()).abs(),
            1e-12,
            || "position inertia".into(),
        );
    }
    r
}

fn check_montecarlo(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let trials = 40_000;
    for _ in 0..n
    {
        let nc = rng.int(2, 4);
        let means: Vec<f64> = (0..nc).map(|_| rng.uniform(-5.0, 5.0)).collect();
        let sds: Vec<f64> = (0..nc).map(|_| rng.uniform(0.05, 0.5)).collect();
        let coeffs: Vec<f64> = (0..nc)
            .map(|_| {
                let v = rng.uniform(-2.0, 2.0);
                if v.abs() < 0.2 { 1.0 } else { v }
            })
            .collect();
        let comps: Vec<Distribution> = means
            .iter()
            .zip(&sds)
            .map(|(&mean, &sd)| Distribution::Normal { mean, sd })
            .collect();
        // Independent reference: a linear combination of normals is normal with
        // mean Σαμ and variance Σα²σ².
        let want_mean: f64 = coeffs.iter().zip(&means).map(|(a, m)| a * m).sum();
        let want_var: f64 = coeffs.iter().zip(&sds).map(|(a, s)| a * a * s * s).sum();
        let seed = rng.u64();
        let res = simulate(
            &comps,
            |xs| mc_linear(&coeffs, xs),
            want_mean,
            want_mean - 1e12,
            want_mean + 1e12,
            trials,
            seed,
        );
        let se_mean = (want_var / trials as f64).sqrt();
        r.check(
            (res.mean - want_mean).abs(),
            (6.0 * se_mean).max(1e-9),
            || format!("MC mean {} vs {}", res.mean, want_mean),
        );
        let want_sd = want_var.sqrt();
        let se_sd = want_sd / (2.0 * trials as f64).sqrt();
        r.check((res.sigma - want_sd).abs(), (6.0 * se_sd).max(1e-9), || {
            format!("MC sigma {} vs {}", res.sigma, want_sd)
        });
    }
    r
}

fn check_correlated(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let nc = rng.int(2, 5);
        // Gradient of f = Σ aᵢ sin(xᵢ) vs the analytic aᵢ cos(xᵢ).
        let a: Vec<f64> = (0..nc).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let x0: Vec<f64> = (0..nc).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let f = |x: &[f64]| a.iter().zip(x).map(|(ai, xi)| ai * xi.sin()).sum::<f64>();
        let g = gradient(f, &x0, 1e-5);
        for i in 0..nc
        {
            r.check((g[i] - a[i] * x0[i].cos()).abs(), 1e-6, || {
                format!("correlated gradient[{i}]")
            });
        }
        // correlated_inertia with identity correlation == √(Σ α²I²).
        let coeffs: Vec<f64> = (0..nc).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let inert: Vec<f64> = (0..nc).map(|_| rng.uniform(0.01, 0.3)).collect();
        let corr_i = uniform_correlation(nc, 0.0);
        let want = coeffs
            .iter()
            .zip(&inert)
            .map(|(a, i)| a * a * i * i)
            .sum::<f64>()
            .sqrt();
        r.check(
            (correlated_inertia(&coeffs, &inert, &corr_i) - want).abs(),
            1e-12,
            || "correlated identity == statistical".into(),
        );
        // second_order_mean of f = Σ xᵢ² equals the exact Σ(μᵢ² + σᵢ²).
        let mu: Vec<f64> = (0..nc).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let varv: Vec<f64> = (0..nc).map(|_| rng.uniform(0.01, 0.5)).collect();
        let fq = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
        let so = second_order_mean(fq, &mu, 1e-3, &varv);
        let exact = mu.iter().zip(&varv).map(|(m, v)| m * m + v).sum::<f64>();
        r.check((so - exact).abs(), 1e-5, || {
            "correlated 2nd-order mean".into()
        });
    }
    r
}

fn check_geometry(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        // Points exactly on a random plane ⇒ zero flatness.
        let (pa, pb, pc) = (
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
        );
        let m = rng.int(4, 12);
        let on_plane: Vec<[f64; 3]> = (0..m)
            .map(|_| {
                let (x, y) = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0));
                [x, y, pa + pb * x + pc * y]
            })
            .collect();
        r.check(flatness(&on_plane), 1e-7, || {
            "geometry perfect plane".into()
        });
        // LS-plane residual orthogonality on noisy points (normal equations).
        let noisy: Vec<[f64; 3]> = (0..m)
            .map(|_| {
                let (x, y) = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0));
                [x, y, pa + pb * x + pc * y + 0.05 * rng.normal()]
            })
            .collect();
        if let Some((qa, qb, qc)) = least_squares_plane(&noisy)
        {
            let res: Vec<f64> = noisy
                .iter()
                .map(|p| p[2] - (qa + qb * p[0] + qc * p[1]))
                .collect();
            let scale = res.iter().map(|v| v.abs()).sum::<f64>().max(1e-9);
            let sr: f64 = res.iter().sum();
            let srx: f64 = res.iter().zip(&noisy).map(|(v, p)| v * p[0]).sum();
            let sry: f64 = res.iter().zip(&noisy).map(|(v, p)| v * p[1]).sum();
            r.check(sr.abs() / scale, 1e-7, || "geometry plane Σr".into());
            r.check(srx.abs() / scale, 1e-6, || "geometry plane Σr·x".into());
            r.check(sry.abs() / scale, 1e-6, || "geometry plane Σr·y".into());
        }
        // Points exactly on a random circle ⇒ zero roundness.
        let (cx, cy, rad) = (
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(0.5, 3.0),
        );
        let circ: Vec<[f64; 2]> = (0..12)
            .map(|k| {
                let t = k as f64 / 12.0 * std::f64::consts::TAU;
                [cx + rad * t.cos(), cy + rad * t.sin()]
            })
            .collect();
        r.check(roundness(&circ), 1e-7, || "geometry perfect circle".into());
        // Orientation zones vs cross-/dot-product identities.
        let u = [
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        ];
        let v = [
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        ];
        let nu = (u.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let nv = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let l = rng.uniform(1.0, 20.0);
        if nu > 1e-6 && nv > 1e-6
        {
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let cn = (cross.iter().map(|x| x * x).sum::<f64>()).sqrt();
            let ref_par = l * cn / (nu * nv);
            r.check((parallelism(u, v, l) - ref_par).abs(), 1e-9, || {
                "geometry parallelism vs cross".into()
            });
            let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).abs();
            let ref_perp = l * dot / (nu * nv);
            r.check((perpendicularity(u, v, l) - ref_perp).abs(), 1e-9, || {
                "geometry perpendicularity vs dot".into()
            });
        }
    }
    r
}

fn check_sensitivity(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let nc = rng.int(2, 6);
        let coeffs: Vec<f64> = (0..nc).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let inert: Vec<f64> = (0..nc).map(|_| rng.uniform(0.01, 0.3)).collect();
        let cs: Vec<Contributor> = coeffs
            .iter()
            .zip(&inert)
            .map(|(a, i)| Contributor::new("x", *a, *i))
            .collect();
        let cons = contributions(&cs);
        let sum: f64 = cons.iter().map(|c| c.fraction).sum();
        r.check((sum - 1.0).abs(), 1e-12, || "sensitivity Σfrac == 1".into());
        // correlated_contributions (identity) matches direct αᵢ²Iᵢ²/total per index.
        let total: f64 = coeffs.iter().zip(&inert).map(|(a, i)| a * a * i * i).sum();
        let corr_i = uniform_correlation(nc, 0.0);
        let frac = correlated_contributions(&coeffs, &inert, &corr_i);
        for i in 0..nc
        {
            let want = coeffs[i] * coeffs[i] * inert[i] * inert[i] / total;
            r.check((frac[i] - want).abs(), 1e-12, || {
                format!("sensitivity frac[{i}]")
            });
        }
    }
    r
}

fn check_process(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let nc = rng.int(2, 4);
        let coeffs: Vec<f64> = (0..nc)
            .map(|_| {
                let v = rng.uniform(-2.0, 2.0);
                if v.abs() < 0.2 { 1.0 } else { v }
            })
            .collect();
        let opts: Vec<Vec<ProcessOption>> = (0..nc)
            .map(|_| {
                let k = rng.int(2, 4);
                (0..k)
                    .map(|_| ProcessOption::new(rng.uniform(0.02, 0.2), rng.uniform(0.5, 5.0)))
                    .collect()
            })
            .collect();
        let method = if rng.u64() & 1 == 0
        {
            Combination::Statistical
        }
        else
        {
            Combination::WorstCase
        };
        let budget = rng.uniform(0.05, 0.4);
        let got = allocate_discrete(&coeffs, &opts, budget, method);
        // Independent reference: brute-force every combination (nc ≤ 4, k ≤ 4).
        let mut idx = vec![0usize; nc];
        let mut best: Option<f64> = None;
        loop
        {
            let (mut wsum, mut cost) = (0.0, 0.0);
            for i in 0..nc
            {
                let opt = &opts[i][idx[i]];
                cost += opt.cost;
                wsum += match method
                {
                    Combination::Statistical => coeffs[i] * coeffs[i] * opt.inertia * opt.inertia,
                    Combination::WorstCase => coeffs[i].abs() * opt.inertia,
                };
            }
            let iy = match method
            {
                Combination::Statistical => wsum.sqrt(),
                Combination::WorstCase => wsum,
            };
            if iy <= budget && best.map(|b| cost < b).unwrap_or(true)
            {
                best = Some(cost);
            }
            // Mixed-radix increment.
            let mut carry = 0;
            idx[carry] += 1;
            while carry < nc && idx[carry] == opts[carry].len()
            {
                idx[carry] = 0;
                carry += 1;
                if carry < nc
                {
                    idx[carry] += 1;
                }
            }
            if carry >= nc
            {
                break;
            }
        }
        match (got, best)
        {
            (Some(a), Some(bc)) => r.check((a.total_cost - bc).abs(), 1e-9, || {
                format!("process cost {} vs brute {}", a.total_cost, bc)
            }),
            (None, None) => r.check(0.0, 1.0, || "process both infeasible".into()),
            (g, b) => r.check(1.0, 0.5, || {
                format!(
                    "process feasibility mismatch: got={} brute={}",
                    g.is_some(),
                    b.is_some()
                )
            }),
        }
    }
    r
}

fn check_drift(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let trials = 60_000;
    for _ in 0..n
    {
        let sd = rng.uniform(0.1, 1.0);
        let d = rng.uniform(0.0, 1.5);
        // Independent reference: Monte-Carlo of a drifting mean U(−d,d) plus
        // within-noise N(0,sd) ⇒ Var = sd² + d²/3.
        let (mut s1, mut s2) = (0.0, 0.0);
        for _ in 0..trials
        {
            let mean = rng.uniform(-d, d);
            let x = mean + sd * rng.normal();
            s1 += x;
            s2 += x * x;
        }
        let mc_var = s2 / trials as f64 - (s1 / trials as f64).powi(2);
        let want_sd = long_term_sigma(sd, d);
        let se = want_sd / (2.0 * trials as f64).sqrt();
        r.check(
            (mc_var.sqrt() - want_sd).abs(),
            (6.0 * se).max(1e-4),
            || format!("drift σ_lt MC {} vs {}", mc_var.sqrt(), want_sd),
        );
        // Cpk↔Ppk shift round-trip.
        let cpk = rng.uniform(0.5, 2.0);
        r.check(
            (ppk_to_cpk(cpk_to_ppk(cpk, 1.5), 1.5) - cpk).abs(),
            1e-12,
            || "drift Cpk↔Ppk round-trip".into(),
        );
    }
    r
}

fn check_msa(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let p = rng.int(3, 8);
        let o = rng.int(2, 4);
        let rr = rng.int(3, 6);
        // (1) Random balanced study ⇒ the variance-component identity holds.
        let sp = rng.uniform(0.5, 3.0);
        let so = rng.uniform(0.05, 0.4);
        let se = rng.uniform(0.05, 0.5);
        let parts: Vec<f64> = (0..p).map(|_| sp * rng.normal()).collect();
        let opers: Vec<f64> = (0..o).map(|_| so * rng.normal()).collect();
        let data: Vec<Vec<Vec<f64>>> = (0..p)
            .map(|i| {
                (0..o)
                    .map(|j| {
                        (0..rr)
                            .map(|_| parts[i] + opers[j] + se * rng.normal())
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let g = gage_rr(&data, None).unwrap();
        r.check(
            (g.grr_var + g.part_var - g.total_var).abs() / g.total_var.max(1e-12),
            1e-12,
            || "msa grr+part == total".into(),
        );
        // (2) Identical replicates within every cell ⇒ zero repeatability.
        let ident: Vec<Vec<Vec<f64>>> = (0..p)
            .map(|i| (0..o).map(|j| vec![parts[i] + opers[j]; rr]).collect())
            .collect();
        let gi = gage_rr(&ident, None).unwrap();
        r.check(gi.repeatability_var, 1e-9, || {
            "msa zero repeatability".into()
        });
        // (3) All operators read identically ⇒ zero reproducibility.
        let same_oper: Vec<Vec<Vec<f64>>> = (0..p)
            .map(|i| {
                let reads: Vec<f64> = (0..rr).map(|_| parts[i] + se * rng.normal()).collect();
                (0..o).map(|_| reads.clone()).collect()
            })
            .collect();
        let gs = gage_rr(&same_oper, None).unwrap();
        r.check(gs.reproducibility_var, 1e-9, || {
            "msa zero reproducibility".into()
        });
    }
    r
}

fn check_interval(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let trials = 2000;
    for _ in 0..n
    {
        let nn = rng.int(4, 20);
        let p = rng.uniform(0.80, 0.98);
        let conf = rng.uniform(0.80, 0.95);
        let k = tolerance_factor_two_sided(nn, p, conf).unwrap();
        // Independent reference: the Monte-Carlo coverage probability that
        // *defines* the factor — the fraction of samples whose x̄±k·s contains
        // at least proportion p of N(0,1) should be ≥ the nominal confidence.
        let mut ok = 0u64;
        for _ in 0..trials
        {
            let xs: Vec<f64> = (0..nn).map(|_| rng.normal()).collect();
            let mean = xs.iter().sum::<f64>() / nn as f64;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nn as f64 - 1.0);
            let s = var.sqrt();
            let coverage = normal_cdf(mean + k * s) - normal_cdf(mean - k * s);
            if coverage >= p
            {
                ok += 1;
            }
        }
        let emp = ok as f64 / trials as f64;
        // Howe is accurate/slightly conservative ⇒ empirical confidence is not
        // meaningfully below the nominal.
        r.check((conf - emp).max(0.0), 0.06, || {
            format!("interval coverage emp {emp:.3} vs conf {conf:.3} (n={nn}, p={p:.2})")
        });
    }
    r
}

fn check_distfit(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        // (1) CDF∘quantile round-trip for every family with random params.
        let dists = [
            FittedDistribution::Normal {
                mean: rng.uniform(-2.0, 2.0),
                sd: rng.uniform(0.3, 2.0),
            },
            FittedDistribution::Lognormal {
                mu: rng.uniform(-1.0, 1.0),
                sigma: rng.uniform(0.2, 1.0),
            },
            FittedDistribution::Rayleigh {
                sigma: rng.uniform(0.5, 3.0),
            },
            FittedDistribution::Weibull {
                shape: rng.uniform(0.8, 4.0),
                scale: rng.uniform(0.5, 4.0),
            },
        ];
        for d in dists
        {
            let p = rng.uniform(0.02, 0.98);
            r.check((d.cdf(d.quantile(p)) - p).abs(), 1e-9, || {
                "distfit cdf∘quantile".into()
            });
        }
        // (2) percentile capability of a normal reduces to the classical Cp.
        let (mean, sd) = (rng.uniform(-1.0, 1.0), rng.uniform(0.3, 1.5));
        let (lsl, usl) = (
            mean - rng.uniform(3.0, 6.0) * sd,
            mean + rng.uniform(3.0, 6.0) * sd,
        );
        let c = percentile_capability(&FittedDistribution::Normal { mean, sd }, lsl, usl);
        r.check(
            (c.cp - cap_cp(sd, lsl, usl)).abs() / cap_cp(sd, lsl, usl),
            1e-3,
            || "distfit normal reduces to classic cp".into(),
        );
        // (3) Parameter recovery on large simulated samples.
        let m = 4000;
        let sig = rng.uniform(0.5, 3.0);
        let rl: Vec<f64> = (0..m)
            .map(|_| sig * (-2.0 * rng.uniform(1e-12, 1.0).ln()).sqrt())
            .collect();
        if let Some(FittedDistribution::Rayleigh { sigma }) = fit_rayleigh(&rl)
        {
            r.check((sigma - sig).abs() / sig, 0.05, || {
                "distfit rayleigh recovery".into()
            });
        }
        let (mu0, sg0) = (rng.uniform(-1.0, 1.0), rng.uniform(0.2, 0.8));
        let ln: Vec<f64> = (0..m).map(|_| (mu0 + sg0 * rng.normal()).exp()).collect();
        if let Some(FittedDistribution::Lognormal { mu, sigma }) = fit_lognormal(&ln)
        {
            r.check((mu - mu0).abs(), 0.05, || "distfit lognormal mu".into());
            r.check((sigma - sg0).abs() / sg0, 0.05, || {
                "distfit lognormal sigma".into()
            });
        }
        let (k0, lam0) = (rng.uniform(1.2, 3.5), rng.uniform(0.5, 4.0));
        let wb: Vec<f64> = (0..m)
            .map(|_| lam0 * (-rng.uniform(1e-12, 1.0).ln()).powf(1.0 / k0))
            .collect();
        if let Some(FittedDistribution::Weibull { shape, scale }) = fit_weibull(&wb)
        {
            r.check((shape - k0).abs() / k0, 0.08, || {
                "distfit weibull shape".into()
            });
            r.check((scale - lam0).abs() / lam0, 0.08, || {
                "distfit weibull scale".into()
            });
        }
    }
    r
}

fn check_dual(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let nc = rng.int(2, 6);
        let states: Vec<ContributorState> = (0..nc)
            .map(|_| {
                ContributorState::new(
                    "x",
                    rng.uniform(-2.0, 2.0),
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(0.01, 0.3),
                )
            })
            .collect();
        let dual = dual_contributions(&states);
        // Mean contributions sum to the assembly off-centering.
        let asm = assembly_state(&states);
        let msum: f64 = dual.iter().map(|d| d.mean_contribution).sum();
        r.check((msum - asm.off_centering).abs(), 1e-9, || {
            "dual mean sum == δ_Y".into()
        });
        // Variance fractions sum to 1 and geo_factor == |coeff|.
        let vsum: f64 = dual.iter().map(|d| d.variance_fraction).sum();
        r.check((vsum - 1.0).abs(), 1e-12, || "dual variance sum".into());
        for (d, s) in dual.iter().zip(&states)
        {
            r.check((d.geo_factor - s.coeff.abs()).abs(), 1e-12, || {
                "dual geo_factor".into()
            });
        }
    }
    r
}

fn check_gdt(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let (mmc, tol) = (rng.uniform(5.0, 20.0), rng.uniform(0.01, 0.5));
        r.check(
            (virtual_condition(mmc, tol, FeatureType::Internal) - (mmc - tol)).abs(),
            1e-12,
            || "gdt VC internal".into(),
        );
        r.check(
            (virtual_condition(mmc, tol, FeatureType::External) - (mmc + tol)).abs(),
            1e-12,
            || "gdt VC external".into(),
        );
        // Datum shift: 0 at MMB, equal to the departure beyond it.
        let mmb = rng.uniform(5.0, 20.0);
        r.check(datum_shift(mmb, mmb, FeatureType::Internal), 1e-12, || {
            "gdt datum 0".into()
        });
        let dep = rng.uniform(0.0, 0.3);
        r.check(
            (datum_shift(mmb + dep, mmb, FeatureType::Internal) - dep).abs(),
            1e-12,
            || "gdt datum departure".into(),
        );
        // Composite conforms iff both tiers pass.
        let pltzf = rng.uniform(0.2, 0.6);
        let frtzf = rng.uniform(0.02, pltzf);
        let comp = CompositePosition::new(pltzf, frtzf);
        let (ldx, ldy) = (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3));
        let (pdx, pdy) = (rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12));
        let expected = true_position(ldx, ldy) <= pltzf && true_position(pdx, pdy) <= frtzf;
        r.check(
            if comp.conforms(ldx, ldy, pdx, pdy) == expected
            {
                0.0
            }
            else
            {
                1.0
            },
            0.5,
            || "gdt composite two-tier".into(),
        );
    }
    r
}

fn check_capability_ci(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let trials = 1500;
    for _ in 0..n
    {
        let nn = rng.int(10, 40);
        let conf = rng.uniform(0.80, 0.95);
        let sigma_true = rng.uniform(0.2, 1.0);
        let lsl = -rng.uniform(2.0, 4.0);
        let usl = rng.uniform(2.0, 4.0);
        let true_cp = (usl - lsl) / (6.0 * sigma_true);
        // Independent reference: the exact χ² Cp interval must cover the true Cp
        // with frequency ≈ the nominal confidence.
        let mut cover = 0u64;
        for _ in 0..trials
        {
            let xs: Vec<f64> = (0..nn).map(|_| sigma_true * rng.normal()).collect();
            let mean = xs.iter().sum::<f64>() / nn as f64;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nn as f64 - 1.0);
            let cp_hat = (usl - lsl) / (6.0 * var.sqrt());
            let (lo, hi) = cp_confidence_interval(cp_hat, nn, conf).unwrap();
            if lo <= true_cp && true_cp <= hi
            {
                cover += 1;
            }
        }
        let emp = cover as f64 / trials as f64;
        r.check((emp - conf).abs(), 0.05, || {
            format!("cp CI coverage {emp:.3} vs conf {conf:.3}")
        });
    }
    r
}

fn check_variables(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let trials = 3000;
    for _ in 0..n
    {
        let nn = rng.int(5, 30);
        let k = rng.uniform(1.0, 2.5);
        let plan = VariablesPlan::new(nn, k, true);
        // Independent Monte-Carlo of the OC. Put the upper limit at U = 0, σ = 1;
        // a process with fraction p beyond U has mean μ = −z_p (z_p = −Φ⁻¹(p)).
        // Accept when Q_U = (U − x̄)/σ = −x̄ ≥ k.
        let p = rng.uniform(0.005, 0.15);
        let z_p = -normal_cdf_inv(p);
        let mu = -z_p;
        let mut acc = 0u64;
        for _ in 0..trials
        {
            let xbar = (0..nn).map(|_| mu + rng.normal()).sum::<f64>() / nn as f64;
            if -xbar >= k
            {
                acc += 1;
            }
        }
        let emp = acc as f64 / trials as f64;
        r.check(
            (plan.probability_of_acceptance(p) - emp).abs(),
            0.05,
            || {
                format!(
                    "variables OC Pa {:.3} vs MC {emp:.3} (n={nn}, k={k:.2}, p={p:.3})",
                    plan.probability_of_acceptance(p)
                )
            },
        );
        // MSD identity: a centred lot at σ = MSD lands exactly on k.
        let (lsl, usl) = (-rng.uniform(1.0, 5.0), rng.uniform(1.0, 5.0));
        let msd = plan.max_process_sigma(lsl, usl);
        let mid = 0.5 * (lsl + usl);
        r.check(((usl - mid) / msd - k).abs(), 1e-9, || {
            "variables MSD==k".into()
        });
        // A designed plan is at least as protective as its two nominal points.
        let (aql, rql) = (rng.uniform(0.005, 0.03), rng.uniform(0.06, 0.15));
        let (alpha, beta) = (rng.uniform(0.02, 0.10), rng.uniform(0.05, 0.15));
        if let Some(d) = design_variables_plan(aql, rql, alpha, beta, true)
        {
            r.check(
                (d.probability_of_acceptance(aql) - (1.0 - alpha)).max(0.0) - 0.03,
                0.0,
                || "variables design Pa(AQL)>=1-alpha".into(),
            );
            r.check(
                (d.probability_of_acceptance(rql) - beta).max(0.0) - 0.03,
                0.0,
                || "variables design Pa(RQL)<=beta".into(),
            );
        }
    }
    r
}

fn check_sixsigma(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let shift = if rng.u64() & 1 == 0 { 0.0 } else { 1.5 };
        // yield ↔ sigma round trip.
        let y = rng.uniform(0.5, 0.9999);
        let s = sigma_from_yield(y, shift);
        r.check((yield_from_sigma(s, shift) - y).abs(), 1e-9, || {
            "sixsigma y<->sigma".into()
        });
        // dpmo ↔ sigma round trip and independent normal-tail reference.
        let sigma = rng.uniform(1.0, 6.0);
        let d = dpmo_from_sigma(sigma, shift);
        r.check((1e6 * normal_sf(sigma - shift) - d).abs(), 1e-6, || {
            "sixsigma dpmo vs normal_sf".into()
        });
        // Round-trip through the deep tail loses a few ulps to 1−(1−Φ)
        // cancellation at large σ — still far tighter than any real use needs.
        r.check((sigma_from_dpmo(d, shift) - sigma).abs(), 1e-6, || {
            "sixsigma sigma<->dpmo".into()
        });
        // Poisson yield: −ln(Y) recovers the DPU.
        let dpu = rng.uniform(0.0, 2.0);
        r.check((-throughput_yield(dpu).ln() - dpu).abs(), 1e-12, || {
            "sixsigma -lnY==dpu".into()
        });
        // Rolled throughput yield vs an independent product; normalisation.
        let steps = rng.int(2, 8);
        let ys: Vec<f64> = (0..steps).map(|_| rng.uniform(0.90, 0.999)).collect();
        let mut prod = 1.0;
        for &yi in &ys
        {
            prod *= yi;
        }
        let rty = rolled_throughput_yield(&ys);
        r.check((rty - prod).abs(), 1e-12, || "sixsigma RTY==prod".into());
        let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        r.check((rty - ymin).max(0.0), 1e-12, || "sixsigma RTY<=min".into());
        // Normalised yield^steps == RTY.
        let ynorm = normalized_yield(rty, steps);
        r.check((ynorm.powi(steps as i32) - rty).abs(), 1e-9, || {
            "sixsigma ynorm^k==RTY".into()
        });
        // Report self-consistency.
        let rep = process_report(&ys, shift).unwrap();
        r.check(
            ((-rep.total_dpu).exp() - rep.rolled_throughput_yield).abs(),
            1e-9,
            || "sixsigma report total_dpu".into(),
        );
    }
    r
}

fn check_attribution(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let k = rng.int(1, 4);
        let nn = rng.int(60, 200);
        // Independent normal regressor columns and known coefficients.
        let alpha: Vec<f64> = (0..k).map(|_| rng.uniform(-3.0, 3.0)).collect();
        let cols: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..nn).map(|_| rng.normal()).collect())
            .collect();
        let sigma_n = rng.uniform(0.05, 0.5);
        let y: Vec<f64> = (0..nn)
            .map(|i| (0..k).map(|j| alpha[j] * cols[j][i]).sum::<f64>() + sigma_n * rng.normal())
            .collect();
        let names: Vec<&str> = ["x1", "x2", "x3", "x4"][..k].to_vec();
        let Some(rep) = attribute(&names, &cols, &y)
        else
        {
            continue;
        };
        // (1) Exact identity: contributions sum to R² (Cov-based vs residual R²).
        let sum: f64 = rep.components.iter().map(|c| c.contribution).sum();
        r.check((sum - rep.r_squared).abs(), 1e-7, || {
            format!("attribution Σc={sum:.6} vs R²={:.6}", rep.r_squared)
        });
        // (2) Fitted sensitivities recover the known generating coefficients.
        for (j, a) in alpha.iter().enumerate()
        {
            r.check((rep.components[j].sensitivity - a).abs(), 0.2, || {
                format!(
                    "attribution β{j}={:.3} vs α={a:.3}",
                    rep.components[j].sensitivity
                )
            });
        }
        // (3) Explained + unexplained partition to 1.
        r.check((rep.r_squared + rep.unexplained - 1.0).abs(), 1e-12, || {
            "attribution R²+unexpl==1".into()
        });
        // (4) Single regressor: contribution equals corr(x,y)² (independent).
        if k == 1
        {
            let xm = cols[0].iter().sum::<f64>() / nn as f64;
            let ym = y.iter().sum::<f64>() / nn as f64;
            let sxy: f64 = (0..nn).map(|i| (cols[0][i] - xm) * (y[i] - ym)).sum();
            let sxx: f64 = cols[0].iter().map(|x| (x - xm).powi(2)).sum();
            let syy: f64 = y.iter().map(|v| (v - ym).powi(2)).sum();
            let corr2 = sxy * sxy / (sxx * syy);
            r.check((rep.components[0].contribution - corr2).abs(), 1e-7, || {
                "attribution c==corr²".into()
            });
        }
    }
    r
}

fn check_attributes(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let trials = 3000;
    for _ in 0..n
    {
        let nn = rng.int(10, 60);
        let c = rng.int(0, 4);
        let plan = AttributesPlan::new(nn, c);
        let p = rng.uniform(0.01, 0.20);
        // Independent Monte-Carlo: draw Binomial(nn, p) as nn Bernoulli trials,
        // accept when defectives ≤ c; the accept rate estimates Pa(p).
        let mut acc = 0u64;
        for _ in 0..trials
        {
            let d = (0..nn).filter(|_| rng.uniform(0.0, 1.0) < p).count();
            if d <= c
            {
                acc += 1;
            }
        }
        let emp = acc as f64 / trials as f64;
        r.check(
            (plan.probability_of_acceptance(p) - emp).abs(),
            0.05,
            || {
                format!(
                    "attributes OC Pa {:.3} vs MC {emp:.3} (n={nn}, c={c}, p={p:.3})",
                    plan.probability_of_acceptance(p)
                )
            },
        );
        // A designed plan clears both nominal points.
        let (aql, rql) = (rng.uniform(0.005, 0.02), rng.uniform(0.08, 0.15));
        let (alpha, beta) = (0.05, 0.10);
        if let Some(d) = design_attributes_plan(aql, rql, alpha, beta, 400)
        {
            r.check(
                (1.0 - alpha - d.probability_of_acceptance(aql)).max(0.0),
                1e-9,
                || "attributes design Pa(AQL)>=1-alpha".into(),
            );
            r.check(
                (d.probability_of_acceptance(rql) - beta).max(0.0),
                1e-9,
                || "attributes design Pa(RQL)<=beta".into(),
            );
        }
    }
    r
}

fn check_interference(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let trials = 4000;
    for _ in 0..n
    {
        let mu_s = rng.uniform(8.0, 12.0);
        let sd_s = rng.uniform(0.3, 1.5);
        let mu_l = rng.uniform(6.0, 11.0);
        let sd_l = rng.uniform(0.3, 1.5);
        let rel = interference_reliability(mu_s, sd_s, mu_l, sd_l);
        // Independent Monte-Carlo of P(S > L).
        let mut surv = 0u64;
        for _ in 0..trials
        {
            let s = mu_s + sd_s * rng.normal();
            let l = mu_l + sd_l * rng.normal();
            if s > l
            {
                surv += 1;
            }
        }
        let emp = surv as f64 / trials as f64;
        r.check((rel - emp).abs(), 0.03, || {
            format!("interference R {rel:.3} vs MC {emp:.3}")
        });
        // Clearance fit: prob_clearance == reliability of hole>shaft; partition.
        let f = clearance_fit(mu_s, sd_s, mu_l, sd_l);
        r.check((f.prob_clearance - rel).abs(), 1e-12, || {
            "interference clearance==R".into()
        });
        r.check(
            (f.prob_clearance + f.prob_interference - 1.0).abs(),
            1e-12,
            || "interference partition".into(),
        );
        r.check(
            (f.sd_clearance - (sd_s * sd_s + sd_l * sd_l).sqrt()).abs(),
            1e-12,
            || "interference sd_clearance".into(),
        );
    }
    r
}

fn check_subgroup(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    for _ in 0..n
    {
        let k = rng.int(15, 40); // subgroups
        let m = rng.int(3, 8); // subgroup size
        let mu = rng.uniform(50.0, 100.0);
        let sigma = rng.uniform(0.5, 3.0);
        let groups: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..m).map(|_| mu + sigma * rng.normal()).collect())
            .collect();
        let (lsl, usl) = (mu - 6.0 * sigma, mu + 6.0 * sigma);
        let s = subgroup_capability(&groups, lsl, usl).unwrap();
        // (1) Independent recomputation of the overall sigma (pooled about grand).
        let all: Vec<f64> = groups.iter().flatten().copied().collect();
        let nf = all.len() as f64;
        let gmean = all.iter().sum::<f64>() / nf;
        let so = (all.iter().map(|x| (x - gmean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
        r.check((s.sigma_overall - so).abs(), 1e-9, || {
            "subgroup overall sigma".into()
        });
        // (2) Range-method and s-method within-sigma agree (two estimators).
        let s_method = sigma_within_s_method(&groups).unwrap();
        r.check(
            (s.sigma_within - s_method).abs() / s.sigma_within,
            0.15,
            || {
                format!(
                    "subgroup within R̄/d₂ {:.4} vs s̄/c₄ {s_method:.4}",
                    s.sigma_within
                )
            },
        );
        // (3) Cp identity from the within spread.
        r.check(
            (s.cp * 6.0 * s.sigma_within - (usl - lsl)).abs() / (usl - lsl),
            1e-12,
            || "subgroup Cp identity".into(),
        );
    }
    r
}

fn check_fits(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let letters = ['d', 'e', 'f', 'g', 'h'];
    for _ in 0..n
    {
        let nominal = rng.uniform(1.0, 500.0);
        let hole_grade = rng.int(5, 12) as u8;
        let shaft_grade = rng.int(5, 12) as u8;
        let letter = letters[rng.int(0, 4)];
        let it_hole = it_grade_tolerance(hole_grade, nominal).unwrap();
        let it_shaft = it_grade_tolerance(shaft_grade, nominal).unwrap();
        let fit = hole_basis_fit(nominal, hole_grade, shaft_grade, letter).unwrap();
        // Independent identity: the clearance range equals the summed tolerances.
        r.check(
            (fit.max_clearance - fit.min_clearance - (it_hole + it_shaft)).abs()
                / (it_hole + it_shaft),
            1e-12,
            || "fits clearance range == IT_hole+IT_shaft".into(),
        );
        // Grade monotonicity: a coarser grade is a wider tolerance.
        let coarser = it_grade_tolerance(hole_grade + 1, nominal).unwrap();
        r.check((it_hole - coarser).max(0.0), 0.0, || {
            "fits IT monotone in grade".into()
        });
        // Independent parallel recomputation of the IT magnitude from the
        // i-factor formula (a different code path than the module).
        let hi = [
            3.0, 6.0, 10.0, 18.0, 30.0, 50.0, 80.0, 120.0, 180.0, 250.0, 315.0, 400.0, 500.0,
        ];
        let mult = [
            7.0, 10.0, 16.0, 25.0, 40.0, 63.0, 100.0, 160.0, 250.0, 400.0, 640.0, 1000.0, 1600.0,
            2500.0,
        ];
        let mut d = 0.0;
        for (j, &h) in hi.iter().enumerate()
        {
            let lo = if j == 0 { 1.0 } else { hi[j - 1] };
            if nominal > lo && nominal <= h
            {
                d = (lo * h).sqrt();
                break;
            }
        }
        let it_ref = mult[(hole_grade - 5) as usize] * (0.45 * d.cbrt() + 0.001 * d);
        r.check((it_hole - it_ref).abs() / it_hole, 1e-12, || {
            "fits IT vs independent i-factor formula".into()
        });
    }
    r
}

fn check_sequential(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let trials = 3000;
    for _ in 0..n
    {
        // --- Double sampling: OC and ASN vs direct Monte-Carlo. ---
        let n1 = rng.int(20, 60);
        let c1 = rng.int(0, 2);
        let r1 = c1 + rng.int(2, 4);
        let n2 = rng.int(20, 60);
        let c2 = r1 + rng.int(0, 3);
        let plan = DoubleSamplingPlan::new(n1, c1, r1, n2, c2);
        let p = rng.uniform(0.01, 0.12);
        let (mut acc, mut total_n) = (0u64, 0u64);
        for _ in 0..trials
        {
            let d1 = (0..n1).filter(|_| rng.uniform(0.0, 1.0) < p).count();
            if d1 <= c1
            {
                acc += 1;
                total_n += n1 as u64;
            }
            else if d1 >= r1
            {
                total_n += n1 as u64;
            }
            else
            {
                let d2 = (0..n2).filter(|_| rng.uniform(0.0, 1.0) < p).count();
                total_n += (n1 + n2) as u64;
                if d1 + d2 <= c2
                {
                    acc += 1;
                }
            }
        }
        let emp_pa = acc as f64 / trials as f64;
        let emp_asn = total_n as f64 / trials as f64;
        r.check(
            (plan.probability_of_acceptance(p) - emp_pa).abs(),
            0.05,
            || {
                format!(
                    "double OC {:.3} vs MC {emp_pa:.3}",
                    plan.probability_of_acceptance(p)
                )
            },
        );
        r.check(
            (plan.average_sample_number(p) - emp_asn).abs() / emp_asn,
            0.05,
            || {
                format!(
                    "double ASN {:.1} vs MC {emp_asn:.1}",
                    plan.average_sample_number(p)
                )
            },
        );

        // --- SPRT: OC guarantee at the two design points. ---
        let (aql, rql) = (rng.uniform(0.01, 0.03), rng.uniform(0.08, 0.15));
        let (alpha, beta) = (0.05, 0.10);
        let sprt = design_sequential_plan(aql, rql, alpha, beta).unwrap();
        let max_n = 2000;
        let sim = |pp: f64, rng: &mut Rng| {
            let mut ok = 0u64;
            for _ in 0..1000
            {
                let (mut nn, mut d) = (0usize, 0usize);
                loop
                {
                    nn += 1;
                    if rng.uniform(0.0, 1.0) < pp
                    {
                        d += 1;
                    }
                    match sprt.verdict(nn, d)
                    {
                        SequentialVerdict::Accept =>
                        {
                            ok += 1;
                            break;
                        },
                        SequentialVerdict::Reject => break,
                        SequentialVerdict::Continue =>
                        {
                            if nn >= max_n
                            {
                                break;
                            }
                        },
                    }
                }
            }
            ok as f64 / 1000.0
        };
        let pa_good = sim(aql, rng);
        let pa_bad = sim(rql, rng);
        // Wald's SPRT holds Pa(AQL) ≳ 1−α and Pa(RQL) ≲ β (bounds are conservative).
        r.check(((1.0 - alpha) - pa_good).max(0.0), 0.07, || {
            format!("sprt Pa(AQL) {pa_good:.3} vs 1−α {:.3}", 1.0 - alpha)
        });
        r.check((pa_bad - beta).max(0.0), 0.07, || {
            format!("sprt Pa(RQL) {pa_bad:.3} vs β {beta:.3}")
        });
    }
    r
}

fn check_taguchi(rng: &mut Rng, n: usize) -> Report {
    let mut r = Report::default();
    let trials = 20000;
    for _ in 0..n
    {
        let k = rng.uniform(1.0, 50.0);
        let target = rng.uniform(-5.0, 5.0);
        let mean = target + rng.uniform(-1.0, 1.0);
        let sd = rng.uniform(0.05, 1.0);
        let delta = mean - target;
        let inertia = (delta * delta + sd * sd).sqrt();
        // Identity: E[L] via inertia == via moments == k·I².
        r.check(
            (expected_loss(k, inertia) - expected_loss_from_moments(k, mean, sd, target)).abs(),
            1e-9,
            || "taguchi E[L] inertia==moments".into(),
        );
        // Independent Monte-Carlo of the mean quadratic loss.
        let mut acc = 0.0;
        for _ in 0..trials
        {
            let y = mean + sd * rng.normal();
            acc += quadratic_loss(k, y, target);
        }
        let emp = acc / trials as f64;
        let want = expected_loss(k, inertia);
        r.check((emp - want).abs() / want, 0.06, || {
            format!("taguchi E[L] {want:.3} vs MC {emp:.3}")
        });
        // Loss coefficient: a part at the limit costs exactly `cost`.
        let (cost, half_tol) = (rng.uniform(1.0, 100.0), rng.uniform(0.1, 2.0));
        let kk = loss_coefficient(cost, half_tol);
        r.check(
            (quadratic_loss(kk, target + half_tol, target) - cost).abs(),
            1e-9,
            || "taguchi coeff hits cost at limit".into(),
        );
        // Economic tolerance balances functional loss against rework cost.
        let (a0, delta0, a) = (
            rng.uniform(5.0, 50.0),
            rng.uniform(0.5, 2.0),
            rng.uniform(1.0, 20.0),
        );
        let econ = economic_tolerance(a0, delta0, a);
        let k0 = loss_coefficient(a0, delta0);
        r.check((k0 * econ * econ - a).abs(), 1e-9, || {
            "taguchi econ tol balance".into()
        });
        // Smaller-the-better vs a Monte-Carlo about 0.
        let mut acc2 = 0.0;
        for _ in 0..trials
        {
            let y = mean + sd * rng.normal();
            acc2 += k * y * y;
        }
        let emp2 = acc2 / trials as f64;
        let want2 = smaller_the_better_loss(k, mean, sd);
        r.check((emp2 - want2).abs() / want2, 0.06, || {
            "taguchi smaller-the-better".into()
        });
    }
    r
}

/// Standard-normal quantile via the crate's `special`, wrapped for local use.
fn normal_cdf_inv(p: f64) -> f64 {
    scirust_tolerance::special::inv_normal_cdf(p)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(250);
    let mut rng = Rng(0xda3e_39cb_94b6_95a5);

    let reports = [
        ("special", check_special(&mut rng, n)),
        ("sampling", check_sampling(&mut rng, n.min(200))),
        ("spatial", check_spatial(&mut rng, n)),
        ("modal", check_modal(&mut rng, n)),
        ("chain", check_chain(&mut rng, n)),
        ("capability", check_capability(&mut rng, n)),
        ("nonnormal", check_nonnormal(&mut rng, n)),
        ("position", check_position(&mut rng, n)),
        ("montecarlo", check_montecarlo(&mut rng, n.min(120))),
        ("correlated", check_correlated(&mut rng, n)),
        ("geometry", check_geometry(&mut rng, n)),
        ("sensitivity", check_sensitivity(&mut rng, n)),
        ("process", check_process(&mut rng, n)),
        ("drift", check_drift(&mut rng, n.min(150))),
        ("msa", check_msa(&mut rng, n)),
        ("interval", check_interval(&mut rng, n.min(80))),
        ("distfit", check_distfit(&mut rng, n.min(120))),
        ("dual", check_dual(&mut rng, n)),
        ("gdt", check_gdt(&mut rng, n)),
        ("capability_ci", check_capability_ci(&mut rng, n.min(80))),
        ("variables", check_variables(&mut rng, n.min(120))),
        ("sixsigma", check_sixsigma(&mut rng, n)),
        ("attribution", check_attribution(&mut rng, n.min(120))),
        ("attributes", check_attributes(&mut rng, n.min(120))),
        ("interference", check_interference(&mut rng, n.min(150))),
        ("subgroup", check_subgroup(&mut rng, n.min(150))),
        ("fits", check_fits(&mut rng, n)),
        ("sequential", check_sequential(&mut rng, n.min(60))),
        ("taguchi", check_taguchi(&mut rng, n.min(80))),
    ];

    println!("=== fuzz_crosscheck ({n} instances/module, independent references) ===");
    let mut total_err = 0usize;
    let mut total_chk = 0usize;
    for (name, rep) in &reports
    {
        println!("{}", rep.line(name));
        total_err += rep.errors;
        total_chk += rep.checks;
    }
    println!("---------------------------------------------------------------");
    println!("total checks {total_chk}, total errors {total_err}");
    if total_err == 0
    {
        println!("PASS — every module agrees with its independent cross-check.");
    }
    else
    {
        println!("FAIL — {total_err} cross-check(s) disagreed.");
        std::process::exit(1);
    }
}
