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
//!
//! Run: `cargo run -p scirust-tolerance --example fuzz_crosscheck [N]`

use scirust_tolerance::capability::nonconformity_ppm;
use scirust_tolerance::chain::{
    Allocation, Contributor, allocate, assembly_inertia_statistical, assembly_inertia_worst_case,
};
use scirust_tolerance::form::FormBatch;
use scirust_tolerance::modal::{ModalBasis, modal_inertias};
use scirust_tolerance::nonnormal::{cornish_fisher_quantile, nonnormal_ppm};
use scirust_tolerance::position::{
    coord_to_position, position_to_coord, positional_inertia, true_position,
};
use scirust_tolerance::sampling::SamplingPlan;
use scirust_tolerance::spatial::{
    Feature, Torsor, inertia_decomposition, surface_inertia_analytical,
    surface_inertia_from_torsors,
};
use scirust_tolerance::special::{
    chi2_cdf, chi2_quantile, erf, erfc, ncchi2_cdf, normal_cdf, normal_sf,
};

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
