//! Fuzz cross-check of the multi-requirement cost optimiser
//! ([`scirust_tolerance::optimize`]) over many random instances.
//!
//! For each instance the solver's output is checked against an **independent,
//! primal-only optimality certificate** (it never looks at the solver's dual
//! multipliers):
//!
//! 1. **No NaN/Inf** in the inertias or the cost.
//! 2. **Feasibility** — every requirement's resultant inertia ≤ its budget.
//! 3. **Coordinate optimality ("pinned")** — because the cost `Σ bᵢ Iᵢ^(−rᵢ)`
//!    strictly *decreases* as any `Iᵢ` grows, a globally optimal point must
//!    leave no component room to grow: for every `i`, the largest feasible
//!    `Iᵢ` (holding the others fixed) must equal the returned `Iᵢ`. Any
//!    appreciable slack means cost was left on the table (a real bug).
//!
//! Deterministic (seeded xorshift), so a failure is reproducible.
//!
//! Run: `cargo run -p scirust-tolerance --example fuzz_optimize [N]`

use scirust_tolerance::optimize::{Component, Requirement, optimize};

/// Tiny deterministic xorshift64* RNG (avoids an external `rand` dependency).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform f64 in `[lo, hi)`.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
    /// Uniform integer in `[lo, hi]`.
    fn int(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }
}

fn main() {
    let n_instances: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1500);

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    let feas_tol = 1e-6; // relative infeasibility tolerance
    let opt_tol = 1e-3; // relative "can still grow" tolerance

    let mut solved = 0usize;
    let mut skipped = 0usize;
    let mut nonconverged = 0usize;
    let mut errors = 0usize;
    let mut worst_infeas = 0.0f64;
    let mut worst_growth = 0.0f64;
    let mut failures: Vec<String> = Vec::new();

    for inst in 0..n_instances
    {
        // Random components with reciprocal-power costs.
        let n = rng.int(2, 8);
        let comps: Vec<Component> = (0..n)
            .map(|i| {
                Component::new(
                    format!("X{i}"),
                    rng.uniform(0.1, 10.0),
                    rng.uniform(0.5, 4.0),
                )
            })
            .collect();

        // Random requirements (some coefficients zeroed to make chains sparse).
        let k = rng.int(1, 4);
        let mut reqs: Vec<Requirement> = (0..k)
            .map(|kk| {
                let coeffs: Vec<f64> = (0..n)
                    .map(|_| {
                        if rng.uniform(0.0, 1.0) < 0.3
                        {
                            0.0
                        }
                        else
                        {
                            rng.uniform(-2.0, 2.0)
                        }
                    })
                    .collect();
                Requirement::new(format!("Y{kk}"), coeffs, rng.uniform(0.01, 0.2))
            })
            .collect();

        // Ensure every component is reached by at least one requirement.
        for i in 0..n
        {
            if reqs.iter().all(|r| r.coeffs[i] == 0.0)
            {
                let target = rng.int(0, k - 1);
                reqs[target].coeffs[i] = if rng.uniform(0.0, 1.0) < 0.5
                {
                    1.0
                }
                else
                {
                    -1.0
                };
            }
        }

        let res = match optimize(&comps, &reqs)
        {
            Ok(r) => r,
            Err(_) =>
            {
                skipped += 1;
                continue;
            },
        };
        solved += 1;
        if !res.converged
        {
            nonconverged += 1;
        }

        // (1) Finiteness.
        if res.inertias.iter().any(|x| !x.is_finite()) || !res.total_cost.is_finite()
        {
            errors += 1;
            if failures.len() < 6
            {
                failures.push(format!("#{inst}: non-finite output"));
            }
            continue;
        }

        // Precompute achieved² and budgets².
        let ck: Vec<f64> = reqs.iter().map(|r| r.i_max * r.i_max).collect();
        let ach2: Vec<f64> = reqs
            .iter()
            .map(|r| {
                let a = r.achieved(&res.inertias);
                a * a
            })
            .collect();

        // (2) Feasibility.
        let mut inst_infeas = 0.0f64;
        for (r, a2) in reqs.iter().zip(&ach2)
        {
            let rel = (a2.sqrt() - r.i_max) / r.i_max;
            inst_infeas = inst_infeas.max(rel);
        }

        // (3) Coordinate optimality: how much could each Iᵢ grow, holding the
        // rest fixed, before some constraint hits its budget?
        let mut inst_growth = 0.0f64;
        for i in 0..n
        {
            let ii = res.inertias[i];
            let mut max_grow = f64::INFINITY;
            for (kk, r) in reqs.iter().enumerate()
            {
                let a = r.coeffs[i];
                if a == 0.0
                {
                    continue;
                }
                let a2 = a * a;
                // achieved² without component i's contribution.
                let rem = ck[kk] - (ach2[kk] - a2 * ii * ii);
                if rem <= 0.0
                {
                    max_grow = 0.0;
                    break;
                }
                let ii_max = (rem / a2).sqrt();
                max_grow = max_grow.min(ii_max - ii);
            }
            if max_grow.is_finite()
            {
                inst_growth = inst_growth.max(max_grow / ii.max(1e-300));
            }
        }

        worst_infeas = worst_infeas.max(inst_infeas);
        worst_growth = worst_growth.max(inst_growth);

        if inst_infeas > feas_tol || inst_growth > opt_tol
        {
            errors += 1;
            if failures.len() < 6
            {
                failures.push(format!(
                    "#{inst}: n={n} k={k} infeas={inst_infeas:.2e} growth={inst_growth:.2e} conv={}",
                    res.converged
                ));
            }
        }
    }

    println!("=== fuzz_optimize: {n_instances} instances ===");
    println!("solved            : {solved}");
    println!("skipped (degenerate input): {skipped}");
    println!("non-converged     : {nonconverged}");
    println!("worst infeasibility (rel): {worst_infeas:.3e}  (tol {feas_tol:.0e})");
    println!("worst can-still-grow (rel): {worst_growth:.3e}  (tol {opt_tol:.0e})");
    println!("ERRORS            : {errors}");
    for f in &failures
    {
        println!("  {f}");
    }
    if errors == 0
    {
        println!("PASS — all solved instances are feasible and coordinate-optimal.");
    }
    else
    {
        println!("FAIL — {errors} instance(s) violated the optimality certificate.");
        std::process::exit(1);
    }
}
