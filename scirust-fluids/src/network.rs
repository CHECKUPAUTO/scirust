//! Looped pipe networks — the **Hardy Cross** method.
//!
//! A network is described by its pipes (each with a head-loss law
//! `h = r·|Q|^{n−1}·Q`, `n = 2` for Darcy–Weisbach, `n = 1.852` for
//! Hazen–Williams), a set of closed loops (signed pipe memberships) and
//! an initial flow distribution that already satisfies continuity at
//! every node. Hardy Cross iteratively corrects each loop's flows until
//! the head loss around every loop closes; loop corrections preserve
//! node continuity exactly, by construction.
//!
//! [`hardy_cross`] takes a fixed resistance per pipe. [`hardy_cross_darcy`]
//! couples this to [`crate::pipe::friction_factor`]: each pipe is given
//! its real diameter, length and roughness, and its Darcy resistance is
//! recomputed from the actual Reynolds number at every outer iteration
//! (laminar, Colebrook–White or the documented blend, exactly as in
//! [`crate::pipe`]) — sizing a real network rather than one with
//! pre-guessed friction factors.

use crate::error::{FluidsError, finite, in_range, non_negative, positive};

/// A pipe's head-loss law `h = r·|Q|^{n−1}·Q` \[m\] for flow `Q` \[m³/s\].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkPipe {
    r: f64,
    n: f64,
}

impl NetworkPipe {
    /// Build a pipe from its resistance coefficient `r > 0` and head-loss
    /// exponent `n ∈ [1, 3]` (2 for Darcy–Weisbach with a fixed friction
    /// factor, 1.852 for Hazen–Williams).
    pub fn new(r: f64, n: f64) -> Result<Self, FluidsError> {
        positive("r", r)?;
        in_range("n", n, 1.0, 3.0)?;
        Ok(Self { r, n })
    }

    /// Signed head loss \[m\] at flow `q` \[m³/s\] (negative for
    /// reversed flow).
    pub fn head_loss(&self, q: f64) -> f64 {
        self.r * q.abs().powf(self.n - 1.0) * q
    }

    /// Derivative `dh/dQ = n·r·|Q|^{n−1}` of the head-loss law.
    pub fn head_loss_slope(&self, q: f64) -> f64 {
        self.n * self.r * q.abs().powf(self.n - 1.0)
    }
}

/// Solve a looped network by **Hardy Cross** iteration.
///
/// * `pipes` — the head-loss law of every pipe
/// * `loops` — each loop is a list of `(pipe index, orientation)` pairs,
///   orientation `+1.0` when the pipe's reference direction follows the
///   loop's traversal direction and `−1.0` when it opposes it
/// * `initial_flows` — starting flow of every pipe \[m³/s\] in its
///   reference direction; **must satisfy node continuity** (the method
///   preserves whatever continuity the input has)
/// * `tolerance` — convergence threshold on the largest loop correction
///   \[m³/s\], > 0
/// * `max_iterations` — iteration budget before giving up
///
/// Returns the corrected flow of every pipe. Deterministic: loops are
/// swept in the given order with a fixed update rule, so identical
/// inputs give identical outputs everywhere. Fails with
/// [`FluidsError::NoConvergence`] if the budget is exhausted or a loop
/// degenerates (all its flows and its correction denominator vanish).
pub fn hardy_cross(
    pipes: &[NetworkPipe],
    loops: &[Vec<(usize, f64)>],
    initial_flows: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<Vec<f64>, FluidsError> {
    if pipes.is_empty()
    {
        return Err(FluidsError::NonPositive {
            name: "pipes.len()",
            value: 0.0,
        });
    }
    if initial_flows.len() != pipes.len()
    {
        return Err(FluidsError::OutOfRange {
            name: "initial_flows.len()",
            value: initial_flows.len() as f64,
            min: pipes.len() as f64,
            max: pipes.len() as f64,
        });
    }
    if loops.is_empty()
    {
        return Err(FluidsError::NonPositive {
            name: "loops.len()",
            value: 0.0,
        });
    }
    for &q in initial_flows
    {
        finite("initial_flows", q)?;
    }
    for lp in loops
    {
        if lp.is_empty()
        {
            return Err(FluidsError::NonPositive {
                name: "loop.len()",
                value: 0.0,
            });
        }
        for &(idx, sign) in lp
        {
            if idx >= pipes.len()
            {
                return Err(FluidsError::OutOfRange {
                    name: "pipe index",
                    value: idx as f64,
                    min: 0.0,
                    max: (pipes.len() - 1) as f64,
                });
            }
            if sign != 1.0 && sign != -1.0
            {
                return Err(FluidsError::OutOfRange {
                    name: "orientation",
                    value: sign,
                    min: -1.0,
                    max: 1.0,
                });
            }
        }
    }
    positive("tolerance", tolerance)?;
    if max_iterations == 0
    {
        return Err(FluidsError::NonPositive {
            name: "max_iterations",
            value: 0.0,
        });
    }

    let mut flows = initial_flows.to_vec();
    for _ in 0..max_iterations
    {
        let mut worst = 0.0f64;
        for lp in loops
        {
            // Head-loss closure and its slope around this loop.
            let mut closure = 0.0;
            let mut slope = 0.0;
            for &(idx, sign) in lp
            {
                closure += sign * pipes[idx].head_loss(flows[idx]);
                slope += pipes[idx].head_loss_slope(flows[idx]);
            }
            if slope <= f64::MIN_POSITIVE
            {
                // All flows of the loop are zero: the correction is
                // undefined — a degenerate input (no flow to balance).
                return Err(FluidsError::NoConvergence {
                    what: "Hardy Cross loop correction (degenerate loop)",
                });
            }
            let dq = -closure / slope;
            for &(idx, sign) in lp
            {
                flows[idx] += sign * dq;
            }
            worst = worst.max(dq.abs());
        }
        if worst < tolerance
        {
            return Ok(flows);
        }
    }
    Err(FluidsError::NoConvergence {
        what: "Hardy Cross iteration budget",
    })
}

/// A physical pipe whose Darcy resistance is derived from its actual
/// dimensions rather than pre-computed, for use with
/// [`hardy_cross_darcy`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalPipe {
    diameter: f64,
    length: f64,
    roughness: f64,
}

impl PhysicalPipe {
    /// Build a physical pipe from its internal diameter `D > 0` \[m\],
    /// length `L > 0` \[m\] and absolute roughness `ε ≥ 0` \[m\], with
    /// `ε/D ≤ 0.1` (the validity range of [`crate::pipe::friction_factor`]).
    pub fn new(diameter: f64, length: f64, roughness: f64) -> Result<Self, FluidsError> {
        positive("diameter", diameter)?;
        positive("length", length)?;
        non_negative("roughness", roughness)?;
        in_range("roughness / diameter", roughness / diameter, 0.0, 0.1)?;
        Ok(Self {
            diameter,
            length,
            roughness,
        })
    }

    /// Cross-sectional area `π D²/4` \[m²\].
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.diameter * self.diameter / 4.0
    }
}

/// Solve a looped network of **physical** pipes, coupling [`hardy_cross`]
/// to [`crate::pipe::friction_factor`].
///
/// At each outer iteration, every pipe's Darcy friction factor is
/// recomputed from its current Reynolds number, producing a resistance
/// `r = f L/(2 g D A²)` (Darcy–Weisbach, `h = r Q²`); an inner
/// [`hardy_cross`] pass then corrects the loop flows for that frozen
/// set of resistances. Repeating this outer loop to convergence is the
/// standard successive-substitution treatment of a Re-dependent `f` in
/// real pipe-network solvers: `h = f (L/D) V²/(2g)` is quadratic in `V`
/// by the Darcy–Weisbach definition itself in every regime — laminar
/// flow's linear-in-`V` head loss falls out of `f` itself scaling as
/// `1/Re`, not from a different power law — so the fixed point of this
/// iteration is the exact physical solution.
///
/// * `pipes` — physical dimensions of each pipe
/// * `loops`, `initial_flows` — as [`hardy_cross`]
/// * `density`, `dyn_viscosity` — fluid properties \[kg/m³\], \[Pa·s\]
/// * `gravity` — \[m/s²\], > 0
/// * `tolerance` — convergence threshold on both the inner loop
///   correction and the outer flow change between successive friction
///   updates \[m³/s\], > 0
/// * `max_outer_iterations`, `max_inner_iterations` — iteration budgets
///
/// Deterministic: the friction factor at (near-)zero flow is evaluated
/// at a floored Reynolds number so it never triggers a domain error;
/// identical inputs give identical outputs everywhere.
#[allow(clippy::too_many_arguments)]
pub fn hardy_cross_darcy(
    pipes: &[PhysicalPipe],
    loops: &[Vec<(usize, f64)>],
    initial_flows: &[f64],
    density: f64,
    dyn_viscosity: f64,
    gravity: f64,
    tolerance: f64,
    max_outer_iterations: usize,
    max_inner_iterations: usize,
) -> Result<Vec<f64>, FluidsError> {
    positive("density", density)?;
    positive("dyn_viscosity", dyn_viscosity)?;
    positive("gravity", gravity)?;
    positive("tolerance", tolerance)?;
    if initial_flows.len() != pipes.len()
    {
        return Err(FluidsError::OutOfRange {
            name: "initial_flows.len()",
            value: initial_flows.len() as f64,
            min: pipes.len() as f64,
            max: pipes.len() as f64,
        });
    }
    if max_outer_iterations == 0
    {
        return Err(FluidsError::NonPositive {
            name: "max_outer_iterations",
            value: 0.0,
        });
    }

    let mut flows = initial_flows.to_vec();
    for _ in 0..max_outer_iterations
    {
        let mut net_pipes = Vec::with_capacity(pipes.len());
        for (p, &q) in pipes.iter().zip(flows.iter())
        {
            let area = p.area();
            let speed = q.abs() / area;
            let re = (density * speed * p.diameter / dyn_viscosity).max(1e-9);
            let f = crate::pipe::friction_factor(re, p.roughness / p.diameter)?;
            let r = f * p.length / (2.0 * gravity * p.diameter * area * area);
            net_pipes.push(NetworkPipe::new(r, 2.0)?);
        }
        let new_flows = hardy_cross(&net_pipes, loops, &flows, tolerance, max_inner_iterations)?;
        let max_change = new_flows
            .iter()
            .zip(flows.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        flows = new_flows;
        if max_change < tolerance
        {
            return Ok(flows);
        }
    }
    Err(FluidsError::NoConvergence {
        what: "Hardy Cross / Darcy-Weisbach outer iteration",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_parallel_pipes_split_evenly() {
        // Two identical pipes between the same two nodes carrying 10 m³/s
        // in total must converge to a 5/5 split whatever the start.
        let p = NetworkPipe::new(3.0, 2.0).unwrap();
        let flows = hardy_cross(
            &[p, p],
            &[vec![(0, 1.0), (1, -1.0)]],
            &[8.0, 2.0],
            1e-12,
            100,
        )
        .unwrap();
        assert!((flows[0] - 5.0).abs() < 1e-9, "q0 = {}", flows[0]);
        assert!((flows[1] - 5.0).abs() < 1e-9, "q1 = {}", flows[1]);
    }

    #[test]
    fn asymmetric_parallel_pipes_analytic_solution() {
        // r₁ = 1, r₂ = 4, n = 2: equal head loss ⇒ q₁ = 2 q₂; with
        // q₁ + q₂ = 9 the exact split is 6 / 3.
        let pipes = [
            NetworkPipe::new(1.0, 2.0).unwrap(),
            NetworkPipe::new(4.0, 2.0).unwrap(),
        ];
        let flows = hardy_cross(
            &pipes,
            &[vec![(0, 1.0), (1, -1.0)]],
            &[8.0, 1.0],
            1e-12,
            100,
        )
        .unwrap();
        assert!((flows[0] - 6.0).abs() < 1e-9, "q0 = {}", flows[0]);
        assert!((flows[1] - 3.0).abs() < 1e-9, "q1 = {}", flows[1]);
        // Total flow (continuity) is preserved exactly by loop updates.
        assert!((flows[0] + flows[1] - 9.0).abs() < 1e-12);
        // Head losses balance.
        assert!((pipes[0].head_loss(flows[0]) - pipes[1].head_loss(flows[1])).abs() < 1e-8);
    }

    #[test]
    fn three_parallel_pipes_two_loops() {
        // r = 1, 2, 4 (n = 2): equal head ⇒ q₁ = 2q₃, q₂ = √2 q₃.
        // Total (2 + √2 + 1) q₃; pick q₃ = 1.
        let total = 3.0 + std::f64::consts::SQRT_2;
        let pipes = [
            NetworkPipe::new(1.0, 2.0).unwrap(),
            NetworkPipe::new(2.0, 2.0).unwrap(),
            NetworkPipe::new(4.0, 2.0).unwrap(),
        ];
        let loops = [vec![(0, 1.0), (1, -1.0)], vec![(1, 1.0), (2, -1.0)]];
        let flows = hardy_cross(&pipes, &loops, &[total - 0.2, 0.1, 0.1], 1e-13, 500).unwrap();
        assert!((flows[0] - 2.0).abs() < 1e-8, "q0 = {}", flows[0]);
        assert!(
            (flows[1] - std::f64::consts::SQRT_2).abs() < 1e-8,
            "q1 = {}",
            flows[1]
        );
        assert!((flows[2] - 1.0).abs() < 1e-8, "q2 = {}", flows[2]);
        // Loop closures vanish at the solution.
        for lp in &loops
        {
            let closure: f64 = lp
                .iter()
                .map(|&(i, s)| s * pipes[i].head_loss(flows[i]))
                .sum();
            assert!(closure.abs() < 1e-7, "loop residual = {closure}");
        }
    }

    #[test]
    fn hazen_williams_exponent_works() {
        // n = 1.852, identical pipes: still an even split.
        let p = NetworkPipe::new(10.0, 1.852).unwrap();
        let flows = hardy_cross(
            &[p, p],
            &[vec![(0, 1.0), (1, -1.0)]],
            &[0.9, 0.1],
            1e-12,
            200,
        )
        .unwrap();
        assert!((flows[0] - 0.5).abs() < 1e-9);
        assert!((flows[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn flow_reversal_is_handled() {
        // Start with a wrong guess whose loop correction must reverse
        // pipe 1's direction: signed head-loss law keeps it consistent.
        let pipes = [
            NetworkPipe::new(1.0, 2.0).unwrap(),
            NetworkPipe::new(1.0, 2.0).unwrap(),
        ];
        let flows = hardy_cross(
            &pipes,
            &[vec![(0, 1.0), (1, 1.0)]],
            // Loop of two pipes traversed in the same direction: the
            // closure forces q0 = -q1 (circulation dies out).
            &[2.0, -6.0],
            1e-12,
            200,
        )
        .unwrap();
        assert!((flows[0] + flows[1]).abs() < 1e-9, "{flows:?}");
    }

    #[test]
    fn rejects_invalid_networks() {
        let p = NetworkPipe::new(1.0, 2.0).unwrap();
        // Bad pipe index.
        assert!(hardy_cross(&[p], &[vec![(3, 1.0)]], &[1.0], 1e-9, 10).is_err());
        // Bad orientation.
        assert!(hardy_cross(&[p, p], &[vec![(0, 0.5), (1, -1.0)]], &[1.0, 1.0], 1e-9, 10).is_err());
        // Length mismatch.
        assert!(hardy_cross(&[p, p], &[vec![(0, 1.0), (1, -1.0)]], &[1.0], 1e-9, 10).is_err());
        // Degenerate all-zero loop.
        assert!(hardy_cross(&[p, p], &[vec![(0, 1.0), (1, -1.0)]], &[0.0, 0.0], 1e-9, 10).is_err());
        // Bad pipe law.
        assert!(NetworkPipe::new(0.0, 2.0).is_err());
        assert!(NetworkPipe::new(1.0, 5.0).is_err());
    }

    // ---------------------------------------------------------------- //
    //  hardy_cross_darcy: real pipes coupled to friction_factor.       //
    // ---------------------------------------------------------------- //

    const WATER_RHO: f64 = 998.0;
    const WATER_MU: f64 = 1.002e-3;
    const G: f64 = 9.81;

    #[test]
    fn darcy_symmetric_pipes_split_evenly() {
        // Two physically identical pipes must split any total flow
        // evenly, whatever their (equal, unknown a priori) friction
        // factor turns out to be.
        let p = PhysicalPipe::new(0.10, 100.0, 4.5e-5).unwrap();
        let flows = hardy_cross_darcy(
            &[p, p],
            &[vec![(0, 1.0), (1, -1.0)]],
            &[0.08, 0.02],
            WATER_RHO,
            WATER_MU,
            G,
            1e-10,
            60,
            200,
        )
        .unwrap();
        assert!((flows[0] - 0.05).abs() < 1e-6, "q0 = {}", flows[0]);
        assert!((flows[1] - 0.05).abs() < 1e-6, "q1 = {}", flows[1]);
    }

    #[test]
    fn darcy_wider_pipe_carries_more_flow_and_closes_physically() {
        // A wider parallel pipe (same length/roughness) is less
        // resistant and must carry the larger share; continuity must be
        // preserved, and the converged flows must satisfy the REAL
        // Darcy-Weisbach head-loss balance (not just the frozen-r
        // balance of a single outer step).
        let wide = PhysicalPipe::new(0.10, 100.0, 4.5e-5).unwrap();
        let narrow = PhysicalPipe::new(0.05, 100.0, 4.5e-5).unwrap();
        let total = 0.10;
        let flows = hardy_cross_darcy(
            &[wide, narrow],
            &[vec![(0, 1.0), (1, -1.0)]],
            &[total * 0.5, total * 0.5],
            WATER_RHO,
            WATER_MU,
            G,
            1e-10,
            60,
            200,
        )
        .unwrap();
        assert!(
            flows[0] > flows[1],
            "wide={}, narrow={}",
            flows[0],
            flows[1]
        );
        assert!((flows[0] + flows[1] - total).abs() < 1e-8, "{flows:?}");

        // Recompute each pipe's head loss from first principles at the
        // converged flow and check the loop closes.
        let head_loss = |p: &PhysicalPipe, q: f64| -> f64 {
            let area = p.area();
            let speed = q.abs() / area;
            let re = (WATER_RHO * speed * p.diameter / WATER_MU).max(1e-9);
            let f = crate::pipe::friction_factor(re, p.roughness / p.diameter).unwrap();
            f * p.length / p.diameter * speed * speed / (2.0 * G)
        };
        let h0 = head_loss(&wide, flows[0]);
        let h1 = head_loss(&narrow, flows[1]);
        assert!((h0 - h1).abs() / h0.max(h1) < 1e-6, "h0={h0}, h1={h1}");
    }

    #[test]
    fn darcy_rejects_invalid_pipes_and_inputs() {
        assert!(PhysicalPipe::new(0.0, 100.0, 1e-5).is_err());
        assert!(PhysicalPipe::new(0.1, 100.0, -1e-5).is_err());
        assert!(PhysicalPipe::new(0.1, 100.0, 0.02).is_err()); // eps/D = 0.2 > 0.1
        let p = PhysicalPipe::new(0.1, 100.0, 4.5e-5).unwrap();
        assert!(
            hardy_cross_darcy(
                &[p, p],
                &[vec![(0, 1.0), (1, -1.0)]],
                &[1.0],
                WATER_RHO,
                WATER_MU,
                G,
                1e-9,
                10,
                50
            )
            .is_err()
        );
        assert!(
            hardy_cross_darcy(
                &[p],
                &[vec![(0, 1.0)]],
                &[1.0],
                -1.0,
                WATER_MU,
                G,
                1e-9,
                10,
                50
            )
            .is_err()
        );
        assert!(
            hardy_cross_darcy(
                &[p],
                &[vec![(0, 1.0)]],
                &[1.0],
                WATER_RHO,
                WATER_MU,
                G,
                1e-9,
                0,
                50
            )
            .is_err()
        );
    }
}
