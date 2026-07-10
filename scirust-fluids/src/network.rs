//! Looped pipe networks — the **Hardy Cross** method.
//!
//! A network is described by its pipes (each with a head-loss law
//! `h = r·|Q|^{n−1}·Q`, `n = 2` for Darcy–Weisbach, `n = 1.852` for
//! Hazen–Williams), a set of closed loops (signed pipe memberships) and
//! an initial flow distribution that already satisfies continuity at
//! every node. Hardy Cross iteratively corrects each loop's flows until
//! the head loss around every loop closes; loop corrections preserve
//! node continuity exactly, by construction.

use crate::error::{FluidsError, finite, in_range, positive};

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
}
