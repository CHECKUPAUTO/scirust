//! Scientific computing: 1-D steady-state heat FEM solver.
//! (ODE solvers live in the `ode` module.)

use crate::{SolverError, SolverResult};

/// Pivot-rejection threshold `n · eps · max|entry|` (Golub & Van Loan,
/// *Matrix Computations*, §3.4.6) — relative to the system's own scale, like
/// the dense LU guard, so a well-posed system at a small physical scale is
/// not declared singular while a genuinely zero pivot is caught.
fn pivot_tol(n: usize, max_abs: f64) -> f64 {
    (n as f64) * f64::EPSILON * max_abs.max(1e-300)
}

/// Rejects NaN/Inf scalars — same contract as the hardened linalg solvers.
fn check_finite(value: f64, _label: &str) -> SolverResult<()> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value });
    }
    Ok(())
}

/// Finite Element Method (FEM) solver for 1D problems.
pub struct FemSolver1D {
    pub nodes: usize,
    pub length: f64,
}

impl FemSolver1D {
    pub fn new(nodes: usize, length: f64) -> Self {
        Self { nodes, length }
    }

    /// Solve the 1D steady-state heat equation: -d^2u/dx^2 = f
    /// with boundary conditions u(0) = 0, u(L) = 0.
    ///
    /// # Errors
    /// - [`SolverError::NanDetected`] if `length` or `source_term` is NaN/Inf,
    ///   or if a non-finite value appears in the assembled system or solution;
    /// - [`SolverError::InvalidInput`] if `length == 0` with ≥ 2 nodes (the
    ///   element size `h` would vanish);
    /// - [`SolverError::Singular`] if a Thomas pivot falls below the
    ///   scale-relative threshold (degenerate stiffness matrix).
    pub fn solve_steady_heat(&self, source_term: f64) -> SolverResult<Vec<f64>> {
        check_finite(self.length, "length")?;
        check_finite(source_term, "source_term")?;

        let n = self.nodes;
        // Degenerate meshes are returned trivially rather than panicking: with 0
        // nodes there is nothing to solve, and with 1 node the single node is a
        // fixed boundary at 0. A real 1-D FEM mesh needs ≥ 2 nodes for `h` to be
        // finite and the interior loop `1..n-1` not to underflow.
        if n == 0
        {
            return Ok(Vec::new());
        }
        if n == 1
        {
            return Ok(vec![0.0]);
        }
        let h = self.length / (n as f64 - 1.0);
        if h == 0.0
        {
            return Err(SolverError::InvalidInput(
                "length must be nonzero for a mesh with >= 2 nodes".into(),
            ));
        }

        // Build the stiffness matrix K and load vector F.
        // K is tri-diagonal for 1D linear elements, so only the three bands
        // are assembled (no dense n×n storage).
        let mut sub = vec![0.0; n]; // sub[i]  = K[i][i-1]
        let mut diag = vec![0.0; n]; // diag[i] = K[i][i]
        let mut sup = vec![0.0; n]; // sup[i]  = K[i][i+1]
        let mut rhs = vec![0.0; n];

        for i in 1..n - 1
        {
            sub[i] = -1.0 / h;
            diag[i] = 2.0 / h;
            sup[i] = -1.0 / h;
            rhs[i] = source_term * h;
        }

        // Boundary conditions (enforced by setting identity in matrix)
        diag[0] = 1.0;
        rhs[0] = 0.0;
        diag[n - 1] = 1.0;
        rhs[n - 1] = 0.0;

        // Solve the linear system K * u = f with the Thomas algorithm — the
        // right tool for a tri-diagonal system (O(n) instead of O(n³)).
        solve_tridiagonal(&sub, diag, &sup, rhs)
    }
}

/// Thomas algorithm for a tri-diagonal system given as bands
/// (`sub[i] = K[i][i-1]`, `diag[i] = K[i][i]`, `sup[i] = K[i][i+1]`).
/// Consumes `diag` and `rhs` (they are mutated by the forward sweep).
///
/// Every pivot is guarded with the scale-relative threshold from
/// [`pivot_tol`]; a pivot at or below it returns [`SolverError::Singular`]
/// instead of silently producing inf/NaN. Inputs and outputs are NaN-checked.
fn solve_tridiagonal(
    sub: &[f64],
    mut diag: Vec<f64>,
    sup: &[f64],
    mut rhs: Vec<f64>,
) -> SolverResult<Vec<f64>> {
    let n = rhs.len();
    debug_assert_eq!(sub.len(), n);
    debug_assert_eq!(diag.len(), n);
    debug_assert_eq!(sup.len(), n);
    if n == 0
    {
        return Ok(Vec::new());
    }

    // NaN-check the assembled system and record its scale for the pivot guard.
    let mut max_abs = 0.0_f64;
    for i in 0..n
    {
        check_finite(sub[i], "sub")?;
        check_finite(diag[i], "diag")?;
        check_finite(sup[i], "sup")?;
        check_finite(rhs[i], "rhs")?;
        max_abs = max_abs
            .max(sub[i].abs())
            .max(diag[i].abs())
            .max(sup[i].abs());
    }
    let tol = pivot_tol(n, max_abs);

    // Forward sweep
    for i in 1..n
    {
        let pivot = diag[i - 1];
        if pivot.abs() <= tol
        {
            return Err(SolverError::Singular { row: i - 1, pivot });
        }
        let m = sub[i] / pivot;
        diag[i] -= m * sup[i - 1];
        rhs[i] -= m * rhs[i - 1];
    }
    let last = diag[n - 1];
    if last.abs() <= tol
    {
        return Err(SolverError::Singular {
            row: n - 1,
            pivot: last,
        });
    }

    // Back substitution (all pivots `diag[i]` were checked above).
    let mut u = vec![0.0; n];
    u[n - 1] = rhs[n - 1] / last;
    for i in (0..n - 1).rev()
    {
        u[i] = (rhs[i] - sup[i] * u[i + 1]) / diag[i];
    }

    for &ui in &u
    {
        check_finite(ui, "u")?;
    }
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For `-u'' = f` on `[0, L]` with `u(0) = u(L) = 0` and constant `f`, the
    /// exact solution is the parabola `u(x) = (f/2)·x·(L − x)`. 1D linear FEM is
    /// nodally exact for this operator, so the computed nodal values must match
    /// the analytic ones to round-off.
    // Degenerate meshes must not panic (0 or 1 node → trivial solution).
    #[test]
    fn steady_heat_handles_degenerate_node_counts() {
        assert!(
            FemSolver1D::new(0, 1.0)
                .solve_steady_heat(1.0)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            FemSolver1D::new(1, 1.0).solve_steady_heat(1.0).unwrap(),
            vec![0.0]
        );
        // 2 nodes = both boundaries, still no panic.
        let u2 = FemSolver1D::new(2, 1.0).solve_steady_heat(1.0).unwrap();
        assert_eq!(u2.len(), 2);
        assert!(u2.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn steady_heat_matches_analytic_parabola() {
        let n = 5;
        let length = 1.0;
        let f = 1.0;
        let fem = FemSolver1D::new(n, length);
        let u = fem.solve_steady_heat(f).unwrap();
        assert_eq!(u.len(), n);
        let h = length / (n as f64 - 1.0);
        for (i, &ui) in u.iter().enumerate()
        {
            let x = i as f64 * h;
            let exact = 0.5 * f * x * (length - x);
            assert!(
                (ui - exact).abs() < 1e-9,
                "node {i}: got {ui}, expected {exact}"
            );
        }
        // Boundary conditions enforced exactly.
        assert_eq!(u[0], 0.0);
        assert_eq!(u[n - 1], 0.0);
    }

    /// The displacement is symmetric about the midpoint and peaks there.
    #[test]
    fn steady_heat_is_symmetric_and_peaks_at_center() {
        let fem = FemSolver1D::new(7, 2.0);
        let u = fem.solve_steady_heat(3.0).unwrap();
        let mid = u.len() / 2;
        for k in 0..mid
        {
            assert!((u[k] - u[u.len() - 1 - k]).abs() < 1e-9);
        }
        assert!(u[mid] >= u[mid - 1] && u[mid] >= u[mid + 1]);
    }

    /// A zero pivot must be reported as `Singular`, never as silent inf/NaN.
    #[test]
    fn tridiagonal_zero_pivot_returns_singular() {
        // First pivot is exactly zero.
        let r = solve_tridiagonal(&[0.0, 1.0], vec![0.0, 1.0], &[1.0, 0.0], vec![1.0, 1.0]);
        assert!(
            matches!(r, Err(SolverError::Singular { row: 0, .. })),
            "expected Singular at row 0, got {r:?}"
        );

        // Elimination cancels the last pivot: diag[1] - (sub[1]/diag[0])·sup[0]
        // = 1 - 1·1 = 0.
        let r = solve_tridiagonal(&[0.0, 1.0], vec![1.0, 1.0], &[1.0, 0.0], vec![1.0, 1.0]);
        assert!(
            matches!(r, Err(SolverError::Singular { row: 1, .. })),
            "expected Singular at row 1, got {r:?}"
        );
    }

    /// A pivot tiny relative to the system scale is near-singular too.
    #[test]
    fn tridiagonal_tiny_pivot_relative_to_scale_is_singular() {
        // Scale is 1e10, pivot 1e-9 → far below n·eps·max|a| ≈ 6.7e-6.
        let r = solve_tridiagonal(
            &[0.0, 1e10, 0.0],
            vec![1e-9, 1e10, 1e10],
            &[1e10, 1e10, 0.0],
            vec![1.0, 1.0, 1.0],
        );
        assert!(matches!(r, Err(SolverError::Singular { row: 0, .. })));
    }

    /// NaN/Inf inputs are rejected with a typed error, not propagated.
    #[test]
    fn steady_heat_rejects_non_finite_inputs() {
        let r = FemSolver1D::new(5, 1.0).solve_steady_heat(f64::NAN);
        assert!(matches!(r, Err(SolverError::NanDetected { .. })));

        let r = FemSolver1D::new(5, f64::INFINITY).solve_steady_heat(1.0);
        assert!(matches!(r, Err(SolverError::NanDetected { .. })));

        // Zero length with >= 2 nodes → h = 0 → typed InvalidInput.
        let r = FemSolver1D::new(5, 0.0).solve_steady_heat(1.0);
        assert!(matches!(r, Err(SolverError::InvalidInput(_))));
    }
}
