//! Scientific Computing: FEM, CFD, and ODE solvers.

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
    pub fn solve_steady_heat(&self, source_term: f64) -> Vec<f64> {
        let n = self.nodes;
        let h = self.length / (n as f64 - 1.0);

        // Build the stiffness matrix K and load vector F
        // K is tri-diagonal for 1D linear elements
        let mut k = vec![vec![0.0; n]; n];
        let mut f = vec![0.0; n];

        for i in 1..n - 1
        {
            k[i][i - 1] = -1.0 / h;
            k[i][i] = 2.0 / h;
            k[i][i + 1] = -1.0 / h;
            f[i] = source_term * h;
        }

        // Boundary conditions (enforced by setting identity in matrix)
        k[0][0] = 1.0;
        f[0] = 0.0;
        k[n - 1][n - 1] = 1.0;
        f[n - 1] = 0.0;

        // Solve the linear system K * u = f (using Gaussian elimination from scirust-learning/lib.rs logic if it were available here)
        // Since we are in solvers, we'll use a local simple Thomas algorithm for tri-diagonal systems
        self.solve_tridiagonal(k, f)
    }

    fn solve_tridiagonal(&self, k: Vec<Vec<f64>>, f: Vec<f64>) -> Vec<f64> {
        let n = f.len();
        let mut u = vec![0.0; n];
        let mut a = vec![0.0; n]; // sub-diagonal
        let mut b = vec![0.0; n]; // diagonal
        let mut c = vec![0.0; n]; // super-diagonal
        let mut d = f;

        for i in 0..n
        {
            b[i] = k[i][i];
            if i > 0
            {
                a[i] = k[i][i - 1];
            }
            if i < n - 1
            {
                c[i] = k[i][i + 1];
            }
        }

        // Forward sweep
        for i in 1..n
        {
            let m = a[i] / b[i - 1];
            b[i] -= m * c[i - 1];
            d[i] -= m * d[i - 1];
        }

        // Back substitution
        u[n - 1] = d[n - 1] / b[n - 1];
        for i in (0..n - 1).rev()
        {
            u[i] = (d[i] - c[i] * u[i + 1]) / b[i];
        }

        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For `-u'' = f` on `[0, L]` with `u(0) = u(L) = 0` and constant `f`, the
    /// exact solution is the parabola `u(x) = (f/2)·x·(L − x)`. 1D linear FEM is
    /// nodally exact for this operator, so the computed nodal values must match
    /// the analytic ones to round-off.
    #[test]
    fn steady_heat_matches_analytic_parabola() {
        let n = 5;
        let length = 1.0;
        let f = 1.0;
        let fem = FemSolver1D::new(n, length);
        let u = fem.solve_steady_heat(f);
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
        let u = fem.solve_steady_heat(3.0);
        let mid = u.len() / 2;
        for k in 0..mid
        {
            assert!((u[k] - u[u.len() - 1 - k]).abs() < 1e-9);
        }
        assert!(u[mid] >= u[mid - 1] && u[mid] >= u[mid + 1]);
    }
}
