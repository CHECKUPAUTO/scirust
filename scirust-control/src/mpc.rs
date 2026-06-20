//! Condensed linear Model Predictive Control with hard input bounds.
//!
//! Regulates `x_{k+1} = A·xₖ + B·uₖ` to a reference by minimising a finite-
//! horizon quadratic cost subject to box input constraints, solved each step as
//! a box-QP ([`crate::qp`]). Because the QP projects onto the actuator box, the
//! applied input is **always feasible by construction** — constraint
//! satisfaction is certified, not merely penalised.

use crate::qp::solve_box_qp;
use scirust_estimation::Mat;

/// `A^k` (with `A^0 = I`).
fn mat_pow(a: &Mat, k: usize) -> Mat {
    let mut m = Mat::identity(a.rows);
    for _ in 0..k
    {
        m = m.matmul(a);
    }
    m
}

/// Block-diagonal matrix repeating `block` `count` times.
fn blkdiag(block: &Mat, count: usize) -> Mat {
    let (br, bc) = (block.rows, block.cols);
    let mut out = Mat::zeros(br * count, bc * count);
    for t in 0..count
    {
        for i in 0..br
        {
            for j in 0..bc
            {
                out.set(t * br + i, t * bc + j, block.get(i, j));
            }
        }
    }
    out
}

/// Linear MPC controller.
pub struct LinearMpc {
    a: Mat,
    b: Mat,
    horizon: usize,
    n: usize,
    m: usize,
    u_min: f64,
    u_max: f64,
    phi: Mat,
    qbar_gamma: Mat, // Q̄·Γ  (reused in g)
    h: Mat,          // 2(Γ'Q̄Γ + R̄)
}

impl LinearMpc {
    /// Build for system `(A, B)`, stage weights `Q` (n×n), `R` (m×m), prediction
    /// `horizon`, and scalar input bounds applied to every channel.
    pub fn new(a: Mat, b: Mat, q: Mat, r: Mat, horizon: usize, u_min: f64, u_max: f64) -> Self {
        let n = a.rows;
        let m = b.cols;
        let nn = n * horizon;
        let mm = m * horizon;

        // Φ = [A; A²; …; A^N]
        let mut phi = Mat::zeros(nn, n);
        for i in 1..=horizon
        {
            let ap = mat_pow(&a, i);
            for r0 in 0..n
            {
                for c0 in 0..n
                {
                    phi.set((i - 1) * n + r0, c0, ap.get(r0, c0));
                }
            }
        }
        // Γ: block (i,j) = A^(i-1-j)·B for j ≤ i-1.
        let mut gamma = Mat::zeros(nn, mm);
        for i in 1..=horizon
        {
            for j in 0..i
            {
                let ab = mat_pow(&a, i - 1 - j).matmul(&b);
                for r0 in 0..n
                {
                    for c0 in 0..m
                    {
                        gamma.set((i - 1) * n + r0, j * m + c0, ab.get(r0, c0));
                    }
                }
            }
        }
        let qbar = blkdiag(&q, horizon);
        let rbar = blkdiag(&r, horizon);
        let qbar_gamma = qbar.matmul(&gamma);
        let base = gamma.t().matmul(&qbar_gamma).add(&rbar);
        let h = base.add(&base); // 2(Γ'Q̄Γ + R̄)
        Self {
            a,
            b,
            horizon,
            n,
            m,
            u_min,
            u_max,
            phi,
            qbar_gamma,
            h,
        }
    }

    /// Compute the control move for the current state `x0` and constant
    /// reference `x_ref` (length n). Returns the first input (length m).
    pub fn control(&self, x0: &[f64], x_ref: &[f64]) -> Vec<f64> {
        let nn = self.n * self.horizon;
        // Φ x0 − Xref  (stacked reference).
        let phi_x = self.phi.matvec(x0);
        let mut e = vec![0.0; nn];
        for i in 0..self.horizon
        {
            for r0 in 0..self.n
            {
                e[i * self.n + r0] = phi_x[i * self.n + r0] - x_ref[r0];
            }
        }
        // g = 2 (Q̄Γ)ᵀ e
        let g: Vec<f64> = {
            let gt = self.qbar_gamma.t();
            gt.matvec(&e).iter().map(|v| 2.0 * v).collect()
        };
        let mm = self.m * self.horizon;
        let lb = vec![self.u_min; mm];
        let ub = vec![self.u_max; mm];
        let u_seq = solve_box_qp(&self.h, &g, &lb, &ub, 600);
        u_seq[..self.m].to_vec()
    }

    /// Convenience: advance the true plant one step under the MPC law.
    pub fn closed_loop_step(&self, x: &[f64], x_ref: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let u = self.control(x, x_ref);
        let ax = self.a.matvec(x);
        let bu = self.b.matvec(&u);
        let x_next: Vec<f64> = ax.iter().zip(&bu).map(|(a, b)| a + b).collect();
        (u, x_next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpc_regulates_to_setpoint_within_input_bounds() {
        let dt = 0.1;
        let a = Mat::new(2, 2, vec![1.0, dt, 0.0, 1.0]); // double integrator
        let b = Mat::new(2, 1, vec![0.5 * dt * dt, dt]);
        let q = Mat::new(2, 2, vec![10.0, 0.0, 0.0, 1.0]);
        let r = Mat::new(1, 1, vec![0.1]);
        let mpc = LinearMpc::new(a, b, q, r, 15, -0.5, 0.5);

        let x_ref = [1.0, 0.0];
        let mut x = vec![0.0, 0.0];
        for _ in 0..600
        {
            let (u, nx) = mpc.closed_loop_step(&x, &x_ref);
            assert!(
                u[0] >= -0.5 - 1e-9 && u[0] <= 0.5 + 1e-9,
                "input {u:?} out of bounds"
            );
            x = nx;
        }
        assert!(
            (x[0] - 1.0).abs() < 0.05,
            "position {} not at setpoint",
            x[0]
        );
        assert!(x[1].abs() < 0.05, "velocity {} not settled", x[1]);
    }
}
