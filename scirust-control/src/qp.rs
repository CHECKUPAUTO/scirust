//! Box-constrained convex QP by projected gradient descent.
//!
//! Solves `min ½ uᵀH u + gᵀu  s.t.  lb ≤ u ≤ ub` for symmetric positive-(semi)
//! definite `H`. Used by the MPC layer; the box projection guarantees the
//! returned input is always feasible (within the actuator limits) by
//! construction — the certified part of "certified constraint satisfaction".

use scirust_estimation::Mat;

/// Upper bound on the spectral radius of `H` (Gershgorin: max absolute row sum),
/// used to pick a safe gradient step `1/L`.
fn lipschitz_bound(h: &Mat) -> f64 {
    let mut l = 0.0_f64;
    for i in 0..h.rows
    {
        let row_sum: f64 = (0..h.cols).map(|j| h.get(i, j).abs()).sum();
        l = l.max(row_sum);
    }
    l.max(1e-12)
}

/// Solve the box-constrained QP. `iters` projected-gradient iterations.
pub fn solve_box_qp(h: &Mat, g: &[f64], lb: &[f64], ub: &[f64], iters: usize) -> Vec<f64> {
    let n = g.len();
    let alpha = 1.0 / lipschitz_bound(h);
    let mut u = vec![0.0; n];
    // Start at the projection of the unconstrained-ish origin.
    for (ui, (&l, &up)) in u.iter_mut().zip(lb.iter().zip(ub))
    {
        *ui = 0.0_f64.clamp(l, up);
    }
    for _ in 0..iters
    {
        let hu = h.matvec(&u);
        for i in 0..n
        {
            let grad = hu[i] + g[i];
            u[i] = (u[i] - alpha * grad).clamp(lb[i], ub[i]);
        }
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_qp_matches_clamped_analytic_solution() {
        // min ½(2u0² + 4u1²) + (-3)u0 + (8)u1  -> unconstrained u = [1.5, -2.0].
        let h = Mat::new(2, 2, vec![2.0, 0.0, 0.0, 4.0]);
        let g = [-3.0, 8.0];
        let lb = [-5.0, -5.0];
        let ub = [5.0, 5.0];
        let u = solve_box_qp(&h, &g, &lb, &ub, 2000);
        assert!(
            (u[0] - 1.5).abs() < 1e-3 && (u[1] - (-2.0)).abs() < 1e-3,
            "u {u:?}"
        );
    }

    #[test]
    fn box_constraint_is_always_respected() {
        // Unconstrained optimum at +∞ direction; bound must hold.
        let h = Mat::new(1, 1, vec![1.0]);
        let g = [-100.0]; // wants u huge positive
        let lb = [-2.0];
        let ub = [2.0];
        let u = solve_box_qp(&h, &g, &lb, &ub, 1000);
        assert!((u[0] - 2.0).abs() < 1e-6, "u {u:?} should hit upper bound");
        assert!(u[0] <= 2.0 + 1e-9 && u[0] >= -2.0 - 1e-9);
    }

    #[test]
    fn coupled_qp_converges_to_the_true_minimum() {
        // H = [[2,1],[1,2]], g = [-1,-1] -> u* = H^-1 [1,1] = [1/3, 1/3].
        let h = Mat::new(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let g = [-1.0, -1.0];
        let u = solve_box_qp(&h, &g, &[-10.0, -10.0], &[10.0, 10.0], 5000);
        assert!(
            (u[0] - 1.0 / 3.0).abs() < 1e-3 && (u[1] - 1.0 / 3.0).abs() < 1e-3,
            "u {u:?}"
        );
    }
}
