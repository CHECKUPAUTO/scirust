//! Discrete-time infinite-horizon LQR via the Riccati recursion.

use scirust_estimation::Mat;

/// Solve the discrete LQR problem: find the gain `K` minimising
/// `Σ xₖᵀQxₖ + uₖᵀRuₖ` for `x_{k+1} = A·xₖ + B·uₖ`, by iterating the Riccati
/// recursion to convergence. The control law is `u = −K·x`. Returns `None` if a
/// required inverse is singular or convergence is not reached.
pub fn dlqr(a: &Mat, b: &Mat, q: &Mat, r: &Mat) -> Option<Mat> {
    let at = a.t();
    let bt = b.t();
    let mut p = q.clone();
    let mut converged = false;
    for _ in 0..2000
    {
        // P⁺ = Q + AᵀPA − AᵀPB (R + BᵀPB)⁻¹ BᵀPA
        let bpb = r.add(&bt.matmul(&p).matmul(b));
        let inv = bpb.inverse()?;
        let atpa = at.matmul(&p).matmul(a);
        let atpb = at.matmul(&p).matmul(b);
        let btpa = bt.matmul(&p).matmul(a);
        let p_next = q.add(&atpa).sub(&atpb.matmul(&inv).matmul(&btpa));
        // Convergence by max element change.
        let delta = p_next
            .data
            .iter()
            .zip(&p.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        p = p_next;
        if delta < 1e-10
        {
            converged = true;
            break;
        }
    }
    if !converged
    {
        return None;
    }
    // K = (R + BᵀPB)⁻¹ BᵀPA
    let bpb = r.add(&bt.matmul(&p).matmul(b));
    Some(bpb.inverse()?.matmul(&bt.matmul(&p).matmul(a)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lqr_stabilizes_an_unstable_scalar_system() {
        // x_{k+1} = 1.2 x + u  (open-loop unstable).
        let a = Mat::new(1, 1, vec![1.2]);
        let b = Mat::new(1, 1, vec![1.0]);
        let q = Mat::new(1, 1, vec![1.0]);
        let r = Mat::new(1, 1, vec![1.0]);
        let k = dlqr(&a, &b, &q, &r).expect("lqr");
        // Closed-loop pole 1.2 - K must be inside the unit circle.
        let pole = 1.2 - k.data[0];
        assert!(pole.abs() < 1.0, "closed-loop pole {pole} not stable");
    }

    #[test]
    fn lqr_drives_a_double_integrator_to_zero() {
        let dt = 0.1;
        // States [pos, vel]; x_{k+1} = A x + B u.
        let a = Mat::new(2, 2, vec![1.0, dt, 0.0, 1.0]);
        let b = Mat::new(2, 1, vec![0.5 * dt * dt, dt]);
        let q = Mat::new(2, 2, vec![1.0, 0.0, 0.0, 0.1]);
        let r = Mat::new(1, 1, vec![0.1]);
        let k = dlqr(&a, &b, &q, &r).expect("lqr");

        // Simulate the closed loop from a displaced state.
        let mut x = vec![1.0, 0.0];
        for _ in 0..400
        {
            let u = -k.matvec(&x)[0];
            let nx = a.matvec(&x);
            let bu = b.matvec(&[u]);
            x = nx.iter().zip(&bu).map(|(a, b)| a + b).collect();
        }
        assert!(
            x[0].abs() < 1e-2 && x[1].abs() < 1e-2,
            "state {x:?} not regulated"
        );
    }
}
