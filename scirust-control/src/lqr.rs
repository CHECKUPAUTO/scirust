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
        // Convergence by max element change, relative to P's own magnitude
        // (plus a small absolute floor for a P that converges near zero):
        // a fixed absolute 1e-10 either stalls for `n·max_iter` on systems
        // where P converges to large entries (e.g. heavily weighted Q, or a
        // system near the stability boundary) — the noise floor of the
        // iteration itself can sit above 1e-10 in that regime — or converges
        // needlessly slowly relative to what double precision can actually
        // resolve. This mirrors the `tol.abs + tol.rel * scale` convention
        // used by the iterative solvers in scirust-solvers.
        let delta = p_next
            .data
            .iter()
            .zip(&p.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let p_scale = p_next.data.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        p = p_next;
        if delta < 1e-10 + 1e-10 * p_scale
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
    fn lqr_converges_for_weak_authority_heavily_weighted_scalar_system() {
        // Regression test for a P1 audit finding: the Riccati fixed-point
        // iteration used a fixed absolute convergence threshold (1e-10 on
        // the max element change), which either stalls for the full 2000
        // iterations (returning None on a solvable problem) or converges far
        // more slowly than necessary once P settles at a large magnitude —
        // an absolute 1e-10 change is a vastly tighter *relative* tolerance
        // at P ~ 1e10 than the arithmetic can usefully resolve. This
        // (a, b, q) triple — weak control authority, heavily weighted state
        // cost — never satisfies the old absolute-only criterion within
        // 2000 iterations (verified empirically), even though the Riccati
        // recursion has already settled after the very first step under a
        // relative criterion.
        let a = Mat::new(1, 1, vec![1.5]);
        let b = Mat::new(1, 1, vec![0.001]);
        let q = Mat::new(1, 1, vec![1e10]);
        let r = Mat::new(1, 1, vec![1.0]);
        let k = dlqr(&a, &b, &q, &r).expect("DARE must converge for this solvable system");
        let pole = 1.5 - b.data[0] * k.data[0];
        assert!(pole.abs() < 1.0, "closed-loop pole {pole} not stable");
    }

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
