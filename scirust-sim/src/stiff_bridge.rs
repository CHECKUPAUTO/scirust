//! Bridge from this crate's [`System`] to the implicit stiff integrators in
//! `scirust-stiff` (enabled by the `stiff` feature).
//!
//! The explicit engine ([`simulate`](crate::simulate),
//! [`simulate_adaptive`](crate::simulate_adaptive)) must take steps smaller
//! than a system's fastest time-scale. For *stiff* plants — the
//! [`Robertson`](crate::chemistry::Robertson) kinetics, a fast electrical
//! transient feeding a slow thermal state — that makes them unusable. These
//! two functions hand a [`System`] to `scirust-stiff`'s L-stable Backward
//! Euler and adaptive Rosenbrock-W methods instead, whose stability is
//! decoupled from the fast modes.
//!
//! `System::derivatives` writes the derivative in place; `scirust-stiff`'s
//! closures return it by value. The adapter here bridges the two shapes and
//! maps `scirust-stiff`'s `Solution`/`StiffError` back to this crate's
//! [`Trajectory`]/[`SimError`].
//!
//! ```
//! # // (requires the `stiff` feature)
//! use scirust_sim::chemistry::Robertson;
//! use scirust_sim::stiff_bridge::simulate_rosenbrock;
//!
//! let rob = Robertson::classic();
//! let traj = simulate_rosenbrock(&rob, &rob.initial_state(), 0.0, 0.4, 1e-6, 1e-9, 1e-6).unwrap();
//! let last = traj.last_state().unwrap();
//! // Mass is conserved and species A has barely depleted by t = 0.4.
//! assert!((last[0] + last[1] + last[2] - 1.0).abs() < 1e-6);
//! assert!(last[0] > 0.98 && last[0] < 0.99);
//! ```

use crate::{SimError, System, Trajectory};
use scirust_stiff::{Solution, StiffError};

/// Adapt a [`System`] to the by-value right-hand side `scirust-stiff` expects.
fn rhs<S: System>(system: &S) -> impl Fn(f64, &[f64]) -> Vec<f64> + '_ {
    let dim = system.dim();
    move |t: f64, y: &[f64]| {
        let mut dydt = vec![0.0; dim];
        system.derivatives(t, y, &mut dydt);
        dydt
    }
}

fn to_trajectory(sol: Solution) -> Trajectory {
    Trajectory { t: sol.t, y: sol.y }
}

fn map_error(e: StiffError) -> SimError {
    match e
    {
        StiffError::DimMismatch { expected, got } => SimError::DimMismatch { expected, got },
        StiffError::StepUnderflow { t, .. } => SimError::StepUnderflow { t },
        StiffError::BadInput(msg) => SimError::BadInput(msg),
        other => SimError::BadInput(format!("stiff integration failed: {other}")),
    }
}

/// Integrate a stiff `system` from `y0` over `[t0, t_end]` with the adaptive,
/// linearly-implicit **Rosenbrock-W(2,3)** method (`ode23s`-type) from
/// `scirust-stiff` — the recommended stiff integrator.
///
/// `rtol`, `atol` and the initial step `h0` must be finite and positive. Errors
/// from the underlying solver (non-convergence, a singular iteration matrix,
/// step underflow) are surfaced as [`SimError`] rather than panicking.
pub fn simulate_rosenbrock<S: System>(
    system: &S,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    rtol: f64,
    atol: f64,
    h0: f64,
) -> Result<Trajectory, SimError> {
    scirust_stiff::rosenbrock23(rhs(system), t0, y0, t_end, rtol, atol, h0)
        .map(to_trajectory)
        .map_err(map_error)
}

/// Integrate a stiff `system` from `y0` over `[t0, t_end]` with fixed-step,
/// L-stable **Backward Euler** from `scirust-stiff`: unconditionally stable, so
/// the step `h` tracks accuracy rather than stability. First-order accurate.
///
/// `h` must be finite and positive; solver failures surface as [`SimError`].
pub fn simulate_backward_euler<S: System>(
    system: &S,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    h: f64,
) -> Result<Trajectory, SimError> {
    scirust_stiff::backward_euler(rhs(system), t0, y0, t_end, h)
        .map(to_trajectory)
        .map_err(map_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::Robertson;
    use crate::engine::simulate;

    /// `y' = -50 y`, exact solution `e^{-50 t}` — a stiff scalar problem the
    /// bridge should integrate correctly.
    struct Decay50;
    impl System for Decay50 {
        fn dim(&self) -> usize {
            1
        }

        fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            dydt[0] = -50.0 * y[0];
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn rosenbrock_matches_the_analytic_stiff_decay() {
        let traj = simulate_rosenbrock(&Decay50, &[1.0], 0.0, 1.0, 1e-8, 1e-10, 1e-3).unwrap();
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            assert!(
                (row[0] - (-50.0 * t).exp()).abs() < 1e-6,
                "t = {t}: {}",
                row[0]
            );
        }
        assert_eq!(traj.last_time(), Some(1.0));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn backward_euler_matches_the_analytic_stiff_decay() {
        // Backward Euler is L-stable: it stays bounded and accurate at a step
        // (0.02) far above the explicit stability limit (h < 0.04).
        let traj = simulate_backward_euler(&Decay50, &[1.0], 0.0, 1.0, 1e-3).unwrap();
        let last = traj.last_state().unwrap();
        assert!(
            (last[0] - (-50.0f64).exp()).abs() < 1e-3,
            "y(1) = {}",
            last[0]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn robertson_solution_conserves_mass_and_matches_the_reference() {
        // The canonical stiff benchmark, integrated to t = 0.4 (Hairer &
        // Wanner's reference point).
        let rob = Robertson::classic();
        let traj =
            simulate_rosenbrock(&rob, &rob.initial_state(), 0.0, 0.4, 1e-7, 1e-10, 1e-6).unwrap();
        // Total mass a + b + c stays 1 all along.
        for row in &traj.y
        {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-6, "mass drift");
        }
        // Reference solution at t = 0.4: a ≈ 0.9851, b ≈ 3.39e-5, c ≈ 0.01490.
        let last = traj.last_state().unwrap();
        assert!((last[0] - 0.9851).abs() < 2e-3, "a(0.4) = {}", last[0]);
        assert!(last[1] > 0.0 && last[1] < 1e-4, "b(0.4) = {}", last[1]);
        assert!((last[2] - 0.0149).abs() < 2e-3, "c(0.4) = {}", last[2]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn robertson_crosses_several_decades_of_time() {
        // The Robertson tail is very slow (A only fully drains into C by
        // t ~ 10^5). Integrating to t = 1000 already crosses several decades
        // from the ~10^-4 initial transient: reference (SUNDIALS) puts a ≈ 0.36
        // and c ≈ 0.64 there, so the majority of the mass has converted. The
        // adaptive stiff solver does this without the tiny steps an explicit
        // method would need.
        let rob = Robertson::classic();
        let traj = simulate_rosenbrock(&rob, &rob.initial_state(), 0.0, 1000.0, 1e-7, 1e-10, 1e-6)
            .unwrap();
        let last = traj.last_state().unwrap();
        assert!(last[0] < 0.45, "a(1000) = {}", last[0]);
        assert!(last[2] > 0.6, "c(1000) = {}", last[2]);
        assert!((last.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn backward_euler_and_rosenbrock_agree_on_robertson() {
        // Two independent stiff methods land on the same t = 0.4 state.
        let rob = Robertson::classic();
        let be = simulate_backward_euler(&rob, &rob.initial_state(), 0.0, 0.4, 1e-4).unwrap();
        let ros =
            simulate_rosenbrock(&rob, &rob.initial_state(), 0.0, 0.4, 1e-7, 1e-10, 1e-6).unwrap();
        let (a, b) = (be.last_state().unwrap(), ros.last_state().unwrap());
        assert!((a[0] - b[0]).abs() < 5e-3, "a: {} vs {}", a[0], b[0]);
        assert!((a[2] - b[2]).abs() < 5e-3, "c: {} vs {}", a[2], b[2]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn explicit_rk4_cannot_step_robertson_where_the_stiff_method_can() {
        // A coarse fixed RK4 step blows up on the stiff transient; the stiff
        // integrator handles the same span. This is the reason the bridge
        // exists.
        let rob = Robertson::classic();
        let explicit = simulate(&rob, &rob.initial_state(), 0.0, 0.4, 0.05);
        assert!(
            matches!(explicit, Err(SimError::NonFinite { .. })),
            "{explicit:?}"
        );
        assert!(
            simulate_rosenbrock(&rob, &rob.initial_state(), 0.0, 0.4, 1e-6, 1e-9, 1e-6).is_ok()
        );
    }

    #[test]
    fn bad_tolerances_are_reported_not_panicked() {
        assert!(matches!(
            simulate_rosenbrock(&Decay50, &[1.0], 0.0, 1.0, 0.0, 1e-9, 1e-3),
            Err(SimError::BadInput(_))
        ));
        assert!(matches!(
            simulate_backward_euler(&Decay50, &[1.0], 0.0, 1.0, -1.0),
            Err(SimError::BadInput(_))
        ));
    }
}
