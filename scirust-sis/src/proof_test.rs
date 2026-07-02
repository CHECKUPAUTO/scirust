//! Proof-test interval sizing: the inverse problem to `Architecture::pfd_avg`
//! — given a target `PFDavg` (from the required SIL band), find the longest
//! proof-test interval `T1` that still meets it. Longer `T1` means cheaper
//! maintenance (fewer proof tests per year) but worse `PFDavg`, so this is
//! the number a process-safety engineer actually wants when sizing a
//! maintenance schedule.

use crate::error::{SisError, SisResult};
use crate::voting::Architecture;
use scirust_solvers::Tolerance;
use scirust_solvers::roots::bisection;

const MAX_BRACKET_DOUBLINGS: usize = 200;

/// Solves for the maximum `T1` (hours) such that `architecture.pfd_avg(
/// lambda_du, T1, beta) <= target_pfd`.
///
/// `PFDavg` is monotonically increasing in `T1` for every architecture in
/// [`crate::voting::Architecture`], so bisection always converges once a
/// bracket is found; a closed form exists for 1oo1/2oo2 (linear in `T1`) but
/// not for the quadratic/cubic 1oo2/2oo3/1oo3 forms, so a numerical root is
/// used uniformly rather than special-casing each architecture.
///
/// Deterministic: the search bracket is found by a fixed-budget doubling
/// loop (`MAX_BRACKET_DOUBLINGS`), and `bisection` itself runs a fixed
/// iteration budget (`Tolerance::default().max_iter`) — no wall-clock or
/// adaptive cutoffs.
pub fn max_proof_test_interval(
    architecture: Architecture,
    lambda_du: f64,
    beta: f64,
    target_pfd: f64,
) -> SisResult<f64> {
    if target_pfd <= 0.0
    {
        return Err(SisError::InvalidInput("target_pfd must be > 0".to_string()));
    }
    if lambda_du <= 0.0
    {
        return Err(SisError::InvalidInput("lambda_du must be > 0".to_string()));
    }
    // Validate the architecture up front so an unsupported one produces a
    // clear error, not an opaque "no sign change" from the bisection below.
    architecture.pfd_avg(lambda_du, 1.0, beta)?;

    let g = |t1: f64| {
        architecture
            .pfd_avg(lambda_du, t1, beta)
            .unwrap_or(f64::INFINITY)
            - target_pfd
    };

    let mut hi = 1.0_f64;
    let mut tries = 0usize;
    while g(hi) < 0.0
    {
        hi *= 2.0;
        tries += 1;
        if tries > MAX_BRACKET_DOUBLINGS
        {
            return Err(SisError::NoBracket {
                target: target_pfd,
                tries,
                last_t1: hi,
                last_pfd: architecture
                    .pfd_avg(lambda_du, hi, beta)
                    .unwrap_or(f64::INFINITY),
            });
        }
    }

    let sol = bisection(g, 0.0, hi, Tolerance::default())
        .map_err(|e| SisError::RootFindingFailed(e.to_string()))?;
    Ok(sol.value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn inverts_1oo1_closed_form() {
        // pfd_1oo1(1e-6, 8760) = 4.38e-3 exactly; the inverse should recover 8760h.
        let t1 = max_proof_test_interval(Architecture::OO1, 1e-6, 0.0, 4.38e-3).unwrap();
        assert_relative_eq!(t1, 8760.0, epsilon = 1e-6);
    }

    #[test]
    fn inverts_1oo2_quadratic_form() {
        // pfd_1oo2(1e-3, 1000, 0.1) = 0.32 exactly (see scirust-reliability tests).
        let t1 = max_proof_test_interval(Architecture::OO2, 1e-3, 0.1, 0.32).unwrap();
        assert_relative_eq!(t1, 1000.0, epsilon = 1e-6);
    }

    #[test]
    fn roundtrips_through_pfd_avg() {
        let architecture = Architecture::TWO_OO3;
        let (lambda_du, beta, target) = (2e-6, 0.05, 1e-3);
        let t1 = max_proof_test_interval(architecture, lambda_du, beta, target).unwrap();
        let pfd = architecture.pfd_avg(lambda_du, t1, beta).unwrap();
        assert_relative_eq!(pfd, target, epsilon = 1e-6);
    }

    #[test]
    fn rejects_non_positive_target() {
        assert!(max_proof_test_interval(Architecture::OO1, 1e-6, 0.0, 0.0).is_err());
        assert!(max_proof_test_interval(Architecture::OO1, 1e-6, 0.0, -1.0).is_err());
    }

    #[test]
    fn rejects_non_positive_lambda() {
        assert!(max_proof_test_interval(Architecture::OO1, 0.0, 0.0, 1e-3).is_err());
    }

    #[test]
    fn rejects_unsupported_architecture_clearly() {
        let unsupported = Architecture::new(2, 4).unwrap();
        let err = max_proof_test_interval(unsupported, 1e-6, 0.0, 1e-3).unwrap_err();
        assert!(matches!(err, SisError::UnsupportedArchitecture { .. }));
    }
}
