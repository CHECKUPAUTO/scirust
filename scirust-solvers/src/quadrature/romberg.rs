//! Quadrature de Romberg : extrapolation de Richardson sur la règle du
//! trapèze. Converge à l'ordre 2k pour k niveaux d'extrapolation. Très
//! efficace sur fonctions très lisses (analytiques).

use crate::{SolverError, SolverResult};

fn validate_inputs(a: f64, b: f64, tol: f64, max_levels: usize) -> SolverResult<()> {
    if !a.is_finite() || !b.is_finite()
    {
        return Err(SolverError::InvalidInput(
            "integration bounds must be finite".into(),
        ));
    }
    if !tol.is_finite() || tol <= 0.0
    {
        return Err(SolverError::InvalidInput(
            "tol must be finite and > 0".into(),
        ));
    }
    if max_levels < 2
    {
        return Err(SolverError::InvalidInput(
            "max_levels must be >= 2 (a Richardson error estimate needs at least two levels)"
                .into(),
        ));
    }
    Ok(())
}

fn eval_finite<F: Fn(f64) -> f64>(f: &F, x: f64) -> SolverResult<f64> {
    let value = f(x);
    if value.is_finite()
    {
        Ok(value)
    }
    else
    {
        Err(SolverError::NanDetected { iter: 0, value })
    }
}

/// Intègre f sur `[a,b]` par Romberg. `max_levels` ~ 10-15 suffit en pratique.
/// `tol` est la tolérance absolue sur l'erreur estimée (différence entre deux
/// niveaux successifs). Best-effort : si la tolérance n'est pas atteinte
/// après `max_levels` niveaux, renvoie tout de même la dernière estimation
/// (voir [`romberg_strict`] pour une variante qui signale l'échec).
///
/// # Errors
/// [`SolverError::InvalidInput`] si les bornes/`tol`/`max_levels` sont
/// invalides ; [`SolverError::NanDetected`] si `f` renvoie une valeur non
/// finie.
pub fn romberg<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_levels: usize,
) -> SolverResult<f64> {
    validate_inputs(a, b, tol, max_levels)?;
    romberg_table(&f, a, b, tol, max_levels).map(|(value, _, _)| value)
}

/// Variante stricte : renvoie [`SolverError::NoConvergence`] si la tolérance
/// n'est pas atteinte après `max_levels` niveaux, plutôt que de renvoyer
/// silencieusement la meilleure estimation disponible.
///
/// # Errors
/// Comme [`romberg`], plus [`SolverError::NoConvergence`] sur non-convergence.
pub fn romberg_strict<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_levels: usize,
) -> SolverResult<f64> {
    validate_inputs(a, b, tol, max_levels)?;
    let (value, converged, residual) = romberg_table(&f, a, b, tol, max_levels)?;
    if converged
    {
        Ok(value)
    }
    else
    {
        Err(SolverError::NoConvergence {
            iterations: max_levels,
            residual,
        })
    }
}

/// Runs the Romberg table and returns `(value, converged, last_residual)`.
fn romberg_table<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    tol: f64,
    max_levels: usize,
) -> SolverResult<(f64, bool, f64)> {
    let mut r = vec![vec![0.0_f64; max_levels]; max_levels];

    // Niveau 0 : trapèze simple
    r[0][0] = 0.5 * (b - a) * (eval_finite(f, a)? + eval_finite(f, b)?);

    let mut last_residual = f64::INFINITY;
    for i in 1..max_levels
    {
        // Trapèze composé avec 2^i intervalles
        let n = 1usize << (i - 1); // nombre de NOUVEAUX points
        let h = (b - a) / (1usize << i) as f64;
        let mut sum = 0.0;
        for k in 0..n
        {
            sum += eval_finite(f, a + (2 * k + 1) as f64 * h)?;
        }
        r[i][0] = 0.5 * r[i - 1][0] + h * sum;

        // Extrapolation de Richardson
        for j in 1..=i
        {
            let denom = (1usize << (2 * j)) as f64 - 1.0;
            r[i][j] = r[i][j - 1] + (r[i][j - 1] - r[i - 1][j - 1]) / denom;
        }

        // Convergence : différence entre deux niveaux d'extrapolation
        if i >= 2
        {
            last_residual = (r[i][i] - r[i - 1][i - 1]).abs();
            if last_residual < tol
            {
                return Ok((r[i][i], true, last_residual));
            }
        }
    }
    Ok((r[max_levels - 1][max_levels - 1], false, last_residual))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn romberg_sin() {
        let v = romberg(|x| x.sin(), 0.0, PI, 1e-14, 15).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn romberg_exp() {
        // ∫₀¹ exp(x) dx = e - 1
        let v = romberg(|x: f64| x.exp(), 0.0, 1.0, 1e-14, 15).unwrap();
        assert_relative_eq!(v, std::f64::consts::E - 1.0, epsilon = 1e-13);
    }

    #[test]
    fn rejects_invalid_numeric_inputs() {
        assert!(romberg(|x| x, 0.0, 1.0, 0.0, 15).is_err());
        assert!(romberg(|x| x, f64::NAN, 1.0, 1e-10, 15).is_err());
        assert!(romberg(|_| f64::NAN, 0.0, 1.0, 1e-10, 15).is_err());
    }

    #[test]
    fn rejects_max_levels_below_two() {
        assert!(matches!(
            romberg(|x| x, 0.0, 1.0, 1e-10, 1),
            Err(SolverError::InvalidInput(_))
        ));
        assert!(matches!(
            romberg(|x| x, 0.0, 1.0, 1e-10, 0),
            Err(SolverError::InvalidInput(_))
        ));
    }

    /// `romberg` is best-effort: an unreachable tolerance still returns the
    /// best available estimate rather than erroring.
    #[test]
    fn romberg_returns_best_effort_on_unreachable_tolerance() {
        let v = romberg(|x| x.sin(), 0.0, PI, 1e-300, 5).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-6);
    }

    /// `romberg_strict` reports the same unreachable tolerance as an error
    /// instead of silently returning the best-effort value.
    #[test]
    fn romberg_strict_reports_non_convergence() {
        let result = romberg_strict(|x| x.sin(), 0.0, PI, 1e-300, 5);
        assert!(matches!(result, Err(SolverError::NoConvergence { .. })));
    }

    #[test]
    fn romberg_strict_matches_romberg_on_convergence() {
        let v = romberg_strict(|x| x.sin(), 0.0, PI, 1e-14, 15).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-12);
    }
}
