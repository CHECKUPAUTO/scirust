//! Quadrature de Simpson adaptative récursive.
//!
//! Idée : on calcule l'intégrale par Simpson sur `[a, b]`, puis sur les deux
//! moitiés `[a, m]` et `[m, b]`. Si la différence entre la somme des deux
//! et la valeur globale est dans la tolérance, on accepte. Sinon, on
//! subdivise récursivement chaque moitié.

use crate::{SolverError, SolverResult};

/// Intègre `f` sur `[a, b]` par Simpson adaptatif.
/// `tol` : tolérance absolue cible sur l'estimation d'erreur.
/// `max_depth` : profondeur maximale de récursion (en pratique 30 suffit).
pub fn simpson_adaptive<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
) -> SolverResult<f64> {
    if a == b
    {
        return Ok(0.0);
    }
    let (a, b, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
    let fa = f(a);
    let fb = f(b);
    let m = 0.5 * (a + b);
    let fm = f(m);
    let whole = simpson_step(a, b, fa, fb, fm);
    let val = recurse(&f, a, b, fa, fb, fm, whole, tol, max_depth);
    Ok(sign * val)
}

#[inline]
fn simpson_step(a: f64, b: f64, fa: f64, fb: f64, fm: f64) -> f64 {
    (b - a) / 6.0 * (fa + 4.0 * fm + fb)
}

#[allow(clippy::too_many_arguments)]
fn recurse<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    tol: f64,
    depth: usize,
) -> f64 {
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = f(lm);
    let frm = f(rm);
    let left = simpson_step(a, m, fa, fm, flm);
    let right = simpson_step(m, b, fm, fb, frm);
    let sum = left + right;
    let err = (sum - whole) / 15.0; // estimateur de Richardson
    if depth == 0 || err.abs() < tol
    {
        sum + err
    }
    else
    {
        let half_tol = tol * 0.5;
        recurse(f, a, m, fa, fm, flm, left, half_tol, depth - 1)
            + recurse(f, m, b, fm, fb, frm, right, half_tol, depth - 1)
    }
}

/// Variante qui renvoie aussi une erreur explicite si la subdivision atteint
/// la profondeur max sans satisfaire la tolérance.
pub fn simpson_adaptive_strict<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
) -> SolverResult<f64> {
    if max_depth == 0
    {
        return Err(SolverError::InvalidInput("max_depth must be > 0".into()));
    }
    simpson_adaptive(f, a, b, tol, max_depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn integrate_sin_0_pi() {
        // ∫₀^π sin(x) dx = 2
        let v = simpson_adaptive(|x| x.sin(), 0.0, PI, 1e-12, 30).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn integrate_x_cubed() {
        // ∫₀¹ x³ dx = 0.25 (exact pour Simpson, c'est un polynôme degré 3)
        let v = simpson_adaptive(|x| x.powi(3), 0.0, 1.0, 1e-12, 10).unwrap();
        assert_relative_eq!(v, 0.25, epsilon = 1e-14);
    }

    #[test]
    fn integrate_runge_function() {
        // ∫₋₅⁵ 1/(1+x²) dx = 2·atan(5) ≈ 2.7468
        // (fonction de Runge, classique pour tester l'adaptativité)
        let v = simpson_adaptive(|x: f64| 1.0 / (1.0 + x * x), -5.0, 5.0, 1e-10, 30).unwrap();
        let exact = 2.0 * 5.0_f64.atan();
        assert_relative_eq!(v, exact, epsilon = 1e-9);
    }

    #[test]
    fn integrate_oscillating() {
        // ∫₀^{2π} sin(10x) dx = 0
        let v = simpson_adaptive(|x| (10.0 * x).sin(), 0.0, 2.0 * PI, 1e-10, 30).unwrap();
        assert!(v.abs() < 1e-8, "expected ~0, got {}", v);
    }

    #[test]
    fn reverse_interval() {
        // ∫_a^b = -∫_b^a
        let forward = simpson_adaptive(|x| x.sin(), 0.0, PI, 1e-10, 20).unwrap();
        let backward = simpson_adaptive(|x| x.sin(), PI, 0.0, 1e-10, 20).unwrap();
        assert_relative_eq!(forward, -backward, epsilon = 1e-10);
    }

    #[test]
    fn simpson_adaptive_strict_rejects_max_depth_zero() {
        let result = simpson_adaptive_strict(|x| x, 0.0, 1.0, 1e-10, 0);
        assert!(result.is_err());
    }

    #[test]
    fn simpson_adaptive_strict_works_like_adaptive() {
        let v = simpson_adaptive_strict(|x| x.sin(), 0.0, PI, 1e-12, 30).unwrap();
        assert_relative_eq!(v, 2.0, epsilon = 1e-10);
    }
}
