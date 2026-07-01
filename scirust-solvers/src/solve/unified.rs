//! Dispatcher unifié `solve(...)`.
//!
//! Reçoit soit une `Expr` symbolique soit une closure Rust, et choisit la
//! stratégie optimale :
//!
//! - Linéaire (degré 1)        → solution exacte directe (1 opération)
//! - Quadratique (degré 2)     → formule discriminant
//! - Polynôme degré ≥ 3        → Durand-Kerner (toutes les racines)
//! - Équation générale         → Brent ou Newton (1 racine, demande un intervalle ou x0)

use crate::polynomial::Polynomial;
use crate::roots::{brent, newton};
use crate::{SolverError, SolverResult, Tolerance};
use scirust_autodiff::Dual;
use scirust_symbolic::{self as sym, Expr};
use std::collections::HashMap;

use super::symbolic_bridge::{expr_to_closure, extract_polynomial_coeffs};

/// Résultat d'un appel `solve` : peut être une liste de racines (cas
/// polynomial, où on les a toutes), ou une racine unique (cas général).
#[derive(Debug, Clone)]
pub enum SolveResult {
    /// Toutes les racines réelles, triées (cas polynomial).
    AllReal(Vec<f64>),
    /// Toutes les racines (réelles et complexes), polynômes.
    AllComplex(Vec<(f64, f64)>),
    /// Racine unique trouvée par méthode numérique.
    Single(f64),
}

impl SolveResult {
    /// Renvoie un `Vec<f64>` des racines réelles, quelle que soit la variante.
    pub fn real_roots(&self) -> Vec<f64> {
        match self
        {
            SolveResult::AllReal(v) => v.clone(),
            SolveResult::AllComplex(v) => v
                .iter()
                .filter(|(_, im)| im.abs() < 1e-6)
                .map(|(re, _)| *re)
                .collect(),
            SolveResult::Single(x) => vec![*x],
        }
    }
}

/// Résout `expr = 0` en `var`. Choisit automatiquement la meilleure stratégie.
///
/// - Si `expr` est un polynôme en `var` avec degré ≤ 2 : solution exacte.
/// - Si `expr` est un polynôme de degré ≥ 3 : Durand-Kerner sur tous les coefficients.
/// - Sinon : pas applicable (cf. `solve_in_interval`).
///
/// `bindings` : valeurs des autres variables (si l'Expr en contient).
pub fn solve(expr: &Expr, var: &str, bindings: HashMap<String, f64>) -> SolverResult<SolveResult> {
    let _ = bindings; // pour le cas polynomial on n'évalue pas — on extrait les coefs
    // 1) Tente l'extraction polynomiale
    if let Some(coeffs) = extract_polynomial_coeffs(expr, var)
    {
        return solve_polynomial(&coeffs);
    }
    Err(SolverError::InvalidInput(format!(
        "L'expression n'est pas un polynôme en `{}` et aucun intervalle de recherche \
         n'a été fourni. Utilise `solve_in_interval(...)` ou `solve_near(...)`.",
        var
    )))
}

/// Résout `expr = 0` en cherchant une racine dans `[a, b]` par Brent.
/// Marche pour n'importe quelle Expr (transcendante, etc.) tant que f(a)·f(b) < 0.
pub fn solve_in_interval(
    expr: &Expr,
    var: &str,
    a: f64,
    b: f64,
    bindings: HashMap<String, f64>,
    tol: Tolerance,
) -> SolverResult<f64> {
    let f = expr_to_closure(expr.clone(), var, bindings);
    let s = brent(f, a, b, tol)?;
    Ok(s.value)
}

/// Résout `expr = 0` par Newton, en partant de `x0`.
/// Dérivée calculée symboliquement via `scirust-symbolic::diff` puis évaluée.
/// (On ne peut pas utiliser autodiff::Dual directement car l'Expr n'est pas
/// générique sur le type.)
pub fn solve_near(
    expr: &Expr,
    var: &str,
    x0: f64,
    bindings: HashMap<String, f64>,
    tol: Tolerance,
) -> SolverResult<f64> {
    let d_expr = sym::diff(expr, var);
    let f_expr = expr.clone();
    let f = expr_to_closure(f_expr, var, bindings.clone());
    let df = expr_to_closure(d_expr, var, bindings);
    let s = crate::roots::newton_with_derivative(f, df, x0, tol)?;
    Ok(s.value)
}

/// Résout `expr = 0` en utilisant Newton avec dérivée par autodiff Dual,
/// pour une closure pure Rust (pas une Expr symbolique).
pub fn solve_closure_newton<F>(f: F, x0: f64, tol: Tolerance) -> SolverResult<f64>
where
    F: Fn(Dual) -> Dual,
{
    let s = newton(f, x0, tol)?;
    Ok(s.value)
}

/// Résout un polynôme par les méthodes les plus adaptées.
fn solve_polynomial(coeffs: &[f64]) -> SolverResult<SolveResult> {
    let p = Polynomial::new(coeffs.to_vec());
    match p.degree()
    {
        0 =>
        {
            // Constante non nulle → pas de racine
            if p.coeffs[0].abs() < 1e-15
            {
                // 0 = 0 : techniquement tout x est solution. On renvoie []
                Ok(SolveResult::AllReal(vec![]))
            }
            else
            {
                Err(SolverError::InvalidInput(format!(
                    "Polynôme constant non nul {} = 0 sans solution",
                    p.coeffs[0]
                )))
            }
        },
        1 =>
        {
            // a + bx = 0  →  x = -a/b
            let a = p.coeffs[0];
            let b = p.coeffs[1];
            if b.abs() < 1e-30
            {
                return Err(SolverError::Singular { row: 0, pivot: b });
            }
            Ok(SolveResult::AllReal(vec![-a / b]))
        },
        2 =>
        {
            // ax² + bx + c = 0
            let c = p.coeffs[0];
            let b = p.coeffs[1];
            let a = p.coeffs[2];
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0
            {
                let sq = disc.sqrt();
                // Formule stable : on évite la soustraction de deux quantités
                // presque égales (annulation catastrophique) qui apparaît dans
                // `(-b ± sq) / (2a)` quand `4ac ≪ b²` (racines bien séparées).
                // On calcule d'abord `q = -½(b + signe(b)·sq)` — toujours une
                // addition de termes de même signe — puis `x1 = q/a`, `x2 = c/q`.
                let sign_b = if b < 0.0 { -1.0 } else { 1.0 };
                let q = -0.5 * (b + sign_b * sq);
                // `q` ne peut valoir 0 que si `disc == 0` et `b == 0`, càd racine
                // double nulle (c == 0) ; on retombe alors sur la formule directe.
                let (mut x1, mut x2) = if q.abs() < 1e-300
                {
                    let x = -b / (2.0 * a);
                    (x, x)
                }
                else
                {
                    (q / a, c / q)
                };
                if x1 > x2
                {
                    std::mem::swap(&mut x1, &mut x2);
                }
                Ok(SolveResult::AllReal(vec![x1, x2]))
            }
            else
            {
                let sq = (-disc).sqrt();
                let re = -b / (2.0 * a);
                let im = sq / (2.0 * a);
                Ok(SolveResult::AllComplex(vec![(re, -im), (re, im)]))
            }
        },
        _ =>
        {
            // Degré 3+ : Durand-Kerner
            let all = crate::polynomial::roots::roots(&p)?;
            // Sépare réelles/complexes
            let any_complex = all.iter().any(|(_, im)| im.abs() > 1e-6);
            if any_complex
            {
                Ok(SolveResult::AllComplex(all))
            }
            else
            {
                let mut reals: Vec<f64> = all.into_iter().map(|(re, _)| re).collect();
                reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Ok(SolveResult::AllReal(reals))
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirust_symbolic as sym;
    use std::collections::HashMap;

    #[test]
    fn solve_linear() {
        // 2x + 4 = 0
        let e = sym::parse("2*x + 4").unwrap();
        let r = solve(&e, "x", HashMap::new()).unwrap();
        let reals = r.real_roots();
        assert_eq!(reals.len(), 1);
        assert_relative_eq!(reals[0], -2.0, epsilon = 1e-12);
    }

    #[test]
    fn solve_quadratic_real() {
        // x² - 5x + 6 = 0  →  x = 2 ou 3
        let e = sym::parse("x^2 - 5*x + 6").unwrap();
        let r = solve(&e, "x", HashMap::new()).unwrap();
        let reals = r.real_roots();
        assert_eq!(reals.len(), 2);
        assert_relative_eq!(reals[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(reals[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn solve_quadratic_well_separated_roots_no_cancellation() {
        // x² - 1e8·x + 1 = 0. Racines très séparées : ≈ 1e8 et ≈ 1e-8.
        // La formule naïve `(-b ± √disc)/(2a)` souffre d'annulation
        // catastrophique sur la petite racine (donne ≈ 7.45e-9 au lieu de 1e-8).
        let e = sym::parse("x^2 - 100000000*x + 1").unwrap();
        let r = solve(&e, "x", HashMap::new()).unwrap();
        let reals = r.real_roots();
        assert_eq!(reals.len(), 2);
        // Petite racine : doit être proche de 1e-8 à la précision machine
        // relative, pas seulement à ~25 % comme avec la formule naïve.
        assert_relative_eq!(reals[0], 1e-8, max_relative = 1e-10);
        assert_relative_eq!(reals[1], 1e8, max_relative = 1e-12);
        // Contrôle direct : le produit des racines vaut c/a = 1.
        assert_relative_eq!(reals[0] * reals[1], 1.0, max_relative = 1e-10);
    }

    #[test]
    fn solve_quadratic_complex() {
        // x² + 1 = 0  →  ±i
        let e = sym::parse("x^2 + 1").unwrap();
        let r = solve(&e, "x", HashMap::new()).unwrap();
        match r
        {
            SolveResult::AllComplex(roots) =>
            {
                assert_eq!(roots.len(), 2);
                // |im| = 1 pour les deux
                for (re, im) in &roots
                {
                    assert!(re.abs() < 1e-10);
                    assert!((im.abs() - 1.0).abs() < 1e-10);
                }
            },
            _ => panic!("expected complex roots, got {:?}", r),
        }
    }

    #[test]
    fn solve_cubic() {
        // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
        let e = sym::parse("x^3 - 6*x^2 + 11*x - 6").unwrap();
        let r = solve(&e, "x", HashMap::new()).unwrap();
        let reals = r.real_roots();
        assert_eq!(reals.len(), 3);
        assert_relative_eq!(reals[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(reals[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(reals[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn solve_transcendental_via_interval() {
        // cos(x) = 0 dans [0, π], racine = π/2
        let e = sym::parse("cos(x)").unwrap();
        let v = solve_in_interval(
            &e,
            "x",
            0.0,
            std::f64::consts::PI,
            HashMap::new(),
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(v, std::f64::consts::PI / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn solve_with_other_var() {
        // 2*x + y = 0 avec y = 6  →  x = -3
        let e = sym::parse("2*x + y").unwrap();
        let mut bindings = HashMap::new();
        bindings.insert("y".to_string(), 6.0);
        let v = solve_in_interval(&e, "x", -10.0, 10.0, bindings, Tolerance::default()).unwrap();
        assert_relative_eq!(v, -3.0, epsilon = 1e-9);
    }

    #[test]
    fn solve_closure_newton_cubic() {
        // x³ - 2x - 5 = 0
        let v = solve_closure_newton(
            |x: Dual| x.powi(3) - x * 2.0 - 5.0,
            2.0,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(v, 2.094_551_481_542_326_6, epsilon = 1e-10);
    }

    #[test]
    fn solve_near_symbolic_diff() {
        // tan(x) - x = 0, partir de 4.5
        let e = sym::parse("sin(x)/cos(x) - x").unwrap();
        let v = solve_near(&e, "x", 4.5, HashMap::new(), Tolerance::default()).unwrap();
        assert_relative_eq!(v, 4.493_409_457_909_064, epsilon = 1e-6);
    }
}
