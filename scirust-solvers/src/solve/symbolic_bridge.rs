//! Ponts entre `scirust-symbolic::Expr` et nos solveurs numériques.
//!
//! - `expr_to_closure` : transforme une Expr en `Fn(f64) -> f64` pour un solveur 1D.
//! - `extract_polynomial_coeffs` : tente d'extraire les coefficients f64 d'une Expr
//!   si celle-ci se réduit à un polynôme en une variable donnée. Renvoie `None`
//!   si l'expression contient des fonctions transcendantes ou des variables
//!   autres que la cible.

use scirust_symbolic::Expr;
use std::collections::HashMap;

/// Construit une closure `|x: f64| -> f64` à partir d'une Expr et du nom
/// de la variable d'intérêt. Toutes les autres variables doivent être
/// fournies dans `bindings`. Si l'évaluation échoue (variable manquante,
/// division par zéro, etc.), la closure renvoie `f64::NAN` — c'est volontaire
/// pour que les solveurs comme Brent puissent réagir au lieu de paniquer.
pub fn expr_to_closure(
    expr: Expr,
    var: &str,
    bindings: HashMap<String, f64>,
) -> impl Fn(f64) -> f64 {
    let var = var.to_string();
    move |x: f64| -> f64 {
        let mut env = bindings.clone();
        env.insert(var.clone(), x);
        scirust_symbolic::eval(&expr, &env).unwrap_or(f64::NAN)
    }
}

/// Tente d'extraire les coefficients polynomiaux d'une Expr en `var`.
/// Renvoie `Some(Vec<f64>)` (ordre croissant des puissances) si l'Expr est
/// effectivement un polynôme en `var`, `None` sinon.
///
/// Reconnaît : Const, Var(var), Add, Sub, Mul, Neg, Pow avec exposant entier.
/// Rejette : Sin/Cos/Exp/Ln/Sqrt/Abs et toute Var ≠ var.
pub fn extract_polynomial_coeffs(expr: &Expr, var: &str) -> Option<Vec<f64>> {
    let coeffs = expand_poly(expr, var)?;
    let mut c = coeffs;
    while c.len() > 1 && c.last() == Some(&0.0)
    {
        c.pop();
    }
    if c.is_empty()
    {
        c.push(0.0);
    }
    Some(c)
}

/// Récursivement : renvoie le vecteur de coefficients du polynôme représenté
/// par `e` en `var`. L'index est le degré.
fn expand_poly(e: &Expr, var: &str) -> Option<Vec<f64>> {
    use Expr::*;
    match e
    {
        Const(c) => Some(vec![*c]),
        Var(name) =>
        {
            if name == var
            {
                Some(vec![0.0, 1.0])
            }
            else
            {
                None
            }
        },
        Add(a, b) => Some(add_poly(expand_poly(a, var)?, expand_poly(b, var)?)),
        Sub(a, b) =>
        {
            let pa = expand_poly(a, var)?;
            let pb = expand_poly(b, var)?;
            Some(add_poly(pa, neg_poly(pb)))
        },
        Neg(a) => Some(neg_poly(expand_poly(a, var)?)),
        Mul(a, b) =>
        {
            let pa = expand_poly(a, var)?;
            let pb = expand_poly(b, var)?;
            Some(mul_poly(&pa, &pb))
        },
        Pow(base, exp) =>
        {
            // L'exposant doit être un entier positif constant
            let n = match **exp
            {
                Const(c) if c.fract() == 0.0 && c >= 0.0 => c as usize,
                _ => return None,
            };
            let pb = expand_poly(base, var)?;
            let mut acc = vec![1.0];
            for _ in 0..n
            {
                acc = mul_poly(&acc, &pb);
            }
            Some(acc)
        },
        Div(a, b) =>
        {
            // Division par une constante uniquement
            let pa = expand_poly(a, var)?;
            let pb = expand_poly(b, var)?;
            if pb.len() == 1 && pb[0] != 0.0
            {
                Some(pa.iter().map(|c| c / pb[0]).collect())
            }
            else
            {
                None
            }
        },
        Sin(_) | Cos(_) | Exp(_) | Ln(_) | Sqrt(_) | Abs(_) => None,
    }
}

fn add_poly(mut a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let n = a.len().max(b.len());
    a.resize(n, 0.0);
    for (i, &v) in b.iter().enumerate()
    {
        a[i] += v;
    }
    a
}

fn neg_poly(mut a: Vec<f64>) -> Vec<f64> {
    for v in &mut a
    {
        *v = -*v;
    }
    a
}

fn mul_poly(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty()
    {
        return vec![0.0];
    }
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate()
    {
        if ai == 0.0
        {
            continue;
        }
        for (j, &bj) in b.iter().enumerate()
        {
            out[i + j] += ai * bj;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirust_symbolic as sym;

    #[test]
    fn extract_linear() {
        // 2x + 3 = 0  →  coeffs [3, 2]
        let e = sym::parse("2*x + 3").unwrap();
        let c = extract_polynomial_coeffs(&e, "x").unwrap();
        assert_eq!(c, vec![3.0, 2.0]);
    }

    #[test]
    fn extract_quadratic() {
        // x² - 3x + 2  →  coeffs [2, -3, 1]
        let e = sym::parse("x^2 - 3*x + 2").unwrap();
        let c = extract_polynomial_coeffs(&e, "x").unwrap();
        assert_eq!(c, vec![2.0, -3.0, 1.0]);
    }

    #[test]
    fn extract_cubic() {
        // (x-1)(x+1)x = x³ - x  →  coeffs [0, -1, 0, 1]
        let e = sym::parse("x^3 - x").unwrap();
        let c = extract_polynomial_coeffs(&e, "x").unwrap();
        assert_eq!(c, vec![0.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn rejects_transcendental() {
        let e = sym::parse("sin(x) + x").unwrap();
        assert!(extract_polynomial_coeffs(&e, "x").is_none());
    }

    #[test]
    fn expr_to_closure_works() {
        let e = sym::parse("x^2 - 5*x + 6").unwrap();
        let f = expr_to_closure(e, "x", HashMap::new());
        assert_relative_eq!(f(0.0), 6.0, epsilon = 1e-12);
        assert_relative_eq!(f(2.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(f(3.0), 0.0, epsilon = 1e-12);
    }
}
