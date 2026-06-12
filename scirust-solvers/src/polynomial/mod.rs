//! Manipulation et résolution de polynômes.
//!
//! Représentation : `Polynomial` stocke les coefficients dans l'ordre
//! **croissant des puissances** : `c[0] + c[1]·x + c[2]·x² + ...`
//!
//! - `eval`     : évaluation Horner
//! - `deriv`    : dérivée formelle
//! - `roots`    : toutes les racines (réelles + complexes) via la matrice
//!   compagnon et l'algorithme QR sur celle-ci
//! - `real_roots`: filtrage des racines réelles

use crate::SolverResult;

pub mod roots;

pub use roots::{durand_kerner, real_roots, roots};

/// Polynôme à coefficients f64 (degré croissant).
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    pub coeffs: Vec<f64>,
}

impl Polynomial {
    /// Crée un polynôme depuis ses coefficients (degré croissant).
    /// Trim les zéros de fin pour normaliser le degré.
    pub fn new(mut coeffs: Vec<f64>) -> Self {
        while coeffs.len() > 1 && coeffs.last() == Some(&0.0)
        {
            coeffs.pop();
        }
        if coeffs.is_empty()
        {
            coeffs.push(0.0);
        }
        Self { coeffs }
    }

    /// Construit depuis l'ordre décroissant (style maths : a·x^n + ... + c).
    pub fn from_descending(mut coeffs: Vec<f64>) -> Self {
        coeffs.reverse();
        Self::new(coeffs)
    }

    /// Degré du polynôme.
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Évaluation par schéma de Horner (numériquement stable).
    pub fn eval(&self, x: f64) -> f64 {
        let mut acc = 0.0;
        for &c in self.coeffs.iter().rev()
        {
            acc = acc * x + c;
        }
        acc
    }

    /// Dérivée formelle : p'(x).
    pub fn deriv(&self) -> Polynomial {
        if self.degree() == 0
        {
            return Polynomial::new(vec![0.0]);
        }
        let coeffs: Vec<f64> = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| c * (i as f64))
            .collect();
        Polynomial::new(coeffs)
    }

    /// Trouve toutes les racines (complexes), via Durand-Kerner.
    pub fn roots_all(&self) -> SolverResult<Vec<(f64, f64)>> {
        roots::roots(self)
    }

    /// Trouve uniquement les racines réelles (parties imaginaires < eps).
    pub fn real_roots(&self, eps: f64) -> SolverResult<Vec<f64>> {
        roots::real_roots(self, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn horner_eval() {
        // p(x) = 1 + 2x + 3x²
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert_relative_eq!(p.eval(0.0), 1.0, epsilon = 1e-14);
        assert_relative_eq!(p.eval(1.0), 6.0, epsilon = 1e-14);
        assert_relative_eq!(p.eval(2.0), 17.0, epsilon = 1e-14);
    }

    #[test]
    fn derivative() {
        // p = x³ + 2x + 5  →  p' = 3x² + 2
        let p = Polynomial::new(vec![5.0, 2.0, 0.0, 1.0]);
        let dp = p.deriv();
        assert_eq!(dp.coeffs, vec![2.0, 0.0, 3.0]);
    }

    #[test]
    fn from_descending() {
        // 2x² + 3x + 5  →  coeffs [5, 3, 2]
        let p = Polynomial::from_descending(vec![2.0, 3.0, 5.0]);
        assert_eq!(p.coeffs, vec![5.0, 3.0, 2.0]);
    }
}
