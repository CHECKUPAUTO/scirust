//! Préconditionneur diagonal de Jacobi.
//!
//! Le préconditionneur le moins cher possible : `z = D⁻¹ · r` où `D` est la
//! diagonale de `A`. Utile en amont de [`crate::linalg::gmres_preconditioned`]
//! ou [`crate::linalg::bicgstab_preconditioned`] quand `A` est diagonalement
//! dominante ou simplement mal conditionnée sur sa diagonale (cas fréquent
//! des laplaciens de différences finies et des matrices de circuits/FEM).

use crate::{SolverError, SolverResult};

/// Préconditionneur `M = diag(A)`, appliqué comme `z = M⁻¹ · r`.
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner {
    inv_diag: Vec<f64>,
}

impl JacobiPreconditioner {
    /// Construit le préconditionneur à partir de la diagonale de `A`.
    ///
    /// Une entrée quasi nulle (`|d| < 1e-300`) est traitée comme une
    /// identité locale (facteur 1.0) plutôt que de diviser par zéro — le
    /// préconditionneur reste alors sans effet sur cette composante, ce qui
    /// est sûr (dégrade la convergence, ne la casse jamais).
    pub fn new(diag: &[f64]) -> SolverResult<Self> {
        if diag.is_empty()
        {
            return Err(SolverError::InvalidInput(
                "jacobi preconditioner: empty diagonal".to_string(),
            ));
        }
        for (i, &d) in diag.iter().enumerate()
        {
            if !d.is_finite()
            {
                return Err(SolverError::NanDetected { iter: i, value: d });
            }
        }
        const EPS: f64 = 1e-300;
        let inv_diag = diag
            .iter()
            .map(|&d| if d.abs() > EPS { 1.0 / d } else { 1.0 })
            .collect();
        Ok(Self { inv_diag })
    }

    pub fn len(&self) -> usize {
        self.inv_diag.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inv_diag.is_empty()
    }

    /// `z = D⁻¹ · r`, en place dans `z`.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        for i in 0..r.len()
        {
            z[i] = r[i] * self.inv_diag[i];
        }
    }

    /// Adaptateur closure pour les solveurs préconditionnés
    /// (`gmres_preconditioned`, `bicgstab_preconditioned`).
    pub fn as_fn(&self) -> impl Fn(&[f64], &mut [f64]) + '_ {
        move |r: &[f64], z: &mut [f64]| self.apply(r, z)
    }
}

/// Préconditionneur identité (pas de préconditionnement) : `z = r`.
/// Utilisé comme valeur par défaut par les variantes non préconditionnées.
pub fn identity_precond(r: &[f64], z: &mut [f64]) {
    z.copy_from_slice(r);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_scales_by_inverse_diagonal() {
        let p = JacobiPreconditioner::new(&[2.0, 4.0, 0.5]).unwrap();
        let mut z = vec![0.0; 3];
        p.apply(&[1.0, 1.0, 1.0], &mut z);
        assert_eq!(z, vec![0.5, 0.25, 2.0]);
    }

    #[test]
    fn jacobi_treats_near_zero_diagonal_as_identity() {
        let p = JacobiPreconditioner::new(&[1e-310, 3.0]).unwrap();
        let mut z = vec![0.0; 2];
        p.apply(&[5.0, 6.0], &mut z);
        assert_eq!(z[0], 5.0);
        assert!((z[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_empty_diagonal() {
        assert!(JacobiPreconditioner::new(&[]).is_err());
    }
}
