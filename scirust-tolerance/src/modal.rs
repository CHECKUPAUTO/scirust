//! Modal decomposition of form defects (*tolérancement modal*), the companion
//! of inertial tolerancing in Adragna's thesis (tel-00403876, arXiv:1002.0251).
//!
//! A form defect measured at `m` points is decomposed — "à la manière des
//! séries de Fourier" — onto an **orthonormal modal basis** `{Qₖ}` of shape
//! vectors ordered by increasing complexity:
//!
//! ```text
//! d = Σₖ λₖ Qₖ ,    λₖ = ⟨d, Qₖ⟩ ,    Σₖ λₖ² = ‖d‖²  (Parseval).
//! ```
//!
//! Low modes are interpretable (mode 0 = size/mean offset, mode 1 = tilt,
//! mode 2 = ovality/curvature, …), so a form defect becomes a short, physical
//! coefficient vector instead of a point cloud. The natural basis is the
//! surface's own vibration eigenmodes; a ready-made discrete-cosine basis
//! (the Fourier analogue the method is built on) is provided for profiles, and
//! any user basis (e.g. from an FEM modal analysis) can be supplied or
//! Gram-Schmidt-orthonormalised.
//!
//! Combined with inertial tolerancing, each mode carries its own inertia
//! `Iₖ` across a batch, and the modal inertias **partition** the surface
//! inertia: `Σₖ Iₖ² = m · I_S²` (see [`modal_inertias`]).

use crate::inertia::Inertia;
use serde::{Deserialize, Serialize};

/// An orthonormal modal basis over `m` sample points: each mode is a unit shape
/// vector of length `m`, mutually orthogonal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalBasis {
    modes: Vec<Vec<f64>>,
    m: usize,
}

impl ModalBasis {
    /// Wrap raw mode vectors as a basis. All modes must share the same non-zero
    /// length; orthonormality is *assumed* (check it with
    /// [`ModalBasis::is_orthonormal`] or build via [`ModalBasis::orthonormalize`]).
    /// Returns `None` on empty or ragged input.
    pub fn from_modes(modes: Vec<Vec<f64>>) -> Option<Self> {
        let m = modes.first()?.len();
        if m == 0 || modes.iter().any(|q| q.len() != m)
        {
            return None;
        }
        Some(Self { modes, m })
    }

    /// The first `k` orthonormal DCT-II modes over `m` points,
    /// `Qₖ[j] = cₖ·cos(π(j+½)k/m)` with `c₀ = √(1/m)`, `cₖ = √(2/m)` — the
    /// discrete-cosine (Fourier-type) basis, exactly orthonormal by
    /// construction. `k` is clamped to `m`.
    pub fn dct(m: usize, k: usize) -> Self {
        let k = k.min(m);
        let modes = (0..k)
            .map(|mode| {
                let c = if mode == 0
                {
                    (1.0 / m as f64).sqrt()
                }
                else
                {
                    (2.0 / m as f64).sqrt()
                };
                (0..m)
                    .map(|j| {
                        c * (core::f64::consts::PI * (j as f64 + 0.5) * mode as f64 / m as f64)
                            .cos()
                    })
                    .collect()
            })
            .collect();
        Self { modes, m }
    }

    /// Orthonormalise raw (possibly non-orthogonal) shape vectors by modified
    /// Gram-Schmidt, dropping any vector that is numerically dependent on the
    /// earlier ones. Returns `None` on empty or ragged input.
    pub fn orthonormalize(raw: Vec<Vec<f64>>) -> Option<Self> {
        let m = raw.first()?.len();
        if m == 0 || raw.iter().any(|q| q.len() != m)
        {
            return None;
        }
        let mut modes: Vec<Vec<f64>> = Vec::new();
        for v in raw
        {
            let orig_norm = dot(&v, &v).sqrt();
            let mut w = v;
            for q in &modes
            {
                let proj = dot(&w, q);
                for (wi, qi) in w.iter_mut().zip(q)
                {
                    *wi -= proj * qi;
                }
            }
            let norm = dot(&w, &w).sqrt();
            // Relative rank tolerance: a vector is dependent on the earlier
            // modes when its orthogonal residual is a negligible fraction of
            // its own length, so the threshold scales with the input magnitude.
            if norm > 1e-9 * orig_norm.max(f64::MIN_POSITIVE)
            {
                for wi in &mut w
                {
                    *wi /= norm;
                }
                modes.push(w);
            }
        }
        if modes.is_empty()
        {
            return None;
        }
        Some(Self { modes, m })
    }

    /// Number of modes.
    pub fn len(&self) -> usize {
        self.modes.len()
    }

    /// Whether the basis has no modes.
    pub fn is_empty(&self) -> bool {
        self.modes.is_empty()
    }

    /// Sample-point count `m`.
    pub fn points(&self) -> usize {
        self.m
    }

    /// Whether the modes are orthonormal to within `tol` (all pairwise inner
    /// products `≈ δₖₗ`).
    pub fn is_orthonormal(&self, tol: f64) -> bool {
        for (a, qa) in self.modes.iter().enumerate()
        {
            for (b, qb) in self.modes.iter().enumerate()
            {
                let want = if a == b { 1.0 } else { 0.0 };
                if (dot(qa, qb) - want).abs() > tol
                {
                    return false;
                }
            }
        }
        true
    }

    /// Modal coefficients `λₖ = ⟨d, Qₖ⟩` of a deviation vector. Returns an empty
    /// vector if `d`'s length does not match the basis.
    pub fn decompose(&self, deviation: &[f64]) -> Vec<f64> {
        if deviation.len() != self.m
        {
            return Vec::new();
        }
        self.modes.iter().map(|q| dot(q, deviation)).collect()
    }

    /// Reconstruct `Σₖ coeffsₖ·Qₖ` (a truncated form defect if fewer modes than
    /// the full basis are used). Extra coefficients beyond the mode count are
    /// ignored.
    pub fn reconstruct(&self, coeffs: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.m];
        for (c, q) in coeffs.iter().zip(&self.modes)
        {
            for (o, qi) in out.iter_mut().zip(q)
            {
                *o += c * qi;
            }
        }
        out
    }

    /// Euclidean norm of the residual `d − Σₖ λₖ Qₖ` — the part of the form
    /// defect the basis cannot represent (zero for a complete basis).
    pub fn residual_norm(&self, deviation: &[f64]) -> f64 {
        if deviation.len() != self.m
        {
            return f64::NAN;
        }
        let recon = self.reconstruct(&self.decompose(deviation));
        deviation
            .iter()
            .zip(&recon)
            .map(|(d, r)| (d - r).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Inner product of two equal-length vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Per-mode inertia of a batch of form defects: projects every part's deviation
/// vector onto the basis and, for each mode `k`, forms the inertia of its
/// coefficient across the batch (off-centering `λ̄ₖ`, dispersion `std(λₖ)`).
///
/// For a **complete** orthonormal basis the modal inertias partition the
/// surface inertia,
///
/// ```text
/// Σₖ Iₖ² = (1/n) Σᵢ ‖dᵢ‖² = m · I_S² ,
/// ```
///
/// so tolerancing the modes (a small, physical set of budgets) is equivalent to
/// tolerancing the whole surface. `parts` are the raw per-part deviation
/// vectors (e.g. [`crate::form::FormBatch::deviations`]); rows whose length does
/// not match the basis are skipped.
pub fn modal_inertias(basis: &ModalBasis, parts: &[Vec<f64>]) -> Vec<Inertia> {
    let coeffs: Vec<Vec<f64>> = parts
        .iter()
        .filter(|p| p.len() == basis.m)
        .map(|p| basis.decompose(p))
        .collect();
    let n = coeffs.len().max(1) as f64;
    (0..basis.len())
        .map(|k| {
            let mean = coeffs.iter().map(|c| c[k]).sum::<f64>() / n;
            let var = coeffs.iter().map(|c| (c[k] - mean).powi(2)).sum::<f64>() / n;
            Inertia::new(mean, var.sqrt())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::form::FormBatch;
    use approx::assert_relative_eq;

    #[test]
    fn dct_basis_is_orthonormal() {
        let basis = ModalBasis::dct(16, 16);
        assert!(basis.is_orthonormal(1e-12));
        assert_eq!(basis.len(), 16);
    }

    #[test]
    fn decompose_reconstruct_round_trips_with_full_basis() {
        let basis = ModalBasis::dct(8, 8);
        let d = vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.15, -0.05];
        let recon = basis.reconstruct(&basis.decompose(&d));
        for (a, b) in d.iter().zip(&recon)
        {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
        assert!(basis.residual_norm(&d) < 1e-12);
    }

    #[test]
    fn parseval_holds_for_complete_basis() {
        let basis = ModalBasis::dct(10, 10);
        let d: Vec<f64> = (0..10).map(|j| (j as f64 * 0.3).sin()).collect();
        let coeffs = basis.decompose(&d);
        let energy_coeff: f64 = coeffs.iter().map(|c| c * c).sum();
        let energy_signal: f64 = d.iter().map(|x| x * x).sum();
        assert_relative_eq!(energy_coeff, energy_signal, epsilon = 1e-12);
    }

    #[test]
    fn truncation_leaves_a_residual_and_mode0_is_the_mean() {
        let basis_full = ModalBasis::dct(6, 6);
        let basis_1 = ModalBasis::dct(6, 1); // mode 0 only (constant)
        let d = vec![1.0, 1.2, 0.9, 1.1, 1.0, 0.8];
        // Mode-0 coefficient reconstructs the mean level: recon is constant = mean.
        let recon = basis_1.reconstruct(&basis_1.decompose(&d));
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        for r in &recon
        {
            assert_relative_eq!(*r, mean, epsilon = 1e-12);
        }
        // A single mode cannot capture the variation ⇒ non-zero residual.
        assert!(basis_1.residual_norm(&d) > 1e-6);
        assert!(basis_full.residual_norm(&d) < 1e-12);
    }

    #[test]
    fn gram_schmidt_orthonormalises_and_drops_dependents() {
        // Two independent + one dependent (sum of the first two).
        let raw = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![2.0, 1.0, 0.0], // = row0 + row1, dependent
        ];
        let basis = ModalBasis::orthonormalize(raw).unwrap();
        assert_eq!(basis.len(), 2); // dependent vector dropped
        assert!(basis.is_orthonormal(1e-12));
    }

    #[test]
    fn modal_inertias_partition_the_surface_inertia() {
        // Σₖ Iₖ² = m · I_S² for a complete basis.
        let parts = vec![
            vec![0.10, -0.05, 0.20, 0.00],
            vec![-0.10, 0.05, 0.10, 0.10],
            vec![0.00, 0.15, -0.10, 0.05],
        ];
        let batch = FormBatch::new(parts.clone()).unwrap();
        let m = batch.points();
        let basis = ModalBasis::dct(m, m);
        let modal = modal_inertias(&basis, batch.deviations());
        let sum_i2: f64 = modal.iter().map(|i| i.mean_squared_deviation()).sum();
        assert_relative_eq!(
            sum_i2,
            m as f64 * batch.surface_inertia().powi(2),
            epsilon = 1e-12
        );
    }

    #[test]
    fn length_mismatch_is_handled() {
        let basis = ModalBasis::dct(4, 4);
        assert!(basis.decompose(&[1.0, 2.0]).is_empty());
        assert!(basis.residual_norm(&[1.0, 2.0]).is_nan());
    }
}
