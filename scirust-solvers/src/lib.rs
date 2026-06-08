//! `scirust-solvers` — résolution d'équations, EDO, optimisation, quadrature.
//!
//! Construit au-dessus de `scirust-autodiff` (gradients exacts) et
//! `scirust-symbolic` (chemin symbolique quand possible). Couvre :
//!
//! - **Algèbre linéaire** : `Matrix`, LU, QR, Cholesky, déterminant, inverse,
//!   gradient conjugué.
//! - **Racines 1D** : bissection, sécante, Newton (autodiff), Brent.
//! - **Systèmes non-linéaires** : Newton-Raphson multivarié (jacobienne via
//!   autodiff), Broyden quasi-Newton.
//! - **EDO** : Runge-Kutta 4 fixe, Dormand-Prince 5(4) adaptatif.
//! - **Optimisation** : BFGS, descente de gradient avec recherche linéaire,
//!   Nelder-Mead sans dérivée.
//! - **Quadrature** : Simpson adaptatif, Gauss-Legendre.
//! - **Polynômes** : évaluation Horner, racines via matrice compagnon.
//! - **API unifiée** : `solve(expr, var)` dispatch symbolique → numérique.
//!
//! Aucun raccourci. Tous les algorithmes sont implémentés en pur Rust.

#![allow(clippy::needless_range_loop)]

pub mod linalg;
pub mod nonlinear;
pub mod ode;
pub mod optimize;
pub mod polynomial;
pub mod quadrature;
pub mod roots;
pub mod scientific;
pub mod solve;

pub use linalg::Matrix;

use thiserror::Error;
use std::fmt::Debug;

/// Erreur unifiée pour tous les solveurs.
#[derive(Debug, Clone, Error)]
pub enum SolverError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("matrix is singular (or near-singular) — pivot {pivot:.3e} at row {row}")]
    Singular { row: usize, pivot: f64 },

    #[error("matrix is not square: {rows}x{cols}")]
    NotSquare { rows: usize, cols: usize },

    #[error("matrix is not symmetric positive definite (Cholesky failed)")]
    NotSpd,

    #[error("no sign change in interval [{a}, {b}] for bisection: f(a)={fa:.3e}, f(b)={fb:.3e}")]
    NoSignChange { a: f64, b: f64, fa: f64, fb: f64 },

    #[error("derivative is zero at x={x} — Newton step undefined")]
    ZeroDerivative { x: f64 },

    #[error("did not converge after {iterations} iterations (last residual: {residual:.3e})")]
    NoConvergence { iterations: usize, residual: f64 },

    #[error("step size became too small ({step:.3e}) — likely stuck")]
    StepUnderflow { step: f64 },

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("ODE integration failed: {0}")]
    IntegrationFailed(String),

    #[error("NaN or Inf detected at iteration {iter} (value: {value:.3e})")]
    NanDetected { iter: usize, value: f64 },

    #[error("divergence detected: residual grew from {from:.3e} to {to:.3e} at iteration {iter}")]
    Divergence {
        iter: usize,
        from: f64,
        to: f64,
    },

    #[error("backup state restored after anomaly at iteration {iter}: {reason}")]
    BackupRestored { iter: usize, reason: String },
}

pub type SolverResult<T> = Result<T, SolverError>;

/// Informations renvoyées par un solveur itératif en cas de succès.
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    pub iterations: usize,
    pub residual: f64,
    pub converged: bool,
}

/// Résultat enrichi : valeur + métadonnées de convergence.
#[derive(Debug, Clone)]
pub struct Solution<T> {
    pub value: T,
    pub info: ConvergenceInfo,
}

impl<T> Solution<T> {
    pub fn new(value: T, iterations: usize, residual: f64) -> Self {
        Self {
            value,
            info: ConvergenceInfo {
                iterations,
                residual,
                converged: true,
            },
        }
    }

    /// Extrait juste la valeur, jette la metadata.
    pub fn into_inner(self) -> T {
        self.value
    }
}

/// Critères de tolérance partagés.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub abs: f64,
    pub rel: f64,
    pub max_iter: usize,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            abs: 1e-10,
            rel: 1e-8,
            max_iter: 200,
        }
    }
}

impl Tolerance {
    pub fn new(abs: f64, rel: f64, max_iter: usize) -> Self {
        Self {
            abs,
            rel,
            max_iter,
        }
    }

    /// Test : `|residual| <= abs + rel * |reference|`.
    pub fn satisfied(&self, residual: f64, reference: f64) -> bool {
        residual.abs() <= self.abs + self.rel * reference.abs()
    }
}
