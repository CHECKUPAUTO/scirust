//! Correctness qualification for explicit SGEMM candidates.
//!
//! Timing evidence is meaningless until a candidate is proven numerically
//! admissible against SciRust's scalar reference on the same inputs. This module
//! performs that gate and returns deterministic error summaries suitable for a
//! higher-level autotuner evidence record.

use super::backend::{ScalarBackend, SimdBackend};
use super::candidate_plan::{CandidateGemmPlanError, CandidateGemmPlanF32};
use super::gemm_candidates::{GemmCandidateDescriptor, GemmProblemSignature};
use super::view::{MatrixView, MatrixViewMut};

/// Numerical acceptance policy for one GEMM candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GemmQualificationPolicy {
    pub abs_tolerance: f32,
    pub rel_tolerance: f32,
}

impl Default for GemmQualificationPolicy {
    fn default() -> Self {
        Self {
            abs_tolerance: 1.0e-4,
            rel_tolerance: 1.0e-3,
        }
    }
}

/// Deterministic qualification summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GemmQualificationReport {
    pub candidate: GemmCandidateDescriptor,
    pub element_count: usize,
    pub max_abs_error: f32,
    pub max_rel_error: f32,
    pub finite: bool,
    pub accepted: bool,
}

/// Run the scalar oracle and the exact requested candidate on identical inputs.
///
/// This function allocates oracle/candidate output buffers because qualification
/// is an out-of-band planning step. Repeated candidate execution itself remains
/// allocation-free through [`CandidateGemmPlanF32`].
pub fn qualify_gemm_candidate_f32(
    problem: GemmProblemSignature,
    candidate: GemmCandidateDescriptor,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    initial_c: &[f32],
    policy: GemmQualificationPolicy,
) -> Result<GemmQualificationReport, GemmQualificationError> {
    if !policy.abs_tolerance.is_finite()
        || !policy.rel_tolerance.is_finite()
        || policy.abs_tolerance < 0.0
        || policy.rel_tolerance < 0.0
    {
        return Err(GemmQualificationError::InvalidTolerance);
    }

    let a_expected = checked_product(problem.m, problem.k)?;
    let b_expected = checked_product(problem.k, problem.n)?;
    let c_expected = checked_product(problem.m, problem.n)?;
    if a.len() != a_expected
    {
        return Err(GemmQualificationError::ALength {
            expected: a_expected,
            actual: a.len(),
        });
    }
    if b.len() != b_expected
    {
        return Err(GemmQualificationError::BLength {
            expected: b_expected,
            actual: b.len(),
        });
    }
    if initial_c.len() != c_expected
    {
        return Err(GemmQualificationError::CLength {
            expected: c_expected,
            actual: initial_c.len(),
        });
    }

    let mut oracle = initial_c.to_vec();
    ScalarBackend.sgemm_f32(
        alpha,
        MatrixView::new(a, problem.m, problem.k),
        MatrixView::new(b, problem.k, problem.n),
        beta,
        MatrixViewMut::new(&mut oracle, problem.m, problem.n),
    );

    let mut observed = initial_c.to_vec();
    let mut plan = CandidateGemmPlanF32::prepare(problem, candidate)
        .map_err(GemmQualificationError::Plan)?;
    plan.execute(alpha, a, b, beta, &mut observed)
        .map_err(GemmQualificationError::Plan)?;

    let mut max_abs_error = 0.0_f32;
    let mut max_rel_error = 0.0_f32;
    let mut finite = true;
    let mut accepted = true;

    for (&expected, &actual) in oracle.iter().zip(&observed)
    {
        if !expected.is_finite() || !actual.is_finite()
        {
            finite = false;
            accepted = false;
            continue;
        }
        let abs_error = (actual - expected).abs();
        let scale = expected.abs().max(actual.abs());
        let rel_error = if scale == 0.0
        {
            0.0
        }
        else
        {
            abs_error / scale
        };
        max_abs_error = max_abs_error.max(abs_error);
        max_rel_error = max_rel_error.max(rel_error);

        let allowed = policy.abs_tolerance + policy.rel_tolerance * expected.abs();
        if abs_error > allowed
        {
            accepted = false;
        }
    }

    Ok(GemmQualificationReport {
        candidate,
        element_count: c_expected,
        max_abs_error,
        max_rel_error,
        finite,
        accepted,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum GemmQualificationError {
    InvalidTolerance,
    ShapeOverflow,
    ALength { expected: usize, actual: usize },
    BLength { expected: usize, actual: usize },
    CLength { expected: usize, actual: usize },
    Plan(CandidateGemmPlanError),
}

impl core::fmt::Display for GemmQualificationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::InvalidTolerance => write!(
                f,
                "GEMM qualification tolerances must be finite and non-negative"
            ),
            Self::ShapeOverflow => write!(f, "GEMM dimensions overflow usize"),
            Self::ALength { expected, actual } => write!(
                f,
                "GEMM A length mismatch: expected {expected}, got {actual}"
            ),
            Self::BLength { expected, actual } => write!(
                f,
                "GEMM B length mismatch: expected {expected}, got {actual}"
            ),
            Self::CLength { expected, actual } => write!(
                f,
                "GEMM C length mismatch: expected {expected}, got {actual}"
            ),
            Self::Plan(error) => write!(f, "GEMM candidate plan failed: {error}"),
        }
    }
}

impl std::error::Error for GemmQualificationError {}

fn checked_product(left: usize, right: usize) -> Result<usize, GemmQualificationError> {
    left.checked_mul(right)
        .ok_or(GemmQualificationError::ShapeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::gemm_candidates::available_gemm_candidates_f32;

    #[test]
    fn every_available_candidate_qualifies_on_deterministic_fixture() {
        let problem = GemmProblemSignature::new(13, 17, 11).unwrap();
        let a: Vec<f32> = (0..problem.m * problem.k)
            .map(|index| (index % 23) as f32 * 0.03125 - 0.25)
            .collect();
        let b: Vec<f32> = (0..problem.k * problem.n)
            .map(|index| (index % 19) as f32 * 0.025 - 0.2)
            .collect();
        let c = vec![0.125_f32; problem.m * problem.n];

        for candidate in available_gemm_candidates_f32(problem)
        {
            let report = qualify_gemm_candidate_f32(
                problem,
                candidate,
                0.875,
                &a,
                &b,
                0.25,
                &c,
                GemmQualificationPolicy::default(),
            )
            .unwrap();
            assert!(
                report.finite,
                "candidate {:?} produced non-finite output",
                candidate.path
            );
            assert!(report.accepted, "candidate {:?}: {report:?}", candidate.path);
        }
    }

    #[test]
    fn invalid_tolerance_is_rejected_before_execution() {
        let problem = GemmProblemSignature::new(1, 1, 1).unwrap();
        let candidate = available_gemm_candidates_f32(problem)[0];
        let error = qualify_gemm_candidate_f32(
            problem,
            candidate,
            1.0,
            &[1.0],
            &[1.0],
            0.0,
            &[0.0],
            GemmQualificationPolicy {
                abs_tolerance: -1.0,
                rel_tolerance: 0.0,
            },
        )
        .unwrap_err();
        assert_eq!(error, GemmQualificationError::InvalidTolerance);
    }
}
