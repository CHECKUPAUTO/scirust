//! Prepared execution of one explicit SGEMM candidate.
//!
//! `GemmPlanF32::prepare` intentionally chooses the preferred implementation.
//! Autotuning needs a different primitive: execute a *specific* candidate so
//! competing implementations can be validated and measured under identical
//! inputs. This module provides that primitive while keeping allocation outside
//! repeated execution.

use super::backend::{ScalarBackend, SimdBackend};
use super::gemm_candidates::{
    GemmCandidateDescriptor, GemmProblemSignature, available_gemm_candidates_f32,
};
use super::gemm_plan::GemmExecutionPath;
use super::view::{MatrixView, MatrixViewMut};
use super::workspace_gemm::{GemmWorkspaceF32, sgemm_tiled_with_workspace};

/// Prepared explicit-candidate plan for fixed `m×k · k×n` dimensions.
#[derive(Debug)]
pub struct CandidateGemmPlanF32 {
    problem: GemmProblemSignature,
    candidate: GemmCandidateDescriptor,
    workspace: Option<GemmWorkspaceF32>,
}

impl CandidateGemmPlanF32 {
    /// Prepare exactly `candidate` when it belongs to the executable candidate
    /// set for this host. Packing storage is allocated here, never by execute.
    pub fn prepare(
        problem: GemmProblemSignature,
        candidate: GemmCandidateDescriptor,
    ) -> Result<Self, CandidateGemmPlanError> {
        if !available_gemm_candidates_f32(problem).contains(&candidate)
        {
            return Err(CandidateGemmPlanError::CandidateUnavailable);
        }
        let workspace = match candidate.path
        {
            GemmExecutionPath::Scalar => None,
            GemmExecutionPath::Avx512Packed | GemmExecutionPath::NeonPacked =>
            {
                Some(GemmWorkspaceF32::new())
            },
        };
        Ok(Self {
            problem,
            candidate,
            workspace,
        })
    }

    pub const fn problem(&self) -> GemmProblemSignature {
        self.problem
    }

    pub const fn candidate(&self) -> GemmCandidateDescriptor {
        self.candidate
    }

    /// Backing-buffer identities for allocation-regression checks.
    pub fn workspace_identities(&self) -> Option<(usize, usize)> {
        self.workspace
            .as_ref()
            .map(GemmWorkspaceF32::buffer_identities)
    }

    /// Execute `C = alpha·A·B + beta·C` with the exact prepared candidate.
    ///
    /// No allocation, resize or candidate selection occurs in this method.
    pub fn execute(
        &mut self,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
    ) -> Result<(), CandidateGemmPlanError> {
        let a_expected = checked_product(self.problem.m, self.problem.k)?;
        let b_expected = checked_product(self.problem.k, self.problem.n)?;
        let c_expected = checked_product(self.problem.m, self.problem.n)?;
        if a.len() != a_expected
        {
            return Err(CandidateGemmPlanError::ALength {
                expected: a_expected,
                actual: a.len(),
            });
        }
        if b.len() != b_expected
        {
            return Err(CandidateGemmPlanError::BLength {
                expected: b_expected,
                actual: b.len(),
            });
        }
        if c.len() != c_expected
        {
            return Err(CandidateGemmPlanError::CLength {
                expected: c_expected,
                actual: c.len(),
            });
        }

        let a_view = MatrixView::new(a, self.problem.m, self.problem.k);
        let b_view = MatrixView::new(b, self.problem.k, self.problem.n);
        let c_view = MatrixViewMut::new(c, self.problem.m, self.problem.n);
        match self.candidate.path
        {
            GemmExecutionPath::Scalar =>
            {
                ScalarBackend.sgemm_f32(alpha, a_view, b_view, beta, c_view);
            },
            GemmExecutionPath::Avx512Packed | GemmExecutionPath::NeonPacked =>
            {
                let workspace = self
                    .workspace
                    .as_mut()
                    .ok_or(CandidateGemmPlanError::MissingWorkspace)?;
                sgemm_tiled_with_workspace(alpha, a_view, b_view, beta, c_view, workspace);
            },
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateGemmPlanError {
    CandidateUnavailable,
    ShapeOverflow,
    MissingWorkspace,
    ALength { expected: usize, actual: usize },
    BLength { expected: usize, actual: usize },
    CLength { expected: usize, actual: usize },
}

impl core::fmt::Display for CandidateGemmPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self
        {
            Self::CandidateUnavailable =>
            {
                write!(f, "GEMM candidate is not executable on this host")
            },
            Self::ShapeOverflow => write!(f, "GEMM dimensions overflow usize"),
            Self::MissingWorkspace => write!(f, "packed GEMM candidate has no prepared workspace"),
            Self::ALength { expected, actual } =>
            {
                write!(
                    f,
                    "GEMM A length mismatch: expected {expected}, got {actual}"
                )
            },
            Self::BLength { expected, actual } =>
            {
                write!(
                    f,
                    "GEMM B length mismatch: expected {expected}, got {actual}"
                )
            },
            Self::CLength { expected, actual } =>
            {
                write!(
                    f,
                    "GEMM C length mismatch: expected {expected}, got {actual}"
                )
            },
        }
    }
}

impl std::error::Error for CandidateGemmPlanError {}

fn checked_product(left: usize, right: usize) -> Result<usize, CandidateGemmPlanError> {
    left.checked_mul(right)
        .ok_or(CandidateGemmPlanError::ShapeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_available_candidate_matches_scalar_reference() {
        let problem = GemmProblemSignature::new(17, 23, 19).unwrap();
        let a: Vec<f32> = (0..problem.m * problem.k)
            .map(|index| (index % 29) as f32 * 0.025 - 0.3)
            .collect();
        let b: Vec<f32> = (0..problem.k * problem.n)
            .map(|index| (index % 31) as f32 * 0.02 - 0.25)
            .collect();
        let initial: Vec<f32> = (0..problem.m * problem.n)
            .map(|index| (index % 7) as f32 * 0.01)
            .collect();

        let mut expected = initial.clone();
        ScalarBackend.sgemm_f32(
            0.875,
            MatrixView::new(&a, problem.m, problem.k),
            MatrixView::new(&b, problem.k, problem.n),
            0.25,
            MatrixViewMut::new(&mut expected, problem.m, problem.n),
        );

        for candidate in available_gemm_candidates_f32(problem)
        {
            let mut plan = CandidateGemmPlanF32::prepare(problem, candidate).unwrap();
            let identities = plan.workspace_identities();
            for _ in 0..2
            {
                let mut got = initial.clone();
                plan.execute(0.875, &a, &b, 0.25, &mut got).unwrap();
                for index in 0..got.len()
                {
                    let tolerance = 1e-3 * (1.0 + expected[index].abs());
                    assert!(
                        (got[index] - expected[index]).abs() <= tolerance,
                        "candidate={:?}, index={index}: {} vs {}",
                        candidate.path,
                        got[index],
                        expected[index]
                    );
                }
                assert_eq!(plan.workspace_identities(), identities);
            }
        }
    }

    #[test]
    fn wrong_lengths_fail_before_output_mutation() {
        let problem = GemmProblemSignature::new(2, 3, 4).unwrap();
        let candidate = available_gemm_candidates_f32(problem)[0];
        let mut plan = CandidateGemmPlanF32::prepare(problem, candidate).unwrap();
        let mut c = [7.0_f32; 8];
        let before = c;
        let error = plan
            .execute(1.0, &[0.0; 5], &[0.0; 12], 0.0, &mut c)
            .unwrap_err();
        assert_eq!(
            error,
            CandidateGemmPlanError::ALength {
                expected: 6,
                actual: 5
            }
        );
        assert_eq!(c, before);
    }
}
