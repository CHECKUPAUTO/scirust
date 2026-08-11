//! Canonical correctness evidence for qualified SGEMM candidates.
//!
//! The encoder is intentionally dependency-free and uses a fixed little-endian
//! binary layout. A higher-level autotuner can store the returned bytes directly
//! as opaque correctness evidence without importing SciRust SIMD internals.

use super::candidate_qualification::{GemmQualificationPolicy, GemmQualificationReport};
use super::gemm_candidates::GemmProblemSignature;
use super::gemm_plan::GemmExecutionPath;

/// Schema version for [`encode_gemm_correctness_evidence`].
pub const GEMM_CORRECTNESS_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Encode an accepted finite qualification report into canonical bytes.
///
/// Rejected/non-finite reports are deliberately not encodable as positive
/// evidence. Every integer is little-endian and every float is represented by
/// its IEEE-754 bit pattern, so identical evidence produces identical bytes.
pub fn encode_gemm_correctness_evidence(
    problem: GemmProblemSignature,
    policy: GemmQualificationPolicy,
    report: GemmQualificationReport,
) -> Result<Vec<u8>, GemmCorrectnessEvidenceError> {
    if !report.finite
    {
        return Err(GemmCorrectnessEvidenceError::NonFiniteReport);
    }
    if !report.accepted
    {
        return Err(GemmCorrectnessEvidenceError::RejectedReport);
    }

    let class_key = problem
        .class_key()
        .map_err(|_| GemmCorrectnessEvidenceError::DimensionTooLarge)?;
    let expected_elements = problem
        .m
        .checked_mul(problem.n)
        .ok_or(GemmCorrectnessEvidenceError::DimensionTooLarge)?;
    if report.element_count != expected_elements
    {
        return Err(GemmCorrectnessEvidenceError::ElementCountMismatch {
            expected: expected_elements,
            actual: report.element_count,
        });
    }

    let mut out = Vec::with_capacity(128);
    push_u32(&mut out, GEMM_CORRECTNESS_EVIDENCE_SCHEMA_VERSION);
    out.extend_from_slice(&class_key);
    push_u8(&mut out, path_code(report.candidate.path));
    push_usize(&mut out, report.candidate.mr)?;
    push_usize(&mut out, report.candidate.nr)?;
    push_usize(&mut out, report.candidate.kc)?;
    push_usize(&mut out, report.candidate.mc)?;
    push_usize(&mut out, report.candidate.nc)?;
    push_usize(&mut out, report.candidate.temporary_bytes)?;
    push_u8(&mut out, u8::from(report.candidate.deterministic));
    push_u32(&mut out, policy.abs_tolerance.to_bits());
    push_u32(&mut out, policy.rel_tolerance.to_bits());
    push_u32(&mut out, report.max_abs_error.to_bits());
    push_u32(&mut out, report.max_rel_error.to_bits());
    push_usize(&mut out, report.element_count)?;
    push_u8(&mut out, u8::from(report.finite));
    push_u8(&mut out, u8::from(report.accepted));
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmCorrectnessEvidenceError {
    NonFiniteReport,
    RejectedReport,
    DimensionTooLarge,
    ElementCountMismatch { expected: usize, actual: usize },
}

impl core::fmt::Display for GemmCorrectnessEvidenceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self
        {
            Self::NonFiniteReport =>
            {
                write!(f, "cannot encode non-finite GEMM qualification evidence")
            },
            Self::RejectedReport => write!(f, "cannot encode rejected GEMM qualification evidence"),
            Self::DimensionTooLarge =>
            {
                write!(f, "GEMM evidence field does not fit canonical u64 encoding")
            },
            Self::ElementCountMismatch { expected, actual } => write!(
                f,
                "GEMM qualification element count mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for GemmCorrectnessEvidenceError {}

fn push_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_usize(out: &mut Vec<u8>, value: usize) -> Result<(), GemmCorrectnessEvidenceError> {
    let value =
        u64::try_from(value).map_err(|_| GemmCorrectnessEvidenceError::DimensionTooLarge)?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

const fn path_code(path: GemmExecutionPath) -> u8 {
    match path
    {
        GemmExecutionPath::Scalar => 0,
        GemmExecutionPath::Avx512Packed => 1,
        GemmExecutionPath::NeonPacked => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::gemm_candidates::available_gemm_candidates_f32;

    fn accepted_fixture() -> (
        GemmProblemSignature,
        GemmQualificationPolicy,
        GemmQualificationReport,
    ) {
        let problem = GemmProblemSignature::new(3, 5, 7).unwrap();
        let candidate = available_gemm_candidates_f32(problem)[0];
        let policy = GemmQualificationPolicy::default();
        let report = GemmQualificationReport {
            candidate,
            element_count: 21,
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            finite: true,
            accepted: true,
        };
        (problem, policy, report)
    }

    #[test]
    fn identical_evidence_is_byte_identical() {
        let (problem, policy, report) = accepted_fixture();
        let first = encode_gemm_correctness_evidence(problem, policy, report).unwrap();
        let second = encode_gemm_correctness_evidence(problem, policy, report).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn policy_or_shape_changes_evidence_identity() {
        let (problem, policy, report) = accepted_fixture();
        let baseline = encode_gemm_correctness_evidence(problem, policy, report).unwrap();

        let changed_policy = GemmQualificationPolicy {
            abs_tolerance: policy.abs_tolerance * 2.0,
            ..policy
        };
        let changed = encode_gemm_correctness_evidence(problem, changed_policy, report).unwrap();
        assert_ne!(baseline, changed);

        let other_problem = GemmProblemSignature::new(7, 5, 3).unwrap();
        let mut other_report = report;
        other_report.element_count = 21;
        let changed =
            encode_gemm_correctness_evidence(other_problem, policy, other_report).unwrap();
        assert_ne!(baseline, changed);
    }

    #[test]
    fn rejected_or_non_finite_reports_cannot_be_positive_evidence() {
        let (problem, policy, mut report) = accepted_fixture();
        report.accepted = false;
        assert_eq!(
            encode_gemm_correctness_evidence(problem, policy, report),
            Err(GemmCorrectnessEvidenceError::RejectedReport)
        );

        report.accepted = true;
        report.finite = false;
        assert_eq!(
            encode_gemm_correctness_evidence(problem, policy, report),
            Err(GemmCorrectnessEvidenceError::NonFiniteReport)
        );
    }

    #[test]
    fn mismatched_element_count_is_rejected() {
        let (problem, policy, mut report) = accepted_fixture();
        report.element_count = 20;
        assert_eq!(
            encode_gemm_correctness_evidence(problem, policy, report),
            Err(GemmCorrectnessEvidenceError::ElementCountMismatch {
                expected: 21,
                actual: 20,
            })
        );
    }
}
