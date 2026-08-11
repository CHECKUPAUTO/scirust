//! Deterministic candidate descriptors for prepared `f32` GEMM planning.
//!
//! This module deliberately does not benchmark or select a winner. It exposes
//! the execution families that are actually executable on the current host,
//! together with stable problem/candidate metadata that a higher-level tuner
//! (for example ElasticAutoTuner) can measure and qualify.

use super::gemm_plan::GemmExecutionPath;

/// Stable schema for [`GemmProblemSignature::class_key`].
pub const GEMM_PROBLEM_SCHEMA_VERSION: u32 = 1;

/// Current packed AVX-512 SGEMM register/cache blocking.
#[cfg(target_arch = "x86_64")]
const AVX512_MR: usize = 8;
#[cfg(target_arch = "x86_64")]
const AVX512_NR: usize = 16;
#[cfg(target_arch = "x86_64")]
const AVX512_KC: usize = 256;
#[cfg(target_arch = "x86_64")]
const AVX512_MC: usize = 256;
#[cfg(target_arch = "x86_64")]
const AVX512_NC: usize = 1024;

/// Current packed AArch64 NEON SGEMM register/cache blocking.
#[cfg(target_arch = "aarch64")]
const NEON_MR: usize = 8;
#[cfg(target_arch = "aarch64")]
const NEON_NR: usize = 8;
#[cfg(target_arch = "aarch64")]
const NEON_KC: usize = 256;
#[cfg(target_arch = "aarch64")]
const NEON_MC: usize = 256;
#[cfg(target_arch = "aarch64")]
const NEON_NC: usize = 512;

/// Shape identity used by tuning/plan caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GemmProblemSignature {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl GemmProblemSignature {
    /// Validate that all dense row-major extents fit in `usize`.
    pub fn new(m: usize, k: usize, n: usize) -> Result<Self, GemmCandidateError> {
        m.checked_mul(k).ok_or(GemmCandidateError::ShapeOverflow)?;
        k.checked_mul(n).ok_or(GemmCandidateError::ShapeOverflow)?;
        m.checked_mul(n).ok_or(GemmCandidateError::ShapeOverflow)?;
        Ok(Self { m, k, n })
    }

    /// Canonical little-endian class key for a tuner/cache.
    ///
    /// The encoding is independent of native `usize` width: dimensions are
    /// widened to `u64`, and construction rejects values that do not fit.
    pub fn class_key(self) -> Result<[u8; 28], GemmCandidateError> {
        let m = u64::try_from(self.m).map_err(|_| GemmCandidateError::DimensionTooLarge)?;
        let k = u64::try_from(self.k).map_err(|_| GemmCandidateError::DimensionTooLarge)?;
        let n = u64::try_from(self.n).map_err(|_| GemmCandidateError::DimensionTooLarge)?;
        let mut bytes = [0_u8; 28];
        bytes[0..4].copy_from_slice(&GEMM_PROBLEM_SCHEMA_VERSION.to_le_bytes());
        bytes[4..12].copy_from_slice(&m.to_le_bytes());
        bytes[12..20].copy_from_slice(&k.to_le_bytes());
        bytes[20..28].copy_from_slice(&n.to_le_bytes());
        Ok(bytes)
    }
}

/// One executable CPU SGEMM family on the current host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmCandidateDescriptor {
    pub path: GemmExecutionPath,
    /// Register tile height.
    pub mr: usize,
    /// Register tile width.
    pub nr: usize,
    /// K cache block.
    pub kc: usize,
    /// M cache block.
    pub mc: usize,
    /// N cache block.
    pub nc: usize,
    /// Packing storage required by the current implementation.
    pub temporary_bytes: usize,
    /// Whether execution preserves the implementation's documented deterministic
    /// accumulation order for a fixed path/target.
    pub deterministic: bool,
}

impl GemmCandidateDescriptor {
    /// Stable integer parameter tuple suitable for external tuner adapters.
    pub const fn tuning_parameters(self) -> [(&'static str, i64); 6] {
        [
            ("path", path_code(self.path)),
            ("mr", self.mr as i64),
            ("nr", self.nr as i64),
            ("kc", self.kc as i64),
            ("mc", self.mc as i64),
            ("nc", self.nc as i64),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmCandidateError {
    ShapeOverflow,
    DimensionTooLarge,
}

impl core::fmt::Display for GemmCandidateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::ShapeOverflow => write!(f, "GEMM dimensions overflow usize"),
            Self::DimensionTooLarge =>
            {
                write!(f, "GEMM dimension does not fit canonical u64 encoding")
            },
        }
    }
}

impl std::error::Error for GemmCandidateError {}

/// Enumerate only execution families that can actually run on this host.
///
/// Planning may allocate the returned `Vec`; repeated GEMM execution remains a
/// separate concern and uses `GemmPlanF32` / `GemmWorkspaceF32`.
pub fn available_gemm_candidates_f32(
    problem: GemmProblemSignature,
) -> Vec<GemmCandidateDescriptor> {
    let mut candidates = Vec::with_capacity(2);

    // Scalar is the always-available correctness/reference family. It carries no
    // packing workspace requirement.
    candidates.push(GemmCandidateDescriptor {
        path: GemmExecutionPath::Scalar,
        mr: 1,
        nr: 1,
        kc: problem.k.max(1),
        mc: problem.m.max(1),
        nc: problem.n.max(1),
        temporary_bytes: 0,
        deterministic: true,
    });

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f")
    {
        let a_elems = AVX512_KC * AVX512_MC.div_ceil(AVX512_MR) * AVX512_MR;
        let b_elems = AVX512_KC * AVX512_NC.div_ceil(AVX512_NR) * AVX512_NR;
        candidates.push(GemmCandidateDescriptor {
            path: GemmExecutionPath::Avx512Packed,
            mr: AVX512_MR,
            nr: AVX512_NR,
            kc: AVX512_KC,
            mc: AVX512_MC,
            nc: AVX512_NC,
            temporary_bytes: (a_elems + b_elems) * core::mem::size_of::<f32>(),
            deterministic: true,
        });
    }

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon")
    {
        let a_elems = NEON_KC * NEON_MC.div_ceil(NEON_MR) * NEON_MR;
        let b_elems = NEON_KC * NEON_NC.div_ceil(NEON_NR) * NEON_NR;
        candidates.push(GemmCandidateDescriptor {
            path: GemmExecutionPath::NeonPacked,
            mr: NEON_MR,
            nr: NEON_NR,
            kc: NEON_KC,
            mc: NEON_MC,
            nc: NEON_NC,
            temporary_bytes: (a_elems + b_elems) * core::mem::size_of::<f32>(),
            deterministic: true,
        });
    }

    candidates
}

const fn path_code(path: GemmExecutionPath) -> i64 {
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

    #[test]
    fn class_key_is_stable_and_dimension_order_sensitive() {
        let a = GemmProblemSignature::new(16, 32, 64)
            .unwrap()
            .class_key()
            .unwrap();
        let b = GemmProblemSignature::new(16, 64, 32)
            .unwrap()
            .class_key()
            .unwrap();
        assert_ne!(a, b);
        assert_eq!(&a[0..4], &GEMM_PROBLEM_SCHEMA_VERSION.to_le_bytes());
    }

    #[test]
    fn scalar_candidate_is_always_present_and_requires_no_scratch() {
        let problem = GemmProblemSignature::new(7, 11, 13).unwrap();
        let candidates = available_gemm_candidates_f32(problem);
        let scalar = candidates
            .iter()
            .find(|candidate| candidate.path == GemmExecutionPath::Scalar)
            .unwrap();
        assert_eq!(scalar.temporary_bytes, 0);
        assert!(scalar.deterministic);
    }

    #[test]
    fn candidate_order_and_identity_are_stable() {
        let problem = GemmProblemSignature::new(64, 64, 64).unwrap();
        let first = available_gemm_candidates_f32(problem);
        let second = available_gemm_candidates_f32(problem);
        assert_eq!(first, second);
        assert_eq!(first[0].path, GemmExecutionPath::Scalar);
        for candidate in first
        {
            assert!(candidate.mr > 0);
            assert!(candidate.nr > 0);
            assert!(candidate.kc > 0);
            assert!(candidate.mc > 0);
            assert!(candidate.nc > 0);
        }
    }

    #[test]
    fn rejects_overflowing_dense_shapes() {
        assert_eq!(
            GemmProblemSignature::new(usize::MAX, 2, 1),
            Err(GemmCandidateError::ShapeOverflow)
        );
    }
}
