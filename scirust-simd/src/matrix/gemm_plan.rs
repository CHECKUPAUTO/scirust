//! Prepared SGEMM execution plans.
//!
//! A [`GemmPlanF32`] freezes matrix dimensions and the preferred execution path
//! once. Repeated execution validates caller buffers, reuses caller-owned packing
//! storage and performs no heap allocation. Architecture feature checks remain in
//! the packed kernel entrypoint as a safety guard; candidate/path selection itself
//! is not rebuilt by the plan.

use super::backend::{ScalarBackend, SimdBackend};
use super::view::{MatrixView, MatrixViewMut};
use super::workspace_gemm::{GemmWorkspaceF32, sgemm_tiled_with_workspace};

/// Execution family selected when a plan is prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmExecutionPath {
    /// Existing deterministic scalar reference backend.
    Scalar,
    /// Packed AVX-512 8×16 path from `workspace_gemm`.
    Avx512Packed,
    /// Packed AArch64 NEON 8×8 path from `workspace_gemm`.
    NeonPacked,
}

/// Shape validation failure before a prepared GEMM touches any output memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmPlanError {
    ShapeOverflow,
    ALength { expected: usize, actual: usize },
    BLength { expected: usize, actual: usize },
    CLength { expected: usize, actual: usize },
}

impl core::fmt::Display for GemmPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self
        {
            Self::ShapeOverflow => write!(f, "GEMM dimensions overflow usize"),
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

impl std::error::Error for GemmPlanError {}

/// Prepared row-major `f32` GEMM plan for fixed `(m, k, n)` dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmPlanF32 {
    m: usize,
    k: usize,
    n: usize,
    path: GemmExecutionPath,
}

impl GemmPlanF32 {
    /// Prepare a plan using the best currently implemented reusable path on this
    /// machine. This performs feature selection once, outside repeated execution.
    pub fn prepare(m: usize, k: usize, n: usize) -> Result<Self, GemmPlanError> {
        checked_product(m, k)?;
        checked_product(k, n)?;
        checked_product(m, n)?;
        Ok(Self {
            m,
            k,
            n,
            path: preferred_path(),
        })
    }

    pub const fn m(&self) -> usize {
        self.m
    }

    pub const fn k(&self) -> usize {
        self.k
    }

    pub const fn n(&self) -> usize {
        self.n
    }

    pub const fn path(&self) -> GemmExecutionPath {
        self.path
    }

    /// Create packing storage suitable for repeated execution of this plan.
    /// Allocation is explicit and occurs outside [`Self::execute`].
    pub fn create_workspace(&self) -> GemmWorkspaceF32 {
        GemmWorkspaceF32::new()
    }

    /// Execute `C = alpha·A·B + beta·C` into caller-owned output and workspace.
    ///
    /// No heap allocation is performed by this method. The packed path delegates
    /// to the reusable kernel introduced by `workspace_gemm`; the scalar path
    /// bypasses packing entirely.
    pub fn execute(
        &self,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
        workspace: &mut GemmWorkspaceF32,
    ) -> Result<(), GemmPlanError> {
        let a_expected = checked_product(self.m, self.k)?;
        let b_expected = checked_product(self.k, self.n)?;
        let c_expected = checked_product(self.m, self.n)?;
        if a.len() != a_expected
        {
            return Err(GemmPlanError::ALength {
                expected: a_expected,
                actual: a.len(),
            });
        }
        if b.len() != b_expected
        {
            return Err(GemmPlanError::BLength {
                expected: b_expected,
                actual: b.len(),
            });
        }
        if c.len() != c_expected
        {
            return Err(GemmPlanError::CLength {
                expected: c_expected,
                actual: c.len(),
            });
        }

        let a_view = MatrixView::new(a, self.m, self.k);
        let b_view = MatrixView::new(b, self.k, self.n);
        let c_view = MatrixViewMut::new(c, self.m, self.n);

        match self.path
        {
            GemmExecutionPath::Scalar =>
            {
                ScalarBackend.sgemm_f32(alpha, a_view, b_view, beta, c_view);
            },
            GemmExecutionPath::Avx512Packed | GemmExecutionPath::NeonPacked =>
            {
                // `sgemm_tiled_with_workspace` retains a hardware feature guard at
                // the unsafe-intrinsic boundary. That guard is intentionally kept
                // even though the plan already selected the path during prepare().
                sgemm_tiled_with_workspace(alpha, a_view, b_view, beta, c_view, workspace);
            },
        }
        Ok(())
    }
}

fn checked_product(left: usize, right: usize) -> Result<usize, GemmPlanError> {
    left.checked_mul(right).ok_or(GemmPlanError::ShapeOverflow)
}

fn preferred_path() -> GemmExecutionPath {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f")
    {
        return GemmExecutionPath::Avx512Packed;
    }

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon")
    {
        return GemmExecutionPath::NeonPacked;
    }

    GemmExecutionPath::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_plan_matches_scalar_reference() {
        let (m, k, n) = (17_usize, 23_usize, 19_usize);
        let a: Vec<f32> = (0..m * k)
            .map(|index| (index % 29) as f32 * 0.025 - 0.3)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|index| (index % 31) as f32 * 0.02 - 0.25)
            .collect();
        let initial: Vec<f32> = (0..m * n).map(|index| (index % 7) as f32 * 0.01).collect();

        let mut expected = initial.clone();
        ScalarBackend.sgemm_f32(
            0.875,
            MatrixView::new(&a, m, k),
            MatrixView::new(&b, k, n),
            0.25,
            MatrixViewMut::new(&mut expected, m, n),
        );

        let plan = GemmPlanF32::prepare(m, k, n).unwrap();
        let selected = plan.path();
        let mut workspace = plan.create_workspace();
        let identities = workspace.buffer_identities();
        let capacities = workspace.capacities();

        for _ in 0..3
        {
            let mut got = initial.clone();
            plan.execute(0.875, &a, &b, 0.25, &mut got, &mut workspace)
                .unwrap();
            for index in 0..got.len()
            {
                let tolerance = 1e-3 * (1.0 + expected[index].abs());
                assert!(
                    (got[index] - expected[index]).abs() <= tolerance,
                    "index={index}: {} vs {}",
                    got[index],
                    expected[index]
                );
            }
            assert_eq!(plan.path(), selected);
            assert_eq!(workspace.buffer_identities(), identities);
            assert_eq!(workspace.capacities(), capacities);
        }
    }

    #[test]
    fn prepared_plan_rejects_wrong_buffer_length_before_execution() {
        let plan = GemmPlanF32::prepare(2, 3, 4).unwrap();
        let mut workspace = plan.create_workspace();
        let mut c = [0.0f32; 8];
        let error = plan
            .execute(1.0, &[0.0; 5], &[0.0; 12], 0.0, &mut c, &mut workspace)
            .unwrap_err();
        assert_eq!(
            error,
            GemmPlanError::ALength {
                expected: 6,
                actual: 5,
            }
        );
    }

    #[test]
    fn plan_rejects_overflowing_dimensions() {
        assert_eq!(
            GemmPlanF32::prepare(usize::MAX, 2, 1),
            Err(GemmPlanError::ShapeOverflow)
        );
    }
}
