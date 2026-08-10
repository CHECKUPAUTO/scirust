//! Reusable packing workspace for the tiled `f32` GEMM hot path.
//!
//! The existing `crate::gemm::sgemm_tiled` API remains the convenient one-shot
//! entry point. This module exposes a prepared form whose packing storage is
//! allocated once and then reused by every execution.

use super::backend::{ScalarBackend, SimdBackend};
use super::view::{MatrixView, MatrixViewMut};

#[cfg(target_arch = "x86_64")]
const MR: usize = 8;
#[cfg(target_arch = "x86_64")]
const NR: usize = 16;
#[cfg(target_arch = "x86_64")]
const KC: usize = 256;
#[cfg(target_arch = "x86_64")]
const MC: usize = 256;
#[cfg(target_arch = "x86_64")]
const NC: usize = 1024;

#[cfg(target_arch = "aarch64")]
const MR_N: usize = 8;
#[cfg(target_arch = "aarch64")]
const NR_N: usize = 8;
#[cfg(target_arch = "aarch64")]
const KC_N: usize = 256;
#[cfg(target_arch = "aarch64")]
const MC_N: usize = 256;
#[cfg(target_arch = "aarch64")]
const NC_N: usize = 512;

/// Caller-reusable packing storage for [`sgemm_tiled_with_workspace`].
#[derive(Debug)]
pub struct GemmWorkspaceF32 {
    a_pack: Vec<f32>,
    b_pack: Vec<f32>,
}

impl GemmWorkspaceF32 {
    /// Allocate the largest packing panels required by the target-specific path.
    pub fn new() -> Self {
        Self::new_for_target()
    }

    #[cfg(target_arch = "x86_64")]
    fn new_for_target() -> Self {
        Self {
            a_pack: vec![0.0; KC * MC.div_ceil(MR) * MR],
            b_pack: vec![0.0; KC * NC.div_ceil(NR) * NR],
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn new_for_target() -> Self {
        Self {
            a_pack: vec![0.0; KC_N * MC_N.div_ceil(MR_N) * MR_N],
            b_pack: vec![0.0; KC_N * NC_N.div_ceil(NR_N) * NR_N],
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn new_for_target() -> Self {
        Self {
            a_pack: Vec::new(),
            b_pack: Vec::new(),
        }
    }

    /// Current capacities, useful for allocation-regression instrumentation.
    pub fn capacities(&self) -> (usize, usize) {
        (self.a_pack.capacity(), self.b_pack.capacity())
    }

    /// Stable backing-buffer identities for allocation-regression tests.
    pub fn buffer_identities(&self) -> (usize, usize) {
        (self.a_pack.as_ptr() as usize, self.b_pack.as_ptr() as usize)
    }
}

impl Default for GemmWorkspaceF32 {
    fn default() -> Self {
        Self::new()
    }
}

/// `C = alpha·A·B + beta·C`, row-major, using caller-owned reusable scratch.
///
/// No allocation, resize, or replacement of `workspace` storage occurs during
/// this call. x86_64 uses the packed AVX-512 8×16 micro-kernel when AVX-512F is
/// available; AArch64 uses the packed NEON 8×8 micro-kernel; other paths retain
/// the existing scalar oracle.
pub fn sgemm_tiled_with_workspace(
    alpha: f32,
    a: MatrixView<f32>,
    b: MatrixView<f32>,
    beta: f32,
    c: MatrixViewMut<f32>,
    workspace: &mut GemmWorkspaceF32,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    assert_eq!(
        b.rows(),
        k,
        "sgemm_tiled_with_workspace: A.cols != B.rows"
    );
    assert_eq!(
        c.rows(),
        m,
        "sgemm_tiled_with_workspace: C.rows != A.rows"
    );
    assert_eq!(
        c.cols(),
        n,
        "sgemm_tiled_with_workspace: C.cols != B.cols"
    );

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f")
    {
        // SAFETY: AVX-512F is checked above and MatrixView validates extents.
        unsafe { sgemm_avx512(alpha, a, b, beta, c, workspace) };
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon")
    {
        // SAFETY: NEON availability is checked above; matrix views are bounded.
        unsafe { sgemm_neon(alpha, a, b, beta, c, workspace) };
        return;
    }

    ScalarBackend.sgemm_f32(alpha, a, b, beta, c);
}

/// Apply beta once before packed K-block accumulation.
unsafe fn scale_c(beta: f32, m: usize, n: usize, c: *mut f32) {
    if beta == 1.0
    {
        return;
    }
    for index in 0..m * n
    {
        *c.add(index) = if beta == 0.0
        {
            0.0
        }
        else
        {
            *c.add(index) * beta
        };
    }
}

/// Pack a B panel as `kc × NR`, padding the final column panel with zeros.
unsafe fn pack_b<const BLOCK_NR: usize>(
    b: *const f32,
    ldb: usize,
    kc: usize,
    nc: usize,
    dst: &mut [f32],
) {
    let panels = nc.div_ceil(BLOCK_NR);
    debug_assert!(dst.len() >= panels * kc * BLOCK_NR);
    for panel in 0..panels
    {
        let j0 = panel * BLOCK_NR;
        let nr = BLOCK_NR.min(nc - j0);
        let base = panel * kc * BLOCK_NR;
        for p in 0..kc
        {
            let source = b.add(p * ldb + j0);
            let output = base + p * BLOCK_NR;
            for j in 0..nr
            {
                dst[output + j] = *source.add(j);
            }
            dst[output + nr..output + BLOCK_NR].fill(0.0);
        }
    }
}

/// Pack an A panel as `kc ×MR`, fusing alpha and padding final rows with zero.
unsafe fn pack_a<const BLOCK_MR: usize>(
    alpha: f32,
    a: *const f32,
    lda: usize,
    mc: usize,
    kc: usize,
    dst: &mut [f32],
) {
    let panels = mc.div_ceil(BLOCK_MR);
    debug_assert!(dst.len() >= panels * kc * BLOCK_MR);
    for panel in 0..panels
    {
        let i0 = panel * BLOCK_MR;
        let mr = BLOCK_MR.min(mc - i0);
        let base = panel * kc * BLOCK_MR;
        for p in 0..kc
        {
            let output = base + p * BLOCK_MR;
            for i in 0..mr
            {
                dst[output + i] = alpha * *a.add((i0 + i) * lda + p);
            }
            dst[output + mr..output + BLOCK_MR].fill(0.0);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn micro_kernel_avx512(
    a_pack: *const f32,
    b_pack: *const f32,
    kc: usize,
    mr: usize,
    nr: usize,
    c: *mut f32,
    ldc: usize,
) {
    use core::arch::x86_64::*;

    let mask = if nr == NR
    {
        u16::MAX
    }
    else
    {
        (1_u16 << nr) - 1
    };
    let mut accumulators = [_mm512_setzero_ps(); MR];
    for p in 0..kc
    {
        let b_vector = _mm512_loadu_ps(b_pack.add(p * NR));
        let a_row = a_pack.add(p * MR);
        for (i, accumulator) in accumulators.iter_mut().enumerate()
        {
            *accumulator =
                _mm512_fmadd_ps(_mm512_set1_ps(*a_row.add(i)), b_vector, *accumulator);
        }
    }
    for (i, accumulator) in accumulators.iter().enumerate().take(mr)
    {
        let c_row = c.add(i * ldc);
        let previous = _mm512_maskz_loadu_ps(mask, c_row);
        _mm512_mask_storeu_ps(c_row, mask, _mm512_add_ps(previous, *accumulator));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_avx512(
    alpha: f32,
    a: MatrixView<f32>,
    b: MatrixView<f32>,
    beta: f32,
    mut c: MatrixViewMut<f32>,
    workspace: &mut GemmWorkspaceF32,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    if m == 0 || n == 0
    {
        return;
    }
    let c_ptr = c.row_slice_mut(0).expect("C base").as_mut_ptr();
    scale_c(beta, m, n, c_ptr);
    if k == 0 || alpha == 0.0
    {
        return;
    }

    let a_ptr = a.row_slice(0).expect("A base").as_ptr();
    let b_ptr = b.row_slice(0).expect("B base").as_ptr();
    let mut jc = 0;
    while jc < n
    {
        let nc = NC.min(n - jc);
        let n_panels = nc.div_ceil(NR);
        let mut pc = 0;
        while pc < k
        {
            let kc = KC.min(k - pc);
            pack_b::<NR>(
                b_ptr.add(pc * n + jc),
                n,
                kc,
                nc,
                &mut workspace.b_pack,
            );
            let mut ic = 0;
            while ic < m
            {
                let mc = MC.min(m - ic);
                pack_a::<MR>(
                    alpha,
                    a_ptr.add(ic * k + pc),
                    k,
                    mc,
                    kc,
                    &mut workspace.a_pack,
                );
                let m_panels = mc.div_ceil(MR);
                for ip in 0..m_panels
                {
                    let i0 = ic + ip * MR;
                    let mr = MR.min(m - i0);
                    let a_panel = workspace.a_pack.as_ptr().add(ip * kc * MR);
                    for jp in 0..n_panels
                    {
                        let j0 = jc + jp * NR;
                        let nr = NR.min(n - j0);
                        let b_panel = workspace.b_pack.as_ptr().add(jp * kc * NR);
                        micro_kernel_avx512(
                            a_panel,
                            b_panel,
                            kc,
                            mr,
                            nr,
                            c_ptr.add(i0 * n + j0),
                            n,
                        );
                    }
                }
                ic += MC;
            }
            pc += KC;
        }
        jc += NC;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn micro_kernel_neon(
    a_pack: *const f32,
    b_pack: *const f32,
    kc: usize,
    mr: usize,
    nr: usize,
    c: *mut f32,
    ldc: usize,
) {
    use core::arch::aarch64::*;

    let mut low = [vdupq_n_f32(0.0); MR_N];
    let mut high = [vdupq_n_f32(0.0); MR_N];
    for p in 0..kc
    {
        let b_low = vld1q_f32(b_pack.add(p * NR_N));
        let b_high = vld1q_f32(b_pack.add(p * NR_N + 4));
        let a_row = a_pack.add(p * MR_N);
        for i in 0..MR_N
        {
            let a_value = vdupq_n_f32(*a_row.add(i));
            low[i] = vfmaq_f32(low[i], a_value, b_low);
            high[i] = vfmaq_f32(high[i], a_value, b_high);
        }
    }

    let mut temporary = [0.0_f32; NR_N];
    for i in 0..mr
    {
        vst1q_f32(temporary.as_mut_ptr(), low[i]);
        vst1q_f32(temporary.as_mut_ptr().add(4), high[i]);
        let c_row = c.add(i * ldc);
        for (j, value) in temporary.iter().copied().enumerate().take(nr)
        {
            *c_row.add(j) += value;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sgemm_neon(
    alpha: f32,
    a: MatrixView<f32>,
    b: MatrixView<f32>,
    beta: f32,
    mut c: MatrixViewMut<f32>,
    workspace: &mut GemmWorkspaceF32,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    if m == 0 || n == 0
    {
        return;
    }
    let c_ptr = c.row_slice_mut(0).expect("C base").as_mut_ptr();
    scale_c(beta, m, n, c_ptr);
    if k == 0 || alpha == 0.0
    {
        return;
    }

    let a_ptr = a.row_slice(0).expect("A base").as_ptr();
    let b_ptr = b.row_slice(0).expect("B base").as_ptr();
    let mut jc = 0;
    while jc < n
    {
        let nc = NC_N.min(n - jc);
        let n_panels = nc.div_ceil(NR_N);
        let mut pc = 0;
        while pc < k
        {
            let kc = KC_N.min(k - pc);
            pack_b::<NR_N>(
                b_ptr.add(pc * n + jc),
                n,
                kc,
                nc,
                &mut workspace.b_pack,
            );
            let mut ic = 0;
            while ic < m
            {
                let mc = MC_N.min(m - ic);
                pack_a::<MR_N>(
                    alpha,
                    a_ptr.add(ic * k + pc),
                    k,
                    mc,
                    kc,
                    &mut workspace.a_pack,
                );
                let m_panels = mc.div_ceil(MR_N);
                for ip in 0..m_panels
                {
                    let i0 = ic + ip * MR_N;
                    let mr = MR_N.min(m - i0);
                    let a_panel = workspace.a_pack.as_ptr().add(ip * kc * MR_N);
                    for jp in 0..n_panels
                    {
                        let j0 = jc + jp * NR_N;
                        let nr = NR_N.min(n - j0);
                        let b_panel = workspace.b_pack.as_ptr().add(jp * kc * NR_N);
                        micro_kernel_neon(
                            a_panel,
                            b_panel,
                            kc,
                            mr,
                            nr,
                            c_ptr.add(i0 * n + j0),
                            n,
                        );
                    }
                }
                ic += MC_N;
            }
            pc += KC_N;
        }
        jc += NC_N;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reusable_workspace_matches_scalar_reference_and_keeps_allocations() {
        let (m, k, n) = (33_usize, 67_usize, 29_usize);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 73) as f32) * 0.013 - 0.4)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 61) as f32) * 0.017 - 0.5)
            .collect();
        let c0: Vec<f32> = (0..m * n).map(|i| ((i % 17) as f32) * 0.03 - 0.2).collect();

        let mut expected = c0.clone();
        ScalarBackend.sgemm_f32(
            0.75,
            MatrixView::new(&a, m, k),
            MatrixView::new(&b, k, n),
            -0.25,
            MatrixViewMut::new(&mut expected, m, n),
        );

        let mut workspace = GemmWorkspaceF32::new();
        let capacities = workspace.capacities();
        let identities = workspace.buffer_identities();

        for _ in 0..4
        {
            let mut got = c0.clone();
            sgemm_tiled_with_workspace(
                0.75,
                MatrixView::new(&a, m, k),
                MatrixView::new(&b, k, n),
                -0.25,
                MatrixViewMut::new(&mut got, m, n),
                &mut workspace,
            );
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
            assert_eq!(workspace.capacities(), capacities);
            assert_eq!(workspace.buffer_identities(), identities);
        }
    }

    #[test]
    fn zero_sized_gemm_preserves_workspace() {
        let mut workspace = GemmWorkspaceF32::new();
        let capacities = workspace.capacities();
        let identities = workspace.buffer_identities();
        let a = [];
        let b = [];
        let mut c = [];
        sgemm_tiled_with_workspace(
            1.0,
            MatrixView::new(&a, 0, 0),
            MatrixView::new(&b, 0, 0),
            0.0,
            MatrixViewMut::new(&mut c, 0, 0),
            &mut workspace,
        );
        assert_eq!(workspace.capacities(), capacities);
        assert_eq!(workspace.buffer_identities(), identities);
    }
}
