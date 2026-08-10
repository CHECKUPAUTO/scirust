//! Reusable packing workspace for the tiled `f32` GEMM hot path.
//!
//! The existing `crate::gemm::sgemm_tiled` API owns its packing buffers and is
//! convenient for one-shot calls. This module exposes the complementary
//! prepared-workspace form: allocate the pack panels once, then reuse them over
//! arbitrarily many GEMM invocations without growing or replacing either buffer.

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

/// Reusable packing storage for [`sgemm_tiled_with_workspace`].
///
/// Construction performs the only heap allocations needed by the packed SIMD
/// kernels. The two buffers retain fixed length and capacity for the lifetime of
/// the workspace, so repeated GEMM calls do not allocate in their hot path.
#[derive(Debug)]
pub struct GemmWorkspaceF32 {
    a_pack: Vec<f32>,
    b_pack: Vec<f32>,
}

impl GemmWorkspaceF32 {
    /// Allocates packing panels sized for the largest tile used by the current
    /// architecture-specific kernel.
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            return Self {
                a_pack: vec![0.0; KC * MC.div_ceil(MR) * MR],
                b_pack: vec![0.0; KC * NC.div_ceil(NR) * NR],
            };
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self {
                a_pack: vec![0.0; KC_N * MC_N.div_ceil(MR_N) * MR_N],
                b_pack: vec![0.0; KC_N * NC_N.div_ceil(NR_N) * NR_N],
            };
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                a_pack: Vec::new(),
                b_pack: Vec::new(),
            }
        }
    }

    /// Current backing-buffer capacities, useful for instrumentation and
    /// allocation-regression tests.
    pub fn capacities(&self) -> (usize, usize) {
        (self.a_pack.capacity(), self.b_pack.capacity())
    }

    /// Backing-buffer addresses, exposed only as integer identities so tests and
    /// profilers can prove that repeated execution keeps the same allocations.
    pub fn buffer_identities(&self) -> (usize, usize) {
        (self.a_pack.as_ptr() as usize, self.b_pack.as_ptr() as usize)
    }
}

impl Default for GemmWorkspaceF32 {
    fn default() -> Self {
        Self::new()
    }
}

/// Tiled SGEMM using caller-owned reusable packing storage.
///
/// `C = alpha·A·B + beta·C`, row-major. On x86_64 this dispatches to the packed
/// AVX-512 8×16 micro-kernel when AVX-512F is present. On AArch64 it uses the
/// packed NEON 8×8 micro-kernel. Other targets use the existing scalar oracle.
/// No buffer in `workspace` is allocated, resized, or replaced by this call.
pub fn sgemm_tiled_with_workspace(
    alpha: f32,
    a: MatrixView<f32>,
    b: MatrixView<f32>,
    beta: f32,
    c: MatrixViewMut<f32>,
    workspace: &mut GemmWorkspaceF32,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    assert_eq!(b.rows(), k, "sgemm_tiled_with_workspace: A.cols != B.rows");
    assert_eq!(c.rows(), m, "sgemm_tiled_with_workspace: C.rows != A.rows");
    assert_eq!(c.cols(), n, "sgemm_tiled_with_workspace: C.cols != B.cols");

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: runtime feature detection gates all AVX-512 instructions;
            // MatrixView/MatrixViewMut validate the row-major extents.
            unsafe { sgemm_avx512(alpha, a, b, beta, c, workspace) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon")
        {
            // SAFETY: NEON is detected above and all pointer arithmetic is
            // bounded by validated row-major matrix views and fixed pack panels.
            unsafe { sgemm_neon(alpha, a, b, beta, c, workspace) };
            return;
        }
    }
    ScalarBackend.sgemm_f32(alpha, a, b, beta, c);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scale_c_avx512(beta: f32, m: usize, n: usize, c: *mut f32) {
    use core::arch::x86_64::*;
    if beta == 1.0
    {
        return;
    }
    let bv = _mm512_set1_ps(beta);
    for i in 0..m
    {
        let row = c.add(i * n);
        let mut j = 0;
        while j + 16 <= n
        {
            let value = if beta == 0.0 {
                _mm512_setzero_ps()
            } else {
                _mm512_mul_ps(_mm512_loadu_ps(row.add(j)), bv)
            };
            _mm512_storeu_ps(row.add(j), value);
            j += 16;
        }
        let rem = n - j;
        if rem != 0
        {
            let mask = (1_u16 << rem) - 1;
            let value = if beta == 0.0 {
                _mm512_setzero_ps()
            } else {
                _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, row.add(j)), bv)
            };
            _mm512_mask_storeu_ps(row.add(j), mask, value);
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn pack_b_avx512(b: *const f32, ldb: usize, kc: usize, nc: usize, dst: &mut [f32]) {
    let panels = nc.div_ceil(NR);
    debug_assert!(dst.len() >= panels * kc * NR);
    for panel in 0..panels
    {
        let j0 = panel * NR;
        let nr = NR.min(nc - j0);
        let base = panel * kc * NR;
        for p in 0..kc
        {
            let src = b.add(p * ldb + j0);
            let out = base + p * NR;
            for j in 0..nr
            {
                dst[out + j] = *src.add(j);
            }
            dst[out + nr..out + NR].fill(0.0);
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn pack_a_avx512(
    alpha: f32,
    a: *const f32,
    lda: usize,
    mc: usize,
    kc: usize,
    dst: &mut [f32],
) {
    let panels = mc.div_ceil(MR);
    debug_assert!(dst.len() >= panels * kc * MR);
    for panel in 0..panels
    {
        let i0 = panel * MR;
        let mr = MR.min(mc - i0);
        let base = panel * kc * MR;
        for p in 0..kc
        {
            let out = base + p * MR;
            for i in 0..mr
            {
                dst[out + i] = alpha * *a.add((i0 + i) * lda + p);
            }
            dst[out + mr..out + MR].fill(0.0);
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
    let mask = if nr == NR { u16::MAX } else { (1_u16 << nr) - 1 };
    let mut acc = [_mm512_setzero_ps(); MR];
    for p in 0..kc
    {
        let bv = _mm512_loadu_ps(b_pack.add(p * NR));
        let av = a_pack.add(p * MR);
        for (i, lane) in acc.iter_mut().enumerate()
        {
            *lane = _mm512_fmadd_ps(_mm512_set1_ps(*av.add(i)), bv, *lane);
        }
    }
    for (i, lane) in acc.iter().enumerate().take(mr)
    {
        let row = c.add(i * ldc);
        let prior = _mm512_maskz_loadu_ps(mask, row);
        _mm512_mask_storeu_ps(row, mask, _mm512_add_ps(prior, *lane));
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
    scale_c_avx512(beta, m, n, c_ptr);
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
            pack_b_avx512(
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
                pack_a_avx512(
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
unsafe fn scale_c_neon(beta: f32, m: usize, n: usize, c: *mut f32) {
    use core::arch::aarch64::*;
    if beta == 1.0
    {
        return;
    }
    let bv = vdupq_n_f32(beta);
    for i in 0..m
    {
        let row = c.add(i * n);
        let mut j = 0;
        while j + 4 <= n
        {
            let value = if beta == 0.0 {
                vdupq_n_f32(0.0)
            } else {
                vmulq_f32(vld1q_f32(row.add(j)), bv)
            };
            vst1q_f32(row.add(j), value);
            j += 4;
        }
        while j < n
        {
            *row.add(j) = if beta == 0.0 { 0.0 } else { *row.add(j) * beta };
            j += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn pack_b_neon(b: *const f32, ldb: usize, kc: usize, nc: usize, dst: &mut [f32]) {
    let panels = nc.div_ceil(NR_N);
    debug_assert!(dst.len() >= panels * kc * NR_N);
    for panel in 0..panels
    {
        let j0 = panel * NR_N;
        let nr = NR_N.min(nc - j0);
        let base = panel * kc * NR_N;
        for p in 0..kc
        {
            let src = b.add(p * ldb + j0);
            let out = base + p * NR_N;
            for j in 0..nr
            {
                dst[out + j] = *src.add(j);
            }
            dst[out + nr..out + NR_N].fill(0.0);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn pack_a_neon(
    alpha: f32,
    a: *const f32,
    lda: usize,
    mc: usize,
    kc: usize,
    dst: &mut [f32],
) {
    let panels = mc.div_ceil(MR_N);
    debug_assert!(dst.len() >= panels * kc * MR_N);
    for panel in 0..panels
    {
        let i0 = panel * MR_N;
        let mr = MR_N.min(mc - i0);
        let base = panel * kc * MR_N;
        for p in 0..kc
        {
            let out = base + p * MR_N;
            for i in 0..mr
            {
                dst[out + i] = alpha * *a.add((i0 + i) * lda + p);
            }
            dst[out + mr..out + MR_N].fill(0.0);
        }
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
    let mut acc0 = [vdupq_n_f32(0.0); MR_N];
    let mut acc1 = [vdupq_n_f32(0.0); MR_N];
    for p in 0..kc
    {
        let b0 = vld1q_f32(b_pack.add(p * NR_N));
        let b1 = vld1q_f32(b_pack.add(p * NR_N + 4));
        let av = a_pack.add(p * MR_N);
        for i in 0..MR_N
        {
            let scalar = vdupq_n_f32(*av.add(i));
            acc0[i] = vfmaq_f32(acc0[i], scalar, b0);
            acc1[i] = vfmaq_f32(acc1[i], scalar, b1);
        }
    }
    let mut tmp = [0.0_f32; NR_N];
    for i in 0..mr
    {
        vst1q_f32(tmp.as_mut_ptr(), acc0[i]);
        vst1q_f32(tmp.as_mut_ptr().add(4), acc1[i]);
        let row = c.add(i * ldc);
        for (j, value) in tmp.iter().copied().enumerate().take(nr)
        {
            *row.add(j) += value;
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
    scale_c_neon(beta, m, n, c_ptr);
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
            pack_b_neon(
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
                pack_a_neon(
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
        let c0: Vec<f32> = (0..m * n)
            .map(|i| ((i % 17) as f32) * 0.03 - 0.2)
            .collect();

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
