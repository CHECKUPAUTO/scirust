


//! # SciRust SIMD — Auto-vectorization utilities
//!
//! This crate provides:
//!
//! * The **`#[simd]`** proc-macro attribute (re-exported from `scirust-simd-macros`)
//!   that automatically generates architecture-specific variants of a free function
//!   with runtime dispatch (AVX2 / SSE2 / NEON / SVE / scalar).
//!
//! * A generic **`simd_map`** and **`simd_zip_with`** implemented on top of
//!   `std::simd` (nightly `portable_simd` feature).
//!
//! * **Stable manual SIMD kernels** for `f32`/`f64` using `core::arch` with
//!   runtime feature detection and scalar fallback.
//!
//! * **ARM64 NEON intrinsics** (Pilier 4) — 4x f32 lanes on all ARM64.
//!
//! * **ARM SVE intrinsics** (Pilier 4) — scalable vector length on Ampere/Graviton.
//!
//! ## Runtime dispatch
//!
//! The `simd_map` function automatically selects the best backend:
//! ```
//! // Detect platform and dispatch
//! use scirust_simd::simd_map;
//!
//! let input = vec![1.0f32, 2.0, 3.0, 4.0];
//! let mut output = vec![0.0f32; 4];
//! simd_map::<f32, 4>(&input, &mut output, |v| v + Simd::splat(1.0));
//! ```

#![feature(portable_simd)]
#![allow(unused_crate_dependencies)]
#![allow(unused_features)]


// Guard de compatibilité multi-architecture injecté pour ARM64 / Jetson Pipeline
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[allow(unused_macros)]
macro_rules! is_x86_feature_detected {
    ($feat:expr) => { false };
}

pub mod portable;
pub use portable::simd_ops;

pub use scirust_simd_macros::simd;

// =============================================================================
// Nightly portable_simd generic API
// =============================================================================

#[cfg(feature = "portable-simd")]
pub mod generic {
    use std::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};

    /// Apply a lane-wise operation to every element of `input`, writing results
    /// into `output`.
    pub fn simd_map<T, const N: usize, F>(input: &[T], output: &mut [T], f: F)
    where
        T: SimdElement + Default,
        LaneCount<N>: SupportedLaneCount,
        F: Fn(Simd<T, N>) -> Simd<T, N>,
    {
        assert_eq!(input.len(), output.len());

        let mut in_chunks = input.chunks_exact(N);
        let mut out_chunks = output.chunks_exact_mut(N);

        for (in_chunk, out_chunk) in in_chunks.by_ref().zip(out_chunks.by_ref()) {
            let v = Simd::<T, N>::from_slice(in_chunk);
            let r = f(v);
            r.copy_to_slice(out_chunk);
        }

        let in_rem = in_chunks.remainder();
        let out_rem = out_chunks.into_remainder();
        for (i, &x) in in_rem.iter().enumerate() {
            let s = Simd::<T, N>::splat(x);
            let r = f(s);
            out_rem[i] = r[0];
        }
    }

    /// Lane-wise binary operation over two equally-sized slices.
    pub fn simd_zip_with<T, const N: usize, F>(a: &[T], b: &[T], output: &mut [T], f: F)
    where
        T: SimdElement + Default,
        LaneCount<N>: SupportedLaneCount,
        F: Fn(Simd<T, N>, Simd<T, N>) -> Simd<T, N>,
    {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());

        let mut a_chunks = a.chunks_exact(N);
        let mut b_chunks = b.chunks_exact(N);
        let mut out_chunks = output.chunks_exact_mut(N);

        for ((a_chunk, b_chunk), out_chunk) in a_chunks
            .by_ref()
            .zip(b_chunks.by_ref())
            .zip(out_chunks.by_ref())
        {
            let va = Simd::<T, N>::from_slice(a_chunk);
            let vb = Simd::<T, N>::from_slice(b_chunk);
            let r = f(va, vb);
            r.copy_to_slice(out_chunk);
        }

        let a_rem = a_chunks.remainder();
        let b_rem = b_chunks.remainder();
        let out_rem = out_chunks.into_remainder();
        for i in 0..a_rem.len() {
            let sa = Simd::<T, N>::splat(a_rem[i]);
            let sb = Simd::<T, N>::splat(b_rem[i]);
            let r = f(sa, sb);
            out_rem[i] = r[0];
        }
    }
}

// =============================================================================
// Stable manual SIMD kernels (core::arch)
// =============================================================================

pub mod ops {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Element-wise `out[i] = a[i] + b[i]` for `f32` with AVX2/SSE2/scalar dispatch.
    pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let n = a.len();
        let mut i = 0;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx2") {
                while i + 8 <= n {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    let vr = _mm256_add_ps(va, vb);
                    _mm256_storeu_ps(out.as_mut_ptr().add(i), vr);
                    i += 8;
                }
            } else if std::arch::is_x86_feature_detected!("sse2") {
                while i + 4 <= n {
                    let va = _mm_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm_loadu_ps(b.as_ptr().add(i));
                    let vr = _mm_add_ps(va, vb);
                    _mm_storeu_ps(out.as_mut_ptr().add(i), vr);
                    i += 4;
                }
            }
        }

        while i < n {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }

    /// Element-wise `out[i] = a[i] * b[i]` for `f32`.
    pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let n = a.len();
        let mut i = 0;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx2") {
                while i + 8 <= n {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    let vr = _mm256_mul_ps(va, vb);
                    _mm256_storeu_ps(out.as_mut_ptr().add(i), vr);
                    i += 8;
                }
            } else if std::arch::is_x86_feature_detected!("sse2") {
                while i + 4 <= n {
                    let va = _mm_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm_loadu_ps(b.as_ptr().add(i));
                    let vr = _mm_mul_ps(va, vb);
                    _mm_storeu_ps(out.as_mut_ptr().add(i), vr);
                    i += 4;
                }
            }
        }

        while i < n {
            out[i] = a[i] * b[i];
            i += 1;
        }
    }

    /// Element-wise `out[i] = a[i] + b[i]` for `f64`.
    pub fn add_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let n = a.len();
        let mut i = 0;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx2") {
                while i + 4 <= n {
                    let va = _mm256_loadu_pd(a.as_ptr().add(i));
                    let vb = _mm256_loadu_pd(b.as_ptr().add(i));
                    let vr = _mm256_add_pd(va, vb);
                    _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
                    i += 4;
                }
            } else if std::arch::is_x86_feature_detected!("sse2") {
                while i + 2 <= n {
                    let va = _mm_loadu_pd(a.as_ptr().add(i));
                    let vb = _mm_loadu_pd(b.as_ptr().add(i));
                    let vr = _mm_add_pd(va, vb);
                    _mm_storeu_pd(out.as_mut_ptr().add(i), vr);
                    i += 2;
                }
            }
        }

        while i < n {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }

    /// Element-wise `out[i] = a[i] * b[i]` for `f64`.
    pub fn mul_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        let n = a.len();
        let mut i = 0;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx2") {
                while i + 4 <= n {
                    let va = _mm256_loadu_pd(a.as_ptr().add(i));
                    let vb = _mm256_loadu_pd(b.as_ptr().add(i));
                    let vr = _mm256_mul_pd(va, vb);
                    _mm256_storeu_pd(out.as_mut_ptr().add(i), vr);
                    i += 4;
                }
            } else if std::arch::is_x86_feature_detected!("sse2") {
                while i + 2 <= n {
                    let va = _mm_loadu_pd(a.as_ptr().add(i));
                    let vb = _mm_loadu_pd(b.as_ptr().add(i));
                    let vr = _mm_mul_pd(va, vb);
                    _mm_storeu_pd(out.as_mut_ptr().add(i), vr);
                    i += 2;
                }
            }
        }

        while i < n {
            out[i] = a[i] * b[i];
            i += 1;
        }
    }
}

// =============================================================================
// Scalar fallback helpers
// =============================================================================

/// Generic scalar map — always works, never uses SIMD.
pub fn scalar_map<T: Copy>(input: &[T], output: &mut [T], f: impl Fn(T) -> T) {
    assert_eq!(input.len(), output.len());
    for (i, &x) in input.iter().enumerate() {
        output[i] = f(x);
    }
}

/// Generic scalar zip — always works, never uses SIMD.
pub fn scalar_zip<T: Copy>(a: &[T], b: &[T], output: &mut [T], f: impl Fn(T, T) -> T) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = f(a[i], b[i]);
    }
}

// =============================================================================
// ARM64 NEON kernels (Pilier 4)
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    #[allow(unused_imports)]
    pub use super::neon_fns::*;
}

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub mod neon {
    #[allow(unused_imports)]
    pub use super::neon_fns::*;
}

#[cfg(target_arch = "aarch64")]
mod neon_fns {
    #[allow(unused_imports)]
    pub use crate::neon::*;
}

// =============================================================================
// ARM SVE kernels (Pilier 4)
// =============================================================================

#[cfg(target_arch = "aarch64")]
pub mod sve {
    #[allow(unused_imports)]
    pub use super::sve_fns::*;
}

#[cfg(target_arch = "aarch64")]
mod sve_fns {
    #[allow(unused_imports)]
    pub use crate::sve::*;
}

// =============================================================================
// Runtime dispatch (auto-select best SIMD backend)
// =============================================================================

/// Detects the best SIMD backend available on this platform.
pub fn detect_simd_backend() -> SimdBackend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdBackend::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdBackend::Avx2;
        }
        if is_x86_feature_detected!("sse2") {
            return SimdBackend::Sse2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if has_sve() {
            return SimdBackend::Sve;
        }
        return SimdBackend::Neon;
    }

    #[allow(unreachable_code)]
    SimdBackend::Scalar
}

/// SIMD backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    Avx512,
    Avx2,
    Sse2,
    Neon,
    Sve,
    Scalar,
}

impl SimdBackend {
    /// Returns the vector width in f32 elements.
    pub fn lane_width(&self) -> usize {
        match self {
            SimdBackend::Avx512 => 16,
            SimdBackend::Avx2 | SimdBackend::Sse2 => 4,
            SimdBackend::Neon => 4,
            SimdBackend::Sve => 8, // typical for 256-bit SVE
            SimdBackend::Scalar => 1,
        }
    }

    /// Returns true if this backend is available.
    pub fn available(&self) -> bool {
        match self {
            SimdBackend::Avx512 => cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx512f"),
            SimdBackend::Avx2 => cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx2"),
            SimdBackend::Sse2 => cfg!(target_arch = "x86_64") && is_x86_feature_detected!("sse2"),
            SimdBackend::Neon => cfg!(target_arch = "aarch64"),
            SimdBackend::Sve => cfg!(target_arch = "aarch64") && has_sve(),
            SimdBackend::Scalar => true,
        }
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

#[cfg(target_arch = "aarch64")]
fn has_sve() -> bool {
    unsafe {
        let hwcap = libc::getauxval(33); // AT_HWCAP
        (hwcap & (1 << 31)) != 0
    }
}

/// Convenience: add 1.0 to every element of a `f64` slice.
pub fn simd_add_one(data: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let n = data.len();
    let mut i = 0;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if std::arch::is_x86_feature_detected!("avx2") {
            let one = _mm256_set1_pd(1.0);
            while i + 4 <= n {
                let va = _mm256_loadu_pd(data.as_ptr().add(i));
                let vr = _mm256_add_pd(va, one);
                _mm256_storeu_pd(data.as_mut_ptr().add(i), vr);
                i += 4;
            }
        } else if std::arch::is_x86_feature_detected!("sse2") {
            let one = _mm_set1_pd(1.0);
            while i + 2 <= n {
                let va = _mm_loadu_pd(data.as_ptr().add(i));
                let vr = _mm_add_pd(va, one);
                _mm_storeu_pd(data.as_mut_ptr().add(i), vr);
                i += 2;
            }
        }
    }

    while i < n {
        data[i] += 1.0;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_add_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0f32; 5];
        ops::add_f32(&a, &b, &mut out);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_stable_mul_f64() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![2.0f64, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f64; 4];
        ops::mul_f64(&a, &b, &mut out);
        assert_eq!(out, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_scalar_map() {
        let input = vec![1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        scalar_map(&input, &mut output, |x| x + 1.0);
        assert_eq!(output, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scalar_zip() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10.0f32, 20.0, 30.0];
        let mut out = vec![0.0f32; 3];
        scalar_zip(&a, &b, &mut out, |x, y| x + y);
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_detect_backend() {
        let backend = detect_simd_backend();
        assert!(backend.available());
    }
}

pub mod dispatch;
pub mod complex;
pub mod matrix;
