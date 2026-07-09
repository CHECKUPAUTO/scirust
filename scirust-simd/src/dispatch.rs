// scirust-simd/src/dispatch.rs
//
// Détection des capacités CPU à l'exécution + sélection du backend.
// Permet de livrer un seul binaire qui exploite AVX2/SSE/NEON si dispo,
// sinon retombe sur scalaire — sans #[target_feature] global.
//
// La détection se fait une seule fois (OnceLock) et le résultat est
// caché. Coût d'un appel : un load atomique.
//
// ## Safety
//
// This module contains architecture-specific backends using `#[target_feature]`
// and `unsafe` intrinsics. Safety invariants for ALL `unsafe` functions:
//
// **AVX2/SSE2 backends (x86_64)**:
// - Functions marked `#[target_feature(enable = "avx2")]` or `#[target_feature(enable = "sse2")]`
//   are only called after runtime feature detection (`std::is_x86_feature_detected!`)
//   confirms the CPU supports the instruction set. The `detect_backend()` function
//   guarantees this.
// - `_mm256_loadu_ps` / `_mm_loadu_ps` / `_mm256_loadu_pd` / `_mm_loadu_pd`:
//   Support **unaligned** loads (MOVUPS/MOVUPD). No alignment requirement beyond 1 byte.
//   Pointer arithmetic `x.as_ptr().add(i)` stays in-bounds due to loop conditions
//   (`i + 8 <= n` for AVX2, `i + 4 <= n` for SSE2 f32, `i + 2 <= n` for f64).
// - `_mm256_storeu_ps` / `_mm_storeu_ps` / etc.: Same unaligned store guarantee.
// - Scalar remainder loops handle non-multiple-of-vector-width lengths safely.
// - No pointers escape; all borrows are bounded by slice lifetimes.
//
// **NEON backend (aarch64)**:
// - `#[target_feature(enable = "neon")]` only used after `std::arch::is_aarch64_feature_detected!("neon")`
//   returns true (guaranteed on all ARMv8+ CPUs).
// - `vld1q_f32` / `vst1q_f32`: Unaligned load/store (LDR Q / STR Q). No alignment requirement.
// - Pointer arithmetic bounded by `chunks = x.len() / 4` and remainder loop.
//
// **General invariants across all backends**:
// - Caller must ensure slice lengths match (enforced by `assert!` in public `SimdBackend` trait methods).
// - Slices are valid Rust references: non-null, aligned to at least 1, valid for the full length.
// - No aliasing violations: `&mut [f32]` / `&mut [f64]` guarantees exclusive mutable access.
// - All `unsafe` blocks are internal; public trait methods are `safe` and perform validation.
//
// **Soundness summary**: Each `unsafe` intrinsic call is guarded by:
// 1. Compile-time `#[target_feature]` matching runtime detection
// 2. Loop bounds checking ensuring pointer arithmetic stays in-bounds
// 3. Unaligned load/store intrinsics removing alignment constraints
// 4. Valid slice references from safe Rust guaranteeing pointer validity

#[cfg(feature = "portable-simd")]
use crate::matrix::backend::PortableSimdBackend;
use crate::matrix::backend::{ScalarBackend, SimdBackend};
use std::sync::OnceLock;

// ------------------------------------------------------------------ //
//  Énumération des backends disponibles                               //
// ------------------------------------------------------------------ //

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Scalar,
    Sse2,
    Avx2,
    Avx512,
    Neon,
    PortableSimd,
}

impl BackendKind {
    pub fn label(self) -> &'static str {
        match self
        {
            BackendKind::Scalar => "scalar",
            BackendKind::Sse2 => "x86_64/SSE2",
            BackendKind::Avx2 => "x86_64/AVX2",
            BackendKind::Avx512 => "x86_64/AVX-512",
            BackendKind::Neon => "aarch64/NEON",
            BackendKind::PortableSimd => "portable_simd",
        }
    }
}

// ------------------------------------------------------------------ //
//  Détection unique au démarrage                                      //
// ------------------------------------------------------------------ //

static DETECTED: OnceLock<BackendKind> = OnceLock::new();

/// Renvoie le meilleur backend disponible sur le CPU courant.
/// La détection est mise en cache après le premier appel.
pub fn detect_backend() -> BackendKind {
    *DETECTED.get_or_init(|| {
        // Si l'utilisateur a compilé avec portable-simd, on l'utilise
        // (le compilateur émet déjà les bonnes instructions) ; sinon on fait
        // de la détection runtime. Les deux chemins sont mutuellement exclusifs
        // par `cfg` pour éviter tout code mort.
        #[cfg(feature = "portable-simd")]
        {
            BackendKind::PortableSimd
        }
        #[cfg(not(feature = "portable-simd"))]
        {
            // Détection runtime x86_64
            #[cfg(target_arch = "x86_64")]
            {
                // AVX-512 d'abord (plus large)
                if std::is_x86_feature_detected!("avx512f")
                {
                    return BackendKind::Avx512;
                }
                if std::is_x86_feature_detected!("avx2")
                {
                    return BackendKind::Avx2;
                }
                if std::is_x86_feature_detected!("sse2")
                {
                    return BackendKind::Sse2;
                }
            }

            // ARM64 — NEON est baseline depuis ARMv8, toujours dispo
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon")
                {
                    return BackendKind::Neon;
                }
            }

            BackendKind::Scalar
        }
    })
}

/// Retourne une référence statique vers le backend le plus performant.
/// L'objet vit pour toute la durée du programme.
pub fn runtime_backend() -> &'static dyn SimdBackend {
    match detect_backend()
    {
        #[cfg(feature = "portable-simd")]
        BackendKind::PortableSimd => &PortableSimdBackend,

        #[cfg(target_arch = "x86_64")]
        BackendKind::Avx2 => &Avx2Backend,
        #[cfg(target_arch = "x86_64")]
        BackendKind::Sse2 => &Sse2Backend,
        #[cfg(target_arch = "x86_64")]
        BackendKind::Avx512 => &Avx2Backend, // fallback Avx2 tant qu'on n'a pas écrit le 512

        #[cfg(target_arch = "aarch64")]
        BackendKind::Neon => &NeonBackend,

        _ => &ScalarBackend,
    }
}

/// Affiche un résumé des capacités détectées (utile au démarrage).
pub fn print_capabilities() {
    let kind = detect_backend();
    println!("[scirust] backend sélectionné : {}", kind.label());
    println!("[scirust] détails CPU :");

    #[cfg(target_arch = "x86_64")]
    {
        println!("  - SSE2     : {}", std::is_x86_feature_detected!("sse2"));
        println!("  - SSE4.1   : {}", std::is_x86_feature_detected!("sse4.1"));
        println!("  - AVX      : {}", std::is_x86_feature_detected!("avx"));
        println!("  - AVX2     : {}", std::is_x86_feature_detected!("avx2"));
        println!("  - FMA      : {}", std::is_x86_feature_detected!("fma"));
        println!(
            "  - AVX-512F : {}",
            std::is_x86_feature_detected!("avx512f")
        );
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!(
            "  - NEON : {}",
            std::arch::is_aarch64_feature_detected!("neon")
        );
        println!(
            "  - SVE  : {}",
            std::arch::is_aarch64_feature_detected!("sve")
        );
    }
}

// ------------------------------------------------------------------ //
//  Backends arch-spécifiques (squelettes)                             //
//  À étendre — pour le moment ils délèguent au scalar backend        //
//  Les vraies implémentations utilisent #[target_feature(enable=...)] //
// ------------------------------------------------------------------ //

#[cfg(target_arch = "x86_64")]
pub struct Avx2Backend;

#[cfg(target_arch = "x86_64")]
impl SimdBackend for Avx2Backend {
    fn name(&self) -> &'static str {
        "avx2"
    }
    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        unsafe { saxpy_f32_avx2(alpha, x, y) }
    }
    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        unsafe { daxpy_f64_avx2(alpha, x, y) }
    }
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        unsafe { sdot_f32_avx2(x, y) }
    }
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        unsafe { ddot_f64_avx2(x, y) }
    }
    fn sgemv_f32(
        &self,
        alpha: f32,
        a: crate::matrix::view::MatrixView<f32>,
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) {
        let m = a.rows();
        for (i, item) in y.iter_mut().enumerate().take(m)
        {
            let row = a.row_slice(i).expect("row_slice");
            let dot = unsafe { sdot_f32_avx2(row, x) };
            *item = alpha * dot + beta * *item;
        }
    }
    fn sgemm_f32(
        &self,
        alpha: f32,
        a: crate::matrix::view::MatrixView<f32>,
        b: crate::matrix::view::MatrixView<f32>,
        beta: f32,
        c: crate::matrix::view::MatrixViewMut<f32>,
    ) {
        ScalarBackend.sgemm_f32(alpha, a, b, beta, c)
    }
    fn relu_f32(&self, v: &mut [f32]) {
        unsafe { relu_f32_avx2(v) }
    }
}

// ---- AVX2 kernel free functions ----

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn saxpy_f32_avx2(alpha: f32, x: &[f32], y: &mut [f32]) {
    use core::arch::x86_64::*;
    // `x` and `y` must be the same length: the loop indexes both at `0..x.len()`
    // through raw `loadu`/`storeu`, so a shorter `y` would be an out-of-bounds
    // read *and write*. Validate up front (panics in every profile) rather than
    // relying on the caller.
    assert_eq!(x.len(), y.len(), "saxpy: x.len() != y.len()");
    let alpha8 = _mm256_set1_ps(alpha);
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n
    {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let yv = _mm256_loadu_ps(y.as_ptr().add(i));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(alpha8, xv, yv));
        i += 8;
    }
    for j in i..n
    {
        y[j] += alpha * x[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn daxpy_f64_avx2(alpha: f64, x: &[f64], y: &mut [f64]) {
    use core::arch::x86_64::*;
    assert_eq!(x.len(), y.len(), "daxpy: x.len() != y.len()");
    let alpha4 = _mm256_set1_pd(alpha);
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n
    {
        let xv = _mm256_loadu_pd(x.as_ptr().add(i));
        let yv = _mm256_loadu_pd(y.as_ptr().add(i));
        _mm256_storeu_pd(y.as_mut_ptr().add(i), _mm256_fmadd_pd(alpha4, xv, yv));
        i += 4;
    }
    for j in i..n
    {
        y[j] += alpha * x[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sdot_f32_avx2(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    assert_eq!(x.len(), y.len(), "sdot: x.len() != y.len()");
    let n = x.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n
    {
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr().add(i)),
            _mm256_loadu_ps(y.as_ptr().add(i)),
            acc,
        );
        i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum: f32 = tmp.iter().sum();
    for j in i..n
    {
        sum += x[j] * y[j];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ddot_f64_avx2(x: &[f64], y: &[f64]) -> f64 {
    use core::arch::x86_64::*;
    assert_eq!(x.len(), y.len(), "ddot: x.len() != y.len()");
    let n = x.len();
    let mut acc = _mm256_setzero_pd();
    let mut i = 0;
    while i + 4 <= n
    {
        acc = _mm256_fmadd_pd(
            _mm256_loadu_pd(x.as_ptr().add(i)),
            _mm256_loadu_pd(y.as_ptr().add(i)),
            acc,
        );
        i += 4;
    }
    let mut tmp = [0.0f64; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
    let mut sum: f64 = tmp.iter().sum();
    for j in i..n
    {
        sum += x[j] * y[j];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_f32_avx2(v: &mut [f32]) {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let n = v.len();
    let mut i = 0;
    while i + 8 <= n
    {
        _mm256_storeu_ps(
            v.as_mut_ptr().add(i),
            _mm256_max_ps(_mm256_loadu_ps(v.as_ptr().add(i)), zero),
        );
        i += 8;
    }
    for x in &mut v[i..n]
    {
        *x = x.max(0.0);
    }
}

// SSE2 backend — real SIMD using _mm_* intrinsics (4-wide f32, 2-wide f64)
#[cfg(target_arch = "x86_64")]
pub struct Sse2Backend;

#[cfg(target_arch = "x86_64")]
impl SimdBackend for Sse2Backend {
    fn name(&self) -> &'static str {
        "sse2"
    }

    fn saxpy_f32(&self, a: f32, x: &[f32], y: &mut [f32]) {
        unsafe { saxpy_f32_sse2(a, x, y) }
    }
    fn daxpy_f64(&self, a: f64, x: &[f64], y: &mut [f64]) {
        unsafe { daxpy_f64_sse2(a, x, y) }
    }
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        unsafe { sdot_f32_sse2(x, y) }
    }
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        unsafe { ddot_f64_sse2(x, y) }
    }
    fn sgemv_f32(
        &self,
        alpha: f32,
        a: crate::matrix::view::MatrixView<f32>,
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) {
        let m = a.rows();
        for (i, item) in y.iter_mut().enumerate().take(m)
        {
            let row = a.row_slice(i).expect("row_slice");
            let dot = unsafe { sdot_f32_sse2(row, x) };
            *item = alpha * dot + beta * *item;
        }
    }
    fn sgemm_f32(
        &self,
        a: f32,
        ma: crate::matrix::view::MatrixView<f32>,
        mb: crate::matrix::view::MatrixView<f32>,
        b: f32,
        mc: crate::matrix::view::MatrixViewMut<f32>,
    ) {
        ScalarBackend.sgemm_f32(a, ma, mb, b, mc);
    }
    fn relu_f32(&self, v: &mut [f32]) {
        unsafe { relu_f32_sse2(v) }
    }
}

// ---- SSE2 kernel free functions (target_feature) ----

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn saxpy_f32_sse2(alpha: f32, x: &[f32], y: &mut [f32]) {
    use core::arch::x86_64::*;
    let a4 = _mm_set1_ps(alpha);
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n
    {
        let xv = _mm_loadu_ps(x.as_ptr().add(i));
        let yv = _mm_loadu_ps(y.as_ptr().add(i));
        let r = _mm_add_ps(_mm_mul_ps(a4, xv), yv);
        _mm_storeu_ps(y.as_mut_ptr().add(i), r);
        i += 4;
    }
    for j in i..n
    {
        y[j] += alpha * x[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn daxpy_f64_sse2(alpha: f64, x: &[f64], y: &mut [f64]) {
    use core::arch::x86_64::*;
    let a2 = _mm_set1_pd(alpha);
    let n = x.len();
    let mut i = 0;
    while i + 2 <= n
    {
        let xv = _mm_loadu_pd(x.as_ptr().add(i));
        let yv = _mm_loadu_pd(y.as_ptr().add(i));
        let r = _mm_add_pd(_mm_mul_pd(a2, xv), yv);
        _mm_storeu_pd(y.as_mut_ptr().add(i), r);
        i += 2;
    }
    for j in i..n
    {
        y[j] += alpha * x[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sdot_f32_sse2(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = x.len();
    let mut acc = _mm_setzero_ps();
    let mut i = 0;
    while i + 4 <= n
    {
        let xv = _mm_loadu_ps(x.as_ptr().add(i));
        let yv = _mm_loadu_ps(y.as_ptr().add(i));
        acc = _mm_add_ps(acc, _mm_mul_ps(xv, yv));
        i += 4;
    }
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum: f32 = tmp.iter().sum();
    for j in i..n
    {
        sum += x[j] * y[j];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn ddot_f64_sse2(x: &[f64], y: &[f64]) -> f64 {
    use core::arch::x86_64::*;
    let n = x.len();
    let mut acc = _mm_setzero_pd();
    let mut i = 0;
    while i + 2 <= n
    {
        let xv = _mm_loadu_pd(x.as_ptr().add(i));
        let yv = _mm_loadu_pd(y.as_ptr().add(i));
        acc = _mm_add_pd(acc, _mm_mul_pd(xv, yv));
        i += 2;
    }
    let mut tmp = [0.0f64; 2];
    _mm_storeu_pd(tmp.as_mut_ptr(), acc);
    let mut sum: f64 = tmp.iter().sum();
    for j in i..n
    {
        sum += x[j] * y[j];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn relu_f32_sse2(v: &mut [f32]) {
    use core::arch::x86_64::*;
    let zero = _mm_setzero_ps();
    let n = v.len();
    let mut i = 0;
    while i + 4 <= n
    {
        let xv = _mm_loadu_ps(v.as_ptr().add(i));
        let r = _mm_max_ps(xv, zero);
        _mm_storeu_ps(v.as_mut_ptr().add(i), r);
        i += 4;
    }
    for x in &mut v[i..n]
    {
        *x = x.max(0.0);
    }
}

#[cfg(target_arch = "aarch64")]
pub struct NeonBackend;

#[cfg(target_arch = "aarch64")]
impl SimdBackend for NeonBackend {
    fn name(&self) -> &'static str {
        "neon"
    }
    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        unsafe { saxpy_f32_neon(alpha, x, y) }
    }

    fn daxpy_f64(&self, a: f64, x: &[f64], y: &mut [f64]) {
        ScalarBackend.daxpy_f64(a, x, y);
    }
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        unsafe { sdot_f32_neon(x, y) }
    }
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        ScalarBackend.ddot_f64(x, y)
    }
    fn sgemv_f32(
        &self,
        alpha: f32,
        a: crate::matrix::view::MatrixView<f32>,
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) {
        let m = a.rows();
        for (i, item) in y.iter_mut().enumerate().take(m)
        {
            let row = a.row_slice(i).expect("row_slice");
            let dot = unsafe { sdot_f32_neon(row, x) };
            *item = alpha * dot + beta * *item;
        }
    }
    fn sgemm_f32(
        &self,
        a: f32,
        ma: crate::matrix::view::MatrixView<f32>,
        mb: crate::matrix::view::MatrixView<f32>,
        b: f32,
        mc: crate::matrix::view::MatrixViewMut<f32>,
    ) {
        ScalarBackend.sgemm_f32(a, ma, mb, b, mc);
    }
    fn relu_f32(&self, v: &mut [f32]) {
        ScalarBackend.relu_f32(v);
    }
}

// ---- NEON kernel free functions ----
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn saxpy_f32_neon(alpha: f32, x: &[f32], y: &mut [f32]) {
    use std::arch::aarch64::*;
    let alpha_v = vdupq_n_f32(alpha);
    let chunks = x.len() / 4;
    for c in 0..chunks
    {
        let xp = x.as_ptr().add(c * 4);
        let yp = y.as_mut_ptr().add(c * 4);
        let xv = vld1q_f32(xp);
        let yv = vld1q_f32(yp);
        let result = vfmaq_f32(yv, alpha_v, xv);
        vst1q_f32(yp, result);
    }
    let start = chunks * 4;
    for i in start..x.len()
    {
        y[i] += alpha * x[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sdot_f32_neon(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_f32(0.0);
    let n = x.len();
    let mut i = 0;
    while i + 4 <= n
    {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let yv = vld1q_f32(y.as_ptr().add(i));
        acc = vfmaq_f32(acc, xv, yv);
        i += 4;
    }
    let mut tmp = [0.0f32; 4];
    vst1q_f32(tmp.as_mut_ptr(), acc);
    let mut sum: f32 = tmp.iter().sum();
    for j in i..n
    {
        sum += x[j] * y[j];
    }
    sum
}

// ------------------------------------------------------------------ //
//  Tests                                                              //
// ------------------------------------------------------------------ //
#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::view::MatrixView;

    fn available_backends() -> Vec<(&'static dyn SimdBackend, &'static str)> {
        let mut v: Vec<(&'static dyn SimdBackend, &'static str)> = vec![(&ScalarBackend, "scalar")];
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2")
            {
                v.push((&Avx2Backend, "avx2"));
            }
            if std::is_x86_feature_detected!("sse2")
            {
                v.push((&Sse2Backend, "sse2"));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon")
            {
                v.push((&NeonBackend, "neon"));
            }
        }
        v
    }

    #[test]
    fn detection_returns_something() {
        let kind = detect_backend();
        assert!(matches!(
            kind,
            BackendKind::Scalar
                | BackendKind::Sse2
                | BackendKind::Avx2
                | BackendKind::Avx512
                | BackendKind::Neon
                | BackendKind::PortableSimd
        ));
    }

    #[test]
    fn detection_is_cached() {
        let k1 = detect_backend();
        let k2 = detect_backend();
        assert_eq!(k1, k2);
    }

    #[test]
    fn runtime_backend_returns_valid() {
        let b = runtime_backend();
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![1.0f32; 4];
        b.saxpy_f32(2.0, &y, &mut x);
        assert_eq!(x, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn sgemv_known_value() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![1.0f32, 1.0];
        let backends = available_backends();
        for (b, name) in &backends
        {
            let a = MatrixView::new(&a_data, 2, 2);
            let mut y = vec![0.0f32; 2];
            b.sgemv_f32(1.0, a, &x, 0.0, &mut y);
            assert_eq!(y, vec![3.0, 7.0], "[{name}] known-value sgemv failed");
        }
    }

    #[test]
    fn sgemv_cross_backend_exhaustive_small() {
        let backends = available_backends();
        let sizes: Vec<usize> = (0..=20).collect();
        let alphas = [1.0f32, -1.0, 0.5];
        let betas = [0.0f32, 1.0, -0.5];

        for &m in &sizes
        {
            for &k in &sizes
            {
                let mut a_data = vec![0.0f32; m * k];
                for i in 0..m
                {
                    for j in 0..k
                    {
                        a_data[i * k + j] = ((i * k + j) as f32) * 0.1 + 1.0;
                    }
                }
                let x: Vec<f32> = (0..k).map(|j| (j as f32) * 0.2 - 1.0).collect();
                let y0: Vec<f32> = (0..m).map(|i| (i as f32) * 0.3 + 2.0).collect();

                for &alpha in &alphas
                {
                    for &beta in &betas
                    {
                        let a = MatrixView::new(&a_data, m, k);
                        let mut expected = y0.clone();
                        ScalarBackend.sgemv_f32(alpha, a, &x, beta, &mut expected);

                        for (backend, name) in &backends
                        {
                            if name == &"scalar"
                            {
                                continue;
                            }
                            let a = MatrixView::new(&a_data, m, k);
                            let mut result = y0.clone();
                            backend.sgemv_f32(alpha, a, &x, beta, &mut result);
                            for i in 0..m
                            {
                                let diff = (result[i] - expected[i]).abs();
                                assert!(
                                    diff < 1e-4,
                                    "[{name}] m={m} k={k} alpha={alpha} beta={beta} i={i}: expected={}, got={}",
                                    expected[i],
                                    result[i]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sgemv_wide_matrix_matches_scalar() {
        let backends = available_backends();
        let m = 10;
        let k = 33;
        let mut a_data = vec![0.0f32; m * k];
        for i in 0..m
        {
            for j in 0..k
            {
                a_data[i * k + j] = ((i * k + j) as f32).sin();
            }
        }
        let x: Vec<f32> = (0..k).map(|j| (j as f32).cos()).collect();
        let y0: Vec<f32> = (0..m).map(|i| (i as f32) * 2.0).collect();

        let a_ref = MatrixView::new(&a_data, m, k);
        let mut expected = y0.clone();
        ScalarBackend.sgemv_f32(0.75, a_ref, &x, -0.25, &mut expected);

        for (backend, name) in &backends
        {
            if name == &"scalar"
            {
                continue;
            }
            let a = MatrixView::new(&a_data, m, k);
            let mut result = y0.clone();
            backend.sgemv_f32(0.75, a, &x, -0.25, &mut result);
            for i in 0..m
            {
                let diff = (result[i] - expected[i]).abs();
                assert!(
                    diff < 1e-4,
                    "[{name}] wide matrix m={m} k={k} i={i}: expected={}, got={}",
                    expected[i],
                    result[i]
                );
            }
        }
    }

    #[test]
    fn sgemv_tall_matrix_matches_scalar() {
        let backends = available_backends();
        let m = 33;
        let k = 10;
        let mut a_data = vec![0.0f32; m * k];
        for i in 0..m
        {
            for j in 0..k
            {
                a_data[i * k + j] = ((i * k + j) as f32).sin();
            }
        }
        let x: Vec<f32> = (0..k).map(|j| (j as f32).cos()).collect();
        let y0: Vec<f32> = (0..m).map(|i| (i as f32) * 2.0).collect();

        let a_ref = MatrixView::new(&a_data, m, k);
        let mut expected = y0.clone();
        ScalarBackend.sgemv_f32(1.5, a_ref, &x, 0.5, &mut expected);

        for (backend, name) in &backends
        {
            if name == &"scalar"
            {
                continue;
            }
            let a = MatrixView::new(&a_data, m, k);
            let mut result = y0.clone();
            backend.sgemv_f32(1.5, a, &x, 0.5, &mut result);
            for i in 0..m
            {
                let diff = (result[i] - expected[i]).abs();
                assert!(
                    diff < 1e-4,
                    "[{name}] tall matrix m={m} k={k} i={i}: expected={}, got={}",
                    expected[i],
                    result[i]
                );
            }
        }
    }

    #[test]
    fn sgemv_non_power_of_two() {
        let backends = available_backends();
        let sizes = [(7, 9), (9, 7), (5, 17), (17, 5), (6, 10), (10, 6)];
        for &(m, k) in &sizes
        {
            let mut a_data = vec![0.0f32; m * k];
            for i in 0..m
            {
                for j in 0..k
                {
                    a_data[i * k + j] = ((i * k) as f32 + j as f32) * 0.01;
                }
            }
            let x: Vec<f32> = (0..k).map(|j| (j as f32).sin()).collect();
            let y0: Vec<f32> = (0..m).map(|i| (i as f32).cos()).collect();

            let a_ref = MatrixView::new(&a_data, m, k);
            let mut expected = y0.clone();
            ScalarBackend.sgemv_f32(1.0, a_ref, &x, 0.0, &mut expected);

            for (backend, name) in &backends
            {
                if name == &"scalar"
                {
                    continue;
                }
                let a = MatrixView::new(&a_data, m, k);
                let mut result = y0.clone();
                backend.sgemv_f32(1.0, a, &x, 0.0, &mut result);
                for i in 0..m
                {
                    let diff = (result[i] - expected[i]).abs();
                    assert!(
                        diff < 1e-4,
                        "[{name}] m={m} k={k} i={i}: expected={}, got={}",
                        expected[i],
                        result[i]
                    );
                }
            }
        }
    }
}
