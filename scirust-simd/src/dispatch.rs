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
                // AVX-512 d'abord (plus large). Le noyau utilise FMA
                // (`_mm512_fmadd_ps`), garanti sur toute puce avx512f réelle.
                if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")
                {
                    return BackendKind::Avx512;
                }
                // Le backend AVX2 fait des `_mm256_fmadd_ps` : n'y router que si
                // FMA est bien présent (sinon instruction illégale sur les rares
                // puces AVX2-sans-FMA). Repli SSE2 le cas échéant.
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
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
        BackendKind::Avx512 => &Avx512Backend,

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
        unsafe { sgemm_f32_avx2(alpha, a, b, beta, c) }
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

/// In-place scale `c[j] *= beta`, vectorized (AVX2). Used by the tiled GEMM to
/// apply `beta * C` before the rank-1 updates.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scal_f32_avx2(beta: f32, c: &mut [f32]) {
    use core::arch::x86_64::*;
    let b8 = _mm256_set1_ps(beta);
    let n = c.len();
    let mut i = 0;
    while i + 8 <= n
    {
        let v = _mm256_loadu_ps(c.as_ptr().add(i));
        _mm256_storeu_ps(c.as_mut_ptr().add(i), _mm256_mul_ps(v, b8));
        i += 8;
    }
    for x in &mut c[i..n]
    {
        *x *= beta;
    }
}

/// AVX2 SGEMM: `C = alpha·A·B + beta·C`, row-major, via the axpy-over-rows
/// (rank-1 update) formulation. Each output row `C_i` is first scaled by `beta`,
/// then for every `p` a single scaled B-row is fused-multiply-added into it —
/// which streams `B` contiguously in row-major order (cache-friendly) and lets
/// the inner loop reuse the FMA `saxpy` kernel. Bit-close to the scalar
/// reference (differs only by summation order), cross-checked in tests.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sgemm_f32_avx2(
    alpha: f32,
    a: crate::matrix::view::MatrixView<f32>,
    b: crate::matrix::view::MatrixView<f32>,
    beta: f32,
    mut c: crate::matrix::view::MatrixViewMut<f32>,
) {
    let m = a.rows();
    let k = a.cols();
    for i in 0..m
    {
        let a_row = a.row_slice(i).expect("A row");
        let c_row = c.row_slice_mut(i).expect("C row");
        scal_f32_avx2(beta, c_row);
        for (p, &a_ip) in a_row.iter().enumerate().take(k)
        {
            let s = alpha * a_ip;
            if s == 0.0
            {
                continue;
            }
            let b_row = b.row_slice(p).expect("B row");
            saxpy_f32_avx2(s, b_row, c_row);
        }
    }
}

// =================================================================== //
//  AVX-512 backend — 16-wide f32 / 8-wide f64, FMA, masked remainders //
// =================================================================== //

/// Marker for the AVX-512 backend. Only constructed after
/// `is_x86_feature_detected!("avx512f")` succeeds (see `runtime_backend`).
#[cfg(target_arch = "x86_64")]
pub struct Avx512Backend;

#[cfg(target_arch = "x86_64")]
impl SimdBackend for Avx512Backend {
    fn name(&self) -> &'static str {
        "avx512"
    }
    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        unsafe { saxpy_f32_avx512(alpha, x, y) }
    }
    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        unsafe { daxpy_f64_avx512(alpha, x, y) }
    }
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        unsafe { sdot_f32_avx512(x, y) }
    }
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        unsafe { ddot_f64_avx512(x, y) }
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
            let dot = unsafe { sdot_f32_avx512(row, x) };
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
        unsafe { sgemm_f32_avx512(alpha, a, b, beta, c) }
    }
    fn relu_f32(&self, v: &mut [f32]) {
        unsafe { relu_f32_avx512(v) }
    }
}

// ---- AVX-512 kernel free functions ----
//
// The tail of every array is handled with an AVX-512 **write-mask** rather than
// a scalar loop: `(1u16 << r) - 1` selects the `r < 16` live f32 lanes (or
// `(1u8 << r) - 1` for the `r < 8` f64 lanes). `maskz` loads zero-fill the
// inactive lanes so they never contribute to an FMA accumulation, and masked
// stores leave the corresponding memory untouched. This keeps the whole kernel
// branch-light and avoids a separate scalar epilogue.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn saxpy_f32_avx512(alpha: f32, x: &[f32], y: &mut [f32]) {
    use core::arch::x86_64::*;
    assert_eq!(x.len(), y.len(), "saxpy: x.len() != y.len()");
    let a16 = _mm512_set1_ps(alpha);
    let n = x.len();
    let mut i = 0;
    while i + 16 <= n
    {
        let xv = _mm512_loadu_ps(x.as_ptr().add(i));
        let yv = _mm512_loadu_ps(y.as_ptr().add(i));
        _mm512_storeu_ps(y.as_mut_ptr().add(i), _mm512_fmadd_ps(a16, xv, yv));
        i += 16;
    }
    let r = n - i;
    if r > 0
    {
        let mask = (1u16 << r) - 1;
        let xv = _mm512_maskz_loadu_ps(mask, x.as_ptr().add(i));
        let yv = _mm512_maskz_loadu_ps(mask, y.as_ptr().add(i));
        let res = _mm512_fmadd_ps(a16, xv, yv);
        _mm512_mask_storeu_ps(y.as_mut_ptr().add(i), mask, res);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn daxpy_f64_avx512(alpha: f64, x: &[f64], y: &mut [f64]) {
    use core::arch::x86_64::*;
    assert_eq!(x.len(), y.len(), "daxpy: x.len() != y.len()");
    let a8 = _mm512_set1_pd(alpha);
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n
    {
        let xv = _mm512_loadu_pd(x.as_ptr().add(i));
        let yv = _mm512_loadu_pd(y.as_ptr().add(i));
        _mm512_storeu_pd(y.as_mut_ptr().add(i), _mm512_fmadd_pd(a8, xv, yv));
        i += 8;
    }
    let r = n - i;
    if r > 0
    {
        let mask = (1u8 << r) - 1;
        let xv = _mm512_maskz_loadu_pd(mask, x.as_ptr().add(i));
        let yv = _mm512_maskz_loadu_pd(mask, y.as_ptr().add(i));
        let res = _mm512_fmadd_pd(a8, xv, yv);
        _mm512_mask_storeu_pd(y.as_mut_ptr().add(i), mask, res);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sdot_f32_avx512(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    assert_eq!(x.len(), y.len(), "sdot: x.len() != y.len()");
    let n = x.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n
    {
        acc = _mm512_fmadd_ps(
            _mm512_loadu_ps(x.as_ptr().add(i)),
            _mm512_loadu_ps(y.as_ptr().add(i)),
            acc,
        );
        i += 16;
    }
    let r = n - i;
    if r > 0
    {
        let mask = (1u16 << r) - 1;
        acc = _mm512_fmadd_ps(
            _mm512_maskz_loadu_ps(mask, x.as_ptr().add(i)),
            _mm512_maskz_loadu_ps(mask, y.as_ptr().add(i)),
            acc,
        );
    }
    _mm512_reduce_add_ps(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn ddot_f64_avx512(x: &[f64], y: &[f64]) -> f64 {
    use core::arch::x86_64::*;
    assert_eq!(x.len(), y.len(), "ddot: x.len() != y.len()");
    let n = x.len();
    let mut acc = _mm512_setzero_pd();
    let mut i = 0;
    while i + 8 <= n
    {
        acc = _mm512_fmadd_pd(
            _mm512_loadu_pd(x.as_ptr().add(i)),
            _mm512_loadu_pd(y.as_ptr().add(i)),
            acc,
        );
        i += 8;
    }
    let r = n - i;
    if r > 0
    {
        let mask = (1u8 << r) - 1;
        acc = _mm512_fmadd_pd(
            _mm512_maskz_loadu_pd(mask, x.as_ptr().add(i)),
            _mm512_maskz_loadu_pd(mask, y.as_ptr().add(i)),
            acc,
        );
    }
    _mm512_reduce_add_pd(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn relu_f32_avx512(v: &mut [f32]) {
    use core::arch::x86_64::*;
    let zero = _mm512_setzero_ps();
    let n = v.len();
    let mut i = 0;
    while i + 16 <= n
    {
        _mm512_storeu_ps(
            v.as_mut_ptr().add(i),
            _mm512_max_ps(_mm512_loadu_ps(v.as_ptr().add(i)), zero),
        );
        i += 16;
    }
    let r = n - i;
    if r > 0
    {
        let mask = (1u16 << r) - 1;
        let vv = _mm512_maskz_loadu_ps(mask, v.as_ptr().add(i));
        _mm512_mask_storeu_ps(v.as_mut_ptr().add(i), mask, _mm512_max_ps(vv, zero));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scal_f32_avx512(beta: f32, c: &mut [f32]) {
    use core::arch::x86_64::*;
    let b16 = _mm512_set1_ps(beta);
    let n = c.len();
    let mut i = 0;
    while i + 16 <= n
    {
        let v = _mm512_loadu_ps(c.as_ptr().add(i));
        _mm512_storeu_ps(c.as_mut_ptr().add(i), _mm512_mul_ps(v, b16));
        i += 16;
    }
    let r = n - i;
    if r > 0
    {
        let mask = (1u16 << r) - 1;
        let v = _mm512_maskz_loadu_ps(mask, c.as_ptr().add(i));
        _mm512_mask_storeu_ps(c.as_mut_ptr().add(i), mask, _mm512_mul_ps(v, b16));
    }
}

/// AVX-512 SGEMM (`C = alpha·A·B + beta·C`, row-major), same rank-1-update
/// structure as [`sgemm_f32_avx2`] but 16 f32 lanes per FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_f32_avx512(
    alpha: f32,
    a: crate::matrix::view::MatrixView<f32>,
    b: crate::matrix::view::MatrixView<f32>,
    beta: f32,
    mut c: crate::matrix::view::MatrixViewMut<f32>,
) {
    let m = a.rows();
    let k = a.cols();
    for i in 0..m
    {
        let a_row = a.row_slice(i).expect("A row");
        let c_row = c.row_slice_mut(i).expect("C row");
        scal_f32_avx512(beta, c_row);
        for (p, &a_ip) in a_row.iter().enumerate().take(k)
        {
            let s = alpha * a_ip;
            if s == 0.0
            {
                continue;
            }
            let b_row = b.row_slice(p).expect("B row");
            saxpy_f32_avx512(s, b_row, c_row);
        }
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
        unsafe { sgemm_f32_sse2(a, ma, mb, b, mc) }
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn scal_f32_sse2(beta: f32, c: &mut [f32]) {
    use core::arch::x86_64::*;
    let b4 = _mm_set1_ps(beta);
    let n = c.len();
    let mut i = 0;
    while i + 4 <= n
    {
        let v = _mm_loadu_ps(c.as_ptr().add(i));
        _mm_storeu_ps(c.as_mut_ptr().add(i), _mm_mul_ps(v, b4));
        i += 4;
    }
    for x in &mut c[i..n]
    {
        *x *= beta;
    }
}

/// SSE2 SGEMM (`C = alpha·A·B + beta·C`, row-major), rank-1-update formulation
/// (4 f32 lanes). See [`sgemm_f32_avx2`] for the algorithm.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sgemm_f32_sse2(
    alpha: f32,
    a: crate::matrix::view::MatrixView<f32>,
    b: crate::matrix::view::MatrixView<f32>,
    beta: f32,
    mut c: crate::matrix::view::MatrixViewMut<f32>,
) {
    let m = a.rows();
    let k = a.cols();
    for i in 0..m
    {
        let a_row = a.row_slice(i).expect("A row");
        let c_row = c.row_slice_mut(i).expect("C row");
        scal_f32_sse2(beta, c_row);
        for (p, &a_ip) in a_row.iter().enumerate().take(k)
        {
            let s = alpha * a_ip;
            if s == 0.0
            {
                continue;
            }
            let b_row = b.row_slice(p).expect("B row");
            saxpy_f32_sse2(s, b_row, c_row);
        }
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
        unsafe { sgemm_f32_neon(a, ma, mb, b, mc) }
    }
    fn relu_f32(&self, v: &mut [f32]) {
        ScalarBackend.relu_f32(v);
    }
}

// ---- NEON kernel free functions ----

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scal_f32_neon(beta: f32, c: &mut [f32]) {
    use std::arch::aarch64::*;
    let bv = vdupq_n_f32(beta);
    let n = c.len();
    let mut i = 0;
    while i + 4 <= n
    {
        let v = vld1q_f32(c.as_ptr().add(i));
        vst1q_f32(c.as_mut_ptr().add(i), vmulq_f32(v, bv));
        i += 4;
    }
    for x in &mut c[i..n]
    {
        *x *= beta;
    }
}

/// NEON SGEMM (`C = alpha·A·B + beta·C`, row-major), même formulation
/// rank-1 / axpy-sur-lignes que le palier x86 — porte le gain multi-plateforme
/// sur Jetson / Raspberry Pi / RK3588 (remplaçait un fallback scalaire).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sgemm_f32_neon(
    alpha: f32,
    a: crate::matrix::view::MatrixView<f32>,
    b: crate::matrix::view::MatrixView<f32>,
    beta: f32,
    mut c: crate::matrix::view::MatrixViewMut<f32>,
) {
    let m = a.rows();
    let k = a.cols();
    for i in 0..m
    {
        let a_row = a.row_slice(i).expect("A row");
        let c_row = c.row_slice_mut(i).expect("C row");
        scal_f32_neon(beta, c_row);
        for (p, &a_ip) in a_row.iter().enumerate().take(k)
        {
            let s = alpha * a_ip;
            if s == 0.0
            {
                continue;
            }
            let b_row = b.row_slice(p).expect("B row");
            saxpy_f32_neon(s, b_row, c_row);
        }
    }
}
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
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma")
            {
                v.push((&Avx512Backend, "avx512"));
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
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

    /// Every non-scalar backend's SGEMM must match the scalar reference for a
    /// spread of shapes (including K/N not a multiple of any vector width) and
    /// non-trivial `alpha`/`beta`. This is what promotes the vectorized GEMM
    /// (and the newly-wired AVX-512 backend) from "compiles" to "correct".
    #[test]
    fn sgemm_cross_backend_matches_scalar() {
        use crate::matrix::view::MatrixViewMut;
        let backends = available_backends();
        let shapes = [
            (1usize, 1usize, 1usize),
            (2, 3, 2),
            (4, 4, 4),
            (7, 5, 9),
            (9, 17, 3),
            (16, 16, 16),
            (17, 31, 13),
            (33, 8, 20),
        ];
        let alphas = [1.0f32, -0.5, 2.0];
        let betas = [0.0f32, 1.0, -0.75];

        for &(m, k, n) in &shapes
        {
            let a_data: Vec<f32> = (0..m * k).map(|t| (t as f32 * 0.017 - 0.3).sin()).collect();
            let b_data: Vec<f32> = (0..k * n).map(|t| (t as f32 * 0.023 + 0.1).cos()).collect();
            let c0: Vec<f32> = (0..m * n).map(|t| (t as f32) * 0.05 - 0.5).collect();

            for &alpha in &alphas
            {
                for &beta in &betas
                {
                    let mut expected = c0.clone();
                    ScalarBackend.sgemm_f32(
                        alpha,
                        MatrixView::new(&a_data, m, k),
                        MatrixView::new(&b_data, k, n),
                        beta,
                        MatrixViewMut::new(&mut expected, m, n),
                    );

                    for (backend, name) in &backends
                    {
                        if name == &"scalar"
                        {
                            continue;
                        }
                        let mut result = c0.clone();
                        backend.sgemm_f32(
                            alpha,
                            MatrixView::new(&a_data, m, k),
                            MatrixView::new(&b_data, k, n),
                            beta,
                            MatrixViewMut::new(&mut result, m, n),
                        );
                        for t in 0..m * n
                        {
                            let diff = (result[t] - expected[t]).abs();
                            let tol = 1e-4 * (1.0 + expected[t].abs());
                            assert!(
                                diff <= tol,
                                "[{name}] sgemm m={m} k={k} n={n} alpha={alpha} beta={beta} t={t}: \
                                 expected={}, got={}",
                                expected[t],
                                result[t]
                            );
                        }
                    }
                }
            }
        }
    }

    /// AVX-512 axpy / dot / relu vs scalar, exercising the masked-remainder path
    /// at every length `0..=40` (so tails of 1..15 f32 lanes are all covered).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_kernels_match_scalar_all_lengths() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma"))
        {
            return; // no AVX-512 on this host — nothing to verify
        }
        for len in 0..=40usize
        {
            let x: Vec<f32> = (0..len).map(|t| (t as f32) * 0.3 - 2.0).collect();
            let y: Vec<f32> = (0..len).map(|t| (t as f32) * -0.11 + 1.0).collect();

            // saxpy
            let mut got = y.clone();
            Avx512Backend.saxpy_f32(1.5, &x, &mut got);
            let mut want = y.clone();
            ScalarBackend.saxpy_f32(1.5, &x, &mut want);
            for t in 0..len
            {
                assert!((got[t] - want[t]).abs() <= 1e-4, "saxpy len={len} t={t}");
            }

            // sdot
            let d = Avx512Backend.sdot_f32(&x, &y);
            let dref = ScalarBackend.sdot_f32(&x, &y);
            assert!(
                (d - dref).abs() <= 1e-3 * (1.0 + dref.abs()),
                "sdot len={len}: {d} vs {dref}"
            );

            // relu
            let mut r = x.clone();
            Avx512Backend.relu_f32(&mut r);
            for t in 0..len
            {
                assert_eq!(r[t], x[t].max(0.0), "relu len={len} t={t}");
            }
        }

        // f64 daxpy / ddot at f64-tail lengths (0..=20).
        for len in 0..=20usize
        {
            let x: Vec<f64> = (0..len).map(|t| (t as f64) * 0.7 - 3.0).collect();
            let y: Vec<f64> = (0..len).map(|t| (t as f64) * 0.2 + 0.5).collect();
            let mut got = y.clone();
            Avx512Backend.daxpy_f64(-0.25, &x, &mut got);
            let mut want = y.clone();
            ScalarBackend.daxpy_f64(-0.25, &x, &mut want);
            for t in 0..len
            {
                assert!((got[t] - want[t]).abs() <= 1e-9, "daxpy len={len} t={t}");
            }
            let d = Avx512Backend.ddot_f64(&x, &y);
            let dref = ScalarBackend.ddot_f64(&x, &y);
            assert!(
                (d - dref).abs() <= 1e-9 * (1.0 + dref.abs()),
                "ddot len={len}: {d} vs {dref}"
            );
        }
    }
}
