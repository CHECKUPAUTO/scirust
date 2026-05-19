// scirust-simd/src/dispatch.rs
//
// Détection des capacités CPU à l'exécution + sélection du backend.
// Permet de livrer un seul binaire qui exploite AVX2/SSE/NEON si dispo,
// sinon retombe sur scalaire — sans #[target_feature] global.
//
// La détection se fait une seule fois (OnceLock) et le résultat est
// caché. Coût d'un appel : un load atomique.

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
        match self {
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
        // (le compilateur émet déjà les bonnes instructions).
        #[cfg(feature = "portable-simd")]
        {
            return BackendKind::PortableSimd;
        }

        // Détection runtime x86_64
        #[cfg(target_arch = "x86_64")]
        {
            // AVX-512 d'abord (plus large)
            if std::is_x86_feature_detected!("avx512f") {
                return BackendKind::Avx512;
            }
            if std::is_x86_feature_detected!("avx2") {
                return BackendKind::Avx2;
            }
            if std::is_x86_feature_detected!("sse2") {
                return BackendKind::Sse2;
            }
        }

        // ARM64 — NEON est baseline depuis ARMv8, toujours dispo
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return BackendKind::Neon;
            }
        }

        BackendKind::Scalar
    })
}

/// Retourne une référence statique vers le backend le plus performant.
/// L'objet vit pour toute la durée du programme.
pub fn runtime_backend() -> &'static dyn SimdBackend {
    match detect_backend() {
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
    fn name(&self) -> &'static str { "avx2" }
    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) { unsafe { saxpy_f32_avx2(alpha, x, y) } }
    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) { unsafe { daxpy_f64_avx2(alpha, x, y) } }
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 { unsafe { sdot_f32_avx2(x, y) } }
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 { unsafe { ddot_f64_avx2(x, y) } }
    fn sgemv_f32(&self, alpha: f32, a: crate::matrix::view::MatrixView<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
        ScalarBackend.sgemv_f32(alpha, a, x, beta, y)
    }
    fn sgemm_f32(&self, alpha: f32, a: crate::matrix::view::MatrixView<f32>, b: crate::matrix::view::MatrixView<f32>, beta: f32, c: crate::matrix::view::MatrixViewMut<f32>) {
        ScalarBackend.sgemm_f32(alpha, a, b, beta, c)
    }
    fn relu_f32(&self, v: &mut [f32]) { unsafe { relu_f32_avx2(v) } }
}

// ---- AVX2 kernel free functions ----

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn saxpy_f32_avx2(alpha: f32, x: &[f32], y: &mut [f32]) {
    use core::arch::x86_64::*;
    let alpha8 = _mm256_set1_ps(alpha);
    let n = x.len(); let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let yv = _mm256_loadu_ps(y.as_ptr().add(i));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(alpha8, xv, yv));
        i += 8;
    }
    for j in i..n { y[j] += alpha * x[j]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn daxpy_f64_avx2(alpha: f64, x: &[f64], y: &mut [f64]) {
    use core::arch::x86_64::*;
    let alpha4 = _mm256_set1_pd(alpha);
    let n = x.len(); let mut i = 0;
    while i + 4 <= n {
        let xv = _mm256_loadu_pd(x.as_ptr().add(i));
        let yv = _mm256_loadu_pd(y.as_ptr().add(i));
        _mm256_storeu_pd(y.as_mut_ptr().add(i), _mm256_fmadd_pd(alpha4, xv, yv));
        i += 4;
    }
    for j in i..n { y[j] += alpha * x[j]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sdot_f32_avx2(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = x.len(); let mut acc = _mm256_setzero_ps(); let mut i = 0;
    while i + 8 <= n {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(x.as_ptr().add(i)), _mm256_loadu_ps(y.as_ptr().add(i)), acc);
        i += 8;
    }
    let mut tmp = [0.0f32; 8]; _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum: f32 = tmp.iter().sum();
    for j in i..n { sum += x[j] * y[j]; }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ddot_f64_avx2(x: &[f64], y: &[f64]) -> f64 {
    use core::arch::x86_64::*;
    let n = x.len(); let mut acc = _mm256_setzero_pd(); let mut i = 0;
    while i + 4 <= n {
        acc = _mm256_fmadd_pd(_mm256_loadu_pd(x.as_ptr().add(i)), _mm256_loadu_pd(y.as_ptr().add(i)), acc);
        i += 4;
    }
    let mut tmp = [0.0f64; 4]; _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
    let mut sum: f64 = tmp.iter().sum();
    for j in i..n { sum += x[j] * y[j]; }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_f32_avx2(v: &mut [f32]) {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_ps(); let n = v.len(); let mut i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(v.as_mut_ptr().add(i), _mm256_max_ps(_mm256_loadu_ps(v.as_ptr().add(i)), zero));
        i += 8;
    }
    for x in &mut v[i..n] { *x = x.max(0.0); }
}



// SSE2 backend — real SIMD using _mm_* intrinsics (4-wide f32, 2-wide f64)
#[cfg(target_arch = "x86_64")]
pub struct Sse2Backend;

#[cfg(target_arch = "x86_64")]
impl SimdBackend for Sse2Backend {
    fn name(&self) -> &'static str { "sse2" }

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
    fn sgemv_f32(&self, a: f32, m: crate::matrix::view::MatrixView<f32>, x: &[f32], b: f32, y: &mut [f32]) {
        ScalarBackend.sgemv_f32(a, m, x, b, y);
    }
    fn sgemm_f32(&self, a: f32, ma: crate::matrix::view::MatrixView<f32>, mb: crate::matrix::view::MatrixView<f32>, b: f32, mc: crate::matrix::view::MatrixViewMut<f32>) {
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
    while i + 4 <= n {
        let xv = _mm_loadu_ps(x.as_ptr().add(i));
        let yv = _mm_loadu_ps(y.as_ptr().add(i));
        let r = _mm_add_ps(_mm_mul_ps(a4, xv), yv);
        _mm_storeu_ps(y.as_mut_ptr().add(i), r);
        i += 4;
    }
    for j in i..n { y[j] += alpha * x[j]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn daxpy_f64_sse2(alpha: f64, x: &[f64], y: &mut [f64]) {
    use core::arch::x86_64::*;
    let a2 = _mm_set1_pd(alpha);
    let n = x.len();
    let mut i = 0;
    while i + 2 <= n {
        let xv = _mm_loadu_pd(x.as_ptr().add(i));
        let yv = _mm_loadu_pd(y.as_ptr().add(i));
        let r = _mm_add_pd(_mm_mul_pd(a2, xv), yv);
        _mm_storeu_pd(y.as_mut_ptr().add(i), r);
        i += 2;
    }
    for j in i..n { y[j] += alpha * x[j]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sdot_f32_sse2(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = x.len();
    let mut acc = _mm_setzero_ps();
    let mut i = 0;
    while i + 4 <= n {
        let xv = _mm_loadu_ps(x.as_ptr().add(i));
        let yv = _mm_loadu_ps(y.as_ptr().add(i));
        acc = _mm_add_ps(acc, _mm_mul_ps(xv, yv));
        i += 4;
    }
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum: f32 = tmp.iter().sum();
    for j in i..n { sum += x[j] * y[j]; }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn ddot_f64_sse2(x: &[f64], y: &[f64]) -> f64 {
    use core::arch::x86_64::*;
    let n = x.len();
    let mut acc = _mm_setzero_pd();
    let mut i = 0;
    while i + 2 <= n {
        let xv = _mm_loadu_pd(x.as_ptr().add(i));
        let yv = _mm_loadu_pd(y.as_ptr().add(i));
        acc = _mm_add_pd(acc, _mm_mul_pd(xv, yv));
        i += 2;
    }
    let mut tmp = [0.0f64; 2];
    _mm_storeu_pd(tmp.as_mut_ptr(), acc);
    let mut sum: f64 = tmp.iter().sum();
    for j in i..n { sum += x[j] * y[j]; }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn relu_f32_sse2(v: &mut [f32]) {
    use core::arch::x86_64::*;
    let zero = _mm_setzero_ps();
    let n = v.len();
    let mut i = 0;
    while i + 4 <= n {
        let xv = _mm_loadu_ps(v.as_ptr().add(i));
        let r = _mm_max_ps(xv, zero);
        _mm_storeu_ps(v.as_mut_ptr().add(i), r);
        i += 4;
    }
    for x in &mut v[i..n] { *x = x.max(0.0); }
}

#[cfg(target_arch = "aarch64")]
pub struct NeonBackend;

#[cfg(target_arch = "aarch64")]
impl SimdBackend for NeonBackend {
    fn name(&self) -> &'static str {
        "neon"
    }
    fn saxpy_f32(&self, a: f32, x: &[f32], y: &mut [f32]) {
        ScalarBackend.saxpy_f32(a, x, y);
    }

    fn daxpy_f64(&self, a: f64, x: &[f64], y: &mut [f64]) {
        ScalarBackend.daxpy_f64(a, x, y);
    }
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        ScalarBackend.sdot_f32(x, y)
    }
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        ScalarBackend.ddot_f64(x, y)
    }
    fn sgemv_f32(
        &self,
        a: f32,
        m: crate::matrix::view::MatrixView<f32>,
        x: &[f32],
        b: f32,
        y: &mut [f32],
    ) {
        ScalarBackend.sgemv_f32(a, m, x, b, y);
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

// ------------------------------------------------------------------ //
//  Tests                                                              //
// ------------------------------------------------------------------ //
#[cfg(test)]
mod tests {
    use super::*;

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
}
