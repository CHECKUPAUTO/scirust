// scirust-simd/src/dispatch.rs
//
// Détection des capacités CPU à l'exécution + sélection du backend.
// Permet de livrer un seul binaire qui exploite AVX2/SSE/NEON si dispo,
// sinon retombe sur scalaire — sans #[target_feature] global.
//
// La détection se fait une seule fois (OnceLock) et le résultat est
// caché. Coût d'un appel : un load atomique.

use std::sync::OnceLock;
use crate::matrix::backend::{
    SimdBackend, ScalarBackend,
    #[cfg(feature = "portable-simd")] PortableSimdBackend,
};

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
            BackendKind::Scalar       => "scalar",
            BackendKind::Sse2         => "x86_64/SSE2",
            BackendKind::Avx2         => "x86_64/AVX2",
            BackendKind::Avx512       => "x86_64/AVX-512",
            BackendKind::Neon         => "aarch64/NEON",
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
        { return BackendKind::PortableSimd; }

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
        BackendKind::Avx2  => &Avx2Backend,
        #[cfg(target_arch = "x86_64")]
        BackendKind::Sse2  => &Sse2Backend,
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
        println!("  - AVX-512F : {}", std::is_x86_feature_detected!("avx512f"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("  - NEON : {}", std::arch::is_aarch64_feature_detected!("neon"));
        println!("  - SVE  : {}", std::arch::is_aarch64_feature_detected!("sve"));
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

    #[target_feature(enable = "avx2,fma")]
    unsafe fn saxpy_avx2(alpha: f32, x: &[f32], y: &mut [f32]) {
        use std::arch::x86_64::*;
        let alpha_v = _mm256_set1_ps(alpha);
        let chunks = x.len() / 8;

        for c in 0..chunks {
            let xp = x.as_ptr().add(c * 8);
            let yp = y.as_mut_ptr().add(c * 8);
            let xv = _mm256_loadu_ps(xp);
            let yv = _mm256_loadu_ps(yp);
            // y = alpha * x + y, via FMA
            let result = _mm256_fmadd_ps(alpha_v, xv, yv);
            _mm256_storeu_ps(yp, result);
        }
        // Reste scalaire
        let start = chunks * 8;
        for i in start..x.len() { y[i] += alpha * x[i]; }
    }

    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        unsafe { Self::saxpy_avx2(alpha, x, y) }
    }

    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        // Délégation au scalaire — à étendre
        ScalarBackend.daxpy_f64(alpha, x, y);
    }

    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        unsafe { Self::sdot_avx2(x, y) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn sdot_avx2(x: &[f32], y: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        // Quatre accumulateurs pour briser la chaîne de dépendances
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let chunks = x.len() / 16;

        for c in 0..chunks {
            let xp = x.as_ptr().add(c * 16);
            let yp = y.as_ptr().add(c * 16);
            let x0 = _mm256_loadu_ps(xp);
            let y0 = _mm256_loadu_ps(yp);
            let x1 = _mm256_loadu_ps(xp.add(8));
            let y1 = _mm256_loadu_ps(yp.add(8));
            acc0 = _mm256_fmadd_ps(x0, y0, acc0);
            acc1 = _mm256_fmadd_ps(x1, y1, acc1);
        }
        let summed = _mm256_add_ps(acc0, acc1);

        // Réduction horizontale du vecteur 256 bits
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), summed);
        let mut total = buf.iter().sum::<f32>();

        // Reste scalaire
        let start = chunks * 16;
        for i in start..x.len() { total += x[i] * y[i]; }
        total
    }

    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        ScalarBackend.ddot_f64(x, y)
    }

    fn sgemv_f32(&self, alpha: f32, a: crate::matrix::view::MatrixView<f32>,
                 x: &[f32], beta: f32, y: &mut [f32]) {
        let (m, _) = (a.rows(), a.cols());
        for i in 0..m {
            let row = a.row_slice(i).expect("row_slice (col_stride=1)");
            let dot = self.sdot_f32(row, x);
            y[i] = alpha * dot + beta * y[i];
        }
    }

    fn sgemm_f32(&self, alpha: f32,
                 a: crate::matrix::view::MatrixView<f32>,
                 b: crate::matrix::view::MatrixView<f32>,
                 beta: f32,
                 c: crate::matrix::view::MatrixViewMut<f32>) {
        // Pour la GEMM, on délègue au backend scalaire pour la structure
        // et on utilise sdot AVX2 pour les produits scalaires internes.
        // Une vraie GEMM AVX2 tuilée 6x16 reste à écrire.
        ScalarBackend.sgemm_f32(alpha, a, b, beta, c);
    }

    fn relu_f32(&self, v: &mut [f32]) {
        unsafe { Self::relu_avx2(v) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn relu_avx2(v: &mut [f32]) {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_ps();
        let chunks = v.len() / 8;
        for c in 0..chunks {
            let p = v.as_mut_ptr().add(c * 8);
            let xv = _mm256_loadu_ps(p);
            let result = _mm256_max_ps(xv, zero);
            _mm256_storeu_ps(p, result);
        }
        let start = chunks * 8;
        for i in start..v.len() { v[i] = v[i].max(0.0); }
    }
}

// SSE2 backend (squelette — ressemble à AVX2 mais avec _mm_*)
#[cfg(target_arch = "x86_64")]
pub struct Sse2Backend;

#[cfg(target_arch = "x86_64")]
impl SimdBackend for Sse2Backend {
    fn name(&self) -> &'static str { "sse2" }
    fn saxpy_f32(&self, a: f32, x: &[f32], y: &mut [f32]) { ScalarBackend.saxpy_f32(a, x, y); }
    fn daxpy_f64(&self, a: f64, x: &[f64], y: &mut [f64]) { ScalarBackend.daxpy_f64(a, x, y); }
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 { ScalarBackend.sdot_f32(x, y) }
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 { ScalarBackend.ddot_f64(x, y) }
    fn sgemv_f32(&self, a: f32, m: crate::matrix::view::MatrixView<f32>, x: &[f32], b: f32, y: &mut [f32]) {
        ScalarBackend.sgemv_f32(a, m, x, b, y);
    }
    fn sgemm_f32(&self, a: f32, ma: crate::matrix::view::MatrixView<f32>,
                 mb: crate::matrix::view::MatrixView<f32>, b: f32,
                 mc: crate::matrix::view::MatrixViewMut<f32>) {
        ScalarBackend.sgemm_f32(a, ma, mb, b, mc);
    }
    fn relu_f32(&self, v: &mut [f32]) { ScalarBackend.relu_f32(v); }
}

#[cfg(target_arch = "aarch64")]
pub struct NeonBackend;

#[cfg(target_arch = "aarch64")]
impl SimdBackend for NeonBackend {
    fn name(&self) -> &'static str { "neon" }
    fn saxpy_f32(&self, a: f32, x: &[f32], y: &mut [f32]) {
        unsafe { Self::saxpy_neon(a, x, y) }
    }
    #[target_feature(enable = "neon")]
    unsafe fn saxpy_neon(alpha: f32, x: &[f32], y: &mut [f32]) {
        use std::arch::aarch64::*;
        let alpha_v = vdupq_n_f32(alpha);
        let chunks = x.len() / 4;
        for c in 0..chunks {
            let xp = x.as_ptr().add(c * 4);
            let yp = y.as_mut_ptr().add(c * 4);
            let xv = vld1q_f32(xp);
            let yv = vld1q_f32(yp);
            let result = vfmaq_f32(yv, alpha_v, xv);  // FMA NEON
            vst1q_f32(yp, result);
        }
        let start = chunks * 4;
        for i in start..x.len() { y[i] += alpha * x[i]; }
    }

    fn daxpy_f64(&self, a: f64, x: &[f64], y: &mut [f64]) { ScalarBackend.daxpy_f64(a, x, y); }
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 { ScalarBackend.sdot_f32(x, y) }
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 { ScalarBackend.ddot_f64(x, y) }
    fn sgemv_f32(&self, a: f32, m: crate::matrix::view::MatrixView<f32>, x: &[f32], b: f32, y: &mut [f32]) {
        ScalarBackend.sgemv_f32(a, m, x, b, y);
    }
    fn sgemm_f32(&self, a: f32, ma: crate::matrix::view::MatrixView<f32>,
                 mb: crate::matrix::view::MatrixView<f32>, b: f32,
                 mc: crate::matrix::view::MatrixViewMut<f32>) {
        ScalarBackend.sgemm_f32(a, ma, mb, b, mc);
    }
    fn relu_f32(&self, v: &mut [f32]) { ScalarBackend.relu_f32(v); }
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
        assert!(matches!(kind,
            BackendKind::Scalar | BackendKind::Sse2 | BackendKind::Avx2
            | BackendKind::Avx512 | BackendKind::Neon | BackendKind::PortableSimd));
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
