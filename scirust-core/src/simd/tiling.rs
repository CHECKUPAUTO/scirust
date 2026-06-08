//! # Cache-Aware Tiling — Pilier 4
//!
//! Ce module implémente le tiling automatique pour les opérations matricielles,
//! adapté à la taille du cache L2 de la machine cible.

use std::simd::f32x4;

/// Paramètres de tiling adaptatifs.
#[derive(Debug, Clone, Copy)]
pub struct TilingConfig {
    /// Taille de bloc pour l'axe M (lignes de A).
    pub tile_m: usize,
    /// Taille de bloc pour l'axe K (colonnes de A / lignes de B).
    pub tile_k: usize,
    /// Taille de bloc pour l'axe N (colonnes de B).
    pub tile_n: usize,
    /// SIMD largeur (lanes).
    pub simd_width: usize,
    /// Taille L2 détectée en bytes.
    pub l2_cache_size: usize,
    /// Cible de remplissage du cache L2 (fraction).
    pub l2_fill_fraction: f32,
}

impl TilingConfig {
    /// Crée les paramètres par défaut pour la plateforme courante.
    pub fn detect() -> Self {
        let l2 = Self::detect_l2_cache();
        let simd = Self::detect_simd_width();

        // Calculer les tiles pour remplir ~90% du cache L2
        let fill = 0.9_f32 * l2 as f32 / 12.0; 
        let tile_base = (fill as f64).sqrt() as usize;
        let tile_base = tile_base.clamp(16, 128);
        let tile_base = (tile_base / simd) * simd; // Aligner sur la largeur SIMD

        Self {
            tile_m: tile_base,
            tile_k: tile_base,
            tile_n: tile_base,
            simd_width: simd,
            l2_cache_size: l2,
            l2_fill_fraction: 0.9,
        }
    }

    /// Détecte la taille du cache L2.
    fn detect_l2_cache() -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(size_str) = std::fs::read_to_string(
                "/sys/devices/system/cpu/cpu0/cache/index2/size"
            ) {
                if let Ok(size) = size_str.trim().trim_end_matches('K').parse::<usize>() {
                    return size * 1024; // Convertir KB → bytes
                }
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            1_048_576 // 1 MB par défaut
        }
        #[cfg(target_arch = "aarch64")]
        {
            4_194_304 // 4 MB (Jetson Thor)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            524_288 // Fallback universel 512 KB
        }
    }

    /// Détecte la largeur SIMD.
    pub fn detect_simd_width() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return 16;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return 8;
            }
            4
        }

        #[cfg(target_arch = "aarch64")]
        {
            if Self::has_sve() {
                return 16;
            }
            4 // NEON
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            4
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn has_sve() -> bool {
        unsafe {
            let hwcap = libc::getauxval(libc::AT_HWCAP);
            (hwcap & (1 << 31)) != 0 // Bit SVE dans HWCAP
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn has_sve() -> bool {
        false
    }
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self::detect()
    }
}

/// Matmul tuilée avec le backend SIMD automatique et portable.
#[inline]
pub fn matmul_tiled_f32(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    config: Option<&TilingConfig>,
) {
    let det_config;
    let config = match config {
        Some(cfg) => cfg,
        None => {
            det_config = TilingConfig::detect();
            &det_config
        }
    };
    let (tile_m, tile_k, tile_n) = (config.tile_m, config.tile_k, config.tile_n);

    // Initialiser C par beta
    for i in 0..m * n {
        c[i] *= beta;
    }

    let mut ii = 0;
    while ii < m {
        let im = (ii + tile_m).min(m);
        let mut pp = 0;
        while pp < k {
            let pk = (pp + tile_k).min(k);
            let mut jj = 0;
            while jj < n {
                let jn = (jj + tile_n).min(n);

                for i in ii..im {
                    let a_row_off = i * k;
                    let c_row_off = i * n;
                    for p in pp..pk {
                        let alpha_a = alpha * a[a_row_off + p];
                        let a_val = f32x4::splat(alpha_a);
                        let b_col_off = p * n;

                        let mut j = jj;
                        while j + 4 <= jn {
                            let c_idx = c_row_off + j;
                            let b_idx = b_col_off + j;

                            let mut cv = f32x4::from_slice(&c[c_idx..c_idx + 4]);
                            let bv = f32x4::from_slice(&b[b_idx..b_idx + 4]);

                            cv += bv * a_val;

                            cv.copy_to_slice(&mut c[c_idx..c_idx + 4]);
                            j += 4;
                        }

                        while j < jn {
                            c[c_row_off + j] += alpha_a * b[b_col_off + j];
                            j += 1;
                        }
                    }
                }
                jj += tile_n;
            }
            pp += tile_k;
        }
        ii += tile_m;
    }
}

/// Matmul tuilée pour ARM64 NEON (Intrinsèques matériels natifs).
#[inline]
#[cfg(target_arch = "aarch64")]
pub fn matmul_neon_tiled_f32(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use std::arch::aarch64::*;

    let (tile_m, tile_k, tile_n) = (64, 64, 64);

    for i in 0..m * n {
        c[i] *= beta;
    }

    let mut ii = 0;
    while ii < m {
        let im = (ii + tile_m).min(m);
        let mut pp = 0;
        while pp < k {
            let pk = (pp + tile_k).min(k);
            let mut jj = 0;
            while jj < n {
                let jn = (jj + tile_n).min(n);

                for i in ii..im {
                    let a_row_off = i * k;
                    let c_row_off = i * n;
                    for p in pp..pk {
                        let alpha_a = alpha * a[a_row_off + p];
                        let b_col_off = p * n;

                        let mut j = jj;
                        unsafe {
                            let a_val_v = vdupq_n_f32(alpha_a);
                            while j + 4 <= jn {
                                let c_ptr = c.as_mut_ptr().add(c_row_off + j);
                                let b_ptr = b.as_ptr().add(b_col_off + j);

                                let vc = vld1q_f32(c_ptr);
                                let vb = vld1q_f32(b_ptr);
                                let vr = vmlaq_f32(vc, vb, a_val_v);

                                vst1q_f32(c_ptr, vr);
                                j += 4;
                            }
                        }

                        while j < jn {
                            c[c_row_off + j] += alpha_a * b[b_col_off + j];
                            j += 1;
                        }
                    }
                }
                jj += tile_n;
            }
            pp += tile_k;
        }
        ii += tile_m;
    }
}

/// Matmul tuilée pour x86_64 AVX2 + FMA.
#[inline]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn matmul_avx2_tiled_f32(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use std::arch::x86_64::*;

    let (tile_m, tile_k, tile_n) = (64, 64, 64);

    if !std::arch::is_x86_feature_detected!("avx2") {
        return matmul_tiled_f32(alpha, a, b, beta, c, m, k, n, None);
    }

    for i in 0..m * n {
        c[i] *= beta;
    }

    let mut ii = 0;
    while ii < m {
        let im = (ii + tile_m).min(m);
        let mut pp = 0;
        while pp < k {
            let pk = (pp + tile_k).min(k);
            let mut jj = 0;
            while jj < n {
                let jn = (jj + tile_n).min(n);

                for i in ii..im {
                    let a_row_off = i * k;
                    let c_row_off = i * n;
                    for p in pp..pk {
                        let alpha_a = alpha * a[a_row_off + p];
                        let b_col_off = p * n;

                        let mut j = jj;
                        unsafe {
                            let va = _mm256_set1_ps(alpha_a);
                            while j + 8 <= jn {
                                let b_ptr = b.as_ptr().add(b_col_off + j);
                                let c_ptr = c.as_mut_ptr().add(c_row_off + j);

                                let vb = _mm256_loadu_ps(b_ptr);
                                let vc = _mm256_loadu_ps(c_ptr);
                                let vr = _mm256_fmadd_ps(va, vb, vc);

                                _mm256_storeu_ps(c_ptr, vr);
                                j += 8;
                            }
                        }

                        while j < jn {
                            c[c_row_off + j] += alpha_a * b[b_col_off + j];
                            j += 1;
                        }
                    }
                }
                jj += tile_n;
            }
            pp += tile_k;
        }
        ii += tile_m;
    }
}

/// Configuration de profilage pour le cache L2/L3.
#[derive(Debug, Clone)]
pub struct CacheProfile {
    pub l1_size: usize,
    pub l2_size: usize,
    pub l3_size: usize,
    pub cache_line: usize,
    pub simd_width: usize,
}

impl CacheProfile {
    pub fn detect() -> Self {
        Self {
            l1_size: detect_l1_cache(),
            l2_size: detect_l2_cache_size(),
            l3_size: detect_l3_cache(),
            cache_line: 64,
            simd_width: TilingConfig::detect_simd_width(),
        }
    }

    pub fn optimal_tile(&self) -> TilingConfig {
        let ideal = (0.9_f32 * self.l2_size as f32 / 12.0).sqrt() as usize;
        let tile = ideal.clamp(16, 128);
        let tile = (tile / self.simd_width) * self.simd_width;

        TilingConfig {
            tile_m: tile,
            tile_k: tile,
            tile_n: tile,
            simd_width: self.simd_width,
            l2_cache_size: self.l2_size,
            l2_fill_fraction: 0.9,
        }
    }
}

fn detect_l1_cache() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size") {
            if let Ok(v) = s.trim().trim_end_matches('K').parse::<usize>() {
                return v * 1024;
            }
        }
    }
    32_768
}

fn detect_l2_cache_size() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size") {
            if let Ok(v) = s.trim().trim_end_matches('K').parse::<usize>() {
                return v * 1024;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        4_194_304
    }
    #[cfg(target_arch = "x86_64")]
    {
        262_144
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        262_144
    }
}

fn detect_l3_cache() -> usize {
    #[cfg(target_os = "linux")]
    {
        let paths = [
            "/sys/devices/system/cpu/cpu0/cache/index3/size",
            "/sys/devices/system/cpu/cache/index3/size",
        ];
        for path in &paths {
            if let Ok(s) = std::fs::read_to_string(path) {
                if let Ok(v) = s.trim().trim_end_matches('K').parse::<usize>() {
                    return v * 1024;
                }
            }
        }
    }
    0
}
