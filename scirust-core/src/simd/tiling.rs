//! # Cache-Aware Tiling — Pilier 4
//!
//! Ce module implémente le tiling automatique pour les opérations matricielles,
//! adapté à la taille du cache L2 de la machine cible.

#[cfg(feature = "portable-simd")]
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
            if let Ok(size_str) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size")
                && let Ok(size) = size_str.trim().trim_end_matches('K').parse::<usize>()
            {
                return size * 1024; // Convertir KB → bytes
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
            if std::arch::is_x86_feature_detected!("avx512f")
            {
                return 16;
            }
            if std::arch::is_x86_feature_detected!("avx2")
            {
                return 8;
            }
            4
        }

        #[cfg(target_arch = "aarch64")]
        {
            if Self::has_sve()
            {
                return 16;
            }
            4 // NEON
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            4
        }
    }

    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "linux", target_os = "android")
    ))]
    #[allow(dead_code)]
    fn has_sve() -> bool {
        // Linux and Android expose AArch64 CPU capabilities through auxv.
        // SVE is bit 22 of AT_HWCAP; other operating systems use different
        // discovery APIs and must fall back to the portable NEON width here.
        const AT_HWCAP: libc::c_ulong = 16;
        const HWCAP_SVE: libc::c_ulong = 1 << 22;
        let hwcap = unsafe { libc::getauxval(AT_HWCAP) };
        (hwcap & HWCAP_SVE) != 0
    }

    #[cfg(not(all(
        target_arch = "aarch64",
        any(target_os = "linux", target_os = "android")
    )))]
    #[allow(dead_code)]
    fn has_sve() -> bool {
        false
    }
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self::detect()
    }
}

/// Validate that the operand slices are large enough for a contiguous
/// (row-major) `m×k · k×n → m×n` product. The tiled kernels index `a`, `b` and
/// `c` through raw/SIMD pointers with no per-access bounds check, so a caller
/// passing a slice shorter than the `(m,k,n)` contract would otherwise read or
/// write out of bounds (UB) rather than panic. `checked_mul` also rejects a
/// dimension product that would overflow `usize`.
#[inline]
fn assert_contiguous_dims(a_len: usize, b_len: usize, c_len: usize, m: usize, k: usize, n: usize) {
    let mk = m.checked_mul(k).expect("matmul tiled: m*k overflows usize");
    let kn = k.checked_mul(n).expect("matmul tiled: k*n overflows usize");
    let mn = m.checked_mul(n).expect("matmul tiled: m*n overflows usize");
    assert!(a_len >= mk, "matmul tiled: a.len() {a_len} < m*k {mk}");
    assert!(b_len >= kn, "matmul tiled: b.len() {b_len} < k*n {kn}");
    assert!(c_len >= mn, "matmul tiled: c.len() {c_len} < m*n {mn}");
}

/// Matmul tuilée avec le backend SIMD automatique et portable.
#[inline]
#[allow(clippy::too_many_arguments)]
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
    assert_contiguous_dims(a.len(), b.len(), c.len(), m, k, n);
    matmul_tiled_strided_f32(alpha, a, b, beta, c, m, k, n, k, 1, n, 1, n, 1, config)
}

/// Matmul tuilée avec support complet des strides pour A, B et C.
/// Permet d'éviter les copies lors des transpositions.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn matmul_tiled_strided_f32(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    rs_a: usize,
    cs_a: usize,
    rs_b: usize,
    cs_b: usize,
    rs_c: usize,
    cs_c: usize,
    config: Option<&TilingConfig>,
) {
    let det_config;
    let config = match config
    {
        Some(cfg) => cfg,
        None =>
        {
            det_config = TilingConfig::detect();
            &det_config
        },
    };
    let (tile_m, tile_k, tile_n) = (config.tile_m, config.tile_k, config.tile_n);

    // Initialiser C par beta en respectant les strides de C, afin de ne
    // toucher que les cellules logiques (i, j) et jamais le padding voisin.
    for i in 0..m
    {
        for j in 0..n
        {
            c[i * rs_c + j * cs_c] *= beta;
        }
    }

    let mut ii = 0;
    while ii < m
    {
        let im = (ii + tile_m).min(m);
        let mut pp = 0;
        while pp < k
        {
            let pk = (pp + tile_k).min(k);
            let mut jj = 0;
            while jj < n
            {
                let jn = (jj + tile_n).min(n);

                for i in ii..im
                {
                    for p in pp..pk
                    {
                        let alpha_a = alpha * a[i * rs_a + p * cs_a];

                        let mut j = jj;

                        // Vectorisation possible uniquement si les colonnes de B et C sont contiguës
                        #[cfg(feature = "portable-simd")]
                        if cs_b == 1 && cs_c == 1
                        {
                            let a_val = f32x4::splat(alpha_a);
                            while j + 4 <= jn
                            {
                                let c_idx = i * rs_c + j;
                                let b_idx = p * rs_b + j;

                                let mut cv = f32x4::from_slice(&c[c_idx..c_idx + 4]);
                                let bv = f32x4::from_slice(&b[b_idx..b_idx + 4]);

                                cv += bv * a_val;

                                cv.copy_to_slice(&mut c[c_idx..c_idx + 4]);
                                j += 4;
                            }
                        }

                        while j < jn
                        {
                            c[i * rs_c + j * cs_c] += alpha_a * b[p * rs_b + j * cs_b];
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
#[allow(clippy::too_many_arguments)]
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

    assert_contiguous_dims(a.len(), b.len(), c.len(), m, k, n);

    let (tile_m, tile_k, tile_n) = (64, 64, 64);

    #[allow(clippy::needless_range_loop)]
    for i in 0..m * n
    {
        c[i] *= beta;
    }

    let mut ii = 0;
    while ii < m
    {
        let im = (ii + tile_m).min(m);
        let mut pp = 0;
        while pp < k
        {
            let pk = (pp + tile_k).min(k);
            let mut jj = 0;
            while jj < n
            {
                let jn = (jj + tile_n).min(n);

                for i in ii..im
                {
                    let a_row_off = i * k;
                    let c_row_off = i * n;
                    for p in pp..pk
                    {
                        let alpha_a = alpha * a[a_row_off + p];
                        let b_col_off = p * n;

                        let mut j = jj;
                        unsafe {
                            let a_val_v = vdupq_n_f32(alpha_a);
                            while j + 4 <= jn
                            {
                                let c_ptr = c.as_mut_ptr().add(c_row_off + j);
                                let b_ptr = b.as_ptr().add(b_col_off + j);

                                let vc = vld1q_f32(c_ptr);
                                let vb = vld1q_f32(b_ptr);
                                let vr = vmlaq_f32(vc, vb, a_val_v);

                                vst1q_f32(c_ptr, vr);
                                j += 4;
                            }
                        }

                        while j < jn
                        {
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
#[allow(clippy::too_many_arguments)]
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

    assert_contiguous_dims(a.len(), b.len(), c.len(), m, k, n);

    let (tile_m, tile_k, tile_n) = (64, 64, 64);

    if !std::arch::is_x86_feature_detected!("avx2")
    {
        return matmul_tiled_f32(alpha, a, b, beta, c, m, k, n, None);
    }

    #[allow(clippy::needless_range_loop)]
    for i in 0..m * n
    {
        c[i] *= beta;
    }

    let mut ii = 0;
    while ii < m
    {
        let im = (ii + tile_m).min(m);
        let mut pp = 0;
        while pp < k
        {
            let pk = (pp + tile_k).min(k);
            let mut jj = 0;
            while jj < n
            {
                let jn = (jj + tile_n).min(n);

                for i in ii..im
                {
                    let a_row_off = i * k;
                    let c_row_off = i * n;
                    for p in pp..pk
                    {
                        let alpha_a = alpha * a[a_row_off + p];
                        let b_col_off = p * n;

                        let mut j = jj;
                        unsafe {
                            let va = _mm256_set1_ps(alpha_a);
                            while j + 8 <= jn
                            {
                                let b_ptr = b.as_ptr().add(b_col_off + j);
                                let c_ptr = c.as_mut_ptr().add(c_row_off + j);

                                let vb = _mm256_loadu_ps(b_ptr);
                                let vc = _mm256_loadu_ps(c_ptr);
                                let vr = _mm256_fmadd_ps(va, vb, vc);

                                _mm256_storeu_ps(c_ptr, vr);
                                j += 8;
                            }
                        }

                        while j < jn
                        {
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
        if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size")
            && let Ok(v) = s.trim().trim_end_matches('K').parse::<usize>()
        {
            return v * 1024;
        }
    }
    32_768
}

fn detect_l2_cache_size() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size")
            && let Ok(v) = s.trim().trim_end_matches('K').parse::<usize>()
        {
            return v * 1024;
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
        for path in &paths
        {
            if let Ok(s) = std::fs::read_to_string(path)
                && let Ok(v) = s.trim().trim_end_matches('K').parse::<usize>()
            {
                return v * 1024;
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Petite config déterministe pour forcer plusieurs tuiles (donc plusieurs
    /// passages sur les mêmes cellules de C via l'axe K).
    fn tiny_config() -> TilingConfig {
        TilingConfig {
            tile_m: 2,
            tile_k: 2,
            tile_n: 2,
            simd_width: 4,
            l2_cache_size: 262_144,
            l2_fill_fraction: 0.9,
        }
    }

    /// Non-régression: le beta-scaling de C doit respecter les strides de C.
    ///
    /// On stocke C comme un sous-bloc 2x2 d'une matrice ambiante 3x3 (rs_c = 3),
    /// avec beta != 0. Avant le correctif, la boucle d'init faisait `c[i] *= beta`
    /// sur les 4 premières cellules contiguës, ce qui (a) ne scalait pas les vraies
    /// cellules C et (b) corrompait des cellules voisines. Après correctif, seules
    /// les cellules logiques C[i*rs_c + j*cs_c] sont scalées et les voisines sont
    /// intactes.
    #[test]
    fn beta_scaling_respects_c_strides() {
        // C = 2x2, rangé dans une matrice 3x3 (ligne stride = 3, col stride = 1).
        // Cellules C = indices {0,1,3,4}. Cellules voisines (padding) = {2,5,6,7,8}.
        let m = 2usize;
        let k = 2usize;
        let n = 2usize;
        let (rs_c, cs_c) = (3usize, 1usize);

        // A (2x2) et B (2x2), contigus row-major.
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];

        // Produit A*B attendu (row-major 2x2):
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        let alpha = 1.0f32;
        let beta = 2.0f32;

        // Sentinelles distinctes pour distinguer chaque cellule.
        let mut c = [10.0f32, 20.0, 999.0, 30.0, 40.0, 998.0, 997.0, 996.0, 995.0];

        // Valeurs C logiques initiales: C[0,0]=10, C[0,1]=20, C[1,0]=30, C[1,1]=40.
        // Résultat attendu = alpha * A*B + beta * C_init.
        let cfg = tiny_config();
        matmul_tiled_strided_f32(
            alpha,
            &a,
            &b,
            beta,
            &mut c,
            m,
            k,
            n,
            k,
            1,
            n,
            1,
            rs_c,
            cs_c,
            Some(&cfg),
        );

        let expected_c00 = 19.0 + 2.0 * 10.0; // 39
        let expected_c01 = 22.0 + 2.0 * 20.0; // 62
        let expected_c10 = 43.0 + 2.0 * 30.0; // 103
        let expected_c11 = 50.0 + 2.0 * 40.0; // 130

        assert_eq!(c[0], expected_c00, "C[0,0]");
        assert_eq!(c[1], expected_c01, "C[0,1]");
        assert_eq!(c[3], expected_c10, "C[1,0]");
        assert_eq!(c[4], expected_c11, "C[1,1]");

        // Les cellules voisines (padding) doivent rester intactes.
        assert_eq!(c[2], 999.0, "padding [0,2] must not be touched");
        assert_eq!(c[5], 998.0, "padding [1,2] must not be touched");
        assert_eq!(c[6], 997.0, "padding [2,0] must not be touched");
        assert_eq!(c[7], 996.0, "padding [2,1] must not be touched");
        assert_eq!(c[8], 995.0, "padding [2,2] must not be touched");
    }

    /// Non-régression: le chemin vectorisé (SIMD) doit lui aussi respecter
    /// `rs_c` lorsque C est un sous-bloc row-strided d'une matrice ambiante
    /// (`rs_c > n`, `cs_c == 1`). Le beta-scaling scale `c[i*rs_c + j]` et le
    /// chemin SIMD accumule dans `c[i*rs_c + j]`: les deux doivent viser les
    /// mêmes cellules. On force `tile_n >= 4` et `n >= 4` pour que le bloc
    /// `while j + 4 <= jn` s'exécute réellement (ce que ne fait pas le test 2x2).
    ///
    /// Le résultat est comparé à une référence naïve stride-aware; les cellules
    /// de padding (`j >= n`) doivent rester intactes.
    #[cfg(feature = "portable-simd")]
    #[test]
    fn simd_path_respects_row_strided_c() {
        // Référence naïve stride-aware: C = alpha*A*B + beta*C.
        #[allow(clippy::too_many_arguments)]
        fn naive_ref(
            alpha: f32,
            a: &[f32],
            b: &[f32],
            beta: f32,
            c: &mut [f32],
            m: usize,
            k: usize,
            n: usize,
            rs_a: usize,
            cs_a: usize,
            rs_b: usize,
            cs_b: usize,
            rs_c: usize,
            cs_c: usize,
        ) {
            for i in 0..m
            {
                for j in 0..n
                {
                    let mut acc = 0.0f32;
                    for p in 0..k
                    {
                        acc += a[i * rs_a + p * cs_a] * b[p * rs_b + j * cs_b];
                    }
                    let idx = i * rs_c + j * cs_c;
                    c[idx] = alpha * acc + beta * c[idx];
                }
            }
        }

        let m = 5usize;
        let k = 4usize;
        let n = 6usize; // >= 4 pour déclencher le bloc SIMD
        let (rs_c, cs_c) = (8usize, 1usize); // C = sous-bloc row-strided (rs_c > n)

        let a: Vec<f32> = (0..m * k).map(|x| x as f32 * 0.5 + 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|x| x as f32 * 0.3 - 0.7).collect();

        // Matrice ambiante pour C, avec padding distinct par cellule.
        let ambient = (m - 1) * rs_c + n; // dernière cellule logique C[m-1, n-1]
        let c_init: Vec<f32> = (0..ambient).map(|x| x as f32 * 0.11 + 0.2).collect();

        let alpha = 1.3f32;
        let beta = 0.7f32;

        // Config qui produit plusieurs tuiles ET un tile_n >= 4 (bloc SIMD actif).
        let cfg = TilingConfig {
            tile_m: 2,
            tile_k: 2,
            tile_n: 4,
            simd_width: 4,
            l2_cache_size: 262_144,
            l2_fill_fraction: 0.9,
        };

        let mut c_simd = c_init.clone();
        matmul_tiled_strided_f32(
            alpha,
            &a,
            &b,
            beta,
            &mut c_simd,
            m,
            k,
            n,
            k,
            1,
            n,
            1,
            rs_c,
            cs_c,
            Some(&cfg),
        );

        let mut c_ref = c_init.clone();
        naive_ref(
            alpha, &a, &b, beta, &mut c_ref, m, k, n, k, 1, n, 1, rs_c, cs_c,
        );

        // Les cellules logiques de C doivent correspondre à la référence.
        for i in 0..m
        {
            for j in 0..n
            {
                let idx = i * rs_c + j * cs_c;
                let diff = (c_simd[idx] - c_ref[idx]).abs();
                assert!(
                    diff < 1e-3,
                    "C[{i},{j}] (idx {idx}): simd={} ref={} diff={diff}",
                    c_simd[idx],
                    c_ref[idx]
                );
            }
        }

        // Les cellules de padding (colonnes j >= n de chaque ligne) doivent
        // rester intactes: ni le beta-scaling ni le chemin SIMD ne doivent y toucher.
        for i in 0..m
        {
            for j in n..rs_c
            {
                let idx = i * rs_c + j;
                if idx >= ambient
                {
                    continue;
                }
                assert_eq!(
                    c_simd[idx], c_init[idx],
                    "padding [{i},{j}] (idx {idx}) must not be touched"
                );
            }
        }
    }
}
