//! # Intel AMX — accélérateur matriciel à tuiles (x86_64)
//!
//! *Advanced Matrix Extensions* : un fichier de **8 tuiles** `tmm0..7` (jusqu'à
//! `16 lignes × 64 octets` chacune) et une unité **TMUL** qui exécute un
//! produit matriciel `C += A·B` **par instruction** (`_tile_dpbssd` pour l'int8).
//! C'est le plafond de performance de l'inférence quantifiée sur Sapphire Rapids
//! / Emerald Rapids / Granite Rapids : ~8× le débit int8 d'AVX-512 VNNI.
//!
//! ## Modèle de calcul
//!
//! Une tuile int8 se décline en trois rôles :
//! * **A** (`src1`) : `M×K` int8, `M ≤ 16` lignes, `K ≤ 64` octets/ligne ;
//! * **B** (`src2`) : disposition **VNNI** `(K/4)×(N·4)` — `b_tile[p][4j+r] =
//!   B[4p+r][j]` — soit `K/4 ≤ 16` lignes, `N ≤ 16` colonnes int32 ;
//! * **C** (`dst`) : `M×N` int32 accumulé.
//!
//! `_tile_dpbssd::<C,A,B>()` calcule
//! `C[m][j] += Σ_{p<K/4} Σ_{r<4} A[m][4p+r]·B_vnni[p][4j+r]`, exactement
//! `Σ_{k<K} A[m][k]·B[k][j]`. Ce module tuile un GEMM `m×k×n` quelconque en
//! panneaux `≤16×≤64×≤16`, accumule sur `K` dans la tuile `C`, puis la stocke.
//!
//! ## Disponibilité & repli
//!
//! Deux conditions runtime : les ISA `amx-tile`+`amx-int8`, **et** la permission
//! noyau `ARCH_REQ_XCOMP_PERM` (Linux gère l'état AMX à la demande). Si l'une
//! manque, repli sur la **référence scalaire** — qui est aussi l'oracle de
//! correction. Le chemin AMX est compilé et prêt (drop-in matériel) ; il
//! s'exécute dès qu'une puce AMX est présente.
//!
//! ## Safety
//!
//! Les fonctions `#[target_feature(enable = "amx-int8,amx-tile")]` ne sont
//! atteintes qu'après [`amx_int8_usable`] (détection ISA + permission noyau
//! obtenue). Les `_tile_loadd`/`_tile_stored` lisent/écrivent des tampons
//! `[i8; 16*64]` / `[i32; 16*16]` de taille fixe, jamais les slices utilisateur
//! directement — le (dé)packing borne tous les accès.

#![allow(clippy::missing_safety_doc)]

use std::sync::OnceLock;

/// Produit matriciel **int8 → i32** `C[m×n] = A[m×k]·B[k×n]` (row-major), via
/// Intel AMX si disponible (ISA + permission noyau), sinon repli scalaire.
///
/// `a` est `m×k`, `b` est `k×n`, tous deux `i8` row-major ; renvoie `C` en
/// `Vec<i32>` de longueur `m·n`.
pub fn amx_matmul_i8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    assert_eq!(a.len(), m * k, "amx_matmul_i8: A shape mismatch");
    assert_eq!(b.len(), k * n, "amx_matmul_i8: B shape mismatch");
    let mut c = vec![0i32; m * n];
    #[cfg(target_arch = "x86_64")]
    {
        if amx_int8_usable()
        {
            // SAFETY: ISA AMX détectée + permission noyau obtenue (amx_int8_usable).
            unsafe { amx_matmul_i8_tiled(a, b, m, k, n, &mut c) };
            return c;
        }
    }
    matmul_i8_scalar_into(a, b, m, k, n, &mut c);
    c
}

/// Référence scalaire de [`amx_matmul_i8`] (aussi l'oracle des tests).
pub fn matmul_i8_scalar(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    matmul_i8_scalar_into(a, b, m, k, n, &mut c);
    c
}

fn matmul_i8_scalar_into(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, c: &mut [i32]) {
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = 0i32;
            for p in 0..k
            {
                acc += (a[i * k + p] as i32) * (b[p * n + j] as i32);
            }
            c[i * n + j] = acc;
        }
    }
}

/// `true` si AMX int8 est réellement utilisable : ISA `amx-tile`+`amx-int8`
/// présentes **et** permission `ARCH_REQ_XCOMP_PERM` obtenue du noyau. Résultat
/// mis en cache (la demande de permission n'est faite qu'une fois).
#[cfg(target_arch = "x86_64")]
pub fn amx_int8_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !std::is_x86_feature_detected!("amx-tile") || !std::is_x86_feature_detected!("amx-int8")
        {
            return false;
        }
        // SAFETY: simple appel système sans effet mémoire côté espace utilisateur.
        unsafe { request_amx_permission() }
    })
}

/// Demande au noyau Linux la permission d'utiliser l'état AMX `XTILEDATA` via
/// `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`. Renvoie `true` si
/// accordée. Sans cet appel, toute instruction de tuile fauterait (`#NM`).
#[cfg(target_arch = "x86_64")]
unsafe fn request_amx_permission() -> bool {
    const SYS_ARCH_PRCTL: i64 = 158;
    const ARCH_REQ_XCOMP_PERM: u64 = 0x1023;
    const XFEATURE_XTILEDATA: u64 = 18;
    let ret: i64;
    core::arch::asm!(
        "syscall",
        inlateout("rax") SYS_ARCH_PRCTL => ret,
        in("rdi") ARCH_REQ_XCOMP_PERM,
        in("rsi") XFEATURE_XTILEDATA,
        lateout("rcx") _,
        lateout("r11") _,
        options(nostack, preserves_flags),
    );
    ret == 0
}

/// Configuration des tuiles AMX (`LDTILECFG`) : palette 1, `rows[t]`/`colsb[t]`
/// par tuile. 64 octets, disposition imposée par l'ISA.
#[cfg(target_arch = "x86_64")]
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct TileConfig {
    palette_id: u8,
    start_row: u8,
    reserved: [u8; 14],
    colsb: [u16; 16],
    rows: [u8; 16],
}

#[cfg(target_arch = "x86_64")]
impl TileConfig {
    const fn zeroed() -> Self {
        TileConfig {
            palette_id: 1,
            start_row: 0,
            reserved: [0; 14],
            colsb: [0; 16],
            rows: [0; 16],
        }
    }
}

// Rôles de tuiles : C=accumulateur i32, A=int8 M×K, B=int8 VNNI (K/4)×(N·4).
#[cfg(target_arch = "x86_64")]
const TMM_C: usize = 0;
#[cfg(target_arch = "x86_64")]
const TMM_A: usize = 1;
#[cfg(target_arch = "x86_64")]
const TMM_B: usize = 2;
#[cfg(target_arch = "x86_64")]
const MAX_M: usize = 16; // lignes/tuile
#[cfg(target_arch = "x86_64")]
const MAX_N: usize = 16; // colonnes int32/tuile
#[cfg(target_arch = "x86_64")]
const MAX_K: usize = 64; // octets int8/ligne

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "amx-int8,amx-tile")]
unsafe fn amx_matmul_i8_tiled(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, c: &mut [i32]) {
    use core::arch::x86_64::*;

    // Tampons de packing (taille max d'une tuile), réutilisés par panneau.
    let mut a_buf = [0i8; MAX_M * MAX_K];
    let mut b_buf = [0i8; (MAX_K / 4) * (MAX_N * 4)];
    let mut c_buf = [0i32; MAX_M * MAX_N];

    let mut mb = 0;
    while mb < m
    {
        let mr = MAX_M.min(m - mb);
        let mut nb = 0;
        while nb < n
        {
            let nr = MAX_N.min(n - nb);
            let stride_b = nr * 4;

            // Config des tuiles pour ce panneau, **une seule fois** : `LDTILECFG`
            // remet à zéro toutes les tuiles, donc reconfigurer dans la boucle `K`
            // effacerait l'accumulateur `C`. On fixe donc les dimensions `A`/`B`
            // au maximum (`K = 64`, `B: 16` lignes) et on zéro-padde le packing
            // des blocs `K` partiels — les produits sur le padding valent 0.
            let mut cfg = TileConfig::zeroed();
            cfg.rows[TMM_C] = mr as u8;
            cfg.colsb[TMM_C] = stride_b as u16;
            cfg.rows[TMM_A] = mr as u8;
            cfg.colsb[TMM_A] = MAX_K as u16; // 64 octets int8/ligne
            cfg.rows[TMM_B] = (MAX_K / 4) as u8; // 16 lignes VNNI
            cfg.colsb[TMM_B] = stride_b as u16;
            _tile_loadconfig(&cfg as *const _ as *const u8);
            _tile_zero::<{ TMM_C as i32 }>();

            let mut kb = 0;
            while kb < k
            {
                let kr = MAX_K.min(k - kb);

                // Pack A : mr lignes × 64 octets (stride fixe MAX_K), zéro au-delà
                // de kr.
                a_buf.fill(0);
                for i in 0..mr
                {
                    for p in 0..kr
                    {
                        a_buf[i * MAX_K + p] = a[(mb + i) * k + kb + p];
                    }
                }
                // Pack B VNNI : b_buf[p*stride_b + 4*j + r] = B[kb + 4p+r][nb+j],
                // lignes au-delà de ceil(kr/4) laissées à zéro.
                b_buf.fill(0);
                let kp = kr.div_ceil(4);
                for p in 0..kp
                {
                    for j in 0..nr
                    {
                        for r in 0..4
                        {
                            let kk = 4 * p + r;
                            if kk < kr
                            {
                                b_buf[p * stride_b + 4 * j + r] = b[(kb + kk) * n + nb + j];
                            }
                        }
                    }
                }

                _tile_loadd::<{ TMM_A as i32 }>(a_buf.as_ptr() as *const u8, MAX_K);
                _tile_loadd::<{ TMM_B as i32 }>(b_buf.as_ptr() as *const u8, stride_b);
                _tile_dpbssd::<{ TMM_C as i32 }, { TMM_A as i32 }, { TMM_B as i32 }>();

                kb += MAX_K;
            }

            // Décharge la tuile C accumulée dans un tampon puis dans la sortie.
            _tile_stored::<{ TMM_C as i32 }>(c_buf.as_mut_ptr() as *mut u8, stride_b);
            for i in 0..mr
            {
                for j in 0..nr
                {
                    c[(mb + i) * n + nb + j] = c_buf[i * nr + j];
                }
            }

            nb += MAX_N;
        }
        mb += MAX_M;
    }
    _tile_release();
}

// ===================================================================== //
//  Couche dense quantifiée int8 (inférence)                              //
// ===================================================================== //

/// **Couche linéaire quantifiée int8** : `Y[f32] = scale ⊙ (X·W) + biais`, avec
/// `X` activations `i8` (`m×k`), `W` poids `i8` (`k×n`), `scale`/`biais` des
/// vecteurs de longueur `n` (déquantification **par canal de sortie** + biais).
///
/// C'est l'analogue int8 de [`crate::gemm::sgemm_bias_act`] : le produit
/// `X·W` accumulé en `i32` passe par le **GEMM AMX** ([`amx_matmul_i8`], donc
/// `_tile_dpbssd` sur silicium AMX, sinon repli scalaire), puis un épilogue
/// `O(m·n)` déquantifie (`i32 → f32` × `scale[j]`) et ajoute le biais. Motif
/// standard de l'inférence quantifiée (activation `u8`/`i8`, poids `i8` par
/// canal).
pub fn qlinear_i8(
    x: &[i8],
    w: &[i8],
    m: usize,
    k: usize,
    n: usize,
    scale: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    assert_eq!(scale.len(), n, "qlinear_i8: scale length != N");
    assert_eq!(bias.len(), n, "qlinear_i8: bias length != N");
    let acc = amx_matmul_i8(x, w, m, k, n);
    let mut y = vec![0f32; m * n];
    for i in 0..m
    {
        let arow = &acc[i * n..i * n + n];
        let yrow = &mut y[i * n..i * n + n];
        for j in 0..n
        {
            yrow[j] = scale[j] * (arow[j] as f32) + bias[j];
        }
    }
    y
}

// ===================================================================== //
//  AMX bf16 : C(f32) = A(bf16)·B(bf16)  (TDPBF16PS)                       //
// ===================================================================== //

/// Produit matriciel **bf16 → f32** `C[m×n] = A[m×k]·B[k×n]` (row-major, entrées
/// `bf16` stockées en `u16`), via Intel AMX (`_tile_dpbf16ps`) si disponible
/// (`amx-tile`+`amx-bf16` + permission noyau), sinon repli scalaire.
///
/// bf16 est le format d'entraînement/inférence de référence : l'AMX en fait un
/// GEMM matriciel accéléré, produits/accumulation `f32` (mêmes garanties de
/// précision que `VDPBF16PS`, cf. [`crate::quant::dot_bf16`]).
pub fn amx_matmul_bf16(a: &[u16], b: &[u16], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "amx_matmul_bf16: A shape mismatch");
    assert_eq!(b.len(), k * n, "amx_matmul_bf16: B shape mismatch");
    let mut c = vec![0f32; m * n];
    #[cfg(target_arch = "x86_64")]
    {
        if amx_bf16_usable()
        {
            // SAFETY: ISA AMX bf16 détectée + permission noyau obtenue.
            unsafe { amx_matmul_bf16_tiled(a, b, m, k, n, &mut c) };
            return c;
        }
    }
    matmul_bf16_scalar_into(a, b, m, k, n, &mut c);
    c
}

/// Référence scalaire de [`amx_matmul_bf16`] (aussi l'oracle des tests) :
/// élargit `bf16 → f32` puis accumule en `f32`.
pub fn matmul_bf16_scalar(a: &[u16], b: &[u16], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    matmul_bf16_scalar_into(a, b, m, k, n, &mut c);
    c
}

fn matmul_bf16_scalar_into(a: &[u16], b: &[u16], m: usize, k: usize, n: usize, c: &mut [f32]) {
    use crate::quant::bf16_to_f32;
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = 0f32;
            for p in 0..k
            {
                acc += bf16_to_f32(a[i * k + p]) * bf16_to_f32(b[p * n + j]);
            }
            c[i * n + j] = acc;
        }
    }
}

/// `true` si AMX bf16 est utilisable : ISA `amx-tile`+`amx-bf16` **et**
/// permission `ARCH_REQ_XCOMP_PERM` obtenue. Mis en cache.
#[cfg(target_arch = "x86_64")]
pub fn amx_bf16_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !std::is_x86_feature_detected!("amx-tile") || !std::is_x86_feature_detected!("amx-bf16")
        {
            return false;
        }
        // SAFETY: simple appel système sans effet mémoire côté espace utilisateur.
        unsafe { request_amx_permission() }
    })
}

/// `K` bf16 max par ligne de tuile A : 64 octets / 2 = 32 ⇒ `B` VNNI a `K/2 = 16`
/// lignes (2 bf16 par groupe int32).
#[cfg(target_arch = "x86_64")]
const MAX_K_BF16: usize = 32;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "amx-bf16,amx-tile")]
unsafe fn amx_matmul_bf16_tiled(a: &[u16], b: &[u16], m: usize, k: usize, n: usize, c: &mut [f32]) {
    use core::arch::x86_64::*;

    // Tampons bf16 (u16) et f32, dimensionnés au pire cas d'une tuile.
    let mut a_buf = [0u16; MAX_M * MAX_K_BF16]; // 16×32 bf16 = 16×64 octets
    let mut b_buf = [0u16; (MAX_K_BF16 / 2) * (MAX_N * 2)]; // 16 lignes × N·2 bf16
    let mut c_buf = [0f32; MAX_M * MAX_N];

    let mut mb = 0;
    while mb < m
    {
        let mr = MAX_M.min(m - mb);
        let mut nb = 0;
        while nb < n
        {
            let nr = MAX_N.min(n - nb);
            let stride_c = nr * 4; // N f32 = N·4 octets
            let stride_b = nr * 2 * 2; // N·2 bf16 = N·4 octets
            let stride_a = MAX_K_BF16 * 2; // 32 bf16 = 64 octets

            // Config unique par panneau (cf. variante int8 : LDTILECFG zéro-padde
            // les tuiles ; dimensions A/B fixées au max, blocs K partiels zéro-
            // paddés).
            let mut cfg = TileConfig::zeroed();
            cfg.rows[TMM_C] = mr as u8;
            cfg.colsb[TMM_C] = stride_c as u16;
            cfg.rows[TMM_A] = mr as u8;
            cfg.colsb[TMM_A] = stride_a as u16;
            cfg.rows[TMM_B] = (MAX_K_BF16 / 2) as u8; // 16 lignes VNNI
            cfg.colsb[TMM_B] = stride_b as u16;
            _tile_loadconfig(&cfg as *const _ as *const u8);
            _tile_zero::<{ TMM_C as i32 }>();

            let mut kb = 0;
            while kb < k
            {
                let kr = MAX_K_BF16.min(k - kb);

                a_buf.fill(0);
                for i in 0..mr
                {
                    for p in 0..kr
                    {
                        a_buf[i * MAX_K_BF16 + p] = a[(mb + i) * k + kb + p];
                    }
                }
                // Pack B VNNI bf16 : b_buf[p*(nr*2) + 2*j + r] = B[kb + 2p+r][nb+j].
                b_buf.fill(0);
                let kp = kr.div_ceil(2);
                for p in 0..kp
                {
                    for j in 0..nr
                    {
                        for r in 0..2
                        {
                            let kk = 2 * p + r;
                            if kk < kr
                            {
                                b_buf[p * (nr * 2) + 2 * j + r] = b[(kb + kk) * n + nb + j];
                            }
                        }
                    }
                }

                _tile_loadd::<{ TMM_A as i32 }>(a_buf.as_ptr() as *const u8, stride_a);
                _tile_loadd::<{ TMM_B as i32 }>(b_buf.as_ptr() as *const u8, stride_b);
                _tile_dpbf16ps::<{ TMM_C as i32 }, { TMM_A as i32 }, { TMM_B as i32 }>();

                kb += MAX_K_BF16;
            }

            _tile_stored::<{ TMM_C as i32 }>(c_buf.as_mut_ptr() as *mut u8, stride_c);
            for i in 0..mr
            {
                for j in 0..nr
                {
                    c[(mb + i) * n + nb + j] = c_buf[i * nr + j];
                }
            }

            nb += MAX_N;
        }
        mb += MAX_M;
    }
    _tile_release();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::f32_to_bf16;

    #[test]
    fn amx_matmul_matches_scalar_or_falls_back() {
        // Sur cette machine sans AMX, `amx_matmul_i8` prend le repli scalaire ;
        // sur une puce AMX, le chemin tuilé. Les deux doivent coïncider avec la
        // référence indépendante — formes couvrant plusieurs tuiles (M/N > 16,
        // K > 64) et des bords partiels.
        let shapes = [
            (1usize, 1usize, 1usize),
            (3, 5, 2),
            (16, 64, 16),
            (17, 65, 19),
            (20, 100, 24),
            (33, 40, 7),
        ];
        for &(m, k, n) in &shapes
        {
            let a: Vec<i8> = (0..m * k)
                .map(|t| ((t as i32 * 7 - 61) % 128) as i8)
                .collect();
            let b: Vec<i8> = (0..k * n)
                .map(|t| ((t as i32 * -5 + 23) % 128) as i8)
                .collect();
            let got = amx_matmul_i8(&a, &b, m, k, n);
            let want = matmul_i8_scalar(&a, &b, m, k, n);
            assert_eq!(got, want, "shape {m}x{k}x{n}");
        }
    }

    #[test]
    fn scalar_known_value() {
        // A(2x3)·B(3x2) = [[58,64],[139,154]] (int8).
        let a = [1i8, 2, 3, 4, 5, 6];
        let b = [7i8, 8, 9, 10, 11, 12];
        assert_eq!(matmul_i8_scalar(&a, &b, 2, 3, 2), vec![58, 64, 139, 154]);
    }

    #[test]
    fn qlinear_i8_matches_reference() {
        // Déquantification par canal + biais au-dessus du GEMM AMX/scalaire.
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (4, 6, 3), (18, 70, 20)]
        {
            let x: Vec<i8> = (0..m * k)
                .map(|t| ((t as i32 * 7 - 61) % 128) as i8)
                .collect();
            let w: Vec<i8> = (0..k * n)
                .map(|t| ((t as i32 * -5 + 23) % 128) as i8)
                .collect();
            let scale: Vec<f32> = (0..n).map(|j| 0.001 * (j as f32 + 1.0)).collect();
            let bias: Vec<f32> = (0..n).map(|j| (j as f32) * 0.1 - 0.5).collect();

            let got = qlinear_i8(&x, &w, m, k, n, &scale, &bias);
            // Référence indépendante : accumulation i32 naïve, puis déquant+biais.
            let mut want = vec![0f32; m * n];
            for i in 0..m
            {
                for j in 0..n
                {
                    let mut acc = 0i32;
                    for p in 0..k
                    {
                        acc += (x[i * k + p] as i32) * (w[p * n + j] as i32);
                    }
                    want[i * n + j] = scale[j] * (acc as f32) + bias[j];
                }
            }
            for t in 0..m * n
            {
                assert!(
                    (got[t] - want[t]).abs() <= 1e-3 * (1.0 + want[t].abs()),
                    "shape {m}x{k}x{n} t={t}: {} vs {}",
                    got[t],
                    want[t]
                );
            }
        }
    }

    #[test]
    fn amx_bf16_matches_scalar_or_falls_back() {
        // Chemin AMX bf16 (si présent) ou repli scalaire — dans les deux cas,
        // coïncidence avec la référence indépendante (bf16→f32). Formes couvrant
        // plusieurs tuiles (M/N > 16, K > 32) et bords partiels. Tolérance bf16.
        let shapes = [
            (1usize, 1usize, 1usize),
            (4, 6, 3),
            (16, 32, 16),
            (18, 33, 20),
            (20, 70, 24),
        ];
        for &(m, k, n) in &shapes
        {
            let af: Vec<f32> = (0..m * k).map(|t| (t as f32 * 0.017).sin() * 0.5).collect();
            let bf: Vec<f32> = (0..k * n).map(|t| (t as f32 * 0.013).cos() * 0.5).collect();
            let a: Vec<u16> = af.iter().map(|&x| f32_to_bf16(x)).collect();
            let b: Vec<u16> = bf.iter().map(|&x| f32_to_bf16(x)).collect();
            let got = amx_matmul_bf16(&a, &b, m, k, n);
            let want = matmul_bf16_scalar(&a, &b, m, k, n);
            for t in 0..m * n
            {
                assert!(
                    (got[t] - want[t]).abs() <= 1e-2 * (1.0 + want[t].abs()),
                    "shape {m}x{k}x{n} t={t}: {} vs {}",
                    got[t],
                    want[t]
                );
            }
        }
    }
}
