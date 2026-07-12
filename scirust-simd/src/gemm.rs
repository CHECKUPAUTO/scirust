//! # GEMM tuilé multi-niveaux (x86_64) — `f32` (SGEMM) et `f64` (DGEMM)
//!
//! `C = alpha·A·B + beta·C`, matrices **row-major**. Le SGEMM (`f32`, tuile
//! registre `8×16`) et le DGEMM (`f64`, tuile `8×8`) partagent la même
//! structure ; les entrées `dgemm_*` sont l'analogue double-précision des
//! `sgemm_*` décrites ci-dessous.
//!
//! Le SGEMM « rank-1 » du module [`crate::dispatch`] est simple et
//! cache-friendly en lecture de `B`, mais il relit et réécrit toute la ligne de
//! `C` à chaque `p` : le trafic sur `C` domine dès que `k` est grand. Ce module
//! implémente la structure classique haute-performance (façon BLIS) :
//!
//! * **Blocking cache** sur `M`/`K`/`N` (`MC`/`KC`/`NC`) pour que les panneaux
//!   travaillés tiennent en L2/L1 et soient réutilisés.
//! * **Packing explicite** des panneaux de `A` (`MC×KC`) et `B` (`KC×NC`) dans
//!   des buffers contigus, re-agencés pour que le micro-kernel les lise en
//!   **stride unitaire** (accès séquentiels, prefetch matériel optimal, plus de
//!   lecture croisée `ldb`/`lda`). `alpha` est fusionné dans le packing de `A`.
//! * **Micro-kernel registre-bloqué** `MR×NR` (`8×16`) : 8 accumulateurs `zmm`
//!   maintiennent une tuile `8×16` de `C` **dans les registres** pendant toute
//!   la dimension `KC`. Les accès à `C` sont amortis sur `KC` produits.
//! * **Bords masqués** : les tuiles partielles (`< 16` colonnes) sont gérées
//!   par masque `k` AVX-512 ; les lignes/colonnes manquantes sont zéro-paddées
//!   dans les buffers de packing.
//! * **Parallélisme** : [`sgemm_parallel`] découpe la dimension `M` en blocs de
//!   lignes disjoints (donc des tranches `row-major` contiguës, sans partage
//!   mutable) confiés à des threads via `std::thread::scope` — aucune
//!   dépendance externe.
//!
//! Repli automatique : hors `avx512f`, on délègue au SGEMM scalaire de
//! référence. Le résultat est vérifié *proche* du scalaire dans les tests.

use crate::matrix::backend::{ScalarBackend, SimdBackend};
use crate::matrix::view::{MatrixView, MatrixViewMut};

/// Dimensions de blocking SGEMM. `MR`/`NR` fixent la tuile registre ;
/// `MC`/`KC`/`NC` dimensionnent les panneaux packés pour la hiérarchie de
/// caches. Utilisées uniquement par le noyau AVX-512 (cf. `cfg` : sur les
/// cibles sans ce noyau elles seraient du code mort → erreur sous `-D warnings`).
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

/// SGEMM tuilé (mono-thread) : `C = alpha·A(m×k)·B(k×n) + beta·C(m×n)`,
/// row-major. Dispatch runtime vers le noyau AVX-512 packé si disponible,
/// sinon la référence scalaire.
pub fn sgemm_tiled(
    alpha: f32,
    a: MatrixView<f32>,
    b: MatrixView<f32>,
    beta: f32,
    c: MatrixViewMut<f32>,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    assert_eq!(b.rows(), k, "sgemm_tiled: A.cols != B.rows");
    assert_eq!(c.rows(), m, "sgemm_tiled: C.rows != A.rows");
    assert_eq!(c.cols(), n, "sgemm_tiled: C.cols != B.cols");

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: gated by the runtime detection just above; slices come
            // from validated MatrixView / MatrixViewMut.
            unsafe { sgemm_tiled_avx512(alpha, a, b, beta, c) };
            return;
        }
    }
    ScalarBackend.sgemm_f32(alpha, a, b, beta, c);
}

/// SGEMM tuilé **multi-thread** sur tranches de slices row-major.
///
/// `a` est `m×k`, `b` est `k×n`, `c` est `m×n`, tous row-major contigus.
/// La dimension `M` est partitionnée en `threads` blocs de lignes disjoints ;
/// chaque bloc est un GEMM indépendant `C_bloc = alpha·A_bloc·B + beta·C_bloc`,
/// exécuté sur son propre thread (aucun chevauchement mémoire). `threads == 0`
/// ou `1`, ou `m` trop petit, retombe sur l'exécution mono-thread.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_parallel(
    alpha: f32,
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    beta: f32,
    c: &mut [f32],
    threads: usize,
) {
    assert_eq!(a.len(), m * k, "sgemm_parallel: A shape mismatch");
    assert_eq!(b.len(), k * n, "sgemm_parallel: B shape mismatch");
    assert_eq!(c.len(), m * n, "sgemm_parallel: C shape mismatch");

    let nt = threads.max(1).min(m.max(1));
    if nt <= 1 || m == 0
    {
        sgemm_tiled(
            alpha,
            MatrixView::new(a, m, k),
            MatrixView::new(b, k, n),
            beta,
            MatrixViewMut::new(c, m, n),
        );
        return;
    }

    // Répartition équilibrée des lignes : les `rem` premiers blocs ont +1 ligne.
    let base = m / nt;
    let rem = m % nt;

    std::thread::scope(|scope| {
        let mut a_rest = a;
        let mut c_rest = &mut c[..];
        let mut row0 = 0;
        for t in 0..nt
        {
            let rows = base + usize::from(t < rem);
            if rows == 0
            {
                continue;
            }
            let (a_chunk, a_tail) = a_rest.split_at(rows * k);
            let (c_chunk, c_tail) = c_rest.split_at_mut(rows * n);
            a_rest = a_tail;
            c_rest = c_tail;
            debug_assert!(row0 + rows <= m);
            row0 += rows;
            scope.spawn(move || {
                sgemm_tiled(
                    alpha,
                    MatrixView::new(a_chunk, rows, k),
                    MatrixView::new(b, k, n),
                    beta,
                    MatrixViewMut::new(c_chunk, rows, n),
                );
            });
        }
    });
}

// ===================================================================== //
//  Noyau AVX-512 packé                                                    //
// ===================================================================== //

/// Applique `beta` sur toute la matrice C (row-major) une bonne fois pour
/// toutes ; le noyau accumule ensuite `alpha·A·B` par-dessus.
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
            let r = if beta == 0.0
            {
                _mm512_setzero_ps()
            }
            else
            {
                _mm512_mul_ps(_mm512_loadu_ps(row.add(j)), bv)
            };
            _mm512_storeu_ps(row.add(j), r);
            j += 16;
        }
        let rem = n - j;
        if rem > 0
        {
            let mask = (1u16 << rem) - 1;
            let r = if beta == 0.0
            {
                _mm512_setzero_ps()
            }
            else
            {
                _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, row.add(j)), bv)
            };
            _mm512_mask_storeu_ps(row.add(j), mask, r);
        }
    }
}

/// Pack un panneau `B[pc.., jc..]` de `kc×nc` en panneaux de `NR` colonnes,
/// row-major par `p`, zéro-paddé à `NR` : `dst[panel*(kc*NR) + p*NR + j]`.
#[cfg(target_arch = "x86_64")]
unsafe fn pack_b(b: *const f32, ldb: usize, kc: usize, nc: usize, dst: &mut [f32]) {
    let n_panels = nc.div_ceil(NR);
    for panel in 0..n_panels
    {
        let j0 = panel * NR;
        let nr = NR.min(nc - j0);
        let base = panel * kc * NR;
        for p in 0..kc
        {
            let src_row = b.add(p * ldb + j0);
            let dst_row = base + p * NR;
            for j in 0..nr
            {
                dst[dst_row + j] = *src_row.add(j);
            }
            for j in nr..NR
            {
                dst[dst_row + j] = 0.0;
            }
        }
    }
}

/// Pack un panneau `A[ic.., pc..]` de `mc×kc` en panneaux de `MR` lignes,
/// re-agencé par `p` (`dst[panel*(kc*MR) + p*MR + i]`), zéro-paddé à `MR`, avec
/// `alpha` fusionné.
#[cfg(target_arch = "x86_64")]
unsafe fn pack_a(alpha: f32, a: *const f32, lda: usize, mc: usize, kc: usize, dst: &mut [f32]) {
    let m_panels = mc.div_ceil(MR);
    for panel in 0..m_panels
    {
        let i0 = panel * MR;
        let mr = MR.min(mc - i0);
        let base = panel * kc * MR;
        for p in 0..kc
        {
            let dst_row = base + p * MR;
            for i in 0..mr
            {
                dst[dst_row + i] = alpha * *a.add((i0 + i) * lda + p);
            }
            for i in mr..MR
            {
                dst[dst_row + i] = 0.0;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_tiled_avx512(
    alpha: f32,
    a: MatrixView<f32>,
    b: MatrixView<f32>,
    beta: f32,
    mut c: MatrixViewMut<f32>,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    if m == 0 || n == 0
    {
        return;
    }
    // MatrixView contiguës row-major : pointeurs de base via la ligne 0.
    let c_ptr = c.row_slice_mut(0).expect("C base").as_mut_ptr();
    scale_c_avx512(beta, m, n, c_ptr);
    if k == 0 || alpha == 0.0
    {
        return;
    }
    let a_ptr = a.row_slice(0).expect("A base").as_ptr();
    let b_ptr = b.row_slice(0).expect("B base").as_ptr();

    // Buffers de packing réutilisés (alloués une fois, dimensionnés au pire cas).
    let mut bpack = vec![0.0f32; KC * NC.div_ceil(NR) * NR];
    let mut apack = vec![0.0f32; KC * MC.div_ceil(MR) * MR];

    let mut jc = 0;
    while jc < n
    {
        let nc = NC.min(n - jc);
        let n_panels = nc.div_ceil(NR);
        let mut pc = 0;
        while pc < k
        {
            let kc = KC.min(k - pc);
            pack_b(b_ptr.add(pc * n + jc), n, kc, nc, &mut bpack);
            let mut ic = 0;
            while ic < m
            {
                let mc = MC.min(m - ic);
                pack_a(alpha, a_ptr.add(ic * k + pc), k, mc, kc, &mut apack);
                let m_panels = mc.div_ceil(MR);
                for ip in 0..m_panels
                {
                    let i0 = ic + ip * MR;
                    let mr = MR.min(m - i0);
                    let apanel = &apack[ip * kc * MR..];
                    for jp in 0..n_panels
                    {
                        let j0 = jc + jp * NR;
                        let nr = NR.min(n - j0);
                        let bpanel = &bpack[jp * kc * NR..];
                        micro_kernel_8x16(
                            apanel.as_ptr(),
                            bpanel.as_ptr(),
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

/// Micro-kernel packé : `C_tile[mr×nr] += Apanel[kc×MR] · Bpanel[kc×NR]`.
///
/// `Apanel`/`Bpanel` sont contigus (stride unitaire), zéro-paddés à `MR`/`NR` ;
/// `alpha` est déjà fusionné dans `Apanel`. Maintient `MR` accumulateurs `zmm`
/// sur toute la dimension `kc`, puis ajoute la tuile à `C` (déjà scalé par
/// `beta`) avec un masque `k` sur les `nr` colonnes utiles et seulement `mr`
/// lignes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_kernel_8x16(
    apanel: *const f32,
    bpanel: *const f32,
    kc: usize,
    mr: usize,
    nr: usize,
    c: *mut f32,
    ldc: usize,
) {
    use core::arch::x86_64::*;
    debug_assert!(mr <= MR && nr <= NR);
    let mask: u16 = if nr >= 16 { 0xffff } else { (1u16 << nr) - 1 };

    let mut acc = [_mm512_setzero_ps(); MR];
    for p in 0..kc
    {
        let bv = _mm512_loadu_ps(bpanel.add(p * NR));
        let arow = apanel.add(p * MR);
        for (i, ai) in acc.iter_mut().enumerate()
        {
            *ai = _mm512_fmadd_ps(_mm512_set1_ps(*arow.add(i)), bv, *ai);
        }
    }
    for (i, ai) in acc.iter().enumerate().take(mr)
    {
        let crow = c.add(i * ldc);
        let cv = _mm512_maskz_loadu_ps(mask, crow);
        _mm512_mask_storeu_ps(crow, mask, _mm512_add_ps(cv, *ai));
    }
}

// ===================================================================== //
//  DGEMM (f64) — même structure tuilée/packée, tuile registre 8×8        //
// ===================================================================== //

/// Tuile registre f64 : un `zmm` contient 8 `f64`, donc `NR_D = 8` colonnes ;
/// `MR_D = 8` lignes → 8 accumulateurs `zmm`. Noyau AVX-512 uniquement (cf.
/// `cfg`, même raison que les constantes SGEMM ci-dessus).
#[cfg(target_arch = "x86_64")]
const MR_D: usize = 8;
#[cfg(target_arch = "x86_64")]
const NR_D: usize = 8;
#[cfg(target_arch = "x86_64")]
const KC_D: usize = 256;
#[cfg(target_arch = "x86_64")]
const MC_D: usize = 256;
#[cfg(target_arch = "x86_64")]
const NC_D: usize = 512;

/// DGEMM tuilé (mono-thread) : `C = alpha·A(m×k)·B(k×n) + beta·C(m×n)` en `f64`,
/// row-major. Dispatch runtime vers le noyau AVX-512 packé si disponible, sinon
/// la référence scalaire.
pub fn dgemm_tiled(
    alpha: f64,
    a: MatrixView<f64>,
    b: MatrixView<f64>,
    beta: f64,
    c: MatrixViewMut<f64>,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    assert_eq!(b.rows(), k, "dgemm_tiled: A.cols != B.rows");
    assert_eq!(c.rows(), m, "dgemm_tiled: C.rows != A.rows");
    assert_eq!(c.cols(), n, "dgemm_tiled: C.cols != B.cols");

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: gated by the runtime detection just above.
            unsafe { dgemm_tiled_avx512(alpha, a, b, beta, c) };
            return;
        }
    }
    dgemm_scalar(alpha, a, b, beta, c);
}

/// Référence/repli scalaire `f64` (triple boucle naïve).
#[allow(clippy::needless_range_loop)]
fn dgemm_scalar(
    alpha: f64,
    a: MatrixView<f64>,
    b: MatrixView<f64>,
    beta: f64,
    mut c: MatrixViewMut<f64>,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    for i in 0..m
    {
        let a_row = a.row_slice(i).expect("A row");
        let c_row = c.row_slice_mut(i).expect("C row");
        for j in 0..n
        {
            let mut acc = 0.0f64;
            for p in 0..k
            {
                acc += a_row[p] * b.row_slice(p).expect("B row")[j];
            }
            c_row[j] = alpha * acc + beta * c_row[j];
        }
    }
}

/// DGEMM tuilé **multi-thread** (analogue `f64` de [`sgemm_parallel`]).
#[allow(clippy::too_many_arguments)]
pub fn dgemm_parallel(
    alpha: f64,
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    beta: f64,
    c: &mut [f64],
    threads: usize,
) {
    assert_eq!(a.len(), m * k, "dgemm_parallel: A shape mismatch");
    assert_eq!(b.len(), k * n, "dgemm_parallel: B shape mismatch");
    assert_eq!(c.len(), m * n, "dgemm_parallel: C shape mismatch");

    let nt = threads.max(1).min(m.max(1));
    if nt <= 1 || m == 0
    {
        dgemm_tiled(
            alpha,
            MatrixView::new(a, m, k),
            MatrixView::new(b, k, n),
            beta,
            MatrixViewMut::new(c, m, n),
        );
        return;
    }

    let base = m / nt;
    let rem = m % nt;
    std::thread::scope(|scope| {
        let mut a_rest = a;
        let mut c_rest = &mut c[..];
        for t in 0..nt
        {
            let rows = base + usize::from(t < rem);
            if rows == 0
            {
                continue;
            }
            let (a_chunk, a_tail) = a_rest.split_at(rows * k);
            let (c_chunk, c_tail) = c_rest.split_at_mut(rows * n);
            a_rest = a_tail;
            c_rest = c_tail;
            scope.spawn(move || {
                dgemm_tiled(
                    alpha,
                    MatrixView::new(a_chunk, rows, k),
                    MatrixView::new(b, k, n),
                    beta,
                    MatrixViewMut::new(c_chunk, rows, n),
                );
            });
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scale_c_d_avx512(beta: f64, m: usize, n: usize, c: *mut f64) {
    use core::arch::x86_64::*;
    if beta == 1.0
    {
        return;
    }
    let bv = _mm512_set1_pd(beta);
    for i in 0..m
    {
        let row = c.add(i * n);
        let mut j = 0;
        while j + 8 <= n
        {
            let r = if beta == 0.0
            {
                _mm512_setzero_pd()
            }
            else
            {
                _mm512_mul_pd(_mm512_loadu_pd(row.add(j)), bv)
            };
            _mm512_storeu_pd(row.add(j), r);
            j += 8;
        }
        let rem = n - j;
        if rem > 0
        {
            let mask = (1u8 << rem) - 1;
            let r = if beta == 0.0
            {
                _mm512_setzero_pd()
            }
            else
            {
                _mm512_mul_pd(_mm512_maskz_loadu_pd(mask, row.add(j)), bv)
            };
            _mm512_mask_storeu_pd(row.add(j), mask, r);
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn pack_b_d(b: *const f64, ldb: usize, kc: usize, nc: usize, dst: &mut [f64]) {
    let n_panels = nc.div_ceil(NR_D);
    for panel in 0..n_panels
    {
        let j0 = panel * NR_D;
        let nr = NR_D.min(nc - j0);
        let base = panel * kc * NR_D;
        for p in 0..kc
        {
            let src_row = b.add(p * ldb + j0);
            let dst_row = base + p * NR_D;
            for j in 0..nr
            {
                dst[dst_row + j] = *src_row.add(j);
            }
            for j in nr..NR_D
            {
                dst[dst_row + j] = 0.0;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn pack_a_d(alpha: f64, a: *const f64, lda: usize, mc: usize, kc: usize, dst: &mut [f64]) {
    let m_panels = mc.div_ceil(MR_D);
    for panel in 0..m_panels
    {
        let i0 = panel * MR_D;
        let mr = MR_D.min(mc - i0);
        let base = panel * kc * MR_D;
        for p in 0..kc
        {
            let dst_row = base + p * MR_D;
            for i in 0..mr
            {
                dst[dst_row + i] = alpha * *a.add((i0 + i) * lda + p);
            }
            for i in mr..MR_D
            {
                dst[dst_row + i] = 0.0;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dgemm_tiled_avx512(
    alpha: f64,
    a: MatrixView<f64>,
    b: MatrixView<f64>,
    beta: f64,
    mut c: MatrixViewMut<f64>,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    if m == 0 || n == 0
    {
        return;
    }
    let c_ptr = c.row_slice_mut(0).expect("C base").as_mut_ptr();
    scale_c_d_avx512(beta, m, n, c_ptr);
    if k == 0 || alpha == 0.0
    {
        return;
    }
    let a_ptr = a.row_slice(0).expect("A base").as_ptr();
    let b_ptr = b.row_slice(0).expect("B base").as_ptr();

    let mut bpack = vec![0.0f64; KC_D * NC_D.div_ceil(NR_D) * NR_D];
    let mut apack = vec![0.0f64; KC_D * MC_D.div_ceil(MR_D) * MR_D];

    let mut jc = 0;
    while jc < n
    {
        let nc = NC_D.min(n - jc);
        let n_panels = nc.div_ceil(NR_D);
        let mut pc = 0;
        while pc < k
        {
            let kc = KC_D.min(k - pc);
            pack_b_d(b_ptr.add(pc * n + jc), n, kc, nc, &mut bpack);
            let mut ic = 0;
            while ic < m
            {
                let mc = MC_D.min(m - ic);
                pack_a_d(alpha, a_ptr.add(ic * k + pc), k, mc, kc, &mut apack);
                let m_panels = mc.div_ceil(MR_D);
                for ip in 0..m_panels
                {
                    let i0 = ic + ip * MR_D;
                    let mr = MR_D.min(m - i0);
                    let apanel = &apack[ip * kc * MR_D..];
                    for jp in 0..n_panels
                    {
                        let j0 = jc + jp * NR_D;
                        let nr = NR_D.min(n - j0);
                        let bpanel = &bpack[jp * kc * NR_D..];
                        micro_kernel_dgemm(
                            apanel.as_ptr(),
                            bpanel.as_ptr(),
                            kc,
                            mr,
                            nr,
                            c_ptr.add(i0 * n + j0),
                            n,
                        );
                    }
                }
                ic += MC_D;
            }
            pc += KC_D;
        }
        jc += NC_D;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_kernel_dgemm(
    apanel: *const f64,
    bpanel: *const f64,
    kc: usize,
    mr: usize,
    nr: usize,
    c: *mut f64,
    ldc: usize,
) {
    use core::arch::x86_64::*;
    debug_assert!(mr <= MR_D && nr <= NR_D);
    let mask: u8 = if nr >= 8 { 0xff } else { (1u8 << nr) - 1 };

    let mut acc = [_mm512_setzero_pd(); MR_D];
    for p in 0..kc
    {
        let bv = _mm512_loadu_pd(bpanel.add(p * NR_D));
        let arow = apanel.add(p * MR_D);
        for (i, ai) in acc.iter_mut().enumerate()
        {
            *ai = _mm512_fmadd_pd(_mm512_set1_pd(*arow.add(i)), bv, *ai);
        }
    }
    for (i, ai) in acc.iter().enumerate().take(mr)
    {
        let crow = c.add(i * ldc);
        let cv = _mm512_maskz_loadu_pd(mask, crow);
        _mm512_mask_storeu_pd(crow, mask, _mm512_add_pd(cv, *ai));
    }
}

// ===================================================================== //
//  GEMM à épilogue fusionné : couche dense + activation (SGEMM f32)      //
// ===================================================================== //

/// Fonction d'activation appliquée dans l'épilogue fusionné.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// Aucune activation (couche linéaire pure).
    Identity,
    /// `max(x, 0)` — appliquée en registre (`_mm512_max_ps`).
    Relu,
    /// GELU (approximation tanh) — via le noyau vectorisé de
    /// [`crate::activations`].
    Gelu,
    /// SiLU / swish (`x·sigmoid(x)`) — via [`crate::activations`].
    Silu,
}

impl Activation {
    /// Applique l'activation scalaire (repli + référence).
    #[inline]
    fn apply_scalar(self, x: f32) -> f32 {
        match self
        {
            Activation::Identity => x,
            Activation::Relu => x.max(0.0),
            Activation::Gelu => crate::activations::gelu_scalar(x),
            Activation::Silu => crate::activations::silu_scalar(x),
        }
    }
}

/// **Couche dense fusionnée** : `out = act(alpha·A·B + biais)`, avec `A` de
/// forme `m×k`, `B` de forme `k×n`, `biais` un vecteur-ligne de longueur `n`
/// diffusé sur chaque ligne, et `out` de forme `m×n` (row-major).
///
/// C'est le calcul exact d'une couche linéaire suivie d'une activation
/// (`Y = act(X·W + b)`). Le produit `alpha·A·B` passe par le **GEMM tuilé/packé
/// complet** ([`sgemm_tiled`]) — donc **n'importe quel `k`** et l'accélération
/// AVX-512/multi-bloc — puis le biais et l'activation sont appliqués en **un
/// seul passage `O(m·n)`** (épilogue vectorisé), négligeable devant le matmul.
/// Dispatch AVX-512 / repli scalaire pour l'épilogue.
pub fn sgemm_bias_act(
    alpha: f32,
    a: MatrixView<f32>,
    b: MatrixView<f32>,
    bias: &[f32],
    act: Activation,
    mut c: MatrixViewMut<f32>,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    assert_eq!(b.rows(), k, "sgemm_bias_act: A.cols != B.rows");
    assert_eq!(c.rows(), m, "sgemm_bias_act: C.rows != A.rows");
    assert_eq!(c.cols(), n, "sgemm_bias_act: C.cols != B.cols");
    assert_eq!(bias.len(), n, "sgemm_bias_act: bias length != N");
    if m == 0 || n == 0
    {
        return;
    }

    // Pointeur de base vers le buffer contigu `m×n` de C, capturé avant que
    // `sgemm_tiled` ne consomme la vue.
    let c_ptr = c.row_slice_mut(0).expect("C base").as_mut_ptr();

    // C = alpha·A·B  (GEMM tuilé complet — tout k). `k == 0` ⇒ C reste nul.
    sgemm_tiled(alpha, a, b, 0.0, c);

    // Épilogue : C[i,j] = act(C[i,j] + biais[j]), en un passage.
    // SAFETY: `c_ptr` désigne le buffer row-major contigu `m×n` de C (col_stride
    // 1) ; la vue `c` a été consommée par `sgemm_tiled` et n'a plus d'emprunt
    // actif, donc on reconstruit une tranche mutable exclusive valide.
    let cs = unsafe { std::slice::from_raw_parts_mut(c_ptr, m * n) };
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: gated by the runtime detection just above.
            unsafe { bias_act_epilogue_avx512(cs, m, n, bias, act) };
            return;
        }
    }
    bias_act_epilogue_scalar(cs, m, n, bias, act);
}

/// Épilogue scalaire : `c[i,j] = act(c[i,j] + bias[j])` (repli portable).
fn bias_act_epilogue_scalar(c: &mut [f32], m: usize, n: usize, bias: &[f32], act: Activation) {
    for i in 0..m
    {
        let row = &mut c[i * n..i * n + n];
        for (j, cj) in row.iter_mut().enumerate()
        {
            *cj = act.apply_scalar(*cj + bias[j]);
        }
    }
}

/// Applique l'activation à un vecteur `__m512` de pré-activations.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn apply_act_ps(
    pre: core::arch::x86_64::__m512,
    act: Activation,
) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    match act
    {
        Activation::Identity => pre,
        Activation::Relu => _mm512_max_ps(pre, _mm512_setzero_ps()),
        Activation::Gelu => crate::activations::gelu_ps(pre),
        Activation::Silu => crate::activations::silu_ps(pre),
    }
}

/// Épilogue AVX-512 : `c[i,j] = act(c[i,j] + bias[j])`, 16 voies + reste masqué.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn bias_act_epilogue_avx512(
    c: &mut [f32],
    m: usize,
    n: usize,
    bias: &[f32],
    act: Activation,
) {
    use core::arch::x86_64::*;
    let bias_ptr = bias.as_ptr();
    for i in 0..m
    {
        let row = c.as_mut_ptr().add(i * n);
        let mut j = 0;
        while j + 16 <= n
        {
            let pre = _mm512_add_ps(
                _mm512_loadu_ps(row.add(j)),
                _mm512_loadu_ps(bias_ptr.add(j)),
            );
            _mm512_storeu_ps(row.add(j), apply_act_ps(pre, act));
            j += 16;
        }
        let rem = n - j;
        if rem > 0
        {
            let mask = (1u16 << rem) - 1;
            let pre = _mm512_add_ps(
                _mm512_maskz_loadu_ps(mask, row.add(j)),
                _mm512_maskz_loadu_ps(mask, bias_ptr.add(j)),
            );
            _mm512_mask_storeu_ps(row.add(j), mask, apply_act_ps(pre, act));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_act_matches_naive() {
        // k=300 et k=600 > KC(256) : depuis la levée de la contrainte, ils
        // passent par le GEMM tuilé multi-bloc-K + épilogue (plus de repli
        // scalaire) — d'où leur présence explicite ici.
        let shapes = [
            (1usize, 1usize, 1usize),
            (2, 3, 5),
            (8, 16, 16),
            (7, 5, 19),
            (17, 33, 13),
            (16, 16, 32),
            (12, 300, 20),
            (9, 600, 40),
        ];
        for &(m, k, n) in &shapes
        {
            let a: Vec<f32> = (0..m * k).map(|t| (t as f32 * 0.017 - 0.5).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|t| (t as f32 * 0.011 + 0.3).cos()).collect();
            let bias: Vec<f32> = (0..n).map(|j| (j as f32) * 0.1 - 0.7).collect();
            for &alpha in &[1.0f32, -0.5, 2.0]
            {
                for &act in &[
                    Activation::Identity,
                    Activation::Relu,
                    Activation::Gelu,
                    Activation::Silu,
                ]
                {
                    // Référence naïve indépendante (activation scalaire).
                    let mut want = vec![0.0f32; m * n];
                    for i in 0..m
                    {
                        for j in 0..n
                        {
                            let mut acc = 0.0f32;
                            for p in 0..k
                            {
                                acc += a[i * k + p] * b[p * n + j];
                            }
                            want[i * n + j] = act.apply_scalar(alpha * acc + bias[j]);
                        }
                    }

                    let mut got = vec![123.0f32; m * n]; // sortie fraîche (écrasée)
                    sgemm_bias_act(
                        alpha,
                        MatrixView::new(&a, m, k),
                        MatrixView::new(&b, k, n),
                        &bias,
                        act,
                        MatrixViewMut::new(&mut got, m, n),
                    );
                    for t in 0..m * n
                    {
                        let tol = 1e-3 * (1.0 + want[t].abs());
                        assert!(
                            (got[t] - want[t]).abs() <= tol,
                            "shape {m}x{k}x{n} a={alpha} act={act:?} t={t}: {} vs {}",
                            got[t],
                            want[t]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bias_act_relu_zeroes_negatives() {
        // A·B = 0 partout (B = 0) → out = relu(bias) : négatifs coupés.
        let (m, k, n) = (3, 4, 5);
        let a = vec![1.0f32; m * k];
        let b = vec![0.0f32; k * n];
        let bias: Vec<f32> = (0..n).map(|j| j as f32 - 2.0).collect(); // -2,-1,0,1,2
        let mut got = vec![0.0f32; m * n];
        sgemm_bias_act(
            1.0,
            MatrixView::new(&a, m, k),
            MatrixView::new(&b, k, n),
            &bias,
            Activation::Relu,
            MatrixViewMut::new(&mut got, m, n),
        );
        for i in 0..m
        {
            assert_eq!(&got[i * n..i * n + n], &[0.0, 0.0, 0.0, 1.0, 2.0]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_ref(
        alpha: f32,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        beta: f32,
        c0: &[f32],
    ) -> Vec<f32> {
        let mut c = c0.to_vec();
        ScalarBackend.sgemm_f32(
            alpha,
            MatrixView::new(a, m, k),
            MatrixView::new(b, k, n),
            beta,
            MatrixViewMut::new(&mut c, m, n),
        );
        c
    }

    #[test]
    fn tiled_matches_scalar_small_and_edge_shapes() {
        let shapes = [
            (1usize, 1usize, 1usize),
            (2, 3, 2),
            (8, 8, 16),
            (7, 5, 9),
            (9, 17, 13),
            (16, 16, 16),
            (17, 31, 19),
            (33, 40, 15),
            (8, 256, 16),
        ];
        let alphas = [1.0f32, -0.5, 2.0];
        let betas = [0.0f32, 1.0, -0.75];
        for &(m, k, n) in &shapes
        {
            let a: Vec<f32> = (0..m * k).map(|t| (t as f32 * 0.013 - 0.4).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|t| (t as f32 * 0.021 + 0.2).cos()).collect();
            let c0: Vec<f32> = (0..m * n).map(|t| (t as f32) * 0.03 - 0.5).collect();
            for &alpha in &alphas
            {
                for &beta in &betas
                {
                    let want = scalar_ref(alpha, &a, m, k, &b, n, beta, &c0);
                    let mut got = c0.clone();
                    sgemm_tiled(
                        alpha,
                        MatrixView::new(&a, m, k),
                        MatrixView::new(&b, k, n),
                        beta,
                        MatrixViewMut::new(&mut got, m, n),
                    );
                    for t in 0..m * n
                    {
                        let tol = 1e-3 * (1.0 + want[t].abs());
                        assert!(
                            (got[t] - want[t]).abs() <= tol,
                            "shape {m}x{k}x{n} a={alpha} b={beta} t={t}: {} vs {}",
                            got[t],
                            want[t]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tiled_matches_scalar_large_multiblock() {
        // Traverse plusieurs blocs MC/NC/KC et des tuiles registre pleines.
        let (m, k, n) = (300, 300, 300);
        let a: Vec<f32> = (0..m * k).map(|t| ((t % 97) as f32) * 0.01 - 0.3).collect();
        let b: Vec<f32> = (0..k * n).map(|t| ((t % 89) as f32) * 0.02 - 0.5).collect();
        let c0: Vec<f32> = (0..m * n).map(|t| ((t % 13) as f32) * 0.1).collect();
        let want = scalar_ref(0.75, &a, m, k, &b, n, -0.25, &c0);
        let mut got = c0.clone();
        sgemm_tiled(
            0.75,
            MatrixView::new(&a, m, k),
            MatrixView::new(&b, k, n),
            -0.25,
            MatrixViewMut::new(&mut got, m, n),
        );
        for t in 0..m * n
        {
            let tol = 1e-2 * (1.0 + want[t].abs());
            assert!(
                (got[t] - want[t]).abs() <= tol,
                "t={t}: {} vs {}",
                got[t],
                want[t]
            );
        }
    }

    #[test]
    fn parallel_matches_single_thread() {
        for &(m, k, n) in &[(1usize, 1, 1), (5, 7, 9), (64, 48, 40), (130, 200, 70)]
        {
            let a: Vec<f32> = (0..m * k)
                .map(|t| ((t % 91) as f32) * 0.011 - 0.4)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|t| ((t % 83) as f32) * 0.017 - 0.6)
                .collect();
            let c0: Vec<f32> = (0..m * n).map(|t| ((t % 11) as f32) * 0.2 - 1.0).collect();

            let mut single = c0.clone();
            sgemm_tiled(
                1.25,
                MatrixView::new(&a, m, k),
                MatrixView::new(&b, k, n),
                -0.5,
                MatrixViewMut::new(&mut single, m, n),
            );

            for threads in [2usize, 3, 8]
            {
                let mut par = c0.clone();
                sgemm_parallel(1.25, &a, m, k, &b, n, -0.5, &mut par, threads);
                for t in 0..m * n
                {
                    assert!(
                        (par[t] - single[t]).abs() <= 1e-4 * (1.0 + single[t].abs()),
                        "m={m} k={k} n={n} threads={threads} t={t}: {} vs {}",
                        par[t],
                        single[t]
                    );
                }
            }
        }
    }

    /// Bench informel (ignoré par défaut) : compare scalaire naïf / tuilé packé
    /// mono-thread / tuilé packé multi-thread sur une grande matrice.
    /// `cargo test -p scirust-simd --release -- --ignored --nocapture gemm_bench`
    #[test]
    #[ignore = "bench manuel"]
    fn gemm_bench() {
        use std::time::Instant;
        let (m, k, n) = (1024usize, 1024, 1024);
        let a: Vec<f32> = (0..m * k)
            .map(|t| ((t % 251) as f32) * 0.004 - 0.5)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|t| ((t % 241) as f32) * 0.004 - 0.5)
            .collect();
        let c0 = vec![0.0f32; m * n];
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        let t = Instant::now();
        let mut cs = c0.clone();
        ScalarBackend.sgemm_f32(
            1.0,
            MatrixView::new(&a, m, k),
            MatrixView::new(&b, k, n),
            0.0,
            MatrixViewMut::new(&mut cs, m, n),
        );
        let dt = t.elapsed().as_secs_f64();
        println!(
            "scalaire naïf   : {:8.1} ms  ({:6.2} GFLOP/s)",
            dt * 1e3,
            flops / dt / 1e9
        );

        let t = Instant::now();
        let mut c1 = c0.clone();
        sgemm_tiled(
            1.0,
            MatrixView::new(&a, m, k),
            MatrixView::new(&b, k, n),
            0.0,
            MatrixViewMut::new(&mut c1, m, n),
        );
        let dt = t.elapsed().as_secs_f64();
        println!(
            "tuilé 1 thread  : {:8.1} ms  ({:6.2} GFLOP/s)",
            dt * 1e3,
            flops / dt / 1e9
        );

        let nt = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(4);
        let t = Instant::now();
        let mut cp = c0.clone();
        sgemm_parallel(1.0, &a, m, k, &b, n, 0.0, &mut cp, nt);
        let dt = t.elapsed().as_secs_f64();
        println!(
            "tuilé {nt} threads : {:8.1} ms  ({:6.2} GFLOP/s)",
            dt * 1e3,
            flops / dt / 1e9
        );

        // Sanity : les trois résultats coïncident.
        for t in (0..m * n).step_by(7919)
        {
            assert!((c1[t] - cs[t]).abs() <= 1e-2 * (1.0 + cs[t].abs()));
            assert!((cp[t] - cs[t]).abs() <= 1e-2 * (1.0 + cs[t].abs()));
        }
    }

    #[test]
    fn tiled_known_value() {
        // A(2x3)·B(3x2) = [[58,64],[139,154]].
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [1.0f32, 1.0, 1.0, 1.0];
        sgemm_tiled(
            2.0,
            MatrixView::new(&a, 2, 3),
            MatrixView::new(&b, 3, 2),
            3.0,
            MatrixViewMut::new(&mut c, 2, 2),
        );
        assert_eq!(c, [119.0, 131.0, 281.0, 311.0]);
    }

    // ---- DGEMM (f64) ----

    #[allow(clippy::too_many_arguments)]
    fn dgemm_naive_ref(
        alpha: f64,
        a: &[f64],
        m: usize,
        k: usize,
        b: &[f64],
        n: usize,
        beta: f64,
        c0: &[f64],
    ) -> Vec<f64> {
        // Référence indépendante (ne passe PAS par dgemm_scalar) pour ne rien
        // partager avec le code testé.
        let mut c = c0.to_vec();
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f64;
                for p in 0..k
                {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = alpha * acc + beta * c[i * n + j];
            }
        }
        c
    }

    #[test]
    fn dgemm_matches_naive_shapes_and_edges() {
        let shapes = [
            (1usize, 1usize, 1usize),
            (2, 3, 2),
            (8, 8, 8),
            (7, 5, 9),
            (9, 17, 13),
            (8, 256, 8),
            (17, 31, 19),
            (33, 40, 15),
        ];
        let alphas = [1.0f64, -0.5, 2.0];
        let betas = [0.0f64, 1.0, -0.75];
        for &(m, k, n) in &shapes
        {
            let a: Vec<f64> = (0..m * k).map(|t| (t as f64 * 0.013 - 0.4).sin()).collect();
            let b: Vec<f64> = (0..k * n).map(|t| (t as f64 * 0.021 + 0.2).cos()).collect();
            let c0: Vec<f64> = (0..m * n).map(|t| (t as f64) * 0.03 - 0.5).collect();
            for &alpha in &alphas
            {
                for &beta in &betas
                {
                    let want = dgemm_naive_ref(alpha, &a, m, k, &b, n, beta, &c0);
                    let mut got = c0.clone();
                    dgemm_tiled(
                        alpha,
                        MatrixView::new(&a, m, k),
                        MatrixView::new(&b, k, n),
                        beta,
                        MatrixViewMut::new(&mut got, m, n),
                    );
                    for t in 0..m * n
                    {
                        let tol = 1e-10 * (1.0 + want[t].abs());
                        assert!(
                            (got[t] - want[t]).abs() <= tol,
                            "shape {m}x{k}x{n} a={alpha} b={beta} t={t}: {} vs {}",
                            got[t],
                            want[t]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dgemm_large_multiblock_and_parallel() {
        let (m, k, n) = (200, 260, 180);
        let a: Vec<f64> = (0..m * k).map(|t| ((t % 97) as f64) * 0.01 - 0.3).collect();
        let b: Vec<f64> = (0..k * n).map(|t| ((t % 89) as f64) * 0.02 - 0.5).collect();
        let c0: Vec<f64> = (0..m * n).map(|t| ((t % 13) as f64) * 0.1).collect();
        let want = dgemm_naive_ref(0.75, &a, m, k, &b, n, -0.25, &c0);

        let mut single = c0.clone();
        dgemm_tiled(
            0.75,
            MatrixView::new(&a, m, k),
            MatrixView::new(&b, k, n),
            -0.25,
            MatrixViewMut::new(&mut single, m, n),
        );
        for threads in [1usize, 2, 4, 8]
        {
            let mut got = c0.clone();
            dgemm_parallel(0.75, &a, m, k, &b, n, -0.25, &mut got, threads);
            for t in 0..m * n
            {
                let tol = 1e-9 * (1.0 + want[t].abs());
                assert!(
                    (got[t] - want[t]).abs() <= tol,
                    "threads={threads} t={t}: {} vs {}",
                    got[t],
                    want[t]
                );
                assert!((got[t] - single[t]).abs() <= tol);
            }
        }
    }

    /// `cargo test -p scirust-simd --release -- --ignored --nocapture dgemm_bench`
    #[test]
    #[ignore = "bench manuel"]
    fn dgemm_bench() {
        use std::time::Instant;
        let (m, k, n) = (1024usize, 1024, 1024);
        let a: Vec<f64> = (0..m * k)
            .map(|t| ((t % 251) as f64) * 0.004 - 0.5)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|t| ((t % 241) as f64) * 0.004 - 0.5)
            .collect();
        let c0 = vec![0.0f64; m * n];
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        let t = Instant::now();
        let mut c1 = c0.clone();
        dgemm_tiled(
            1.0,
            MatrixView::new(&a, m, k),
            MatrixView::new(&b, k, n),
            0.0,
            MatrixViewMut::new(&mut c1, m, n),
        );
        let dt = t.elapsed().as_secs_f64();
        println!(
            "DGEMM tuilé 1 thread  : {:8.1} ms  ({:6.2} GFLOP/s)",
            dt * 1e3,
            flops / dt / 1e9
        );

        let nt = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(4);
        let t = Instant::now();
        let mut cp = c0.clone();
        dgemm_parallel(1.0, &a, m, k, &b, n, 0.0, &mut cp, nt);
        let dt = t.elapsed().as_secs_f64();
        println!(
            "DGEMM tuilé {nt} threads : {:8.1} ms  ({:6.2} GFLOP/s)",
            dt * 1e3,
            flops / dt / 1e9
        );
    }

    #[test]
    fn dgemm_known_value() {
        // A(2x3)·B(3x2) = [[58,64],[139,154]].
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [1.0f64, 1.0, 1.0, 1.0];
        dgemm_tiled(
            2.0,
            MatrixView::new(&a, 2, 3),
            MatrixView::new(&b, 3, 2),
            3.0,
            MatrixViewMut::new(&mut c, 2, 2),
        );
        assert_eq!(c, [119.0, 131.0, 281.0, 311.0]);
    }
}
