//! # SGEMM tuilé multi-niveaux (x86_64)
//!
//! `C = alpha·A·B + beta·C`, matrices **row-major**, en `f32`.
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

/// Dimensions de blocking. `MR`/`NR` fixent la tuile registre ; `MC`/`KC`/`NC`
/// dimensionnent les panneaux packés pour la hiérarchie de caches.
const MR: usize = 8;
const NR: usize = 16;
const KC: usize = 256;
const MC: usize = 256;
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
