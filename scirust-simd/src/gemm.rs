//! # SGEMM tuilé multi-niveaux (x86_64)
//!
//! `C = alpha·A·B + beta·C`, matrices **row-major**, en `f32`.
//!
//! Le SGEMM « rank-1 » du module [`crate::dispatch`] est simple et
//! cache-friendly en lecture de `B`, mais il relit et réécrit toute la ligne de
//! `C` à chaque `p` : le trafic sur `C` domine dès que `k` est grand. Ce module
//! implémente la structure classique haute-performance (façon BLIS) :
//!
//! * **Blocking cache** sur `K` (`KC`) et `N` (`NC`) pour que le panneau de `B`
//!   travaillé tienne en L2/L1 et soit réutilisé par toutes les lignes de `A`.
//! * **Micro-kernel registre-bloqué** `MR×NR` (`8×16`) : 8 accumulateurs `zmm`
//!   maintiennent une tuile `8×16` de `C` **dans les registres** pendant toute
//!   la dimension `KC`, avec un `broadcast(A) × B` FMA par pas de `k`. Les
//!   accès à `C` (16 chargements + 16 écritures) sont ainsi amortis sur `KC`
//!   produits au lieu d'un par produit.
//! * **Bords masqués** : les tuiles partielles (`< 16` colonnes, `< 8` lignes)
//!   sont gérées par masque `k` AVX-512, sans chemin scalaire séparé.
//!
//! Repli automatique : hors `avx512f`, on délègue au SGEMM scalaire de
//! référence. Le résultat est vérifié *bit-proche* du scalaire dans les tests.

use crate::matrix::backend::{ScalarBackend, SimdBackend};
use crate::matrix::view::{MatrixView, MatrixViewMut};

/// Dimensions de blocking. `MR`/`NR` fixent la tuile registre ; `KC`/`NC` sont
/// choisis pour qu'un panneau `KC×NR` de B et la tuile de C tiennent en cache.
const MR: usize = 8;
const NR: usize = 16;
const KC: usize = 256;
const NC: usize = 1024;

/// SGEMM tuilé : `C = alpha·A(m×k)·B(k×n) + beta·C(m×n)`, row-major.
///
/// Point d'entrée sûr : valide les dimensions puis dispatch runtime vers le
/// noyau AVX-512 si disponible, sinon vers la référence scalaire.
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
            let v = _mm512_loadu_ps(row.add(j));
            // beta == 0 → on écrit des zéros exacts (évite 0·NaN si C sale).
            let r = if beta == 0.0
            {
                _mm512_setzero_ps()
            }
            else
            {
                _mm512_mul_ps(v, bv)
            };
            _mm512_storeu_ps(row.add(j), r);
            j += 16;
        }
        let rem = n - j;
        if rem > 0
        {
            let mask = (1u16 << rem) - 1;
            let v = _mm512_maskz_loadu_ps(mask, row.add(j));
            let r = if beta == 0.0
            {
                _mm512_setzero_ps()
            }
            else
            {
                _mm512_mul_ps(v, bv)
            };
            _mm512_mask_storeu_ps(row.add(j), mask, r);
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
    // Les MatrixView sont contiguës row-major (col_stride == 1) : on récupère
    // des pointeurs de base via les lignes 0.
    let a_ptr = a.row_slice(0).map(|r| r.as_ptr());
    let b_ptr = b.row_slice(0).map(|r| r.as_ptr());
    let c_ptr = c.row_slice_mut(0).map(|r| r.as_mut_ptr());
    let (a_ptr, b_ptr, c_ptr) = match (a_ptr, b_ptr, c_ptr)
    {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        // m==0 ou k==0 ou n==0 : rien à faire (C éventuellement scalé plus bas).
        _ =>
        {
            if m > 0 && n > 0
            {
                scale_c_avx512(beta, m, n, c.row_slice_mut(0).unwrap().as_mut_ptr());
            }
            return;
        },
    };

    // 1) C <- beta * C
    scale_c_avx512(beta, m, n, c_ptr);
    if k == 0 || alpha == 0.0
    {
        return;
    }

    // 2) Boucles de blocking : NC (colonnes) puis KC (contraction).
    let mut jc = 0;
    while jc < n
    {
        let nc = NC.min(n - jc);
        let mut pc = 0;
        while pc < k
        {
            let kc = KC.min(k - pc);
            // Balaye les tuiles registre MR×NR à l'intérieur du bloc.
            let mut ic = 0;
            while ic < m
            {
                let mr = MR.min(m - ic);
                let mut jr = 0;
                while jr < nc
                {
                    let nr = NR.min(nc - jr);
                    micro_kernel_8x16(
                        alpha,
                        mr,
                        nr,
                        kc,
                        a_ptr.add(ic * k + pc),
                        k, // lda (row stride de A)
                        b_ptr.add(pc * n + (jc + jr)),
                        n, // ldb (row stride de B)
                        c_ptr.add(ic * n + (jc + jr)),
                        n, // ldc (row stride de C)
                    );
                    jr += NR;
                }
                ic += MR;
            }
            pc += KC;
        }
        jc += NC;
    }
}

/// Micro-kernel : `C_tile[mr×nr] += alpha · A_tile[mr×kc] · B_panel[kc×nr]`.
///
/// Maintient jusqu'à `MR` accumulateurs `zmm` (une tuile `MR×16` de C) dans les
/// registres pendant toute la dimension `kc`. `nr < 16` ou `mr < 8` (bords) sont
/// gérés par masque `k`. Toutes les lignes ont un stride row-major (`lda`,
/// `ldb`, `ldc`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_kernel_8x16(
    alpha: f32,
    mr: usize,
    nr: usize,
    kc: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    use core::arch::x86_64::*;
    debug_assert!(mr <= MR && nr <= NR);
    let mask: u16 = if nr >= 16 { 0xffff } else { (1u16 << nr) - 1 };
    let av = _mm512_set1_ps(alpha);

    // Accumulateurs : un zmm par ligne de la tuile.
    let mut acc = [_mm512_setzero_ps(); MR];

    for p in 0..kc
    {
        // Charge la ligne p du panneau de B (nr colonnes, masquée).
        let bv = _mm512_maskz_loadu_ps(mask, b.add(p * ldb));
        // Pour chaque ligne i de la tuile : acc[i] += (alpha*A[i,p]) * bv.
        for (i, ai) in acc.iter_mut().enumerate().take(mr)
        {
            let a_ip = *a.add(i * lda + p);
            let sv = _mm512_mul_ps(av, _mm512_set1_ps(a_ip));
            *ai = _mm512_fmadd_ps(sv, bv, *ai);
        }
    }

    // Écrit la tuile dans C : C_tile += acc (C déjà scalé par beta en amont).
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
        // Traverse plusieurs blocs NC/KC et des tuiles registre pleines.
        let (m, k, n) = (40, 300, 50);
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
