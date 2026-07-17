//! # ARM SVE — kernels **scalables** (Pilier 4, aarch64)
//!
//! *Scalable Vector Extension* : SIMD à **longueur vectorielle inconnue à la
//! compilation** (128 à 2048 bits par pas de 128), déterminée par le matériel.
//! Le même binaire tourne à pleine largeur sur A64FX (512 b), Graviton 3
//! (256 b), Neoverse V2… sans recompilation ni gestion de bord spécifique.
//!
//! ## Le modèle scalable en pratique
//!
//! Chaque kernel ici est écrit **sans jamais nommer la largeur** :
//!
//! * la boucle avance de `svcntw()` éléments `f32` (= voies par vecteur, valeur
//!   runtime) ;
//! * le **prédicat** `svwhilelt_b32_u64(i, n)` active exactement les voies
//!   `i..min(i+VL, n)` — le dernier pas partiel est géré par le prédicat, **sans
//!   épilogue scalaire** ;
//! * chargements/écritures prédiqués (`svld1`/`svst1`) : les voies inactives ne
//!   touchent pas la mémoire et se lisent comme `0`.
//!
//! Contrairement à la précédente *sonde* (qui lisait seulement la longueur via
//! `rdvl`), ce module fournit de **vrais kernels de calcul** (`saxpy`, `sdot`,
//! `sscal`) validés à l'exécution sous `qemu-aarch64`.
//!
//! ## Safety
//!
//! Les fonctions `#[target_feature(enable = "sve")]` ne sont appelées qu'après
//! `is_aarch64_feature_detected!("sve")`. Les accès mémoire sont prédiqués et
//! bornés par `n` (le prédicat masque tout dépassement), donc aucune lecture ou
//! écriture hors des slices fournies.

use crate::matrix::backend::{ScalarBackend, SimdBackend};
use crate::matrix::view::{MatrixView, MatrixViewMut};

/// Longueur vectorielle SVE en éléments de type `T`, ou `0` si SVE est absent.
///
/// Lit la longueur architecturale avec `rdvl` (asm inline *stable*).
/// L'instruction n'est exécutée qu'après détection runtime : sûr sur tout cœur
/// aarch64.
pub fn sve_vector_length_elements<T>() -> usize {
    if !std::arch::is_aarch64_feature_detected!("sve")
    {
        return 0;
    }
    let vl_bytes: u64;
    // SAFETY: rdvl n'est atteint que si le CPU rapporte le support SVE.
    unsafe {
        core::arch::asm!(
            ".arch_extension sve",
            "rdvl {0}, #1",
            out(reg) vl_bytes,
            options(nomem, nostack, preserves_flags)
        );
    }
    vl_bytes as usize / core::mem::size_of::<T>()
}

/// AXPY scalable : `y[i] += alpha * x[i]`. Chemin SVE si disponible, repli
/// scalaire sinon (référence de correction).
pub fn saxpy_f32_sve(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "saxpy_f32_sve: length mismatch");
    if std::arch::is_aarch64_feature_detected!("sve")
    {
        // SAFETY: gated by the runtime detection just above.
        unsafe { saxpy_f32_sve_impl(alpha, x, y) };
        return;
    }
    for (yi, &xi) in y.iter_mut().zip(x)
    {
        *yi += alpha * xi;
    }
}

/// Produit scalaire scalable : `sum(x[i] * y[i])`. Chemin SVE si disponible,
/// repli scalaire sinon.
pub fn sdot_f32_sve(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "sdot_f32_sve: length mismatch");
    if std::arch::is_aarch64_feature_detected!("sve")
    {
        // SAFETY: gated by the runtime detection just above.
        return unsafe { sdot_f32_sve_impl(x, y) };
    }
    x.iter().zip(y).map(|(&a, &b)| a * b).sum()
}

/// Mise à l'échelle scalable : `x[i] *= alpha`. Chemin SVE si disponible, repli
/// scalaire sinon.
pub fn sscal_f32_sve(alpha: f32, x: &mut [f32]) {
    if std::arch::is_aarch64_feature_detected!("sve")
    {
        // SAFETY: gated by the runtime detection just above.
        unsafe { sscal_f32_sve_impl(alpha, x) };
        return;
    }
    for xi in x.iter_mut()
    {
        *xi *= alpha;
    }
}

/// Cœur SVE de [`saxpy_f32_sve`] : boucle prédiquée, `_x` (don't-care) car le
/// store prédiqué n'écrit que les voies actives — aucune fuite de voie inactive.
/// # Safety
/// Caller must ensure SVE is available
/// (`is_aarch64_feature_detected!("sve")`). `x.len() == y.len()` is
/// required ([`saxpy_f32_sve`] asserts this before dispatching here).
/// Bounds are otherwise self-contained: `svwhilelt_b32_u64` builds a
/// predicate that masks every lane at or past `n`, so `svld1_f32`/
/// `svst1_f32` never touch memory past `x`/`y`'s end regardless of the
/// hardware vector length.
#[target_feature(enable = "sve")]
unsafe fn saxpy_f32_sve_impl(alpha: f32, x: &[f32], y: &mut [f32]) {
    use core::arch::aarch64::*;
    let n = x.len();
    let step = svcntw() as usize;
    let alpha_v = svdup_n_f32(alpha);
    let xp = x.as_ptr();
    let yp = y.as_mut_ptr();
    let mut i = 0usize;
    while i < n
    {
        let pg = svwhilelt_b32_u64(i as u64, n as u64);
        let vx = svld1_f32(pg, xp.add(i));
        let vy = svld1_f32(pg, yp.add(i));
        // svmla(acc, a, b) = acc + a*b ⇒ vy + vx*alpha = alpha*x + y.
        let r = svmla_f32_x(pg, vy, vx, alpha_v);
        svst1_f32(pg, yp.add(i), r);
        i += step;
    }
}

/// Cœur SVE de [`sdot_f32_sve`]. **`_m` (merge)** pour l'accumulation : les voies
/// inactives du dernier pas partiel **conservent** leur somme partielle (un `_x`
/// les rendrait indéfinies et corromprait le `svaddv` final sur toutes les
/// voies).
/// # Safety
/// Same contract as [`saxpy_f32_sve_impl`]: caller must ensure SVE is
/// available, and `x.len() == y.len()` ([`sdot_f32_sve`] asserts this
/// before dispatching here). Predicated load bounds are self-contained as
/// described there.
#[target_feature(enable = "sve")]
unsafe fn sdot_f32_sve_impl(x: &[f32], y: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let n = x.len();
    let step = svcntw() as usize;
    let mut acc = svdup_n_f32(0.0);
    let xp = x.as_ptr();
    let yp = y.as_ptr();
    let mut i = 0usize;
    while i < n
    {
        let pg = svwhilelt_b32_u64(i as u64, n as u64);
        let vx = svld1_f32(pg, xp.add(i));
        let vy = svld1_f32(pg, yp.add(i));
        acc = svmla_f32_m(pg, acc, vx, vy);
        i += step;
    }
    // Toutes les voies portent une somme partielle valide → réduction complète.
    svaddv_f32(svptrue_b32(), acc)
}

/// Cœur SVE de [`sscal_f32_sve`] : `_x` suffit (store prédiqué).
/// # Safety
/// Caller must ensure SVE is available. Bounds are self-contained
/// (predicated load/store, as in [`saxpy_f32_sve_impl`]) — no length
/// precondition beyond `x` itself since there's only one slice involved.
#[target_feature(enable = "sve")]
unsafe fn sscal_f32_sve_impl(alpha: f32, x: &mut [f32]) {
    use core::arch::aarch64::*;
    let n = x.len();
    let step = svcntw() as usize;
    let alpha_v = svdup_n_f32(alpha);
    let xp = x.as_mut_ptr();
    let mut i = 0usize;
    while i < n
    {
        let pg = svwhilelt_b32_u64(i as u64, n as u64);
        let vx = svld1_f32(pg, xp.add(i));
        let r = svmul_f32_x(pg, vx, alpha_v);
        svst1_f32(pg, xp.add(i), r);
        i += step;
    }
}

// ===================================================================== //
//  SGEMM packé / register-blocked SVE (au-delà du rank-1)                 //
// ===================================================================== //

/// Lignes de la tuile registre SVE. **Constante de compilation** : les types SVE
/// sont *sizeless* (pas de `[svfloat32_t; N]`, pas de `Vec`, pas d'indexation
/// runtime), donc les `MR_SVE` accumulateurs sont des variables **nommées**
/// déroulées à la main. `MR_SVE + 2` registres `Z` vivants (accs + 1 vecteur B +
/// 1 broadcast A) ⇒ 10/32, large marge (comme les tuiles `8×…` x86/NEON).
const MR_SVE: usize = 8;

/// SGEMM **packé, register-blocked, scalable** (SVE) : `C = alpha·A·B + beta·C`,
/// row-major. Remplace la formulation rank-1 (`sscal`+`saxpy` par ligne) : ici
/// une tuile `MR_SVE × VL` de `C` est maintenue **dans les registres** sur toute
/// la dimension `K`, donc `C` n'est écrite **qu'une fois** (le trafic sur `C` est
/// amorti sur `K`, comme les noyaux AVX-512/NEON packés). `VL = svcntw()` est la
/// largeur vectorielle **runtime** ; les bords `n % VL` tombent par prédicat, les
/// bords `m % MR_SVE` par zéro-padding de `A`. Repli scalaire hors SVE.
pub fn sgemm_f32_sve(
    alpha: f32,
    a: MatrixView<f32>,
    b: MatrixView<f32>,
    beta: f32,
    c: MatrixViewMut<f32>,
) {
    if std::arch::is_aarch64_feature_detected!("sve")
    {
        // SAFETY: gated by the runtime detection just above.
        unsafe { sgemm_f32_sve_packed(alpha, a, b, beta, c) };
        return;
    }
    ScalarBackend.sgemm_f32(alpha, a, b, beta, c);
}

/// `C *= beta` prédiqué (cas `k == 0` ou `alpha == 0`, où `C = beta·C`).
/// `beta == 0` écrit des zéros sans **lire** `C` (évite un `0·NaN` si `C` est
/// non initialisé).
/// # Safety
/// Caller must ensure SVE is available. `c` must point to a valid,
/// exclusively-borrowed row-major `m×n` `f32` buffer (row stride `n`),
/// writable for the whole call. Column bounds are self-contained via the
/// predicate (`svwhilelt_b32_u64`), which masks every lane at or past `n`.
#[target_feature(enable = "sve")]
unsafe fn scale_c_sve(beta: f32, m: usize, n: usize, c: *mut f32) {
    use core::arch::aarch64::*;
    if beta == 1.0
    {
        return;
    }
    let vl = svcntw() as usize;
    let bv = svdup_n_f32(beta);
    for i in 0..m
    {
        let row = c.add(i * n);
        let mut j = 0;
        while j < n
        {
            let pg = svwhilelt_b32_u64(j as u64, n as u64);
            let r = if beta == 0.0
            {
                svdup_n_f32(0.0)
            }
            else
            {
                svmul_f32_x(pg, svld1_f32(pg, row.add(j)), bv)
            };
            svst1_f32(pg, row.add(j), r);
            j += vl;
        }
    }
}

/// Cœur packé. Pour chaque panneau de `MR_SVE` lignes : `A` est empaqueté une
/// fois (`p`-majeur, `alpha` fusionné, lignes `mr..MR_SVE` mises à zéro), puis
/// pour chaque bande de `VL` colonnes on maintient `MR_SVE` accumulateurs sur
/// tout `K` et on stocke la tuile (bord colonne par prédicat, `beta` fondu).
/// # Safety
/// Caller must ensure SVE is available. `a`/`b`/`c` are `MatrixView`s, so
/// their bounds are already validated by construction. Row bounds beyond
/// `MR_SVE` are handled by zero-padding `apack` (missing rows contribute
/// zero to every accumulator); column bounds are handled by the predicate
/// (`svwhilelt_b32_u64`), which masks every lane at or past `n`.
#[target_feature(enable = "sve")]
unsafe fn sgemm_f32_sve_packed(
    alpha: f32,
    a: MatrixView<f32>,
    b: MatrixView<f32>,
    beta: f32,
    mut c: MatrixViewMut<f32>,
) {
    use core::arch::aarch64::*;
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    if m == 0 || n == 0
    {
        return;
    }
    let c_ptr = c.row_slice_mut(0).expect("C base").as_mut_ptr();
    if k == 0 || alpha == 0.0
    {
        scale_c_sve(beta, m, n, c_ptr);
        return;
    }
    let a_ptr = a.row_slice(0).expect("A base").as_ptr();
    let b_ptr = b.row_slice(0).expect("B base").as_ptr();
    let vl = svcntw() as usize;

    // Tampon de packing de A (MR_SVE × k), réutilisé par panneau de lignes.
    let mut apack = vec![0.0f32; MR_SVE * k];

    let mut i0 = 0;
    while i0 < m
    {
        let mr = MR_SVE.min(m - i0);
        // Pack A[i0.., 0..k] : apack[p*MR_SVE + i] = alpha·A[i0+i][p], lignes
        // manquantes (i >= mr) à zéro (accumulateurs correspondants restent nuls).
        for p in 0..k
        {
            let base = p * MR_SVE;
            for i in 0..mr
            {
                apack[base + i] = alpha * *a_ptr.add((i0 + i) * k + p);
            }
            for slot in apack[base + mr..base + MR_SVE].iter_mut()
            {
                *slot = 0.0;
            }
        }

        let mut j0 = 0;
        while j0 < n
        {
            let pg = svwhilelt_b32_u64(j0 as u64, n as u64);
            let bbase = b_ptr.add(j0);
            let cbase = c_ptr.add(i0 * n + j0);

            // MR_SVE accumulateurs nommés (types SVE sizeless : pas d'array).
            let z = svdup_n_f32(0.0);
            let mut a0 = z;
            let mut a1 = z;
            let mut a2 = z;
            let mut a3 = z;
            let mut a4 = z;
            let mut a5 = z;
            let mut a6 = z;
            let mut a7 = z;

            for p in 0..k
            {
                // Une bande VL de la ligne p de B (voies inactives lues 0).
                let bv = svld1_f32(pg, bbase.add(p * n));
                let ap = apack.as_ptr().add(p * MR_SVE);
                // acc_i += bv · (alpha·A[i][p]) ; `_x` : aucune réduction inter-voies,
                // seules les voies actives sont stockées (svst1 prédiqué).
                a0 = svmla_f32_x(pg, a0, bv, svdup_n_f32(*ap.add(0)));
                a1 = svmla_f32_x(pg, a1, bv, svdup_n_f32(*ap.add(1)));
                a2 = svmla_f32_x(pg, a2, bv, svdup_n_f32(*ap.add(2)));
                a3 = svmla_f32_x(pg, a3, bv, svdup_n_f32(*ap.add(3)));
                a4 = svmla_f32_x(pg, a4, bv, svdup_n_f32(*ap.add(4)));
                a5 = svmla_f32_x(pg, a5, bv, svdup_n_f32(*ap.add(5)));
                a6 = svmla_f32_x(pg, a6, bv, svdup_n_f32(*ap.add(6)));
                a7 = svmla_f32_x(pg, a7, bv, svdup_n_f32(*ap.add(7)));
            }

            // Épilogue : C[i, bande] = acc_i + beta·C_old, seulement les `mr`
            // lignes valides. `beta == 0` : store direct (pas de lecture de C).
            let bv_beta = svdup_n_f32(beta);
            macro_rules! store_row {
                ($i:expr, $acc:expr) => {
                    if mr > $i
                    {
                        let cp = cbase.add($i * n);
                        let out = if beta == 0.0
                        {
                            $acc
                        }
                        else
                        {
                            svmla_f32_x(pg, $acc, svld1_f32(pg, cp), bv_beta)
                        };
                        svst1_f32(pg, cp, out);
                    }
                };
            }
            store_row!(0, a0);
            store_row!(1, a1);
            store_row!(2, a2);
            store_row!(3, a3);
            store_row!(4, a4);
            store_row!(5, a5);
            store_row!(6, a6);
            store_row!(7, a7);

            j0 += vl;
        }
        i0 += MR_SVE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_length_is_positive_multiple_of_four() {
        // Sur un cœur SVE, VL(f32) est un multiple de 4 (128 b mini) > 0 ; sinon 0.
        let vl = sve_vector_length_elements::<f32>();
        if std::arch::is_aarch64_feature_detected!("sve")
        {
            assert!(vl >= 4 && vl.is_multiple_of(4), "vl={vl}");
        }
        else
        {
            assert_eq!(vl, 0);
        }
    }

    #[test]
    fn saxpy_matches_scalar_all_lengths() {
        // Couvre plusieurs pas vectoriels + bords prédiqués, indépendamment de VL.
        for n in 0..=300usize
        {
            let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013 - 0.4).collect();
            let y0: Vec<f32> = (0..n).map(|i| (i as f32) * -0.021 + 0.7).collect();
            let mut got = y0.clone();
            saxpy_f32_sve(1.75, &x, &mut got);
            for i in 0..n
            {
                let want = y0[i] + 1.75 * x[i];
                assert!(
                    (got[i] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "n={n} i={i}"
                );
            }
        }
    }

    #[test]
    fn sdot_matches_scalar_all_lengths() {
        for n in 0..=300usize
        {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin()).collect();
            let y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011).cos()).collect();
            let got = sdot_f32_sve(&x, &y);
            let want: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
            assert!(
                (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
                "n={n}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn sscal_matches_scalar_all_lengths() {
        for n in 0..=300usize
        {
            let base: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 5.0).collect();
            let mut got = base.clone();
            sscal_f32_sve(-0.5, &mut got);
            for i in 0..n
            {
                assert!((got[i] - base[i] * -0.5).abs() <= 1e-4, "n={n} i={i}");
            }
        }
    }

    #[test]
    fn sgemm_packed_matches_scalar() {
        // Le GEMM packé SVE doit coïncider avec la référence scalaire (à l'ordre
        // de sommation près, tolérance GEMM du repo). Couvre : lignes partielles
        // (m % MR_SVE), bandes de colonnes partielles (n non multiple de VL, y
        // compris n < VL), plusieurs panneaux MR + bandes VL, et les bords
        // `k == 0` / `alpha == 0` (⇒ `C = beta·C`).
        let shapes = [
            (1usize, 1usize, 1usize),
            (3, 4, 2),
            (8, 8, 8),
            (7, 5, 9),
            (9, 17, 13),
            (16, 16, 16),
            (17, 31, 19),
            (33, 40, 15),
            (40, 24, 48),
            (20, 0, 10), // k == 0 : C = beta·C
        ];
        let alphas = [1.0f32, -0.5, 2.0, 0.0];
        let betas = [0.0f32, 1.0, -0.75];
        for &(m, k, n) in &shapes
        {
            let a: Vec<f32> = (0..m * k).map(|t| (t as f32 * 0.017 - 0.3).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|t| (t as f32 * 0.023 + 0.1).cos()).collect();
            let c0: Vec<f32> = (0..m * n).map(|t| (t as f32) * 0.05 - 0.5).collect();
            for &alpha in &alphas
            {
                for &beta in &betas
                {
                    // Référence scalaire indépendante.
                    let mut want = c0.clone();
                    for i in 0..m
                    {
                        for j in 0..n
                        {
                            let mut acc = 0.0f32;
                            for p in 0..k
                            {
                                acc += a[i * k + p] * b[p * n + j];
                            }
                            want[i * n + j] = alpha * acc + beta * want[i * n + j];
                        }
                    }
                    let mut got = c0.clone();
                    sgemm_f32_sve(
                        alpha,
                        MatrixView::new(&a, m, k),
                        MatrixView::new(&b, k, n),
                        beta,
                        MatrixViewMut::new(&mut got, m, n),
                    );
                    for t in 0..m * n
                    {
                        let tol = 1e-4 * (1.0 + want[t].abs());
                        assert!(
                            (got[t] - want[t]).abs() <= tol,
                            "m={m} k={k} n={n} a={alpha} b={beta} t={t}: {} vs {}",
                            got[t],
                            want[t]
                        );
                    }
                }
            }
        }
    }
}
