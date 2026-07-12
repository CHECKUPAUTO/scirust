//! # ARM SME — accélérateur matriciel scalable (aarch64)
//!
//! *Scalable Matrix Extension* : le pendant ARM d'Intel AMX. Un accumulateur
//! matriciel **ZA** (`SVL×SVL` bits, largeur scalable comme SVE) et un **mode
//! streaming SVE**. L'opération de base est l'**outer-product accumulate**
//! `FMOPA` : `ZA += vᵀ · u` (produit externe d'un vecteur ligne et d'un vecteur
//! colonne), qui construit un GEMM par accumulation de rangs-1.
//!
//! ## État de l'écosystème (drop-in en attente)
//!
//! Contrairement à AMX (intrinsèques `_tile_*` disponibles, cf. [`crate::amx`])
//! et à SVE (intrinsèques `sv*` désormais présentes, cf. [`crate::sve`]), les
//! **intrinsèques SME de `core::arch::aarch64` n'existent pas encore** dans la
//! toolchain (ni `svcntsw`, ni les accès ZA `svmopa`/`svst1_hor`…), et le
//! *target-feature* `sme` reste `aarch64_unstable_target_feature`. On ne peut
//! donc pas encore écrire le noyau ZA natif comme on l'a fait pour AMX.
//!
//! Ce module fournit donc, dans le même esprit que l'amorce SVE historique :
//! * [`sme_available`] — détection runtime de la présence matérielle de SME ;
//! * [`matmul_f32_rank1`] — la **référence** de l'opération que `FMOPA`
//!   accélérera : GEMM `f32` par accumulation d'outer-products (rang-1), qui est
//!   *exactement* le motif ZA. C'est le repli portable et l'oracle de correction
//!   du futur noyau SME natif.
//!
//! Dès que `core::arch::aarch64` exposera les intrinsèques ZA/streaming, le
//! noyau natif viendra se brancher derrière [`sme_available`], validé contre
//! [`matmul_f32_rank1`] — le même schéma que le port SVE (sonde → kernels réels).

/// `true` si le cœur courant expose **SME** (Scalable Matrix Extension).
///
/// Détection runtime via `is_aarch64_feature_detected!` ; sûr sur tout cœur
/// aarch64 (renvoie `false` sans SME).
pub fn sme_available() -> bool {
    std::arch::is_aarch64_feature_detected!("sme")
}

/// GEMM `f32` `C[m×n] = A[m×k]·B[k×n]` (row-major) construit par **accumulation
/// d'outer-products** (rang-1), le motif exact de l'accumulateur ZA de SME :
/// pour chaque `p`, `C += A[:,p] ⊗ B[p,:]`.
///
/// C'est la **référence** de l'opération que `FMOPA` accélérera, et le repli
/// portable tant que les intrinsèques SME ne sont pas disponibles. L'ordre des
/// boucles (`p` externe) reproduit la sémantique ZA plutôt que le produit
/// scalaire classique — numériquement équivalent à l'addition près.
pub fn matmul_f32_rank1(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul_f32_rank1: A shape mismatch");
    assert_eq!(b.len(), k * n, "matmul_f32_rank1: B shape mismatch");
    let mut c = vec![0f32; m * n];
    // ZA-style : accumulation de produits externes A[:,p] ⊗ B[p,:].
    for p in 0..k
    {
        for i in 0..m
        {
            let a_ip = a[i * k + p];
            if a_ip == 0.0
            {
                continue;
            }
            let crow = &mut c[i * n..i * n + n];
            let brow = &b[p * n..p * n + n];
            for (cj, &bj) in crow.iter_mut().zip(brow)
            {
                *cj += a_ip * bj;
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sme_available_is_callable() {
        // Ne présume rien du matériel : l'appel doit juste être sûr et renvoyer
        // un booléen cohérent (le noyau qemu par défaut n'expose pas SME).
        let _ = sme_available();
    }

    #[test]
    fn rank1_matmul_matches_dot_reference() {
        // L'accumulation d'outer-products doit égaler le produit matriciel
        // classique (à tolérance flottante), sur formes variées.
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (3, 5, 4), (8, 13, 7), (16, 20, 9)]
        {
            let a: Vec<f32> = (0..m * k).map(|t| (t as f32 * 0.017).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|t| (t as f32 * 0.011).cos()).collect();
            let got = matmul_f32_rank1(&a, &b, m, k, n);
            for i in 0..m
            {
                for j in 0..n
                {
                    let mut want = 0f32;
                    for p in 0..k
                    {
                        want += a[i * k + p] * b[p * n + j];
                    }
                    assert!(
                        (got[i * n + j] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "m={m} k={k} n={n} i={i} j={j}"
                    );
                }
            }
        }
    }
}
