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

#![allow(clippy::missing_safety_doc)]

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
}
