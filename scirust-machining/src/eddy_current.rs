//! Contrôle non destructif (CND) par **courants de Foucault** — profondeur
//! standard de pénétration, déphasage en fonction de la profondeur et fréquence
//! d'inspection requise pour une profondeur cible.
//!
//! ```text
//! profondeur standard   δ = 1 / sqrt(π·f·μ·σ)
//! déphasage             θ = z / δ
//! fréquence pour z      f = 1 / (π·z²·μ·σ)
//! ```
//!
//! `f` fréquence d'excitation (Hz), `μ` perméabilité magnétique **absolue** du
//! matériau (H/m, soit `μ = μ_r·μ_0`), `σ` conductivité électrique (S/m), `δ`
//! profondeur standard de pénétration (m), `z` profondeur sous la surface (m),
//! `θ` déphasage du courant induit par rapport au courant de surface (rad).
//! À la profondeur standard `δ`, l'amplitude du courant induit est tombée à
//! `1/e ≈ 37 %` de sa valeur en surface et le déphasage vaut exactement 1 rad ;
//! le déphasage croît linéairement avec la profondeur exprimée en `δ`. La
//! fréquence requise pour une profondeur cible est l'inverse de la relation de
//! `δ`.
//!
//! **Convention** : SI cohérent (Hz, H/m, S/m, m, rad). **Limite honnête** :
//! matériau conducteur homogène, non magnétique ou de perméabilité linéaire,
//! demi-espace plan ; la perméabilité `μ` et la conductivité `σ` sont **fournies
//! par l'appelant** — aucune valeur matériau, ni `μ_0`, ni fréquence « par
//! défaut » n'est inventée ici. Modèle plan à onde plane : pas de prise en
//! compte de la courbure de la pièce, du diamètre de la sonde, du lift-off ni de
//! la saturation magnétique.

use core::f64::consts::PI;

/// Profondeur standard de pénétration `δ = 1 / sqrt(π·f·μ·σ)` (m).
///
/// Profondeur à laquelle l'amplitude du courant induit vaut `1/e ≈ 37 %` de
/// celle en surface.
///
/// Panique si `frequency <= 0`, `permeability <= 0` ou `conductivity <= 0`.
pub fn eddy_standard_depth_of_penetration(
    frequency: f64,
    permeability: f64,
    conductivity: f64,
) -> f64 {
    assert!(
        frequency > 0.0,
        "la fréquence doit être strictement positive"
    );
    assert!(
        permeability > 0.0,
        "la perméabilité doit être strictement positive"
    );
    assert!(
        conductivity > 0.0,
        "la conductivité doit être strictement positive"
    );
    1.0 / (PI * frequency * permeability * conductivity).sqrt()
}

/// Déphasage du courant induit `θ = z / δ` (rad).
///
/// Le déphasage par rapport au courant de surface croît linéairement avec la
/// profondeur `z` exprimée en profondeurs standard `δ` ; il vaut 1 rad à `z = δ`.
///
/// Panique si `depth < 0` ou `standard_depth <= 0`.
pub fn eddy_phase_lag(depth: f64, standard_depth: f64) -> f64 {
    assert!(depth >= 0.0, "la profondeur doit être positive");
    assert!(
        standard_depth > 0.0,
        "la profondeur standard doit être strictement positive"
    );
    depth / standard_depth
}

/// Fréquence d'inspection `f = 1 / (π·z²·μ·σ)` pour placer la profondeur
/// standard à `z` (Hz).
///
/// Inverse de [`eddy_standard_depth_of_penetration`] : renvoie la fréquence pour
/// laquelle `δ` est égale à la profondeur cible `target_depth`.
///
/// Panique si `target_depth <= 0`, `permeability <= 0` ou `conductivity <= 0`.
pub fn eddy_frequency_for_depth(target_depth: f64, permeability: f64, conductivity: f64) -> f64 {
    assert!(
        target_depth > 0.0,
        "la profondeur cible doit être strictement positive"
    );
    assert!(
        permeability > 0.0,
        "la perméabilité doit être strictement positive"
    );
    assert!(
        conductivity > 0.0,
        "la conductivité doit être strictement positive"
    );
    1.0 / (PI * target_depth.powi(2) * permeability * conductivity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Cuivre recuit : σ = 5,8e7 S/m, non magnétique donc μ = μ_0 = 4π·1e-7 H/m.
    const MU_0: f64 = 4.0 * PI * 1e-7;
    const SIGMA_CU: f64 = 5.8e7;

    #[test]
    fn standard_depth_realistic_copper_at_1_khz() {
        // δ = 1/sqrt(π·1000·μ_0·5,8e7).
        // π·1000·μ_0·5,8e7 = 228974,8 → sqrt = 478,5136 → δ = 2,0898e-3 m.
        let delta = eddy_standard_depth_of_penetration(1000.0, MU_0, SIGMA_CU);
        assert_relative_eq!(delta, 2.089_81e-3, epsilon = 1e-7);
    }

    #[test]
    fn frequency_and_depth_are_reciprocal() {
        // δ(f(z)) = z : la fréquence renvoyée replace bien la profondeur standard
        // exactement sur la profondeur cible.
        let z = 1.5e-3;
        let f = eddy_frequency_for_depth(z, MU_0, SIGMA_CU);
        let delta = eddy_standard_depth_of_penetration(f, MU_0, SIGMA_CU);
        assert_relative_eq!(delta, z, epsilon = 1e-12);
    }

    #[test]
    fn depth_scales_as_inverse_sqrt_of_frequency() {
        // δ ∝ 1/sqrt(f) : quadrupler la fréquence divise δ par deux.
        let d1 = eddy_standard_depth_of_penetration(1000.0, MU_0, SIGMA_CU);
        let d4 = eddy_standard_depth_of_penetration(4000.0, MU_0, SIGMA_CU);
        assert_relative_eq!(d1 / d4, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn phase_lag_is_one_radian_at_standard_depth() {
        // θ = z/δ : à z = δ le déphasage vaut exactement 1 rad.
        let delta = eddy_standard_depth_of_penetration(2000.0, MU_0, SIGMA_CU);
        assert_relative_eq!(eddy_phase_lag(delta, delta), 1.0, epsilon = 1e-12);
        // À deux profondeurs standard, le déphasage double.
        assert_relative_eq!(eddy_phase_lag(2.0 * delta, delta), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn frequency_scales_as_inverse_square_of_depth() {
        // f ∝ 1/z² : viser une profondeur deux fois moindre quadruple la fréquence.
        let f_deep = eddy_frequency_for_depth(2.0e-3, MU_0, SIGMA_CU);
        let f_shallow = eddy_frequency_for_depth(1.0e-3, MU_0, SIGMA_CU);
        assert_relative_eq!(f_shallow / f_deep, 4.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "fréquence doit être strictement positive")]
    fn zero_frequency_panics() {
        eddy_standard_depth_of_penetration(0.0, MU_0, SIGMA_CU);
    }
}
