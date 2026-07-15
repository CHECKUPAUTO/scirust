//! Longueur d'engagement de filet — dimensionnement contre l'**arrachement**
//! (cisaillement) des filets d'un assemblage vis/écrou.
//!
//! On compare la résistance au cisaillement des filets engagés à la résistance
//! en traction de la vis. L'aire cisaillée est approchée par un cylindre au
//! diamètre nominal, corrigée par un facteur de forme de filet fourni :
//!
//! ```text
//! aire de cisaillement des filets   As_stripping = π·d·Le·k
//! résistance à l'arrachement        F_strip = As_stripping·τ_nut
//! résistance en traction de la vis  F_tens  = At·Rm
//! longueur d'engagement requise     Le_req  = (At·Rm) / (π·d·k·τ_nut)
//! règle acier/acier                 Le_full ≈ d·r   (r = 1 pour τ égales)
//! ```
//!
//! `d` diamètre nominal, `Le` longueur d'engagement, `At` section résistante de
//! la vis, `Rm` résistance en traction de la vis, `τ_nut` résistance au
//! cisaillement du matériau de l'écrou, `k` facteur de forme de filet
//! (adimensionnel, fraction cisaillante par unité de longueur, typiquement
//! ≈ 0,5–0,88), `r` rapport de résistance vis/écrou (adimensionnel).
//!
//! **Convention d'unités** : base cohérente **N–mm–MPa** (`d`, `Le` en mm,
//! `At` en mm², `Rm`, `τ_nut` en MPa, aires en mm²). Toute autre base cohérente
//! convient car les contraintes se simplifient dans `Le_req`.
//!
//! **Limite honnête** : modèle de cisaillement de filet **simplifié** — l'aire
//! réelle dépend du profil (ISO 68-1), du jeu de flanc et de la répartition non
//! uniforme de charge entre filets ; toute cette géométrie est condensée dans le
//! facteur `k` **fourni par l'appelant**. Les propriétés des matériaux vis/écrou
//! (`Rm`, `τ_nut`, `At`, rapport `r`) sont elles aussi **fournies** — aucune
//! valeur matériau ou de procédé n'est inventée ici. Complète
//! [`crate::threads`] (géométrie et section résistante du filetage).

use core::f64::consts::PI;

/// Aire de cisaillement des filets engagés `As = π·d·Le·k` (mm²).
///
/// Cylindre au diamètre nominal `d` sur la longueur d'engagement `Le`, corrigé
/// par le facteur de forme de filet `k`. Dans ce modèle simplifié, le pas
/// n'intervient qu'à travers `k` (il est validé mais non recombiné ici).
///
/// Panique si `nominal_diameter <= 0`, `engagement_length < 0`, `pitch <= 0`
/// ou `thread_shear_factor <= 0`.
pub fn thread_eng_stripping_area(
    nominal_diameter: f64,
    engagement_length: f64,
    pitch: f64,
    thread_shear_factor: f64,
) -> f64 {
    assert!(
        nominal_diameter > 0.0,
        "le diamètre nominal doit être strictement positif"
    );
    assert!(
        engagement_length >= 0.0,
        "la longueur d'engagement doit être positive"
    );
    assert!(pitch > 0.0, "le pas doit être strictement positif");
    assert!(
        thread_shear_factor > 0.0,
        "le facteur de forme de filet doit être strictement positif"
    );
    PI * nominal_diameter * engagement_length * thread_shear_factor
}

/// Longueur d'engagement requise `Le = (At·Rm)/(π·d·k·τ_nut)` (mm).
///
/// Longueur minimale telle que la résistance à l'arrachement des filets de
/// l'écrou (`π·d·Le·k·τ_nut`) atteigne la résistance en traction de la vis
/// (`At·Rm`). Les contraintes `Rm` et `τ_nut` se simplifient : le résultat est
/// homogène à une longueur quelle que soit la base cohérente choisie.
///
/// Panique si l'un des arguments est `<= 0`.
pub fn thread_eng_required_length(
    bolt_tensile_area: f64,
    tensile_strength: f64,
    nut_shear_strength: f64,
    nominal_diameter: f64,
    shear_factor: f64,
) -> f64 {
    assert!(
        bolt_tensile_area > 0.0,
        "la section résistante de la vis doit être strictement positive"
    );
    assert!(
        tensile_strength > 0.0,
        "la résistance en traction de la vis doit être strictement positive"
    );
    assert!(
        nut_shear_strength > 0.0,
        "la résistance au cisaillement de l'écrou doit être strictement positive"
    );
    assert!(
        nominal_diameter > 0.0,
        "le diamètre nominal doit être strictement positif"
    );
    assert!(
        shear_factor > 0.0,
        "le facteur de forme de filet doit être strictement positif"
    );
    (bolt_tensile_area * tensile_strength)
        / (PI * nominal_diameter * shear_factor * nut_shear_strength)
}

/// Longueur d'engagement de pleine résistance `Le ≈ d·r` (mm).
///
/// Règle d'atelier : pour un assemblage acier sur acier de résistances
/// équivalentes (`strength_ratio = 1`), une longueur d'engagement de l'ordre du
/// diamètre nominal suffit à faire rompre la vis avant l'arrachement des
/// filets. Le rapport `r` (résistance vis / résistance de l'écrou) allonge
/// l'engagement lorsque l'écrou est plus tendre (`r > 1`).
///
/// Panique si `nominal_diameter <= 0` ou `strength_ratio <= 0`.
pub fn thread_eng_length_for_full_strength(nominal_diameter: f64, strength_ratio: f64) -> f64 {
    assert!(
        nominal_diameter > 0.0,
        "le diamètre nominal doit être strictement positif"
    );
    assert!(
        strength_ratio > 0.0,
        "le rapport de résistance doit être strictement positif"
    );
    nominal_diameter * strength_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn stripping_area_is_linear_in_engagement_length() {
        // As = π·d·Le·k : doubler Le double l'aire cisaillée.
        let a1 = thread_eng_stripping_area(10.0, 6.0, 1.5, 0.5);
        let a2 = thread_eng_stripping_area(10.0, 12.0, 1.5, 0.5);
        assert_relative_eq!(a2, 2.0 * a1, epsilon = 1e-12);
        // Valeur explicite : π·10·6·0,5 = 30·π.
        assert_relative_eq!(a1, 30.0 * PI, epsilon = 1e-12);
    }

    #[test]
    fn required_length_equalizes_stripping_and_tensile_resistances() {
        // À Le = Le_req, la résistance à l'arrachement égale la traction vis.
        let (at, rm, tau, d, k) = (58.0, 800.0, 464.0, 10.0, 0.5);
        let le = thread_eng_required_length(at, rm, tau, d, k);
        let strip = thread_eng_stripping_area(d, le, 1.5, k) * tau;
        let tensile = at * rm;
        assert_relative_eq!(strip, tensile, epsilon = 1e-6);
    }

    #[test]
    fn required_length_realistic_m10_class_88() {
        // M10 classe 8.8 : At = 58 mm², Rm = 800 MPa, écrou τ = 464 MPa,
        // d = 10 mm, k = 0,5. Le = 58·800 / (π·10·0,5·464)
        //                        = 46400 / (2320·π) ≈ 6,3662 mm.
        let le = thread_eng_required_length(58.0, 800.0, 464.0, 10.0, 0.5);
        assert_relative_eq!(le, 46400.0 / (2320.0 * PI), epsilon = 1e-12);
        assert_relative_eq!(le, 6.366_2, epsilon = 1e-3);
    }

    #[test]
    fn required_length_scales_inversely_with_shear_factor() {
        // Le_req ∝ 1/k : un facteur de forme deux fois plus grand halve Le.
        let le1 = thread_eng_required_length(58.0, 800.0, 464.0, 10.0, 0.5);
        let le2 = thread_eng_required_length(58.0, 800.0, 464.0, 10.0, 1.0);
        assert_relative_eq!(le1, 2.0 * le2, epsilon = 1e-12);
    }

    #[test]
    fn full_strength_length_matches_diameter_for_equal_materials() {
        // r = 1 (acier/acier) → Le ≈ d ; r = 1,5 (écrou tendre) → Le ≈ 1,5·d.
        assert_relative_eq!(
            thread_eng_length_for_full_strength(12.0, 1.0),
            12.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            thread_eng_length_for_full_strength(12.0, 1.5),
            18.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "diamètre nominal")]
    fn zero_diameter_panics() {
        thread_eng_stripping_area(0.0, 6.0, 1.5, 0.5);
    }
}
