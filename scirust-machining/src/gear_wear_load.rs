//! Charge limite d'usure d'un engrenage — **équation de Buckingham** : effort
//! tangentiel admissible vis-à-vis de la piqûre/usure de la denture, déduit du
//! facteur de rapport `Q` et du facteur charge-contrainte `K`.
//!
//! ```text
//! charge d'usure admissible   F_w = Dp·b·Q·K
//! facteur de rapport (ext.)   Q   = 2·Zg / (Zg + Zp)
//! facteur charge-contrainte   K   = (σ_es²·sin α / 1,4)·(1/E_p + 1/E_g)
//! ```
//!
//! `Dp` diamètre primitif du pignon (m), `b` largeur de denture (m), `Q` facteur
//! de rapport (adimensionnel), `K` facteur charge-contrainte de Buckingham (Pa),
//! `Zg`/`Zp` nombres de dents de la roue et du pignon (adimensionnels), `σ_es`
//! limite d'endurance superficielle du couple de matériaux (Pa), `α` angle de
//! pression au primitif (rad), `E_p`/`E_g` modules d'Young du pignon et de la roue
//! (Pa). `F_w` est en newtons (`m·m·Pa = N`).
//!
//! **Convention** : SI cohérent (N, m, Pa, rad). **Limite honnête** : équation
//! d'usure de Buckingham pour denture **extérieure** — le facteur `Q` a une forme
//! différente pour un engrenage intérieur (`Q = 2·Zg / (Zg − Zp)`), non traité
//! ici. La limite d'endurance superficielle `σ_es` (couple de matériaux, dureté,
//! traitement), l'angle de pression `α`, les modules d'Young `E_p`/`E_g` et le
//! facteur `1,4` de Buckingham fixent des données de **procédé/matériau fournies
//! par l'appelant** — aucune valeur n'est inventée ici. Le dimensionnement exige
//! que la charge d'usure admissible `F_w` **dépasse** la charge dynamique
//! effective sur la denture ; cette comparaison relève de l'appelant.

use core::f64::consts::PI;

/// Charge d'usure admissible `F_w = Dp·b·Q·K` (N).
///
/// Produit du diamètre primitif du pignon, de la largeur de denture, du facteur
/// de rapport `Q` et du facteur charge-contrainte `K` de Buckingham.
///
/// Panique si `pinion_pitch_diameter <= 0`, `face_width <= 0`, `ratio_factor <= 0`
/// ou `load_stress_factor < 0`.
pub fn gearwear_limiting_load(
    pinion_pitch_diameter: f64,
    face_width: f64,
    ratio_factor: f64,
    load_stress_factor: f64,
) -> f64 {
    assert!(
        pinion_pitch_diameter > 0.0,
        "le diamètre primitif du pignon doit être positif"
    );
    assert!(face_width > 0.0, "la largeur de denture doit être positive");
    assert!(
        ratio_factor > 0.0,
        "le facteur de rapport doit être positif"
    );
    assert!(
        load_stress_factor >= 0.0,
        "le facteur charge-contrainte doit être positif"
    );
    pinion_pitch_diameter * face_width * ratio_factor * load_stress_factor
}

/// Facteur de rapport `Q = 2·Zg / (Zg + Zp)` pour denture **extérieure**
/// (adimensionnel).
///
/// Croît de 1 (dentures égales) vers 2 (crémaillère, `Zg → ∞`). Pour une denture
/// intérieure la formule diffère (`2·Zg / (Zg − Zp)`) et n'est pas traitée ici.
///
/// Panique si `gear_teeth <= 0` ou `pinion_teeth <= 0`.
pub fn gearwear_ratio_factor(gear_teeth: f64, pinion_teeth: f64) -> f64 {
    assert!(
        gear_teeth > 0.0,
        "le nombre de dents de la roue doit être positif"
    );
    assert!(
        pinion_teeth > 0.0,
        "le nombre de dents du pignon doit être positif"
    );
    2.0 * gear_teeth / (gear_teeth + pinion_teeth)
}

/// Facteur charge-contrainte de Buckingham
/// `K = (σ_es²·sin α / 1,4)·(1/E_p + 1/E_g)` (Pa).
///
/// Regroupe la limite d'endurance superficielle du couple de matériaux, l'angle
/// de pression et la souplesse combinée des deux dentures. Le facteur `1,4`
/// (constante de Buckingham) et `σ_es` sont **fournis par l'appelant**.
///
/// Panique si `surface_endurance_limit < 0`, si `pressure_angle_rad ∉ ]0, π/2[`,
/// `youngs_modulus_pinion <= 0` ou `youngs_modulus_gear <= 0`.
pub fn gearwear_load_stress_factor(
    surface_endurance_limit: f64,
    pressure_angle_rad: f64,
    youngs_modulus_pinion: f64,
    youngs_modulus_gear: f64,
) -> f64 {
    assert!(
        surface_endurance_limit >= 0.0,
        "la limite d'endurance superficielle doit être positive"
    );
    assert!(
        pressure_angle_rad > 0.0 && pressure_angle_rad < PI / 2.0,
        "l'angle de pression doit être dans ]0, π/2["
    );
    assert!(
        youngs_modulus_pinion > 0.0,
        "le module d'Young du pignon doit être positif"
    );
    assert!(
        youngs_modulus_gear > 0.0,
        "le module d'Young de la roue doit être positif"
    );
    (surface_endurance_limit.powi(2) * pressure_angle_rad.sin() / 1.4)
        * (1.0 / youngs_modulus_pinion + 1.0 / youngs_modulus_gear)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn ratio_factor_unity_for_equal_teeth() {
        // Cas limite Zg = Zp : Q = 2·Z / (2·Z) = 1.
        let q = gearwear_ratio_factor(24.0, 24.0);
        assert_relative_eq!(q, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn ratio_factor_tends_to_two_for_rack() {
        // Zg ≫ Zp : Q → 2 (roue quasi-crémaillère face à un petit pignon).
        let q = gearwear_ratio_factor(1.0e9, 20.0);
        assert!(q < 2.0);
        assert_relative_eq!(q, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn ratio_factor_realistic_value() {
        // Zg = 60, Zp = 20 : Q = 2·60 / 80 = 1,5.
        let q = gearwear_ratio_factor(60.0, 20.0);
        assert_relative_eq!(q, 1.5, epsilon = 1e-12);
    }

    #[test]
    fn load_stress_factor_scales_with_square_of_endurance_limit() {
        // K ∝ σ_es² : doubler σ_es quadruple le facteur charge-contrainte.
        let alpha = 20.0_f64 * PI / 180.0;
        let k1 = gearwear_load_stress_factor(600.0e6, alpha, 200.0e9, 200.0e9);
        let k2 = gearwear_load_stress_factor(1200.0e6, alpha, 200.0e9, 200.0e9);
        assert_relative_eq!(k2 / k1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn load_stress_factor_realistic_value() {
        // σ_es = 600 MPa, α = 20°, E_p = E_g = 200 GPa (acier/acier).
        // K = (600e6²·sin20°/1,4)·(2/200e9) ≈ 8,7948e5 Pa.
        let alpha = 20.0_f64 * PI / 180.0;
        let k = gearwear_load_stress_factor(600.0e6, alpha, 200.0e9, 200.0e9);
        assert_relative_eq!(k, 879_480.368_551_72, epsilon = 1e-2);
    }

    #[test]
    fn limiting_load_realistic_value_and_linearity() {
        // Dp = 100 mm, b = 40 mm, Q = 1,5, avec le K acier/acier ci-dessus.
        // F_w = 0,100·0,040·1,5·8,7948e5 ≈ 5276,88 N.
        let alpha = 20.0_f64 * PI / 180.0;
        let q = gearwear_ratio_factor(60.0, 20.0);
        let k = gearwear_load_stress_factor(600.0e6, alpha, 200.0e9, 200.0e9);
        let fw = gearwear_limiting_load(0.100, 0.040, q, k);
        assert_relative_eq!(fw, 5_276.882_211_310_3, epsilon = 1e-3);
        // F_w ∝ b : doubler la largeur de denture double la charge admissible.
        let fw2 = gearwear_limiting_load(0.100, 0.080, q, k);
        assert_relative_eq!(fw2 / fw, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "angle de pression")]
    fn out_of_range_pressure_angle_panics() {
        gearwear_load_stress_factor(600.0e6, PI, 200.0e9, 200.0e9);
    }
}
