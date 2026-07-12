//! Dureté — essais **Brinell** (HB) et **Vickers** (HV), et estimation de la
//! résistance à la traction d'un acier à partir de la dureté Brinell.
//!
//! ```text
//! Brinell   HB = 2·F / [π·D·(D − √(D² − d²))]      (F en kgf, D,d en mm)
//! Vickers   HV = 1,854·F / d²                       (F en kgf, d en mm)
//! acier     Rm ≈ 3,45·HB   (MPa)                    (corrélation empirique)
//! ```
//!
//! `F` charge d'essai (kgf), `D` diamètre de la bille Brinell (mm), `d`
//! diamètre de l'empreinte Brinell ou diagonale moyenne Vickers (mm), `HB`/`HV`
//! duretés, `Rm` résistance à la traction estimée.
//!
//! **Convention** : unités traditionnelles des essais (kgf, mm) ; `Rm` rendu en
//! **MPa**. **Limite honnête** : formules **normalisées** des empreintes ; la
//! relation `Rm ≈ 3,45·HB` est une **corrélation empirique** valable pour les
//! aciers au carbone courants, pas une loi — l'appelant l'ajuste au matériau.

use core::f64::consts::PI;

/// Dureté Brinell `HB = 2·F/[π·D·(D − √(D² − d²))]`.
///
/// Panique si `indent_diameter >= ball_diameter` ou dimensions `<= 0`.
pub fn brinell_hardness(load_kgf: f64, ball_diameter_mm: f64, indent_diameter_mm: f64) -> f64 {
    assert!(
        ball_diameter_mm > 0.0 && indent_diameter_mm > 0.0,
        "les diamètres doivent être strictement positifs"
    );
    assert!(
        indent_diameter_mm < ball_diameter_mm,
        "l'empreinte ne peut dépasser la bille"
    );
    let d = ball_diameter_mm;
    let sqrt = (d * d - indent_diameter_mm * indent_diameter_mm).sqrt();
    2.0 * load_kgf / (PI * d * (d - sqrt))
}

/// Dureté Vickers `HV = 1,854·F/d²`, `d` diagonale moyenne de l'empreinte (mm).
///
/// Panique si `diagonal <= 0`.
pub fn vickers_hardness(load_kgf: f64, diagonal_mm: f64) -> f64 {
    assert!(
        diagonal_mm > 0.0,
        "la diagonale doit être strictement positive"
    );
    1.854 * load_kgf / (diagonal_mm * diagonal_mm)
}

/// Résistance à la traction estimée d'un acier `Rm ≈ 3,45·HB` (MPa).
///
/// Panique si `hb < 0`.
pub fn tensile_strength_from_brinell(hb: f64) -> f64 {
    assert!(hb >= 0.0, "la dureté doit être positive");
    3.45 * hb
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn brinell_of_a_standard_test() {
        // F=3000 kgf, D=10 mm, d=4 mm → HB ≈ 229.
        let hb = brinell_hardness(3000.0, 10.0, 4.0);
        let sqrt = (100.0f64 - 16.0).sqrt();
        assert_relative_eq!(
            hb,
            2.0 * 3000.0 / (PI * 10.0 * (10.0 - sqrt)),
            epsilon = 1e-9
        );
        assert!(hb > 220.0 && hb < 240.0);
    }

    #[test]
    fn smaller_indent_means_harder() {
        // Une empreinte plus petite (matériau plus dur) → HB plus élevé.
        assert!(brinell_hardness(3000.0, 10.0, 3.0) > brinell_hardness(3000.0, 10.0, 5.0));
    }

    #[test]
    fn vickers_definition() {
        // F=30 kgf, d=0,5 mm → HV = 1,854·30/0,25 = 222,48.
        assert_relative_eq!(
            vickers_hardness(30.0, 0.5),
            1.854 * 30.0 / 0.25,
            epsilon = 1e-9
        );
    }

    #[test]
    fn tensile_from_hardness_correlation() {
        // HB=200 → Rm ≈ 690 MPa.
        assert_relative_eq!(tensile_strength_from_brinell(200.0), 690.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "empreinte ne peut dépasser")]
    fn oversized_indent_panics() {
        brinell_hardness(3000.0, 10.0, 12.0);
    }
}
