//! Tournage conique — réglages machine pour usiner un cône droit
//! (conicité, demi-angle au sommet, décalage de contre-poupée, angle du chariot
//! supérieur).
//!
//! Un cône est décrit par son grand diamètre `D`, son petit diamètre `d` et la
//! longueur conique `L` séparant les deux sections. On en tire la **conicité**
//! `C`, le **demi-angle au sommet** `β` (entre la génératrice et l'axe), le
//! **décalage de contre-poupée** `t` (usinage entre pointes) et l'**angle du
//! chariot supérieur** (identique à `β`) :
//!
//! ```text
//! C = (D − d) / L                          (conicité, sans dimension)
//! β = atan( (D − d) / (2·L) )              (demi-angle au sommet, rad)
//! t = L_tot · (D − d) / (2·L)              (décalage de contre-poupée, m)
//! θ = atan( (D − d) / (2·L) ) = β          (angle du chariot supérieur, rad)
//! ```
//!
//! Légende (unités SI cohérentes — longueurs en mètres) :
//! - `D`, `d` : grand et petit diamètres (m), `D ≥ d`.
//! - `L` : longueur conique axiale entre les deux sections (m), `L > 0`.
//! - `L_tot` : longueur totale entre pointes de la pièce (m), `L_tot ≥ L`.
//! - `C` : conicité (m/m, sans dimension). Ex. cône Morse ≈ 0,05.
//! - `β`, `θ` : demi-angle au sommet et angle du chariot supérieur (rad).
//! - `t` : décalage transversal de la contre-poupée (m).
//!
//! **Limite honnête** : ce module suppose un **cône circulaire droit idéal**
//! (génératrice rectiligne, axe droit, sections circulaires). Le décalage de
//! contre-poupée suppose un **usinage entre pointes sur toute la longueur**
//! `L_tot`. Il ne modélise ni les défauts de forme, ni la flexion de la pièce,
//! ni la reprise élastique. Aucune constante matériau, procédé ou tolérance
//! normalisée (Morse, métrique…) n'est supposée : diamètres et longueurs sont
//! **fournis par l'appelant** ; ce module n'invente aucune valeur « par défaut ».

/// Conicité `C = (D − d) / L` (sans dimension) d'un cône droit, à partir du
/// grand diamètre `large_diameter`, du petit diamètre `small_diameter` et de la
/// longueur conique `length`. Longueurs en unités cohérentes (m).
///
/// Panique si un diamètre est négatif, si `large_diameter < small_diameter`, ou
/// si `length <= 0`.
pub fn taper_ratio(large_diameter: f64, small_diameter: f64, length: f64) -> f64 {
    assert!(
        small_diameter >= 0.0 && large_diameter >= 0.0,
        "les diamètres doivent être positifs ou nuls"
    );
    assert!(
        large_diameter >= small_diameter,
        "le grand diamètre doit être supérieur ou égal au petit diamètre"
    );
    assert!(
        length > 0.0,
        "la longueur conique doit être strictement positive"
    );
    (large_diameter - small_diameter) / length
}

/// Demi-angle au sommet `β = atan( (D − d) / (2·L) )` (rad), angle entre la
/// génératrice et l'axe du cône, à partir des diamètres et de la longueur
/// conique `length`.
///
/// Panique si un diamètre est négatif, si `large_diameter < small_diameter`, ou
/// si `length <= 0`.
pub fn taper_half_angle(large_diameter: f64, small_diameter: f64, length: f64) -> f64 {
    assert!(
        small_diameter >= 0.0 && large_diameter >= 0.0,
        "les diamètres doivent être positifs ou nuls"
    );
    assert!(
        large_diameter >= small_diameter,
        "le grand diamètre doit être supérieur ou égal au petit diamètre"
    );
    assert!(
        length > 0.0,
        "la longueur conique doit être strictement positive"
    );
    ((large_diameter - small_diameter) / (2.0 * length)).atan()
}

/// Décalage transversal de la contre-poupée `t = L_tot · (D − d) / (2·L)` (m)
/// pour tourner un cône par la méthode du décalage de contre-poupée, la pièce
/// étant usinée entre pointes sur toute la longueur `total_length`. `taper_length`
/// est la longueur conique.
///
/// Panique si un diamètre est négatif, si `large_diameter < small_diameter`, si
/// `taper_length <= 0`, ou si `total_length < taper_length`.
pub fn taper_tailstock_offset(
    large_diameter: f64,
    small_diameter: f64,
    taper_length: f64,
    total_length: f64,
) -> f64 {
    assert!(
        small_diameter >= 0.0 && large_diameter >= 0.0,
        "les diamètres doivent être positifs ou nuls"
    );
    assert!(
        large_diameter >= small_diameter,
        "le grand diamètre doit être supérieur ou égal au petit diamètre"
    );
    assert!(
        taper_length > 0.0,
        "la longueur conique doit être strictement positive"
    );
    assert!(
        total_length >= taper_length,
        "la longueur totale doit être supérieure ou égale à la longueur conique"
    );
    total_length * (large_diameter - small_diameter) / (2.0 * taper_length)
}

/// Angle de réglage du chariot supérieur `θ = atan( (D − d) / (2·L) )` (rad)
/// pour tourner un cône par pivotement du chariot ; identique au demi-angle au
/// sommet [`taper_half_angle`]. `length` est la longueur conique.
///
/// Panique si un diamètre est négatif, si `large_diameter < small_diameter`, ou
/// si `length <= 0`.
pub fn taper_compound_rest_angle(large_diameter: f64, small_diameter: f64, length: f64) -> f64 {
    assert!(
        small_diameter >= 0.0 && large_diameter >= 0.0,
        "les diamètres doivent être positifs ou nuls"
    );
    assert!(
        large_diameter >= small_diameter,
        "le grand diamètre doit être supérieur ou égal au petit diamètre"
    );
    assert!(
        length > 0.0,
        "la longueur conique doit être strictement positive"
    );
    ((large_diameter - small_diameter) / (2.0 * length)).atan()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ratio_matches_definition() {
        // D=50 mm, d=40 mm, L=100 mm → C = 10/100 = 0,1.
        assert_relative_eq!(taper_ratio(0.050, 0.040, 0.100), 0.1, epsilon = 1e-12);
    }

    #[test]
    fn half_angle_consistent_with_ratio() {
        // tan(β) = (D − d) / (2·L) = C / 2 (lien exact angle ↔ conicité).
        let (big, small, l) = (0.050_f64, 0.040_f64, 0.100_f64);
        let beta = taper_half_angle(big, small, l);
        let c = taper_ratio(big, small, l);
        assert_relative_eq!(beta.tan(), c / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn compound_rest_angle_equals_half_angle() {
        // L'angle du chariot supérieur est, par définition, le demi-angle.
        let (big, small, l) = (0.050_f64, 0.040_f64, 0.100_f64);
        assert_relative_eq!(
            taper_compound_rest_angle(big, small, l),
            taper_half_angle(big, small, l),
            epsilon = 1e-12
        );
    }

    #[test]
    fn tailstock_offset_worked_case_and_full_length_limit() {
        // Cas chiffré : D=50, d=40, L=100, L_tot=250 mm →
        // t = 0,250 · 0,010 / (2 · 0,100) = 0,250 · 0,05 = 0,0125 m.
        assert_relative_eq!(
            taper_tailstock_offset(0.050, 0.040, 0.100, 0.250),
            0.0125,
            epsilon = 1e-12
        );
        // Limite L_tot = L : le décalage vaut la demi-différence de rayon (D − d)/2.
        assert_relative_eq!(
            taper_tailstock_offset(0.050, 0.040, 0.100, 0.100),
            (0.050 - 0.040) / 2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn tailstock_offset_proportional_to_total_length() {
        // t ∝ L_tot : doubler la longueur totale double le décalage.
        let t1 = taper_tailstock_offset(0.050, 0.040, 0.100, 0.200);
        let t2 = taper_tailstock_offset(0.050, 0.040, 0.100, 0.400);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-12);
    }

    #[test]
    fn cylinder_has_zero_taper_angle_and_offset() {
        // D = d → conicité, demi-angle et décalage tous nuls.
        assert_relative_eq!(taper_ratio(0.025, 0.025, 0.100), 0.0, epsilon = 1e-12);
        assert_relative_eq!(taper_half_angle(0.025, 0.025, 0.100), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            taper_tailstock_offset(0.025, 0.025, 0.100, 0.250),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "grand diamètre")]
    fn inverted_diameters_panic() {
        taper_ratio(0.010, 0.020, 0.100);
    }
}
