//! Moletage (knurling) — **géométrie du motif** (pas circonférentiel, nombre de
//! crans) et **croissance diamétrale** par refoulement de matière.
//!
//! ```text
//! pas circonférentiel    p  = π / DP
//! nombre de crans        N  = π·D / p
//! croissance diamétrale  Δd = h
//! diamètre avant moletage Dp = Df − h
//! ```
//!
//! `DP` pas diamétral de la molette (1/m, densité de crans imposée par l'outil),
//! `p` pas circonférentiel du motif (m, longueur d'arc entre deux crans),
//! `D` diamètre de la pièce à moleter (m), `N` nombre de crans sur le pourtour
//! (adimensionnel, valeur réelle à arrondir), `h` profondeur du cran (m),
//! `Δd` croissance diamétrale (m, la matière est **refoulée** donc le diamètre
//! **augmente**), `Df` diamètre fini après moletage (m), `Dp` diamètre du brut
//! avant moletage (m).
//!
//! **Convention** : SI cohérent (m, 1/m, adimensionnel). **Limite honnête** :
//! moletage par **déformation** (refoulement de matière, le diamètre
//! **augmente**), pas par enlèvement ; le **pas diamétral** `DP` de la molette
//! est **fourni par l'appelant** — aucune valeur « par défaut » n'est inventée.
//! Le nombre de crans `N` **doit être entier** pour éviter un motif fuyant
//! (« motif qui décale ») : cette contrainte de compatibilité pas/diamètre est
//! **à vérifier par l'appelant** ; la fonction renvoie la valeur réelle. La
//! croissance diamétrale est **approximée par la profondeur du cran**.

use core::f64::consts::PI;

/// Pas circonférentiel du motif `p = π / DP` (m).
///
/// Longueur d'arc entre deux crans consécutifs, imposée par le pas diamétral
/// `DP` de la molette.
///
/// Panique si `diametral_pitch <= 0`.
pub fn knurl_circular_pitch(diametral_pitch: f64) -> f64 {
    assert!(
        diametral_pitch > 0.0,
        "le pas diamétral doit être strictement positif"
    );
    PI / diametral_pitch
}

/// Nombre de crans sur le pourtour `N = π·D / p` (adimensionnel).
///
/// Valeur **réelle** (à arrondir) du nombre de crans reportés sur la
/// circonférence `π·D` ; elle **doit être entière** pour un motif régulier,
/// contrainte à vérifier par l'appelant.
///
/// Panique si `workpiece_diameter <= 0` ou `circular_pitch <= 0`.
pub fn knurl_teeth_count(workpiece_diameter: f64, circular_pitch: f64) -> f64 {
    assert!(
        workpiece_diameter > 0.0,
        "le diamètre de la pièce doit être strictement positif"
    );
    assert!(
        circular_pitch > 0.0,
        "le pas circonférentiel doit être strictement positif"
    );
    PI * workpiece_diameter / circular_pitch
}

/// Croissance diamétrale `Δd = h` (m).
///
/// La matière étant **refoulée** (et non enlevée), le diamètre augmente d'une
/// quantité approximée par la profondeur du cran `h`.
///
/// Panique si `tooth_depth < 0`.
pub fn knurl_diameter_growth(tooth_depth: f64) -> f64 {
    assert!(
        tooth_depth >= 0.0,
        "la profondeur du cran doit être positive ou nulle"
    );
    tooth_depth
}

/// Diamètre du brut avant moletage `Dp = Df − h` (m).
///
/// Diamètre à usiner avant l'opération : le refoulement de la matière fera
/// croître le diamètre de `h` pour atteindre le diamètre fini `Df`.
///
/// Panique si `finished_diameter <= 0`, `tooth_depth < 0` ou
/// `tooth_depth >= finished_diameter` (diamètre avant moletage non strictement
/// positif).
pub fn knurl_pre_roll_diameter(finished_diameter: f64, tooth_depth: f64) -> f64 {
    assert!(
        finished_diameter > 0.0,
        "le diamètre fini doit être strictement positif"
    );
    assert!(
        tooth_depth >= 0.0,
        "la profondeur du cran doit être positive ou nulle"
    );
    assert!(
        tooth_depth < finished_diameter,
        "la profondeur du cran doit être inférieure au diamètre fini"
    );
    finished_diameter - tooth_depth
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn circular_pitch_reciprocity_with_diametral_pitch() {
        // Réciprocité p = π/DP ⇔ DP = π/p : on reconstitue le pas diamétral.
        let dp = 3_927.0_f64;
        let p = knurl_circular_pitch(dp);
        assert_relative_eq!(PI / p, dp, epsilon = 1e-9);
    }

    #[test]
    fn teeth_count_realistic_value() {
        // Pièce Ø20 mm, pas circonférentiel 0,8 mm :
        // N = π·0,020 / 0,0008 = π·25 = 78,53981633974483.
        let n = knurl_teeth_count(0.020, 0.000_8);
        assert_relative_eq!(n, 78.539_816_339_744_83, epsilon = 1e-9);
    }

    #[test]
    fn teeth_count_proportional_to_diameter() {
        // N ∝ D : doubler le diamètre double le nombre de crans.
        let p = 0.000_8_f64;
        let n1 = knurl_teeth_count(0.020, p);
        let n2 = knurl_teeth_count(0.040, p);
        assert_relative_eq!(n2 / n1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn diameter_growth_equals_tooth_depth() {
        // Identité Δd = h : la croissance vaut exactement la profondeur du cran.
        let h = 0.000_25_f64;
        assert_relative_eq!(knurl_diameter_growth(h), h, epsilon = 1e-15);
    }

    #[test]
    fn pre_roll_and_growth_reconstitute_finished_diameter() {
        // Réciprocité : Dp + Δd = Df (le refoulement ramène au diamètre fini).
        let (df, h) = (0.020_f64, 0.000_3_f64);
        let dp = knurl_pre_roll_diameter(df, h);
        assert_relative_eq!(dp + knurl_diameter_growth(h), df, epsilon = 1e-15);
    }

    #[test]
    fn pre_roll_realistic_value() {
        // Ø fini 20 mm, cran 0,3 mm → brut = 0,020 − 0,0003 = 0,0197 m.
        assert_relative_eq!(
            knurl_pre_roll_diameter(0.020, 0.000_3),
            0.019_7,
            epsilon = 1e-15
        );
    }

    #[test]
    #[should_panic(expected = "pas diamétral")]
    fn non_positive_diametral_pitch_panics() {
        knurl_circular_pitch(0.0);
    }
}
