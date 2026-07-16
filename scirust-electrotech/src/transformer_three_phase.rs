//! **Transformateur triphasé (couplages)** — rapport des tensions **composées**
//! selon le couplage (Yy, Yd, Dy, Dd) et déphasage donné par l'indice horaire.
//!
//! ```text
//! Yy ou Dd : rapport tensions composées   m = N1 / N2
//! Yd (étoile → triangle) :                 m = √3 · N1 / N2
//! Dy (triangle → étoile) :                 m = N1 / (√3 · N2)
//! déphasage (indice horaire) :             φ = clock_number · 30°
//! ```
//!
//! `N1` nombre de spires primaires **par colonne** (sans dimension), `N2` nombre
//! de spires secondaires **par colonne** (sans dimension), `m` rapport des
//! tensions **composées** (de ligne) primaire/secondaire (sans dimension),
//! `clock_number` indice horaire du couplage (entier `∈ [0, 11]`, sans
//! dimension), `φ` déphasage entre tensions homologues primaire et secondaire
//! (degrés). Le facteur `√3` provient du passage entre grandeur de phase (par
//! colonne) et grandeur de ligne : un enroulement en étoile relève la tension de
//! ligne d'un facteur `√3` par rapport à la tension d'enroulement, un enroulement
//! en triangle non.
//!
//! **Convention** : SI ; spires sans dimension, rapports sans dimension,
//! déphasage en **degrés** (pas de 30°). **Limite honnête** : transformateur
//! triphasé **équilibré** et **idéal** (pas de pertes, de fuite ni de saturation) ;
//! le **rapport de spires par colonne** `N1`, `N2` est **fourni par l'appelant**
//! (plaque signalétique, bobinage réel) ; le **couplage** (Yy, Yd, Dy, Dd) et
//! l'**indice horaire** sont eux aussi **fournis** — aucune valeur « par défaut »
//! de composant ou de couplage n'est inventée.

/// Rapport des tensions **composées** d'un couplage **Yd** (primaire étoile,
/// secondaire triangle) `m = √3 · N1 / N2` (sans dimension).
///
/// Panique si `primary_turns <= 0` ou si `secondary_turns <= 0`.
pub fn xfmr3_line_voltage_ratio_yd(primary_turns: f64, secondary_turns: f64) -> f64 {
    assert!(
        primary_turns > 0.0,
        "le nombre de spires primaires N1 doit être > 0"
    );
    assert!(
        secondary_turns > 0.0,
        "le nombre de spires secondaires N2 doit être > 0"
    );
    3.0_f64.sqrt() * primary_turns / secondary_turns
}

/// Rapport des tensions **composées** d'un couplage **Dy** (primaire triangle,
/// secondaire étoile) `m = N1 / (√3 · N2)` (sans dimension).
///
/// Panique si `primary_turns <= 0` ou si `secondary_turns <= 0`.
pub fn xfmr3_line_voltage_ratio_dy(primary_turns: f64, secondary_turns: f64) -> f64 {
    assert!(
        primary_turns > 0.0,
        "le nombre de spires primaires N1 doit être > 0"
    );
    assert!(
        secondary_turns > 0.0,
        "le nombre de spires secondaires N2 doit être > 0"
    );
    primary_turns / (3.0_f64.sqrt() * secondary_turns)
}

/// Rapport des tensions **composées** d'un couplage **Yy** (ou **Dd**), où le
/// facteur `√3` se compense de part et d'autre `m = N1 / N2` (sans dimension).
///
/// Panique si `primary_turns <= 0` ou si `secondary_turns <= 0`.
pub fn xfmr3_line_voltage_ratio_yy(primary_turns: f64, secondary_turns: f64) -> f64 {
    assert!(
        primary_turns > 0.0,
        "le nombre de spires primaires N1 doit être > 0"
    );
    assert!(
        secondary_turns > 0.0,
        "le nombre de spires secondaires N2 doit être > 0"
    );
    primary_turns / secondary_turns
}

/// Déphasage entre tensions homologues donné par l'**indice horaire** (clock
/// number) `φ = clock_number · 30°` (degrés).
///
/// Panique si `clock_number` n'est pas dans `[0, 11]`.
pub fn xfmr3_phase_shift_degrees(clock_number: f64) -> f64 {
    assert!(
        (0.0..=11.0).contains(&clock_number),
        "l'indice horaire clock_number doit être dans [0, 11]"
    );
    clock_number * 30.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn yd_is_sqrt3_times_yy() {
        // Identité : à spires égales, le rapport Yd vaut √3 fois le rapport Yy.
        let n1 = 1000.0_f64;
        let n2 = 400.0_f64;
        let yd = xfmr3_line_voltage_ratio_yd(n1, n2);
        let yy = xfmr3_line_voltage_ratio_yy(n1, n2);
        assert_relative_eq!(yd / yy, 3.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn dy_is_yy_over_sqrt3() {
        // Identité : à spires égales, le rapport Dy vaut le rapport Yy divisé par √3.
        let n1 = 800.0_f64;
        let n2 = 100.0_f64;
        let dy = xfmr3_line_voltage_ratio_dy(n1, n2);
        let yy = xfmr3_line_voltage_ratio_yy(n1, n2);
        assert_relative_eq!(dy * 3.0_f64.sqrt(), yy, epsilon = 1e-12);
    }

    #[test]
    fn yd_and_dy_are_reciprocal_products() {
        // Réciprocité : Yd(N1,N2)·Dy(N2,N1) = (√3·N1/N2)·(N2/(√3·N1)) = 1.
        let n1 = 550.0_f64;
        let n2 = 275.0_f64;
        let yd = xfmr3_line_voltage_ratio_yd(n1, n2);
        let dy = xfmr3_line_voltage_ratio_dy(n2, n1);
        assert_relative_eq!(yd * dy, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn yy_ratio_scales_with_primary_turns() {
        // Proportionnalité : doubler N1 double le rapport Yy.
        let m1 = xfmr3_line_voltage_ratio_yy(500.0, 200.0);
        let m2 = xfmr3_line_voltage_ratio_yy(1000.0, 200.0);
        assert_relative_eq!(m2 / m1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn clock_number_zero_and_six() {
        // Cas limites de l'indice horaire : 0 → 0°, 6 → 180°.
        assert_relative_eq!(xfmr3_phase_shift_degrees(0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(xfmr3_phase_shift_degrees(6.0), 180.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_dyn11_ratio_and_shift() {
        // Cas chiffré Dyn11 : primaire triangle N1 = 2000 spires/colonne,
        // secondaire étoile N2 = 100 spires/colonne, indice horaire 11.
        //   m = N1 / (√3·N2) = 2000 / (√3·100) = 20 / √3
        //     = 20 / 1,7320508075688772 = 11,547005383792515
        //   φ = 11 · 30 = 330°
        let m = xfmr3_line_voltage_ratio_dy(2000.0, 100.0);
        assert_relative_eq!(m, 20.0_f64 / 3.0_f64.sqrt(), epsilon = 1e-9);
        assert_relative_eq!(m, 11.547005383792515, epsilon = 1e-3);
        let phi = xfmr3_phase_shift_degrees(11.0);
        assert_relative_eq!(phi, 330.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "l'indice horaire clock_number doit être dans [0, 11]")]
    fn clock_number_out_of_range_panics() {
        xfmr3_phase_shift_degrees(12.0);
    }
}
