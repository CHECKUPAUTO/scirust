//! **Mélange adiabatique de deux débits d'air humide** — état psychrométrique
//! du flux résultant comme barycentre massique des deux flux entrants.
//!
//! ```text
//! débit d'air sec        m   = m1 + m2                        (conservation)
//! température sèche       T   = (m1·T1 + m2·T2) / (m1 + m2)
//! teneur en eau           w   = (m1·w1 + m2·w2) / (m1 + m2)
//! enthalpie spécifique    h   = (m1·h1 + m2·h2) / (m1 + m2)
//! ```
//!
//! `m1`, `m2` débits massiques d'**air sec** des deux flux (kg·s⁻¹), `m` débit
//! d'air sec du mélange (kg·s⁻¹) ; `T1`, `T2`, `T` températures sèches (°C) ;
//! `w1`, `w2`, `w` teneurs en eau (kg d'eau par kg d'air **sec**) ; `h1`, `h2`,
//! `h` enthalpies spécifiques rapportées au kg d'air **sec** (kJ·kg⁻¹).
//!
//! **Convention** : chaque grandeur d'état du mélange est la moyenne pondérée par
//! les débits d'air sec des grandeurs correspondantes des deux flux ; le point
//! résultant se situe sur le segment reliant les deux états sur le diagramme
//! psychrométrique. Unités SI par ailleurs.
//!
//! **Limite honnête** : mélange **adiabatique à pression constante** de deux
//! flux d'air humide, sans apport ni retrait de chaleur ni d'eau. Les débits
//! massiques d'**air sec** ainsi que les états (température, teneur en eau,
//! enthalpie) des deux flux sont des **données fournies par l'appelant** : aucune
//! valeur « par défaut » n'est inventée ici. La **saturation n'est pas vérifiée**
//! — si le barycentre tombe sous la courbe de saturation, il y a formation de
//! brouillard et le modèle linéaire ci-dessus n'est plus valable ; c'est à
//! l'appelant de le contrôler. Complète [`crate::psychrometrics`].

/// Débit d'air sec du mélange `m = m1 + m2` (conservation de la masse d'air sec).
///
/// `mass_flow1`, `mass_flow2` débits massiques d'air **sec** des deux flux
/// (kg·s⁻¹) ; renvoie le débit d'air sec du mélange (kg·s⁻¹).
///
/// Panique si `mass_flow1 <= 0` ou `mass_flow2 <= 0`.
pub fn airmix_mass_flow_out(mass_flow1: f64, mass_flow2: f64) -> f64 {
    assert!(
        mass_flow1 > 0.0,
        "débit d'air sec du premier flux strictement positif requis"
    );
    assert!(
        mass_flow2 > 0.0,
        "débit d'air sec du second flux strictement positif requis"
    );
    mass_flow1 + mass_flow2
}

/// Température sèche du mélange `T = (m1·T1 + m2·T2) / (m1 + m2)`
/// (moyenne pondérée par les débits d'air sec).
///
/// `mass_flow1`, `mass_flow2` débits massiques d'air **sec** (kg·s⁻¹),
/// `temperature1`, `temperature2` températures sèches des deux flux (°C) ;
/// renvoie la température sèche du mélange (°C).
///
/// Panique si `mass_flow1 <= 0` ou `mass_flow2 <= 0`.
pub fn airmix_temperature(
    mass_flow1: f64,
    temperature1: f64,
    mass_flow2: f64,
    temperature2: f64,
) -> f64 {
    assert!(
        mass_flow1 > 0.0,
        "débit d'air sec du premier flux strictement positif requis"
    );
    assert!(
        mass_flow2 > 0.0,
        "débit d'air sec du second flux strictement positif requis"
    );
    (mass_flow1 * temperature1 + mass_flow2 * temperature2) / (mass_flow1 + mass_flow2)
}

/// Teneur en eau du mélange `w = (m1·w1 + m2·w2) / (m1 + m2)`
/// (moyenne pondérée par les débits d'air sec).
///
/// `mass_flow1`, `mass_flow2` débits massiques d'air **sec** (kg·s⁻¹),
/// `humidity_ratio1`, `humidity_ratio2` teneurs en eau des deux flux (kg d'eau
/// par kg d'air sec) ; renvoie la teneur en eau du mélange (kg d'eau par kg
/// d'air sec).
///
/// Panique si `mass_flow1 <= 0`, `mass_flow2 <= 0`, `humidity_ratio1 < 0` ou
/// `humidity_ratio2 < 0`.
pub fn airmix_humidity_ratio(
    mass_flow1: f64,
    humidity_ratio1: f64,
    mass_flow2: f64,
    humidity_ratio2: f64,
) -> f64 {
    assert!(
        mass_flow1 > 0.0,
        "débit d'air sec du premier flux strictement positif requis"
    );
    assert!(
        mass_flow2 > 0.0,
        "débit d'air sec du second flux strictement positif requis"
    );
    assert!(
        humidity_ratio1 >= 0.0,
        "teneur en eau du premier flux négative interdite"
    );
    assert!(
        humidity_ratio2 >= 0.0,
        "teneur en eau du second flux négative interdite"
    );
    (mass_flow1 * humidity_ratio1 + mass_flow2 * humidity_ratio2) / (mass_flow1 + mass_flow2)
}

/// Enthalpie spécifique du mélange `h = (m1·h1 + m2·h2) / (m1 + m2)`
/// (moyenne pondérée par les débits d'air sec).
///
/// `mass_flow1`, `mass_flow2` débits massiques d'air **sec** (kg·s⁻¹),
/// `enthalpy1`, `enthalpy2` enthalpies spécifiques des deux flux rapportées au
/// kg d'air sec (kJ·kg⁻¹) ; renvoie l'enthalpie spécifique du mélange
/// (kJ·kg⁻¹). L'enthalpie totale est conservée (mélange adiabatique).
///
/// Panique si `mass_flow1 <= 0` ou `mass_flow2 <= 0`.
pub fn airmix_enthalpy(mass_flow1: f64, enthalpy1: f64, mass_flow2: f64, enthalpy2: f64) -> f64 {
    assert!(
        mass_flow1 > 0.0,
        "débit d'air sec du premier flux strictement positif requis"
    );
    assert!(
        mass_flow2 > 0.0,
        "débit d'air sec du second flux strictement positif requis"
    );
    (mass_flow1 * enthalpy1 + mass_flow2 * enthalpy2) / (mass_flow1 + mass_flow2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mass_flow_out_conserves_dry_air() {
        // Conservation : m = m1 + m2 = 1,5 + 0,5 = 2,0 kg/s.
        let m = airmix_mass_flow_out(1.5, 0.5);
        assert_relative_eq!(m, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn equal_flows_give_arithmetic_mean_temperature() {
        // Débits égaux : le barycentre est la moyenne arithmétique.
        // T = (10 + 30) / 2 = 20 °C.
        let t = airmix_temperature(2.0, 10.0, 2.0, 30.0);
        assert_relative_eq!(t, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn temperature_matches_hand_calc() {
        // m1 = 1, T1 = 10 °C, m2 = 3, T2 = 30 °C :
        // T = (1·10 + 3·30) / (1 + 3) = (10 + 90) / 4 = 100 / 4 = 25 °C.
        let t = airmix_temperature(1.0, 10.0, 3.0, 30.0);
        assert_relative_eq!(t, 25.0, epsilon = 1e-12);
    }

    #[test]
    fn mixing_two_identical_states_returns_the_same_state() {
        // Barycentre de deux états identiques = cet état (teneur en eau).
        let w = airmix_humidity_ratio(1.2, 0.008, 3.4, 0.008);
        assert_relative_eq!(w, 0.008, epsilon = 1e-12);
    }

    #[test]
    fn mixing_is_symmetric_under_stream_swap() {
        // Échanger les deux flux ne change pas l'enthalpie du mélange.
        let h_ab = airmix_enthalpy(1.0, 30.0, 2.0, 60.0);
        let h_ba = airmix_enthalpy(2.0, 60.0, 1.0, 30.0);
        assert_relative_eq!(h_ab, h_ba, epsilon = 1e-12);
    }

    #[test]
    fn enthalpy_matches_hand_calc() {
        // m1 = 1, h1 = 30, m2 = 2, h2 = 60 kJ/kg :
        // h = (1·30 + 2·60) / (1 + 2) = (30 + 120) / 3 = 150 / 3 = 50 kJ/kg.
        let h = airmix_enthalpy(1.0, 30.0, 2.0, 60.0);
        assert_relative_eq!(h, 50.0, epsilon = 1e-12);
    }

    #[test]
    fn humidity_ratio_lies_between_the_two_inlets() {
        // Le barycentre est encadré par les deux teneurs en eau d'entrée.
        let (w1, w2) = (0.005_f64, 0.015_f64);
        let w = airmix_humidity_ratio(1.0, w1, 4.0, w2);
        assert!(w > w1 && w < w2);
        // Pondération vers le flux le plus abondant (m2 = 4·m1) :
        // w = (1·0,005 + 4·0,015) / 5 = (0,005 + 0,060) / 5 = 0,065 / 5 = 0,013.
        assert_relative_eq!(w, 0.013, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "débit d'air sec du premier flux strictement positif requis")]
    fn temperature_rejects_zero_mass_flow() {
        let _ = airmix_temperature(0.0, 10.0, 2.0, 30.0);
    }
}
