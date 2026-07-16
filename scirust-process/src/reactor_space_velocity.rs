//! Vitesse spatiale et temps de contact d'un réacteur — rapports simples entre
//! débit d'alimentation et volume (ou masse) du réacteur/catalyseur : vitesse
//! spatiale, temps de passage `τ`, GHSV et WHSV.
//!
//! ```text
//! vitesse spatiale (inverse du temps de passage)
//!   SV = Q_in / V_r                                         [s⁻¹]
//! temps de passage (temps spatial)
//!   τ  = V_r / Q_in                                         [s]
//! vitesse spatiale horaire gazeuse (Gas Hourly Space Velocity)
//!   GHSV = Q_gaz,STP / V_cat                                [h⁻¹] (si Q en m³·h⁻¹)
//! vitesse spatiale horaire massique (Weight Hourly Space Velocity)
//!   WHSV = ṁ_in / m_cat                                     [h⁻¹] (si ṁ en kg·h⁻¹)
//! ```
//!
//! `Q_in` débit volumique **d'entrée** de l'alimentation [m³·s⁻¹], `V_r` volume
//! du réacteur [m³], `SV` vitesse spatiale [s⁻¹], `τ` temps de passage [s] ;
//! `Q_gaz,STP` débit volumique de gaz ramené aux **conditions normales** (STP)
//! [m³·h⁻¹], `V_cat` volume de catalyseur (ou du lit) [m³], `GHSV` [h⁻¹] ;
//! `ṁ_in` débit **massique** d'alimentation [kg·h⁻¹], `m_cat` masse de
//! catalyseur [kg], `WHSV` [h⁻¹].
//!
//! **Limite honnête** : la vitesse spatiale est simplement l'**inverse du temps
//! de passage** ; elle est bâtie sur le **débit d'ENTRÉE** aux conditions
//! spécifiées et ne tient **pas** compte de l'expansion (ou de la contraction)
//! volumique de la réaction dans le réacteur — ce n'est donc pas le temps de
//! séjour réel du fluide. Pour la **GHSV**, le débit gazeux doit être ramené aux
//! **conditions normales** (STP) par l'appelant ; pour la **WHSV**, c'est un
//! débit **massique**. Tous les **volumes, débits et masses** (réacteur,
//! catalyseur, alimentation) sont **FOURNIS** par l'appelant : aucune propriété
//! (masse volumique, enthalpie, constante cinétique, isotherme, volatilité…)
//! n'est calculée ni inventée par ce module.

/// Vitesse spatiale `SV = Q_in / V_r` (s⁻¹), inverse du temps de passage.
///
/// `volumetric_feed_rate` (Q_in) débit volumique d'entrée [m³·s⁻¹],
/// `reactor_volume` (V_r) volume du réacteur [m³].
///
/// Panique si `Q_in < 0` ou si `V_r ≤ 0`.
pub fn rsv_space_velocity(volumetric_feed_rate: f64, reactor_volume: f64) -> f64 {
    assert!(
        volumetric_feed_rate >= 0.0,
        "Q_in ≥ 0 requis (débit volumique d'entrée)"
    );
    assert!(reactor_volume > 0.0, "V_r > 0 requis (volume du réacteur)");
    volumetric_feed_rate / reactor_volume
}

/// Temps de passage `τ = V_r / Q_in` (s), inverse de la vitesse spatiale.
///
/// `reactor_volume` (V_r) volume du réacteur [m³], `volumetric_feed_rate` (Q_in)
/// débit volumique d'entrée [m³·s⁻¹].
///
/// Panique si `V_r < 0` ou si `Q_in ≤ 0`.
pub fn rsv_space_time(reactor_volume: f64, volumetric_feed_rate: f64) -> f64 {
    assert!(reactor_volume >= 0.0, "V_r ≥ 0 requis (volume du réacteur)");
    assert!(
        volumetric_feed_rate > 0.0,
        "Q_in > 0 requis (débit volumique d'entrée)"
    );
    reactor_volume / volumetric_feed_rate
}

/// Vitesse spatiale horaire gazeuse `GHSV = Q_gaz,STP / V_cat` (h⁻¹ si le débit
/// est exprimé en m³·h⁻¹), débit gazeux ramené aux conditions normales (STP).
///
/// `gas_volumetric_flow_stp` (Q_gaz,STP) débit volumique de gaz aux conditions
/// normales [m³·h⁻¹], `catalyst_volume` (V_cat) volume de catalyseur [m³].
///
/// Panique si `Q_gaz,STP < 0` ou si `V_cat ≤ 0`.
pub fn rsv_ghsv(gas_volumetric_flow_stp: f64, catalyst_volume: f64) -> f64 {
    assert!(
        gas_volumetric_flow_stp >= 0.0,
        "Q_gaz,STP ≥ 0 requis (débit gazeux aux conditions normales)"
    );
    assert!(
        catalyst_volume > 0.0,
        "V_cat > 0 requis (volume de catalyseur)"
    );
    gas_volumetric_flow_stp / catalyst_volume
}

/// Vitesse spatiale horaire massique `WHSV = ṁ_in / m_cat` (h⁻¹ si le débit est
/// exprimé en kg·h⁻¹), rapport du débit massique d'alimentation à la masse de
/// catalyseur.
///
/// `mass_feed_rate` (ṁ_in) débit massique d'alimentation [kg·h⁻¹],
/// `catalyst_mass` (m_cat) masse de catalyseur [kg].
///
/// Panique si `ṁ_in < 0` ou si `m_cat ≤ 0`.
pub fn rsv_whsv(mass_feed_rate: f64, catalyst_mass: f64) -> f64 {
    assert!(
        mass_feed_rate >= 0.0,
        "ṁ_in ≥ 0 requis (débit massique d'alimentation)"
    );
    assert!(
        catalyst_mass > 0.0,
        "m_cat > 0 requis (masse de catalyseur)"
    );
    mass_feed_rate / catalyst_mass
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn space_velocity_and_space_time_are_reciprocal() {
        // SV = 1/τ : la vitesse spatiale est exactement l'inverse du temps de
        // passage pour le même couple (Q_in, V_r).
        let (q, v) = (0.02_f64, 0.5_f64);
        let sv = rsv_space_velocity(q, v);
        let tau = rsv_space_time(v, q);
        assert_relative_eq!(sv * tau, 1.0, max_relative = 1e-12);
        assert_relative_eq!(sv, 1.0 / tau, max_relative = 1e-12);
    }

    #[test]
    fn space_time_realistic_case() {
        // Q_in = 0.02 m³/s, V_r = 0.5 m³ ⇒ τ = 0.5/0.02 = 25 s et
        // SV = 0.02/0.5 = 0.04 s⁻¹.
        assert_relative_eq!(rsv_space_time(0.5_f64, 0.02_f64), 25.0, max_relative = 1e-3);
        assert_relative_eq!(
            rsv_space_velocity(0.02_f64, 0.5_f64),
            0.04,
            max_relative = 1e-3
        );
    }

    #[test]
    fn ghsv_realistic_case() {
        // Q_gaz,STP = 1000 m³/h, V_cat = 2 m³ ⇒ GHSV = 1000/2 = 500 h⁻¹.
        assert_relative_eq!(rsv_ghsv(1000.0_f64, 2.0_f64), 500.0, max_relative = 1e-3);
    }

    #[test]
    fn whsv_realistic_case() {
        // ṁ_in = 120 kg/h, m_cat = 50 kg ⇒ WHSV = 120/50 = 2.4 h⁻¹.
        assert_relative_eq!(rsv_whsv(120.0_f64, 50.0_f64), 2.4, max_relative = 1e-3);
    }

    #[test]
    fn space_velocity_scales_linearly_with_feed_rate() {
        // SV ∝ Q_in à volume fixé : doubler le débit d'entrée double la vitesse
        // spatiale (et halve le temps de passage).
        let single = rsv_space_velocity(0.01_f64, 0.4_f64);
        let double = rsv_space_velocity(0.02_f64, 0.4_f64);
        assert_relative_eq!(double, 2.0 * single, max_relative = 1e-12);
    }

    #[test]
    fn zero_feed_rate_gives_zero_space_velocity() {
        // Q_in = 0 ⇒ aucune alimentation ⇒ vitesse spatiale nulle (cas limite).
        assert_relative_eq!(rsv_space_velocity(0.0_f64, 0.5_f64), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "V_r > 0 requis")]
    fn space_velocity_panics_on_zero_volume() {
        // V_r = 0 ⇒ division par zéro ⇒ entrée rejetée.
        let _ = rsv_space_velocity(0.02_f64, 0.0_f64);
    }
}
