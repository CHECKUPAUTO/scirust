//! Intégration énergétique — analyse du pincement (méthode du tableau des
//! problèmes) : charge thermique et débit de capacité thermique d'un courant,
//! températures décalées de ±ΔTmin/2, et besoin minimal d'utilité chaude déduit
//! de la récupération maximale.
//!
//! ```text
//! charge thermique d'un courant   Q      = CP · ΔT = ṁ · cp · ΔT   [W]
//! débit de capacité thermique     CP     = ṁ · cp                  [W·K⁻¹]
//! température décalée (chaud)      T*_h   = T − ΔTmin / 2           [K]
//! température décalée (froid)      T*_c   = T + ΔTmin / 2           [K]
//! utilité chaude minimale          Q_hu   = Q_c,tot − Q_rec,max     [W]
//! ```
//!
//! `ṁ` débit massique du courant [kg·s⁻¹], `cp` chaleur massique du courant
//! [J·kg⁻¹·K⁻¹], `ΔT` variation de température subie par le courant
//! [K, valeur ≥ 0], `Q` charge thermique du courant [W] ; `CP` débit de capacité
//! thermique [W·K⁻¹] ; `T` température réelle du courant [K], `ΔTmin` écart minimal
//! d'approche imposé [K, ≥ 0], `T*_h`/`T*_c` températures décalées d'un courant
//! chaud / froid [K] ; `Q_c,tot` charge froide totale à satisfaire [W],
//! `Q_rec,max` récupération thermique maximale déduite de la cascade [W],
//! `Q_hu` besoin minimal d'utilité chaude [W].
//!
//! **Limite honnête** : ce module ne fournit que les **briques élémentaires** de
//! la méthode du **pincement** (tableau des problèmes). Les **débits de capacité
//! thermique** `CP` sont supposés **CONSTANTS** par courant (pas de changement de
//! phase ni de dépendance de `cp` à la température) ; les chaleurs massiques `cp`
//! et l'écart minimal d'approche `ΔTmin` sont **FOURNIS** par l'appelant, jamais
//! inventés. Les températures décalées de ±ΔTmin/2 rendent chaud et froid
//! comparables et permettent de bâtir la **cascade de chaleur** ; la
//! **récupération maximale** `Q_rec,max` (et donc la position du pincement) se
//! déduit de cette cascade, **à construire par l'appelant** — elle n'est pas
//! calculée ici. Aucune propriété physique (enthalpies, chaleurs latentes,
//! coefficient global d'échange, surface…) n'est estimée.

/// Charge thermique d'un courant `Q = ṁ · cp · ΔT` (W), où `ΔT` est la
/// **valeur** de la variation de température subie (échauffement ou
/// refroidissement), prise ≥ 0.
///
/// `mass_flow` (ṁ) débit massique [kg·s⁻¹], `specific_heat` (cp) chaleur massique
/// [J·kg⁻¹·K⁻¹], `temperature_change` (ΔT) variation de température [K, ≥ 0].
///
/// Panique si `mass_flow < 0`, si `specific_heat < 0`, ou si
/// `temperature_change < 0`.
pub fn pinch_stream_heat_duty(mass_flow: f64, specific_heat: f64, temperature_change: f64) -> f64 {
    assert!(mass_flow >= 0.0, "ṁ ≥ 0 requis (débit massique)");
    assert!(specific_heat >= 0.0, "cp ≥ 0 requis (chaleur massique)");
    assert!(
        temperature_change >= 0.0,
        "ΔT ≥ 0 requis (variation de température en valeur)"
    );
    mass_flow * specific_heat * temperature_change
}

/// Débit de capacité thermique d'un courant `CP = ṁ · cp` (W·K⁻¹), supposé
/// constant sur la plage de température du courant.
///
/// `mass_flow` (ṁ) débit massique [kg·s⁻¹], `specific_heat` (cp) chaleur massique
/// [J·kg⁻¹·K⁻¹].
///
/// Panique si `mass_flow < 0` ou si `specific_heat < 0`.
pub fn pinch_heat_capacity_flowrate(mass_flow: f64, specific_heat: f64) -> f64 {
    assert!(mass_flow >= 0.0, "ṁ ≥ 0 requis (débit massique)");
    assert!(specific_heat >= 0.0, "cp ≥ 0 requis (chaleur massique)");
    mass_flow * specific_heat
}

/// Température décalée d'un courant **chaud** `T*_h = T − ΔTmin / 2` (K),
/// abaissée de la demi-approche pour rendre chauds et froids comparables dans le
/// tableau des problèmes.
///
/// `actual_temperature` (T) température réelle du courant [K, > 0],
/// `minimum_approach` (ΔTmin) écart minimal d'approche [K, ≥ 0].
///
/// Panique si `actual_temperature <= 0` ou si `minimum_approach < 0`.
pub fn pinch_shifted_temperature_hot(actual_temperature: f64, minimum_approach: f64) -> f64 {
    assert!(
        actual_temperature > 0.0,
        "T > 0 K requis (température absolue du courant chaud)"
    );
    assert!(
        minimum_approach >= 0.0,
        "ΔTmin ≥ 0 requis (écart minimal d'approche)"
    );
    actual_temperature - minimum_approach / 2.0
}

/// Température décalée d'un courant **froid** `T*_c = T + ΔTmin / 2` (K),
/// relevée de la demi-approche pour rendre chauds et froids comparables dans le
/// tableau des problèmes.
///
/// `actual_temperature` (T) température réelle du courant [K, > 0],
/// `minimum_approach` (ΔTmin) écart minimal d'approche [K, ≥ 0].
///
/// Panique si `actual_temperature <= 0` ou si `minimum_approach < 0`.
pub fn pinch_shifted_temperature_cold(actual_temperature: f64, minimum_approach: f64) -> f64 {
    assert!(
        actual_temperature > 0.0,
        "T > 0 K requis (température absolue du courant froid)"
    );
    assert!(
        minimum_approach >= 0.0,
        "ΔTmin ≥ 0 requis (écart minimal d'approche)"
    );
    actual_temperature + minimum_approach / 2.0
}

/// Besoin minimal d'utilité chaude `Q_hu = Q_c,tot − Q_rec,max` (W), une fois la
/// récupération thermique maximale déterminée par la cascade de chaleur.
///
/// `total_cold_duty` (Q_c,tot) charge froide totale à satisfaire [W, ≥ 0],
/// `total_hot_duty` (Q_h,tot) charge chaude totale disponible [W, ≥ 0],
/// `maximum_heat_recovery` (Q_rec,max) récupération maximale déduite de la cascade
/// [W, ≥ 0]. La récupération ne peut excéder ni la charge froide ni la charge
/// chaude ; elle est donc bornée par leur minimum.
///
/// Panique si l'une des charges est négative, si `maximum_heat_recovery < 0`, ou
/// si `maximum_heat_recovery` dépasse `total_cold_duty` ou `total_hot_duty`.
pub fn pinch_minimum_hot_utility(
    total_cold_duty: f64,
    total_hot_duty: f64,
    maximum_heat_recovery: f64,
) -> f64 {
    assert!(
        total_cold_duty >= 0.0,
        "Q_c,tot ≥ 0 requis (charge froide totale)"
    );
    assert!(
        total_hot_duty >= 0.0,
        "Q_h,tot ≥ 0 requis (charge chaude totale)"
    );
    assert!(
        maximum_heat_recovery >= 0.0,
        "Q_rec,max ≥ 0 requis (récupération maximale)"
    );
    assert!(
        maximum_heat_recovery <= total_cold_duty,
        "Q_rec,max ≤ Q_c,tot requis (récupération bornée par la charge froide)"
    );
    assert!(
        maximum_heat_recovery <= total_hot_duty,
        "Q_rec,max ≤ Q_h,tot requis (récupération bornée par la charge chaude)"
    );
    total_cold_duty - maximum_heat_recovery
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn heat_duty_factors_through_capacity_flowrate() {
        // Q = CP · ΔT avec CP = ṁ · cp : les deux briques doivent concorder.
        let (m, cp, dt) = (2.0_f64, 3500.0_f64, 25.0_f64);
        let cp_flow = pinch_heat_capacity_flowrate(m, cp);
        let q = pinch_stream_heat_duty(m, cp, dt);
        assert_relative_eq!(q, cp_flow * dt, max_relative = 1e-12);
    }

    #[test]
    fn heat_duty_scales_linearly_with_mass_flow() {
        // Q ∝ ṁ : doubler le débit double la charge thermique.
        let (cp, dt) = (2000.0_f64, 40.0_f64);
        let q1 = pinch_stream_heat_duty(1.5_f64, cp, dt);
        let q2 = pinch_stream_heat_duty(3.0_f64, cp, dt);
        assert_relative_eq!(q2, 2.0 * q1, max_relative = 1e-12);
    }

    #[test]
    fn heat_duty_realistic_water_stream() {
        // ṁ = 2 kg/s, cp = 4180 J/kg/K, ΔT = 30 K
        //   ⇒ Q = 2 · 4180 · 30 = 250 800 W.
        let q = pinch_stream_heat_duty(2.0_f64, 4180.0_f64, 30.0_f64);
        assert_relative_eq!(q, 250_800.0, max_relative = 1e-9);
    }

    #[test]
    fn shifted_temperatures_span_minimum_approach() {
        // Pour une même température réelle, l'écart froid − chaud vaut ΔTmin,
        // et leur moyenne redonne la température réelle.
        let (t, dtmin) = (400.0_f64, 10.0_f64);
        let th = pinch_shifted_temperature_hot(t, dtmin);
        let tc = pinch_shifted_temperature_cold(t, dtmin);
        assert_relative_eq!(tc - th, dtmin, max_relative = 1e-12);
        assert_relative_eq!(0.5 * (th + tc), t, max_relative = 1e-12);
    }

    #[test]
    fn shifted_temperatures_collapse_when_approach_zero() {
        // ΔTmin = 0 ⇒ aucune décalage : T*_h = T*_c = T.
        let t = 350.0_f64;
        assert_relative_eq!(
            pinch_shifted_temperature_hot(t, 0.0_f64),
            t,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            pinch_shifted_temperature_cold(t, 0.0_f64),
            t,
            max_relative = 1e-12
        );
    }

    #[test]
    fn minimum_hot_utility_from_recovery() {
        // Q_c,tot = 1000 kW, Q_h,tot = 800 kW, Q_rec,max = 600 kW
        //   ⇒ Q_hu = 1000 − 600 = 400 kW. Récupération totale (rec = cold)
        //   annulerait le besoin d'utilité chaude.
        let q_hu = pinch_minimum_hot_utility(1000.0_f64, 800.0_f64, 600.0_f64);
        assert_relative_eq!(q_hu, 400.0, max_relative = 1e-12);
        assert_relative_eq!(
            pinch_minimum_hot_utility(1000.0_f64, 1000.0_f64, 1000.0_f64),
            0.0,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "ṁ ≥ 0 requis")]
    fn heat_duty_rejects_negative_mass_flow() {
        let _ = pinch_stream_heat_duty(-1.0_f64, 4180.0_f64, 30.0_f64);
    }
}
