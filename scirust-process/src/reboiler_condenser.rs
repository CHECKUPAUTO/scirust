//! Charges thermiques de rebouilleur et de condenseur — dimensionnement des
//! échangeurs de tête et de fond d'une colonne de distillation : chaleur à
//! évacuer au condenseur, chaleur à fournir au rebouilleur, consommation de
//! vapeur vive, débit d'eau de refroidissement et surface d'échange requise.
//!
//! ```text
//! charge du condenseur (total)     Q_c = V · λ                    [W]
//! charge du rebouilleur (total)    Q_r = V' · λ'                  [W]
//! consommation de vapeur vive      S   = Q_r / λ_s                [kg·s⁻¹]
//! débit d'eau de refroidissement   m_w = Q_c / (c_p · ΔT_w)       [kg·s⁻¹]
//! surface du rebouilleur           A   = Q_r / (U · ΔT_lm)        [m²]
//! ```
//!
//! `V` débit de vapeur montant en tête, condensé en totalité [kg·s⁻¹], `λ`
//! chaleur latente de condensation du distillat [J·kg⁻¹], `Q_c` charge du
//! condenseur [W] ; `V'` débit de vapeur (« boil-up ») produit au fond [kg·s⁻¹],
//! `λ'` chaleur latente de vaporisation du résidu [J·kg⁻¹], `Q_r` charge du
//! rebouilleur [W] ; `λ_s` chaleur latente de la vapeur vive de chauffe
//! [J·kg⁻¹], `S` débit de vapeur vive [kg·s⁻¹] ; `c_p` capacité thermique
//! massique de l'eau de refroidissement [J·kg⁻¹·K⁻¹], `ΔT_w` échauffement de
//! l'eau [K], `m_w` débit d'eau de refroidissement [kg·s⁻¹] ; `U` coefficient
//! global d'échange [W·m⁻²·K⁻¹], `ΔT_lm` écart de température moyen (moteur
//! thermique) [K], `A` surface d'échange [m²].
//!
//! **Limite honnête** : les débits de vapeur (`V`, `V'`), les chaleurs latentes
//! (`λ`, `λ'`, `λ_s`), la capacité thermique (`c_p`), le coefficient global
//! d'échange (`U`) et l'écart moteur (`ΔT_lm`, à établir par ailleurs, p. ex. par
//! LMTD) sont **FOURNIS** par l'appelant d'après des tables, des corrélations ou
//! des essais : aucune propriété physique ni aucun coefficient de transfert
//! n'est inventé ici. On suppose un **condenseur et un rebouilleur totaux**
//! (changement de phase complet, isobare, à la chaleur latente, sans
//! sous-refroidissement ni surchauffe), en **régime permanent** et **pertes
//! thermiques négligées**. Ce module complète les modules de distillation
//! (`distillation_mccabe`, `flash_distillation`…) qui fournissent les débits de
//! vapeur ; il ne recalcule pas les propriétés d'état (cf. `scirust-thermo`).

/// Charge du condenseur total `Q_c = V · λ` (W) : chaleur à évacuer pour
/// condenser en totalité le débit de vapeur de tête.
///
/// `vapor_flow` (V) débit de vapeur montant en tête [kg·s⁻¹], `latent_heat`
/// (λ) chaleur latente de condensation du distillat [J·kg⁻¹].
///
/// Panique si `vapor_flow < 0` ou si `latent_heat <= 0`.
pub fn rbc_condenser_duty(vapor_flow: f64, latent_heat: f64) -> f64 {
    assert!(vapor_flow >= 0.0, "V ≥ 0 requis (débit de vapeur de tête)");
    assert!(
        latent_heat > 0.0,
        "λ > 0 requis (chaleur latente de condensation)"
    );
    vapor_flow * latent_heat
}

/// Charge du rebouilleur total `Q_r = V' · λ'` (W) : chaleur à fournir pour
/// vaporiser en totalité le débit de « boil-up » au fond de colonne.
///
/// `boilup_flow` (V') débit de vapeur produit au fond [kg·s⁻¹], `latent_heat`
/// (λ') chaleur latente de vaporisation du résidu [J·kg⁻¹].
///
/// Panique si `boilup_flow < 0` ou si `latent_heat <= 0`.
pub fn rbc_reboiler_duty(boilup_flow: f64, latent_heat: f64) -> f64 {
    assert!(boilup_flow >= 0.0, "V' ≥ 0 requis (débit de boil-up)");
    assert!(
        latent_heat > 0.0,
        "λ' > 0 requis (chaleur latente de vaporisation)"
    );
    boilup_flow * latent_heat
}

/// Consommation de vapeur vive de chauffe `S = Q_r / λ_s` (kg·s⁻¹) : débit de
/// vapeur vive dont la condensation apporte la charge du rebouilleur.
///
/// `reboiler_duty` (Q_r) charge du rebouilleur [W], `steam_latent_heat` (λ_s)
/// chaleur latente de la vapeur vive à sa pression de chauffe [J·kg⁻¹].
///
/// Panique si `reboiler_duty < 0` ou si `steam_latent_heat <= 0`.
pub fn rbc_steam_consumption(reboiler_duty: f64, steam_latent_heat: f64) -> f64 {
    assert!(
        reboiler_duty >= 0.0,
        "Q_r ≥ 0 requis (charge du rebouilleur)"
    );
    assert!(
        steam_latent_heat > 0.0,
        "λ_s > 0 requis (chaleur latente de la vapeur vive)"
    );
    reboiler_duty / steam_latent_heat
}

/// Débit d'eau de refroidissement `m_w = Q_c / (c_p · ΔT_w)` (kg·s⁻¹) : débit
/// d'eau nécessaire pour absorber la charge du condenseur en s'échauffant de
/// `ΔT_w` en chaleur sensible.
///
/// `condenser_duty` (Q_c) charge du condenseur [W], `water_heat_capacity` (c_p)
/// capacité thermique massique de l'eau [J·kg⁻¹·K⁻¹], `temperature_rise` (ΔT_w)
/// échauffement admis de l'eau [K].
///
/// Panique si `condenser_duty < 0`, si `water_heat_capacity <= 0` ou si
/// `temperature_rise <= 0`.
pub fn rbc_cooling_water_flow(
    condenser_duty: f64,
    water_heat_capacity: f64,
    temperature_rise: f64,
) -> f64 {
    assert!(
        condenser_duty >= 0.0,
        "Q_c ≥ 0 requis (charge du condenseur)"
    );
    assert!(
        water_heat_capacity > 0.0,
        "c_p > 0 requis (capacité thermique de l'eau)"
    );
    assert!(
        temperature_rise > 0.0,
        "ΔT_w > 0 requis (échauffement de l'eau)"
    );
    condenser_duty / (water_heat_capacity * temperature_rise)
}

/// Surface d'échange du rebouilleur `A = Q / (U · ΔT_lm)` (m²) : surface requise
/// pour transférer la charge `Q` sous un écart moteur `ΔT_lm` et un coefficient
/// global `U`.
///
/// `duty` (Q) charge thermique à transférer [W], `overall_coefficient` (U)
/// coefficient global d'échange [W·m⁻²·K⁻¹], `temperature_difference` (ΔT_lm)
/// écart de température moteur [K].
///
/// Panique si `duty < 0`, si `overall_coefficient <= 0` ou si
/// `temperature_difference <= 0`.
pub fn rbc_reboiler_area(duty: f64, overall_coefficient: f64, temperature_difference: f64) -> f64 {
    assert!(duty >= 0.0, "Q ≥ 0 requis (charge thermique)");
    assert!(
        overall_coefficient > 0.0,
        "U > 0 requis (coefficient global d'échange)"
    );
    assert!(
        temperature_difference > 0.0,
        "ΔT_lm > 0 requis (écart de température moteur)"
    );
    duty / (overall_coefficient * temperature_difference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn condenser_duty_scales_linearly_with_vapor_flow() {
        // Q_c = V · λ : proportionnel à V à λ fixé.
        let lambda = 2.0e6_f64;
        let q1 = rbc_condenser_duty(2.0_f64, lambda);
        let q2 = rbc_condenser_duty(4.0_f64, lambda);
        // V = 2, λ = 2e6 ⇒ Q_c = 4.0e6 W (recalcul : 2·2e6 = 4e6).
        assert_relative_eq!(q1, 4.0e6, max_relative = 1e-12);
        assert_relative_eq!(q2, 2.0 * q1, max_relative = 1e-12);
    }

    #[test]
    fn condenser_and_reboiler_share_the_same_form() {
        // Q = débit · λ : les deux charges coïncident pour mêmes arguments.
        let (flow, lambda) = (3.0_f64, 1.5e6_f64);
        assert_relative_eq!(
            rbc_condenser_duty(flow, lambda),
            rbc_reboiler_duty(flow, lambda),
            max_relative = 1e-12
        );
        // 3 · 1.5e6 = 4.5e6 W.
        assert_relative_eq!(rbc_reboiler_duty(flow, lambda), 4.5e6, max_relative = 1e-12);
    }

    #[test]
    fn steam_consumption_inverts_reboiler_duty() {
        // S = Q_r / λ_s ; réciproquement S · λ_s = Q_r.
        let (duty, lambda_s) = (4.0e6_f64, 2.0e6_f64);
        let s = rbc_steam_consumption(duty, lambda_s);
        // 4e6 / 2e6 = 2.0 kg/s.
        assert_relative_eq!(s, 2.0, max_relative = 1e-12);
        assert_relative_eq!(s * lambda_s, duty, max_relative = 1e-12);
    }

    #[test]
    fn cooling_water_flow_worked_case() {
        // Q_c = 4.18e6 W, c_p = 4180 J/kg/K, ΔT_w = 10 K
        //   ⇒ m_w = 4.18e6 / (4180 · 10) = 4.18e6 / 41800 = 100.0 kg/s.
        let m_w = rbc_cooling_water_flow(4.18e6_f64, 4180.0_f64, 10.0_f64);
        assert_relative_eq!(m_w, 100.0, max_relative = 1e-12);
        // Bilan de chaleur sensible : m_w · c_p · ΔT_w = Q_c.
        assert_relative_eq!(m_w * 4180.0 * 10.0, 4.18e6, max_relative = 1e-12);
    }

    #[test]
    fn reboiler_area_worked_case_and_inverse_in_u() {
        // Q = 1.0e6 W, U = 1000 W/m²/K, ΔT_lm = 20 K
        //   ⇒ A = 1e6 / (1000 · 20) = 1e6 / 20000 = 50.0 m².
        let a = rbc_reboiler_area(1.0e6_f64, 1000.0_f64, 20.0_f64);
        assert_relative_eq!(a, 50.0, max_relative = 1e-12);
        // Doubler U à charge et écart fixés divise la surface par deux.
        let a2 = rbc_reboiler_area(1.0e6_f64, 2000.0_f64, 20.0_f64);
        assert_relative_eq!(a2, a / 2.0, max_relative = 1e-12);
    }

    #[test]
    fn zero_flow_gives_zero_duty() {
        // Débit nul admis ⇒ charge nulle (borne physique).
        assert_relative_eq!(
            rbc_condenser_duty(0.0_f64, 2.0e6_f64),
            0.0,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "ΔT_w > 0 requis")]
    fn cooling_water_flow_panics_on_zero_temperature_rise() {
        // Échauffement nul ⇒ division par zéro physique ⇒ panique.
        let _ = rbc_cooling_water_flow(4.18e6_f64, 4180.0_f64, 0.0_f64);
    }
}
