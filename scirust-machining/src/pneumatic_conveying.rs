//! Transport **pneumatique** en **phase diluée** — taux de charge, débit-masse
//! de gaz porteur, chute de pression d'accélération des solides et nombre de
//! Froude de saltation (seuil de dépôt en conduite horizontale).
//!
//! ```text
//! taux de charge     µ = ṁ_s / ṁ_g                (phase diluée si µ ⪅ 15)
//! débit-masse gaz    ṁ_g = ρ_g · A · U            (vitesse superficielle U)
//! Δp d'accélération  Δp_acc = ṁ_s · v_p / A       (mise en vitesse des solides)
//! Froude saltation   Fr_s = v_salt / √(g · D)     (seuil de dépôt horizontal)
//! ```
//!
//! `ṁ_s` débit-masse des solides (kg/s), `ṁ_g` débit-masse du gaz (kg/s),
//! `µ` taux de charge (sans dimension), `ρ_g` masse volumique du gaz (kg/m³),
//! `A` section de la conduite (m²), `U` vitesse superficielle du gaz (m/s),
//! `v_p` vitesse des particules (m/s), `v_salt` vitesse de saltation (m/s),
//! `D` diamètre de la conduite (m), `g` accélération de la pesanteur (m/s²),
//! `Δp_acc` chute de pression d'accélération (Pa), `Fr_s` nombre de Froude.
//!
//! **Convention** : unités SI cohérentes.
//! **Limite honnête** : modèle de **phase diluée** (particules en suspension) ;
//! débits, sections et vitesses sont **fournis** par l'appelant. La vitesse de
//! saltation/reprise (seuil de dépôt) provient d'une **corrélation ou d'un essai**
//! dépendant du produit et n'est **pas** calculée ici. La phase **dense**
//! (bouchons, dunes) n'est **pas** modélisée. Aucune masse volumique de gaz ni
//! constante de procédé « par défaut » n'est supposée.

/// Taux de charge (rapport solides/gaz) `µ = ṁ_s / ṁ_g`.
///
/// Phase diluée usuellement pour `µ` inférieur à ≈ 15.
///
/// Panique si `gas_mass_flow <= 0` ou `solids_mass_flow < 0`.
pub fn pneuconvey_solids_loading_ratio(solids_mass_flow: f64, gas_mass_flow: f64) -> f64 {
    assert!(
        gas_mass_flow > 0.0,
        "le débit-masse de gaz doit être strictement positif"
    );
    assert!(
        solids_mass_flow >= 0.0,
        "le débit-masse de solides doit être positif ou nul"
    );
    solids_mass_flow / gas_mass_flow
}

/// Débit-masse du gaz porteur `ṁ_g = ρ_g · A · U`.
///
/// Panique si un paramètre est `<= 0`.
pub fn pneuconvey_gas_mass_flow(
    gas_density: f64,
    pipe_area: f64,
    superficial_velocity: f64,
) -> f64 {
    assert!(
        gas_density > 0.0 && pipe_area > 0.0 && superficial_velocity > 0.0,
        "ρ_g, A et U doivent être strictement positifs"
    );
    gas_density * pipe_area * superficial_velocity
}

/// Chute de pression d'**accélération** des solides `Δp_acc = ṁ_s · v_p / A`.
///
/// Représente la pression nécessaire pour amener les particules à la vitesse
/// `v_p` depuis le repos dans la zone d'accélération.
///
/// Panique si `pipe_area <= 0`, `particle_velocity <= 0` ou
/// `solids_mass_flow < 0`.
pub fn pneuconvey_acceleration_pressure_drop(
    solids_mass_flow: f64,
    pipe_area: f64,
    particle_velocity: f64,
) -> f64 {
    assert!(
        pipe_area > 0.0 && particle_velocity > 0.0,
        "A et v_p doivent être strictement positifs"
    );
    assert!(
        solids_mass_flow >= 0.0,
        "le débit-masse de solides doit être positif ou nul"
    );
    solids_mass_flow * particle_velocity / pipe_area
}

/// Nombre de **Froude de saltation** `Fr_s = v_salt / √(g · D)`.
///
/// Caractérise le seuil de dépôt des particules en conduite **horizontale** ;
/// `v_salt` est fournie par corrélation ou essai (dépend du produit).
///
/// Panique si un paramètre est `<= 0`.
pub fn pneuconvey_saltation_froude(
    saltation_velocity: f64,
    pipe_diameter: f64,
    gravity: f64,
) -> f64 {
    assert!(
        saltation_velocity > 0.0 && pipe_diameter > 0.0 && gravity > 0.0,
        "v_salt, D et g doivent être strictement positifs"
    );
    saltation_velocity / (gravity * pipe_diameter).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn loading_ratio_dilute_case() {
        // ṁ_s = 4,8 kg/s, ṁ_g = 0,48 kg/s → µ = 10 (phase diluée, µ < 15).
        let mu = pneuconvey_solids_loading_ratio(4.8, 0.48);
        assert_relative_eq!(mu, 10.0, epsilon = 1e-12);
        assert!(mu < 15.0);
    }

    #[test]
    fn loading_ratio_reciprocity_with_gas_flow() {
        // µ · ṁ_g = ṁ_s : le produit du taux de charge par le débit gaz
        // restitue le débit solides.
        let (solids, gas) = (3.2_f64, 0.4_f64);
        let mu = pneuconvey_solids_loading_ratio(solids, gas);
        assert_relative_eq!(mu * gas, solids, epsilon = 1e-12);
    }

    #[test]
    fn gas_mass_flow_numeric() {
        // ρ_g = 1,2 kg/m³, A = 0,02 m², U = 20 m/s → ṁ_g = 0,48 kg/s.
        let m = pneuconvey_gas_mass_flow(1.2, 0.02, 20.0);
        assert_relative_eq!(m, 0.48, epsilon = 1e-12);
    }

    #[test]
    fn acceleration_pressure_drop_numeric() {
        // ṁ_s = 1,0 kg/s, A = 0,02 m², v_p = 15 m/s → Δp = 1·15/0,02 = 750 Pa.
        let dp = pneuconvey_acceleration_pressure_drop(1.0, 0.02, 15.0);
        assert_relative_eq!(dp, 750.0, epsilon = 1e-9);
    }

    #[test]
    fn acceleration_pressure_drop_proportional_to_velocity() {
        // Δp ∝ v_p à débit et section fixés : doubler v_p double Δp.
        let dp1 = pneuconvey_acceleration_pressure_drop(1.0, 0.02, 15.0);
        let dp2 = pneuconvey_acceleration_pressure_drop(1.0, 0.02, 30.0);
        assert_relative_eq!(dp2 / dp1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn saltation_froude_numeric() {
        // v_salt = 10 m/s, D = 0,1 m, g = 9,81 m/s² :
        // Fr_s = 10 / √(9,81·0,1) = 10 / √0,981 ≈ 10,0964.
        let fr = pneuconvey_saltation_froude(10.0, 0.1, 9.81);
        assert_relative_eq!(fr, 10.096_375_546_923, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "débit-masse de gaz")]
    fn zero_gas_flow_panics() {
        pneuconvey_solids_loading_ratio(4.8, 0.0);
    }
}
