//! **Bilan énergétique / enthalpique en régime permanent** — chaleur sensible,
//! chaleur latente, déséquilibre d'un bilan enthalpique global et température de
//! mélange adiabatique de deux courants.
//!
//! ```text
//! chaleur sensible   Q_s = m·cp·ΔT                                  (W)
//! chaleur latente    Q_l = m·L                                      (W)
//! bilan enthalpique  R   = Σ H_out − Σ H_in − Q + W                 (W)
//! mélange adiabat.   T   = (m1·cp1·T1 + m2·cp2·T2)/(m1·cp1 + m2·cp2) (K)
//! ```
//!
//! `m` débit massique (kg/s) ; `cp` capacité thermique massique (J·kg⁻¹·K⁻¹) ;
//! `ΔT` variation de température (K, signée) ; `Q_s` flux de chaleur sensible (W) ;
//! `L` chaleur latente de changement d'état (J/kg) ; `Q_l` flux de chaleur latente
//! (W) ; `H_in`/`H_out` flux enthalpiques entrant/sortant (W) ; `Q` chaleur ajoutée
//! au système (W) ; `W` travail fourni par le système (W) ; `R` résidu du bilan (W,
//! nul à l'équilibre) ; `T1`/`T2` températures des courants mélangés (K, absolues) ;
//! `T` température de mélange (K).
//!
//! **Convention** : SI (débits en kg/s, capacités en J·kg⁻¹·K⁻¹, chaleurs latentes
//! en J/kg, flux et puissances en W, températures en K). Signe du bilan : `ΔH = Q − W`
//! (première loi en système ouvert), d'où `R = Σ H_out − Σ H_in − Q + W`.
//!
//! **Limite honnête** : régime **permanent** ; les capacités thermiques (`cp`) et les
//! chaleurs latentes (`L`) sont **fournies par l'appelant** et supposées **constantes**
//! sur l'intervalle considéré (aucune valeur matériau « par défaut » n'est inventée) ;
//! les énergies cinétique et potentielle sont négligées sauf à être injectées via le
//! terme de travail ; la **référence d'enthalpie** est cohérente et fournie par
//! l'appelant. Le mélange adiabatique suppose l'absence de pertes thermiques, de
//! chaleur de mélange et de changement d'état.

/// Flux de chaleur sensible `Q_s = m·cp·ΔT` (W).
///
/// `ΔT` est signé : positif pour un échauffement, négatif pour un refroidissement.
///
/// Panique si `mass_flow < 0` ou si `specific_heat <= 0`.
pub fn enbal_sensible_heat(mass_flow: f64, specific_heat: f64, temperature_change: f64) -> f64 {
    assert!(mass_flow >= 0.0, "le débit massique doit être positif");
    assert!(
        specific_heat > 0.0,
        "la capacité thermique massique doit être strictement positive"
    );
    mass_flow * specific_heat * temperature_change
}

/// Flux de chaleur latente d'un changement d'état `Q_l = m·L` (W).
///
/// Panique si `mass_flow < 0` ou si `latent_heat < 0`.
pub fn enbal_latent_heat(mass_flow: f64, latent_heat: f64) -> f64 {
    assert!(mass_flow >= 0.0, "le débit massique doit être positif");
    assert!(
        latent_heat >= 0.0,
        "la chaleur latente de changement d'état doit être positive"
    );
    mass_flow * latent_heat
}

/// Résidu du bilan enthalpique global en régime permanent
/// `R = Σ H_out − Σ H_in − Q + W` (W), nul à l'équilibre.
///
/// Les tranches contiennent les flux enthalpiques (W) de chaque courant ; `heat_added`
/// est la chaleur reçue par le système et `work_done` le travail fourni par le système
/// (convention `ΔH = Q − W`).
///
/// Panique si `inlet_enthalpy_flows` ou `outlet_enthalpy_flows` est vide.
pub fn enbal_enthalpy_balance(
    inlet_enthalpy_flows: &[f64],
    outlet_enthalpy_flows: &[f64],
    heat_added: f64,
    work_done: f64,
) -> f64 {
    assert!(
        !inlet_enthalpy_flows.is_empty(),
        "il faut au moins un flux enthalpique entrant"
    );
    assert!(
        !outlet_enthalpy_flows.is_empty(),
        "il faut au moins un flux enthalpique sortant"
    );
    let inlet_sum: f64 = inlet_enthalpy_flows.iter().sum();
    let outlet_sum: f64 = outlet_enthalpy_flows.iter().sum();
    outlet_sum - inlet_sum - heat_added + work_done
}

/// Température de mélange adiabatique de deux courants
/// `T = (m1·cp1·T1 + m2·cp2·T2)/(m1·cp1 + m2·cp2)` (K).
///
/// Panique si un débit est négatif, si une capacité thermique n'est pas strictement
/// positive, si une température n'est pas strictement positive (K absolus), ou si le
/// débit thermique total `m1·cp1 + m2·cp2` est nul (deux débits nuls).
pub fn enbal_adiabatic_mixing_temperature(
    m1: f64,
    cp1: f64,
    t1: f64,
    m2: f64,
    cp2: f64,
    t2: f64,
) -> f64 {
    assert!(
        m1 >= 0.0,
        "le débit massique du courant 1 doit être positif"
    );
    assert!(
        m2 >= 0.0,
        "le débit massique du courant 2 doit être positif"
    );
    assert!(
        cp1 > 0.0,
        "la capacité thermique du courant 1 doit être strictement positive"
    );
    assert!(
        cp2 > 0.0,
        "la capacité thermique du courant 2 doit être strictement positive"
    );
    assert!(
        t1 > 0.0,
        "la température du courant 1 doit être strictement positive (K absolus)"
    );
    assert!(
        t2 > 0.0,
        "la température du courant 2 doit être strictement positive (K absolus)"
    );
    let capacity_flow_1 = m1 * cp1;
    let capacity_flow_2 = m2 * cp2;
    let total_capacity_flow = capacity_flow_1 + capacity_flow_2;
    assert!(
        total_capacity_flow > 0.0,
        "le débit thermique total doit être strictement positif (au moins un débit non nul)"
    );
    (capacity_flow_1 * t1 + capacity_flow_2 * t2) / total_capacity_flow
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sensible_heat_reciprocity_between_heating_and_cooling() {
        // Chauffer un courant de T1 à T2 puis le ramener de T2 à T1 : flux opposés.
        let heating = enbal_sensible_heat(2.0, 4180.0, 60.0);
        let cooling = enbal_sensible_heat(2.0, 4180.0, -60.0);
        assert_relative_eq!(heating, -cooling, epsilon = 1e-9);
        // Cas chiffré : 2 kg/s · 4180 J·kg⁻¹·K⁻¹ · 60 K = 501 600 W.
        assert_relative_eq!(heating, 501_600.0, epsilon = 1e-6);
    }

    #[test]
    fn latent_heat_is_proportional_to_mass_flow() {
        // Q_l = m·L : doubler le débit double le flux latent.
        let q1 = enbal_latent_heat(0.5, 2_260_000.0);
        let q2 = enbal_latent_heat(1.0, 2_260_000.0);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-6);
        // Cas chiffré : 0.5 kg/s · 2 260 000 J/kg = 1 130 000 W.
        assert_relative_eq!(q1, 1_130_000.0, epsilon = 1e-6);
    }

    #[test]
    fn enthalpy_balance_is_zero_at_equilibrium() {
        // Entrées = 100 + 200 = 300 W ; Q = 50 W ; W = 30 W.
        // Équilibre : sortie = 300 + 50 − 30 = 320 W  ⇒  R = 320 − 300 − 50 + 30 = 0.
        let residual = enbal_enthalpy_balance(&[100.0, 200.0], &[320.0], 50.0, 30.0);
        assert_relative_eq!(residual, 0.0, epsilon = 1e-9);
        // Un déséquilibre se reporte à l'identique : sortie de 400 au lieu de 320.
        let unbalanced = enbal_enthalpy_balance(&[100.0, 200.0], &[400.0], 50.0, 30.0);
        assert_relative_eq!(unbalanced, 80.0, epsilon = 1e-9);
    }

    #[test]
    fn mixing_of_equal_capacity_flows_gives_midpoint() {
        // Débits thermiques égaux (m·cp identiques) : T = (T1 + T2)/2.
        let t = enbal_adiabatic_mixing_temperature(1.0, 4180.0, 300.0, 1.0, 4180.0, 360.0);
        assert_relative_eq!(t, 330.0, epsilon = 1e-9);
    }

    #[test]
    fn mixing_realistic_weighted_average() {
        // Courant 1 : 2 kg/s, cp 1000, 400 K → 2·1000·400 = 800 000.
        // Courant 2 : 1 kg/s, cp 2000, 300 K → 1·2000·300 = 600 000.
        // Débit thermique total = 2000 + 2000 = 4000.
        // T = 1 400 000 / 4000 = 350 K, borné entre 300 K et 400 K.
        let t = enbal_adiabatic_mixing_temperature(2.0, 1000.0, 400.0, 1.0, 2000.0, 300.0);
        assert_relative_eq!(t, 350.0, epsilon = 1e-9);
        assert!((300.0..=400.0).contains(&t));
    }

    #[test]
    #[should_panic(expected = "débit massique")]
    fn sensible_heat_rejects_negative_mass_flow() {
        enbal_sensible_heat(-1.0, 4180.0, 10.0);
    }
}
