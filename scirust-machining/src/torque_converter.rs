//! **Convertisseur de couple hydrodynamique** (multiplication de couple) —
//! rapport de couple, rapport de vitesse, rendement et facteur de capacité `K`
//! d'un convertisseur à réacteur (stator), en régime permanent, à partir des
//! couples et vitesses de la pompe (impulseur) et de la turbine.
//!
//! ```text
//! rapport de couple   TR = T_turbine / T_pompe
//! rapport de vitesse  SR = N_turbine / N_pompe
//! rendement           η  = TR · SR
//! facteur de capacité K  = N_pompe / sqrt(T_pompe)
//! ```
//!
//! `T_pompe` couple absorbé par la pompe / impulseur (N·m), `T_turbine` couple
//! délivré par la turbine (N·m), `N_pompe` vitesse de rotation de la pompe
//! (tr/min ou rad/s, unité cohérente), `N_turbine` vitesse de rotation de la
//! turbine (même unité que `N_pompe`), `TR` rapport de couple (sans unité,
//! supérieur à 1 au calage grâce au réacteur), `SR` rapport de vitesse (sans
//! unité, 0 ≤ SR ≤ 1), `η` rendement (sans unité, 0 ≤ η ≤ 1), `K` facteur de
//! capacité (unité cohérente `[N]/sqrt([N·m])`).
//!
//! **Convention** : unités SI cohérentes. Le réacteur (**stator**) renvoie le
//! fluide vers la pompe et permet un rapport de couple `TR > 1` au calage —
//! contrairement à [`crate::fluid_coupling`] où `TR = 1`. Le rendement s'annule
//! au calage (`SR = 0`) comme en roue libre (`TR → 1`, la roue libre du stator
//! transformant le convertisseur en simple coupleur).
//!
//! **Limite honnête** : modèle de **régime permanent** (pas de transitoire ni
//! d'inertie). Les couples et vitesses de la pompe et de la turbine sont des
//! données **fournies par l'appelant** (elles dépendent du point de
//! fonctionnement, de la géométrie et du remplissage) ; aucune constante
//! physique, matériau ou procédé n'est supposée « par défaut ».

/// Rapport de couple `TR = T_turbine / T_pompe` du convertisseur (> 1 au calage
/// grâce à la réaction du stator).
///
/// Panique si `pump_torque <= 0` ou si `turbine_torque < 0`.
pub fn tc_torque_ratio(turbine_torque: f64, pump_torque: f64) -> f64 {
    assert!(
        turbine_torque >= 0.0,
        "le couple de turbine T_turbine ne peut pas être négatif"
    );
    assert!(
        pump_torque > 0.0,
        "le couple de pompe T_pompe doit être strictement positif"
    );
    turbine_torque / pump_torque
}

/// Rapport de vitesse `SR = N_turbine / N_pompe` du convertisseur.
///
/// Panique si `pump_speed <= 0` ou si `turbine_speed < 0`.
pub fn tc_speed_ratio(turbine_speed: f64, pump_speed: f64) -> f64 {
    assert!(
        turbine_speed >= 0.0,
        "la vitesse de turbine N_turbine ne peut pas être négative"
    );
    assert!(
        pump_speed > 0.0,
        "la vitesse de pompe N_pompe doit être strictement positive"
    );
    turbine_speed / pump_speed
}

/// Rendement `η = TR · SR` du convertisseur (produit du rapport de couple par le
/// rapport de vitesse) ; nul au calage (`SR = 0`).
///
/// Panique si `torque_ratio < 0` ou si `speed_ratio < 0`.
pub fn tc_efficiency(torque_ratio: f64, speed_ratio: f64) -> f64 {
    assert!(
        torque_ratio >= 0.0,
        "le rapport de couple TR ne peut pas être négatif"
    );
    assert!(
        speed_ratio >= 0.0,
        "le rapport de vitesse SR ne peut pas être négatif"
    );
    torque_ratio * speed_ratio
}

/// Facteur de capacité `K = N_pompe / sqrt(T_pompe)` (aptitude du convertisseur
/// à absorber le couple à une vitesse de pompe donnée).
///
/// Panique si `pump_torque <= 0` ou si `pump_speed < 0`.
pub fn tc_capacity_factor(pump_speed: f64, pump_torque: f64) -> f64 {
    assert!(
        pump_speed >= 0.0,
        "la vitesse de pompe N_pompe ne peut pas être négative"
    );
    assert!(
        pump_torque > 0.0,
        "le couple de pompe T_pompe doit être strictement positif"
    );
    pump_speed / pump_torque.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn torque_ratio_greater_than_one_at_stall() {
        // Au calage, T_turbine > T_pompe grâce au réacteur : TR > 1.
        let tr = tc_torque_ratio(420.0, 200.0);
        assert_relative_eq!(tr, 2.1, epsilon = 1e-12);
        assert!(tr > 1.0);
    }

    #[test]
    fn efficiency_is_product_of_ratios() {
        // η = TR · SR : identité de définition.
        let (tr, sr) = (1.05, 0.80);
        assert_relative_eq!(tc_efficiency(tr, sr), tr * sr, epsilon = 1e-12);
        // Point de fonctionnement réaliste : 0,84.
        assert_relative_eq!(tc_efficiency(tr, sr), 0.84, epsilon = 1e-12);
    }

    #[test]
    fn efficiency_vanishes_at_stall() {
        // Au calage la turbine est immobile (SR = 0) : η = TR · 0 = 0.
        assert_relative_eq!(tc_efficiency(2.1, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn capacity_factor_realistic_case() {
        // N_pompe = 2000 (tr/min) ; T_pompe = 400 N·m :
        // K = 2000 / sqrt(400) = 2000 / 20 = 100.
        let k = tc_capacity_factor(2000.0, 400.0);
        assert_relative_eq!(k, 100.0, epsilon = 1e-9);
    }

    #[test]
    fn capacity_factor_scales_with_pump_speed() {
        // K ∝ N_pompe à couple fixé : doubler N double K.
        let t = 250.0;
        let k1 = tc_capacity_factor(1500.0, t);
        let k2 = tc_capacity_factor(3000.0, t);
        assert_relative_eq!(k2, 2.0 * k1, epsilon = 1e-9);
    }

    #[test]
    fn speed_ratio_reciprocity_with_efficiency() {
        // À partir de couples et vitesses cohérents, on retrouve η par les deux
        // voies : η = TR · SR avec TR = T_t/T_p et SR = N_t/N_p.
        let (t_turbine, t_pump) = (240.0, 200.0);
        let (n_turbine, n_pump) = (1600.0, 2000.0);
        let tr = tc_torque_ratio(t_turbine, t_pump);
        let sr = tc_speed_ratio(n_turbine, n_pump);
        // TR = 1,2 ; SR = 0,8 ; η = 0,96.
        assert_relative_eq!(tr, 1.2, epsilon = 1e-12);
        assert_relative_eq!(sr, 0.8, epsilon = 1e-12);
        assert_relative_eq!(tc_efficiency(tr, sr), 0.96, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le couple de pompe T_pompe doit être strictement positif")]
    fn zero_pump_torque_panics() {
        tc_torque_ratio(300.0, 0.0);
    }
}
