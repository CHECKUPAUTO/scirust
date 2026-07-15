//! **Turbine hydraulique** — puissance récupérée et lois de similitude d'une
//! turbine hydraulique (Pelton, Francis, Kaplan) à partir de la hauteur nette,
//! du débit et du rendement global.
//!
//! ```text
//! puissance disponible   P_dispo = ρ·g·Q·H
//! puissance à l'arbre     P       = ρ·g·Q·H·η
//! vitesse spécifique      Ns      = N·√P / H^1.25   (forme dimensionnelle)
//! vitesse unitaire        N11'    = N / √H
//! ```
//!
//! `ρ` masse volumique du fluide (kg/m³), `g` accélération de la pesanteur
//! (m/s²), `Q` débit turbiné (m³/s), `H` hauteur nette (m, chute utile après
//! pertes de charge amont/aval), `η` rendement global (sans unité, 0 < η ≤ 1),
//! `P_dispo` puissance hydraulique disponible dans la chute (W), `P` puissance
//! mécanique à l'arbre (W), `N` vitesse de rotation (**tr/min**, rpm), `Ns`
//! vitesse spécifique de puissance (**grandeur dimensionnelle**, unité composite
//! rpm·W^0,5·m^−1,25), `N11'` vitesse unitaire (rpm·m^−0,5).
//!
//! **Convention** : unités SI cohérentes sauf la vitesse de rotation, exprimée
//! en **tr/min** (rpm) conformément à l'usage sur les turbines. La vitesse
//! spécifique `Ns` calculée ici est la **forme dimensionnelle** (avec `P` en
//! watts) : sa valeur numérique dépend du système d'unités choisi pour `P` et
//! `H`, elle n'est comparable qu'entre turbines exprimées dans les **mêmes**
//! unités.
//!
//! **Limite honnête** : la hauteur nette `H`, le débit `Q`, le rendement global
//! `η`, la masse volumique `ρ` et la pesanteur `g` sont des données **fournies
//! par l'appelant** ; aucune valeur « par défaut » n'est supposée (le rendement
//! dépend du point de fonctionnement, `H` intègre les pertes de charge du circuit
//! qui relèvent d'une étude hydraulique séparée). La classification de type
//! [`turbine_type_from_specific_speed`] se contente de **ranger** la machine par
//! plage de `Ns` ; les **bornes** de plage (qui dépendent du système d'unités et
//! des conventions constructeur) sont elles aussi **fournies par l'appelant**.
//! Ce module ne dimensionne pas l'aubage.

/// Type de turbine hydraulique selon la plage de vitesse spécifique.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurbineType {
    /// Basse vitesse spécifique — grande chute, faible débit (turbine à action).
    Pelton,
    /// Vitesse spécifique moyenne — chute et débit intermédiaires.
    Francis,
    /// Haute vitesse spécifique — faible chute, grand débit (turbine à hélice).
    Kaplan,
}

/// Puissance hydraulique disponible dans la chute `P_dispo = ρ·g·Q·H` (W),
/// avant tout prélèvement par la machine.
///
/// Panique si `density <= 0`, `gravity <= 0`, `flow_rate < 0`, ou
/// `net_head <= 0`.
pub fn turbine_available_power(density: f64, gravity: f64, flow_rate: f64, net_head: f64) -> f64 {
    assert!(
        density > 0.0,
        "la masse volumique ρ doit être strictement positive"
    );
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur g doit être strictement positive"
    );
    assert!(flow_rate >= 0.0, "le débit Q ne peut pas être négatif");
    assert!(
        net_head > 0.0,
        "la hauteur nette H doit être strictement positive"
    );
    density * gravity * flow_rate * net_head
}

/// Puissance mécanique à l'arbre `P = ρ·g·Q·H·η` (W), une fois le rendement
/// global appliqué à la puissance disponible.
///
/// Panique si `density <= 0`, `gravity <= 0`, `flow_rate < 0`, `net_head <= 0`,
/// ou si `efficiency` n'est pas dans `]0, 1]`.
pub fn turbine_hydraulic_power(
    density: f64,
    gravity: f64,
    flow_rate: f64,
    net_head: f64,
    efficiency: f64,
) -> f64 {
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement global η doit être dans ]0, 1]"
    );
    turbine_available_power(density, gravity, flow_rate, net_head) * efficiency
}

/// Vitesse spécifique de puissance `Ns = N·√P / H^1.25` (forme **dimensionnelle**,
/// `N` en tr/min, `P` en watts, `H` en mètres).
///
/// La valeur n'est comparable qu'entre turbines exprimées dans les mêmes unités.
///
/// Panique si `rotational_speed_rpm < 0`, `power_watt < 0`, ou `net_head <= 0`.
pub fn turbine_specific_speed(rotational_speed_rpm: f64, power_watt: f64, net_head: f64) -> f64 {
    assert!(
        rotational_speed_rpm >= 0.0,
        "la vitesse de rotation N ne peut pas être négative"
    );
    assert!(
        power_watt >= 0.0,
        "la puissance P ne peut pas être négative"
    );
    assert!(
        net_head > 0.0,
        "la hauteur nette H doit être strictement positive"
    );
    rotational_speed_rpm * power_watt.sqrt() / net_head.powf(1.25)
}

/// Vitesse unitaire `N11' = N / √H` (rpm·m^−0,5), utilisée dans les lois de
/// similitude à diamètre de roue fixé.
///
/// Panique si `rotational_speed_rpm < 0` ou `net_head <= 0`.
pub fn turbine_unit_speed(rotational_speed_rpm: f64, net_head: f64) -> f64 {
    assert!(
        rotational_speed_rpm >= 0.0,
        "la vitesse de rotation N ne peut pas être négative"
    );
    assert!(
        net_head > 0.0,
        "la hauteur nette H doit être strictement positive"
    );
    rotational_speed_rpm / net_head.sqrt()
}

/// Classe la turbine selon sa vitesse spécifique `Ns` : `Pelton` si
/// `Ns < pelton_francis_boundary`, `Kaplan` si `Ns >= francis_kaplan_boundary`,
/// `Francis` entre les deux.
///
/// Les deux bornes sont **fournies par l'appelant** car elles dépendent du
/// système d'unités retenu pour `Ns` et des conventions constructeur ; ce module
/// n'en invente aucune.
///
/// Panique si `specific_speed < 0`, ou si les bornes ne vérifient pas
/// `0 < pelton_francis_boundary < francis_kaplan_boundary`.
pub fn turbine_type_from_specific_speed(
    specific_speed: f64,
    pelton_francis_boundary: f64,
    francis_kaplan_boundary: f64,
) -> TurbineType {
    assert!(
        specific_speed >= 0.0,
        "la vitesse spécifique Ns ne peut pas être négative"
    );
    assert!(
        pelton_francis_boundary > 0.0 && pelton_francis_boundary < francis_kaplan_boundary,
        "les bornes doivent vérifier 0 < pelton_francis_boundary < francis_kaplan_boundary"
    );
    if specific_speed < pelton_francis_boundary
    {
        TurbineType::Pelton
    }
    else if specific_speed < francis_kaplan_boundary
    {
        TurbineType::Francis
    }
    else
    {
        TurbineType::Kaplan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hydraulic_power_is_available_power_scaled_by_efficiency() {
        // Identité : P = P_dispo·η. Le rendement relie exactement les deux.
        let (rho, g, q, h, eta) = (1000.0, 9.81, 10.0, 20.0, 0.9);
        let p_dispo = turbine_available_power(rho, g, q, h);
        let p = turbine_hydraulic_power(rho, g, q, h, eta);
        assert_relative_eq!(p, p_dispo * eta, epsilon = 1e-9);
    }

    #[test]
    fn unit_efficiency_recovers_available_power() {
        // Cas limite η = 1 : la puissance à l'arbre égale la puissance disponible.
        let (rho, g, q, h) = (998.0, 9.81, 3.5, 45.0);
        assert_relative_eq!(
            turbine_hydraulic_power(rho, g, q, h, 1.0),
            turbine_available_power(rho, g, q, h),
            epsilon = 1e-9
        );
    }

    #[test]
    fn available_power_is_proportional_to_flow_rate() {
        // P_dispo ∝ Q à ρ, g, H fixés : doubler le débit double la puissance.
        let (rho, g, h) = (1000.0, 9.81, 30.0);
        let p1 = turbine_available_power(rho, g, 5.0, h);
        let p2 = turbine_available_power(rho, g, 10.0, h);
        assert_relative_eq!(p2, 2.0 * p1, epsilon = 1e-9);
    }

    #[test]
    fn specific_speed_is_proportional_to_sqrt_power() {
        // Ns ∝ √P à N et H fixés : quadrupler P double Ns.
        let (n, h) = (500.0, 16.0);
        let ns1 = turbine_specific_speed(n, 10_000.0, h);
        let ns2 = turbine_specific_speed(n, 40_000.0, h);
        assert_relative_eq!(ns2, 2.0 * ns1, epsilon = 1e-9);
    }

    #[test]
    fn realistic_specific_and_unit_speed_case() {
        // N = 500 tr/min, P = 10 000 W, H = 16 m.
        // H^1.25 = 16·16^0.25 = 16·2 = 32 ; √P = 100.
        // Ns = 500·100/32 = 1562,5 ; N11' = 500/√16 = 500/4 = 125.
        let ns = turbine_specific_speed(500.0, 10_000.0, 16.0);
        assert_relative_eq!(ns, 1562.5, epsilon = 1e-9);
        let n11 = turbine_unit_speed(500.0, 16.0);
        assert_relative_eq!(n11, 125.0, epsilon = 1e-12);
    }

    #[test]
    fn classification_ranges_by_specific_speed() {
        // Bornes fournies par l'appelant : chaque plage tombe dans le bon type.
        let (b1, b2) = (70.0, 400.0);
        assert_eq!(
            turbine_type_from_specific_speed(30.0, b1, b2),
            TurbineType::Pelton
        );
        assert_eq!(
            turbine_type_from_specific_speed(200.0, b1, b2),
            TurbineType::Francis
        );
        assert_eq!(
            turbine_type_from_specific_speed(800.0, b1, b2),
            TurbineType::Kaplan
        );
        // La borne haute est inclusive côté Kaplan.
        assert_eq!(
            turbine_type_from_specific_speed(b2, b1, b2),
            TurbineType::Kaplan
        );
    }

    #[test]
    #[should_panic(expected = "le rendement global η doit être dans ]0, 1]")]
    fn efficiency_above_one_panics() {
        turbine_hydraulic_power(1000.0, 9.81, 10.0, 20.0, 1.2);
    }
}
