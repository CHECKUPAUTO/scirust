//! Dimensionnement hydraulique d'une pompe : puissance hydraulique transmise au
//! liquide, puissance à l'arbre à partir du rendement, NPSH disponible à
//! l'aspiration, vitesse spécifique et loi de similitude (affinité) sur le
//! débit.
//!
//! ```text
//! puissance hydraulique
//!   Ph      = ρ · g · Q · H                                    [W]
//! puissance à l'arbre
//!   Pa      = Ph / η                                           [W]
//! NPSH disponible (hauteur nette d'aspiration)
//!   NPSHa   = (P_atm − P_vap) / (ρ · g) + h_stat − h_pertes    [m]
//! vitesse spécifique
//!   Ns      = N · √Q / H^0.75                                  [unités]
//! loi de similitude (affinité) sur le débit
//!   Q₂      = Q₁ · N₂ / N₁                                     [m³·s⁻¹]
//! ```
//!
//! `ρ` masse volumique du liquide [kg·m⁻³], `g` accélération de la pesanteur
//! [m·s⁻²], `Q` débit volumique [m³·s⁻¹], `H` hauteur manométrique
//! (« head ») [m], `Ph` puissance hydraulique utile transmise au fluide [W],
//! `η` rendement global de la pompe [sans dimension, 0 < η ≤ 1], `Pa`
//! puissance mécanique à l'arbre [W] ; `P_atm` pression au plan d'aspiration
//! [Pa], `P_vap` pression de vapeur saturante du liquide [Pa], `h_stat` charge
//! statique d'aspiration [m] (positive en charge, négative en dépression),
//! `h_pertes` pertes de charge de la conduite d'aspiration [m], `NPSHa` charge
//! nette absolue à l'aspiration disponible [m] ; `N` vitesse de rotation
//! [tr·min⁻¹ ou rad·s⁻¹ selon la convention retenue], `Ns` vitesse spécifique
//! [unités dépendant de la convention de `N`, `Q`, `H`] ; les indices `₁` et
//! `₂` désignent deux régimes de rotation d'une même pompe.
//!
//! **Limite honnête** : modèle au niveau des **opérations unitaires**, sans
//! recouvrir la mécanique des fluides fondamentale (scirust-fluids) ni les
//! propriétés d'état (scirust-thermo). Le **rendement** `η`, la **pression de
//! vapeur** `P_vap` et les **pertes de charge d'aspiration** `h_pertes` sont
//! **FOURNIS** par l'appelant : ce module n'estime aucune corrélation de pertes,
//! ni courbe de rendement, ni tension de vapeur. Le NPSH **disponible** calculé
//! ici doit être comparé au NPSH **requis** de la pompe (lu sur la courbe
//! **constructeur**, donc FOURNI) pour prévenir la **cavitation** : la marge
//! `NPSHa − NPSHr` n'est pas calculée par ce module. La **vitesse spécifique**
//! `Ns` dépend du système d'unités choisi pour `N`, `Q` et `H` (SI, tr·min⁻¹,
//! usage américain…) : la valeur numérique n'a de sens qu'à convention fixée par
//! l'appelant. Les **lois de similitude** (affinité) ne sont valables qu'à
//! **rendement constant** et pour des points homologues d'une même roue. La
//! **gravité** `g` est FOURNIE. Aucune propriété physique n'est inventée.

/// Puissance hydraulique utile transmise au liquide
/// `Ph = ρ · g · Q · H` (W).
///
/// `density` (ρ) [kg·m⁻³], `gravity` (g) [m·s⁻²], `flow_rate` (Q) [m³·s⁻¹],
/// `head` (H) [m].
///
/// Panique si `ρ ≤ 0`, si `g ≤ 0`, si `Q < 0` ou si `H < 0` (grandeurs
/// physiques d'un refoulement non physiques sinon).
pub fn pump_hydraulic_power(density: f64, gravity: f64, flow_rate: f64, head: f64) -> f64 {
    assert!(density > 0.0, "ρ > 0 requis (masse volumique du liquide)");
    assert!(gravity > 0.0, "g > 0 requis (pesanteur)");
    assert!(flow_rate >= 0.0, "Q ≥ 0 requis (débit volumique)");
    assert!(head >= 0.0, "H ≥ 0 requis (hauteur manométrique)");
    density * gravity * flow_rate * head
}

/// Puissance mécanique à l'arbre à partir de la puissance hydraulique et du
/// rendement global
/// `Pa = Ph / η` (W).
///
/// `hydraulic_power` (Ph) [W], `efficiency` (η) rendement global
/// [sans dimension].
///
/// Panique si `Ph < 0` ou si `η` hors de `]0, 1]` (rendement non physique ou
/// division par zéro).
pub fn pump_shaft_power(hydraulic_power: f64, efficiency: f64) -> f64 {
    assert!(
        hydraulic_power >= 0.0,
        "Ph ≥ 0 requis (puissance hydraulique)"
    );
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "0 < η ≤ 1 requis (rendement global)"
    );
    hydraulic_power / efficiency
}

/// NPSH disponible (charge nette absolue à l'aspiration)
/// `NPSHa = (P_atm − P_vap) / (ρ · g) + h_stat − h_pertes` (m).
///
/// `atmospheric_pressure` (P_atm) [Pa], `vapor_pressure` (P_vap) [Pa],
/// `density` (ρ) [kg·m⁻³], `gravity` (g) [m·s⁻²], `static_head` (h_stat) [m]
/// (positif en charge, négatif en dépression), `friction_loss` (h_pertes) [m].
///
/// Panique si `ρ ≤ 0`, si `g ≤ 0`, si `P_atm < 0`, si `P_vap < 0`, si
/// `P_vap > P_atm` (charge de pression absolue négative, non physique) ou si
/// `h_pertes < 0`. La valeur retournée doit être comparée au NPSH **requis**
/// (FOURNI par la courbe constructeur) pour vérifier la marge anti-cavitation.
pub fn pump_npsh_available(
    atmospheric_pressure: f64,
    vapor_pressure: f64,
    density: f64,
    gravity: f64,
    static_head: f64,
    friction_loss: f64,
) -> f64 {
    assert!(density > 0.0, "ρ > 0 requis (masse volumique du liquide)");
    assert!(gravity > 0.0, "g > 0 requis (pesanteur)");
    assert!(
        atmospheric_pressure >= 0.0,
        "P_atm ≥ 0 requis (pression au plan d'aspiration)"
    );
    assert!(
        vapor_pressure >= 0.0,
        "P_vap ≥ 0 requis (pression de vapeur)"
    );
    assert!(
        vapor_pressure <= atmospheric_pressure,
        "P_vap ≤ P_atm requis (charge de pression absolue non négative)"
    );
    assert!(
        friction_loss >= 0.0,
        "h_pertes ≥ 0 requis (pertes d'aspiration)"
    );
    (atmospheric_pressure - vapor_pressure) / (density * gravity) + static_head - friction_loss
}

/// Vitesse spécifique de la pompe
/// `Ns = N · √Q / H^0.75` (unités dépendant de la convention de `N`, `Q`, `H`).
///
/// `rotation_speed` (N) [tr·min⁻¹ ou rad·s⁻¹], `flow_rate` (Q) [m³·s⁻¹],
/// `head` (H) [m]. La valeur n'a de sens qu'à système d'unités fixé par
/// l'appelant.
///
/// Panique si `N < 0`, si `Q < 0` ou si `H ≤ 0` (élévation de `H` à une
/// puissance non entière et division exigent `H` strictement positif).
pub fn pump_specific_speed(rotation_speed: f64, flow_rate: f64, head: f64) -> f64 {
    assert!(rotation_speed >= 0.0, "N ≥ 0 requis (vitesse de rotation)");
    assert!(flow_rate >= 0.0, "Q ≥ 0 requis (débit volumique)");
    assert!(head > 0.0, "H > 0 requis (hauteur manométrique)");
    rotation_speed * flow_rate.sqrt() / head.powf(0.75)
}

/// Débit homologue par la loi de similitude (affinité) à rendement constant
/// `Q₂ = Q₁ · N₂ / N₁` (m³·s⁻¹).
///
/// `flow_1` (Q₁) débit au régime 1 [m³·s⁻¹], `speed_1` (N₁) vitesse au
/// régime 1 [même unité que `speed_2`], `speed_2` (N₂) vitesse au régime 2.
///
/// Panique si `Q₁ < 0`, si `N₁ ≤ 0` (division par zéro) ou si `N₂ < 0`. Valable
/// uniquement à rendement constant entre points homologues d'une même roue.
pub fn pump_affinity_flow(flow_1: f64, speed_1: f64, speed_2: f64) -> f64 {
    assert!(flow_1 >= 0.0, "Q₁ ≥ 0 requis (débit du régime 1)");
    assert!(speed_1 > 0.0, "N₁ > 0 requis (vitesse du régime 1)");
    assert!(speed_2 >= 0.0, "N₂ ≥ 0 requis (vitesse du régime 2)");
    flow_1 * speed_2 / speed_1
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hydraulic_and_shaft_power_are_reciprocal() {
        // Ph = 1000·9.81·0.05·20 = 9810 W ; avec η = 0.75, Pa = 9810/0.75 = 13080 W.
        let ph = pump_hydraulic_power(1000.0_f64, 9.81_f64, 0.05_f64, 20.0_f64);
        assert_relative_eq!(ph, 9810.0, max_relative = 1e-12);
        let pa = pump_shaft_power(ph, 0.75_f64);
        assert_relative_eq!(pa, 13080.0, max_relative = 1e-12);
        // Réciprocité : Pa · η = Ph.
        assert_relative_eq!(pa * 0.75_f64, ph, max_relative = 1e-12);
    }

    #[test]
    fn hydraulic_power_linear_in_flow_and_head() {
        let base = pump_hydraulic_power(998.0_f64, 9.81_f64, 0.03_f64, 15.0_f64);
        // Doubler Q double Ph.
        let double_q = pump_hydraulic_power(998.0_f64, 9.81_f64, 0.06_f64, 15.0_f64);
        assert_relative_eq!(double_q, 2.0 * base, max_relative = 1e-12);
        // Doubler H double Ph.
        let double_h = pump_hydraulic_power(998.0_f64, 9.81_f64, 0.03_f64, 30.0_f64);
        assert_relative_eq!(double_h, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn npsh_available_realistic_case() {
        // Eau à 20 °C : P_atm = 101325 Pa, P_vap = 2339 Pa, ρ = 998, g = 9.81,
        // h_stat = 3 m, h_pertes = 1.5 m.
        //   (101325 − 2339) / (998·9.81) = 98986 / 9790.38 = 10.110537 m
        //   NPSHa = 10.110537 + 3 − 1.5 = 11.610537 m.
        let npsh = pump_npsh_available(
            101325.0_f64,
            2339.0_f64,
            998.0_f64,
            9.81_f64,
            3.0_f64,
            1.5_f64,
        );
        assert_relative_eq!(npsh, 11.610537, max_relative = 1e-3);
    }

    #[test]
    fn specific_speed_realistic_case_and_proportional() {
        // N = 1450, Q = 0.05, H = 20 :
        //   √0.05 = 0.22360680, 20^0.75 = 8000^0.25 = 9.4574161
        //   Ns = 1450·0.22360680 / 9.4574161 = 324.229857 / 9.4574161 = 34.283133.
        let ns = pump_specific_speed(1450.0_f64, 0.05_f64, 20.0_f64);
        assert_relative_eq!(ns, 34.283133, max_relative = 1e-3);
        // Proportionnalité à N : doubler N double Ns.
        let ns2 = pump_specific_speed(2900.0_f64, 0.05_f64, 20.0_f64);
        assert_relative_eq!(ns2, 2.0 * ns, max_relative = 1e-12);
    }

    #[test]
    fn affinity_flow_scales_with_speed() {
        // Q₁ = 0.05, N₁ = 1450, N₂ = 1740 ⇒ Q₂ = 0.05·1740/1450 = 0.06.
        let q2 = pump_affinity_flow(0.05_f64, 1450.0_f64, 1740.0_f64);
        assert_relative_eq!(q2, 0.06, max_relative = 1e-12);
        // À vitesse égale, le débit est inchangé.
        let q_same = pump_affinity_flow(0.05_f64, 1450.0_f64, 1450.0_f64);
        assert_relative_eq!(q_same, 0.05, max_relative = 1e-12);
        // Doubler N₂ double Q₂ (proportionnalité).
        let q_double = pump_affinity_flow(0.05_f64, 1450.0_f64, 3480.0_f64);
        assert_relative_eq!(q_double, 2.0 * q2, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "0 < η ≤ 1 requis")]
    fn shaft_power_panics_on_invalid_efficiency() {
        // η = 0 ⇒ division par zéro ⇒ entrée rejetée.
        let _ = pump_shaft_power(9810.0_f64, 0.0_f64);
    }
}
