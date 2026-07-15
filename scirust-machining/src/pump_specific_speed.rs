//! **Vitesse spécifique d'une pompe** : nombre de similitude qui caractérise la
//! forme de la roue au point de meilleur rendement, indépendamment de la taille de
//! la machine, et permet de comparer des pompes géométriquement semblables.
//!
//! ```text
//! vitesse spécifique adimensionnelle (N en rad/s) :
//!   ns = N · √Q / (g·H)^0.75
//! vitesse spécifique d'aspiration (même forme, NPSH requis au lieu de H) :
//!   nss = N · √Q / (g·NPSHr)^0.75
//! forme dimensionnelle usuelle (N en tr/min, Q en m³/s, H en m) :
//!   Ns = rpm · √Q / H^0.75
//! conversion (mêmes Q, H et vitesse cohérente) :
//!   ns = Ns · (2π/60) / g^0.75
//! ```
//!
//! `N` fréquence de rotation angulaire (rad/s), `rpm` fréquence de rotation
//! (tr/min), `Q` débit volumique au point de meilleur rendement (m³/s), `H` hauteur
//! manométrique (m), `NPSHr` charge nette absolue à l'aspiration requise (m), `g`
//! accélération de la pesanteur (m/s²), `ns`/`nss` sont adimensionnels, `Ns` a la
//! dimension résiduelle des unités choisies (tr·min⁻¹·m³·⁵·s⁻⁰·⁵·m⁻⁰·⁷⁵).
//!
//! **Convention** : SI cohérent pour la forme adimensionnelle, qui utilise l'énergie
//! massique `g·H` (J/kg). La forme dimensionnelle `Ns` dépend explicitement du choix
//! d'unités (ici tr/min, m³/s, m) et n'est donc comparable qu'à unités identiques.
//!
//! **Limite honnête** : la vitesse spécifique n'a de sens que sous **similitude
//! géométrique** (roues de la même famille) et **au point de meilleur rendement**.
//! Les grandeurs de fonctionnement (`N`, `Q`, `H`, `NPSHr`) et la pesanteur `g` sont
//! **fournies par l'appelant** ; aucune valeur « par défaut » (fluide, machine,
//! procédé) n'est inventée. Le classement du type de roue utilise des **bornes de
//! transition fournies par l'appelant** : ces conventions ne sont pas universelles.
//! Ce module **classe** la forme de roue mais ne la **dimensionne** pas.

/// Type de roue déduit de la vitesse spécifique, par ordre croissant de `ns`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PumpSpecificImpellerType {
    /// Roue radiale (pompe centrifuge), faible vitesse spécifique.
    Radial,
    /// Roue à écoulement mixte (hélico-centrifuge), vitesse spécifique moyenne.
    MixedFlow,
    /// Roue axiale (hélice), forte vitesse spécifique.
    Axial,
}

/// Vitesse spécifique adimensionnelle `ns = N·√Q / (g·H)^0.75` (`N` en rad/s).
///
/// Panique si `rotational_speed < 0`, `flow_rate < 0`, `head <= 0` ou `gravity <= 0`.
pub fn pump_specific_speed(rotational_speed: f64, flow_rate: f64, head: f64, gravity: f64) -> f64 {
    assert!(
        rotational_speed >= 0.0,
        "la vitesse de rotation ne peut pas être négative"
    );
    assert!(flow_rate >= 0.0, "le débit ne peut pas être négatif");
    assert!(
        head > 0.0,
        "la hauteur manométrique doit être strictement positive"
    );
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur doit être strictement positive"
    );
    rotational_speed * flow_rate.sqrt() / (gravity * head).powf(0.75)
}

/// Vitesse spécifique d'aspiration adimensionnelle
/// `nss = N·√Q / (g·NPSHr)^0.75` (`N` en rad/s, `NPSHr` charge d'aspiration requise).
///
/// Panique si `rotational_speed < 0`, `flow_rate < 0`, `npsh_required <= 0`
/// ou `gravity <= 0`.
pub fn pump_specific_suction_speed(
    rotational_speed: f64,
    flow_rate: f64,
    npsh_required: f64,
    gravity: f64,
) -> f64 {
    assert!(
        rotational_speed >= 0.0,
        "la vitesse de rotation ne peut pas être négative"
    );
    assert!(flow_rate >= 0.0, "le débit ne peut pas être négatif");
    assert!(
        npsh_required > 0.0,
        "la charge nette d'aspiration requise doit être strictement positive"
    );
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur doit être strictement positive"
    );
    rotational_speed * flow_rate.sqrt() / (gravity * npsh_required).powf(0.75)
}

/// Vitesse spécifique sous forme dimensionnelle usuelle
/// `Ns = rpm·√Q / H^0.75` (`rpm` en tr/min, `Q` en m³/s, `H` en m).
///
/// Panique si `rotational_speed_rpm < 0`, `flow_rate_m3s < 0` ou `head_m <= 0`.
pub fn pump_specific_dimensional_ns_rpm(
    rotational_speed_rpm: f64,
    flow_rate_m3s: f64,
    head_m: f64,
) -> f64 {
    assert!(
        rotational_speed_rpm >= 0.0,
        "la vitesse de rotation ne peut pas être négative"
    );
    assert!(flow_rate_m3s >= 0.0, "le débit ne peut pas être négatif");
    assert!(
        head_m > 0.0,
        "la hauteur manométrique doit être strictement positive"
    );
    rotational_speed_rpm * flow_rate_m3s.sqrt() / head_m.powf(0.75)
}

/// Classe le type de roue à partir de la vitesse spécifique `specific_speed`
/// et des **bornes de transition fournies** `radial_upper` (radial → mixte) et
/// `mixed_upper` (mixte → axial) : `Radial` si `ns ≤ radial_upper`, `MixedFlow`
/// si `ns ≤ mixed_upper`, sinon `Axial`.
///
/// Panique si `specific_speed < 0`, `radial_upper <= 0`
/// ou `mixed_upper <= radial_upper`.
pub fn pump_specific_impeller_class(
    specific_speed: f64,
    radial_upper: f64,
    mixed_upper: f64,
) -> PumpSpecificImpellerType {
    assert!(
        specific_speed >= 0.0,
        "la vitesse spécifique ne peut pas être négative"
    );
    assert!(
        radial_upper > 0.0,
        "la borne radial→mixte doit être strictement positive"
    );
    assert!(
        mixed_upper > radial_upper,
        "la borne mixte→axial doit dépasser la borne radial→mixte"
    );
    if specific_speed <= radial_upper
    {
        PumpSpecificImpellerType::Radial
    }
    else if specific_speed <= mixed_upper
    {
        PumpSpecificImpellerType::MixedFlow
    }
    else
    {
        PumpSpecificImpellerType::Axial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn dimensional_worked_case() {
        // rpm = 1000, Q = 0,04 m³/s (√Q = 0,2), H = 16 m (16^0.75 = 2³ = 8).
        // Ns = 1000 · 0,2 / 8 = 200 / 8 = 25 (exact).
        assert_relative_eq!(
            pump_specific_dimensional_ns_rpm(1000.0, 0.04, 16.0),
            25.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn dimensionless_matches_dimensional_via_conversion() {
        // Identité : ns = Ns · (2π/60) / g^0.75 pour la même vitesse cohérente.
        let (rpm, q, h, g) = (1450.0_f64, 0.05_f64, 30.0_f64, 9.81_f64);
        let n_rad = rpm * 2.0 * PI / 60.0;
        let ns = pump_specific_speed(n_rad, q, h, g);
        let ns_dim = pump_specific_dimensional_ns_rpm(rpm, q, h);
        let expected = ns_dim * (2.0 * PI / 60.0) / g.powf(0.75);
        assert_relative_eq!(ns, expected, epsilon = 1e-12);
    }

    #[test]
    fn suction_speed_shares_speed_formula() {
        // La vitesse spécifique d'aspiration est la même forme, avec NPSHr au lieu
        // de H : les deux fonctions coïncident pour les mêmes arguments.
        let (n, q, x, g) = (151.0_f64, 0.06_f64, 4.5_f64, 9.81_f64);
        assert_relative_eq!(
            pump_specific_suction_speed(n, q, x, g),
            pump_specific_speed(n, q, x, g),
            epsilon = 1e-15
        );
    }

    #[test]
    fn scales_linearly_with_speed_and_sqrt_of_flow() {
        // Proportionnalités : ns ∝ N et ns ∝ √Q. Doubler N double ns ;
        // quadrupler Q double ns.
        let base = pump_specific_speed(120.0, 0.05, 25.0, 9.81);
        assert_relative_eq!(
            pump_specific_speed(240.0, 0.05, 25.0, 9.81),
            2.0 * base,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            pump_specific_speed(120.0, 0.20, 25.0, 9.81),
            2.0 * base,
            epsilon = 1e-12
        );
    }

    #[test]
    fn head_exponent_is_three_quarters() {
        // ns ∝ (g·H)^-0.75 : le rapport de deux hauteurs suit l'exposant -3/4,
        // soit ns(H2)/ns(H1) = (H1/H2)^0.75.
        let (n, q, g) = (150.0_f64, 0.05_f64, 9.81_f64);
        let (h1, h2) = (20.0_f64, 80.0_f64);
        let ns1 = pump_specific_speed(n, q, h1, g);
        let ns2 = pump_specific_speed(n, q, h2, g);
        assert_relative_eq!(ns2 / ns1, (h1 / h2).powf(0.75), epsilon = 1e-12);
    }

    #[test]
    fn impeller_class_uses_supplied_bounds() {
        // Bornes conventionnelles fournies : radial→mixte à 1,0 ; mixte→axial à 3,0.
        assert_eq!(
            pump_specific_impeller_class(0.5, 1.0, 3.0),
            PumpSpecificImpellerType::Radial
        );
        assert_eq!(
            pump_specific_impeller_class(2.0, 1.0, 3.0),
            PumpSpecificImpellerType::MixedFlow
        );
        assert_eq!(
            pump_specific_impeller_class(4.0, 1.0, 3.0),
            PumpSpecificImpellerType::Axial
        );
    }

    #[test]
    #[should_panic(expected = "la hauteur manométrique doit être strictement positive")]
    fn zero_head_panics() {
        pump_specific_speed(150.0, 0.05, 0.0, 9.81);
    }
}
