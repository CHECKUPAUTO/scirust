//! Agitation d'un liquide — **puissance d'un mobile** (turbine, hélice) et
//! **Reynolds d'agitation** caractérisant le régime d'écoulement dans la cuve.
//!
//! ```text
//! Reynolds agitation   Re = ρ·N·D² / μ                     [sans dimension]
//! puissance turbulente P  = Np·ρ·N³·D⁵                      [W]
//! puissance laminaire  P  = Kp·μ·N²·D³                      [W]   (Np = Kp/Re)
//! vitesse périphérique v  = π·D·N                           [m/s]
//! ```
//!
//! `Np` nombre de puissance (turbulent) [sans dimension], `Kp` constante de
//! régime laminaire [sans dimension], `ρ` masse volumique du liquide [kg/m³],
//! `μ` viscosité dynamique [Pa·s], `N` vitesse de rotation [tr/s], `D` diamètre
//! du mobile [m], `P` puissance dissipée à l'arbre [W], `v` vitesse en bout de
//! pale [m/s], `Re` nombre de Reynolds d'agitation [sans dimension].
//!
//! **Limite honnête** : le nombre de puissance `Np` (régime turbulent) et la
//! constante `Kp` (régime laminaire) sont **fournis par l'appelant** ; ils
//! dépendent du type de mobile et de la géométrie de cuve (courbe `Np`–`Re`
//! propre à chaque installation) et ne sont jamais supposés par défaut. La
//! masse volumique `ρ` et la viscosité `μ` sont également **fournies**. La
//! vitesse de rotation `N` est exprimée en **tours par seconde** (tr/s), pas en
//! rad/s ni en tr/min.

use core::f64::consts::PI;

/// Nombre de Reynolds d'agitation `Re = ρ·N·D² / μ` [sans dimension].
///
/// `density` `ρ` en kg/m³, `rotational_speed` `N` en tr/s, `impeller_diameter`
/// `D` en m, `dynamic_viscosity` `μ` en Pa·s ; le résultat est sans dimension.
///
/// Panique si un argument est négatif ou nul.
pub fn mixing_reynolds(
    density: f64,
    rotational_speed: f64,
    impeller_diameter: f64,
    dynamic_viscosity: f64,
) -> f64 {
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive (kg/m³)"
    );
    assert!(
        rotational_speed > 0.0,
        "la vitesse de rotation doit être strictement positive (tr/s)"
    );
    assert!(
        impeller_diameter > 0.0,
        "le diamètre du mobile doit être strictement positif (m)"
    );
    assert!(
        dynamic_viscosity > 0.0,
        "la viscosité dynamique doit être strictement positive (Pa·s)"
    );
    density * rotational_speed * impeller_diameter.powi(2) / dynamic_viscosity
}

/// Puissance dissipée en régime turbulent `P = Np·ρ·N³·D⁵` [W].
///
/// `power_number` `Np` (sans dimension, fourni par l'appelant), `density` `ρ`
/// en kg/m³, `rotational_speed` `N` en tr/s, `impeller_diameter` `D` en m ; la
/// puissance rendue est en watts (W).
///
/// Panique si un argument est négatif ou nul.
pub fn mixing_power_turbulent(
    power_number: f64,
    density: f64,
    rotational_speed: f64,
    impeller_diameter: f64,
) -> f64 {
    assert!(
        power_number > 0.0,
        "le nombre de puissance doit être strictement positif"
    );
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive (kg/m³)"
    );
    assert!(
        rotational_speed > 0.0,
        "la vitesse de rotation doit être strictement positive (tr/s)"
    );
    assert!(
        impeller_diameter > 0.0,
        "le diamètre du mobile doit être strictement positif (m)"
    );
    power_number * density * rotational_speed.powi(3) * impeller_diameter.powi(5)
}

/// Puissance dissipée en régime laminaire `P = Kp·μ·N²·D³` [W], où
/// `Np = Kp/Re` (le nombre de puissance varie en `1/Re`).
///
/// `laminar_constant` `Kp` (sans dimension, fourni par l'appelant),
/// `dynamic_viscosity` `μ` en Pa·s, `rotational_speed` `N` en tr/s,
/// `impeller_diameter` `D` en m ; la puissance rendue est en watts (W).
///
/// Panique si un argument est négatif ou nul.
pub fn mixing_power_laminar(
    laminar_constant: f64,
    dynamic_viscosity: f64,
    rotational_speed: f64,
    impeller_diameter: f64,
) -> f64 {
    assert!(
        laminar_constant > 0.0,
        "la constante de régime laminaire doit être strictement positive"
    );
    assert!(
        dynamic_viscosity > 0.0,
        "la viscosité dynamique doit être strictement positive (Pa·s)"
    );
    assert!(
        rotational_speed > 0.0,
        "la vitesse de rotation doit être strictement positive (tr/s)"
    );
    assert!(
        impeller_diameter > 0.0,
        "le diamètre du mobile doit être strictement positif (m)"
    );
    laminar_constant * dynamic_viscosity * rotational_speed.powi(2) * impeller_diameter.powi(3)
}

/// Vitesse périphérique en bout de pale `v = π·D·N` [m/s].
///
/// `rotational_speed` `N` en tr/s, `impeller_diameter` `D` en m ; la vitesse
/// rendue est en m/s.
///
/// Panique si un argument est négatif ou nul.
pub fn mixing_tip_speed(rotational_speed: f64, impeller_diameter: f64) -> f64 {
    assert!(
        rotational_speed > 0.0,
        "la vitesse de rotation doit être strictement positive (tr/s)"
    );
    assert!(
        impeller_diameter > 0.0,
        "le diamètre du mobile doit être strictement positif (m)"
    );
    PI * impeller_diameter * rotational_speed
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn turbulent_power_known_case() {
        // Np = 5, ρ = 1000 kg/m³, N = 2 tr/s, D = 0,3 m.
        // P = 5·1000·2³·0,3⁵ = 5·1000·8·0,00243 = 97,2 W.
        let (np, rho, n, d) = (5.0_f64, 1000.0_f64, 2.0_f64, 0.3_f64);
        assert_relative_eq!(mixing_power_turbulent(np, rho, n, d), 97.2, epsilon = 1e-9);
    }

    #[test]
    fn reynolds_known_case() {
        // ρ = 1000, N = 2, D = 0,3, μ = 1e-3 (eau).
        // Re = 1000·2·0,09 / 1e-3 = 180 / 1e-3 = 180000.
        let (rho, n, d, mu) = (1000.0_f64, 2.0_f64, 0.3_f64, 1e-3_f64);
        assert_relative_eq!(mixing_reynolds(rho, n, d, mu), 180_000.0, epsilon = 1e-6);
    }

    #[test]
    fn laminar_power_known_case() {
        // Kp = 64, μ = 1,0 Pa·s, N = 2 tr/s, D = 0,3 m.
        // P = 64·1,0·2²·0,3³ = 64·4·0,027 = 6,912 W.
        let (kp, mu, n, d) = (64.0_f64, 1.0_f64, 2.0_f64, 0.3_f64);
        assert_relative_eq!(mixing_power_laminar(kp, mu, n, d), 6.912, epsilon = 1e-9);
    }

    #[test]
    fn tip_speed_known_case() {
        // v = π·D·N = π·0,3·2 = 0,6·π ≈ 1,884955592 m/s.
        let (n, d) = (2.0_f64, 0.3_f64);
        assert_relative_eq!(mixing_tip_speed(n, d), 0.6 * PI, epsilon = 1e-12);
        assert_relative_eq!(mixing_tip_speed(n, d), 1.884_955_592, epsilon = 1e-9);
    }

    #[test]
    fn laminar_equals_turbulent_when_np_is_kp_over_re() {
        // Cohérence des deux régimes : en posant Np = Kp/Re, la formule
        // turbulente redonne exactement la puissance laminaire.
        let (kp, rho, mu, n, d) = (64.0_f64, 1000.0_f64, 0.5_f64, 3.0_f64, 0.25_f64);
        let re = mixing_reynolds(rho, n, d, mu);
        let p_lam = mixing_power_laminar(kp, mu, n, d);
        let p_turb = mixing_power_turbulent(kp / re, rho, n, d);
        assert_relative_eq!(p_turb, p_lam, epsilon = 1e-12);
    }

    #[test]
    fn turbulent_power_scales_as_speed_cubed() {
        // Doubler N à géométrie fixe multiplie la puissance turbulente par 2³ = 8.
        let (np, rho, d) = (4.0_f64, 1200.0_f64, 0.4_f64);
        let base = mixing_power_turbulent(np, rho, 1.5, d);
        let doubled = mixing_power_turbulent(np, rho, 3.0, d);
        assert_relative_eq!(doubled, 8.0 * base, epsilon = 1e-9);
    }

    #[test]
    fn tip_speed_is_linear_in_diameter() {
        // À N fixe, v est proportionnelle à D : v(2D) = 2·v(D).
        let n = 1.7_f64;
        let single = mixing_tip_speed(n, 0.2);
        let double = mixing_tip_speed(n, 0.4);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn zero_viscosity_panics() {
        mixing_reynolds(1000.0, 2.0, 0.3, 0.0);
    }
}
