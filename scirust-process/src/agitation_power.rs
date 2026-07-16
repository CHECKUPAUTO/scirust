//! Puissance d'agitation d'une cuve mécaniquement agitée (mobile tournant dans
//! un liquide) — nombre de Reynolds d'agitation, puissance dissipée à l'arbre,
//! nombre de puissance en régime laminaire, vitesse périphérique en bout de pale
//! et débit de circulation (pompage) du mobile.
//!
//! ```text
//! Reynolds d'agitation
//!   Re = ρ · N · D² / μ                                        [-]
//! puissance dissipée
//!   P  = Np · ρ · N³ · D⁵                                      [W]
//! nombre de puissance, régime laminaire
//!   Np = K_L / Re                                              [-]
//! vitesse périphérique (bout de pale)
//!   v_tip = π · D · N                                          [m·s⁻¹]
//! débit de circulation (pompage)
//!   Q  = Nq · N · D³                                           [m³·s⁻¹]
//! ```
//!
//! `ρ` masse volumique du liquide [kg·m⁻³], `N` vitesse de rotation en **tours
//! par seconde** [s⁻¹] (et non rad·s⁻¹), `D` diamètre du mobile d'agitation [m],
//! `μ` viscosité dynamique [Pa·s] ; `Re` nombre de Reynolds d'agitation
//! [sans dimension] ; `Np` nombre de puissance [sans dimension], `P` puissance
//! mécanique dissipée dans le liquide [W] ; `K_L` constante géométrique du mobile
//! en régime laminaire [sans dimension] ; `v_tip` vitesse linéaire en bout de
//! pale [m·s⁻¹] ; `Nq` nombre de débit (pompage) [sans dimension], `Q` débit de
//! circulation refoulé par le mobile [m³·s⁻¹].
//!
//! **Limite honnête** : corrélations à l'échelle des **opérations unitaires**
//! pour une cuve **standard munie de chicanes** et un fluide **newtonien**. Le
//! **nombre de puissance** `Np` (lu sur la courbe Np–Re propre au mobile et à la
//! géométrie) et le **nombre de débit** `Nq` sont **FOURNIS** par l'appelant :
//! ils dépendent du type de turbine, du rapport D/T, du dégagement au fond et du
//! chicanage, et ne sont **jamais** supposés « par défaut ». En **régime
//! turbulent** (Re ≳ 10⁴) `Np` est **constant** ; en **régime laminaire**
//! (Re ≲ 10) `Np = K_L/Re` avec `K_L` **FOURNI** (constante géométrique). Les
//! **propriétés physiques** du liquide (masse volumique, viscosité) sont
//! **FOURNIES** : aucune valeur physique n'est calculée ni inventée par ce
//! module. La formule `P = Np·ρ·N³·D⁵` n'est valable que si `Np` correspond bien
//! au régime (valeur de Re) considéré.

use core::f64::consts::PI;

/// Nombre de Reynolds d'agitation `Re = ρ · N · D² / μ` (sans dimension).
///
/// La vitesse de rotation `N` est exprimée en **tours par seconde** [s⁻¹].
///
/// `density` (ρ) masse volumique du liquide [kg·m⁻³], `rotation_speed` (N)
/// vitesse de rotation [s⁻¹], `impeller_diameter` (D) diamètre du mobile [m],
/// `viscosity` (μ) viscosité dynamique [Pa·s].
///
/// Panique si `ρ ≤ 0`, si `N < 0`, si `D ≤ 0`, ou si `μ ≤ 0`.
pub fn agit_reynolds(
    density: f64,
    rotation_speed: f64,
    impeller_diameter: f64,
    viscosity: f64,
) -> f64 {
    assert!(density > 0.0, "ρ > 0 requis (masse volumique du liquide)");
    assert!(rotation_speed >= 0.0, "N ≥ 0 requis (vitesse de rotation)");
    assert!(impeller_diameter > 0.0, "D > 0 requis (diamètre du mobile)");
    assert!(viscosity > 0.0, "μ > 0 requis (viscosité dynamique)");
    density * rotation_speed * impeller_diameter * impeller_diameter / viscosity
}

/// Puissance dissipée à l'arbre par le mobile
/// `P = Np · ρ · N³ · D⁵` (W).
///
/// La vitesse de rotation `N` est exprimée en **tours par seconde** [s⁻¹]. Le
/// nombre de puissance `Np` est **FOURNI** par l'appelant (courbe Np–Re du
/// mobile) et doit correspondre au régime (valeur de Re) considéré.
///
/// `power_number` (Np) nombre de puissance [sans dimension], `density` (ρ)
/// [kg·m⁻³], `rotation_speed` (N) [s⁻¹], `impeller_diameter` (D) [m].
///
/// Panique si `Np < 0`, si `ρ ≤ 0`, si `N < 0`, ou si `D ≤ 0`.
pub fn agit_power(
    power_number: f64,
    density: f64,
    rotation_speed: f64,
    impeller_diameter: f64,
) -> f64 {
    assert!(power_number >= 0.0, "Np ≥ 0 requis (nombre de puissance)");
    assert!(density > 0.0, "ρ > 0 requis (masse volumique du liquide)");
    assert!(rotation_speed >= 0.0, "N ≥ 0 requis (vitesse de rotation)");
    assert!(impeller_diameter > 0.0, "D > 0 requis (diamètre du mobile)");
    power_number * density * rotation_speed.powi(3) * impeller_diameter.powi(5)
}

/// Nombre de puissance en **régime laminaire** `Np = K_L / Re` (sans dimension).
///
/// La constante géométrique `K_L` est **FOURNIE** par l'appelant (elle dépend du
/// type de mobile et de la géométrie de la cuve). Valable pour `Re ≲ 10`.
///
/// `constant` (K_L) constante géométrique laminaire [sans dimension],
/// `reynolds` (Re) nombre de Reynolds d'agitation [sans dimension].
///
/// Panique si `K_L < 0` ou si `Re ≤ 0`.
pub fn agit_power_number_laminar(constant: f64, reynolds: f64) -> f64 {
    assert!(constant >= 0.0, "K_L ≥ 0 requis (constante géométrique)");
    assert!(reynolds > 0.0, "Re > 0 requis (Reynolds d'agitation)");
    constant / reynolds
}

/// Vitesse périphérique en bout de pale `v_tip = π · D · N` (m·s⁻¹).
///
/// La vitesse de rotation `N` est exprimée en **tours par seconde** [s⁻¹], de
/// sorte que `π · D` est la circonférence décrite par l'extrémité de la pale et
/// `v_tip` la distance parcourue par seconde.
///
/// `rotation_speed` (N) [s⁻¹], `impeller_diameter` (D) [m].
///
/// Panique si `N < 0` ou si `D ≤ 0`.
pub fn agit_tip_speed(rotation_speed: f64, impeller_diameter: f64) -> f64 {
    assert!(rotation_speed >= 0.0, "N ≥ 0 requis (vitesse de rotation)");
    assert!(impeller_diameter > 0.0, "D > 0 requis (diamètre du mobile)");
    PI * impeller_diameter * rotation_speed
}

/// Débit de circulation (pompage) refoulé par le mobile
/// `Q = Nq · N · D³` (m³·s⁻¹).
///
/// La vitesse de rotation `N` est exprimée en **tours par seconde** [s⁻¹]. Le
/// nombre de débit `Nq` est **FOURNI** par l'appelant (il dépend du mobile et de
/// la géométrie).
///
/// `flow_number` (Nq) nombre de débit [sans dimension], `rotation_speed` (N)
/// [s⁻¹], `impeller_diameter` (D) [m].
///
/// Panique si `Nq < 0`, si `N < 0`, ou si `D ≤ 0`.
pub fn agit_pumping_flow(flow_number: f64, rotation_speed: f64, impeller_diameter: f64) -> f64 {
    assert!(flow_number >= 0.0, "Nq ≥ 0 requis (nombre de débit)");
    assert!(rotation_speed >= 0.0, "N ≥ 0 requis (vitesse de rotation)");
    assert!(impeller_diameter > 0.0, "D > 0 requis (diamètre du mobile)");
    flow_number * rotation_speed * impeller_diameter.powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reynolds_reference_case() {
        // ρ = 1000, N = 2 tr/s, D = 0.5 m, μ = 1e-3 Pa·s ⇒
        //   Re = 1000·2·0.5² / 1e-3 = 1000·2·0.25 / 1e-3
        //      = 500 / 1e-3 = 500000.
        // Recalcul : 0.5² = 0.25 ; 1000·2·0.25 = 500 ; 500/0.001 = 500000.
        let re = agit_reynolds(1000.0_f64, 2.0_f64, 0.5_f64, 1.0e-3_f64);
        assert_relative_eq!(re, 500_000.0, max_relative = 1e-12);
    }

    #[test]
    fn reynolds_proportional_to_rotation_speed() {
        // Re ∝ N : doubler N double Re.
        let single = agit_reynolds(1000.0_f64, 2.0_f64, 0.5_f64, 1.0e-3_f64);
        let double = agit_reynolds(1000.0_f64, 4.0_f64, 0.5_f64, 1.0e-3_f64);
        assert_relative_eq!(double, 2.0 * single, max_relative = 1e-12);
    }

    #[test]
    fn power_reference_case() {
        // Np = 5, ρ = 1000, N = 2 tr/s, D = 0.5 m ⇒
        //   P = 5·1000·2³·0.5⁵ = 5·1000·8·0.03125.
        // Recalcul : 2³ = 8 ; 0.5⁵ = 0.03125 ; 5·1000·8 = 40000 ;
        //   40000·0.03125 = 1250 W.
        let p = agit_power(5.0_f64, 1000.0_f64, 2.0_f64, 0.5_f64);
        assert_relative_eq!(p, 1250.0, max_relative = 1e-12);
    }

    #[test]
    fn power_scales_with_speed_cubed() {
        // P ∝ N³ : doubler N multiplie la puissance par 8.
        let base = agit_power(5.0_f64, 1000.0_f64, 2.0_f64, 0.5_f64);
        let fast = agit_power(5.0_f64, 1000.0_f64, 4.0_f64, 0.5_f64);
        assert_relative_eq!(fast, 8.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn laminar_power_number_reciprocity() {
        // Np·Re = K_L : identité de réciprocité de la loi laminaire.
        // K_L = 64, Re = 10 ⇒ Np = 6.4 ; puis Np·Re = 6.4·10 = 64.
        let np = agit_power_number_laminar(64.0_f64, 10.0_f64);
        assert_relative_eq!(np, 6.4, max_relative = 1e-12);
        assert_relative_eq!(np * 10.0, 64.0, max_relative = 1e-12);
    }

    #[test]
    fn tip_speed_equals_pi_case_and_scales_with_diameter() {
        // N = 2 tr/s, D = 0.5 m ⇒ v_tip = π·0.5·2 = π m/s.
        let v = agit_tip_speed(2.0_f64, 0.5_f64);
        assert_relative_eq!(v, PI, max_relative = 1e-12);
        // v_tip ∝ D : doubler D double la vitesse en bout de pale.
        let v2 = agit_tip_speed(2.0_f64, 1.0_f64);
        assert_relative_eq!(v2, 2.0 * v, max_relative = 1e-12);
    }

    #[test]
    fn pumping_flow_reference_case() {
        // Nq = 0.8, N = 2 tr/s, D = 0.5 m ⇒
        //   Q = 0.8·2·0.5³ = 0.8·2·0.125.
        // Recalcul : 0.5³ = 0.125 ; 0.8·2 = 1.6 ; 1.6·0.125 = 0.2 m³/s.
        let q = agit_pumping_flow(0.8_f64, 2.0_f64, 0.5_f64);
        assert_relative_eq!(q, 0.2, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "μ > 0 requis")]
    fn reynolds_panics_on_zero_viscosity() {
        // μ = 0 ⇒ division par zéro non physique ⇒ entrée rejetée.
        let _ = agit_reynolds(1000.0_f64, 2.0_f64, 0.5_f64, 0.0_f64);
    }
}
