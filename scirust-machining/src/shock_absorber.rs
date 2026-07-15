//! **Amortisseur hydraulique** (dashpot à orifice) — effort d'amortissement à
//! étranglement (quadratique en vitesse) ou visqueux linéaire, énergie dissipée
//! sur une course et amortissement critique du système monté.
//!
//! ```text
//! effort à orifice     F_o = 0,5·ρ·A_p³·v² / (C_d·A_o)²
//! effort visqueux      F_v = c·v
//! énergie dissipée     E   = F_moy·s
//! amortissement crit.  c_c = 2·√(m·k)
//! ```
//!
//! `ρ` masse volumique du fluide (kg·m⁻³), `A_p` aire du piston (m²), `A_o` aire
//! de l'orifice (m²), `v` vitesse du piston (m·s⁻¹), `C_d` coefficient de
//! décharge de l'orifice (sans dimension), `F_o`/`F_v` effort d'amortissement
//! (N), `c` coefficient d'amortissement visqueux (N·s·m⁻¹), `F_moy` effort moyen
//! sur la course (N), `s` course (m), `E` énergie dissipée (J), `m` masse montée
//! (kg), `k` raideur montée (N·m⁻¹), `c_c` amortissement critique (N·s·m⁻¹).
//!
//! **Convention** : unités SI cohérentes. La vitesse du fluide à l'orifice
//! découle de la continuité `A_p·v = A_o·v_o`, d'où l'effort d'orifice
//! quadratique en `v`.
//!
//! **Limite honnête** : l'amortissement à orifice est **quadratique** en
//! vitesse (coefficient de décharge `C_d` et aires **fournis par l'appelant**),
//! l'amortissement visqueux est **linéaire** (coefficient `c` **fourni**).
//! Fluide **incompressible**, régime **établi** ; la cavitation et la
//! compressibilité du gaz de rappel **ne sont pas modélisées**. Toutes les
//! constantes matériau / procédé (masse volumique, coefficient de décharge,
//! aires, coefficient d'amortissement) sont des **données fournies par
//! l'appelant** — aucune valeur « par défaut » n'est inventée ici. Complète
//! [`crate::air_spring`] et [`crate::vibration_isolation`].

/// Effort d'amortissement d'un dashpot à orifice, quadratique en vitesse
/// `F_o = 0,5·ρ·A_p³·v² / (C_d·A_o)²`.
///
/// `discharge_coefficient` coefficient de décharge de l'orifice (sans dimension),
/// `piston_area` aire du piston (m²), `orifice_area` aire de l'orifice (m²),
/// `fluid_density` masse volumique du fluide (kg·m⁻³), `piston_velocity` vitesse
/// du piston (m·s⁻¹, quelconque) ; renvoie l'effort d'amortissement en N.
///
/// Panique si `discharge_coefficient`, `piston_area`, `orifice_area` ou
/// `fluid_density` est `<= 0`, ou si `piston_velocity` n'est pas fini.
pub fn shockabs_orifice_force(
    discharge_coefficient: f64,
    piston_area: f64,
    orifice_area: f64,
    fluid_density: f64,
    piston_velocity: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0
            && piston_area > 0.0
            && orifice_area > 0.0
            && fluid_density > 0.0,
        "coefficient de décharge, aire de piston, aire d'orifice et masse volumique strictement positifs requis"
    );
    assert!(
        piston_velocity.is_finite(),
        "vitesse de piston finie requise"
    );
    let throat = discharge_coefficient * orifice_area;
    0.5 * fluid_density * piston_area.powi(3) * piston_velocity * piston_velocity / throat.powi(2)
}

/// Effort d'amortissement visqueux **linéaire** `F_v = c·v`.
///
/// `damping_coefficient` coefficient d'amortissement visqueux (N·s·m⁻¹),
/// `piston_velocity` vitesse du piston (m·s⁻¹, quelconque) ; renvoie l'effort
/// d'amortissement en N (de même signe que la vitesse).
///
/// Panique si `damping_coefficient` est `<= 0` ou si `piston_velocity` n'est pas
/// fini.
pub fn shockabs_linear_damping_force(damping_coefficient: f64, piston_velocity: f64) -> f64 {
    assert!(
        damping_coefficient > 0.0,
        "coefficient d'amortissement visqueux strictement positif requis"
    );
    assert!(
        piston_velocity.is_finite(),
        "vitesse de piston finie requise"
    );
    damping_coefficient * piston_velocity
}

/// Énergie dissipée par l'amortisseur sur une course `E = F_moy·s`.
///
/// `average_force` effort d'amortissement moyen sur la course (N), `stroke`
/// course (m) ; renvoie l'énergie dissipée en J.
///
/// Panique si un paramètre est `<= 0`.
pub fn shockabs_energy_dissipated(average_force: f64, stroke: f64) -> f64 {
    assert!(
        average_force > 0.0 && stroke > 0.0,
        "effort moyen et course strictement positifs requis"
    );
    average_force * stroke
}

/// Amortissement critique du système masse-ressort monté `c_c = 2·√(m·k)`.
///
/// `mass` masse montée (kg), `stiffness` raideur montée (N·m⁻¹) ; renvoie
/// l'amortissement critique en N·s·m⁻¹.
///
/// Panique si un paramètre est `<= 0`.
pub fn shockabs_critical_damping(mass: f64, stiffness: f64) -> f64 {
    assert!(
        mass > 0.0 && stiffness > 0.0,
        "masse et raideur strictement positives requises"
    );
    2.0 * (mass * stiffness).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn orifice_force_realistic_hand_calc() {
        // ρ = 1000 kg/m³ ; A_p = 0,01 m² ; A_o = 0,001 m² ; C_d = 1,0 ; v = 2 m/s.
        // A_p³ = 1e-6 ; v² = 4 ; (C_d·A_o)² = 1e-6 :
        // F = 0,5·1000·1e-6·4 / 1e-6 = 0,5·1000·4 = 2000 N.
        let f = shockabs_orifice_force(1.0, 0.01, 0.001, 1000.0, 2.0);
        assert_relative_eq!(f, 2000.0, epsilon = 1e-6);
    }

    #[test]
    fn orifice_force_quadratic_in_velocity() {
        // F ∝ v² : doubler la vitesse quadruple l'effort ; le signe est indifférent.
        let base = shockabs_orifice_force(0.6, 0.01, 0.001, 850.0, 1.5);
        let doubled = shockabs_orifice_force(0.6, 0.01, 0.001, 850.0, 3.0);
        assert_relative_eq!(doubled, 4.0 * base, epsilon = 1e-9);
        let reversed = shockabs_orifice_force(0.6, 0.01, 0.001, 850.0, -1.5);
        assert_relative_eq!(reversed, base, epsilon = 1e-9);
    }

    #[test]
    fn orifice_force_scales_with_piston_area_cubed() {
        // F ∝ A_p³ : doubler l'aire du piston multiplie l'effort par 8.
        let base = shockabs_orifice_force(0.6, 0.01, 0.001, 850.0, 2.0);
        let doubled = shockabs_orifice_force(0.6, 0.02, 0.001, 850.0, 2.0);
        assert_relative_eq!(doubled, 8.0 * base, epsilon = 1e-9);
    }

    #[test]
    fn orifice_force_inverse_square_in_orifice_area() {
        // F ∝ 1/(C_d·A_o)² : doubler l'aire d'orifice divise l'effort par 4.
        let base = shockabs_orifice_force(0.6, 0.01, 0.001, 850.0, 2.0);
        let wider = shockabs_orifice_force(0.6, 0.01, 0.002, 850.0, 2.0);
        assert_relative_eq!(wider, base / 4.0, epsilon = 1e-9);
    }

    #[test]
    fn linear_damping_and_energy_identities() {
        // F_v = c·v linéaire : c = 5000 N·s/m ; v = 2 m/s → 10 000 N.
        let fv = shockabs_linear_damping_force(5000.0, 2.0);
        assert_relative_eq!(fv, 10_000.0, epsilon = 1e-9);
        assert_relative_eq!(
            shockabs_linear_damping_force(5000.0, 4.0),
            2.0 * fv,
            epsilon = 1e-9
        );
        // E = F_moy·s : F = 2000 N sur s = 0,05 m → 100 J.
        let e = shockabs_energy_dissipated(2000.0, 0.05);
        assert_relative_eq!(e, 100.0, epsilon = 1e-9);
    }

    #[test]
    fn critical_damping_hand_calc_and_sqrt_scaling() {
        // c_c = 2·√(m·k) : m = 100 kg ; k = 10 000 N/m → 2·√(1e6) = 2000 N·s/m.
        let cc = shockabs_critical_damping(100.0, 10_000.0);
        assert_relative_eq!(cc, 2000.0, epsilon = 1e-9);
        // c_c ∝ √k : quadrupler la raideur double l'amortissement critique.
        let stiffer = shockabs_critical_damping(100.0, 40_000.0);
        assert_relative_eq!(stiffer, 2.0 * cc, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "strictement positifs requis")]
    fn orifice_force_rejects_zero_orifice_area() {
        let _ = shockabs_orifice_force(0.6, 0.01, 0.0, 850.0, 2.0);
    }
}
