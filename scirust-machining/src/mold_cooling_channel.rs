//! Canaux de refroidissement de moule — **débit du fluide caloporteur** et
//! **puissance thermique évacuée** en conduite circulaire.
//!
//! ```text
//! nombre de Reynolds        Re = ρ·v·D / μ
//! débit volumique           Q  = v·(π/4)·D²
//! vitesse depuis le débit   v  = Q / ((π/4)·D²)
//! puissance évacuée         Q̇  = qm·c·ΔT
//! débit massique requis     qm = Q̇ / (c·ΔT)
//! ```
//!
//! `ρ` masse volumique du fluide (kg/m³), `v` vitesse moyenne débitante (m/s),
//! `D` diamètre hydraulique du canal (m), `μ` viscosité dynamique (Pa·s),
//! `Re` nombre de Reynolds (sans dimension), `Q` débit volumique (m³/s),
//! `qm` débit massique (kg/s), `c` capacité thermique massique (J/(kg·K)),
//! `ΔT` échauffement du fluide entre entrée et sortie (K, ou °C — seule la
//! différence intervient), `Q̇` puissance thermique évacuée (W).
//!
//! **Convention** : SI cohérent (mètre, kilogramme, seconde, kelvin). **Limite
//! honnête** : écoulement **établi** en conduite circulaire pleine, propriétés
//! du fluide (`ρ`, `μ`, `c`) supposées constantes et **fournies par
//! l'appelant** — aucune valeur matériau ni de fluide n'est supposée ici. Pour
//! une évacuation efficace on vise un régime **turbulent** (`Re > 4000`) ;
//! [`coolant_is_turbulent`] ne fait qu'appliquer ce seuil usuel, sans corriger
//! la zone de transition.

use core::f64::consts::PI;

/// Seuil usuel au-delà duquel l'écoulement en conduite est tenu pour turbulent.
pub const COOLANT_TURBULENT_REYNOLDS: f64 = 4000.0;

/// Nombre de Reynolds du fluide caloporteur `Re = ρ·v·D / μ` (sans dimension).
///
/// Panique si `density <= 0`, `diameter <= 0` ou `dynamic_viscosity <= 0`
/// (vitesse nulle admise : `Re = 0`).
pub fn coolant_reynolds(density: f64, velocity: f64, diameter: f64, dynamic_viscosity: f64) -> f64 {
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive"
    );
    assert!(diameter > 0.0, "le diamètre doit être strictement positif");
    assert!(
        dynamic_viscosity > 0.0,
        "la viscosité dynamique doit être strictement positive"
    );
    density * velocity * diameter / dynamic_viscosity
}

/// Indique si le régime est **turbulent** selon le seuil usuel
/// `Re > 4000` ([`COOLANT_TURBULENT_REYNOLDS`]).
///
/// Panique si `reynolds < 0`.
pub fn coolant_is_turbulent(reynolds: f64) -> bool {
    assert!(reynolds >= 0.0, "le nombre de Reynolds doit être positif");
    reynolds > COOLANT_TURBULENT_REYNOLDS
}

/// Débit volumique en conduite circulaire `Q = v·(π/4)·D²` (m³/s).
/// Réciproque de [`coolant_velocity_from_flow_rate`].
///
/// Panique si `diameter <= 0` (vitesse nulle admise : `Q = 0`).
pub fn coolant_flow_rate(velocity: f64, diameter: f64) -> f64 {
    assert!(diameter > 0.0, "le diamètre doit être strictement positif");
    velocity * (PI / 4.0) * diameter * diameter
}

/// Vitesse moyenne débitante déduite du débit volumique
/// `v = Q / ((π/4)·D²)` (m/s). Inverse de [`coolant_flow_rate`].
///
/// Panique si `flow_rate < 0` ou `diameter <= 0`.
pub fn coolant_velocity_from_flow_rate(flow_rate: f64, diameter: f64) -> f64 {
    assert!(flow_rate >= 0.0, "le débit volumique doit être positif");
    assert!(diameter > 0.0, "le diamètre doit être strictement positif");
    flow_rate / ((PI / 4.0) * diameter * diameter)
}

/// Puissance thermique évacuée par le fluide `Q̇ = qm·c·ΔT` (W).
/// Réciproque de [`mold_mass_flow_for_heat_removal`].
///
/// Panique si `mass_flow < 0`, `specific_heat <= 0` ou `temp_rise < 0`.
pub fn mold_heat_removal_rate(mass_flow: f64, specific_heat: f64, temp_rise: f64) -> f64 {
    assert!(mass_flow >= 0.0, "le débit massique doit être positif");
    assert!(
        specific_heat > 0.0,
        "la capacité thermique massique doit être strictement positive"
    );
    assert!(
        temp_rise >= 0.0,
        "l'échauffement du fluide doit être positif"
    );
    mass_flow * specific_heat * temp_rise
}

/// Débit massique requis pour évacuer une puissance donnée
/// `qm = Q̇ / (c·ΔT)` (kg/s). Inverse de [`mold_heat_removal_rate`].
///
/// Panique si `heat_rate < 0`, `specific_heat <= 0` ou `temp_rise <= 0`.
pub fn mold_mass_flow_for_heat_removal(heat_rate: f64, specific_heat: f64, temp_rise: f64) -> f64 {
    assert!(heat_rate >= 0.0, "la puissance évacuée doit être positive");
    assert!(
        specific_heat > 0.0,
        "la capacité thermique massique doit être strictement positive"
    );
    assert!(
        temp_rise > 0.0,
        "l'échauffement du fluide doit être strictement positif"
    );
    heat_rate / (specific_heat * temp_rise)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reynolds_scales_linearly_with_velocity() {
        // Re ∝ v : doubler la vitesse double le Reynolds.
        let re1 = coolant_reynolds(1000.0, 1.0, 0.01, 1.0e-3);
        let re2 = coolant_reynolds(1000.0, 2.0, 0.01, 1.0e-3);
        assert_relative_eq!(re2 / re1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn flow_rate_and_velocity_are_reciprocal() {
        // Réciprocité : v -> Q -> v redonne la vitesse de départ.
        let (v, d) = (1.5_f64, 0.012_f64);
        let q = coolant_flow_rate(v, d);
        assert_relative_eq!(coolant_velocity_from_flow_rate(q, d), v, epsilon = 1e-12);
    }

    #[test]
    fn flow_rate_scales_with_diameter_squared() {
        // Q ∝ D² à vitesse fixée : doubler D quadruple le débit.
        let q1 = coolant_flow_rate(2.0, 0.008);
        let q2 = coolant_flow_rate(2.0, 0.016);
        assert_relative_eq!(q2 / q1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn heat_removal_and_mass_flow_are_reciprocal() {
        // Réciprocité : qm -> Q̇ -> qm redonne le débit massique de départ.
        let (qm, c, dt) = (0.2_f64, 4180.0_f64, 5.0_f64);
        let q_dot = mold_heat_removal_rate(qm, c, dt);
        assert_relative_eq!(
            mold_mass_flow_for_heat_removal(q_dot, c, dt),
            qm,
            epsilon = 1e-12
        );
    }

    #[test]
    fn realistic_case_matches_closed_form() {
        // Cas chiffré : eau à 20 °C, D = 10 mm, v = 1 m/s.
        // ρ = 998 kg/m³, μ = 1.0e-3 Pa·s -> Re = 998·1·0.01/1e-3 = 9980 (turbulent).
        let re = coolant_reynolds(998.0, 1.0, 0.010, 1.0e-3);
        assert_relative_eq!(re, 9980.0, epsilon = 1e-9);
        assert!(coolant_is_turbulent(re));
        // Débit : Q = 1·(π/4)·0.01² m³/s.
        let q = coolant_flow_rate(1.0, 0.010);
        assert_relative_eq!(q, (PI / 4.0) * 0.010_f64.powi(2), epsilon = 1e-15);
        // Puissance évacuée : qm = ρ·Q, c = 4182 J/(kg·K), ΔT = 5 K.
        let qm = 998.0 * q;
        let q_dot = mold_heat_removal_rate(qm, 4182.0, 5.0);
        assert_relative_eq!(q_dot, qm * 4182.0 * 5.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "viscosité")]
    fn zero_viscosity_panics() {
        // μ = 0 rend le Reynolds indéfini.
        coolant_reynolds(1000.0, 1.0, 0.01, 0.0);
    }
}
