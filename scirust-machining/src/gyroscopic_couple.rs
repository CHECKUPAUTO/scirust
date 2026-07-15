//! **Couple gyroscopique** — la réaction de précession d'un rotor rapide dont
//! l'axe de rotation change d'orientation.
//!
//! ```text
//! moment cinétique     H = I·ω_spin                         (N·m·s)
//! couple gyroscopique  C = I·ω_spin·ω_precession            (N·m)
//! taux de précession   ω_precession = C / (I·ω_spin)        (rad/s)
//! ```
//!
//! `I` moment d'inertie polaire du rotor autour de son axe de rotation
//! (kg·m²), `ω_spin` vitesse de rotation propre (rad/s), `ω_precession`
//! vitesse angulaire de précession de l'axe (rad/s), `H` moment cinétique
//! (kg·m²/s = N·m·s), `C` couple gyroscopique (couple de réaction, N·m).
//!
//! **Convention** : SI, axes orthogonaux. **Limite honnête** : rotor
//! symétrique en rotation **rapide** (spin ≫ précession), axes de spin et de
//! précession **orthogonaux**, réaction gyroscopique idéale (paliers rigides,
//! pas de nutation ni de flexibilité d'arbre). Le moment d'inertie polaire et
//! les vitesses angulaires sont des données **fournies par l'appelant** ;
//! aucune valeur matériau/rotor n'est supposée par défaut. Voir
//! [`crate::balancing`] et [`crate::reflected_inertia`].

/// Moment cinétique d'un rotor `H = I·ω_spin` (N·m·s).
///
/// Panique si `polar_inertia < 0`.
pub fn gyroscopic_angular_momentum(polar_inertia: f64, spin_speed_rad: f64) -> f64 {
    assert!(
        polar_inertia >= 0.0,
        "le moment d'inertie polaire doit être positif ou nul"
    );
    polar_inertia * spin_speed_rad
}

/// Couple gyroscopique de réaction `C = I·ω_spin·ω_precession` (N·m).
///
/// Panique si `polar_inertia < 0`.
pub fn gyroscopic_couple(
    polar_inertia: f64,
    spin_speed_rad: f64,
    precession_speed_rad: f64,
) -> f64 {
    assert!(
        polar_inertia >= 0.0,
        "le moment d'inertie polaire doit être positif ou nul"
    );
    polar_inertia * spin_speed_rad * precession_speed_rad
}

/// Taux de précession déduit du couple `ω_precession = C / (I·ω_spin)` (rad/s).
///
/// Panique si `polar_inertia <= 0` ou `spin_speed_rad == 0`.
pub fn gyroscopic_precession_rate(couple: f64, polar_inertia: f64, spin_speed_rad: f64) -> f64 {
    assert!(
        polar_inertia > 0.0,
        "le moment d'inertie polaire doit être strictement positif"
    );
    assert!(
        spin_speed_rad != 0.0,
        "la vitesse de rotation propre doit être non nulle"
    );
    couple / (polar_inertia * spin_speed_rad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn couple_is_momentum_times_precession() {
        // C = (I·ω_spin)·ω_precession = H·ω_precession.
        let i = 0.8;
        let ws = 300.0;
        let wp = 1.5;
        let h = gyroscopic_angular_momentum(i, ws);
        assert_relative_eq!(gyroscopic_couple(i, ws, wp), h * wp, epsilon = 1e-12);
    }

    #[test]
    fn precession_rate_inverts_couple() {
        // Réciprocité : retrouver ω_precession à partir du couple qu'il produit.
        let i = 0.8;
        let ws = 300.0;
        let wp = 1.5;
        let c = gyroscopic_couple(i, ws, wp);
        assert_relative_eq!(gyroscopic_precession_rate(c, i, ws), wp, epsilon = 1e-12);
    }

    #[test]
    fn couple_scales_linearly_with_precession() {
        // Doubler la vitesse de précession double le couple gyroscopique.
        let c1 = gyroscopic_couple(0.8, 300.0, 1.5);
        let c2 = gyroscopic_couple(0.8, 300.0, 3.0);
        assert_relative_eq!(c2 / c1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_rotor_couple() {
        // Rotor I = 0,8 kg·m², spin 3000 tr/min = 3000·2π/60 = 314,159… rad/s,
        // précession 1,5 rad/s.
        // C = 0,8 · 314,159265… · 1,5 = 376,991118… N·m.
        let i = 0.8;
        let ws = 3000.0 * 2.0 * core::f64::consts::PI / 60.0;
        let wp = 1.5;
        let c = gyroscopic_couple(i, ws, wp);
        assert_relative_eq!(c, 376.991_118_430_775_1, epsilon = 1e-9);
    }

    #[test]
    fn momentum_is_linear_in_inertia() {
        // H proportionnel à I à vitesse de spin fixée.
        let h1 = gyroscopic_angular_momentum(0.4, 300.0);
        let h2 = gyroscopic_angular_momentum(0.8, 300.0);
        assert_relative_eq!(h2 / h1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn zero_precession_gives_zero_couple() {
        // Pas de changement d'orientation → aucun couple gyroscopique.
        assert_relative_eq!(gyroscopic_couple(0.8, 300.0, 0.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "vitesse de rotation propre")]
    fn zero_spin_precession_rate_panics() {
        gyroscopic_precession_rate(100.0, 0.8, 0.0);
    }
}
