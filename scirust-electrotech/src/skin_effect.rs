//! **Effet de peau** — module de profondeur de peau, de rapport de résistance
//! alternatif/continu et d'aire effective de conduction pour un conducteur
//! cylindrique parcouru par un courant sinusoïdal.
//!
//! ```text
//! profondeur de peau (ω)   δ      = √(2·ρ / (ω·µ))
//! profondeur de peau (f)   δ      = √(ρ / (π·f·µ))
//! rapport R_ac/R_dc épais  ratio  = r / (2·δ)          (valable pour r >> δ)
//! aire effective couronne  A_eff  = π·(r² − (r − δ)₊²) (approximation)
//! ```
//!
//! `δ` profondeur de peau (m), `ρ` résistivité électrique du matériau (Ω·m),
//! `ω` pulsation du régime sinusoïdal (rad/s), `f` fréquence du régime
//! sinusoïdal (Hz), `µ` perméabilité magnétique absolue du matériau (H/m),
//! `r` rayon du conducteur cylindrique (m), `ratio` rapport R_ac/R_dc (sans
//! dimension), `A_eff` aire effective de conduction (m²) ; la notation `(·)₊`
//! désigne la partie positive `max(·, 0)`.
//!
//! **Convention** : SI ; résistivités en Ω·m, perméabilités en H/m, pulsations
//! en rad/s, fréquences en Hz, longueurs et rayons en m, aires en m² ; l'angle
//! implicite du régime est en **radians**. **Limite honnête** : conducteur
//! **cylindrique** en **régime sinusoïdal permanent** ; la profondeur de peau
//! `δ` concentre le courant en surface. Le rapport `R_ac/R_dc ≈ r/(2·δ)` n'est
//! qu'une **approximation haute fréquence valable pour r >> δ** (il tend vers
//! zéro à basse fréquence, ce qui est non physique — R_ac/R_dc ≥ 1 en réalité),
//! et l'aire effective en couronne est également **approchée**. Ce module
//! **néglige l'effet de proximité**. La résistivité `ρ` et la perméabilité `µ`
//! du matériau, ainsi que les grandeurs réseau (`ω`, `f`) et la géométrie (`r`),
//! sont **fournies par l'appelant** (fiches matériau, mesures) — aucune valeur
//! « par défaut » n'est inventée.

/// Profondeur de peau à partir de la pulsation `δ = √(2·ρ / (ω·µ))` (m).
///
/// Panique si `resistivity < 0`, si `angular_frequency <= 0` ou si
/// `permeability <= 0` (division par zéro ou racine d'un nombre négatif).
pub fn skin_depth(resistivity: f64, angular_frequency: f64, permeability: f64) -> f64 {
    assert!(resistivity >= 0.0, "la résistivité ρ doit être ≥ 0");
    assert!(
        angular_frequency > 0.0,
        "la pulsation ω doit être strictement positive"
    );
    assert!(
        permeability > 0.0,
        "la perméabilité µ doit être strictement positive"
    );
    (2.0 * resistivity / (angular_frequency * permeability)).sqrt()
}

/// Profondeur de peau à partir de la fréquence `δ = √(ρ / (π·f·µ))` (m).
///
/// Panique si `resistivity < 0`, si `frequency <= 0` ou si `permeability <= 0`
/// (division par zéro ou racine d'un nombre négatif).
pub fn skin_depth_from_frequency(resistivity: f64, frequency: f64, permeability: f64) -> f64 {
    assert!(resistivity >= 0.0, "la résistivité ρ doit être ≥ 0");
    assert!(
        frequency > 0.0,
        "la fréquence f doit être strictement positive"
    );
    assert!(
        permeability > 0.0,
        "la perméabilité µ doit être strictement positive"
    );
    (resistivity / (core::f64::consts::PI * frequency * permeability)).sqrt()
}

/// Rapport approché `R_ac/R_dc ≈ r/(2·δ)` d'un conducteur épais (sans
/// dimension), valable uniquement pour `r >> δ` (haute fréquence).
///
/// Panique si `conductor_radius < 0` ou si `skin_depth <= 0` (division par
/// zéro).
pub fn skin_ac_resistance_ratio_thick(conductor_radius: f64, skin_depth: f64) -> f64 {
    assert!(conductor_radius >= 0.0, "le rayon r doit être ≥ 0");
    assert!(
        skin_depth > 0.0,
        "la profondeur de peau δ doit être strictement positive"
    );
    conductor_radius / (2.0 * skin_depth)
}

/// Aire effective de conduction en couronne
/// `A_eff = π·(r² − (r − δ)₊²)` (m²), approximation ; si `δ ≥ r` la couronne
/// couvre toute la section et `A_eff = π·r²`.
///
/// Panique si `conductor_radius < 0` ou si `skin_depth < 0` (grandeurs
/// géométriques sans sens physique si négatives).
pub fn skin_effective_area(conductor_radius: f64, skin_depth: f64) -> f64 {
    assert!(conductor_radius >= 0.0, "le rayon r doit être ≥ 0");
    assert!(skin_depth >= 0.0, "la profondeur de peau δ doit être ≥ 0");
    let inner = (conductor_radius - skin_depth).max(0.0);
    core::f64::consts::PI * (conductor_radius * conductor_radius - inner.powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn both_skin_depth_forms_agree() {
        // Réciprocité : avec ω = 2·π·f, δ = √(2ρ/(ω·µ)) = √(ρ/(π·f·µ)) ; les deux
        // formes donnent la même profondeur de peau.
        let rho = 1.68e-8_f64;
        let mu = 4.0 * core::f64::consts::PI * 1e-7;
        let f = 50.0_f64;
        let omega = 2.0 * core::f64::consts::PI * f;
        let d1 = skin_depth(rho, omega, mu);
        let d2 = skin_depth_from_frequency(rho, f, mu);
        assert_relative_eq!(d1, d2, epsilon = 1e-12);
    }

    #[test]
    fn skin_depth_scales_with_inverse_sqrt_frequency() {
        // Proportionnalité : δ ∝ 1/√f ; quadrupler la fréquence divise la
        // profondeur de peau par deux.
        let rho = 0.5_f64;
        let mu = 1.0_f64;
        let d_low = skin_depth_from_frequency(rho, 1.0, mu);
        let d_high = skin_depth_from_frequency(rho, 4.0, mu);
        assert_relative_eq!(d_high, d_low / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn copper_skin_depth_at_50_hz() {
        // Cas chiffré réaliste : cuivre ρ = 1,68e-8 Ω·m, µ = µ₀ = 4π·10⁻⁷ H/m,
        // f = 50 Hz.
        //   πfµ = π·50·(4π·10⁻⁷) = 200·π²·10⁻⁷ = 1,973921e-4
        //   δ   = √(1,68e-8 / 1,973921e-4) = √(8,51099e-5) = 9,2255e-3 m ≈ 9,23 mm
        let rho = 1.68e-8_f64;
        let mu = 4.0 * core::f64::consts::PI * 1e-7;
        let d = skin_depth_from_frequency(rho, 50.0, mu);
        assert_relative_eq!(d, 9.2255e-3, epsilon = 1e-6);
    }

    #[test]
    fn thick_ratio_is_linear_in_radius() {
        // Cas chiffré et proportionnalité : ratio = r/(2δ) est linéaire en r.
        //   r = 2, δ = 1 → ratio = 2/(2·1) = 1
        //   doubler r double le ratio.
        assert_relative_eq!(
            skin_ac_resistance_ratio_thick(2.0, 1.0),
            1.0,
            epsilon = 1e-15
        );
        let ra = skin_ac_resistance_ratio_thick(2.0, 1.0);
        let rb = skin_ac_resistance_ratio_thick(4.0, 1.0);
        assert_relative_eq!(rb, 2.0 * ra, epsilon = 1e-15);
    }

    #[test]
    fn effective_area_annulus_and_full_section() {
        // Cas chiffré : r = 2, δ = 1 → couronne A_eff = π·(2² − (2−1)²)
        //   = π·(4 − 1) = 3π ≈ 9,424778.
        assert_relative_eq!(
            skin_effective_area(2.0, 1.0),
            3.0 * core::f64::consts::PI,
            epsilon = 1e-12
        );
        // Cas limite : δ ≥ r (basse fréquence) → toute la section conduit,
        //   A_eff = π·r² = π·2² = 4π.
        assert_relative_eq!(
            skin_effective_area(2.0, 5.0),
            4.0 * core::f64::consts::PI,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "la pulsation ω doit être strictement positive")]
    fn zero_angular_frequency_panics() {
        skin_depth(1.68e-8, 0.0, 1.0);
    }
}
