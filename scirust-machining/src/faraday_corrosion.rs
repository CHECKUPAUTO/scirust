//! **Vitesse de corrosion électrochimique (loi de Faraday)** — perte de masse,
//! vitesse de pénétration et courant réciproque pour une attaque uniforme.
//!
//! ```text
//! perte de masse       dm/dt = I·M / (z·F)              (g/s si M en g/mol)
//! vitesse pénétration  v     = i·M / (z·F·ρ)            (m/s)
//! courant réciproque   i     = v·z·F·ρ / M              (A/m²)
//! ```
//!
//! `I` courant de corrosion (A = C/s), `i` densité de courant de corrosion
//! (A/m²), `M` masse molaire (g/mol pour la perte de masse en g/s, kg/mol pour
//! la vitesse en m/s), `z` valence (nombre d'électrons échangés, sans
//! dimension), `F` constante de Faraday (C/mol, voir [`FARADAY_CONSTANT`]), `ρ`
//! masse volumique du métal (kg/m³ pour une vitesse en m/s), `v` vitesse de
//! pénétration (m/s), `dm/dt` débit massique perdu.
//!
//! **Convention** : les unités sont **portées par l'appelant** et doivent être
//! cohérentes. Pour la perte de masse en g/s, prendre `M` en g/mol et `I` en A.
//! Pour la vitesse de pénétration en m/s, prendre le système SI cohérent (`M`
//! en kg/mol, `i` en A/m², `ρ` en kg/m³).
//!
//! **Limite honnête** : modèle de **corrosion électrochimique uniforme** (loi
//! de Faraday, rendement faradique unitaire de 100 %). Il ne prédit **ni** la
//! corrosion localisée (piqûration, crevasse, galvanique), **ni** la cinétique
//! électrochimique : le courant est **fourni**. La masse molaire, la valence,
//! la masse volumique et la constante de Faraday sont **fournies par
//! l'appelant** ; aucune valeur « par défaut » n'est inventée. Voir
//! [`crate::corrosion_rate`].

/// Constante de Faraday `F = 96485` C/mol (charge d'une mole d'électrons).
pub const FARADAY_CONSTANT: f64 = 96485.0;

/// Débit massique perdu par corrosion `dm/dt = I·M / (z·F)`.
///
/// Résultat en g/s si `atomic_mass` est en g/mol et `current` en A ; l'appelant
/// garantit la cohérence des unités avec `faraday_constant`.
///
/// Panique si `current < 0`, `atomic_mass <= 0`, `valence <= 0` ou
/// `faraday_constant <= 0`.
pub fn faraday_mass_loss_rate(
    current: f64,
    atomic_mass: f64,
    valence: f64,
    faraday_constant: f64,
) -> f64 {
    assert!(current >= 0.0, "le courant de corrosion I ≥ 0 requis");
    assert!(
        atomic_mass > 0.0,
        "la masse molaire M doit être strictement positive"
    );
    assert!(valence > 0.0, "la valence z doit être strictement positive");
    assert!(
        faraday_constant > 0.0,
        "la constante de Faraday F doit être strictement positive"
    );
    current * atomic_mass / (valence * faraday_constant)
}

/// Vitesse de pénétration de la corrosion `v = i·M / (z·F·ρ)` (m/s en SI).
///
/// SI cohérent attendu : `current_density` en A/m², `atomic_mass` en kg/mol,
/// `density` en kg/m³ ; le résultat est alors en m/s.
///
/// Panique si `current_density < 0`, `atomic_mass <= 0`, `valence <= 0`,
/// `density <= 0` ou `faraday_constant <= 0`.
pub fn faraday_penetration_rate(
    current_density: f64,
    atomic_mass: f64,
    valence: f64,
    density: f64,
    faraday_constant: f64,
) -> f64 {
    assert!(
        current_density >= 0.0,
        "la densité de courant i ≥ 0 requise"
    );
    assert!(
        atomic_mass > 0.0,
        "la masse molaire M doit être strictement positive"
    );
    assert!(valence > 0.0, "la valence z doit être strictement positive");
    assert!(
        density > 0.0,
        "la masse volumique ρ doit être strictement positive"
    );
    assert!(
        faraday_constant > 0.0,
        "la constante de Faraday F doit être strictement positive"
    );
    current_density * atomic_mass / (valence * faraday_constant * density)
}

/// Densité de courant de corrosion associée à une vitesse de pénétration
/// `i = v·z·F·ρ / M` (A/m², réciproque de [`faraday_penetration_rate`]).
///
/// SI cohérent attendu : `penetration_rate` en m/s, `atomic_mass` en kg/mol,
/// `density` en kg/m³ ; le résultat est alors en A/m².
///
/// Panique si `penetration_rate < 0`, `atomic_mass <= 0`, `valence < 0`,
/// `density < 0` ou `faraday_constant < 0`.
pub fn faraday_corrosion_current_from_rate(
    penetration_rate: f64,
    atomic_mass: f64,
    valence: f64,
    density: f64,
    faraday_constant: f64,
) -> f64 {
    assert!(
        penetration_rate >= 0.0,
        "la vitesse de pénétration v ≥ 0 requise"
    );
    assert!(
        atomic_mass > 0.0,
        "la masse molaire M doit être strictement positive"
    );
    assert!(valence >= 0.0, "la valence z ≥ 0 requise");
    assert!(density >= 0.0, "la masse volumique ρ ≥ 0 requise");
    assert!(
        faraday_constant >= 0.0,
        "la constante de Faraday F ≥ 0 requise"
    );
    penetration_rate * valence * faraday_constant * density / atomic_mass
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mass_loss_matches_definition() {
        // dm/dt = I·M/(z·F), identité de définition.
        let (i, m, z) = (1.5_f64, 55.85_f64, 2.0_f64);
        assert_relative_eq!(
            faraday_mass_loss_rate(i, m, z, FARADAY_CONSTANT),
            i * m / (z * FARADAY_CONSTANT),
            epsilon = 1e-15
        );
    }

    #[test]
    fn mass_loss_iron_reference() {
        // Fer : I=1 A, M=55.85 g/mol, z=2, F=96485 C/mol.
        // dm/dt = 55.85/(2·96485) ≈ 2.89423e-4 g/s.
        let rate = faraday_mass_loss_rate(1.0, 55.85, 2.0, FARADAY_CONSTANT);
        assert_relative_eq!(rate, 2.894232e-4, epsilon = 1e-9);
    }

    #[test]
    fn mass_loss_proportional_to_current() {
        // dm/dt ∝ I : tripler le courant triple le débit massique perdu.
        let base = faraday_mass_loss_rate(1.0, 63.5, 2.0, FARADAY_CONSTANT);
        let tripled = faraday_mass_loss_rate(3.0, 63.5, 2.0, FARADAY_CONSTANT);
        assert_relative_eq!(tripled, 3.0 * base, epsilon = 1e-15);
    }

    #[test]
    fn penetration_and_current_are_reciprocal() {
        // Réciprocité : v(i) → i doit reproduire i initial (fer, SI cohérent).
        let (i, m, z, rho) = (1.0e-2_f64, 55.85e-3_f64, 2.0_f64, 7870.0_f64);
        let v = faraday_penetration_rate(i, m, z, rho, FARADAY_CONSTANT);
        let i_back = faraday_corrosion_current_from_rate(v, m, z, rho, FARADAY_CONSTANT);
        assert_relative_eq!(i_back, i, epsilon = 1e-15);
    }

    #[test]
    fn penetration_rate_iron_reference() {
        // Fer : i=1e-2 A/m², M=55.85e-3 kg/mol, z=2, ρ=7870 kg/m³.
        // v = M·i/(z·F·ρ) ≈ 3.67757e-13 m/s.
        let v = faraday_penetration_rate(1.0e-2, 55.85e-3, 2.0, 7870.0, FARADAY_CONSTANT);
        assert_relative_eq!(
            v,
            55.85e-3 * 1.0e-2 / (2.0 * FARADAY_CONSTANT * 7870.0),
            epsilon = 1e-20
        );
        assert_relative_eq!(v, 3.67757e-13, epsilon = 1e-17);
    }

    #[test]
    fn penetration_inversely_proportional_to_valence() {
        // v ∝ 1/z : doubler la valence divise la vitesse par deux.
        let z1 = faraday_penetration_rate(1.0e-2, 63.5e-3, 1.0, 8960.0, FARADAY_CONSTANT);
        let z2 = faraday_penetration_rate(1.0e-2, 63.5e-3, 2.0, 8960.0, FARADAY_CONSTANT);
        assert_relative_eq!(z2, z1 / 2.0, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "la valence z doit être strictement positive")]
    fn zero_valence_panics() {
        faraday_mass_loss_rate(1.0, 55.85, 0.0, FARADAY_CONSTANT);
    }
}
