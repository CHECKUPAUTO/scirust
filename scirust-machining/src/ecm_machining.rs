//! **Usinage électrochimique (ECM, loi de Faraday)** — débit volumétrique
//! enlevé, vitesse d'avance normale et jeu inter-électrode d'équilibre pour une
//! dissolution anodique à rendement de courant unitaire.
//!
//! ```text
//! débit enlevé     Q     = I·M / (z·F·ρ)              (m³/s)
//! vitesse d'avance v     = i·M / (z·F·ρ)              (m/s)
//! jeu d'équilibre  g     = U·κ / i                    (m)
//! courant / jeu    i     = U·κ / g                    (A/m²)
//! ```
//!
//! `I` courant total (A = C/s), `i` densité de courant (A/m²), `M` masse
//! atomique molaire (kg/mol pour un débit en m³/s), `z` valence de dissolution
//! (nombre d'électrons échangés, sans dimension), `F` constante de Faraday
//! (C/mol, voir [`ECM_FARADAY_CONSTANT`]), `ρ` masse volumique de l'anode
//! (kg/m³), `Q` débit volumétrique enlevé (m³/s), `v` vitesse d'avance normale
//! de l'outil (m/s), `U` tension inter-électrode (V), `κ` conductivité de
//! l'électrolyte (S/m), `g` jeu inter-électrode d'équilibre (m).
//!
//! **Convention** : SI cohérent. Pour un débit en m³/s, prendre `M` en kg/mol,
//! `ρ` en kg/m³ et `I` en A ; pour un jeu en m, prendre `U` en V, `κ` en S/m et
//! `i` en A/m².
//!
//! **Limite honnête** : modèle de **dissolution anodique** à **rendement de
//! courant unitaire** (loi de Faraday, 100 %). Le jeu d'équilibre correspond à
//! une avance donnée. Le modèle **ignore** la polarisation d'électrode, la
//! surtension et l'échauffement (ou le dégazage) de l'électrolyte qui modifient
//! la conductivité `κ`. La masse atomique, la valence, la masse volumique, la
//! conductivité de l'électrolyte et la constante de Faraday sont **fournies par
//! l'appelant** ; aucune valeur « par défaut » n'est inventée. Voir
//! [`crate::faraday_corrosion`].

/// Constante de Faraday `F = 96485` C/mol (charge d'une mole d'électrons).
pub const ECM_FARADAY_CONSTANT: f64 = 96485.0;

/// Débit volumétrique enlevé par dissolution anodique
/// `Q = I·M / (z·F·ρ)` (m³/s en SI).
///
/// SI cohérent attendu : `current` en A, `atomic_mass` en kg/mol, `density` en
/// kg/m³ ; le résultat est alors en m³/s.
///
/// Panique si `current < 0`, `atomic_mass <= 0`, `valence <= 0`,
/// `density <= 0` ou `faraday_constant <= 0`.
pub fn ecm_material_removal_rate(
    current: f64,
    atomic_mass: f64,
    valence: f64,
    density: f64,
    faraday_constant: f64,
) -> f64 {
    assert!(current >= 0.0, "le courant total I ≥ 0 requis");
    assert!(
        atomic_mass > 0.0,
        "la masse atomique M doit être strictement positive"
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
    current * atomic_mass / (valence * faraday_constant * density)
}

/// Vitesse d'avance normale de l'outil `v = i·M / (z·F·ρ)` (m/s en SI).
///
/// SI cohérent attendu : `current_density` en A/m², `atomic_mass` en kg/mol,
/// `density` en kg/m³ ; le résultat est alors en m/s.
///
/// Panique si `current_density < 0`, `atomic_mass <= 0`, `valence <= 0`,
/// `density <= 0` ou `faraday_constant <= 0`.
pub fn ecm_penetration_rate(
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
        "la masse atomique M doit être strictement positive"
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

/// Jeu inter-électrode d'équilibre `g = U·κ / i` (m).
///
/// SI cohérent attendu : `voltage` en V, `conductivity` en S/m,
/// `current_density` en A/m² ; le résultat est alors en m. Réciproque de
/// [`ecm_current_from_gap`].
///
/// Panique si `voltage < 0`, `conductivity < 0` ou `current_density <= 0`.
pub fn ecm_equilibrium_gap(voltage: f64, conductivity: f64, current_density: f64) -> f64 {
    assert!(voltage >= 0.0, "la tension U ≥ 0 requise");
    assert!(conductivity >= 0.0, "la conductivité κ ≥ 0 requise");
    assert!(
        current_density > 0.0,
        "la densité de courant i doit être strictement positive"
    );
    voltage * conductivity / current_density
}

/// Densité de courant pour un jeu inter-électrode donné `i = U·κ / g` (A/m²).
///
/// SI cohérent attendu : `voltage` en V, `conductivity` en S/m, `gap` en m ; le
/// résultat est alors en A/m². Réciproque de [`ecm_equilibrium_gap`].
///
/// Panique si `voltage < 0`, `conductivity < 0` ou `gap <= 0`.
pub fn ecm_current_from_gap(voltage: f64, conductivity: f64, gap: f64) -> f64 {
    assert!(voltage >= 0.0, "la tension U ≥ 0 requise");
    assert!(conductivity >= 0.0, "la conductivité κ ≥ 0 requise");
    assert!(gap > 0.0, "le jeu g doit être strictement positif");
    voltage * conductivity / gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn removal_rate_matches_definition() {
        // Q = I·M/(z·F·ρ), identité de définition (fer, SI cohérent).
        let (i, m, z, rho) = (1.0e3_f64, 55.85e-3_f64, 2.0_f64, 7870.0_f64);
        assert_relative_eq!(
            ecm_material_removal_rate(i, m, z, rho, ECM_FARADAY_CONSTANT),
            i * m / (z * ECM_FARADAY_CONSTANT * rho),
            epsilon = 1e-25
        );
    }

    #[test]
    fn removal_rate_iron_reference() {
        // Fer : I=1000 A, M=55.85e-3 kg/mol, z=2, F=96485 C/mol, ρ=7870 kg/m³.
        // Q = 1000·55.85e-3/(2·96485·7870) = 55.85/1.5186739e9 ≈ 3.67755e-8 m³/s.
        let q = ecm_material_removal_rate(1.0e3, 55.85e-3, 2.0, 7870.0, ECM_FARADAY_CONSTANT);
        assert_relative_eq!(q, 3.67755e-8, epsilon = 1e-12);
    }

    #[test]
    fn removal_rate_is_penetration_times_area() {
        // Q(I) = A·v(i) avec i = I/A : le débit vaut la vitesse d'avance fois
        // l'aire usinée, cohérence entre les deux formes de la loi de Faraday.
        let (current, area) = (500.0_f64, 5.0e-4_f64);
        let (m, z, rho) = (63.5e-3_f64, 2.0_f64, 8960.0_f64);
        let q = ecm_material_removal_rate(current, m, z, rho, ECM_FARADAY_CONSTANT);
        let v = ecm_penetration_rate(current / area, m, z, rho, ECM_FARADAY_CONSTANT);
        assert_relative_eq!(q, area * v, epsilon = 1e-18);
    }

    #[test]
    fn gap_and_current_are_reciprocal() {
        // Réciprocité : g(i) → i doit reproduire i initial (conditions ECM).
        let (u, kappa, i) = (15.0_f64, 20.0_f64, 1.0e6_f64);
        let g = ecm_equilibrium_gap(u, kappa, i);
        let i_back = ecm_current_from_gap(u, kappa, g);
        assert_relative_eq!(i_back, i, epsilon = 1e-6);
    }

    #[test]
    fn equilibrium_gap_reference() {
        // U=15 V, κ=20 S/m, i=1e6 A/m² : g = 15·20/1e6 = 3e-4 m = 0.3 mm.
        let g = ecm_equilibrium_gap(15.0, 20.0, 1.0e6);
        assert_relative_eq!(g, 3.0e-4, epsilon = 1e-15);
    }

    #[test]
    fn penetration_inversely_proportional_to_density() {
        // v ∝ 1/ρ : doubler la masse volumique divise la vitesse par deux.
        let v1 = ecm_penetration_rate(1.0e6, 55.85e-3, 2.0, 7870.0, ECM_FARADAY_CONSTANT);
        let v2 = ecm_penetration_rate(1.0e6, 55.85e-3, 2.0, 15740.0, ECM_FARADAY_CONSTANT);
        assert_relative_eq!(v2, v1 / 2.0, epsilon = 1e-18);
    }

    #[test]
    #[should_panic(expected = "la densité de courant i doit être strictement positive")]
    fn zero_current_density_gap_panics() {
        ecm_equilibrium_gap(15.0, 20.0, 0.0);
    }
}
