//! **Galvanoplastie** — dépôt électrolytique par la **loi de Faraday** : masse
//! déposée, épaisseur de revêtement et durée nécessaire pour une épaisseur visée.
//!
//! ```text
//! masse déposée     m = M·I·t / (z·F)              (kg)
//! épaisseur         e = m / (ρ·A)                   (m)
//! durée visée       t = e·ρ·A·z·F / (M·I)           (s)
//! ```
//!
//! `M` masse molaire du métal déposé (kg/mol), `I` intensité du courant (A),
//! `t` durée d'électrolyse (s), `z` valence (nombre d'électrons échangés, sans
//! dimension), `F` constante de Faraday (C/mol, voir [`FARADAY`]), `m` masse
//! déposée (kg), `e` épaisseur du dépôt (m), `ρ` masse volumique du métal
//! (kg/m³), `A` aire cathodique revêtue (m²).
//!
//! **Convention** : SI. **Limite honnête** : modèle à **rendement de courant de
//! 100 %**. En pratique le rendement faradique est `< 1` (réactions parasites,
//! dégagement d'hydrogène…) ; multipliez la masse/épaisseur obtenue par le
//! facteur de rendement **fourni par l'appelant**. Le dépôt est supposé
//! **uniforme** sur toute l'aire cathodique (pas d'effet de bords ni de
//! répartition de courant). Les constantes physiques et les propriétés du métal
//! (`M`, `ρ`, `z`) sont **fournies par l'appelant** ; aucune valeur matériau
//! « par défaut » n'est inventée. Voir [`crate::thermal`].

/// Constante de Faraday `F = 96485` C/mol (charge d'une mole d'électrons).
pub const FARADAY: f64 = 96485.0;

/// Masse déposée par la loi de Faraday `m = M·I·t / (z·F)` (kg).
///
/// Panique si `molar_mass <= 0`, `current < 0`, `time < 0` ou `valence <= 0`.
pub fn plating_deposited_mass(molar_mass: f64, current: f64, time: f64, valence: f64) -> f64 {
    assert!(
        molar_mass > 0.0,
        "la masse molaire M doit être strictement positive"
    );
    assert!(current >= 0.0, "l'intensité I ≥ 0 requise");
    assert!(time >= 0.0, "la durée t ≥ 0 requise");
    assert!(valence > 0.0, "la valence z doit être strictement positive");
    molar_mass * current * time / (valence * FARADAY)
}

/// Épaisseur du dépôt supposé uniforme `e = m / (ρ·A)` (m).
///
/// Panique si `mass < 0`, `density <= 0` ou `area <= 0`.
pub fn plating_thickness(mass: f64, density: f64, area: f64) -> f64 {
    assert!(mass >= 0.0, "la masse m ≥ 0 requise");
    assert!(
        density > 0.0,
        "la masse volumique ρ doit être strictement positive"
    );
    assert!(
        area > 0.0,
        "l'aire cathodique A doit être strictement positive"
    );
    mass / (density * area)
}

/// Durée d'électrolyse pour atteindre une épaisseur visée
/// `t = e·ρ·A·z·F / (M·I)` (s).
///
/// Réciproque de la chaîne [`plating_deposited_mass`] → [`plating_thickness`].
///
/// Panique si `thickness < 0`, `density <= 0`, `area <= 0`, `molar_mass <= 0`,
/// `current <= 0` ou `valence <= 0`.
pub fn plating_time_for_thickness(
    thickness: f64,
    density: f64,
    area: f64,
    molar_mass: f64,
    current: f64,
    valence: f64,
) -> f64 {
    assert!(thickness >= 0.0, "l'épaisseur e ≥ 0 requise");
    assert!(
        density > 0.0,
        "la masse volumique ρ doit être strictement positive"
    );
    assert!(
        area > 0.0,
        "l'aire cathodique A doit être strictement positive"
    );
    assert!(
        molar_mass > 0.0,
        "la masse molaire M doit être strictement positive"
    );
    assert!(
        current > 0.0,
        "l'intensité I doit être strictement positive"
    );
    assert!(valence > 0.0, "la valence z doit être strictement positive");
    thickness * density * area * valence * FARADAY / (molar_mass * current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn deposited_mass_matches_faraday_law() {
        // Cuivre : M=63.5e-3 kg/mol, I=10 A, t=3600 s, z=2.
        // m = 63.5e-3·10·3600 / (2·96485) = 2286 / 192970 ≈ 1.1846e-2 kg.
        let m = plating_deposited_mass(63.5e-3, 10.0, 3600.0, 2.0);
        assert_relative_eq!(
            m,
            63.5e-3 * 10.0 * 3600.0 / (2.0 * FARADAY),
            epsilon = 1e-12
        );
        assert_relative_eq!(m, 1.184639e-2, epsilon = 1e-6);
    }

    #[test]
    fn mass_proportional_to_charge() {
        // m ∝ I·t : doubler la charge double la masse.
        let base = plating_deposited_mass(63.5e-3, 5.0, 1800.0, 2.0);
        let double_current = plating_deposited_mass(63.5e-3, 10.0, 1800.0, 2.0);
        let double_time = plating_deposited_mass(63.5e-3, 5.0, 3600.0, 2.0);
        assert_relative_eq!(double_current, 2.0 * base, epsilon = 1e-12);
        assert_relative_eq!(double_time, 2.0 * base, epsilon = 1e-12);
    }

    #[test]
    fn mass_inversely_proportional_to_valence() {
        // m ∝ 1/z : doubler la valence divise la masse par deux.
        let z1 = plating_deposited_mass(58.7e-3, 8.0, 1200.0, 1.0);
        let z2 = plating_deposited_mass(58.7e-3, 8.0, 1200.0, 2.0);
        assert_relative_eq!(z2, z1 / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn thickness_recovers_mass_over_rho_area() {
        // e = m/(ρ·A), identité de définition.
        let m = 1.1846e-2_f64;
        let (rho, a) = (8960.0_f64, 0.05_f64);
        assert_relative_eq!(plating_thickness(m, rho, a), m / (rho * a), epsilon = 1e-15);
    }

    #[test]
    fn time_is_reciprocal_of_deposition_chain() {
        // Réciprocité : m(t) → e → t doit reproduire t initial.
        let (m_mol, i, t, z) = (63.5e-3_f64, 10.0_f64, 3600.0_f64, 2.0_f64);
        let (rho, a) = (8960.0_f64, 0.05_f64);
        let m = plating_deposited_mass(m_mol, i, t, z);
        let e = plating_thickness(m, rho, a);
        let t_back = plating_time_for_thickness(e, rho, a, m_mol, i, z);
        assert_relative_eq!(t_back, t, epsilon = 1e-6);
    }

    #[test]
    fn realistic_copper_thickness_case() {
        // Cuivre sur A=0.05 m², I=10 A, t=1 h, ρ=8960 kg/m³, M=63.5e-3, z=2.
        // m ≈ 1.1846e-2 kg ; e = m/(ρ·A) ≈ 1.1846e-2 / 448 ≈ 2.644e-5 m ≈ 26.4 µm.
        let m = plating_deposited_mass(63.5e-3, 10.0, 3600.0, 2.0);
        let e = plating_thickness(m, 8960.0, 0.05);
        assert_relative_eq!(e, 2.6443e-5, epsilon = 1e-8);
    }

    #[test]
    #[should_panic(expected = "la valence z doit être strictement positive")]
    fn zero_valence_panics() {
        plating_deposited_mass(63.5e-3, 10.0, 3600.0, 0.0);
    }
}
