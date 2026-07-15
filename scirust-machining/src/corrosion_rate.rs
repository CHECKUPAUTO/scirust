//! **Vitesse de corrosion uniforme** — taux de pénétration à partir d'une perte
//! de masse (CPR) et conversion électrochimique courant ↔ vitesse par la **loi
//! de Faraday**.
//!
//! ```text
//! taux de pénétration     CPR = K·Δm / (ρ·A·t)          (unité fixée par K)
//! perte de masse          Δm  = CPR·ρ·A·t / K            (réciproque du CPR)
//! vitesse ↔ courant       CR  = M·i / (z·F·ρ)            (m/s)
//! courant ↔ vitesse       i   = CR·z·F·ρ / M             (A/m²)
//! ```
//!
//! `Δm` perte de masse (unité au choix de l'appelant, cohérente avec `K`), `ρ`
//! masse volumique du métal, `A` aire exposée, `t` durée d'exposition, `K`
//! **facteur d'unités fourni** convertissant vers l'unité de CPR visée (p. ex.
//! `mm/an`), `CPR` taux de pénétration, `M` masse molaire (kg/mol), `i` densité
//! de courant de corrosion (A/m²), `z` valence (nombre d'électrons échangés,
//! sans dimension), `F` constante de Faraday (C/mol, voir [`CORROSION_FARADAY`]),
//! `CR` vitesse de corrosion (m/s).
//!
//! **Convention** : pour le lien électrochimique, SI cohérent (`M` en kg/mol,
//! `i` en A/m², `ρ` en kg/m³, `CR` en m/s). Pour le CPR, les unités sont
//! **entièrement portées par `K`** : l'appelant garantit la cohérence entre
//! `K`, `Δm`, `ρ`, `A` et `t`.
//!
//! **Limite honnête** : modèle de **corrosion uniforme** (attaque homogène sur
//! toute l'aire, rendement faradique de 100 %). Il ne décrit **ni** la
//! corrosion par piqûration, **ni** la corrosion galvanique localisée, **ni**
//! les effets de bords ou de répartition de courant. Les constantes physiques,
//! les propriétés matière (`M`, `ρ`, `z`) et le facteur d'unités `K` sont
//! **fournis par l'appelant** ; aucune valeur « par défaut » n'est inventée.
//! Voir [`crate::electroplating`].

/// Constante de Faraday `F = 96485` C/mol (charge d'une mole d'électrons).
pub const CORROSION_FARADAY: f64 = 96485.0;

/// Taux de pénétration de la corrosion uniforme `CPR = K·Δm / (ρ·A·t)`.
///
/// L'unité du résultat est celle induite par le facteur d'unités `unit_factor`
/// (`K`), qui doit être cohérent avec les unités de `mass_loss`, `density`,
/// `area` et `time`.
///
/// Panique si `unit_factor < 0`, `mass_loss < 0`, `density <= 0`, `area <= 0`
/// ou `time <= 0`.
pub fn corrosion_penetration_rate(
    unit_factor: f64,
    mass_loss: f64,
    density: f64,
    area: f64,
    time: f64,
) -> f64 {
    assert!(unit_factor >= 0.0, "le facteur d'unités K ≥ 0 requis");
    assert!(mass_loss >= 0.0, "la perte de masse Δm ≥ 0 requise");
    assert!(
        density > 0.0,
        "la masse volumique ρ doit être strictement positive"
    );
    assert!(
        area > 0.0,
        "l'aire exposée A doit être strictement positive"
    );
    assert!(time > 0.0, "la durée t doit être strictement positive");
    unit_factor * mass_loss / (density * area * time)
}

/// Perte de masse correspondant à un taux de pénétration
/// `Δm = CPR·ρ·A·t / K` (réciproque de [`corrosion_penetration_rate`]).
///
/// Panique si `cpr < 0`, `density < 0`, `area < 0`, `time < 0` ou
/// `unit_factor <= 0`.
pub fn cpr_mass_loss(cpr: f64, density: f64, area: f64, time: f64, unit_factor: f64) -> f64 {
    assert!(cpr >= 0.0, "le taux de pénétration CPR ≥ 0 requis");
    assert!(density >= 0.0, "la masse volumique ρ ≥ 0 requise");
    assert!(area >= 0.0, "l'aire exposée A ≥ 0 requise");
    assert!(time >= 0.0, "la durée t ≥ 0 requise");
    assert!(
        unit_factor > 0.0,
        "le facteur d'unités K doit être strictement positif"
    );
    cpr * density * area * time / unit_factor
}

/// Vitesse de corrosion déduite d'une densité de courant `CR = M·i / (z·F·ρ)`
/// (m/s, SI cohérent).
///
/// Panique si `current_density < 0`, `molar_mass <= 0`, `density <= 0` ou
/// `valence <= 0`.
pub fn corrosion_rate_from_current(
    current_density: f64,
    molar_mass: f64,
    density: f64,
    valence: f64,
) -> f64 {
    assert!(
        current_density >= 0.0,
        "la densité de courant i ≥ 0 requise"
    );
    assert!(
        molar_mass > 0.0,
        "la masse molaire M doit être strictement positive"
    );
    assert!(
        density > 0.0,
        "la masse volumique ρ doit être strictement positive"
    );
    assert!(valence > 0.0, "la valence z doit être strictement positive");
    molar_mass * current_density / (valence * CORROSION_FARADAY * density)
}

/// Densité de courant de corrosion associée à une vitesse
/// `i = CR·z·F·ρ / M` (A/m², réciproque de [`corrosion_rate_from_current`]).
///
/// Panique si `corrosion_rate < 0`, `density < 0`, `molar_mass <= 0` ou
/// `valence < 0`.
pub fn faraday_corrosion_current(
    corrosion_rate: f64,
    density: f64,
    molar_mass: f64,
    valence: f64,
) -> f64 {
    assert!(
        corrosion_rate >= 0.0,
        "la vitesse de corrosion CR ≥ 0 requise"
    );
    assert!(density >= 0.0, "la masse volumique ρ ≥ 0 requise");
    assert!(
        molar_mass > 0.0,
        "la masse molaire M doit être strictement positive"
    );
    assert!(valence >= 0.0, "la valence z ≥ 0 requise");
    corrosion_rate * valence * CORROSION_FARADAY * density / molar_mass
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cpr_matches_definition() {
        // CPR = K·Δm/(ρ·A·t), identité de définition.
        let (k, dm, rho, a, t) = (87.6_f64, 250.0_f64, 7.87_f64, 12.0_f64, 720.0_f64);
        assert_relative_eq!(
            corrosion_penetration_rate(k, dm, rho, a, t),
            k * dm / (rho * a * t),
            epsilon = 1e-12
        );
    }

    #[test]
    fn cpr_and_mass_loss_are_reciprocal() {
        // Réciprocité : Δm → CPR → Δm doit reproduire Δm initial.
        let (k, rho, a, t) = (87.6_f64, 7.87_f64, 12.0_f64, 720.0_f64);
        let dm = 250.0_f64;
        let cpr = corrosion_penetration_rate(k, dm, rho, a, t);
        let dm_back = cpr_mass_loss(cpr, rho, a, t, k);
        assert_relative_eq!(dm_back, dm, epsilon = 1e-9);
    }

    #[test]
    fn cpr_proportional_to_mass_loss() {
        // CPR ∝ Δm : doubler la perte de masse double le taux de pénétration.
        let base = corrosion_penetration_rate(87.6, 100.0, 7.87, 10.0, 500.0);
        let doubled = corrosion_penetration_rate(87.6, 200.0, 7.87, 10.0, 500.0);
        assert_relative_eq!(doubled, 2.0 * base, epsilon = 1e-12);
    }

    #[test]
    fn current_and_rate_are_reciprocal() {
        // Réciprocité électrochimique : CR(i) → i doit reproduire i initial.
        let (i, m, rho, z) = (1.0e-2_f64, 55.85e-3_f64, 7870.0_f64, 2.0_f64);
        let cr = corrosion_rate_from_current(i, m, rho, z);
        let i_back = faraday_corrosion_current(cr, rho, m, z);
        assert_relative_eq!(i_back, i, epsilon = 1e-12);
    }

    #[test]
    fn realistic_iron_corrosion_rate() {
        // Fer : M=55.85e-3 kg/mol, ρ=7870 kg/m³, z=2, i=1e-2 A/m² (≈1 µA/cm²).
        // CR = M·i/(z·F·ρ) ≈ 3.677e-13 m/s (soit ≈ 0.0116 mm/an).
        let cr = corrosion_rate_from_current(1.0e-2, 55.85e-3, 7870.0, 2.0);
        assert_relative_eq!(
            cr,
            55.85e-3 * 1.0e-2 / (2.0 * CORROSION_FARADAY * 7870.0),
            epsilon = 1e-18
        );
        assert_relative_eq!(cr, 3.67757e-13, epsilon = 1e-17);
    }

    #[test]
    fn rate_inversely_proportional_to_valence() {
        // CR ∝ 1/z : doubler la valence divise la vitesse par deux.
        let z1 = corrosion_rate_from_current(1.0e-2, 63.5e-3, 8960.0, 1.0);
        let z2 = corrosion_rate_from_current(1.0e-2, 63.5e-3, 8960.0, 2.0);
        assert_relative_eq!(z2, z1 / 2.0, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "la valence z doit être strictement positive")]
    fn zero_valence_panics() {
        corrosion_rate_from_current(1.0e-2, 55.85e-3, 7870.0, 0.0);
    }
}
