//! Débitmètre à diaphragme (plaque à orifice) — mesure de débit par perte de
//! charge à travers une restriction calibrée, avec facteur d'approche, débit
//! volumique associé, perte de charge visée et fraction de perte permanente.
//!
//! ```text
//! rapport des diamètres (β)
//!   β = d / D                                                    [sans dim.]
//! débit massique (avec facteur d'approche E = 1/√(1 − β⁴))
//!   ṁ = Cd / √(1 − β⁴) · A₀ · √(2·ρ·ΔP)                          [kg·s⁻¹]
//! débit volumique
//!   Q = ṁ / ρ                                                    [m³·s⁻¹]
//! perte de charge différentielle pour un débit visé
//!   ΔP = [ ṁ·√(1 − β⁴) / (Cd·A₀) ]² / (2·ρ)                      [Pa]
//! fraction de perte de charge permanente (approchée)
//!   ω = 1 − β²                                                   [sans dim.]
//! ```
//!
//! `d` diamètre de l'orifice [m], `D` diamètre intérieur de la conduite [m],
//! `β` rapport des diamètres [sans dimension, 0 < β < 1], `Cd` coefficient de
//! décharge [sans dimension], `A₀` section de l'orifice [m²], `ρ` masse
//! volumique du fluide [kg·m⁻³], `ΔP` pression différentielle mesurée aux prises
//! [Pa], `ṁ` débit massique [kg·s⁻¹], `Q` débit volumique [m³·s⁻¹], `ω` fraction
//! de la pression différentielle non récupérée en aval [sans dimension].
//!
//! **Limite honnête** : modèle à l'échelle des **opérations unitaires** pour un
//! **liquide** (facteur de dilatation ε = 1), en écoulement **incompressible,
//! turbulent établi**, conduite **pleine**. Le **coefficient de décharge** `Cd`
//! est **empirique** (il dépend du nombre de Reynolds, du type de prises de
//! pression et de la géométrie de la plaque) : il est **FOURNI** par l'appelant
//! d'après une norme (p. ex. ISO 5167 / Reader-Harris–Gallagher) ou un
//! étalonnage — jamais supposé ni calculé ici. La **masse volumique** `ρ` est
//! elle aussi **FOURNIE**. La fraction de perte permanente `ω = 1 − β²` est une
//! **approximation** commode ; la perte réelle dépend de la géométrie et se lit
//! sur les corrélations normatives. Aucune propriété d'état, aucune corrélation
//! de coefficient n'est inventée par ce module.

/// Rapport des diamètres β d'un diaphragme
/// `β = d / D`, orifice sur conduite (sans dimension).
///
/// `orifice_diameter` (d) diamètre de l'orifice [m], `pipe_diameter` (D)
/// diamètre intérieur de la conduite [m].
///
/// Panique si `d ≤ 0`, si `D ≤ 0`, ou si `d > D` (orifice plus large que la
/// conduite, β physiquement impossible).
pub fn orif_beta_ratio(orifice_diameter: f64, pipe_diameter: f64) -> f64 {
    assert!(
        orifice_diameter > 0.0,
        "d > 0 requis (diamètre de l'orifice)"
    );
    assert!(
        pipe_diameter > 0.0,
        "D > 0 requis (diamètre de la conduite)"
    );
    assert!(
        orifice_diameter <= pipe_diameter,
        "d ≤ D requis (l'orifice ne peut être plus large que la conduite)"
    );
    orifice_diameter / pipe_diameter
}

/// Débit massique à travers un diaphragme
/// `ṁ = Cd/√(1 − β⁴) · A₀ · √(2·ρ·ΔP)` (kg·s⁻¹), équation de la plaque à
/// orifice pour un liquide (ε = 1), le terme `1/√(1 − β⁴)` étant le facteur
/// d'approche de la vitesse.
///
/// `discharge_coefficient` (Cd) coefficient de décharge FOURNI [sans dimension],
/// `beta_ratio` (β) rapport des diamètres [sans dimension], `orifice_area` (A₀)
/// section de l'orifice [m²], `density` (ρ) masse volumique FOURNIE [kg·m⁻³],
/// `differential_pressure` (ΔP) pression différentielle mesurée [Pa].
///
/// Panique si `Cd ≤ 0`, si `β` hors de `[0, 1[`, si `A₀ ≤ 0`, si `ρ ≤ 0`, ou si
/// `ΔP < 0`.
pub fn orif_mass_flow(
    discharge_coefficient: f64,
    beta_ratio: f64,
    orifice_area: f64,
    density: f64,
    differential_pressure: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0,
        "Cd > 0 requis (coefficient de décharge)"
    );
    assert!(
        (0.0..1.0).contains(&beta_ratio),
        "0 ≤ β < 1 requis (rapport des diamètres ; β = 1 annule 1 − β⁴)"
    );
    assert!(orifice_area > 0.0, "A₀ > 0 requis (section de l'orifice)");
    assert!(density > 0.0, "ρ > 0 requis (masse volumique du fluide)");
    assert!(
        differential_pressure >= 0.0,
        "ΔP ≥ 0 requis (pression différentielle)"
    );
    (discharge_coefficient / (1.0 - beta_ratio.powi(4)).sqrt())
        * orifice_area
        * (2.0 * density * differential_pressure).sqrt()
}

/// Débit volumique à partir du débit massique
/// `Q = ṁ / ρ` (m³·s⁻¹), conversion à masse volumique constante.
///
/// `mass_flow` (ṁ) débit massique [kg·s⁻¹], `density` (ρ) masse volumique FOURNIE
/// [kg·m⁻³].
///
/// Panique si `ρ ≤ 0`.
pub fn orif_volumetric_flow(mass_flow: f64, density: f64) -> f64 {
    assert!(density > 0.0, "ρ > 0 requis (masse volumique du fluide)");
    mass_flow / density
}

/// Perte de charge différentielle pour un débit massique visé
/// `ΔP = [ṁ·√(1 − β⁴)/(Cd·A₀)]² / (2·ρ)` (Pa), inversion de l'équation du
/// diaphragme (réciproque de [`orif_mass_flow`]).
///
/// `mass_flow` (ṁ) débit massique visé [kg·s⁻¹], `discharge_coefficient` (Cd)
/// coefficient de décharge FOURNI [sans dimension], `beta_ratio` (β) rapport des
/// diamètres [sans dimension], `orifice_area` (A₀) section de l'orifice [m²],
/// `density` (ρ) masse volumique FOURNIE [kg·m⁻³].
///
/// Panique si `ṁ < 0`, si `Cd ≤ 0`, si `β` hors de `[0, 1[`, si `A₀ ≤ 0`, ou si
/// `ρ ≤ 0`.
pub fn orif_differential_pressure(
    mass_flow: f64,
    discharge_coefficient: f64,
    beta_ratio: f64,
    orifice_area: f64,
    density: f64,
) -> f64 {
    assert!(mass_flow >= 0.0, "ṁ ≥ 0 requis (débit massique)");
    assert!(
        discharge_coefficient > 0.0,
        "Cd > 0 requis (coefficient de décharge)"
    );
    assert!(
        (0.0..1.0).contains(&beta_ratio),
        "0 ≤ β < 1 requis (rapport des diamètres ; β = 1 annule 1 − β⁴)"
    );
    assert!(orifice_area > 0.0, "A₀ > 0 requis (section de l'orifice)");
    assert!(density > 0.0, "ρ > 0 requis (masse volumique du fluide)");
    let numerator =
        mass_flow * (1.0 - beta_ratio.powi(4)).sqrt() / (discharge_coefficient * orifice_area);
    numerator.powi(2) / (2.0 * density)
}

/// Fraction de perte de charge permanente (approchée)
/// `ω = 1 − β²` (sans dimension), part de la pression différentielle non
/// récupérée en aval du diaphragme.
///
/// `beta_ratio` (β) rapport des diamètres [sans dimension] : plus β est petit
/// (orifice serré), plus ω tend vers 1 (perte quasi totale) ; ω → 0 quand β → 1.
///
/// Panique si `β` hors de `[0, 1]`.
pub fn orif_permanent_loss_fraction(beta_ratio: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&beta_ratio),
        "0 ≤ β ≤ 1 requis (rapport des diamètres)"
    );
    1.0 - beta_ratio * beta_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn beta_ratio_basic_and_bounds() {
        // β = d/D ; à d = D on atteint la borne β = 1 ; demi-diamètre ⇒ β = 0.5.
        assert_relative_eq!(
            orif_beta_ratio(0.05_f64, 0.10_f64),
            0.5,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            orif_beta_ratio(0.10_f64, 0.10_f64),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn mass_flow_numeric_case() {
        // Cd = 0.6, β = 0.5, A₀ = 0.01 m², ρ = 1000 kg·m⁻³, ΔP = 20000 Pa.
        // β⁴ = 0.0625 ; 1 − β⁴ = 0.9375 ; √0.9375 = 0.9682458366.
        // Facteur d'approche : Cd/√(1−β⁴) = 0.6/0.9682458366 = 0.6196770.
        // √(2·ρ·ΔP) = √(2·1000·20000) = √4.0e7 = 6324.5553203.
        // ṁ = 0.6196770 · 0.01 · 6324.5553203 = 39.19181.
        // Recalcul indépendant : 0.01·6324.5553203 = 63.245553203 ;
        //   63.245553203 · 0.6196770 = 39.191815.
        assert_relative_eq!(
            orif_mass_flow(0.6_f64, 0.5_f64, 0.01_f64, 1000.0_f64, 20000.0_f64),
            39.1918,
            max_relative = 1e-3
        );
    }

    #[test]
    fn mass_flow_and_pressure_are_reciprocal() {
        // ΔP → ṁ → ΔP : orif_differential_pressure est l'inverse exact de
        // orif_mass_flow à (Cd, β, A₀, ρ) fixés.
        let (cd, beta, area, rho, dp) = (0.62_f64, 0.45_f64, 0.008_f64, 998.0_f64, 12500.0_f64);
        let mdot = orif_mass_flow(cd, beta, area, rho, dp);
        let dp_back = orif_differential_pressure(mdot, cd, beta, area, rho);
        assert_relative_eq!(dp_back, dp, max_relative = 1e-9);
    }

    #[test]
    fn mass_flow_scales_with_sqrt_of_pressure() {
        // ṁ ∝ √ΔP : quadrupler ΔP double le débit massique (autres paramètres
        // fixés), car ṁ contient √(2·ρ·ΔP).
        let base = orif_mass_flow(0.61_f64, 0.4_f64, 0.005_f64, 1000.0_f64, 5000.0_f64);
        let quad = orif_mass_flow(0.61_f64, 0.4_f64, 0.005_f64, 1000.0_f64, 20000.0_f64);
        assert_relative_eq!(quad, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn volumetric_flow_consistency() {
        // Q = ṁ/ρ : à ρ = 800 kg·m⁻³, un débit de 40 kg·s⁻¹ donne 0.05 m³·s⁻¹.
        assert_relative_eq!(
            orif_volumetric_flow(40.0_f64, 800.0_f64),
            0.05,
            max_relative = 1e-12
        );
    }

    #[test]
    fn permanent_loss_fraction_limits() {
        // ω = 1 − β² : borne β → 0 ⇒ ω = 1 (perte quasi totale) ; β = 1 ⇒ ω = 0
        // (aucune restriction) ; β = 0.5 ⇒ ω = 1 − 0.25 = 0.75.
        assert_relative_eq!(
            orif_permanent_loss_fraction(0.0_f64),
            1.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            orif_permanent_loss_fraction(1.0_f64),
            0.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            orif_permanent_loss_fraction(0.5_f64),
            0.75,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "0 ≤ β < 1 requis")]
    fn mass_flow_panics_on_beta_one() {
        // β = 1 ⇒ 1 − β⁴ = 0 ⇒ division par zéro dans le facteur d'approche.
        let _ = orif_mass_flow(0.6_f64, 1.0_f64, 0.01_f64, 1000.0_f64, 20000.0_f64);
    }
}
