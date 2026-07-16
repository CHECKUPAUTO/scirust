//! Séparation centrifuge — décantation accélérée par un champ centrifuge :
//! facteur de séparation `g` (nombre de fois la pesanteur), vitesse de migration
//! radiale d'une particule en régime de Stokes, facteur Sigma d'une centrifugeuse
//! tubulaire (indice de capacité géométrique) et diamètre de coupure à 50 %.
//!
//! ```text
//! facteur de séparation (nombre de « g »)
//!   g_c = ω²·r / g                                              [sans dimension]
//! vitesse de migration radiale (régime de Stokes)
//!   u_r = d²·Δρ·ω²·r / (18·μ)                                   [m·s⁻¹]
//! facteur Sigma d'une centrifugeuse tubulaire (forme retenue)
//!   Σ   = V·ω²·(r_o + r_i) / (2·g·|ln(r_o − r_i)|)              [m²]
//! diamètre de coupure à 50 %
//!   d_c = √( 18·μ·Q / (2·Δρ·g·Σ) )                              [m]
//! ```
//!
//! `ω` vitesse angulaire du bol [rad·s⁻¹], `r` rayon de rotation [m], `g`
//! accélération de la pesanteur (ou de référence) [m·s⁻²] ; `g_c` facteur de
//! séparation, rapport de l'accélération centrifuge à `g` [sans dimension] ; `d`
//! diamètre de la particule [m], `Δρ` différence de masse volumique entre la
//! particule et le liquide [kg·m⁻³], `μ` viscosité dynamique du liquide [Pa·s],
//! `u_r` vitesse de migration radiale [m·s⁻¹] ; `V` volume de liquide retenu dans
//! le bol [m³], `r_o` rayon externe (paroi du bol) [m], `r_i` rayon interne
//! (surface libre du liquide) [m], `Σ` facteur Sigma, aire d'un décanteur
//! gravitaire équivalent [m²] ; `Q` débit volumique d'alimentation [m³·s⁻¹], `d_c`
//! diamètre de coupure (50 % de capture) [m].
//!
//! **Limite honnête** : modèle de séparation centrifuge à l'échelle des
//! **opérations unitaires**, en **régime de Stokes** (particules fines, nombre de
//! Reynolds particulaire faible, suspensions **diluées** sans entrave mutuelle).
//! Les propriétés physiques — différence de masse volumique `Δρ` et viscosité `μ`
//! — sont **FOURNIES** par l'appelant, jamais inventées ni recalculées ici. La
//! vitesse angulaire `ω` est exprimée en **rad·s⁻¹** (convertir depuis les tr·min⁻¹
//! en amont). L'accélération `g` est **FOURNIE** (elle vaut la pesanteur terrestre
//! pour le facteur de séparation, ou une accélération de référence choisie). Le
//! **facteur Sigma** est un indice de **mise à l'échelle géométrique** permettant
//! de comparer deux machines à performance de séparation égale (`Q₁/Σ₁ = Q₂/Σ₂`) ;
//! la forme retenue ci-dessus est celle **imposée par l'appelant** et n'est pas
//! substituée à une autre corrélation de bol. Aucune propriété d'état ni
//! corrélation de transport n'est fabriquée par ce module.

/// Facteur de séparation centrifuge `g_c = ω²·r / g` (sans dimension), soit le
/// nombre de fois la pesanteur ressenti par une particule dans le champ centrifuge.
///
/// `angular_velocity` (ω) vitesse angulaire [rad·s⁻¹], `radius` (r) rayon de
/// rotation [m], `gravity` (g) accélération de référence [m·s⁻²].
///
/// Panique si `r < 0` ou si `g ≤ 0` (accélération de référence strictement
/// positive requise).
pub fn centf_g_factor(angular_velocity: f64, radius: f64, gravity: f64) -> f64 {
    assert!(radius >= 0.0, "r ≥ 0 requis (rayon de rotation)");
    assert!(gravity > 0.0, "g > 0 requis (accélération de référence)");
    angular_velocity * angular_velocity * radius / gravity
}

/// Vitesse de migration radiale d'une particule en **régime de Stokes**
/// `u_r = d²·Δρ·ω²·r / (18·μ)` (m·s⁻¹) dans un champ centrifuge.
///
/// `particle_diameter` (d) diamètre de la particule [m], `density_difference` (Δρ)
/// différence de masse volumique particule − liquide [kg·m⁻³], `angular_velocity`
/// (ω) vitesse angulaire [rad·s⁻¹], `radius` (r) rayon de rotation [m], `viscosity`
/// (μ) viscosité dynamique du liquide [Pa·s].
///
/// Panique si `d < 0`, si `r < 0`, ou si `μ ≤ 0` (viscosité strictement positive
/// requise). `Δρ` peut être négatif (particule plus légère : migration vers l'axe).
pub fn centf_terminal_velocity(
    particle_diameter: f64,
    density_difference: f64,
    angular_velocity: f64,
    radius: f64,
    viscosity: f64,
) -> f64 {
    assert!(
        particle_diameter >= 0.0,
        "d ≥ 0 requis (diamètre de la particule)"
    );
    assert!(radius >= 0.0, "r ≥ 0 requis (rayon de rotation)");
    assert!(viscosity > 0.0, "μ > 0 requis (viscosité dynamique)");
    particle_diameter
        * particle_diameter
        * density_difference
        * angular_velocity
        * angular_velocity
        * radius
        / (18.0 * viscosity)
}

/// Facteur Sigma d'une centrifugeuse tubulaire (forme retenue)
/// `Σ = V·ω²·(r_o + r_i) / (2·g·|ln(r_o − r_i)|)` (m²), aire d'un décanteur
/// gravitaire équivalent servant à la mise à l'échelle géométrique.
///
/// `volume` (V) volume de liquide retenu dans le bol [m³], `gravity` (g)
/// accélération de référence [m·s⁻²], `angular_velocity` (ω) vitesse angulaire
/// [rad·s⁻¹], `outer_radius` (r_o) rayon externe [m], `inner_radius` (r_i) rayon
/// interne [m].
///
/// Le logarithme est borné par `max(|ln(r_o − r_i)|, 1e-9)` afin d'éviter une
/// division par zéro lorsque `r_o − r_i` approche 1.
///
/// Panique si `V < 0`, si `g ≤ 0`, si `r_i ≤ 0`, ou si `r_o ≤ r_i` (le rayon
/// externe doit dépasser strictement le rayon interne, et `r_o − r_i` doit rester
/// positif pour que le logarithme soit défini).
pub fn centf_sigma_factor(
    volume: f64,
    gravity: f64,
    angular_velocity: f64,
    outer_radius: f64,
    inner_radius: f64,
) -> f64 {
    assert!(volume >= 0.0, "V ≥ 0 requis (volume de liquide)");
    assert!(gravity > 0.0, "g > 0 requis (accélération de référence)");
    assert!(inner_radius > 0.0, "r_i > 0 requis (rayon interne)");
    assert!(
        outer_radius > inner_radius,
        "r_o > r_i requis (rayon externe strictement supérieur au rayon interne)"
    );
    let log_term = (outer_radius - inner_radius).ln().abs().max(1e-9);
    volume * angular_velocity * angular_velocity * (outer_radius + inner_radius)
        / (2.0 * gravity * log_term)
}

/// Diamètre de coupure à 50 % `d_c = √( 18·μ·Q / (2·Δρ·g·Σ) )` (m) : diamètre de
/// la particule capturée à 50 % pour un débit et un facteur Sigma donnés.
///
/// `flow_rate` (Q) débit volumique d'alimentation [m³·s⁻¹], `sigma_factor` (Σ)
/// facteur Sigma de la machine [m²], `density_difference` (Δρ) différence de masse
/// volumique particule − liquide [kg·m⁻³], `viscosity` (μ) viscosité dynamique
/// [Pa·s], `gravity` (g) accélération de référence [m·s⁻²].
///
/// Panique si `Q < 0`, si `Σ ≤ 0`, si `Δρ ≤ 0`, si `μ ≤ 0`, ou si `g ≤ 0` (le
/// radicande doit être positif : particule plus dense que le liquide requise).
pub fn centf_cut_diameter(
    flow_rate: f64,
    sigma_factor: f64,
    density_difference: f64,
    viscosity: f64,
    gravity: f64,
) -> f64 {
    assert!(flow_rate >= 0.0, "Q ≥ 0 requis (débit d'alimentation)");
    assert!(sigma_factor > 0.0, "Σ > 0 requis (facteur Sigma)");
    assert!(
        density_difference > 0.0,
        "Δρ > 0 requis (particule plus dense que le liquide)"
    );
    assert!(viscosity > 0.0, "μ > 0 requis (viscosité dynamique)");
    assert!(gravity > 0.0, "g > 0 requis (accélération de référence)");
    (18.0 * viscosity * flow_rate / (2.0 * density_difference * gravity * sigma_factor)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn g_factor_scales_quadratically_with_speed() {
        // g_c ∝ ω² : doubler la vitesse angulaire quadruple le facteur de séparation.
        let base = centf_g_factor(100.0_f64, 0.1_f64, 9.81_f64);
        let doubled = centf_g_factor(200.0_f64, 0.1_f64, 9.81_f64);
        assert_relative_eq!(doubled, 4.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn g_factor_numeric_case() {
        // ω = 200 rad/s, r = 0.1 m, g = 10 m/s² ⇒
        //   g_c = 200² · 0.1 / 10 = 40000 · 0.1 / 10 = 4000 / 10 = 400.
        let g_c = centf_g_factor(200.0_f64, 0.1_f64, 10.0_f64);
        assert_relative_eq!(g_c, 400.0, max_relative = 1e-12);
    }

    #[test]
    fn terminal_velocity_numeric_case() {
        // d = 1e-5 m, Δρ = 200 kg/m³, ω = 100 rad/s, r = 0.1 m, μ = 0.001 Pa·s ⇒
        //   num = (1e-5)² · 200 · 100² · 0.1 = 1e-10 · 200 · 1e4 · 0.1 = 2e-5
        //   u_r = 2e-5 / (18 · 0.001) = 2e-5 / 0.018 ≈ 1.111111e-3 m/s.
        let u_r = centf_terminal_velocity(1e-5_f64, 200.0_f64, 100.0_f64, 0.1_f64, 0.001_f64);
        assert_relative_eq!(u_r, 1.111111e-3, max_relative = 1e-3);
    }

    #[test]
    fn terminal_velocity_via_g_factor_identity() {
        // u_r = d²·Δρ·(ω²·r)/(18μ) = d²·Δρ·(g_c·g)/(18μ) : la vitesse de migration
        // se factorise par le facteur de séparation g_c = ω²r/g.
        let (d, drho, omega, r, mu, g) = (
            2e-5_f64, 150.0_f64, 80.0_f64, 0.06_f64, 0.0012_f64, 9.81_f64,
        );
        let direct = centf_terminal_velocity(d, drho, omega, r, mu);
        let g_c = centf_g_factor(omega, r, g);
        let via = d * d * drho * (g_c * g) / (18.0 * mu);
        assert_relative_eq!(direct, via, max_relative = 1e-9);
    }

    #[test]
    fn sigma_factor_numeric_case() {
        // V = 0.002 m³, g = 9.81, ω = 50 rad/s, r_o = 0.12 m, r_i = 0.02 m ⇒
        //   r_o − r_i = 0.10, |ln(0.10)| = 2.302585
        //   num = 0.002 · 2500 · 0.14 = 5 · 0.14 = 0.7
        //   den = 2 · 9.81 · 2.302585 = 19.62 · 2.302585 ≈ 45.17672
        //   Σ = 0.7 / 45.17672 ≈ 0.0154948 m².
        let sigma = centf_sigma_factor(0.002_f64, 9.81_f64, 50.0_f64, 0.12_f64, 0.02_f64);
        assert_relative_eq!(sigma, 0.0154948, max_relative = 1e-3);
    }

    #[test]
    fn cut_diameter_reciprocity_with_sigma() {
        // Réciprocité : d_c = √(18μQ / (2Δρ g Σ)) ⇒ 2Δρ g Σ d_c² = 18μQ.
        // Cas chiffré : μ = 0.001, Q = 1e-4, Δρ = 100, g = 10, Σ = 1000 ⇒
        //   radicande = 18·0.001·1e-4 / (2·100·10·1000) = 1.8e-6 / 2e6 = 9e-13
        //   d_c = √(9e-13) = 3·3.162278e-7 ≈ 9.486833e-7 m.
        let d_c = centf_cut_diameter(1e-4_f64, 1000.0_f64, 100.0_f64, 0.001_f64, 10.0_f64);
        assert_relative_eq!(d_c, 9.486833e-7, max_relative = 1e-3);
        // Inversion : on retrouve 18μQ à partir de d_c.
        assert_relative_eq!(
            2.0 * 100.0 * 10.0 * 1000.0 * d_c * d_c,
            18.0 * 0.001 * 1e-4,
            max_relative = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "r_o > r_i requis")]
    fn sigma_factor_panics_when_outer_not_greater() {
        // r_o = r_i ⇒ ln(0) non défini ⇒ géométrie rejetée.
        let _ = centf_sigma_factor(0.002_f64, 9.81_f64, 50.0_f64, 0.05_f64, 0.05_f64);
    }
}
