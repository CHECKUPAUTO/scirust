//! **Contrôle non destructif par radiographie** (CND / RT) — grandeurs de base
//! pour l'inspection d'une pièce traversée par un faisceau de rayonnement
//! ionisant (rayons X ou gamma) impressionnant un film ou un détecteur.
//!
//! ```text
//! intensité transmise    I  = I0 · exp(-mu · x)      (loi de Beer-Lambert)
//! couche de demi-atténuation  HVL = ln(2) / mu
//! flou géométrique       Ug = f · OFD / SOD
//! ```
//!
//! `I` intensité (ou flux de photons) transmise après la traversée (W/m² ou
//! coups/s), `I0` intensité incidente à l'entrée de la pièce (mêmes unités que
//! `I`), `mu` coefficient d'atténuation linéaire du matériau (1/m), `x`
//! épaisseur traversée (m), `HVL` couche de demi-atténuation, épaisseur qui
//! divise l'intensité par deux (m), `Ug` flou géométrique (pénombre) de l'image
//! (m), `f` dimension de la source (foyer) du tube ou de la source gamma (m),
//! `OFD` distance objet-film (m), `SOD` distance source-objet (m).
//!
//! **Convention** : SI (intensités homogènes entre elles, distances en m,
//! coefficient `mu` en 1/m). **Limite honnête** : faisceau supposé
//! **monochromatique** (énergie unique) et **collimaté** ; l'atténuation est
//! purement **exponentielle et linéaire** (le rayonnement **diffusé** et le
//! durcissement de faisceau sont **négligés**). Le coefficient d'atténuation
//! `mu` (qui dépend du matériau et de l'énergie du rayonnement), la dimension de
//! source `f` et la géométrie de tir (`SOD`, `OFD`) sont des **données de
//! l'appelant** et ne sont jamais supposés par défaut.

/// Intensité transmise `I = I0 · exp(-mu · x)` (loi de Beer-Lambert).
///
/// Décroissance exponentielle de l'intensité incidente `I0` avec l'épaisseur
/// traversée `x` ; le résultat est exprimé dans la même unité que `I0`.
///
/// Panique si `incident_intensity < 0`, `attenuation_coefficient < 0` ou
/// `thickness < 0`.
pub fn rt_transmitted_intensity(
    incident_intensity: f64,
    attenuation_coefficient: f64,
    thickness: f64,
) -> f64 {
    assert!(
        incident_intensity >= 0.0,
        "l'intensité incidente I0 doit être ≥ 0"
    );
    assert!(
        attenuation_coefficient >= 0.0,
        "le coefficient d'atténuation mu doit être ≥ 0"
    );
    assert!(thickness >= 0.0, "l'épaisseur x doit être ≥ 0");
    incident_intensity * (-attenuation_coefficient * thickness).exp()
}

/// Couche de demi-atténuation `HVL = ln(2) / mu` (m).
///
/// Épaisseur de matériau qui réduit de moitié l'intensité incidente ; c'est
/// l'épaisseur `x` telle que `exp(-mu · x) = 1/2`.
///
/// Panique si `attenuation_coefficient <= 0`.
pub fn rt_half_value_layer(attenuation_coefficient: f64) -> f64 {
    assert!(
        attenuation_coefficient > 0.0,
        "le coefficient d'atténuation mu doit être > 0"
    );
    core::f64::consts::LN_2 / attenuation_coefficient
}

/// Flou géométrique `Ug = f · OFD / SOD` (m).
///
/// Pénombre de l'image due à la taille finie de la source : l'ombre d'un bord
/// est étalée d'autant plus que la source est grande, l'objet éloigné du film
/// (`OFD` grand) ou proche de la source (`SOD` petit).
///
/// Panique si `source_size < 0`, `object_to_film < 0` ou `source_to_object <= 0`.
pub fn rt_geometric_unsharpness(
    source_size: f64,
    object_to_film: f64,
    source_to_object: f64,
) -> f64 {
    assert!(source_size >= 0.0, "la dimension de source f doit être ≥ 0");
    assert!(
        object_to_film >= 0.0,
        "la distance objet-film OFD doit être ≥ 0"
    );
    assert!(
        source_to_object > 0.0,
        "la distance source-objet SOD doit être > 0"
    );
    source_size * object_to_film / source_to_object
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn transmitted_intensity_at_zero_thickness_equals_incident() {
        // Cas limite x = 0 : exp(0) = 1, aucune atténuation.
        let i0 = 1000.0_f64;
        let i = rt_transmitted_intensity(i0, 100.0, 0.0);
        assert_relative_eq!(i, i0, epsilon = 1e-9);
    }

    #[test]
    fn half_value_layer_halves_the_intensity() {
        // Réciprocité entre les deux fonctions : à x = HVL, l'intensité vaut I0/2.
        let mu = 100.0_f64;
        let i0 = 1000.0_f64;
        let hvl = rt_half_value_layer(mu);
        let i = rt_transmitted_intensity(i0, mu, hvl);
        assert_relative_eq!(i, i0 / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn transmitted_intensity_proportional_to_incident() {
        // I ∝ I0 à mu et x fixés : doubler I0 double I.
        let mu = 80.0_f64;
        let x = 0.015_f64;
        let i1 = rt_transmitted_intensity(500.0, mu, x);
        let i2 = rt_transmitted_intensity(1000.0, mu, x);
        assert_relative_eq!(i2, 2.0 * i1, epsilon = 1e-9);
    }

    #[test]
    fn unsharpness_linear_in_source_size() {
        // Ug ∝ f à géométrie fixée : tripler la source triple le flou.
        let ofd = 0.025_f64;
        let sod = 0.5_f64;
        let u1 = rt_geometric_unsharpness(0.001, ofd, sod);
        let u3 = rt_geometric_unsharpness(0.003, ofd, sod);
        assert_relative_eq!(u3, 3.0 * u1, epsilon = 1e-12);
    }

    #[test]
    fn steel_radiograph_realistic_case() {
        // Acier, source gamma : mu = 50 /m, foyer f = 2 mm.
        // HVL = ln(2)/50 = 0,693147.../50 = 0,013862943 m (≈ 13,86 mm).
        let mu = 50.0_f64;
        let hvl = rt_half_value_layer(mu);
        assert_relative_eq!(hvl, 0.013_862_943_611, epsilon = 1e-9);
        // Épaisseur x = 0,02 m → mu·x = 1 → I/I0 = exp(-1) = 0,367879441.
        let i0 = 1.0e5_f64;
        let i = rt_transmitted_intensity(i0, mu, 0.02);
        assert_relative_eq!(i, i0 * core::f64::consts::E.recip(), epsilon = 1e-3);
        // Géométrie SOD = 0,5 m, OFD = 0,025 m, f = 0,002 m :
        // Ug = 0,002·0,025/0,5 = 5e-5/0,5 = 1e-4 m (0,1 mm).
        let ug = rt_geometric_unsharpness(0.002, 0.025, 0.5);
        assert_relative_eq!(ug, 1.0e-4, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le coefficient d'atténuation mu doit être > 0")]
    fn zero_attenuation_coefficient_panics_in_hvl() {
        rt_half_value_layer(0.0);
    }
}
