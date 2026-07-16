//! **Longueur de flambement d'un poteau comprimé (Eurocode 3, EN 1993-1-1
//! §6.3.1)** : longueur de flambement `Lcr = K·L`, rayon de giration `i`,
//! élancement géométrique `λ = Lcr/i`, contrainte critique d'Euler
//! `σcr = π²·E/λ²` et élancement réduit `λ̄` servant aux courbes de flambement.
//!
//! ```text
//! longueur de flambement    Lcr = K·L
//! rayon de giration         i   = √(I / A)
//! élancement géométrique    λ   = Lcr / i
//! contrainte critique       σcr = π²·E / λ²
//! élancement réduit         λ̄   = (λ / π)·√(fy / E)
//! ```
//!
//! `L` = `length` longueur d'épure du poteau (m), `K` =
//! `effective_length_factor` facteur de longueur de flambement selon les
//! conditions d'appui (sans dimension), `Lcr` = `effective_length` longueur de
//! flambement (m), `I` = `inertia` moment quadratique de la section autour de
//! l'axe de flambement (m⁴), `A` = `cross_section_area` aire de la section (m²),
//! `i` = `radius_of_gyration` rayon de giration (m), `λ` = `slenderness`
//! élancement géométrique (sans dimension), `E` = `elastic_modulus` module
//! d'élasticité longitudinal (Pa = N/m²), `σcr` = contrainte critique d'Euler
//! (Pa), `fy` = `yield_strength` limite d'élasticité (Pa), `λ̄` = élancement
//! réduit (sans dimension).
//!
//! **Convention** : unités SI cohérentes N, m, Pa (`1 Pa = 1 N/m²`) ; `K`, `λ`
//! et `λ̄` sont sans dimension ; aucun angle n'intervient. Les fonctions sont
//! homogènes : toute paire d'unités cohérente (par ex. N, mm, MPa) donne un
//! résultat correct pourvu que `E`, `fy`, `I` et les longueurs soient exprimés
//! dans le même système.
//!
//! **Limite honnête** : modèle **élastique** de flambement d'Euler par flexion
//! d'une barre droite idéale, sans imperfection ni plastification. Le facteur de
//! longueur de flambement `K` (0,5 encastré-encastré, 0,7 encastré-articulé, 1,0
//! bi-articulé, 2,0 console…), le module d'élasticité `E`, la limite d'élasticité
//! caractéristique `fy` et les caractéristiques de section `I`, `A` sont
//! **fournis par l'appelant** d'après l'Eurocode, son Annexe Nationale et les
//! catalogues de profilés ; aucune valeur n'est inventée. L'élancement réduit
//! `λ̄` calculé ici alimente les **courbes de flambement** (facteur de réduction
//! `χ`), traitées ailleurs (voir `steel_compression`) ; le flambement par torsion
//! ou par flexion-torsion n'est **pas** couvert.

/// Longueur de flambement `Lcr = K·L` (m) d'un poteau, avec `length` = `L` la
/// longueur d'épure (m) et `effective_length_factor` = `K` le facteur de longueur
/// de flambement selon les conditions d'appui (sans dimension).
///
/// Panique si `length <= 0` ou `effective_length_factor <= 0`.
pub fn efflen_effective_length(length: f64, effective_length_factor: f64) -> f64 {
    assert!(
        length > 0.0,
        "la longueur d'épure L doit être strictement positive (m)"
    );
    assert!(
        effective_length_factor > 0.0,
        "le facteur de longueur de flambement K doit être strictement positif"
    );
    length * effective_length_factor
}

/// Rayon de giration `i = √(I / A)` (m) d'une section, avec `inertia` = `I` le
/// moment quadratique autour de l'axe considéré (m⁴) et `cross_section_area` =
/// `A` l'aire de la section (m²).
///
/// Panique si `inertia <= 0` ou `cross_section_area <= 0`.
pub fn efflen_radius_of_gyration(inertia: f64, cross_section_area: f64) -> f64 {
    assert!(
        inertia > 0.0,
        "le moment quadratique I doit être strictement positif (m⁴)"
    );
    assert!(
        cross_section_area > 0.0,
        "l'aire de la section A doit être strictement positive (m²)"
    );
    (inertia / cross_section_area).sqrt()
}

/// Élancement géométrique `λ = Lcr / i` (sans dimension), avec `effective_length`
/// = `Lcr` la longueur de flambement (m) et `radius_of_gyration` = `i` le rayon
/// de giration (m).
///
/// Panique si `effective_length <= 0` ou `radius_of_gyration <= 0`.
pub fn efflen_slenderness(effective_length: f64, radius_of_gyration: f64) -> f64 {
    assert!(
        effective_length > 0.0,
        "la longueur de flambement Lcr doit être strictement positive (m)"
    );
    assert!(
        radius_of_gyration > 0.0,
        "le rayon de giration i doit être strictement positif (m)"
    );
    effective_length / radius_of_gyration
}

/// Contrainte critique de flambement d'Euler `σcr = π²·E / λ²` (Pa), avec
/// `elastic_modulus` = `E` le module d'élasticité longitudinal (Pa) et
/// `slenderness` = `λ` l'élancement géométrique (sans dimension).
///
/// Panique si `elastic_modulus <= 0` ou `slenderness <= 0`.
pub fn efflen_euler_critical_stress(elastic_modulus: f64, slenderness: f64) -> f64 {
    assert!(
        elastic_modulus > 0.0,
        "le module d'élasticité E doit être strictement positif (Pa)"
    );
    assert!(
        slenderness > 0.0,
        "l'élancement λ doit être strictement positif"
    );
    core::f64::consts::PI * core::f64::consts::PI * elastic_modulus / (slenderness * slenderness)
}

/// Élancement réduit `λ̄ = (λ / π)·√(fy / E)` (sans dimension) de l'Eurocode 3,
/// avec `slenderness` = `λ` l'élancement géométrique (sans dimension),
/// `yield_strength` = `fy` la limite d'élasticité (Pa) et `elastic_modulus` = `E`
/// le module d'élasticité longitudinal (Pa).
///
/// Panique si `slenderness <= 0`, `yield_strength <= 0` ou `elastic_modulus <= 0`.
pub fn efflen_relative_slenderness(
    slenderness: f64,
    yield_strength: f64,
    elastic_modulus: f64,
) -> f64 {
    assert!(
        slenderness > 0.0,
        "l'élancement λ doit être strictement positif"
    );
    assert!(
        yield_strength > 0.0,
        "la limite d'élasticité fy doit être strictement positive (Pa)"
    );
    assert!(
        elastic_modulus > 0.0,
        "le module d'élasticité E doit être strictement positif (Pa)"
    );
    (slenderness / core::f64::consts::PI) * (yield_strength / elastic_modulus).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn effective_length_proportional_to_support_factor() {
        // Lcr = K·L : proportionnalité stricte en K et en L.
        let length = 4.0_f64; // m
        // Bi-articulé (K = 1) : Lcr = L.
        assert_relative_eq!(
            efflen_effective_length(length, 1.0),
            length,
            epsilon = 1e-12
        );
        // Console (K = 2) : Lcr double.
        assert_relative_eq!(
            efflen_effective_length(length, 2.0),
            2.0 * efflen_effective_length(length, 1.0),
            epsilon = 1e-12
        );
        // Encastré-encastré (K = 0,5) : Lcr moitié.
        assert_relative_eq!(efflen_effective_length(length, 0.5), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn radius_of_gyration_inverts_inertia() {
        // Réciprocité i = √(I/A)  ⇔  I = A·i², quelles que soient les entrées.
        let inertia = 8.356e-5_f64; // m⁴ (≈ HEB 200 axe fort)
        let area = 7.808e-3_f64; // m²
        let i = efflen_radius_of_gyration(inertia, area);
        assert_relative_eq!(area * i.powi(2), inertia, epsilon = 1e-12);
        // Cas chiffré : I = 16, A = 4  ⇒  i = √4 = 2 exactement.
        assert_relative_eq!(efflen_radius_of_gyration(16.0, 4.0), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn slenderness_chains_length_and_radius() {
        // λ = Lcr/i, avec Lcr = K·L : λ = K·L/i, homogène et sans dimension.
        let length = 3.0_f64; // m
        let factor = 2.0_f64; // console
        let i = 0.05_f64; // m
        let lcr = efflen_effective_length(length, factor);
        let lambda = efflen_slenderness(lcr, i);
        // K·L/i = 2·3/0,05 = 120 exactement.
        assert_relative_eq!(lambda, 120.0, epsilon = 1e-9);
        // Doubler i divise l'élancement par deux.
        assert_relative_eq!(
            efflen_slenderness(lcr, 2.0 * i),
            0.5 * lambda,
            epsilon = 1e-9
        );
    }

    #[test]
    fn euler_stress_worked_case_and_inverse_square() {
        // σcr = π²·E/λ². E = 210 GPa, λ = 100 :
        // σcr = π²·210e9/10000 = π²·2,1e7.
        // π² = 9,869604401 ; 9,869604401·2,1e7 = 2,072616924e8 Pa (≈ 207,3 MPa).
        let e = 210.0e9_f64; // Pa
        let sigma = efflen_euler_critical_stress(e, 100.0);
        assert_relative_eq!(sigma, 2.072_616_924e8, epsilon = 1.0e3);
        // Loi en 1/λ² : doubler λ divise σcr par quatre.
        assert_relative_eq!(
            efflen_euler_critical_stress(e, 200.0),
            sigma / 4.0,
            epsilon = 1.0
        );
    }

    #[test]
    fn relative_slenderness_reference_value_and_scaling() {
        // λ̄ = (λ/π)·√(fy/E). Pour l'acier S235 (fy = 235 MPa, E = 210 GPa),
        // l'élancement de référence est λ1 = π·√(E/fy) = π·√(893,617) ≈ 93,913,
        // donc λ = 93,913 donne λ̄ = 1,0 par construction.
        let fy = 235.0e6_f64; // Pa
        let e = 210.0e9_f64; // Pa
        let lambda_ref = 93.913_202_f64;
        assert_relative_eq!(
            efflen_relative_slenderness(lambda_ref, fy, e),
            1.0,
            epsilon = 1e-3
        );
        // λ̄ est proportionnel à λ : doubler λ double λ̄.
        let base = efflen_relative_slenderness(50.0, fy, e);
        assert_relative_eq!(
            efflen_relative_slenderness(100.0, fy, e),
            2.0 * base,
            epsilon = 1e-9
        );
    }

    #[test]
    fn relative_slenderness_matches_euler_definition() {
        // Identité EC3 : λ̄ = √(fy/σcr). On le vérifie via σcr = π²E/λ².
        let fy = 355.0e6_f64; // Pa
        let e = 210.0e9_f64; // Pa
        let lambda = 80.0_f64;
        let sigma_cr = efflen_euler_critical_stress(e, lambda);
        let lambda_bar = efflen_relative_slenderness(lambda, fy, e);
        assert_relative_eq!(lambda_bar, (fy / sigma_cr).sqrt(), epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "l'élancement λ doit être strictement positif")]
    fn euler_stress_rejects_non_positive_slenderness() {
        // Élancement nul : division par zéro, entrée refusée.
        efflen_euler_critical_stress(210.0e9, 0.0);
    }
}
