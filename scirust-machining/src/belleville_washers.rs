//! Rondelles **Belleville** (ressorts coniques) — loi effort-flèche **non
//! linéaire** d'Almen-László (DIN 2092) et charge d'aplatissement.
//!
//! ```text
//! facteur K1 = (1/π)·[(C−1)/C]² / [ (C+1)/(C−1) − 2/ln C ]   (C = De/Di)
//! effort     P(δ) = 4E·δ / [(1−ν²)·K1·De²] · [ (h₀−δ)(h₀−δ/2)·t + t³ ]
//! aplatissement (δ = h₀)   P = 4E·h₀·t³ / [(1−ν²)·K1·De²]
//! ```
//!
//! `E` module de Young (Pa), `ν` coefficient de Poisson, `De`/`Di` diamètres
//! extérieur/intérieur (m), `h₀` hauteur du cône libre (m, = flèche à plat), `t`
//! épaisseur (m), `δ` flèche (m), `K1` facteur de forme géométrique. La courbe
//! est **non linéaire** : pour `h₀/t` élevé elle peut présenter un palier, d'où
//! l'usage de rondelles Belleville pour des efforts quasi constants.
//!
//! **Convention** : SI cohérent. **Limite honnête** : modèle **élastique**
//! d'Almen-László (une rondelle, sans appui de bord ni frottement) ; les
//! empilages en série/parallèle et la fatigue relèvent d'un calcul distinct que
//! l'appelant compose à partir de ces primitives.

/// Facteur géométrique `K1` d'une rondelle Belleville, `C = De/Di`.
///
/// Panique si `outer_diameter <= inner_diameter` ou `inner_diameter <= 0`.
pub fn k1_factor(outer_diameter: f64, inner_diameter: f64) -> f64 {
    assert!(
        inner_diameter > 0.0 && outer_diameter > inner_diameter,
        "0 < Di < De requis"
    );
    let c = outer_diameter / inner_diameter;
    let num = ((c - 1.0) / c).powi(2);
    let den = (c + 1.0) / (c - 1.0) - 2.0 / c.ln();
    num / (core::f64::consts::PI * den)
}

/// Effort d'une rondelle Belleville à la flèche `δ` (loi d'Almen-László, N).
///
/// Panique si `(1−ν²)·K1·De² <= 0`.
pub fn load(
    youngs_modulus: f64,
    poisson: f64,
    outer_diameter: f64,
    k1: f64,
    cone_height: f64,
    thickness: f64,
    deflection: f64,
) -> f64 {
    let coeff_den = (1.0 - poisson * poisson) * k1 * outer_diameter * outer_diameter;
    assert!(
        coeff_den > 0.0,
        "(1−ν²)·K1·De² doit être strictement positif"
    );
    let coeff = 4.0 * youngs_modulus / coeff_den;
    let bracket = (cone_height - deflection) * (cone_height - deflection / 2.0) * thickness
        + thickness.powi(3);
    coeff * deflection * bracket
}

/// Charge d'**aplatissement** (`δ = h₀`) `P = 4E·h₀·t³/[(1−ν²)·K1·De²]` (N).
///
/// Panique si `(1−ν²)·K1·De² <= 0`.
pub fn flatten_load(
    youngs_modulus: f64,
    poisson: f64,
    outer_diameter: f64,
    k1: f64,
    cone_height: f64,
    thickness: f64,
) -> f64 {
    let coeff_den = (1.0 - poisson * poisson) * k1 * outer_diameter * outer_diameter;
    assert!(
        coeff_den > 0.0,
        "(1−ν²)·K1·De² doit être strictement positif"
    );
    4.0 * youngs_modulus * cone_height * thickness.powi(3) / coeff_den
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn k1_for_diameter_ratio_two() {
        // C=2 → K1 ≈ 0,69 (valeur tabulée usuelle).
        let k1 = k1_factor(0.040, 0.020);
        assert!(k1 > 0.65 && k1 < 0.72);
    }

    #[test]
    fn zero_deflection_gives_zero_load() {
        let k1 = k1_factor(0.040, 0.020);
        assert_relative_eq!(
            load(210e9, 0.3, 0.040, k1, 0.001, 0.002, 0.0),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn load_at_full_deflection_equals_flatten_load() {
        // À δ = h₀, la loi générale doit redonner la charge d'aplatissement.
        let k1 = k1_factor(0.040, 0.020);
        let (e, nu, de, h0, t) = (210e9, 0.3, 0.040, 0.001, 0.002);
        let at_flat = load(e, nu, de, k1, h0, t, h0);
        assert_relative_eq!(
            at_flat,
            flatten_load(e, nu, de, k1, h0, t),
            max_relative = 1e-9
        );
        assert!(at_flat > 0.0);
    }

    #[test]
    fn load_rises_from_zero() {
        // Une petite flèche produit un effort positif croissant.
        let k1 = k1_factor(0.040, 0.020);
        let p_small = load(210e9, 0.3, 0.040, k1, 0.001, 0.002, 0.0002);
        let p_more = load(210e9, 0.3, 0.040, k1, 0.001, 0.002, 0.0004);
        assert!(p_small > 0.0 && p_more > p_small);
    }

    #[test]
    #[should_panic(expected = "0 < Di < De")]
    fn inverted_diameters_panic() {
        k1_factor(0.020, 0.040);
    }
}
