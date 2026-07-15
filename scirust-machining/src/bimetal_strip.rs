//! Bilame thermique (deux métaux de coefficients de dilatation différents) —
//! courbure, rayon et flèche en bout selon la formule simplifiée de Timoshenko.
//!
//! ```text
//! courbure       k = 6·(α₂ − α₁)·ΔT / t        (lames d'épaisseurs égales)
//! rayon          R = 1 / k
//! flèche en bout δ = k·L² / 2                   (poutre en arc de cercle)
//! ```
//!
//! `α₁`, `α₂` coefficients de dilatation linéaire des deux lames (1/K, la lame 2
//! étant la plus dilatante pour une courbure vers la lame 1), `ΔT` variation de
//! température (K), `t` épaisseur **totale** du bilame (m), `k` courbure (1/m),
//! `R` rayon de courbure (m), `L` longueur libre de la lame (m), `δ` flèche
//! transversale en bout (m).
//!
//! **Convention** : SI cohérent. **Limite honnête** : formule de Timoshenko
//! **simplifiée** valable pour deux lames d'**épaisseurs égales** et de modules
//! d'Young **voisins**, en **petites courbures** (arc de cercle) ; les
//! coefficients de dilatation, épaisseurs et longueurs sont **fournis par
//! l'appelant** — aucune valeur matériau n'est supposée par défaut.

/// Courbure `k = 6·(α₂ − α₁)·ΔT / t` d'un bilame thermique (1/m).
///
/// Le signe suit celui de `(α₂ − α₁)·ΔT` : positif quand la lame 2 se dilate
/// plus que la lame 1 (courbure vers la lame 1).
///
/// Panique si `total_thickness <= 0`.
pub fn bimetal_curvature(
    alpha1: f64,
    alpha2: f64,
    delta_temperature: f64,
    total_thickness: f64,
) -> f64 {
    assert!(
        total_thickness > 0.0,
        "l'épaisseur totale du bilame doit être strictement positive"
    );
    6.0 * (alpha2 - alpha1) * delta_temperature / total_thickness
}

/// Flèche transversale en bout `δ = k·L² / 2` d'une lame en arc de cercle (m).
///
/// Panique si `length < 0`.
pub fn bimetal_tip_deflection(curvature: f64, length: f64) -> f64 {
    assert!(
        length >= 0.0,
        "la longueur libre de la lame doit être positive ou nulle"
    );
    curvature * length * length / 2.0
}

/// Rayon de courbure `R = 1 / k` d'un bilame thermique (m).
///
/// Panique si `curvature == 0` (rayon infini, lame rectiligne).
pub fn bimetal_radius(curvature: f64) -> f64 {
    assert!(
        curvature != 0.0,
        "la courbure doit être non nulle pour définir un rayon fini"
    );
    1.0 / curvature
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn curvature_realistic_case() {
        // α₁=12 µm/m/K (acier), α₂=20 µm/m/K (laiton), ΔT=100 K, t=0,5 mm.
        // Δα = 8e-6 ; k = 6·8e-6·100 / 5e-4 = 6·1,6 = 9,6 1/m.
        let k = bimetal_curvature(12e-6, 20e-6, 100.0, 0.5e-3);
        assert_relative_eq!(k, 9.6, max_relative = 1e-12);
    }

    #[test]
    fn radius_is_reciprocal_of_curvature() {
        // Réciprocité : R = 1/k et k = 1/R.
        let k = bimetal_curvature(12e-6, 20e-6, 100.0, 0.5e-3);
        let r = bimetal_radius(k);
        assert_relative_eq!(r, 1.0 / 9.6, max_relative = 1e-12);
        assert_relative_eq!(1.0 / r, k, max_relative = 1e-12);
    }

    #[test]
    fn tip_deflection_realistic_case() {
        // k=9,6 1/m, L=50 mm : δ = 9,6·0,05²/2 = 9,6·0,00125 = 0,012 m.
        let k = bimetal_curvature(12e-6, 20e-6, 100.0, 0.5e-3);
        let d = bimetal_tip_deflection(k, 0.05);
        assert_relative_eq!(d, 0.012, max_relative = 1e-12);
    }

    #[test]
    fn curvature_proportional_to_delta_temperature() {
        // k ∝ ΔT : doubler ΔT double la courbure.
        let k1 = bimetal_curvature(12e-6, 20e-6, 50.0, 0.5e-3);
        let k2 = bimetal_curvature(12e-6, 20e-6, 100.0, 0.5e-3);
        assert_relative_eq!(k2, 2.0 * k1, max_relative = 1e-12);
    }

    #[test]
    fn equal_alphas_give_no_curvature() {
        // Deux lames identiques ne se courbent pas : k = 0.
        let k = bimetal_curvature(17e-6, 17e-6, 200.0, 1.0e-3);
        assert_relative_eq!(k, 0.0, epsilon = 1e-18);
    }

    #[test]
    fn deflection_quadratic_in_length() {
        // δ ∝ L² : doubler la longueur quadruple la flèche.
        let k = 9.6;
        let d1 = bimetal_tip_deflection(k, 0.05);
        let d2 = bimetal_tip_deflection(k, 0.10);
        assert_relative_eq!(d2, 4.0 * d1, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "l'épaisseur totale du bilame doit être strictement positive")]
    fn zero_thickness_panics() {
        bimetal_curvature(12e-6, 20e-6, 100.0, 0.0);
    }
}
