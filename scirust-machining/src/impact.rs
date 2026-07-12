//! Chocs et charges dynamiques — coefficient de restitution, choc de deux masses,
//! et facteur d'amplification dynamique d'une charge appliquée brusquement ou
//! tombant d'une hauteur sur une structure élastique.
//!
//! ```text
//! restitution           e = (v2' − v1')/(v1 − v2)      0 ≤ e ≤ 1
//! choc direct (qté mvt) m1·v1 + m2·v2 = m1·v1' + m2·v2'
//! charge subitement app. facteur = 2   (δ_dyn = 2·δ_st)
//! charge tombant de h   n = 1 + √(1 + 2h/δ_st)
//! énergie perdue (e<1)  ΔE = ½·(m1·m2/(m1+m2))·(1−e²)·(v1−v2)²
//! ```
//!
//! `e` coefficient de restitution (`e=1` élastique, `e=0` parfaitement plastique),
//! `δ_st` flèche statique sous la charge appliquée lentement, `h` hauteur de
//! chute, `n` facteur d'amplification dynamique (multiplie contrainte et flèche
//! statiques).
//!
//! **Convention** : SI cohérent, chocs colinéaires (1D). **Limite honnête** :
//! choc direct central le long d'un axe ; le facteur de chute suppose une
//! structure **élastique linéaire** sans amortissement et toute l'énergie
//! cinétique convertie en énergie de déformation (majorant).

/// Vitesses après un **choc direct central** de deux masses, coefficient de
/// restitution `e`. Renvoie `(v1', v2')`.
///
/// Panique si `e` sort de `[0, 1]` ou si `m1 + m2 <= 0`.
pub fn direct_impact_velocities(m1: f64, v1: f64, m2: f64, v2: f64, e: f64) -> (f64, f64) {
    assert!(
        (0.0..=1.0).contains(&e),
        "le coefficient de restitution doit être dans [0, 1]"
    );
    let total = m1 + m2;
    assert!(
        total > 0.0,
        "la masse totale doit être strictement positive"
    );
    // Conservation de la quantité de mouvement + loi de restitution.
    let v1p = (m1 * v1 + m2 * v2 - m2 * e * (v1 - v2)) / total;
    let v2p = (m1 * v1 + m2 * v2 + m1 * e * (v1 - v2)) / total;
    (v1p, v2p)
}

/// Énergie cinétique dissipée dans un choc `ΔE = ½·μ·(1−e²)·(v1−v2)²` (J),
/// `μ = m1·m2/(m1+m2)` masse réduite.
///
/// Panique si `e` sort de `[0, 1]` ou `m1 + m2 <= 0`.
pub fn energy_lost(m1: f64, v1: f64, m2: f64, v2: f64, e: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&e),
        "le coefficient de restitution doit être dans [0, 1]"
    );
    let total = m1 + m2;
    assert!(
        total > 0.0,
        "la masse totale doit être strictement positive"
    );
    let mu = m1 * m2 / total;
    0.5 * mu * (1.0 - e * e) * (v1 - v2) * (v1 - v2)
}

/// Facteur d'amplification dynamique d'une charge **appliquée subitement**
/// (sans vitesse initiale) : `n = 2` (`δ_dyn = 2·δ_st`).
pub fn suddenly_applied_factor() -> f64 {
    2.0
}

/// Facteur d'amplification dynamique d'une charge **tombant d'une hauteur** `h`
/// sur une structure de flèche statique `δ_st` : `n = 1 + √(1 + 2h/δ_st)`.
///
/// Multiplie la contrainte et la flèche statiques. Panique si `δ_st <= 0` ou
/// `h < 0`.
pub fn falling_load_factor(drop_height: f64, static_deflection: f64) -> f64 {
    assert!(
        static_deflection > 0.0 && drop_height >= 0.0,
        "δ_st > 0 et h ≥ 0 requis"
    );
    1.0 + (1.0 + 2.0 * drop_height / static_deflection).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn perfectly_plastic_impact_gives_common_velocity() {
        // e=0 : les deux masses repartent ensemble. m1=2,v1=3 ; m2=1,v2=0.
        // v' = (2·3)/(3) = 2 m/s pour les deux.
        let (v1p, v2p) = direct_impact_velocities(2.0, 3.0, 1.0, 0.0, 0.0);
        assert_relative_eq!(v1p, 2.0, epsilon = 1e-12);
        assert_relative_eq!(v2p, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn elastic_equal_masses_exchange_velocities() {
        // e=1, masses égales : échange des vitesses (v1=4, v2=0 → 0 et 4).
        let (v1p, v2p) = direct_impact_velocities(1.0, 4.0, 1.0, 0.0, 1.0);
        assert_relative_eq!(v1p, 0.0, epsilon = 1e-12);
        assert_relative_eq!(v2p, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn momentum_is_conserved() {
        // La quantité de mouvement se conserve quel que soit e.
        let (m1, v1, m2, v2) = (3.0, 5.0, 2.0, -1.0);
        let (v1p, v2p) = direct_impact_velocities(m1, v1, m2, v2, 0.6);
        assert_relative_eq!(m1 * v1 + m2 * v2, m1 * v1p + m2 * v2p, epsilon = 1e-9);
    }

    #[test]
    fn elastic_impact_loses_no_energy() {
        assert_relative_eq!(energy_lost(2.0, 3.0, 1.0, 0.0, 1.0), 0.0, epsilon = 1e-12);
        // Plastique : perte maximale ½·μ·(v1−v2)² ; μ=2/3, Δv=3 → ½·(2/3)·9 = 3 J.
        assert_relative_eq!(energy_lost(2.0, 3.0, 1.0, 0.0, 0.0), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn dynamic_load_factors() {
        // Charge subite : facteur 2. Chute nulle → mêmes 2 (n = 1+√1 = 2).
        assert_relative_eq!(suddenly_applied_factor(), 2.0, epsilon = 1e-12);
        assert_relative_eq!(falling_load_factor(0.0, 1e-3), 2.0, epsilon = 1e-12);
        // Chute h=50 mm sur δ_st=1 mm → n = 1+√(1+100) ≈ 11,05.
        let n = falling_load_factor(0.05, 1e-3);
        assert_relative_eq!(n, 1.0 + (1.0f64 + 100.0).sqrt(), epsilon = 1e-9);
        assert!(n > 11.0 && n < 11.1);
    }

    #[test]
    #[should_panic(expected = "restitution")]
    fn restitution_above_one_panics() {
        direct_impact_velocities(1.0, 1.0, 1.0, 0.0, 1.5);
    }
}
