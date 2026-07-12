//! Treillis (systèmes réticulés) — barres à deux forces : contrainte axiale,
//! allongement élastique, et résolution d'un **nœud** par la méthode des nœuds.
//!
//! ```text
//! contrainte axiale   σ = N/A          (N > 0 traction, N < 0 compression)
//! allongement         ΔL = N·L/(E·A)
//! équilibre d'un nœud (2 barres, effort extérieur Fx,Fy) :
//!   N1·cosθ1 + N2·cosθ2 + Fx = 0
//!   N1·sinθ1 + N2·sinθ2 + Fy = 0
//! ```
//!
//! `N` effort normal dans la barre (N), `A` aire de section (m²), `L` longueur,
//! `E` module de Young, `θ` angle de la barre orientée du nœud vers l'autre
//! extrémité, mesuré depuis l'axe `x` (rad). Convention : effort de barre positif
//! en **traction** (la barre tire sur le nœud le long de sa direction sortante).
//!
//! **Convention** : SI cohérent, traction positive. **Limite honnête** : barres
//! droites articulées (efforts axiaux purs, pas de flexion), petites
//! déformations, matériau élastique linéaire ; la résolution fournie traite un
//! nœud isolé à deux inconnues — un treillis complet s'assemble nœud par nœud.

/// Contrainte axiale dans une barre `σ = N/A`.
///
/// Panique si `area <= 0`.
pub fn axial_stress(normal_force: f64, area: f64) -> f64 {
    assert!(
        area > 0.0,
        "l'aire de section doit être strictement positive"
    );
    normal_force / area
}

/// Allongement élastique d'une barre `ΔL = N·L/(E·A)`.
///
/// Panique si `e*area <= 0`.
pub fn member_elongation(normal_force: f64, length: f64, e: f64, area: f64) -> f64 {
    assert!(
        e * area > 0.0,
        "la rigidité E·A doit être strictement positive"
    );
    normal_force * length / (e * area)
}

/// Résout l'équilibre d'un nœud chargé `(Fx, Fy)` relié à **deux barres** de
/// directions sortantes `θ1` et `θ2` (rad). Renvoie `(N1, N2)`, efforts normaux
/// positifs en traction.
///
/// Panique si les deux barres sont colinéaires (`sin(θ2 − θ1) ≈ 0`), cas
/// singulier sans solution unique.
pub fn two_member_joint(fx: f64, fy: f64, theta1_rad: f64, theta2_rad: f64) -> (f64, f64) {
    let (c1, s1) = (theta1_rad.cos(), theta1_rad.sin());
    let (c2, s2) = (theta2_rad.cos(), theta2_rad.sin());
    // Système : [c1 c2; s1 s2]·[N1; N2] = [−Fx; −Fy].
    let det = c1 * s2 - c2 * s1; // = sin(θ2 − θ1)
    assert!(
        det.abs() > 1e-12,
        "barres colinéaires : équilibre du nœud indéterminé"
    );
    let n1 = (-fx * s2 - (-fy) * c2) / det;
    let n2 = (c1 * (-fy) - s1 * (-fx)) / det;
    (n1, n2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    #[test]
    fn axial_stress_and_elongation() {
        // N=10 kN, A=100 mm²=1e-4 m² → σ=100 MPa. L=2 m, E=210 GPa → ΔL.
        assert_relative_eq!(axial_stress(10_000.0, 1e-4), 100e6, epsilon = 1.0);
        let dl = member_elongation(10_000.0, 2.0, 210e9, 1e-4);
        assert_relative_eq!(dl, 10_000.0 * 2.0 / (210e9 * 1e-4), epsilon = 1e-15);
    }

    #[test]
    fn compression_gives_negative_stress() {
        assert!(axial_stress(-5_000.0, 1e-4) < 0.0);
    }

    #[test]
    fn joint_with_horizontal_and_vertical_members() {
        // Nœud tiré vers le bas par Fy=−1000 N. Barre 1 horizontale (θ=0),
        // barre 2 verticale vers le haut (θ=π/2). Seule la verticale reprend :
        // N2·sin(π/2) − 1000 = 0 → N2 = 1000 (traction), N1 = 0.
        let (n1, n2) = two_member_joint(0.0, -1000.0, 0.0, FRAC_PI_2);
        assert_relative_eq!(n1, 0.0, epsilon = 1e-9);
        assert_relative_eq!(n2, 1000.0, epsilon = 1e-9);
    }

    #[test]
    fn symmetric_two_bar_joint() {
        // Charge verticale Fy=−2000 N, deux barres à ±45° au-dessus du nœud
        // (θ1=45°, θ2=135°). Par symétrie N1=N2, et leurs composantes verticales
        // équilibrent la charge : 2·N·sin45° = 2000 → N = 1000/sin45° ≈ 1414 N.
        let (n1, n2) = two_member_joint(0.0, -2000.0, FRAC_PI_4, PI - FRAC_PI_4);
        assert_relative_eq!(n1, n2, epsilon = 1e-9);
        assert_relative_eq!(n1, 1000.0 / FRAC_PI_4.sin(), epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "colinéaires")]
    fn collinear_members_panic() {
        two_member_joint(100.0, 0.0, 0.0, PI);
    }
}
