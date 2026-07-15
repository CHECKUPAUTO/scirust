//! Robotique — **jacobien** d'un bras planaire **2R** (deux liaisons rotoïdes) :
//! matrice reliant les vitesses articulaires à la vitesse de l'outil, son
//! déterminant et la détection des configurations **singulières**.
//!
//! ```text
//! jacobien (2×2)   J = [ −l1·sinθ1 − l2·sin(θ1+θ2)   −l2·sin(θ1+θ2) ]
//!                      [  l1·cosθ1 + l2·cos(θ1+θ2)    l2·cos(θ1+θ2) ]
//! vitesse outil    (vx, vy)ᵀ = J · (ω1, ω2)ᵀ
//! déterminant      det J = l1·l2·sin θ2
//! singularité      sin θ2 ≈ 0  ⇔  bras tendu (θ2 = 0) ou replié (θ2 = ±π)
//! ```
//!
//! `l1`, `l2` longueurs des deux segments (m, ≥ 0), `θ1` angle de la première
//! liaison depuis l'axe `+x` (rad), `θ2` angle de la seconde liaison relatif au
//! segment 1 (rad), `ω1`, `ω2` vitesses articulaires (rad/s), `(vx, vy)` vitesse
//! du *tool center point* (m/s), `J` matrice jacobienne 2×2 aplatie en ligne
//! `[J11, J12, J21, J22]` (unité m/rad), `det J` déterminant (m²/rad).
//!
//! **Convention** : angles en rad (sens trigonométrique), longueurs en m,
//! vitesses articulaires en rad/s, vitesse cartésienne en m/s (SI cohérent). La
//! matrice est stockée **en ligne** : `[J11, J12, J21, J22]`.
//!
//! **Limite honnête** : cinématique planaire idéale — segments rigides, liaisons
//! sans jeu ni flexion. Aux configurations alignées (`sin θ2 = 0`) le jacobien est
//! singulier : sa matrice inverse n'existe pas et le bras perd un degré de liberté
//! cartésien. Les longueurs `l1`, `l2`, les angles et la tolérance de singularité
//! sont FOURNIS par l'appelant ; ce module n'invente aucune géométrie de bras ni
//! seuil « par défaut ».

/// Jacobien 2×2 du bras 2R, aplati en ligne `[J11, J12, J21, J22]` (m/rad).
///
/// `J11 = −l1·sinθ1 − l2·sin(θ1+θ2)`, `J12 = −l2·sin(θ1+θ2)`,
/// `J21 =  l1·cosθ1 + l2·cos(θ1+θ2)`, `J22 =  l2·cos(θ1+θ2)`.
///
/// Panique si `l1 < 0` ou si `l2 < 0`.
pub fn jac2r_jacobian(l1: f64, l2: f64, theta1: f64, theta2: f64) -> [f64; 4] {
    assert!(
        l1 >= 0.0,
        "la longueur du premier segment doit être positive ou nulle"
    );
    assert!(
        l2 >= 0.0,
        "la longueur du second segment doit être positive ou nulle"
    );
    let phi = theta1 + theta2;
    let (s1, c1) = (theta1.sin(), theta1.cos());
    let (sp, cp) = (phi.sin(), phi.cos());
    let j11 = -l1 * s1 - l2 * sp;
    let j12 = -l2 * sp;
    let j21 = l1 * c1 + l2 * cp;
    let j22 = l2 * cp;
    [j11, j12, j21, j22]
}

/// Vitesse `(vx, vy)` de l'outil : `(vx, vy)ᵀ = J·(ω1, ω2)ᵀ` (m/s).
///
/// `vx = J11·ω1 + J12·ω2`, `vy = J21·ω1 + J22·ω2`, avec `jacobian` aplati en
/// ligne `[J11, J12, J21, J22]`.
///
/// Ne panique jamais (défini pour toute matrice et toute vitesse articulaire).
pub fn jac2r_tip_velocity(jacobian: [f64; 4], omega1: f64, omega2: f64) -> (f64, f64) {
    let [j11, j12, j21, j22] = jacobian;
    let vx = j11 * omega1 + j12 * omega2;
    let vy = j21 * omega1 + j22 * omega2;
    (vx, vy)
}

/// Déterminant du jacobien 2R : `det J = l1·l2·sin θ2` (m²/rad).
///
/// Indépendant de `θ1` (invariance par rotation globale du bras). S'annule aux
/// configurations alignées (`sin θ2 = 0`).
///
/// Panique si `l1 < 0` ou si `l2 < 0`.
pub fn jac2r_determinant(l1: f64, l2: f64, theta2: f64) -> f64 {
    assert!(
        l1 >= 0.0,
        "la longueur du premier segment doit être positive ou nulle"
    );
    assert!(
        l2 >= 0.0,
        "la longueur du second segment doit être positive ou nulle"
    );
    l1 * l2 * theta2.sin()
}

/// Indique si la configuration est **singulière** : `|sin θ2| ≤ tol`.
///
/// Vrai lorsque le bras est tendu (`θ2 = 0`) ou replié (`θ2 = ±π`) : le jacobien
/// n'est plus inversible et un degré de liberté cartésien est perdu.
///
/// Panique si `tol < 0`.
pub fn jac2r_is_singular(theta2: f64, tol: f64) -> bool {
    assert!(tol >= 0.0, "la tolérance doit être positive ou nulle");
    theta2.sin().abs() <= tol
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn determinant_matches_flattened_jacobian_cross_product() {
        // Identité : det J = J11·J22 − J12·J21 doit valoir l1·l2·sin θ2.
        let (l1, l2, t1, t2) = (0.42_f64, 0.27_f64, 0.7_f64, 0.9_f64);
        let [j11, j12, j21, j22] = jac2r_jacobian(l1, l2, t1, t2);
        let det_matrix = j11 * j22 - j12 * j21;
        assert_relative_eq!(det_matrix, jac2r_determinant(l1, l2, t2), epsilon = 1e-12);
        assert_relative_eq!(det_matrix, l1 * l2 * t2.sin(), epsilon = 1e-12);
    }

    #[test]
    fn determinant_is_independent_of_first_joint_angle() {
        // det J ne dépend pas de θ1 : invariance par rotation globale du bras.
        let (l1, l2, t2) = (0.33_f64, 0.19_f64, 1.2_f64);
        let ref_det = l1 * l2 * t2.sin();
        for &t1 in &[-1.0_f64, 0.0, 0.5, 2.3, PI]
        {
            let [j11, j12, j21, j22] = jac2r_jacobian(l1, l2, t1, t2);
            assert_relative_eq!(j11 * j22 - j12 * j21, ref_det, epsilon = 1e-12);
        }
    }

    #[test]
    fn tip_velocity_is_linear_in_joint_rates() {
        // Linéarité : J·(ω1,ω2) = ω1·(colonne 1) + ω2·(colonne 2).
        let jac = jac2r_jacobian(0.30, 0.20, 0.4, 0.6);
        let (vx1, vy1) = jac2r_tip_velocity(jac, 1.0, 0.0);
        let (vx2, vy2) = jac2r_tip_velocity(jac, 0.0, 1.0);
        let (w1, w2) = (2.5_f64, -1.5_f64);
        let (vx, vy) = jac2r_tip_velocity(jac, w1, w2);
        assert_relative_eq!(vx, w1 * vx1 + w2 * vx2, epsilon = 1e-12);
        assert_relative_eq!(vy, w1 * vy1 + w2 * vy2, epsilon = 1e-12);
    }

    #[test]
    fn tip_speed_at_right_angle_is_computable_case() {
        // Cas chiffré : l1=0,3 l2=0,4 θ1=0 θ2=π/2, seul ω1=1 rad/s actif.
        // colonne 1 = (−l1·sinθ1 − l2·sin(θ1+θ2), l1·cosθ1 + l2·cos(θ1+θ2))
        //           = (−0,4 , 0,3). Vitesse outil = (−0,4 ; 0,3) m/s.
        let jac = jac2r_jacobian(0.30, 0.40, 0.0, PI / 2.0);
        let (vx, vy) = jac2r_tip_velocity(jac, 1.0, 0.0);
        assert_relative_eq!(vx, -0.40, epsilon = 1e-12);
        assert_relative_eq!(vy, 0.30, epsilon = 1e-12);
        // Norme de la vitesse = ‖colonne 1‖ = √(l1²+l2²) car les colonnes sont ⊥ ? Non :
        // ici seule la 1re colonne agit, sa norme vaut √(0,4²+0,3²)=0,5 m/s.
        assert_relative_eq!((vx * vx + vy * vy).sqrt(), 0.50, epsilon = 1e-12);
    }

    #[test]
    fn extended_and_folded_configurations_are_singular() {
        // sin θ2 = 0 aux configurations alignées : det J nul et is_singular vrai.
        let (l1, l2) = (0.35_f64, 0.20_f64);
        for &t2 in &[0.0_f64, PI, -PI]
        {
            assert_relative_eq!(jac2r_determinant(l1, l2, t2), 0.0, epsilon = 1e-12);
            assert!(jac2r_is_singular(t2, 1e-9));
        }
        // À θ2 = π/2 le bras est loin de la singularité.
        assert!(!jac2r_is_singular(PI / 2.0, 1e-9));
    }

    #[test]
    fn determinant_is_odd_in_theta2() {
        // Antisymétrie : det J(−θ2) = −det J(θ2) car sin est impair.
        let (l1, l2) = (0.28_f64, 0.22_f64);
        for &t2 in &[0.3_f64, 1.1, 2.0]
        {
            assert_relative_eq!(
                jac2r_determinant(l1, l2, -t2),
                -jac2r_determinant(l1, l2, t2),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    #[should_panic(expected = "tolérance doit être positive")]
    fn negative_tolerance_panics() {
        jac2r_is_singular(0.5, -1e-6);
    }
}
