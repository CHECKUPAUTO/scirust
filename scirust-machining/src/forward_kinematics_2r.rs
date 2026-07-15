//! Cinématique directe d'un bras robotisé planaire **2R** (deux segments rigides
//! reliés par deux liaisons rotoïdes).
//!
//! ```text
//! coude (fin du segment 1)   (x_e, y_e) = (l1·cos θ1, l1·sin θ1)
//! outil TCP (fin du segment 2)
//!     x = l1·cos θ1 + l2·cos(θ1 + θ2)
//!     y = l1·sin θ1 + l2·sin(θ1 + θ2)
//! distance radiale (portée)  r = √(x² + y²)
//! ```
//!
//! `l1`, `l2` longueurs des deux segments (m, ≥ 0), `θ1` angle de la première
//! liaison mesuré depuis l'axe `+x` (rad), `θ2` angle de la seconde liaison mesuré
//! relativement au segment 1 (rad), `(x_e, y_e)` position du coude (m), `(x, y)`
//! position de l'outil / *tool center point* (m), `r` distance radiale de l'origine
//! au point visé (m). TCP = *tool center point*.
//!
//! **Convention** : angles en rad (sens trigonométrique), longueurs en m (SI
//! cohérent, tout facteur d'échelle est conservé). L'origine est la base du bras.
//!
//! **Limite honnête** : bras planaire idéal — segments parfaitement rigides,
//! liaisons ponctuelles sans jeu ni flexion, mouvement dans un seul plan. Les
//! longueurs `l1`, `l2` et les angles `θ1`, `θ2` sont FOURNIS par l'appelant ; ce
//! module n'invente aucune valeur « par défaut » de géométrie de bras, de course
//! articulaire ni de limite mécanique.

/// Position `(x_e, y_e)` du coude (extrémité du premier segment).
///
/// `(x_e, y_e) = (l1·cos θ1, l1·sin θ1)` (m).
///
/// Panique si `l1 < 0`.
pub fn fk2r_elbow_position(l1: f64, theta1: f64) -> (f64, f64) {
    assert!(
        l1 >= 0.0,
        "la longueur du premier segment doit être positive ou nulle"
    );
    (l1 * theta1.cos(), l1 * theta1.sin())
}

/// Position `(x, y)` de l'outil (*tool center point*) au bout du bras 2R.
///
/// `x = l1·cos θ1 + l2·cos(θ1 + θ2)`, `y = l1·sin θ1 + l2·sin(θ1 + θ2)` (m).
///
/// Panique si `l1 < 0` ou si `l2 < 0`.
pub fn fk2r_tcp_position(l1: f64, l2: f64, theta1: f64, theta2: f64) -> (f64, f64) {
    assert!(
        l1 >= 0.0,
        "la longueur du premier segment doit être positive ou nulle"
    );
    assert!(
        l2 >= 0.0,
        "la longueur du second segment doit être positive ou nulle"
    );
    let (xe, ye) = fk2r_elbow_position(l1, theta1);
    let phi = theta1 + theta2;
    (xe + l2 * phi.cos(), ye + l2 * phi.sin())
}

/// Distance radiale (portée) de l'origine à un point : `r = √(x² + y²)` (m).
///
/// Utile pour mesurer l'allonge d'un point atteint par l'outil du bras.
///
/// Ne panique jamais (défini pour tout `(x, y)`).
pub fn fk2r_reach_distance(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn fully_extended_arm_reaches_sum_of_segments() {
        // θ1 = θ2 = 0 : bras aligné sur +x, l'outil est en (l1 + l2, 0).
        let (x, y) = fk2r_tcp_position(0.30, 0.20, 0.0, 0.0);
        assert_relative_eq!(x, 0.50, epsilon = 1e-12);
        assert_relative_eq!(y, 0.0, epsilon = 1e-12);
        // La portée vaut alors exactement l1 + l2.
        assert_relative_eq!(fk2r_reach_distance(x, y), 0.50, epsilon = 1e-12);
    }

    #[test]
    fn tcp_equals_elbow_plus_second_segment() {
        // Identité : TCP = coude + vecteur du second segment.
        let (l1, l2, t1, t2) = (0.42_f64, 0.27_f64, 0.7_f64, -0.5_f64);
        let (xe, ye) = fk2r_elbow_position(l1, t1);
        let (x, y) = fk2r_tcp_position(l1, l2, t1, t2);
        assert_relative_eq!(x - xe, l2 * (t1 + t2).cos(), epsilon = 1e-12);
        assert_relative_eq!(y - ye, l2 * (t1 + t2).sin(), epsilon = 1e-12);
        // La longueur du second segment est la distance coude→TCP.
        assert_relative_eq!(fk2r_reach_distance(x - xe, y - ye), l2, epsilon = 1e-12);
    }

    #[test]
    fn folded_back_arm_reaches_difference_of_segments() {
        // θ2 = π : le second segment se replie sur le premier, portée = |l1 − l2|.
        let (l1, l2) = (0.35_f64, 0.20_f64);
        for &t1 in &[0.0_f64, 0.6, 1.3, PI]
        {
            let (x, y) = fk2r_tcp_position(l1, l2, t1, PI);
            assert_relative_eq!(fk2r_reach_distance(x, y), l1 - l2, epsilon = 1e-12);
        }
    }

    #[test]
    fn elbow_lies_on_first_segment_circle() {
        // Le coude est toujours à distance l1 de l'origine.
        let l1 = 0.25_f64;
        for &t1 in &[-1.0_f64, 0.0, 0.4, 2.1, 3.0]
        {
            let (xe, ye) = fk2r_elbow_position(l1, t1);
            assert_relative_eq!(fk2r_reach_distance(xe, ye), l1, epsilon = 1e-12);
        }
    }

    #[test]
    fn right_angle_elbow_gives_pythagorean_reach() {
        // θ1 = 0, θ2 = π/2 : segments perpendiculaires, portée = √(l1² + l2²).
        let (l1, l2) = (0.30_f64, 0.40_f64);
        let (x, y) = fk2r_tcp_position(l1, l2, 0.0, PI / 2.0);
        assert_relative_eq!(x, l1, epsilon = 1e-12);
        assert_relative_eq!(y, l2, epsilon = 1e-12);
        assert_relative_eq!(
            fk2r_reach_distance(x, y),
            (l1 * l1 + l2 * l2).sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn reach_is_rotation_invariant() {
        // Une rotation globale θ1 → θ1 + α ne change pas la portée du TCP.
        let (l1, l2, t2) = (0.33_f64, 0.19_f64, 0.9_f64);
        let (x0, y0) = fk2r_tcp_position(l1, l2, 0.0, t2);
        let r0 = fk2r_reach_distance(x0, y0);
        for &alpha in &[0.5_f64, 1.7, 2.8, 4.5]
        {
            let (x, y) = fk2r_tcp_position(l1, l2, alpha, t2);
            assert_relative_eq!(fk2r_reach_distance(x, y), r0, epsilon = 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "premier segment doit être positive")]
    fn negative_first_segment_panics() {
        fk2r_tcp_position(-0.1, 0.2, 0.0, 0.0);
    }
}
