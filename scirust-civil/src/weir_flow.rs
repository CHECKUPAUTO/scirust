//! Débit sur **déversoir** (weir) — évacuateurs et dispositifs de mesure à
//! surface libre : déversoir rectangulaire à mince paroi, déversoir triangulaire
//! en V, seuil épais (broad-crested), et charge amont déduite du débit.
//!
//! ```text
//! rectangulaire (mince paroi)  Q = (2/3)·Cd·b·√(2g)·H^{3/2}
//! triangulaire en V            Q = (8/15)·Cd·tan(θ/2)·√(2g)·H^{5/2}
//! seuil épais (crit.)          Q = Cd·b·√g·((2/3)·H)^{3/2}
//! charge amont (rect.)         H = ( Q / ((2/3)·Cd·b·√(2g)) )^{2/3}
//! ```
//!
//! `Q` débit (m³/s), `Cd` coefficient de débit (empirique, sans dimension),
//! `b` largeur de la crête ou du seuil (m), `H` charge amont mesurée dans le
//! plan d'eau non perturbé (m), `θ` angle d'ouverture du V (rad), `g`
//! accélération de la pesanteur (m/s²).
//!
//! **Convention** : SI strict et cohérent — mètres (m) et secondes (s), l'angle
//! d'échancrure `θ` est exprimé en **radians**. Types `f64`.
//!
//! **Limite honnête** : déversoirs de **mesure** ou d'**évacuation**. Le
//! coefficient de débit `Cd` est **empirique et fourni par l'appelant** d'après
//! l'abaque ou la norme applicable au type de déversoir (Rehbock, Kindsvater-
//! Carter, etc. — jamais une valeur « par défaut » inventée), et l'accélération
//! de la pesanteur `g` est **fournie**. La charge `H` est supposée **mesurée en
//! amont dans un écoulement non perturbé** (loin de la crête). Les formules
//! supposent une **nappe aérée** et un **déversoir dénoyé** (écoulement libre à
//! l'aval, sans influence du niveau aval). Ce module relève de l'hydraulique
//! classique et ne traite ni le noyage, ni les effets de viscosité/tension
//! superficielle aux très faibles charges.

use core::f64::consts::PI;

/// Débit d'un **déversoir rectangulaire à mince paroi**
/// `Q = (2/3)·Cd·b·√(2g)·H^{3/2}` (m³/s).
///
/// Panique si `discharge_coefficient <= 0`, `width < 0`, `head < 0` ou
/// `gravity <= 0`.
pub fn weir_rectangular_flow(
    discharge_coefficient: f64,
    width: f64,
    head: f64,
    gravity: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0,
        "le coefficient de débit Cd doit être strictement positif"
    );
    assert!(width >= 0.0, "la largeur b doit être positive ou nulle");
    assert!(head >= 0.0, "la charge amont H doit être positive ou nulle");
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur g doit être strictement positive"
    );
    (2.0 / 3.0) * discharge_coefficient * width * (2.0 * gravity).sqrt() * head.powf(1.5)
}

/// Débit d'un **déversoir triangulaire en V**
/// `Q = (8/15)·Cd·tan(θ/2)·√(2g)·H^{5/2}` (m³/s), angle d'ouverture `θ` en
/// **radians**.
///
/// Panique si `discharge_coefficient <= 0`, `notch_angle` hors de `]0, π[`,
/// `head < 0` ou `gravity <= 0`.
pub fn weir_triangular_flow(
    discharge_coefficient: f64,
    notch_angle: f64,
    head: f64,
    gravity: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0,
        "le coefficient de débit Cd doit être strictement positif"
    );
    assert!(
        notch_angle > 0.0 && notch_angle < PI,
        "l'angle d'échancrure θ doit être dans ]0, π[ radians"
    );
    assert!(head >= 0.0, "la charge amont H doit être positive ou nulle");
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur g doit être strictement positive"
    );
    (8.0 / 15.0)
        * discharge_coefficient
        * (notch_angle / 2.0).tan()
        * (2.0 * gravity).sqrt()
        * head.powf(2.5)
}

/// Débit d'un **seuil épais** (broad-crested weir) en condition critique
/// `Q = Cd·b·√g·((2/3)·H)^{3/2}` (m³/s).
///
/// Panique si `discharge_coefficient <= 0`, `width < 0`, `head < 0` ou
/// `gravity <= 0`.
pub fn weir_broad_crested_flow(
    discharge_coefficient: f64,
    width: f64,
    head: f64,
    gravity: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0,
        "le coefficient de débit Cd doit être strictement positif"
    );
    assert!(width >= 0.0, "la largeur b doit être positive ou nulle");
    assert!(head >= 0.0, "la charge amont H doit être positive ou nulle");
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur g doit être strictement positive"
    );
    discharge_coefficient * width * gravity.sqrt() * ((2.0 / 3.0) * head).powf(1.5)
}

/// Charge amont d'un **déversoir rectangulaire** déduite du débit
/// `H = ( Q / ((2/3)·Cd·b·√(2g)) )^{2/3}` (m). Réciproque de
/// [`weir_rectangular_flow`].
///
/// Panique si `flow < 0`, `discharge_coefficient <= 0`, `width <= 0` ou
/// `gravity <= 0`.
pub fn weir_head_from_flow_rectangular(
    flow: f64,
    discharge_coefficient: f64,
    width: f64,
    gravity: f64,
) -> f64 {
    assert!(flow >= 0.0, "le débit Q doit être positif ou nul");
    assert!(
        discharge_coefficient > 0.0,
        "le coefficient de débit Cd doit être strictement positif"
    );
    assert!(width > 0.0, "la largeur b doit être strictement positive");
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur g doit être strictement positive"
    );
    (flow / ((2.0 / 3.0) * discharge_coefficient * width * (2.0 * gravity).sqrt())).powf(2.0 / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_PI_2;

    #[test]
    fn rectangular_head_reciprocity() {
        // Aller-retour Q(H) puis H(Q) : on doit retrouver la charge de départ.
        let cd = 0.62;
        let b = 1.0;
        let h = 0.3;
        let g = 9.81;
        let q = weir_rectangular_flow(cd, b, h, g);
        let h_back = weir_head_from_flow_rectangular(q, cd, b, g);
        assert_relative_eq!(h_back, h, max_relative = 1e-9);
    }

    #[test]
    fn rectangular_scales_with_head_power_three_halves() {
        // Q ∝ H^{3/2} : quadrupler la charge multiplie le débit par 4^{3/2} = 8.
        let cd = 0.62;
        let b = 1.5;
        let g = 9.81;
        let q1 = weir_rectangular_flow(cd, b, 0.2, g);
        let q4 = weir_rectangular_flow(cd, b, 0.8, g);
        assert_relative_eq!(q4, 8.0 * q1, max_relative = 1e-9);
        // Q ∝ b : doubler la largeur double le débit (charge fixée).
        let q2b = weir_rectangular_flow(cd, 2.0 * b, 0.2, g);
        assert_relative_eq!(q2b, 2.0 * q1, max_relative = 1e-9);
    }

    #[test]
    fn triangular_ninety_degree_notch_worked_case() {
        // V à 90° : θ = π/2, tan(θ/2) = tan(45°) = 1, Cd = 0,58, H = 0,2, g = 9,81.
        // √(2g) = √19,62 = 4,429447.
        // H^{5/2} = 0,2^{2,5} = 0,04·√0,2 = 0,04·0,4472136 = 0,017888544.
        // Q = (8/15)·0,58·1·4,429447·0,017888544
        //   = 0,309333·4,429447·0,017888544 = 0,0245152 m³/s.
        let q = weir_triangular_flow(0.58, FRAC_PI_2, 0.2, 9.81);
        assert_relative_eq!(q, 0.0245152, max_relative = 1e-3);
    }

    #[test]
    fn triangular_scales_with_head_power_five_halves() {
        // Q ∝ H^{5/2} : quadrupler la charge multiplie le débit par 4^{5/2} = 32.
        let q1 = weir_triangular_flow(0.58, FRAC_PI_2, 0.15, 9.81);
        let q4 = weir_triangular_flow(0.58, FRAC_PI_2, 0.6, 9.81);
        assert_relative_eq!(q4, 32.0 * q1, max_relative = 1e-9);
    }

    #[test]
    fn broad_crested_worked_case() {
        // Seuil épais : Cd = 0,85, b = 1 m, H = 0,3 m, g = 9,81.
        // √g = 3,132092 ; ((2/3)·0,3)^{1,5} = 0,2^{1,5} = 0,2·√0,2 = 0,089442719.
        // Q = 0,85·1·3,132092·0,089442719 = 0,238122 m³/s.
        let q = weir_broad_crested_flow(0.85, 1.0, 0.3, 9.81);
        assert_relative_eq!(q, 0.238122, max_relative = 1e-3);
    }

    #[test]
    fn zero_head_gives_no_flow() {
        // Charge nulle : aucun débit sur le déversoir.
        assert_relative_eq!(
            weir_rectangular_flow(0.62, 1.0, 0.0, 9.81),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    #[should_panic(expected = "le coefficient de débit Cd doit être strictement positif")]
    fn zero_discharge_coefficient_panics() {
        let _ = weir_rectangular_flow(0.0, 1.0, 0.3, 9.81);
    }
}
