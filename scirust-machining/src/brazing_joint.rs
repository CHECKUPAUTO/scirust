//! Joint brasé (**brazed joint**) — capillarité de remplissage du jeu et
//! résistance en cisaillement d'un assemblage à recouvrement.
//!
//! ```text
//! aire de cisaillement   A     = L·w
//! effort admissible      F_adm = tau_f·L·w
//! recouvrement requis    L_req = F·n / (tau_f·w)
//! pression capillaire    p_cap = 2·gamma·cos(theta) / g
//! ```
//!
//! `L` longueur de recouvrement (m), `w` largeur du joint (m), `A` aire de
//! cisaillement collée (m²), `tau_f` résistance au cisaillement du métal d'apport
//! (Pa), `F_adm` effort de cisaillement admissible transmis par le joint (N),
//! `F` charge de service (N), `n` facteur de sécurité (sans dimension),
//! `L_req` longueur de recouvrement nécessaire (m), `gamma` tension superficielle
//! du métal d'apport fondu (N/m), `theta` angle de contact de mouillage (rad),
//! `g` jeu capillaire (m), `p_cap` pression capillaire de remplissage (Pa).
//!
//! **Convention** : SI cohérent — longueurs en m, contraintes et pressions en Pa,
//! charges en N, tension superficielle en N/m, angle en rad.
//!
//! **Limite honnête** : joint à recouvrement sollicité en **cisaillement uniforme
//! idéalisé** sur l'aire `L·w` (pas de pics d'extrémité ni de pelage). La
//! **résistance au cisaillement** `tau_f` du métal d'apport et la **tension
//! superficielle** `gamma` avec l'**angle de contact** `theta` dépendent du couple
//! apport/métal de base, de la température et de l'atmosphère de brasage : elles
//! sont **fournies par l'appelant**. Le **jeu capillaire optimal** (typiquement
//! 0,05–0,2 mm) est lui aussi **fourni** — aucune valeur « par défaut » n'est
//! inventée ici. Distinct de [`crate::adhesive_lap_joint`] (collage structural).

/// Aire de cisaillement d'un joint à recouvrement `A = L·w`.
///
/// `overlap_length` = `L` (m), `joint_width` = `w` (m) ; renvoie une aire (m²).
///
/// Panique si `overlap_length <= 0` ou `joint_width <= 0`.
pub fn brazing_shear_area(overlap_length: f64, joint_width: f64) -> f64 {
    assert!(
        overlap_length > 0.0 && joint_width > 0.0,
        "L > 0 et w > 0 requis"
    );
    overlap_length * joint_width
}

/// Effort de cisaillement admissible du joint `F_adm = tau_f·L·w`.
///
/// `filler_shear_strength` = `tau_f` (Pa), `overlap_length` = `L` (m),
/// `joint_width` = `w` (m) ; renvoie une charge (N).
///
/// Panique si `filler_shear_strength < 0`, `overlap_length <= 0` ou
/// `joint_width <= 0`.
pub fn brazing_joint_shear_strength(
    filler_shear_strength: f64,
    overlap_length: f64,
    joint_width: f64,
) -> f64 {
    assert!(
        filler_shear_strength >= 0.0 && overlap_length > 0.0 && joint_width > 0.0,
        "tau_f ≥ 0, L > 0 et w > 0 requis"
    );
    filler_shear_strength * overlap_length * joint_width
}

/// Longueur de recouvrement nécessaire `L_req = F·n / (tau_f·w)`.
///
/// `load` = `F` (N), `filler_shear_strength` = `tau_f` (Pa),
/// `joint_width` = `w` (m), `safety_factor` = `n` (sans dimension) ; renvoie une
/// longueur (m).
///
/// Panique si `load < 0`, `filler_shear_strength <= 0`, `joint_width <= 0` ou
/// `safety_factor <= 0`.
pub fn brazing_required_overlap(
    load: f64,
    filler_shear_strength: f64,
    joint_width: f64,
    safety_factor: f64,
) -> f64 {
    assert!(
        load >= 0.0 && filler_shear_strength > 0.0 && joint_width > 0.0 && safety_factor > 0.0,
        "F ≥ 0, tau_f > 0, w > 0 et n > 0 requis"
    );
    load * safety_factor / (filler_shear_strength * joint_width)
}

/// Pression capillaire de remplissage du jeu `p_cap = 2·gamma·cos(theta) / g`.
///
/// `surface_tension` = `gamma` (N/m), `contact_angle_rad` = `theta` (rad),
/// `gap` = `g` (m) ; renvoie une pression (Pa), positive si le métal d'apport
/// mouille (`theta < pi/2`).
///
/// Panique si `surface_tension < 0`, `contact_angle_rad` hors de `[0, pi]` ou
/// `gap <= 0`.
pub fn brazing_capillary_gap_pressure(
    surface_tension: f64,
    contact_angle_rad: f64,
    gap: f64,
) -> f64 {
    assert!(
        surface_tension >= 0.0
            && (0.0..=core::f64::consts::PI).contains(&contact_angle_rad)
            && gap > 0.0,
        "gamma ≥ 0, 0 ≤ theta ≤ pi et g > 0 requis"
    );
    2.0 * surface_tension * contact_angle_rad.cos() / gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn strength_is_area_times_stress() {
        // F_adm = tau_f·A : l'effort admissible est l'aire de cisaillement
        // multipliée par la résistance du métal d'apport (cohérence des deux fn).
        let (tau, l, w) = (150e6_f64, 0.005, 0.020);
        let area = brazing_shear_area(l, w);
        assert_relative_eq!(
            brazing_joint_shear_strength(tau, l, w),
            tau * area,
            epsilon = 1e-3
        );
    }

    #[test]
    fn required_overlap_reciprocal_of_strength() {
        // Avec n = 1, le recouvrement requis pour une charge F redonne, injecté
        // dans la capacité, exactement F : réciprocité effort <-> longueur.
        let (f, tau, w) = (15000.0_f64, 150e6, 0.020);
        let l_req = brazing_required_overlap(f, tau, w, 1.0);
        assert_relative_eq!(
            brazing_joint_shear_strength(tau, l_req, w),
            f,
            epsilon = 1e-6
        );
    }

    #[test]
    fn required_overlap_proportional_to_safety_factor() {
        // L_req ∝ n : doubler le facteur de sécurité double le recouvrement.
        let single = brazing_required_overlap(8000.0, 120e6, 0.015, 1.5);
        let double = brazing_required_overlap(8000.0, 120e6, 0.015, 3.0);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-9);
    }

    #[test]
    fn capillary_pressure_vanishes_at_ninety_degrees() {
        // theta = pi/2 : cos = 0, pas de moteur capillaire (limite de mouillage).
        let p = brazing_capillary_gap_pressure(0.5, core::f64::consts::FRAC_PI_2, 1e-4);
        assert_relative_eq!(p, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn realistic_brazed_lap_joint() {
        // Recouvrement 5 mm × 20 mm, apport tau_f = 150 MPa.
        // A = 0,005·0,020 = 1e-4 m² ; F_adm = 150e6·1e-4 = 15000 N.
        let area = brazing_shear_area(0.005, 0.020);
        assert_relative_eq!(area, 1e-4, epsilon = 1e-12);
        let f_adm = brazing_joint_shear_strength(150e6, 0.005, 0.020);
        assert_relative_eq!(f_adm, 15000.0, epsilon = 1e-6);
    }

    #[test]
    fn realistic_capillary_pressure() {
        // gamma = 0,5 N/m, theta = 20° = 0,349066 rad, jeu g = 0,1 mm.
        // p = 2·0,5·cos(20°)/1e-4 = cos(20°)·1e4 = 0,9396926·1e4 ≈ 9396,926 Pa.
        let theta = 20.0_f64.to_radians();
        let p = brazing_capillary_gap_pressure(0.5, theta, 1e-4);
        assert_relative_eq!(p, 9396.926_f64, epsilon = 1e-2);
    }

    #[test]
    #[should_panic(expected = "g > 0 requis")]
    fn zero_gap_panics() {
        // Jeu nul : pression capillaire non définie -> panique attendue.
        brazing_capillary_gap_pressure(0.5, 0.0, 0.0);
    }
}
