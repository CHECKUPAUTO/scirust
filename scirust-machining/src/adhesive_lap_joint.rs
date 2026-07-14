//! Collage à simple recouvrement (**single lap joint**) — contrainte de
//! cisaillement moyenne, résistance du joint et longueur de recouvrement requise.
//!
//! ```text
//! contrainte moyenne   tau_avg = F / (w·L)
//! résistance du joint   F_max  = tau·w·L
//! recouvrement requis   L_req  = F / (tau_adm·w)
//! ```
//!
//! `F` charge de traction transmise par le joint (N), `w` largeur du recouvrement
//! (m), `L` longueur de recouvrement (m), `tau_avg` contrainte de cisaillement
//! moyenne dans le film de colle (Pa), `tau` résistance au cisaillement du joint
//! collé (Pa), `tau_adm` contrainte de cisaillement admissible (Pa), `F_max`
//! charge de rupture / capacité du joint (N), `L_req` longueur de recouvrement
//! nécessaire (m).
//!
//! **Convention** : SI cohérent — charges en N, dimensions en m, contraintes en Pa.
//!
//! **Limite honnête** : modèle de contrainte de cisaillement **moyenne uniforme**
//! sur toute l'aire de recouvrement `w·L`. La réalité est très différente : le
//! cisaillement présente des **pics aux extrémités** du recouvrement (analyse de
//! Volkersen/Goland-Reissner) avec un cœur peu chargé, et des contraintes de
//! **pelage** dues à l'excentricité de la ligne d'action ; la capacité ne croît
//! donc pas indéfiniment avec `L`. Ce modèle sert au pré-dimensionnement, pas à la
//! prédiction de rupture. La **résistance au cisaillement** `tau` et la contrainte
//! **admissible** `tau_adm` dépendent de l'adhésif, des substrats, de la
//! préparation de surface et des conditions d'emploi : elles sont **fournies par
//! l'appelant** — aucune valeur « par défaut » n'est inventée ici.

/// Contrainte de cisaillement moyenne `tau_avg = F / (w·L)` dans le film de colle.
///
/// `load` = `F` (N), `width` = `w` (m), `overlap_length` = `L` (m) ; renvoie une
/// contrainte (Pa).
///
/// Panique si `load < 0`, `width <= 0` ou `overlap_length <= 0`.
pub fn adhesive_average_shear_stress(load: f64, width: f64, overlap_length: f64) -> f64 {
    assert!(
        load >= 0.0 && width > 0.0 && overlap_length > 0.0,
        "F ≥ 0, w > 0 et L > 0 requis"
    );
    load / (width * overlap_length)
}

/// Capacité (charge de rupture) du joint `F_max = tau·w·L`.
///
/// `shear_strength` = `tau` (Pa), `width` = `w` (m), `overlap_length` = `L` (m) ;
/// renvoie une charge (N).
///
/// Panique si `shear_strength < 0`, `width <= 0` ou `overlap_length <= 0`.
pub fn adhesive_joint_strength(shear_strength: f64, width: f64, overlap_length: f64) -> f64 {
    assert!(
        shear_strength >= 0.0 && width > 0.0 && overlap_length > 0.0,
        "tau ≥ 0, w > 0 et L > 0 requis"
    );
    shear_strength * width * overlap_length
}

/// Longueur de recouvrement requise `L_req = F / (tau_adm·w)`.
///
/// `load` = `F` (N), `allowable_shear` = `tau_adm` (Pa), `width` = `w` (m) ;
/// renvoie une longueur (m).
///
/// Panique si `load < 0`, `allowable_shear <= 0` ou `width <= 0`.
pub fn required_overlap_length(load: f64, allowable_shear: f64, width: f64) -> f64 {
    assert!(
        load >= 0.0 && allowable_shear > 0.0 && width > 0.0,
        "F ≥ 0, tau_adm > 0 et w > 0 requis"
    );
    load / (allowable_shear * width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn stress_and_strength_are_reciprocal() {
        // Charger le joint à sa capacité (F = F_max) redonne tau en contrainte
        // moyenne : réciprocité entre capacité et contrainte.
        let (tau, w, l) = (12e6_f64, 0.025, 0.020);
        let f_max = adhesive_joint_strength(tau, w, l);
        assert_relative_eq!(
            adhesive_average_shear_stress(f_max, w, l),
            tau,
            epsilon = 1e-3
        );
    }

    #[test]
    fn required_length_reaches_allowable_stress() {
        // À L = L_req, la contrainte moyenne vaut exactement l'admissible.
        let (f, tau_adm, w) = (3000.0_f64, 10e6, 0.030);
        let l = required_overlap_length(f, tau_adm, w);
        assert_relative_eq!(
            adhesive_average_shear_stress(f, w, l),
            tau_adm,
            epsilon = 1e-3
        );
    }

    #[test]
    fn average_stress_inversely_proportional_to_area() {
        // Doubler la longueur de recouvrement (donc l'aire) halve la contrainte.
        let single = adhesive_average_shear_stress(2000.0, 0.02, 0.015);
        let double = adhesive_average_shear_stress(2000.0, 0.02, 0.030);
        assert_relative_eq!(double, single / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn strength_is_bilinear_in_dimensions() {
        // F_max ∝ w et ∝ L : doubler chacun quadruple la capacité.
        let base = adhesive_joint_strength(15e6, 0.02, 0.010);
        let quad = adhesive_joint_strength(15e6, 0.04, 0.020);
        assert_relative_eq!(quad, 4.0 * base, epsilon = 1e-6);
    }

    #[test]
    fn realistic_epoxy_joint() {
        // Recouvrement 25 mm × 20 mm, adhésif époxy tau = 20 MPa.
        // Aire = 5e-4 m² → capacité 10 kN.
        let f_max = adhesive_joint_strength(20e6, 0.025, 0.020);
        assert_relative_eq!(f_max, 10e3, epsilon = 1.0);
        // Sous 5 kN, contrainte moyenne = 10 MPa (moitié de la résistance).
        assert_relative_eq!(
            adhesive_average_shear_stress(5e3, 0.025, 0.020),
            10e6,
            epsilon = 1.0
        );
    }

    #[test]
    fn required_length_matches_capacity_inverse() {
        // Dimensionner pour F puis évaluer la capacité au même tau redonne F.
        let (f, tau, w) = (4000.0_f64, 8e6, 0.025);
        let l = required_overlap_length(f, tau, w);
        assert_relative_eq!(adhesive_joint_strength(tau, w, l), f, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "tau_adm > 0")]
    fn zero_allowable_shear_panics() {
        required_overlap_length(1000.0, 0.0, 0.02);
    }
}
