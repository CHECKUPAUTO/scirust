//! Cordon de soudure d'angle (**soudure en filet**) — gorge, contrainte de
//! cisaillement, effort admissible et longueur de cordon requise.
//!
//! ```text
//! gorge (cordon isocèle)  a      = leg / √2   (= 0,707·leg)
//! contrainte cisaillement tau    = F / (a·L)
//! effort admissible       F_adm  = tau_adm·0,707·leg·L = tau_adm·a·L
//! longueur requise        L_req  = F / (tau_adm·0,707·leg) = F / (tau_adm·a)
//! ```
//!
//! `leg` = jambe (côté) du triangle du cordon isocèle (m), `a` = gorge utile,
//! épaisseur de calcul du cordon (m), `F` = effort transmis par le cordon (N),
//! `L` = longueur du cordon (m), `tau` = contrainte de cisaillement sur la gorge
//! (Pa), `tau_adm` = contrainte de cisaillement admissible du cordon (Pa),
//! `F_adm` = effort admissible / capacité du cordon (N), `L_req` = longueur de
//! cordon nécessaire (m).
//!
//! **Convention** : SI cohérent — dimensions en m, efforts en N, contraintes en Pa.
//!
//! **Limite honnête** : modèle du **cordon d'angle isocèle** sollicité en
//! **cisaillement sur la gorge** avec répartition d'effort **uniforme idéalisée**
//! le long du cordon. Le facteur `0,707 = 1/√2` suppose un cordon à jambes égales
//! et une gorge à 45° ; il ne s'applique pas aux cordons dissymétriques ni aux
//! cordons à pénétration. Ce modèle **ne traite pas** la flexion d'un groupe de
//! cordons (excentricité de l'effort, répartition non uniforme, méthode du moment
//! polaire d'inertie) ni les concentrations aux extrémités. La contrainte
//! **admissible** `tau_adm` dépend du métal d'apport, du métal de base, du procédé
//! et du code de calcul retenu : elle est **fournie par l'appelant** — aucune
//! valeur « par défaut » n'est inventée ici.

use core::f64::consts::SQRT_2;

/// Gorge utile du cordon isocèle `a = leg / √2` (soit `0,707·leg`).
///
/// `leg_size` = `leg` jambe du cordon (m) ; renvoie la gorge `a` (m).
///
/// Panique si `leg_size < 0`.
pub fn fillet_weld_throat(leg_size: f64) -> f64 {
    assert!(leg_size >= 0.0, "leg ≥ 0 requis");
    leg_size / SQRT_2
}

/// Contrainte de cisaillement sur la gorge `tau = F / (a·L)`.
///
/// `load` = `F` (N), `throat` = `a` gorge (m), `length` = `L` longueur du cordon
/// (m) ; renvoie une contrainte (Pa).
///
/// Panique si `load < 0`, `throat <= 0` ou `length <= 0`.
pub fn fillet_weld_shear_stress(load: f64, throat: f64, length: f64) -> f64 {
    assert!(
        load >= 0.0 && throat > 0.0 && length > 0.0,
        "F ≥ 0, a > 0 et L > 0 requis"
    );
    load / (throat * length)
}

/// Effort admissible (capacité) du cordon `F_adm = tau_adm·0,707·leg·L`.
///
/// `allowable_shear` = `tau_adm` (Pa), `leg_size` = `leg` jambe (m), `length` = `L`
/// longueur du cordon (m) ; renvoie un effort (N).
///
/// Panique si `allowable_shear < 0`, `leg_size < 0` ou `length <= 0`.
pub fn fillet_weld_capacity(allowable_shear: f64, leg_size: f64, length: f64) -> f64 {
    assert!(
        allowable_shear >= 0.0 && leg_size >= 0.0 && length > 0.0,
        "tau_adm ≥ 0, leg ≥ 0 et L > 0 requis"
    );
    allowable_shear * fillet_weld_throat(leg_size) * length
}

/// Longueur de cordon requise `L_req = F / (tau_adm·0,707·leg)`.
///
/// `load` = `F` (N), `allowable_shear` = `tau_adm` (Pa), `leg_size` = `leg` jambe
/// (m) ; renvoie une longueur (m).
///
/// Panique si `load < 0`, `allowable_shear <= 0` ou `leg_size <= 0`.
pub fn fillet_weld_required_length(load: f64, allowable_shear: f64, leg_size: f64) -> f64 {
    assert!(
        load >= 0.0 && allowable_shear > 0.0 && leg_size > 0.0,
        "F ≥ 0, tau_adm > 0 et leg > 0 requis"
    );
    load / (allowable_shear * fillet_weld_throat(leg_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn throat_is_leg_over_sqrt2() {
        // La gorge multipliée par √2 redonne la jambe (cordon isocèle).
        let leg = 0.006_f64;
        let a = fillet_weld_throat(leg);
        assert_relative_eq!(a * SQRT_2, leg, epsilon = 1e-12);
        // Et vaut bien (1/√2)·leg ≈ 0,707·leg.
        assert_relative_eq!(a, FRAC_1_SQRT_2 * leg, epsilon = 1e-12);
    }

    #[test]
    fn stress_and_capacity_are_reciprocal() {
        // Charger le cordon à sa capacité (F = F_adm) redonne tau_adm sur la gorge.
        let (tau_adm, leg, l) = (100e6_f64, 0.006, 0.100);
        let a = fillet_weld_throat(leg);
        let f_adm = fillet_weld_capacity(tau_adm, leg, l);
        assert_relative_eq!(
            fillet_weld_shear_stress(f_adm, a, l),
            tau_adm,
            epsilon = 1e-3
        );
    }

    #[test]
    fn required_length_reaches_allowable_stress() {
        // À L = L_req, la contrainte sur la gorge vaut exactement l'admissible.
        let (f, tau_adm, leg) = (20e3_f64, 120e6, 0.008);
        let l = fillet_weld_required_length(f, tau_adm, leg);
        let a = fillet_weld_throat(leg);
        assert_relative_eq!(fillet_weld_shear_stress(f, a, l), tau_adm, epsilon = 1e-3);
    }

    #[test]
    fn capacity_proportional_to_length() {
        // F_adm ∝ L : doubler la longueur double la capacité.
        let single = fillet_weld_capacity(100e6, 0.006, 0.050);
        let double = fillet_weld_capacity(100e6, 0.006, 0.100);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-6);
    }

    #[test]
    fn realistic_fillet_weld() {
        // Cordon isocèle jambe 6 mm, longueur 100 mm, tau_adm = 100 MPa.
        // Gorge a = 6/√2 = 4,2426 mm → capacité = 100e6·0,0042426·0,1 ≈ 42,43 kN.
        let f_adm = fillet_weld_capacity(100e6, 0.006, 0.100);
        assert_relative_eq!(f_adm, 42_426.406_871_2_f64, epsilon = 1.0);
        // Sous 21,213 kN (moitié de la capacité), la contrainte vaut 50 MPa.
        let a = fillet_weld_throat(0.006);
        assert_relative_eq!(
            fillet_weld_shear_stress(f_adm / 2.0, a, 0.100),
            50e6,
            epsilon = 10.0
        );
    }

    #[test]
    fn required_length_matches_capacity_inverse() {
        // Dimensionner pour F puis évaluer la capacité au même tau_adm redonne F.
        let (f, tau_adm, leg) = (30e3_f64, 90e6, 0.005);
        let l = fillet_weld_required_length(f, tau_adm, leg);
        assert_relative_eq!(fillet_weld_capacity(tau_adm, leg, l), f, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "tau_adm > 0")]
    fn zero_allowable_shear_panics() {
        fillet_weld_required_length(1000.0, 0.0, 0.006);
    }
}
