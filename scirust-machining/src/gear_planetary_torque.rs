//! Répartition des **couples** dans un train épicycloïdal (planétaire) idéal —
//! équilibre des couples soleil / couronne / porte-satellites en régime permanent.
//!
//! ```text
//! couple couronne (porte-satellites fixe)  T_R = T_S · N_R/N_S
//! équilibre des couples (somme nulle)      T_S + T_R + T_C = 0
//! réaction du porte-satellites             T_C = −(T_S + T_R)
//! rapport de couple sortie porte-satellites (couronne fixe)
//!                                          T_C/T_S = (N_S + N_R)/N_S
//! effort tangentiel sur un organe          Ft = T / r
//! ```
//!
//! `T_S` couple sur le soleil, `T_R` sur la couronne, `T_C` sur le
//! porte-satellites (carrier) — N·m, couples algébriques (signe = sens).
//! `N_S`, `N_R` nombres de dents soleil et couronne. `r` rayon primitif (m),
//! `Ft` effort tangentiel au cercle primitif (N).
//!
//! **Convention** : unités SI (N·m, m, N), couples algébriques de même
//! convention de signe. **Limite honnête** : train épicycloïdal **idéal SANS
//! PERTES** — la somme des couples est nulle en régime permanent (statique de
//! réaction), inertie négligée. Les nombres de dents et les rayons primitifs
//! sont **fournis par l'appelant** ; le **rendement** (frottement engrènement,
//! paliers) est à appliquer **par l'appelant** et n'est jamais supposé ici.
//! Aucune valeur « par défaut » de matériau, de procédé ou de géométrie n'est
//! inventée. Complète [`crate::epicyclic`] (cinématique de Willis).

/// Couple sur la **couronne** déduit du couple soleil, porte-satellites fixe
/// `T_R = T_S · N_R/N_S` (N·m).
///
/// `sun_torque` en N·m ; `ring_teeth`, `sun_teeth` nombres de dents.
///
/// Panique si `sun_teeth == 0`.
pub fn planetary_torque_ring_from_sun(sun_torque: f64, ring_teeth: u32, sun_teeth: u32) -> f64 {
    assert!(sun_teeth > 0, "le soleil doit avoir au moins une dent");
    sun_torque * (ring_teeth as f64) / (sun_teeth as f64)
}

/// Couple de **réaction** du porte-satellites `T_C = −(T_S + T_R)` (N·m),
/// équilibre des couples d'un train sans pertes (somme nulle).
///
/// `sun_torque`, `ring_torque` en N·m.
///
/// Panique si l'un des couples n'est pas fini.
pub fn planetary_torque_carrier_reaction(sun_torque: f64, ring_torque: f64) -> f64 {
    assert!(
        sun_torque.is_finite() && ring_torque.is_finite(),
        "les couples doivent être finis"
    );
    -(sun_torque + ring_torque)
}

/// Rapport de **couple** sortie porte-satellites, **couronne fixe**
/// `T_C/T_S = (N_S + N_R)/N_S = 1 + N_R/N_S` (sans dimension).
///
/// Égal au rapport de réduction soleil → porte-satellites de
/// [`crate::epicyclic`] (conservation de la puissance à rendement unité).
///
/// Panique si `sun_teeth == 0`.
pub fn planetary_torque_ratio_carrier_output(sun_teeth: u32, ring_teeth: u32) -> f64 {
    assert!(sun_teeth > 0, "le soleil doit avoir au moins une dent");
    (sun_teeth as f64 + ring_teeth as f64) / (sun_teeth as f64)
}

/// Effort **tangentiel** au cercle primitif `Ft = T / r` (N).
///
/// `member_torque` en N·m, `pitch_radius` en m.
///
/// Panique si `pitch_radius <= 0`.
pub fn planetary_torque_tangential_force(member_torque: f64, pitch_radius: f64) -> f64 {
    assert!(
        pitch_radius > 0.0,
        "le rayon primitif doit être strictement positif"
    );
    member_torque / pitch_radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ring_torque_scales_with_teeth_ratio() {
        // N_S=20, N_R=50, T_S=10 N·m → T_R = 10·50/20 = 25 N·m.
        assert_relative_eq!(
            planetary_torque_ring_from_sun(10.0, 50, 20),
            25.0,
            epsilon = 1e-12
        );
        // Proportionnalité : doubler le couple soleil double le couple couronne.
        let t1 = planetary_torque_ring_from_sun(10.0, 50, 20);
        let t2 = planetary_torque_ring_from_sun(20.0, 50, 20);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-12);
    }

    #[test]
    fn three_torques_sum_to_zero() {
        // Équilibre : T_S + T_R + T_C = 0 par construction de la réaction.
        let ts = 10.0;
        let tr = planetary_torque_ring_from_sun(ts, 50, 20);
        let tc = planetary_torque_carrier_reaction(ts, tr);
        assert_relative_eq!(ts + tr + tc, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn carrier_reaction_matches_output_ratio() {
        // Avec T_R = T_S·N_R/N_S, on a |T_C| = T_S·(N_S+N_R)/N_S = T_S · rapport.
        let ts = 10.0;
        let tr = planetary_torque_ring_from_sun(ts, 50, 20);
        let tc = planetary_torque_carrier_reaction(ts, tr);
        let ratio = planetary_torque_ratio_carrier_output(20, 50);
        assert_relative_eq!(-tc, ts * ratio, epsilon = 1e-12);
        // Cas chiffré : rapport = 70/20 = 3,5 → T_C = −35 N·m.
        assert_relative_eq!(ratio, 3.5, epsilon = 1e-12);
        assert_relative_eq!(tc, -35.0, epsilon = 1e-12);
    }

    #[test]
    fn output_ratio_matches_epicyclic_reduction() {
        // Le rapport de couple (couronne fixe) est 1 + N_R/N_S, cohérent avec
        // le rapport de réduction cinématique du module epicyclic.
        assert_relative_eq!(
            planetary_torque_ratio_carrier_output(20, 50),
            1.0 + 50.0 / 20.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn tangential_force_is_torque_over_radius() {
        // T_C = 35 N·m au rayon primitif r = 0,05 m → Ft = 700 N.
        let ft = planetary_torque_tangential_force(35.0, 0.05);
        assert_relative_eq!(ft, 700.0, epsilon = 1e-9);
        // Réciprocité : T = Ft·r redonne le couple d'origine.
        assert_relative_eq!(ft * 0.05, 35.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "au moins une dent")]
    fn zero_sun_teeth_panics() {
        planetary_torque_ring_from_sun(10.0, 50, 0);
    }
}
