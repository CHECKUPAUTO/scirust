//! **Embrayage centrifuge à patins** — force centrifuge d'un patin, force nette
//! contre le ressort de rappel, régime d'engagement et couple transmis.
//!
//! ```text
//! force centrifuge d'un patin   F  = m·r·ω²
//! force nette pressante         Fn = m·r·ω² − Fs        (nulle si Fn < 0)
//! régime d'engagement           ωe = √( Fs / (m·r) )    (F = Fs)
//! couple transmis               C  = μ·Fn·R·n
//! ```
//!
//! `m` masse d'un patin (kg), `r` rayon du centre de gravité du patin (m), `ω`
//! vitesse angulaire de rotation (rad/s), `F` force centrifuge d'un patin (N),
//! `Fs` force du ressort de rappel maintenant le patin rétracté (N), `Fn` force
//! nette pressant le patin contre le tambour (N), `ωe` régime d'engagement
//! (rad/s), `μ` coefficient de frottement patin/tambour (sans dimension), `R`
//! rayon intérieur du tambour (m), `n` nombre de patins, `C` couple transmis
//! (N·m).
//!
//! **Convention** : SI. **Limite honnête** : patins **identiques** régulièrement
//! répartis, ressort de rappel `Fs`, coefficient de frottement `μ` et rayons
//! **fournis par l'appelant** (aucune valeur matériau/procédé inventée par
//! défaut) ; régime **permanent** (ω constant). En dessous du régime
//! d'engagement `ωe` la force nette est **nulle** et le couple transmis est
//! **nul** (le patin reste rétracté). Le modèle ignore l'usure, la dépendance de
//! `μ` à la vitesse/température et la flexibilité des patins. Voir
//! [`crate::clutch_engagement`] (synchronisation en glissement).

/// Force centrifuge d'un patin `F = m·r·ω²`.
///
/// Force radiale dirigée vers le tambour engendrée par la rotation d'un patin de
/// masse `m` dont le centre de gravité est au rayon `r`.
///
/// Panique si `shoe_mass < 0`, `center_of_gravity_radius < 0` ou
/// `angular_speed < 0`.
pub fn centrifugal_clutch_shoe_force(
    shoe_mass: f64,
    center_of_gravity_radius: f64,
    angular_speed: f64,
) -> f64 {
    assert!(shoe_mass >= 0.0, "la masse du patin m doit être positive");
    assert!(
        center_of_gravity_radius >= 0.0,
        "le rayon du centre de gravité r doit être positif"
    );
    assert!(
        angular_speed >= 0.0,
        "la vitesse angulaire ω doit être positive"
    );
    shoe_mass * center_of_gravity_radius * angular_speed.powi(2)
}

/// Force nette pressant un patin `Fn = max(0, m·r·ω² − Fs)`.
///
/// Excédent de la force centrifuge sur la force du ressort de rappel `Fs` ;
/// **nulle** en dessous du régime d'engagement (le patin reste rétracté et ne
/// transmet aucun couple).
///
/// Panique si `shoe_mass < 0`, `cg_radius < 0`, `angular_speed < 0` ou
/// `spring_force < 0`.
pub fn centrifugal_clutch_net_force(
    shoe_mass: f64,
    cg_radius: f64,
    angular_speed: f64,
    spring_force: f64,
) -> f64 {
    assert!(
        spring_force >= 0.0,
        "la force du ressort Fs doit être positive"
    );
    let centrifugal = centrifugal_clutch_shoe_force(shoe_mass, cg_radius, angular_speed);
    (centrifugal - spring_force).max(0.0)
}

/// Régime d'engagement `ωe = √( Fs / (m·r) )`.
///
/// Vitesse angulaire à laquelle la force centrifuge d'un patin équilibre
/// exactement la force du ressort `Fs` ; au-delà le patin vient au contact.
///
/// Panique si `spring_force < 0`, `shoe_mass <= 0` ou `cg_radius <= 0`.
pub fn centrifugal_clutch_engagement_speed(
    spring_force: f64,
    shoe_mass: f64,
    cg_radius: f64,
) -> f64 {
    assert!(
        spring_force >= 0.0,
        "la force du ressort Fs doit être positive"
    );
    assert!(
        shoe_mass > 0.0,
        "la masse du patin m doit être strictement positive"
    );
    assert!(
        cg_radius > 0.0,
        "le rayon du centre de gravité r doit être strictement positif"
    );
    (spring_force / (shoe_mass * cg_radius)).sqrt()
}

/// Couple transmis `C = μ·Fn·R·n`.
///
/// Couple total transmis par `n` patins identiques pressés avec la force nette
/// `Fn` sur le tambour de rayon `R`, coefficient de frottement `μ`.
///
/// Panique si `friction_coefficient < 0`, `net_force < 0`, `drum_radius < 0` ou
/// `shoe_count < 0`.
pub fn centrifugal_clutch_torque(
    friction_coefficient: f64,
    net_force: f64,
    drum_radius: f64,
    shoe_count: f64,
) -> f64 {
    assert!(
        friction_coefficient >= 0.0,
        "le coefficient de frottement μ doit être positif"
    );
    assert!(net_force >= 0.0, "la force nette Fn doit être positive");
    assert!(
        drum_radius >= 0.0,
        "le rayon du tambour R doit être positif"
    );
    assert!(shoe_count >= 0.0, "le nombre de patins n doit être positif");
    friction_coefficient * net_force * drum_radius * shoe_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn shoe_force_scales_quadratically_with_speed() {
        // F ∝ ω² : doubler ω quadruple la force centrifuge.
        let f1 = centrifugal_clutch_shoe_force(0.3, 0.05, 100.0);
        let f2 = centrifugal_clutch_shoe_force(0.3, 0.05, 200.0);
        assert_relative_eq!(f2, 4.0 * f1, epsilon = 1e-9);
    }

    #[test]
    fn shoe_force_realistic_case() {
        // m=0,3 kg, r=0,05 m, ω=200 rad/s → F = 0,3·0,05·200² = 600 N.
        let f = centrifugal_clutch_shoe_force(0.3, 0.05, 200.0);
        assert_relative_eq!(f, 600.0, epsilon = 1e-9);
    }

    #[test]
    fn shoe_force_equals_spring_at_engagement_speed() {
        // À ω = ωe, la force centrifuge équilibre exactement le ressort (Fn = 0).
        let spring = 150.0_f64;
        let (m, r) = (0.3_f64, 0.05_f64);
        let we = centrifugal_clutch_engagement_speed(spring, m, r);
        let f = centrifugal_clutch_shoe_force(m, r, we);
        assert_relative_eq!(f, spring, epsilon = 1e-9);
        assert_relative_eq!(
            centrifugal_clutch_net_force(m, r, we, spring),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn net_force_is_zero_below_engagement() {
        // En dessous de ωe la force nette est bridée à zéro (patin rétracté).
        let spring = 150.0_f64;
        let (m, r) = (0.3_f64, 0.05_f64);
        let we = centrifugal_clutch_engagement_speed(spring, m, r);
        let fn_below = centrifugal_clutch_net_force(m, r, 0.5 * we, spring);
        assert_relative_eq!(fn_below, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn engagement_speed_realistic_case() {
        // Fs=150 N, m=0,3 kg, r=0,05 m → ωe = √(150/0,015) = √10000 = 100 rad/s.
        let we = centrifugal_clutch_engagement_speed(150.0, 0.3, 0.05);
        assert_relative_eq!(we, 100.0, epsilon = 1e-9);
    }

    #[test]
    fn torque_realistic_case_and_proportionality() {
        // ω=200 rad/s : F=600 N, Fs=150 N → Fn=450 N.
        // C = μ·Fn·R·n = 0,35·450·0,06·4 = 37,8 N·m.
        let fnet = centrifugal_clutch_net_force(0.3, 0.05, 200.0, 150.0);
        assert_relative_eq!(fnet, 450.0, epsilon = 1e-9);
        let c = centrifugal_clutch_torque(0.35, fnet, 0.06, 4.0);
        assert_relative_eq!(c, 37.8, epsilon = 1e-9);
        // C ∝ n : doubler le nombre de patins double le couple.
        let c_double = centrifugal_clutch_torque(0.35, fnet, 0.06, 8.0);
        assert_relative_eq!(c_double, 2.0 * c, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la masse du patin m doit être strictement positive")]
    fn zero_mass_engagement_speed_panics() {
        centrifugal_clutch_engagement_speed(150.0, 0.0, 0.05);
    }
}
