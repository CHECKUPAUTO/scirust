//! Choc direct central (1D) entre deux corps rigides — coefficient de
//! restitution, vitesses après choc et énergie dissipée.
//!
//! ```text
//! coefficient        e  = -(v₁' − v₂') / (v₁ − v₂)   (rapport séparation/rapprochement)
//! quantité de mvt    m₁·v₁ + m₂·v₂ = m₁·v₁' + m₂·v₂'
//! vitesse finale 1   v₁' = [(m₁ − e·m₂)·v₁ + (1 + e)·m₂·v₂] / (m₁ + m₂)
//! vitesse finale 2   v₂' = [(m₂ − e·m₁)·v₂ + (1 + e)·m₁·v₁] / (m₁ + m₂)
//! énergie dissipée   ΔE = ½ · (m₁·m₂ / (m₁ + m₂)) · (1 − e²) · (v₁ − v₂)²
//! ```
//!
//! `m₁`, `m₂` masses des deux corps (kg), `v₁`, `v₂` vitesses **avant** choc
//! (m/s, algébriques sur l'axe du choc), `v₁'`, `v₂'` vitesses **après** choc
//! (m/s), `e` coefficient de restitution (sans dimension, ∈ [0, 1]), `ΔE`
//! énergie cinétique dissipée (J).
//!
//! **Convention** : SI cohérent, axe unique orienté ; vitesses algébriques.
//! **Limite honnête** : choc **direct central** (1D), corps **rigides**,
//! frottement et rotation **négligés** ; le coefficient `e` est **fourni par
//! l'appelant** (1 = élastique, 0 = parfaitement plastique) — aucune valeur de
//! matériau, de procédé ou de constante physique n'est supposée par défaut.

/// Coefficient de restitution `e = -v_after / v_before` (sans dimension).
///
/// `relative_velocity_before = v₁ − v₂` est la vitesse relative de
/// rapprochement (avant choc) et `relative_velocity_after = v₁' − v₂'` la
/// vitesse relative d'éloignement (après choc).
///
/// Panique si `relative_velocity_before == 0` (pas de rapprochement défini).
pub fn restitution_coefficient(relative_velocity_before: f64, relative_velocity_after: f64) -> f64 {
    assert!(
        relative_velocity_before != 0.0,
        "la vitesse relative de rapprochement doit être non nulle"
    );
    -relative_velocity_after / relative_velocity_before
}

/// Vitesse `v₁'` du corps 1 après choc direct (m/s).
///
/// `v₁' = [(m₁ − e·m₂)·v₁ + (1 + e)·m₂·v₂] / (m₁ + m₂)`, issue de la
/// conservation de la quantité de mouvement et du coefficient `e`.
///
/// Panique si `mass1 <= 0`, `mass2 <= 0`, ou `e` hors de `[0, 1]`.
pub fn restitution_final_velocity_1(
    mass1: f64,
    mass2: f64,
    velocity1: f64,
    velocity2: f64,
    e: f64,
) -> f64 {
    assert!(
        mass1 > 0.0,
        "la masse du corps 1 doit être strictement positive"
    );
    assert!(
        mass2 > 0.0,
        "la masse du corps 2 doit être strictement positive"
    );
    assert!(
        (0.0..=1.0).contains(&e),
        "le coefficient de restitution doit appartenir à [0, 1]"
    );
    ((mass1 - e * mass2) * velocity1 + (1.0 + e) * mass2 * velocity2) / (mass1 + mass2)
}

/// Vitesse `v₂'` du corps 2 après choc direct (m/s).
///
/// `v₂' = [(m₂ − e·m₁)·v₂ + (1 + e)·m₁·v₁] / (m₁ + m₂)`, issue de la
/// conservation de la quantité de mouvement et du coefficient `e`.
///
/// Panique si `mass1 <= 0`, `mass2 <= 0`, ou `e` hors de `[0, 1]`.
pub fn restitution_final_velocity_2(
    mass1: f64,
    mass2: f64,
    velocity1: f64,
    velocity2: f64,
    e: f64,
) -> f64 {
    assert!(
        mass1 > 0.0,
        "la masse du corps 1 doit être strictement positive"
    );
    assert!(
        mass2 > 0.0,
        "la masse du corps 2 doit être strictement positive"
    );
    assert!(
        (0.0..=1.0).contains(&e),
        "le coefficient de restitution doit appartenir à [0, 1]"
    );
    ((mass2 - e * mass1) * velocity2 + (1.0 + e) * mass1 * velocity1) / (mass1 + mass2)
}

/// Énergie cinétique dissipée par le choc direct `ΔE` (J).
///
/// `ΔE = ½ · (m₁·m₂ / (m₁ + m₂)) · (1 − e²) · (v₁ − v₂)²` ; nulle pour un choc
/// élastique (`e = 1`), maximale pour un choc parfaitement plastique (`e = 0`).
///
/// Panique si `mass1 <= 0`, `mass2 <= 0`, ou `e` hors de `[0, 1]`.
pub fn restitution_energy_loss(
    mass1: f64,
    mass2: f64,
    velocity1: f64,
    velocity2: f64,
    e: f64,
) -> f64 {
    assert!(
        mass1 > 0.0,
        "la masse du corps 1 doit être strictement positive"
    );
    assert!(
        mass2 > 0.0,
        "la masse du corps 2 doit être strictement positive"
    );
    assert!(
        (0.0..=1.0).contains(&e),
        "le coefficient de restitution doit appartenir à [0, 1]"
    );
    let reduced_mass = mass1 * mass2 / (mass1 + mass2);
    let relative_velocity = velocity1 - velocity2;
    0.5 * reduced_mass * (1.0 - e * e) * relative_velocity * relative_velocity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Cas de référence chiffré : m₁=2 kg, m₂=3 kg, v₁=4 m/s, v₂=-1 m/s, e=0,5.
    // v₁' = ((2-0,5·3)·4 + 1,5·3·(-1))/5 = (0,5·4 - 4,5)/5 = (2-4,5)/5 = -0,5 m/s.
    // v₂' = ((3-0,5·2)·(-1) + 1,5·2·4)/5 = (-2 + 12)/5 = 2,0 m/s.

    #[test]
    fn final_velocities_realistic_case() {
        let v1p = restitution_final_velocity_1(2.0, 3.0, 4.0, -1.0, 0.5);
        let v2p = restitution_final_velocity_2(2.0, 3.0, 4.0, -1.0, 0.5);
        assert_relative_eq!(v1p, -0.5, max_relative = 1e-12);
        assert_relative_eq!(v2p, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn momentum_is_conserved() {
        // m₁·v₁ + m₂·v₂ = m₁·v₁' + m₂·v₂' quelles que soient les valeurs.
        let (m1, m2, v1, v2, e) = (2.0, 3.0, 4.0, -1.0, 0.5);
        let v1p = restitution_final_velocity_1(m1, m2, v1, v2, e);
        let v2p = restitution_final_velocity_2(m1, m2, v1, v2, e);
        assert_relative_eq!(m1 * v1 + m2 * v2, m1 * v1p + m2 * v2p, max_relative = 1e-12);
    }

    #[test]
    fn coefficient_reproduces_separation_ratio() {
        // Réciprocité : les vitesses finales redonnent e via -(v₁'-v₂')/(v₁-v₂).
        let (m1, m2, v1, v2, e) = (2.0, 3.0, 4.0, -1.0, 0.5);
        let v1p = restitution_final_velocity_1(m1, m2, v1, v2, e);
        let v2p = restitution_final_velocity_2(m1, m2, v1, v2, e);
        let recovered = restitution_coefficient(v1 - v2, v1p - v2p);
        assert_relative_eq!(recovered, e, max_relative = 1e-12);
    }

    #[test]
    fn energy_loss_matches_kinetic_balance() {
        // ΔE = KE_avant - KE_après pour le cas de référence.
        // KE_avant = 0,5·2·4² + 0,5·3·1² = 16 + 1,5 = 17,5 J.
        // KE_après = 0,5·2·0,5² + 0,5·3·2² = 0,25 + 6 = 6,25 J → ΔE = 11,25 J.
        let (m1, m2, v1, v2, e) = (2.0, 3.0, 4.0, -1.0, 0.5);
        let loss = restitution_energy_loss(m1, m2, v1, v2, e);
        assert_relative_eq!(loss, 11.25, max_relative = 1e-12);

        let v1p = restitution_final_velocity_1(m1, m2, v1, v2, e);
        let v2p = restitution_final_velocity_2(m1, m2, v1, v2, e);
        let ke_before = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2;
        let ke_after = 0.5 * m1 * v1p * v1p + 0.5 * m2 * v2p * v2p;
        assert_relative_eq!(loss, ke_before - ke_after, max_relative = 1e-12);
    }

    #[test]
    fn elastic_collision_conserves_energy() {
        // Choc élastique (e=1) : énergie dissipée nulle.
        let loss = restitution_energy_loss(2.0, 3.0, 4.0, -1.0, 1.0);
        assert_relative_eq!(loss, 0.0, epsilon = 1e-18);
    }

    #[test]
    fn plastic_collision_gives_common_velocity() {
        // Choc parfaitement plastique (e=0) : v₁' = v₂' = quantité de mvt / masse totale.
        let (m1, m2, v1, v2) = (2.0, 3.0, 4.0, -1.0);
        let v1p = restitution_final_velocity_1(m1, m2, v1, v2, 0.0);
        let v2p = restitution_final_velocity_2(m1, m2, v1, v2, 0.0);
        let common = (m1 * v1 + m2 * v2) / (m1 + m2);
        assert_relative_eq!(v1p, common, max_relative = 1e-12);
        assert_relative_eq!(v2p, common, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le coefficient de restitution doit appartenir à [0, 1]")]
    fn coefficient_above_one_panics() {
        restitution_final_velocity_1(2.0, 3.0, 4.0, -1.0, 1.5);
    }
}
