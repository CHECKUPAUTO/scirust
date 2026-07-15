//! Laminage à plat — **effort et couple** de laminage : longueur d'arc de
//! contact projeté, effort de laminage, couple par cylindre et puissance des
//! deux cylindres.
//!
//! ```text
//! longueur de contact   L = sqrt(R·Δh)          Δh = h0 − hf
//! effort de laminage    F = Ȳ·w·L
//! couple par cylindre   M = F·L/2               (bras de levier à mi-arc)
//! puissance (2 cyl.)    P = 2·M·ω
//! ```
//!
//! `R` rayon du cylindre (m), `Δh` réduction d'épaisseur (draft, m) égale à
//! l'épaisseur d'entrée `h0` moins l'épaisseur de sortie `hf`, `L` longueur d'arc
//! de contact projeté (m), `Ȳ` contrainte d'écoulement moyenne du matériau sur la
//! passe (Pa), `w` largeur de la bande (m), `F` effort de laminage (N), `M` couple
//! sur un cylindre (N·m), `ω` vitesse angulaire du cylindre (rad/s), `P` puissance
//! totale (W). L'effort est l'aire de contact projetée `w·L` multipliée par la
//! contrainte d'écoulement moyenne ; le couple applique cet effort à un bras de
//! levier pris à mi-arc `L/2` ; la puissance somme les deux cylindres entraînés.
//!
//! **Convention** : SI cohérent (m, Pa, N, N·m, rad/s, W). **Limite honnête** :
//! laminage à plat, contrainte d'écoulement moyenne `Ȳ` **fournie par
//! l'appelant** — aucune valeur matériau ni loi d'écrouissage n'est inventée ici ;
//! largeur constante (déformation plane, pas d'élargissement) ; frottement
//! suffisant pour l'entraînement sans glissement ; arc de contact **projeté**
//! approché (`L ≈ sqrt(R·Δh)`, aplatissement élastique des cylindres négligé).

/// Longueur d'arc de contact projeté `L = sqrt(R·Δh)` (m).
///
/// `Δh = h0 − hf` est la réduction d'épaisseur (draft) sur la passe.
///
/// Panique si `roll_radius < 0` ou `draft < 0`.
pub fn rolling_contact_length(roll_radius: f64, draft: f64) -> f64 {
    assert!(roll_radius >= 0.0, "le rayon du cylindre doit être positif");
    assert!(draft >= 0.0, "la réduction d'épaisseur doit être positive");
    (roll_radius * draft).sqrt()
}

/// Effort de laminage `F = Ȳ·w·L` (N).
///
/// Contrainte d'écoulement moyenne `Ȳ` appliquée sur l'aire de contact projetée
/// `w·L`.
///
/// Panique si `average_flow_stress < 0`, `width < 0` ou `contact_length < 0`.
pub fn rolling_force(average_flow_stress: f64, width: f64, contact_length: f64) -> f64 {
    assert!(
        average_flow_stress >= 0.0,
        "la contrainte d'écoulement moyenne doit être positive"
    );
    assert!(width >= 0.0, "la largeur doit être positive");
    assert!(
        contact_length >= 0.0,
        "la longueur de contact doit être positive"
    );
    average_flow_stress * width * contact_length
}

/// Couple sur un cylindre `M = F·L/2` (N·m).
///
/// L'effort de laminage `F` s'applique à un bras de levier pris à mi-arc `L/2`.
///
/// Panique si `rolling_force < 0` ou `contact_length < 0`.
pub fn rolling_torque_per_roll(rolling_force: f64, contact_length: f64) -> f64 {
    assert!(
        rolling_force >= 0.0,
        "l'effort de laminage doit être positif"
    );
    assert!(
        contact_length >= 0.0,
        "la longueur de contact doit être positive"
    );
    rolling_force * contact_length / 2.0
}

/// Puissance totale des deux cylindres `P = 2·M·ω` (W).
///
/// Somme la puissance des deux cylindres entraînés, chacun développant le couple
/// `M` à la vitesse angulaire `ω`.
///
/// Panique si `rolling_torque_per_roll < 0` ou `roll_angular_speed < 0`.
pub fn rolling_power(rolling_torque_per_roll: f64, roll_angular_speed: f64) -> f64 {
    assert!(
        rolling_torque_per_roll >= 0.0,
        "le couple par cylindre doit être positif"
    );
    assert!(
        roll_angular_speed >= 0.0,
        "la vitesse angulaire doit être positive"
    );
    2.0 * rolling_torque_per_roll * roll_angular_speed
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn contact_length_squared_equals_radius_times_draft() {
        // Identité définissante : L² = R·Δh.
        let r = 0.25;
        let draft = 0.005;
        let l = rolling_contact_length(r, draft);
        assert_relative_eq!(l * l, r * draft, epsilon = 1e-15);
    }

    #[test]
    fn force_scales_linearly_with_width() {
        // F ∝ w : doubler la largeur double l'effort.
        let f1 = rolling_force(200e6, 0.30, 0.035);
        let f2 = rolling_force(200e6, 0.60, 0.035);
        assert_relative_eq!(f2 / f1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn torque_is_force_times_half_contact_length() {
        // M = F·(L/2) : le bras de levier est bien la demi-longueur d'arc.
        let l = 0.035_355_339_059_327;
        let f = rolling_force(200e6, 0.30, l);
        let m = rolling_torque_per_roll(f, l);
        assert_relative_eq!(m, f * l / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn realistic_flat_rolling_pass() {
        // Passe : R = 0,25 m, Δh = 5 mm, Ȳ = 200 MPa, w = 0,30 m, ω = 2 rad/s.
        // L  = sqrt(0,25·0,005) = 0,035355339 m
        // F  = 200e6·0,30·L = 2,12132 MN
        // M  = F·L/2 = 200e6·0,30·L²/2 = 200e6·0,30·0,00125/2 = 37 500 N·m
        // P  = 2·M·ω = 2·37500·2 = 150 000 W
        let l = rolling_contact_length(0.25, 0.005);
        let f = rolling_force(200e6, 0.30, l);
        let m = rolling_torque_per_roll(f, l);
        let p = rolling_power(m, 2.0);
        assert_relative_eq!(l, 0.035_355_339_059_327, epsilon = 1e-12);
        assert_relative_eq!(f, 2_121_320.343_559, epsilon = 1e-3);
        assert_relative_eq!(m, 37_500.0, epsilon = 1e-6);
        assert_relative_eq!(p, 150_000.0, epsilon = 1e-6);
    }

    #[test]
    fn power_scales_linearly_with_angular_speed() {
        // P ∝ ω : doubler la vitesse double la puissance.
        let p1 = rolling_power(37_500.0, 2.0);
        let p2 = rolling_power(37_500.0, 4.0);
        assert_relative_eq!(p2 / p1, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "réduction d'épaisseur")]
    fn negative_draft_panics() {
        rolling_contact_length(0.25, -0.001);
    }
}
