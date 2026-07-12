//! Mise en forme — **laminage à plat** : réduction, longueur de contact, effort
//! de laminage et couple d'entraînement.
//!
//! ```text
//! réduction (draft)   Δh = h0 − hf
//! longueur de contact L = √(R·Δh)
//! effort de laminage  F = w·L·Ȳ
//! couple (par cylindre) C = 0,5·F·L
//! réduction max        Δh_max = µ²·R
//! ```
//!
//! `h0`/`hf` épaisseurs entrée/sortie (m), `R` rayon des cylindres (m), `w`
//! largeur de la bande (m), `Ȳ` contrainte d'écoulement **moyenne** en
//! déformation plane (Pa), `L` longueur de l'arc de contact (m), `µ` frottement.
//! La réduction maximale par passe est fixée par la condition de mordage
//! `Δh_max = µ²·R`.
//!
//! **Convention** : SI cohérent. **Limite honnête** : laminage à plat, largeur
//! constante (déformation plane), effort par la pression moyenne `Ȳ·L·w`
//! (modèle simplifié — les modèles de Bland-Ford/Orowan raffinent la
//! distribution). `Ȳ` et `µ` sont fournis par l'appelant.

/// Réduction d'épaisseur (draft) `Δh = h0 − hf` (m).
pub fn draft(initial_thickness: f64, final_thickness: f64) -> f64 {
    initial_thickness - final_thickness
}

/// Longueur de l'arc de contact `L = √(R·Δh)` (m).
///
/// Panique si `roll_radius·draft < 0`.
pub fn contact_length(roll_radius: f64, draft: f64) -> f64 {
    assert!(roll_radius * draft >= 0.0, "R·Δh doit être positif");
    (roll_radius * draft).sqrt()
}

/// Effort de laminage `F = w·L·Ȳ` (N).
pub fn roll_force(width: f64, contact_length: f64, avg_flow_stress: f64) -> f64 {
    width * contact_length * avg_flow_stress
}

/// Couple par cylindre `C = 0,5·F·L` (N·m).
pub fn roll_torque(force: f64, contact_length: f64) -> f64 {
    0.5 * force * contact_length
}

/// Réduction maximale par passe (condition de mordage) `Δh_max = µ²·R` (m).
pub fn max_draft(friction: f64, roll_radius: f64) -> f64 {
    friction * friction * roll_radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn draft_and_contact_length() {
        // h0=10, hf=8 mm → Δh=2 mm ; R=250 mm → L = √(0,25·0,002) = √5e-4 ≈ 22,4 mm.
        assert_relative_eq!(draft(0.010, 0.008), 0.002, epsilon = 1e-9);
        let l = contact_length(0.25, 0.002);
        assert_relative_eq!(l, (0.25f64 * 0.002).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn force_and_torque() {
        // w=200 mm, L≈22,4 mm, Ȳ=250 MPa.
        let l = contact_length(0.25, 0.002);
        let f = roll_force(0.2, l, 250e6);
        assert_relative_eq!(f, 0.2 * l * 250e6, epsilon = 1.0);
        // couple = 0,5·F·L.
        assert_relative_eq!(roll_torque(f, l), 0.5 * f * l, epsilon = 1e-3);
    }

    #[test]
    fn bite_condition_limits_draft() {
        // µ=0,2, R=250 mm → Δh_max = 0,04·0,25 = 0,01 m = 10 mm.
        assert_relative_eq!(max_draft(0.2, 0.25), 0.01, epsilon = 1e-12);
        // Une réduction plus fine que Δh_max est acceptable.
        assert!(draft(0.010, 0.008) < max_draft(0.2, 0.25));
    }

    #[test]
    fn larger_reduction_lengthens_contact() {
        // Plus la réduction est grande, plus l'arc de contact est long.
        assert!(contact_length(0.25, 0.004) > contact_length(0.25, 0.002));
    }

    #[test]
    #[should_panic(expected = "R·Δh")]
    fn negative_product_panics() {
        contact_length(0.25, -0.002);
    }
}
