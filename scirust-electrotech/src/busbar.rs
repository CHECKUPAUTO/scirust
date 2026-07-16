//! **Jeu de barres** — module de dimensionnement d'un jeu de barres rectiligne :
//! résistance en régime continu, densité de courant, force électrodynamique
//! entre deux barres parallèles lors d'un court-circuit et pertes Joule.
//!
//! ```text
//! résistance continue     R  = ρ·L / A
//! densité de courant      J  = I / A
//! force électrodynamique  F  = µ₀·Î²·L / (2·π·d)
//! pertes Joule            P  = I²·R
//! ```
//!
//! `R` résistance en continu (Ω), `ρ` résistivité électrique du matériau (Ω·m),
//! `L` longueur de la barre (m), `A` aire de la section droite (m²), `J` densité
//! de courant (A/m²), `I` courant efficace parcourant la barre (A), `F` force
//! électrodynamique par barre (N), `µ₀` perméabilité du vide (H/m), `Î` courant
//! de crête de court-circuit (A), `d` entraxe entre les deux barres parallèles
//! (m), `P` pertes Joule dissipées (W).
//!
//! **Convention** : SI ; résistivités en Ω·m, longueurs et entraxes en m, aires
//! en m², courants en A, densités en A/m², forces en N, puissances en W,
//! perméabilité en H/m ; l'angle implicite `2·π` du champ magnétique est en
//! **radians**. **Limite honnête** : jeu de barres **rectiligne** en **régime
//! établi** ; la résistivité `ρ` et la géométrie (`L`, `A`, `d`) sont
//! **fournies par l'appelant** (fiches matériau, plans mécaniques). La force
//! électrodynamique entre deux **barres parallèles** — calculée à partir du
//! **courant de crête de court-circuit `Î` fourni** par l'étude réseau —
//! dimensionne les **supports isolants** ; le modèle suppose deux conducteurs
//! rectilignes parallèles infiniment fins (approximation de fil). La **densité
//! de courant admissible** (limite d'échauffement) est **fournie par la norme**
//! selon la ventilation et le matériau — ce module ne fait que **calculer** la
//! densité réelle, il n'invente aucune valeur admissible « par défaut ».

/// Perméabilité magnétique du vide `µ₀` (H/m), constante physique fournie pour
/// le calcul de la force électrodynamique entre barres parallèles.
pub const BUSBAR_MU0: f64 = 1.256_637_061_4e-6;

/// Résistance en régime continu d'une barre `R = ρ·L / A` (Ω).
///
/// Panique si `resistivity < 0`, si `length < 0` ou si
/// `cross_section_area <= 0` (division par zéro ou résistance négative).
pub fn busbar_dc_resistance(resistivity: f64, length: f64, cross_section_area: f64) -> f64 {
    assert!(resistivity >= 0.0, "la résistivité ρ doit être ≥ 0");
    assert!(length >= 0.0, "la longueur L doit être ≥ 0");
    assert!(
        cross_section_area > 0.0,
        "l'aire de section A doit être strictement positive"
    );
    resistivity * length / cross_section_area
}

/// Densité de courant dans la section `J = I / A` (A/m²).
///
/// Panique si `cross_section_area <= 0` (division par zéro).
pub fn busbar_current_density(current: f64, cross_section_area: f64) -> f64 {
    assert!(
        cross_section_area > 0.0,
        "l'aire de section A doit être strictement positive"
    );
    current / cross_section_area
}

/// Force électrodynamique entre deux barres parallèles en court-circuit
/// `F = µ₀·Î²·L / (2·π·d)` (N).
///
/// Panique si `conductor_spacing <= 0` (division par zéro), si `length < 0` ou
/// si `vacuum_permeability < 0` (force négative non physique).
pub fn busbar_short_circuit_force(
    peak_short_circuit_current: f64,
    conductor_spacing: f64,
    length: f64,
    vacuum_permeability: f64,
) -> f64 {
    assert!(
        conductor_spacing > 0.0,
        "l'entraxe d entre barres doit être strictement positif"
    );
    assert!(length >= 0.0, "la longueur L doit être ≥ 0");
    assert!(
        vacuum_permeability >= 0.0,
        "la perméabilité µ₀ doit être ≥ 0"
    );
    vacuum_permeability * peak_short_circuit_current * peak_short_circuit_current * length
        / (2.0 * core::f64::consts::PI * conductor_spacing)
}

/// Pertes Joule dissipées dans la barre `P = I²·R` (W).
///
/// Panique si `resistance < 0` (pertes négatives non physiques).
pub fn busbar_power_loss(current: f64, resistance: f64) -> f64 {
    assert!(resistance >= 0.0, "la résistance R doit être ≥ 0");
    current * current * resistance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dc_resistance_copper_case() {
        // Cas chiffré : cuivre ρ = 1,68e-8 Ω·m, L = 10 m, A = 1e-4 m² (100 mm²).
        //   R = 1,68e-8·10 / 1e-4 = 1,68e-7 / 1e-4 = 1,68e-3 Ω.
        let r = busbar_dc_resistance(1.68e-8, 10.0, 1.0e-4);
        assert_relative_eq!(r, 1.68e-3, epsilon = 1e-9);
    }

    #[test]
    fn dc_resistance_is_proportional_to_length_inverse_area() {
        // Proportionnalités : R ∝ L (doubler L double R) et R ∝ 1/A (doubler A
        // divise R par deux).
        let base = busbar_dc_resistance(2.0e-8, 5.0, 2.0e-4);
        let long = busbar_dc_resistance(2.0e-8, 10.0, 2.0e-4);
        let wide = busbar_dc_resistance(2.0e-8, 5.0, 4.0e-4);
        assert_relative_eq!(long, 2.0 * base, epsilon = 1e-15);
        assert_relative_eq!(wide, base / 2.0, epsilon = 1e-15);
    }

    #[test]
    fn current_density_case_and_scaling() {
        // Cas chiffré : I = 1000 A, A = 1e-4 m² → J = 1000 / 1e-4 = 1e7 A/m².
        assert_relative_eq!(
            busbar_current_density(1000.0, 1.0e-4),
            1.0e7,
            epsilon = 1e-3
        );
        // Proportionnalité : J ∝ 1/A ; doubler la section divise J par deux.
        let j = busbar_current_density(1000.0, 1.0e-4);
        let j2 = busbar_current_density(1000.0, 2.0e-4);
        assert_relative_eq!(j2, j / 2.0, epsilon = 1e-3);
    }

    #[test]
    fn short_circuit_force_reference_case() {
        // Cas chiffré : µ₀ = 1,2566370614e-6 H/m donne µ₀/(2π) = 2e-7 (constante
        // classique). Avec Î = 10000 A, d = 0,1 m, L = 1 m :
        //   F = 2e-7·Î²·L / d = 2e-7·1e8·1 / 0,1 = 20 / 0,1 = 200 N.
        let f = busbar_short_circuit_force(10000.0, 0.1, 1.0, BUSBAR_MU0);
        assert_relative_eq!(f, 200.0, epsilon = 1e-3);
    }

    #[test]
    fn short_circuit_force_scalings() {
        // Proportionnalités : F ∝ Î² (doubler Î quadruple F), F ∝ L (doubler L
        // double F) et F ∝ 1/d (doubler l'entraxe divise F par deux).
        let base = busbar_short_circuit_force(5000.0, 0.2, 1.0, BUSBAR_MU0);
        let dbl_i = busbar_short_circuit_force(10000.0, 0.2, 1.0, BUSBAR_MU0);
        let dbl_l = busbar_short_circuit_force(5000.0, 0.2, 2.0, BUSBAR_MU0);
        let dbl_d = busbar_short_circuit_force(5000.0, 0.4, 1.0, BUSBAR_MU0);
        assert_relative_eq!(dbl_i, 4.0 * base, epsilon = 1e-9);
        assert_relative_eq!(dbl_l, 2.0 * base, epsilon = 1e-9);
        assert_relative_eq!(dbl_d, base / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn power_loss_case_and_quadratic() {
        // Cas chiffré : I = 1000 A, R = 1,68e-3 Ω → P = 1000²·1,68e-3
        //   = 1e6·1,68e-3 = 1680 W.
        assert_relative_eq!(busbar_power_loss(1000.0, 1.68e-3), 1680.0, epsilon = 1e-3);
        // Proportionnalité quadratique : doubler I quadruple les pertes.
        let p = busbar_power_loss(1000.0, 1.68e-3);
        let p2 = busbar_power_loss(2000.0, 1.68e-3);
        assert_relative_eq!(p2, 4.0 * p, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "l'entraxe d entre barres doit être strictement positif")]
    fn zero_spacing_panics() {
        busbar_short_circuit_force(10000.0, 0.0, 1.0, BUSBAR_MU0);
    }
}
