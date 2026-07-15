//! Étanchéité par **joint labyrinthe** — fuite de gaz par détente successive sur
//! `n` dents (modèle de **Martin**, gaz parfait).
//!
//! ```text
//! fuite massique (Martin)   ṁ = A·√( (p_up² − p_down²)·ρ_up / (n·p_up) )
//! facteur de report         Kc = √( n / (n − 1) )            (n > 1)
//! chute par dent            Δp = (p_up − p_down) / n
//! ```
//!
//! `A` aire de la section de jeu (fente annulaire) (m²), `p_up` pression amont
//! (Pa), `p_down` pression aval (Pa), `ρ_up` masse volumique du gaz amont
//! (kg·m⁻³), `n` nombre de dents (sans dimension), `ṁ` débit massique de fuite
//! (kg/s), `Kc` facteur de report cinétique (sans dimension), `Δp` chute de
//! pression moyenne par dent (Pa).
//!
//! **Convention** : SI cohérent (m, Pa, kg·m⁻³ → kg/s). **Limite honnête** :
//! joint labyrinthe à `n` dents avec détente supposée **répartie uniformément**
//! (formule de Martin, gaz parfait). L'aire de jeu `A` et la masse volumique
//! amont `ρ_up` sont **fournies par l'appelant** ; le résultat est une
//! **estimation** — le report cinétique (énergie non totalement dissipée entre
//! dents) et la géométrie réelle des dents (rayon de bec, pas, angle) modulent la
//! fuite réelle. Aucune constante matériau, fluide ou procédé n'est supposée par
//! défaut.

/// Débit massique de fuite (formule de Martin)
/// `ṁ = A·√((p_up² − p_down²)·ρ_up / (n·p_up))` (kg/s).
///
/// Panique si `clearance_area < 0`, `upstream_pressure <= 0`,
/// `downstream_pressure < 0`, `downstream_pressure > upstream_pressure`,
/// `upstream_density <= 0` ou `number_of_teeth <= 0`.
pub fn labyrinth_leakage_flow(
    clearance_area: f64,
    upstream_pressure: f64,
    downstream_pressure: f64,
    upstream_density: f64,
    number_of_teeth: f64,
) -> f64 {
    assert!(
        clearance_area >= 0.0,
        "l'aire de jeu doit être positive ou nulle"
    );
    assert!(
        upstream_pressure > 0.0,
        "la pression amont doit être strictement positive"
    );
    assert!(
        downstream_pressure >= 0.0,
        "la pression aval ne peut être négative"
    );
    assert!(
        downstream_pressure <= upstream_pressure,
        "la pression aval ne peut dépasser la pression amont"
    );
    assert!(
        upstream_density > 0.0,
        "la masse volumique amont doit être strictement positive"
    );
    assert!(
        number_of_teeth > 0.0,
        "le nombre de dents doit être strictement positif"
    );
    let squared_difference =
        upstream_pressure * upstream_pressure - downstream_pressure * downstream_pressure;
    clearance_area
        * (squared_difference * upstream_density / (number_of_teeth * upstream_pressure)).sqrt()
}

/// Facteur de report cinétique `Kc = √(n/(n−1))` (sans dimension), `n > 1`.
///
/// Panique si `number_of_teeth <= 1`.
pub fn labyrinth_carry_over_factor(number_of_teeth: f64) -> f64 {
    assert!(
        number_of_teeth > 1.0,
        "le nombre de dents doit être strictement supérieur à 1"
    );
    (number_of_teeth / (number_of_teeth - 1.0)).sqrt()
}

/// Chute de pression moyenne par dent `Δp = (p_up − p_down)/n` (Pa).
///
/// Panique si `downstream_pressure > upstream_pressure` ou `number_of_teeth <= 0`.
pub fn labyrinth_pressure_drop_per_tooth(
    upstream_pressure: f64,
    downstream_pressure: f64,
    number_of_teeth: f64,
) -> f64 {
    assert!(
        downstream_pressure <= upstream_pressure,
        "la pression aval ne peut dépasser la pression amont"
    );
    assert!(
        number_of_teeth > 0.0,
        "le nombre de dents doit être strictement positif"
    );
    (upstream_pressure - downstream_pressure) / number_of_teeth
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn leakage_reference_case() {
        // A = 1e-4 m², p_up = 5e5 Pa, p_down = 3e5 Pa, ρ = 5 kg/m³, n = 10.
        // p_up² − p_down² = 2,5e11 − 9e10 = 1,6e11.
        // (1,6e11·5)/(10·5e5) = 8e11/5e6 = 1,6e5 ; √ = 400 ; ·1e-4 = 0,04 kg/s.
        let flow = labyrinth_leakage_flow(1.0e-4, 5.0e5, 3.0e5, 5.0, 10.0);
        assert_relative_eq!(flow, 0.04, epsilon = 1e-12);
    }

    #[test]
    fn leakage_zero_without_pressure_difference() {
        // p_up = p_down → p_up² − p_down² = 0 → aucune fuite.
        let flow = labyrinth_leakage_flow(2.0e-4, 4.0e5, 4.0e5, 6.0, 8.0);
        assert_relative_eq!(flow, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn leakage_linear_in_area() {
        // ṁ ∝ A à conditions fixées : doubler l'aire double la fuite.
        let base = labyrinth_leakage_flow(1.0e-4, 6.0e5, 2.0e5, 4.0, 12.0);
        let double = labyrinth_leakage_flow(2.0e-4, 6.0e5, 2.0e5, 4.0, 12.0);
        assert_relative_eq!(double, 2.0 * base, epsilon = 1e-12);
    }

    #[test]
    fn leakage_scales_with_inverse_sqrt_teeth() {
        // ṁ ∝ 1/√n : quadrupler le nombre de dents divise la fuite par 2.
        let few = labyrinth_leakage_flow(1.0e-4, 5.0e5, 1.0e5, 3.0, 5.0);
        let many = labyrinth_leakage_flow(1.0e-4, 5.0e5, 1.0e5, 3.0, 20.0);
        assert_relative_eq!(many, few / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn carry_over_factor_reference_and_limit() {
        // n = 2 → Kc = √(2/1) = √2.
        assert_relative_eq!(
            labyrinth_carry_over_factor(2.0),
            2.0_f64.sqrt(),
            epsilon = 1e-12
        );
        // n = 5 → Kc = √(5/4).
        assert_relative_eq!(
            labyrinth_carry_over_factor(5.0),
            (5.0_f64 / 4.0).sqrt(),
            epsilon = 1e-12
        );
        // Kc décroît vers 1 quand n augmente (report négligeable).
        assert!(labyrinth_carry_over_factor(100.0) < labyrinth_carry_over_factor(5.0));
        assert!(labyrinth_carry_over_factor(100.0) > 1.0);
    }

    #[test]
    fn pressure_drop_sums_back_to_total() {
        // n·Δp doit reconstituer la détente totale p_up − p_down.
        let p_up = 7.0e5;
        let p_down = 1.0e5;
        let n = 12.0;
        let drop = labyrinth_pressure_drop_per_tooth(p_up, p_down, n);
        assert_relative_eq!(n * drop, p_up - p_down, epsilon = 1e-9);
        // Cas chiffré direct : (7e5 − 1e5)/12 = 5e4 Pa.
        assert_relative_eq!(drop, 5.0e4, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "strictement supérieur à 1")]
    fn carry_over_single_tooth_panics() {
        labyrinth_carry_over_factor(1.0);
    }
}
