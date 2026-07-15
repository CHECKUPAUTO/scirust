//! Rendement de première passe et **RTY** (qualité d'un procédé multi-étapes) :
//! rendement au premier passage, rendement laminé sur toute la chaîne, rendement
//! normalisé par étape et défauts moyens par unité (DPU).
//!
//! ```text
//! rendement 1re passe   FPY = units_passed / units_entering
//! rendement laminé      RTY = Π_{i=1}^{n} Y_i
//! rendement normalisé   Y_norm = RTY^(1/n)
//! défauts par unité     DPU = −ln(RTY)          (approximation de Poisson)
//! ```
//!
//! `FPY` (`first_pass_yield`) fraction d'unités bonnes du premier coup à une
//! étape (sans dimension, dans `[0, 1]`), `units_passed` nombre d'unités passant
//! sans reprise, `units_entering` nombre d'unités entrant dans l'étape, `RTY`
//! (`rolled_throughput_yield`) probabilité qu'une unité traverse les `n` étapes
//! sans aucun défaut (sans dimension), `Y_i` rendement de l'étape `i` (dans
//! `[0, 1]`), `Y_norm` (`normalized_yield`) rendement moyen géométrique par étape,
//! `DPU` (`total_defects_per_unit`) nombre moyen de défauts par unité (comptage,
//! sans unité physique). Toutes ces grandeurs sont des fractions ou des comptes.
//!
//! **Limite honnête** : les étapes sont supposées **indépendantes** (le RTY est le
//! produit des rendements) et le DPU découle de l'**approximation de Poisson**
//! `Y = e^(−DPU)`, valable quand les défauts sont rares et indépendants. Les
//! rendements d'étape `Y_i` (dans `[0, 1]`) et les comptages d'unités sont
//! **fournis** par l'appelant (mesurés sur la ligne) ; ce module n'invente aucune
//! valeur « par défaut » ni aucune constante de procédé.

/// Rendement de première passe `FPY = units_passed / units_entering` : fraction
/// d'unités bonnes du premier coup (sans reprise) à une étape.
///
/// Panique si `units_entering == 0` ou si `units_passed > units_entering`.
pub fn rty_first_pass_yield(units_passed_first_time: u64, units_entering: u64) -> f64 {
    assert!(
        units_entering > 0,
        "le nombre d'unités entrantes doit être strictement positif"
    );
    assert!(
        units_passed_first_time <= units_entering,
        "le nombre d'unités passées ne peut pas dépasser le nombre d'unités entrantes"
    );
    units_passed_first_time as f64 / units_entering as f64
}

/// Rendement laminé de bout en bout `RTY = Π Y_i` : produit des rendements des
/// étapes supposées indépendantes.
///
/// Panique si `yields` est vide ou si un rendement sort de `[0, 1]`.
pub fn rty_rolled_throughput_yield(yields: &[f64]) -> f64 {
    assert!(
        !yields.is_empty(),
        "la chaîne doit comporter au moins une étape"
    );
    assert!(
        yields.iter().all(|&y| (0.0..=1.0).contains(&y)),
        "chaque rendement d'étape doit être dans [0, 1]"
    );
    yields.iter().product()
}

/// Rendement normalisé par étape `Y_norm = RTY^(1/n)` : moyenne géométrique
/// représentant le rendement « équivalent » d'une étape moyenne.
///
/// Panique si `rty` sort de `[0, 1]` ou si `n_steps == 0`.
pub fn rty_normalized_yield(rty: f64, n_steps: u32) -> f64 {
    assert!(
        (0.0..=1.0).contains(&rty),
        "le rendement laminé RTY doit être dans [0, 1]"
    );
    assert!(
        n_steps > 0,
        "le nombre d'étapes doit être strictement positif"
    );
    rty.powf(1.0_f64 / f64::from(n_steps))
}

/// Défauts moyens par unité `DPU = −ln(RTY)`, déduit du rendement laminé via
/// l'approximation de Poisson `RTY = e^(−DPU)`.
///
/// Panique si `rty` sort de `]0, 1]` (un RTY nul donnerait un DPU infini).
pub fn rty_total_defects_per_unit(rty: f64) -> f64 {
    assert!(
        rty > 0.0 && rty <= 1.0,
        "le rendement laminé RTY doit être dans ]0, 1] pour un DPU fini"
    );
    -rty.ln()
}

/// Rendement d'étape reconstruit à partir des défauts par unité
/// `Y = e^(−DPU)` (réciproque de [`rty_total_defects_per_unit`]).
///
/// Panique si `dpu` est négatif.
pub fn rty_yield_from_defects_per_unit(dpu: f64) -> f64 {
    assert!(
        dpu >= 0.0,
        "le nombre de défauts par unité doit être positif ou nul"
    );
    (-dpu).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn first_pass_yield_bounds_and_realistic_case() {
        // Unité toute bonne → FPY = 1 ; cas chiffré 940/1000 = 0.94.
        assert_relative_eq!(rty_first_pass_yield(1000, 1000), 1.0, epsilon = 1e-12);
        assert_relative_eq!(rty_first_pass_yield(940, 1000), 0.94, epsilon = 1e-12);
    }

    #[test]
    fn rolled_yield_is_product_of_steps() {
        // RTY = Π Y_i : 0.99·0.98·0.97 calculé directement.
        let ys = [0.99, 0.98, 0.97];
        let expected = 0.99_f64 * 0.98 * 0.97;
        assert_relative_eq!(
            rty_rolled_throughput_yield(&ys),
            expected,
            max_relative = 1e-12
        );
    }

    #[test]
    fn identical_steps_recover_via_normalized_yield() {
        // Si toutes les étapes valent y, alors RTY = y^n et Y_norm = y (réciprocité).
        let y = 0.95;
        let n = 6_u32;
        let ys = [y; 6];
        let rty = rty_rolled_throughput_yield(&ys);
        assert_relative_eq!(rty, y.powi(n as i32), max_relative = 1e-12);
        assert_relative_eq!(rty_normalized_yield(rty, n), y, max_relative = 1e-12);
    }

    #[test]
    fn dpu_and_yield_are_reciprocal() {
        // Y → DPU → Y doit boucler : e^(−(−ln Y)) = Y.
        let rty = 0.90;
        let dpu = rty_total_defects_per_unit(rty);
        assert_relative_eq!(
            rty_yield_from_defects_per_unit(dpu),
            rty,
            max_relative = 1e-12
        );
        // Cas chiffré : RTY = e^(−1) → DPU = 1 défaut/unité.
        assert_relative_eq!(
            rty_total_defects_per_unit((-1.0_f64).exp()),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn perfect_process_has_zero_defects() {
        // RTY = 1 (aucune perte) → DPU = 0.
        assert_relative_eq!(rty_total_defects_per_unit(1.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(rty_yield_from_defects_per_unit(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn dpu_is_additive_over_independent_steps() {
        // Défauts additifs : DPU(RTY) = Σ DPU(Y_i) puisque ln d'un produit = Σ ln.
        let ys = [0.97, 0.95, 0.92];
        let rty = rty_rolled_throughput_yield(&ys);
        let sum_dpu: f64 = ys.iter().map(|&y| rty_total_defects_per_unit(y)).sum();
        assert_relative_eq!(
            rty_total_defects_per_unit(rty),
            sum_dpu,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "au moins une étape")]
    fn empty_chain_panics() {
        rty_rolled_throughput_yield(&[]);
    }
}
