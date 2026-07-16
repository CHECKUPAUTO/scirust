//! Lixiviation (extraction solide-liquide) — répartition du soluté entre le
//! débordement et le sous-débit à l'équilibre, solution entraînée avec le solide
//! inerte, rendement d'étage et taux de récupération d'une cascade idéale à
//! contre-courant.
//!
//! ```text
//! fraction de soluté au débordement   f_o = L_o / (L_o + L_u)          [-]
//! solution entraînée (sous-débit)     L_u = m_s · r_su                 [kg]
//! rendement d'étage                   η   = X_a / X_eq                 [-]
//! récupération, N étages à            R   = 1 − r^(N+1)                [-]
//!   contre-courant
//! ```
//!
//! `L_o` solution (liquide) quittant l'étage au **débordement** [kg ou kg·s⁻¹],
//! `L_u` solution entraînée au **sous-débit** avec le solide [même unité que
//! `L_o`], `f_o` fraction du soluté partant au débordement [sans dimension,
//! 0 ≤ f_o ≤ 1] — le soluté est supposé à **même concentration** dans les deux
//! phases liquides ; `m_s` masse de solide **inerte** (insoluble) [kg ou kg·s⁻¹],
//! `r_su` rapport de solution entraînée par unité de solide inerte [kg solution ·
//! kg solide⁻¹] ; `X_a`/`X_eq` extractions **réelle**/à l'**équilibre** [fractions
//! sans dimension], `η` rendement (efficacité) d'étage [sans dimension] ; `r`
//! rapport de rétention de solution par étage (fraction de soluté conservée dans
//! le sous-débit à chaque lavage) [sans dimension, 0 ≤ r ≤ 1], `N` nombre d'étages
//! [étages], `R` taux de récupération du soluté [sans dimension, 0 ≤ R ≤ 1].
//!
//! **Limite honnête** : ces relations décrivent une lixiviation à **étages
//! idéaux** où la solution du **débordement** et celle du **sous-débit** ont la
//! **MÊME concentration** (équilibre atteint), le solide **inerte** est
//! **insoluble** et la **rétention de solution** par ce solide (rapport `r_su`
//! ou `r`) est **FOURNIE** par l'appelant. Le **rendement d'étage réel** (`η`,
//! efficacité de Murphree ou globale) est lui aussi **FOURNI**. Aucune propriété
//! physique (enthalpies, volatilités, coefficients de partage, constantes
//! cinétiques, diffusivités, solubilités…) n'est **jamais** supposée « par
//! défaut » : elles proviennent de tables, d'essais ou de l'appelant. La forme
//! `R = 1 − r^(N+1)` suppose une **rétention constante** identique à chaque étage.

/// Fraction du soluté partant au débordement `f_o = L_o / (L_o + L_u)` (sans
/// dimension), sous l'hypothèse d'une **même concentration** de soluté dans la
/// solution du débordement et dans celle du sous-débit. La fraction restant au
/// sous-débit vaut `1 − f_o`.
///
/// `solution_in_overflow` (L_o) et `solution_in_underflow` (L_u) quantités (ou
/// débits) de solution, exprimées dans la **même unité cohérente** (kg ou
/// kg·s⁻¹).
///
/// Panique si `solution_in_overflow < 0`, `solution_in_underflow < 0` ou si la
/// solution totale `L_o + L_u <= 0`.
pub fn leach_overflow_solute_fraction(
    solution_in_overflow: f64,
    solution_in_underflow: f64,
) -> f64 {
    assert!(
        solution_in_overflow >= 0.0,
        "L_o ≥ 0 requis (solution au débordement)"
    );
    assert!(
        solution_in_underflow >= 0.0,
        "L_u ≥ 0 requis (solution au sous-débit)"
    );
    let total = solution_in_overflow + solution_in_underflow;
    assert!(
        total > 0.0,
        "L_o + L_u > 0 requis (solution totale non nulle)"
    );
    solution_in_overflow / total
}

/// Solution entraînée avec le solide inerte au sous-débit `L_u = m_s · r_su`
/// (même unité que `m_s`).
///
/// `inert_solid_mass` (m_s) masse (ou débit) de solide **inerte insoluble**
/// [kg ou kg·s⁻¹] ; `solution_to_solid_ratio` (r_su) rapport **FOURNI** de
/// solution retenue par unité de solide [kg solution · kg solide⁻¹].
///
/// Panique si `inert_solid_mass < 0` ou `solution_to_solid_ratio < 0`.
pub fn leach_underflow_solution(inert_solid_mass: f64, solution_to_solid_ratio: f64) -> f64 {
    assert!(
        inert_solid_mass >= 0.0,
        "m_s ≥ 0 requis (masse de solide inerte)"
    );
    assert!(
        solution_to_solid_ratio >= 0.0,
        "r_su ≥ 0 requis (solution retenue par unité de solide)"
    );
    inert_solid_mass * solution_to_solid_ratio
}

/// Rendement (efficacité) d'étage `η = X_a / X_eq` (sans dimension), rapport de
/// l'extraction **réelle** à l'extraction à l'**équilibre**.
///
/// `actual_extraction` (X_a) extraction réellement obtenue et
/// `equilibrium_extraction` (X_eq) extraction à l'équilibre théorique,
/// exprimées dans la **même grandeur** (fractions cohérentes).
///
/// Panique si `actual_extraction < 0` ou si `equilibrium_extraction <= 0`.
pub fn leach_stage_efficiency(actual_extraction: f64, equilibrium_extraction: f64) -> f64 {
    assert!(
        actual_extraction >= 0.0,
        "X_a ≥ 0 requis (extraction réelle)"
    );
    assert!(
        equilibrium_extraction > 0.0,
        "X_eq > 0 requis (extraction à l'équilibre)"
    );
    actual_extraction / equilibrium_extraction
}

/// Taux de récupération du soluté d'une cascade idéale à **contre-courant**
/// `R = 1 − r^(N+1)` (sans dimension), avec `r` le rapport de **rétention** de
/// solution par étage (fraction de soluté conservée au sous-débit à chaque
/// lavage). Pour `r < 1`, `R → 1` quand `N → ∞` ; à `N = 0` (lavage unique),
/// `R = 1 − r`.
///
/// `underflow_solution_ratio` (r) rapport de rétention par étage [sans
/// dimension, 0 ≤ r ≤ 1] ; `stages` (N) nombre d'étages de lavage.
///
/// Panique si `underflow_solution_ratio < 0` ou `underflow_solution_ratio > 1`.
pub fn leach_countercurrent_solute_recovery(underflow_solution_ratio: f64, stages: u32) -> f64 {
    assert!(
        underflow_solution_ratio >= 0.0,
        "r ≥ 0 requis (rapport de rétention par étage)"
    );
    assert!(
        underflow_solution_ratio <= 1.0,
        "r ≤ 1 requis (rétention fractionnaire ≤ 1)"
    );
    1.0 - underflow_solution_ratio.powi(stages as i32 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn overflow_fraction_and_complement_sum_to_one() {
        // L_o = 8, L_u = 2 ⇒ f_o = 8/10 = 0.8, fraction au sous-débit = 0.2.
        let f_o = leach_overflow_solute_fraction(8.0_f64, 2.0_f64);
        assert_relative_eq!(f_o, 0.8, max_relative = 1e-12);
        // Réciprocité : fractions débordement + sous-débit = 1.
        let f_u = leach_overflow_solute_fraction(2.0_f64, 8.0_f64);
        assert_relative_eq!(f_o + f_u, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn overflow_fraction_limit_no_underflow() {
        // Sans solution au sous-débit, tout le soluté part au débordement : f_o = 1.
        assert_relative_eq!(
            leach_overflow_solute_fraction(5.0_f64, 0.0_f64),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn underflow_solution_is_proportional() {
        // m_s = 100, r_su = 0.5 ⇒ L_u = 50.
        assert_relative_eq!(
            leach_underflow_solution(100.0_f64, 0.5_f64),
            50.0,
            max_relative = 1e-12
        );
        // Proportionnalité : doubler le solide double la solution entraînée.
        let base = leach_underflow_solution(100.0_f64, 0.5_f64);
        let doubled = leach_underflow_solution(200.0_f64, 0.5_f64);
        assert_relative_eq!(doubled, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn stage_efficiency_realistic_and_unity() {
        // X_a = 0.72, X_eq = 0.90 ⇒ η = 0.8.
        assert_relative_eq!(
            leach_stage_efficiency(0.72_f64, 0.90_f64),
            0.8,
            max_relative = 1e-12
        );
        // Étage idéal atteignant l'équilibre : η = 1.
        assert_relative_eq!(
            leach_stage_efficiency(0.65_f64, 0.65_f64),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn countercurrent_recovery_realistic_case() {
        // r = 0.5, N = 3 ⇒ R = 1 − 0.5^4 = 1 − 0.0625 = 0.9375.
        assert_relative_eq!(
            leach_countercurrent_solute_recovery(0.5_f64, 3),
            0.9375,
            max_relative = 1e-12
        );
        // r = 0 (aucune rétention) ⇒ récupération totale R = 1.
        assert_relative_eq!(
            leach_countercurrent_solute_recovery(0.0_f64, 5),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn single_stage_recovery_matches_overflow_fraction() {
        // À N = 0, R = 1 − r ; avec r la fraction retenue au sous-débit,
        // 1 − r est exactement la fraction partant au débordement.
        // L_o = 8, L_u = 2 ⇒ r = 2/10 = 0.2, R(N=0) = 0.8 = f_o.
        let r = 0.2_f64;
        let recovery = leach_countercurrent_solute_recovery(r, 0);
        let f_o = leach_overflow_solute_fraction(8.0_f64, 2.0_f64);
        assert_relative_eq!(recovery, f_o, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "L_o + L_u > 0 requis")]
    fn overflow_fraction_panics_on_zero_solution() {
        // Aucune solution présente ⇒ fraction indéfinie (0/0) ⇒ panique.
        let _ = leach_overflow_solute_fraction(0.0_f64, 0.0_f64);
    }
}
