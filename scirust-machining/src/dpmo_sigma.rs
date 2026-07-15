//! Indicateurs **Six Sigma** au niveau défauts : DPMO (défauts par million
//! d'opportunités), rendement première passe associé, défauts par unité (DPU) et
//! rendement de Poisson déduit du DPU.
//!
//! ```text
//! DPMO   = defects · 1_000_000 / (units · opportunities_per_unit)
//! yield  = 1 − DPMO / 1_000_000                 (rendement première passe)
//! DPU    = defects / units                      (défauts par unité)
//! yield  = e^(−DPU)                             (rendement, loi de Poisson)
//! ```
//!
//! `defects` nombre total de défauts observés (comptage), `units` nombre d'unités
//! inspectées (comptage, > 0), `opportunities_per_unit` nombre d'occasions de
//! défaut par unité (comptage, > 0, définit le périmètre), `DPMO`
//! (`dpmo_from_defects`) défauts rapportés à un million d'opportunités (sans
//! dimension), `yield` (`dpmo_yield`, `dpmo_yield_from_dpu`) rendement dans
//! `[0, 1]`, `DPU` (`dpmo_defects_per_unit`) nombre moyen de défauts par unité
//! (comptage, sans unité physique). Toutes ces grandeurs sont des fractions ou
//! des comptes purs.
//!
//! **Limite honnête** : le nombre d'`opportunities_per_unit` (la **définition du
//! périmètre** de comptage des défauts) et les comptages `defects`/`units` sont
//! **fournis** par l'appelant (mesurés sur la ligne) ; ce module n'invente aucune
//! valeur « par défaut ». Le rendement issu du DPU suppose des défauts **rares et
//! indépendants** (approximation de **Poisson** `yield = e^(−DPU)`). Le passage
//! au « niveau sigma » avec décalage conventionnel de 1,5 σ exige une **table de
//! la loi normale** non incluse ici. Distinct de
//! [`crate::rolled_throughput_yield`], qui agrège des rendements d'étapes.

/// Défauts par million d'opportunités
/// `DPMO = defects · 1_000_000 / (units · opportunities_per_unit)`.
///
/// Panique si `units == 0` ou `opportunities_per_unit == 0`.
pub fn dpmo_from_defects(defects: u64, units: u64, opportunities_per_unit: u64) -> f64 {
    assert!(
        units > 0,
        "le nombre d'unités inspectées doit être strictement positif"
    );
    assert!(
        opportunities_per_unit > 0,
        "le nombre d'opportunités de défaut par unité doit être strictement positif"
    );
    defects as f64 * 1_000_000.0_f64 / (units as f64 * opportunities_per_unit as f64)
}

/// Rendement première passe `yield = 1 − DPMO / 1_000_000` déduit du DPMO.
///
/// Panique si `dpmo` sort de `[0, 1_000_000]` (rendement hors de `[0, 1]`).
pub fn dpmo_yield(dpmo: f64) -> f64 {
    assert!(
        (0.0..=1_000_000.0).contains(&dpmo),
        "le DPMO doit être dans [0, 1_000_000] pour un rendement dans [0, 1]"
    );
    1.0_f64 - dpmo / 1_000_000.0_f64
}

/// Défauts par unité `DPU = defects / units` (moyenne de défauts par pièce).
///
/// Panique si `units == 0`.
pub fn dpmo_defects_per_unit(defects: u64, units: u64) -> f64 {
    assert!(
        units > 0,
        "le nombre d'unités inspectées doit être strictement positif"
    );
    defects as f64 / units as f64
}

/// Rendement selon la loi de Poisson `yield = e^(−DPU)` : probabilité qu'une unité
/// ne présente aucun défaut lorsque les défauts sont rares et indépendants.
///
/// Panique si `dpu` est négatif.
pub fn dpmo_yield_from_dpu(dpu: f64) -> f64 {
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
    fn dpmo_realistic_case() {
        // 15 défauts sur 1000 unités à 5 opportunités chacune :
        // DPMO = 15·1e6 / (1000·5) = 15_000_000 / 5000 = 3000.
        assert_relative_eq!(dpmo_from_defects(15, 1000, 5), 3000.0, epsilon = 1e-9);
        // Rendement associé : 1 − 3000/1e6 = 0.997.
        assert_relative_eq!(dpmo_yield(3000.0), 0.997, epsilon = 1e-12);
    }

    #[test]
    fn dpmo_limits_are_perfect_and_worst() {
        // Aucun défaut → DPMO nul → rendement parfait de 1.
        assert_relative_eq!(dpmo_from_defects(0, 1000, 5), 0.0, epsilon = 1e-12);
        assert_relative_eq!(dpmo_yield(0.0), 1.0, epsilon = 1e-12);
        // Un défaut à chaque opportunité → DPMO = 1e6 → rendement nul.
        assert_relative_eq!(dpmo_from_defects(50, 10, 5), 1_000_000.0, epsilon = 1e-6);
        assert_relative_eq!(dpmo_yield(1_000_000.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn dpmo_single_opportunity_matches_dpu() {
        // Avec une seule opportunité par unité, DPMO = DPU · 1e6.
        let defects = 15;
        let units = 1000;
        let dpu = dpmo_defects_per_unit(defects, units);
        assert_relative_eq!(
            dpmo_from_defects(defects, units, 1),
            dpu * 1_000_000.0_f64,
            epsilon = 1e-9
        );
    }

    #[test]
    fn dpmo_is_proportional_to_defects() {
        // Doubler les défauts double DPMO et double le DPU (linéarité).
        let single = dpmo_from_defects(15, 1000, 5);
        let double = dpmo_from_defects(30, 1000, 5);
        assert_relative_eq!(double, 2.0_f64 * single, epsilon = 1e-9);
        assert_relative_eq!(
            dpmo_defects_per_unit(30, 1000),
            2.0_f64 * dpmo_defects_per_unit(15, 1000),
            epsilon = 1e-12
        );
    }

    #[test]
    fn dpmo_poisson_yield_reciprocity() {
        // yield = e^(−DPU) ; DPU nul → rendement 1 ; réciprocité −ln(yield) = DPU.
        assert_relative_eq!(dpmo_yield_from_dpu(0.0), 1.0, epsilon = 1e-12);
        let dpu = 0.015_f64;
        let y = dpmo_yield_from_dpu(dpu);
        assert_relative_eq!(-y.ln(), dpu, epsilon = 1e-12);
        // Cas chiffré : e^(−0.015) ≈ 0.985111940.
        assert_relative_eq!(y, 0.985_111_940_f64, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le nombre d'unités inspectées doit être strictement positif")]
    fn dpmo_zero_units_panics() {
        let _ = dpmo_from_defects(3, 0, 5);
    }
}
