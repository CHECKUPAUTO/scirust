//! Contrôleur "run-to-run" (R2R) par moyenne mobile pondérée
//! exponentiellement (EWMA) — l'algorithme canonique de contrôle de
//! recette entre lots en fabrication de semi-conducteurs (Sachs, Hu &
//! Ingolfsson, "Run by Run Process Control: Combining SPC and Feedback
//! Control", IEEE Trans. Semicond. Manuf. 8(1), 1995 ; Butler & Stefani,
//! "Supervisory Run-to-Run Control of Polysilicon Gate Etch Using In
//! Situ Ellipsometry", IEEE Trans. Semicond. Manuf. 7(2), 1994).
//!
//! ## Modèle et boucle
//! Modèle linéaire procédé→sortie : `y = a + b·u` (`a` = décalage/dérive
//! du procédé, inconnu et supposé lentement variable ; `b` = gain,
//! identifié hors ligne et supposé stable). Le contrôleur EWMA estime `a`
//! après chaque run et recalcule la recette pour annuler l'écart à la
//! cible au run suivant :
//!
//! `â_n = λ·(y_n - b̂·u_n) + (1-λ)·â_{n-1}`
//! `u_{n+1} = (target - â_n) / b̂`
//!
//! Vérifié contre un exemple travaillé : cible 500 nm, `b̂=5`, `λ=0.3`,
//! `u1=100s` (recette nominale), `y1=510nm` mesuré → `â1=3.0`,
//! `u2=99.4s`.
//!
//! **Limite honnête** : suppose un gain `b̂` connu et stable (identifié
//! hors ligne, par ex. par un plan d'expérience) — ce module ne fait pas
//! d'identification de gain en ligne, et ne modélise pas les dérives de
//! gain (seulement les dérives de décalage `a`).

/// Contrôleur EWMA run-to-run à état (estimation courante de la dérive).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EwmaR2rController {
    target: f64,
    gain: f64,
    lambda: f64,
    drift_estimate: f64,
}

impl EwmaR2rController {
    /// `target` = valeur de sortie visée, `gain` = `b̂` (identifié hors
    /// ligne, doit être non nul), `lambda ∈ (0,1]` = facteur d'oubli
    /// EWMA, `initial_drift_estimate` = `â_0` (souvent `0.0` en
    /// l'absence d'information préalable).
    pub fn new(target: f64, gain: f64, lambda: f64, initial_drift_estimate: f64) -> Self {
        Self {
            target,
            gain,
            lambda: lambda.clamp(1e-3, 1.0),
            drift_estimate: initial_drift_estimate,
        }
    }

    /// Recette recommandée pour le prochain run, compte tenu de l'état courant.
    pub fn next_recipe(&self) -> f64 {
        (self.target - self.drift_estimate) / self.gain
    }

    /// Met à jour l'estimation de dérive avec le run observé
    /// (`applied_recipe` = `u_n` réellement utilisée, `measured_output` =
    /// `y_n` mesurée) et renvoie la nouvelle recette recommandée.
    pub fn update(&mut self, applied_recipe: f64, measured_output: f64) -> f64 {
        let prediction_error = measured_output - self.gain * applied_recipe;
        self.drift_estimate =
            self.lambda * prediction_error + (1.0 - self.lambda) * self.drift_estimate;
        self.next_recipe()
    }

    pub fn drift_estimate(&self) -> f64 {
        self.drift_estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn matches_the_worked_example() {
        let controller = EwmaR2rController::new(500.0, 5.0, 0.3, 0.0);
        assert_relative_eq!(controller.next_recipe(), 100.0, epsilon = 1e-9);

        let mut controller = controller;
        let u2 = controller.update(100.0, 510.0);
        assert_relative_eq!(controller.drift_estimate(), 3.0, epsilon = 1e-9);
        assert_relative_eq!(u2, 99.4, epsilon = 1e-9);
    }

    /// Avec un gain parfaitement connu (`b̂=b`) et une dérive de procédé
    /// constante `a_true`, l'estimation EWMA d'une constante converge
    /// géométriquement vers cette constante (facteur `(1-λ)` par pas) —
    /// donc la sortie converge vers la cible. Vérifié : après 50 pas à
    /// `λ=0.3`, `(1-0.3)^50 ≈ 4e-8`, largement assez pour une tolérance
    /// de 1e-4 sur la sortie.
    #[test]
    fn converges_to_target_under_a_constant_unmodeled_offset() {
        let (target, gain, a_true) = (500.0, 5.0, 20.0);
        let mut controller = EwmaR2rController::new(target, gain, 0.3, 0.0);
        let mut u = controller.next_recipe();
        let mut y = a_true + gain * u;
        for _ in 0..50
        {
            u = controller.update(u, y);
            y = a_true + gain * u;
        }
        assert_relative_eq!(y, target, epsilon = 1e-4);
    }
}
