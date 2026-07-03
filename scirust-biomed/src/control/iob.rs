//! Suivi de l'insuline active ("insulin on board", IOB) par décroissance
//! mono-exponentielle — le principe utilisé par les systèmes en boucle
//! fermée hybride pour éviter l'empilement de doses ("stacking"), sous sa
//! forme la plus simple.
//!
//! **Avertissement** : les systèmes commerciaux utilisent des courbes
//! empiriquement ajustées et spécifiques au profil pharmacocinétique de
//! chaque insuline — par ex. la courbe biexponentielle de Walsh (projets
//! communautaires OpenAPS) ou le modèle exponentiel documenté par
//! LoopKit/Loop. Ce module n'en reproduit aucune : il expose le principe
//! (décroissance mono-exponentielle, constante de temps `tau` réglable)
//! pour piloter la logique de sécurité de
//! [`super::insulin_safety`], pas une courbe cliniquement validée.

/// Suivi de l'insuline active par sommation de doses à décroissance
/// exponentielle indépendante (chaque dose décroît avec sa propre
/// constante de temps `tau`, en minutes — permet de mélanger des
/// insulines de cinétiques différentes).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InsulinOnBoard {
    /// `(instant d'administration, dose, tau)` pour chaque bolus enregistré.
    doses: Vec<(f64, f64, f64)>,
}

impl InsulinOnBoard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enregistre une dose de `units` administrée à l'instant `t_min`,
    /// avec une constante de temps de décroissance `tau_min` (minutes).
    pub fn record_dose(&mut self, t_min: f64, units: f64, tau_min: f64) {
        self.doses.push((t_min, units, tau_min));
    }

    /// Insuline active totale à l'instant `now_min` :
    /// `Σ dose_i · exp(-(now - t_i)/tau_i)` sur les doses déjà
    /// administrées (`t_i <= now_min`).
    pub fn active_at(&self, now_min: f64) -> f64 {
        self.doses
            .iter()
            .filter(|(t, _, _)| *t <= now_min)
            .map(|(t, units, tau)| units * (-(now_min - t) / tau).exp())
            .sum()
    }

    /// Retire les doses dont la contribution résiduelle à `now_min` est
    /// sous `epsilon` unités, pour ne pas laisser l'historique grossir
    /// indéfiniment sur une session longue.
    pub fn prune(&mut self, now_min: f64, epsilon: f64) {
        self.doses.retain(|(t, units, tau)| {
            *t > now_min || units * (-(now_min - t) / tau).exp() >= epsilon
        });
    }

    pub fn dose_count(&self) -> usize {
        self.doses.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn single_dose_decays_exponentially() {
        let mut iob = InsulinOnBoard::new();
        iob.record_dose(0.0, 5.0, 180.0);
        assert_relative_eq!(iob.active_at(0.0), 5.0, epsilon = 1e-12);
        assert_relative_eq!(
            iob.active_at(180.0),
            5.0 / std::f64::consts::E,
            epsilon = 1e-9
        );
        assert!(iob.active_at(10_000.0) < 1e-10);
    }

    #[test]
    fn doses_before_now_are_ignored() {
        let mut iob = InsulinOnBoard::new();
        iob.record_dose(100.0, 5.0, 180.0);
        assert_relative_eq!(iob.active_at(50.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn stacked_doses_sum() {
        let mut iob = InsulinOnBoard::new();
        iob.record_dose(0.0, 3.0, 180.0);
        iob.record_dose(60.0, 2.0, 180.0);
        let expected = 3.0 * (-60.0_f64 / 180.0).exp() + 2.0;
        assert_relative_eq!(iob.active_at(60.0), expected, epsilon = 1e-9);
    }

    #[test]
    fn prune_drops_negligible_doses() {
        let mut iob = InsulinOnBoard::new();
        iob.record_dose(0.0, 1.0, 10.0); // decays fast
        iob.record_dose(0.0, 5.0, 10_000.0); // decays very slowly
        iob.prune(1000.0, 1e-6);
        assert_eq!(iob.dose_count(), 1);
    }
}
