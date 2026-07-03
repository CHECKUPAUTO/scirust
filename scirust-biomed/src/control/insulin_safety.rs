//! Logique de supervision de sécurité par seuils pour pompe à insuline en
//! boucle fermée : suspension sur glycémie basse (avec variante
//! prédictive), plafond de dose par insuline active (IOB), et sortie de
//! mode automatique — les trois familles de garde-fous documentées
//! publiquement pour les systèmes hybrides de première génération (par
//! ex. Medtronic 670G/770G "Suspend on Low" / SmartGuard prédictif, résumé
//! FDA de sécurité et d'efficacité de ces dispositifs).
//!
//! **Avertissement** : reproduit le *principe* de ces garde-fous à des
//! fins de démonstration — les seuils sont des paramètres fournis par
//! l'appelant, pas les valeurs cliniques réelles d'un dispositif
//! homologué.

/// Suspend la délivrance d'insuline si la glycémie mesurée est sous le
/// seuil (unité cohérente avec `threshold` — mg/dL ou mmol/L selon
/// l'appelant, le module ne fait que comparer).
pub fn suspend_on_low(glucose: f64, threshold: f64) -> bool {
    glucose < threshold
}

/// Suspension prédictive : extrapole linéairement la tendance actuelle
/// (`trend_per_min`, même unité que `glucose` par minute) sur
/// `horizon_min` et suspend si la valeur prédite passerait sous le seuil
/// (le principe de SmartGuard/"Suspend before low").
pub fn predictive_suspend(
    glucose: f64,
    trend_per_min: f64,
    threshold: f64,
    horizon_min: f64,
) -> bool {
    glucose + trend_per_min * horizon_min < threshold
}

/// Dose maximale sûre compte tenu de l'insuline déjà active : ne jamais
/// dépasser `max_iob` au total. Renvoie la dose à délivrer, toujours dans
/// `[0, requested]`.
pub fn max_safe_bolus(requested: f64, current_iob: f64, max_iob: f64) -> f64 {
    let headroom = (max_iob - current_iob).max(0.0);
    requested.max(0.0).min(headroom)
}

/// Moniteur de sortie de mode automatique : accumule le temps passé hors
/// de la plage de sécurité `[low, high]` et signale une sortie de mode
/// requise une fois `max_duration_min` dépassé — la logique "hors plage
/// trop longtemps" documentée pour les systèmes hybrides de première
/// génération. Fenêtre glissante simple (pas de moyenne pondérée) : le
/// compteur se remet à zéro dès le retour dans la plage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoModeMonitor {
    low: f64,
    high: f64,
    max_duration_min: f64,
    out_of_range_min: f64,
}

impl AutoModeMonitor {
    pub fn new(low: f64, high: f64, max_duration_min: f64) -> Self {
        Self {
            low,
            high,
            max_duration_min,
            out_of_range_min: 0.0,
        }
    }

    /// Avance l'horloge de `dt_min` avec la lecture `glucose` courante ;
    /// renvoie `true` si le mode automatique doit être quitté maintenant.
    pub fn step(&mut self, glucose: f64, dt_min: f64) -> bool {
        if glucose < self.low || glucose > self.high
        {
            self.out_of_range_min += dt_min;
        }
        else
        {
            self.out_of_range_min = 0.0;
        }
        self.out_of_range_min >= self.max_duration_min
    }

    pub fn out_of_range_duration(&self) -> f64 {
        self.out_of_range_min
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn suspend_on_low_triggers_below_threshold_only() {
        assert!(suspend_on_low(65.0, 70.0));
        assert!(!suspend_on_low(70.0, 70.0));
        assert!(!suspend_on_low(90.0, 70.0));
    }

    #[test]
    fn predictive_suspend_extrapolates_the_trend() {
        // Falling 2/min for 20 min from 100 -> predicted 60 < 70: suspend.
        assert!(predictive_suspend(100.0, -2.0, 70.0, 20.0));
        // Falling slowly: predicted 90 >= 70: no suspend.
        assert!(!predictive_suspend(100.0, -0.5, 70.0, 20.0));
    }

    #[test]
    fn max_safe_bolus_respects_headroom() {
        assert_relative_eq!(max_safe_bolus(5.0, 3.0, 6.0), 3.0); // headroom 3 < requested 5
        assert_relative_eq!(max_safe_bolus(2.0, 3.0, 6.0), 2.0); // requested 2 < headroom 3
        assert_relative_eq!(max_safe_bolus(2.0, 8.0, 6.0), 0.0); // already over the limit
    }

    #[test]
    fn auto_mode_monitor_exits_only_after_sustained_excursion() {
        let mut mon = AutoModeMonitor::new(70.0, 180.0, 30.0);
        assert!(!mon.step(60.0, 10.0)); // 10 min out of range
        assert!(!mon.step(60.0, 10.0)); // 20 min
        assert!(mon.step(60.0, 10.0)); // 30 min: exit
    }

    #[test]
    fn auto_mode_monitor_resets_on_return_to_range() {
        let mut mon = AutoModeMonitor::new(70.0, 180.0, 30.0);
        mon.step(60.0, 20.0);
        assert!(!mon.step(100.0, 1.0)); // back in range: counter resets
        assert_relative_eq!(mon.out_of_range_duration(), 0.0);
        assert!(!mon.step(60.0, 20.0)); // only 20 min again, not 40
    }
}
