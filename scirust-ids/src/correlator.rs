use crate::alert::Alert;
use crate::detectors::DetectorResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration du correlateur d'alertes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelatorConfig {
    /// Fenêtre temporelle de corrélation en secondes
    pub correlation_window_secs: f64,
    /// Nombre minimal d'alertes corrélées pour un incident
    pub min_correlated_alerts: usize,
    /// Seuil de confiance pour la corrélation
    pub correlation_confidence: f32,
    /// TTL des alertes en mémoire (secondes)
    pub alert_ttl_secs: f64,
}

impl Default for CorrelatorConfig {
    fn default() -> Self {
        Self {
            correlation_window_secs: 300.0,
            min_correlated_alerts: 2,
            correlation_confidence: 0.6,
            alert_ttl_secs: 3600.0,
        }
    }
}

/// Type de corrélation détectée.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationType {
    /// Même source IP, attaques multiples
    MultiAttack,
    /// Escalade: scan puis brute force
    Escalation,
    /// Coordinated: attaques sur plusieurs cibles
    Coordinated,
    /// Pattern: séquence prédictive d'attaques
    Pattern,
}

impl CorrelationType {
    pub fn label_en(&self) -> &'static str {
        match self
        {
            CorrelationType::MultiAttack => "multi_attack",
            CorrelationType::Escalation => "escalation",
            CorrelationType::Coordinated => "coordinated_attack",
            CorrelationType::Pattern => "attack_pattern",
        }
    }

    pub fn label_fr(&self) -> &'static str {
        match self
        {
            CorrelationType::MultiAttack => "attaque_multiple",
            CorrelationType::Escalation => "escalade",
            CorrelationType::Coordinated => "attaque_coordonnée",
            CorrelationType::Pattern => "pattern_attaque",
        }
    }
}

/// Résultat de corrélation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    /// Type de corrélation
    pub correlation_type: CorrelationType,
    /// IDs des alertes corrélées
    pub alert_ids: Vec<u64>,
    /// IP source commune
    pub source_ip: String,
    /// Confiance de la corrélation
    pub confidence: f32,
    /// Description de la corrélation
    pub description: String,
    /// Recommandation
    pub recommendation: String,
}

/// Moteur de corrélation d'alertes IDS.
///
/// Identifie les patterns d'attaques complexes en croisant les alertes
/// dans le temps et par source/destination.
pub struct AlertCorrelator {
    pub config: CorrelatorConfig,
    /// Alertes actives (en mémoire)
    active_alerts: Vec<Alert>,
    /// Résultats de corrélation
    correlations: Vec<CorrelationResult>,
    /// Compteur d'incidents
    incident_count: u64,
}

impl AlertCorrelator {
    pub fn new(config: CorrelatorConfig) -> Self {
        Self {
            config,
            active_alerts: Vec::new(),
            correlations: Vec::new(),
            incident_count: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(CorrelatorConfig::default())
    }

    /// Ajouter une alerte au correlateur et tenter la corrélation.
    pub fn add_alert(&mut self, alert: Alert) -> Vec<CorrelationResult> {
        let now = alert.timestamp;
        self.active_alerts.push(alert);

        // Nettoyer les alertes expirées
        self.active_alerts
            .retain(|a| now - a.timestamp < self.config.alert_ttl_secs);

        // Tenter la corrélation
        self.correlate(now)
    }

    /// Ajouter des résultats de détecteur directement.
    pub fn add_results(
        &mut self,
        results: &[DetectorResult],
        timestamp: f64,
    ) -> Vec<CorrelationResult> {
        let mut correlations = Vec::new();
        for r in results
        {
            let alert = Alert::from_detector_result(r, 0, timestamp);
            let mut corr = self.add_alert(alert);
            correlations.append(&mut corr);
        }
        correlations
    }

    /// Corréler les alertes actives.
    fn correlate(&mut self, now: f64) -> Vec<CorrelationResult> {
        let mut new_correlations = Vec::new();

        // Grouper par source IP
        let mut by_source: HashMap<String, Vec<&Alert>> = HashMap::new();
        for alert in &self.active_alerts
        {
            if now - alert.timestamp <= self.config.correlation_window_secs
            {
                by_source
                    .entry(alert.source_ip.clone())
                    .or_default()
                    .push(alert);
            }
        }

        for (source_ip, alerts) in &by_source
        {
            if alerts.len() < self.config.min_correlated_alerts
            {
                continue;
            }

            // Corrélation 1: Multi-attack (même source, détecteurs différents)
            let detectors: Vec<&str> = alerts.iter().map(|a| a.detector.as_str()).collect();
            let unique_detectors: Vec<&str> = detectors
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            if unique_detectors.len() >= 2
            {
                let alert_ids = alerts.iter().map(|a| a.id).collect();
                let detector_names: Vec<&str> = unique_detectors.to_vec();
                let confidence = 0.7 + (alerts.len() as f32 * 0.05).min(0.3);

                new_correlations.push(CorrelationResult {
                    correlation_type: CorrelationType::MultiAttack,
                    alert_ids,
                    source_ip: source_ip.clone(),
                    confidence,
                    description: format!(
                        "Attaques multiples depuis {}: {}",
                        source_ip,
                        detector_names.join(", ")
                    ),
                    recommendation: format!(
                        "Bloquer l'IP {} immédiatement. Analyser l'ensemble des vecteurs d'attaque.",
                        source_ip
                    ),
                });
            }

            // Corrélation 2: Escalation (scan -> brute force)
            let has_scan = alerts.iter().any(|a| a.detector == "port_scan");
            let has_bf = alerts.iter().any(|a| a.detector == "brute_force");
            if has_scan && has_bf
            {
                let alert_ids = alerts.iter().map(|a| a.id).collect();
                new_correlations.push(CorrelationResult {
                    correlation_type: CorrelationType::Escalation,
                    alert_ids,
                    source_ip: source_ip.clone(),
                    confidence: 0.85,
                    description: format!(
                        "Escalade d'attaque depuis {}: scan de ports suivi de force brute",
                        source_ip
                    ),
                    recommendation: format!(
                        "Priorité maximale. L'attaquant {} a identifié les services ouverts puis tente l'authentification. Vérifier les comptes compromis.",
                        source_ip
                    ),
                });
            }

            // Corrélation 3: Escalation (scan -> beacon C2)
            let has_beacon = alerts.iter().any(|a| a.detector == "beacon");
            if has_scan && has_beacon
            {
                let alert_ids = alerts.iter().map(|a| a.id).collect();
                new_correlations.push(CorrelationResult {
                    correlation_type: CorrelationType::Escalation,
                    alert_ids,
                    source_ip: source_ip.clone(),
                    confidence: 0.80,
                    description: format!(
                        "Post-compromission suspect {}: scan de ports puis beaconing C2",
                        source_ip
                    ),
                    recommendation: format!(
                        "Isoler la machine {}. Un malware actif a pu être installé après le scan initial.",
                        source_ip
                    ),
                });
            }
        }

        // Corrélation 4: Coordinated (plusieurs sources, même cible)
        let mut by_dest: HashMap<String, Vec<&Alert>> = HashMap::new();
        for alert in &self.active_alerts
        {
            if now - alert.timestamp <= self.config.correlation_window_secs
            {
                by_dest
                    .entry(alert.destination_ip.clone())
                    .or_default()
                    .push(alert);
            }
        }

        for (dest_ip, alerts) in &by_dest
        {
            let sources: Vec<&str> = alerts.iter().map(|a| a.source_ip.as_str()).collect();
            let unique_sources: std::collections::HashSet<&str> = sources.into_iter().collect();
            if unique_sources.len() >= 3
            {
                let alert_ids = alerts.iter().map(|a| a.id).collect();
                let confidence = 0.7 + (unique_sources.len() as f32 * 0.05).min(0.25);
                new_correlations.push(CorrelationResult {
                    correlation_type: CorrelationType::Coordinated,
                    alert_ids,
                    source_ip: format!("{}+ sources", unique_sources.len()),
                    confidence,
                    description: format!(
                        "Attaque coordonnée sur {}: {} sources différentes",
                        dest_ip,
                        unique_sources.len()
                    ),
                    recommendation: format!(
                        "Activer les mitigations DDoS sur {}. Attaque distribuée détectée.",
                        dest_ip
                    ),
                });
            }
        }

        self.incident_count += new_correlations.len() as u64;
        self.correlations.extend(new_correlations.clone());
        new_correlations
    }

    /// Nombre total d'incidents corrélés.
    pub fn incident_count(&self) -> u64 {
        self.incident_count
    }

    /// Historique des corrélations.
    pub fn correlations(&self) -> &[CorrelationResult] {
        &self.correlations
    }

    /// Alertes actives.
    pub fn active_alerts(&self) -> &[Alert] {
        &self.active_alerts
    }

    /// Réinitialiser.
    pub fn reset(&mut self) {
        self.active_alerts.clear();
        self.correlations.clear();
        self.incident_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alert::AlertSeverity;

    fn make_alert(source: &str, dest: &str, detector: &str, ts: f64) -> Alert {
        Alert {
            id: 0,
            timestamp: ts,
            detector: detector.to_string(),
            attack_type: "test".to_string(),
            attack_type_fr: "test".to_string(),
            severity: AlertSeverity::Critical,
            confidence: 0.9,
            source_ip: source.to_string(),
            destination_ip: dest.to_string(),
            details: "test".to_string(),
            recommendation: "test".to_string(),
        }
    }

    #[test]
    fn test_multi_attack_correlation() {
        let mut corr = AlertCorrelator::with_defaults();

        let a1 = make_alert("10.0.0.1", "10.0.0.2", "port_scan", 100.0);
        let a2 = make_alert("10.0.0.1", "10.0.0.2", "brute_force", 150.0);

        let r1 = corr.add_alert(a1);
        assert!(r1.is_empty(), "one alert not enough");

        let r2 = corr.add_alert(a2);
        assert!(!r2.is_empty(), "should correlate multi-attack");
        assert!(matches!(
            r2[0].correlation_type,
            CorrelationType::MultiAttack
        ));
    }

    #[test]
    fn test_escalation_correlation() {
        let mut corr = AlertCorrelator::with_defaults();

        let a1 = make_alert("10.0.0.1", "10.0.0.2", "port_scan", 100.0);
        let a2 = make_alert("10.0.0.1", "10.0.0.2", "brute_force", 150.0);

        corr.add_alert(a1);
        let results = corr.add_alert(a2);

        let has_escalation = results
            .iter()
            .any(|r| matches!(r.correlation_type, CorrelationType::Escalation));
        assert!(has_escalation, "should detect escalation");
    }

    #[test]
    fn test_coordinated_attack() {
        let mut corr = AlertCorrelator::with_defaults();

        for src in &["10.0.0.1", "10.0.0.2", "10.0.0.3"]
        {
            let a = make_alert(src, "10.0.0.10", "ddos", 100.0);
            corr.add_alert(a);
        }

        // The third alert should trigger coordinated attack
        let a = make_alert("10.0.0.4", "10.0.0.10", "ddos", 100.0);
        let results = corr.add_alert(a);

        let has_coordinated = results
            .iter()
            .any(|r| matches!(r.correlation_type, CorrelationType::Coordinated));
        assert!(has_coordinated, "should detect coordinated attack");
    }

    #[test]
    fn test_no_correlation_different_sources() {
        let mut corr = AlertCorrelator::with_defaults();

        let a1 = make_alert("10.0.0.1", "10.0.0.2", "port_scan", 100.0);
        let a2 = make_alert("10.0.0.3", "10.0.0.4", "brute_force", 150.0);

        corr.add_alert(a1);
        let results = corr.add_alert(a2);
        // Different sources, different destinations: no correlation
        let has_multi = results
            .iter()
            .any(|r| matches!(r.correlation_type, CorrelationType::MultiAttack));
        assert!(!has_multi, "should not correlate different sources");
    }
}
