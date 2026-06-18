use crate::detectors::DetectorResult;
use serde::{Deserialize, Serialize};

/// Niveau de sévérité d'une alerte IDS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl AlertSeverity {
    pub fn label_en(&self) -> &'static str {
        match self
        {
            AlertSeverity::Info => "INFO",
            AlertSeverity::Warning => "WARNING",
            AlertSeverity::Critical => "CRITICAL",
        }
    }

    pub fn label_fr(&self) -> &'static str {
        match self
        {
            AlertSeverity::Info => "INFO",
            AlertSeverity::Warning => "ALERTE",
            AlertSeverity::Critical => "CRITIQUE",
        }
    }
}

impl From<&str> for AlertSeverity {
    fn from(s: &str) -> Self {
        match s.to_uppercase().as_str()
        {
            "CRITICAL" => AlertSeverity::Critical,
            "WARNING" => AlertSeverity::Warning,
            _ => AlertSeverity::Info,
        }
    }
}

/// Une alerte IDS structurée.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// ID unique de l'alerte
    pub id: u64,
    /// Horodatage
    pub timestamp: f64,
    /// Détecteur source
    pub detector: String,
    /// Type d'attaque (label_en)
    pub attack_type: String,
    /// Type d'attaque (label_fr)
    pub attack_type_fr: String,
    /// Sévérité
    pub severity: AlertSeverity,
    /// Confiance
    pub confidence: f32,
    /// IP source (attaquant)
    pub source_ip: String,
    /// IP destination (cible)
    pub destination_ip: String,
    /// Détails textuels
    pub details: String,
    /// Recommandation d'action
    pub recommendation: String,
}

impl Alert {
    pub fn from_detector_result(result: &DetectorResult, id: u64, timestamp: f64) -> Self {
        let severity = AlertSeverity::from(result.severity.as_str());
        let recommendation = generate_recommendation(&result.detector, &severity);

        Self {
            id,
            timestamp,
            detector: result.detector.clone(),
            attack_type: result.label_en.clone(),
            attack_type_fr: result.label_fr.clone(),
            severity,
            confidence: result.confidence,
            source_ip: result.source_ip.clone(),
            destination_ip: result.destination_ip.clone(),
            details: result.details.clone(),
            recommendation,
        }
    }

    pub fn is_critical(&self) -> bool {
        self.severity == AlertSeverity::Critical
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("JSON error: {}", e))
    }
}

fn generate_recommendation(detector: &str, severity: &AlertSeverity) -> String {
    match detector {
        "port_scan" => match severity {
            AlertSeverity::Critical => {
                "Bloquer immédiatement l'IP source. Analyser les ports ciblés pour identifier la cible potentielle.".to_string()
            }
            AlertSeverity::Warning => {
                "Surveiller l'IP source. Vérifier les logs de connexion cible.".to_string()
            }
            _ => {
                "Enregistrer pour corrélation future. Pas d'action immédiate requise.".to_string()
            }
        },
        "ddos" => match severity {
            AlertSeverity::Critical => {
                "Activer les mitigations DDoS (rate limiting, geo-blocking si applicable). Vérifier l'impact sur les services.".to_string()
            }
            AlertSeverity::Warning => {
                "Augmenter la surveillance. Préparer les mitigations si l'attaque s'intensifie.".to_string()
            }
            _ => {
                "Surveiller l'évolution du trafic. Pas d'action immédiate requise.".to_string()
            }
        },
        "brute_force" => match severity {
            AlertSeverity::Critical => {
                "Verrouiller le compte ciblé. Forcer la réinitialisation du mot de passe. Vérifier les logs d'authentification.".to_string()
            }
            AlertSeverity::Warning => {
                "Appliquer un rate-limiting sur l'authentification. Surveiller les tentatives depuis cette IP.".to_string()
            }
            _ => {
                "Enregistrer pour analyse de tendance. Pas d'action immédiate requise.".to_string()
            }
        },
        "dns_tunnel" => match severity {
            AlertSeverity::Critical => {
                "Bloquer le trafic DNS vers ce serveur de l'IP source. Analyser les requêtes DNS pour identifier les données exfiltrées.".to_string()
            }
            AlertSeverity::Warning => {
                "Inspecter les requêtes DNS de l'IP source. Vérifier les patterns de sous-domaines.".to_string()
            }
            _ => {
                "Surveiller le volume DNS de cette source. Pas d'action immédiate requise.".to_string()
            }
        },
        "beacon" => match severity {
            AlertSeverity::Critical => {
                "Isoler la machine source. Analyser le trafic vers la destination pour identifier le malware C2.".to_string()
            }
            AlertSeverity::Warning => {
                "Capture paquets sur la connexion source-destination. Analyser les patterns temporels.".to_string()
            }
            _ => {
                "Surveiller la fréquence des connexions. Pas d'action immédiate requise.".to_string()
            }
        },
        _ => "Aucune recommandation spécifique. Analyser manuellement.".to_string(),
    }
}

/// Système de journalisation des alertes IDS.
#[derive(Debug)]
pub struct AlertLog {
    alerts: Vec<Alert>,
    next_id: u64,
    /// Filtre de sévérité minimal
    min_severity: AlertSeverity,
}

impl AlertLog {
    pub fn new(min_severity: AlertSeverity) -> Self {
        Self {
            alerts: Vec::new(),
            next_id: 1,
            min_severity,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(AlertSeverity::Info)
    }

    /// Enregistrer une alerte.
    pub fn push(&mut self, mut alert: Alert) -> bool {
        if alert.severity >= self.min_severity
        {
            alert.id = self.next_id;
            self.next_id += 1;
            self.alerts.push(alert);
            true
        }
        else
        {
            false
        }
    }

    /// Convertir et enregistrer un résultat de détecteur.
    pub fn push_result(&mut self, result: &DetectorResult, timestamp: f64) -> bool {
        let alert = Alert::from_detector_result(result, self.next_id, timestamp);
        self.push(alert)
    }

    /// Convertir et enregistrer une liste de résultats.
    pub fn push_results(&mut self, results: &[DetectorResult], timestamp: f64) -> usize {
        results
            .iter()
            .filter(|r| self.push_result(r, timestamp))
            .count()
    }

    /// Nombre total d'alertes.
    pub fn len(&self) -> usize {
        self.alerts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.alerts.is_empty()
    }

    /// Alertes par sévérité.
    pub fn by_severity(&self, severity: AlertSeverity) -> Vec<&Alert> {
        self.alerts
            .iter()
            .filter(|a| a.severity == severity)
            .collect()
    }

    /// Alertes pour une IP source spécifique.
    pub fn by_source_ip(&self, ip: &str) -> Vec<&Alert> {
        self.alerts.iter().filter(|a| a.source_ip == ip).collect()
    }

    /// Dernières N alertes.
    pub fn recent(&self, n: usize) -> &[Alert] {
        let start = self.alerts.len().saturating_sub(n);
        &self.alerts[start..]
    }

    /// Exporter en JSON.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.alerts).map_err(|e| format!("JSON error: {}", e))
    }

    /// Réinitialiser le journal.
    pub fn clear(&mut self) {
        self.alerts.clear();
        self.next_id = 1;
    }
}

impl Default for AlertLog {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(severity: &str) -> DetectorResult {
        DetectorResult {
            detector: "port_scan".to_string(),
            label_en: "vertical_port_scan".to_string(),
            label_fr: "scan_vertical".to_string(),
            confidence: 0.9,
            severity: severity.to_string(),
            source_ip: "10.0.0.1".to_string(),
            destination_ip: "10.0.0.2".to_string(),
            details: "ports=50 connections=50".to_string(),
        }
    }

    #[test]
    fn test_alert_creation() {
        let result = make_result("CRITICAL");
        let alert = Alert::from_detector_result(&result, 1, 1000.0);
        assert!(alert.is_critical());
        assert_eq!(alert.source_ip, "10.0.0.1");
        assert!(!alert.recommendation.is_empty());
    }

    #[test]
    fn test_alert_log_push() {
        let mut log = AlertLog::with_defaults();
        let result = make_result("CRITICAL");
        let pushed = log.push_result(&result, 1000.0);
        assert!(pushed);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_alert_log_filter_severity() {
        let mut log = AlertLog::new(AlertSeverity::Warning);
        let r1 = make_result("INFO");
        let r2 = make_result("WARNING");
        let r3 = make_result("CRITICAL");

        log.push_result(&r1, 1.0);
        log.push_result(&r2, 2.0);
        log.push_result(&r3, 3.0);

        // INFO should be filtered out
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_alert_log_by_source_ip() {
        let mut log = AlertLog::with_defaults();
        let mut r1 = make_result("CRITICAL");
        r1.source_ip = "10.0.0.1".to_string();
        let mut r2 = make_result("WARNING");
        r2.source_ip = "10.0.0.2".to_string();

        log.push_result(&r1, 1.0);
        log.push_result(&r2, 2.0);

        let from_1 = log.by_source_ip("10.0.0.1");
        assert_eq!(from_1.len(), 1);
    }

    #[test]
    fn test_alert_json() {
        let result = make_result("CRITICAL");
        let alert = Alert::from_detector_result(&result, 1, 1000.0);
        let json = alert.to_json().unwrap();
        assert!(json.contains("port_scan"));
        assert!(json.contains("10.0.0.1"));
    }
}
