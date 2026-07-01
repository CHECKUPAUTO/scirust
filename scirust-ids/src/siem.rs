use crate::alert::Alert;
use crate::correlator::CorrelationResult;
use crate::detectors::DetectorResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration de l'export SIEM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiemConfig {
    /// Format de sortie
    pub output_format: SiemFormat,
    /// Endpoint Elasticsearch (URL de base)
    pub elasticsearch_url: Option<String>,
    /// Index Elasticsearch
    pub elasticsearch_index: String,
    /// Endpoint Syslog (host:port)
    pub syslog_endpoint: Option<String>,
    /// Facultatif: token d'authentification
    pub auth_token: Option<String>,
    /// Batch size pour l'envoi groupé
    pub batch_size: usize,
    /// Timeout en secondes
    pub timeout_secs: u64,
    /// Champs personnalisés ajoutés à chaque événement
    pub custom_fields: HashMap<String, String>,
}

impl Default for SiemConfig {
    fn default() -> Self {
        Self {
            output_format: SiemFormat::Json,
            elasticsearch_url: None,
            elasticsearch_index: "scirust-ids".to_string(),
            syslog_endpoint: None,
            auth_token: None,
            batch_size: 100,
            timeout_secs: 30,
            custom_fields: HashMap::new(),
        }
    }
}

/// Format de sortie SIEM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SiemFormat {
    /// JSON (Elasticsearch, Splunk, Logstash)
    Json,
    /// NDJSON (newline-delimited JSON, pour Elasticsearch bulk API)
    NdJson,
    /// CEF (Common Event Format, pour ArcSight, QRadar)
    Cef,
    /// Syslog (RFC 5424)
    Syslog,
    /// LEEF (Log Event Extended Format, pour QRadar)
    Leef,
}

/// Événement SIEM formaté.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiemEvent {
    /// Timestamp ISO 8601
    pub timestamp: String,
    /// Hostname de la source IDS
    pub host: String,
    /// Source de l'événement
    pub source: String,
    /// Sévérité (1-10)
    pub severity: u8,
    /// Catégorie
    pub category: String,
    /// Message principal
    pub message: String,
    /// Champs structurés
    pub fields: HashMap<String, String>,
}

impl SiemEvent {
    /// Convertir une alerte en événement SIEM.
    pub fn from_alert(alert: &Alert, host: &str) -> Self {
        let severity = match alert.severity
        {
            crate::alert::AlertSeverity::Info => 3,
            crate::alert::AlertSeverity::Warning => 6,
            crate::alert::AlertSeverity::Critical => 9,
        };

        let mut fields = HashMap::new();
        fields.insert("source_ip".to_string(), alert.source_ip.clone());
        fields.insert("destination_ip".to_string(), alert.destination_ip.clone());
        fields.insert("detector".to_string(), alert.detector.clone());
        fields.insert("confidence".to_string(), alert.confidence.to_string());
        fields.insert("attack_type".to_string(), alert.attack_type.clone());

        Self {
            timestamp: format_epoch(alert.timestamp),
            host: host.to_string(),
            source: format!("scirust-ids/{}", alert.detector),
            severity,
            category: "intrusion-detection".to_string(),
            message: format!(
                "[{}] {} from {} -> {} ({:.0}% confidence)",
                alert.severity.label_en(),
                alert.attack_type,
                alert.source_ip,
                alert.destination_ip,
                alert.confidence * 100.0
            ),
            fields,
        }
    }

    /// Convertir un résultat de corrélation en événement SIEM.
    ///
    /// `timestamp` est l'instant Unix (secondes depuis l'epoch UTC) de la
    /// corrélation ; il est propagé tel quel dans l'événement afin que
    /// l'horodatage reflète le moment réel de détection.
    pub fn from_correlation(corr: &CorrelationResult, host: &str, timestamp: f64) -> Self {
        let mut fields = HashMap::new();
        fields.insert(
            "correlation_type".to_string(),
            corr.correlation_type.label_en().to_string(),
        );
        fields.insert("source_ip".to_string(), corr.source_ip.clone());
        fields.insert("alert_count".to_string(), corr.alert_ids.len().to_string());

        Self {
            timestamp: format_epoch(timestamp),
            host: host.to_string(),
            source: "scirust-ids/correlator".to_string(),
            severity: 8,
            category: "correlation".to_string(),
            message: corr.description.clone(),
            fields,
        }
    }

    /// Convertir en JSON.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("JSON error: {}", e))
    }

    /// Convertir en CEF.
    pub fn to_cef(&self) -> String {
        format!(
            "CEF:0|SciRust|IDS|1.0|{}|{}|{}|{}",
            self.category,
            self.message,
            self.severity,
            self.to_fields_string()
        )
    }

    /// Convertir en LEEF.
    pub fn to_leef(&self) -> String {
        format!(
            "LEEF:2.0|SciRust|IDS|1.0|{}|{}|severity={}",
            self.category, self.message, self.severity
        )
    }

    fn to_fields_string(&self) -> String {
        self.fields
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Formater un timestamp Unix (secondes depuis l'epoch UTC) en date-heure
/// ISO 8601 / RFC 3339 complète, p.ex. `2023-11-14T22:13:20.000Z`.
///
/// La conversion jour civil utilise l'algorithme de Howard Hinnant, sans
/// dépendance externe, et gère correctement les années bissextiles. Elle
/// est déterministe et ne panique jamais (les timestamps négatifs, avant
/// l'epoch, sont pris en charge).
fn format_epoch(ts: f64) -> String {
    // Partie entière (secondes) et fractionnaire (millisecondes), en gardant
    // les millisecondes dans [0, 1000) même pour les timestamps négatifs.
    let millis_total = (ts * 1000.0).round() as i64;
    let mut total_secs = millis_total.div_euclid(1000);
    let millis = millis_total.rem_euclid(1000);

    let seconds = total_secs.rem_euclid(60);
    total_secs = total_secs.div_euclid(60);
    let minutes = total_secs.rem_euclid(60);
    total_secs = total_secs.div_euclid(60);
    let hours = total_secs.rem_euclid(24);
    let days = total_secs.div_euclid(24);

    let (year, month, day) = civil_from_days(days);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hours, minutes, seconds, millis
    )
}

/// Convertir un nombre de jours depuis l'epoch Unix (1970-01-01) en date
/// civile `(année, mois, jour)`. Algorithme de Howard Hinnant.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // Décaler l'ère de sorte que le jour 0 soit le 0000-03-01.
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day)
}

/// Moteur d'export SIEM.
///
/// Convertit les alertes et corrélations en formats compatibles SIEM
/// et gère l'envoi groupé (batch).
pub struct SiemExporter {
    pub config: SiemConfig,
    /// Buffer d'événements en attente d'envoi
    buffer: Vec<SiemEvent>,
    /// Nombre total d'événements exportés
    exported_count: u64,
    /// Dernière erreur
    last_error: Option<String>,
    /// Statistiques par format
    stats: HashMap<String, u64>,
}

impl SiemExporter {
    pub fn new(config: SiemConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
            exported_count: 0,
            last_error: None,
            stats: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(SiemConfig::default())
    }

    /// Ajouter une alerte au buffer.
    pub fn push_alert(&mut self, alert: &Alert, host: &str) {
        let event = SiemEvent::from_alert(alert, host);
        self.buffer.push(event);
        if self.buffer.len() >= self.config.batch_size
        {
            let _ = self.flush();
        }
    }

    /// Ajouter des résultats de détecteur au buffer.
    pub fn push_results(&mut self, results: &[DetectorResult], timestamp: f64, host: &str) {
        for r in results
        {
            let alert = crate::alert::Alert::from_detector_result(r, 0, timestamp);
            self.push_alert(&alert, host);
        }
    }

    /// Ajouter une corrélation au buffer.
    ///
    /// `timestamp` est l'instant Unix (secondes depuis l'epoch UTC) de la
    /// corrélation, propagé dans l'événement SIEM.
    pub fn push_correlation(&mut self, corr: &CorrelationResult, timestamp: f64, host: &str) {
        let event = SiemEvent::from_correlation(corr, host, timestamp);
        self.buffer.push(event);
    }

    /// Vider le buffer et formater les événements.
    pub fn flush(&mut self) -> Result<String, String> {
        if self.buffer.is_empty()
        {
            return Ok(String::new());
        }

        let output = match self.config.output_format
        {
            SiemFormat::Json => self.format_json()?,
            SiemFormat::NdJson => self.format_ndjson()?,
            SiemFormat::Cef => self.format_cef()?,
            SiemFormat::Syslog => self.format_syslog()?,
            SiemFormat::Leef => self.format_leef()?,
        };

        let count = self.buffer.len() as u64;
        self.exported_count += count;
        *self
            .stats
            .entry(format!("{:?}", self.config.output_format))
            .or_insert(0) += count;
        self.buffer.clear();

        Ok(output)
    }

    fn format_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.buffer).map_err(|e| format!("JSON error: {}", e))
    }

    fn format_ndjson(&self) -> Result<String, String> {
        let mut lines = Vec::new();
        for event in &self.buffer
        {
            lines.push(serde_json::to_string(event).map_err(|e| format!("JSON error: {}", e))?);
        }
        Ok(lines.join("\n"))
    }

    fn format_cef(&self) -> Result<String, String> {
        Ok(self
            .buffer
            .iter()
            .map(|e| e.to_cef())
            .collect::<Vec<_>>()
            .join("\n"))
    }

    fn format_syslog(&self) -> Result<String, String> {
        let mut lines = Vec::new();
        for event in &self.buffer
        {
            let severity = match event.severity
            {
                1..=3 => 5,  // notice
                4..=6 => 4,  // warning
                7..=8 => 3,  // error
                9..=10 => 2, // critical
                _ => 6,
            };
            let msg = format!(
                "<{}>1 {} {} scirust-ids - - - {}",
                16 + severity, // facility=1 (kern) + severity
                event.timestamp,
                event.host,
                event.message
            );
            lines.push(msg);
        }
        Ok(lines.join("\n"))
    }

    fn format_leef(&self) -> Result<String, String> {
        Ok(self
            .buffer
            .iter()
            .map(|e| e.to_leef())
            .collect::<Vec<_>>()
            .join("\n"))
    }

    /// Nombre d'événements exportés.
    pub fn exported_count(&self) -> u64 {
        self.exported_count
    }

    /// Taille du buffer.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Dernière erreur.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Statistiques par format.
    pub fn stats(&self) -> &HashMap<String, u64> {
        &self.stats
    }

    /// Réinitialiser.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.exported_count = 0;
        self.last_error = None;
        self.stats.clear();
    }
}

impl Default for SiemExporter {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alert::AlertSeverity;

    fn make_test_alert() -> Alert {
        Alert {
            id: 42,
            timestamp: 1700000000.0,
            detector: "port_scan".to_string(),
            attack_type: "vertical_port_scan".to_string(),
            attack_type_fr: "scan_vertical".to_string(),
            severity: AlertSeverity::Critical,
            confidence: 0.95,
            source_ip: "10.0.0.1".to_string(),
            destination_ip: "10.0.0.2".to_string(),
            details: "ports=50 connections=50".to_string(),
            recommendation: "Block immediately".to_string(),
        }
    }

    #[test]
    fn test_siem_event_from_alert() {
        let alert = make_test_alert();
        let event = SiemEvent::from_alert(&alert, "ids-host");
        assert_eq!(event.host, "ids-host");
        assert_eq!(event.severity, 9); // Critical
        assert!(event.message.contains("vertical_port_scan"));
    }

    #[test]
    fn test_siem_json_export() {
        let mut exporter = SiemExporter::with_defaults();
        let alert = make_test_alert();
        exporter.push_alert(&alert, "ids-host");
        let json = exporter.flush().unwrap();
        assert!(json.contains("vertical_port_scan"));
        assert!(json.contains("10.0.0.1"));
        assert_eq!(exporter.exported_count(), 1);
    }

    #[test]
    fn test_siem_cef_export() {
        let config = SiemConfig {
            output_format: SiemFormat::Cef,
            ..Default::default()
        };
        let mut exporter = SiemExporter::new(config);
        let alert = make_test_alert();
        exporter.push_alert(&alert, "ids-host");
        let cef = exporter.flush().unwrap();
        assert!(cef.contains("CEF:0|SciRust|IDS|1.0"));
    }

    #[test]
    fn test_siem_ndjson_export() {
        let config = SiemConfig {
            output_format: SiemFormat::NdJson,
            ..Default::default()
        };
        let mut exporter = SiemExporter::new(config);
        exporter.push_alert(&make_test_alert(), "ids-host");
        exporter.push_alert(&make_test_alert(), "ids-host");
        let ndjson = exporter.flush().unwrap();
        assert_eq!(ndjson.lines().count(), 2);
    }

    #[test]
    fn test_siem_batch_flush() {
        let config = SiemConfig {
            batch_size: 3,
            ..Default::default()
        };
        let mut exporter = SiemExporter::new(config);

        exporter.push_alert(&make_test_alert(), "host");
        assert_eq!(exporter.buffer_size(), 1);

        exporter.push_alert(&make_test_alert(), "host");
        assert_eq!(exporter.buffer_size(), 2);

        exporter.push_alert(&make_test_alert(), "host");
        assert_eq!(exporter.buffer_size(), 0, "should auto-flush at batch_size");
    }

    #[test]
    fn test_siem_syslog_export() {
        let config = SiemConfig {
            output_format: SiemFormat::Syslog,
            ..Default::default()
        };
        let mut exporter = SiemExporter::new(config);
        exporter.push_alert(&make_test_alert(), "ids-host");
        let syslog = exporter.flush().unwrap();
        assert!(syslog.contains("<"));
        assert!(syslog.contains("scirust-ids"));
    }

    fn make_test_correlation() -> CorrelationResult {
        CorrelationResult {
            correlation_type: crate::correlator::CorrelationType::MultiAttack,
            alert_ids: vec![1, 2, 3],
            source_ip: "10.0.0.1".to_string(),
            confidence: 0.9,
            description: "multi attack".to_string(),
            recommendation: "block".to_string(),
        }
    }

    // Regression: format_epoch must produce a full ISO 8601 / RFC 3339 UTC
    // datetime (with date), not a time-only string, and must not wrap the hour
    // component modulo 24h. 1_700_000_000 s == 2023-11-14T22:13:20Z.
    #[test]
    fn test_format_epoch_full_iso8601() {
        let s = format_epoch(1_700_000_000.0);
        assert_eq!(s, "2023-11-14T22:13:20.000Z");
        // Must carry a date, not just a time (old bug produced "T00:53:20.000Z").
        assert!(
            s.starts_with("2023-11-14T"),
            "missing date component: {}",
            s
        );
        assert!(!s.starts_with('T'), "time-only string regressed: {}", s);
        // Epoch itself.
        assert_eq!(format_epoch(0.0), "1970-01-01T00:00:00.000Z");
        // Milliseconds are preserved.
        assert_eq!(format_epoch(1_700_000_000.5), "2023-11-14T22:13:20.500Z");
        // Leap-year date handling (2024 is a leap year): 2024-02-29T00:00:00Z.
        assert_eq!(format_epoch(1_709_164_800.0), "2024-02-29T00:00:00.000Z");
    }

    // Regression: from_correlation must use the supplied timestamp rather than
    // hardcoding the epoch (0.0), so correlation events keep their real time.
    #[test]
    fn test_from_correlation_uses_real_timestamp() {
        let corr = make_test_correlation();
        let event = SiemEvent::from_correlation(&corr, "ids-host", 1_700_000_000.0);
        assert_eq!(event.timestamp, "2023-11-14T22:13:20.000Z");
        assert_ne!(
            event.timestamp,
            format_epoch(0.0),
            "correlation event still hardcodes the epoch"
        );
    }

    // Regression: the same real timestamp must survive through the exporter path.
    #[test]
    fn test_push_correlation_preserves_timestamp() {
        let mut exporter = SiemExporter::with_defaults();
        let corr = make_test_correlation();
        exporter.push_correlation(&corr, 1_700_000_000.0, "ids-host");
        let json = exporter.flush().unwrap();
        assert!(
            json.contains("2023-11-14T22:13:20.000Z"),
            "exported correlation lost its timestamp: {}",
            json
        );
    }
}
