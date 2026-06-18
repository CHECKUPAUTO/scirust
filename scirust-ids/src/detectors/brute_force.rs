use crate::detectors::DetectorResult;
use crate::flow::FlowWindow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration du détecteur de brute force.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BruteForceConfig {
    /// Nombre maximal de tentatives échouées par fenêtre avant alerte
    pub max_failed_attempts: usize,
    /// Fenêtre temporelle en secondes
    pub time_window_secs: f64,
    /// Protocoles cibles (SSH, FTP, Telnet, HTTP)
    pub target_protocols: Vec<String>,
    /// Seuil minimal de confiance
    pub min_confidence: f32,
}

impl Default for BruteForceConfig {
    fn default() -> Self {
        Self {
            max_failed_attempts: 5,
            time_window_secs: 60.0,
            target_protocols: vec![
                "SSH".to_string(),
                "FTP".to_string(),
                "Telnet".to_string(),
                "HTTP".to_string(),
                "SMTP".to_string(),
            ],
            min_confidence: 0.7,
        }
    }
}

/// Type de brute force détecté.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BruteForceType {
    /// Password brute force: tentatives répétées de mots de passe
    Password,
    /// Credential stuffing: tentatives avec des paires user/pass différentes
    CredentialStuffing,
    /// Dictionary attack: essais basés sur un dictionnaire
    Dictionary,
}

impl BruteForceType {
    pub fn label_en(&self) -> &'static str {
        match self
        {
            BruteForceType::Password => "password_brute_force",
            BruteForceType::CredentialStuffing => "credential_stuffing",
            BruteForceType::Dictionary => "dictionary_attack",
        }
    }

    pub fn label_fr(&self) -> &'static str {
        match self
        {
            BruteForceType::Password => "brute_force_mot_de_passe",
            BruteForceType::CredentialStuffing => "réutilisation_identifiants",
            BruteForceType::Dictionary => "attaque_dictionnaire",
        }
    }
}

/// Détecteur de brute force.
///
/// Identifie les tentatives d'authentification répétées échouées
/// sur les services d'authentification (SSH, FTP, Telnet, HTTP).
#[derive(Debug, Clone)]
pub struct BruteForceDetector {
    pub config: BruteForceConfig,
    detected_count: u64,
}

impl BruteForceDetector {
    pub fn new(config: BruteForceConfig) -> Self {
        Self {
            config,
            detected_count: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(BruteForceConfig::default())
    }

    /// Analyser une fenêtre de flux pour détecter les attaques par force brute.
    ///
    /// Logique: on regroupe les flux par (src_ip, dst_ip, dst_port) et on
    /// identifie les patterns d'échec (RST après SYN, error_count élevé,
    /// protocole d'authentification ciblé).
    pub fn analyze(&mut self, window: &FlowWindow) -> Vec<DetectorResult> {
        let mut results = Vec::new();

        // Grouper par (src_ip, dst_ip, dst_port)
        let mut groups: HashMap<(String, String, u16), Vec<&crate::flow::Flow>> = HashMap::new();
        for f in &window.flows
        {
            groups
                .entry((f.src_ip.clone(), f.dst_ip.clone(), f.dst_port))
                .or_default()
                .push(f);
        }

        for ((src_ip, dst_ip, dst_port), flows) in &groups
        {
            // Filtrer sur les protocoles d'authentification
            let protocol = crate::flow::Protocol::from_port(*dst_port);
            let proto_name = protocol.to_string();
            if !self
                .config
                .target_protocols
                .iter()
                .any(|p| p.eq_ignore_ascii_case(&proto_name))
            {
                continue;
            }

            let total_connections = flows.len();
            if total_connections < self.config.max_failed_attempts
            {
                continue;
            }

            // Compter les indicateurs d'échec
            let total_rst: u32 = flows.iter().map(|f| f.rst_count).sum();
            let total_errors: u32 = flows.iter().map(|f| f.error_count).sum();
            let _total_syn: u32 = flows.iter().map(|f| f.syn_count).sum();

            // Un RST après SYN indique un échec d'authentification
            let failure_score =
                (total_rst as f64 + total_errors as f64) / (total_connections as f64 * 2.0);

            if failure_score < 0.3
            {
                continue;
            }

            // Classifier le type
            let unique_src_ports = flows
                .iter()
                .map(|f| f.src_port)
                .collect::<std::collections::HashSet<_>>()
                .len();
            let bf_type = if unique_src_ports > total_connections * 2 / 3
            {
                BruteForceType::CredentialStuffing
            }
            else if total_errors > total_rst
            {
                BruteForceType::Dictionary
            }
            else
            {
                BruteForceType::Password
            };

            let confidence = (0.5
                + failure_score as f32 * 0.3
                + (total_connections as f32 / self.config.max_failed_attempts as f32).min(0.2))
            .min(1.0);

            if confidence >= self.config.min_confidence
            {
                self.detected_count += 1;
                results.push(DetectorResult {
                    detector: "brute_force".to_string(),
                    label_en: bf_type.label_en().to_string(),
                    label_fr: bf_type.label_fr().to_string(),
                    confidence,
                    severity: if confidence >= 0.9
                    {
                        "CRITICAL".to_string()
                    }
                    else if confidence >= 0.8
                    {
                        "WARNING".to_string()
                    }
                    else
                    {
                        "INFO".to_string()
                    },
                    source_ip: src_ip.clone(),
                    destination_ip: dst_ip.clone(),
                    details: format!(
                        "port={} protocol={} connections={} rst={} errors={} type={}",
                        dst_port,
                        proto_name,
                        total_connections,
                        total_rst,
                        total_errors,
                        bf_type.label_en()
                    ),
                });
            }
        }

        results
    }

    pub fn detected_count(&self) -> u64 {
        self.detected_count
    }

    pub fn reset(&mut self) {
        self.detected_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::Flow;

    fn make_brute_force_window() -> FlowWindow {
        let mut window = FlowWindow::new(0.0, 60.0);
        // 10 tentatives SSH échouées depuis la même IP, même port source
        for i in 0..10
        {
            let mut f = Flow::new("10.0.0.1", "10.0.0.2", 50000, 22);
            f.syn_count = 1;
            f.rst_count = 1; // connexion rejetée
            f.packets_out = 2;
            f.packets_in = 1;
            f.bytes_out = 120;
            f.bytes_in = 60;
            f.start_time = i as f64 * 5.0;
            f.end_time = f.start_time + 0.5;
            window.push(f);
        }
        window
    }

    #[test]
    fn test_detect_ssh_brute_force() {
        let window = make_brute_force_window();
        let mut det = BruteForceDetector::with_defaults();
        let results = det.analyze(&window);
        assert!(!results.is_empty(), "should detect brute force");
        assert_eq!(results[0].label_en, "password_brute_force");
    }

    #[test]
    fn test_no_brute_force_normal() {
        let mut window = FlowWindow::new(0.0, 60.0);
        for i in 0..3
        {
            let mut f = Flow::new("10.0.0.1", "10.0.0.2", 40000, 22);
            f.packets_out = 10;
            f.packets_in = 10;
            f.bytes_out = 500;
            f.bytes_in = 500;
            f.start_time = i as f64 * 20.0;
            f.end_time = f.start_time + 5.0;
            window.push(f);
        }
        let mut det = BruteForceDetector::with_defaults();
        let results = det.analyze(&window);
        assert!(results.is_empty(), "normal SSH should not trigger");
    }
}
