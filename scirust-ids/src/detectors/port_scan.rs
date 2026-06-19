use crate::detectors::DetectorResult;
use crate::flow::FlowWindow;
use serde::{Deserialize, Serialize};

/// Configuration du détecteur de scan de ports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortScanConfig {
    /// Nombre minimal de ports uniques ciblés par une source pour déclencher une alerte
    pub min_unique_ports: usize,
    /// Nombre minimal de connexions par source pour déclencher une alerte
    pub min_connections: usize,
    /// Fenêtre temporelle en secondes pour la détection
    pub time_window_secs: f64,
    /// Seuil de confiance minimal
    pub min_confidence: f32,
}

impl Default for PortScanConfig {
    fn default() -> Self {
        Self {
            min_unique_ports: 15,
            min_connections: 20,
            time_window_secs: 60.0,
            min_confidence: 0.7,
        }
    }
}

/// Type de scan de ports détecté.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScanType {
    /// Scan horizontal: un seul port, plusieurs destinations
    Horizontal,
    /// Scan vertical: une seule destination, plusieurs ports
    Vertical,
    /// Scan dégressif: tentatives croissantes
    Decrementing,
    /// Scan complet: tous les ports
    Full,
}

impl ScanType {
    pub fn label_en(&self) -> &'static str {
        match self
        {
            ScanType::Horizontal => "horizontal_port_scan",
            ScanType::Vertical => "vertical_port_scan",
            ScanType::Decrementing => "decrementing_port_scan",
            ScanType::Full => "full_port_scan",
        }
    }

    pub fn label_fr(&self) -> &'static str {
        match self
        {
            ScanType::Horizontal => "scan_horizontal",
            ScanType::Vertical => "scan_vertical",
            ScanType::Decrementing => "scan_dégressif",
            ScanType::Full => "scan_complet",
        }
    }
}

/// Détecteur de scan de ports.
///
/// Analyse les flux dans une fenêtre temporelle pour identifier
/// les comportements de scan de ports (vertical, horizontal, complet).
#[derive(Debug, Clone)]
pub struct PortScanDetector {
    pub config: PortScanConfig,
    /// Nombre de scans détectés
    detected_count: u64,
}

impl PortScanDetector {
    pub fn new(config: PortScanConfig) -> Self {
        Self {
            config,
            detected_count: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(PortScanConfig::default())
    }

    /// Analyser une fenêtre de flux et retourner les scan détectés.
    pub fn analyze(&mut self, window: &FlowWindow) -> Vec<DetectorResult> {
        let mut results = Vec::new();
        let conns_per_source = window.connections_per_source();
        let dests_per_source = window.destinations_per_source();

        for (src_ip, &conn_count) in &conns_per_source
        {
            if conn_count < self.config.min_connections
            {
                continue;
            }

            let unique_dst_ports = window
                .flows
                .iter()
                .filter(|f| f.src_ip == *src_ip)
                .map(|f| f.dst_port)
                .collect::<std::collections::HashSet<_>>()
                .len();

            if unique_dst_ports < self.config.min_unique_ports
            {
                continue;
            }

            // Classifier le type de scan
            let scan_type = if let Some(dests) = dests_per_source.get(src_ip)
            {
                if dests.len() == 1 && unique_dst_ports > self.config.min_unique_ports
                {
                    ScanType::Vertical
                }
                else if dests.len() > 1 && unique_dst_ports < 100
                {
                    ScanType::Horizontal
                }
                else if unique_dst_ports >= 100
                {
                    ScanType::Full
                }
                else
                {
                    ScanType::Vertical
                }
            }
            else
            {
                ScanType::Vertical
            };

            // Score de confiance basé sur la densité de ports scannés
            let port_ratio = unique_dst_ports as f32 / self.config.min_unique_ports as f32;
            let conn_ratio = conn_count as f32 / self.config.min_connections as f32;
            let confidence =
                (0.4 + (port_ratio * 0.3).min(0.3) + (conn_ratio * 0.3).min(0.3)).min(1.0);

            if confidence >= self.config.min_confidence
            {
                self.detected_count += 1;
                results.push(DetectorResult {
                    detector: "port_scan".to_string(),
                    label_en: scan_type.label_en().to_string(),
                    label_fr: scan_type.label_fr().to_string(),
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
                    destination_ip: window
                        .flows
                        .iter()
                        .find(|f| f.src_ip == *src_ip)
                        .map(|f| f.dst_ip.clone())
                        .unwrap_or_default(),
                    details: format!(
                        "ports={} connections={} type={}",
                        unique_dst_ports,
                        conn_count,
                        scan_type.label_en()
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

    fn make_scan_window() -> FlowWindow {
        let mut window = FlowWindow::new(0.0, 60.0);
        for port in 1..=50
        {
            let mut f = Flow::new("10.0.0.1", "10.0.0.2", 40000 + port, port);
            f.start_time = port as f64 * 0.5;
            f.end_time = f.start_time + 0.1;
            f.packets_out = 1;
            window.push(f);
        }
        window
    }

    #[test]
    fn test_detect_vertical_scan() {
        let window = make_scan_window();
        let mut det = PortScanDetector::with_defaults();
        let results = det.analyze(&window);
        assert!(!results.is_empty(), "should detect scan");
        assert_eq!(results[0].label_en, "vertical_port_scan");
        assert!(results[0].confidence > 0.7);
    }

    #[test]
    fn test_no_scan_normal_traffic() {
        let mut window = FlowWindow::new(0.0, 60.0);
        for i in 0..10
        {
            let mut f = Flow::new("10.0.0.1", "10.0.0.2", 40000, 80);
            f.start_time = i as f64;
            f.end_time = i as f64 + 0.1;
            f.packets_out = 5;
            window.push(f);
        }
        let mut det = PortScanDetector::with_defaults();
        let results = det.analyze(&window);
        assert!(results.is_empty(), "normal traffic should not trigger");
    }
}
