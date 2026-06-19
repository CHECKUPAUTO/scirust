use crate::capture::{CaptureConfig, NetworkCapture};
use crate::correlator::{AlertCorrelator, CorrelationResult, CorrelatorConfig};
use crate::detectors::*;
use crate::flow::{FlowWindow, Protocol};
use crate::learner::{IdsLearner, LearnerConfig};
use crate::parsers::{MultiParser, ParsedPayload};
use crate::siem::{SiemConfig, SiemExporter};
use serde::{Deserialize, Serialize};

/// Configuration globale du moteur IDS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdsConfig {
    pub port_scan: port_scan::PortScanConfig,
    pub ddos: ddos::DdosConfig,
    pub brute_force: brute_force::BruteForceConfig,
    pub dns_tunnel: dns_tunnel::DnsTunnelConfig,
    pub beacon: beacon::BeaconConfig,
    pub correlator: CorrelatorConfig,
    pub learner: LearnerConfig,
    pub siem: SiemConfig,
    pub capture: CaptureConfig,
    /// Seuil de confiance minimal global (filtrage final)
    pub global_min_confidence: f32,
    /// Activer le parsing protocole (HTTP/DNS/SSH)
    pub enable_parsers: bool,
    /// Activer la corrélation d'alertes
    pub enable_correlation: bool,
    /// Activer le module ML d'anomalie
    pub enable_ml: bool,
    /// Activer l'export SIEM
    pub enable_siem: bool,
    /// Hostname de ce nœud IDS
    pub hostname: String,
}

impl Default for IdsConfig {
    fn default() -> Self {
        Self {
            port_scan: port_scan::PortScanConfig::default(),
            ddos: ddos::DdosConfig::default(),
            brute_force: brute_force::BruteForceConfig::default(),
            dns_tunnel: dns_tunnel::DnsTunnelConfig::default(),
            beacon: beacon::BeaconConfig::default(),
            correlator: CorrelatorConfig::default(),
            learner: LearnerConfig::default(),
            siem: SiemConfig::default(),
            capture: CaptureConfig::default(),
            global_min_confidence: 0.5,
            enable_parsers: true,
            enable_correlation: true,
            enable_ml: true,
            enable_siem: true,
            hostname: "scirust-ids".to_string(),
        }
    }
}

/// Rapport d'exécution d'un cycle IDS complet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdsReport {
    /// Nombre de flux analysés
    pub flows_analyzed: usize,
    /// Résultats par détecteur
    pub results: Vec<DetectorResult>,
    /// Alertes du correlateur
    pub correlations: Vec<CorrelationResult>,
    /// Nombre total d'alertes
    pub alert_count: usize,
    /// Nombre d'alertes critiques
    pub critical_count: usize,
    /// Nombre de corrélations
    pub correlation_count: usize,
    /// Score ML moyen (si activé)
    pub ml_avg_score: f64,
    /// Nombre d'anomalies ML détectées
    pub ml_anomaly_count: usize,
    /// Événements SIEM exportés
    pub siem_exported: usize,
    /// Timestamp du rapport
    pub timestamp: f64,
}

/// Moteur IDS production intégré.
///
/// Orchestre: capture -> parsing -> détection -> corrélation -> ML -> SIEM.
pub struct IdsEngine {
    pub config: IdsConfig,
    port_scan: PortScanDetector,
    ddos: DdosDetector,
    brute_force: BruteForceDetector,
    dns_tunnel: DnsTunnelDetector,
    beacon: BeaconDetector,
    correlator: AlertCorrelator,
    learner: IdsLearner,
    parser: MultiParser,
    siem: SiemExporter,
    event_counter: u64,
    reports: Vec<IdsReport>,
}

impl IdsEngine {
    pub fn new(config: IdsConfig) -> Self {
        Self {
            correlator: AlertCorrelator::new(config.correlator.clone()),
            learner: IdsLearner::new(config.learner.clone()),
            siem: SiemExporter::new(config.siem.clone()),
            parser: MultiParser::default(),
            port_scan: PortScanDetector::new(config.port_scan.clone()),
            ddos: DdosDetector::new(config.ddos.clone()),
            brute_force: BruteForceDetector::new(config.brute_force.clone()),
            dns_tunnel: DnsTunnelDetector::new(config.dns_tunnel.clone()),
            beacon: BeaconDetector::new(config.beacon.clone()),
            config,
            event_counter: 0,
            reports: Vec::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(IdsConfig::default())
    }

    /// Analyser une fenêtre de flux avec tous les sous-systèmes.
    pub fn analyze(&mut self, window: &FlowWindow, timestamp: f64) -> IdsReport {
        let mut all_results = Vec::new();

        // 1. Détecteurs statistiques
        let mut results = self.port_scan.analyze(window);
        all_results.append(&mut results);
        let mut results = self.ddos.analyze(window);
        all_results.append(&mut results);
        let mut results = self.brute_force.analyze(window);
        all_results.append(&mut results);
        let mut results = self.dns_tunnel.analyze(window);
        all_results.append(&mut results);
        let mut results = self.beacon.analyze(window);
        all_results.append(&mut results);

        // 2. Parsing protocole (si activé)
        if self.config.enable_parsers
        {
            for flow in &window.flows
            {
                // Le parsing nécessiterait le payload brut — ici on simule
                // avec les métadonnées du flux
                if let Protocol::Http | Protocol::Https = flow.protocol
                {
                    if flow.error_count > 0
                    {
                        all_results.push(DetectorResult {
                            detector: "http_parser".to_string(),
                            label_en: "http_error_rate".to_string(),
                            label_fr: "taux_erreur_http".to_string(),
                            confidence: 0.7,
                            severity: "WARNING".to_string(),
                            source_ip: flow.src_ip.clone(),
                            destination_ip: flow.dst_ip.clone(),
                            details: format!("errors={}", flow.error_count),
                        });
                    }
                }
            }
        }

        // 3. Filtrer par confiance globale
        all_results.retain(|r| r.confidence >= self.config.global_min_confidence);

        // 4. Corrélation (si activée)
        let mut correlations = Vec::new();
        if self.config.enable_correlation
        {
            correlations = self.correlator.add_results(&all_results, timestamp);
        }

        // 5. ML anomaly detection (si activé, sur features agrégées)
        let mut ml_anomaly_count = 0;
        let mut ml_avg_score = 0.0;
        if self.config.enable_ml && self.learner.is_trained()
        {
            let features = crate::protocols::window_features(window);
            if features.len() == self.learner.config.input_features
            {
                if let Some(result) = self.learner.detect(&features)
                {
                    ml_anomaly_count += 1;
                    all_results.push(result);
                }
                ml_avg_score = self.learner.score(&features);
            }
        }

        // 6. SIEM export (si activé)
        let mut siem_exported = 0;
        if self.config.enable_siem
        {
            self.siem
                .push_results(&all_results, timestamp, &self.config.hostname);
            for corr in &correlations
            {
                self.siem.push_correlation(corr, &self.config.hostname);
            }
            if let Ok(output) = self.siem.flush()
            {
                if !output.is_empty()
                {
                    siem_exported = all_results.len() + correlations.len();
                }
            }
        }

        let critical_count = all_results.iter().filter(|r| r.is_critical()).count();
        let alert_count = all_results.len();
        self.event_counter += all_results.len() as u64;

        let report = IdsReport {
            flows_analyzed: window.len(),
            correlation_count: correlations.len(),
            ml_avg_score,
            ml_anomaly_count,
            siem_exported,
            results: all_results,
            correlations,
            alert_count,
            critical_count,
            timestamp,
        };

        self.reports.push(report.clone());
        report
    }

    /// Analyser depuis un captureur réseau (boucle de capture en continu).
    pub fn analyze_capture<T: NetworkCapture>(
        &mut self,
        capture: &mut T,
        cycles: usize,
    ) -> Vec<IdsReport> {
        let mut reports = Vec::new();
        let window_secs = self.config.capture.window_duration_secs;

        for cycle in 0..cycles
        {
            let window_start = cycle as f64 * window_secs;
            if let Ok(window) = capture.capture_window(&self.config.capture, window_start)
            {
                let report = self.analyze(&window, window_start + window_secs);
                reports.push(report);
            }
        }

        reports
    }

    /// Parser un payload brut avec les décodeurs protocoles.
    pub fn parse_payload(
        &self,
        payload: &[u8],
        src_port: u16,
        dst_port: u16,
    ) -> Option<ParsedPayload> {
        self.parser.parse(payload, src_port, dst_port)
    }

    /// Accéder au correlateur.
    pub fn correlator(&self) -> &AlertCorrelator {
        &self.correlator
    }

    /// Accéder au module ML.
    pub fn learner(&self) -> &IdsLearner {
        &self.learner
    }

    /// Accéder au module ML (mutable, pour entraînement).
    pub fn learner_mut(&mut self) -> &mut IdsLearner {
        &mut self.learner
    }

    /// Accéder à l'exporteur SIEM.
    pub fn siem(&self) -> &SiemExporter {
        &self.siem
    }

    pub fn total_events(&self) -> u64 {
        self.event_counter
    }

    pub fn report_count(&self) -> usize {
        self.reports.len()
    }

    pub fn last_report(&self) -> Option<&IdsReport> {
        self.reports.last()
    }

    pub fn reset(&mut self) {
        self.port_scan.reset();
        self.ddos.reset();
        self.brute_force.reset();
        self.dns_tunnel.reset();
        self.beacon.reset();
        self.correlator.reset();
        self.siem.reset();
        self.event_counter = 0;
        self.reports.clear();
    }

    pub fn report_to_events(&mut self, report: &IdsReport) -> Vec<scirust_events_core::Event> {
        report
            .results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                r.to_event(
                    self.event_counter - report.results.len() as u64 + i as u64 + 1,
                    report.timestamp,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::SimulatedCapture;
    use crate::flow::Flow;

    fn make_mixed_window() -> FlowWindow {
        let mut window = FlowWindow::new(0.0, 60.0);

        for port in 1..=30
        {
            let mut f = Flow::new("10.0.0.1", "10.0.0.2", 40000 + port, port);
            f.start_time = port as f64 * 0.1;
            f.end_time = f.start_time + 0.05;
            f.packets_out = 1;
            window.push(f);
        }

        for i in 0..8
        {
            let mut f = Flow::new("10.0.0.5", "10.0.0.2", 50000, 22);
            f.syn_count = 1;
            f.rst_count = 1;
            f.packets_out = 2;
            f.packets_in = 1;
            f.bytes_out = 120;
            f.bytes_in = 60;
            f.start_time = 30.0 + i as f64 * 2.0;
            f.end_time = f.start_time + 0.3;
            window.push(f);
        }

        window
    }

    #[test]
    fn test_engine_analyze() {
        let window = make_mixed_window();
        let mut engine = IdsEngine::with_defaults();
        let report = engine.analyze(&window, 1000.0);

        assert!(report.flows_analyzed > 0);
        assert!(report.alert_count > 0);
        assert_eq!(report.timestamp, 1000.0);
    }

    #[test]
    fn test_engine_with_correlation() {
        let window = make_mixed_window();
        let mut engine = IdsEngine::with_defaults();
        let report = engine.analyze(&window, 1000.0);
        // Correlation requires multiple alert types from same source
        // May or may not trigger depending on source IPs
        assert!(report.alert_count > 0);
    }

    #[test]
    fn test_engine_with_siem() {
        let window = make_mixed_window();
        let mut engine = IdsEngine::with_defaults();
        let _report = engine.analyze(&window, 1000.0);
        // SIEM export should have happened
        assert!(engine.siem().exported_count() > 0);
    }

    #[test]
    fn test_engine_capture_integration() {
        let mut cap = SimulatedCapture::new();
        let config = CaptureConfig::default();

        for i in 0..20
        {
            let pkt = SimulatedCapture::fake_tcp_packet(
                "10.0.0.1",
                "10.0.0.2",
                40000 + i,
                1 + i,
                0x02,
                0,
                i as f64,
            );
            cap.inject(vec![pkt]);
        }

        cap.start(&config).unwrap();
        let mut engine = IdsEngine::with_defaults();
        let reports = engine.analyze_capture(&mut cap, 1);

        assert_eq!(reports.len(), 1);
        assert!(reports[0].flows_analyzed > 0);
    }

    #[test]
    fn test_engine_clear_results_below_threshold() {
        let mut window = FlowWindow::new(0.0, 60.0);
        for i in 0..5
        {
            let mut f = Flow::new("10.0.0.1", "10.0.0.2", 40000, 80);
            f.packets_out = 1;
            f.start_time = i as f64;
            f.end_time = i as f64 + 0.1;
            window.push(f);
        }
        let mut engine = IdsEngine::with_defaults();
        let report = engine.analyze(&window, 1000.0);
        assert_eq!(report.alert_count, 0);
    }

    #[test]
    fn test_engine_parse_payload() {
        let engine = IdsEngine::with_defaults();

        let http_payload = b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let parsed = engine.parse_payload(http_payload, 40000, 80);
        assert!(parsed.is_some());
        assert_eq!(parsed.unwrap().command, "GET");

        let ssh_payload = b"SSH-2.0-OpenSSH_8.9\r\n";
        let parsed = engine.parse_payload(ssh_payload, 40000, 22);
        assert!(parsed.is_some());
    }
}
