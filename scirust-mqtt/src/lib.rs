//! SciRust MQTT Bridge
//!
//! Publishes detected events from the SciRust event pipeline to MQTT brokers
//! for Industry 4.0 dashboards, SCADA integration, and alerting systems.
//!
//! Supports MQTT v3.1.1 / v5 semantics with SparkPlug B-compatible payloads.
//!
//! ## Architecture
//! ```text
//! EventDetector -> Event -> MqttPublisher -> [MQTT Broker] -> Dashboard/SCADA
//! ```

use scirust_events_core::Event;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// MQTT Payload Format
// ---------------------------------------------------------------------------

/// Standard MQTT event payload following SparkPlug B conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayload {
    /// Source identifier (e.g. "line3-spindle-vibration")
    pub source: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Event ID
    pub event_id: u64,
    /// English label
    pub label_en: String,
    /// French label
    pub label_fr: String,
    /// Confidence score 0.0-1.0
    pub confidence: f32,
    /// Optional data snapshot
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_snapshot: Option<Vec<f32>>,
    /// Severity: INFO, WARNING, CRITICAL
    pub severity: EventSeverity,
    /// Structured metadata (e.g. {"bearing_fault": "BPFO", "harmonics": "1"})
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Event severity levels for industrial alerting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum EventSeverity {
    Info,
    Warning,
    Critical,
}

impl EventSeverity {
    /// Derive severity from confidence score.
    /// > 0.95 → Critical, > 0.8 → Warning, else Info
    pub fn from_confidence(confidence: f32) -> Self {
        if confidence >= 0.95
        {
            EventSeverity::Critical
        }
        else if confidence >= 0.8
        {
            EventSeverity::Warning
        }
        else
        {
            EventSeverity::Info
        }
    }

    /// MQTT topic suffix for this severity level.
    pub fn topic_suffix(&self) -> &str {
        match self
        {
            EventSeverity::Info => "info",
            EventSeverity::Warning => "warning",
            EventSeverity::Critical => "critical",
        }
    }
}

/// Convert a SciRust `Event` into an MQTT payload.
pub fn event_to_payload(
    event: &Event,
    source: &str,
    metadata: Option<serde_json::Value>,
) -> EventPayload {
    let severity = EventSeverity::from_confidence(event.confidence);
    EventPayload {
        source: source.to_string(),
        timestamp: format_unix_timestamp(event.timestamp),
        event_id: event.id,
        label_en: event.label_en.clone(),
        label_fr: event.label_fr.clone(),
        confidence: event.confidence,
        data_snapshot: event.data_snapshot.clone(),
        severity,
        metadata,
    }
}

fn format_unix_timestamp(ts: f64) -> String {
    // Simple ISO 8601-like formatting
    let total_secs = ts as i64;
    let hours = (total_secs / 3600) % 24;
    let minutes = (total_secs / 60) % 60;
    let seconds = total_secs % 60;
    let millis = ((ts - total_secs as f64) * 1000.0).round() as u32;
    format!("T{:02}:{:02}:{:02}.{:03}Z", hours, minutes, seconds, millis)
}

// ---------------------------------------------------------------------------
// MQTT Client Abstraction
// ---------------------------------------------------------------------------

/// Configuration for an MQTT connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttConfig {
    /// Broker hostname or IP
    pub host: String,
    /// Broker port (default 1883)
    pub port: u16,
    /// Client identifier
    pub client_id: String,
    /// Base topic for publishing events
    pub base_topic: String,
    /// Username (optional)
    pub username: Option<String>,
    /// Password (optional)
    pub password: Option<String>,
    /// Keep-alive interval in seconds
    pub keep_alive_secs: u16,
    /// QoS level (0, 1, or 2)
    pub qos: u8,
}

impl Default for MqttConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 1883,
            client_id: "scirust-monitor".to_string(),
            base_topic: "scirust/events".to_string(),
            username: None,
            password: None,
            keep_alive_secs: 60,
            qos: 1,
        }
    }
}

/// The MQTT client abstraction.
///
/// Implement this trait to connect to real MQTT brokers or to provide
/// a simulated backend for testing.
pub trait MqttPublisher {
    /// Connect to the MQTT broker.
    fn connect(&mut self, config: &MqttConfig) -> Result<(), String>;

    /// Disconnect from the broker.
    fn disconnect(&mut self) -> Result<(), String>;

    /// Publish a message to a specific topic.
    fn publish(&mut self, topic: &str, payload: &[u8], qos: u8, retain: bool)
    -> Result<(), String>;

    /// Publish an `EventPayload` as JSON on the configured base topic.
    fn publish_event(&mut self, event_payload: &EventPayload) -> Result<(), String>;

    /// Check if the client is connected.
    fn is_connected(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Simulated MQTT Publisher
// ---------------------------------------------------------------------------

/// A simulated MQTT publisher that logs messages to an in-memory buffer.
///
/// Useful for development, testing, and CI without requiring a real broker.
#[derive(Debug)]
pub struct SimulatedMqttPublisher {
    config: Option<MqttConfig>,
    connected: bool,
    /// All published messages: (topic, payload, qos, retain)
    pub messages: Vec<(String, Vec<u8>, u8, bool)>,
    /// Number of publishes attempted
    pub publish_count: u64,
    /// Last error message
    pub last_error: Option<String>,
}

impl SimulatedMqttPublisher {
    pub fn new() -> Self {
        Self {
            config: None,
            connected: false,
            messages: Vec::new(),
            publish_count: 0,
            last_error: None,
        }
    }

    /// Count events by severity.
    pub fn count_by_severity(&self) -> (usize, usize, usize) {
        let mut info = 0;
        let mut warn = 0;
        let mut crit = 0;
        for (topic, _, _, _) in &self.messages
        {
            if topic.contains("/critical")
            {
                crit += 1;
            }
            else if topic.contains("/warning")
            {
                warn += 1;
            }
            else if topic.contains("/info")
            {
                info += 1;
            }
        }
        (info, warn, crit)
    }

    /// Get all published payloads deserialized.
    pub fn get_events(&self) -> Vec<EventPayload> {
        self.messages
            .iter()
            .filter_map(|(_, payload, _, _)| serde_json::from_slice(payload).ok())
            .collect()
    }
}

impl Default for SimulatedMqttPublisher {
    fn default() -> Self {
        Self::new()
    }
}

impl MqttPublisher for SimulatedMqttPublisher {
    fn connect(&mut self, config: &MqttConfig) -> Result<(), String> {
        self.config = Some(config.clone());
        self.connected = true;
        self.messages.clear();
        self.publish_count = 0;
        Ok(())
    }

    fn disconnect(&mut self) -> Result<(), String> {
        self.connected = false;
        Ok(())
    }

    fn publish(
        &mut self,
        topic: &str,
        payload: &[u8],
        qos: u8,
        retain: bool,
    ) -> Result<(), String> {
        if !self.connected
        {
            self.last_error = Some("Not connected".to_string());
            return Err("Not connected to broker".to_string());
        }
        if topic.is_empty()
        {
            self.last_error = Some("Empty topic".to_string());
            return Err("Topic cannot be empty".to_string());
        }
        self.messages
            .push((topic.to_string(), payload.to_vec(), qos, retain));
        self.publish_count += 1;
        Ok(())
    }

    fn publish_event(&mut self, event_payload: &EventPayload) -> Result<(), String> {
        let cfg = self.config.as_ref().ok_or("Not configured")?;
        let topic = format!(
            "{}/{}/{}",
            cfg.base_topic,
            event_payload.source,
            event_payload.severity.topic_suffix()
        );
        let payload = serde_json::to_vec(event_payload)
            .map_err(|e| format!("JSON serialization error: {}", e))?;
        self.publish(&topic, &payload, cfg.qos, false)
    }

    fn is_connected(&self) -> bool {
        self.connected
    }
}

// ---------------------------------------------------------------------------
// High-level bridge functions
// ---------------------------------------------------------------------------

/// Publish a batch of SciRust `Event`s to an MQTT broker.
///
/// Each event is serialized as JSON and published on a topic:
/// `{base_topic}/{source}/{severity}`
pub fn publish_events(
    publisher: &mut dyn MqttPublisher,
    events: &[Event],
    source: &str,
    metadata: Option<serde_json::Value>,
) -> Result<usize, String> {
    let mut published = 0usize;
    for event in events
    {
        let payload = event_to_payload(event, source, metadata.clone());
        publisher.publish_event(&payload)?;
        published += 1;
    }
    Ok(published)
}

/// Filter events by minimum confidence threshold.
pub fn filter_by_confidence(events: &[Event], min_confidence: f32) -> Vec<Event> {
    events
        .iter()
        .filter(|e| e.confidence >= min_confidence)
        .cloned()
        .collect()
}

/// Generate a SparkPlug B-compatible birth certificate payload.
///
/// The birth certificate announces the device's capabilities to the broker
/// on first connection.
pub fn sparkplug_birth_certificate(
    group_id: &str,
    edge_node_id: &str,
    device_id: &str,
) -> serde_json::Value {
    serde_json::json!({
        "timestamp": 0u64,
        "metrics": [
            {
                "name": "Node Control/Rebirth",
                "timestamp": 0u64,
                "dataType": "Boolean",
                "value": false
            }
        ],
        "seq": 0u64,
        "uuid": format!("{}_{}_{}", group_id, edge_node_id, device_id)
    })
}

// ---------------------------------------------------------------------------
// Industrial Integration Helpers
// ---------------------------------------------------------------------------

/// Configuration for an industrial monitoring station.
///
/// Maps a physical station (machine, line, cell) to its sensor configuration,
/// MQTT topics, and detection parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStation {
    /// Station identifier (e.g. "line3-station12-spindle")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// MQTT base topic for this station
    pub mqtt_topic: String,
    /// OPC-UA node IDs to monitor
    pub sensor_node_ids: Vec<String>,
    /// Minimum confidence to publish an event
    pub min_confidence: f32,
    /// Sampling interval in milliseconds
    pub sampling_interval_ms: f64,
    /// Event detection threshold
    pub detection_threshold: f64,
    /// EMA smoothing factor for SpikeDetector
    pub ema_alpha: f64,
    /// Sliding window size for EventStream
    pub window_size: usize,
    /// Sliding window stride
    pub window_stride: usize,
}

impl Default for MonitoringStation {
    fn default() -> Self {
        Self {
            id: "station-1".to_string(),
            name: "Default Station".to_string(),
            mqtt_topic: "scirust/events/station-1".to_string(),
            sensor_node_ids: vec![],
            min_confidence: 0.8,
            sampling_interval_ms: 100.0,
            detection_threshold: 1.0,
            ema_alpha: 0.8,
            window_size: 32,
            window_stride: 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_event() -> Event {
        Event {
            id: 1,
            timestamp: 1000.5,
            label_en: "spike".to_string(),
            label_fr: "pic".to_string(),
            confidence: 0.96,
            data_snapshot: Some(vec![1.0, 2.0]),
        }
    }

    #[test]
    fn test_event_severity_mapping() {
        assert_eq!(
            EventSeverity::from_confidence(0.96),
            EventSeverity::Critical
        );
        assert_eq!(EventSeverity::from_confidence(0.85), EventSeverity::Warning);
        assert_eq!(EventSeverity::from_confidence(0.50), EventSeverity::Info);
    }

    #[test]
    fn test_event_to_payload() {
        let event = make_test_event();
        let payload = event_to_payload(&event, "test-source", None);
        assert_eq!(payload.source, "test-source");
        assert_eq!(payload.event_id, 1);
        assert_eq!(payload.severity, EventSeverity::Critical);
        assert_eq!(payload.confidence, 0.96);
        assert!(payload.data_snapshot.is_some());
    }

    #[test]
    fn test_simulated_publisher_connect_publish_disconnect() {
        let mut pubr = SimulatedMqttPublisher::new();
        let cfg = MqttConfig::default();
        pubr.connect(&cfg).unwrap();
        assert!(pubr.is_connected());

        pubr.publish("test/topic", b"hello", 1, false).unwrap();
        assert_eq!(pubr.publish_count, 1);
        assert_eq!(pubr.messages.len(), 1);
        assert_eq!(pubr.messages[0].0, "test/topic");

        pubr.disconnect().unwrap();
        assert!(!pubr.is_connected());
    }

    #[test]
    fn test_publish_event() {
        let mut pubr = SimulatedMqttPublisher::new();
        pubr.connect(&MqttConfig::default()).unwrap();

        let event = make_test_event();
        let payload = event_to_payload(&event, "motor1", None);
        pubr.publish_event(&payload).unwrap();

        assert_eq!(pubr.publish_count, 1);
        let topic = &pubr.messages[0].0;
        assert!(topic.contains("motor1"));
        assert!(topic.contains("critical"));
    }

    #[test]
    fn test_count_by_severity() {
        let mut pubr = SimulatedMqttPublisher::new();
        pubr.connect(&MqttConfig::default()).unwrap();

        for severity in [
            EventSeverity::Info,
            EventSeverity::Warning,
            EventSeverity::Critical,
        ]
        {
            let payload = EventPayload {
                source: "test".to_string(),
                timestamp: "T00:00:00.000Z".to_string(),
                event_id: 1,
                label_en: "test".to_string(),
                label_fr: "test".to_string(),
                confidence: 0.9,
                data_snapshot: None,
                severity,
                metadata: None,
            };
            pubr.publish_event(&payload).unwrap();
        }

        let (info, warn, crit) = pubr.count_by_severity();
        assert_eq!(info, 1);
        assert_eq!(warn, 1);
        assert_eq!(crit, 1);
    }

    #[test]
    fn test_publish_not_connected_errors() {
        let mut pubr = SimulatedMqttPublisher::new();
        let result = pubr.publish("test", b"data", 1, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparkplug_birth_certificate() {
        let cert = sparkplug_birth_certificate("g1", "n1", "d1");
        assert!(cert["uuid"].as_str().unwrap().contains("g1_n1_d1"));
        assert!(!cert["metrics"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_filter_by_confidence() {
        let events = vec![
            Event {
                id: 1,
                timestamp: 0.0,
                label_en: "a".into(),
                label_fr: "a".into(),
                confidence: 0.5,
                data_snapshot: None,
            },
            Event {
                id: 2,
                timestamp: 0.0,
                label_en: "b".into(),
                label_fr: "b".into(),
                confidence: 0.9,
                data_snapshot: None,
            },
            Event {
                id: 3,
                timestamp: 0.0,
                label_en: "c".into(),
                label_fr: "c".into(),
                confidence: 0.95,
                data_snapshot: None,
            },
        ];
        let filtered = filter_by_confidence(&events, 0.8);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].id, 2);
    }

    #[test]
    fn test_publish_events_batch() {
        let mut pubr = SimulatedMqttPublisher::new();
        pubr.connect(&MqttConfig::default()).unwrap();
        let events = vec![make_test_event(), make_test_event()];
        let count = publish_events(&mut pubr, &events, "station1", None).unwrap();
        assert_eq!(count, 2);
        assert_eq!(pubr.publish_count, 2);
    }
}
