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
    // Full UTC ISO 8601 timestamp: `YYYY-MM-DDThh:mm:ss.sssZ`.
    //
    // `ts` is a Unix timestamp (seconds since 1970-01-01T00:00:00Z), so the
    // rendered string must carry the *date* as well as the time-of-day —
    // emitting time-of-day only (and wrapping modulo 24h) would drop the day
    // and mislabel any timestamp beyond the first day.
    //
    // Round to whole milliseconds *first*, then decompose, so that a
    // fractional part rounding up to 1000 ms carries into the seconds field
    // instead of being emitted as an invalid 4-digit ".1000Z". Using
    // div/rem_euclid keeps every field non-negative even for pre-epoch
    // (negative) inputs.
    let total_millis = (ts * 1000.0).round() as i64;
    let total_secs = total_millis.div_euclid(1000);
    let millis = total_millis.rem_euclid(1000);

    let days = total_secs.div_euclid(86_400);
    let secs_of_day = total_secs.rem_euclid(86_400);
    let hours = secs_of_day / 3600;
    let minutes = (secs_of_day % 3600) / 60;
    let seconds = secs_of_day % 60;

    let (year, month, day) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hours, minutes, seconds, millis
    )
}

/// Convert a count of days since the Unix epoch (1970-01-01) into the
/// proleptic Gregorian `(year, month, day)`, where `month` is 1..=12.
///
/// Based on Howard Hinnant's `civil_from_days` algorithm; valid for the full
/// range of `i64` day counts and fully deterministic.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // Shift the epoch so the era starts on a 400-year boundary (0000-03-01).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11] (March = 0)
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day)
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
    /// Number of messages successfully published (failed publishes — e.g.
    /// while disconnected or with an empty topic — are *not* counted and are
    /// not recorded in `messages`). Always equals `messages.len()`.
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

    /// Count events by severity, returned as `(info, warning, critical)`.
    ///
    /// Event topics have the shape `{base_topic}/{source}/{severity}`, so the
    /// severity is always the final `/`-delimited segment. Matching that
    /// trailing segment exactly avoids miscounting a source name that happens
    /// to contain a severity word (e.g. a `line3-critical-spindle/info` topic
    /// must count as *info*, not *critical*).
    pub fn count_by_severity(&self) -> (usize, usize, usize) {
        let mut info = 0;
        let mut warn = 0;
        let mut crit = 0;
        for (topic, _, _, _) in &self.messages
        {
            match topic.rsplit('/').next().unwrap_or("")
            {
                "critical" => crit += 1,
                "warning" => warn += 1,
                "info" => info += 1,
                _ =>
                {},
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
        // A fresh publisher is not connected; publishing must fail with the
        // exact broker error and must not record or count anything.
        let result = pubr.publish("test", b"data", 1, false);
        assert_eq!(result, Err("Not connected to broker".to_string()));
        assert_eq!(pubr.last_error.as_deref(), Some("Not connected"));
        assert_eq!(pubr.publish_count, 0);
        assert!(pubr.messages.is_empty());
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

    // -- Oracle tests: state machine, message accounting, flag fidelity ------

    #[test]
    fn test_connect_sets_connected_state() {
        let mut pubr = SimulatedMqttPublisher::new();
        // A fresh publisher starts disconnected.
        assert!(!pubr.is_connected());
        pubr.connect(&MqttConfig::default()).unwrap();
        assert!(pubr.is_connected());
    }

    #[test]
    fn test_publish_increments_count_by_exactly_n() {
        let mut pubr = SimulatedMqttPublisher::new();
        pubr.connect(&MqttConfig::default()).unwrap();
        // connect() resets the buffers, so both start at zero.
        assert_eq!(pubr.publish_count, 0);
        assert_eq!(pubr.messages.len(), 0);

        const N: u64 = 5;
        for i in 0..N
        {
            pubr.publish(&format!("t/{i}"), b"x", 0, false).unwrap();
        }
        // Exactly N successful publishes recorded and counted.
        assert_eq!(pubr.publish_count, N);
        assert_eq!(pubr.messages.len(), N as usize);
    }

    #[test]
    fn test_publish_preserves_topic_payload_qos_retain() {
        let mut pubr = SimulatedMqttPublisher::new();
        pubr.connect(&MqttConfig::default()).unwrap();

        // Distinct values per field so a swap/drop bug would be caught.
        pubr.publish("plant/line3/temp", &[0xDE, 0xAD], 2, true)
            .unwrap();
        pubr.publish("plant/line3/vibe", &[0xBE, 0xEF, 0x01], 0, false)
            .unwrap();

        assert_eq!(pubr.messages.len(), 2);

        let (topic0, payload0, qos0, retain0) = &pubr.messages[0];
        assert_eq!(topic0, "plant/line3/temp");
        assert_eq!(payload0.as_slice(), &[0xDE, 0xAD]);
        assert_eq!(*qos0, 2);
        assert!(*retain0);

        let (topic1, payload1, qos1, retain1) = &pubr.messages[1];
        assert_eq!(topic1, "plant/line3/vibe");
        assert_eq!(payload1.as_slice(), &[0xBE, 0xEF, 0x01]);
        assert_eq!(*qos1, 0);
        assert!(!*retain1);
    }

    #[test]
    fn test_publish_empty_topic_errors_without_recording() {
        let mut pubr = SimulatedMqttPublisher::new();
        pubr.connect(&MqttConfig::default()).unwrap();
        let result = pubr.publish("", b"data", 1, false);
        assert_eq!(result, Err("Topic cannot be empty".to_string()));
        assert_eq!(pubr.last_error.as_deref(), Some("Empty topic"));
        // Failed publish must not be counted or recorded.
        assert_eq!(pubr.publish_count, 0);
        assert!(pubr.messages.is_empty());
    }

    #[test]
    fn test_disconnect_resets_connected_and_blocks_publish() {
        let mut pubr = SimulatedMqttPublisher::new();
        pubr.connect(&MqttConfig::default()).unwrap();
        pubr.publish("t/a", b"x", 1, false).unwrap();
        assert!(pubr.is_connected());
        assert_eq!(pubr.publish_count, 1);

        pubr.disconnect().unwrap();
        assert!(!pubr.is_connected());

        // After disconnect, publishing is rejected and nothing new is recorded.
        let result = pubr.publish("t/b", b"y", 1, false);
        assert_eq!(result, Err("Not connected to broker".to_string()));
        assert_eq!(pubr.publish_count, 1);
        assert_eq!(pubr.messages.len(), 1);
    }

    #[test]
    fn test_publish_event_after_disconnect_errors() {
        let mut pubr = SimulatedMqttPublisher::new();
        pubr.connect(&MqttConfig::default()).unwrap();
        pubr.disconnect().unwrap();

        // config is still Some after disconnect, so the not-connected guard in
        // publish() (not the not-configured guard) must be what rejects this.
        let payload = event_to_payload(&make_test_event(), "motor1", None);
        let result = pubr.publish_event(&payload);
        assert_eq!(result, Err("Not connected to broker".to_string()));
        assert!(pubr.messages.is_empty());
    }

    #[test]
    fn test_reconnect_clears_previous_messages() {
        let mut pubr = SimulatedMqttPublisher::new();
        pubr.connect(&MqttConfig::default()).unwrap();
        pubr.publish("t/a", b"x", 1, false).unwrap();
        assert_eq!(pubr.messages.len(), 1);

        // A second connect() must reset the message buffer and counter.
        pubr.connect(&MqttConfig::default()).unwrap();
        assert!(pubr.is_connected());
        assert_eq!(pubr.publish_count, 0);
        assert!(pubr.messages.is_empty());
    }

    #[test]
    fn test_publish_event_uses_config_qos_and_no_retain() {
        let mut pubr = SimulatedMqttPublisher::new();
        let cfg = MqttConfig {
            qos: 2,
            base_topic: "scirust/events".to_string(),
            ..MqttConfig::default()
        };
        pubr.connect(&cfg).unwrap();

        let event = make_test_event(); // confidence 0.96 -> Critical
        let payload = event_to_payload(&event, "motor1", None);
        pubr.publish_event(&payload).unwrap();

        let (topic, recorded, qos, retain) = &pubr.messages[0];
        // Topic = {base}/{source}/{severity-suffix}.
        assert_eq!(topic, "scirust/events/motor1/critical");
        // QoS comes from config; events are never retained.
        assert_eq!(*qos, 2);
        assert!(!*retain);
        // Recorded payload round-trips back to the same event payload.
        let decoded: EventPayload = serde_json::from_slice(recorded).unwrap();
        assert_eq!(decoded.source, "motor1");
        assert_eq!(decoded.event_id, event.id);
        assert_eq!(decoded.severity, EventSeverity::Critical);
        assert_eq!(decoded.confidence, event.confidence);
    }

    #[test]
    fn test_event_payload_json_roundtrip_preserves_fields() {
        let event = make_test_event();
        let meta = serde_json::json!({ "bearing_fault": "BPFO" });
        let payload = event_to_payload(&event, "spindle", Some(meta.clone()));

        let bytes = serde_json::to_vec(&payload).unwrap();
        let back: EventPayload = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(back.source, "spindle");
        assert_eq!(back.event_id, 1);
        assert_eq!(back.label_en, "spike");
        assert_eq!(back.label_fr, "pic");
        assert_eq!(back.confidence, 0.96);
        assert_eq!(back.data_snapshot, Some(vec![1.0, 2.0]));
        assert_eq!(back.severity, EventSeverity::Critical);
        assert_eq!(back.metadata, Some(meta));
        // UPPERCASE severity rename is honored on the wire.
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("\"severity\":\"CRITICAL\""));
    }

    // -- Regression: count_by_severity matches the trailing topic segment ----

    #[test]
    fn test_count_by_severity_handles_source_named_like_severity() {
        let mut pubr = SimulatedMqttPublisher::new();
        pubr.connect(&MqttConfig::default()).unwrap();

        // Sources whose names embed a severity word must be classified by the
        // *suffix*, not by a substring of the source segment.
        // info x2, warning x2, critical x2 by hand.
        let topics = [
            "scirust/events/motor1/critical",
            "scirust/events/info-sensor/critical",
            "scirust/events/motor1/warning",
            "scirust/events/warning-light/warning",
            "scirust/events/motor1/info",
            "scirust/events/critical-pump/info",
        ];
        for t in topics
        {
            pubr.publish(t, b"{}", 1, false).unwrap();
        }

        let (info, warn, crit) = pubr.count_by_severity();
        assert_eq!((info, warn, crit), (2, 2, 2));
    }

    // -- Regression: millisecond rounding carries into seconds ---------------

    #[test]
    fn test_format_timestamp_values() {
        // Full UTC ISO 8601 timestamps, hand-derived from the Unix epoch.
        assert_eq!(format_unix_timestamp(0.0), "1970-01-01T00:00:00.000Z");
        assert_eq!(format_unix_timestamp(1000.5), "1970-01-01T00:16:40.500Z");
        assert_eq!(format_unix_timestamp(3661.25), "1970-01-01T01:01:01.250Z");
        assert_eq!(format_unix_timestamp(86399.999), "1970-01-01T23:59:59.999Z");
    }

    #[test]
    fn test_format_timestamp_millis_carry() {
        // Fractional parts that round up to 1000 ms must carry into the next
        // second and never emit an invalid 4-digit ".1000Z".
        // 1000.9999 s = 16 min 40.9999 s -> rounds to 16:41.000.
        assert_eq!(format_unix_timestamp(1000.9999), "1970-01-01T00:16:41.000Z");
        // 59.9996 s -> rounds to 1 min 00.000 s.
        assert_eq!(format_unix_timestamp(59.9996), "1970-01-01T00:01:00.000Z");
        // 3599.9999 s = 59 min 59.9999 s -> rounds to 1 h 00:00.000.
        assert_eq!(format_unix_timestamp(3599.9999), "1970-01-01T01:00:00.000Z");
        // 86399.9999 s carries the seconds -> minutes -> hours -> *day*.
        assert_eq!(
            format_unix_timestamp(86399.9999),
            "1970-01-02T00:00:00.000Z"
        );
        // No produced millisecond field ever has 4 digits.
        for ts in [0.0_f64, 1000.9999, 59.9996, 3599.9999, 86399.9999]
        {
            let s = format_unix_timestamp(ts);
            let frac = s.trim_start_matches(|c| c != '.').trim_matches(['.', 'Z']);
            assert_eq!(frac.len(), 3, "millis field not 3 digits for ts={ts}: {s}");
        }
    }

    // -- Regression: timestamp is a full ISO 8601 date+time, not time-of-day --

    #[test]
    fn test_format_timestamp_includes_date_and_does_not_wrap_24h() {
        // A real-world Unix timestamp (2023-11-14T22:13:20.123Z). The buggy
        // implementation emitted only the time-of-day wrapped modulo 24h
        // ("T22:13:20.123Z"), silently dropping ~19700 days of date.
        assert_eq!(
            format_unix_timestamp(1_700_000_000.123),
            "2023-11-14T22:13:20.123Z"
        );

        // Two timestamps exactly 24h apart must differ in the *date* field, not
        // collapse to the same string as they did under modulo-24h wrapping.
        let day0 = format_unix_timestamp(1_700_000_000.0);
        let day1 = format_unix_timestamp(1_700_000_000.0 + 86_400.0);
        assert_ne!(day0, day1);
        assert_eq!(day0, "2023-11-14T22:13:20.000Z");
        assert_eq!(day1, "2023-11-15T22:13:20.000Z");

        // The output must be a parseable ISO 8601 calendar timestamp: a
        // 4-digit year, `-`-separated date, `T`, time, and trailing `Z`.
        let s = format_unix_timestamp(1_700_000_000.123);
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[7..8], "-");
        assert_eq!(&s[10..11], "T");
        assert!(s.ends_with('Z'));
        assert_eq!(s.len(), "1970-01-01T00:00:00.000Z".len());
    }

    #[test]
    fn test_format_timestamp_pre_epoch() {
        // Negative (pre-1970) timestamps must borrow across the day boundary
        // and never produce negative fields.
        assert_eq!(format_unix_timestamp(-1.0), "1969-12-31T23:59:59.000Z");
        assert_eq!(format_unix_timestamp(-0.5), "1969-12-31T23:59:59.500Z");
    }
}
