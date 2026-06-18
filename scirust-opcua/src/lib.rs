//! SciRust OPC-UA Bridge
//!
//! Connects industrial PLC/SCADA systems to the SciRust event detection pipeline.
//! Provides a trait-based abstraction (`OpcuaClient`) with a simulated backend
//! for development/testing. Ready for swap-in of a real OPC-UA protocol stack
//! (e.g., `opcua` crate) in production.
//!
//! ## Architecture
//! ```text
//! PLC/SCADA -> OpcuaClient -> EventStream -> EventDetector -> Events
//! ```

use scirust_events_core::EventStream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single OPC-UA variable node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpcuaNode {
    /// OPC-UA NodeId as a string (e.g. "ns=2;s=Motor.Vibration")
    pub node_id: String,
    /// Human-readable display name
    pub display_name: String,
    /// Engineering unit (e.g. "m/s²", "°C", "bar")
    pub unit: String,
    /// Description / purpose
    pub description: String,
}

/// A snapshot of a variable's value at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpcuaValue {
    pub node_id: String,
    pub value: f64,
    pub timestamp: f64, // Unix timestamp in seconds
    pub quality: OpcuaQuality,
}

/// OPC-UA data quality indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpcuaQuality {
    Good,
    Uncertain,
    Bad,
}

impl OpcuaQuality {
    pub fn is_good(&self) -> bool {
        matches!(self, OpcuaQuality::Good)
    }
}

/// Configuration for an OPC-UA connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpcuaConfig {
    /// Endpoint URL (e.g. "opc.tcp://192.168.1.100:4840")
    pub endpoint: String,
    /// Application name for the OPC-UA session
    pub application_name: String,
    /// Session timeout in milliseconds
    pub session_timeout_ms: u32,
    /// Sampling interval for subscriptions in milliseconds
    pub sampling_interval_ms: f64,
}

impl Default for OpcuaConfig {
    fn default() -> Self {
        Self {
            endpoint: "opc.tcp://localhost:4840".to_string(),
            application_name: "SciRust-Monitor".to_string(),
            session_timeout_ms: 60_000,
            sampling_interval_ms: 100.0,
        }
    }
}

/// The main OPC-UA client abstraction.
///
/// Implement this trait to connect to real OPC-UA servers or to provide
/// simulated data for testing.
pub trait OpcuaClient {
    /// Connect to the OPC-UA server.
    fn connect(&mut self, config: &OpcuaConfig) -> Result<(), String>;

    /// Disconnect from the server.
    fn disconnect(&mut self) -> Result<(), String>;

    /// List available variable nodes matching a filter pattern.
    fn browse(&self, path_filter: &str) -> Result<Vec<OpcuaNode>, String>;

    /// Read the current value of a single node.
    fn read(&self, node_id: &str) -> Result<OpcuaValue, String>;

    /// Read current values of multiple nodes in one call.
    fn read_many(&self, node_ids: &[String]) -> Result<Vec<OpcuaValue>, String> {
        node_ids.iter().map(|id| self.read(id)).collect()
    }

    /// Subscribe to a set of nodes and return a time-ordered stream of values.
    /// The implementation should buffer values internally and return them
    /// when polled.
    fn subscribe(&mut self, node_ids: &[String]) -> Result<(), String>;

    /// Poll the subscription buffer for new values.
    fn poll_subscription(&mut self) -> Result<Vec<OpcuaValue>, String>;
}

/// Convert a batch of `OpcuaValue`s into a SciRust `EventStream`.
///
/// Groups values by timestamp, then produces a flat vector of numeric values
/// ordered by node_id consistently.
///
/// `values`: all values from a subscription poll window.
/// `node_order`: canonical node_id ordering (must match across calls).
/// `window_size`: sliding window size for the EventStream.
/// `stride`: sliding window stride.
pub fn values_to_event_stream(
    values: &[OpcuaValue],
    node_order: &[String],
    window_size: usize,
    stride: usize,
) -> EventStream {
    // Group latest values per node
    let mut latest: HashMap<&str, f64> = HashMap::new();
    for v in values
    {
        if v.quality.is_good()
        {
            latest.insert(&v.node_id, v.value);
        }
    }
    // Produce flat array in canonical node order
    let flat: Vec<f32> = node_order
        .iter()
        .map(|id| latest.get(id.as_str()).copied().unwrap_or(f64::NAN) as f32)
        .collect();
    EventStream::new(flat, window_size, stride)
}

// ---------------------------------------------------------------------------
// Simulated OPC-UA Client
// ---------------------------------------------------------------------------

/// A simulated OPC-UA client that generates synthetic sensor data.
///
/// Useful for development, testing, and CI without requiring a real PLC.
///
/// ## Sensor types simulated:
/// - **Vibration sensor** (random walk + periodic sine at shaft rate)
/// - **Temperature sensor** (slow drift + noise)
/// - **Pressure sensor** (step changes + noise)
/// - **Current sensor** (load-dependent sine + noise)
/// - **Flow sensor** (steady with occasional dips)
#[derive(Debug)]
pub struct SimulatedOpcuaClient {
    config: OpcuaConfig,
    connected: bool,
    nodes: Vec<OpcuaNode>,
    subscribed_nodes: Vec<String>,
    /// Buffer of simulated values waiting to be polled
    buffer: Vec<OpcuaValue>,
    /// Internal state for each sensor's simulation
    states: HashMap<String, SimulatedSensorState>,
    /// Monotonic timer (seconds)
    sim_time: f64,
}

#[derive(Debug, Clone)]
struct SimulatedSensorState {
    value: f64,
    last_value: f64,
    /// For random-walk sensors (vibration, temperature)
    trend: f64,
    /// For periodic sensors
    phase: f64,
    /// Sensor type
    sensor_type: SensorType,
}

#[derive(Debug, Clone, PartialEq)]
enum SensorType {
    Vibration,
    Temperature,
    Pressure,
    Current,
    Flow,
}

impl SimulatedSensorState {
    fn update(&mut self, dt: f64) -> f64 {
        self.last_value = self.value;
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let noise: f64 = rng.gen_range(-0.1..0.1);

        match self.sensor_type
        {
            SensorType::Vibration =>
            {
                // Random walk + 30 Hz sine (simulated shaft vibration)
                self.trend += rng.gen_range(-0.02..0.02);
                self.trend = self.trend.clamp(-1.0, 1.0);
                self.phase += dt * 30.0 * std::f64::consts::TAU;
                self.value = self.trend + 0.5 * self.phase.sin() + noise * 0.05;
            },
            SensorType::Temperature =>
            {
                // Slow drift toward setpoint
                let setpoint = 75.0;
                self.value += (setpoint - self.value) * 0.01 + noise * 0.02;
            },
            SensorType::Pressure =>
            {
                // Steps with noise, occasional drops (simulating actuator cycles)
                self.value = 6.0 + noise;
                // Inject a sudden drop every ~50 steps
                if rng.gen_bool(0.02)
                {
                    self.value -= rng.gen_range(2.0..4.0);
                }
                self.value = self.value.max(0.5);
            },
            SensorType::Current =>
            {
                // Load-dependent: sine at 50 Hz grid frequency + noise
                self.phase += dt * 50.0 * std::f64::consts::TAU;
                self.value = 150.0 * (0.5 + 0.3 * self.phase.sin()) + noise * 5.0;
            },
            SensorType::Flow =>
            {
                // Steady flow with occasional dips (simulating pump cavitation)
                self.value = 42.0 + noise * 0.5;
                if rng.gen_bool(0.01)
                {
                    self.value -= rng.gen_range(5.0..15.0);
                }
                self.value = self.value.max(5.0);
            },
        }
        self.value
    }
}

impl SimulatedOpcuaClient {
    pub fn new() -> Self {
        let nodes = vec![
            OpcuaNode {
                node_id: "ns=2;s=Vibration.X".to_string(),
                display_name: "Vibration X-axis".to_string(),
                unit: "m/s²".to_string(),
                description: "Accelerometer X-axis on spindle motor".to_string(),
            },
            OpcuaNode {
                node_id: "ns=2;s=Vibration.Y".to_string(),
                display_name: "Vibration Y-axis".to_string(),
                unit: "m/s²".to_string(),
                description: "Accelerometer Y-axis on spindle motor".to_string(),
            },
            OpcuaNode {
                node_id: "ns=2;s=Vibration.Z".to_string(),
                display_name: "Vibration Z-axis".to_string(),
                unit: "m/s²".to_string(),
                description: "Accelerometer Z-axis on spindle motor".to_string(),
            },
            OpcuaNode {
                node_id: "ns=2;s=Temperature.Motor".to_string(),
                display_name: "Motor Temperature".to_string(),
                unit: "°C".to_string(),
                description: "Winding temperature of main drive motor".to_string(),
            },
            OpcuaNode {
                node_id: "ns=2;s=Temperature.Coolant".to_string(),
                display_name: "Coolant Temperature".to_string(),
                unit: "°C".to_string(),
                description: "Coolant return temperature".to_string(),
            },
            OpcuaNode {
                node_id: "ns=2;s=Pressure.Hydraulic".to_string(),
                display_name: "Hydraulic Pressure".to_string(),
                unit: "bar".to_string(),
                description: "Main hydraulic circuit pressure".to_string(),
            },
            OpcuaNode {
                node_id: "ns=2;s=Current.Motor".to_string(),
                display_name: "Motor Current".to_string(),
                unit: "A".to_string(),
                description: "Motor phase current RMS".to_string(),
            },
            OpcuaNode {
                node_id: "ns=2;s=Flow.Coolant".to_string(),
                display_name: "Coolant Flow".to_string(),
                unit: "L/min".to_string(),
                description: "Coolant flow rate".to_string(),
            },
        ];

        let mut states = HashMap::new();
        states.insert(
            "ns=2;s=Vibration.X".to_string(),
            SimulatedSensorState {
                value: 0.0,
                last_value: 0.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Vibration,
            },
        );
        states.insert(
            "ns=2;s=Vibration.Y".to_string(),
            SimulatedSensorState {
                value: 0.0,
                last_value: 0.0,
                trend: 0.0,
                phase: 1.0,
                sensor_type: SensorType::Vibration,
            },
        );
        states.insert(
            "ns=2;s=Vibration.Z".to_string(),
            SimulatedSensorState {
                value: 0.0,
                last_value: 0.0,
                trend: 0.0,
                phase: 2.0,
                sensor_type: SensorType::Vibration,
            },
        );
        states.insert(
            "ns=2;s=Temperature.Motor".to_string(),
            SimulatedSensorState {
                value: 25.0,
                last_value: 25.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Temperature,
            },
        );
        states.insert(
            "ns=2;s=Temperature.Coolant".to_string(),
            SimulatedSensorState {
                value: 22.0,
                last_value: 22.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Temperature,
            },
        );
        states.insert(
            "ns=2;s=Pressure.Hydraulic".to_string(),
            SimulatedSensorState {
                value: 6.0,
                last_value: 6.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Pressure,
            },
        );
        states.insert(
            "ns=2;s=Current.Motor".to_string(),
            SimulatedSensorState {
                value: 100.0,
                last_value: 100.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Current,
            },
        );
        states.insert(
            "ns=2;s=Flow.Coolant".to_string(),
            SimulatedSensorState {
                value: 42.0,
                last_value: 42.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Flow,
            },
        );

        Self {
            config: OpcuaConfig::default(),
            connected: false,
            nodes,
            subscribed_nodes: Vec::new(),
            buffer: Vec::new(),
            states,
            sim_time: 0.0,
        }
    }

    /// Advance the simulation clock and generate new values.
    pub fn tick(&mut self) {
        if !self.connected || self.subscribed_nodes.is_empty()
        {
            return;
        }
        let dt = self.config.sampling_interval_ms / 1000.0;
        self.sim_time += dt;

        let mut values = Vec::with_capacity(self.subscribed_nodes.len());
        for node_id in &self.subscribed_nodes
        {
            if let Some(state) = self.states.get_mut(node_id)
            {
                let value = state.update(dt);
                values.push(OpcuaValue {
                    node_id: node_id.clone(),
                    value,
                    timestamp: self.sim_time,
                    quality: OpcuaQuality::Good,
                });
            }
        }
        self.buffer.extend(values);
    }
}

impl Default for SimulatedOpcuaClient {
    fn default() -> Self {
        Self::new()
    }
}

impl OpcuaClient for SimulatedOpcuaClient {
    fn connect(&mut self, config: &OpcuaConfig) -> Result<(), String> {
        self.config = config.clone();
        self.connected = true;
        self.sim_time = 0.0;
        Ok(())
    }

    fn disconnect(&mut self) -> Result<(), String> {
        self.connected = false;
        self.subscribed_nodes.clear();
        self.buffer.clear();
        Ok(())
    }

    fn browse(&self, path_filter: &str) -> Result<Vec<OpcuaNode>, String> {
        let filter = path_filter.to_lowercase();
        Ok(self
            .nodes
            .iter()
            .filter(|n| {
                n.node_id.to_lowercase().contains(&filter)
                    || n.display_name.to_lowercase().contains(&filter)
                    || n.unit.to_lowercase().contains(&filter)
            })
            .cloned()
            .collect())
    }

    fn read(&self, node_id: &str) -> Result<OpcuaValue, String> {
        if let Some(state) = self.states.get(node_id)
        {
            Ok(OpcuaValue {
                node_id: node_id.to_string(),
                value: state.value,
                timestamp: self.sim_time,
                quality: OpcuaQuality::Good,
            })
        }
        else
        {
            Err(format!("Node not found: {}", node_id))
        }
    }

    fn subscribe(&mut self, node_ids: &[String]) -> Result<(), String> {
        for id in node_ids
        {
            if !self.states.contains_key(id.as_str())
            {
                return Err(format!("Unknown node: {}", id));
            }
        }
        self.subscribed_nodes = node_ids.to_vec();
        Ok(())
    }

    fn poll_subscription(&mut self) -> Result<Vec<OpcuaValue>, String> {
        self.tick();
        let values = std::mem::take(&mut self.buffer);
        Ok(values)
    }
}

// ---------------------------------------------------------------------------
// High-level bridge function
// ---------------------------------------------------------------------------

/// Run a complete OPC-UA → SciRust event detection loop.
///
/// 1. Connects to the OPC-UA server (or simulator)
/// 2. Browses for motor-related nodes
/// 3. Subscribes to them
/// 4. Runs `n_iterations` polling cycles
/// 5. Feeds each batch into an `EventStream`
/// 6. Returns all raw `OpcuaValue`s for downstream processing
///
/// The caller should feed the resulting `EventStream` into an `EventDetector`.
pub fn run_opcua_loop(
    client: &mut dyn OpcuaClient,
    config: &OpcuaConfig,
    node_pattern: &str,
    n_iterations: usize,
) -> Result<Vec<Vec<OpcuaValue>>, String> {
    client.connect(config)?;

    let nodes = client.browse(node_pattern)?;
    if nodes.is_empty()
    {
        return Err(format!("No nodes matching pattern '{}'", node_pattern));
    }

    let node_ids: Vec<String> = nodes.iter().map(|n| n.node_id.clone()).collect();
    client.subscribe(&node_ids)?;

    let mut batches = Vec::with_capacity(n_iterations);

    for _ in 0..n_iterations
    {
        let values = client.poll_subscription()?;
        if !values.is_empty()
        {
            batches.push(values);
        }
    }

    client.disconnect()?;
    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_client_connect_disconnect() {
        let mut client = SimulatedOpcuaClient::new();
        let cfg = OpcuaConfig::default();
        client.connect(&cfg).unwrap();
        assert!(client.connected);
        client.disconnect().unwrap();
        assert!(!client.connected);
    }

    #[test]
    fn test_simulated_client_browse() {
        let client = SimulatedOpcuaClient::new();
        let results = client.browse("vibration").unwrap();
        assert_eq!(results.len(), 3); // X, Y, Z
    }

    #[test]
    fn test_simulated_client_subscribe_and_poll() {
        let mut client = SimulatedOpcuaClient::new();
        client.connect(&OpcuaConfig::default()).unwrap();

        let nodes = client.browse("vibration").unwrap();
        let ids: Vec<String> = nodes.iter().map(|n| n.node_id.clone()).collect();
        client.subscribe(&ids).unwrap();

        let values = client.poll_subscription().unwrap();
        assert!(!values.is_empty());
        // All values should have Good quality
        for v in &values
        {
            assert!(v.quality.is_good());
        }
    }

    #[test]
    fn test_values_to_event_stream() {
        let values = vec![
            OpcuaValue {
                node_id: "a".to_string(),
                value: 1.0,
                timestamp: 0.0,
                quality: OpcuaQuality::Good,
            },
            OpcuaValue {
                node_id: "b".to_string(),
                value: 2.0,
                timestamp: 0.0,
                quality: OpcuaQuality::Good,
            },
            OpcuaValue {
                node_id: "a".to_string(),
                value: 1.5,
                timestamp: 0.1,
                quality: OpcuaQuality::Bad,
            }, // bad quality, ignored
        ];
        let order = vec!["a".to_string(), "b".to_string()];
        let stream = values_to_event_stream(&values, &order, 2, 1);
        assert_eq!(stream.data, vec![1.0, 2.0]); // latest good a=1.0, b=2.0
    }

    #[test]
    fn test_run_opcua_loop() {
        let mut client = SimulatedOpcuaClient::new();
        let cfg = OpcuaConfig::default();
        let batches = run_opcua_loop(&mut client, &cfg, "vibration", 3).unwrap();
        assert!(!batches.is_empty(), "should have received data batches");
    }

    #[test]
    fn test_opcua_quality() {
        assert!(OpcuaQuality::Good.is_good());
        assert!(!OpcuaQuality::Bad.is_good());
        assert!(!OpcuaQuality::Uncertain.is_good());
    }
}
