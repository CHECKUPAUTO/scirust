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

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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

    /// Report whether the client currently has a usable OPC-UA session.
    ///
    /// The default is deliberately fail-closed so existing third-party
    /// implementations keep compiling but cannot be presented as ready until
    /// they expose their real connection state.
    fn is_connected(&self) -> bool {
        false
    }

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
    // Keep, per node, the latest good sample *by timestamp* — not merely the
    // last one encountered — so the result is correct even if `values` is not
    // already time-ordered. Bad/uncertain samples never overwrite a good one.
    let mut latest: HashMap<&str, (f64, f64)> = HashMap::new();
    for v in values
    {
        if v.quality.is_good()
        {
            latest
                .entry(v.node_id.as_str())
                .and_modify(|(ts, val)| {
                    if v.timestamp >= *ts
                    {
                        *ts = v.timestamp;
                        *val = v.value;
                    }
                })
                .or_insert((v.timestamp, v.value));
        }
    }
    // Produce flat array in canonical node order
    let flat: Vec<f32> = node_order
        .iter()
        .map(|id| {
            latest
                .get(id.as_str())
                .map(|(_, val)| *val)
                .unwrap_or(f64::NAN) as f32
        })
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
    /// Seed used to (re)initialize the deterministic generator.
    seed: u64,
    /// Deterministic generator backing all synthetic noise/events.
    rng: StdRng,
}

#[derive(Debug, Clone)]
struct SimulatedSensorState {
    value: f64,
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
    /// Advance the sensor model by `dt` seconds, drawing all randomness from the
    /// supplied seeded generator so the simulation is fully deterministic for a
    /// given seed and call sequence.
    fn update(&mut self, dt: f64, rng: &mut StdRng) -> f64 {
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

/// Default seed for the deterministic simulator. Fixed so that, absent an
/// explicit override, every process produces the identical synthetic stream.
const DEFAULT_SIM_SEED: u64 = 0x5C1A_0C4A_DEAD_BEEF_u64;

impl SimulatedOpcuaClient {
    /// Create a simulator seeded with [`DEFAULT_SIM_SEED`].
    pub fn new() -> Self {
        Self::with_seed(DEFAULT_SIM_SEED)
    }

    /// Create a simulator with an explicit seed. Two clients built with the same
    /// seed and driven through the same calls produce byte-identical values.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            config: OpcuaConfig::default(),
            connected: false,
            nodes: Self::catalog_nodes(),
            subscribed_nodes: Vec::new(),
            buffer: Vec::new(),
            states: Self::initial_states(),
            sim_time: 0.0,
            seed,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// The fixed catalog of variable nodes exposed by the simulator.
    fn catalog_nodes() -> Vec<OpcuaNode> {
        vec![
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
        ]
    }

    /// The pristine per-sensor state, used both at construction and on every
    /// fresh `connect` so a reconnection restarts an identical deterministic run.
    fn initial_states() -> HashMap<String, SimulatedSensorState> {
        let mut states = HashMap::new();
        states.insert(
            "ns=2;s=Vibration.X".to_string(),
            SimulatedSensorState {
                value: 0.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Vibration,
            },
        );
        states.insert(
            "ns=2;s=Vibration.Y".to_string(),
            SimulatedSensorState {
                value: 0.0,
                trend: 0.0,
                phase: 1.0,
                sensor_type: SensorType::Vibration,
            },
        );
        states.insert(
            "ns=2;s=Vibration.Z".to_string(),
            SimulatedSensorState {
                value: 0.0,
                trend: 0.0,
                phase: 2.0,
                sensor_type: SensorType::Vibration,
            },
        );
        states.insert(
            "ns=2;s=Temperature.Motor".to_string(),
            SimulatedSensorState {
                value: 25.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Temperature,
            },
        );
        states.insert(
            "ns=2;s=Temperature.Coolant".to_string(),
            SimulatedSensorState {
                value: 22.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Temperature,
            },
        );
        states.insert(
            "ns=2;s=Pressure.Hydraulic".to_string(),
            SimulatedSensorState {
                value: 6.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Pressure,
            },
        );
        states.insert(
            "ns=2;s=Current.Motor".to_string(),
            SimulatedSensorState {
                value: 100.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Current,
            },
        );
        states.insert(
            "ns=2;s=Flow.Coolant".to_string(),
            SimulatedSensorState {
                value: 42.0,
                trend: 0.0,
                phase: 0.0,
                sensor_type: SensorType::Flow,
            },
        );
        states
    }

    /// Advance the simulation clock and generate new values.
    pub fn tick(&mut self) {
        if !self.connected || self.subscribed_nodes.is_empty()
        {
            return;
        }
        let dt = self.config.sampling_interval_ms / 1000.0;
        self.sim_time += dt;

        // Borrow disjoint fields explicitly so the seeded RNG can be threaded
        // through each per-node update while we read the subscription order.
        let SimulatedOpcuaClient {
            subscribed_nodes,
            states,
            rng,
            buffer,
            sim_time,
            ..
        } = self;

        for node_id in subscribed_nodes.iter()
        {
            if let Some(state) = states.get_mut(node_id)
            {
                let value = state.update(dt, rng);
                buffer.push(OpcuaValue {
                    node_id: node_id.clone(),
                    value,
                    timestamp: *sim_time,
                    quality: OpcuaQuality::Good,
                });
            }
        }
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
        // Restore a pristine, reproducible simulation: same seed, same initial
        // sensor states, clock back to zero, no carried-over subscription/buffer.
        self.sim_time = 0.0;
        self.rng = StdRng::seed_from_u64(self.seed);
        self.states = Self::initial_states();
        self.subscribed_nodes.clear();
        self.buffer.clear();
        Ok(())
    }

    fn disconnect(&mut self) -> Result<(), String> {
        self.connected = false;
        self.subscribed_nodes.clear();
        self.buffer.clear();
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
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

    /// Every node the simulator is expected to expose, in catalog order.
    const ALL_NODE_IDS: [&str; 8] = [
        "ns=2;s=Vibration.X",
        "ns=2;s=Vibration.Y",
        "ns=2;s=Vibration.Z",
        "ns=2;s=Temperature.Motor",
        "ns=2;s=Temperature.Coolant",
        "ns=2;s=Pressure.Hydraulic",
        "ns=2;s=Current.Motor",
        "ns=2;s=Flow.Coolant",
    ];

    fn ids(slice: &[&str]) -> Vec<String> {
        slice.iter().map(|s| s.to_string()).collect()
    }

    /// Drive a freshly-connected client through `n` polls of every node and
    /// return the flattened value stream.
    fn run_all_nodes(seed: u64, n: usize) -> Vec<OpcuaValue> {
        let mut client = SimulatedOpcuaClient::with_seed(seed);
        client.connect(&OpcuaConfig::default()).unwrap();
        client.subscribe(&ids(&ALL_NODE_IDS)).unwrap();
        let mut out = Vec::new();
        for _ in 0..n
        {
            out.extend(client.poll_subscription().unwrap());
        }
        out
    }

    // --- connection lifecycle --------------------------------------------

    #[test]
    fn connect_then_is_connected_then_disconnect_flips_state() {
        let mut client = SimulatedOpcuaClient::new();
        assert!(!client.connected, "starts disconnected");

        client.connect(&OpcuaConfig::default()).unwrap();
        assert!(client.connected, "connect() must report connected");

        client.disconnect().unwrap();
        assert!(!client.connected, "disconnect() must clear connected");
    }

    #[test]
    fn connect_records_supplied_config() {
        let mut client = SimulatedOpcuaClient::new();
        let cfg = OpcuaConfig {
            endpoint: "opc.tcp://plc.local:4840".to_string(),
            application_name: "unit-test".to_string(),
            session_timeout_ms: 12_345,
            sampling_interval_ms: 250.0,
        };
        client.connect(&cfg).unwrap();
        assert_eq!(client.config.sampling_interval_ms, 250.0);
        assert_eq!(client.config.session_timeout_ms, 12_345);
        assert_eq!(client.config.endpoint, "opc.tcp://plc.local:4840");
    }

    #[test]
    fn disconnect_clears_subscription_and_buffer() {
        let mut client = SimulatedOpcuaClient::new();
        client.connect(&OpcuaConfig::default()).unwrap();
        client.subscribe(&ids(&["ns=2;s=Vibration.X"])).unwrap();
        client.tick(); // populate the buffer
        assert!(!client.buffer.is_empty());

        client.disconnect().unwrap();
        assert!(client.subscribed_nodes.is_empty(), "subscription cleared");
        assert!(client.buffer.is_empty(), "buffer drained on disconnect");
    }

    // --- browse ----------------------------------------------------------

    #[test]
    fn browse_vibration_returns_exactly_the_three_axes() {
        let client = SimulatedOpcuaClient::new();
        let results = client.browse("vibration").unwrap();

        let got_ids: Vec<&str> = results.iter().map(|n| n.node_id.as_str()).collect();
        assert_eq!(
            got_ids,
            vec![
                "ns=2;s=Vibration.X",
                "ns=2;s=Vibration.Y",
                "ns=2;s=Vibration.Z",
            ],
            "vibration filter must match the three accelerometer axes only"
        );

        let got_names: Vec<&str> = results.iter().map(|n| n.display_name.as_str()).collect();
        assert_eq!(
            got_names,
            vec!["Vibration X-axis", "Vibration Y-axis", "Vibration Z-axis"]
        );
    }

    #[test]
    fn browse_is_case_insensitive() {
        let client = SimulatedOpcuaClient::new();
        assert_eq!(client.browse("VIBRATION").unwrap().len(), 3);
        assert_eq!(client.browse("Vibration").unwrap().len(), 3);
    }

    #[test]
    fn browse_temperature_returns_motor_and_coolant() {
        let client = SimulatedOpcuaClient::new();
        let results = client.browse("temperature").unwrap();
        let got: Vec<&str> = results.iter().map(|n| n.node_id.as_str()).collect();
        assert_eq!(
            got,
            vec!["ns=2;s=Temperature.Motor", "ns=2;s=Temperature.Coolant"]
        );
    }

    #[test]
    fn browse_empty_filter_lists_every_node() {
        let client = SimulatedOpcuaClient::new();
        let results = client.browse("").unwrap();
        let got: Vec<&str> = results.iter().map(|n| n.node_id.as_str()).collect();
        assert_eq!(got, ALL_NODE_IDS.to_vec(), "empty filter is a wildcard");
    }

    #[test]
    fn browse_unknown_pattern_returns_empty() {
        let client = SimulatedOpcuaClient::new();
        assert!(client.browse("does-not-exist").unwrap().is_empty());
    }

    // --- read ------------------------------------------------------------

    #[test]
    fn read_unknown_node_errors() {
        let client = SimulatedOpcuaClient::new();
        let err = client.read("ns=2;s=Nope").unwrap_err();
        assert!(err.contains("Node not found"), "got: {err}");
    }

    #[test]
    fn read_reflects_initial_then_ticked_value() {
        let mut client = SimulatedOpcuaClient::with_seed(7);
        client.connect(&OpcuaConfig::default()).unwrap();

        // Before any tick, Temperature.Motor sits at its documented start (25 °C).
        let before = client.read("ns=2;s=Temperature.Motor").unwrap();
        assert_eq!(before.value, 25.0);
        assert!(before.quality.is_good());

        client
            .subscribe(&ids(&["ns=2;s=Temperature.Motor"]))
            .unwrap();
        client.tick();

        // After a tick the read reflects the freshly advanced value, which has
        // drifted up toward the 75 °C setpoint.
        let after = client.read("ns=2;s=Temperature.Motor").unwrap();
        assert!(
            after.value > 25.0 && after.value < 26.0,
            "after one tick: {} (expected just above 25)",
            after.value
        );
    }

    // --- subscribe / poll ------------------------------------------------

    #[test]
    fn subscribe_rejects_unknown_node() {
        let mut client = SimulatedOpcuaClient::new();
        client.connect(&OpcuaConfig::default()).unwrap();
        let err = client
            .subscribe(&ids(&["ns=2;s=Vibration.X", "ns=2;s=Bogus"]))
            .unwrap_err();
        assert!(err.contains("Unknown node"), "got: {err}");
        // A failed subscribe must not have partially registered anything.
        assert!(client.subscribed_nodes.is_empty());
    }

    #[test]
    fn poll_yields_one_value_per_subscribed_node_all_good() {
        let mut client = SimulatedOpcuaClient::with_seed(1);
        client.connect(&OpcuaConfig::default()).unwrap();
        let subscribed = ids(&ALL_NODE_IDS);
        client.subscribe(&subscribed).unwrap();

        let values = client.poll_subscription().unwrap();
        assert_eq!(
            values.len(),
            subscribed.len(),
            "one sample per subscribed node per poll"
        );
        let got: Vec<&str> = values.iter().map(|v| v.node_id.as_str()).collect();
        assert_eq!(got, ALL_NODE_IDS.to_vec(), "samples in subscription order");
        for v in &values
        {
            assert!(v.quality.is_good(), "{} not Good", v.node_id);
        }
    }

    #[test]
    fn poll_timestamp_advances_by_sampling_interval() {
        let mut client = SimulatedOpcuaClient::with_seed(3);
        client.connect(&OpcuaConfig::default()).unwrap();
        client.subscribe(&ids(&["ns=2;s=Vibration.X"])).unwrap();
        // Default sampling interval is 100 ms => 0.1 s of sim-time per poll.
        for step in 1..=5u32
        {
            let v = client.poll_subscription().unwrap();
            let expected = 0.1 * step as f64;
            assert!(
                (v[0].timestamp - expected).abs() < 1e-9,
                "step {step}: ts={} expected {expected}",
                v[0].timestamp
            );
        }
    }

    #[test]
    fn poll_without_connection_yields_nothing() {
        let mut client = SimulatedOpcuaClient::new();
        // Never connected: tick must be a no-op, poll returns an empty batch.
        assert!(client.poll_subscription().unwrap().is_empty());
    }

    #[test]
    fn poll_without_subscription_yields_nothing() {
        let mut client = SimulatedOpcuaClient::new();
        client.connect(&OpcuaConfig::default()).unwrap();
        assert!(client.poll_subscription().unwrap().is_empty());
    }

    // --- simulated value ranges (derived by hand from the generators) ----

    #[test]
    fn simulated_values_stay_within_documented_ranges() {
        // 200 polls of every sensor; assert each stays inside the analytic
        // envelope of its generator.
        let values = run_all_nodes(11, 200);
        assert!(values.len() >= 8 * 200);

        for v in &values
        {
            let x = v.value;
            assert!(v.quality.is_good());
            match v.node_id.as_str()
            {
                // trend in [-1,1] + 0.5*sin in [-0.5,0.5] + noise*0.05 in
                // [-0.005,0.005) => |x| < 1.51
                "ns=2;s=Vibration.X" | "ns=2;s=Vibration.Y" | "ns=2;s=Vibration.Z" =>
                {
                    assert!(x.abs() < 1.51, "vibration out of range: {x}");
                },
                // Starts at 25/22, drifts toward 75 setpoint, small noise.
                "ns=2;s=Temperature.Motor" | "ns=2;s=Temperature.Coolant" =>
                {
                    assert!((21.0..=75.5).contains(&x), "temperature out of range: {x}");
                },
                // 6.0 + noise[-0.1,0.1), optional drop [2,4], clamped to >= 0.5.
                "ns=2;s=Pressure.Hydraulic" =>
                {
                    assert!((0.5..6.1).contains(&x), "pressure out of range: {x}");
                },
                // 75 + 45*sin + noise*5 in [29.5, 120.5).
                "ns=2;s=Current.Motor" =>
                {
                    assert!((29.5..120.5).contains(&x), "current out of range: {x}");
                },
                // 42 + noise*0.5, optional dip [5,15], clamped to >= 5.0.
                "ns=2;s=Flow.Coolant" =>
                {
                    assert!((5.0..42.1).contains(&x), "flow out of range: {x}");
                },
                other => panic!("unexpected node in stream: {other}"),
            }
        }
    }

    #[test]
    fn temperature_drifts_monotonically_toward_setpoint() {
        // With noise*0.02 dwarfed by the 1%-of-gap pull while far from 75 °C,
        // the motor temperature must climb on every early step.
        let mut client = SimulatedOpcuaClient::with_seed(123);
        client.connect(&OpcuaConfig::default()).unwrap();
        client
            .subscribe(&ids(&["ns=2;s=Temperature.Motor"]))
            .unwrap();

        let mut prev = 25.0;
        for _ in 0..20
        {
            let v = client.poll_subscription().unwrap();
            let cur = v[0].value;
            assert!(cur > prev, "temperature should rise: {prev} -> {cur}");
            assert!(cur < 75.0, "must not overshoot setpoint: {cur}");
            prev = cur;
        }
    }

    // --- determinism -----------------------------------------------------

    #[test]
    fn same_seed_produces_bit_identical_streams() {
        let a = run_all_nodes(2024, 25);
        let b = run_all_nodes(2024, 25);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter())
        {
            assert_eq!(x.node_id, y.node_id);
            // Exact bit equality: the seeded generator must be reproducible.
            assert_eq!(
                x.value.to_bits(),
                y.value.to_bits(),
                "value diverged for {} ({} vs {})",
                x.node_id,
                x.value,
                y.value
            );
            assert_eq!(x.timestamp, y.timestamp);
        }
    }

    #[test]
    fn different_seeds_produce_different_streams() {
        let a = run_all_nodes(1, 25);
        let b = run_all_nodes(999, 25);
        let any_diff = a
            .iter()
            .zip(b.iter())
            .any(|(x, y)| x.value.to_bits() != y.value.to_bits());
        assert!(any_diff, "distinct seeds must yield distinct noise");
    }

    #[test]
    fn reconnect_restarts_an_identical_deterministic_run() {
        // A reconnect must restore the pristine simulation, so the second run
        // reproduces the first exactly.
        let mut client = SimulatedOpcuaClient::with_seed(55);

        client.connect(&OpcuaConfig::default()).unwrap();
        client.subscribe(&ids(&ALL_NODE_IDS)).unwrap();
        let first: Vec<u64> = (0..10)
            .flat_map(|_| client.poll_subscription().unwrap())
            .map(|v| v.value.to_bits())
            .collect();
        client.disconnect().unwrap();

        client.connect(&OpcuaConfig::default()).unwrap();
        client.subscribe(&ids(&ALL_NODE_IDS)).unwrap();
        let second: Vec<u64> = (0..10)
            .flat_map(|_| client.poll_subscription().unwrap())
            .map(|v| v.value.to_bits())
            .collect();

        assert_eq!(first, second, "reconnect did not reset the simulation");
    }

    #[test]
    fn first_batch_matches_known_reference_values() {
        // Reference values captured from the seeded (seed=42) generator on the
        // first poll (dt = 0.1 s). These pin the deterministic output so an
        // accidental change to the generator or RNG wiring is caught.
        let values = run_all_nodes(42, 1);
        let by_id = |id: &str| -> f64 { values.iter().find(|v| v.node_id == id).unwrap().value };
        let approx = |got: f64, want: f64| {
            assert!(
                (got - want).abs() < 1e-9,
                "got {got}, want {want} (delta {})",
                (got - want).abs()
            );
        };
        approx(by_id("ns=2;s=Vibration.X"), 0.001_974_582_486_153_126_3);
        approx(by_id("ns=2;s=Vibration.Y"), 0.418_336_213_724_618_04);
        approx(by_id("ns=2;s=Vibration.Z"), 0.446_590_415_439_805);
        approx(by_id("ns=2;s=Temperature.Motor"), 25.500_949_697_710_897);
        approx(by_id("ns=2;s=Current.Motor"), 75.006_149_362_446_42);
    }

    // --- values_to_event_stream ------------------------------------------

    #[test]
    fn event_stream_keeps_latest_good_and_drops_bad() {
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
            }, // bad quality => must be ignored
        ];
        let order = ids(&["a", "b"]);
        let stream = values_to_event_stream(&values, &order, 2, 1);
        assert_eq!(stream.data, vec![1.0_f32, 2.0]);
        assert_eq!(stream.window_size, 2);
        assert_eq!(stream.stride, 1);
    }

    #[test]
    fn event_stream_picks_latest_good_by_timestamp_not_slice_order() {
        // The newer good sample (t=0.2) appears *before* the older one (t=0.1)
        // in the slice; the result must still be the t=0.2 value.
        let values = vec![
            OpcuaValue {
                node_id: "a".to_string(),
                value: 9.0,
                timestamp: 0.2,
                quality: OpcuaQuality::Good,
            },
            OpcuaValue {
                node_id: "a".to_string(),
                value: 3.0,
                timestamp: 0.1,
                quality: OpcuaQuality::Good,
            },
        ];
        let order = ids(&["a"]);
        let stream = values_to_event_stream(&values, &order, 1, 1);
        assert_eq!(stream.data, vec![9.0_f32], "latest is by timestamp");
    }

    #[test]
    fn event_stream_missing_node_is_nan() {
        let values = vec![OpcuaValue {
            node_id: "a".to_string(),
            value: 5.0,
            timestamp: 0.0,
            quality: OpcuaQuality::Good,
        }];
        let order = ids(&["a", "missing"]);
        let stream = values_to_event_stream(&values, &order, 2, 1);
        assert_eq!(stream.data[0], 5.0);
        assert!(stream.data[1].is_nan(), "absent node must be NaN-filled");
    }

    #[test]
    fn event_stream_all_bad_yields_all_nan() {
        let values = vec![OpcuaValue {
            node_id: "a".to_string(),
            value: 5.0,
            timestamp: 0.0,
            quality: OpcuaQuality::Bad,
        }];
        let order = ids(&["a"]);
        let stream = values_to_event_stream(&values, &order, 1, 1);
        assert!(stream.data[0].is_nan(), "no good sample => NaN");
    }

    // --- end-to-end loop -------------------------------------------------

    #[test]
    fn run_opcua_loop_collects_one_batch_per_iteration() {
        let mut client = SimulatedOpcuaClient::with_seed(8);
        let cfg = OpcuaConfig::default();
        let batches = run_opcua_loop(&mut client, &cfg, "vibration", 3).unwrap();

        assert_eq!(batches.len(), 3, "one non-empty batch per iteration");
        for batch in &batches
        {
            assert_eq!(batch.len(), 3, "three vibration axes per batch");
            for v in batch
            {
                assert!(v.node_id.starts_with("ns=2;s=Vibration."));
                assert!(v.quality.is_good());
            }
        }
        // The loop disconnects when finished.
        assert!(!client.connected, "loop must disconnect at the end");
    }

    #[test]
    fn run_opcua_loop_errors_when_pattern_matches_nothing() {
        let mut client = SimulatedOpcuaClient::new();
        let cfg = OpcuaConfig::default();
        let err = run_opcua_loop(&mut client, &cfg, "no-such-sensor", 3).unwrap_err();
        assert!(err.contains("No nodes matching"), "got: {err}");
    }

    // --- quality ---------------------------------------------------------

    #[test]
    fn quality_is_good_only_for_good() {
        assert!(OpcuaQuality::Good.is_good());
        assert!(!OpcuaQuality::Bad.is_good());
        assert!(!OpcuaQuality::Uncertain.is_good());
    }
}
