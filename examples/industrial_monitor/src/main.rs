//! Industrial Monitor — End-to-End Integration Example
//!
//! Demonstrates the full pipeline:
//! OPC-UA (simulated PLC) → Signal processing → Event detection →
//! Predictive maintenance → MQTT publishing → Functional safety
//!
//! ```text
//! [PLC Simulator] → [Signal Features] → [Event Detector] → [Health Index]
//! → [RUL Estimation] → [Fault Detectors] → [MQTT Publisher] → [Audit Log]
//! ```

use scirust_events_core::EventStream;
use scirust_events_models::SpikeDetector;
use scirust_events_runtime::EventRuntime;
use scirust_func_safety::audit::AuditLog;
use scirust_mqtt::{MqttConfig, MqttPublisher, SimulatedMqttPublisher, event_to_payload};
use scirust_opcua::{OpcuaClient, OpcuaConfig, SimulatedOpcuaClient, values_to_event_stream};
use scirust_pdm::change_detection::CUSUM;
use scirust_pdm::health::HealthIndex;
use scirust_pdm::rul::{LinearRulEstimator, RulEstimator};
use scirust_signal::{crest_factor, kurtosis, rms};

fn main() {
    println!("=== SciRust Industrial Monitor — End-to-End Demo ===\n");

    // 1. Connect to OPC-UA (simulated PLC with vibration sensors)
    let mut opcua = SimulatedOpcuaClient::new();
    let opcua_cfg = OpcuaConfig::default();
    opcua.connect(&opcua_cfg).expect("OPC-UA connection failed");
    println!("[OPC-UA] Connected to {}", opcua_cfg.endpoint);

    let nodes = opcua.browse("vibration").expect("Browse failed");
    println!("[OPC-UA] Found {} vibration sensors:", nodes.len());
    for n in &nodes
    {
        println!("  - {} ({})", n.display_name, n.unit);
    }

    let node_ids: Vec<String> = nodes.iter().map(|n| n.node_id.clone()).collect();
    opcua.subscribe(&node_ids).expect("Subscribe failed");

    // 2. MQTT publisher for alerts
    let mut mqtt = SimulatedMqttPublisher::new();
    mqtt.connect(&MqttConfig::default())
        .expect("MQTT connection failed");
    println!(
        "\n[MQTT] Connected to broker, base topic: {}",
        MqttConfig::default().base_topic
    );

    // 3. Event detector (spike detector on vibration)
    let mut runtime = EventRuntime::new(Box::new(SpikeDetector::new(1.0, 0.8)));

    // 4. Health Index (baseline = normal vibration, threshold = failure)
    let mut health = HealthIndex::new(
        vec![0.1, 0.1, 0.1], // baseline: low vibration on all axes
        vec![3.0, 3.0, 3.0], // failure threshold
        vec![0.33, 0.33, 0.34],
        0.3, // EMA smoothing
    );
    println!("\n[Health] Health Index estimator initialized");

    // 5. RUL estimator
    let mut rul = LinearRulEstimator::new(100, 10);
    println!("[RUL] Linear RUL estimator initialized");

    // 6. CUSUM change detector
    let mut cusum = CUSUM::new(0.1, 0.05, 0.5);
    println!("[CUSUM] Change detector initialized (target=0.1)");

    // 7. Audit log
    let mut audit = AuditLog::new(1000);
    println!("[Audit] Audit log initialized (hash-chained)\n");

    // 8. Run monitoring loop
    let n_cycles = 50;
    println!("=== Running {} monitoring cycles ===\n", n_cycles);

    let mut sim_time = 0.0_f64;
    let mut total_events = 0u64;
    let mut health_history: Vec<(f64, f64)> = Vec::new();

    for cycle in 0..n_cycles
    {
        // Poll OPC-UA for new sensor values
        let values = opcua.poll_subscription().expect("Poll failed");
        if values.is_empty()
        {
            continue;
        }

        // Convert to EventStream
        let _stream = values_to_event_stream(&values, &node_ids, 2, 1);

        // Extract features per axis
        let vibration_x = values
            .iter()
            .filter(|v| v.node_id.contains("Vibration.X"))
            .map(|v| v.value)
            .next_back()
            .unwrap_or(0.0);
        let vibration_y = values
            .iter()
            .filter(|v| v.node_id.contains("Vibration.Y"))
            .map(|v| v.value)
            .next_back()
            .unwrap_or(0.0);
        let vibration_z = values
            .iter()
            .filter(|v| v.node_id.contains("Vibration.Z"))
            .map(|v| v.value)
            .next_back()
            .unwrap_or(0.0);

        let features = vec![vibration_x, vibration_y, vibration_z];

        // Update Health Index
        let hi = health.update(&features);
        let state = health.state();
        health_history.push((sim_time, hi));

        // Update RUL
        rul.update(hi, sim_time);

        // CUSUM on vibration X
        let _ = cusum.update(vibration_x, 0.1);

        // Run event detection
        let mut det_stream = EventStream::new(features.iter().map(|f| *f as f32).collect(), 2, 1);
        let events = runtime.process_all(&mut det_stream);

        // Publish events to MQTT
        for event in events
        {
            if event.confidence >= 0.8
            {
                let payload = event_to_payload(&event, "spindle-motor", None);
                let _ = mqtt.publish_event(&payload);
                total_events += 1;
            }
        }

        // Audit log entry
        audit.add(
            "monitoring_cycle",
            &format!("Cycle {}: HI={:.3}, state={}", cycle, hi, state.label()),
            "spindle-motor",
            if hi < 0.5 { "alert" } else { "pass" },
            hi as f32,
            sim_time,
        );

        sim_time += 0.1;

        // Print summary every 10 cycles
        if cycle % 10 == 0 || cycle == n_cycles - 1
        {
            println!(
                "Cycle {:3} | Vibration(X={:.2}, Y={:.2}, Z={:.2}) | HI={:.3} ({}) | RUL={:.0}h | Events={}",
                cycle,
                vibration_x,
                vibration_y,
                vibration_z,
                hi,
                state.label(),
                rul.predict().remaining_hours,
                total_events
            );
        }
    }

    // 9. Final report
    println!("\n=== Final Report ===\n");
    println!("Total cycles: {}", n_cycles);
    println!("Total events published: {}", total_events);
    println!(
        "Final Health Index: {:.3} ({})",
        health.value(),
        health.state().label()
    );
    let rul_pred = rul.predict();
    println!(
        "RUL estimate: {:.1} hours (95% CI: {:.1} - {:.1})",
        rul_pred.remaining_hours, rul_pred.lower_bound_hours, rul_pred.upper_bound_hours
    );
    println!("RUL method: {}", rul_pred.method);
    println!("Audit log entries: {}", audit.len());
    println!("Audit chain valid: {}", audit.verify_chain());
    println!("MQTT messages published: {}", mqtt.publish_count);

    let (info, warn, crit) = mqtt.count_by_severity();
    println!("  - Info: {}", info);
    println!("  - Warning: {}", warn);
    println!("  - Critical: {}", crit);

    // 10. Feature summary
    println!("\n=== Signal Features ===\n");
    let raw_vib: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).sin() * 0.5).collect();
    println!("RMS of test signal: {:.4}", rms(&raw_vib));
    println!("Kurtosis: {:.4}", kurtosis(&raw_vib));
    println!("Crest factor: {:.4}", crest_factor(&raw_vib));

    // 11. Functional safety summary
    println!("\n=== Functional Safety ===\n");
    use scirust_func_safety::asil::{AsilConfig, AsilLevel};
    let asil_cfg = AsilConfig::for_level("spindle-monitor", AsilLevel::B);
    println!("Component: {}", asil_cfg.component_id);
    println!("ASIL Level: {}", asil_cfg.asil_level.label());
    println!(
        "Required MC/DC coverage: {}%",
        asil_cfg.asil_level.required_mcdc_coverage()
    );
    println!(
        "Required fault injection tests: {}",
        asil_cfg.asil_level.required_fault_injection_count()
    );
    println!("Dual lockstep: {}", asil_cfg.dual_lockstep);
    println!("Max latency: {} ms", asil_cfg.max_latency_ms);

    use scirust_func_safety::degraded_mode::DegradedModeController;
    let mut degr = DegradedModeController::new();
    degr.min_dwell_time_ms = 0.0;
    let action = degr.update(health.value() as f32, 0, 1.0);
    println!("Current degradation level: {}", action.level.label());
    println!("Production active: {}", degr.production_active());

    // 12. Drift detection summary
    println!("\n=== MLOps ===\n");
    use scirust_mlops::drift::DataDriftDetector;
    let ref_data: Vec<f64> = (0..100).map(|_| rand::random::<f64>() * 0.2).collect();
    let mut drift = DataDriftDetector::from_reference(&ref_data, 10, 50, 0.25);
    for &h in health_history.iter().map(|(_, h)| h).take(50)
    {
        drift.add_sample(h);
    }
    if let Some(report) = drift.check()
    {
        println!(
            "Data drift PSI: {:.4} (threshold: {})",
            report.data_drift_score, report.threshold
        );
        println!("Drift detected: {:?}", report.drift_type);
    }

    println!("\n=== Demo Complete ===");

    // Cleanup
    opcua.disconnect().expect("OPC-UA disconnect failed");
    mqtt.disconnect().expect("MQTT disconnect failed");
}
