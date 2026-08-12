# Automotive / Industry 4.0 extension for SciRust

## Context
SciRust has a deterministic inference foundation, temporal-stream event detection, and edge/embedded deployment (`no_std`, Q16.16 fixed point, Kani certification). However, integration into a real automotive production line is blocked by the lack of industrial connectors, business feature extractors, and ready-to-use application modules.

---

## Axis 1 — Industrial protocol connectors (`scirust-bridge`)

```
Implement a crate `scirust-opcua`:
  - OPC-UA client (real-time node reading, variables, events)
  - OPC-UA data model support (NodeId, BrowsePath)
  - Subscription to value changes → feeding an EventStream
  - Output: normalized stream compatible with scirust-events-core::EventStream

Implement a crate `scirust-mqtt`:
  - MQTT v3.1.1 / v5 client (SparkPlug B for Industry 4.0)
  - Publication of detected events (Event → JSON/CBOR payload)
  - QoS 1 minimum, keep-alive, last will testament

Advanced optional:
  - scirust-modbus (Modbus TCP/RTU)
  - scirust-can (CAN 2.0 / CAN FD for vehicle bus)
```

**Constraint:** everything `no_std` compatible with `scirust-edge` and `scirust-embedded`.

---

## Axis 2 — Sensor feature extractors (`scirust-signal`)

```
Implement a crate `scirust-signal` (or `scirust-features`) with:

1. Frequency domain:
   - FFT (radix-2, Hanning/Hamming/Blackman windows)
   - Power spectral density (PSD)
   - Cepstrum for gearbox diagnostics

2. Time domain:
   - RMS, peak, crest factor, kurtosis, skewness
   - Envelope (Hilbert transform)
   - Zero-crossing rate, autocorrelation

3. Automotive-specific:
   - Order tracking for rotating machinery
   - Bearing fault detection (BPFO, BPFI, BSF)
   - Normalized ISO 10816 / ISO 13374 indicators

Output: feature vectors compatible with an EventClassifier or an enriched SpikeDetector.
```

---

## Axis 3 — Application modules (`scirust-predictive-maintenance`)

```
Implement a crate `scirust-pdm` (predictive maintenance):

1. Degradation detector:
   - Health Index computation from a feature stream
   - Remaining Useful Life (RUL) estimation by regression
   - Regime change detection (CUSUM, Page-Hinkley)

2. Specialized detectors:
   - Motor unbalance (1x RPM + harmonics)
   - Misalignment fault (2x, 3x RPM)
   - Bearing fault (BPFO/BPFI high frequencies)
   - Pump cavitation
   - Pneumatic leak (ultrasonic analysis)

3. Multi-sensor classification:
   - Feature fusion via Kalman
   - Trainable diagnostic model (1D CNN / Transformer on spectrograms)
   - SRT1 export of the trained model for edge deployment

4. Standardized output:
   - Event enriched with: severity (INFO/WARNING/CRITICAL), remaining_life_hours,
     fault_type, component_id, maintenance_action_recommended
```

---

## Axis 4 — Automotive certification (`scirust-functional-safety`)

```
Adapt SciRust to ISO 26262 / IEC 61508 constraints:

1. ASIL levels:
   - Integrity level configuration (ASIL A/B/C/D)
   - Parametrizable hardware/software redundancy (dual lockstep)
   - Watchdog timer for the inference loop

2. Requirements → code traceability:
   - #[requirement("REQ-SAF-042")] annotation on critical functions
   - Traceability matrix generation (requirements → tests → code)
   - ReqIF or spreadsheet export for certification dossiers

3. Coverage tests:
   - MC/DC (Modified Condition/Decision Coverage) on critical paths
   - Fault injection on tensors and weights
   - Worst-case latency tests (WCET) with `scirust-edge`

4. Degraded mode:
   - Deterministic fallback if confidence < threshold
   - Failed sensor isolation (graceful degradation)
   - Immutable audit log (hash chain) of all decisions
```

---

## Axis 5 — Industrial continuous integration (`scirust-mlops`)

```
1. Training → deployment pipeline:
   - Training on historical data (CSV, Parquet, real-time database)
   - Automatic conversion → SRT1 / QSR1 int8
   - Model signing (Ed25519 or ECDSA P-256)
   - OTA (Over-The-Air) deployment to edge fleet

2. Production monitoring:
   - Data drift: distribution shift detection
   - Model drift: prediction vs reality divergence
   - Automatic alert if drift > configurable threshold
   - Dashboard (command-line JSON export, Grafana compatible)

3. Continuous validation:
   - Performance benchmark on target hardware (cycles, RAM, latency)
   - Accuracy regression vs reference model
   - Parallel execution current model / new model (shadow deployment)
   - Automatic rollback on degradation
```

---

## Prioritization

| Priority | Axis | Justification |
|----------|-----|---------------|
| **P0** | Axis 1 (OPC-UA + MQTT) | Without a connector, no real data enters the system |
| **P0** | Axis 2 (FFT + features) | Without features, the detectors have no usable signal |
| **P1** | Axis 3 (PDM) | Direct business value, builds on P0 |
| **P1** | Axis 5 (MLOps) | Required to iterate in real conditions |
| **P2** | Axis 4 (ISO 26262) | Regulatory prerequisite but heavy, can start in parallel |

---

## Success metrics per axis

- **Axis 1**: 1 OPC-UA control loop → EventStream in < 10 ms on ARM Cortex-A72
- **Axis 2**: 1024-point FFT in < 100 µs on Cortex-M7 (Q16.16 fixed-point)
- **Axis 3**: F1 > 0.90 on C-MAPSS (NASA turbofan degradation dataset)
- **Axis 4**: 100% MC/DC on the 20 critical edge inference functions
- **Axis 5**: Drift detection at < 1% false positives on synthetic dataset

---

## Implementation status (June 2026)

### Implemented axes

| Axis | Crate(s) | Status | Tests | Description |
|-----|----------|--------|-------|-------------|
| **P0 - Axis 1** | `scirust-opcua`, `scirust-mqtt` | ✅ Completed | 6 + 9 | `OpcuaClient` / `MqttPublisher` traits + simulators. Feature flags for real backends via `opcua` / `rumqttc` |
| **P0 - Axis 2** | `scirust-signal` | ✅ Completed | 24 | radix-2 FFT, 5 windows, time/frequency features, BPFO/BPFI/BSF, order tracking |
| **P1 - Axis 3** | `scirust-pdm` | ✅ Completed | 24 | Health Index (ISO 13374), linear+exponential RUL, CUSUM, Page-Hinkley, 4 detectors |
| **P1 - Axis 5** | `scirust-mlops` | ✅ Completed | 19 | Data drift (PSI), model drift, shadow deployment (Promote/Keep/Inconclusive), signed OTA |
| **P2 - Axis 4** | `scirust-func-safety` | ✅ Completed | 33 | ASIL A-D, requirements traceability, fault injection (6 types), degraded mode (4 levels), hash-chained audit log |

### Additional integration crates

| Crate | Description | Tests |
|-------|-------------|-------|
| `scirust-integration` | Unifying library: `Backend`, `BackendFactory`, `PipelineConfig`, `Pipeline`, code templates | 32 |
| `scirust-industrial` | 7-command CLI: discover, test-opcua, test-mqtt, gen-config, scaffold, run, doctor | — |
| `examples/industrial_monitor` | End-to-end demo: OPC-UA → Signal → Events → Health → RUL → MQTT → Safety → MLOps | — |

### Total

- **115 new tests** for the industrial crates (1047 in the whole workspace, 0 failures)
- **Documentation**: 8 languages (FR/EN/ES/DE/ZH/JA/KO/AR) updated under `docs/translations/` and in the technical report
- **Next steps**: Integration of the `opcua` and `rumqttc` crates for real backends, C-MAPSS dataset, edge performance benchmark

### Quick test commands

```bash
# Test the full pipeline in simulated mode
cargo run -p industrial-monitor

# Discover available sensors
cargo run -p scirust-industrial -- discover --simulated

# Generate and test an automotive config
cargo run -p scirust-industrial -- gen-config --template automotive --stations 3 --output /tmp/cfg.json
cargo run -p scirust-industrial -- doctor --config /tmp/cfg.json
cargo run -p scirust-industrial -- run --config /tmp/cfg.json --cycles 50

# Scaffold a project
cargo run -p scirust-industrial -- scaffold --name my-monitor --template automotive --output /tmp
```
