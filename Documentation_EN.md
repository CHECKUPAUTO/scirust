# SciRust Documentation 🦀

Welcome to the documentation for **SciRust**, a Deep Learning and scientific computing framework written entirely in **pure Rust**.

## 1. What is SciRust?

SciRust is a research and development platform for Artificial Intelligence. Unlike many other tools (such as PyTorch or TensorFlow) that rely on complex C++ or Python libraries, SciRust is built from the ground up in Rust.

**Why does this matter?**
- **Total Transparency**: You can read every line of computation code, from the network layer to the mathematical kernel.
- **Security and Reliability**: Benefits from Rust's memory and safety guarantees.
- **Independence**: No complex external dependencies (FFI) are required.

## 2. Philosophy and Key Advantages

SciRust does not try to replace industry giants, but offers a different approach focused on **trust** and **reproducibility**.

### Bit-for-Bit Determinism
In many frameworks, running the same calculation twice can yield slightly different results (due to parallelism). SciRust guarantees **bit-for-bit determinism**: the result will be strictly identical, regardless of the number of processors used. This is crucial for auditability.

### Auditability
Since everything is in Rust, it is easy to verify that the code does exactly what it says. There is no software "black box".

### Validation Oracles
Every mathematical function in SciRust is validated against a "validator oracle" (a trusted reference). We do not assume the result is correct; we measure it.

## 3. Application Domains

SciRust is particularly useful in fields where precision, security, and small software footprint are critical:

- **Embedded Systems (Edge AI)**: Thanks to its low footprint and quantization capabilities (reducing model size), it runs perfectly on small devices.
- **Regulated Sectors (Aerospace, Medical, Finance)**: Where every AI decision must be reproducible and explainable for safety or compliance reasons.
- **Scientific Research**: To discover mathematical laws from data through symbolic regression.
- **Security Audit**: For companies that need to certify their entire computational chain.

## 4. What You Can Achieve

SciRust covers a wide range of modern techniques:

- **Deep Learning**: Building neural networks (MLP, CNN, Transformers) with automatic differentiation (autograd).
- **Reinforcement Learning (RL)**: Full stack support for Tabular Q-Learning, DQN, and PPO with clipping.
- **Advanced Computer Vision**: ResNet-18/34 architectures and Vision Transformer (ViT) with global pooling.
- **Generative AI (VAE)**: Variational Autoencoders with reparameterization trick for latent generation.
- **Transformers and MoE**: Mixture of Experts layers with Top-k routing for model scalability.
- **Graph Neural Networks (GNN)**: Graph Convolutional Networks (GCN) for structured data.
- **Speech AI and Audio**: Audio encoders and CTC loss function for speech recognition.
- **PEFT Adaptation (LoRA)**: Low-Rank Adaptation for efficient fine-tuning of pre-trained models.
- **Advanced Scientific Computing**: 1D FEM (Finite Element Method) solver for physical equations.
- **Symbolic Regression**: Discovering mathematical formulas (e.g., `f(x) = sin(x) + x^2`) from observations.
- **Evolutionary Optimization**: Using nature-inspired algorithms (like NSGA-II) to solve complex problems.
- **int8 Quantization**: Shrinking model size by 4x to fit on small processors without losing accuracy.
- **GPU Acceleration**: Harnessing the power of graphics cards via WebGPU (wgpu) or NVIDIA Tensor Cores (cuBLAS).
- **Physics-Informed Neural Networks (PINN)**: Integration of physical laws (differential equations) directly into the loss function for modeling complex phenomena.
- **Formal Invariant Contracts**: Mathematical guarantees (absence of NaN/Inf, value bounds) for critical applications (medical, aerospace).
- **CSR Tensors and SpMM Kernels**: Memory and computation optimization for sparse models on embedded targets.
- **Secure Enclave Execution (TEE)**: Hardened #![no_std] compatible runtime for isolated execution (TrustZone/SGX) without OS allocator.

## 5. Command Guide

SciRust is primarily used via the terminal with `cargo`, Rust's standard tool.

### Installation
Add this to your `Cargo.toml` file:
```toml
[dependencies]
scirust-core = { path = "..." }
```

### Compile and Test
- **Check the project**: `cargo check --workspace`
- **Run all tests** (over 250 tests validate the framework): `cargo test --workspace`
- **Compile in optimized mode** (recommended for AI): `cargo build --release`
- **Enable GPU support**: Add `--features wgpu` to your commands.

### Execution Examples
- **MNIST Training (handwritten digits)**:
  ```bash
  cargo run --example mnist_classifier --release
  ```
- **Transformer Compression Demo**:
  ```bash
  cargo run -p transformer_compress --release
  ```
- **Matrix Multiplication Benchmark**:
  ```bash
  cargo run -p scirust-core --example bench_matmul --release
  ```

## 6. Code Example (Quick Start)

Here is how to create and train a very simple model in a few lines:

```rust
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Sequential, Linear, ReLU, PcgEngine};

fn main() {
    let mut rng = PcgEngine::new(42);

    // Create a simple model
    let mut model = Sequential::new()
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    // Training loop
    for epoch in 0..100 {
        let tape = Tape::new();
        // ... (data loading and gradient calculation)
        println!("Epoch {}: calculation in progress...", epoch);
    }
}
```

## 7. scirust-tensor — Tensor Algebra and Graph Optimization

The `scirust-tensor` module introduces a high-level abstraction layer for manipulating complex tensors while ensuring maximum performance through graph compilation.

### Why use scirust-tensor?
- **Einsum**: Write complex operations (Multi-Head Attention, Contractions) in a single, readable line of code.
- **Operator Fusion**: Reduce memory access by merging activations and biases directly into the computation kernels.
- **Guaranteed Determinism**: Like all of SciRust, every calculation is bit-for-bit reproducible.

### Example: Multi-Head Attention
```rust
use scirust_tensor_einsum::einsum;

// Einstein signature for attention: Batch, Heads, SeqLen, Dim
// (b, h, i, d) , (b, h, j, d) -> (b, h, i, j)
let attention_scores = einsum("bhid,bhjd->bhij", &[&queries, &keys]).unwrap();
```

### Installation
Add this to your `Cargo.toml`:
```toml
[dependencies]
scirust-tensor-core = { path = "scirust-tensor-core" }
scirust-tensor-einsum = { path = "scirust-tensor-einsum" }
```

## 8. Industrial & Automotive Monitoring (v0.14-dev)

SciRust now includes a set of crates for **industrial production line monitoring**, particularly in the automotive domain.

### 8.1 Signal Processing (`scirust-signal`)

Pure-Rust signal processing for vibration analysis and machine diagnostics:

- **Radix-2 FFT** (Cooley-Tukey, forward + inverse)
- **Windows**: Hanning, Hamming, Blackman, Blackman-Harris, Flat-top
- **Time-domain features**: RMS, crest factor, kurtosis, skewness, zero-crossing rate, autocorrelation, energy, entropy
- **Frequency-domain features**: PSD, spectral centroid, spread, spectral entropy, rolloff, band power, flatness
- **Bearing diagnostics**: BPFO, BPFI, BSF, FTF calculation, fault frequency detection in envelope spectrum
- **Order analysis**: order tracking, angle resampling, order spectrum for variable-speed rotating machinery

### 8.2 OPC-UA Connector (`scirust-opcua`)

Connects industrial PLCs/SCADA to the SciRust pipeline:

- **`OpcuaClient` trait**: abstraction for variable reading, subscription, browsing
- **`SimulatedOpcuaClient`**: 8 simulated sensors (3-axis vibration, motor/coolant temperature, hydraulic pressure, motor current, coolant flow)
- **Bridge**: converts OPC-UA values → SciRust `EventStream`
- Ready for real OPC-UA stack integration (via `opcua` crate) using feature flags

### 8.3 MQTT Publishing (`scirust-mqtt`)

Publishes detected events to MQTT brokers for Industry 4.0:

- **`MqttPublisher` trait**: publishing abstraction
- **SparkPlug B format**: Industry 4.0-compatible payloads
- **Severity**: Info / Warning / Critical (derived from confidence score)
- **`SimulatedMqttPublisher`**: test backend without real broker
- **`MonitoringStation`**: station configuration

### 8.4 Predictive Maintenance (`scirust-pdm`)

Predictive maintenance modules for industrial machinery:

- **Health Index**: 0..1 score combining multiple sensor indicators, with EMA smoothing and ISO 13374 classification (Good/Degraded/Warning/Critical/Failed)
- **RUL (Remaining Useful Life)**: linear and exponential estimators with 95% confidence intervals
- **Change detection**: CUSUM (ISO 7870) and Page-Hinkley for regime shift detection
- **Specialized detectors**: `ImbalanceDetector`, `MisalignmentDetector`, `BearingFaultDetector`, `CavitationDetector`

### 8.5 Industrial MLOps (`scirust-mlops`)

ML operations for continuous industrial deployment:

- **Drift detection**: Data drift via Population Stability Index (PSI), Model drift via relative MAE
- **Shadow deployment**: parallel production/candidate model execution, Promote/Keep/Inconclusive recommendation
- **Signed OTA**: Over-The-Air model distribution with cryptographic signature and integrity verification

### 8.6 Functional Safety (`scirust-func-safety`)

ISO 26262 / IEC 61508 compliance for automotive AI:

- **ASIL A-D**: integrity levels, auto-configuration (lockstep, watchdog, max latency, redundancy)
- **Requirement traceability**: requirements → code → tests matrix, JSON export, certification report
- **Fault injection**: 6 fault types (bit-flip, stuck-at, noise, zero-out, scale-shift, overflow), batch testing
- **Degraded mode**: 4 levels (Full → Reduced → Safety → Emergency), hysteresis, safe state
- **Hash-chained audit log**: immutable safety decision journal, chain integrity verification

### 8.7 Integration Kit (`scirust-integration`)

Unifying library to simplify industrial integration:

- **`Backend`**: unified OPC-UA + MQTT abstraction with feature flags (`real-opcua`, `real-mqtt`)
- **`BackendFactory`**: automatic creation, simulated → real fallback
- **`PipelineConfig`**: complete JSON configuration (backend, stations, sensors, Health Index, RUL, drift)
- **`Pipeline`**: full pipeline Backend → Signal → Events → Health → RUL → MQTT → Audit
- **Templates**: project generation (`minimal`, `automotive`, `bearing`, `pdm`)

### 8.8 Industrial CLI (`scirust-industrial`)

Command-line tool to facilitate integration:

```bash
scirust-industrial discover --simulated                    # Browse available PLC sensors
scirust-industrial test-opcua --simulated --samples 5       # Test OPC-UA connection
scirust-industrial test-mqtt --simulated                    # Test MQTT connection
scirust-industrial gen-config --output config.json --template automotive --stations 3
scirust-industrial scaffold --name line3-monitor --template automotive
scirust-industrial run --config config.json --cycles 100 --report report.json
scirust-industrial doctor --config config.json             # Diagnose integration issues
```

### 8.9 Full Integration Example (`industrial-monitor`)

The `industrial_monitor` example demonstrates the complete chain:

```
OPC-UA (PLC) → Signal Processing → Event Detection → Health Index
→ RUL Estimation → CUSUM → MQTT Publishing → Audit Log → Functional Safety → MLOps Drift
```

```bash
cargo run -p industrial-monitor
```

## 9. Conclusion

SciRust is the framework of choice for those who prioritize **understanding** and **rigor** over raw speed or the ease of Python. It is a powerful tool for building trustworthy AI, from research to embedded systems.

---
*For more technical details, see the full report in `paper/SciRust-technical-report.md`.*

## 13. Research → Functions (N-D autograd extensions)

The N-D autograd tape now carries a complete deep-learning stack, every piece
backed by a research paper and a test (gradient check or oracle). See
[`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) (14/20 delivered).

- **Causal decoder LM**, trained end-to-end (token + positional embeddings,
  causal attention, fused softmax cross-entropy); overfits a sequence exactly.
- **LLaMA-family layers**: RMSNorm, SwiGLU, LLaMA block, RoPE, grouped/
  multi-query attention (GQA/MQA).
- **Deterministic optimizers**: Adam, AdamW, Lion, Muon (Newton–Schulz), Schedule-Free, AdEMAMix, and SOAP (Adam in Shampoo's eigenbasis).
- **Certifiable AI**: Interval Bound Propagation **and CROWN** (tighter
  linear-relaxation bounds) — *provable* output bounds and
  a robustness certificate.
- **Reproducible reductions**, order-independent (bit-identical regardless of
  thread count).
- **Exact speculative decoding**; **FlashAttention** (online softmax);
  **DeltaNet** (delta-rule linear attention);
  **Mamba** (selective state-space / selective scan);
  **RetNet** (retention / linear attention);
  **GLA** (gated linear attention);
  **HGRN** (gated linear RNN);
  **Neural ODE** (backprop through an RK4 solver); a Physics-Informed Neural Network (PINN) that solves a boundary-value problem with the PDE residual in the loss.
- **Compression**: Wanda (activation-aware) pruning, SmoothQuant, GPTQ (second-order error-feedback int8 weight quantization), AWQ (activation-aware search-based int8 weight quantization).

New CLI commands:
- `scirust certify [--seed N] [--eps E]` — provable ReLU-MLP bounds (IBP **and** CROWN, the tighter linear-relaxation bounds, side by side).
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead|lamb|adan|adafactor|shampoo|prodigy]` — train the N-D decoder LM.
- `scirust deltanet [--seed N] [--steps S]` — train a single-head DeltaNet (delta-rule linear attention) layer to fit a sequence; reports the MSE reduction.
- `scirust mamba [--seed N] [--steps S]` — train a Mamba selective state-space layer (S6 scan) to fit a sequence; reports the MSE reduction.
- `scirust retnet [--seed N] [--steps S]` — train a RetNet retention layer (linear attention, recurrent form ≡ parallel form) to fit a sequence; reports the MSE reduction.
- `scirust gla [--seed N] [--steps S]` — train a Gated Linear Attention layer (data-dependent forget gate) to fit a sequence; reports the MSE reduction.
- `scirust hgrn [--seed N] [--steps S]` — train an HGRN gated-linear-RNN token mixer (lower-bounded forget gate) to fit a sequence; reports the MSE reduction.
- `scirust rwkv [--seed N] [--steps S]` — train a single RWKV time-mixing (WKV) layer (per-channel time decay + bonus) to fit a sequence; reports the MSE reduction.
- `scirust conformal [--seed N] [--alpha A]` — conformal intervals with a guaranteed, distribution-free coverage level.
- `scirust calibrate [--seed N]` — temperature scaling; fit T to lower the expected calibration error (ECE) without changing accuracy.
- `scirust pinn [--seed N] [--steps S]` — physics-informed network; solve the BVP `u''=−u` (PDE residual in the loss), checked against `sin x`.
- `scirust gptq [--seed N] [--samples S] [--damp D]` — GPTQ int8 weight quantization; reports the calibration-error reduction vs round-to-nearest.
- `scirust awq [--seed N] [--samples S] [--grid G]` — AWQ activation-aware int8 weight quantization; reports the selected scaling exponent and the calibration-error reduction vs round-to-nearest.
- `scirust bitnet [--seed N]` — BitNet b1.58 ternary {-1,0,+1} weight quantization (~1.58 bit/weight); verifies the multiplication-free matmul.

## 14. Industrial CLI — Complete Reference

The `scirust-industrial` CLI facilitates integrating SciRust with real industrial systems.

### Installation

```bash
cargo install --path scirust-industrial   # provides the `scirust-industrial` binary
# or in place: cargo run -p scirust-industrial -- <command>
```

### Commands

| Command | Description | Options |
|----------|-------------|---------|
| `discover` | Lists available sensors on OPC-UA server | `--endpoint`, `--filter`, `--simulated` |
| `test-opcua` | Tests OPC-UA connection and reads values | `--endpoint`, `--simulated`, `--samples N` |
| `test-mqtt` | Tests MQTT broker connection and publishes a message | `--host`, `--port`, `--simulated`, `--topic` |
| `gen-config` | Generates a pipeline configuration file | `--output`, `--template`, `--stations N`, `--line-id` |
| `scaffold` | Generates a complete monitoring project | `--name`, `--output`, `--template` |
| `run` | Runs a monitoring pipeline from config | `--config`, `--cycles N`, `--report` |
| `doctor` | Diagnoses integration issues | `--config` |

### Templates

| Template | Description |
|----------|-------------|
| `minimal` | 1 station, simulated backend, spike detection |
| `automotive` | Multi-station automotive line with bearing diagnostics, RUL, MQTT, audit |
| `bearing` | Bearing fault detection (FFT envelope, BPFO/BPFI/BSF) |
| `pdm` | Predictive maintenance (Health Index, RUL, CUSUM) |

### Recommended Integration Flow

```bash
# 1. Scaffold a project
scirust-industrial scaffold --name line3-monitor --template automotive

# 2. Verify everything works
cd line3-monitor
scirust-industrial doctor --config config.json

# 3. Customize configuration
# Edit config.json: OPC-UA endpoint, MQTT broker, sensors, thresholds

# 4. Switch to real mode (optional)
# Edit Cargo.toml: uncomment real-opcua / real-mqtt features
# Edit config.json: backend_type "opcua"

# 5. Start monitoring
scirust-industrial run --config config.json --cycles 1000
```

### Switching from Simulated to Real Mode

The simulated mode works without any hardware. To go to production:

1. **Real OPC-UA**: Add `features = ["real-opcua"]` to `scirust-integration` in `Cargo.toml`, add `opcua = "0.13"` dependency, and change `backend_type` to `"opcua"` in `config.json`.
2. **Real MQTT**: Add `features = ["real-mqtt"]`, add `rumqttc = "0.24"`, and configure broker `host`/`port`.

The `BackendFactory` handles automatic fallback: if the real backend fails, it falls back to simulated mode.
