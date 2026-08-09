# SciRust Quantum roadmap

This document distinguishes the current deterministic foundation from planned
capabilities. It makes no claim of quantum advantage.

## Implemented

- Auditable `Complex32` arithmetic for simulator amplitudes.
- Exact CPU dense state-vector simulation with `I`, `H`, `X`, `Y`, `Z`, `S`,
  `Sdg`, `T`, `Tdg`, `Rx`, `Ry`, `Rz`, `PhaseShift`, `CNOT`, `CZ`, and `SWAP`.
- Little-endian state-vector indexing: index bit `q` is qubit `q`; for two
  qubits, index 1 is `|01>` and index 2 is `|10>`.
- Typed circuit IR with validated qubit operands and symbolic parameters.
- Pauli products and exact real expectation values with residual-imaginary
  validation.
- Deterministic seeded shot sampling, separate from exact expectations.
- Central finite difference as a numerical validation oracle and
  parameter-shift as an independent exact validation oracle for `Rx`, `Ry`,
  and `Rz`.
- Exact dense adjoint differentiation for every symbolic `Rx`, `Ry`, and `Rz`
  occurrence. One dense backward execution and one reverse circuit traversal
  replace the two shifted executions previously required per parameter
  occurrence. Reused symbolic parameters accumulate deterministically in
  ascending circuit-operation order.
- Deterministic SciRust reverse-mode integration for batched, ordered
  multi-observable expectations: features are `[batch, inputs]`, one
  `[1, parameters]` row is shared across the batch, and row-major outputs are
  `[batch, observables]`. Exact adjoint gradients reach encoded classical
  inputs and sum shared-parameter contributions across samples without
  implicit batch averaging.
- A high-level `scirust_core::vqnet` facade composes that existing execution
  path without duplicating it. `VariationalCircuitBuilder` allocates stable
  symbolic parameter IDs automatically, provides ordered angle encoding,
  hardware-efficient `Ry`/`Rz` layers with nearest-neighbour CNOT entanglement,
  validates ordered measurements, and builds a reusable differentiable
  `VariationalCircuit` backed by `QuantumLayer`.
- Reusable variational templates support caller-ordered `Rx`/`Ry`/`Rz`
  rotation families, `CNOT` or `CZ` entanglers, and deterministic `None`,
  `Linear`, `Ring`, and `Full` connectivity. The historical hardware-efficient
  helper remains exactly representable as `[Ry,Rz] + Linear + CNOT`.
- `AngleEncodingHandle` exposes deterministic data re-uploading: later encoding
  layers reuse the original input `ParameterId` values rather than allocating
  duplicate feature columns, while the existing adjoint path accumulates every
  repeated symbolic occurrence into the original classical-feature gradient.
- `QuantumModule` adds persistent trainable quantum state above the fresh-tape
  execution model. `ParameterInitializer` provides zero, finite constant, and
  deterministic seeded-uniform initialization; `VariationalParameters` owns the
  values between tapes; and `QuantumForward` exposes the exact tape parameter
  node for validated synchronization back into the module.
- `OptimizerSlot` gives trainable quantum state a stable string identity that is
  independent of temporary reverse-mode tape node indices. The
  `PersistentParameterOptimizer` adapter reuses SciRust's existing raw-slice
  AdamW and LAMB implementations, keeps their moment state keyed across fresh
  tapes, and commits cloned optimizer/parameter state only after finite-output
  validation.
- `QuantumModule` implements the existing `nn::Module` contract, so a quantum
  layer can be placed directly inside `nn::Sequential` between classical
  modules. Parameter indices participate in ordinary tape optimizers, `sync`
  persists updated quantum values, and `state_dict`/`load_state_dict` integrate
  quantum parameters into the existing hierarchical checkpoint namespace.
- A deterministic optimizer-backed two-sample hybrid binary-classifier example
  at `scirust-core/examples/quantum_hybrid_classifier.rs`; this compatibility
  example continues to use the backward-compatible single-sample,
  single-observable layer API.
- A deterministic four-class hybrid classifier at
  `scirust-core/examples/quantum_multifeature_classifier.rs`: one four-row full
  batch supplies two raw classical features to a trainable `2 × 2` classical
  encoder, then one `forward_batch` call per epoch evaluates two ordered
  observables with two shared trainable quantum parameters. Deterministic
  nearest-codeword decoding uses the two observable values directly, and
  reverse-mode gradients reach both the classical encoder and quantum
  parameters.

## Partially implemented

- The VQNet-like facade currently covers deterministic circuit construction,
  parameter-role mapping, angle encoding and data re-uploading, configurable
  rotation/entanglement ansatz topologies, ordered Pauli measurement,
  reverse-mode execution, deterministic parameter initialization, persistent
  module-owned quantum values, stable optimizer identity across fresh tapes for
  raw-slice AdamW/LAMB, direct `nn::Module` composition, ordinary tape-optimizer
  participation, and checkpoint state. Broader encoding families,
  probability/readout abstractions, higher-level training helpers, and
  remote-hardware execution remain future facade work.
- A real-amplitude MPS simulator remains available for real gates and adjacent
  two-qubit operations. It is not a complex quantum backend and reports no
  general phase support.
- Dense execution is an exact model but has exponential memory: `2^n` complex
  `f32` amplitudes require approximately `2^n * 8` bytes before allocation
  overhead. The backend applies an explicit allocation ceiling.
- The backend trait and capabilities describe only the dense CPU features that
  actually exist today.
- Dense adjoint differentiation retains exponential `2^n` state memory and
  stores one adjoint state per ordered observable during the reverse sweep.
  It does not apply to the real-amplitude MPS simulator.

## Designed, not implemented

- Complex MPS tensors and complex truncated SVD, with explicit truncation error.
- Differentiable shot-estimation policies.
- Circuit serialization and OpenQASM 3/QIR lowering.

## Future work

- Expand `scirust_core::vqnet` with broader reusable encoders,
  probability/readout abstractions and higher-level training helpers.
- Density-matrix and noise simulation.
- GPU kernels, distributed simulation, stabilizer and tensor-network backends.
- Hardware topology routing, gate decomposition, remote QPU execution, and
  hardware-result uncertainty/error mitigation.

All seeded execution guarantees apply to the same backend and build; no remote
hardware determinism is implied.
