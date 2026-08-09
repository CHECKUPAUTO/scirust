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
  The public feature tensor width is therefore independent of re-upload depth.
- `HamiltonianTerm`, `Hamiltonian`, and `HamiltonianReadout` provide fixed real
  linear combinations of the circuit's measured Pauli products, including an
  optional identity offset and multiple Hamiltonian outputs per batch. Pauli
  factors are matched semantically independent of factor order; repeated terms
  accumulate deterministically. Projection is ordinary reverse-mode `matmul`
  plus `add_bias`, so the existing quantum adjoint remains the only quantum
  differentiation implementation and gradients propagate through the readout.
  Hamiltonian coefficients are fixed problem-definition data, not trainable
  module parameters. `HamiltonianReadout` also implements `nn::Module`, so it
  composes directly as `QuantumModule → HamiltonianReadout → Linear` while
  contributing no trainable indices or checkpoint state of its own.
- `ComputationalBasisReadout` reconstructs exact-model computational-basis
  probabilities from the complete non-empty Pauli-Z moment basis via a fixed
  Walsh projection. Moment columns use ascending binary-mask order, probability
  columns use little-endian basis-index order, and the identity contribution is
  a fixed bias. No clipping or post-renormalization is applied, preserving a
  strictly linear reverse-mode path into the existing quantum adjoint. Because
  the explicit Walsh matrix scales as `O(4^n)` and the dense adjoint stores one
  state per observable, this exact facade is deliberately capped at 10 qubits.
  `ComputationalBasisReadout` is also a stateless `nn::Module`, so probability
  vectors compose directly with classical layers and `TrainingSession` without
  adding trainable indices or checkpoint state.
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
- `TrainingSession<O>` provides a minimal guarded fresh-tape training step while
  reusing the existing `Module`, `Loss`, `Tape`, and tape `Optimizer` contracts.
  It validates finite inputs, targets, predictions, scalar loss, gradients, and
  optimizer-updated parameters; then persists model state with `Module::sync`.
  The first successful step pins the exact ordered `parameter_indices()` layout,
  and later graph drift is rejected before `Optimizer::step`, preventing silent
  moment reassociation for tape optimizers keyed by temporary node indices.
- `TrainingSession::train_epoch` adds deterministic epoch orchestration over an
  ordered iterator of already-batched `(Tensor, Tensor)` pairs. It performs one
  existing guarded `train_step` and therefore one fresh tape per batch, applies
  no implicit shuffling, batching, prefetching, or parallel reduction, and
  computes the reported mean loss by sequential `f64` accumulation. Empty epochs
  are rejected. Epoch execution is intentionally non-transactional: if a later
  batch fails, earlier successful parameter updates remain committed.
- `TrainingSession::train_loader_epoch` connects that same training path directly
  to SciRust's native `data::DataLoader`: the core loader selects the requested
  epoch and owns sampling/batching order, then its iterator is delegated to
  `train_epoch`. No VQNet dataset, loader, sampler, or shuffle hierarchy is
  introduced. Together with the epoch-addressable core loader shuffle contract,
  `(dataset, seed, epoch)` is sufficient to reproduce the same resumed batch
  order without replaying previous epochs.
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
  Hamiltonian linear readout, exact computational-basis probability readout,
  reverse-mode execution, deterministic parameter initialization, persistent
  module-owned quantum values, stable optimizer identity across fresh tapes for
  raw-slice AdamW/LAMB, direct `nn::Module` composition, guarded fresh-tape
  training steps, ordered epoch orchestration, and direct integration with the
  native SciRust `DataLoader`, plus ordinary tape-optimizer participation and
  checkpoint state. Broader encoding families, scalable probability
  reconstruction, richer data-pipeline conveniences, and remote-hardware
  execution remain future facade work.
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

- Expand `scirust_core::vqnet` with broader reusable encoders, scalable
  probability reconstruction and richer explicit data-pipeline conveniences.
- Density-matrix and noise simulation.
- GPU kernels, distributed simulation, stabilizer and tensor-network backends.
- Hardware topology routing, gate decomposition, remote QPU execution, and
  hardware-result uncertainty/error mitigation.

All seeded execution guarantees apply to the same backend and build; no remote
hardware determinism is implied.
