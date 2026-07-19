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
- Central finite difference as a validation oracle and parameter-shift for
  `Rx`, `Ry`, and `Rz`.
- SciRust reverse-mode integration for one sample and one expectation output,
  including gradients to encoded classical inputs and quantum parameters.
- A deterministic optimizer-backed hybrid example at
  `scirust-core/examples/quantum_hybrid_classifier.rs`.

## Partially implemented

- A real-amplitude MPS simulator remains available for real gates and adjacent
  two-qubit operations. It is not a complex quantum backend and reports no
  general phase support.
- Dense execution is an exact model but has exponential memory: `2^n` complex
  `f32` amplitudes require approximately `2^n * 8` bytes before allocation
  overhead. The backend applies an explicit allocation ceiling.
- The backend trait and capabilities describe only the dense CPU features that
  actually exist today.

## Designed, not implemented

- Complex MPS tensors and complex truncated SVD, with explicit truncation error.
- Batched hybrid layers, multiple observable outputs, adjoint differentiation,
  and differentiable shot-estimation policies.
- Circuit serialization and OpenQASM 3/QIR lowering.

## Future work

- Density-matrix and noise simulation.
- GPU kernels, distributed simulation, stabilizer and tensor-network backends.
- Hardware topology routing, gate decomposition, remote QPU execution, and
  hardware-result uncertainty/error mitigation.

All seeded execution guarantees apply to the same backend and build; no remote
hardware determinism is implied.
