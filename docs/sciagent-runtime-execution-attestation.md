# SciAgent runtime execution-attestation builder

The protocol-level `ExecutionProfile` and the compute-layer canonical profile encodings are intentionally separated. SciAgent needs a narrow bridge that combines them after backend selection without taking ownership of probing, planning, or model-file I/O.

`build_runtime_execution_attestation` is that bridge.

## Inputs

`RuntimeExecutionAttestationInputs` requires explicit runtime facts:

- backend kind and logical device ordinal;
- semantic architecture family/name;
- canonical hardware-capability bytes;
- canonical system-topology bytes;
- optional caller memory budget;
- selected numeric mode;
- selected reproducibility mode;
- kernel semantic version;
- optional sampler semantic version;
- precomputed model SHA-256;
- precomputed tokenizer SHA-256.

The two profile byte slices must come from `scirust-compute::canonical_hardware_profile_bytes` and `canonical_topology_profile_bytes`. The builder hashes those bytes with SciAgent's architecture-independent FIPS-180-4 SHA-256 and places the resulting canonical lowercase digests into the protocol profile.

## Route-B CUDA semantics

`build_route_b_cuda_execution_attestation` is a narrower bridge for the resident Route-B CUDA path. It fixes only execution semantics already guaranteed by that path:

- backend `Cuda`;
- logical device ordinal 0, matching `CudaChain::new`;
- architecture family `NvidiaGpu` after successful CUDA acquisition;
- numeric mode `bf16-fp32-accum-v1`;
- reproducibility `NumericallyEquivalent` rather than a cross-device bit-exact claim;
- versioned Route-B CUDA kernel semantics.

The optional architecture name remains caller-supplied. This layer never parses a product string or guesses a compute capability. Likewise, the hardware-capability and topology fingerprints remain derived from canonical compute-profile bytes supplied by the execution layer. Building the attestation itself opens no CUDA context.

## Deliberate non-responsibilities

The builder performs no:

- backend selection;
- CPU/GPU probing;
- ISA inspection;
- topology discovery;
- benchmark or timing;
- checkpoint/tokenizer file read;
- model serialization;
- fallback policy.

This separation prevents an attestation helper from silently changing execution behavior or adding per-token I/O.

## Model and tokenizer provenance

Model and tokenizer hashes are required `Sha256Digest` inputs. They are computed once at the relevant load/provenance boundary and then carried with the loaded runtime state. The builder does not substitute a placeholder hash and does not re-read large files during generation.

`load_checkpoint_with_provenance` reads `model.safetensors` once and hashes the exact byte buffer that is passed to the safetensors deserializer, eliminating a hash/load TOCTOU gap. The legacy `load_checkpoint` API remains available and delegates to that provenance-bearing path.

SciAgent also exposes deterministic identities for the built-in raw-byte tokenizer and the embedded `tokenizer/bpe.json`. The raw-byte path hashes a versioned semantic identifier because it has no artifact file; the embedded BPE path hashes the exact `include_bytes!` payload. External tokenizer files remain a follow-up so the active ElasticTokeniser work can settle without introducing a competing parser path.

## Integrity and trust

The returned `ExecutionAttestation` is self-checking through its profile fingerprint. As defined by protocol v1, this is an integrity digest rather than a digital signature. COGNO trust remains attached to the `RuntimeDiscovery` / `DeterministicKernel` protocol path that emits the attestation.
