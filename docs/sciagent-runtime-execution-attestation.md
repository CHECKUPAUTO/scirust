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

Model and tokenizer hashes are required `Sha256Digest` inputs. They must be computed once at the relevant load/provenance boundary and then carried with the loaded runtime state. The builder does not substitute a placeholder hash and does not re-read large files during generation.

A follow-up integration phase will add provenance-bearing checkpoint/tokenizer loaders for the resident production paths and pass those hashes into this builder.

## Integrity and trust

The returned `ExecutionAttestation` is self-checking through its profile fingerprint. As defined by protocol v1, this is an integrity digest rather than a digital signature. COGNO trust remains attached to the `RuntimeDiscovery` / `DeterministicKernel` protocol path that emits the attestation.
