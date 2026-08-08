# Execution attestation v1

SciRust needs a stable, architecture-neutral execution identity that SciAgent can emit and COGNO-1 can consume without exposing planner internals or low-level ISA details.

`scirust-agent-protocol::ExecutionProfile` is that wire-level semantic contract. It is deliberately separate from `HardwareCapabilities` and `SystemTopology`: those rich runtime structures justify backend selection internally, while the attestation carries their fingerprints plus the semantic facts relevant to replay and audit.

## Profile fields

Version 1 records:

- backend kind and logical device ordinal;
- architecture family plus an optional semantic architecture name;
- SHA-256 fingerprints of the capability profile and topology profile used for selection;
- semantic numeric mode;
- reproducibility level;
- kernel semantic version;
- optional sampler semantic version;
- model SHA-256;
- tokenizer SHA-256.

The schema intentionally contains no ISA feature list, vector width/model, PCIe topology, benchmark result, timing history, or device-name heuristic.

## Canonical fingerprint

JSON is the transport representation, not the fingerprint representation.

`ExecutionProfile::canonical_bytes()` emits a versioned, fixed-order binary encoding with:

- domain separator `scirust.execution-profile.v1\0`;
- explicit numeric tags for enums;
- little-endian fixed-width integers;
- presence tags for optional text;
- length-prefixed UTF-8 strings.

`ExecutionProfile::fingerprint()` computes SHA-256 over those canonical bytes. The implementation is self-contained pure Rust and is checked against NIST SHA-256 vectors. A public integration test pins a golden v1 profile fingerprint, so accidental wire-semantic changes require an explicit schema/version decision.

## Digest versus authenticity

`ExecutionAttestation` contains an `ExecutionProfile` plus its `profile_sha256` and can detect mutation through `verify()`.

This digest is **not a digital signature** and does not establish who produced the profile. Authenticity and trust remain protocol concerns. A later runtime integration should emit the attestation from a `RuntimeDiscovery` or `DeterministicKernel` trust path, where existing agent-protocol sender policy can enforce provenance boundaries.

## Validation

Version 1 fails closed on:

- unsupported schema versions;
- non-canonical SHA-256 strings (anything other than 64 lowercase hex characters);
- empty or malformed semantic identifiers;
- malformed architecture names;
- `Other` architecture without a semantic name;
- profile/digest mismatch.

## Runtime integration boundary

This PR defines only the stable protocol contract. A follow-up runtime builder should map compute-layer facts into it:

- `DeviceKind` → backend kind;
- `ArchitectureFamily` → execution architecture family;
- selected reproducibility guarantee → execution reproducibility;
- canonical capability/topology snapshots → their SHA-256 fingerprints;
- selected kernel/sampler semantic versions;
- model/tokenizer content hashes.

That mapping must remain architecture-neutral and must not forward raw ISA capabilities to COGNO-1.
