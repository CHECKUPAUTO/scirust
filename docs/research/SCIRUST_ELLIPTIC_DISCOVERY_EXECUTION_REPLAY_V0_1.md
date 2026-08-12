# SciRust Elliptic Discovery — local execution and replay v0.1

## Status

This document defines phase 6, after the integration of phases 0 through 5. It does
not modify the security scope of the main design document:
only locally generated toy curves may be evaluated.

## Objective

The library can build the corpora, generate relations, falsify them, and produce
justifications. What is still missing is a single execution
boundary that:

1. executes a validated `SearchPlan`;
2. summarizes exactly all evaluated candidates;
3. produces a canonical receipt with domain separation and a SHA-256 fingerprint;
4. replays the same plan and detects any divergence;
5. decodes no external data.

The receipt is neither a mathematical proof nor a claim of novelty. It
only attests that a given local plan produced a precise automated
result with this version of the schema.

## Decisions

### Library API, without an external protocol

Phase 6 adds a Rust API in `scirust-elliptic-discovery`. It
does not add a general-purpose CLI, an incoming JSON format, a server, networking,
or a plugin system. The only entry point is a `SearchPlan` already
bounded by the crate.

This decision prevents a replay path from implicitly becoming a parser
for addresses, public keys, SEC 1 encodings, or blockchain targets.

### Canonical receipt

An `ExecutionReceipt` contains:

- the exact plan;
- the ordered fingerprints of the three built-in corpora;
- an ordered fingerprint of each candidate evaluation;
- a summary per authorized status;
- the number of recorded counterexamples.

The encoding is binary, big-endian, with explicit lengths, and separated by the
domain `SCIRUST-ELLIPTIC-DISCOVERY/EXECUTION-RECEIPT/V1` in the original
definition. Phase 7 evolves the generation behavior and moves the
plan, evaluation, and receipt domains to `V2`; see
[`SCIRUST_ELLIPTIC_DISCOVERY_HARDENING_V0_1.md`](SCIRUST_ELLIPTIC_DISCOVERY_HARDENING_V0_1.md).
Relations are encoded by their typed syntax tree, never by `Debug`
or `Display`.

### Replay

`replay_local` re-executes the plan carried by a receipt, recomputes a complete receipt,
and compares the canonical bytes. The result exposes the expected and observed
fingerprints as well as a boolean of concordance. It does not replace the expected
receipt, so as to keep the divergence for audit.

## Invariants

| Invariant | Verification |
|---|---|
| Local inputs only | The API accepts only `SearchPlan`. |
| Finite bounds | The construction of `SearchPlan` keeps all phase 4 limits. |
| Exactness | No floating point is added to the execution or receipt path. |
| Stable order | Corpora and candidates remain in their existing canonical orders. |
| Strict replay | The comparison covers all bytes of the receipt, not a partial summary. |
| Non-novelty | The summary exclusively reuses `ClassificationStatus`. |
| Pure Rust | No `unsafe`, no FFI, and no new dependency. |
| No hidden I/O | Execution reads no file, no environment variable, and no network. |

## Exit tests

The phase is finished when:

- two executions of the same plan produce identical receipts;
- two distinct seeds produce distinct fingerprints;
- replaying an intact receipt agrees;
- replay detects tampering with the receipt;
- the encoding covers every relation, status, gate, and
  counterexample variant;
- the workspace CI matrix stays green on the declared MSRV.

## Out of scope

- importing or exporting arbitrary curves;
- reading Bitcoin addresses, keys, or point encodings;
- secret recovery;
- connecting to a blockchain;
- claiming discovery or proof from the receipt;
- deserializing a receipt coming from an untrusted source.
