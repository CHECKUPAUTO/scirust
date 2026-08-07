# Elastic Latent KV — Phase 10 → Phase 13 runtime handoff

## Status

This follow-up connects deterministic online basis learning from Phase 10 to the
integrated Phase 13 decode runtime and the legacy incremental Transformer bridge.

The handoff is deliberately epoch-scoped: a basis commit never mutates an
already-running cache. New committed bases become eligible only when a fresh
`ElasticLatentDecodeRuntime` or `ElasticLatentEncoderSession` is constructed.

## Why an archive is required

`DeterministicBasisLearner` continuously updates its live basis after every
observation. Its Phase 10 metadata records a version number and a fingerprint,
but the live basis can continue moving after that commit. A runtime therefore
must not borrow `basis()` and label it with an older committed version.

`CommittedBasisLearner` wraps the learner with a fixed-capacity archive. Storage
for every possible version is allocated at construction from
`maximum_versions × basis_len`. When `observe` returns a commit, the exact basis
bits are copied into the preallocated slot associated with that version. No
archive allocation occurs on the observation path.

The archived snapshot fingerprint is required to match the `BasisVersion`
fingerprint emitted by the underlying learner.

## Key/value synchronization

Phase 13 lifecycle metadata currently associates one `basis_version` with a
resident token. Key and value learners can commit at different rates, so a new
runtime resolves each head to:

```text
common_version = min(key.current_version(), value.current_version())
```

and loads the immutable key and value snapshots for that same version. This is
the newest version known to be committed by both channels. A faster-moving
channel therefore cannot silently pair a newer basis with an older one under a
single token version.

## Runtime factories

The public handoff API exposes:

- `CommittedBasisLearner` — deterministic Phase 10 learner plus preallocated
  immutable committed snapshots;
- `LearnedHeadBasis` — key/value learners and quality profile for one attention
  head;
- `resolve_committed_head_calibration` — resolves one head to the newest common
  committed version;
- `runtime_from_committed_bases` — creates a fresh
  `ElasticLatentDecodeRuntime` directly from learned heads;
- `LearnedLayerBasis` — runtime policy plus learned heads for one Transformer
  layer;
- `encoder_session_from_committed_bases` — creates a fresh
  `ElasticLatentEncoderSession` directly from learned layers.

Existing `HeadCalibration`, `ElasticLatentDecodeRuntime::new`,
`ElasticLatentEncoderSession::new`, and the historical dense Transformer APIs
remain unchanged.

## Basis shape contract

The current Phase 13 tiered backend accepts a full row-major
`[d_head, d_head]` basis because the Phase 9 planner may select any prefix rank
up to the dense head dimension. Consequently, a learner used by this handoff
must currently be configured with `dimension == rank == d_head`.

Lower-rank online learners remain valid Phase 10 learners, but they cannot be
fed directly into this Phase 13 factory until the backend supports an explicit
maximum learned rank or a deterministic orthogonal completion policy.

## Epoch semantics

1. Observe key/value samples through `CommittedBasisLearner`.
2. Phase 10 quality gates decide independently whether each channel commits.
3. Open a new Phase 13 runtime/session.
4. The factory resolves the highest common committed K/V version for each head.
5. Phase 13 copies those bases into its preallocated tiered caches.
6. Later observations may advance the online learners, but the running runtime
   remains bound to its construction epoch.
7. Open another runtime/session to consume later common commits.

This preserves the Phase 10 rule that committed versions apply only to future
cache epochs and the Phase 11 rule that resident tokens retain the basis version
under which they were admitted.

## Determinism and allocation

The archive performs no growth in `observe`: both the wrapped learner's version
metadata capacity and the basis snapshot storage are reserved during
construction. Snapshot lookup is deterministic and scans the bounded version
metadata in commit order.

Runtime/session construction may allocate because Phase 13 already allocates
its bounded cache structures at construction. No new allocation claim is made
for construction itself or for the surrounding legacy tape-based Transformer.

## Validation

The dedicated CI gate verifies:

```bash
cargo +nightly-2026-07-02 fmt --all -- --check
cargo +nightly-2026-07-02 clippy -p scirust-core --all-targets --locked -- -D warnings
cargo +1.89.0 check -p scirust-core --all-targets --locked
cargo +1.89.0 test -p scirust-core elastic_latent_basis_handoff --locked -- --test-threads=1
cargo +1.89.0 test -p scirust-core elastic_latent_encoder_handoff --locked -- --test-threads=1
```

The tests cover immutable snapshot preservation across later training, highest
common K/V version resolution, direct Phase 13 runtime construction, and direct
Transformer session construction.
