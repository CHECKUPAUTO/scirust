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
the newest ordinal version known to be committed by both channels. A
faster-moving channel therefore cannot silently pair its newest basis with an
older channel under a single token version.

## Runtime factories

The public handoff API exposes:

- `CommittedBasisLearner` — deterministic Phase 10 learner plus preallocated
  immutable committed snapshots;
- `LearnedHeadBasis` — key/value learners and quality profile for one attention
  head;
- `ResolvedHeadCalibration` — construction-time owned carrier for one resolved
  committed K/V version;
- `resolve_committed_head_calibration` — resolves one head to the newest common
  committed version and validates the learned rank against the runtime policy;
- `runtime_from_committed_bases` — creates a fresh
  `ElasticLatentDecodeRuntime` directly from learned heads;
- `LearnedLayerBasis` — runtime policy plus learned heads for one Transformer
  layer;
- `encoder_session_from_committed_bases` — creates a fresh
  `ElasticLatentEncoderSession` directly from learned layers.

Existing `HeadCalibration`, `ElasticLatentDecodeRuntime::new`,
`ElasticLatentEncoderSession::new`, and the historical dense Transformer APIs
remain unchanged.

## Lower-rank learned bases

Online learning is useful when `rank < d_head`; forcing a square learned basis
would make a full orthonormal basis span the whole head space and leave no
reconstruction residual for the Oja update to learn from.

Phase 13 currently accepts a square row-major `[d_head, d_head]` basis carrier.
The handoff therefore materializes a construction-time square carrier without
changing the learned subspace:

1. the exact committed `[d_head, learned_rank]` columns are copied into the
   leading columns of the square carrier;
2. columns `learned_rank..d_head` are zero-filled;
3. the handoff rejects the runtime if `maximum_rank` exceeds either the learned
   key rank or learned value rank.

The zero-filled columns are therefore structural padding only and can never be
selected by the Phase 9 planner. Tiered Phase 11 storage receives exactly the
same learned prefix columns that were archived at the Phase 10 commit.

This keeps the existing Phase 13 backend API stable while allowing actual
lower-rank online bases to drive future cache epochs.

## Epoch semantics

1. Observe key/value samples through `CommittedBasisLearner`.
2. Phase 10 quality gates decide independently whether each channel commits.
3. Open a new Phase 13 runtime/session.
4. The factory resolves the highest common committed K/V version for each head.
5. It validates that the runtime's `maximum_rank` is supported by both learned
   bases and materializes square construction-time carriers.
6. Phase 13 copies the selected learned prefix columns into its preallocated
   tiered caches.
7. Later observations may advance the online learners, but the running runtime
   remains bound to its construction epoch.
8. Open another runtime/session to consume later common commits.

This preserves the Phase 10 rule that committed versions apply only to future
cache epochs and the Phase 11 rule that resident tokens retain the basis version
under which they were admitted.

## Determinism and allocation

The archive performs no growth in `observe`: both the wrapped learner's version
metadata capacity and the basis snapshot storage are reserved during
construction. Snapshot lookup is deterministic and scans the bounded version
metadata in commit order.

Square carrier materialization occurs only while opening a fresh runtime/session
and follows fixed row-major copy order. Runtime/session construction may allocate
because Phase 13 already allocates its bounded cache structures at construction.
No new allocation claim is made for construction itself or for the surrounding
legacy tape-based Transformer.

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
common K/V version resolution, lower-rank square-carrier preservation, direct
Phase 13 runtime construction from learned lower-rank bases, and direct
Transformer session construction from learned lower-rank bases.
