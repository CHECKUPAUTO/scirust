# Changelog

All notable changes to CCOS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Content-addressed embedding cache (`ccos::embed_cache::CachedEmbedder`,
  `octasoma` feature)** — the documented follow-up of the Pro semantic tier:
  wrap ANY `octasoma::Embedder` (notably the local `OllamaEmbedder`) and
  identical content is embedded once (`sha256(text) → vector`), making the
  derived semantic index and the OctaCore cascade practical behind a real
  neural embedder instead of one embedding call per node per rebuild.
  Optional durable persistence (`save`/`load`, temp+fsync+atomic-rename like
  every checkpoint) is **fail-closed**: a cache built by a different embedder
  (label or dimension) or a malformed file is refused with an explicit error,
  never served silently and never downgraded to an empty cache. Hit/miss
  stats for observability; the deterministic `HashEmbedder` stays
  bit-replayable with or without the cache (transparency covered by tests,
  incl. identical cascade rankings). 5 unit tests + doctest.

### Changed

- **Documentation refresh (post-0.4.0 doc audit)** — the user-facing docs now
  match the fused product: `docs/DEPLOYMENT.md` (0.4.0 doctor sample, the full
  feature table incl. every fusion feature + the `pro-default`/`all-full`
  bundles and premium build commands, `CCOS_LICENSE_FILE` in the license
  resolution order, §4c now lists all **9** Pro features);
  `docs/MEMORY_INTERFACE.md` (the MCP catalogue: the sixteen core tools incl.
  `get`/`sync`, plus the Pro `slha.*`/`octa.*`/`rsi.*` namespaces) and the
  matching README §3 count; `LICENSING.md` (new §2b separating the *copyright
  license* from the *Pro runtime token*); `CONTRIBUTING.md` (the real fusion CI
  matrix, premium dev-loop commands); `docs/ARCHITECTURE.md` (the fused
  premium modules/crates map); CHANGELOG compare links; a ROADMAP pointer to
  the fusion plan. `ccos doctor` additionally reports the compiled **premium**
  features (`slhav2`/`slhav2-full`/`octasoma`/`octacore`/`rsi`/`rsi-dgm`,
  plus `signed-sync`) in both text and JSON output, so a premium deployment can
  verify at a glance that its build compiled the kernels it licensed.

## [0.4.0] — 2026-07-08 — CCOS_EXTENDED premium fusion

The first CCOS_EXTENDED release: the CCOS 0.3 core fused with SLHAv2, OctaSoma
and CERVO/RSI (plan `docs/FUSION_PLAN.md`, P0–P6 all closed; audit
`docs/AUDIT_FUSION_2026-07.md`). The **default build's behaviour and dependency
tree are unchanged from CCOS 0.3** — every premium kernel is opt-in
(cargo feature) and Pro-gated (offline license); the version bump names the
fused product.

### Added

- **CCOS_EXTENDED fusion audit + P5/P6 close-out (the premium merger of CCOS +
  SLHAv2 + OctaSoma + CERVO/RSI).** (1) *Base sync*: the CCOS core was caught up
  with upstream #151 — `ccos.recall` aligned with the OpenClaw contract
  (`query`/`limit`/`minScore`/`sessionKey` aliases + `structuredContent`) and the
  new `get` / `sync` MCP tools. (2) *Vendored-source refresh*: the scirust family
  was re-based onto SLHAv2 HEAD, ingesting the TurboQuant series — MIXED/TQ3/MIX3
  latent codecs with AVX2/AVX-512/NEON score paths, `fit_joint`, COLD→EventLog
  persistence, the `slha.compress` codec parameter, and the llama.cpp Phase-2
  codec FFI (`slha_weights_load`/`slha_encode_key`/`slha_decode_latent`) — with
  the P4 hardening re-applied on top (`#![deny(unsafe_code)]` + audited allow
  zones + FFI alignment guard). (3) *P5 — unified premium surface*: one MCP
  server multiplexing four namespaces (`ccos.*` + `slha.*` + `octa.*` + `rsi.*`,
  `src/mcp_ext.rs`), each kernel tool behind its offline Pro gate with visible
  refusals, and the matching CLI (`ccos slha|octa|rsi …`, refusal exit 3) sharing
  the same implementation. **DGM self-modification stays unreachable over MCP and
  the CLI** — typed `GuardedDgm` API only. (4) *P6 — CI matrix*: byte-identity
  guard (default tree links no premium kernel), `pro-default` / `all-full` test
  profiles, fused-member tests, community-tier CLI refusal smoke.
  New: `tests/fusion_unified_mcp.rs` (6 tests), `docs/AUDIT_FUSION_2026-07.md`
  (the full audit report).

- **Cryptographic agent identity for the multi-agent store (`signed-sync`, off by default).** The
  hash chain proves a bundle's *integrity*; only a signature proves *who* recorded it. The new
  feature adds exactly that: `ccos sync keygen` creates a per-workspace ed25519 identity
  (`<workspace>.ccos.key` — the seed never travels and never enters the timeline sidecar), `sync
  export` then signs the bundle's canonical payload automatically, and `sync import` verifies and
  **TOFU-pins** the key per agent id (persisted additively in the sidecar). From the first signed
  contact on: a bundle under the same id with a *different* key refuses (`KeyMismatch` — spoof or
  unannounced swap), an *unsigned* bundle from a pinned agent refuses (`UnsignedFromPinned` — a
  stripped signature does not bypass pinning), a bad or half-present signature refuses
  (`BadSignature`, enforced on **every** build), and a signed bundle on a crypto-free build refuses
  with a rebuild hint (`SignedUnsupported`) rather than being silently accepted unverified. Pinning
  happens only after the whole import succeeds — a refused bundle changes no state, keys included.
  Key rotation is an explicit human act (`keygen` refuses to overwrite). Zero new crates: reuses the
  same optional `ed25519-dalek` as `license` plus the in-tree `getrandom` (uuid already draws from
  it); the default build stays crypto-free and unsigned federations work unchanged. 5 new tests
  (signed round-trip + pin persistence across restart, chain-valid-but-forged refused by signature,
  spoof + stripped-signature refusals, keygen overwrite refusal, field-pairing on every build) +
  CLI smoke; `docs/SYNC.md` gains the signed-bundles contract; paper §9 item 4 (new list) marked
  delivered.

- **Distributed multi-agent store (`ccos sync`) — paper §9 item 5 landed; the future-work list is
  now fully cleared.** Every agent keeps exactly one append-only, hash-chained timeline of its own
  ops; sharing is the exchange of **chain-verified segments** (`SyncBundle`, a plain JSON file — any
  transport including sneakernet, so the air-gappable posture survives federation). Imports
  re-verify every link and refuse tampering (`SyncError::Tampered`), gaps (`Gap`), self-imports,
  and **equivocation** — one agent id publishing two divergent histories is caught link-for-link
  (`Diverged`), the distributed payoff of the op-log chain. Imported logs stay per-agent (grow-only,
  persisted additively in the sidecar); the *shared brain* is `AgentSession::merged_view()` — a pure
  function replaying all known timelines from empty in canonical agent order with `replay_to`'s
  exact semantics (shared `apply_op`), so agents holding the same logs materialize **bit-identical**
  views: a state-based CRDT of grow-only per-agent logs, no consensus protocol, no network stack,
  no new dependency. Convergence is checked with the new `CcosMemory::state_fingerprint()` —
  canonical SHA-256 of the replayable state (graph + sources + both chain heads; only the
  by-design-non-deterministic audit UUIDs are excluded). CLI: `ccos sync
  export|import|status|materialize`. `examples/sync_crux.rs` measures the four claims live
  (isolation → verified exchange → bit-identical convergence + the cross-agent `api→db` call edge
  neither agent had alone → tamper refusal), deterministic across runs. 7 new tests (convergence
  incl. a third agent, tamper, equivocation+gap, incremental==full, sidecar round-trip,
  export/self-import guards, JSON round-trip). Contract notes in `docs/SYNC.md` (compaction vs
  federation, local baseline stays local, declarative identity). Paper §4.7/§7/§9 rewritten: all
  five future-work items of the previous edition are delivered; §9 now lists the genuinely-next
  items (data-flow direction, trait-object dispatch, spectral pass, cryptographic identity, belief
  fusion).

- **Richer receiver inference (Slice 4) — paper §9 item 3 landed.** The `syn` call graph now types
  **compound receivers** from the same syntactically-certain declarations Slice 3 trusts, via two
  new per-scope fact families (never persisted — they only enrich the minted `Type::method`
  callees): struct **field types** and declared **return types** (`-> Self` resolved to the impl's
  concrete type). New shapes, all resolve-uniquely-or-skip: field receivers `self.field.m()` /
  `x.field.m()` (the dominant method-call shape in real Rust), call returns `f().m()` /
  `let x = f(); x.m()`, assoc-fn chains `A::make().t()`, and single/multi-hop method chains
  `x.b().q()`. Two precision rules: *evidence beats convention* (an in-scope `C::new() -> Option<C>`
  refutes the returns-`Self` convention → skip) and *wrappers never unwrap* (`Vec<_>`/`Option<_>`/
  generic fields and returns are never receivers). Minted callees resolve through the existing
  graph-wide `(type, method)` unique-or-skip index, so field receivers link **across files** with no
  new false-edge surface. Measured on CCOS's own `src/` (51 files, identical corpus):
  **1007 → 1114 `Calls` edges (+107, ≈ +10.6 %)** — the single largest recall gain of the whole
  call-graph arc (the `&T` peel gave +7 %). `examples/resolution_coverage.rs` grows to **14/14
  resolving idioms + 4/4 precision-skips**; 4 new parser tests (field/chain positives, uncertainty
  skips, cross-file resolution). Trait-object dynamic dispatch remains skipped *by design* — a
  `dyn Trait` receiver has no statically-certain concrete type. See
  `docs/MEASUREMENT_resolution_coverage.md`.

- **Tamper-evident chain on the session op-log — paper §9 item 2 fully landed.** CCOS had three
  logs and two of them were hash-chained; the third — the `AgentSession` op-log (the
  `<workspace>.ccos.oplog` sidecar), the timeline that actually carries `replay == live` — was not.
  It now is: every recorded `Op` links into a canonical SHA-256 chain
  (`hashᵢ = H(prevᵢ | stepᵢ | opᵢ)`, no wall-clock fields, so the chain is bit-reproducible), the
  replay **baseline** is pinned by its own commitment (a baseline edit is as detectable as an op
  edit), and compaction hands the folded prefix's head to the `anchor` — the chain head never moves,
  so one hash commits to the entire history since genesis. Enforcement has teeth:
  `AgentSession::open` **refuses** a workspace whose sidecar fails the check (new
  `MemoryError::TimelineTampered`), leaving the sidecar intact for forensics — previously a
  tampered op-log was silently *self-healed away*, destroying the evidence. Chain-valid staleness
  (an out-of-band `ccos memory` write) still self-heals exactly as before, and pre-chain sidecars
  load unchanged (chain backfilled on open, persisted on the next checkpoint). `ccos verify
  <workspace>` audits a sidecar *without* opening it (`agent_session::audit_workspace`); replay
  never reads the chain, so `replay == live` is untouched (the whole property suite passes
  unmodified). 7 new tests: payload tamper, reorder, mid-deletion, baseline tamper, legacy
  backfill, head continuity across compaction+restart, fork prefix validity.

### Removed

- **External dense-retrieval backend (`scirust-retrieval` feature).** Removed the optional bridge to the
  external `scirust-retrieval` crate (`src/scirust_bridge.rs`) and its `scirust-dense` eval strategy. CI
  could no longer authenticate to the private `Memorithm/scirust` repo (the pinned revision became
  unreachable), which failed `cargo clippy --all-features --locked` at **dependency resolution** on every
  PR — before any code compiled. As the dependency was optional and off by default, removing it (dep,
  feature, `scirust_bridge` module, `scirust-dense` eval arm, the dedicated CI step, and the lock entries)
  unblocks the whole repo's CI without touching the default build. The **distilled, dependency-free**
  retrieval (`src/retrieval.rs`: exact dense / BM25 / RRF / LSA over CCOS's own embeddings) is unaffected
  and remains the moat-aligned path; the external backend can be re-added once `scirust` access is restored.

- **Dead Neural Store source files (`src/brain`, `src/core`, `src/ffi`, `src/storage`) — 1065 LOC of
  uncompiled orphan code.** An unrelated-histories merge flattened a separate "Neural Store" crate
  (SIMD engine, LSM-tree, brain workers, zero-copy FFI) into `src/`, but its own `Cargo.toml`/`lib.rs`
  were dropped in the merge, so the files were **never wired into the crate**: not declared in
  `lib.rs`, not in `Cargo.toml` (no `rayon` dependency despite `src/core/search.rs` importing it),
  never referenced, and provably **not compiled** (`cargo check --all-features` passed *because* they
  were dead). A moat audit confirmed `replay == live`, the zero-dependency / air-gappable identity, and
  the FFI-free build were all untouched — but 1065 LOC carrying `extern "C"`, `rayon`, SIMD intrinsics,
  and `unsafe` sat in `src/` as misleading cruft. Removed to keep the tree honest and matching CCOS's
  deterministic, no-FFI identity. (The subsystem remains in git history if ever wanted — it would
  return **feature-gated**, off by default, so the default build stays pure.)

### Added

- **Quarantined neural embedder (`neural-embed` feature, off by default).** Lands the paper's
  future-work item 1 *as a quarantine*: `src/neural_embed.rs` provides `NeuralEncoder`, a
  `retrieval::Encoder` over a **local** Ollama-style `/api/embeddings` endpoint, so it drops into any
  `SemanticRetriever`/`HybridRetriever` unchanged. The contract is explicit — the default build
  compiles none of it and stays bit-for-bit replayable; the feature pulls **no new crate** (only the
  in-tree `reqwest`'s blocking client); the endpoint is local, so nothing leaves the host; and the
  neural path is **not replay-exact** (weights/server/hardware-dependent), which is exactly why it
  lives behind the flag. Fail-fast constructor (no silent fallback that would fake semantics),
  visible degradation (`errors()` counter, zero-vector substitution ranks last), response parser
  accepts both Ollama dialects and is offline-unit-tested. `examples/neural_vs_lsa.rs` compares
  lexical / LSA / neural on the synonym crux (graceful with setup instructions when no server is
  running); `docs/NEURAL_EMBED.md` states the quarantine design. No measured neural numbers are
  committed on purpose — the measurement docs only carry numbers reproducible from the repo alone.

- **BEIR-style benchmark harness (`examples/beir_eval.rs` + `docs/MEASUREMENT_beir.md`).** CCOS's
  deterministic retrievers evaluated on **standard IR benchmarks** in the native BEIR format
  (`corpus.jsonl` + `queries.jsonl` + `qrels/test.tsv`; datasets fetched locally, never committed —
  `/data/` is git-ignored). Four systems over the same corpus — exact BM25, hashed TF-IDF dense, its
  LSA projection, and their RRF fusion — scored with nDCG@10 / R@10 / R@100 / MRR@10 / MAP. Headline,
  measured: **CCOS's zero-dependency BM25 scores nDCG@10 0.662 on SciFact vs 0.665 published for the
  tuned Anserini baseline** (and 0.307 vs 0.325 on NFCorpus) with a plain lowercase tokenizer — no
  stemming, no stopwords, no tuning. The doc reads the results honestly: BM25 dominates where query
  and document share vocabulary (the mirror of the synonym crux, where the same LSA encoder wins),
  hybrid fusion dilutes a dominant system, and the residual gap to Anserini is stemming. Output is
  bit-for-bit identical across runs (timings on stderr); zero new dependencies. Closes the paper's
  future-work item 4.

- **README "See it run" section.** The repo's front door now surfaces the three one-command
  demonstrations: `flagship` (replay==live + contested beliefs + LSA-beats-RAG in one deterministic
  run), `resolution_coverage` (the resolver measured: 10/10 idioms, 963+43 edges on `src/`), and the
  opt-in zero-dependency SLHAv2 backend example (`-p ccos-memory-runtime --example slha_backend`).

- **`docs/MEASUREMENT_resolution_coverage.md`** — the resolver arc's capstone doc: the 10/10 resolving
  idioms (each tagged with the slice that added it: #113 → #122 → #124 → #126), the 3/3 deliberate
  precision-skips and why each is Rust-correct, the structural yield on CCOS's own `src/` (2474 call
  refs → 963 `Calls` edges; 80 data refs → 43 `DataFlow` edges; the `&T` fix alone: 903 → 963, ≈ +7 %),
  and the adversarial-review provenance. The example's footer now cross-references this doc.

- **Resolution-coverage measurement (`examples/resolution_coverage.rs`).** A deterministic example that
  enumerates every Rust path shape the call/data-flow resolver handles — crate-rooted, `use`-import
  (fn and module), local submodule path, nested submodule, receiver-type method, `Self::` method, bare
  const, import-scoped const, renamed const — each tagged with the slice that added it, confirms the
  shapes it *deliberately skips* (bare `extern::fn()` without `use`, globally-ambiguous, unknown
  module), and reports the structural yield on CCOS's own `src/` (fn→fn `Calls` + fn→const `DataFlow`
  edges resolved vs. references parsed). The honest "what a RAG index cannot represent" number, replay-
  exact. (Building this surfaced the reference-receiver gap fixed below.)

- **Receiver-type inference now handles reference-typed receivers (`x: &T`, `x: &mut T`).** Method
  calls on a reference-annotated receiver (`fn f(x: &T) { x.bar() }`) — the pervasive `&self`/`&T`
  pattern — previously inferred no type, so `x.bar()` resolved to nothing. `annotation_type` now peels
  leading references to the underlying type (`&T`/`&mut T`/`&&T`/`&'a T` ⇒ `T`), the same inference and
  the same precision as the owned `x: T` case: the `(type, method)` cardinality guard still gates every
  edge, and `&dyn Trait` / `&Box<T>` / `&[T]` / `&GenericParam` remain (correctly) uninferred.

- **Data-flow resolution now follows *renamed* imports (`use m::MAX as LIMIT; … LIMIT`).** A renamed
  const/static import bound a local alias the data-flow pass didn't understand, so the reference
  resolved to nothing — even though the call resolver already handled renamed imports. `resolve_data_flow`
  now builds the same per-file alias index and rewrites an aliased leading segment onto its target
  before resolving (mirroring `resolve_call`), so an aliased data ref links to its real `static`/`const`.
  The rewrite is applied at most once, feeds the existing resolve-uniquely-or-skip resolvers (no new
  false-edge surface), and is deterministic (sorted `pending_aliases`, first-binding-wins); non-aliased
  refs are byte-identical. Two precision tests (resolves-through-alias, and skips when the target is not
  a data symbol).

- **Call graph now resolves non-`use` local module-path calls (`submod::fn()`, `outer::inner::fn()`).**
  Previously a qualified call resolved only when crate-rooted (`crate::…`) or mediated by a matching
  `use` import; the very common idiom of calling through a bare **local submodule** path with no `use`
  produced **no `Calls` edge** (measured: three of six common call shapes were unresolved).
  `resolve_qualified` now falls back — only when no import matches — to `resolve_bare_modpath`, which
  resolves the leading path as a submodule of the **caller's own crate** (module-file-must-exist,
  exact — no ancestor shortening — then symbol-must-exist; a present-but-symbol-less module skips).
  Only the caller's crate is consulted: a bare path is deliberately **not** resolved into an unrelated
  external crate, because a local `mod` shadows a same-named extern crate in Rust — an adversarial
  multi-agent review proved the external reading minted false edges (a symbol-less local module falling
  through to a same-named crate) and stole valid type-method edges, so it is excluded. Composes with
  the module-vs-type reconciliation (type methods unaffected) and applies to qualified data-flow refs
  too. **resolve-uniquely-or-skip**, deterministic (indices over sorted ids → replay == live), no new
  dependencies. Designed and adversarially reviewed via multi-agent workflows; 13 new precision tests.

- **Flagship end-to-end example (`examples/flagship.rs`).** One deterministic run that demonstrates,
  on a single event-sourced agent session, three properties a RAG stack cannot offer: (1) **replay ==
  live** — the session is reconstructed bit-for-bit from its op log (auditable, time-travel-debuggable);
  (2) **contested knowledge** — a lone refutation is a typed `Contradicts` edge and `qbelief.conflict`
  flags the claim, where a similarity retriever's cosine puts the dissent *inside* the confirmation
  band (polarity-blind); (3) **beating RAG on its own turf** — the deterministic LSA encoder recovers
  synonym recall (17% vs 0% Recall@1, MRR 0.458 vs 0.185) a lexical retriever structurally misses.
  Pure-Rust, zero external dependencies, byte-exact reproducible. `cargo run --release --example flagship`.

- **Data-flow resolution now links *bare* global refs through imports and same-module scope, not only
  when globally unique.** A bare `static`/`const` reference previously became a `DataFlow` edge
  only when exactly one symbol of that name existed graph-wide, so a common name like
  `CONFIG`/`MAX`/`LIMIT` shared across modules resolved to nothing. `resolve_data_flow` now runs the
  same **Tier A → B → C** ladder as the call resolver — import-scoped (`use m::CONFIG` pins the
  defining module), then the reader's own module, then global-unique — against the data-symbol-only
  index. A shared global reached through an explicit import (or defined alongside the reader) links
  even when its name is not unique; ambiguous imports and unresolvable names still
  **resolve-uniquely-or-skip** (no guessed edge). Deterministic, no new dependencies.

- **Post-quantum Pro license verifier (`license-pq` feature / SLH-DSA, FIPS 205).** A second,
  fully independent offline license-signature verifier alongside ed25519, behind the orthogonal
  `license-pq` cargo feature. A license token can now be signed/verified with **SLH-DSA**
  (NIST FIPS 205, formerly SPHINCS+) — stateless hash-based signatures conjectured secure against
  a large-scale quantum computer, where ed25519 (Discrete-Log) is not. Parameter set
  **SLH-DSA-SHAKE-128s**: a 32-byte public key (the same shape as ed25519, so the fail-closed
  all-zero placeholder transfers verbatim) and a 7,856-byte (~10.5 KB base64url) signature — the
  smallest FIPS 205 signature, NIST PQ category 1, a like-for-like PQ upgrade of ed25519's
  classical 128-bit. The token format is `slhdsa.<payload>.<sig>` — a `slhdsa.` scheme tag that
  both dispatches [`Licensing::detect`] to the right verifier (a build may compile in one, the
  other, or both via `--features license,license-pq`) and is bound into the signed message, so a
  signature made under one scheme can never be replayed as the other. New `SlhDsaVerifier`,
  `sign_token_slhdsa`, `LICENSE_SLH_DSA_PUBLIC_KEY` placeholder, vendor tool
  `cargo run --features license-pq --example license_sign_pq`, `ccos doctor` scheme surfacing, and
  a full mirror of the ed25519 test suite plus cross-scheme isolation tests. **Crate choice:** the
  `lattice-slh-dsa` crate (pure Rust, `#![forbid(unsafe_code)]`), not RustCrypto's `slh-dsa` — the
  latter pins a pre-release `signature` crate that cannot coexist with `ed25519-dalek` in one build
  (which would break `--all-features`); `lattice-slh-dsa` depends only on stable `sha2`/`sha3`, so
  the two license features compose. **Caveat:** `lattice-slh-dsa` is not independently audited — see
  `docs/DEPLOYMENT.md` §4b before trusting it to gate production features. (ROADMAP slice 29c.)

### Changed

- **Research paper (`docs/PAPER.md`) refreshed with everything landed since its last update.** New
  **§6.8 "Call/data-flow resolution coverage"** (10/10 path shapes resolve, 3/3 precision-skips hold,
  963 `Calls` + 43 `DataFlow` edges on CCOS's own `src/`, the adversarial-review provenance); §5 now
  describes the Cargo **workspace** and the opt-in zero-dependency **SLHAv2 memory backend**
  (`ccos-memory-runtime`, `slhav2` feature) and drops the stale claim that an optional
  `scirust-retrieval` bridge still exists (removed when its private pin became unfetchable); §4.6
  gains the **DTW timeline alignment** (cross-run regression hunting over recorded cognitive
  histories); §6.5 self-hosting figures re-measured (~2,400 nodes / ~3,900 edges, was ~350/~400);
  §7/§9 receiver-inference limitation and future-work items updated to the current resolver; counts
  refreshed (~37 KLoC, **649 tests** across all targets); Reproducibility block gains
  `resolution_coverage` and `flagship`.

- **Research paper (`docs/PAPER.md`) brought up to date with the current system.** The paper
  described an earlier CCOS (line-based heuristic parser, no semantic edges, ~6 KLoC, 364 tests). It
  now reflects reality: the `syn` AST as the default parser, the call-graph + data-flow semantic edges,
  the **Q-Page dual-evidence belief** layer (§4.10, with decay + propagation + the temporal "fever
  curve"), **deterministic semantic retrieval** (§4.11, TF-IDF/LSA encoders + causal-topology-weighted
  LSA), and a new evaluation section **§6.7 "Retrieval: challenging RAG, deterministically"** with the
  measured results — ties lexical RAG, **beats it on semantic recall** (Recall@3 17%→83%), suppresses a
  refuted contradiction (precision@1 2/2 vs 1/2), and self-improves (Recall@1 8%→100%), all bit-for-bit
  reproducible with zero extra dependencies. Solved items removed from Limitations / Future work, and
  the numbers refreshed (~35 KLoC, 480+ tests). The `docs/paper/` multilingual `ccos_regions.*`
  versions remain a follow-up re-render.

- **SLHAv2 grouped-INT4 embeddings are now a Pro feature (`Feature::SlhAv2Embeddings`).** The
  adaptive per-group INT4 quantization (group size 16, the "SLHAv2 two-level INT4" distilled from
  SCIRUST's KV-cache) that powers semantic recall is now gated behind the Pro license. A
  **community** session falls back to **uniform** INT4 (a single per-vector absmax scale — the same
  4× storage win, slightly less faithful on heterogeneous vectors); a **Pro** session keeps the
  grouped scheme. The core recall path is unchanged — only the *precision* of the semantic embedding
  store reflects the tier, exactly like custom authority weights. The scheme is decided silently at
  session open from the host tier (community `new()`/`open` → uniform; Pro `open` → grouped) and via
  the explicit gated `AgentSession::enable_slhav2_embeddings`; it is runtime-only (never persisted),
  so `replay == live` holds. The `Int4Embedding.group_size` field already discriminated the two
  schemes (16 = grouped, `dim` = uniform), so there is no persistence-schema change; old snapshots
  deserialize with the grouped default.

### Fixed

- **CI unblocked after the `neural_store` unrelated-histories merge.** That merge
  force-added the entire `target/` build tree (722 files — 77% of the tracked tree)
  despite `/target` being in `.gitignore`, and pulled in unformatted `neural_store`
  integration tests that broke `cargo fmt --all --check`. The build artifacts are now
  untracked (`git rm -r --cached target/` — already gitignored, so local builds are
  unaffected) and the offending tests were removed in the prior commit. CI runs green
  again once the repo's GitHub Actions billing is restored (a public repo has unlimited
  free minutes).
- **`ci.yml` / `audit.yml`: bump `actions/checkout` and `actions/cache` to `v5`.** The
  `v4` actions target Node 20, which GitHub Actions has deprecated (runs are forced to
  Node 24 with a warning that will become a hard error). `v5` targets Node 24 natively.
- **`eval::tests::pipeline_runs_offline_stub` is now hermetic.** `provider_label()` picks
  the LLM provider from the process env (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
  `OLLAMA_ENDPOINT`), so the test — which asserts the offline `none` stub — failed for
  any contributor with a local Ollama server configured, even though it passed in CI's
  clean env. The test now strips those vars up front (it is the only test calling
  `run_eval`, so there is no parallel-test race), matching CONTRIBUTING's "tests run
  fully offline" contract.

### Performance

- **Ingestion is no longer ~O(N³): O(1) `add_edge` de-duplication + the `ingest_profile` profiler.**
  Profiling (`examples/ingest_profile.rs`) found the ingestion hot spot is the whole-graph **resolve
  passes** (data-flow ~49%, calls ~23%; parse is only ~5%) — *not* cache layout — and that `add_edge`
  de-duplicated with an **O(E) linear scan of every edge**. Since the resolve passes re-run after each
  ingested file and add an edge per ref, that made ingesting N files roughly **cubic** (600 files ≈
  216 s of resolution). Replacing the scan with an **O(1) membership-set index** (`edge_set`, a
  `serde(skip)` `HashSet<(source, target, type)>` rebuilt lazily on a length mismatch) cut a single
  ingest pass **~11×** (the data-flow pass ~70×) and dropped scaling to a clean **O(N²)** (×~4.3 per
  file-count doubling; 600 files ≈ 11 s). The remaining quadratic — the per-file whole-graph
  re-resolution — is the next slice (incremental resolution → O(N)). Measuring first redirected the
  work from a speculative SoA/cache rewrite to the real bottleneck.

- **B2-batch: deferred whole-graph resolution — ~174× faster batch ingestion (O(N²)→O(N)).** The
  three resolve passes are order-independent pure functions of the *final* node + pending-ref set, so
  running them **once at the batch boundary** instead of after every file collapses the remaining
  quadratic to a single linear pass. The new `CcosMemory::ingest_deferred` (record a file, mark
  resolution pending) + `CcosMemory::resolve` (run the passes once, idempotent/near-free when clean)
  expose this; the profiler's new `# B2-batch` table measures **15,596 ms → 89.5 ms at 600 files**,
  scaling ~×2.5 per doubling (linear) instead of ~×4.9 (quadratic). The eager `ingest_source` is
  unchanged — it is now literally `ingest_deferred` + `resolve`, so a single ingest still leaves a
  fully-resolved graph (a `debug_assert` in `recall`/`to_json`/`checkpoint` guards the deferred
  contract). Surfaced and **measured** an honest semantic subtlety: eager (incremental, add-only)
  resolution keeps an order-dependent `Calls` edge that batch (final-state, resolve-uniquely-or-skip)
  correctly drops under late-arriving name ambiguity — so the **replayable `AgentSession` path stays
  eager** and `replay == live` is exact. Order-independent resolution (prune resolution-owned edges
  before each rebuild → eager ≡ batch everywhere, replay can batch too; edge ownership mapped) is the
  scoped follow-up. See `docs/MEASUREMENT_batch_resolution.md`.

- **B2-full: order-independent resolution — eager ≡ batch, the divergence is gone.** Made resolution
  *idempotent-with-removal* via `MemoryGraph::resolve_all` (now behind `CcosMemory::resolve`): it
  **prunes the resolution-owned edges, then rebuilds from the final state**, so a name that became
  ambiguous after a caller was linked is no longer left as an order-dependent stale `Calls` edge. The
  prune is **selective** to respect the `serde(skip)` pending-ref indices (empty after a checkpoint
  load): `file:→file:` import / hierarchy edges always rebuild from the durable node set, while
  `Calls`/`DataFlow` are pruned only for files whose pending refs are present (this session / a replay
  re-ingest) — a loaded file with no pending refs keeps its edges (they can't be rebuilt). So eager
  (per-file), batch (deferred) and a replay re-ingest now converge on the **identical** graph, **and
  `replay == live` stays exact** (replay sees the same pending-presence pattern as live). New tests:
  `eager_and_batch_agree_under_late_ambiguity`, `checkpoint_load_then_ingest_keeps_loaded_call_edges`.
  This removes the semantic blocker, so batching the replayable path (O(N) time-travel) +
  `AgentSession::ingest_batch` is now a safe mechanical follow-up. See
  `docs/MEASUREMENT_batch_resolution.md`.

- **B2-replay: the replayable/agent path now batches too — O(N) time-travel.** With resolution
  order-independent (B2-full), `AgentSession::replay_to` and the counterfactual `retrieval_reward`
  **defer** every `Ingest` op and run the resolve passes **once** — before each op that reads
  cross-file edges (a recall page-in, a failure / page-fault propagation) and once at the end —
  instead of resolving after every ingest, turning the O(N²) reconstruction into O(N). The new
  `AgentSession::ingest_batch` applies the same single-resolve batch to the live ingest path.
  `examples/replay_batch_crux.rs` measures a reconstruction speedup of **12× → 23× → 47.5× at
  150/300/600 ops** (eager ~×4 per doubling = quadratic; batched ~×2 = linear), asserting both paths
  rebuild the byte-identical graph. `replay == live` is preserved **exactly**: ingestion never demotes
  to COLD (so deferring the resolve cannot reorder paging), and `tests/replay_equivalence_property.rs`
  still passes byte-for-byte over any interleaving of ingests, failures, recalls and page-faults. See
  `docs/MEASUREMENT_batch_resolution.md`.

### Changed

- **The real `syn` AST parser is now the default ingestion path** (was opt-in behind
  the `syn-parser` feature). On real code the old line heuristic is **36.5% wrong**
  structurally — import recall only 66.9% (grouped `use a::{b,c}` collapsed, so a third
  of the cross-file dependency edges were invisible) plus 145 hallucinated symbols
  (local consts promoted to top-level) — see `docs/MEASUREMENT_ast.md`. `syn` /
  `proc-macro2` are already in the dependency tree via serde, so defaulting to the AST
  pulls **no new dependency**; `--no-default-features` keeps the zero-extra-dependency
  heuristic, retained as the fallback for non-Rust / unparseable input. Still no async
  runtime and no TLS in the default build.

### Added

- **`LsaEncoder` — semantic retrieval that *beats* lexical RAG on synonym recall.** The dense retriever
  over ccos's TF-IDF *ties* the lexical RAG (same signal); swapping the encoder to project TF-IDF
  through ccos's deterministic **LSA** latent space (`crate::lsa`) captures the synonymy a literal-term
  retriever structurally cannot. On a corpus where each query and its answer share **zero vocabulary**
  (linked only by co-occurrence *bridge* docs), `examples/semantic_retrieval_crux.rs` measures
  **Recall@3 17% → 83%, MRR 0.185 → 0.458 (2.5×)**: lexical RAG cannot retrieve the answer, LSA recovers
  it, bit-for-bit reproducibly. This is RAG's *own* turf — semantic recall — won by a deterministic,
  zero-dependency encoder. Always-compiled; the encoder chooses the axis (TF-IDF lexical / LSA semantic)
  over the same index / fusion / metrics machinery. See `docs/MEASUREMENT_pure_retrieval.md`.

- **Adaptive retrieval — the self-improving `ImprovementLoop` (premium tier) + license gate.** The
  `ccos::retrieval` core (dense/BM25/hybrid + metrics) is free; the **premium** tier learns a linear
  projection of the embedding space from confirmed `(query, relevant-doc)` pairs by deterministic
  contrastive (InfoNCE) training, so Recall@k climbs as feedback accumulates. Distilled from
  `scirust-retrieval`'s `contrastive` + `feedback` (which use `scirust-core`'s autodiff) — reimplemented
  with a **seeded** xorshift RNG, **fixed-order `f32`**, and a **hand-derived analytic gradient**
  (gradient-checked against finite differences, so the math is verified not trusted): no `scirust-core`,
  no rayon. `examples/retrieval_improvement.rs` shows Recall@1 climbing **8% → 100%** across cycles on a
  deliberate *disjoint-vocabulary* gap (query and answer share no term — base retrieval is at chance),
  bit-for-bit reproducible. Gated by `RetrievalAccess::unlock` behind CCOS's own #29 ed25519 license
  (new `Feature::AdaptiveRetrieval`) — reusing the offline, deterministic, no-FFI license rather than
  linking `scirust-license` (one fewer dep; the node-locked `$1/machine/month` model would come from the
  clean `scirust-license` if wanted). 5 tests; always-compiled, zero new dependencies.

- **Pure semantic retrieval (`ccos::retrieval`) — distilled from SciRust, challenges RAG, measured.**
  A dependency-free distillation of SciRust's `scirust-retrieval` pure modules over the embeddings CCOS
  already owns: `vector` primitives, an exact-cosine `DenseIndex`, a classic `Bm25Index`,
  `reciprocal_rank_fusion`, `SemanticRetriever` / `HybridRetriever`, the five ranking `metrics`
  (Recall@k, Precision@k, MRR, MAP, nDCG@k), and a `CcosEncoder` bridge (`Encoder` over the TF-IDF
  embedder). **Distilled, not linked** — `scirust-retrieval` depends on `scirust-core`
  (`default = ["rayon"]` + `nalgebra`/`ndarray`), and linking it would drag rayon's non-deterministic
  parallel `f32` reductions into the build, breaking `replay == live` and CCOS's zero-dep/air-gappable
  identity (the exact #14 trap); the retrieval algorithms themselves are pure, so they're reimplemented
  with fixed-order `f32` and hand-derived oracle tests (matching SciRust's own vectors). The benchmark
  `examples/pure_retrieval_vs_rag.rs` scores all three retrievers on CCOS's own `src/` corpus + AST
  dependency ground truth: **pure dense reproduces ccos's lexical RAG bit-for-bit** (24/52/66
  Recall@1/5/10, MRR 0.626 — a faithful-distillation check; absolute figures track the live `src/`
  corpus, qualitative result is stable), the hybrid trades slightly on this structural task (an honest
  negative), and the decisive win is **determinism** (every number reproducible bit-for-bit, zero RNG,
  zero generative step). Zero new dependencies; always-compiled.
  See `docs/MEASUREMENT_pure_retrieval.md`.

- **Call-graph Slice 3 (#23) — `x.bar()` receiver-type inference.** A method call `x.bar()` names the
  method but not the type `x` belongs to, and CCOS stores a method as a flat `sym:<file>:bar` symbol, so
  when two types both define `bar` the name is ambiguous and the resolver (precision-first) skipped it —
  dropping the `caller → callee` edge. #23 closes this in two **resolve-uniquely-or-skip** halves. The
  parser infers a local's concrete type from four syntactically-certain idioms only — a typed param, a
  `let` annotation, a constructor `Foo::new()`/`default()`/`with_*()`, and a single-segment struct
  literal — guarded by a PascalCase head (separates a type `Foo::new()` from a module fn `foo::new()`),
  generic-param + std-wrapper exclusion, and **poison-on-conflict** (a name bound to two types,
  re-`let`, or reassigned is dropped), then emits a `Foo::bar` callee. The resolver builds a
  `(type, method) → symbol` index from each `impl` block (carrying per-bucket cardinality, so a
  same-final-name type homonym is ambiguous → skipped, never last-writer-wins) and resolves a 2-segment
  `A::b` callee by trying **both** interpretations — `A`-as-module and `A`-as-type — linking only when
  they agree or exactly one resolves. A wrong inference would mint a *false* call edge (strictly worse
  than the data-ref case), so everything outside the idiom whitelist is dropped; the bonus is that
  explicit `Type::assoc()` calls now resolve too. The new edges are resolution-owned, so `replay == live`
  and eager ≡ batch hold (the property test and a real-codebase `analyze → replay` round-trip both pass).
  `examples/method_crux.rs` + `docs/MEASUREMENT_method_crux.md` measure it on an **adversarial twin**
  (`render` on two types): **3/3 cross-file method edges recovered, 100 % precision, zero false edges**.

- **`ccos stdin` — pipe a JSON op-stream through an ephemeral in-memory graph.** The persistence-free,
  pipe-friendly sibling of `ccos memory`: reads the same newline-delimited ops (`ingest` / `recall` /
  `failure` / `verify` / `stats` / …) from stdin and prints one JSON response per op, with no workspace
  file. The op-loop is factored into a shared `run_op_stream`, so `ccos memory` (persistent) and
  `ccos stdin` (in-memory) stay in lockstep. (Also un-breaks the CI smoke step, which already invoked it.)

- **SciRust fusion (#14a) — distilled incremental LSA: linear ingestion + contradiction-aware
  retrieval.** After inspecting the SciRust repo, the verdict was **distill, not link** — its SVD is a
  `nalgebra` wrapper with no incremental update, and depending on `scirust-core` pulls rayon-parallel
  non-determinism that would break `replay == live`. The key insight: CCOS's LSA factors through the
  Gram matrix `C = MᵀM` (fixed `dim × dim`), a **sum of per-document outer products** — so a batch just
  *adds* its contributions. New `lsa::IncrementalLsa` folds a batch in **O(batch)** (vs the O(N) full
  recompute) and is **bit-exact** versus a single batch over the same documents (so `replay == live`
  holds); `lsa::weighted_lsa_projection` scales each document by its causal authority *before*
  reduction. The judge `examples/scirust_vs_rag_crux.rs` measures both axes: **ingestion ~5.5× faster
  at 600 docs** (incremental O(N) vs full O(N²), the gap growing with N), and **contradiction-aware
  retrieval 2/2 vs blind 512-chunk RAG 1/2** on a Conflict of Origins (the refuted source crushed to the
  bottom) — with the honest finding that the *retrieval-time* belief gate (`latent cosine × authority`),
  not the pre-reduction weighting alone, is what suppresses the contradiction. Deterministic,
  dependency-free, SciRust never modified. See `docs/MEASUREMENT_scirust_fusion.md`. The live wiring lands
  in **#14b** (below).

- **SciRust fusion (#14b) — the causally-weighted latent space, wired into live recall.** `CcosMemory`'s
  semantic-recall re-ranking now builds its LSA projection from a **causal-topology-weighted** Gram: each
  document is scaled by `(1 + λc·centrality)·(1 + λa·authority)` — `spectral::eigenvector_centrality`
  (max-normalised to `[0,1]`) × the node's Q-Page net belief (new batched `MemoryGraph::qbeliefs`, one
  `O(edges + nodes)` pass instead of `O(N·edges)`) — *before* the reduction, so the latent space is shaped
  by what the causal graph deems important and the Q-Page deems trustworthy, not raw term frequency. It is
  **version-cached** (an `O(1)` hit between graph mutations, replacing the full per-query LSA recompute the
  old path paid). The honest design call: a *global*-weight Gram cannot be both ingest-order-incremental
  *and* bit-exact-rebuildable from a snapshot (adding a doc changes every doc's centrality, and an `f64`
  sum is order-sensitive), so live recall **re-folds in canonical id order per version** — buying bit-exact
  **`live == reload`** and **eager ≡ batch** (both property-tested), while the `O(batch)` as-of-ingest
  `IncrementalLsa` stays the append-only **streaming** primitive. Four tests pin the moat
  (`weighted_lsa_model_is_order_independent`, `…survives_a_reload`,
  `causal_weights_are_deterministic_and_rise_with_evidence`, and a recall-path integration); the refined
  `examples/scirust_vs_rag_crux.rs` + `docs/MEASUREMENT_scirust_fusion.md` §C document it. Always-on (no
  new feature gate), deterministic, dependency-free, SciRust never modified.

- **`ccos doctor` + deployment guide — frictionless server install (deployment-DX).** A read-only
  self-check command (`ccos doctor [--json]`) reports the build profile (debug vs release), target
  arch/os, compiled features (llm / license / syn-parser / learned-embed / mimalloc), active parser,
  license tier + whether a real vendor key is baked in (vs the fail-closed placeholder) + token
  presence, MCP readiness, and actionable **warnings** (debug build, missing feature, placeholder key,
  unverified token) — the first thing to run on a new host. New `docs/DEPLOYMENT.md` (the
  `--release --features llm,license` build, the install, the MCP config pointing at the *release*
  binary, the Pro-key setup, the fsync-durability note) and `scripts/install.sh` (one-shot build →
  install → doctor). Surfaces the real gotchas: the `ccos` bin **requires `llm`** (a bare
  `cargo build` makes no binary), and Pro is fail-closed until a vendor key replaces the placeholder.
  Adds `license::embedded_key_is_set`.

- **`spectral::temporal_profile` — the belief "fever curve" as a reusable primitive (#13).** The
  `temporal_tensor_crux` measurement (sharp, exploitable signal) is now a core API: `temporal_profile(
  `temporal_tensor_crux` measurement (sharp, exploitable signal) is now a core API: `temporal_profile(
  graphs, claims, half_life)` returns the dynamic-profile tensor `Θ[claim, {Belief, Tension}, t]` —
  each tracked claim's belief and tension (`QBelief.conflict`) across an ordered sequence of graph
  states — with accessors `tension_series` / `belief_series` / `temperature` (the aggregate system
  "fever curve") / `peak_temperature`. `AgentSession::belief_tension_timeline(claims, stride, half_life)`
  builds it over the **real recorded timeline** (replay per sampled step, offline like
  `retrieval_reward`). Pure, deterministic, ungated core — the conflict-resolution-oriented temporal
  view (how belief & tension evolve under injected contradiction → propagation → decay), as opposed to
  the flat structural-centrality reading. Tests cover the spike-on-contradiction trajectory and the
  timeline path.

- **Temporal-tensor measurement — the "fever curve" of belief (#13, design pass).** The
  spectral/centrality direction was found flat on CCOS's own small, densely-coupled graph, so the
  "temporal tensor" is re-aimed at what CCOS actually *is* — a conflict-resolution engine.
  `examples/temporal_tensor_crux.rs` records the dynamic-profile tensor `Θ[node, component, t]`,
  `component ∈ {Belief, Tension}`, across a deterministic **Conflict-of-Origins** crisis: a believed
  source and a conflicting (refuted) source both *cause* three decisions; on injection the refutation
  propagates one causal hop and the decisions' **tension spikes together** (0 → 0.49), then the
  knowledge half-life **decays** it back (0.49 → 0.20) — the fever breaks on its own. The origins stay
  cool (each is one-sided); the heat emerges only where conflicting origins *meet*; and a contested
  node halts the wavefront (no cascade — conflict is localized, not spread). The signal is sharp and
  legible, so the dynamic belief/tension profile is a real primitive — a client-facing real-time fever
  chart of the knowledge base facing injected misinformation. Deterministic (logical clock, sorted
  propagation, no RNG) ⇒ `replay == live`. See `docs/MEASUREMENT_temporal_tensor.md`. Productionizing
  it (a `spectral::temporal_profile` primitive + a CLI / MCP surface) is the next slice.

- **The three Pro license behaviors, built and gated through `require()` (license slice 29b —
  completes #29).** Each gated feature now has a real implementation; the **core is never touched**,
  only the advanced surface:
  - **Custom per-source authority weights** — `AgentSession::set_custom_authorities` (a
    `CustomAuthorityMap` of source → weight), gated by `Feature::CustomAuthorityWeights`. Gated at
    **install-time**, not assert-time, so an unlicensed session is **never degraded**: assertions always
    apply, just with their uniform per-call authority. The override is folded into the logged
    `Op::Assert` weight, so **`replay == live` stays exact** with no map to persist.
  - **Tension visualization** — `ccos tensions <snapshot> [--min N] [--limit N]`: the contested Q-Page
    claims (`conflict ≥ min`) ranked by tension with a compact bar (`MemoryGraph::claim_beliefs` +
    `memory::render_tension_bar`). Gated by `Feature::TensionVisualization`.
  - **Audit reports** — `ccos audit <snapshot> [--json] [--min N]`: a belief / conflict / provenance
    report per asserted claim (supporting + contradicting evidence) plus hash-chain integrity. Gated by
    `Feature::AuditReports`.
  `Licensing` is threaded onto `AgentSession` (loaded fresh at `open`, never serialized → replay-safe);
  CLI commands obtain it via the new `Licensing::detect(now)`. A locked feature emits exactly the
  announced `require()` refusal and the command exits 0 — **announced, never silently degraded**. Tests:
  the community-refuses / Pro-applies / replay-matches gate, `claim_beliefs` conflict-ranking, and the
  tension renderer; a CLI smoke confirms the locked path.

- **Offline Pro-license verifier — ed25519, zero-knowledge, fail-closed (`src/license.rs`, the
  `license` feature; license slice 29a).** The gate scaffolding (tiers, the three Pro `Feature`s,
  `Licensing::require()` with explicit *no-silent-degradation* logging) gains its actual trust spine:
  an `Ed25519Verifier` that checks a locally-signed token against a **baked-in public key** — no
  network, no telemetry, nothing leaves the host (a customer can run air-gapped). The token is a
  JWT-like `base64url(payload).base64url(signature)` over `{licensee, exp}` (base64url hand-rolled, so
  the only new dependency is `ed25519-dalek`, optional and absent from the default build). A single
  `load_license_blob` loader reads `$CCOS_LICENSE` (inline token) or the license file
  (`$CCOS_LICENSE_FILE` / `~/.config/ccos/license`); a new `ccos license` command reports the active
  tier, licensee and expiry. The public key shipped in this tree is an **all-zero placeholder, so the
  default build licenses nothing (fail-closed)**; a deployment pastes its own key with the
  `examples/license_sign` keygen/sign tool (the private seed never lives in this tree). A
  signature-valid but expired token reads as community while keeping the licensee for the audit log —
  gated, never silently degraded. Tested (CI runs the `license` feature): sign→verify→Pro,
  tamper / wrong-key / malformed → rejected, expiry, base64url round-trip, fail-closed placeholder.
  **Next (slice 29b):** build + gate the three Pro behaviors (custom authority weights, tension
  visualization, audit reports) through `require()`.

- **Cognitive distillation — the `Extractor` pipeline + Conflict-of-Origins resolution
  (`src/extractor.rs`).** Turns raw text into Q-Page `Assertion`s (`{claim, source, stance,
  authority}`) — the auto-detection of `Supports`/`Contradicts` edges that slice 1 left as manual
  assertions. The `Extractor` trait keeps it **provider-agnostic**: a deterministic `MockExtractor`
  drives the bench and tests with no model, and an `llm`-feature `LlmExtractor` distills the same shape
  from text via the configurable LLM backend. Extraction is the only non-deterministic step and runs
  once at ingest; its output is recorded as replayable `assert_*` / `Op::Assert` events, so a replay
  never re-calls the model (`replay == live`). Each assertion carries a per-source **authority** in
  `[0, 1]` (the evidence edge weight), and `QBelief::is_validated(min_belief, max_conflict)` is the
  strategic gate — believed-enough AND not-too-contested. Measured by `examples/conflict_of_origins.rs`
  / `docs/MEASUREMENT_conflict_of_origins.md`: as a dissenting source's authority `β` rises, the
  claim's belief slides `+0.47 → −0.03` (the more credible origin wins the direction), `conflict`
  climbs `0 → 0.65`, and validation flips off at `β = 0.30` — a defensible, inspectable resolution a
  flat or majority store cannot express.

- **Q-Page belief propagation — single deterministic hop (`MemoryGraph::propagate_beliefs`).** Belief
  revision across the causal graph: for every `Causes` edge `A → B` whose source claim `A` is
  *resolved* (`|qbelief.belief| ≥ resolve_threshold`), a derived, **attenuated** evidence edge is added
  on the effect `B` — `Supports` from a believed cause, `Contradicts` from a refuted one, weight
  `edge.weight · damping · |belief|`. So a claim with no direct evidence inherits a weaker,
  correctly-signed belief from the causes it depends on — something a flat evidence store cannot do.
  Deterministic (collect read-only, sort, add; `add_edge` dedups ⇒ idempotent); self-loops and
  unresolved causes are skipped. **One hop:** the signal attenuates below the threshold, so the
  wavefront stops rather than cascading (measured in `docs/MEASUREMENT_propagation_crux.md`: an effect
  inherits `±0.31` from a `±0.75` cause, while a 2-hop claim stays `0`). Multi-hop accumulation with a
  scheduler, and an `Op::Propagate` for replay, are the next slice.

- **Q-Page decay — knowledge half-life (`MemoryGraph::qbelief_decayed`).** A time-decayed view of a
  claim's belief: each evidence edge's weight is scaled by `0.5^(age / half_life)`, where `age` is the
  clock ticks since the edge was asserted (`created_at` vs the current `clock`). Lazy and pure
  (computed on demand, no stored decay state), so it stays deterministic and `replay == live` holds,
  and it never mutates or deletes history — only the *current* weight of an old edge fades. A fresh
  (re-)assertion carries full weight, so recent evidence outweighs an ageing one: a stale,
  never-reaffirmed dissent that plain `qbelief` would treat as an eternal deadlock resolves on its own
  as it ages. Measured in `docs/MEASUREMENT_decay_crux.md`: with a one-off objection aged against a
  fresh support, `conflict` collapses `0.67 → 0.06` (and `belief` climbs `0 → +0.50`) as the
  objection ages, versus a frozen `0.67` under plain `qbelief`. `half_life` is a caller parameter
  (domain-dependent); per-class half-life and retrieval-path decay are follow-ups.

- **Q-Page dual-evidence belief layer — contested-knowledge memory (`EdgeType::Supports` /
  `EdgeType::Contradicts`).** A claim node carries two opposing, explicitly-asserted evidence
  surfaces — the affirmative `S_A` (`Supports`) and the negative `S_¬A` (`Contradicts`) — and
  `MemoryGraph::qbelief` derives `{support, contradiction, belief, conflict}` from a claim's incoming
  edges (each edge's weight is the asserting **source authority**, clamped to `[0, 1]`). It is **pure
  and derived** (no stored state, so snapshots are unchanged and `replay == live` holds): `belief` is
  the **signed** support fraction `(s − c)/(s + c + ε)` ∈ `[−1, 1]` (`0` at no/balanced evidence; sign
  = direction, magnitude = strength), `conflict` the **geometric** balance `2·√(s·c)/(s + c + ε)` ∈
  `[0, 1]` — high *only* when both surfaces carry weight, the resolution signal a similarity index
  cannot represent (relatedness has no polarity); `ε = 1` is a unit prior (sparse evidence stays near
  neutral). The two `EdgeType`
  variants are appended additively (old snapshots never contain them). Contradictions are **explicit
  cognitive events** — `CcosMemory::assert_support` / `assert_contradiction` (agent API, recorded in
  the hash-chained audit) and an `AgentSession` `Op::Assert` replayed in `replay_to`, so an
  agent-asserted contradiction reconstructs identically (`replay == live` for contested knowledge,
  not just for ingested structure). Measured in `docs/MEASUREMENT_contradiction_crux.md`: a
  refutation's lexical similarity to its claim falls *inside* the band of the confirmations, so no
  cosine threshold separates support from refutation — the typed edge does, and `conflict` flags the
  contested claim. Auto-detection (rules / NLI), resolution propagation, and decay are later slices.

- **Data-flow semantic edges — `EdgeType::DataFlow` (ROADMAP P1.3, the second half of "semantic
  edges").** The `syn` AST captures in-body references to module-level `static`/`const` items
  (Slice 1: bare `SCREAMING_SNAKE` value paths — the Rust convention, which precisely excludes
  PascalCase types and snake_case fns/locals). A deterministic whole-graph pass
  (`MemoryGraph::resolve_data_flow`, run after call resolution) links each `reader → item` with a
  `DataFlow` edge when **exactly one** resident `static`/`const` of that name exists graph-wide
  (**resolve-uniquely-or-skip**, so a wrong edge is never invented) — the shared-global-state
  channel that call and import edges miss (a function reads a global defined in a file it never
  imports by name). The graph node carries `NodeType` not `SymbolKind`, so the parser marks the
  data-symbols at ingest; the references live in a transient `#[serde(skip)]` field (only the edges
  persist, rebuilt on the replay re-ingest → `replay == live` holds). Off on the heuristic path.
  A **scope guard** excludes locally-bound names (parameters, `let`s, fn-local `const`/`static`)
  from capture, so a local never mislinks to a same-named global — closing the cardinal false-edge
  an adversarial review found. Slice 1 covers bare references resolved global-unique; **Slice 2**
  (below) adds qualified `m::CONST`. Same-module disambiguation, write/read direction, and the rare
  residual (a bare `SCREAMING`-cased `use`-imported enum variant coinciding with a global const)
  remain later slices.

- **Data-flow Slice 2 — qualified `m::CONST` references.** In-body value paths whose *last* segment
  is `SCREAMING_SNAKE` (`config::MAX_RETRIES`, `crate::limits::MAX`, `self::FOO`) are now captured
  with their full `::`-path and resolved through a shared `resolve_qualified` helper — the *same*
  machinery qualified calls use, but against a **data-symbol-only** index, so a qualified ref can
  only ever land on a `static`/`const`, never a fn. **Resolve-uniquely-or-skip**: the module prefix
  is pinned to a defining file (crate-rooted, or an alias expanded through the file's imports), with
  no fallback to the bare global index — an unresolvable/ambiguous qualified ref adds no edge. The
  local-binding scope guard extends to qualified paths (a locally-bound head segment is skipped).

- **`data_flow_crux` measurement** (`examples/data_flow_crux.rs`, `docs/MEASUREMENT_data_flow_crux.md`).
  The data-flow analogue of the call/import crux: a reader names the const it reads (partial lexical
  signal), but two **co-readers** of the same global share only that one concept — swamped by their
  disjoint domain vocabulary, a true co-reader typically ranks below an unrelated decoy (lexical
  recall@1 ≈25 %, MRR ≈0.49). The data-flow graph recovers the shared-state link by construction —
  the cross-vocabulary channel a vector retriever cannot see.

- **Call-graph polish — renamed-import aliases & cross-impl-block self-calls.** Two precision gains,
  both resolve-uniquely-or-skip and deterministic: (1) `use a::b as c` now binds the local alias `c`
  to target `a::b` (top-level, in groups, nested groups), so a call `c()` / `c::X` rewrites onto the
  real target and never mislinks to a same-named sibling; (2) `self.method()` / `Self::method` now
  resolves across **all** impl blocks of a type — a `BTreeMap<type, methods>` unions every inherent
  and trait impl, so a self-call reaches a method defined in a *different* block of the same type,
  while a blanket `impl<T> .. for T` (type-variable Self) and two distinct types sharing a method
  name are strictly kept from cross-linking.

- **Spectral primitive — deterministic eigenvector centrality (`src/spectral.rs`, #13 first slice).**
  `eigenvector_centrality` computes the textbook `A x = λ x` ranking by power iteration on the
  **symmetrized**, `A + I`-shifted adjacency (the shift defeats the bipartite oscillation a DAG-like
  code graph would otherwise cause), L2-normalized, processed in sorted node order for byte-identical
  runs. Dependency-free and pure (read-only, not wired into scoring/CLI) — a clean brick complementary
  to the damped `MemoryGraph::eigencentrality`. Spectral regions, the temporal tensor, and any
  `scirust` fusion are deliberately deferred to a later design pass.

- **Call-graph semantic edges — `EdgeType::Calls` (ROADMAP P1.3, Slice 1).** The `syn` AST
  now extracts in-body function-call sites; a deterministic whole-graph pass
  (`MemoryGraph::resolve_symbol_calls`) resolves each `caller → callee` via a strict
  import-scoped → same-module → global-unique ladder (**resolve-uniquely-or-skip**, so a wrong
  edge is never invented) and adds a `Calls` edge — the fn→fn structure import edges miss.
  Slices 1–3 cover bare (`foo()`), qualified (`crate::m::foo()`, and `alias::foo()` expanded
  through the file's imports), and **`self.method()` / `Self::assoc()`** calls (resolved in the
  caller's own module, never via imports); arbitrary `x.bar()` (unknown receiver) stays deferred.
  Off on the heuristic path; call-sites held in a transient field so only the edges persist
  (snapshots unchanged, `replay == live` holds). Measured (`docs/MEASUREMENT_call_crux.md`,
  adversarially reviewed): a vector retriever recovers **direct** calls (it names the callee,
  recall@1 75 %) but collapses on **transitive** 2-hop calls (recall@1 0 %), which the call
  graph reaches by traversal — the call-level analogue of the import crux.

- **Node lifecycle state (`NodeState`: `Stable` / `Working` / `Orphan`).** Separates a
  node's *health/attention* from graph *topology* so it can't pollute the structural
  signal — a per-node enum field (not a tensor dimension; a node's state is single-valued).
  `Orphan` is excluded from the centrality calc and evicted first regardless of recency;
  `Working` is pinned resident as the current focus even as recency decays. Off by default
  (`Stable`) ⇒ centrality, score and snapshot are byte-identical until a state is set;
  `set_node_state` invalidates the centrality caches. See `docs/MEASUREMENT_node_lifecycle.md`
  (pillar in-degree 12→6 once dead dependents are excluded; real-work retention 1/6→6/6 when
  freshly-edited dead code is labeled). Companion to the off-by-default **eigenvector
  centrality** mode (`CentralityMode::Eigenvector`) added earlier in the series.

- **COLD entry-count bound — an on-disk husk index (slice 5c, "Lever 2"; the
  `O(1)`-resident COLD tier).** Slices 3–5b bounded each COLD entry's *size*; this
  bounds their *count*. The deep-spill tier no longer keeps one `BTreeMap` node per
  husk in RAM — husks live in a hand-rolled, dependency-free LSM-lite
  (`src/cold_index.rs`): immutable sorted segments with a sparse resident index, a
  memtable + flush, tombstone deletes + compaction, and a bounded LRU read cache, each
  verified standalone by a model-check property test before wiring. `MemoryGraph`'s
  resident `cold_deep` map is gone; `cold_neighbours` is answered `O(degree)` by a
  keyed on-disk **reverse-adjacency** index (`<dir>.radj`), and `flush_cold_tier`
  durabilises the indices at checkpoint. Measured (`examples/cold_count.rs`): **≈2 B
  per husk resident** (vs 146 B fully resident), 1 GiB at **~537 M husks**. Lossless
  round-trip, no-leak GC and crash recovery are property-/model-checked;
  dependency-free (`std` only); `replay == live` is untouched (the event log is the
  source of truth, the cold tier a rebuildable cache). See
  `docs/DESIGN_cold_entry_count.md`.
- **Natural-language queries match code identifiers (subword tokenization).** The
  TF-IDF tokenizer now splits each token on `snake_case` and `camelCase` boundaries,
  so `connection_pool_acquire` yields `connection`, `pool`, … — a query like
  "connection pool acquire" shared *zero* tokens with it before, making the semantic
  signal zero. Measured (`examples/identifier_recall.rs`): 6/6 NL queries recall their
  identifier-named target at rank ≤2 (overlap 0 → 3/3); on the `lsa_rerank` corpus the
  topic target's mean rank improves 11.8 → 2.0. Deterministic.
- **LSA re-ranking stage for recall (`set_lsa_rerank`, opt-in).** Wires the LSA
  embedder where #39 measured it earns its keep — *re-ranking* the recalled region
  (recall@k≥5), not entry selection (recall@1=0). A node's score is multiplied by
  `1 + w·max(0, cosine)` (only ever promotes). Measured (`examples/lsa_rerank.rs`):
  target mean rank 11.8 → 2.1; the honest limiter is entry selection (synonyms score
  ≈0), which re-ranking can't repair. Deterministic, `replay == live` untouched.

### Changed

- **Spill stubs hold a raw `[u8; 32]` hash, not a 64-char hex `String`** — −56 B and
  one fewer heap allocation per COLD spill/husk stub (serialized form unchanged via
  serde-hex). **Snapshots are byte-canonical** — the resident `nodes` `HashMap` now
  serializes in sorted key order, so identical state ⇒ byte-identical snapshot, not
  merely identical *sorted* hash. Both verified by property tests.

### Fixed

- **A COLD spill blob leaked on page-in.** When `page_in` faulted a blob back and
  dropped its last reference (content folded inline, or a husk removed), the on-disk
  blob became unreferenced but was never reclaimed — a slow disk leak no later
  `remove` could find. Caught by a new cross-tier hardening property test (lossless
  round-trip + no orphaned blobs under random op streams); `page_in` now reclaims the
  dropped blob in both paths. The headline `replay == live` invariant is now also
  **fuzzed** (byte-identical full-graph hash over random op streams), as is snapshot
  round-trip and the on-disk index's model.
- **An on-disk lossless codec for the spill store (LZSS, dependency-free).** Spill
  blobs are LZSS-compressed on write and verified on read (the key is the original
  content's SHA-256, so dedup is unchanged and any codec bug is a recoverable
  cold-miss); a `proptest` round-trip pins `decompress(compress(x)) == x`. Closes the
  "no codec yet" gap.

- **Structural-centrality scoring term** (from a design discussion — the one idea in
  that conversation CCOS's score didn't already have). `compute_node_score` gains a
  `w_centrality · ln(1 + in_degree)` term: a hub (a shared module / interface many
  nodes depend on) is structurally more important than a leaf, independent of recency.
  **Off by default** (`w_centrality = 0.0`, `skip_serializing_if` elides it) ⇒ the
  score is byte-identical to before and replay/snapshots are unchanged. In-degree is
  computed via a cache keyed on `edges.len()` (edges are append/`retain`-only) and is
  only built when the term is enabled, so the default path pays nothing.
  `CCOS_W_CENTRALITY` overrides it, and the log-tuner
  (`AgentSession::tune_recall_weights`) now learns it too (absolute candidates, since
  a multiplicative move can't escape 0). Deterministic.
- **COLD-tier deep-spill — bound the per-entry *resident* metadata, losslessly**
  (slices 5 & 5b; measure-then-fix, see `docs/MEASUREMENT_cold_ram.md`, reproduce with
  `examples/cold_ram.rs`). A measurement first showed slice 3 left the COLD tier's
  dominant RAM cost as per-entry **metadata** — ~2.8× the spilled content, ~60% of it
  edges — and that lossy edge-contraction is the *wrong* lever (it inflates that edge
  cost on hubs). So `set_cold_resident_budget(Some(b))` drives resident COLD metadata
  toward `b` by **deep-spilling** the coldest entries: each is archived *whole* to the
  content-addressed store and represented in RAM only by a compact `DeepHusk`
  (body-blob stub + the neighbour **ids** that `cold_neighbours`/region paging need),
  held in a separate `cold_deep` map. Because the husk is far smaller than a full
  `ColdNode`, *every* entry shrinks when spilled and the budget is actually reached —
  resident COLD metadata **halves (−50%)** on the 120K-node fixture (slice 5's
  full-husk first cut had stalled at ~11% against the `size_of::<ColdNode>()` floor).
  **Lossless** (the node faults back, hash-verified, on `page_in`; a missing/tampered
  body is a cold-miss, never a half-restore), **deterministic** (coldest-first), and
  **off by default** (`cold_deep` is `serde`-elided when empty and the budget is a
  runtime knob ⇒ byte-identical default snapshot/replay). Deep husks are *terminal*
  (excluded from further spill/compaction). Shrinks edges to ids — never adds bridge
  edges — so hubs get cheaper, not the O(degree²) blow-up contraction would cause.
  Observable via `cold_deep_spilled_count` / `is_deep_spilled`. **Honest scope:** this
  bounds the per-entry resident *size*; bounding the entry *count* (an on-disk husk
  index) remains future work.

### Performance

- **Per-recall caches make recall up to ~5700× faster at scale** (the perf pass —
  measure-then-fix; see `docs/MEASUREMENT_latency.md`, reproduce with
  `examples/recall_latency.rs`). A latency benchmark showed recall was super-linear
  in corpus size because every query recall rebuilt derived structures from scratch:
  `around`/`task` re-ran the whole **region clustering** (`initialize_regions`), and
  `semantic`/`hybrid` additionally re-fit the **embedding store** (and the LSA
  eigensolve under `learned-embed`). `CcosMemory` now memoises both behind a **graph
  version counter** bumped on every resident-graph mutation; a cache is reused only
  at the same version, so it is **never stale** and the result is byte-identical to a
  fresh rebuild — **determinism and `replay == live` are preserved** (a new test
  asserts a post-warm ingest is visible to the next recall; the full replay suite
  still passes). At 2000 nodes: `around` 75 ms → 13 µs, `semantic` ~42×, `hybrid`
  ~21×. (The first recall after a mutation still rebuilds; the win is on the repeated
  recalls between mutations — the common pattern.)

### Added

- **Recall-strategy measurement (`examples/recall_eval.rs`) + honest findings**
  (`docs/MEASUREMENT_recall.md`). An LLM-free benchmark on a synthetic corpus with
  ground-truth relevant files, comparing working-set / lexical / semantic / hybrid
  recall at a tight budget across three task types. Result: **hybrid fusion is
  measurably the best query strategy** (overall hit-rate 58% vs lexical 17% /
  semantic 21%; it alone recovers the target in the decoy+failure case) —
  validating slice A in measurement. The **opt-in LSA embedder does *not* help and
  can hurt** in CCOS's entry-selection use (drops hybrid to 38%), so it correctly
  stays off by default; the data, not assumption, sets the recommendation.
- **Opt-in learned semantic embedder (`learned-embed` feature)** — slice B of better
  retrieval, completing the arc. A new `src/lsa.rs` distils the deterministic INT4
  TF-IDF into a learned **latent-semantic (LSA / truncated-SVD) projection**: the top
  singular vectors of the corpus's document–term matrix, found by a fixed
  cyclic-Jacobi sweep (zero new dependencies, fully deterministic). It captures
  synonymy/transitivity raw TF-IDF can't — a query term that only *co-occurs* with a
  document's terms still matches it. `CausalEmbeddings::fit_and_embed_lsa` stores the
  projected vectors and `embed_query` projects queries the same way;
  `build_embeddings` uses it only under `--features learned-embed`, so the **default
  build stays raw INT4 TF-IDF, byte-identical and replayable** (the embedder's
  `projection` field is `skip_serializing_if = None`). *Honest scope:* LSA is a
  linear distillation, not a neural model; it helps most with enough documents to
  truncate; the eigensolve adds per-build cost, hence opt-in.
- **Self-improving retrieval from the replayable log** (slice C of better retrieval —
  the CCOS-native gem). A retrieval **reward** is read straight off the hash-chained
  timeline: for each recorded recall, was the node the agent engaged *next* (a
  failure signal / page-fault) present — at file granularity — in the window that
  recall would have produced? `AgentSession::retrieval_hit_rate` reports it;
  `tune_recall_weights` learns the `ScoringWeights` that maximise it by
  **deterministic coordinate ascent, evaluated by replay** (same log ⇒ same
  weights); `adopt_tuned_recall_weights` applies them **and records an `Op::Retune`**,
  so the learned policy is auditable and **reproduced on replay** — `replay == live`
  still holds. This is retrieval that trains on CCOS's own moat: the deterministic,
  replayable causal history. *Honest scope:* the reward is a proxy (the next failing
  node = the context recall should have surfaced); the optimiser is greedy (a local
  optimum) over the four scoring weights; evaluation is one replay per candidate, so
  it is an offline/maintenance call, not a hot path.
- **Hybrid entry fusion for recall** (slice A of better retrieval). A new
  `Recall::Hybrid(text)` resolves a free-text task's entry node by
  **reciprocal-rank fusion** of three independent rankings — lexical token
  overlap, semantic INT4-TF-IDF cosine, and the causal **active-failure focus** —
  before the usual causal-region expansion. RRF compares ranks (no cross-signal
  score calibration), so a node strong on any one axis can still surface while a
  node decent across several wins; `K = 60`. The causal vote is **sparse** — it
  ranks only nodes under failure pressure, so it abstains on a quiet graph (no
  spurious id-ordered bias) and speaks for the active problem region once a
  failure is signalled (the CCOS-native attention signal). Deterministic; wired
  through `recall()`, the MCP `recall` tool (`strategy:"hybrid"`), and the runtime
  recall CLI. `Recall::hybrid(text)` constructs it.
- **Compact the coldest COLD tail → a frugal backing store** (slice 4 of unbounded
  working memory, the deepest tier). A new, opt-in
  `CcosMemory::set_cold_content_budget(Some(bytes))` keeps total COLD *content*
  (inline + spilled) toward `bytes` by **lossily compacting** the coldest entries —
  routed by kind, code is skeletonised / prose summarised / JSON crushed
  (`CausalAst` / `CausalSumm` / `CausalCrusher`, reused as pure functions), and the
  full original is discarded. Deterministic (coldest-first by causal score), and
  **observable**: `is_compacted` and `MemoryStats.cold_compacted` report the lossy
  tier. This is where "infinite working memory as a *direction*" bottoms out — at
  the floor frugality wins, and CCOS compacts to a summary, **never silently
  drops**. **Off by default** ⇒ COLD stays lossless and serialization byte-identical
  (the `ColdNode.compacted` flag is `skip_serializing_if = false`; the budget is
  `serde(skip)`). *Honest scope:* this bounds the cold **content** footprint, not
  the entry **count** (the in-RAM stub map is still O(N) — an on-disk index is
  future work); compaction is lossy and, like spill, an operational mode layered on
  the deterministic default path, not part of replay.
- **Spill COLD content to disk → RAM-bounded content, disk-unbounded** (slice 3
  of unbounded working memory). A new, opt-in
  `CcosMemory::attach_cold_spill(dir, inline_budget)` flushes the coldest COLD
  *content* blobs to a content-addressed on-disk store (SHA-256 keys — the same
  addressing as the CCR store) once resident COLD content exceeds `inline_budget`
  bytes, dropping the blob from RAM and leaving a hash **stub**. `page_in` faults
  it back **hash-verified**: a tampered, truncated, or missing blob is a cold-miss,
  never a silent empty restore — so disk spill *extends* the integrity story.
  Identical content is **deduplicated**; the flush is lossless and deterministic
  (coldest-first by causal score, ties on id). **Off by default** ⇒ no spill,
  byte-identical serialization, replay/snapshot invariants untouched (the new
  `spill` stub is `skip_serializing_if = None`; the store handle is `serde(skip)`).
  `MemoryStats.cold_spilled` / `cold_spilled_bytes` surface it (via `ccos stats` /
  the MCP `stats` tool). *Honest scope:* only the unbounded **content** moves to
  disk — per-cold-node metadata still grows in RAM (slice 4); blobs are stored
  verbatim (dedup, no compression codec yet); a snapshot taken with spill active
  references blobs by hash and needs the `dir` re-attached to restore (a sidecar,
  like a swapfile).
- **Page-fault from the COLD tier on the read paths** (slice 2 of unbounded
  working memory). A `page_fault` now resurrects cold *faulting* files (its
  per-file `signal_failure` is cold-aware), and a `recall` **around** a demoted
  node pages it — and its cold neighbours (`MemoryGraph::cold_neighbours`) — back
  into the resident graph via the new `CcosMemory::ensure_resident`, wired into
  `AgentSession::recall` / `recall_compressed` / `recall_compressed_with_feedback`.
  The page-in is a deterministic, **replayed** side effect (`Op::Recall` reproduces
  it), so `replay == live` holds. New `CcosMemory::set_max_resident` configures the
  frugal resident-window size.
- **Non-destructive eviction → a COLD tier (the "swap").** First slice of the
  *unbounded working memory* direction (frugality × available RAM). Evicting a
  node from the resident graph now **demotes** it — with its incident edges — into
  a COLD tier instead of dropping it: the resident set stays capped by
  `max_in_memory_nodes`, the backing store grows into RAM, and any node can be
  paged back (`MemoryGraph::page_in`). A `signal_failure` on a demoted node
  **resurrects it from COLD** (a page fault) instead of erroring. `MemoryStats.cold`
  surfaces the tier (via `ccos stats` / the MCP `stats` tool). Deterministic
  (sorted demotion, `BTreeMap` COLD store); snapshots stay reproducible. See
  ROADMAP for the arc (disk-spill + compaction next).
- **Wired the recent modules onto the live path.** Three capabilities that were
  in-tree but unreachable from the live recall/ingest core are now connected:
  (1) **semantic recall** — a new `Recall::Semantic` strategy resolves a
  free-text task to its entry node by INT4 TF-IDF cosine (`embeddings`), exposed
  via the MCP `recall` tool and `ccos memory`; (2) **injection signal at ingest**
  — every `IngestReport` now carries `injection_score` / `injection_flagged` from
  a shared `InjectionDetector`, so the signal is recorded on the live path, not
  only in `ccos sanitize`; (3) **learned eviction** — `MemoryGraph::enforce_paging`
  now consults `EvictionPolicy`, blending its learned keep/evict preference into
  the eviction order. The policy is **untrained by default**, in which case paging
  is byte-identical to the deterministic greedy (never worse); `train_eviction_policy`
  fits it offline. All three preserve determinism/replay; each has a wiring test.
- **Input hardening — deterministic Unicode de-obfuscation + an injection
  signal** (`sanitizer`, `hashing_tokenizer`, `injection_classifier` modules).
  Hidden-character injection vectors — Trojan-Source bidi overrides
  (CVE-2021-42574), zero-width formatting, Unicode-Tags ASCII smuggling — are
  surfaced as explicit, auditable literals (`[U+202E RLO]`) at ingest
  (default-on in `ingest_source`; clean source is borrowed unchanged, zero
  copy), with findings in `IngestReport.anomalies` and the event-log hash taken
  over the cleaned form so a replay reproduces it. A deterministic
  feature-hashing tokenizer feeds a linear log-space (multinomial-Naive-Bayes)
  injection **signal** whose weights are locked in an immutable,
  SHA-256-verified blob, with a forensic per-feature explanation of every score;
  a held-out red-team measures F1 0.90 (precision 0.87, recall 0.93). Labelled a
  *signal, not a shield* by design (evaded by paraphrase; no character pass
  solves semantic injection). New `ccos sanitize` CLI command and the
  `train_injection` / `injection_redteam` examples. See
  [`docs/SECURITY.md`](docs/SECURITY.md).
- **Reversible context compression pipeline** (`compressor` module) — the real
  *compression* pass CCOS historically lacked, sitting downstream of the causal
  MMU's selection so the graph, the scoring, the paging and the hash-chain
  replay are untouched. Three deterministic compressors: `CausalCrusher`
  (columnar JSON collapse + null-drop + string back-refs), `CausalAST`
  (skeletonizes code — strips comments / blank lines / `use` imports, collapses
  long signature runs, renames `_`-temporaries to `$n`), `CausalSumm` (TextRank
  extractive summary **biased by the causal score**). No ML model, no
  stochastic step: everything is seed-stable and total-order tie-broken, so
  the replay / `postmortem` invariants hold. Measured on this repo's source:
  30–50 % token reduction on real Rust code (run `cargo run --example
  bench_compress --release`). Zero new dependencies.
- **CCR store + `ccos_retrieve` MCP tool** — every compressed item carries a
  12-char `ccr_ref` (truncated SHA-256 of the original); the host LLM calls
  `ccos_retrieve` to fetch the full text on demand (the CCOS equivalent of
  headroom's `headroom_retrieve`). Nothing is ever lost. `RecallItem` gains an
  optional `ccr_ref` field (serde-skipped when absent, so old snapshots still
  load).
- **Cross-item near-duplicate suppression** — a distilled MinHash (64 hashes,
  3-char shingles, FNV-1a + double-hashing, seed-stable) estimates Jaccard
  similarity over the *compressed* forms within a window; near-dup items are
  replaced by `// ~dup of <uri>` (their original stays retrievable). The causal
  graph dedups cross-file; this dedups *within* a window.
- **Budget feedback loop** (`CcosMemory::recall_compressed_with_feedback` /
  `AgentSession::recall_compressed_with_feedback`) — when compression shrinks
  the window below the token budget, the freed space is *re-spent* on more
  causal nodes (a second recall pass with a grown effective budget), so the
  host gets strictly more causal signal at the same emitted-token cost.
  Monotonic and bounded (max 3 rounds); stops at convergence. Measured: +11
  causal nodes on a 4096-token task recall vs a single compressed pass, while
  staying under budget.
- **`CausalAST` v2 knobs** — `enable_ast_v2` drops pure `use` lines (the causal
  graph already encodes the dependency) and `ast_signature_collapse_after`
  collapses a run of >N one-line `fn` signatures into the first N + `// (+M
  more signatures)`. `pub use` re-exports are kept.
- **Auto-tuner** (`CausalCompressor::auto_tune`) — deterministic coordinate
  descent over the config knobs (dedup threshold/on, AST v2/collapse, summary
  length, prose on, min-chars) to minimise the compressed-token count on a
  representative sample. `eval_config` is public for external benchmarks.
- **`ccos://session/context` compressed by default** — the resource now runs
  through `recall_compressed` unless `CCOS_COMPRESS_CONTEXT=0` (A/B escape
  hatch). The linearised form appends `// ccr_ref=… (call ccos_retrieve for
  full)` so the host knows the handle.
- **SCIRUST counterparts** — the algorithms were distilled from
  `scirust-nlp-advanced`, which gains four new modules: `bloom` (Bloom filter),
  `lsh` (MinHash-LSH band-and-bucket), `trie` (byte-radix shared-prefix
  compaction), `huffman` (canonical reversible entropy coding).
- **Causal embeddings** (`embeddings` module) — a zero-dependency TF-IDF
  embedder with a hashed vocabulary (128-dim default) whose vectors are
  **INT4-quantized** (distilled from SCIRUST's `elastic_kv_cache.rs` SLHAv2
  scheme: grouped absmax symmetric INT4, cosine error < 0.01). The
  [`CausalEmbeddings`] store is ~4× smaller than `f32` and powers a
  [`CcosMemory::semantic_entry`] for `Recall::Task` that down-weights
  ubiquitous tokens via IDF (catches "connection pool" → `db.rs` where a
  raw lexical overlap is distracted by the ubiquitous `fn`). Deterministic:
  the hashed vocab + `BTreeMap` store serialize bit-stable.
- **RL eviction policy** (`eviction_policy` module) — a tabular Q-learning
  agent (distilled from SCIRUST's `scirust-rl-algo::TabularQLearner`) that
  learns when to evict a node from the paging window based on a 4-bucket
  state (score / recency / failure-pressure / size). 162 cells max, serializes
  as a `BTreeMap`, bit-reproducible. **Advisory**: [`should_evict`] returns
  `false` when untrained, so the deterministic greedy stays the authority
  until the policy has learned a preference — turning it on is never worse
  than the status quo. Training is offline (`fit` over a replayed timeline
  with reward shaping for keep/evict decisions).

### Fixed

- **Audit pass 4 (hardening the unbounded-memory + retrieval slices).** Four
  adversarial auditors (one each for determinism, `replay == live`, default-path
  byte-identity, and resource bounds) confirmed the crown invariants hold on the
  default path, and surfaced three real issues now fixed:
  - **Spill-blob garbage collection (was an unbounded disk leak).** The on-disk
    spill store (`ColdSpill`) had no deletion path, so re-ingesting, removing, or
    compacting a previously-spilled node orphaned its blob forever. Added
    `ColdSpill::remove` and a **dedup-safe** `release_blob_if_orphan` (a blob is
    deleted only once no COLD entry still references its hash), wired into
    `upsert_node`, `remove_node`, and compaction. (Off by default — only matters
    when a spill store is attached.)
  - **Compaction floor no longer busy-loops.** An un-shrinkable cold entry was
    re-selected (and its blob re-read from disk) on every ingest while the tier
    stayed over budget. Such entries are now parked with a new `ColdNode.at_floor`
    flag and excluded from future compaction candidates (a fresh ingest drops the
    shadow, so the flag never goes stale). `skip_serializing_if` keeps the default
    serialization byte-identical.
  - **LSA corpus order pinned for determinism.** `build_embeddings` now sorts
    nodes by id before fitting, so the `learned-embed` LSA Gram-matrix f64 sum is
    independent of `HashMap` iteration order (the one place determinism rested on
    float-associativity rather than a fixed order). The default TF-IDF path was
    already order-free.

  Deferred to a perf pass (documented, not regressions): per-ingest `O(cold)`
  budget re-scans, per-recall `cold_neighbours` scan, and the per-recall
  embedding-store rebuild — all to be addressed with incremental counters/indices
  and a cached, dirty-invalidated embedding store.

### Changed

- **Unified the two snapshot types.** `persistence::RuntimeState` was a
  field-for-field duplicate of `persist::KernelSnapshot`; it is now a type alias
  for it (one state type, two on-disk layouts — single-file vs three-file
  directory). The load-time integrity check (both hash chains valid + no dangling
  edges) moved into the shared `KernelSnapshot::verify_integrity`, now also
  reachable via `KernelSnapshot::load_verified` and reused by the runtime restore.
  No caller changes (audit pass 3, section B).
- **Encapsulated `MemoryGraph.{nodes,edges}`** (now `pub(crate)`). External
  callers go through read accessors — `node`, `node_mut`, `node_ids`,
  `node_entries`, `node_values`, `contains_node`, `edges()` — instead of touching
  the maps directly, so the `edges ⊆ nodes²` invariant can no longer be broken
  from outside the crate (audit pass 3, section C). Internal behaviour is
  unchanged; a minor breaking change for any external consumer that read the
  fields.
- **Repositioned, honestly.** Measurements refute "causal regions retrieve better
  than RAG": on 70 real bug-fix commits causal selection ties (and at a tight
  budget loses to) a lexical TF-IDF retriever, and the crash-trace pivot is beaten
  by RAG-over-the-error-message. End-to-end (Phase 4, 30B + compiler-in-the-loop)
  CCOS and RAG resolve equally (2/10), **but CCOS uses 6.9× fewer context tokens
  (776 vs 5366)** — efficiency, not retrieval quality, is its measured advantage.
  CCOS's contribution is relocated from *retrieval* to a **frugal, deterministic,
  replayable, auditable** agent memory. README and the paper (title, abstract,
  contributions, time-travel section, Phase-4 efficiency result, conclusion)
  rewritten accordingly.

### Added

- **Deeper page-fault propagation.** A page-fault now injects failure pressure to
  depth **3** (was 2), configurable via `CCOS_PAGE_FAULT_DEPTH` — a Jetson field run
  showed depth 2 left a 3-hop-deep cause un-pressurised (the symptom got hot, the
  cause stayed cold and was evicted under a tight budget). The depth is recorded in
  the op-log so replay reproduces the exact pressure (old logs default to the
  historical depth of 2); determinism preserved.
- **Field-data collection.** `ccos postmortem <workspace> --json` dumps an
  analytics-ready field record (version, stats, hash-chain integrity, timeline,
  compaction floor, current working set) and exits — the non-interactive way to
  archive a session (e.g. on a cron, before compaction folds older steps away).
  `scripts/fleet_collect.sh` pulls workspaces from a fleet over `rsync` and writes a
  `session.json` per node (local-first; integrity is verified offline). Because the
  timeline replays bit-for-bit, a copied workspace reproduces the field run off-site.
  See [`docs/SELF_ANALYSIS.md`](docs/SELF_ANALYSIS.md) → *Collecting field data*.
- **Durable checkpoints + bare-metal notes.** Snapshots (`.ccos`) and the op-log
  (`.oplog`) are now written **durably and atomically** (`util::write_durable`: temp
  + `fsync` + atomic rename + directory `fsync`), so a power loss or killed daemon
  can't leave a truncated file — hardening the "replayable after a crash" guarantee
  (a plain `std::fs::write` only reaches the page cache). On by default. Adds
  `scripts/jetson_repro_env.sh` (pin a Jetson to max clocks for reproducible
  measurement — `nvpmodel`/`jetson_clocks`, no `nvidia-smi`/NUMA on Tegra), an
  optional `mimalloc` allocator feature and a `target-cpu=native` build note for
  bare-metal A/B benchmarking, and [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — an
  honest triage (the kernel is <1% of an agent loop, so most low-level knobs don't
  move the needle; what matters is durability and reproducible measurement).
- **Self-analysis dogfood loop** (`.mcp.json`, `scripts/ccos_self_feed.py`,
  `docs/SELF_ANALYSIS.md`) — wires CCOS into a coding agent (Claude Code) as its
  causal memory. A project `.mcp.json` registers `ccos mcp` so the agent gets the
  memory tools natively (Mode A), and a **PostToolUse hook** is the transparent
  "hardware intercept" (Mode B): every source file the agent reads/writes becomes an
  `ingest` and every failed `cargo test/build` becomes a `page_fault`, with zero
  cognitive overhead — so `workspace.ccos` + `.oplog` accumulate a replayable record
  you then debug with `ccos postmortem`. Verified end-to-end: simulated tool events
  feed the memory and the session is walkable post-mortem.
- **MCP server** (`ccos mcp`, `mcp` module) — exposes the external-memory façade
  as [Model Context Protocol](https://modelcontextprotocol.io) tools over **stdio
  JSON-RPC 2.0**, so any MCP-compatible agent (Claude, a local agent on the Jetson)
  can use CCOS as native working memory. Dependency-free (`serde_json` only); speaks
  the standard `initialize` / `tools/list` / `tools/call` / `resources/list` /
  `resources/read` / `ping` handshake. Advertises **eight tools** (`ingest`,
  `recall`, `signal_failure`, `page_fault`, `stats`, `verify`, plus the time-travel
  pair `timeline` / `recall_what_if` — rewind to a past step and re-run a recall) and
  **two resources** (`ccos://session/context`, the self-bounding working set
  linearised for direct system-prompt injection, and `ccos://session/timeline`),
  backed by an event-sourced `AgentSession`. Optional **persistence**: `ccos mcp
  [workspace.ccos]` (or `CCOS_MCP_WORKSPACE`) reloads the checkpoint on start and
  re-checkpoints after every memory-changing call — the same snapshot format as
  `ccos memory`, so the two transports share one workspace. The **cognitive timeline
  persists too** in a `<workspace>.oplog` sidecar (the op-log plus its replay
  baseline), so `timeline` / `recall_what_if` time-travel spans the whole recorded
  history **across restarts**; a stale sidecar that no longer reproduces the snapshot
  self-heals to the snapshot (the memory is never corrupted by a stale log). The
  op-log **compacts** to stay bounded for a long-running daemon — older ops fold into
  the baseline past `CCOS_OPLOG_MAX` (default 512), keeping the last `CCOS_OPLOG_KEEP`
  (default 128) replayable; compaction is index-stable and never touches the live
  memory (only deep historical rewind is traded away). Point a client's stdio
  transport at it: `{"command":"ccos","args":["mcp","workspace.ccos"]}`. See
  [`MEMORY_INTERFACE.md`](docs/MEMORY_INTERFACE.md#serving-over-mcp-ccos-mcp).
- **Interactive post-mortem debugger** (`ccos postmortem [workspace.ccos]`,
  `postmortem` module) — a "GDB for the agent's memory": load a persisted timeline
  (`<workspace>.oplog`, even after a crashed run) or a built-in drifting session and
  walk it by hand. A REPL cursor time-travels the cognitive timeline (`timeline`,
  `goto`/`next`/`prev`, `recall`/`around`/`task` at the cursor) and two drift views
  surface how the working set moved: `diff A B` (files that entered/left) and
  `energy A B` (node-level Δscore + failure-pressure — the migration of causal heat
  through the AST as failures propagate, visible even when the file set is stable).
  `missing <node> [budget]` is an **eviction watchpoint**: it finds the first step a
  node drops out of the budgeted window, with the triggering op, the token gap, and a
  status strip (`·●●●●●○○●●`); it reports cleanly against the compaction floor when
  the eviction lies in folded history. Every command reconstructs state
  deterministically via `recall_what_if`/replay, so it is exact and side-effect free.
- **Time-travel debugging demo** (`examples/time_travel.rs`, `cargo run --example
  time_travel`) — an agent session that drifts (a tight-budget recall evicts the
  cause two hops away), then is debugged by rewinding to the exact recall and
  replaying it under a larger budget; `replay_to` reconstructs the state exactly.
- **Robust efficiency number** — `phase4_eval.py` prints a context-efficiency
  report (works in `--dry-run`, no model). Across 51 single-file fixes from
  `fd`/`bat`/`hyperfine`, CCOS assembles 700–1600 context tokens vs RAG's
  budget-filling ~6000 — a **4.1–9.1× reduction** (it self-bounds at the causal
  region; the baseline fills the budget by construction).
- **Event-sourced agent session** (`agent_session` module) — `AgentSession`
  records every cognitive op (ingest / failure / recall / page-fault) as a
  timeline; `replay_to(step)` reconstructs the exact state, and
  `recall_what_if(step, q, b)` re-runs a recall under different parameters:
  **time-travel debugging** for an agent's context, the capability a probabilistic
  retrieval stack lacks.
- **Context page fault** (`AgentSession::page_fault`) — feed `cargo test` /
  compiler output back in: parse the faulting locations (`trace`), inject failure
  pressure, recall a refreshed window — the MMU "demand paging on a fault" step,
  logged and replayable. `scripts/phase4_eval.py` now uses it as a
  **compiler-in-the-loop** retry (patch → test → page-fault → enriched context →
  retry, `--max-attempts`).
- **`ccos trace`** + **module-hierarchy linking** — parse `cargo test` / panic /
  backtrace (stdin) into the crash's source files (`trace` module); and
  `link_module_imports` now adds parent→sub-module edges so sub-modules reached
  only via a re-export aren't orphaned. (Both from the crash-trace pivot PoC, whose
  verdict was that RAG-over-the-error-message still wins.)
- **Phase-4 prototype** (`scripts/phase4_eval.py`) — the *sufficient*-condition
  harness: for a real single-file fix it builds the agent's context two ways at an
  equal token budget (CCOS causal region vs lexical-RAG top files), asks a model
  to rewrite the buggy file, applies it, and runs `cargo test`, comparing CCOS vs
  RAG resolved-rate. Validated in `--dry-run` offline; the model (Ollama) + test
  grading run on a machine with a toolchain (the Jetson). Dry-run already shows a
  caveat: CCOS's region is often *just the target file* (sparse cross-file edges),
  so it gives a thinner context than RAG at equal budget — the verdict hinges on
  whether targeted-thin beats broad-lexical for the model.
- **Thesis check in the validation harness** — measures seed↔target lexical
  similarity per scenario and reports Δ(CCOS−RAG) for far vs near seeds. On the
  available data (fd, n=8) it is *unsupportive*: CCOS does worse, not better, when
  the seed is lexically far from its targets (corr +0.45, thesis predicts −).
- **Bidirectional failure propagation** — `MemoryGraph::propagate_failure_bidirectional`
  / `ccos failure --bidirectional` spread failure pressure to *upstream causes*
  (callers/importers) as well as downstream dependencies, and `ccos analyze` now
  links cross-file imports into the snapshot it writes. Measured on the
  causal-validation harness across three mature crates (`fd`, `bat`, `hyperfine`;
  70 mined fix commits), at a sufficient budget (`K≥50`) `R_cov` reaches
  **0.85–1.0** (recovering the large majority of the files each fix touched), up
  from `0.50–0.84` downstream-only, while diluting to `0.19–0.28` at a tight
  `K=20` — an honest, systematic trade-off (see
  `scripts/causal_validation/README.md`).
- **Lexical-RAG baseline in the harness** (TF-IDF cosine, same file budget) — and
  the honest result it gives: causal selection has **no net coverage advantage**
  over lexical similarity on these real repos (CCOS/RAG ties at `K≥50`; RAG is
  clearly better at `K=20`). On real bugs a fix's files are lexically similar to
  each other, so TF-IDF finds them too; the high `R_cov` is the *necessary*
  condition, not a CCOS win. Reported, not buried. Also: crate-aware import
  resolution (multi-crate workspaces + absolute paths).
- **Cross-file import linking** — `MemoryGraph::link_module_imports()` resolves
  intra-crate imports (`use:<file>:<path>` nodes) into `file→file` dependency
  edges by mapping each file to its module path and longest-prefix-matching the
  import. The kernel previously connected causally-related files only through
  shared `dep:` hubs, so failure propagation and region recall could not reach a
  fix's cross-file cause; now they do (opt-in, idempotent; called by the external
  memory façade on ingest). On a `db→repo→api` workspace, `recall(Around api.rs)`
  returns the cause `db.rs` and excludes unrelated files, and injected failure
  attenuates along the chain (0.85 → 0.78 → 0.65) above the 0.375 noise floor.
- **Agent-loop demo** (`scripts/agent_demo.py`) — a runnable, stdlib-only demo of
  CCOS as an agent's external memory: a bug whose cause is two lexically-distant
  files away is recalled by the causal region (not by a top-k/lexical retriever).
  Runs offline; uses a local Ollama model for the fix step if `OLLAMA_ENDPOINT` is
  set.
- **External memory interface** (`external_memory` module) — a single, documented
  façade (`ExternalMemory` trait + `CcosMemory`) an agent uses to treat CCOS as
  its external working memory, unifying the kernel's separate pieces (causal
  graph, incremental parser, hash-chained logs, causal queries, region engine)
  behind a handful of verbs: `ingest_source`, `signal_failure`, `recall`
  (`WorkingSet` / `Around` region-anchored / `Task` lexical), `verify`, `stats`,
  `checkpoint` (+ inherent `open`, `impact`/`causes`, `tick`). Deterministic
  recall, tamper-evident persistence that round-trips, all result types
  `Serialize`. Also exposed as **`ccos memory`** — a stdio JSON-Lines command
  (one request per line → one JSON response) so any language can use CCOS as
  memory via a subprocess, no server required. Reference guide in
  [`docs/MEMORY_INTERFACE.md`](docs/MEMORY_INTERFACE.md); 5 tests + a doctest.
- **`ccos eval --model M`** + live progress — override the active provider's model
  from the CLI (defaults to a local Ollama server if no provider env is set), and
  a live `[scenario] i/N tasks…` counter on stderr so long cloud-model runs no
  longer look hung.
- **Anthropic reasoning-model support in `ccos eval`** — read the `text` content
  block past a `<thinking>` block, larger `max_tokens`, no `temperature`; the
  grader also strips inline `<think>…</think>` blocks. Lets reasoning models
  (deepseek-v4-pro, qwen3.x, …) be graded on their final answer.
- **Causal-validation harness** (`scripts/causal_validation/`) — a closed-loop,
  LLM-free harness that tests CCOS's failure-propagation claim against the
  repository's **own Git history**. Phase 1 mines fix commits, reconstructs the
  pre-fix world in a throwaway worktree, and injects the fault at a changed file;
  Phase 2 scores `R_cov = |F_target ∩ WorkingSet_K| / |F_target|` per node budget
  `K` (arithmetic + geometric mean). Has a `--dry-run`; standard-library only.
  First run (on this thin history) honestly reports `R_cov ≈ 0.30`, flat across
  `K` — only the seed file is recovered — which localises a real limitation
  (failure pressure flows downstream only) and gives Phase 3 a concrete objective.
- **Tunable scoring weights** — the causal-score coefficients and the
  failure-propagation decay are now a `ScoringWeights` value on `MemoryGraph`
  (defaults reproduce the shipped constants exactly, regression-tested), settable
  via `set_scoring_weights` or the environment (`CCOS_W_BASE`, `CCOS_W_FAILURE`,
  `CCOS_W_RECENCY`, `CCOS_W_ACCESS`, `CCOS_FAILURE_DECAY`). `ccos analyze` and
  `ccos failure` honour them, so a hyperparameter search needs no recompile.
- **`ccos failure --max-nodes K --json`** — re-pages the graph to the bounded
  **WorkingSet_K** after fault injection and emits it (plus the affected set and
  the weights used) as JSON: the measurement hook the validation harness drives.
- **Anthropic Messages provider** for `ccos eval` — the real-LLM harness now also
  speaks `/v1/messages` (`ANTHROPIC_API_KEY` + optional `ANTHROPIC_BASE_URL` /
  `ANTHROPIC_MODEL`), so it can drive any Anthropic-compatible endpoint (e.g.
  DeepSeek at `https://api.deepseek.com/anthropic`, model `deepseek-v4-pro`).
- **Context Region Engine** (CCOS v0.3) — a spatial memory model above the causal
  graph. New modules `context_region`, `region_engine`, `context_policy`,
  `region_metrics`: nodes are embedded in a 3-D context space and clustered into
  **regions** (connected components of the cross-file causal-link graph) with a
  temperature and causal density; a region is hydrated as a `ContextWindow` under
  a **dynamic admission policy**. Five new event types
  (`RegionCreated/Activated/Merged/Evicted/ContextWindowGenerated`) keep it
  event-sourced and deterministically replayable (`replay_from`). New `ccos
  regions` CLI (cluster / activate / metrics), `scripts/region_benchmark.sh`,
  `docs/context_regions.md`, and an arXiv research paper in `docs/paper/`.
  Measured: region selection covers 97% of a task's causal neighbourhood vs 35%
  flat at ≈48% fewer tokens; regions 95.5% internally connected.
- **Hypothesis harness** (`experiment` module + `ccos experiment` CLI) — a
  deterministic, LLM-free simulation testing the *necessary condition* of the
  research thesis on modular synthetic repos with cross-file causal tasks of
  growing diameter, six strategies (RAG-dense/hybrid, GraphRAG-1hop/BFS,
  CCOS-from-query, CCOS-region), under an explicit success oracle, across two
  scenarios. **Clean query:** lexical RAG solves 0% while structure-aware methods
  (graph-BFS, CCOS) solve 100% — the lever is causal *structure*, not CCOS per se.
  **Noisy query** (a decoy out-scores the target lexically): every lexically-seeded
  method collapses to 0% — including graph-BFS and the `ccos-from-query` ablation —
  while only `ccos-region`, anchored on the workspace signal, survives at 100%. The
  ablation isolates the differentiator: the *anchor source*, not the region
  machinery. Folded into the paper (`docs/paper/` §8, two-scenario table).
- **Real-LLM evaluation harness** (`eval` module + `ccos eval` CLI) — tests the
  *sufficient* condition: auto-gradable multi-file "arithmetic causal chain" tasks
  whose answer requires the distant cause, six strategies assembling a budgeted
  window, sent to any OpenAI-compatible or Ollama endpoint. Reports task success,
  model-independent **oracle coverage**, and symbol-hallucination per diameter.
  Runs offline against a no-model stub (reproducing the coverage result on real
  file text) so the pipeline is CI-checked; real success numbers await a reachable
  model. Paper §9 updated (harness implemented; results pending a model).
- **Canonical tamper-evident `EventLog`** (ROADMAP P1.2): every appended event is
  linked into a SHA-256 hash chain over its replayable content (sequence + type +
  payload), so integrity now covers *all* runs, not just persisted snapshots.
  `EventLog::verify_integrity` detects payload tampering, reordering, insertion or
  deletion; `ccos verify` and `ccos replay` check it. The chain excludes the
  non-deterministic `id`/`timestamp`, so logs stay reproducible.
- **Optional `syn`-based AST parser** behind the `syn-parser` feature (ROADMAP
  P0.1): accurate parsing of nested-module bodies, multi-line signatures, grouped
  `use` and impl methods, with the zero-dependency line-based parser as the
  fallback (used when the feature is off or a file does not parse). CI lints
  (`--all-features`) and tests both paths.
- **Graph inspection commands** backed by a new read-only `query` module:
  - `ccos top <path> [--limit N] [--json]` — the hottest nodes by causal score
    (the working set the kernel would page in first).
  - `ccos blame <snapshot> <node-id> [--depth N] [--json]` — a node's upstream
    **causes** and downstream **blast radius**, walked deterministically in each
    edge direction.
  - `ccos export <snapshot> [--out FILE]` — export the causal graph as
    **GraphML** for Gephi / yEd / Cytoscape / networkx (deterministic, id-sorted).
- `query` module API: `impact_set`, `source_set`, `walk`, `hot_set`,
  `to_graphml`, plus `Reached` and `Direction` types (unit-tested).
- New docs: [`docs/USAGE.md`](docs/USAGE.md) (full command reference, end-to-end
  walkthrough, troubleshooting FAQ), [`CONTRIBUTING.md`](CONTRIBUTING.md), and
  this changelog.
- Annotated research **bibliography** ([`docs/BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md))
  — ~60 web-verified papers across 12 themes, each mapped to a CCOS module
  (context paging, causal graph, agents, guard/consensus/adversarial, hash-chained
  log & failure propagation).

### Changed

- The CI pipeline is **consolidated into a single cached job** (Format → Clippy
  `--all-features` → tests on both parser paths → Docs → CLI smoke) to keep
  GitHub Actions minute usage low on the private repo; `cargo audit` moved to a
  **weekly** `audit.yml` (and on-demand) instead of every push. Uses only
  GitHub-authored actions (`actions/checkout`, `actions/cache`).
- `README.md` and `docs/ARCHITECTURE.md` updated for the `query` module and the
  new commands.

### Fixed

- **Compressor CCR reversibility under eviction.** `store` evicted the
  lowest-hash entry as soon as the store passed `ccr_capacity`, so a single
  recall window with more compressed items than the capacity could evict refs it
  had *just handed back* — breaking the "nothing is lost, call `ccos_retrieve`"
  guarantee (latent: the default capacity is 4096, larger than any real window).
  Eviction is now deferred to *after* an item/window is produced
  (`enforce_ccr_capacity`) and never drops a live ref — the cap is a floor
  against older entries, lifted when the current window exceeds it. Regression
  test: `compress_window_keeps_every_ref_retrievable_below_capacity`.
- **Parser:** `strip_comments` now also removes inline `/* … */` block comments
  (string-aware), so symbols hidden in block comments are no longer extracted as
  real nodes. Multi-line block comments remain a known limitation of the
  line-based parser.

## [0.3.0] — Autonomous Context Runtime

### Added

- `scan`, `agents`, `benchmark` and `runtime` commands.
- New modules: `scheduler` (HOT/WARM/COLD context paging), `workspace` (async
  real-filesystem delta scanner), `agents` (Coder/Reviewer/Security behind an
  `Agent` trait), `persistence` (durable runtime state with integrity verify),
  and `benchmark` (cycle harness → JSON report).
- See [`CCOS_v0.3_REPORT.md`](CCOS_v0.3_REPORT.md) for the full report.

## [0.2.0] — Causal Kernel

### Added

- Causal memory graph with scoring, deterministic paging and failure
  propagation; incremental `O(Δ)` updates; append-only `EventLog` with
  deterministic replay and graph reconstruction; hash-chained
  `DistributedEventLog`; `GuardLayer`; multi-model `consensus`; `adversarial`
  fault injection; single-file `persist` snapshots.
- CLI: `demo`, `analyze`, `verify`, `replay`, `diff`, `failure`, `chaos`.

### Fixed

- Unbounded edge leak, guard prefix-bypass, non-deterministic eviction, and
  `max_nesting_depth` enforcement (see [`ROADMAP.md`](ROADMAP.md) → *Done*).

[Unreleased]: https://github.com/Memorithm/CCOS_EXTENDED/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Memorithm/CCOS_EXTENDED/compare/v0.3.0...v0.4.0
