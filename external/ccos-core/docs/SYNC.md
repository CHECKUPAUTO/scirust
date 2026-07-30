# The distributed multi-agent store (`ccos sync`)

> Run the measured demo: `cargo run --release --example sync_crux`

Lands the paper's future-work item 5 — *"an optional distributed store for multi-agent settings"* —
without giving up any moat property. The paper's §7 called single-node working memory *"a design
choice for an auditable, air-gappable kernel, not a horizontally-scaled vector database"* — this
store keeps that choice and federates anyway:

| moat property | how federation keeps it |
|---|---|
| zero dependencies | no network stack, no consensus library — a bundle is a **plain JSON file** |
| air-gappable | any transport works, including none (sneakernet a file between machines) |
| tamper-evident | bundles carry the op-log's SHA-256 chain (PR #134); **every link is re-verified on import** |
| `replay == live` | imports never touch the own timeline; the merged view is a *pure function* — replay semantics unchanged |
| deterministic | two agents holding the same logs materialize **bit-identical** views (tested, and measured in `sync_crux`) |

## The model

Every agent keeps exactly **one** append-only, hash-chained timeline of its *own* ops (the
`.oplog` sidecar). Sharing is the exchange of chain-verified segments:

```text
A: ccos sync export wsA.ccos --agent alice --out a.json     # segment + chain links
   (any transport: scp, USB stick, email…)
B: ccos sync import wsB.ccos a.json                          # re-verify every link, store per-agent
B: ccos sync status wsB.ccos                                 # own + foreign logs, merged-view stats
B: ccos sync materialize wsB.ccos --out shared.ccos          # the shared brain, as a normal store
```

Imported logs are stored **per agent** and never mixed into the own timeline. The *shared brain*
is `AgentSession::merged_view()`: replay every known timeline from empty, in canonical
(sorted-agent-id) order, with the exact `replay_to` semantics. Because it is a pure function of
the known logs, agents that hold the same logs **converge bit-for-bit** — verified with
`CcosMemory::state_fingerprint()`, the canonical SHA-256 of the replayable state (graph + sources
+ both chain heads; the only fields excluded are the audit UUIDs that are non-deterministic *by
design*). This is a state-based CRDT of grow-only per-agent logs: commutative, associative,
idempotent — no consensus protocol needed.

## What import refuses (all tested)

- **Tampered bundle** — any link that does not recompute from the anchor (`SyncError::Tampered`).
- **Equivocation** — one agent publishing two different histories under the same identity: the
  overlap between the bundle and the locally-known chain must agree link-for-link
  (`SyncError::Diverged`). This is the distributed payoff of PR #134's chain.
- **Gaps** — a segment starting past the known end (`SyncError::Gap`): import the earlier bundle
  first (incremental exports via `--since N` are supported and idempotent on overlap).
- **Self-import** and **identity-less** sessions/bundles.
- **Bad signature** — a signed bundle whose ed25519 signature does not verify over its canonical
  payload (`SyncError::BadSignature`) — including a half-present key/signature pair on *every*
  build, crypto or not.
- **Key mismatch** — a bundle signed with a different key than the one TOFU-pinned for that agent
  id (`SyncError::KeyMismatch`): identity spoof or unannounced key swap, both refused.
- **Stripped signature** — an *unsigned* bundle from an agent whose key is pinned
  (`SyncError::UnsignedFromPinned`): removing the signature does not bypass pinning.
- **Signed bundle on a crypto-free build** (`SyncError::SignedUnsupported`) — refused with a
  rebuild hint, never silently accepted unverified.

## Contract notes

- **Compaction and federation.** A compacted prefix is folded into the local baseline and is no
  longer separable into verifiable ops, so `export` refuses ranges below the compaction floor.
  Federated agents should run with compaction off (`CCOS_OPLOG_MAX=0`) or export before compacting.
- **The local baseline stays local.** A seeded (`ccos memory`-initialized) baseline is not part of
  the exchanged history — the *log* is the shared truth. The merged view reconstructs the union of
  recorded timelines.
- **Identity is declarative by default, cryptographic on request.** In the default build agent
  ids are plain strings and the chain enforces *consistency* (no divergent histories under one id
  against the same peer). The `signed-sync` feature upgrades that to *authenticity* — see below.

## Signed bundles (`--features signed-sync`)

The hash chain proves a segment's **integrity**; only a signature proves **who** recorded it. The
`signed-sync` feature adds exactly that, without touching the default posture (it reuses the same
optional `ed25519-dalek` as `license` plus the in-tree `getrandom` — **no new crate**; the default
build stays crypto-free and keeps working with unsigned federations):

```bash
cargo build --features llm,signed-sync
ccos sync keygen  work/a.ccos            # one-time identity: seed → <workspace>.ccos.key
ccos sync export  work/a.ccos --agent agent-a --out a.json    # now signed automatically
ccos sync import  work/b.ccos a.json     # verifies, then TOFU-pins agent-a → that key
ccos sync status  work/b.ccos            # shows "key pinned …" per foreign agent
```

- **Sign**: the exporter signs the bundle's canonical JSON (signature fields blanked) with the
  workspace seed; `pubkey`/`sig` ride in the bundle (additive serde — old readers ignore them).
- **Verify + TOFU pin**: the first *signed* bundle from an agent pins that key (persisted in the
  sidecar, additively). From then on a different key, or an unsigned bundle, refuses. Pinning
  happens only after the whole import succeeds — a refused bundle changes no state, keys included.
- **Key rotation is an explicit human act**: `keygen` refuses to overwrite an existing key, and
  peers that pinned the old key will refuse the new one until they re-pin (delete + re-import
  deliberately).
- **The seed never travels**: it lives in `<workspace>.ccos.key`, next to — never inside — the
  timeline sidecar, so sharing an `.oplog` or a bundle cannot leak it.
- **Threat line**: unsigned federation = tamper-evidence + equivocation-evidence (the chain);
  signed federation adds spoof-refusal and key-swap-refusal. What it deliberately does *not* add:
  PKI, revocation, or key servers — trust bootstraps on first contact (TOFU), like SSH.

## The crux measurement (`examples/sync_crux.rs`)

Two agents with disjoint knowledge (A owns `db.rs`; B owns `api.rs`, which calls into `db.rs`).
Before sync, **neither** graph holds the `api → db` call edge. After a two-bundle exchange, both
merged views hold it, and their fingerprints are equal — printed live, bit-identical across runs.
A bundle mutated in transit (the `timeout` body changed `30 → 0`) is refused with the exact broken
link. Deterministic end to end.
