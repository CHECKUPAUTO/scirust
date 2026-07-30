# Deploying CCOS

CCOS ships as a single Rust binary (`ccos`) that also hosts the MCP server. This is the recommended
path for running it on a server, behind an AI agent.

> **First command on any new host:** `ccos doctor` — it reports the build profile, compiled features,
> active parser, license status, and any deployment warnings.

## 1. Build

The `ccos` binary **requires the `llm` feature** (it drives the async MCP server + runtime), so a bare
`cargo build --release` produces **no binary at all**. Build with the deployment features:

```sh
# Community / core deployment:
cargo build --release --features llm,license

# CCOS_EXTENDED premium deployment — every deterministic Pro tier, replay-safe:
cargo build --release --features llm,pro-default

# Everything incl. the REPLAY-RELAX full kernels (test/CI; see docs/DETERMINISM.md):
cargo build --release --features llm,all-full
```

| feature | gives you | default |
|---|---|---|
| `syn-parser` | the accurate `syn` AST parser | **on** |
| `llm` | the `ccos` binary itself + MCP server + Ollama backend | required for the bin |
| `license` | the offline ed25519 Pro-license verifier (`tensions` / `audit` + a Pro tier) | recommended |
| `license-pq` | the offline **post-quantum** (SLH-DSA, FIPS 205) Pro-license verifier — composes with `license` (§4b) | recommended |
| `signed-sync` | per-workspace ed25519 identities for `ccos sync` (signed, TOFU-pinned bundles) | optional |
| `learned-embed` | the LSA semantic re-ranker | optional |
| `mimalloc` | a faster allocator (benchmarking only) | optional |
| `neural-embed` | quarantined **local** neural embedder (Ollama `/api/embeddings`; REPLAY-RELAX) | optional |
| `slhav2` | the distilled zero-dep SLHAv2 tile backend (`replay == live`) | optional |
| `slhav2-full` | the REAL `ccos-scirust` kernel — SIMD scoring, `ElasticKvCache`, `LatentSafetyGuard`, the `slha.*` MCP tools + `ccos slha` (Pro-gated; REPLAY-RELAX) | optional |
| `octasoma` | the OctaSoma semantic-memory backend + the `octa-semantic` recall strategy (Pro-gated) | optional |
| `octacore` | the causal-narrow → cosine-rerank cascade — the `octa.*` MCP tools + `ccos octa` (Pro-gated; implies `octasoma`) | optional |
| `rsi` | the CERVO/RSI self-improvement core — the `rsi.*` MCP status tools + `ccos rsi` (Pro-gated) | optional |
| `rsi-dgm` | the Linux OS-sandboxed Darwin–Gödel Machine loop (typed API only; Pro-gated; REPLAY-RELAX) | optional |
| `rsi-full` | RSI + a local LLM proposer backend (REPLAY-RELAX) | optional |
| `pro-default` | **bundle**: every deterministic premium tier (`license`, `license-pq`, `signed-sync`, `slhav2`, `octasoma`, `octacore`, `rsi`, `learned-embed`) — replays bit-identically | premium |
| `all-full` | **bundle**: `pro-default` + every REPLAY-RELAX full kernel | test/CI |

The **default build** (nothing beyond `syn-parser`) compiles none of the premium crates — its
dependency tree is byte-identical to core CCOS, and the CI's byte-identity guard enforces that.

## 2. Install

```sh
install -m 755 target/release/ccos /usr/local/bin/ccos
ccos doctor
```

### 2a. DGM and Forge host requirements

Generated-code evaluation is supported on Linux with `/usr/bin/bwrap`,
`/usr/bin/prlimit`, and the exact Rust 1.89.0 rustup toolchain. Install the host
packages on Debian/Ubuntu with:

```sh
sudo apt-get install --yes --no-install-recommends bubblewrap util-linux
```

Run the service as a dedicated non-root account. UID 0 is refused unless the
development-only `CCOS_UNSAFE_ALLOW_ROOT_SANDBOX=1` override is supplied to an
individual test command. The override never permits direct execution or GPU,
network, home, credential, or socket access. Non-Linux deployment can run the
deterministic CCOS core, but generated-code execution fails closed.

The Jetson GPU is deliberately unavailable to untrusted Forge/DGM candidates.
No production profile in this repository mounts NVIDIA device nodes.

`ccos doctor` (or `ccos doctor --json` for machines) prints, e.g.:

```
ccos doctor — deployment self-check

  version      0.4.0
  build        release
  target       x86_64-linux
  parser       syn AST (accurate)
  features     llm=yes license=yes license-pq=yes syn-parser=yes learned-embed=no mimalloc=no signed-sync=no
  premium      slhav2=no slhav2-full=no octasoma=no octacore=no rsi=no rsi-dgm=no
  mcp          ready  (ccos mcp <workspace>)

  license
    verifier   slh-dsa (SLH-DSA-128s) + ed25519 (both compiled in)
    key profile none (fail-closed)
    tier       community
    token      none

  ⚠ 1 warning(s):
    - no license verification keyring was injected at build time — Pro is fail-closed
```

### 2b. `ccos setup` — wire the agent host & certify the install

After `doctor`, one more command wires and certifies the deployment
(`scripts/install.sh` runs both):

```sh
cd /path/to/project && ccos setup --yes
```

`setup` probes the host, registers the MCP server in the project's `.mcp.json`
(an idempotent merge — consent-gated, fail-closed on anything unparseable),
runs a deterministic first-run self-test battery against the real kernel
(ingest → causal recall → failure propagation → hash-chain integrity →
checkpoint determinism → MCP handshake), and seals the verdict into
`setup_report.json`. An MCP agent relays that verdict to the user through the
`ccos://setup/report` resource — the report file, not the model, is the source
of truth. Exit 0 only when every check passed. Full guide: [`SETUP.md`](SETUP.md).

## 3. MCP server

Point your MCP gateway at the **installed release** binary and a workspace path:

```
/usr/local/bin/ccos mcp /var/lib/ccos/workspace.ccos
```

> ⚠️ Do **not** point the gateway at `target/debug/ccos`: a debug build is slower and may diverge from
> your installed release. `ccos doctor` flags a debug build. Keep the MCP command and your install on
> the same release binary.

The workspace is one `.ccos` snapshot + a `.oplog` timeline sidecar, shared with the CLI.

## 4. Pro license (optional)

The public tree contains no production verification key. With no configured
keyring the generated build metadata reports profile `none`, and every Pro
license fails closed. Release builders inject public verification keys through
an explicit, provenance-controlled file; no source edit or private key belongs
in the application repository:

```sh
# 1. Generate a keypair on the vendor's offline signing host. Keep the seed secret.
cargo run --features license --example license_sign -- keygen

# 2. On the controlled release builder, create an untracked public-key file.
install -m 600 /dev/null /secure/build/ccos-license-public-keys
cat > /secure/build/ccos-license-public-keys <<'KEYS'
profile production
ed25519 vendor-2026 <64-lowercase-hex-public-key>
KEYS

# 3. Build with the explicit public keyring. The build fails on malformed,
#    duplicate, empty, all-zero, or test-profile release configuration.
CCOS_LICENSE_PUBLIC_KEYS_FILE=/secure/build/ccos-license-public-keys \
  cargo build --release --features llm,license

# 4. Sign customer licenses only on the signing host.
CCOS_LICENSE_SIGNING_SEED=<64-hex-seed> \
  cargo run --features license --example license_sign -- sign --licensee "Acme Corp" --days 365

# 5. Install the token on the customer host.
export CCOS_LICENSE="<token>"                 # inline (containers / CI), or:
export CCOS_LICENSE_FILE=/etc/ccos/license    # an explicit path, or:
#     write it to ~/.config/ccos/license      # ($XDG_CONFIG_HOME/ccos/license)
ccos doctor                                   # tier should now read: PRO
```

Resolution order: `$CCOS_LICENSE` (token text inline) → `$CCOS_LICENSE_FILE` → the XDG default.

Verification is **fully offline** — no network, no telemetry — so a customer can run air-gapped.

**Selling annual, single-seat licenses at scale**: instead of mailing tokens by
hand, run the claim counter — the vendor hands the customer a one-time
`CCOS-XXXXX-…` code, and `ccos license claim <CODE> --from <url>` redeems it
for a machine-bound token (verified locally before install; the runtime never
contacts the counter again). See [`LICENSING_SERVER.md`](LICENSING_SERVER.md).

### 4b. Post-quantum licenses (SLH-DSA / FIPS 205, optional)

For deployments that want a license signature that is conjectured secure against a large-scale
quantum computer, CCOS ships a **second**, independent offline verifier based on **SLH-DSA**
(NIST FIPS 205, formerly SPHINCS+) behind the `license-pq` cargo feature. It is orthogonal to the
ed25519 `license` feature — a build may compile in one, the other, or both
(`--features llm,license,license-pq`). A token's `slhdsa.` scheme tag dispatches it to the SLH-DSA
verifier; an untagged token still goes to ed25519. The tag is also bound into the signed message,
so a signature made under one scheme can never be replayed as the other.

Parameter set **SLH-DSA-SHAKE-128s**: 32-byte public key, 64-byte secret key, **7,856-byte
signature** (~10.5 KB base64url) — the smallest FIPS 205 signature, NIST PQ category 1 (~128-bit
post-quantum), a like-for-like PQ upgrade of ed25519's classical 128-bit. Signing is deterministic;
verification is fast (it runs on every `ccos` invocation that reads the license).

```sh
# 1. Generate a post-quantum keypair on the offline signing host.
cargo run --features license-pq --example license_sign_pq -- keygen

# 2. Add only the public key to the controlled release keyring:
# slh-dsa-shake-128s vendor-pq-2026 <64-lowercase-hex-public-key>
# Then build with CCOS_LICENSE_PUBLIC_KEYS_FILE as above.

# 3. Sign a license. The token is ~10.5 KB; prefer a file over an env var.
CCOS_LICENSE_PQ_SIGNING_SEED=<128-hex-secret-key> \
  cargo run --features license-pq --example license_sign_pq -- sign --licensee "Acme Corp" --days 365 \
  > /tmp/acme.pqlicense
mkdir -p ~/.config/ccos && cp /tmp/acme.pqlicense ~/.config/ccos/license

# 4. Verify on the host (build with the feature that matches the token's scheme).
cargo run --features llm,license-pq -- doctor    # verifier: slh-dsa; tier: PRO
```

> ⚠️ **Unaudited cryptography.** The `lattice-slh-dsa` crate is pure Rust
> (`#![forbid(unsafe_code)]`, `zeroize`-backed) but **not independently audited**. It was chosen over
> RustCrypto's `slh-dsa` because the latter pins a pre-release `signature` crate that cannot coexist
> with `ed25519-dalek` in a single build (it would break `--all-features`). Treat the PQ verifier as
> defence-in-depth or an opt-in for post-quantum-readiness, not a drop-in replacement for an audited
> ed25519 stack, until an independent audit of `lattice-slh-dsa` exists.

### 4c. Pro features

The Pro license unlocks, all verified locally and gated through `Licensing::require` (the core
causal graph, Q-Page, and recall are **never** gated):

- **custom-authority-weights** — per-source authority weighting (vs. the uniform default).
- **tension-visualization** — cognitive-tension rendering in the logs.
- **audit-reports** — belief / conflict / provenance audit-report generation.
- **slhav2-embeddings** — the adaptive **grouped** INT4 quantization (group size 16) for the
  semantic embedding store. A community session falls back to **uniform** INT4 (a single per-vector
  scale); the core recall path is unchanged, only the embedding precision reflects the tier.
- **adaptive-retrieval** — the `ccos::retrieval` self-improving feedback loop (`ImprovementLoop`).
  The core retrieval (dense / BM25 / hybrid + metrics) is free and fully functional; only the
  continuous-improvement tier is gated.
- **octasoma-memory** — the OctaSoma-backed, region-sharded semantic-anchor index
  (`ccos::octa_index`, compiled behind the `octasoma` cargo feature). The free core recall
  strategies (working-set / around / task / INT4 TF-IDF semantic / hybrid) are untouched; only the
  true-embedding OctaSoma backend is Pro. The tier includes the **explicit relevance-feedback
  channel** (`SemanticFeedback`, and the `octa_feedback` MCP tool): labels from the agent loop
  certify a conformal anchor-score floor (miscoverage ≤ α), and `recall_semantic_calibrated` /
  the MCP `octa-semantic` strategy then trust an anchor only when it clears the floor — refusals
  are visible (`octa-semantic-below-floor-fallback-task`), never a silent downgrade, and with too
  few labels no floor is fabricated.
- **slhav2-full-kernel** *(CCOS_EXTENDED, `slhav2-full` cargo feature)* — the REAL `ccos-scirust`
  attention kernel as a `MemoryProvider` backend: runtime-dispatched SIMD scoring, `ElasticKvCache`
  HOT/WARM/COLD soft-paging with informed eviction, the `LatentSafetyGuard`, and the `slha.*` MCP
  tools / `ccos slha` CLI. A documented REPLAY-RELAX (see `docs/DETERMINISM.md`); the distilled
  `slhav2` backend stays the replay-exact store.
- **rsi-self-improvement** *(CCOS_EXTENDED, `rsi` cargo feature)* — running the CERVO/RSI agent
  with CCOS audit (`CcosAudit`: rsi's audit log over CCOS's hash-chained `EventLog`), plus the
  `rsi.*` MCP status tools / `ccos rsi` CLI. The std-only core keeps `replay == live`.
- **rsi-dgm** *(CCOS_EXTENDED, `rsi-dgm` cargo feature)* — the Linux OS-sandboxed Darwin–Gödel Machine
  loop (`GuardedDgm`: editable-file allowlist, GuardLayer sanitation, air-gapped
  `cargo --offline --frozen` evaluator, hash-chain-audited promotion). Deliberately reachable
  **only** through the typed API — no MCP tool and no CLI one-liner can trigger self-modification.
  A documented REPLAY-RELAX.

`ccos license` enumerates the active set; `ccos doctor` reports the compiled verifier scheme(s).

## 5. Durability (what survives a crash / power cut)

Every checkpoint is **crash- and power-safe**. `util::write_durable` writes a temp file, `fsync`s it,
**atomically renames** it into place, then `fsync`s the parent directory — the snapshot is never left
half-written, and the hash-chained event log detects any tampering on reload. Durability is at
**checkpoint granularity**: the agent / MCP flow checkpoints to the workspace, so both the causal
memory and the replayable timeline survive a restart or a sudden power loss.

## One-shot

`scripts/install.sh` does build → install → `ccos doctor` in one step
(`PREFIX=/usr/local/bin CCOS_FEATURES=llm,license sh scripts/install.sh`; add `,license-pq` to also
compile the post-quantum SLH-DSA verifier — see §4b).
