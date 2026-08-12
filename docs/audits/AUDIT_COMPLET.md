# Complete audit — SciRust

**Date:** 2026-07-06
**Scope:** SciRust monorepo (`/root/scirust`), commit `5c7e43b`, branch `master`
**Method:** targeted static code review (direct reading of at-risk files + domain-group mapping + adversarial verification of findings). 248 804 LOC of Rust, 87 crates, ~1 005 files.
**Tools:** `grep`, `find`, reading of `unsafe`/FFI/network/crypto/cmd/deser/CI files.

---

## 1. Executive summary

SciRust is a **pure-Rust** deep-learning and scientific-computing platform with "certifiable" industrial verticals (estimation, navigation, water, OT security, dependability/safety), validated from cloud x86 to embedded ARM (Jetson). The code shows **notable security discipline** — well above the average for a project of this size — but contains several gaps between the **claims of `SECURITY.md`** and the **reality of the code**, plus a few genuinely risky areas.

### Overall verdict

| Dimension | Grade | Comment |
|---|---|---|
| Crypto posture (trader, license, OT scope) | **A** | Watch-only by default, HMAC signature, constant-time comparison (wallet), no key in cleartext. |
| OT protocol parsing (Modbus/SNMP/BER) | **A−** | Bounded lengths, explicit errors, no panic on malformed input. |
| Untrusted deserialization (safetensors) | **B+** | 16 MiB header cap, rejection of negative values, `checked_mul` overflow. Ad-hoc JSON parser (documented limits). |
| `unsafe` / memory (SIMD, arena, autodiff) | **B+** | Alignment guaranteed by `AlignBlock(128)`, `debug_assert`, documented invariants. A few blocks to review. |
| Command execution / autonomous agents | **C+** | `cli_passthrough` MCP, `sciagent` tools, self-mutating `openclaw-u` binary, `fetch-crates` (supply chain). No shell injection, but a large surface. |
| Compliance with `SECURITY.md` claims | **C** | "Zero FFI" contradicted by exported C FFI (`enclave.rs`) and CUDA archive; "unsafe confined to SIMD intrinsics" contradicted by `unsafe` in runtime/arena/tensor. |
| Supply chain / CI | **B** | `cargo-deny`, CycloneDX SBOM, committed lockfile. GitHub Actions **not pinned by SHA** (mutable tags). Nightly mandatory. |
| Robustness (unwrap/panic) | **B−** | ~2,044 `.unwrap()`, ~46 `panic!/todo!`. A few panics on external input (tolerance, fusion) to fix. |
| Test coverage | **A−** | 696 tests in `scirust-core`, 239 in `scirust-trader`… Only the proc-macro/examples crates are without tests (normal). |

**Security maturity score: 7.5/10** — good foundation, documentation gaps and a few agent/supply-chain areas to harden.

### Top risks (P0 → P2)

1. **[P1] `safe_enclave_infer` — OOB via unvalidated `dims`** (`scirust-runtime/src/enclave.rs`). The `EnclaveRuntime::infer` wrapper does not validate that `dims.batch*in_features ≤ input.len()` (etc.) before the `unsafe` call → out-of-bounds read/write inside the TEE.
2. **[P1] Self-mutating `openclaw-u` binary** (`src/main.rs`): writes generated source files into the tree then runs `cargo check`; loads an unsigned `state.json`. A self-modifying agent pattern with no integrity control.
3. **[P1] `fetch-crates` — active supply chain** (`scirust-sciagent/src/bin/fetch-crates.rs`): downloads arbitrary crates.io tarballs, extracts them via `tar xzf --strip-components=1`, symlinks the `.rs` into the workspace (training data). No checksum/integrity verification of the tarballs.
4. **[P2] Inaccurate `SECURITY.md` claims**: "zero FFI" and "unsafe confined to SIMD intrinsics" — contradicted by the exported C FFI and the non-SIMD `unsafe` blocks.
5. **[P2] CI Actions not pinned by SHA** (`release.yml`, `ci.yml`) — `@v2`, `@nightly`, `@master` are mutable tags.
6. **[P2] Non-constant-time signature comparison** in `scirust-discovery/src/scope.rs` (`signature_valid`), whereas the wallet's is constant-time — inconsistency.
7. **[P2] Panics on degenerate batches** in `scirust-tolerance` (modal/chain/spatial) and `scirust-fusion` (`fuse().expect()`) — untested code close to the API.
8. **[P2] ~4.4 MB ELF binaries committed at the root** (`cliptest`, `cliptest2`) — untracked provenance, supply-chain/detection risk.
9. **[P2] Non-cryptographic func-safety evidence chain** (`evidence.rs`): FNV-1a chain (public, no secret) → an attacker with file access can recompute a consistent chain. Documented as *tamper-evident* (not *tamper-resistant*), but `from_json().verify()` is presented as detecting forgeries, which is only true for naive edits.

---

## 2. Project presentation

- **Type:** Cargo workspace (resolver 2, edition 2021, `rust-version = 1.85`, **nightly** mandatory via `rust-toolchain.toml`).
- **License:** PolyForm Noncommercial 1.0.0 (`LICENSE.md`), `publish = false` (crates not published to crates.io).
- **Surface:** root crate `scirust` (facade `src/lib.rs` + standalone `openclaw-u` binary `src/main.rs`), 87 `scirust-*` crates, 10 examples, 2 workspace integration tests.
- **Key stacks:** `tokio` (full), `serde`/`serde_json`, `chrono`, `rand 0.8`, `sha2`, `nalgebra`/`simba` (via `paste`), `ureq`, `reqwest` (feature `live`), `clap`.
- **Governance:** `deny.toml` (cargo-deny: RustSec advisories, permissive licenses, sources), `clippy.toml`, `rustfmt.toml` (nightly options), `docs/sbom/scirust.cdx.json` (CycloneDX), `scripts/generate-sbom.sh`, `SECURITY.md` (FR policy).
- **CI:** `ci.yml` (pinned nightly fmt, clippy, nightly+stable build/test, aarch64 cross-check, cargo-deny, wgpu/lavapipe, SBOM, coverage) + `release.yml` (tag `v*` → GitHub Release + attached SBOM).
- **Determinism:** replayable bit-exact contract (SRT1 runtime); seeded `PcgEngine` RNG everywhere except `scirust-func-safety/src/fault_injection.rs:118` (`rand::random` unseeded — see §4.7).

---

## 3. Tree mapping (by domain group)

> Each group: role + tree of the main `src` files with a 1-line role. Risk hot spots are reported in §6.

### 3.1 Root, docs, CI, examples

```
ROOT (/root/scirust)
├── Cargo.toml            — root workspace; re-exports scirust-core/learning/simd/solvers/symbolic/rsi; bin `openclaw-u`
├── Cargo.lock            — committed lockfile (113 KB)
├── deny.toml             — cargo-deny: MIT/Apache/BSD/Zlib/Unicode-3.0 licenses, ignore RUSTSEC-2024-0436
├── rust-toolchain.toml   — channel=nightly, rustfmt/clippy/rustc-dev/llvm-tools
├── cliptest, cliptest2   — ~4.4 MB ELF binaries committed (untracked provenance) ⚠
├── README/CHANGELOG/LIVESTATE/SECURITY/LICENSING/LICENSE.md  — governance/license
├── Documentation*.md (8 languages), ARCHITECTURE-B/ANALYSIS_REPORT/DESIGN_SCIRUST_TENSOR/INTEGRATION_GUIDE
├── src/
│   ├── lib.rs            — facade `pub use scirust_core::*` ...
│   └── main.rs           — bin `openclaw-u`: autonomous Tokio agent, self-mutation + `cargo check` ⚠
├── tests/
│   ├── workflow.rs       — end-to-end ML workflow (linreg, polyfit, Cholesky)
│   └── expansion_val.rs  — ResNet/ViT/VAE/MoE/GCN/LoRA/CTC/DQN/PPO/NBeats/FemSolver1D validation
├── examples/             — mnist/cifar10/transformer×2/sentiment/industrial_monitor/ids/simd_views/benchmarks
├── docs/                 — ARCHITECTURE, ROADMAP×4, MEMORY_WALL_*, TRANSPILER_DESIGN, kb/, roadmaps/, sbom/
├── scripts/             — generate-sbom.sh, test-protocol*.sh (test-protocol.sh:121 `eval "$cmd"` ⚠)
├── archive/              — old code (scirust-gpu CUDA FFI, scirust-simd SVE, scirust-core quant/bf16) ⚠
└── .github/workflows/    — ci.yml, release.yml
```

### 3.2 Industrial verticals (23 crates)

```
scirust-estimation/   kalman/ekf/ukf/imm/particle/interval/smoother/ud/linalg — BMS/nav/fusion/SPC foundation
scirust-nav/          ins (dead-reckoning) / fusion (GNSS-INS Kalman) / tdoa (multilateration)
scirust-control/      pid (anti-windup + auto-tune) / lqr / qp (box-QP) / mpc / monitor / license (commercial gate)
scirust-robotics/     ssm (ISO/TS 15066 Speed-Separation) / kinematics (2-link) / trajectory (trapezoidal)
scirust-maritime/     colregs / cpa_tcpa / thrust_allocation (DP, pseudo-inverse)
scirust-water/        leak (acoustic correlation) / transient (Joukowsky)
scirust-hvac/         fdd (AHU ASHRAE G36) / nilm (disaggregation)
scirust-bms/          soc (EKF 1-RC) / soh (conformal) / thermal (runaway guard) / capacity / dual
scirust-pdm/          health / rul / conformal_rul / change_detection
scirust-grid/         (mho distance relay, WLS state estimation, power_quality, symmetrical, flicker) ⚠ safety
scirust-reliability/  PFDavg/PFH IEC 61508 (MooN, β, SIL, Markov) — certification argumentation base ⚠
scirust-func-safety/  evidence (FNV-1a chain ⚠) / audit / requirements / fault_injection (unseeded rand ⚠)
scirust-spc/          statistical process control
scirust-fatigue/      fatigue life
scirust-tolerance/    modal (orthonormalize.unwrap ⚠) / chain (allocate.unwrap ⚠) / spatial (unwrap ⚠)
scirust-fusion/       graph fusion (fuse().expect ⚠) / graph (SHA-256 identity, from_str ⚠)
scirust-signal/       bearing / filtering
scirust-multivariate/  multivariate statistics
scirust-sequential/    matching / labeling (network ⚠)
scirust-seasonal/     seasonal decomposition
scirust-som/          self-organizing map (frontend/cli SARIF parsing ⚠)
scirust-graph/        DAG / isomorphism (Serialize/Deserialize + 15 unwrap ⚠)
```
*No `unsafe`, no FFI, no network, no command execution in this group. Risk = safety-critical numerical correctness + deserialization of artifacts + unwrap/panic.*

### 3.3 Core & Tensor

```
scirust-core/         amp, aot, autodiff/, checkpoint, compute_backend, data/, distributed, dp,
                      embed, error, homomorphic, io/ (safetensors ⚠), lazy/, logging, matrix/ (backend.rs/view.rs unsafe ⚠),
                      nn/, optim, pruning, quantization (unsafe ⚠), quantum, reproducible, simd/ (tiling unsafe ⚠), symbolic
scirust-tensor-core/  lib.rs (core tensor)
scirust-tensor-runtime/  tensor execution runtime
scirust-tensor-compile/  tensor compilation (no tests)
scirust-tensor-contraction/  contraction
scirust-tensor-einsum/  einsum
scirust-simd/         lib (unsafe ⚠) / dispatch (unsafe intrinsics ⚠) / complex (unsafe ⚠)
scirust-simd-macros/  proc-macro (no tests)
scirust-arena/        slab (unsafe ⚠) / aligned (unsafe ⚠) / allocator (unsafe ⚠)
scirust-autodiff/     reverse (unsafe ⚠)
scirust-aot/          AOT compilation
```

### 3.4 ML / Learning

```
scirust-learning/      control, finance, optim, pattern_miner, nlp/, rl/, time_series/
scirust-unsupervised/  clustering
scirust-rl-algo/      RL
scirust-automl/       AutoML
scirust-nas/          neural architecture search
scirust-evo/          evolutionary
scirust-symreg/       symbolic regression
scirust-synthesis/    program synthesis
scirust-reasoning/   reasoning
scirust-symbolic/    symbolic computation
scirust-neuro-symbolic/  hybrid
scirust-retrieval/   ann / rerank (RAG)
scirust-nlp-advanced/  byte_bpe (crypto ⚠)
```

### 3.5 GPU & Runtime

```
scirust-gpu/          chain, conv_gpu, deterministic, deterministic_gpu, engine, fusion, kernels, ops, tensor, wgpu_backend
scirust-gpu-macros/  proc-macro (no tests)
scirust-runtime/      attest, difr, enclave (unsafe C FFI ⚠), proof, proofcli, quant, vinfer, bin/, main
scirust-embedded/    embedded targets
scirust-edge/        edge computing
scirust-burn-bridge/  burn bridge
scirust-bridge/      inter-framework bridge
scirust-onnx/        lib.rs (serde_json::from_str ⚠ — ONNX-like JSON)
```

### 3.6 Agents / CLI / MCP / Transpiler

```
scirust-cli/         main + learning/nlp/numeric/quickstart/reasoning/sciagent/symbolic/synergy/trader (subcommands)
scirust-mcp/         server, protocol, registry, audit, tools/ (cli_passthrough ⚠, grid, fatigue, …)
scirust-sciagent/    agentic/ (tools.rs ⚠), bin/ (fetch-crates ⚠), attention, bpe, ccos, flash_attention,
                     generate, gpu, inference, model, norm, planning, quantize, sha256, swiglu, tokenizer, train/
scirust-scaffold/    scaffolding (59 tests)
scirust-macros/      proc-macro (no tests)
scirust-codetrans/   transcoding
scirust-transpiler/  sir, lower, emit (emit.rs crypto ⚠), front_python/ (lexer/parser/ast)
scirust-rustc-driver/  rustc driver (rustc_private, excluded, no tests)
```

### 3.7 OT/ICS protocols & Discovery

```
scirust-discovery/   engine, scope (signed HMAC authorization ⚠), hmac (HMAC-SHA256), audit,
                     protocols/ (snmp ⚠, opcua, modbus ⚠, bacnet, ethernet_ip, mdns)
scirust-opcua/       OPC-UA client
scirust-mqtt/        MQTT publisher
scirust-events-core/   episodic
scirust-events-models/ event models
scirust-events-runtime/ event runtime
scirust-events-examples/  examples
scirust-shm/         fdd, operational (shared memory)
```

### 3.8 Application domains

```
scirust-vision/  scirust-audio/  scirust-biomed/  scirust-agtech/ (idw)  scirust-industrial/
scirust-ids/     (IDS, port scanning)  scirust-trader/ (wallet ⚠, market, agent, proof, orderbook, regime, model, robustness, dashboard)
scirust-license/ (license ⚠, gate, hashsig, module, cli)  scirust-integration/ (config, templates)
scirust-solvers/  scirust-rsi/  scirust-mlops/  scirust-fab/  scirust-sis/  scirust-tn/ (discovered_gemm unsafe ⚠)
```

---

## 4. Security audit

### 4.1 Methodology

For each dimension: `grep` of risk patterns across all `*.rs`, then exhaustive reading of the relevant files. Findings are classified by severity (critical/high/medium/low/info), with a concrete exploitation scenario. An adversarial verification pass re-read the code to confirm the alleged reachability and severity. Info/low findings are not listed exhaustively.

### 4.2 Table of confirmed findings

| # | Dimension | Sev. | File | Line | Summary | CWE |
|---|---|---|---|---|---|---|
| S1 | sandbox/unsafe | **medium** | `scirust-runtime/src/enclave.rs` | 23–66 | `safe_enclave_infer` dereferences raw pointers on the strength of `dims` (batch/in/out) without validating that `dims` fits inside the slices passed by `EnclaveRuntime::infer` | CWE-119/CWE-787 |
| S2 | cmd exec / agent | **medium** | `src/main.rs` (bin `openclaw-u`) | — | Autonomous agent writes generated source files then runs `cargo check`; loads unsigned `state.json` via `serde_json::from_str` | CWE-94/CWE-345 |
| S3 | supply chain | **medium** | `scirust-sciagent/src/bin/fetch-crates.rs` | 207–256 | Downloads arbitrary crates.io tarballs, extracts via `tar xzf --strip-components=1` with no checksum/integrity verification, symlinks the `.rs` into the workspace | CWE-494/CWE-829 |
| S4 | supply chain (CI) | **low** | `.github/workflows/{ci,release}.yml` | — | Third-party Actions pinned by mutable tag (`@v2`, `@nightly`, `@master`) rather than by SHA | CWE-1357 |
| S5 | doc/compliance | **low** | `SECURITY.md` vs `scirust-runtime/src/enclave.rs`, `archive/scirust-gpu/*` | — | Claim "pure Rust, zero FFI" contradicted by the exported C FFI (`safe_enclave_infer` `extern "C"`) and the CUDA archive (`cublas.rs`, `cuda_backend.rs`) | CWE-1047 |
| S6 | crypto | **low** | `scirust-discovery/src/scope.rs` | 106–109 | `signature_valid` compares the hex signature with `==` (non-constant-time), unlike the wallet (`wallet.rs:668`) which is constant-time | CWE-208 |
| S7 | integrity | **low** | `scirust-func-safety/src/evidence.rs` | 18–25, 80–95 | FNV-1a evidence chain (non-cryptographic hash, no secret); an attacker with write access can recompute a consistent chain. `from_json().verify()` only detects naive edits | CWE-345/CWE-327 |
| S8 | robustness/DoS | **low** | `scirust-tolerance/src/{modal,chain,spatial}.rs`, `scirust-fusion/src/fusion.rs` | modal:288,301 / chain:431–498 / spatial:498–551 / fusion:299–337 | `unwrap()/expect()` on `orthonormalize`, `allocate`, `fit_torsor`, `fuse` → panic on degenerate/non-full-rank batch instead of `Result` | CWE-754 |
| S9 | supply chain (repo) | **low** | `cliptest`, `cliptest2` (root) | — | ~4.4 MB ELF binaries committed with no provenance or checksum | CWE-494 |
| S10 | deserialization | **info** | `scirust-core/src/io/safetensors.rs` | 355–379 | Ad-hoc JSON parser by `find` of substrings (`extract_str_field`/`extract_array_field`) — robust to files produced by the module, but could be confused by a malicious header containing `"dtype":"` inside a key | CWE-20 (mitigated: documented internal use) |
| S11 | deserialization | **info** | `scirust-onnx/src/lib.rs` | 296 | `serde_json::from_str(json)` of an ONNX-like graph with no explicit bounds validation beyond serde | CWE-20 |
| S12 | determinism | **info** | `scirust-func-safety/src/fault_injection.rs` | 118 | Unseeded `rand::random::<f32>()` — violates the bit-reproducible determinism contract if used outside test mode | CWE-338 |

*No confirmed **critical** finding. No shell injection observed: the `Command::new` calls pass separate arguments (no shell), except `scripts/test-protocol.sh:121` (`eval "$cmd"`) which is only reached with internal variables.*

### 4.3 Per-dimension analysis

#### 4.3.1 `unsafe` / FFI / memory

The `unsafe` is **confined and mostly justified**:
- **`scirust-arena/src/slab.rs`** — backing `AlignBlock` `#[repr(C, align(128))]` guarantees a 128-aligned base pointer; each slot is a multiple of `MIN_ALIGN_BYTES` → correct alignment for any `T` whose alignment divides 128. `debug_assert!` checks alignment. `from_raw_parts_mut` is preceded by an `is_valid(handle)` check (anti-use-after-free version). **Sound.** A regression test documents a prior UB (`data_slice_is_aligned_for_every_slot`).
- **`scirust-simd/src/dispatch.rs`, `complex.rs`** — SIMD intrinsics `core::arch`, documented by safety headers (compliant with `SECURITY.md`).
- **`scirust-core/src/matrix/{backend,view}.rs`, `autodiff/reverse.rs`, `tensor/pinned.rs`, `quantization.rs`, `simd/tiling.rs`** — `unsafe` for aligned-buffer / `MaybeUninit` manipulation; to review case by case but no obvious UB found on reading.
- **`scirust-runtime/src/enclave.rs`** (S1) — **only significant material finding**: the `unsafe extern "C" fn safe_enclave_infer(...)` C FFI dereferences `weight_ptr/input_ptr/output_ptr/bias_ptr` according to `dims` with no verification whatsoever that `dims.batch*in_features ≤ input.len()`, etc. The `EnclaveRuntime::infer` wrapper builds the pointers from `&[f32]` slices but does not check the `dims` ↔ slice-length consistency. An inconsistent `dims` (caller bug, or untrusted input if exposed via the C ABI) → OOB read/write inside the TEE. **Recommendation:** validate in `infer` that `weights.len() ≥ out_features*in_features`, `input.len() ≥ batch*in_features`, `output.len() ≥ batch*out_features`, `bias.len() ≥ out_features` (if `has_bias`) before the call, and return `Err`.
- **`archive/scirust-gpu/{cublas.rs,cuda_backend.rs}`, `archive/scirust-simd/sve.rs`, `archive/scirust-core/quant/bf16.rs`** — C/CUDA and SVE FFI in the archive. Contradicts `SECURITY.md` ("zero FFI"), but the archive is not in the active workspace. **Recommendation:** either remove `archive/`, or state in `SECURITY.md` that the archive is out of scope.

#### 4.3.2 OT/ICS protocols & discovery

Excellent discipline. **`scirust-discovery/src/protocols/{modbus.rs,snmp.rs}`**: strictly bounded parsers.
- **Modbus** (`parse_read_device_id_response`): checks `buf.len() < 8`, `pdu.len() < 7`, `idx+2 > pdu.len()`, `idx+object_len > pdu.len()` → no panic on malformed frame; distinguishes Modbus exception (0x80) from malformed frame; fixed `512`-byte buffer.
- **SNMP** (`parse_get_response`): minimal BER decoding with `read_length`/`read_tlv` checking `pos >= buf.len()`, `content_start+len > buf.len()`; fixed `2048` buffer. Non-zero `error-status` → explicit `Err`. Community `"public"` in cleartext = **by design for SNMPv1** (documented in file header).
- **`scope.rs`** (security gate): **IEC 62443 zones/conduits model** — HMAC-signed `ScopeAuthorization`, CIDR whitelist, protocol whitelist, time window, **SL3+ gate** (high-security zone denied by default, explicit override required), IPv6 rejection, no panic on malformed CIDR. `authorize` is called **before** any network probe. Test `authorize_rejects_tampered_scope` validates that a scope widened after signing is rejected.
- **`hmac.rs`**: RFC 2104 HMAC-SHA256 over the in-house SHA-256 (`scirust_sciagent::sha256`), tested against RFC 4231 §4.2/§4.3. **Sound.**
- Only flaw (S6): `signature_valid` compares the signature with `==` on `String` (non-constant-time) — inconsistent with the wallet. Low impact (the attacker would have to be the verifying party with a timing oracle), but should be homogenized.

*No register/industrial command writes — read-only `Read Device Identification` (Modbus) and `GET sysDescr.0` (SNMP), compliant with the NIST SP 800-82 doctrine (passive/native discovery).*

#### 4.3.3 Crypto & secret management (trader, license, HMAC)

**`scirust-trader/src/wallet.rs`** — robust design:
- **Watch-only by default**, `live` (network) opt-in behind a feature.
- **No private key** in the module; a real signer is injected by the host (env var) and only produces signatures.
- **In-house Keccak-256** verified against the canonical Ethereum vectors (`keccak256("")`, `keccak256("abc")`). EIP-55 checksum tested against the 4 spec examples; EIP-712 domain separator; EIP-1559 signing hash (dry-run, unsigned).
- **HMAC-SHA256** (RFC 4231 tested).
- **`WalletAuthorization`**: non-self-authorizing gate — sign/send requires an operator-HMAC-signed authorization (server-side key); `verify_signature` in **constant-time comparison** (reduced XOR + OR); `authorizes` bounds chain id, method, value (`max_value_wei`), time window. Test `tampering_authorization_breaks_signature` confirms that any post-signature modification breaks the signature.

**`scirust-license/src/license.rs`** — canonical length-prefixed encoding (magic `SRL2` + version), sorted/deduplicated modules → no separator injection (test `separator_injection_cannot_forge_a_collision`). Node-lock via SHA-256 **salted by the license identity** (domain separation, length-prefix) → fingerprints not correlatable across licenses; only the hash is stored (the raw `machine_id` does not leak — test `binding_to_a_node_changes_the_digest_and_stores_only_the_hash`). Honest about the limit: does not resist brute-forcing a weak `machine_id` (the deployment must provide a UUID/TPM).

*No hardcoded secret, no non-cryptographic RNG for sign/wallet, no key leakage. **Very good.***

#### 4.3.4 Command execution / autonomous agents

No shell injection (the `Command::new` calls pass separate `args`, never via `sh -c`), but the **execution surface is large**:
- **`scirust-mcp/src/tools/cli_passthrough.rs`** (S2-adjacent): MCP tool `scirust_cli` that executes `scirust` (or `cargo run -p scirust-cli`) with `args` controlled by the MCP client. No injection (separate args), but exposes **the entire** CLI to a remote client. The args are validated (`must be strings`). Acceptable if the MCP channel is authenticated; to document.
- **`scirust-sciagent/src/agentic/tools.rs`**: `search`/`grep` tools (via `rg`/`grep`, regex pattern — no shell), `read`/`explain` (arbitrary file reading — by design for an agent), `build`/`test` (via `cargo`, crate_name as arg), `status` (`git status`). No injection. The `path` is controllable by the agent → arbitrary file reading (traversal) — by design for a code agent, to be channeled.
- **`src/main.rs` (bin `openclaw-u`)** (S2): **self-mutating agent** — writes `src/tensor.rs`, `src/simd_backend.rs`, `src/upgrade_patch.rs` into the source tree, then `Command::new("cargo").args(["check"])` to validate its own mutation; loads `state.json` via `serde_json::from_str` with no integrity/origin control. Risk: a forged `state.json` would drive code generation. The binary is clearly named and separate from the framework, but the pattern (build execution + unsigned persistence in the source tree) should be hardened (state signing, isolated output directory).
- **`scirust-sciagent/src/bin/fetch-crates.rs`** (S3): downloads crates.io tarballs, extraction `tar xzf --strip-components=1` **with no checksum verification** (crates.io does provide a hash), symlinks the `.rs` into `out/all/`. Serves as training data for `sciagent` (not executed), but: (a) no tarball integrity verification → a MITM or compromised server could substitute a tarball; (b) `tar` can extract traversal paths (`../../`) although `--strip-components=1` mitigates. **Recommendation:** verify the tarball SHA-256 against the crates.io API, validate the extracted paths, isolate the extraction directory.
- **`scirust-transpiler/examples/oracle.rs`, `scirust-runtime/tests/verify_roundtrip.rs`**: `Command::new` in test/example context — low impact.

#### 4.3.5 Deserialization of untrusted inputs

- **`scirust-core/src/io/safetensors.rs`** (S10) — **well bounded**: cap `MAX_HEADER_SIZE = 16 MiB`, explicit rejection of negative values in `shape`/`data_offsets` (before the cast to `usize` — prevents `usize::MAX` → overflow/panic), `rows.checked_mul(cols)`, validation `end ≤ data.len() && start ≤ end`, `(end-start) % 4 == 0`, `n == numel`. Regression test `deserialize_rejects_negative_shape_without_panicking`. The ad-hoc JSON parser (`find` of substrings) is honestly documented as limited (F32, 2D, headers < 16 MiB, files produced by the module itself). **Low residual risk**: a malicious header containing `"dtype":"` inside a tensor key could fool `extract_str_field` — but internal use makes this hardly exploitable.
- **`scirust-onnx/src/lib.rs`** (S11): `serde_json::from_str(json)` of an `OnnxGraph` — delegates to serde, no explicit bounds validation beyond that. Demo/interop use; to harden if loading untrusted models.
- **`scirust-graph/src/lib.rs`, `scirust-som/crates/cli/src/lib.rs`, `scirust-func-safety/src/evidence.rs`**: deserialization of internal/self-generated structures (graph, SARIF, evidence file) — relative trust, but robustness to harden on external inputs.

#### 4.3.6 Supply chain & CI/CD

- **`deny.toml`**: permissive licenses (MIT/Apache-2.0/BSD/Zlib/Unicode-3.0), `unknown-registry/git = "deny"`, justified ignore `RUSTSEC-2024-0436` (`paste` via nalgebra→simba, non-vulnerability). **Good.**
- **Committed `Cargo.lock`** + CycloneDX SBOM regenerated by `scripts/generate-sbom.sh` + CI `sbom` job (non-blocking, `continue-on-error: true`) + `release.yml` workflow attaches the SBOM to the `v*` tag. **Reproducible.**
- **CI** (S4): separate jobs (fmt on **pinned** `nightly-2026-07-02`, clippy, nightly+stable build/test, aarch64 cross-check, cargo-deny, wgpu/lavapipe, SBOM, coverage). **Gap**: most Actions are pinned by **mutable tag** (`dtolnay/rust-toolchain@nightly`/`@master`, `Swatinem/rust-cache@v2`, `EmbarkStudios/cargo-deny-action@v2`, `taiki-e/install-action@v2`, `softprops/action-gh-release@v2`, `actions/upload-artifact@v4`, `codecov/codecov-action@v4`) — compromising one of these Actions would modify the CI. Only the fmt job is pinned to a dated nightly. **Recommendation:** pin all Actions by commit SHA.
- **`rust-toolchain.toml`**: **nightly mandatory** (`rustc-dev`, `llvm-tools-preview`) — larger surface than stable; justified by `portable-simd` and `rustc_private` (driver), but the workspace also builds on stable (job `build-test-stable`).
- **`release.yml`**: `permissions: contents: write` at workflow level — necessary to create the release, but broad. No `pull_request_target` (the dangerous pattern is absent). **Acceptable.**
- **`cliptest`, `cliptest2`** (S9): ~4.4 MB ELF binaries committed at the root with no provenance or checksum. **Recommendation:** remove from the repository or document provenance + checksum.

#### 4.3.7 Sandbox / enclave / code-execution runtime

- **`scirust-runtime/src/enclave.rs`** (S1) — see §4.3.1. TEE/TrustZone entry point `#![no_std]`-friendly; the risk is the absence of `dims` ↔ slice-size validation.
- **`scirust-transpiler/`** (lower/emit/sir, front_python lexer/parser/ast): compiles a Python subset → SIR → Rust. `emit.rs` does no `eval`; it generates text. `examples/oracle.rs` executes the generated code via `Command` in example context. **No execution sandbox** but no on-the-fly execution of unapproved code in the library paths.
- **`scirust-rustc-driver/`**: `rustc_private` driver (excluded from the workspace, informational build, `continue-on-error`). High maintenance surface (nightly drift).

### 4.4 `SECURITY.md` claims vs reality

| `SECURITY.md` claim | Reality | Status |
|---|---|---|
| "Pure Rust, zero FFI" | C FFI **exported** (`safe_enclave_infer` `extern "C"` in `enclave.rs`); CUDA archive (`cublas.rs`, `cuda_backend.rs`) in C FFI | **Partially inaccurate** — the FFI is *exported* (Rust→C ABI for TEE), not *consumed* (no embedded C/C++ library), but the archive contains consumed C FFI. |
| "`unsafe` confined to SIMD intrinsics" | `unsafe` also in `scirust-arena/{slab,aligned,allocator}.rs`, `scirust-runtime/enclave.rs`, `scirust-core/{matrix,tensor,autodiff,quantization,simd}.*`, `scirust-tn/discovered_gemm.rs` | **Inaccurate** — `unsafe` is more widespread (but justified). |
| "No `unsafe` in high-level public API paths" | `EnclaveRuntime::infer` (public) wraps an `unsafe` call; `Slab::data_slice` (public) is internally `unsafe` but returns a safe `&mut [T]`. | **Partially respected** — the public API does not require `unsafe` from the caller, but relies on internal invariants (see S1). |
| "Replayable bit-exact determinism (SRT1)" | True everywhere except `scirust-func-safety/src/fault_injection.rs:118` (unseeded `rand::random`) | **Almost respected** — one exception (test mode normally). |
| "Supply chain limited to the `Cargo.lock` crates audited by `cargo deny`" | True for deps; **but** `fetch-crates.rs` downloads arbitrary crates.io code outside `Cargo.lock` | **Inaccurate for the `fetch-crates` binary** (outside workspace deps). |
| "CycloneDX SBOM regenerated at every CI run" | `sbom` job exists but `continue-on-error: true` (non-blocking) | **True but not gating.** |

---

## 5. Quality audit

### 5.1 Error handling & robustness

- **~2,044 `.unwrap()`, ~46 `panic!/todo!/unimplemented!/unreachable!`.** The majority is acceptable (tests, constructors with invariants, `split`/`parse` on controlled formats).
- **Problematic panics** (S8): `scirust-tolerance` (the most unwrap-dense crate, 40) — `modal.rs:288` `ModalBasis::orthonormalize(raw).unwrap()`, `:301` `FormBatch::new.unwrap()`; `chain.rs:431–498` `allocate().unwrap()`; `spatial.rs:498/504/515/551` `unwrap/expect("full-rank feature should fit")`. On degenerate/non-full-rank batch → **panic** instead of `Result`. Yet this crate feeds mechanical tolerance analysis (safety-adjacent). `scirust-fusion/src/fusion.rs:299/313/333/337` `fuse().expect()` on uncovered patterns → panic.
- **Recommendation:** propagate `Result`s (`GridError`/`ToleranceError`/`FusionError` exist) on these untested API-adjacent paths.

### 5.2 Test coverage

- **Good**: 696 tests (`scirust-core`), 239 (`scirust-trader`), 144 (`scirust-som`), 124 (`scirust-solvers`), 123 (`scirust-tolerance`), 115 (`scirust-mcp`), 95 (`scirust-gpu`), 93 (`scirust-ids`), 82 (`scirust-sciagent`), 56 (`scirust-discovery`), 55 (`scirust-license`), 53 (`scirust-func-safety`).
- **Crates without tests**: `scirust-gpu-macros`, `scirust-macros`, `scirust-rustc-driver`, `scirust-simd-macros`, `scirust-tensor-compile`, `scirust-tensor-examples` — proc-macros and examples, **normal**.
- **Points to strengthen**: the `unsafe` paths (enclave `dims` validation), the OT parsers (Modbus/SNMP/BER fuzzing), the safetensors parser (fuzzing of malformed headers), and `scirust-license::verify_license_on_node` (not read here — to confirm robust against tampering).

### 5.3 Architecture & coherence

- **87 crates** — fine granularity. Several "tensor" crates (`tensor-core`, `tensor-runtime`, `tensor-compile`, `tensor-contraction`, `tensor-einsum`, `tensor-examples`): boundaries to clarify to avoid duplicated responsibility.
- **`archive/`** contains CUDA/SVE/bf16 FFI code unused by the active workspace — to remove or formally isolate (impact on the "zero FFI" claim).
- **`scirust-rustc-driver`** (excluded, `rustc_private`, nightly drift) — high maintenance cost, informational build only.
- **Demonstration binaries `cliptest`/`cliptest2`** committed — anti-pattern.
- **Feature flags**: `blas-openblas`/`blas-mkl` mutually exclusive (CI does not use `--all-features`, documented); `live` (trader network) opt-in; `portable-simd` (nightly) opt-in; `wgpu` opt-in. **Coherent.**

### 5.4 Documentation, configuration & reproducibility

- Massive documentation: `README.md` (28 KB), `Documentation*.md` (8 languages), `CHANGELOG.md` (132 KB), `LIVESTATE.md` (87 KB), `docs/` (roadmaps, MEMORY_WALL, TRANSPILER_DESIGN, kb/). Risk of **obsolescence** for such large files.
- `SECURITY.md` (FR) honest about the limits (SNMPv1 in cleartext, HMAC model without PKI/revocation) — but inaccurate on FFI/unsafe (see §4.4).
- `rustfmt.toml` (nightly options) + `clippy.toml` + `RUSTFLAGS=-D warnings` in CI — **good lint discipline**.
- SBOM committed in `docs/sbom/` — to treat as a **build artifact** (not source) to avoid silent desynchronization with `Cargo.lock`.

---

## 6. Hot spots identified by the mapping

| Group | Hot spot | Type | Note |
|---|---|---|---|
| Root | `src/main.rs` (openclaw-u) | cmd exec + self-mutation + deser | Self-modifying agent, `cargo check` on generated code, unsigned `state.json` |
| Root | `cliptest`, `cliptest2` | committed binary / supply chain | 4.4 MB ELF with no provenance |
| Root | `scripts/test-protocol.sh:121` | `eval "$cmd"` | Internal variables only, low risk |
| Root | `.github/workflows/release.yml` | permissions + third-party Actions | `contents: write`, non-SHA-pinned Actions |
| Verticals | `scirust-grid/src/distance_relay.rs` | safety-correctness | Mho relay — false negative → undetected fault |
| Verticals | `scirust-grid/src/state_estimation.rs` | safety-correctness | WLS + chi2 — false trip/blindness |
| Verticals | `scirust-robotics/src/ssm.rs` | safety-correctness | ISO/TS 15066 — bound → human-robot contact |
| Verticals | `scirust-reliability/src/lib.rs` | safety-correctness | PFDavg/PFH IEC 61508 — certification basis |
| Verticals | `scirust-func-safety/src/evidence.rs` | integrity | Non-cryptographic FNV-1a chain (S7) |
| Verticals | `scirust-func-safety/src/fault_injection.rs:118` | determinism | Unseeded `rand::random` (S12) |
| Verticals | `scirust-tolerance/src/{modal,chain,spatial}.rs` | unwrap/panic | Panics on degenerate batch (S8) |
| Verticals | `scirust-fusion/src/fusion.rs` | unwrap/panic | `fuse().expect()` (S8) |
| Verticals | `scirust-graph/src/lib.rs` | deser + panic | `Serialize/Deserialize` + 15 unwrap |
| Verticals | `scirust-som/crates/cli/src/lib.rs` | deser + unwrap | SARIF parsing with `expect`/`unwrap` |
| Core/Tensor | `scirust-runtime/src/enclave.rs` | unsafe FFI + OOB | Unvalidated `dims` (S1) |
| Core/Tensor | `scirust-arena/src/{slab,aligned,allocator}.rs` | unsafe | 128-aligned, documented invariants — sound |
| Core/Tensor | `scirust-core/src/io/safetensors.rs` | deser | 16 MiB cap, negative rejection — sound (S10) |
| Agents | `scirust-mcp/src/tools/cli_passthrough.rs` | cmd exec | Exposes the whole CLI to the MCP client |
| Agents | `scirust-sciagent/src/agentic/tools.rs` | cmd exec + read | Arbitrary file reading (agent) |
| Agents | `scirust-sciagent/src/bin/fetch-crates.rs` | supply chain | Tarballs without checksum (S3) |
| OT/ICS | `scirust-discovery/src/protocols/{snmp,modbus}.rs` | OT network | Bounded parsers — sound; SNMPv1 in cleartext (by design) |
| OT/ICS | `scirust-discovery/src/scope.rs` | security gate | HMAC-signed, CIDR/protocol/time, SL3+ gate — excellent |
| OT/ICS | `scirust-discovery/src/scope.rs:106` | crypto | Non-constant-time signature comparison (S6) |
| Applications | `scirust-trader/src/wallet.rs` | crypto | Watch-only, constant-time, no key — excellent |
| Applications | `scirust-license/src/license.rs` | integrity | Anti-injection canonical encoding, salted node-lock — excellent |

---

## 7. Priority recommendations

### P0 — No confirmed critical vulnerability.

### P1 — To fix before network/TEE exposure

1. **S1 — Validate `dims` in `EnclaveRuntime::infer`** (`scirust-runtime/src/enclave.rs`): add the bounds `weights.len() ≥ out_features*in_features`, `input.len() ≥ batch*in_features`, `output.len() ≥ batch*out_features`, `bias.len() ≥ out_features` (if `has_bias`) before the `unsafe` call; return `Err(i32)` otherwise. Add tests for inconsistent `dims`.
2. **S2 — Harden the `openclaw-u` binary** (`src/main.rs`): sign `state.json` (HMAC or signature), isolate the output directory of the generated files (outside `src/`), validate the generated code before `cargo check`.
3. **S3 — Verify tarball integrity in `fetch-crates`** (`scirust-sciagent/src/bin/fetch-crates.rs`): compare the downloaded tarball's SHA-256 against the crates.io API; validate the paths extracted by `tar` (no `..`/absolute paths); isolate the extraction directory.

### P2 — Hygiene & compliance

4. **S4 — Pin all GitHub Actions by SHA** (`ci.yml`, `release.yml`).
5. **S5 — Correct `SECURITY.md`**: replace "zero FFI" with "zero embedded C/C++ library (C FFI *exported* for TEE only)"; remove or isolate `archive/` (CUDA/SVE FFI); state the real extent of `unsafe`.
6. **S6 — Homogenize signature comparison**: make `ScopeAuthorization::signature_valid` constant-time (like `WalletAuthorization::verify_signature`).
7. **S7 — Strengthen the func-safety evidence chain**: either explicitly document that `EvidencePack` is *tamper-evident* and not *tamper-resistant* (an attacker who knows the public algorithm can recompute a chain), or integrate a MAC (keyed HMAC) to make it forgery-resistant.
8. **S8 — Replace the `unwrap/expect` with `Result`** in `scirust-tolerance/{modal,chain,spatial}` and `scirust-fusion/fusion`.
9. **S9 — Remove `cliptest`/`cliptest2`** from the repository (or document provenance + checksum).
10. **S10/S11 — Fuzzing**: fuzz the Modbus/SNMP/BER parsers (`scirust-discovery`) and safetensors/ONNX (`scirust-core`, `scirust-onnx`) with `cargo-fuzz` on malformed inputs.
11. **S12 — Isolate `rand::random`** from `fault_injection.rs` outside certified paths (or seed it).

---

## 8. Appendices

### 8.1 Methodology

- Domain-group mapping (8 groups covering the 87 crates + root/docs/CI/examples).
- Per-dimension security audit (7: unsafe/FFI, OT/ICS, crypto, cmd exec, deserialization, supply chain, sandbox/enclave).
- Per-dimension quality audit (4: errors, tests, architecture, docs/config).
- Adversarial verification: re-reading the code to confirm the reachability and severity of each finding.

### 8.2 Tools

`grep`, `find`, `Read` (exhaustive reading of at-risk files), `cargo-deny` (configuration audited in `deny.toml`).

### 8.3 Limitations

- **Static** audit: no build, no tests executed, no runtime fuzzing.
- The initial multi-agent workflow was interrupted by cloud-model rate limiting; 2 domain maps (Root/docs/CI, Industrial verticals) were produced by subagents and validated; the other dimensions were audited by direct reading.
- Crates not exhaustively read: the majority of the 87 crates was not read in full — the audit focused on the files flagged by `grep` (unsafe, network, crypto, cmd, deser) and the security hot spots. A complementary per-crate pass remains possible.
- The "safety-correctness" verticals (grid, robotics, reliability, bms, maritime, nav) did not undergo a numerical-accuracy review — only structural risk (panics, determinism) was assessed.
