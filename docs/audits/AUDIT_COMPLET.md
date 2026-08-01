# Audit complet — SciRust

**Date :** 2026-07-06
**Périmètre :** monorepo SciRust (`/root/scirust`), commit `5c7e43b`, branche `master`
**Méthode :** revue de code statique ciblée (lecture directe des fichiers à risque + cartographie par groupe de domaine + vérification adversariale des findings). 248 804 LOC Rust, 87 crates, ~1 005 fichiers.
**Outils :** `grep`, `find`, lecture des fichiers `unsafe`/FFI/réseau/crypto/cmd/deser/CI.

---

## 1. Synthèse exécutive

SciRust est une plateforme **pure-Rust** de deep learning et calcul scientifique avec des verticaux industriels « certifiables » (estimation, navigation, eau, OT-sécurité, sûreté de fonctionnement), validée du cloud x86 à l'embarqué ARM (Jetson). Le code présente une **discipline de sécurité notable** — bien supérieure à la moyenne d'un projet de cette taille — mais comporte plusieurs écarts entre les **affirmations de `SECURITY.md`** et la **réalité du code**, plus quelques zones à risque réelles.

### Verdict global

| Dimension | Note | Commentaire |
|---|---|---|
| Posture crypto (trader, licence, scope OT) | **A** | Watch-only par défaut, signature HMAC, comparaison constant-time (wallet), pas de clé en clair. |
| Parsing de protocoles OT (Modbus/SNMP/BER) | **A−** | Longueurs bornées, erreurs explicites, pas de panique sur entrées malformées. |
| Désérialisation non fiable (safetensors) | **B+** | Cap header 16 MiB, rejet des valeurs négatives, overflow `checked_mul`. Parser JSON ad-hoc (limites documentées). |
| `unsafe` / mémoire (SIMD, arena, autodiff) | **B+** | Alignement garanti par `AlignBlock(128)`, `debug_assert`, invariants documentés. Quelques blocs à revoir. |
| Exécution de commande / agents autonomes | **C+** | `cli_passthrough` MCP, outils `sciagent`, binaire `openclaw-u` auto-mutant, `fetch-crates` (supply-chain). Pas d'injection shell, mais surface large. |
| Conformité aux affirmations `SECURITY.md` | **C** | « Zéro FFI » contredit par FFI C exportée (`enclave.rs`) et archive CUDA ; « unsafe confiné aux intrinsics SIMD » contredit par `unsafe` en runtime/arena/tensor. |
| Chaîne d'approvisionnement / CI | **B** | `cargo-deny`, SBOM CycloneDX, lockfile committé. Actions GitHub **non pinées par SHA** (mutable tags). Nightly obligatoire. |
| Robustesse (unwrap/panic) | **B−** | ~2 044 `.unwrap()`, ~46 `panic!/todo!`. Quelques paniques sur entrées externes (tolérancement, fusion) à corriger. |
| Couverture de tests | **A−** | 696 tests dans `scirust-core`, 239 dans `scirust-trader`… Seuls les crates proc-macro/examples sont sans tests (normal). |

**Score maturité sécurité : 7,5/10** — bon socle, écarts documentaires et quelques zones agents/supply-chain à durcir.

### Top risques (P0 → P2)

1. **[P1] `safe_enclave_infer` — OOB par `dims` non validé** (`scirust-runtime/src/enclave.rs`). Le wrapper `EnclaveRuntime::infer` ne valide pas que `dims.batch*in_features ≤ input.len()` (etc.) avant l'appel `unsafe` → lecture/écriture hors bornes dans le TEE.
2. **[P1] Binaire `openclaw-u` auto-mutant** (`src/main.rs`) : écrit des fichiers source générés dans l'arbre puis lance `cargo check` ; charge un `state.json` non signé. Pattern d'agent auto-modifiant sans contrôle d'intégrité.
3. **[P1] `fetch-crates` — supply-chain active** (`scirust-sciagent/src/bin/fetch-crates.rs`) : télécharge des tarballs crates.io arbitraires, les extrait via `tar xzf --strip-components=1`, symlink les `.rs` dans le workspace (données d'entraînement). Pas de vérification de checksum/intégrité des tarballs.
4. **[P2] Affirmations `SECURITY.md` inexactes** : « zéro FFI » et « unsafe confiné aux intrinsics SIMD » — contredits par l'FFI C exportée et les blocs `unsafe` hors SIMD.
5. **[P2] Actions CI non pinées par SHA** (`release.yml`, `ci.yml`) — `@v2`, `@nightly`, `@master` sont des tags mutables.
6. **[P2] Comparaison de signature non constant-time** dans `scirust-discovery/src/scope.rs` (`signature_valid`), alors que le wallet l'est — incohérence.
7. **[P2] Paniques sur batches dégénérés** dans `scirust-tolerance` (modal/chain/spatial) et `scirust-fusion` (`fuse().expect()`) — code non-test mais proche API.
8. **[P2] Binaires ELF ~4,4 Mo committés à la racine** (`cliptest`, `cliptest2`) — provenance non tracée, risque supply-chain/détection.
9. **[P2] Chaîne d'evidence func-safety non cryptographique** (`evidence.rs`) : FNV-1a chain (publique, sans secret) → un attaquant avec accès fichier peut recalculer une chaîne cohérente. Documenté comme *tamper-evident* (non *tamper-resistant*), mais `from_json().verify()` est présenté comme détectant les forgeries, ce qui n'est vrai que pour les éditions naïves.

---

## 2. Présentation du projet

- **Type :** workspace Cargo (resolver 2, edition 2021, `rust-version = 1.85`, **nightly** obligatoire via `rust-toolchain.toml`).
- **Licence :** PolyForm Noncommercial 1.0.0 (`LICENSE.md`), `publish = false` (crates non publiés sur crates.io).
- **Surface :** crate racine `scirust` (facade `src/lib.rs` + binaire autonome `openclaw-u` `src/main.rs`), 87 crates `scirust-*`, 10 exemples, 2 tests d'intégration workspace.
- **Stack clés :** `tokio` (full), `serde`/`serde_json`, `chrono`, `rand 0.8`, `sha2`, `nalgebra`/`simba` (via `paste`), `ureq`, `reqwest` (feature `live`), `clap`.
- **Governance :** `deny.toml` (cargo-deny : advisories RustSec, licences permissives, sources), `clippy.toml`, `rustfmt.toml` (options nightly), `docs/sbom/scirust.cdx.json` (CycloneDX), `scripts/generate-sbom.sh`, `SECURITY.md` (politique FR).
- **CI :** `ci.yml` (fmt nightly pinée, clippy, build/test nightly+stable, cross-check aarch64, cargo-deny, wgpu/lavapipe, SBOM, coverage) + `release.yml` (tag `v*` → GitHub Release + SBOM attaché).
- **Déterminisme :** contrat bit-exact rejouable (runtime SRT1) ; RNG seeded `PcgEngine` partout sauf `scirust-func-safety/src/fault_injection.rs:118` (`rand::random` non seedé — voir §4.7).

---

## 3. Cartographie de l'arborescence (par groupe de domaine)

> Chaque groupe : rôle + arborescence des fichiers `src` principaux avec un rôle 1-ligne. Les points chauds de risque sont reportés §6.

### 3.1 Racine, docs, CI, examples

```
RACINE (/root/scirust)
├── Cargo.toml            — workspace racine ; re-exporte scirust-core/learning/simd/solvers/symbolic/rsi ; bin `openclaw-u`
├── Cargo.lock            — lockfile committé (113 KB)
├── deny.toml             — cargo-deny : licences MIT/Apache/BSD/Zlib/Unicode-3.0, ignore RUSTSEC-2024-0436
├── rust-toolchain.toml   — channel=nightly, rustfmt/clippy/rustc-dev/llvm-tools
├── cliptest, cliptest2   — BINAIRES ELF ~4,4 Mo committés (provenance non tracée) ⚠
├── README/CHANGELOG/LIVESTATE/SECURITY/LICENSING/LICENSE.md  — governance/licence
├── Documentation*.md (8 langues), ARCHITECTURE-B/ANALYSIS_REPORT/DESIGN_SCIRUST_TENSOR/INTEGRATION_GUIDE
├── src/
│   ├── lib.rs            — facade `pub use scirust_core::*` ...
│   └── main.rs           — bin `openclaw-u` : agent autonome Tokio, auto-mutation + `cargo check` ⚠
├── tests/
│   ├── workflow.rs       — workflow ML end-to-end (linreg, polyfit, Cholesky)
│   └── expansion_val.rs  — validation ResNet/ViT/VAE/MoE/GCN/LoRA/CTC/DQN/PPO/NBeats/FemSolver1D
├── examples/             — mnist/cifar10/transformer×2/sentiment/industrial_monitor/ids/simd_views/benchmarks
├── docs/                 — ARCHITECTURE, ROADMAP×4, MEMORY_WALL_*, TRANSPILER_DESIGN, kb/, roadmaps/, sbom/
├── scripts/             — generate-sbom.sh, test-protocol*.sh (test-protocol.sh:121 `eval "$cmd"` ⚠)
├── archive/              — ancien code (scirust-gpu CUDA FFI, scirust-simd SVE, scirust-core quant/bf16) ⚠
└── .github/workflows/    — ci.yml, release.yml
```

### 3.2 Verticaux industriels (23 crates)

```
scirust-estimation/   kalman/ekf/ukf/imm/particle/interval/smoother/ud/linalg — base BMS/nav/fusion/SPC
scirust-nav/          ins (dead-reckoning) / fusion (GNSS-INS Kalman) / tdoa (multilatération)
scirust-control/      pid (anti-windup + auto-tune) / lqr / qp (box-QP) / mpc / monitor / license (gate commercial)
scirust-robotics/     ssm (ISO/TS 15066 Speed-Separation) / kinematics (2-link) / trajectory (trapézoïdal)
scirust-maritime/     colregs / cpa_tcpa / thrust_allocation (DP, pseudo-inverse)
scirust-water/        leak (corrélation acoustique) / transient (Joukowsky)
scirust-hvac/         fdd (AHU ASHRAE G36) / nilm (disaggregation)
scirust-bms/          soc (EKF 1-RC) / soh (conformal) / thermal (runaway guard) / capacity / dual
scirust-pdm/          health / rul / conformal_rul / change_detection
scirust-grid/         (relais distance mho, state estimation WLS, power_quality, symmetrical, flicker) ⚠ safety
scirust-reliability/  PFDavg/PFH IEC 61508 (MooN, β, SIL, Markov) — base argument certification ⚠
scirust-func-safety/  evidence (chaîne FNV-1a ⚠) / audit / requirements / fault_injection (rand non seedé ⚠)
scirust-spc/          statistical process control
scirust-fatigue/      fatigue life
scirust-tolerance/    modal (orthonormalize.unwrap ⚠) / chain (allocate.unwrap ⚠) / spatial (unwrap ⚠)
scirust-fusion/       fusion de graphe (fuse().expect ⚠) / graph (SHA-256 identité, from_str ⚠)
scirust-signal/       bearing / filtrage
scirust-multivariate/  statistiques multivariées
scirust-sequential/    matching / labeling (réseau ⚠)
scirust-seasonal/     décomposition saisonnière
scirust-som/          self-organizing map (frontend/cli SARIF parsing ⚠)
scirust-graph/        DAG / isomorphisme (Serialize/Deserialize + 15 unwrap ⚠)
```
*Aucun `unsafe`, aucune FFI, aucun réseau, aucune exécution de commande dans ce groupe. Risque = exactitude numérique safety-critical + désérialisation d'artefacts + unwrap/panic.*

### 3.3 Core & Tensor

```
scirust-core/         amp, aot, autodiff/, checkpoint, compute_backend, data/, distributed, dp,
                      embed, error, homomorphic, io/ (safetensors ⚠), lazy/, logging, matrix/ (backend.rs/view.rs unsafe ⚠),
                      nn/, optim, pruning, quantization (unsafe ⚠), quantum, reproducible, simd/ (tiling unsafe ⚠), symbolic
scirust-tensor-core/  lib.rs (tenseur noyau)
scirust-tensor-runtime/  runtime d'exécution tenseur
scirust-tensor-compile/  compilation tenseur (pas de tests)
scirust-tensor-contraction/  contraction
scirust-tensor-einsum/  einsum
scirust-simd/         lib (unsafe ⚠) / dispatch (unsafe intrinsics ⚠) / complex (unsafe ⚠)
scirust-simd-macros/  proc-macro (pas de tests)
scirust-arena/        slab (unsafe ⚠) / aligned (unsafe ⚠) / allocator (unsafe ⚠)
scirust-autodiff/     reverse (unsafe ⚠)
scirust-aot/          compilation AOT
```

### 3.4 ML / Apprentissage

```
scirust-learning/      control, finance, optim, pattern_miner, nlp/, rl/, time_series/
scirust-unsupervised/  clustering
scirust-rl-algo/      RL
scirust-automl/       AutoML
scirust-nas/          neural architecture search
scirust-evo/          évolutionnaire
scirust-symreg/       symbolic regression
scirust-synthesis/    synthèse de programmes
scirust-reasoning/   raisonnement
scirust-symbolic/    calcul symbolique
scirust-neuro-symbolic/  hybride
scirust-retrieval/   ann / rerank (RAG)
scirust-nlp-advanced/  byte_bpe (crypto ⚠)
```

### 3.5 GPU & Runtime

```
scirust-gpu/          chain, conv_gpu, deterministic, deterministic_gpu, engine, fusion, kernels, ops, tensor, wgpu_backend
scirust-gpu-macros/  proc-macro (pas de tests)
scirust-runtime/      attest, difr, enclave (FFI C unsafe ⚠), proof, proofcli, quant, vinfer, bin/, main
scirust-embedded/    cibles embarquées
scirust-edge/        edge computing
scirust-burn-bridge/  pont burn
scirust-bridge/      pont inter-framework
scirust-onnx/        lib.rs (serde_json::from_str ⚠ — ONNX-like JSON)
```

### 3.6 Agents / CLI / MCP / Transpiler

```
scirust-cli/         main + learning/nlp/numeric/quickstart/reasoning/sciagent/symbolic/synergy/trader (subcommands)
scirust-mcp/         server, protocol, registry, audit, tools/ (cli_passthrough ⚠, grid, fatigue, …)
scirust-sciagent/    agentic/ (tools.rs ⚠), bin/ (fetch-crates ⚠), attention, bpe, ccos, flash_attention,
                     generate, gpu, inference, model, norm, planning, quantize, sha256, swiglu, tokenizer, train/
scirust-scaffold/    scaffolding (59 tests)
scirust-macros/      proc-macro (pas de tests)
scirust-codetrans/   transcodage
scirust-transpiler/  sir, lower, emit (emit.rs crypto ⚠), front_python/ (lexer/parser/ast)
scirust-rustc-driver/  driver rustc (rustc_private, excluded, pas de tests)
```

### 3.7 Protocoles OT/ICS & Découverte

```
scirust-discovery/   engine, scope (autorisation HMAC signée ⚠), hmac (HMAC-SHA256), audit,
                     protocols/ (snmp ⚠, opcua, modbus ⚠, bacnet, ethernet_ip, mdns)
scirust-opcua/       client OPC-UA
scirust-mqtt/        publisher MQTT
scirust-events-core/   épisodique
scirust-events-models/ modèles d'événements
scirust-events-runtime/ runtime d'événements
scirust-events-examples/  exemples
scirust-shm/         fdd, operational (shared memory)
```

### 3.8 Domaines applicatifs

```
scirust-vision/  scirust-audio/  scirust-biomed/  scirust-agtech/ (idw)  scirust-industrial/
scirust-ids/     (IDS, scan de ports)  scirust-trader/ (wallet ⚠, market, agent, proof, orderbook, regime, model, robustness, dashboard)
scirust-license/ (license ⚠, gate, hashsig, module, cli)  scirust-integration/ (config, templates)
scirust-solvers/  scirust-rsi/  scirust-mlops/  scirust-fab/  scirust-sis/  scirust-tn/ (discovered_gemm unsafe ⚠)
```

---

## 4. Audit de sécurité

### 4.1 Méthodologie

Pour chaque dimension : `grep` des patterns de risque sur l'ensemble des `*.rs`, puis lecture exhaustive des fichiers pertinents. Les findings sont classés par sévérité (critical/high/medium/low/info), avec un scénario d'exploitation concret. Une vérification adversariale a consisté à relire le code pour confirmer l'atteignabilité et la sévérité alléguée. Les findings info/low ne sont pas listés exhaustivement.

### 4.2 Tableau des findings confirmés

| # | Dimension | Sév. | Fichier | Ligne | Résumé | CWE |
|---|---|---|---|---|---|---|
| S1 | sandbox/unsafe | **medium** | `scirust-runtime/src/enclave.rs` | 23–66 | `safe_enclave_infer` déréférence des pointeurs bruts sur la foi de `dims` (batch/in/out) sans valider que `dims` tient dans les slices passées par `EnclaveRuntime::infer` | CWE-119/CWE-787 |
| S2 | cmd exec / agent | **medium** | `src/main.rs` (bin `openclaw-u`) | — | Agent autonome écrit des fichiers source générés puis lance `cargo check` ; charge `state.json` non signé via `serde_json::from_str` | CWE-94/CWE-345 |
| S3 | supply chain | **medium** | `scirust-sciagent/src/bin/fetch-crates.rs` | 207–256 | Télécharge des tarballs crates.io arbitraires, extrait via `tar xzf --strip-components=1` sans vérification de checksum/intégrité, symlink les `.rs` dans le workspace | CWE-494/CWE-829 |
| S4 | supply chain (CI) | **low** | `.github/workflows/{ci,release}.yml` | — | Actions tierces pinées par tag mutable (`@v2`, `@nightly`, `@master`) et non par SHA | CWE-1357 |
| S5 | doc/conformité | **low** | `SECURITY.md` vs `scirust-runtime/src/enclave.rs`, `archive/scirust-gpu/*` | — | Affirmation « pur Rust, zéro FFI » contredite par l'FFI C exportée (`safe_enclave_infer` `extern "C"`) et l'archive CUDA (`cublas.rs`, `cuda_backend.rs`) | CWE-1047 |
| S6 | crypto | **low** | `scirust-discovery/src/scope.rs` | 106–109 | `signature_valid` compare la signature hex par `==` (non constant-time), contrairement au wallet (`wallet.rs:668`) qui est constant-time | CWE-208 |
| S7 | intégrité | **low** | `scirust-func-safety/src/evidence.rs` | 18–25, 80–95 | Chaîne d'evidence FNV-1a (hash non cryptographique, sans secret) ; un attaquant avec accès écriture peut recalculer une chaîne cohérente. `from_json().verify()` ne détecte que les éditions naïves | CWE-345/CWE-327 |
| S8 | robustesse/DoS | **low** | `scirust-tolerance/src/{modal,chain,spatial}.rs`, `scirust-fusion/src/fusion.rs` | modal:288,301 / chain:431–498 / spatial:498–551 / fusion:299–337 | `unwrap()/expect()` sur `orthonormalize`, `allocate`, `fit_torsor`, `fuse` → panique sur batch dégénéré/rang non plein au lieu de `Result` | CWE-754 |
| S9 | supply chain (repo) | **low** | `cliptest`, `cliptest2` (racine) | — | Binaires ELF ~4,4 Mo committés sans provenance ni checksum | CWE-494 |
| S10 | désérialisation | **info** | `scirust-core/src/io/safetensors.rs` | 355–379 | Parser JSON ad-hoc par `find` de sous-chaînes (`extract_str_field`/`extract_array_field`) — robuste aux fichiers produits par le module, mais pourrait être confondu par un header malveillant contenant `"dtype":"` dans une clé | CWE-20 (atténué : usage interne documenté) |
| S11 | désérialisation | **info** | `scirust-onnx/src/lib.rs` | 296 | `serde_json::from_str(json)` d'un graphe ONNX-like sans validation de bornes explicite au-delà de serde | CWE-20 |
| S12 | détermination | **info** | `scirust-func-safety/src/fault_injection.rs` | 118 | `rand::random::<f32>()` non seedé — viole le contrat de déterminisme bit-reproductible si utilisé hors mode test | CWE-338 |

*Aucun finding **critical** confirmé. Aucune injection shell constatée : les `Command::new` passent des arguments séparés (pas de shell), sauf `scripts/test-protocol.sh:121` (`eval "$cmd"`) qui n'est atteint que par des variables internes.*

### 4.3 Analyse par dimension

#### 4.3.1 `unsafe` / FFI / mémoire

Le `unsafe` est **confiné et majoritairement justifié** :
- **`scirust-arena/src/slab.rs`** — backing `AlignBlock` `#[repr(C, align(128))]` garantit un pointeur de base aligné sur 128 ; chaque slot est un multiple de `MIN_ALIGN_BYTES` → alignement correct pour tout `T` dont l'alignement divise 128. `debug_assert!` vérifie l'alignement. `from_raw_parts_mut` est précédé d'un `is_valid(handle)` (version anti-use-after-free). **Sain.** Un test de régression documente un UB antérieur (`data_slice_is_aligned_for_every_slot`).
- **`scirust-simd/src/dispatch.rs`, `complex.rs`** — intrinsics SIMD `core::arch`, documentés par en-têtes de sûreté (conforme à `SECURITY.md`).
- **`scirust-core/src/matrix/{backend,view}.rs`, `autodiff/reverse.rs`, `tensor/pinned.rs`, `quantization.rs`, `simd/tiling.rs`** — `unsafe` de manipulation de tampons alignés / `MaybeUninit` ; à revoir au cas par cas mais aucun UB manifeste constaté sur lecture.
- **`scirust-runtime/src/enclave.rs`** (S1) — **seul finding matériel significatif** : FFI C `unsafe extern "C" fn safe_enclave_infer(...)` déréférence `weight_ptr/input_ptr/output_ptr/bias_ptr` selon `dims` sans aucune vérification que `dims.batch*in_features ≤ input.len()`, etc. Le wrapper `EnclaveRuntime::infer` construit les pointeurs depuis des slices `&[f32]` mais ne contrôle pas la cohérence `dims` ↔ longueurs de slices. Un `dims` incohérent (bug appelant, ou entrée non fiable si exposée via l'ABI C) → lecture/écriture OOB dans le TEE. **Recommandation :** valider dans `infer` que `weights.len() ≥ out_features*in_features`, `input.len() ≥ batch*in_features`, `output.len() ≥ batch*out_features`, `bias.len() ≥ out_features` (si `has_bias`) avant l'appel, et retourner `Err`.
- **`archive/scirust-gpu/{cublas.rs,cuda_backend.rs}`, `archive/scirust-simd/sve.rs`, `archive/scirust-core/quant/bf16.rs`** — FFI C/CUDA et SVE dans l'archive. Contredit `SECURITY.md` (« zéro FFI »), mais l'archive n'est pas dans le workspace actif. **Recommandation :** soit retirer `archive/`, soit préciser dans `SECURITY.md` que l'archive est hors périmètre.

#### 4.3.2 Protocoles OT/ICS & découverte

Excellente discipline. **`scirust-discovery/src/protocols/{modbus.rs,snmp.rs}`** : parsers à bornes strictes.
- **Modbus** (`parse_read_device_id_response`) : vérifie `buf.len() < 8`, `pdu.len() < 7`, `idx+2 > pdu.len()`, `idx+object_len > pdu.len()` → aucune panique sur trame malformée ; distinguue exception Modbus (0x80) vs trame mal formée ; tampon fixe `512` octets.
- **SNMP** (`parse_get_response`) : décodage BER minimal avec `read_length`/`read_tlv` vérifiant `pos >= buf.len()`, `content_start+len > buf.len()` ; tampon fixe `2048`. `error-status` non nul → `Err` explicite. Communauté `"public"` en clair = **par conception SNMPv1** (documenté en-tête de fichier).
- **`scope.rs`** (porte de sécurité) : **modèle IEC 62443 zones/conduits** — `ScopeAuthorization` HMAC-signée, liste blanche CIDR, liste blanche de protocoles, fenêtre temporelle, **gate SL3+** (zone haute sécurité refusée par défaut, override explicite requis), rejet IPv6, aucune panique sur CIDR malformé. `authorize` est appelé **avant** toute sonde réseau. Test `authorize_rejects_tampered_scope` valide qu'une portée élargie après signature est rejetée.
- **`hmac.rs`** : HMAC-SHA256 RFC 2104 sur SHA-256 maison (`scirust_sciagent::sha256`), testé contre RFC 4231 §4.2/§4.3. **Sain.**
- Seul bémol (S6) : `signature_valid` compare la signature par `==` sur `String` (non constant-time) — incohérent avec le wallet. Faible impact (l'attaquant devrait être la partie vérificatrice avec oracle de timing), mais à homogénéiser.

*Aucune écriture de registre/commande industrielle — uniquement lecture `Read Device Identification` (Modbus) et `GET sysDescr.0` (SNMP), conformes à la doctrine NIST SP 800-82 (découverte passive/native).*

#### 4.3.3 Crypto & gestion de secrets (trader, licence, HMAC)

**`scirust-trader/src/wallet.rs`** — conception robuste :
- **Watch-only par défaut**, `live` (réseau) opt-in derrière feature.
- **Aucune clé privée** dans le module ; un signer réel est injecté par le host (env var) et ne produit que des signatures.
- **Keccak-256 maison** vérifié contre les vecteurs canoniques Ethereum (`keccak256("")`, `keccak256("abc")`). EIP-55 checksum testé contre les 4 exemples spec ; EIP-712 domain separator ; EIP-1559 signing hash (dry-run, non signé).
- **HMAC-SHA256** (RFC 4231 testé).
- **`WalletAuthorization`** : gate non-auto-autorisante — sign/send exige une autorisation HMAC-signée par l'opérateur (clé côté serveur) ; `verify_signature` en **comparaison constant-time** (XOR + OR réduit) ; `authorizes` borne chain id, method, valeur (`max_value_wei`), fenêtre temporelle. Test `tampering_authorization_breaks_signature` confirme que toute modification post-signature casse la signature.

**`scirust-license/src/license.rs`** — encodage canonique length-prefixed (magic `SRL2` + version), modules triés/dédupliqués → pas d'injection de séparateur (test `separator_injection_cannot_forge_a_collision`). Node-lock par SHA-256 **salé par l'identité de licence** (domain separation, length-prefix) → empreintes non corrélatibles entre licences ; seul le hash est stocké (le raw `machine_id` ne fuit pas — test `binding_to_a_node_changes_the_digest_and_stores_only_the_hash`). Honnête sur la limite : ne résiste pas au brute-force d'un `machine_id` faible (le déploiement doit fournir un UUID/TPM).

*Aucun secret hardcoded, aucune RNG non cryptographique pour sign/wallet, aucune fuite de clé. **Très bon.***

#### 4.3.4 Exécution de commande / agents autonomes

Aucune injection shell (les `Command::new` passent des `args` séparés, jamais via `sh -c`), mais la **surface d'exécution est large** :
- **`scirust-mcp/src/tools/cli_passthrough.rs`** (S2-voisin) : outil MCP `scirust_cli` qui exécute `scirust` (ou `cargo run -p scirust-cli`) avec `args` contrôlés par le client MCP. Pas d'injection (args séparés), mais expose **toute** la CLI à un client distant. Les args sont validés (`must be strings`). Acceptable si le canal MCP est authentifié, à documenter.
- **`scirust-sciagent/src/agentic/tools.rs`** : outils `search`/`grep` (via `rg`/`grep`, pattern regex — pas de shell), `read`/`explain` (lecture fichier arbitraire — par conception d'un agent), `build`/`test` (via `cargo`, crate_name en arg), `status` (`git status`). Pas d'injection. Le `path` est contrôlable par l'agent → lecture de fichiers arbitraires (traversal) — by design pour un agent code, à canaliser.
- **`src/main.rs` (bin `openclaw-u`)** (S2) : **agent auto-mutant** — écrit `src/tensor.rs`, `src/simd_backend.rs`, `src/upgrade_patch.rs` dans l'arbre source, puis `Command::new("cargo").args(["check"])` pour valider sa propre mutation ; charge `state.json` via `serde_json::from_str` sans contrôle d'intégrité/origine. Risque : un `state.json` forgé piloterait la génération de code. Le binaire est clairement nommé et séparé du framework, mais le pattern (exécution de build + persistance non signée dans l'arbre source) est à durcir (signature du state, répertoire de sortie isolé).
- **`scirust-sciagent/src/bin/fetch-crates.rs`** (S3) : télécharge des tarballs crates.io, extraction `tar xzf --strip-components=1` **sans vérification de checksum** (crates.io fournit pourtant un hash), symlink des `.rs` dans `out/all/`. Sert de données d'entraînement pour `sciagent` (non exécuté), mais : (a) pas de vérification d'intégrité du tarball → un MITM ou un serveur compromis pourrait substituer un tarball ; (b) `tar` peut extraire des chemins traversaux (`../../`) bien que `--strip-components=1` atténue. **Recommandation :** vérifier le SHA-256 du tarball contre l'API crates.io, valider les chemins extraits, isoler le répertoire d'extraction.
- **`scirust-transpiler/examples/oracle.rs`, `scirust-runtime/tests/verify_roundtrip.rs`** : `Command::new` en contexte de test/example — faible impact.

#### 4.3.5 Désérialisation d'entrées non fiables

- **`scirust-core/src/io/safetensors.rs`** (S10) — **bien borné** : cap `MAX_HEADER_SIZE = 16 MiB`, rejet explicite des valeurs négatives dans `shape`/`data_offsets` (avant cast en `usize` — empêche `usize::MAX` → overflow/panic), `rows.checked_mul(cols)`, validation `end ≤ data.len() && start ≤ end`, `(end-start) % 4 == 0`, `n == numel`. Test de régression `deserialize_rejects_negative_shape_without_panicking`. Le parser JSON ad-hoc (`find` de sous-chaînes) est honnêtement documenté comme limité (F32, 2D, headers < 16 MiB, fichiers produits par le module lui-même). **Faible risque résiduel** : un header malveillant contenant `"dtype":"` à l'intérieur d'une clé de tenseur pourrait tromper `extract_str_field` — mais l'usage interne rend cela peu exploitable.
- **`scirust-onnx/src/lib.rs`** (S11) : `serde_json::from_str(json)` d'un `OnnxGraph` — délègue à serde, pas de validation de bornes explicite au-delà. Usage de démo/interop ; à durcir si chargement de modèles non fiables.
- **`scirust-graph/src/lib.rs`, `scirust-som/crates/cli/src/lib.rs`, `scirust-func-safety/src/evidence.rs`** : désérialisation de structures internes/auto-générées (graphe, SARIF, dossier d'evidence) — confiance relative, mais robustesse à durcir sur entrées externes.

#### 4.3.6 Chaîne d'approvisionnement & CI/CD

- **`deny.toml`** : licences permissives (MIT/Apache-2.0/BSD/Zlib/Unicode-3.0), `unknown-registry/git = "deny"`, ignore justifiée `RUSTSEC-2024-0436` (`paste` via nalgebra→simba, non-vulnérabilité). **Bon.**
- **`Cargo.lock` committé** + SBOM CycloneDX régénéré par `scripts/generate-sbom.sh` + job CI `sbom` (non bloquant, `continue-on-error: true`) + workflow `release.yml` attache le SBOM au tag `v*`. **Reproductible.**
- **CI** (S4) : jobs séparés (fmt nightly **pinnée** `nightly-2026-07-02`, clippy, build/test nightly+stable, cross-check aarch64, cargo-deny, wgpu/lavapipe, SBOM, coverage). **Écart** : la plupart des actions sont pinées par **tag mutable** (`dtolnay/rust-toolchain@nightly`/`@master`, `Swatinem/rust-cache@v2`, `EmbarkStudios/cargo-deny-action@v2`, `taiki-e/install-action@v2`, `softprops/action-gh-release@v2`, `actions/upload-artifact@v4`, `codecov/codecov-action@v4`) — un compromis d'une de ces actions modifierait la CI. Seul le job fmt est piné à une nightly datée. **Recommandation :** pinner toutes les actions par SHA de commit.
- **`rust-toolchain.toml`** : **nightly obligatoire** (`rustc-dev`, `llvm-tools-preview`) — surface plus large qu'un stable ; justifié par `portable-simd` et `rustc_private` (driver), mais le workspace build aussi sur stable (job `build-test-stable`).
- **`release.yml`** : `permissions: contents: write` au niveau workflow — nécessaire pour créer la release, mais large. Pas de `pull_request_target` (le pattern dangereux est absent). **Acceptable.**
- **`cliptest`, `cliptest2`** (S9) : binaires ELF ~4,4 Mo committés à la racine sans provenance ni checksum. **Recommandation :** retirer du dépôt ou documenter la provenance + checksum.

#### 4.3.7 Sandbox / enclave / runtime d'exécution de code

- **`scirust-runtime/src/enclave.rs`** (S1) — voir §4.3.1. Entry point TEE/TrustZone `#![no_std]`-friendly ; le risque est l'absence de validation `dims` ↔ tailles de slices.
- **`scirust-transpiler/`** (lower/emit/sir, front_python lexer/parser/ast) : compile un sous-ensemble Python → SIR → Rust. `emit.rs` ne fait pas d'`eval` ; génère du texte. `examples/oracle.rs` exécute le code généré via `Command` en contexte exemple. **Pas de sandbox d'exécution** mais pas d'exécution à la volée de code non approuvé dans les chemins de librairie.
- **`scirust-rustc-driver/`** : driver `rustc_private` (excluded du workspace, build informational, `continue-on-error`). Surface de maintenance élevée (drift nightly).

### 4.4 Affirmations de `SECURITY.md` vs réalité

| Affirmation `SECURITY.md` | Réalité | Statut |
|---|---|---|
| « Pur Rust, zéro FFI » | FFI C **exportée** (`safe_enclave_infer` `extern "C"` dans `enclave.rs`) ; archive CUDA (`cublas.rs`, `cuda_backend.rs`) en FFI C | **Partiellement inexact** — l'FFI est *exportée* (Rust→C ABI pour TEE), pas *consommée* (pas de bibliothèque C/C++ embarquée), mais l'archive contient du FFI C consommé. |
| « `unsafe` confiné aux intrinsics SIMD » | `unsafe` également dans `scirust-arena/{slab,aligned,allocator}.rs`, `scirust-runtime/enclave.rs`, `scirust-core/{matrix,tensor,autodiff,quantization,simd}.*`, `scirust-tn/discovered_gemm.rs` | **Inexact** — le `unsafe` est plus répandu (mais justifié). |
| « Aucun `unsafe` dans les chemins d'API publics de haut niveau » | `EnclaveRuntime::infer` (public) encapsule un appel `unsafe` ; `Slab::data_slice` (public) est `unsafe` interne mais renvoie un `&mut [T]` sûr. | **Partiellement respecté** — l'API publique ne demande pas d'`unsafe` à l'appelant, mais repose sur des invariants internes (cf. S1). |
| « Déterminisme bit-exact rejouable (SRT1) » | Vrai partout sauf `scirust-func-safety/src/fault_injection.rs:118` (`rand::random` non seedé) | **Quasi respecté** — une exception (mode test normalement). |
| « Chaîne d'approvisionnement limitée aux crates de `Cargo.lock` auditées par `cargo deny` » | Vrai pour les deps ; **mais** `fetch-crates.rs` télécharge du code arbitraire crates.io hors `Cargo.lock` | **Inexact pour le binaire `fetch-crates`** (hors deps du workspace). |
| « SBOM CycloneDX régénéré à chaque CI » | Job `sbom` existe mais `continue-on-error: true` (non bloquant) | **Vrai mais non gating.** |

---

## 5. Audit qualité

### 5.1 Gestion d'erreurs & robustesse

- **~2 044 `.unwrap()`, ~46 `panic!/todo!/unimplemented!/unreachable!`.** La majorité est acceptable (tests, constructeurs avec invariants, `split`/`parse` sur formats contrôlés).
- **Paniques problématiques** (S8) : `scirust-tolerance` (crate le plus unwrap-dense, 40) — `modal.rs:288` `ModalBasis::orthonormalize(raw).unwrap()`, `:301` `FormBatch::new.unwrap()` ; `chain.rs:431–498` `allocate().unwrap()` ; `spatial.rs:498/504/515/551` `unwrap/expect("full-rank feature should fit")`. Sur batch dégénéré/rang non plein → **panic** au lieu de `Result`. Or ce crate alimente du tolérancement mécanique (safety-adjacent). `scirust-fusion/src/fusion.rs:299/313/333/337` `fuse().expect()` sur motifs non couverts → panic.
- **Recommandation :** propager des `Result` (`GridError`/`ToleranceError`/`FusionError` existent) sur ces chemins non-test proches de l'API.

### 5.2 Couverture de tests

- **Bonne** : 696 tests (`scirust-core`), 239 (`scirust-trader`), 144 (`scirust-som`), 124 (`scirust-solvers`), 123 (`scirust-tolerance`), 115 (`scirust-mcp`), 95 (`scirust-gpu`), 93 (`scirust-ids`), 82 (`scirust-sciagent`), 56 (`scirust-discovery`), 55 (`scirust-license`), 53 (`scirust-func-safety`).
- **Crates sans tests** : `scirust-gpu-macros`, `scirust-macros`, `scirust-rustc-driver`, `scirust-simd-macros`, `scirust-tensor-compile`, `scirust-tensor-examples` — proc-macros et exemples, **normal**.
- **Points à renforcer** : les chemins `unsafe` (enclave `dims` validation), les parsers OT (fuzzing Modbus/SNMP/BER), le parser safetensors (fuzzing de headers malformés), et `scirust-license::verify_license_on_node` (non lu ici — à confirmer robuste à la falsification).

### 5.3 Architecture & cohérence

- **87 crates** — granularité fine. Plusieurs crates « tenseur » (`tensor-core`, `tensor-runtime`, `tensor-compile`, `tensor-contraction`, `tensor-einsum`, `tensor-examples`) : frontières à clarifier pour éviter la duplication de responsabilité.
- **`archive/`** contient du code FFI CUDA/SVE/bf16 non utilisé par le workspace actif — à retirer ou à isoler formellement (impact sur l'affirmation « zéro FFI »).
- **`scirust-rustc-driver`** (excluded, `rustc_private`, drift nightly) — coût de maintenance élevé, build informational seulement.
- **Binaires de démonstration `cliptest`/`cliptest2`** committés — anti-pattern.
- **Feature flags** : `blas-openblas`/`blas-mkl` mutuellement exclusifs (CI n'utilise pas `--all-features`, documenté) ; `live` (trader réseau) opt-in ; `portable-simd` (nightly) opt-in ; `wgpu` opt-in. **Cohérent.**

### 5.4 Documentation, configuration & reproductibilité

- Documentation massive : `README.md` (28 Ko), `Documentation*.md` (8 langues), `CHANGELOG.md` (132 Ko), `LIVESTATE.md` (87 Ko), `docs/` (roadmaps, MEMORY_WALL, TRANSPILER_DESIGN, kb/). Risque d'**obsolescence** de fichiers aussi volumineux.
- `SECURITY.md` (FR) honnête sur les limites (SNMPv1 en clair, modèle HMAC sans PKI/révocation) — mais inexact sur FFI/unsafe (voir §4.4).
- `rustfmt.toml` (options nightly) + `clippy.toml` + `RUSTFLAGS=-D warnings` en CI — **bonne discipline lint**.
- SBOM committé dans `docs/sbom/` — à traiter en **artefact de build** (non source) pour éviter la désynchronisation silencieuse avec `Cargo.lock`.

---

## 6. Points chauds identifiés par la cartographie

| Groupe | Point chaud | Type | Note |
|---|---|---|---|
| Racine | `src/main.rs` (openclaw-u) | cmd exec + auto-mutation + deser | Agent auto-modifiant, `cargo check` sur code généré, `state.json` non signé |
| Racine | `cliptest`, `cliptest2` | binaire committé / supply chain | ELF 4,4 Mo sans provenance |
| Racine | `scripts/test-protocol.sh:121` | `eval "$cmd"` | Variables internes uniquement, faible risque |
| Racine | `.github/workflows/release.yml` | permissions + actions tierces | `contents: write`, actions non SHA-pinnées |
| Verticaux | `scirust-grid/src/distance_relay.rs` | safety-correctness | Relais mho — faux négatif → défaut non détecté |
| Verticaux | `scirust-grid/src/state_estimation.rs` | safety-correctness | WLS + chi2 — faux déclenchement/aveuglement |
| Verticaux | `scirust-robotics/src/ssm.rs` | safety-correctness | ISO/TS 15066 — borne → contact humain-robot |
| Verticaux | `scirust-reliability/src/lib.rs` | safety-correctness | PFDavg/PFH IEC 61508 — base certification |
| Verticaux | `scirust-func-safety/src/evidence.rs` | intégrité | Chaîne FNV-1a non cryptographique (S7) |
| Verticaux | `scirust-func-safety/src/fault_injection.rs:118` | déterminisme | `rand::random` non seedé (S12) |
| Verticaux | `scirust-tolerance/src/{modal,chain,spatial}.rs` | unwrap/panic | Paniques sur batch dégénéré (S8) |
| Verticaux | `scirust-fusion/src/fusion.rs` | unwrap/panic | `fuse().expect()` (S8) |
| Verticaux | `scirust-graph/src/lib.rs` | deser + panic | `Serialize/Deserialize` + 15 unwrap |
| Verticaux | `scirust-som/crates/cli/src/lib.rs` | deser + unwrap | Parsing SARIF avec `expect`/`unwrap` |
| Core/Tensor | `scirust-runtime/src/enclave.rs` | unsafe FFI + OOB | `dims` non validé (S1) |
| Core/Tensor | `scirust-arena/src/{slab,aligned,allocator}.rs` | unsafe | Aligné 128, invariants documentés — sain |
| Core/Tensor | `scirust-core/src/io/safetensors.rs` | deser | Cap 16 MiB, rejet négatifs — sain (S10) |
| Agents | `scirust-mcp/src/tools/cli_passthrough.rs` | cmd exec | Expose toute la CLI au client MCP |
| Agents | `scirust-sciagent/src/agentic/tools.rs` | cmd exec + read | Lecture fichier arbitraire (agent) |
| Agents | `scirust-sciagent/src/bin/fetch-crates.rs` | supply chain | Tarballs sans checksum (S3) |
| OT/ICS | `scirust-discovery/src/protocols/{snmp,modbus}.rs` | réseau OT | Parsers bornés — sain ; SNMPv1 en clair (par conception) |
| OT/ICS | `scirust-discovery/src/scope.rs` | gate sécurité | HMAC signé, CIDR/protocole/temps, gate SL3+ — excellent |
| OT/ICS | `scirust-discovery/src/scope.rs:106` | crypto | Comparaison signature non constant-time (S6) |
| Applicatifs | `scirust-trader/src/wallet.rs` | crypto | Watch-only, constant-time, pas de clé — excellent |
| Applicatifs | `scirust-license/src/license.rs` | intégrité | Encodage canonique anti-injection, node-lock salé — excellent |

---

## 7. Recommandations prioritaires

### P0 — Aucune vulnérabilité critique confirmée.

### P1 — À corriger avant exposition réseau/TEE

1. **S1 — Valider `dims` dans `EnclaveRuntime::infer`** (`scirust-runtime/src/enclave.rs`) : ajouter les bornes `weights.len() ≥ out_features*in_features`, `input.len() ≥ batch*in_features`, `output.len() ≥ batch*out_features`, `bias.len() ≥ out_features` (si `has_bias`) avant l'appel `unsafe` ; retourner `Err(i32)` sinon. Ajouter des tests pour `dims` incohérents.
2. **S2 — Durcir le binaire `openclaw-u`** (`src/main.rs`) : signer `state.json` (HMAC ou signature), isoler le répertoire de sortie des fichiers générés (hors `src/`), valider le code généré avant `cargo check`.
3. **S3 — Vérifier l'intégrité des tarballs dans `fetch-crates`** (`scirust-sciagent/src/bin/fetch-crates.rs`) : comparer le SHA-256 du tarball téléchargé à celui de l'API crates.io ; valider les chemins extraits par `tar` (pas de `..`/absolus) ; isoler le répertoire d'extraction.

### P2 — Hygiène & conformité

4. **S4 — Pinner toutes les actions GitHub par SHA** (`ci.yml`, `release.yml`).
5. **S5 — Corriger `SECURITY.md`** : remplacer « zéro FFI » par « zéro bibliothèque C/C++ embarquée (FFI C *exportée* pour TEE uniquement) » ; retirer ou isoler `archive/` (FFI CUDA/SVE) ; préciser l'étendue réelle du `unsafe`.
6. **S6 — Homogénéiser la comparaison de signature** : rendre `ScopeAuthorization::signature_valid` constant-time (comme `WalletAuthorization::verify_signature`).
7. **S7 — Renforcer la chaîne d'evidence func-safety** : soit documenter explicitement que `EvidencePack` est *tamper-evident* et non *tamper-resistant* (l'attaquant qui connaît l'algorithme public peut recalculer une chaîne), soit intégrer un MAC (HMAC à clé) pour la rendre forgery-resistant.
8. **S8 — Remplacer les `unwrap/expect` par `Result`** dans `scirust-tolerance/{modal,chain,spatial}` et `scirust-fusion/fusion`.
9. **S9 — Retirer `cliptest`/`cliptest2`** du dépôt (ou documenter provenance + checksum).
10. **S10/S11 — Fuzzing** : fuzzer les parsers Modbus/SNMP/BER (`scirust-discovery`) et safetensors/ONNX (`scirust-core`, `scirust-onnx`) avec `cargo-fuzz` sur entrées malformées.
11. **S12 — Isoler `rand::random`** de `fault_injection.rs` hors des chemins certifiés (ou le seedé).

---

## 8. Annexes

### 8.1 Méthodologie

- Cartographie par groupe de domaine (8 groupes couvrant les 87 crates + racine/docs/CI/examples).
- Audit sécurité par dimension (7 : unsafe/FFI, OT/ICS, crypto, cmd exec, désérialisation, supply chain, sandbox/enclave).
- Audit qualité par dimension (4 : erreurs, tests, architecture, docs/config).
- Vérification adversariale : relecture du code pour confirmer l'atteignabilité et la sévérité de chaque finding.

### 8.2 Outils

`grep`, `find`, `Read` (lecture exhaustive des fichiers à risque), `cargo-deny` (configuration auditée dans `deny.toml`).

### 8.3 Limites

- Audit **statique** : pas de build, pas de tests exécutés, pas de fuzzing runtime.
- Le workflow multi-agent initial a été interrompu par rate-limit du modèle cloud ; 2 cartes de domaine (Racine/docs/CI, Verticaux industriels) ont été produites par subagents et validées ; les autres dimensions ont été auditées par lecture directe.
- Crates non lus exhaustivement : la majorité des 87 crates n'a pas été lue en entier — l'audit s'est concentré sur les fichiers signalés par `grep` (unsafe, réseau, crypto, cmd, deser) et les points chauds de sécurité. Une passe complémentaire par crate reste possible.
- Les verticaux « safety-correctness » (grid, robotics, reliability, bms, maritime, nav) n'ont pas subi de revue d'exactitude numérique — seul le risque structurel (paniques, déterminisme) a été évalué.