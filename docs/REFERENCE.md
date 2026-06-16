# SciRust — Référence des commandes et de l'API

Référence opérationnelle exhaustive du workspace : commandes, gates,
binaires, features, et points d'entrée d'API. La documentation
**exhaustive des fonctions** est générée par rustdoc (voir §7) — chaque
fonction publique y est documentée depuis le code source.

> Toolchains : le dépôt épingle **nightly** (rust-toolchain.toml) pour la
> feature optionnelle `portable-simd`, mais **tout le workspace compile et
> passe ses tests sur Rust STABLE** (686 tests vérifiés ; job CI
> `build-test-stable`). Pur Rust, zéro FFI, aucune dépendance système.

---

## 1. Gates qualité (identiques en local et en CI)

Ce sont les commandes exactes exécutées par `.github/workflows/ci.yml`.
Un changement n'est livrable que si les six passent :

| Gate | Commande | Vérifie |
|---|---|---|
| Format | `cargo fmt --all -- --check` | style rustfmt du workspace |
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` | zéro lint, code + tests + benches |
| Build | `cargo build --workspace --all-targets` | compilation complète |
| Tests | `cargo test --workspace` | toute la suite (650+ tests) |
| Multi-arch | `cargo check --workspace --all-targets --target aarch64-unknown-linux-gnu` | chemins NEON/SVE (`rustup target add aarch64-unknown-linux-gnu` une fois) |
| Licences/Sécurité | `cargo deny check` | advisories, licences, sources (`cargo install cargo-deny`) |

La CI exporte `RUSTFLAGS="-D warnings"` : tout warning est une erreur.
`--all-features` est volontairement proscrit : `blas-openblas` et
`blas-mkl` sont des backends mutuellement exclusifs de `blas-src`.

## 2. Binaires exécutables

### `scirust` — CLI unifiée (point d'entrée recommandé)

```bash
cargo install --path scirust-cli   # fournit le binaire `scirust`
scirust help                       # liste toutes les commandes
scirust quickstart                 # entraîne le classifieur démo (déterministe) → 4/4
scirust analyze <file.rs> [--sarif]
scirust verify emit|verify <args...>
scirust version
```

Dispatcher mince au-dessus de capacités déjà testées (aucun nouveau
calcul). Sans installation : `cargo run -p scirust-cli -- <commande>`.
Codes de sortie : 0 succès, 1 échec métier (faute/MISMATCH), 2 usage/IO.

| Commande | Effet | Adossé à |
|---|---|---|
| `quickstart` | entraîne le MLP 2→8→2 (XOR), bit-déterministe, 4/4 | `scirust-core` |
| `som train [--seed N] [--epochs E]` | entraîne le modèle d'ownership, accuracy vs baseline | `scirust-som-*` |
| `evo [--seed N] [--gens G]` | minimise la sphère par algorithme génétique seedé | `scirust-evo` |
| `cmaes [--seed N] [--steps S]` | minimise la sphère par CMA-ES seedé | `scirust-evo` |
| `diff <expr> [var]` | dérivée symbolique | `scirust-symbolic` |
| `simplify <expr>` | simplification algébrique | `scirust-symbolic` |
| `eval <expr> [x=..]` | évaluation numérique | `scirust-symbolic` |
| `solve <expr> [var]` | racines réelles symboliques (linéaire/quadratique) | `scirust-symbolic` |
| `prove <a> <b>` | preuve best-effort d'équivalence | `scirust-symbolic` |
| `gradient <expr> x=.. [y=..]` | gradient numérique (1–2 variables) | `scirust-symbolic` |
| `to-rust <expr>` | transpile une expression en Rust | `scirust-symbolic` |
| `regress <xs> <ys> [deg]` | régression moindres carrés (linéaire/polynomiale) | `scirust-symbolic` |
| `symreg <xs> <ys> [--seed N]` | découverte de loi close (programmation génétique) | `scirust-symreg` |
| `sat "c;c"` | satisfiabilité DPLL | `scirust-neuro-symbolic` |
| `trig <expr>` | identités trigonométriques + simplification | `scirust-symbolic` |
| `patterns "v1,v2,.."` | détection de tendance dans une série | `scirust-symbolic` |
| `integrate <expr> <a> <b> [var] [--method]` | intégrale définie (Romberg/Simpson/Gauss) | `scirust-solvers` |
| `root <expr> <a> <b> [var] [--method]` | racine (Brent/bisection/secant/newton) | `scirust-solvers`(+`-symbolic` pour newton) |
| `minimize <expr> <a> <b> [var]` | minimum local 1D (racine de la dérivée) | `scirust-solvers`+`-symbolic` |
| `optimize <expr> --start a,b --vars x,y` | minimum multi-D (Nelder–Mead) | `scirust-solvers`+`-symbolic` |
| `linsolve "r;r" "b"` | résout A·x=b (LU) | `scirust-solvers` |
| `lstsq "r;r;r" "b"` | moindres carrés A·x≈b (QR) | `scirust-solvers` |
| `det "r;r"` | déterminant | `scirust-solvers` |
| `cholesky "r;r"` | facteur L de Cholesky (SPD) | `scirust-solvers` |
| `qr "r;r"` | décomposition QR (Q, R) | `scirust-solvers` |
| `cg "r;r" "b"` | gradient conjugué (SPD, itératif) | `scirust-solvers` |
| `inverse "r;r"` | inverse d'une matrice carrée (LU) | `scirust-solvers` |
| `solve-system "f1;f2" --vars x,y --start a,b` | système non-linéaire F(x)=0 (Broyden) | `scirust-solvers`+`-symbolic` |
| `polyroots "c0,c1,.."` | racines réelles d'un polynôme | `scirust-solvers` |
| `ode <f(t,y)> <y0> <t0> <t1> [h] [--method]` | intègre dy/dt=f (RK4 / DOPRI5 adaptatif) | `scirust-solvers`+`-symbolic` |
| `fem-heat <nodes> <length> <source>` | chaleur 1D −u″=source (éléments finis linéaires) | `scirust-solvers` |
| `tt "r;r" [--factors d] [--max-rank r] [--tol t] [--max-err e]` | compression tensor-train (TT-SVD) d'une matrice | `scirust-tn` |
| `pinn [--seed N] [--steps S]` | réseau physics-informed : résout le BVP `u''=−u` (résidu de PDE dans la loss), vérifié vs `sin x` | `scirust-core::nn::pinn` |
| `bpe "<corpus>" [--vocab N] [--encode "<text>"] [--bytes]` | tokenizer BPE déterministe (entraînement + encode/decode ; `--bytes` = byte-level lossless) | `scirust-learning` |
| `lm ["t0,t1,.."] [--seed N] [--steps S] [--lr R] [--opt adam\|adamw\|lion\|schedule-free\|ademamix\|soap]` | entraîne un petit LM décodeur causal (tape N-D) à mémoriser une séquence de tokens | `scirust-core` |
| `deltanet [--seed N] [--steps S]` | entraîne une couche DeltaNet (attention linéaire à règle delta) à ajuster une séquence ; rapporte la baisse de MSE | `scirust-core::nn::nd_layers` |
| `mamba [--seed N] [--steps S]` | entraîne une couche Mamba (scan sélectif S6, état-espace) à ajuster une séquence ; rapporte la baisse de MSE | `scirust-core::nn::nd_layers` |
| `retnet [--seed N] [--steps S]` | entraîne une couche RetNet (rétention, attention linéaire récurrente ≡ parallèle) à ajuster une séquence | `scirust-core::nn::nd_layers` |
| `analyze <file.rs> [--sarif]` | analyse d'ownership de vrai Rust | `scirust-som-cli` |
| `verify emit\|verify <args>` | certificats d'inférence | `scirust_runtime::proofcli` |
| `certify [--seed N] [--eps E]` | bornes de sortie prouvées d'un MLP ReLU sur une boîte L∞ — **IBP** (couche par couche) **et CROWN** (relaxation linéaire, plus serrée) côte à côte | `scirust-core::nn::ibp` |
| `conformal [--seed N] [--alpha A]` | intervalles conformes à couverture garantie sans hypothèse de distribution | `scirust-core::nn::conformal` |
| `calibrate [--seed N]` | temperature scaling : ajuste `T` pour réduire l'erreur de calibration (ECE) sans changer l'accuracy | `scirust-core::nn::calibration` |
| `gptq [--seed N] [--samples S] [--damp D]` | quantification int8 GPTQ (feedback d'erreur d'ordre 2) ; affiche la réduction d'erreur de calibration vs round-to-nearest | `scirust-core::quantization` |
| `awq [--seed N] [--samples S] [--grid G]` | quantification int8 AWQ (scaling per-canal par recherche, conscient des activations) ; affiche l'`alpha` retenu et la réduction d'erreur vs round-to-nearest | `scirust-core::quantization` |
| `info` / `help` / `version` | méta | — |

Les binaires ci-dessous restent disponibles individuellement ; `scirust`
ne fait que les regrouper derrière une interface unique et découvrable.

### `som-analyze` — analyse d'ownership d'un fichier Rust réel

```bash
cargo run -p scirust-som-cli -- chemin/vers/fichier.rs
```

Parse le fichier avec la grammaire Rust réelle (`syn`), exécute l'oracle
d'ownership déterministe, affiche la table par token
(ownership/borrow/faute) et les diagnostics (use-after-move E0382,
conflits d'emprunt E0502, lecture sous `&mut` E0503, emprunt échappé,
non-déclarés). **Codes de sortie** : `0` aucun défaut, `1` ≥ 1 défaut
(utilisable comme check de script), `2` erreur d'usage/IO/syntaxe.
Exemples fournis : `scirust-som/examples/*.rs`.

### `scirust-verify` — certificats d'inférence vérifiables (preuve)

```bash
cargo run -p scirust-runtime --bin scirust_verify -- emit   model.qsr1 model.proof [batch] [seeds...]
cargo run -p scirust-runtime --bin scirust_verify -- verify model.proof model.qsr1
```

`emit` scelle un certificat canonique `SCIRUST-PROOF-1` (sha256 artefact,
certificat de ressources, empreintes FNV+sha256 des sorties sur entrées
seedées, après preuve d'égalité bit-exacte std/no_std). `verify` re-dérive
tout depuis les octets et sort 0 (MATCH) ou 1 (MISMATCH) — toute
altération de l'artefact ou du certificat est détectée (testé). La
ré-émission est bit-identique.

### `cargo som` / sortie SARIF — linter d'ownership pour la CI

```bash
cargo install --path scirust-som/crates/cli   # installe som-analyze + cargo-som
cargo som --sarif src/lib.rs > som.sarif       # SARIF 2.1.0 (code scanning)
```

Limite documentée : localisations au niveau fichier (les régions
ligne-précises arrivent avec les spans du frontend).

### `openclaw-u` — démo agent autonome (hors framework)

```bash
cargo run --bin openclaw-u
```

Binaire expérimental indépendant du framework (voir README racine).

### Binaires d'audit du runtime (`scirust-runtime/src/bin/`)

Chaque audit vérifie une garantie par exécution et oracle :

```bash
cargo run -p scirust-runtime --bin <nom>
```

`bench_latency` (latence bornée p99/p50), `bn_check`, `cnn_audit`,
`edge_oracle`, `eval_artifact`, `generic_check`, `layers_check`,
`neon_bench` (aarch64), `proof_bundle` (empreinte 64-bit reproductible),
`train_artifact` (SRT1), et la famille quantization int8 :
`quant_audit`, `quant_artifact_audit`, `quant_conv_audit`,
`quant_conv_int8_audit`, `quant_depthwise_audit`, `quant_fullint_audit`,
`quant_lib_audit`, `quant_pointwise_audit`, `quant_static_audit`.

### Exemples (packages du workspace)

```bash
cargo run -p quickstart_v2            # classif 2 classes en ~50 lignes
cargo run -p mnist_classifier         # MNIST réel (97,70 % mesuré)
cargo run -p cifar10_classifier
cargo run -p transformer_demo
cargo run -p transformer_compress
cargo run -p sentiment_demo
```

Hors workspace par défaut : `examples/benchmarks` (criterion),
`examples/simd_views_demo`, `scirust-burn-bridge`,
`scirust-rustc-driver` (nécessite `rustc-dev` ; `setup-rustc-dev.sh`).

## 3. Mesures et sondes reproductibles

```bash
# Métriques SOM (train 200 prog. seed 42, éval 50 prog. held-out seed 9042)
cargo test -p scirust-som-inference --release -- --ignored --nocapture
```

Les tests `#[ignore]` sont des sondes de mesure, jamais des gates.

## 4. Features Cargo

| Crate | Feature | Effet |
|---|---|---|
| scirust-core | `rayon` *(défaut)* | data-parallélisme CPU |
| scirust-core | `portable-simd` | kernels `std::simd` (nightly) |
| scirust-core | `blas-openblas` / `blas-mkl` | matmul via BLAS système — exclusifs, exigent la toolchain système |
| scirust-gpu | `wgpu` / `cuda` | **vides actuellement** : les kernels en `src/` sont archivés, non câblés (audit §5) |

## 5. Carte des crates (points d'entrée)

| Crate | Rôle | Entrée principale |
|---|---|---|
| `scirust` (racine) | façade `scirust::{core,simd,symbolic,learning,solvers}` | `src/lib.rs` |
| `scirust-core` | tenseur 2D, tape autodiff, couches NN, quant int8, data, AMP/DP/pruning/distributed | `autodiff::reverse::{Tape,Tensor,Var}`, `nn::*` |
| `scirust-simd` | kernels AVX2/SSE2/NEON + dispatch runtime | `dispatch::runtime_backend()` |
| `scirust-runtime` | inférence déterministe SRT1 + manifeste | `lib.rs` + bins d'audit |
| `scirust-solvers` | linalg, solveurs | `linalg::*` |
| `scirust-learning` | optim/contrôle/NLP pipeline | `nlp::sentiment` |
| `scirust-symbolic` | différentiation symbolique | `lib.rs` |
| `scirust-neuro-symbolic` | datalog, CSP, SAT/SMT, KG, prob. | `Reasoner` |
| `scirust-evo` | GA / CMA-ES / OpenES / NSGA-II seedés | `GeneticAlgorithm`, `Nsga2` |
| `scirust-tn` | Tensor-Train (réexporte `core::tn`) | `TTLinear` |

## 6. API SOM (référence rapide)

| Crate | Fonctions/types clés |
|---|---|
| `scirust-som-frontend` | `lower_str(&str) -> Result<Lowered, syn::Error>` — Rust réel → IR ; `Lowered{ast, unsupported, approximations}` |
| `scirust-som-pcg` | `ast::*` (IR), `PcgBuilder::build`, `Pcg::{to_dot,to_json}` |
| `scirust-som-symbolic` | `OwnershipOracle::analyze(&SomAst) -> Analysis` ; `type_is_copy(&Type)` ; `Analysis::{ownership_ids,borrow_ids,invalid_flags}` ; `FaultKind`, `TokenLabel`, constantes de classes |
| `scirust-som-tokenizer` | `StructuredTokenizer::{tokenize_ast, tokenize_ast_with_drops, tokenize_pcg}` ; `SomVocab::{encode, vocab_size}` ; `MAX_VARS` |
| `scirust-som-dataset` | `build_training_set(seed,n,max_len)` ; `ProgramGenerator` ; `TrainingSample` ; `DatasetBuilder` (PCG) |
| `scirust-som-model` | `SomModel::{new, forward, parameter_indices, sync}` ; `SomModelConfig` ; `SomLogits` |
| `scirust-som-trainer` | `train(&mut SomModel, &[TrainingSample], &TrainerConfig) -> TrainReport` |
| `scirust-som-inference` | `evaluate`, `ownership_majority_baseline`, `predict_program`, **`predict_rust_source(&mut SomModel, &str)`** ; `EvalReport`, `InferenceReport` |
| `scirust-som-visualizer` | `render_markdown(&Analysis) -> String` |

Sémantique typée de l'oracle (contrat) : voir `scirust-som/README.md` et
le rustdoc de `scirust-som-symbolic`.

## 7. Documentation exhaustive des fonctions (rustdoc)

```bash
cargo doc --workspace --no-deps --open
```

Génère la référence complète de **toutes les fonctions publiques** du
workspace, à partir des commentaires du code (la seule source qui ne peut
pas diverger du code). C'est la documentation de fonctions faisant foi ;
ce fichier n'en est que l'index opérationnel.

## 8. Documents du dépôt

| Document | Contenu |
|---|---|
| `README.md` | positionnement, quick start, statut des features |
| `scirust_complete_audit_report.md` | audit vérifié par exécution + mise à jour fiabilisation |
| `scirust-som/README.md` | pipeline SOM, sémantique typée, métriques mesurées, limites |
| `LIVESTATE.md` | journal de bord inter-sessions (état mesuré) |
| `docs/QUICKSTART.md`, `docs/MNIST.md`, `docs/ARCHITECTURE.md`, `docs/GPU.md` | guides |
| `paper/SciRust-technical-report*.md` | rapport technique (8 langues) |
