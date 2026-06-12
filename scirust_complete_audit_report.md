# RAPPORT COMPLET D'AUDIT — SciRust

**Framework Deep Learning / calcul scientifique en Rust pur**

- Date : 2026-06-12
- Repository : https://github.com/CHECKUPAUTO/scirust
- Référence auditée : `master` @ `02f38cc` (merge PR #48), vérifiée sur x86_64 Linux, toolchain `nightly-1.98.0` (2026-06-11)
- Méthode : audit **vérifié par exécution** (compilation, tests, lints, CI GitHub), pas une analyse spéculative.

---

## 1. RÉSUMÉ EXÉCUTIF

SciRust est un projet **réel et substantiel** : 61 505 lignes de Rust réparties
sur ~45 crates, avec un cœur (`scirust-core`, 29 652 lignes, 124 fichiers) qui
implémente effectivement un moteur tensoriel 2D, un autodiff reverse-mode sur
tape, une bibliothèque de couches complète (MLP, Conv2d, LSTM, Transformer,
ViT, GNN, Flash-Attention, TT-Linear), de la quantization int8 déterministe et
un runtime d'inférence bit-exact. La précision MNIST de 97,70 % annoncée est
documentée par un log d'exécution versionné (`mnist_run_evidence.log`).

**Verdict de la vérification à l'état initial (`02f38cc`) : ÉCHEC.**

| Gate (défini par le README du projet) | État à `02f38cc` | État après correctifs (cette session) |
|---|---|---|
| `cargo check --workspace --all-targets` | ❌ E0425 ×4, E0559 ×1 | ✅ exit 0 |
| `cargo clippy --workspace --lib -- -D warnings` | ❌ 10 violations / 5 crates | ✅ exit 0 |
| `cargo test --workspace` | ❌ ne compile pas | ✅ **630 passés, 0 échec, 18 ignorés** (116 suites) |
| `cargo fmt --check` | ❌ 58 écarts / 14 fichiers | ✅ exit 0 |
| CI GitHub Actions (master) | ❌ **6 runs / 6 en échec, jamais verte** | (à confirmer au prochain push) |

La branche `master` au moment de l'audit **ne compilait sur aucune
architecture** (x86_64 ET aarch64) à cause d'une résolution de conflit de merge
défectueuse introduite le jour même (merge `1c55c79` → PR #48). Détail en §8.
Tous les correctifs ont été appliqués dans cette session et l'intégralité des
gates repasse au vert.

---

## 2. MÉTHODOLOGIE DE VÉRIFICATION

Commandes exécutées (celles du contrat qualité affiché dans le README) :

```bash
cargo check  --workspace --all-targets
cargo clippy --workspace --lib -- -D warnings
cargo test   --workspace
cargo fmt    --check
```

Compléments : inspection de l'historique git (`git log`, `git show` pour
reconstituer l'origine de la régression), interrogation de l'API GitHub
Actions (états des 6 runs CI de `master`), comptage de lignes par crate,
lecture du code des crates stratégiques (gpu, som, neuro-symbolic, runtime).

Environnement : Linux x86_64, `nightly-1.98.0` (le projet exige nightly pour
`portable_simd` et `rustc_private`). 324 dépendances verrouillées dans
`Cargo.lock`. ~114 blocs `unsafe` (essentiellement intrinsics SIMD), avec
en-têtes de sûreté documentés dans `scirust-simd/src/dispatch.rs`.

---

## 3. STRUCTURE RÉELLE DES MODULES

Mesures effectives (lignes de code Rust, tests passés lors du run complet) :

| Crate | Lignes | Tests | État réel |
|---|---:|---:|---|
| **scirust-core** | 29 652 | 359 | ✅ Cœur réel : tensor, autodiff tape, nn complet, quant, data, AMP, DP-SGD, pruning, distributed, lazy/AOT, TN |
| scirust-solvers | 5 139 | 76 | ✅ Substantiel (linalg, ODE…) |
| scirust-runtime | 3 933 | 1 | ✅ Runtime d'inférence déterministe SRT1 + manifeste |
| scirust-simd | 2 976 | 22 | ✅ Kernels AVX2/SSE2/NEON + dispatch runtime (était **cassé**, réparé) |
| scirust-learning | 2 699 | 26 | ✅ Optimiseurs/contrôle |
| scirust-gpu | 2 654* | 0 | ⚠️ **Trompeur** — voir §5 : lib réelle = 67 lignes, le reste est non câblé |
| scirust-neuro-symbolic | 1 961 | 33 | ✅ Réel mais compact : datalog, CSP, SAT/SMT, KG, théorèmes, logique probabiliste |
| scirust-symbolic | 1 217 | 9 | ✅ Différentiation symbolique |
| scirust-fusion | 1 324 | 1 | 🟡 Prototype |
| scirust-arena | 911 | 11 | ✅ Allocateur aligné/slab (était **cassé**, réparé) |
| scirust-evo | 764 | 6 | ✅ GA / CMA-ES / OpenES (clippy réparé) |
| scirust-tn | 555 | 2 | ✅ Tensor-Train (réexporte core::tn) |
| scirust-symreg | 546 | 1+3 ign. | 🟡 Régression symbolique |
| scirust-autodiff | 537 | 6 | 🟡 AD expérimental — **doublon conceptuel** de core::autodiff |
| scirust-reasoning | 364 | 16 | 🟡 Preuves d'égalités, identités trigonométriques |
| scirust-nas | 376 | 3 | 🟡 NAS évolutionnaire simple |
| scirust-onnx | 331 | 1 | 🟡 Export minimal |
| scirust-som/* (8 crates) | ~680 | 5 | 🔴 **Squelette** — voir §6 |
| scirust-tensor-* (6 crates) | 26–205 | 11 | 🟡 Prototypes pédagogiques (TensorND, einsum, fusion, mini-runtime) |
| scirust-events-* (4 crates) | 28–151 | 3 | 🔴 Embryonnaire |
| scirust-edge / embedded / bridge / macros… | 75–360 | — | 🔴 Embryonnaire |
| scirust-rustc-driver | 295 | — | ⚠️ Hors workspace (exige `rustc-dev`) |

\* dont ~2 600 lignes **hors module tree** (code mort, §5).

**Lecture honnête** : le ratio est d'environ un tiers de crates réellement
développées et testées, pour deux tiers de prototypes ou squelettes. Le cœur
(core + simd + solvers + runtime + learning) est sérieux ; la périphérie est
de l'ambition déclarée.

---

## 4. LE CŒUR RÉEL : scirust-core

Contrairement à l'idée d'un « moteur tensoriel à construire », le moteur
existe et il est le centre de gravité du projet :

- **Tensor** : 2D row-major (`rows`/`cols`) — pas N-D génériques ; le N-D vit
  dans le prototype `scirust-tensor-core::TensorND` (non unifié avec core).
- **Autodiff** : tape reverse-mode dynamique (`autodiff/reverse.rs`), avec
  variantes `try_*` (migration assert→Result faite), data-parallélisme (1 tape
  par thread), mixed precision, schedulers, optimiseurs (SGD/Adam/AdamW/LAMB/RMSprop).
- **NN** : linear, conv2d (+transpose, im2col HPC), pooling, batch/layer norm,
  dropout, LSTM, embedding, RoPE, transformer (MHA, GQA, KV-cache),
  flash-attention, ViT, GNN, TT-Linear (contraction on-tape avec gradients à
  travers les cores), fused ops, PEFT, loss strictes.
- **Au-delà du DL** : quantization int8 déterministe, DP-SGD, pruning,
  checkpointing, DataLoader, distributed (all-reduce), XAI, logging,
  lazy graph + AOT, et même un module homomorphique.

C'est la matérialisation des « Phases 1-2-3 » que viserait une roadmap type :
**tensor + autodiff + début de compilation (lazy/fusion) sont déjà faits**, au
moins en 2D. La vraie dette du cœur est l'absence de tenseur N-D unifié.

---

## 5. ANALYSE GPU — ÉCART CLAIMS / CODE LE PLUS IMPORTANT

Constat vérifié dans `scirust-gpu` :

- `lib.rs` fait **67 lignes**. Les backends exposés `WgpuBackend` et
  `CudaBackend` retournent `vec![0.0; m*n]` — **des stubs qui renvoient des
  zéros**. Seul `CpuFallback` calcule un vrai matmul (naïf).
- Les fichiers riches (`wgpu_backend.rs` avec kernels WGSL, `cublas.rs` avec
  cuBLAS via `cudarc`, `gpu_tensor.rs`, `quantize.rs`…) ne sont **déclarés
  dans aucun `mod`** : ils ne font pas partie de la crate compilée.
- Les features `cuda = []` et `wgpu = []` sont **vides** ; aucune dépendance
  `wgpu` ni `cudarc` n'existe dans tout le workspace (vérifié sur tous les
  `Cargo.toml`). `cargo check -p scirust-gpu --features wgpu` « réussit »
  précisément parce que rien n'est branché.

Conséquence : les lignes du README « GPU forward (wgpu) ✅ Stable », « GPU
backward ✅ Stable », et le claim phare du rapport technique (« ~63 TFLOPS
BF16 via cuBLAS sur Jetson Thor ») ne sont **pas reproductibles depuis le
workspace actuel**. Le code qui a pu produire cette mesure existe dans le
dépôt mais est débranché. C'est le point n°1 à corriger en termes de
crédibilité : soit recâbler (`mod` + deps optionnelles réelles), soit
requalifier les claims en « archivé / non câblé ».

---

## 6. ANALYSE SOM (SciRust Ownership Model)

Le concept (prédire ownership/borrow/lifetime à partir d'un Place Capability
Graph) est la partie la plus originale du projet. État réel des 8 crates :

| Crate SOM | Lignes | Réalité |
|---|---:|---|
| pcg | 366 | ✅ AST jouet + graphe PCG (nœuds Variable/Memory/Function/Region, arêtes Owns/Borrows/MutBorrows/Moves/Aliases/Drops), 3 tests |
| dataset | 135 | 🟡 Génération d'exemples |
| tokenizer | 108 | ✅ Tokenisation structurelle AST+PCG, 1 test |
| model | 70 | ⚠️ Étiqueté « Graph Transformer backbone » mais c'est un **MLP Linear+ReLU** : aucune attention, aucune structure de graphe |
| trainer / inference / symbolic / visualizer | 1 chacune | 🔴 Crates vides (un commentaire) |

Les messages de commit « SOM Phases 1-5 implémentées avec Graph Transformer »
**surévaluent l'état réel**. Le pipeline bout-en-bout (vrai parsing Rust via
`scirust-rustc-driver`, vrai transformer — qui existe pourtant dans
`core::nn::transformer` —, entraînement, inférence) reste à construire.
Potentiel réel, exécution ~10 %.

---

## 7. SYMBOLIQUE, NEURO-SYMBOLIQUE, RAISONNEMENT

- `scirust-symbolic` (1 217 l., 9 tests) : différentiation symbolique réelle,
  utilisée par la régression symbolique (claim du paper validé par les tests).
- `scirust-neuro-symbolic` (1 961 l., 33 tests) : modules datalog, règles,
  CSP, SAT/SMT, knowledge graph, théorèmes, logique probabiliste/causale.
  Réel, testé, mais de l'ordre du moteur d'enseignement, pas d'un solveur
  industriel.
- `scirust-reasoning` (364 l., 16 tests) : preuves d'égalité et identités.

L'axe « neuro-symbolique différenciant » du projet est donc **amorcé et
honnête en code**, mais sans pont réel avec les réseaux du core (le couplage
neural↔symbolique reste superficiel).

---

## 8. BUGS TROUVÉS ET CORRIGÉS DANS CETTE SESSION

### 8.1 Régression bloquante : master ne compilait plus (E0425 ×4 + E0559)

Origine reconstituée par git : le commit `71f9edf` (« workspace Clippy
cleanup ») avait réécrit les boucles `sgemv` en forme itérateur
(`for (i, item) in y.iter_mut().enumerate()`); le merge `1c55c79`
(« Merge branch 'master' into som-phase-1-2…», 2026-06-12) a résolu le
conflit en gardant **l'ancien en-tête de boucle avec le nouveau corps** :

```rust
for i in 0..m {                       // en-tête pré-clippy
    ...
    *item = alpha * dot + beta * *item;   // corps post-clippy → `item` n'existe pas
}
```

Touchés : les **trois** backends SIMD (`Avx2`, `Sse2`, `Neon`) dans
`scirust-simd/src/dispatch.rs` → compilation impossible sur x86_64 **et**
aarch64. Même mécanique dans `scirust-arena/src/slab.rs` (champ `_size:` au
site de construction vs `size` dans l'enum → E0559). Correctifs : restauration
de la forme itérateur d'origine (3 sites) et du nom de champ.

**Leçon process** : la machine de dev étant aarch64 (Jetson), le code x86_64
était invisible localement ; et comme la CI n'a jamais été verte (§9), rien
n'a bloqué le merge.

### 8.2 Gate clippy (`-D warnings`) : 10 violations réparées

- `scirust-evo` : 2× `ptr_arg` (`&mut Vec<f64>` → `&mut [f64]`), 2×
  `needless_range_loop` (formes itérateur restaurées) ;
- `scirust-core` : `needless_range_loop` (backward TT, `reverse.rs:2348`),
  `manual_div_ceil` (`dataloader.rs:147`) ;
- `scirust-nas` : cast `usize as usize` inutile ;
- `scirust-som-tokenizer` : `new_without_default` (impl `Default` ajoutée) ;
- `scirust-neuro-symbolic` : `needless_range_loop` sur l'élimination
  gaussienne — `#[allow]` ciblé et justifié (emprunts simultanés de deux
  lignes de la matrice), conforme à la pratique existante du dépôt.

### 8.3 Formatage

`cargo fmt` appliqué : 20 fichiers normalisés (58 écarts initiaux, surtout
les crates SOM et des résidus du merge).

### 8.4 Après correctifs

`check` ✅ · `clippy -D warnings` ✅ · `test` ✅ (630/630, 18 ignorés —
LIVESTATE.md annonce encore « 334 tests », chiffre obsolète) · `fmt --check` ✅.

---

## 9. CI, PROCESS ET GOUVERNANCE — LE VRAI POINT FAIBLE

Faits vérifiés via l'API GitHub Actions :

- **6 runs CI sur master, 6 échecs — la CI n'a jamais été verte.**
- Dernier run (#9, `02f38cc`) : **les 5 jobs en échec** (Format Check, Clippy,
  Build & Test, License & Security Audit/cargo-deny, Code Coverage).
- Les échecs précèdent la régression du merge : la config elle-même est
  vraisemblablement irréalisable en l'état (`--all-features` active
  `blas-openblas`/`blas-mkl` qui exigent des toolchains système absentes du
  runner ; coverage et deny jamais passés).
- Pas de protection de branche effective : un merge cassant tout le workspace
  est arrivé sur master le jour même.
- Incohérences d'hygiène : `Cargo.lock` versionné **mais** listé dans
  `.gitignore` ; LIVESTATE.md périmé ; binaire expérimental `openclaw-u`
  (agent autonome) embarqué dans le package principal (clairement étiqueté,
  mais du périmètre en plus).

C'est ici que le projet perd le plus de valeur : **le filet de sécurité
affiché n'a jamais fonctionné**, et les documents de claims (README « ✅
Stable », messages de commit « Phases 1-5 ») devancent le code.

---

## 10. CONFRONTATION AVEC LE RAPPORT SPÉCULATIF INITIAL

Le brouillon d'audit fourni (architecture « probable », états « attendus »)
est corrigé point par point par les faits :

| Affirmation du brouillon | Réalité vérifiée |
|---|---|
| « scirust-tensor : moteur tensoriel, CRITIQUE » | Le moteur vit dans **scirust-core** (Tensor 2D + tape). Les `scirust-tensor-*` sont 6 petits prototypes (26–205 l.) |
| « scirust-autodiff : la différentiation automatique » | L'autodiff de production est `core::autodiff` (tape reverse). `scirust-autodiff` (537 l.) est un doublon expérimental à clarifier |
| « GPU : ajouter CUDA/Vulkan/Metal » | Des kernels cuBLAS et WGSL **existent déjà** dans le dépôt… mais débranchés (code mort, features vides) — §5 |
| « Il manque une couche IR » | Partiellement faux : lazy graph (`core/lazy`), AOT (`core/aot.rs`) et fusion d'opérateurs (`scirust-tensor-compile`) existent en germe. Rien d'un MLIR, mais la brique n'est pas absente |
| « Ajouter benchmarks/ » | `examples/benchmarks` existe (exclu du build par défaut) + `bench_results_v11_2.txt` versionné + benches criterion dans neuro-symbolic |
| « SOM : potentiel d'IA d'analyse de code » | Concept présent (PCG/tokenizer ✅) mais 4 crates sur 8 vides et « Graph Transformer » = MLP — §6 |
| « Risque : trop de domaines, beaucoup de prototypes » | **Confirmé et mesuré** : ~2/3 des crates < 500 lignes ; events-*, edge, bridge, embedded embryonnaires |
| « Phase 1-2 (tensor, autodiff) à construire » | **Déjà construites et testées** (630 tests). La roadmap réelle commence plus loin — §11 |

---

## 11. RISQUES ET RECOMMANDATIONS PRIORISÉES

1. **(Fait dans cette session)** Réparer master : compilation, clippy, fmt,
   630 tests verts.
2. **Rendre la CI verte et la verrouiller** — c'est LA priorité absolue :
   - matrice de build x86_64 **et** aarch64 (le bug du jour était
     arch-dépendant et indétectable depuis le Jetson seul) ;
   - remplacer `--all-features` par des combinaisons réalistes (les features
     BLAS exigent des prérequis système : les installer ou les sortir du job) ;
   - protection de branche : merge interdit sans CI verte.
3. **Vérité des claims** : recâbler `scirust-gpu` (déclarer les `mod`, ajouter
   `wgpu`/`cudarc` en dépendances optionnelles réelles, CI feature-gated) ou
   requalifier README/paper. Un claim « 63 TFLOPS » non reproductible depuis
   le build coûte plus cher en crédibilité qu'il ne rapporte.
4. **Un seul autodiff** : fusionner ou requalifier `scirust-autodiff` vs
   `core::autodiff`.
5. **Réduire la surface** : archiver ou marquer `experimental` les crates
   squelettes (events-*, edge, bridge, tensor-examples, embedded…) ; le
   workspace y gagnera en lisibilité et en temps de CI.
6. **SOM** : brancher le modèle sur `core::nn::transformer` (l'attention
   existe déjà !), implémenter trainer/inference, connecter
   `scirust-rustc-driver` pour du vrai code Rust → PCG. Sinon, requalifier
   les commits « Phases 1-5 ».
7. **Tenseur N-D** : unifier `TensorND` (prototype) avec le Tensor 2D du core —
   c'est la vraie dette du cœur, prérequis des ambitions compilateur.
8. **Hygiène** : retirer `Cargo.lock` du `.gitignore` (il est versionné),
   mettre à jour LIVESTATE.md (334 → 630 tests), tenir le tableau README
   synchronisé avec le code.

Roadmap corrigée (l'ancienne « Phase 1 : créer un Tensor stable » est
obsolète — c'est fait) : **(1)** CI verte multi-arch verrouillée → **(2)**
GPU recâblé et mesuré en CI → **(3)** Tensor N-D unifié → **(4)** fusion/IR
sur cette base → **(5)** SOM réel (transformer + rustc-driver) → **(6)**
pont neuro-symbolique ↔ réseaux du core.

---

## 12. ÉVALUATION GLOBALE RÉVISÉE

| Axe | Note | Justification mesurée |
|---|---|---|
| Cœur DL (tensor/autodiff/nn/quant/runtime) | **8 / 10** | 630 tests verts, MNIST 97,70 % documenté, déterminisme bit-exact, int8 sérieux |
| Architecture workspace | **6,5 / 10** | Bonne modularité par crates, mais doublons (autodiff, tensor) et ~2/3 de squelettes |
| Process / CI / gouvernance | **3 / 10** | CI jamais verte (6/6 échecs), merge cassant arrivé sur master, docs d'état périmées |
| Véracité documentation vs code | **5 / 10** | Cœur honnête et sourcé ; GPU et SOM survendus (code mort / stubs à zéros) |
| Innovation / différenciation | **8 / 10** | Déterminisme auditables + quant int8 bit-exact + symbolique + concept SOM : réellement différenciant |
| Faisabilité de la vision complète | **6 / 10** | Le cœur prouve la capacité d'exécution ; la dispersion la menace |

**Conclusion.** SciRust n'est ni un vaporware ni le « PyTorch+MLIR+LLVM en
Rust » de sa vision maximale : c'est un **cœur deep-learning pur Rust réel,
testé et différencié** (déterminisme, quantization, symbolique), entouré
d'une périphérie d'ambitions inégalement matérialisées, et fragilisé par un
process qui a laissé master cassé avec une CI jamais verte. La valeur
stratégique reste celle d'une plateforme IA compilée et auditable de bout en
bout — mais elle se gagnera moins en ouvrant de nouveaux chantiers qu'en
**verrouillant la qualité (CI multi-arch), en recâblant le GPU, et en
alignant chaque claim sur du code exécutable**. Le cœur est solide ; c'est la
discipline autour du cœur qui décidera du reste.

---

*Audit réalisé par exécution réelle des gates du projet le 2026-06-12.
Correctifs associés livrés sur la branche `claude/great-pascal-5bmfcw`.*

---

# MISE À JOUR FIABILISATION — 2026-06-12 (fin de journée)

État vérifié par exécution après les travaux de fiabilisation menés sur la
branche `claude/great-pascal-5bmfcw` (commits `bdf4f3e` → ce commit).

## A. Gates : tous verts, et reproductibles en CI

| Gate | État | Note |
|---|---|---|
| `cargo fmt --all -- --check` | ✅ | |
| `cargo clippy --workspace --all-targets -- -D warnings` | ✅ | élargi de `--lib` à `--all-targets` |
| `cargo build` + `cargo test --workspace` | ✅ | **672 tests, 0 échec, 19 ignorés** |
| `cargo check --target aarch64-unknown-linux-gnu --all-targets` | ✅ | nouveau gate multi-arch |
| `cargo deny check` | ✅ | deny.toml réécrit (l'ancien était invalide), validé cargo-deny 0.19.8 |
| `cargo doc --workspace --no-deps` | ✅ **0 warning** | 22 warnings rustdoc corrigés |

La CI (`.github/workflows/ci.yml`) exécute exactement ces commandes ;
`--all-features` (impossible à construire : blas-openblas ⊕ blas-mkl) a
été retiré et documenté. Restant côté GitHub : protection de branche.

## B. SOM : du langage jouet au Rust réel, sémantique typée

- **Frontend `syn`** (`scirust-som-frontend`) : vrais fichiers `.rs` →
  IR ; constructions non couvertes **signalées**, jamais devinées.
- **CLI `som-analyze <file.rs>`** : table ownership/borrow/faute par
  token, diagnostics (E0382/E0502/E0503-style), exit 1 si faute.
- **Oracle type-aware** : sémantique Copy/move exacte sur le vocabulaire
  de types de l'IR (`i32`/`f64`/`bool`/`*T`/`&T` copient ;
  `String`/inconnus/`&mut T` déplacent), inférence locale des bindings
  non annotés, faute « lecture sous `&mut` » ajoutée. La
  sur-signalisation des types Copy relevée dans la v1 est **corrigée et
  testée** (double usage d'`i32` légal, hérité sans annotation).
- Métriques re-mesurées (held-out seed 9042, 850 tokens) : ownership
  **87,3 %** (baseline 33,1 %), borrow 94,0 %, fautes 88,6 %.
- Limites restantes, documentées : emprunts lexicaux (pas NLL), code
  rectiligne (pas de branches), `let x = f();` non annoté = move
  conservateur. Levée complète = chemin `scirust-rustc-driver`.

## C. Véracité documentaire : README aligné sur le code

Les claims GPU du README racine (« GPU forward/backward ✅ Stable »,
« tiled WGSL compute », « 63 TFLOPS ») sont requalifiés **Archived — not
wired**, avec renvoi au §5 du présent rapport. Le statut redevient
conforme à la philosophie « claims backed by measurements ».

## D. Documentation exhaustive

- `docs/REFERENCE.md` (nouveau) : référence opérationnelle complète —
  6 gates, tous les binaires (som-analyze, openclaw-u, 19 audits
  runtime, 6 exemples), features, carte des crates, API SOM, sondes.
- `rustdoc` : 0 warning ; `cargo doc --workspace --no-deps --open` est
  la référence de fonctions faisant foi.
- `scirust-som/README.md` : contrat sémantique typé + métriques à jour.

## E. Risques résiduels (inchangés, à traiter ensuite)

1. GPU : recâblage réel (mod + deps optionnelles) ou suppression des
   sources archivées — décision produit à prendre.
2. Protection de branche master (réglage GitHub, non scriptable d'ici).
3. Doublon `scirust-autodiff` vs `core::autodiff` ; crates squelettes
   (`events-*`, `edge`, `bridge`…) à archiver ou marquer expérimentales.
4. SOM : NLL/branches via rustc-driver ; attention sur graphe PCG ;
   persistance des poids (SRT1).

FIN DE LA MISE À JOUR
