# Changelog

Le format suit [Keep a Changelog](https://keepachangelog.com/) ;
versions sémantiques à partir de la prochaine release taguée.

## [Non publié]

### Ajouté — campagne « faire grandir scirust »
- **LLM bout-en-bout** : décodage KV-cache O(n) (`MiniLLM::generate_ids_cached`,
  `TransformerBlock/Encoder::infer_step`, `PositionalEncoding::encoding_at`)
  **prouvé équivalent** au recalcul complet ; génération découplée du tokenizer
  (`MiniLLM::generate_ids`) → un BPE peut piloter la génération (test
  d'intégration dans `scirust-learning`). Décodage glouton (sampling à venir).
- **CLI `bpe`** : entraîne un tokenizer BPE déterministe sur un corpus
  (documents séparés par `;`), encode/decode, rapporte la taille de vocab et le
  round-trip. Adossé à `scirust-learning` (38 → 39 commandes ; nouveau groupe
  NLP).
- **Matmul par lots N-D** (`NdVar::bmm`) : `(…,m,k)·(…,k,n)→(…,m,n)` avec axes
  batch broadcastés — la capacité que la tape 2D ne sait pas exprimer
  (scores d'attention par tête). Forward + backward gradient-checkés.
- **Autograd N-D (MVP, P2.4)** : `autodiff::nd` — `NdTape`/`NdVar` sur
  `TensorND` (add/mul broadcastés, matmul 2D, relu, sum), à côté de la tape 2D
  de production. Validé par un **gradient check numérique** (différences
  finies vs backward) sur `sum(relu(X·W+b)·V)`.
- **Ops GPU élargies** : kernel elementwise wgpu (add/mul/relu) ; une couche
  entière (matmul → +biais → relu) reste **résidente en VRAM**, validée contre
  l'oracle CPU sur lavapipe.
- **ONNX import** : `import_onnx_json` + `OnnxGraph::weights` — les poids
  font un aller-retour export→import **bit-exact** (format de checkpoint).
- **KV-cache vérifié** : test prouvant que le décodage incrémental
  (`MultiHeadAttention::infer_step`) donne le même dernier token que le forward
  complet — décodage O(n) désormais testé.
- **BPE déterministe** : tie-break par paire (`(count, Reverse(pair))`) — le
  `max_by_key(count)` dépendait de l'ordre d'itération du HashMap ; +5 tests.

### Réparé
- **Revue de code (max-effort) — durcissement** : (1) chemin GPU résident
  (`GpuChain`) : les dimensions dégénérées (`m`/`n`/`k == 0`) faisaient paniquer
  wgpu (buffers de taille nulle) ; gardes ajoutées (placeholder 4 octets,
  dispatch sauté, `download` court-circuité) + test. (2) `scirust ode` :
  `h = 0` provoquait un dépassement de capacité (panique, code 101), `t1 ≤ t0`
  renvoyait silencieusement `y0` (code 0, réponse fausse) et dopri5/rk4
  divergeaient sur de mauvaises bornes ; garde unifiée (`t1 > t0`, `h > 0` fini
  → code 2) + tests. Les autres axes de revue (maths GEMM/transpose, routage
  des gradients Conv2d, `matmul_gpu` av/ar, déterminisme de la réduction
  threadée, restructuration cfg SIMD) ont été tracés à la main : corrects.
- **`scirust-rustc-driver` recompile (P2.3, infra)** : le driver (exclu du
  workspace, `rustc_private`) ne compilait plus sur la nightly courante
  (`get_attrs` renvoie un itérateur, plus un slice). Réparé + warning-clean ;
  job CI informatif `rustc-driver` (continue-on-error) pour rendre la dérive
  d'API future visible ; `scirust-rustc-driver/target/` retiré du suivi git
  (artefacts de build) et ignoré.

### Ajouté
- **Primitives d'inférence de forme N-D (P2.4, fondation)** : `TensorND`
  gagne `broadcast_shape`, `matmul_shape` (matmul batché, broadcast des axes
  batch) et `broadcast_to` (matérialisation numpy) — les briques d'inférence
  de forme « au-delà de rows/cols » que la future tape/IR N-D utilisera, avec
  le pont `from/to_tensor_2d` existant. 3 tests. (La fusion de la tape 2D
  elle-même reste le gros chantier, à faire par incréments testés.)
- **Entraînement data-parallèle à déterminisme certifié (P2.1)** :
  `DataParallelTrainer::train_batch_threaded(n_threads, ..)` exécute les
  workers sur N threads OS (vol de tâches via compteur atomique) mais réduit
  les gradients dans un ordre fixe (worker 0,1,…,n-1), indépendant de
  l'ordonnanceur. L'addition flottante n'étant pas associative, le résultat
  est **bit-identique pour 1/2/4/8 threads** et identique au séquentiel —
  garantie testée que les frameworks grand public n'offrent pas. Trois tests
  CI : contributions sensibles à l'ordre (±1e16), vrai backward autograd, et
  une **boucle SGD multi-pas complète** dont la trajectoire de poids est
  bit-identique pour 1/2/4 threads (l'invariance se compose sur l'entraînement).

## [0.14.0] — 2026-06-13

### Réparé
- **`scirust-gpu` honnête (P2.2, étape « trancher »)** : les backends
  `WgpuBackend`/`CudaBackend` renvoyaient `vec![0.0; m*n]` — des résultats
  **fabriqués** (zéros) sous une étiquette « wgpu »/« cuda », en violation
  de la politique « 100 % câblé/testé, zéro sur-promesse ». Remplacés par
  un vrai backend CPU de référence **testé** (oracle GEMM bit-déterministe)
  et des chemins device qui signalent honnêtement `BackendError::Unavailable`
  (jamais de sortie inventée), à l'image de `scirust_core::compute_backend`.
  Crate passée de 0 à 6 tests. (Le câblage wgpu réel a suivi dans une étape
  séparée — voir « Ajouté » : GEMM WGSL testé sur Vulkan logiciel.)
- **`docs/GPU.md` honnête** : la page décrivait une API GPU en une ligne
  (`GpuContext::try_init`, `ConvGpuPipelines`, `Conv2d::on_gpu`…) qui
  n'existe pas (modules archivés ; `--features wgpu` ne compile rien).
  Réécrite en page de statut + roadmap honnête (ce qui existe = backend CPU
  de référence testé ; pourquoi le GPU n'est pas revendiqué ; plan P2.2).
- Régression de merge cassant la compilation sur toutes architectures
  (sgemv AVX2/SSE2/NEON, champ slab arena).
- CI rendue réalisable : retrait de `--all-features` (features BLAS
  mutuellement exclusives), `deny.toml` réécrit (TOML invalide),
  cross-check aarch64 ajouté ; 6 gates verts localement.
- Fusion d'opérateurs du graphe lazy : les chaînes pointwise fusionnent
  réellement (chaque maillon devenait sa propre chaîne de longueur 1).
- `RandomCrop` écrivait son résultat dans le vide (no-op silencieux).
- 22 warnings rustdoc ; warnings rustc/clippy ramenés à zéro
  (`-D warnings` tenable sur tous les targets).

### Changé
- **Statut GPU** retiré du tableau des features livrées du README (il
  listait du non-câblé) → remplacé par une note honnête « Not included
  yet » pointant la roadmap P2.2.
- **Augmentation de données 100 % déterministe** : RNG `PcgEngine`
  injecté, flux par échantillon indépendant de l'ordre, `with_seed`
  effectif, vrai bruit gaussien (Box-Muller).
- README aligné sur le code : statut GPU requalifié « Archived — not
  wired », compte de tests mesuré.
- `publish = false` sur les 51 manifestes (deps par chemin, licence
  non commerciale).

### Ajouté
- **GPU wgpu réel et testé (P2.2, étape « recâbler »)** : vrai GEMM `f32`
  en WGSL (`C = A·B`) derrière la feature `wgpu`, exécuté sur adaptateur
  Vulkan/Metal/DX12/GL via wgpu 0.20. **Validé contre l'oracle CPU**
  (tolérance flottante documentée, l'accumulation GPU n'étant pas
  bit-identique) et **testé en CI** sur Vulkan logiciel Mesa lavapipe
  (`llvmpipe`) — aucun GPU matériel requis, « pas de claim sans test »
  respecté. `cargo deny` passe sur l'arbre de deps wgpu ; dépendance
  optionnelle (les 8 gates par défaut ne la compilent pas). Nouveau job CI
  `GPU (wgpu / lavapipe)`.
- **GPU wgpu branché dans la tape autograd (P2.2, étape « tape »)** :
  `WgpuEngine` implémente le hook `GpuEngine` du `Tape` (kernel GEMM
  général `C = α·op(A)·op(B) + β·C` avec transposition). `Var::matmul_gpu`
  exécute **forward ET backward** (`dA = g·Bᵀ`, `dB = Aᵀ·g`) sur le GPU,
  device/pipeline mis en cache, repli CPU si un dispatch échoue. Validé
  bout-en-bout contre la tape CPU (forward + 2 gradients, tolérance) sur
  lavapipe. Opt-in (feature + `matmul_gpu`) → garantie bit-exacte par
  défaut intacte.
- **Conv2d GPU (P2.2, étape « Conv2d »)** : les GEMM im2col de Conv2d
  (forward `W·col`, backward `dW = dout·colᵀ` et `dInput = Wᵀ·dout`) passent
  par l'engine via le nouvel helper `Tape::gemm_ab` (chemin transpose natif),
  quand un `WgpuEngine` est attaché. Validé bout-en-bout contre la Conv2d CPU
  sur lavapipe (forward + dInput + dWeight, tolérance). Repli CPU
  bit-identique sans engine (aucune régression). im2col/col2im restent CPU.
- **Activations résidentes en VRAM (P2.2, étape « résidence »)** : API
  `GpuChain` — upload des entrées une fois, chaîne de `matmul` sur des
  handles `GpuMatrix`, un intermédiaire reste en mémoire GPU et alimente le
  GEMM suivant sans aller-retour CPU ; seul le résultat final est téléchargé.
  Validé contre l'oracle CPU sur lavapipe (chaîne 2 GEMM + transpose). La
  résidence transparente dans la tape (DeviceTensor matérialisé paresseusement
  en GPU) reste un chantier futur — sans bénéfice mesurable hors GPU matériel.
- **SBOM CycloneDX + automatisation de release** : SBOM CycloneDX 1.5
  reproductible (`docs/sbom/scirust.cdx.json`, horodatage figé via
  `SOURCE_DATE_EPOCH`, sans serial aléatoire → octet-identique pour une
  source donnée), généré par `./scripts/generate-sbom.sh`. Nouveau job CI
  `sbom` (artefact à chaque build) et workflow `release.yml` (sur tag `v*` :
  rejoue les gates, génère le SBOM, crée la release et y attache le SBOM).
  Section SBOM dans `SECURITY.md`, `docs/sbom/README.md` (provenance).
- **CLI : 5e vague** — `tt` (compression tensor-train TT-SVD d'une matrice,
  `scirust-tn` ; rapporte cœurs, rangs de liaison, ratio de compression et
  erreur de reconstruction, sortie 1 si `--max-err` dépassé), `solve-system`
  (système non-linéaire F(x)=0 par Broyden, `scirust-solvers`), `inverse`
  (inverse de matrice LU), `fem-heat` (chaleur 1D −u″=source par éléments
  finis linéaires), et méthode `dopri5` (Dormand–Prince adaptatif) pour `ode`.
  `FemSolver1D` était non testé : 2 tests ajoutés (oracle parabolique
  −u″=f exact aux nœuds + symétrie). Nouveau groupe TENSOR NETWORKS.
  `reconstruct_matrix` réexporté depuis `scirust-tn` (paire de
  `tt_decompose_matrix`). `newton_system` non exposé (closure `Fn(&[Dual])`
  comme `bfgs`).
- **CLI : 4e vague** — `trig` (identités trigonométriques), `patterns`
  (tendance d'une série), `qr` (décomposition QR), `cg` (gradient
  conjugué SPD). `bfgs` délibérément non exposé (closure `Fn(&[Dual])`
  non constructible depuis une expression symbolique évaluée en f64).
- **CLI : 3e vague** — `symreg` (régression symbolique par programmation
  génétique, `scirust-symreg`), `sat` (satisfiabilité DPLL,
  `scirust-neuro-symbolic`), et deux méthodes de plus pour `root`
  (`secant`, `newton` via dérivée symbolique). Nouveau groupe LOGIC.
- **CLI : 2e vague de commandes** (29 → toutes testées) : `integrate
  --method simpson|gauss`, `root --method bisection`, `optimize`
  (Nelder–Mead multi-variable), `lstsq` (moindres carrés QR), `cholesky`,
  `prove` (équivalence symbolique), `gradient` (numérique 1–2 var). Les
  commandes à expression réutilisent `scirust-symbolic::eval`.
- **CLI massivement étoffée** (19 commandes, toutes adossées à du code
  testé) : ajout de `cmaes` ; maths symboliques `to-rust`, `regress` ;
  solveurs numériques `integrate` (Romberg), `root`/`minimize` (Brent,
  via dérivée symbolique), `linsolve`/`det` (LU), `polyroots`,
  `ode` (RK4). Les commandes pilotées par expression utilisent
  `scirust-symbolic::eval` comme pont vers les solveurs `scirust-solvers`.
  +10 tests CLI ; bug d'ordre (intercept,slope) de `regress` corrigé et
  épinglé par un test.
- **CLI `scirust` étoffée** (niveau industriel) : nouvelles commandes
  groupées et documentées — `som train` (modèle d'ownership, accuracy vs
  baseline), `evo` (optimiseur génétique seedé), `diff`/`simplify`/`eval`/
  `solve` (maths symboliques), `info` (garanties). `scirust help` les
  liste par thème. Chaque commande est adossée à du code déjà testé.
- **Flash Attention réellement testé** : 4 tests dans
  `nn/transformer/flash_attention.rs` (forward vs oracle d'attention
  dense, masque causal, déterminisme bit-exact, gradients finis) — la
  ligne de statut passe de revendiquée à vérifiée.
- **CLI unifiée `scirust`** (`scirust-cli`) : point d'entrée unique et
  découvrable (`scirust help`) regroupant `quickstart` (démo MLP 2→8→2
  bit-déterministe, 4/4), `analyze` (ownership, délègue à som-cli),
  `verify` (certificats, délègue à `proofcli`), `version`. Logique verify
  factorisée dans `scirust_runtime::proofcli` (zéro duplication ;
  `scirust-verify` délègue désormais). Quickstart du README réécrit
  autour de la CLI (plus de copier-coller de 40 lignes d'API), exemple
  bibliothèque corrigé pour l'API réelle.
- **Support Rust stable** : `#![feature(portable_simd)]` rendu réellement
  optionnel (`cfg_attr`), fallback scalaire du tiling ; les 683 tests
  passent sur stable ; job CI `build-test-stable`. La feature nightly
  `portable-simd` (cassée par la migration d'API std::simd) est réparée.
- **`scirust-verify`** : certificats d'inférence `SCIRUST-PROOF-1`
  fichier-à-fichier (emit/verify, exit codes), détection d'altération
  artefact/certificat testée, ré-émission bit-identique.
- **`cargo som` + `--sarif`** : le linter d'ownership en sous-commande
  cargo avec sortie SARIF 2.1.0 pour le code scanning CI.
- **SOM opérationnel sur du vrai Rust** : frontend `syn`
  (`scirust-som-frontend`), oracle d'ownership **type-aware**
  (Copy/move exact, E0382/E0502/E0503-style), CLI `som-analyze`,
  pipeline Transformer entraîné/évalué contre l'oracle (ownership
  87,3 % vs baseline 33,1 % sur held-out), bit-déterminisme testé.
- Modules recâblés et réparés : `core::lazy` (fusion), 
  `core::tensor::{broadcast,device}`, `scirust_symbolic::prelude`.
- `archive/` : sources historiques retirées du build avec état documenté
  (GPU non câblé, NEON/SVE dupliqués, brouillon quant incorrect).
- Docs industrielles : `docs/REFERENCE.md` (commandes/binaires/API
  exhaustifs), `CONTRIBUTING.md`, `SECURITY.md`, audit
  `scirust_complete_audit_report.md`.
