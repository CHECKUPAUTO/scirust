# SciRust — Feuille de route « adoption industrielle »

Propositions d'implémentation classées par valeur pour un industriel,
fondées sur ce que SciRust possède déjà et que PyTorch/Burn/Candle n'ont
pas : **déterminisme bit-exact mesuré, auditabilité totale (pur Rust,
zéro FFI), quantization int8 bit-exacte embarquée, et un analyseur
d'ownership (SOM)**. La stratégie n'est pas de concourir sur les TFLOPS :
c'est de posséder le créneau « IA certifiable et reproductible ».

Chaque proposition précise : le client visé, le livrable, et la
définition de fini (toujours : gates verts + oracle + doc).

---

## P0 — Le socle de confiance (débloque tout le reste)

### P0.1 Inférence porteuse de preuve (« Proof-Carrying Inference »)
- **Client** : finance (audit modèles), médical/aéro (traçabilité),
  assurance, conformité IA (EU AI Act art. 12 — journalisation).
- **Quoi** : étendre `scirust-runtime` pour que chaque inférence émette
  un **certificat** : hash du manifeste d'architecture, hash SRT1 des
  poids, hash des entrées, empreinte 64-bit des sorties (déjà
  existante), seed, versions. Vérifieur indépendant
  `scirust-verify <certificat> <artefacts>` qui rejoue et compare
  bit à bit. Les briques existent (`proof_bundle`, fingerprint) — il
  manque le format stable + le vérifieur + la doc contractuelle.
- **Fini quand** : un tiers reproduit une inférence sur une autre
  machine x86/aarch64 et le vérifieur dit MATCH, en CI.

### P0.2 Release engineering : v1.0 outillée
- **Client** : tout acheteur — personne n'embarque un dépôt sans
  versions.
- **Quoi** : tags semver + `cargo-dist` ou releases GitHub avec binaires
  (`som-analyze`, vérifieur), CHANGELOG tenu (créé), politique MSRV
  affichée, **SBOM** (CycloneDX via `cargo-sbom`/`cargo auditable`)
  attaché à chaque release — l'argument « supply chain 100 % Rust,
  Cargo.lock committé, cargo-deny en CI » devient vérifiable d'un clic.
- **Fini quand** : `v0.14.0` taguée avec binaires + SBOM + notes.

### P0.3 Story « stable » : sortir du nightly pour les consommateurs
- **Client** : équipes d'industrialisation (politiques internes
  interdisent souvent nightly).
- **Quoi** : `portable-simd` est la seule vraie dépendance nightly du
  cœur. La rendre réellement optionnelle de bout en bout et prouver en
  CI un build **stable** de `scirust-core` (+ SOM, déjà compatible
  stable via `syn`) avec le dispatch runtime AVX2/NEON existant
  (intrinsics stables). Job CI `stable-build` dédié.
- **Fini quand** : `cargo +stable test -p scirust-core -p scirust-som-*`
  vert en CI.

## P1 — Les produits qui font signer

### P1.1 « SciRust Edge Pack » : int8 déterministe industrialisé
- **Client** : embarqué/IoT/automobile — c'est la capacité la plus
  différenciante déjà **validée** (int8 bit-exact, NEON ×10, QSR1).
- **Quoi** : transformer les 19 binaires d'audit en un produit : CLI
  `scirust-quantize <model.srt1>` → artefact QSR1 + rapport d'écart
  (bit-exact ou bornes), exemple cross-compilé aarch64 + taille binaire
  mesurée (`no_std`-friendly pour `scirust-embedded` à terme), guide
  « du float au int8 certifié en 30 minutes ».
- **Fini quand** : un README de 1 page reproduit la chaîne complète sur
  le MNIST du dépôt, avec les hashes attendus publiés.

### P1.2 SOM comme linter CI : `cargo som`
- **Client** : toute équipe Rust (au-delà du ML !) — porte d'entrée
  commerciale la plus large du dépôt.
- **Quoi** : empaqueter `som-analyze` en sous-commande cargo
  (`cargo-som`) + GitHub Action (`som-action`) : sortie SARIF pour
  l'onglet Security de GitHub, budget de fautes par PR, et le rapport
  pédagogique par token (déjà fait) en artefact. Étendre le frontend
  aux branches `if/else` (jointure conservatrice : état = pire des deux
  branches) — c'est la limite la plus visible aujourd'hui.
- **Fini quand** : l'action tourne sur ce dépôt même et commente une PR.

### P1.3 Benchmarks publics contre Burn/Candle/tch
- **Client** : décideurs techniques en phase d'évaluation.
- **Quoi** : `examples/benchmarks` réintégré au workspace en job CI
  nightly informatif ; matrice (matmul, conv, MNIST epoch, inférence
  int8) × (SciRust, Burn, Candle) × (x86 AVX2, aarch64 NEON), publiée
  dans `docs/BENCHMARKS.md` avec la méthodologie. Assumer les défaites
  en vitesse brute ; afficher les victoires (déterminisme, variance
  nulle, empreinte, build 100 % Rust).
- **Fini quand** : chiffres reproductibles par `cargo bench` documenté.

## P2 — Profondeur technique (différenciation durable)

### P2.1 Mode « déterminisme certifié » du training
- Étendre la garantie bit-exacte de l'inférence au **training**
  multi-thread : réductions à arbre fixe dans `data_parallel`
  (l'addition flottante n'est pas associative — ordonnancement figé),
  test CI « même seed, 1/2/4 threads ⇒ pertes bit-identiques ».
  Aucun framework grand public n'offre cette garantie testée.

### P2.2 GPU : trancher et recâbler proprement
- **FAIT (étape 1 — trancher)** : suppression des stubs GPU mensongers
  (`gemm_f32` renvoyait des zéros) ; `scirust-gpu` expose un backend CPU
  de référence testé + des chemins device honnêtes (`Unavailable`).
- **FAIT (étape 2 — recâbler wgpu)** : vrai GEMM WGSL derrière la feature
  `wgpu`, exécuté sur adaptateur Vulkan, **validé contre l'oracle CPU**
  (tolérance flottante documentée) et **testé en CI** sur Vulkan logiciel
  (Mesa lavapipe) — « pas de claim sans test » respecté. `cargo deny`
  passe sur l'arbre de deps wgpu. Dépendance optionnelle (les 8 gates par
  défaut ne la compilent pas).
- **FAIT (étape 3 — tape autograd)** : `WgpuEngine` implémente le hook
  `GpuEngine` du `Tape` ; `Var::matmul_gpu` exécute **forward ET backward**
  (`dA = g·Bᵀ`, `dB = Aᵀ·g`) sur le GPU, device/pipeline mis en cache.
  Validé bout-en-bout contre la tape CPU (forward + 2 gradients) sur
  lavapipe. Chemin opt-in (feature + `matmul_gpu`) → garantie bit-exacte
  par défaut intacte.
- **Reste** : router `Conv2d` (im2col + GEMM) via `matmul_gpu`, puis garder
  les activations en VRAM entre couches (pipelines im2col archivés en
  référence) ; plus d'ops (elementwise, réductions). CUDA/cuBLAS reste
  hors périmètre tant qu'un runner GPU matériel n'existe pas.

### P2.3 SOM précision rustc (HIR/MIR)
- Brancher `scirust-rustc-driver` pour : types résolus (fin du
  `let x = f();` conservateur), NLL réels, jointures de branches.
  L'oracle `syn` actuel devient le « mode rapide », le mode rustc le
  « mode précis » — même format de rapport.

### P2.4 Tenseur N-D unifié
- Fusionner `tensor::TensorND` (déjà dans core) avec la tape 2D :
  prérequis aux ambitions compilateur (shape inference au-delà de
  rows/cols), à faire **avant** tout IR de training.

## Ce qu'on ne propose PAS (anti-objectifs)
- Courir après les TFLOPS de PyTorch/TensorRT : créneau perdu d'avance
  et hors philosophie.
- Multiplier les crates : la valeur vient de la profondeur des
  garanties, pas de la surface. (`events-*`, `edge`, `bridge`
  restent gelées tant qu'un client ne les tire pas.)

## Ordre d'exécution recommandé
P0.2 (1 j) → P0.3 (2-3 j) → P0.1 (1 sem) → P1.2 (1 sem) →
P1.1 (1-2 sem) → P1.3 (continu) → P2.x selon traction.
