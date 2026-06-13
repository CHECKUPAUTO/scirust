# LIVESTATE — scirust

> Fichier de bord partagé entre agents.
> Dernière mise à jour : 2026-06-13

## Session 2026-06-13 — volet 28 : bloc transformer N-D complet, entraînable
- op nd : layernorm(axe final, backward dx=rstd(g-mean_g-y·mean_gy)) gradient-
  checké. nd ops complètes : add/sub/mul/matmul/bmm/relu/softmax/transpose_last2
  /reshape/permute/layernorm/sum.
- nn::nd_layers : +NdLayerNorm (affine γ/β) +NdTransformerBlock (Pre-LN,
  résidus) ; +sgd_step partout (attn, ln, block). Test : **bloc transformer
  N-D complet qui APPREND** (perte<70%). « voici le bloc transformer ».
- la tape N-D = mini-framework transformer entraînable, coexiste avec la 2D.
- 765 tests ; 8 gates verts.

## Session 2026-06-13 — volet 27 : couches N-D réutilisables + generate_sampled public
- ops nd : reshape + permute (général, backward = perm inverse).
- nn::nd_layers : NdLinear (entraînable, sgd_step) + NdMultiHeadAttention
  (q/k/v/o + bloc attention). Tests : grad check entrée, MLP N-D qui APPREND
  (perte<70%), grad check couche attention complète. « voici les couches ».
- MiniLLM::generate_sampled(&str) : API publique sampling+KV-cache, greedy =
  generate, déterministe par graine. « sampling branché dans generate public ».
- 763 tests ; 8 gates verts. GROWTH_PLAN court terme : nd::Linear/Attention +
  sampling-in-generate = FAITS.

## Session 2026-06-13 — volet 26 : traiter à fond les 3 « parts honnêtes »
- (C) BPE byte-level (ByteBpeTokenizer, GPT-2) : base 256 octets → 0 OOV,
  round-trip lossless tout UTF-8 (emoji/accents/scripts inconnus) ; 5 tests ;
  CLI `bpe --bytes`. Remplace « BPE basique ».
- (A) sampling seedé (nn::sampling : temp/top-k/top-p, PcgEngine) ; greedy =
  argmax ; MiniLLM::generate_ids_cached_sampled (O(n) KV-cache + sampling,
  déterministe par graine) ; 5 tests. Remplace « décodage glouton ».
- (B) autograd N-D capable : +softmax(axe final, backward Jacobien) +
  transpose_last2 ; **attention multi-tête complète** softmax(Q·Kᵀ/√d)·V sur
  (têtes,seq,d) GRADIENT-CHECKÉE. La N-D = sur-ensemble capable ; 2D = défaut
  par choix d'archi (coexistence, pas TODO). Remplace « tape N-D = ops/toy ».
- principe tenu : aucun TODO laissé, chaque sujet traité à fond + testé.
- 759 tests workspace ; 8 gates verts. roadmap P2.4 / GROWTH_PLAN / CHANGELOG.

## Session 2026-06-13 — volet 25 : fusion N-D + LLM bout-en-bout + CLI bpe
- chantier 1 (fusion N-D) : NdVar::bmm (matmul batché broadcast) + sub ;
  forward + backward gradient-checkés. La tape N-D devient le sur-ensemble
  capable (ce que la 2D ne sait pas : scores d'attention par tête).
- chantier 2 (LLM e2e) : generate_ids (découplé du tokenizer → BPE pilote) ;
  KV-cache bout-en-bout (block/encoder::infer_step, PositionalEncoding::
  encoding_at, MiniLLM::generate_ids_cached) PROUVÉ équivalent au recalcul
  complet (seule la dernière position sert → attend tout dans les 2 cas, même
  encoder BERT non-causal). Test BPE→generate dans scirust-learning.
- chantier 3 (CLI) : commande `bpe` (scirust-learning ajouté en dep CLI),
  groupe NLP, REFERENCE/CHANGELOG. Pas de commande `generate` (modèle non
  entraîné = gibberish → ne pas sur-promettre).
- honnête : decode glouton (pas de sampling), tokenizer char/BPE basique,
  tape N-D = ops (pas la fusion complète de reverse.rs).
- 747 core + 29 CLI ; 8 gates verts.

## Session 2026-06-13 — volet 24 : campagne « faire grandir scirust » (TOUS les chantiers)
- réponse honnête à « scirust assez conséquent ? » : oui pour le créneau
  déterministe/auditable/embarqué ; lacunes = tape 2D, GPU immature, pas de
  KV-cache/tokenizer prod, ONNX export-only, pas de distribué. → lancé tout.
- BPE (scirust-learning) : bug déterminisme (max_by_key(count) dépend de
  l'ordre HashMap) → tie-break (count, Reverse(pair)) ; +5 tests.
- autodiff::nd (NOUVEAU) : autograd N-D reverse sur TensorND (add/mul
  broadcast, matmul2d, relu, sum) ; gradient check numérique. Coexiste avec
  la tape 2D ; utilise broadcast_shape/broadcast_to/matmul_shape (vol. 22).
- GPU elementwise : 2e pipeline wgpu (add/mul/relu) ; GpuChain.add/mul/relu ;
  couche matmul→+biais→relu 100% résidente VRAM, testée lavapipe.
- ONNX import : import_onnx_json + OnnxGraph::weights ; round-trip poids
  bit-exact (checkpoint) ; testé sur Linear KaimingNormal réel.
- KV-cache : infer_step existait (non testé) → test d'équivalence dernier
  token vs forward complet (causal). Correct → décodage O(n) validé.
- honnête : tape N-D = MVP (pas la fusion), ONNX = poids (pas graphe arbitraire
  ni protobuf), GPU = ops de base (pas tout l'op-set). Incréments testés.
- 743 tests workspace ; 18 wgpu ; 8 gates verts.

## Session 2026-06-13 — volet 23 : P2.3 infra — rustc-driver recompile + visible
- scirust-rustc-driver (exclu du workspace, rustc_private) ne compilait plus
  sur nightly courante : get_attrs renvoie un itérateur (plus un slice) →
  .is_empty() inexistant. Fix : .next().is_some() (+ allow(deprecated),
  car get_attrs déprécié au profit de find_attr!). +2 warnings triviaux
  (unused_extern_crate typo, import MirPass). Driver build = 0 warning.
- env : rustc-dev-x86_64 + rust-src INSTALLÉS → le driver buildable ici.
- job CI informatif rustc-driver (continue-on-error, installe rustc-dev/
  rust-src) : la casse était invisible car crate exclu/non-gaté.
- scirust-rustc-driver/target/ était SUIVI par git (artefacts) → git rm
  --cached + .gitignore.
- honnête : P2.3 réel (passe MIR ownership/NLL → format rapport SOM) reste
  le gros chantier, fragile, hors 8 gates. Livré l'infra, pas la passe.

## Session 2026-06-13 — volet 22 : P2.4 fondation — primitives d'inférence de forme N-D
- constat : TensorND a déjà reshape/transpose/slice/can_broadcast_to +
  pont from/to_tensor_2d. Manquaient les primitives d'inférence de forme.
- ajout (TensorND) : broadcast_shape(a,b) numpy ; matmul_shape(a,b) batché
  ((…,m,k)·(…,k,n)→(…,m,n), broadcast batch) ; broadcast_to(target)
  matérialisation. 3 tests (12 au total dans tensor_nd).
- honnête : la FUSION tape 2D↔ND (réécriture reverse.rs ~4700 l sur TensorND)
  reste le gros chantier ; livré la fondation testée, pas le bloc.
- clippy : needless_range_loop → enumerate sur out_strides. 732 tests.

## Session 2026-06-13 — volet 21 : revue de code max-effort + durcissement
- 4 angles finder (agents //) sur le diff de branche (~8,8k lignes) : GPU
  wgpu/résidence, routage autodiff GPU, threading déterminisme + cfg SIMD,
  numerics CLI. 2 angles : 0 bug (maths GEMM/transpose, gradients Conv2d,
  matmul_gpu, réduction threadée, cfg SIMD — tracés à la main, corrects).
- corrigé (GPU résident) : dims dégénérées m/n/k==0 → panique wgpu (buffers
  taille 0). Gardes : upload placeholder 4o, gemm_resident skip dispatch +
  buffer min 1 elem, download court-circuit vide. +test. (commit c141c5a)
- corrigé (run_ode) : h=0 → overflow panique (101) ; t1<=t0 → renvoyait y0
  en silence (code 0, faux) ; dopri5 code 1 au lieu de 2 sur bornes invalides.
  Garde unifiée t1>t0 & h>0 fini → code 2. +5 asserts. clippy: h<=0 (pas
  !(h>0), neg_cmp_op_on_partial_ord).
- 728 tests workspace ; wgpu 16. 8 gates verts.

## Session 2026-06-13 — volet 20 : déterminisme certifié du training multi-thread (P2.1)
- constat : DataParallelTrainer.train_batch était SÉQUENTIEL (pas de threads) ;
  réduction déjà en ordre worker fixe mais aucune garantie testée sous threads.
- ajout train_batch_threaded(n_threads, bf) : workers sur N threads OS
  (thread::scope + compteur atomique, vol de tâches), résultats écrits dans
  slots indexés par worker, reduce_mean en ordre fixe → indépendant du nombre
  de threads ET de l'ordre de terminaison (l'add flottante n'est pas assoc.).
- 2 tests : thread_count_invariant (contributions ±1e16 sensibles à l'ordre,
  1/2/4/8 threads == séquentiel bit-à-bit) + parallel_tape...deterministic
  (vrai backward ParallelTape sur 1/2/4 threads). Couverts par le job CI
  build-test existant.
- 728 tests (nightly+stable). 8 gates verts. roadmap P2.1 = FAIT.
- reste P2.1 : boucle multi-couches complète + bench de scaling.

## Session 2026-06-13 — volet 19 : activations résidentes en VRAM (P2.2 résidence)
- constat : DeviceTensor = { inner: Tensor } (CPU only). Résidence
  transparente dans la tape forcerait as_cpu() (≈50 sites) à matérialiser
  → gros refactor, bénéfice mesurable seulement sur GPU réel. Hors scope.
- livré le mécanisme honnête et testé : WgpuContext.upload/gemm_resident/
  download (+ encode_gemm partagé) ; GpuMatrix (buffer+shape) ; API publique
  GpuChain { new, upload, matmul, matmul_t, download }.
- test resident_chain : (A·B)·C garde T=A·B en VRAM, alimente le 2e GEMM
  sans download, == oracle CPU (rel<1e-4) sur llvmpipe. +resident_transpose.
  15 tests --features wgpu. scirust-gpu seul (core intact), défaut 726/6.
- 8 gates verts (fmt, clippy déf+wgpu, doc déf+wgpu, deny, default tests).
- reste P2.2 : résidence transparente dans la tape (DeviceTensor paresseux)
  + im2col/col2im GPU + ops elementwise.

## Session 2026-06-13 — volet 18 : Conv2d sur GPU (P2.2 étape Conv2d)
- nouvel helper Tape::gemm_ab(a,b,ta,tb) : engine.gemm si attaché (transpose
  natif), sinon match CPU bit-identique aux .matmul/.transpose d'origine.
- 3 GEMM de Conv2d routés : forward W·col (gemm_ab false,false), backward
  dW=dout·colᵀ (false,true), dInput=Wᵀ·dout (true,false) — dans
  try_conv2d_forward + l'arm Op::Conv2dForward de backward().
- test conv2d_gpu_matches_cpu (scirust-gpu, via try_conv2d_forward) : forward
  + dInput + dWeight == CPU (rel<1e-4) sur llvmpipe. 13 tests --features wgpu.
- défaut bit-identique (conv2d tests core OK, 726 inchangé). 8 gates verts.
- reste P2.2 : activations en VRAM entre couches + im2col/col2im GPU.

## Session 2026-06-13 — volet 17 : GPU wgpu branché dans la tape autograd (P2.2)
- découverte : scirust-core avait DÉJÀ tout le câblage (trait GpuEngine,
  Tape.gpu_engine + with/set/clear, Op::MatMulGpu avec backward GPU/CPU,
  Var::matmul_gpu) — mais AUCUN implémenteur, et le forward de
  try_matmul_gpu restait CPU.
- WgpuEngine (scirust-gpu, feature wgpu) implémente GpuEngine via un kernel
  GEMM général C=α·op(A)·op(B)+β·C (transpose/alpha/beta) partagé avec
  WgpuBackend (refactor WgpuContext, device/pipeline en cache). Repli CPU
  si dispatch échoue (jamais de faux résultat).
- scirust-core: try_matmul_gpu forward utilise l'engine si présent (sinon
  CPU). Pas de cycle : scirust-gpu→scirust-core (optionnel, feature wgpu) ;
  core ne dépend que de gpu-macros.
- test bout-en-bout : matmul_gpu forward+backward == tape CPU (rel<1e-4)
  exécuté sur llvmpipe (confirmé "wgpu engine on: llvmpipe"). +3 tests
  (12 total --features wgpu). 726 défaut inchangé.
- 8 gates verts (clippy défaut+wgpu, deny avec core-dep-de-gpu, doc,
  stable/nightly 726, aarch64, portable-simd). docs GPU.md/README/roadmap
  /CHANGELOG mis à jour. Reste P2.2 : Conv2d (im2col) via matmul_gpu.
- note infra : disque saturé pendant build wgpu → purge target/doc+incremental.

## Session 2026-06-13 — volet 16 : GPU wgpu RÉEL (P2.2 recâbler), testé sur lavapipe
- décision user : « pursue real wgpu now ». Débloqué en installant Mesa
  lavapipe (mesa-vulkan-drivers) → adaptateur Vulkan logiciel llvmpipe
  présent dans le conteneur → wgpu testable ici.
- vrai GEMM WGSL (C=A·B, f32) derrière feature `wgpu` (wgpu 0.20.1 +
  pollster + bytemuck, deps optionnelles). WgpuBackend::gemm_f32 exécute
  le shader sur Vulkan ; sinon/erreur adaptateur → Unavailable (jamais
  inventé). 3 tests wgpu validés contre l'oracle CpuBackend (rel.err<1e-4)
  — exécutés réellement (pas skippés) sur llvmpipe.
- cargo deny PASSE sur l'arbre wgpu (advisories/bans/licenses/sources ok).
  clippy --features wgpu clean. doc clean. no_std + default intacts.
- CI : job `GPU (wgpu / lavapipe)` (apt install mesa-vulkan-drivers +
  cargo test -p scirust-gpu --features wgpu). Gates par défaut ne
  compilent pas wgpu (optionnel) → aarch64/etc. inchangés.
- docs : GPU.md réécrit (wgpu réel+testé, déterminisme tolérant, supply
  chain), README (5 endroits), roadmap P2.2 (étapes FAIT), CHANGELOG.
- reste P2.2 : brancher wgpu dans la tape autograd / Conv2d (im2col
  archivés en réf). 726 tests (défaut) ; +9 en scirust-gpu --features wgpu.

## Session 2026-06-13 — volet 15 : SBOM CycloneDX + release v0.14 (prép) + GPU.md honnête
- SBOM : CycloneDX 1.5 reproductible (docs/sbom/scirust.cdx.json, 78
  composants, façade scirust 0.13.0). SOURCE_DATE_EPOCH figé + pas de
  serial → octet-identique vérifié. scripts/generate-sbom.sh (garde le
  SBOM façade, purge les SBOM par-membre que cargo-cyclonedx essaime).
  .gitignore : *.cdx.json sauf docs/sbom/.
- CI : job sbom (artefact, informatif). release.yml : sur tag v*, rejoue
  les gates + génère + attache le SBOM à la release (le tag = action
  humaine ; l'auto le suit). SECURITY.md + docs/sbom/README.md.
- docs/GPU.md : décrivait une API GPU une-ligne INEXISTANTE (modules
  archivés) → réécrite en statut+roadmap honnête (CPU ref testé ; pas de
  claim GPU ; pourquoi ; plan P2.2). README : liens SBOM/SECURITY.
- reste pour la release : bump 0.13→0.14 + push tag (gated, perm requise) ;
  protection de branche = réglage GitHub (non scriptable ici).
- gates non-Rust → fmt clean ; build/test inchangés (726).

## Session 2026-06-13 — volet 14 : P2.2 « trancher » — scirust-gpu honnête
- diagnostic : scirust-gpu/src/lib.rs livrait WgpuBackend/CudaBackend dont
  gemm_f32 renvoyait vec![0.0; m*n] — résultats FABRIQUÉS sous étiquette
  GPU, en violation directe de la politique du repo (la même crate core
  compute_backend.rs faisait déjà ça correctement : Err honnête + tests).
- env. probé : pas de /dev/dri, pas d'ICD Vulkan (loader libvulkan présent
  mais aucun driver lavapipe) → wgpu NON testable ici. Donc « pas de claim
  sans test » interdit d'ajouter un chemin wgpu non testable maintenant.
- correctif (in-philosophy, débloqué) : vrai backend CPU de référence testé
  (GEMM bit-déterministe, oracle), device paths → BackendError::Unavailable
  (jamais de sortie inventée), BackendError {Unavailable, ShapeMismatch}.
  0 → 6 tests. std + no_std compilent. README requalifié (3 endroits).
- types scirust-gpu non importés ailleurs (vérifié) → refactor sans casse.
- reste P2.2 (vrai wgpu) = décision produit : deps lourdes (wgpu/naga) vs
  philosophie auditable + non-déterminisme FP GPU + runner CI absent.
- 726 tests workspace ; 8 gates verts.

## Session 2026-06-13 — volet 13 : CLI vague 5 (tt, solve-system, inverse, fem-heat, dopri5)
- +4 commandes + 1 méthode : tt (compression tensor-train TT-SVD,
  scirust-tn — cœurs/rangs/ratio/erreur, sortie 1 si --max-err dépassé),
  solve-system (système non-linéaire F(x)=0 via Broyden), inverse (LU),
  fem-heat (chaleur 1D -u''=source, éléments finis), ode --method dopri5
  (Dormand-Prince adaptatif). 37 commandes au total. Nouveau groupe
  TENSOR NETWORKS.
- testes manquants : FemSolver1D était non testé → 2 tests (oracle
  parabole -u''=f exact aux nœuds + symétrie). reconstruct_matrix
  réexporté depuis scirust-tn (paire de tt_decompose_matrix).
- newton_system non exposé (closure Fn(&[Dual]) comme bfgs — honnête)
- vérifs main : √2 (solve-system), inverse 2×2 exacte, e à 1e-8 (dopri5),
  FEM == (f/2)x(L-x), tt rel.err 2.4e-7 (ratio honnête <1 sur petite
  matrice : surcoût des cœurs domine)
- 32 tests CLI ; 720 tests workspace (nightly + stable) ; 8 gates verts

## Session 2026-06-13 — volet 12 : CLI vague 4 (trig, patterns, qr, cg)
- +4 commandes : trig (apply_trig_identity), patterns (discover_patterns),
  qr (qr_decompose), cg (conjugate_gradient SPD). 33 commandes au total.
- bfgs NON exposé : Fn(&[Dual])->Dual non câblable depuis eval(f64) — honnête
- vérifs : cg == linsolve (0.0909,0.6364), QR orthogonal, patterns trend_up
- 43 tests CLI ; 8 gates verts

## Session 2026-06-13 — volet 11 : CLI vague 3 (symreg, sat, root methods)
- +4 commandes/méthodes : symreg (scirust-symreg : GP + fit constantes
  symbolique), sat (scirust-neuro-symbolic : DPLL), root --method
  secant|newton (newton via dérivée symbolique). Nouveau groupe LOGIC.
- module reasoning.rs ; +deps scirust-symreg, scirust-neuro-symbolic
- vérifs main : secant/newton→√2, SAT {1,2}/UNSAT, symreg y≈2x MSE≈0
- CSP/datalog laissés de côté (closures/règles non exprimables en CLI
  sans inventer un DSL non testé — hors politique)
- 39 tests CLI ; 8 gates verts

## Session 2026-06-13 — volet 10 : CLI vague 2 (capacités testées non exposées)
- +9 commandes : integrate --method simpson|gauss, root --method
  bisection, optimize (Nelder–Mead multi-D), lstsq (QR), cholesky,
  prove (équiv. symbolique), gradient (num. 1-2 var) ; aide enrichie
- Réponse MTP : Multi-Token Prediction NON nécessaire (hors niche
  déterministe/embarquée ; leviers réels = int8/SIMD/fusion/KV-cache/GPU)
- 34 tests CLI ; 8 gates verts

## Session 2026-06-13 — volet 9 : CLI massive (19 commandes)
- +10 commandes adossées à du code testé : cmaes ; to-rust, regress ;
  integrate/root/minimize/linsolve/det/polyroots/ode (scirust-solvers,
  pont via scirust-symbolic::eval pour les commandes à expression)
- module numeric.rs ; +deps scirust-solvers ; 27 tests CLI au total
- bug réel attrapé : regress sortait 1x+2 au lieu de 2x+1 (ordre du
  tuple (intercept,slope) inversé) → corrigé + test de convention
- aide groupée en 5 sections (LEARNING & OPTIMIZATION / SYMBOLIC /
  NUMERICAL SOLVERS / CODE ANALYSIS / INFERENCE / META)

## Session 2026-06-13 — volet 8 : CLI industrielle + flash attention testé
- Flash Attention RÉELLEMENT testé : 4 tests (forward vs oracle dense,
  causal, bit-déterminisme, gradients finis) → statut « ✅ Stable » honnête
- 2 lignes GPU retirées du tableau Status (listaient du non-câblé) ;
  note « Not included yet » + renvoi roadmap P2.2
- CLI `scirust` étoffée : `som train`, `evo`, `diff`/`simplify`/`eval`/
  `solve`, `info` — aide groupée par thème ; +modules symbolic.rs,
  learning.rs ; 11 tests CLI ; chaque commande adossée à du code testé
- README/REFERENCE/CHANGELOG mis à jour

## Session 2026-06-13 — volet 7 : CLI unifiée (UX)
- `scirust-cli` (nouveau) : binaire `scirust` + lib, dispatcher
  découvrable (`help`/`version`/`quickstart`/`analyze`/`verify`) au-dessus
  du code déjà testé ; 7 tests (help/version, commande inconnue→2,
  quickstart 4/4 + bit-déterministe, usage analyze/verify)
- `scirust_runtime::proofcli` : logique emit/verify extraite en lib
  (DRY) ; `scirust-verify` délègue ; test verify_roundtrip toujours vert
- README : Quickstart réécrit autour de la CLI (fini les 40 lignes
  d'API à copier) ; section « Library API » avec snippet corrigé pour
  l'API réelle (.add / loss_fn.forward(&tape,..) / tape.backward(idx))
- REFERENCE/CHANGELOG mis à jour

## Session 2026-06-12 — volet 6 : exécution roadmap P0/P1
- P0.3 STABLE : feature(portable_simd) optionnelle via cfg_attr +
  fallback scalaire tiling → workspace entier compile ET passe 683
  tests sur Rust stable ; job CI build-test-stable ajouté ; feature
  nightly portable-simd réparée (migration std::simd : SimdFloat→num::,
  LaneCount/SupportedLaneCount supprimés) — 763 tests verts feature ON
- P0.1 PREUVE : binaire scirust-verify (emit/verify de certificats
  SCIRUST-PROOF-1) ; test E2E : MATCH propre, altération artefact
  détectée, certificat falsifié détecté, ré-émission bit-identique
- P1.2 LINTER CI : crate cli refactorée en lib partagée + 2 binaires
  (som-analyze, cargo-som) ; sortie --sarif SARIF 2.1.0 validée par
  test JSON ; localisations niveau fichier (spans = prochain jalon)

## Session 2026-06-12 — volet 5 : tour de code « philosophie & câblage »
- 20 fichiers morts confirmés par test de corruption, traités :
  recâblés (core::lazy + fix réel de la fusion pointwise qui ne
  fusionnait jamais, tensor::broadcast/device, symbolic::prelude) ou
  archivés hors build avec état documenté (`archive/` : gpu ×8,
  neon/sve dupliqués, brouillon quant aux kernels faux)
- Déterminisme restauré dans data::augment : RNG PcgEngine injecté,
  flux par échantillon indépendant de l'ordre, with_seed effectif,
  vrai gaussien Box-Muller, et fix du RandomCrop no-op (résultat jeté)
- Standards industriels : CONTRIBUTING.md, SECURITY.md, CHANGELOG.md,
  docs/INDUSTRIAL_ROADMAP.md (propositions P0–P2)
- **683 tests, 0 échec, 19 ignorés** ; 7 vérifications vertes ;
  plus aucun fichier non compilé sous */src/

## Session 2026-06-12 — volet 4 : fiabilisation industrielle
- Oracle SOM type-aware : Copy/move exact (i32/f64/bool/*T/&T copient ;
  String/&mut T déplacent), inférence locale, faute UseWhileMutBorrowed
  (E0503-style) ; 3 nouveaux tests oracle + 3 tests bout-en-bout CLI sur
  vrai Rust (double usage i32 légal, inférence, lecture sous &mut)
- Métriques re-mesurées : ownership 87,3 % (baseline 33,1 %), borrow
  94,0 %, fautes 88,6 % (held-out 9042, 850 tokens)
- README racine : claims GPU requalifiés « Archived — not wired »
  (véracité claims=code restaurée)
- docs/REFERENCE.md : référence exhaustive commandes/binaires/API
- rustdoc : 22 warnings corrigés → cargo doc --workspace : 0 warning
- Audit : mise à jour fiabilisation ajoutée au rapport
- Bilan volet 4 : **672 tests workspace, 0 échec, 19 ignorés** ;
  7 vérifications vertes (fmt, clippy --all-targets, build, test,
  cross-check aarch64, cargo-deny, rustdoc 0 warning)

## Session 2026-06-12 — volet 3 : SOM sur du VRAI Rust
- `scirust-som-frontend` (nouveau) : parser `syn` (grammaire Rust réelle,
  stable) → abaisse un sous-ensemble vers l'IR de l'oracle. Couvre fn /
  let / move / &,&mut / blocs / return / appels / impl-méthodes ; signale
  honnêtement ce qui est sauté (if/match/loops/closures/macros) ou
  approximé (receveur de méthode = emprunt partagé). 6 tests.
- `scirust-som-cli` (nouveau) : binaire `som-analyze <file.rs>` — analyse
  d'ownership d'un vrai fichier Rust, table par token + diagnostics,
  exit 1 si faute (utilisable comme linter). 4 tests d'intégration
  bout-en-bout (vrai source Rust → oracle). Exemples dans
  scirust-som/examples/ (use_after_move.rs détecté E0382, borrow_conflict.rs).
- `inference::predict_rust_source` : entraîne sur synthétique, prédit sur
  vrai Rust ; test bout-en-bout (accord modèle/oracle > 0,4 sur fichier réel).
- Honnêteté documentée (README) : emprunts LEXICAUX (pas NLL,
  conservateur) ; types Copy sur-signalés (uniform move) ; code
  rectiligne seulement. La précision NLL/Copy/branches = chantier
  rustc-driver (HIR/MIR), hors workspace.
- SOM passe de 25 à 35 tests ; workspace ~665 tests, 6 gates verts.

## Session 2026-06-12 — volet 2 : réparation CI
- `.github/workflows/ci.yml` réécrit pour être réalisable :
  - suppression de `--all-features` (blas-openblas et blas-mkl sont
    mutuellement exclusifs → la CI ne pouvait JAMAIS compiler)
  - nouveau job `cross-check-aarch64` (cargo check --target aarch64) :
    type-vérifie tous les chemins NEON/SVE — la classe de bug du merge
    du 12/06 devient détectable sur PR
  - coverage passé en informatif (continue-on-error), cargo-llvm-cov
    pré-compilé via taiki-e/install-action
- `deny.toml` réécrit (l'ancien n'était pas du TOML valide → le job
  License/Security échouait au parsing) ; validé en local avec
  cargo-deny 0.19.8 : advisories/bans/licenses/sources tous ok ;
  RUSTSEC-2024-0436 (`paste` unmaintained via nalgebra→simba) ignoré
  avec justification — c'est l'alerte Dependabot ouverte
- `publish = false` ajouté aux 51 manifestes (réalité : deps par chemin,
  licence non-commerciale) — active l'exemption licences des crates
  privées dans cargo-deny
- Tous les warnings restants éliminés (RUSTFLAGS -D warnings tenable) ;
  gate clippy étendu à --all-targets : 14 lints de code de test corrigés
- Cross-check aarch64 a aussi révélé/réparé : test SVE cassé
  (`sve_vector_length_elements` inexistante — modules sve/sve_fns en
  ré-export circulaire vide) → implémentation réelle via `rdvl` (asm
  stable, gardée par détection runtime)
- Reste à faire côté GitHub (non scriptable depuis le repo) : protection
  de branche master exigeant fmt/clippy/build-test/cross-check/deny

## Session 2026-06-12 (audit + SOM tranche verticale)
- Audit complet exécuté : voir `scirust_complete_audit_report.md`
- Régression de merge réparée (sgemv AVX2/SSE2/NEON, arena slab) ; gates
  check / clippy -D warnings / test / fmt tous verts en local
- **655 tests workspace passent** (630 avant SOM, +25 SOM), 0 échec
- SOM : tranche verticale réelle livrée (voir `scirust-som/README.md`) —
  oracle d'ownership déterministe, tokenizer+vocab fermé, générateur de
  dataset seedé, backbone TransformerEncoder réel (attention du core),
  trainer bit-déterministe, éval vs oracle : ownership 83,7 % vs baseline
  31,4 % (held-out seed 9042), visualizer markdown
- Anciens stubs SOM remplacés : trainer/inference/symbolic/visualizer
  n'étaient que des fichiers d'1 ligne ; le « Graph Transformer » (MLP
  étiqueté) est devenu un vrai Transformer séquence — l'attention sur
  graphe PCG reste un travail futur et est documentée comme telle

## Référence
- Branche de travail : `claude/great-pascal-5bmfcw` (sessions 2026-06-12)
- L'état mesuré fait foi : sections ci-dessus + `scirust_complete_audit_report.md`
- Notes durables : commentaires FR/EN mixtes ; nightly requis
  (portable_simd) ; TT d>2 backward non implémenté (gradients zéro, cas
  rare) ; Cargo.lock versionné (retiré du .gitignore à confirmer)
