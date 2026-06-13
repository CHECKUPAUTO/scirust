# LIVESTATE — scirust

> Fichier de bord partagé entre agents.
> Dernière mise à jour : 2026-06-12

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
