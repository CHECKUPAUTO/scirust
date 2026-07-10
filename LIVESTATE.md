# LIVESTATE — scirust

> Fichier de bord partagé entre agents.
> Dernière mise à jour : 2026-07-10

## Session 2026-07-10 — fluides & thermo, volet 2 (IF97 complet, convection, réseaux)
- **Contexte** : PR #281 (volet 1) MERGÉE ; « continu » → exécution des trois
  « suites possibles » du volet 1. Branche repartie de master
  (même nom, procédure branche-mergée).
- **IF97 régions 1 & 2** (`scirust-thermo::steam`) : Gibbs complet
  (v,h,u,s,cp,cv,w), liquide comprimé + vapeur surchauffée + frontière B23.
  Coefficients extraits par script depuis le paquet Python `iapws`
  (implémentation de référence, scratchpad) — zéro transcription manuelle ;
  vérifiés en Python pur AVANT écriture du Rust (6 points tables 5/15 OK),
  puis les mêmes oracles en Rust passent à 1e-8 du premier coup.
- **Cycle de Rankine** (`cycles::rankine_ideal`) : struct RankineCycle
  (travaux, chaleurs, η, titre), échappement humide OU surchauffé
  (bissection déterministe). Oracle Cengel 10-1 : η ≈ 0,260 ✓ x₄ ≈ 0,886 ✓.
- **Convection** (`scirust-thermo::convection`) : plaque plane
  laminaire/mixte, Churchill–Bernstein, Ranz–Marshall, Churchill–Chu
  (×2), Rayleigh. Raccord laminaire/mixte à Re=5e5 vérifié (<0,2 %).
- **Hardy Cross** (`scirust-fluids::network`) : NetworkPipe (h=r·|Q|ⁿ⁻¹·Q),
  hardy_cross() déterministe, continuité préservée par construction.
  Solutions analytiques 2 et 3 conduites parallèles à 1e-8.
- **Vérifié** : 111 tests unitaires + 2 doctests verts ; clippy
  `--all-targets -- -D warnings` propre ; fmt appliqué.
- **Suite possible** : IF97 région 3 (autour du point critique, équation
  de Helmholtz) et région 5 (>1073 K) ; équations backward T(p,h)/T(p,s)
  (turbines réelles à rendement isentropique < 1) ; pertes de charge
  réelles dans Hardy Cross via friction_colebrook (couplage
  fluids ↔ conduites) — non entamé.

## Session 2026-07-10 — mécanique des fluides & thermodynamique
- **Contexte** : demande utilisateur — « scirust n'offre pas de solutions aux
  problématiques de mécanique des fluides, la thermodynamique… ». Constat
  vérifié : aucun crate ne couvrait ces domaines (seule mention : primitives
  CFD/thermo « à compléter » dans docs/TRANSPILER_DESIGN.md).
- **Livré** : deux nouveaux crates métier sur le modèle des crates volet
  roadmap (Rust pur, zéro dépendance, `forbid(unsafe_code)`,
  `deny(missing_docs)`, constructeurs validés, erreurs typées, solveurs
  itératifs déterministes — Newton/bissection à algorithme fixe).
  - `scirust-fluids` : adimensionnels, friction en conduite
    (Colebrook–White + explicites), Darcy–Weisbach, Bernoulli/Venturi/
    orifice/Pitot, traînée (Stokes, Clift–Gauvin, vitesse terminale),
    couche limite plaque plane, compressible (isentropique + choc normal),
    canaux (Manning, ressaut, profondeurs critique/normale). 49 tests.
  - `scirust-thermo` : gaz parfait (processus + entropie), cycles (Carnot,
    Otto, Diesel, Brayton), transfert thermique (résistances, DTLM, NUT,
    rayonnement, Dittus–Boelter), psychrométrie ASHRAE (Hyland–Wexler,
    point de rosée), saturation eau IAPWS-IF97 région 4 (tables 35/36
    reproduites à 1e-8). 40 tests.
- **Vérifié** : `cargo test` 87 unitaires + 2 doctests verts ;
  `cargo clippy --all-targets -- -D warnings` propre ; `cargo fmt` appliqué ;
  membres ajoutés au workspace racine.
- **Suite possible** : IF97 régions 1/2 (enthalpies vapeur → cycle de
  Rankine complet), corrélations de convection externes (Churchill–
  Bernstein), réseaux de conduites (Hardy Cross) — non entamé.

## Session 2026-07-10 — volet 114 : les 4 chantiers restants de la cartographie
- **Contexte** : PR #273 (volet 113) MERGÉE ; branche repartie de master.
  Demande utilisateur : « continu sur les 4 points » (RNG contre-basé, GEMM
  reproductible rapide, RoPE/FFT portables, arrondi correct prouvé).
- **Point 1 — Philox4x32-10 (`scirust-core::philox`)** : clean-room depuis le
  papier SC'11, validé contre les vecteurs publiés Random123 (+ impl Python
  indépendante). Fonction pure (clé, compteur) ⇒ aléa order-independent
  (dropout/init/shuffle parallèles bit-identiques). Empreinte-contrat
  0xf96c6b6aeca699f5. QEMU aarch64 : 6/6.
- **Point 2 — accumulateur exact de Kulisch (`scirust-core::exact_acc`)** :
  dot/GEMM à somme EXACTE (virgule fixe 704 bits, bit 0 = 2⁻³⁵²), arrondi
  unique ⇒ correctement arrondi + indépendant de l'ordre + fusion associative
  (multithread bit-exact). Vérifié bit à bit contre la référence Shewchuk.
  NB : le debug_assert de résolution a attrapé un mauvais ancrage initial
  (LSB de mantisse à 2⁻³⁵⁰, pas 2⁻²⁹⁸) — corrigé, assert durci en public.
  QEMU aarch64 : 6/6.
- **Point 3 — RoPE portable (`NdVar::rope_portable`, empreinte
  0xfffeed24261eb5d6) + FFT portable (`scirust-signal::{fft,ifft}_portable`,
  twiddles via la nouvelle API `portable_f32::sincos_small_f64`, empreinte
  0x0acde0a67b427c67)**. QEMU aarch64 : verts.
- **Point 4 — certification d'arrondi correct** (`portable_f32::certify`,
  modes `--certify`/`--eval`, `scripts/verify-certify-offline.py`).
  Campagne exhaustive x86-64 (7 × 2³² entrées, 824 s) — entrées PROUVÉES
  correctement arrondies par certificat d'intervalle (gardes analytiques
  comprises) : exp 99,99915 % (36 512 non certifiées), ln 99,99998 % (695),
  tanh (214), sigmoid (1 007), sin (3 630), cos (3 680), erf (49 244 — zone
  de cancellation x∈[2,4], borne par entrée volontairement large). Les
  94 982 non certifiées ont été tranchées hors ligne en précision arbitraire
  (Decimal 60 chiffres, milieux exacts en rationnels — script corrigé pour
  le seuil d'overflow 2¹²⁸−2¹⁰³ et l'entrée f32 exacte via son rationnel).
  **VERDICT FINAL : 94 517 sont en fait correctement arrondies ; il reste
  465 entrées fidèles à 1 ulp — ZÉRO cas au-delà — sur les 30 064 771 072
  évaluations.** Par fonction (misroundings / 2³²) : exp 2, ln 5, tanh 20,
  sigmoid 78, sin 2, cos 6, erf 352. Taux global : 1,5×10⁻⁸. Comme les
  fonctions sont bit-identiques inter-plates-formes (contrats exhaustifs),
  ce verdict vaut sur TOUTES les plates-formes conformes. Claim mise à jour
  dans la doc du module : « correctement arrondi pour 99,9999985 % des
  entrées, vérifié exhaustivement ; le reste fidèle (1 ulp), listé ».
  L'évaluateur interne est revalidé contre la fonction expédiée sur chacune
  des 30 milliards d'entrées.

## Session 2026-07-10 — volet 113 : entraînement 100 % portable + tanh/sigmoid (lot 1 carto)
- **Contexte** : PR #272 (volet 112) MERGÉE ; branche repartie de master. Programme
  utilisateur acté : CE portable → entraînement MNIST-like portable avec contrat
  de poids → cartographie des trous (lot 1 : transcendantales).
- **CrossEntropyLoss portable** : nouveaux nœuds opt-in `Var::{exp_portable,
  ln_portable, matmul_portable}` (backwards sans libm : Exp depuis la sortie
  stockée, Ln = g⊙1/x, MatMul via le GEMM portable + transpose) dans reverse.rs
  ET parallel.rs ; `CrossEntropyLoss::new_portable()` bascule le log-softmax
  interne sur exp/ln portables. Test : perte+gradient ≡ voie libm (1e-6) +
  empreinte figée 0x40b66c65dceb9772.
- **Entraînement 100 % portable — `proof_portable_training`** : MLP 32×16×10,
  batch 8, 30 pas Adam, données/init PCG déterministes ; chaque nœud du graphe
  est portable (matmul_portable, ReLU, CE portable ; Adam = IEEE + powi/sqrt).
  Contrats commis (x86-64) : trajectoire de perte 0x531f63eb50666b8a, **poids
  finaux 0x4bbd3d8dc162b305**. Intégré au script de preuve (report-training.txt
  dans le bundle) et au job CI QEMU. C'est LA réponse à « l'entraînement
  reproductible cross-platform » : mêmes poids au bit près sur toute machine.
- **Lot 1 carto — tanh/sigmoid portables** : `tanh_f32`/`sigmoid_f32` dans
  portable_f32 (cœur exp_f64 factorisé ; formes stables sans cancellation ;
  saturations analysées ; tanh impaire exacte, ±0 préservés). Oracles ≤ 1 ulp
  vs libm f64 sur 200 k points ; contrats commis : tanh contract
  0x418f903e10257c1e / dense 0xa25de6342faed6e8 / exhaustif 0xd6f9e8508d19f785,
  sigmoid contract 0xea084f0622bdfec4 / dense 0xb82676717c581433 / exhaustif
  0x6796eabedfe7cb02. Binaire de preuve étendu (4 fonctions balayées).
  Débloque : LSTM/GRU portables, GELU-tanh.
- **Lot 1 (suite) — sin/cos portables avec Payne–Hanek** : `sin_f32`/`cos_f32`
  dans portable_f32. Réduction d'argument de Payne & Hanek en **arithmétique
  entière pure u128** (exacte pour tout f32 fini jusqu'à 3,4e38) : produit
  mantisse × 448 bits de 2/π — bits GÉNÉRÉS par nos soins (π par Chudnovsky
  en Decimal, vérifié par recomposition ; aucune table copiée) — quadrant +
  128 bits de fraction signée, r = fraction·(π/2) à ~2⁻⁵² relatif (conversion
  i128→f64 correctement arrondie ⇒ fidèle même aux pires cas de réduction f32,
  |r| ≳ 2⁻³²). Polynômes de Taylor sin(deg 15)/cos(deg 16) sur [−π/4, π/4].
  Oracle ≤ 1 ulp vs libm f64 sur 200 k points TOUTES magnitudes ; parités
  bit-exactes ; sin²+cos² = 1 à 1e38. Contrats commis : sin contract
  0x39c99b71fdbce247 / dense 0x084d235e4d8ddac7 / exhaustif 0xc0719c2d610d8685,
  cos contract 0xcdc07dac0d401d29 / dense 0xcde8a193db4b2f5c / exhaustif
  0xb9b0750ee67e5475. Binaire de preuve : 6 fonctions balayées. Débloque :
  RoPE portable (transformers), FFT portable (scirust-signal), encodages
  positionnels.
- **Lot 1 (fin) — erf + GELU exact portables** : `erf_f32` (série de Maclaurin
  f64, arrêt relatif déterministe, saturation |x| ≥ 4 → ±1, raccourci
  |x| < 1e-4 qui préserve ±0 — bug de signe de zéro attrapé par le test
  specials et corrigé avant commit des goldens) et `gelu_f32` (x/2·(1+erf(x/√2))
  via le cœur f64, sans cast intermédiaire ; gelu(−∞) = −0). Précision
  vérifiée contre une **table de référence indépendante** (série en Decimal
  60 chiffres, générée par nos soins — pas la libm, qui n'a pas d'erf en Rust
  std). Contrats : erf contract 0xfe817b5a5db40dc8 / dense 0xb7d54a90605132c5 /
  exhaustif 0x37655614b70cf42d ; gelu contract 0x8f06fb9eb406d63f / dense
  0xf1a6e6ae9f03349b. Binaire de preuve : 8 balayages. **LOT 1 COMPLET** :
  la voie portable offre exp, ln, tanh, sigmoid, sin, cos, erf, GELU —
  soit STRICTEMENT PLUS que les transcendantales de RepDL (exp/log), toutes
  sous contrat exhaustif ou dense. Prochains candidats de la carto : RNG
  contre-basé (Philox), GEMM classe ReproBLAS, erf dans scirust-special
  (remplacer la claim libm « partout » par la voie portable).
- Validation avant commit : preuve native x86 PASS + QEMU aarch64 PASS
  (tests, proof_portable_f32, proof_portable_training).

## Session 2026-07-10 — volet 112 : preuve cross-platform exécutable de portable_f32 (x86_64 ↔ Jetson)
- **Contexte** : PR #271 (volet 111) MERGÉE ; branche repartie de master (protocole
  branche-mergée). Demande utilisateur : « on doit prouver sur jetson et x86_64
  debian ».
- **Livré** : `scirust-core --bin proof_portable_f32` (auto-vérifiant : goldens +
  balayages contrat 65 537 / dense 257 / **exhaustif pas 1 = 2³² entrées** avec
  `--full` + composites softmax/GEMM, vs constantes `PROOF_*` commises ; exit 0 ⇔
  verdict=PASS ; lignes canoniques vs contexte `#`, SHA-256 comparable entre
  machines) + `scripts/proof-portable-f32.sh` (bundle d'évidence à la convention
  O1, `.gitignore`d) + section `docs/TEST_PROTOCOL.md`. Contrat rendu public et
  partagé tests/binaire ; empreintes denses (exp 0x6495da04866c1c4b,
  ln 0x19e7fd497cffd94b) et exhaustives (exp 0xda65ffaf8fe9f4f4,
  ln 0xb9ad67e08ae8f0fa) ajoutées au contrat.
- **Volet x86_64 exécuté** (conteneur Ubuntu 24.04 x86_64, rustc nightly du
  toolchain épinglé) : `scripts/proof-portable-f32.sh --full` →
  **verdict=PASS** (goldens, contrat, dense, softmax, GEMM, exhaustif — tout
  OK), balayage exhaustif ≈ 89 s, SHA canonique `--full` =
  `e9ac206146dc0b0e3aeb95e3a75880564649fd09043ab5d5c76b1f07bac5b7ae`.
  Les autres machines (Debian x86_64, Jetson) doivent reproduire verdict=PASS
  ET ce SHA exact en mode `--full`.
- **PREUVE CONSTATÉE SUR LES 3 PLATES-FORMES (2026-07-10, commit dc8918e,
  sorties fournies par l'utilisateur)** — mode `--full`, SHA canonique
  IDENTIQUE partout : `e9ac206146dc0b0e3aeb95e3a75880564649fd09043ab5d5c76b1f07bac5b7ae`.
  - **Jetson (aarch64)** : verdict=PASS, exhaustif 83,2 s,
    bundle `proof-portable-f32-20260710T152117Z` (sur l'appareil).
  - **Debian x86_64** : verdict=PASS, exhaustif 97,4 s,
    bundle `proof-portable-f32-20260710T152258Z` (sur la machine).
  - **Conteneur Ubuntu x86_64** : verdict=PASS, exhaustif 89,2 s,
    bundle `proof-portable-f32-20260710T144512Z`.
  Conclusion : l'identité bit à bit x86-64 ↔ aarch64 de la voie f32 portable
  est **constatée sur la totalité des 2³² entrées** de exp_f32 et ln_f32
  (goldens, balayages contrat/dense/exhaustif, softmax, GEMM — tout OK).
  La claim « bit-exact inter-plates-formes par construction » est désormais
  adossée à une exécution multi-machines, conformément à la discipline
  claims → évidence du dépôt.
- **Preuve aarch64 EN CI (suite de session, ferme le « reste ouvert » CI
  check-only)** : le job `cross-check-aarch64` exécute désormais réellement
  du code aarch64 — qemu-user + gcc-aarch64, `cargo test portable_f32` +
  binaire de preuve (mode standard : goldens + contrat + dense + composites)
  sur target aarch64-unknown-linux-gnu. Validé localement avant commit :
  13/13 tests + verdict=PASS sous qemu (dense 5,8 s). Chaque run CI est donc
  une vérification x86↔ARM réelle du contrat.
- **Softmax portable branché dans la tape AD (opt-in)** :
  `Var::softmax_portable()` / `Tensor::softmax_portable()` +
  `Op::SoftmaxPortable` (reverse.rs et parallel.rs) — forward via
  `portable_f32::softmax_f32`, backward **depuis la sortie stockée** (aucun
  appel libm dans le jacobien) ⇒ nœud complet forward+gradient bit-exact
  inter-plates-formes. Le softmax libm existant est inchangé (aucune
  régression d'empreintes). Tests : forward bit-identique à portable_f32,
  gradient ≈ gradient libm (1e-5) + empreinte gradient figée
  0x5ba09810fa590787 (contrat cross-platform du backward).

## Session 2026-07-10 — volet 111 : audit de couverture RepDL + fermeture des écarts (clean-room)
- **Audit complet** : `AUDIT_REPDL_2026-07-10.md` — matrice élément par élément des
  23 items de l'API publique de RepDL (ops/func/nn/optim/utils/from_torch_module)
  contre SciRust. 18 déjà couverts (dont conv2d + les 2 gradients, BatchNorm1d/2d,
  CrossEntropy, softmax, réductions 1D/2D, Adam), 2 couverts par composition
  (sum/mean 4D dims 0,2,3 via reshape ; expand_as via broadcast 2D), 1 N/A par
  conception (`from_torch_module` — SciRust n'est pas une surcouche PyTorch ;
  safetensors/SRT1/ONNX-JSON en tiennent lieu), 3 écarts réels **fermés ce volet**.
- **Écarts fermés (implémentations clean-room, specs publiques uniquement)** :
  (1) `Adam::with_amsgrad()` (Reddi et al. 2018 ; buffer v_max bias-corrigé ;
  oracle de convergence + test anti-pic pas AMSGrad < 10 % pas Adam) dans
  `scirust-core/src/autodiff/optim.rs` ; (2) `scirust_runtime::hash` —
  `sha256_hex_f32/tensor/state_dict` (équiv. `repdl.utils.get_hash`, encodage LE
  des bits IEEE-754, clés triées, 5 tests) ; (3) `reproducible::{exp_via_f64,
  ln_via_f64}` (même classe de technique que RepDL, doc honnête : fidèlement
  arrondi, déterminisme inter-plates-formes probable non prouvé — le CR prouvé en
  Rust pur reste travail futur du volet 108).
- **Copyright : zéro risque constaté et politique actée** — aucun code RepDL dans
  le dépôt (grep exhaustif : 7 fichiers documentaires seulement, citations) ;
  audit mené sur spécification (API + prose + arXiv:2510.09180), aucun code lu →
  copié/traduit ; règle écrite §3 de l'audit : jamais de copie/traduction de code
  RepDL (MIT ≠ incompatible mais attribution + confusion PolyForm), clean-room
  systématique.
- **Voie f32 portable (2e lot de la session, demande utilisateur)** :
  `scirust-core/src/portable_f32.rs` — exp/ln/softmax/dot/gemm **bit-exacts
  inter-plates-formes par construction** (Rust pur, zéro libm, opérations
  IEEE-754 de base en ordre fixe ; exp = réduction k·ln2 hi/lo + Taylor 13,
  ln = mantisse [√2/2,√2] + série atanh, interne f64 ⇒ fidèlement arrondi,
  ≤ 1 ulp vérifié vs oracle libm sur 200 k points). 13 tests : goldens
  bit-à-bit + empreintes FNV du balayage complet de l'espace f32 (pas 65 537 ;
  exp 0x71e63f5e1688a7f1, ln 0x8892b8ba72ffb8b6) = contrat de portabilité à
  exécuter sur ARM ; identiques debug/release. Clean-room (méthodes maths
  publiques, coefficients 1/n! ; aucun code fdlibm/musl/RepDL consulté —
  un algorithme n'est pas protégeable, seule son expression l'est).
  `paper/RELATED_WORK.md` mis à jour (voie portable réalisée ; CR *prouvé*
  reste futur) + post-scriptum dans l'audit.
- Reste ouvert : CI aarch64 = check only (exécuter les empreintes portable_f32
  sur ARM dès qu'un runner existe) ; SIMD/GPU f32 en tolérance (pas bit-exact) ;
  arrondi correct PROUVÉ des transcendantales = travail futur ; brancher
  softmax_f32 dans la tape si la portabilité f32 devient un objectif produit.
- NB numérotation : cette session s'était initialement étiquetée « volet 109 » ;
  renumérotée 111 au merge (109 = Correctness '26 et 110 = identité, mergés avant).

## Session 2026-07-10 — volet 110 : acteur CHECKUPAUTO → TAREK ZEKRITI
- **Décision utilisateur (définitive)** : l'acteur CHECKUPAUTO est remplacé par
  TAREK ZEKRITI. Appliqué : identité git locale (`user.name` TAREK ZEKRITI,
  `user.email` zekrititarek@gmail.com — scirust ET CCOS_EXTENDED) ; champ
  `authors` de scirust-burn-bridge → "Tarek Zekriti" ; toutes les URLs/slugs
  GitHub `CHECKUPAUTO/*` (26 fichiers : Cargo.toml des crates, README, LICENSE.md,
  RELEASING, SBOM, rapports techniques ×8 langues, scripts, scirust-rsi docs,
  SARIF de scirust-som) → `Memorithm/*` (l'org qui héberge — un nom de personne
  n'est pas une URL valide).
- **2e passe (confirmée par l'utilisateur : « oui continu »)** : marque aussi —
  emails `contact@checkupauto.fr` → `zekrititarek@gmail.com` (LICENSE, LICENSING,
  SECURITY, plaquette, en-têtes des rapports ×8) et SPDX
  `LicenseRef-CheckupAuto-Dual` → `LicenseRef-TarekZekriti-Dual` (LICENSE +
  3 Cargo.toml ; deny.toml/SBOM sans référence). git grep -i checkupauto → 0
  hors entrées narratives CHANGELOG/LIVESTATE.
- **Reste** : sweep équivalent dans CCOS_EXTENDED (dépôt séparé) — fait dans la
  même session (PR Memorithm/CCOS_EXTENDED#7, deps git + 3 Cargo.lock basculés,
  `cargo metadata --locked` vert ×3). NB fusion : le volet 109 (Correctness '26,
  mergé avant celui-ci) contient `paper/correctness26/` — occurrences CHECKUPAUTO
  résiduelles de ce nouveau contenu balayées dans le commit de merge.

## Session 2026-07-10 — volet 109 : draft de soumission Correctness '26
- **Contexte** : PR #268 (volet 108) MERGÉE dans master. Décisions utilisateur actées :
  bug reports extérieurs clos ; pas de re-licence JOSS ; plateforme d'évaluation du
  paper = Jetson AGX Thor ; GO d'écriture. Venue identifiée et vérifiée en ligne :
  **Correctness '26** (SC26 Chicago), **deadline 23 JUILLET 2026**, notification 1/9,
  ACM sigconf 7-8 pages hors réfs (repli court 4 p). Thèmes du CFP alignés
  (« contrôle du non-déterminisme » listé explicitement).
- **Livré** : `paper/correctness26/` — `main.tex` (draft complet ~8 p : abstract,
  intro déterminisme-comme-évidence + contributions, related work (pivot RepDL),
  §3 régimes + σ, §4 training T1-T4, §5 inférence/artefacts, §6 int8/NEON, §7 gate σ
  + étude négative dead guards, §8 coût mesuré (table Thor+x86, empreintes
  cross-platform, threats to validity), §9 limitations, §10 conclusion, table*
  claims→évidence) ; `references.bib` (7 réfs, métadonnées VÉRIFIÉES sur arxiv.org
  le 2026-07-10 : RepDL=Xie/Zhang/Chen 2025, EasyScale=Li et al. 2022,
  GPU numerics=Zahid/Laguna/Le 2024) ; `README.md` (build + TODO soumission).
  Pas de TeX dans le conteneur → contrôle structurel python (begin/end, accolades,
  cites↔bib, refs↔labels : tout équilibré/résolu) ; compilation à faire sur
  Overleaf/latexmk.
- **TODO soumission (humain)** : affiliation exacte ; lien artefact relecteurs ;
  anonymat selon CFP ; contrôle de longueur post-compilation (couper §2/§7 si > 8 p).
- Branche claude/new-session-n8bf71 repartie de master post-merge (protocole
  branche-mergée) ; nouvelle PR draft.

## Session 2026-07-10 — volet 108 : honnêteté README (RepDL) + étude « dead guards » (NO-GO) + positionnement paper
- **Lot 1 (bloquant, fait)** : claim d'unicité « No mainstream framework ships this
  guarantee tested » falsifiée par RepDL (arXiv:2510.09180, reproductibilité bit-à-bit
  cross-platform f32 par arrondi correct, surcouche PyTorch). Remplacée partout
  (README.md, docs/INDUSTRIAL_ROADMAP.md, docs/DOSSIER_FINANCEURS.md ; entrée 0.14.0 du
  CHANGELOG rectifiée avec note datée) par la formulation « à notre connaissance, seul
  framework auto-contenu (100 % Rust, zéro FFI) offrant simultanément multi-thread
  bit-identique CI + int8 embarqué + pièces d'audit ; RepDL plus fort sur cross-platform
  f32 ». Preuve : `grep -rn "No mainstream framework"` et `"framework grand public"` → 0.
  Le rapport technique ne portait PAS la claim (« treat as best-effort » — défendable).
- **Lot 2a (fait)** : `epsilon-audit --mine <dir>` — nouveau module public
  `scirust-sigma::mine` (std-only). M1 = f32 sous `f32::MIN_POSITIVE` (flush FTZ/DAZ),
  M2 = sous `1/f32::MAX` (1/d → inf sans FTZ), M2 ⊂ M1 classés séparément. Typage :
  Rust suffixe/ligne ; C nu = double (jamais compté) ; shaders f32 par défaut. Piège
  résolu : comparaison au seuil sur la valeur ARRONDIE f32 (`1.17549435e-38` parse f64
  sous σ exact mais arrondit À `f32::MIN_POSITIVE` → garde licite, non capturée).
  Exclusions test*/bench*/vendor ; fast-math (`-ffast-math`, `use_fast_math`,
  `-funsafe-math-optimizations`, `ftz`) dans les fichiers de build. 27 tests fixtures.
  fmt + clippy -D warnings + 27 tests verts. Gate `--check` existant intact.
- **Lot 2b (fait)** : campagne `/tmp/mining` sur 22 dépôts (SHA notés dans l'étude),
  clones sparse pour les géants, 22 450 fichiers / 9 160 848 lignes. 14 candidats bruts,
  TOUS relus en contexte : 12 = tolérances de test approx de ndarray (`#[cfg(test)]`
  inline dans src/), 2 = constantes `*_SUBNORMAL_F32` du lexer WGSL de naga (tests).
  → **0 CONFIRMED_DEAD_GUARD, verdict NO-GO** (règle ≥3 confirmés dans ≥2 dépôts).
  Modèle de menace FTZ néanmoins confirmé : 9/22 dépôts activent fast-math/FTZ en build
  (ggml/llama.cpp/whisper CPU+CUDA+HIP, pytorch QNNPACK, tensorflow kernels mlir ftz,
  candle flash-attn, OpenBLAS power, ncnn, tract bench). Étude complète, méthodo, table
  SHA+LOC, findings, limitations, règle de décision : `docs/DEAD_GUARDS_STUDY.md`.
  Aucun bug report posté (branche GO non prise) ; aucune issue/PR externe.
- **Lot 3 (fait)** : `paper/RELATED_WORK.md` (FR, citable ; pivot RepDL honnête ;
  motivation voie sanitized via arXiv:2410.09172 ; PAS de section prévalence — NO-GO)
  + `paper/PAPER_PLAN.md` (titre + 2 variantes ; recommandation VENUE : atelier
  correctness/reproducibility d'abord — **JOSS bloqué par la licence PolyForm
  non-OSI** ; table claims→évidence T1-P1 avec test exact + commande par claim ;
  TODO-EVIDENCE : R4 fingerprint thread-count en CI, S2 gate epsilon-audit à câbler
  en job CI, O1 banc overhead ordre-figé vs libre ; réponses rapporteurs a/b/c).
- **Décisions actées (recommandations acceptées par l'utilisateur)** : (1) bug reports
  extérieurs : clos, zéro contact (NO-GO) ; (2) venue : atelier correctness/reproducibility,
  PAS de re-licence pour JOSS ; (3) paper : GO conditionnel engagé → S2/R4/O1 exécutés :
  - **S2 fait** : job CI `epsilon-audit` (gate `--check` σ_sanitized sur scirust-gpu/src)
    ajouté à `.github/workflows/ci.yml`.
  - **R4 fait** : `scirust-runtime/tests/fingerprint_thread_invariance.rs` — fingerprint
    du forward bit-identique sous pools rayon 1/2/4/8 (batches synthétiques entiers,
    modèle construit DANS la fermeture install : Sequential contient des Box<dyn Module>
    non-Send). rayon en dev-dep de scirust-runtime (déjà au lockfile). Test vert.
  - **O1 fait (volet x86)** : `scirust-core/src/bin/bench_reduction_overhead.rs` —
    ordre figé (slots indexés, pattern train_batch_threaded) vs ordre d'arrivée (canal
    mpsc), ±1e16, empreintes bit-à-bit. Mesure x86 4 cœurs release (dim=100 352,
    30 reps) : figé/arrivée = 0,930×/0,895×/0,756×/0,846× à 1/2/4/8 threads → le
    déterminisme du pattern de réduction est GRATUIT (même plus rapide : pas de canal,
    pas de contention) ; empreinte figée unique par n ; baseline arrivée = 3 empreintes
    distinctes à 8 threads (non-déterminisme observé). PAPER_PLAN §4 (R4/S2/O1 → [CI]/
    mesuré) et §6 (décisions) mis à jour. **Reste humain** : atelier précis + run Jetson
    du banc O1 + déclenchement de l'écriture du paper.
  - **Protocole Jetson prêt** : `scripts/bench-o1-jetson.sh` (auto-contenu : rapport
    plateforme avec nvpmodel consigné, `--pin-clocks` explicite jamais silencieux,
    3 runs du banc, tests natifs Q3 NEON + R4 fingerprint, bundle d'évidence horodaté
    ignoré par git). Smoke-testé sur x86 : chiffres cohérents avec le run initial ET
    empreintes « figé » identiques entre processus (0x60daf62c…/0x9bf7c3f3…/
    0xd5b8e15f…/0x7e99a9d0… aux 4 nombres de threads) — stabilité inter-processus du
    banc vérifiée. Section « On-device Jetson bench (O1) » ajoutée à TEST_PROTOCOL.md.
  - **O1 volet Jetson EXÉCUTÉ (2026-07-10, par l'utilisateur, sortie collée en session)** :
    Jetson AGX Thor Dev Kit, 14 cœurs, 128 Go, L4T R38.4.0, noyau 6.8.12-tegra, MAXN
    (horloges épinglées par la 1re invocation), rustc nightly af3d95584, commit 0c2f1bf,
    3 runs × 30 reps. Overhead figé/arrivée : ≈0,99× (1 thr), 0,93-0,95× (2), 1,01-1,03×
    (4), 1,06-1,11× (8) ; arrivée non déterministe (2 empreintes distinctes à 8 thr,
    runs 2-3). **RÉSULTAT CLÉ : empreintes « figé » bit-identiques x86_64 ↔ aarch64**
    aux 4 nombres de threads — la réduction f32 à ordre figé est cross-platform, mesuré.
    Deux corrections de script au passage : (1) sous sudo, secure_path n'a pas
    ~/.cargo/bin → le script source $HOME/.cargo/env (et /home/$SUDO_USER/.cargo/env) ;
    (2) Q3 : `cargo test -p scirust-core <filtre>` exécute TOUS les targets et le
    tail -4 n'affichait que le dernier résumé (0 match) → `--lib` ajouté (sur x86 :
    « 697 filtered out » confirme le bon target ; re-run Jetson conseillé pour
    l'évidence Q3 propre). PAPER_PLAN O1 + §6 mis à jour (volet Jetson : fait).
  - **Re-run Jetson au commit 014795f (script corrigé, --pin-clocks opérationnel)** :
    Q3 EXÉCUTÉ SUR CIBLE — `quantization::tests_neon::neon_matches_scalar_bit_exact
    ... ok` (1 passed, 697 filtered out) + R4 vert nativement sur ARM. Banc reconfirmé
    (0,93-1,01× à 1-4 thr, 1,06-1,10× à 8), mêmes 4 empreintes → identité cross-platform
    x86_64 ↔ aarch64 revérifiée sur run indépendant. Toutes les lignes de la table
    claims→évidence relevant du Jetson sont désormais adossées à une exécution réelle.

## Session 2026-07-09 — volet 107 : déterminisme — bornes σ (`scirust-sigma`) + audit epsilon
- **Nouvel invariant nommé (déterminisme)** : `scirust-sigma` (crate feuille, ZÉRO dépendance
  externe, `std` seul) encode σ = « couvercle de zéro » par régime numérique. σ = plus petit
  positif représentable dans la voie : entier `1`, Q15.16 `2⁻¹⁶`, Q31.32 `2⁻³²`, f32 sanitized
  `f32::MIN_POSITIVE`, f32/f64 bruts = plus petit sous-normal. Invariant central
  `is_valid_guard_f32` : une garde anti-zéro SOUS σ est morte sur la voie 3 (sanitize_f32 l'écrase).
- Constantes `SIGMA_*` + `sigma_f32/f64` + `guard_denominator_f32/f64`. Bords (0/négatif/NaN/régime
  sans σ f32) définis+testés. 12 tests lib (valeurs bit-à-bit) + 1 test d'alignement qui affirme,
  SANS coupler à scirust-gpu, que le seuil de `sanitize_f32` == `SIGMA_SANITIZED_F32` bit-à-bit
  (casse si l'un bouge sans l'autre). N'a PAS touché `sanitize_f32` ni les defaults Adam/AdamW/Lion.
- **Binaire `epsilon-audit`** (std-only ; `sha2` déjà au lockfile pour sceller le rapport) : scanner
  lexical maison (hors commentaires/chaînes), classe ~14 425 littéraux `<1.0` en A/B/C/D/U →
  `docs/EPSILON_AUDIT.md` (déterministe, SHA-256 en pied, reproductible bit-à-bit vérifié).
  Totaux : A=112, B=364, C=10229, D=25, U=3695. Ne modifie AUCUN fichier (lecture seule).
- **Gate CI (une ligne à ajouter au job `clippy`/nouveau job)** :
  `cargo run -q -p scirust-sigma --bin epsilon-audit -- --root . --check`
  → exit ≠ 0 ssi une garde f32 sous σ_sanitized subsiste hors test dans `scirust-gpu/src`.
  **Exit 0 vérifié** sur l'arbre actuel (686 littéraux gpu/src, 0 violation, 0 warning ambigu).
- **Sécurité** : contrôle préventif d'un défaut de sûreté latent (garde morte → Inf/NaN silencieux,
  invisible en revue), zéro nouveau crate (deny.toml/lockfile intacts), rapport scellé SHA-256.
- **Migration σ future (top candidats B, hors périmètre de ce volet)** : 9 gardes `1e-300` en f64
  (hmm.rs, svd.rs:98, fdd.rs:40, chain.rs:254, optimize.rs:298) — mortes si un jour portées sur la
  voie f32 sanitized. Audit d'abord, migration ensuite.
- docs : CHANGELOG + LIVESTATE + docs/EPSILON_AUDIT.md. fmt+clippy(-D warnings)+13 tests verts.

## Session 2026-06-18 — volet 106 : simulateur quantique MPS / Tensor-Train
- `quantum::Mps`/`MpsNode` : état n-qubits en chaîne de tenseurs rang-3 (au lieu de 2ⁿ) ⇒ coût
  O(n·χ³) tant que l'intrication est modérée. Porte 1q in place ; porte 2q = contraction θ +
  application 4×4 + SVD tronquée (tn::ops::truncated_svd, Rust pur/nalgebra, ZÉRO FFI) à χ.
  Amplitudes réelles f32 (H/X/Z/CNOT/CZ/Ry) ; complexe = futur.
- Décision archi (réponse au msg validation) : PAS d'openblas/cuSOLVER (FFI, brise zéro-FFI +
  déterminisme) ni faer (redondant nalgebra). SVD maison existante. Le code de Gemini avait un
  pseudo_svd MOCK + la porte non appliquée → corrigés (vraie SVD + vraie application).
- Réutilise : la TT-compression existe déjà (tn::tt_decompose, nn::tt_linear) ⇒ même machinerie.
- Bibliothèque seule (nouveau module quantum). Pas de CLI ni multilingue (pour l'instant).
- Tests (4, core) : MPS ≡ statevector dense (circuit aléatoire 5q/40 portes, oracle vérité-terrain)
  + Bell (bond 2) + GHZ + troncation saine (produit→bond1, cap χ=1 fidélité>0.5) + déterminisme.
- docs : CHANGELOG. 624 tests core (+4) ; 8 gates verts (à confirmer).

## Session 2026-06-18 — volet 105 : synergie — commandes CLI kvcache/guard/attest (8 langues)
- `scirust-cli::synergy` : 3 commandes (kvcache/guard/attest) exposant les primitives de synergie,
  déterministes par --seed. kvcache : ratio compression + fidélité cosinus attention (+ --budget
  soft-paging) ; guard : couverture empirique ≥ 1−α + verdicts ; attest : journal hash-chaîné +
  vérif + rejet falsification + tamper-evidence.
- Nouvelles commandes CLI ⇒ docs multilingues : docs/REFERENCE.md (3 lignes) + les 8
  Documentation*.md (FR base + EN/AR/DE/ES/JA/KO/ZH, 3 bullets chacun, via sous-agents).
- Vérifié en exécution : kvcache 2.67× + cosinus 0.99998 (budget 32 ⇒ 96 évincés) ; guard 91.3%
  (≥90%) Accept/Abstain/Reject ; attest chaîne OK + falsification rejetée + têtes distinctes.
- docs : CHANGELOG + REFERENCE + Documentation×8. 8 gates verts (à confirmer).

## Session 2026-06-18 — volet 104 : synergie SLHAv2 — codec KV accéléré SIMD (bit-exact)
- `scirust_simd::ops::dequantize_int4_into` (câblé dans nn::elastic_kv_cache::dequantize_int4) :
  déquant INT4 via kernel SIMD mul_f32 ; élémentaire (pas de réduction) ⇒ bit-identique scalaire et
  inter-plateformes (déterminisme préservé ; réductions cosinus/attention restent scalaires).
- Bibliothèque seule (scirust-simd + core). Pas de CLI ni multilingue.
- Tests (1, scirust-simd) : SIMD ≡ scalaire bit-exact pour toute longueur (y compris <1 lane) +
  plage d'échelles. (gate 5 portable-simd + gate 4 AVX2 runtime.)
- docs : CHANGELOG. 620 core + 1 simd ; 8 gates verts (à confirmer).

## Session 2026-06-18 — volet 103 : synergie CCOS — guard à garantie statistique
- `nn::guard::StatisticalGuard` : porte Accept/Abstain/Reject sur l'ensemble conforme (#21).
  Couverture vraie classe ≥ 1−α sans hypothèse de distribution ⇒ pour le guard.rs de CCOS.
- Bibliothèque seule (pas de CLI ni multilingue). Nouveau module nn::guard.
- Tests (2, core) : couverture empirique ≥ 1−α (fraîches, déterministe) ; verdicts
  confiant→Accept / partagé→Abstain / plat→Reject.
- docs : CHANGELOG. 620 tests core (+2) ; 8 gates verts (à confirmer).

## Session 2026-06-18 — volet 102 : synergie CCOS — pont d'attestation hash-chaîné
- `scirust_runtime::attest` : journal d'inférences hash-chaîné (SHA-256), rejouable, à la forme
  de l'event_log de CCOS. InferenceEvent {seq, engagement modèle, hash entrée/sortie, entry_hash} ;
  entry = H(prev ‖ seq ‖ commitment ‖ in ‖ out). attest_and_record vérifie l'authenticité
  (Freivalds vinfer #80) AVANT d'ajouter ⇒ chaîne d'inférences réelles, ingérable par CCOS.
- Bibliothèque seule (scirust-runtime ; pas de CLI ni multilingue). Nouveau module.
- Tests (3, runtime) : chaîne vérifie + replay (même tête) ; falsification/réordonnancement
  détectés ; inférence authentique attestée + falsifiée rejetée (journal inchangé).
- docs : CHANGELOG. 618 core + 7 runtime (+3) ; 8 gates verts (à confirmer).

## Session 2026-06-18 — volet 101 : synergie CCOS/SLHAv2 — KV-cache compressé élastique
- `nn::elastic_kv_cache` : primitive partagée SLHAv2 (compression KV-cache CPU) + CCOS (paging
  borné). Tuile INT4 deux niveaux (base + résidu, « residual tracking » SLHAv2) à **échelles
  adaptatives par groupe** (cosine-aware, quantize_int4_grouped) ⇒ fidélité cosinus >0.99 ;
  ElasticKvCache à budget (évince la plus ancienne, soft-paging ; new_grouped pour le grouping) ;
  attention via contiguous_attention (#63) ⇒ écart = uniquement l'erreur de compression. Codec
  exposé (quantize_int4[_grouped]/dequantize/KvTile/cosine_similarity) pour SLHAv2/CCOS.
- Contexte : SLHAv2 (public) = compression KV-cache déjà bâtie sur un crate « scirust »
  (SciRustSlhaTile) ; CCOS = causal-context OS (graphe causal pagé, event-log hash-chaîné).
  Première brique de synergie ; suites possibles : pont d'attestation/event-log (vinfer #80 + DiFR
  #5 → event_log CCOS), guard à garantie statistique (conformal/ensembles), signature PQ (SLHAv2).
- Bibliothèque seule (pas de CLI ni multilingue). Nouveau module nn::elastic_kv_cache.
- Tests (5, core) : fidélité cosinus >0.95 (résidu > base) + déterminisme ; **grouping cosine-aware
  améliore la fidélité de base** (magnitudes hétérogènes, tile jamais pire) ; attention compressée ≈
  pleine (cosinus >0.99) ; ratio ≥3× vs f32 ; budget borné + bit-exact.
- docs : CHANGELOG. 618 tests core (+5) ; 8 gates verts (à confirmer).

## Session 2026-06-18 — volet 100 : Reluplex (#4) — vérification complète SMT — ROADMAP 80/80 ✅
- `nn::ibp::reluplex_verify`/`reluplex_unstable_count` (Katz 2017) : recherche SAT d'un
  contre-exemple par case-splitting **paresseux** des phases ReLU (neurones stables = phase forcée,
  jamais scindés ; seuls les instables scindés ⇒ 2^instables feuilles vs 2^cachés du MILP). Feuille
  = patron affine, LP 2D exact (partagé #31). Distinct de #26 (split entrée) et #31 (eager).
- Bibliothèque seule (pas de CLI ni multilingue). Module nn::ibp.
- Tests (3, core) : accord avec MILP (balayage rayons, 2 méthodes exactes) ; contre-exemple réel
  (SAT) ; scinde moins que tous les neurones (élimination par bornes) à petit rayon ; déterministe.
- docs : roadmap #4 📋→✅ ; CHANGELOG. 613 tests core (+3) ; 8 gates verts (à confirmer).
- **🎉 LES 80 ITEMS DE LA ROADMAP RECHERCHE SONT ✅ (80/80).**

## Session 2026-06-18 — volet 99 : Inférence vérifiable (#80) — Freivalds + engagement + Fiat-Shamir
- `scirust_runtime::vinfer` : modèle linéaire entier sur GF(2³¹−1) engagé par hash ; vérif sortie
  batchée Y par Freivalds (W·(X·r)=Y·r, r par Fiat-Shamir de hash(engagement,X,Y)). Compact
  (O(out·in+in·b) vs O(out·in·b)), sain (faux Y passe ≤ (1/p)^k). Pas de ZK (vérifieur a les poids).
- Bibliothèque seule (scirust-runtime ; pas de CLI ni multilingue). Nouveau module.
- Tests (4, runtime) : accepte inférence correcte + déterministe ; 1000 falsifications toutes
  rejetées (soundness) ; engagement lie le modèle ; Fiat-Shamir lie la sortie (sortie d'autres
  entrées rejetée).
- docs : roadmap #80 📋→✅ ; CHANGELOG. 610 core + 7 runtime ; 8 gates verts ✓ ; commit 4dc3436.

## Session 2026-06-18 — volet 98 : DiFR (#5) — vérification d'inférence malgré le non-déterminisme
- `scirust_runtime::difr::difr_verify` (2025) : référence canonique via reproducible_dot (f64,
  indépendant de l'ordre) + enveloppe d'erreur FP saine (γ·Σ|termes| propagée, ReLU 1-Lipschitz).
  Accepte tout ordre de sommation f32 honnête, rejette la falsification au-delà de l'enveloppe.
  Prolonge proof bit-exact (qui rejetterait une sortie honnête entre matériels). Nouveau module.
- Bibliothèque seule (scirust-runtime ; pas de CLI ni multilingue).
- Tests (3, runtime) : accepte un ordre de sommation différent ; enveloppe saine (1000 ordres
  tous acceptés) & fine (<0.001 échelle) ; rejette falsification (change la classe) + déterminisme.
- docs : roadmap #5 📋→✅ ; CHANGELOG. 610 core + 3 runtime ; 8 gates verts ✓ ; commit 831746a.

## Session 2026-06-18 — volet 97 : MILP (#31) — vérification exacte
- `nn::ibp::milp_min_margin`/`milp_verify_robustness` (Tjeng 2019) : réseau ReLU 2-entrées
  1-couche. Patrons d'activation ReLU = binaires MILP ⇒ énumérés ; par patron le réseau est affine,
  marge logitₜ−logⱼ minimisée sur boîte ∩ demi-espaces d'activation par énumération de sommets 2D
  (lp_min_2d). Min global exact ; >0 ⇒ Robust sinon Unsafe(contre-exemple exact).
- Bibliothèque seule (pas de CLI ni multilingue). Module nn::ibp.
- Tests (2, core) : = force brute (grille 120², min ≤ tout échantillon + proche) + témoin atteint
  + déterminisme ; contre-exemple réel (grande boîte) + borne ≥ DeepPoly (sain) & strictement plus
  serré quelque part.
- docs : roadmap #31 📋→✅ ; CHANGELOG. 610 tests core (+2) ; 8 gates verts ✓ ; commit 2e160fa.

## Session 2026-06-18 — volet 96 : Branch-and-bound (#26) — vérification complète
- `nn::ibp::verify_robustness`/`BabResult` (GCP-CROWN, Zhang 2022) : BaB sur le domaine d'entrée.
  Marges par classe (fusionnées en dernière couche) bornées par DeepPoly ; toutes >0 ⇒ Robust ;
  sinon sonde centre ⇒ contre-exemple ; sinon scinde axe le plus large + récurse. Décide
  (Robust/Unsafe/Unknown). Split ReLU + plans coupants NON implémentés (documenté).
- CLI : exposé dans `certify`. Pas de nouvelle commande ⇒ pas de multilingue.
- Tests (3, core) : Robust sain (5000 pts) + déterministe ; rayon BaB > DeepPoly seul (+ région
  sup. saine, 3000 pts) ; Unsafe = vrai contre-exemple (mal classé, dans boîte).
- docs : roadmap #26 📋→✅ ; CHANGELOG. 608 tests core (+3) ; 8 gates verts ✓ ; commit e10de3a.

## Session 2026-06-18 — volet 95 : DeepPoly (#28) — domaine abstrait relationnel
- `nn::ibp::deeppoly_certify`/`IbpMlp::certify_deeppoly` (Singh 2019) : bornes basse/haute
  **affines en les entrées** par neurone (back-substitution), relaxation ReLU asymétrique
  (corde sup (u/(u−l))(y−l), inf λy à aire min λ=1 si u>−l sinon 0). Plus serré qu'IBP à toute
  profondeur (vs crown_bounds limité 2 couches).
- CLI : exposé dans `certify` (à côté IBP/CROWN/zonotope/smoothing). Pas de nouvelle commande
  ⇒ pas de multilingue.
- Tests (2, core) : sain (4000 pts ∈ boîte, MLP 3 couches) + déterministe ; plus serré qu'IBP
  sur relu(x)+relu(−x)=|x| (DeepPoly exact [0,1] vs IBP [0,2]).
- docs : roadmap #28 📋→✅ ; CHANGELOG. 605 tests core (+2) ; 8 gates verts ✓ ; commit dec7fbc.

## Session 2026-06-18 — volet 94 : CROWN-IBP (#30) — entraînement certifié
- `nn::crown_ibp::CrownIbpMlp` (Zhang 2020) : propagation IBP **différentiable** sur la tape
  (centre·W+b, rayon·|W| avec |W|=relu(W)+relu(−W) ; ReLU-intervalle [relu(l),relu(u)]) ⇒ logits
  robustes (vraie classe borne inf, autres borne sup), loss = cross-entropy ⇒ réseau prouvablement
  robuste. Mesure du rayon certifié via IbpMlp (plain f32) existant. Nouveau module.
- Bibliothèque seule (entraînement, pas de CLI ni multilingue).
- Tests (2, core) : IBP tape ≡ IbpMlp référence + sain (2000 pts ∈ boîte) ; rayon certifié croît
  (robuste-entraîné >> accuracy-only, +0.2 ℓ∞, tous deux 100 % justes) + déterminisme.
- docs : roadmap #30 📋→✅ ; CHANGELOG. 603 tests core (+2) ; 8 gates verts ✓ ; commit a185017.

## Session 2026-06-18 — volet 93 : Sophia (#44) — optimiseur 2e ordre clippé
- `nn::nd_optim::NdSophia` (Liu 2023) : θ←θ−lr·clip(m/max(γ·h,eps),ρ), h=EMA Hessienne diagonale
  par Hutchinson (ĥ=v⊙Hv, v∈{±1} seedé) via produit Hessien-vecteur en différences finies
  (Hv≈(∇L(θ+εv)−∇L(θ))/ε, exact pour quadratique). Ancien blocage « abs tape op » infondé
  (clipping en f32 dans l'optimiseur). 2 gradients/pas (probe+step orchestrés) ⇒ hors lm --opt.
- Bibliothèque seule (comme SAM, hors boucle lm). Module nd_optim.
- Tests (1, core) : converge sur quadratique mal conditionné (courbures 4 vs 0.25, cond. 16)
  + déterminisme bit-exact (probe seedé).
- docs : roadmap #44 📋→✅ ; CHANGELOG. 601 tests core (+1) ; 8 gates verts ✓ ; commit 387a304.

## Session 2026-06-18 — volet 92 : QuIP# (#64) — incohérence Hadamard + lattice E8
- `quantization::quantize_quip`/`nearest_e8`/`random_hadamard_transform` (Tseng 2024) :
  (1) incohérence = Hadamard randomisée (signes ±1 seedés + FWHT, orthogonale) ⇒ étale aberrants,
  rétrécit la plage que les 2^bits niveaux couvrent (à budget égal) ; (2) codebook lattice E8
  (D8 ∪ D8+½, décodeur Conway-Sloane) plus dense que la grille cubique à densité égale.
- Bibliothèque seule (pas de CLI ni multilingue). Module quantization existant.
- Tests (3, core) : RHT orthogonale (round-trip) + réduit plage aberrants (<0.6×) ; E8 valide
  (coords alignées + somme paire) & < grille cubique en moyenne (4000 vecteurs) ; bout-en-bout
  QuIP# < RTN scalaire 2-bit sur poids à aberrants + déterminisme (codes identiques).
- docs : roadmap #64 📋→✅ ; CHANGELOG. 600 tests core (+3) ; 8 gates verts ✓ ; commit d1567d9.

## Session 2026-06-18 — volet 91 : AQLM (#70) — quantification additive multi-codebook
- `quantization::quantize_aqlm`/`AqlmResult` (Egiazarian 2024) : groupes de dim g, chaque groupe
  ≈ somme de M mots de code (un par codebook, K mots chacun) ; codebooks appris par k-means
  résiduel + optimisation alternée (ré-encodage glouton + ré-ajustement LS). Vectoriel ⇒ capte
  la structure inter-dim. Module quantization existant.
- Bibliothèque seule (pas de CLI ni multilingue).
- Tests (2, core) : < 0.7× RTN scalaire à ~2-bit égal (M·log₂K/g) sur poids structurés ;
  round-trip (longueur non divisible) + déterminisme (codes + bits identiques).
- docs : roadmap #70 📋→✅ ; CHANGELOG. 597 tests core (+2) ; 8 gates verts ✓ ; commit 40a5be0.

## Session 2026-06-18 — volet 90 : S5 (#52) — SSM MIMO + scan associatif parallèle
- `nn::nd_layers::s5_scan`/`s5_parallel_scan`/`NdS5` (Smith 2023) : SSM MIMO diagonal (état
  partagé n piloté par toutes entrées via B, lu via C ; hₜ=Ā⊙hₜ₋₁+xₜB, yₜ=hₜC) ; scan associatif
  Hillis-Steele (combine (a₁,u₁)∘(a₂,u₂)=(a₂a₁,a₂u₁+u₂), ordre doublage fixe ⇒ déterministe).
- Bibliothèque seule (couche gradient-checkée, pas de CLI ni multilingue). Module nd_layers.
- Tests (4, core) : scan parallèle ≡ séquentiel (aₜ variable ⇒ vrai associatif) ; s5_scan ≡
  référence MIMO ; gradient check (x,Ā,B,C) ; NdS5 entraîne (MSE↓ <0.6×) + déterminisme.
- docs : roadmap #52 📋→✅ ; CHANGELOG. 595 tests core (+4) ; 8 gates verts ✓ ; commit 55a31ad.

## Session 2026-06-18 — volet 89 : Mamba-2 / SSD (#50) — dualité espace-d'états ↔ attention
- `nn::nd_layers::ssd_dual`/`NdMamba2` (Dao & Gu 2024) : A scalaire par pas ⇒ récurrence
  Hₜ=aₜHₜ₋₁+xₜBₜᵀ, yₜ=HₜCₜ **exactement** = forme quadratique masquée Y=(L⊙CBᵀ)X,
  L[i,j]=∏_{j<k≤i}aₖ. cumlog = préfixe-somme (matmul triangulaire), L=exp(diff) masquée AVANT
  exp (diff⊙mask→exp→⊙mask) ⇒ pas d'inf·0=NaN. a_log=log a (=Δ·A), aucun op log.
- Bibliothèque seule (couche gradient-checkée, pas de CLI ni multilingue). Module nd_layers.
- Tests (3, core) : dual ≡ récurrence séquentielle (la dualité) ; gradient check (x,B,C,a_log) ;
  NdMamba2 entraîne (MSE↓ <0.6×) + déterminisme bit-exact.
- docs : roadmap #50 📋→✅ ; CHANGELOG. 591 tests core (+3) ; 8 gates verts ✓ ; commit ad0bfb2.

## Session 2026-06-18 — volet 88 : FNO (#75) — opérateur neuronal de Fourier
- `nn::fno::FnoSpectralConv1d`/`NdFno` (Li 2021) : DFT réelle = matrices cos/sin fixes (matmul
  déterministe, différentiable, sans FFT ni complexe) ; garder `modes` basses fréqs, poids
  complexe par mode R_k=Ar_k+iAi_k (mélange canaux via bmm), DFT⁻¹ (inverse unilatéral facteur-2).
  Bloc σ(spectral+local). Nouveau module nn::fno.
- Bibliothèque seule (couches gradient-checkées, pas de CLI ni multilingue).
- Tests (4, core) : reconstruction exacte band-limité (DFT⁻¹∘DFT) ; gradient check (v, Ar, Ai) ;
  **apprend la dérivation** (d/dx↔×ik) et généralise à phase non vue (MSE test <0.02, convexe) ;
  NdFno entraîne (MSE↓ <0.6×) + déterminisme. Bug harnais évité : 1 forward/tape (sinon un
  paramètre ré-inputé N fois ⇒ gradient éclaté sur N nœuds).
- docs : roadmap #75 📋→✅ ; CHANGELOG. 588 tests core (+4) ; 8 gates verts ✓ ; commit 451e6be.

## Session 2026-06-18 — volet 87 : Hyena (#56) — convolutions longues implicites + gating
- `nn::nd_layers::hyena_long_conv`/`NdHyena` (Poli 2023) : mélangeur sans attention. Conv
  causale par canal y[t,c]=Σ_τ h[τ,c]u[t−τ,c] = Σ_τ h[τ,:]⊙(Sτ·u) (matrices décalage Sτ
  constantes ⇒ différentiable sans scatter). Filtre **implicite** : MLP(encodage positionnel)
  ⊙ fenêtre exp(−γ·t̄) apprenable. Opérateur ordre 2 : z=x1⊙(h1*v), z=x2⊙(h2*z).
- Bibliothèque seule (couche gradient-checkée, pas de CLI ni multilingue). Module nd_layers.
- Tests (3, core) : conv ≡ référence causale écrite à la main ; gradient check (u, h) ;
  NdHyena entraîne (MSE↓ <0.6×) + déterminisme bit-exact.
- docs : roadmap #56 📋→✅ ; CHANGELOG. 584 tests core (+3) ; 8 gates verts ✓ ; commit 8ff9c98.

## Session 2026-06-18 — volet 86 : xLSTM (#57) — sLSTM scalaire + mLSTM matriciel
- `nn::nd_layers::slstm_scan`/`mlstm_scan`/`NdXlstm` (Beck 2024) : sLSTM (porte entrée
  exponentielle iₜ=exp(ĩₜ) + normaliseur nₜ, hₜ=oₜ⊙cₜ/nₜ ; tanh=2σ(2x)−1, sortie bornée
  (−1,1) ⇒ stable sans stabilisateur log omis) ; mLSTM (mémoire covariance d×d par produits
  externes, dénominateur max(|nₜ·qₜ|,1) **exact** via |a|=relu(a)+relu(−a), max(a,1)=relu(a−1)+1).
- Bibliothèque seule (couches gradient-checkées, pas de CLI ni multilingue). Module nd_layers.
- Tests (4, core) : mLSTM ≡ référence (dénominateur actif) ; gradient check sLSTM (4 portes)
  et mLSTM (q,k,v,iₜ,fₜ, régime lisse |nₜ·qₜ|<1) ; NdXlstm entraîne (MSE↓ <0.6×) + déterminisme.
- docs : roadmap #57 📋→✅ ; CHANGELOG. 581 tests core (+4) ; 8 gates verts ✓ ; commit 27bf173.

## Session 2026-06-18 — volet 85 : OmniQuant (#65) — clipping de poids apprenable
- `quantization::omniquant_quantize` (Shao 2024) : facteur de coupe γ∈(0,1] par canal
  (plage γ·max|w|), recherche sur grille incluant γ=1=RTN ⇒ ≤ RTN garanti.
- Bibliothèque seule (pas de CLI ni multilingue). Module quantization existant.
- Tests (2, core) : < RTN sur poids queue lourde (≥1 canal coupe) ; jamais pire que RTN
  (uniforme→RTN) + déterminisme (codes/scales identiques).
- docs : roadmap #65 📋→✅ ; CHANGELOG. 577 tests core (+2) ; 8 gates verts ✓ ; commit 38e3fb4.

## Session 2026-06-18 — volet 84 : S4 / S4D (#51) — espace d'états structuré diagonal
- `nn::nd_layers::s4_scan`/`NdS4` (Gu 2022) : SSM LTI diagonal (Ā=exp(Δ⊙A), B̄=Δ⊙B,
  h_t=Ā⊙h_{t−1}+B̄⊙x_t, y_t=Σ_n C⊙h_t) ; init HiPPO diag A[:,j]=−(j+1). Paramètres fixes (vs Mamba sélectif).
- Bibliothèque seule (couche gradient-checkée, pas de CLI ni multilingue). Module nd_layers.
- Tests (2, core) : gradient check (x, a_log, B, C, log_dt vs diff. finies, tol 3e-2) ;
  NdS4 entraîne (MSE↓ <0.6×) + déterminisme bit-exact.
- docs : roadmap #51 📋→✅ ; CHANGELOG. 575 tests core (+2) ; 8 gates verts ✓ ; commit aafa3b3.

## Session 2026-06-18 — volet 83 : AI²/zonotopes (#29) — domaine abstrait vérification
- `nn::ibp::Zonotope`/`IbpMlp::certify_zonotope` (Gehr 2018) : affine exact, ReLU DeepZ
  (λx+μ±μ, 1 générateur/neurone instable). Générateurs partagés ⇒ corrélations.
- CLI : exposé dans `certify` (à côté IBP/CROWN). Pas de nouvelle commande ⇒ pas de multilingue.
- Tests (3, core) : affine exact (=intervalle) ; soundness (4000 points ∈ boîte zonotope, MLP 3 couches) ;
  plus serré qu'IBP sous corrélation (relu(x)−relu(x)≡0 : zono [−0.5,0.5] vs IBP [−1,1], sains).
- docs : roadmap #29 📋→✅ ; CHANGELOG. 573 tests core (+3) ; 8 gates verts ✓ ; commit 7963ee5.

## Session 2026-06-18 — volet 82 : EAGLE (#62) — décodage spéculatif niveau features
- `nn::nd_decoder::EagleHead`/`generate_eagle` (Li 2024) : tête (feature, embed) → feature
  suivant, autorégressée + tête LM gelée ⇒ brouillon, vérifié (préfixe + correction).
  Ajout accesseurs `token_embedding`/`head_logits`/`d_model` ; `EagleHead::train` (MSE features, base gelée).
- Bibliothèque seule (algorithme de décodage, pas de CLI ni multilingue). Module nd_decoder.
- Tests (2, core) : exact = greedy pour tête **quelconque** (random) + déterminisme ;
  tête entraînée (séquence périodique) ⇒ blocs acceptent >1 token (forwards<2·n), exact.
  8/8 tests nd_decoder verts.
- docs : roadmap #62 📋→✅ ; CHANGELOG. 571 tests core (+2) ; 8 gates verts ✓ ; commit dc62241.

## Session 2026-06-18 — volet 81 : Medusa (#61) — décodage à têtes multiples
- `nn::nd_decoder::MedusaHeads`/`generate_medusa` (Cai 2024) : têtes (tête j → token +j+2
  depuis l'état caché) ⇒ brouillon multi-token 1 forward, vérifié (préfixe + correction).
  Ajout `NdDecoderLM::forward_hidden`/`forward_with_hidden` ; `MedusaHeads::train` (base gelée).
- Bibliothèque seule (algorithme de décodage, pas de CLI ni multilingue). Module nd_decoder.
- Tests (2, core) : exact = greedy pour têtes **quelconques** (random) + déterminisme ;
  têtes entraînées (séquence périodique mémorisée) ⇒ blocs acceptent >1 token (forwards<2·n),
  toujours exact. 6/6 tests nd_decoder verts.
- docs : roadmap #61 📋→✅ ; CHANGELOG. 569 tests core (+2) ; 8 gates verts ✓ ; commit 48459a0.

## Session 2026-06-18 — volet 80 : PagedAttention (#63) — KV-cache paginé
- `nn::paged_attention::PagedKvCache` (Kwon/vLLM 2023) : blocs d'un pool + table de blocs ;
  append/gather/attention indexée via la table (softmax(qKᵀ/√d)·V). reserve_decoy() = fragmentation.
- Bibliothèque seule (mécanisme interne, pas de CLI ni multilingue). Nouveau module.
- Tests (3, core) : gather bit-identique sous fragmentation (leurres interleavés) + comptabilité
  blocs ⌈len/bs⌉ ; attention paginée **bit-identique** au cache contigu (même ordre arith) +
  déterminisme ; cas vide + division exacte en blocs.
- docs : roadmap #63 📋→✅ ; CHANGELOG. 567 tests core (+3) ; 8 gates verts ✓ ; commit 66a48be.

## Session 2026-06-18 — volet 79 : DoRA (#73) — LoRA magnitude/direction
- `nn::dora::DoraLinear` (Liu 2024) : W'=m⊙(W₀+BA)/‖W₀+BA‖_col ; W₀ gelé, m/A/B entraînés.
  Backward de la normalisation par colonne en forme close (∂L/∂V=(m/‖V‖)(gw−u·s), ∂L/∂m=s).
- Bibliothèque seule (pas de CLI ni multilingue). Nouveau module nn::dora.
- Tests (3, core) : init B=0,m=‖W₀‖_col ⇒ W'=W₀ exact (+ forward = map de base) ;
  gradient check (diff. finies centrales eps=1e-3, tol 3e-2, params génériques B≠0) ;
  récupère une cible DoRA (perte ÷100, GD) + déterminisme.
- docs : roadmap #73 📋→✅ ; CHANGELOG. 564 tests core (+3) ; 8 gates verts ✓ ; commit 3521006.

## Session 2026-06-18 — volet 78 : GaLore (#48) — projection low-rank des gradients
- `nn::nd_optim::NdGalore`/`galore_subspace` (Zhao 2024) : Adam dans le sous-espace
  dominant rang-r du gradient (top-r vec. singuliers via jacobi_eigenvectors,
  rafraîchi tous update_gap pas) ⇒ états compressés rank×max(m,n). Vecteurs → Adam.
- CLI : `lm --opt galore` (famille d'optimiseurs). REFERENCE mise à jour.
- Tests (5 core + 1 CLI) : P orthonormal + projection orthogonale optimale (Pythagore,
  erreur↓ en r, nulle au rang plein) ; gradient bas-rang reconstruit exact (sous-rang→résidu) ;
  convergence sur cible bas-rang, état 2×4≠4×4 ; sous-rang n'atteint pas ; fallback vecteur=Adam ;
  déterminisme. lm_command vert (galore).
- docs : roadmap #48 📋→✅ ; CHANGELOG ; REFERENCE. 561 tests core (+5) ; 8 gates verts ✓ ; commit 2a41f3e.

## Session 2026-06-18 — volet 77 : YaRN (#60) — extension de contexte RoPE
- `nn::yarn` (Peng 2023) : `yarn_frequencies` (interpolation NTK-by-parts : garde
  haute fréq, interpole basse fréq θ/s, rampe γ), `rope_apply_freqs`/`rope_yarn`
  (rotation, convention emboîtée = RoPE existante), `yarn_attention_scale` (0.1·ln(s)+1).
- Bibliothèque seule (primitive positionnelle, pas de CLI ni multilingue). Nouveau module.
- Tests (6, core) : position relative préservée (⟨rope(q,m),rope(k,n)⟩=g(m−n)) ;
  angle basse-fréq à s·L revient exactement à l'entraînement (L) ; bornes NTK-by-parts
  (haute inchangée, basse=θ/s, rampe monotone) ; scale=1 ≡ RoPE simple ; déterminisme.
- docs : roadmap #60 📋→✅ ; CHANGELOG. 556 tests core (+6) ; 8 gates verts ✓ ; commit b875205.

## Session 2026-06-18 — volet 76 : Learn then Test (#37) — contrôle de risques multiples
- `nn::conformal::learn_then_test`/`hoeffding_pvalue` (Angelopoulos 2021) : contrôle
  distribution-free de risques multiples **arbitraires** (non emboîtés). p-value de
  Hoeffding p=exp(−2n(α−R̂)₊²) pour H₀:R(λ)>α + correction Bonferroni (p≤δ/m) ⇒ FWER≤δ.
  Plus général que RCPS #36 (pas d'hypothèse de monotonie). Réutilise l'infra conformal.
- Bibliothèque seule (pas de CLI ni multilingue). Module `nn::conformal` existant.
- Tests (3, core) : p-value forme close (saturation à 1) ; **FWER≤δ vérifié par
  simulation** (2000 essais, toutes configs sur frontière R=α : FWER mesuré ≤ 0,1 vs
  sélection naïve qui échoue >90 %) ; puissance (sûres retenues, non-sûres rejetées) +
  déterminisme. 16/16 tests conformal verts.
- docs : roadmap #37 📋→✅ ; CHANGELOG. 550 tests core (+3) ; 8 gates verts ✓ ; commit 8d0d766.

## Session 2026-06-17 — volet 75 : Rényi DP accountant (#78) — budget confidentialité
- `dp::gaussian_rdp`/`rdp_to_dp`/`rdp_gaussian_epsilon` (Mironov 2017) : RDP gaussien
  α/(2σ²) + conversion Mironov ε=RDP+ln(1/δ)/(α−1) optimisée sur α. Renforce DP-SGD.
- Bibliothèque seule (pas de CLI ni multilingue). Module `dp` existant.
- Tests (2, core) : RDP/conversion exactes (formes closes) ; ε ≪ composition basique
  (steps=100,σ=4,δ=1e-5 : ~15 vs ~143) + monotonie.
- docs : roadmap #78 📋→✅ ; CHANGELOG. 547 tests core (+2) ; 8 gates verts ✓ ; commit 1af31eb.

## Session 2026-06-17 — volet 74 : Watermark LLM (#79) — provenance auditable
- `nn::watermark` (Kirchenbauer 2023) : partition vert/rouge seedée (hash
  (seed,prev,token)), `apply_green_bias` (biais logits verts), `detect_z` (test z
  (g−γn)/√(nγ(1−γ)) sans accès au modèle).
- Bibliothèque seule (pas de CLI ni multilingue). Pilier audit/provenance (nouveau).
- Tests (3, core) : fraction verte ≈ γ ; biais sur tokens verts seulement ; texte
  filigrané détecté (z≫8) vs naturel (z≈0) + mauvais seed non détecté + déterminisme.
- Note : piège Rust 2024 `gen` mot-clé réservé (renommé `draw`) — déjà vu (RWKV).
- docs : roadmap #79 📋→✅ ; CHANGELOG. 545 tests core (+3) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 73 : DeepONet (#76) — apprentissage d'opérateurs
- `nn::deeponet::DeepONet` (Lu 2021) : G(u)(y)=Σ b_k(u)·t_k(y) ; trunk cosinus fixe +
  branch linéaire (POD-DeepONet) ⇒ convexe, exact pour opérateurs linéaires.
- Bibliothèque seule (pas de CLI ni multilingue). Standalone (GD analytique).
- Tests (2, core) : apprend l'antidérivée, généralise à fonctions non vues (MSE test
  < 0,01, < 0,1× baseline) ; déterminisme.
- docs : roadmap #76 📋→✅ ; CHANGELOG. 542 tests core (+2) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 72 : Deep Ensembles (#40) — incertitude épistémique
- `nn::ensemble::DeepEnsemble` (Lakshminarayanan 2017) : N MLP ReLU seedés (tape +
  NdAdam) ; predict→(moy, écart-type) = estimation + incertitude épistémique.
- Bibliothèque seule (pas de CLI ni multilingue). Réutilise NdLinear/relu/NdAdam.
- Tests (2, core) : MSE ensemble ≤ moy membres (Jensen) + écart-type ≫ OOD (x=4
  vs x=0) ; déterminisme bit-exact.
- docs : roadmap #40 📋→✅ ; CHANGELOG. 540 tests core (+2) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 71 : LLM.int8() (#71) — matmul mixte int8/fp32
- `quantization::int8_mixed_matmul` (Dettmers 2022) : colonnes de features outliers
  (>seuil) en fp32, reste en int8 ; X·W = int8(normal) + fp32(outlier). Réutilise
  compute_scale/quantize_tensor/matmul_int8.
- Bibliothèque seule (pas de CLI ni multilingue).
- Tests (2, core) : erreur vs fp < 0,5× int8 simple (activations à outliers ×75) ;
  sans outliers = int8 pur ; déterminisme.
- docs : roadmap #71 📋→✅ ; CHANGELOG. 538 tests core (+2) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 70 : fix SIMD — bug d'alignement (corrigé)
- `scirust-simd::portable` : `add_f32/f64_inplace`, `dot_f32/f64`, `fma_f32`
  découpaient chaque opérande indépendamment (`as_simd`) ⇒ lanes décalées si
  alignements différents ⇒ résultats **faux non déterministes** (flake
  `test_add_f32_inplace` ~30–50 %). Réécrit en `chunks_exact` (appariement bloc k
  identique). + `needless_return` complex.rs corrigé.
- Test de régression : tous les décalages relatifs (add/dot/fma vs scalaire) ;
  12/12 lancers verts. Déterminisme rétabli (cœur de la thèse scirust).
- docs : CHANGELOG (Corrigé). Pas de changement de roadmap (bug fix). 8 gates (à confirmer).

## Session 2026-06-17 — volet 69 : RCPS (#36) — contrôle de risque (PAC)
- `nn::conformal::hoeffding_ucb` + `rcps_select` (Bates 2021) : contrôle d'un risque
  borné (au-delà de la couverture) via borne Hoeffding ; plus petit λ dont UCB ≤ α
  ⇒ R(λ̂)≤α w.p. 1−δ.
- Bibliothèque seule (comme APS/RAPS/ACI). Complète le pilier conformal.
- Tests (2, core) : UCB = moyenne + slack exact, rcps_select choisit le bon λ ;
  risque test ≤ α sur données fraîches (borne conservatrice).
- docs : roadmap #36 📋→✅ ; CHANGELOG. 536 tests core (+2) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 68 : Prodigy (#46) — Adam sans learning-rate
- `nn::nd_optim::NdProdigy` + `ProdigyConfig` (Mishchenko & Defazio 2023) :
  parameter-free ; estime d≈‖x₀−x*‖ en ligne (corrélation globale ⟨g,x₀−x⟩),
  l'utilise comme taux. d, r, ‖s‖₁ scalaires globaux. Deux passes/step.
- CLI : `lm --opt prodigy` (γ=0.1 par défaut). 8 Documentation + REFERENCE.
- Tests (1, core) : d s'adapte à l'échelle de distance + perte chute (γ=0.1, bande
  ∝ γd sur quadratique déterministe) + déterminisme.
- docs : roadmap #46 📋→✅ ; CHANGELOG. 534 tests core (+1) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 67 : KVQuant (#68) — quant KV-cache (clés per-canal)
- `quantization::kvquant_kv` (Hooper 2024) : clés per-canal (per-colonne), valeurs
  per-token (per-ligne), symétrique bits-bit. Épouse les outliers de canal des clés.
- Bibliothèque seule (pas de CLI ni multilingue).
- Tests (2, core) : erreur d'attention KVQuant < 0,6× per-tensor (clés à outliers
  de canal ×12) ; per-canal résout les petites colonnes (<0,1× erreur) + déterminisme.
- docs : roadmap #68 📋→✅ ; CHANGELOG. 533 tests core (+2) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 66 : ALiBi (#59) — biais d'attention linéaire
- `nn::nd_layers::alibi_slopes` (pentes `2^(−8h/H)`) + `alibi_bias` (biais
  `(H,seq,seq)` = `−pente·(i−j)` causal) + `NdMultiHeadAttention::with_alibi`.
- Branché dans l'attention N-D (builder, inclut le masque causal). MHA standard.
- Tests (4, core) : pentes géométriques ; biais linéaire/causal/Toeplitz ; poids
  softmax décroissants (∝ exp(−pente·dist)) ; attention with_alibi déterministe.
- docs : roadmap #59 📋→✅ ; CHANGELOG. 531 tests core (+4) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 65 : ACI (#38) — Adaptive Conformal Inference
- `nn::conformal::AdaptiveConformal` (Gibbs & Candès 2021) : conformal en ligne ;
  niveau αₜ adapté par rétroaction αₜ₊₁=αₜ+γ(α−errₜ) ⇒ couverture ≈1−α sous dérive.
- Bibliothèque seule (comme APS/RAPS). Complète CQR/APS/RAPS dans `nn::conformal`.
- Tests (2, core) : règle de mise à jour αₜ exacte (miss→−γ(1−α), cover→+γα) ;
  couverture maintenue ≈0,9 sous changement de variance (statique chute) + déterminisme.
- docs : roadmap #38 📋→✅ ; CHANGELOG. 527 tests core (+2) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 64 : KAN (#77) — Kolmogorov-Arnold Networks (RBF)
- `nn::kan::KanLayer` (Liu 2024 ; base RBF de FastKAN, Li 2024) : activations
  apprenables sur arêtes y_j=Σᵢφᵢⱼ(xᵢ), φ = Σ RBF gaussiennes + base SiLU.
- Sortie linéaire en coeffs ⇒ ajustement convexe par GD analytique (standalone).
- Bibliothèque seule (pas de CLI ni multilingue). Variante RBF/FastKAN (pas B-splines).
- Tests (2, core) : ajuste sin(2x₀)+x₁² (MSE<0,02, <0,2× linéaire) ; base RBF
  localisée (pic=1 au centre) + déterminisme bit-exact.
- docs : roadmap #77 📋→✅ ; CHANGELOG. 525 tests core (+2) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 63 : RWKV (#53) — mélange temporel WKV + op `div`
- Nouvel op autograd `div` (division élémentaire broadcast, ∂a=g/b ∂b=−g·a/b²,
  gradient-checké) dans `autodiff::nd`.
- `nn::nd_layers::rwkv_wkv` + `NdRwkv` (Peng 2023) : WKV = attention linéaire à
  décroissance expo. par canal + bonus courant, normalisée ; récurrence sur tape.
  Couche : WKV gaté réception r=σ(W_r·x), decay=σ(·)/bonus=exp(·) par canal apprenables.
- CLI : `rwkv [--seed N] [--steps S]` (groupe NLP). 8 Documentation mises à jour.
- Tests (4) : div gradient-check ; WKV récurrent ≡ formule explicite ; WKV gradient
  check (k,v,decay,bonus) ; NdRwkv entraîne (MSE↓) + déterminisme.
- docs : roadmap #53 📋→✅ ; CHANGELOG. 523 tests core (+4) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 62 : GloRo (#32) — robustesse certifiée par Lipschitz
- `nn::lipschitz` (Leino 2021) : `spectral_norm` (power iteration), `spectral_normalize`
  (couche 1-Lipschitz), `GloroClassifier` (rayon L2 = marge/(√2‖W‖₂), exact-pour-linéaire).
- Bibliothèque seule (pas de CLI ni multilingue). Complète IBP/CROWN/smoothing/GloRo.
- Tests (3, core) : normes spectrales connues (diag/rect) ; norme ≈1 post-normalisation ;
  rayon sain (pire δ ne bascule pas) + conservateur (≤ distance exacte) + déterminisme.
- docs : roadmap #32 📋→✅ ; CHANGELOG. 519 tests core (+3) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 61 : Randomized Smoothing (#27) — robustesse certifiée L2
- `nn::smoothing` (Cohen 2019) : classifieur lissé `g(x)=argmax_c P(f(x+ε)=c)`,
  rayon L2 prouvé `σ·Φ⁻¹(pₐ)` ; `pₐ` minorée par Clopper-Pearson (betai/lgamma
  exact) ; `Φ⁻¹` Acklam. `SmoothedClassifier::{predict,certify}` +
  `clopper_pearson_lower` + `inv_normal_cdf` pub.
- CLI : `certify` enrichi — IBP/CROWN (déterministe) + smoothing (probabiliste,
  demi-espace : rayon ≈ distance). Même signature, pas de multilingue.
- Tests (5, core) : Φ⁻¹ repères ; betai = CDF ; Clopper-Pearson (0.416 + inversion)
  ; rayon = distance demi-espace (indép. σ) + déterminisme ; soundness/abstention.
- docs : roadmap #27 📋→✅ ; CHANGELOG. 516 tests core (+5) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 60 : SpQR (#67) — sparse-quantized (outliers fp)
- `quantization::SpqrOutliers` (Dettmers 2023) : garde la fraction d'outliers (plus
  grosses erreurs |w−q|) en fp, reste en dense ; reconstruction = dense + outliers.
- Bibliothèque seule (comme NF4/SqueezeLLM ; pas de CLI ni multilingue). Scales
  groupés bi-niveaux non modélisés.
- Tests (2, core) : erreur lourde-queue (gaussien + outliers ±12 clampés) divisée
  > 3× en gardant 1 % ; reconstruction n'augmente jamais l'erreur + déterminisme.
- docs : roadmap #67 📋→✅ ; CHANGELOG. 511 tests core (+2) ; 8 gates (à confirmer).
- Note : Randomized Smoothing (#27, `nn::smoothing`) repéré comme prochain gros
  item (Clopper-Pearson ⇒ betai/lgamma/probit ; oracle exact halfspace R=distance).

## Session 2026-06-17 — volet 59 : SqueezeLLM (#66) — quantif non-uniforme sensible
- `quantization::SqueezeLlmCodebook` + `weighted_quant_error` (Kim 2023) : codebook
  `2^bits` par k-means **pondéré sensibilité** (proxy Hessien diag.) ; init quantile
  + Lloyd déterministe ; ties→index bas.
- Bibliothèque seule (comme NF4 ; pas de CLI ni multilingue). Branche sparse non
  modélisée.
- Tests (2, core) : erreur pondérée < RTN (gaussien 3 bits, <0,85×) ; round-trip
  exact sur valeurs du codebook + codebook trié 16 entrées + déterminisme.
- docs : roadmap #66 📋→✅ ; CHANGELOG. 509 tests core (+2) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 58 : APS/RAPS (#34/#35) — ensembles de prédiction
- `nn::conformal::AdaptivePredictionSets` (Romano/Sesia/Candès 2020 ; Angelopoulos
  2021) : conformal classification par score cumulatif `s(x,c)` = masse des classes
  ≥ probables que c. Set `{c : s≤q̂}` ⇒ couverture marginale ≥ 1−α + taille adaptative.
- RAPS : `calibrate_raps(k_reg, λ)` ajoute `λ·max(0, rang−k_reg)` ⇒ ensembles plus
  petits à couverture égale (démontré sur classifieur correct, beaucoup de classes —
  longue queue ; data near-uniforme gonfle q̂ et casse l'effet).
- Bibliothèque seule (comme `ConformalClassifier` ; pas de CLI ni multilingue).
- Tests (3, core) : score cumulatif exact (cas main) ; couverture + adaptativité
  (facile vs ambigu) + déterminisme ; RAPS < APS en taille à couverture ≥ 1−α.
- docs : roadmap #34/#35 📋→✅ ; CHANGELOG. 507 tests core (+3) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 57 : CQR (#33) — Conformalized Quantile Regression
- `nn::conformal::ConformalQuantileRegressor` (Romano, Patterson & Candès 2019) :
  conformalise un régresseur de quantiles. Score `Eᵢ=max(q_lo−y, y−q_hi)`,
  correction finie `Q` (réutilise `conformal_quantile`), intervalle adaptatif
  `[q_lo−Q, q_hi+Q]`. Largeur variable selon x (vs split-conformal constant).
- CLI : `conformal` enrichi — affiche split (largeur constante) **et** CQR
  (largeur adaptative : région faible vs fort bruit) ; même signature (`--seed`,
  `--alpha`), pas de changement multilingue.
- Tests (2, core) : sémantique exacte du score (cas main) ; couverture marginale
  ≥ 1−α + adaptativité (whigh > 1.5·wlow) + déterminisme.
- docs : roadmap #33 📋→✅ ; CHANGELOG. 504 tests core (+2) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 56 : SAM (#47) — Sharpness-Aware Minimization
- `nn::nd_optim::NdSam` + `SamConfig` (Foret 2021) : 2 phases — `ascent` perturbe
  vers `θ+ρ·g/‖g‖` (norme globale du gradient), `descent` restaure θ + pas SGD au
  gradient perturbé. Biais vers les minima plats.
- Bibliothèque seule : 2 gradients/pas ⇒ incompatible avec la boucle `lm --opt`
  (gradient unique). Pas de CLI ni de cycle multilingue (comme NF4).
- Tests (2, core) : perturbation = ρ·g/‖g‖ (‖ε‖=ρ) ; convergence quadratique
  (bande ∝ lr·ρ) + déterminisme.
- docs : roadmap #47 📋→✅ ; CHANGELOG. 502 tests core (+2) ; 8 gates (à confirmer).

## Session 2026-06-17 — volet 55 : Shampoo (#41) — préconditionneur Kronecker
- `nn::nd_optim::NdShampoo` + `ShampooConfig` + helper `inverse_pth_root` (Gupta/
  Koren/Singer 2018) : facteurs `L=E[GGᵀ]`, `R=E[GᵀG]` ; update préconditionné
  `W − lr·L^(−1/4) G R^(−1/4)` ; racines inverses via Jacobi (réutilise
  `jacobi_eigenvectors`), cachées/rafraîchies tous les `precond_freq` pas.
  Vecteurs : Adagrad diagonal.
- CLI : `lm --opt shampoo` (11e valeur `--opt`).
- Tests (3, core) : `inverse_pth_root` `A^(−1/2)²·A≈I` ; convergence quadratique
  matricielle + déterminisme ; repli Adagrad converge.
- docs : roadmap #41 📋→✅ ; 8 Documentation (ligne `--opt`) ; REFERENCE ;
  CHANGELOG.
- 500 tests core (+3) ; 8 gates verts (à confirmer).

## Session 2026-06-17 — volet 54 : Adafactor (#42) — moments 2e ordre factorisés
- `nn::nd_optim::NdAdafactor` + `AdafactorConfig` (Shazeer & Stern 2018) : pour une
  matrice, sommes ligne/colonne du carré du gradient (mémoire `rows+cols`),
  reconstruction rang-1 `V[i,j]=R[i]·C[j]/ΣR` ; update `G/√V` clippé en RMS ;
  planning β2ₜ=1−t^(−0.8). Vecteurs : 2e moment complet (RMSProp).
- CLI : `lm --opt adafactor` (10e valeur `--opt`).
- Tests (3, core) : reconstruction rang-1 exacte ; convergence (bande) +
  déterminisme ; chemin matriciel factorisé réduit ½‖W−T‖².
- docs : roadmap #42 📋→✅ ; 8 Documentation (ligne `--opt`) ; REFERENCE (liste
  `--opt` complétée lookahead/lamb/adan/adafactor) ; CHANGELOG.
- 497 tests core (+3) ; 8 gates verts (à confirmer).

## Session 2026-06-16 — volet 53 : NF4 (#74) — NormalFloat 4-bit
- `quantization::nf4_quantize`/`nf4_dequantize` + `NF4_LEVELS` (QLoRA, Dettmers
  2023) : 16 niveaux = quantiles d'une normale, échelle absmax par bloc.
- Couche de bibliothèque (pas de CLI dédiée — primitive de quantif).
- Tests : sur poids gaussiens (Box-Muller seedé) NF4 < int4 uniforme ; round-trip
  exact sur les niveaux ; déterminisme.
- docs : roadmap #74 📋→✅ ; README int8 ; CHANGELOG. Pas de cycle multilingue
  (bibliothèque).
- 858 tests ; 8 gates verts (à confirmer).

## Session 2026-06-16 — volet 52 : BitNet b1.58 (#69) — quantif ternaire
- `quantization::ternary_quantize` + `ternary_matmul` (Ma 2024) : poids ternaires
  {−1,0,+1} (échelle absmean ~1,58 bit/poids) ; matmul sans multiplication
  (add/sub/skip selon signe).
- CLI : `bitnet [--seed N]` dans le groupe COMPRESSION (en direct : ~20×, 986/4096
  zéros, max err mult-free vs déquant 1,4e-6, reconstruction 0,19 — lossy). 52 cmd.
- Tests : ternaire ∈ {−1,0,1} ; mult-free = forme somme-de-signes (bit-exact) +
  = produit déquant (à la réassociation fp près) ; déterminisme.
- docs : roadmap #69 📋→✅ ; README int8 ; REFERENCE bitnet ; GROWTH_PLAN 52 ;
  CHANGELOG. Multilingue (bitnet) : lot suivant.
- 856 tests ; 8 gates verts (à confirmer).

## Session 2026-06-16 — volet 51 : HGRN (#58) — RNN linéaire gaté
- `nn::nd_layers::hgrn` + `NdHgrn` (Qin 2023) : intégration leaky par canal
  (hₜ=fₜ⊙h+ (1−fₜ)⊙cₜ), porte d'oubli bornée f=lb+(1−lb)σ(·). Pas d'état
  matriciel ; déroulé sur la tape (réutilise cat0 + sub via tenseur ones).
- CLI : `hgrn [--seed N] [--steps S]` (en direct, seed 9/150 : MSE 27.37→4.59).
  51 commandes.
- Tests : match référence Vec ; gradient check (c,f) ; couche entraîne (< 0.7×) +
  déterminisme.
- docs : roadmap #58 📋→✅ ; README stack ; REFERENCE hgrn ; GROWTH_PLAN 51 ;
  CHANGELOG. Multilingue (hgrn) : lot suivant.
- 854 tests ; 8 gates verts (à confirmer).

## Session 2026-06-16 — volet 50 : GLA (#55) — attention linéaire gatée
- `nn::nd_layers::gated_linear_attention` + `NdGla` (Yang 2024) : porte d'oubli
  par canal dépendante de l'entrée α=σ(·) (S_t=diag(α)S_{t-1}+kᵀv, o_t=qS),
  déroulée sur la tape (réutilise cat0).
- CLI : `gla [--seed N] [--steps S]` (en direct, seed 8/150 : MSE 27.16→0.0000).
  50 commandes.
- Tests : match référence Vec ; gradient check (q,k,v,α) ; couche entraîne +
  déterminisme.
- docs : roadmap #55 📋→✅ ; README stack ; REFERENCE gla ; GROWTH_PLAN 50 ;
  CHANGELOG. Multilingue (gla) : lot suivant.
- 851 tests ; 8 gates verts (à confirmer).

## Session 2026-06-16 — volet 49 : RetNet (#54) — rétention (séquence)
- `nn::nd_layers::retention` + `NdRetention` (Sun 2023) : attention linéaire
  récurrente à décroissance γ (S_t=γS_{t-1}+kᵀv, o_t=qS), déroulée sur la tape
  (réutilise cat0).
- CLI : `retnet [--seed N] [--steps S]` (en direct, seed 6/150 : MSE 24.63→0.0002).
  49 commandes.
- Tests : **oracle de dualité** récurrent ≡ parallèle (QKᵀ⊙D)V ; gradient check
  (q,k,v) ; couche entraîne + déterminisme.
- docs : roadmap #54 📋→✅ ; README stack ; REFERENCE retnet ; GROWTH_PLAN 49 ;
  CHANGELOG. Multilingue (retnet) : lot suivant.
- 848 tests ; 8 gates verts (à confirmer).

## Session 2026-06-16 — volet 48 : LAMB (#43) + Adan (#49) — optimiseurs
- `nn::nd_optim::NdLamb` (You 2020) : Adam + ratio de confiance ‖θ‖/‖r‖ par
  tenseur. `NdAdan` (Xie 2022) : Nesterov adaptatif (3 EMA m,v,n + g_prev).
- CLI : `lm --opt lamb|adan` (8e/9e variantes). `--opt` mis à jour dans les 16
  fichiers + code.
- Tests : LAMB converge dans une bande (∝ lr — pas de norme ≈ lr·‖θ‖, comme les
  méthodes par signe) + déterminisme ; Adan converge sur quadratique + déterminisme.
- docs : roadmap #43,#49 📋→✅ ; README optimiseurs ; CHANGELOG.
- 845 tests ; 8 gates verts (à confirmer).

## Session 2026-06-16 — volet 47 : LoRA (#72) — couche PEFT low-rank
- `nn::nd_layers::LoraLinear` (Hu 2022) : base W gelée + ΔW=(α/r)·A·B ; seuls A,B
  entraînés ; B=0 init ⇒ = base. Couche de la tape (forward sur le dernier axe).
- Couche de bibliothèque (pas de commande CLI dédiée — comme RMSNorm/SwiGLU) ;
  exposée + testée.
- Tests : init = base (== x·W), gradient check sur A et B, parameters() = {A,B}.
- docs : roadmap #72 📋→✅ ; README couches ; CHANGELOG.
- 843 tests ; 8 gates verts (à confirmer).

## Session 2026-06-16 — volet 46 : Temperature scaling / calibration (#39)
- `nn::calibration` (Guo 2017) : `temperature_scale` (golden-section sur NLL),
  `expected_calibration_error`, `nll`. Recalibration post-hoc sans changer
  l'argmax (accuracy). Déterministe.
- CLI : nouveau `calibrate [--seed N]` dans le groupe INFERENCE INTEGRITY (avec
  certify/conformal). 48 commandes. En direct : ECE 0.29→0.004 (−98,5 %), T=2,70.
- Tests : ECE baisse + accuracy inchangée + déterminisme.
- docs : roadmap #39 📋→✅ ; README certifiable ; REFERENCE calibrate ;
  GROWTH_PLAN 48 ; CHANGELOG. Multilingue (ligne CLI `calibrate`) : lot suivant.
- 842 tests ; 8 gates verts (à confirmer).

## Session 2026-06-16 — volet 45 : Lookahead (#45) — 1er du pool de candidats
- `nn::nd_optim::NdLookahead` (Zhang 2019) : wrapper poids lents/rapides autour
  d'Adam (k pas rapides, sync φ←φ+α(θ−φ), θ←φ). Déterministe.
- CLI : `lm --opt lookahead` (7e variante d'opt). `--opt` mis à jour dans les 16
  fichiers (code + Documentation×8 + paper×8) par sed (token script-agnostique).
- Tests : convergence quadratique, déterminisme bit-à-bit.
- docs : roadmap #45 📋→✅ ; README optimiseurs ; CHANGELOG.
- Cadence pool : code+test+CLI+roadmap+CHANGELOG/LIVESTATE+README/REFERENCE par
  item ; les bullets prose multilingues d'optimiseurs seront rafraîchies en lot
  après quelques optimiseurs (le `--opt` suffit à l'exposition CLI multilingue).
- 840 tests ; 8 gates verts (à confirmer).

## Session 2026-06-16 — volet 44 : recherche ~55 papers → roadmap (Tier 8-14)
- Ajout de **55 candidats vérifiés** (#26-#80) dans `docs/RESEARCH_ROADMAP.md`,
  arXiv vérifié par recherche web, chacun traduit en fonction scirust concrète +
  module + effort, tous 📋 (au standard test/oracle + 8 gates).
- Tiers : 8 vérification/robustesse certifiée (GCP-CROWN, randomized smoothing,
  DeepPoly, AI², CROWN-IBP, MILP, Lipschitz) · 9 conformal/incertitude (CQR, APS,
  RAPS, RCPS, LtT, ACI, temp-scaling, deep ensembles) · 10 optimiseurs (Shampoo,
  Adafactor, LAMB, Sophia, Lookahead, Prodigy, SAM, GaLore, Adan) · 11 séquence
  (Mamba-2, S4/S5, RWKV, RetNet, GLA, Hyena, xLSTM, HGRN, ALiBi, YaRN) · 12
  décodage (Medusa, EAGLE, PagedAttention) · 13 quantif (QuIP#, OmniQuant,
  SqueezeLLM, SpQR, KVQuant, BitNet b1.58, AQLM, LLM.int8, LoRA, DoRA, NF4) ·
  14 scientifique/audit (FNO, DeepONet, KAN, Rényi-DP, watermark LLM, ZK-ML).
- « Ordre d'attaque » mis à jour : pool de candidats trié par tractabilité.
- Docs uniquement (roadmap) ; gates Rust inchangés (838 tests).

## Session 2026-06-15 — volet 43 : PINN (#17) + CLI/doc
- `nn::pinn` (`Pinn1D` MLP 1→16→16→1 sigmoid, `solve_harmonic`) (Raissi 2019) :
  résout u''=−u, u(0)=0, u(π/2)=1 (= sin x) ; résidu PDE par différences finies
  dans l'entrée (réseau partagé, grad params exact par autodiff inverse) + CL.
- CLI : `pinn [--seed N] [--steps S]` (en direct, seed 1/4000 pas : loss 30.70 →
  0.0003, erreur max vs sin = 0.0036). Groupe NUMERICAL SOLVERS. 47 commandes.
- Tests : loss < 5% de l'initial + erreur max < 0.05 vs sin(x) ; déterminisme.
- docs : roadmap #17 📋→✅ (18/20 + #21..#25) ; README stack N-D ; REFERENCE
  pinn ; GROWTH_PLAN 47 ; CHANGELOG ; Documentation (8) + paper (8).
- 838 tests ; 8 gates verts.

## Session 2026-06-15 — volet 42 : Mamba (#18) + op tape exp + CLI/doc
- `nn::nd_layers::selective_scan` + `NdMamba` (Gu & Dao 2023) : scan sélectif S6
  (Δ,B,C dépendants de l'entrée ; A diagonal ; Ā=exp(Δ·A)) ; récurrence linéaire-
  temps déroulée sur la tape ; init S4D-réelle ; saut D⊙x.
- nouvel op autograd `NdVar::exp` (backward g·exp(x)), gradient-checké.
- CLI : `mamba [--seed N] [--steps S]` (en direct, seed 5/150 pas : MSE 24.53 →
  0.00). 46 commandes.
- Tests : selective_scan match référence ; gradient check (x,Δ,A,B,C) ; couche
  entraîne (MSE↓) + déterminisme ; exp gradient check.
- docs : roadmap #18 📋→✅ (17/20 + #21..#25) ; README stack N-D ; REFERENCE
  mamba ; GROWTH_PLAN 46 ; CHANGELOG ; Documentation (8) + paper (8).
- 836 tests ; 8 gates verts.

## Session 2026-06-15 — volet 41 : DeltaNet (#25) + op tape cat0 + CLI/doc
- `nn::nd_layers::delta_rule` + `NdDeltaNet` (Yang 2024) : attention linéaire
  récurrente à règle delta (mémoire poids rapides S ; S_t = S_{t-1} +
  β_t(v_t − S_{t-1}k_t)k_tᵀ ; o_t = S_t q_t). Récurrence déroulée sur la tape.
- nouvel op autograd `NdVar::cat0` (concat axe 0 + backward par découpe),
  gradient-checké — nécessaire pour réassembler les sorties par pas de temps.
- CLI : `deltanet [--seed N] [--steps S]` (en direct, seed 7/150 pas : MSE
  23.76 → 0.00). 45 commandes.
- Tests : delta_rule match référence Vec ; gradient check (q,k,v,β) ; couche
  entraîne (MSE↓) + déterminisme ; cat0 gradient check.
- docs : roadmap #25 📋→✅ (16/20 + #21..#25) ; README stack N-D ; REFERENCE
  deltanet ; GROWTH_PLAN 45 ; CHANGELOG ; Documentation (8) + paper (8).
- 832 tests ; 8 gates verts.

## Session 2026-06-15 — volet 40 : SOAP (#24) + CLI/doc
- `nn::nd_optim::NdSoap` + `jacobi_eigenvectors` (Vyas 2024) : Adam dans la base
  propre de Shampoo (L=E[GGᵀ], R=E[GᵀG]) ; eigensolveur Jacobi cyclique
  déterministe ; base rafraîchie tous precond_freq pas (moments tournés). Repli
  Adam pour params non matriciels.
- CLI : `lm --opt soap` (en direct, "1,2,3,1,2,3" 60 pas : loss 1.6168→0.0023,
  rappel exact). 6 variantes d'opt dans lm. 44 commandes (inchangé).
- Tests : Jacobi diagonalise (orthogonalité + reconstruction 5×5), SOAP converge
  sur quadratique matricielle (precond_freq=2), déterminisme bit-à-bit.
- docs : roadmap #24 📋→✅ (16/20 + #21..#24) ; README optimiseurs (liste complète) ;
  REFERENCE lm --opt ; CHANGELOG ; Documentation (8) + paper (8) option --opt.
- 828 tests ; 8 gates verts.

## Session 2026-06-15 — volet 39 : AWQ (#15) + CLI/doc
- `quantization::awq_quantize` + `awq_act_scale` + `AwqResult` (Lin 2023) :
  quantification int8 consciente des activations. Importance a_j=moyenne|x_:,j| ;
  scaling s_j=a_j^alpha (moyenne géom. unité) sur les poids avant quant int8
  per-canal ; alpha choisi par grille sur [0,1] (alpha=0 = RTN) minimisant
  l'erreur de sortie pondérée calibration.
- CLI : `awq [--seed N] [--samples S] [--grid G]` (en direct, seed 1 : alpha 0.400,
  RTN 1.70819 → AWQ 0.78211, **−54,2 %** en protégeant 3 canaux saillants ×20).
  44 commandes. **#15 complet : SmoothQuant + GPTQ + AWQ**.
- Tests : protège canaux saillants → erreur < RTN (alpha>0 choisi) + déterminisme.
- docs : roadmap #15 (AWQ ajouté, « Ensuite » vidé de #15) ; README int8 ;
  REFERENCE awq ; GROWTH_PLAN 44 ; CHANGELOG ; Documentation (8) + paper (8).
- 825 tests ; 8 gates verts.

## Session 2026-06-15 — volet 38 : GPTQ (#15) + CLI/doc
- `quantization::quantize_gptq` + `gptq_hessian` (Frantar 2022) : quantification
  int8 par feedback d'erreur 2e ordre. H=XᵀX (calibration) → inverse Cholesky
  f64 → boucle OBQ/GPTQ par canal de sortie (propagation d'erreur + complément
  de Schur). Scale symétrique per-canal de sortie.
- CLI : `gptq [--seed N] [--samples S] [--damp D]` (en direct, seed 1 : RTN
  0.04549 → GPTQ 0.00689, **−84,9 %** d'erreur de calibration). Nouveau groupe
  CLI « COMPRESSION ». 43 commandes.
- Tests : erreur pondérée calibration < RTN (< 0,9·RTN sur données corrélées) +
  soundness (jamais pire) + déterminisme bit-à-bit.
- docs : roadmap #15 🔨→✅ (16/20) ; README int8 ; REFERENCE gptq ; GROWTH_PLAN
  43 ; CHANGELOG ; Documentation (8 langues) + paper (8 langues).
- 823 tests ; 8 gates verts.

## Session 2026-06-15 — volet 37 : CROWN (#2) + doc/CLI
- `nn::ibp::crown_bounds(l1, l2, box)` (Zhang 2018) : bornes de sortie d'un MLP
  ReLU à 1 couche cachée par **relaxation linéaire** + back-substitution.
  Relaxation par neurone (stable = exact ; instable = chorde sup / pente inf).
  **Plus serrée qu'IBP** : prouvé par test (largeur CROWN ≤ largeur IBP +
  soundness par échantillonnage de la boîte L∞).
- CLI : `certify` affiche IBP **et** CROWN côte à côte (en direct, eps=0.05 :
  IBP largeur 0.0847 NON certifié vs CROWN largeur 0.0366 CERTIFIÉ).
- docs : roadmap #2 📋→✅ (15/20 du tier robustesse/opt) ; README certifiable ;
  REFERENCE certify ; CHANGELOG ; Documentation (8 langues) + paper (8 langues).
- 821 tests ; 8 gates verts.

## Session 2026-06-14 — volet 36 : AdEMAMix (#23) + nettoyage code mort
- `nn::nd_optim::NdAdEMAMix` (Pagliardini 2024) : Adam à deux EMA (β1 rapide +
  β3 lente, mélange α) ; déterministe. CLI `lm --opt ademamix`. Tests :
  convergence quadratique (bande), déterminisme.
- **nettoyage** : suppression de `src/nn/.legacy/` (2363 lignes, non câblé,
  dotfile, 0 référence, superposé par les vraies impls). Vérif : analyseur
  CodeFlow = faux positifs (traits/ops/pub API/kernels SIMD par-archi ;
  `archive/` exclu du build) → rien d'autre à supprimer sans casser l'API.
- 820 tests ; 8 gates verts.

## Session 2026-06-14 — volet 35 : Schedule-Free (#22) + doc/CLI/paper
- `nn::nd_optim::NdScheduleFree` (Defazio 2024) : sans planning LR ; base z,
  moyenne Polyak x (point d'éval), gradient en y=(1−β)z+βx. `write_eval_point`.
  Tests : convergence quadratique, déterminisme.
- CLI : `lm --opt schedule-free` (finalize() charge x avant predict ; rappel
  exact en direct). 4 variantes d'opt dans lm.
- docs : roadmap #22 ✅ ; REFERENCE/CHANGELOG ; option --opt mise à jour dans
  Documentation (8) + paper (8).
- NOTE analyseur CodeFlow : voir réponse — faux positifs (outil non-Rust :
  prétend « 0 test » alors que 816 tests passent ; « doublons » = noyaux SIMD
  par-archi + méthodes de traits).
- 818 tests ; 8 gates verts.

## Session 2026-06-14 — volet 34 : conformal prediction (#21) + doc/CLI/paper
- `nn::conformal` (Angelopoulos & Bates) : `conformal_quantile`,
  `ConformalRegressor`, `ConformalClassifier` ; couverture garantie sans
  hypothèse de distribution. Tests : couverture empirique atteint la cible
  (régression + classification).
- CLI : `scirust conformal [--seed N] [--alpha A]` (couverture en direct ;
  90,8 % pour cible 90 %). 42 commandes.
- docs : roadmap #21 ✅ ; README/REFERENCE/GROWTH_PLAN ; Documentation (8 langues)
  et paper (8 langues) — commande conformal ajoutée.
- 816 tests ; 8 gates verts.

## Session 2026-06-14 — volet 33 : CLI + docs multilingues + papers (cycle 2)
- CLI : `scirust certify` (bornes IBP) + `scirust lm --opt adam|adamw|lion`.
  41 commandes. Tests + 8 gates verts.
- docs multilingues : section « Recherche → Fonctions » ajoutée à README,
  Documentation.md + 7 traductions (EN/ES/DE/ZH/JA/KO/AR), et au paper
  (8 langues). Compteur tests README 683→810.
- 2ᵉ recherche de papers → RESEARCH_ROADMAP Tier 7 (#21-#25) : conformal
  prediction (#21, fort fit certifiable), Schedule-Free (#22), AdEMAMix (#23),
  SOAP (#24), DeltaNet (#25).
- 810 tests ; 8 gates verts.

## Session 2026-06-14 — volet 32 : recherche → fonctions (lot 3)
- **Muon** (`nn::nd_optim`) : momentum + Newton-Schulz (quintique, sans SVD) sur
  matrices 2-D ; `newton_schulz_orthogonalize` pub ; déterministe. Tests :
  orthogonalité (déviation s'effondre), perte matricielle, déterminisme.
- **Wanda** (`pruning::prune_wanda`) : élagage one-shot |W|·‖X‖ par ligne ;
  diffère de magnitude sur canaux aberrants.
- **SmoothQuant** (`quantization`) : lissage par canal, préserve X·W ; réduit la
  dispersion des activations. (GPTQ/AWQ encore à faire → #15 partiel.)
- roadmap : **14 des 20** livrés/présents.
- 810 tests ; 8 gates verts.

## Session 2026-06-14 — volet 31 : recherche → fonctions (lot 2)
- **RoPE** (`autodiff::nd`) : op `rope` (paires, backward = rotation inverse) ;
  gradient-check + conservation norme + propriété position relative ; branchée
  dans l'attention via `with_rope`.
- **GQA/MQA** (`nn::nd_layers`) : `new_gqa(num_kv_heads)` — partage K/V via
  broadcast `bmm`, aucune nouvelle op ; gradient-check (kv=2 et kv=1).
- **Neural ODE** (`nn::neural_ode`) : `rk4_integrate` + `NeuralOde`, backprop à
  travers RK4 sur la tape ; RK4 validé (→ e), grad-check, apprend (Adam).
- découverte : FlashAttention online-softmax (#9) **déjà** dans
  `nn::transformer::flash_attention` → marqué ✅, pas de doublon.
- roadmap : **11 des 20** items livrés/présents.
- 802 tests ; 8 gates verts.

## Session 2026-06-14 — volet 30 : recherche → fonctions (lot 1, 7 features)
- `docs/RESEARCH_ROADMAP.md` : 20 papers réels → fonctions, statut + effort.
- **IBP certifié** (`nn::ibp`, Gowal 2018) : intervalles → boîte de sortie
  prouvée ; `certified_robust` ; soundness testée (4000 échantillons ∈ boîte).
  Le pilier « IA certifiable » concret. +`NdLinear::bias()`.
- **réductions reproductibles** (`reproducible`, Demmel-Nguyen) : sum/mean/dot
  bit-identiques quel que soit l'ordre (tri canonique + expansion Shewchuk) ;
  survit à l'annulation catastrophique.
- ops nd : `rmsnorm` + `sigmoid` gradient-checkées. Couches : `NdRmsNorm`,
  `NdSwiGLU`, `NdLlamaBlock` (Pre-RMSNorm+attn causale+SwiGLU), entraînables.
- **décodage spéculatif exact** (`nn::nd_decoder`) : `generate_speculative`
  = greedy cible exact pour tout brouillon, moins de forwards ; +`generate_greedy`.
- optimiseurs : **AdamW** (wd découplé) + **Lion** (déterministe).
- DP-SGD (#19) déjà présent dans `dp.rs` (marqué ✅ dans la roadmap).
- 794 tests ; 8 gates verts.

## Session 2026-06-14 — volet 29 : LM décodeur causal N-D + Adam N-D
- attention causale : `NdMultiHeadAttention { causal }` (masque triangulaire
  -1e9 avant softmax, propagé à `NdTransformerBlock`) ; aucune nouvelle op.
  Test de causalité : perturber le dernier token n'altère AUCUNE sortie
  antérieure (bit-à-bit), la sortie perturbée bouge.
- ops nd : `gather` (embedding, backward scatter-add ; indices répétés
  s'accumulent, lignes inutilisées = grad 0) + `cross_entropy` (softmax+NLL
  fusionné, log-sum-exp ; backward (softmax-onehot)/n). Gradient-checkées.
- `nn::nd_decoder` : **NdDecoderLM** (GPT-style : embeddings tok+pos appris,
  N blocs causals, LN final, tête lm) entraîné en cross-entropy token-suivant.
  Test phare : **sur-apprend une séquence et la reprédit exactement**. « voici
  le LM ». +NdEmbedding (couche réutilisable, nd_layers).
- `nn::nd_optim` : **NdAdam** déterministe + `parameters()` sur toutes les
  couches (compose jusqu'au modèle) ⇒ un `step()` met à jour tout le LM.
  Tests : quadratique (oracle), déterminisme bit-à-bit, LM entraîné par Adam
  (<10 % en 150 pas, prédictions exactes).
- CLI : **commande `lm`** (`scirust lm [..] [--seed/--steps/--lr]`) — entraîne
  le LM décodeur N-D + Adam, rapporte perte + rappel exact ; déterministe.
  39 → 40 commandes. Docs CLI (REFERENCE.md, README, GROWTH_PLAN) à jour.
- fix doc : lien intra-doc `[encode]` cassé dans byte_bpe.rs (gate doc).
- 776 tests ; 8 gates verts.

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
