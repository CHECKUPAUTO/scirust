# Plan du paper — déterminisme comme évidence de certification

> Lot 3 de la mission 2026-07-10. Pièce maîtresse : la table claims →
> évidence (§4) — le paper ne contiendra **aucune claim non prouvée par
> exécution** ; toute claim sans évidence exécutable est supprimée ou
> marquée `TODO-EVIDENCE` avec le test à écrire.

## 1. Titre

Titre de travail :

> **Determinism as Certification Evidence: a Fully Auditable Rust Stack for
> Bit-Reproducible Training and Quantized Edge Inference**

Variantes proposées :

1. *Bit-Reproducible by Construction, Auditable by Design: Deterministic
   Training and Int8 Edge Inference in Pure Rust* — met l'accent sur la
   double propriété (construction + audit).
2. *From Best-Effort to Evidence: a Zero-FFI Rust Stack Where Every
   Determinism Claim Ships With Its Test* — met l'accent sur la discipline
   claims-mesurées, qui est le vrai différenciateur face à RepDL (dont le
   rapport ne comporte pas d'évaluation).

## 2. Venues cibles et recommandation

### Option A — JOSS (Journal of Open Source Software), paper logiciel

- **Pour** : format court centré artefact ; nos points forts collent aux
  critères (tests substantiels, CI, documentation multi-langue, install
  reproductible) ; faisable à court terme ; DOI citable.
- **Contre / BLOQUANT en l'état** : JOSS exige une licence **OSI-approved**.
  Le dépôt est sous **PolyForm Noncommercial 1.0.0** (dual licensing,
  `LICENSING.md`) — non-OSI. Sans re-licence (globale ou d'un sous-ensemble
  publié, ex. `scirust-core` + `scirust-runtime` + `scirust-sigma` sous
  MIT/Apache-2.0), la soumission JOSS est **irrecevable**. Décision
  humaine requise.

### Option B — atelier correctness / reproducibility (version recherche)

- Exemples de familles visées : ateliers « Correctness » (SC), sessions
  reproducibility/artefacts des conférences systèmes-ML. (Le choix précis
  de l'édition/année est une décision humaine ; aucune deadline n'est
  supposée ici.)
- **Pour** : la contribution recherche est réelle mais ciblée — (i) le
  déterminisme *comme pièce d'évidence de certification* (fingerprints,
  journaux hash-chaînés, manifeste) plutôt que comme propriété d'exécution ;
  (ii) le **coût mesuré du déterminisme sur edge ARM** ; (iii) la discipline
  « une claim = un test ». Pas de conflit de licence.
- **Contre** : moins de reconnaissance artefact-logiciel qu'un JOSS ; le banc
  d'overhead O1 est livré côté x86 (voir §4), il reste son volet Jetson.

### Recommandation argumentée

**Option B d'abord.** Deux raisons : (1) la licence PolyForm rend JOSS
irrecevable aujourd'hui — la re-licence est une décision propriétaire, pas
éditoriale ; (2) depuis le verdict **NO-GO** de l'étude « dead guards »
(`docs/DEAD_GUARDS_STUDY.md` : 22 dépôts, ~9,2 M LOC, 0 garde morte
confirmée), la motivation « classe de bug répandue » n'est pas disponible —
le paper doit se positionner sur le **coût mesuré du déterminisme** et
l'**architecture d'évidence**, un angle qui correspond mieux à un atelier
correctness/reproducibility qu'à un journal logiciel. JOSS reste la cible
naturelle *ensuite* si la re-licence d'un sous-ensemble est décidée.

## 3. Plan de sections

1. **Introduction** — le déterminisme traité comme évidence de
   certification, pas comme mode d'exécution ; contrat « chaque claim est
   prouvée par une exécution du dépôt ».
2. **Travaux connexes** — reprendre `paper/RELATED_WORK.md` (Goldberg,
   Monniaux, ReproBLAS ; PyTorch deterministic mode, EasyScale, RepDL —
   paragraphe pivot ; divergences GPU inter-constructeurs). Une phrase de
   due diligence sur l'étude négative « dead guards » (méthode + chiffres).
3. **Architecture** — trois régimes numériques : entier/virgule fixe
   (bit-exact cross-platform), f32 *sanitized* (déterministe
   intra-architecture, σ = `f32::MIN_POSITIVE`), f32 brut (mesuré, non
   garanti) ; zéro FFI dans le chemin de calcul ; TCB = rustc + std.
4. **Entraînement bit-reproductible** — réduction en ordre de worker figé ;
   invariance 1/2/4/8 threads == séquentiel, composée sur une boucle SGD
   multi-pas ; comparaison de conception avec ReproBLAS (ordre figé vs
   somme insensible à l'ordre) et EasyScale.
5. **Inférence comme artefact d'audit** — fingerprint 64 bits,
   reconstruction par manifeste + SRT1/QSR1, verrou de régression,
   détection d'altération, journaux hash-chaînés.
6. **Int8 déterministe pour l'embarqué** — pipeline entièrement entier,
   noyau NEON bit-exact contre la référence scalaire, validation Jetson
   (aarch64).
7. **Évaluation : le coût du déterminisme** — overhead mesuré des choix
   déterministes sur edge et x86 (voir O1) ; latence bornée p99/p50 ;
   précision (MNIST/CIFAR) inchangée.
8. **Limitations** — voie f32 intra-architecture (vs arrondi correct
   RepDL) ; périmètre d'ops ; GPU wgpu validé sur adaptateur logiciel en
   CI ; wall-clock hors CI par nature.
9. **Conclusion et travaux futurs** — transcendantales correctement
   arrondies en Rust pur ; extension multi-nœuds à arbre de réduction figé.

## 4. Table claims → évidence (pièce maîtresse)

Règle : chaque claim ci-dessous est liée au test/benchmark **exact** du dépôt
qui la prouve, avec la commande. `[CI]` = exécuté par
`.github/workflows/ci.yml` ; `[protocole]` = exécution documentée
(`docs/TEST_PROTOCOL.md`), reproductible mais hors CI (wall-clock ou matériel
spécifique). Toute claim sans évidence est marquée `TODO-EVIDENCE`.

| # | Claim du paper | Évidence exacte | Commande |
|---|---|---|---|
| T1 | Un batch d'entraînement multi-thread est bit-identique pour 1/2/4/8 threads et égal au séquentiel, avec contributions sensibles à l'ordre (±1e16) | `train_batch_threaded_is_thread_count_invariant`, `scirust-core/src/autodiff/data_parallel.rs:399` [CI] | `cargo test -p scirust-core train_batch_threaded_is_thread_count_invariant` |
| T2 | L'invariance tient avec un vrai backward autograd (ParallelTape) | `parallel_tape_training_is_deterministic_across_threads`, `scirust-core/src/autodiff/data_parallel.rs:424` [CI] | `cargo test -p scirust-core parallel_tape_training` |
| T3 | L'invariance se compose sur une boucle SGD multi-pas (trajectoire de poids bit-identique 1/2/4 threads) | `multi_step_training_is_thread_count_invariant`, `scirust-core/src/autodiff/data_parallel.rs:458` [CI] | `cargo test -p scirust-core multi_step_training` |
| T4 | L'all-reduce moyenne bit-exactement quel que soit le nombre de threads | `all_reduce_averages_across_threads_bit_exactly`, `scirust-core/src/distributed.rs:251` [CI] | `cargo test -p scirust-core all_reduce_averages` |
| R1 | Un artefact émis est reconstruit depuis le manifeste et toute altération est détectée | `emit_then_verify_roundtrip_and_tamper_detection`, `scirust-runtime/tests/verify_roundtrip.rs:30` [CI] | `cargo test -p scirust-runtime --test verify_roundtrip` |
| R2 | L'artefact quantifié QModel/QSR1 fait un aller-retour déterministe | `qmodel_roundtrip_and_deterministic`, `scirust-runtime/src/quant.rs:285` [CI] | `cargo test -p scirust-runtime qmodel_roundtrip` |
| R3 | Une inférence vérifiable rejette sainement l'altération (sortie, engagement de modèle, substitution) | `vinfer_rejects_tampering_soundly` + 2 tests voisins, `scirust-runtime/src/vinfer.rs:219-257` [CI] | `cargo test -p scirust-runtime vinfer` |
| R4 | Le fingerprint 64 bits du forward est identique entre threads **et entre processus** (0 divergence sur 5120 logits) | volet threads : `forward_fingerprint_is_thread_count_invariant`, `scirust-runtime/tests/fingerprint_thread_invariance.rs` (pools rayon 1/2/4/8, batches synthétiques) [CI] ; volet inter-processus : binaire `scirust-runtime` (fn `fingerprint`, `src/main.rs:14`) + rapport technique §6.2 [protocole] | `cargo test -p scirust-runtime --test fingerprint_thread_invariance` |
| Q1 | La couche linéaire quantifiée int8 reproduit le fp32 dans la tolérance et est déterministe | `test_quantized_linear_matches_fp32` (`scirust-core/src/quantization.rs:214`), `test_quantized_linear_deterministic` (`:243`) [CI] | `cargo test -p scirust-core quantized_linear` |
| Q2 | Le matmul ternaire BitNet sans multiplication égale bit-exactement la référence déquantifiée | `ternary_matmul_equals_dequant_bit_exact`, `scirust-core/src/quantization.rs:1033` [CI] | `cargo test -p scirust-core ternary_matmul` |
| Q3 | Le noyau NEON int8 (aarch64) est bit-exact contre la référence scalaire | `quantization::tests_neon::neon_matches_scalar_bit_exact`, `scirust-core/src/quantization.rs:1959` [protocole : **exécuté sur cible 2026-07-10**, Jetson AGX Thor, commit `014795f` — `ok. 1 passed` ; relançable via `scripts/bench-o1-jetson.sh`] | `cargo test --release -p scirust-core --lib neon_matches_scalar_bit_exact` (sur ARM) |
| S1 | Le seuil de `sanitize_f32` (voie GPU sanitized) est exactement σ = `f32::MIN_POSITIVE`, aligné par test | `sanitize_threshold_matches_sigma_sanitized_f32`, `scirust-sigma/tests/sanitize_alignment.rs:21` [CI] | `cargo test -p scirust-sigma --test sanitize_alignment` |
| S2 | Aucune garde f32 sous σ ne peut entrer dans `scirust-gpu/src` sans casser le build | gate `epsilon-audit --check`, câblé comme job CI `epsilon-audit` (`.github/workflows/ci.yml`) [CI] | `cargo run -q -p scirust-sigma --bin epsilon-audit -- --root . --check` |
| S3 | Le miner « dead guards » classe correctement M1/M2/f64/tests sur fixtures synthétiques | 27 tests unitaires, `scirust-sigma/src/mine.rs` (module `tests`) [CI] | `cargo test -p scirust-sigma mine` |
| G1 | Le GEMM WGSL f32 (wgpu) égale l'oracle CPU, sur adaptateur Vulkan logiciel | job CI `gpu-wgpu` (Mesa lavapipe), tests `scirust-gpu` feature `wgpu` [CI] | `cargo test -p scirust-gpu --features wgpu` |
| A1 | Le journal d'audit hash-chaîné détecte l'altération d'un maillon | `test_chain_tamper_detection`, `scirust-func-safety/src/audit.rs:236` [CI] ; chaîne jumelle `scirust-ids/src/hashchain.rs` | `cargo test -p scirust-func-safety chain_tamper` |
| O1 | Coût du déterminisme : overhead mesuré des réductions à ordre figé vs ordre d'arrivée (x86 + Jetson) | banc `scirust-core/src/bin/bench_reduction_overhead.rs` (slots indexés + réduction ordre 0..n — le pattern de `train_batch_threaded` — vs accumulation en ordre d'arrivée par canal ; magnitudes ±1e16 pour rendre l'ordre observable ; empreintes bit-à-bit par répétition). **Mesuré x86 (4 cœurs, 2026-07-10, release, dim=100 352, 30 reps)** : overhead figé/arrivée = 0,930× (1 thr), 0,895× (2), 0,756× (4), 0,846× (8) — le déterminisme est *gratuit ici, l'ordre figé est même plus rapide* (slots sans contention vs canal) ; empreinte figée unique à chaque n sur 30 reps ; la baseline « arrivée » a produit **3 empreintes distinctes** à 8 threads (non-déterminisme observé en pratique). **Mesuré Jetson AGX Thor (aarch64, 14 cœurs, L4T R38.4.0, MAXN, 2026-07-10, commit `0c2f1bf`, 3 runs × 30 reps ; reconfirmé au commit `014795f` avec `--pin-clocks` opérationnel : 0,93-1,01× à 1-4 thr, 1,06-1,10× à 8, mêmes empreintes)** : overhead figé/arrivée ≈ 0,99× (1 thr), 0,93-0,95× (2), 1,01-1,03× (4), 1,06-1,11× (8) — gratuit jusqu'à 2 threads, ~1-3 % à 4, ~6-11 % à 8 ; baseline « arrivée » non déterministe là aussi (2 empreintes distinctes à 8 threads). **Résultat clé : les 4 empreintes « figé » sont bit-identiques entre x86_64 et aarch64** (`0x60daf62c…`, `0x9bf7c3f3…`, `0xd5b8e15f…`, `0x7e99a9d0…`) — la réduction f32 à ordre figé (add/mul IEEE, sans FMA ni réassociation) est reproductible **cross-platform**, mesuré et pas seulement attendu. Wall-clock ⇒ [protocole], jamais CI | x86 : `cargo run -q --release -p scirust-core --bin bench_reduction_overhead` ; Jetson : `sudo scripts/bench-o1-jetson.sh --pin-clocks` |
| O2 | Latence bornée : p99/p50 ≈ 1,15 (MLP), 1,20 (CNN) | `scirust-runtime/src/bin/bench_latency.rs` + rapport technique §6 [protocole] | `cargo run -p scirust-runtime --bin bench_latency --release` |
| P1 | 1718+ tests workspace, 0 échec (x86) ; 1884/1886 exécutés x86/Jetson | job CI `build-test` + `docs/TEST_PROTOCOL.md` [CI + protocole] | `cargo test --workspace` |

Claims **exclues** du paper par la règle (« aucune claim non prouvée par
exécution ») : le chiffre historique « ~63 TFLOPS BF16 sur Jetson Thor »
(chemin CUDA archivé, non reproductible depuis le build actuel — déjà
marqué comme tel dans le README) ; toute claim d'unicité absolue (« seul
framework à… ») — remplacée par la formulation « à notre connaissance +
périmètre auto-contenu » actée au Lot 1.

## 5. Faiblesses anticipées des rapporteurs, et réponses

- **(a) « Votre voie f32 n'est déterministe qu'intra-architecture ; RepDL
  fait du bit-à-bit cross-platform par arrondi correct. »** Réponse
  assumée : c'est exact, et dit tel quel dans le paper (§8). SciRust offre
  un **spectre de régimes** — entier/virgule fixe déjà bit-exacts
  cross-platform, f32 sanitized intra-architecture — et la roadmap
  explicite : **transcendantales correctement arrondies en Rust pur** pour
  faire converger la voie f32, sans réintroduire de TCB C++/Python. La
  contrepartie que RepDL n'offre pas : pile auditable zéro-FFI, int8
  entier, artefacts d'évidence, et chaque garantie testée en CI (le rapport
  RepDL n'a pas de section d'évaluation).
- **(b) « Quelle est la question de recherche ? »** Réponse : *combien coûte
  le déterminisme, mesuré, sur edge ?* Les chiffres existants du dépôt
  (p99/p50 = 1,15/1,20 ; NEON int8 ~10× vs scalaire à bit-exactitude
  égale ; int8 4× plus petit sans perte — rapport technique §6,
  `docs/TEST_PROTOCOL.md`) forment la base ; le banc O1 livre la première
  mesure directe : **sur x86, la réduction à ordre figé est même plus
  rapide que l'accumulation en ordre d'arrivée** (0,76–0,93×), pendant que
  la baseline non déterministe diverge réellement (3 empreintes distinctes
  à 8 threads) — « le déterminisme du pattern de réduction est gratuit »
  est une réponse mesurée, pas un slogan. Le volet Jetson complète avant
  soumission. Second angle : l'architecture
  **déterminisme-comme-évidence** (fingerprint + hash-chain + manifeste)
  comme objet de conception réutilisable pour la certification.
- **(c) « Périmètre des ops couvertes ? »** Réponse : le périmètre est
  volontairement un sous-ensemble accepté *op par op* contre un oracle
  (politique « une op = un gradient check / un test bit-exact »), listé dans
  le rapport technique ; le paper publie la table de couverture plutôt que
  de la masquer, et positionne SciRust comme framework de référence
  auditable, pas comme concurrent de production de PyTorch (déjà la ligne
  du README).

## 6. Décisions — état au 2026-07-10

Décisions actées (recommandations acceptées) :

- **Bug reports extérieurs** : clos — zéro contact extérieur (verdict NO-GO ;
  le résultat négatif se cite en une phrase de due diligence dans le paper).
- **Venue** : Option B (atelier correctness/reproducibility) ; **pas de
  re-licence** pour JOSS — la licence PolyForm est un choix stratégique,
  JOSS ne le pilote pas.
- **Paper** : GO conditionnel engagé — S2 câblé en CI, R4 verrouillé en CI,
  banc O1 livré avec chiffres x86 (voir table §4).

Décisions restant humaines :

1. Choix de l'atelier précis et de l'édition (deadlines).
2. ~~Exécution du volet Jetson/aarch64 du protocole O1~~ — **fait 2026-07-10**
   (AGX Thor, voir ligne O1 : déterminisme quasi gratuit, et réduction figée
   bit-identique x86_64 ↔ aarch64).
3. Déclenchement de l'écriture du paper complet (tout le matériel amont est
   prêt : RELATED_WORK citable, table claims → évidence sans TODO bloquant,
   les deux volets d'O1 mesurés).
