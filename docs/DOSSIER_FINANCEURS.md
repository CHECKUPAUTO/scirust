# SciRust — Dossier à destination des financeurs

> Document de présentation projet · pré-amorçage / amorçage deep-tech
> Confidentiel — destiné aux investisseurs et partenaires financiers
> Données techniques mesurées sur le dépôt à la date d'édition (voir §10, Méthodologie)

---

## 1. Résumé exécutif (le pitch en 60 secondes)

**SciRust est une plateforme logicielle de calcul scientifique et d'intelligence
artificielle, écrite intégralement en Rust, conçue pour les environnements où l'on
ne peut pas se tromper : automobile, énergie, santé, eau, défense, industrie 4.0.**

Là où les outils d'IA dominants (Python/PyTorch, TensorFlow) sont conçus pour la
recherche — non déterministes, lourds, difficiles à auditer et à certifier —
SciRust apporte trois garanties que personne d'autre ne réunit dans un seul socle :

1. **Déterminisme bit-à-bit** — le même calcul donne exactement le même résultat,
   octet pour octet, quel que soit le nombre de cœurs, la machine ou l'exécution.
   Indispensable pour la traçabilité, l'audit et la certification.
2. **Certifiabilité** — bornes de sortie *prouvées* sur les réseaux de neurones,
   couverture statistique *garantie*, journal d'audit chaîné par hachage, conformité
   visée ISO 26262 (auto) / IEC 61508 / DO-178C (aéro) / 21 CFR Part 11 (pharma).
3. **Pureté & portabilité** — 100 % Rust, **zéro dépendance C/C++/CUDA**, du cloud
   x86 au calculateur embarqué ARM (NVIDIA Jetson) avec un résultat **identique**.

Le socle est déjà **réel et mesuré** : ~158 000 lignes de Rust, **89 modules**,
**~1 900 tests automatiques (0 échec)**, validé de bout en bout sur serveur x86
**et** sur cible embarquée **NVIDIA Jetson AGX Thor**.

**La demande** : financer le passage de ce socle technologique validé à des produits
commerciaux par verticale (voir §8, modèle économique, et §9, emploi des fonds).

---

## 2. Le problème : l'IA industrielle n'est pas « certifiable »

L'IA est en train d'entrer dans des systèmes critiques (conduite assistée, réseaux
électriques, dispositifs médicaux, usines). Mais la pile logicielle dominante a été
pensée pour la recherche, pas pour la sûreté :

| Exigence du monde critique | Réalité de la pile Python/CUDA actuelle |
|---|---|
| Résultat reproductible et auditable | Non déterministe par défaut (ordre des threads, GPU) |
| Bornes garanties sur les sorties | Aucune garantie formelle native |
| Traçabilité / preuve d'intégrité | Pas de chaîne de preuve intégrée |
| Petite surface d'audit (sécurité) | Des millions de lignes C/C++/CUDA + Python |
| Embarquable sur calculateur contraint | Runtime lourd, dépendances GPU propriétaires |
| Indépendance fournisseur | Verrouillage CUDA (NVIDIA) |

**Conséquence** : aujourd'hui, faire certifier un système à base d'IA pour
l'automobile, l'aéronautique ou le médical coûte des années et des millions, parce
que le socle logiciel n'a pas été conçu pour ça. **C'est exactement le vide que
SciRust occupe.**

---

## 3. La solution & le positionnement

SciRust se place **à l'intersection** de trois marchés habituellement séparés :

```
        IA / Deep Learning            Calcul industriel / verticales métier
        (PyTorch, JAX, Burn)          (MATLAB/Simulink, dSPACE, ANSYS)
                    \                 /
                     \               /
                      \   SciRust   /
                       \  (le seul  /
                        \ socle qui /
                         \ unifie  /
                          \ les 3) /
                           \      /
                    Sûreté & certification logicielle
                    (SCADE, outils DO-178C / ISO 26262)
```

**Positionnement en une phrase** : *le socle de calcul et d'IA déterministe,
certifiable et 100 % Rust, du cloud à l'embarqué, pour les systèmes industriels
critiques.*

Ce positionnement est **défendable** parce qu'il repose sur des choix d'architecture
profonds (déterminisme, pureté Rust, zéro FFI) qu'un concurrent établi ne peut pas
ajouter après coup sans réécrire son socle.

---

## 4. Capacités actuelles — les chiffres

### 4.1 Le socle technologique (mesuré sur le dépôt)

| Indicateur | Valeur |
|---|---:|
| Langage | **100 % Rust** |
| Lignes de code Rust | **157 510** |
| Fichiers source Rust | **515** |
| Modules / crates | **89** |
| dont crates de verticales scientifiques/industrielles (`scirust-*`) | **82** |
| Fonctions de test automatiques | **1 970** |
| Tests exécutés au dernier run (x86 / Jetson) | **1 884 / 1 886** |
| Échecs de test | **0** |
| Fichiers source C / C++ / CUDA / Fortran | **0** |
| Dépendances externes (crates.io) | ~65 (auditées, `cargo-deny` en CI) |
| Couverture multi-langue de la documentation | 8 langues (EN, FR, DE, ES, AR, JA, KO, ZH) |

### 4.2 Les garanties différenciantes

- **Déterminisme certifié multi-thread** : l'entraînement réparti sur 1/2/4/8
  threads produit un résultat **bit-identique** (l'addition flottante n'étant pas
  associative, l'ordre de réduction est figé). *Aucun framework grand public ne
  livre cette garantie testée.*
- **Empreinte d'inférence 64 bits** stable entre threads **et** entre processus.
- **Inférence certifiable** : propagation d'intervalles (IBP) et **CROWN**
  (relaxation linéaire, bornes *prouvées* plus serrées) ; **prédiction conforme**
  (couverture garantie sans hypothèse de distribution) ; recalibration de
  température.
- **Quantification déterministe** : int8 sans perte (noyau ARM NEON ~10× plus
  rapide, bit-exact contre la référence scalaire), GPTQ, AWQ, **BitNet b1.58**
  (poids ternaires, matmul sans multiplication), NF4.
- **Auditabilité de bout en bout** : pur Rust, `Cargo.lock` figé, SBOM CycloneDX,
  `cargo-deny` (licences + vulnérabilités) en intégration continue, journal d'audit
  chaîné par hachage (anti-altération).

### 4.3 Couverture fonctionnelle — IA / Deep Learning

Pile d'autodifférentiation N-dimensionnelle, *chaque opération validée par
vérification de gradient* :

- **Transformeurs & LLM** : décodeur causal entraînable, couches type LLaMA
  (RMSNorm, SwiGLU, RoPE, attention groupée/multi-requêtes), **LoRA**, FlashAttention,
  décodage spéculatif exact.
- **Architectures linéaires récentes** : **Mamba** (state-space sélectif), **RetNet**,
  **GLA**, **HGRN**, **RWKV**, **DeltaNet** — temps linéaire.
- **Physique & sciences** : **Neural ODE** (RK4), **PINN** (réseau informé par la
  physique, résout une EDP avec le résidu dans la perte), opérateurs de Fourier (FNO),
  DeepONet, KAN.
- **Optimiseurs déterministes** : Adam, AdamW, Lion, Muon, SOAP, Schedule-Free,
  Lookahead, LAMB, Adan…
- **GPU portable** : noyau GEMM **wgpu/WGSL** validé contre l'oracle CPU — confirmé
  sur adaptateur Vulkan logiciel (CI) **et sur le GPU réel du Jetson AGX Thor**.

### 4.4 Couverture fonctionnelle — verticales industrielles

| Domaine | Crate | Capacités clés | Garantie associée |
|---|---|---|---|
| Sûreté fonctionnelle auto | `scirust-func-safety` | ISO 26262 ASIL A-D, injection de fautes, mode dégradé, **audit chaîné**, comparateur de lot « golden » (GMP / 21 CFR Part 11) | Traçabilité certifiable |
| Maintenance prédictive | `scirust-pdm`, `scirust-signal` | FFT, diagnostics de roulements (BPFO/BPFI/BSF), indice de santé, RUL, CUSUM | Déterminisme |
| Estimation & navigation | `scirust-estimation`, `scirust-nav` | Filtres de Kalman / IMM / **racine-carrée UD** (covariance PSD par construction), fusion **GNSS/INS**, multilatération **TDOA** | Bornes certifiées |
| Réseaux d'eau | `scirust-water` | Localisation acoustique de fuites, physique du coup de bélier (Joukowsky, Korteweg) | Déterminisme |
| Cybersécurité OT | `scirust-ids` | Détection d'intrusion, **attestation de firmware**, intégrité d'automates PLC (détection du motif Stuxnet) | Chaîne de hachage anti-altération |
| Énergie / batteries | `scirust-grid`, `scirust-bms` | Réseau électrique, gestion de batterie (SoC déterministe) | Déterminisme |
| Connectivité usine | `scirust-opcua`, `scirust-mqtt` | OPC-UA (8 capteurs simulés), MQTT Sparkplug B | — |
| MLOps industriel | `scirust-mlops` | Dérive, déploiement « shadow », OTA signé | Intégrité signée |
| + Santé, HVAC, SHM, métrologie, robotique, SPC, fiabilité | `scirust-biomed`, `-hvac`, `-shm`, `-metrology`, `-robotics`, `-spc`, `-reliability` | Lois métier validées contre références | Déterminisme |

### 4.5 Création d'algorithmes (IA générative appliquée au code)

AutoML (optimisation bayésienne), synthèse de programmes, génération d'algorithmes
(tri/recherche/graphes/DP), transformation de code (AST, 20 règles, transpilation
Rust→Python/C), découverte par apprentissage par renforcement.

---

## 5. Preuves de validation (ce n'est pas une maquette)

SciRust est livré avec un **protocole d'acceptation exécutable en une commande**
(`scripts/test-protocol.sh`) qui certifie l'ensemble de la plateforme : tous les
contrôles qualité, tous les tests-oracles de chaque module, le déterminisme
inter-processus, la compilation multi-architecture, la documentation et l'audit de
sécurité — avec un verdict PASS/FAIL et un dossier de preuves horodaté.

| Cible | Verdict | Détail |
|---|---|---|
| Serveur **x86_64** (cloud) | **PASS 12/12** | 1 884 tests, 0 échec ; 92 oracles déterministes reproduits sur 2 process |
| Embarqué **NVIDIA Jetson AGX Thor (ARM aarch64)** | **PASS 10/10** | 1 886 tests, 0 échec ; **GPU Vulkan validé** ; audit licences propre ; 44 noyaux ARM (NEON/SIMD) exécutés *nativement* |

> Point clé pour un industriel : **le même socle déterministe passe au vert,
> identique, du serveur cloud au calculateur embarqué.** C'est la condition
> nécessaire au déploiement « train dans le cloud, exécute dans le véhicule/l'usine
> avec preuve d'équivalence ».

---

## 6. Comparaison concurrentielle

### 6.1 Tableau de positionnement

| Critère | **SciRust** | PyTorch / TF / JAX | Burn (Rust) | Candle (Rust) | tch-rs / libtorch | TensorRT / TFLite | MATLAB+Simulink / SCADE |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 100 % Rust, sans C/C++/CUDA | ✅ | ❌ | ✅ | ⚠️ | ❌ (FFI C++) | ❌ | ❌ |
| Déterminisme bit-à-bit garanti | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ partiel |
| Bornes de sortie *prouvées* (IBP/CROWN) | ✅ | ❌ (ext.) | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| Couverture statistique garantie (conformal) | ✅ | ❌ (ext.) | ❌ | ❌ | ❌ | ❌ | ❌ |
| Sûreté fonctionnelle intégrée (ISO 26262…) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Verticales industrielles prêtes | ✅ (15+) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Embarqué x86 ↔ ARM, résultat identique | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ (inférence) | ✅ |
| IA / entraînement deep learning | ✅ | ✅✅ | ✅ | ⚠️ inférence | ✅ | ❌ inférence | ⚠️ |
| Auditabilité (SBOM, zéro FFI) | ✅ | ❌ | ⚠️ | ⚠️ | ❌ | ❌ | ⚠️ |
| Coût de licence | Open-core | Gratuit | Gratuit | Gratuit | Gratuit | Gratuit/HW | **Très élevé (€€€)** |

Légende : ✅ natif · ⚠️ partiel/via extension · ❌ absent · « ext. » = bibliothèque tierce séparée.

### 6.2 Lecture stratégique

- **PyTorch / TensorFlow / JAX** dominent la *recherche* et l'entraînement à grande
  échelle. Mais leur pile (Python + des millions de lignes C++/CUDA) est non
  déterministe, lourde à auditer et **non certifiable en l'état**. SciRust ne les
  attaque pas sur la recherche : il occupe le créneau « déploiement critique » qu'ils
  ne couvrent pas.
- **Burn / Candle** (Rust) valident la thèse « l'IA en Rust monte » — mais restent
  des frameworks d'IA *généralistes*, sans déterminisme garanti, sans verticales
  industrielles ni sûreté fonctionnelle. SciRust est plus large (calcul + IA +
  métier + certification) et plus profond (garanties formelles).
- **tch-rs** n'est qu'une liaison vers libtorch (C++) : il hérite des défauts de
  PyTorch (non-déterminisme, FFI, surface d'audit).
- **TensorRT / TFLite** sont des moteurs d'*inférence* propriétaires/verrouillés
  matériel ; pas d'entraînement, pas de certification formelle.
- **MATLAB/Simulink + SCADE** sont la référence de la certification industrielle,
  mais propriétaires, **très coûteux** (licences à 5 chiffres/an/poste), cloisonnés
  et **non « IA-natifs »**. SciRust apporte la même rigueur, ouverte, et IA-native.

**Le « moat » (douve défensive)** : aucun concurrent ne réunit *Rust pur +
déterminisme + certifiabilité + verticales + IA + portabilité cloud↔edge*. Ces
propriétés sont architecturales : impossibles à rajouter sans refonte.

---

## 7. Marché adressable

> Les montants ci-dessous sont des **estimations sectorielles publiques** (ordres de
> grandeur), citées pour dimensionner l'opportunité — ce ne sont pas des revenus
> SciRust.

| Segment cible | Taille marché (ordre de grandeur) | Croissance | Pertinence SciRust |
|---|---|---|---|
| IA / ML (logiciel & plateformes) | ~200 Md$ (2024) → ~1 000 Md$ (2030) | ~35 %/an | IA déterministe & certifiable |
| **Edge AI** (IA embarquée) | ~20 Md$ → ~140 Md$ (2030) | ~25 %/an | Cœur de cible (cloud↔edge) |
| Sûreté fonctionnelle (functional safety) | ~6 Md$ → ~10 Md$ | ~8 %/an | Différenciateur direct |
| Automatisation industrielle / Industrie 4.0 | ~150-300 Md$ | ~10 %/an | Verticales métier |
| Maintenance prédictive | ~5 Md$ → ~40 Md$ (2030) | ~25 %/an | `scirust-pdm`/`-signal` |
| Logiciel automobile / ADAS | dizaines de Md$ | élevé | Estimation/nav + ISO 26262 |
| Cybersécurité OT/industrielle | ~20 Md$ → ~45 Md$ | ~15 %/an | `scirust-ids` (OT) |

**Stratégie d'entrée (SOM)** : ne pas viser tout le marché, mais **2-3 verticales
« tête de pont »** où le déterminisme/la certification sont rédhibitoires et chers à
obtenir autrement — typiquement **maintenance prédictive certifiable** et
**estimation/navigation embarquée pour l'automobile/défense** — puis élargir.

---

## 8. Modèle économique

Modèle **open-core** (le cœur attire la communauté et la confiance ; la valeur
métier est commerciale) :

1. **Licences commerciales / duales** pour l'usage propriétaire et embarqué (par
   produit/par appareil).
2. **Kits de certification** : dossiers de preuves prêts à l'audit (ISO 26262, IEC
   61508, DO-178C, 21 CFR Part 11) — c'est ici que se trouve la plus forte valeur,
   car SciRust automatise ce qui coûte aujourd'hui des mois d'ingénierie.
3. **Support, SLA et intégration** (déploiement OPC-UA/MQTT, mise en service usine).
4. **Produits verticaux** : monitoring industriel, protocole d'acceptation « as a
   service », sécurité OT.
5. **Licence edge/embarqué** par calculateur déployé (modèle récurrent).

Marges logicielles élevées ; revenus récurrents (licences + support + certif).

---

## 9. Feuille de route (emploi des fonds)

Le socle technique étant validé, les fonds servent à **transformer la technologie en
revenus** :

**Court terme (0-12 mois)**
- Industrialiser 2 verticales tête de pont (maintenance prédictive certifiable,
  estimation/navigation embarquée) jusqu'au pilote client payant.
- Premier kit de certification ISO 26262 packagé et opposable.
- Déterminisme GPU (étendre la garantie bit-exacte au GPU ; aujourd'hui CPU-exact +
  GPU portable validé contre l'oracle — le chemin cuBLAS BF16 ~63 TFLOPS démontré
  historiquement sur Jetson Thor est archivé et fait partie de la feuille de route).

**Moyen terme (12-24 mois)**
- Élargir aux verticales énergie, santé (21 CFR Part 11), eau, OT.
- LLM/edge : inférence déterministe et certifiable de modèles de langage embarqués.
- Partenariats équipementiers (automobile, défense, énergie).

**Long terme (24 mois+)**
- Devenir le standard de fait du « calcul critique reproductible » — la couche de
  confiance sous l'IA industrielle.

---

## 10. Risques & mitigations (transparence)

| Risque | Mitigation |
|---|---|
| Écosystème IA moins fourni que Python | On ne concourt pas sur la recherche SOTA, mais sur les garanties (déterminisme/certif) que Python n'a pas |
| Maturité GPU vs CUDA | CPU déterministe déjà livré ; GPU portable (wgpu) validé ; déterminisme GPU = jalon financé de la feuille de route |
| Effort de certification long | Les kits de preuves sont précisément le produit à plus forte valeur ; le socle (audit chaîné, SBOM, déterminisme) est déjà aligné |
| Adoption / go-to-market | Entrée par verticales à douleur forte + open-core pour la confiance et le « bottom-up » |
| Dépendance personnes-clés | Code 100 % Rust, testé (~1 900 tests), documenté en 8 langues : faible « bus factor » technique |

---

## 11. Méthodologie des chiffres

Tous les indicateurs du §4.1 sont **mesurés automatiquement sur le dépôt** (comptage
des lignes/fichiers/tests via les outils Git/Cargo, et non déclaratifs). Les verdicts
du §5 proviennent d'exécutions réelles du protocole d'acceptation (`test-protocol.sh`
sur x86, `test-protocol-jetson.sh` sur Jetson AGX Thor), dont les journaux horodatés
constituent le dossier de preuves. Les tailles de marché du §7 sont des estimations
sectorielles publiques, citées en ordre de grandeur.

---

*SciRust — le socle de calcul et d'IA déterministe, certifiable et 100 % Rust,
du cloud à l'embarqué. Contact : voir le dépôt.*
