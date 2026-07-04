# SciRust — Conception du transpileur scientifique (source → Rust)

> Statut : **conception + Phase 0 livrée**. Ce document décrit l'architecture
> d'un transpileur *entrant* (Python / MATLAB / Julia / Fortran / C++ → Rust)
> déterministe, sûr et **vérifié par oracle**, aligné sur la doctrine du dépôt
> (« aucune affirmation sans test »). Il distingue rigoureusement ce qui
> **existe déjà** de ce qui **reste à construire**, et ne revendique aucune
> capacité non livrée.
>
> **Mise à jour — Phase 0 (MVP) implémentée et prouvée.** Le crate
> [`scirust-transpiler`](../scirust-transpiler) réalise le pipeline entrant
> complet (front-end Python/NumPy → SIR typée → émission Rust déterministe),
> gated par un **oracle différentiel contre NumPy réel** :
> `cargo run -p scirust-transpiler --example oracle` → **7/7 cas, 200 essais
> chacun, conformes** (rk4, dot, norm, weighted-mean, cumsum, saxpy, tanh).
> L'oracle est non-vacuous (l'injection d'un opérateur faux fait passer 4/7 cas
> au ROUGE). Voir la §9-bis « État d'implémentation ».

---

## 0. Résumé honnête (à lire en premier)

La vision demande un outil capable de **convertir automatiquement** des
algorithmes scientifiques écrits en Python, MATLAB, Julia, Fortran ou C++ en
Rust « performant, déterministe et sûr », pour 15 secteurs régulés.

État réel du dépôt aujourd'hui :

| Brique nécessaire                              | Statut | Où |
|------------------------------------------------|--------|----|
| Front-ends de langue (source → AST)            | ❌ absent | — |
| IR scientifique typée (formes, unités, effets) | ❌ absent | — |
| Backend d'émission Rust (AST → source Rust)    | 🟡 **réutilisable** | `scirust-codetrans` (`Expr` + pretty-printer) |
| Passes d'optimisation sur l'IR                 | 🟡 **réutilisable** | `scirust-codetrans` (20 règles : CSE, DCE, LICM…) |
| **Vocabulaire cible vérifié par oracle**       | ✅ **présent** | ~90 crates `scirust-*` (voir §5) |
| **Doctrine de validation par oracle**          | ✅ **présente** | tout le dépôt ; audit hash-chaîné CCOS / MCP |
| Harnais d'oracle *de transpilation*            | ❌ absent | — |
| SLM / agent assistant                          | ✅ présent | `scirust-sciagent`, `scirust-mcp` |

> ⚠️ **Point crucial et honnête.** `scirust-codetrans` transpile **Rust → Python**
> et **Rust → C** (sens *sortant*), c'est-à-dire l'**inverse** de ce que la vision
> demande. Ses fonctions `parse_expr` / `parse_pattern` lisent un AST S-expression
> **interne**, pas du code source Python/MATLAB/Fortran. Il n'existe donc
> aujourd'hui **aucune** capacité de transpilation entrante.

**Verdict.** Le transpileur entrant n'est pas encore livré. Mais deux des trois
briques les plus difficiles le sont déjà : (1) un **vocabulaire cible** de
primitives numériques prouvées bit-exactes contre un oracle de référence, et
(2) la **discipline de preuve** qui distingue SciRust d'un traducteur
« LLM ligne à ligne ». La pièce manquante est le pipeline
*front-end → IR → émission* et le *harnais d'oracle de transpilation*. Ce
document en fixe l'architecture et la feuille de route.

---

## 1. Pourquoi un transpileur scientifique n'est PAS un traducteur syntaxique

Le piège évident — un LLM ou un jeu de règles regex qui « traduit ligne à
ligne » — produit du Rust *plausible* mais potentiellement **faux, non
déterministe et non vérifié**. C'est précisément ce que les secteurs visés
interdisent :

- **DO-178C** (aéronautique) et **IEC 62304 Ed.2** (dispositifs médicaux)
  exigent une traçabilité qui « suppose un comportement déterministe ».
- **ISO 26262** (automobile) impose des tests MIL/SIL/PIL/HIL redondants
  *parce que* la correspondance modèle ⇄ code généré n'est garantie
  qu'« à une tolérance près » (MathWorks le documente lui-même).
- La non-associativité du flottant + le threading BLAS non déterministe
  cassent la reproductibilité (cf. `docs/DOMAIN_ROADMAP.md`, bug OpenBLAS
  #1844).

La thèse de SciRust, appliquée à la transpilation, tient en trois exigences
non négociables :

1. **Comprendre la sémantique numérique** (formes, types, ordre des
   réductions, source d'aléa) — pas seulement la syntaxe.
2. **Émettre vers des primitives déjà prouvées** bit-exactes, plutôt que de
   re-dériver la numérique dans du Rust neuf non testé.
3. **Prouver l'équivalence** source ⇄ Rust par un oracle *avant* d'accepter le
   port — exactement la règle « aucune affirmation sans test » du reste du
   dépôt.

Un port qui ne passe pas l'oracle est **rejeté**, pas « probablement bon ».

---

## 2. Architecture cible (pipeline en 5 étages)

```
  Source scientifique                                          Rust vérifié
  (Python/MATLAB/                                              déterministe, sûr
   Julia/Fortran/C++)                                          (+ rapport signé)
        │                                                              ▲
        ▼                                                              │
 ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐
 │ 1. FRONT-END│──▶│ 2. SIR        │──▶│ 3. ANALYSES   │──▶│ 4. LOWERING       │
 │ (1 par      │   │ Scientific IR │   │ formes, types,│   │ SIR → codetrans:: │
 │  langue)    │   │ typée         │   │ RNG, aliasing,│   │ Expr → Rust src   │
 │  → AST      │   │ (shapes,      │   │ ordre de      │   │ (routé vers les   │
 │             │   │  dtypes,      │   │ réduction     │   │  primitives       │
 │             │   │  unités,      │   │               │   │  scirust-*)       │
 │             │   │  effets)      │   │               │   │                   │
 └─────────────┘   └──────────────┘   └───────────────┘   └──────────────────┘
                                                                    │
                                                                    ▼
                                                        ┌────────────────────────┐
                                                        │ 5. ORACLE DE            │
                                                        │    TRANSPILATION        │
                                                        │ exécute source ⇄ Rust   │
                                                        │ sur N entrées, compare  │
                                                        │ sous tolérance déclarée │
                                                        │ → accepte / rejette     │
                                                        │ → rapport hash-chaîné   │
                                                        └────────────────────────┘
```

### Étage 1 — Front-ends
Un analyseur par langue produisant un AST spécifique. On ne vise **jamais**
« tout le langage » mais un **sous-ensemble scientifique contractuel**,
statiquement analysable (voir §6). Chaque front-end déclare explicitement ce
qu'il accepte et **refuse** (avec diagnostic) ce qu'il ne comprend pas —
plutôt que de deviner.

### Étage 2 — Scientific IR (SIR)
Une IR typée, indépendante de la langue source, où chaque valeur porte :
- **forme** (shape) et **dtype** (f32/f64/i32/complex…),
- **unité physique optionnelle** (m, s, kg… — utile en aéro/spatial/énergie),
- **effets** : pureté, I/O, source d'aléa (RNG), aliasing potentiel,
- **ordre de réduction** requis (pour les sommes/produits).

Le SIR est le seul endroit où la sémantique numérique est raisonnée. C'est
aussi la frontière stable : ajouter une langue = ajouter un front-end vers le
SIR, sans toucher au lowering ni à l'oracle.

### Étage 3 — Analyses
Inférence de formes/types (indispensable depuis Python/MATLAB dynamiques),
détection des sources d'aléa, détection de l'aliasing, fixation de l'ordre
des réductions. Ces analyses transforment un SIR « possiblement dynamique »
en un SIR **statiquement émettable**.

### Étage 4 — Lowering
Abaissement du SIR vers `scirust-codetrans::Expr` (backend d'émission **déjà
présent**, dont le `Display` imprime du Rust), en **routant chaque opération
vers une primitive `scirust-*` vérifiée** (voir §4). Les 20 règles
d'optimisation de `codetrans` (constant folding, DCE, CSE, LICM, réduction de
force, inlining, TCO) s'appliquent ici.

### Étage 5 — Oracle de transpilation (cœur de la confiance)
Détaillé en §8. Sans oracle vert, aucun port n'est accepté.

---

## 3. Contrat de déterminisme et de sûreté (par construction)

Le transpileur n'ajoute pas le déterminisme *après coup* — il **n'émet que du
Rust déterministe par construction**, en s'appuyant sur des garanties déjà
tenues ailleurs dans le dépôt :

- **Ordre de réduction fixe.** Les sommes/produits/moyennes sont émis avec un
  ordre pinné (déjà garanti par `scirust-core` : réductions flottantes
  indépendantes du nombre de threads, fingerprint 64 bits identique).
- **PRNG germé.** Toute source `np.random`, `rand`, `randn`, MATLAB `rand` est
  mappée sur un flux `SplitMix64` germé explicitement — jamais d'entropie
  système implicite.
- **Anti-aliasing.** Le SIR trace l'aliasing ; l'émission produit des
  emprunts `&` / `&mut` sûrs, ou insère des copies explicites documentées.
  Objectif : **zéro `unsafe` non justifié**.
- **Tolérance déclarée.** Chaque port porte une tolérance numérique explicite
  (ex. `rel ≤ 1e-12`) ; le **mode bit-exact** est activé quand la primitive
  cible le permet.
- **Cible embarquée optionnelle.** Pour l'IA embarquée / NVIDIA Jetson, le
  lowering peut cibler `scirust-edge` / `scirust-embedded` (`no_std`,
  sans allocation).

---

## 4. La cible : router les opérations vers des primitives prouvées

C'est le différenciateur central. **On ne re-dérive pas la numérique dans du
Rust neuf ; on route chaque opération source vers un noyau déjà validé contre
un oracle de référence.** Extrait du mapping (à compléter au fil des phases) :

| Opération source (NumPy/SciPy/MATLAB/BLAS…)         | Primitive `scirust-*` cible |
|-----------------------------------------------------|-----------------------------|
| `np.linalg.solve` / `\` MATLAB / LU                 | `scirust-solvers` (LU, QR, Cholesky) |
| `np.linalg.svd` / `eig` / `qr`                      | `scirust-solvers` (SVD Jacobi, eig Householder+QL) |
| `scipy.sparse.linalg` (GMRES/BiCGSTAB)              | `scirust-solvers` (GMRES restart, BiCGSTAB) |
| `np.fft` / `scipy.signal`                           | `scirust-signal` (FFT, fenêtres, features) |
| `scipy.integrate.odeint` / MATLAB `ode45`           | `scirust-solvers::ode` (RK, autodiff) |
| Kalman/EKF (`filterpy`, MATLAB)                     | `scirust-estimation` (KF/EKF/UD square-root) |
| GNSS/INS, TDOA                                       | `scirust-nav` |
| PID/LQR/MPC                                          | `scirust-control` |
| optimisation (`scipy.optimize`, `fmincon`)          | `scirust-solvers`, `scirust-evo` |
| PCA/ICA/K-Means/clustering                           | `scirust-multivariate`, `scirust-unsupervised` |
| réseaux de neurones / inférence                     | `scirust-core`, `scirust-onnx`, `scirust-sciagent` |
| traitement d'images / CNN / segmentation            | `scirust-vision` |
| rainflow / Palmgren-Miner (fatigue)                 | `scirust-fatigue` |
| réseaux électriques / WLS                            | `scirust-grid` |
| biosignaux / ECG / dosing                            | `scirust-biomed` |

Là où aucune primitive n'existe, le transpileur **ne devine pas** : il émet un
`TODO` typé et signale le trou (cohérent avec la frontière « not delivered »
du reste du dépôt).

---

## 5. Couverture des 15 secteurs (matrice honnête)

« Vocabulaire cible » = les primitives Rust vérifiées vers lesquelles émettre.
`✅` = primitives déjà présentes ; `🟡` = partiel ; `❌` = à construire.
La transpilation elle-même reste **à construire pour tous** (colonne omise).

| # | Secteur | Vocabulaire cible présent ? | Crates d'ancrage |
|---|---------|------------------------------|------------------|
| 1 | Pharma / biotech (simulation moléculaire, génomique, PK, jumeaux bio) | 🟡 | `scirust-biomed`, `scirust-solvers`, `scirust-multivariate`, `scirust-tn` |
| 2 | Robotique industrielle (trajectoire, SLAM, fusion, temps réel, vision) | ✅ | `scirust-robotics`, `scirust-fusion`, `scirust-control`, `scirust-vision`, `scirust-estimation` |
| 3 | Aéronautique (guidage, nav, Kalman, contrôle de vol, simulation) | ✅ | `scirust-nav`, `scirust-estimation`, `scirust-control`, `scirust-func-safety` |
| 4 | Spatial (nav satellite, orbite, contrôle embarqué, télémétrie) | 🟡 | `scirust-nav`, `scirust-estimation`, `scirust-embedded`, `scirust-signal` |
| 5 | Automobile (ADAS, fusion lidar/radar, vision, moteur, batterie) | ✅ | `scirust-fusion`, `scirust-vision`, `scirust-bms`, `scirust-func-safety`, `scirust-control` |
| 6 | Finance quantitative (pricing, Monte Carlo, risque, portefeuille) | 🟡 | `scirust-solvers`, `scirust-evo`, `scirust-trader` |
| 7 | Énergie (réseaux, smart grid, prévision, éolien, nucléaire, hydro) | ✅ | `scirust-grid`, `scirust-sis`, `scirust-reliability`, `scirust-seasonal`, `scirust-water` |
| 8 | Géophysique (sismologie, exploration, tomographie, signaux) | 🟡 | `scirust-signal`, `scirust-solvers`, `scirust-shm` |
| 9 | Météorologie (prévision numérique, assimilation, climat) | 🟡 | `scirust-solvers`, `scirust-estimation` (assimilation ≈ filtrage), `scirust-tn` |
| 10 | IA embarquée (prétraitement, pipelines ML, inférence déterministe) | ✅ | `scirust-edge`, `scirust-embedded`, `scirust-core`, `scirust-onnx` |
| 11 | Industrie chimique (réacteurs, CFD, thermo, optimisation procédés) | 🟡 | `scirust-solvers`, `scirust-fab`, `scirust-sis`, `scirust-spc` |
| 12 | Imagerie médicale (reconstruction CT/IRM, segmentation, filtrage) | 🟡 | `scirust-vision`, `scirust-signal`, `scirust-solvers` |
| 13 | Défense (simulation, radar, sonar, guerre élec., fusion) | ✅ | `scirust-signal`, `scirust-fusion`, `scirust-estimation`, `scirust-nav` |
| 14 | Physique (Monte Carlo, quantique, astrophysique, particules) | 🟡 | `scirust-tn`, `scirust-solvers`, `scirust-tensor-*` |
| 15 | Industrie 4.0 (jumeaux numériques, PdM, optimisation prod., vision) | ✅ | `scirust-pdm`, `scirust-mlops`, `scirust-opcua`, `scirust-mqtt`, `scirust-vision` |

**Lecture.** Pour ~8 secteurs sur 15, le vocabulaire cible est déjà là et de
qualité oracle : le travail est le pipeline d'entrée, pas la numérique. Pour
les 🟡, il faudra compléter quelques primitives (CFD, reconstruction
tomographique, Monte Carlo financier…) en parallèle des front-ends.

---

## 6. Front-ends : stratégie par langue (difficulté croissante)

Ordre de priorité guidé par (a) le volume de code scientifique réellement
concerné et (b) la tractabilité de l'analyse statique.

| Langue | Priorité | Sous-ensemble visé | Difficulté | Piste de parsing |
|--------|----------|--------------------|------------|------------------|
| **Python/NumPy** | 1 (MVP) | fonctions typées, NumPy/SciPy, pas d'`eval`/réflexion/monkeypatch | moyenne | AST via `rustpython-parser` (Rust pur) — à évaluer côté licence/déps |
| **MATLAB** | 2 | fonctions, matrices, indexation 1-based, broadcasting implicite | moyenne-haute | parser dédié (grammaire propre, copy-on-write) |
| **Fortran** (77/90+) | 3 | routines numériques, tableaux column-major | haute | parser dédié ; attention `COMMON`/`EQUIVALENCE` |
| **Julia** | 4 | déjà typé, dispatch multiple | moyenne | intérêt moindre (Julia est déjà rapide) |
| **C/C++** | 5 | sous-ensemble numérique | très haute | pré-passe `c2rust` puis raffinage vers idiomes SciRust |

Principe commun : **contrat de sous-ensemble explicite**, refus diagnostiqué
hors périmètre, jamais de traduction devinée.

Pourquoi cet ordre : Python/MATLAB portent le prototypage de recherche
(pharma, robotique, finance, imagerie médicale — « développé en Python puis
réécrit ») ; Fortran porte le code hérité (météo, géophysique, spatial,
physique — « des millions de lignes ») ; C/C++ est le plus dur et le moins
rentable en premier (l'UB et les templates rendent l'équivalence prouvable
coûteuse).

---

## 7. Rôle du LLM / SLM : **assistant, jamais oracle**

Le dépôt possède déjà un SLM spécialisé Rust (`scirust-sciagent`) et une
couche MCP (`scirust-mcp`) pilotable par un LLM externe. Leur place dans le
transpileur :

- **Utiles pour** : combler les trous sémantiques (idiomes ambigus), proposer
  un mapping d'opérations, **générer les cas de test** de l'oracle, rédiger la
  documentation du port.
- **Jamais** source de vérité : **toute** sortie assistée passe l'oracle de
  transpilation (§8). C'est la posture déjà tenue par `scirust-trader`
  (« certified predictions, LLM narration, proof-sealed decisions »),
  transposée à la transpilation.

Un LLM accélère la *proposition* ; l'oracle décide de l'*acceptation*.

---

## 8. Le harnais d'oracle de transpilation

C'est la brique qui transforme « transpileur » en « transpileur *de confiance* ».

1. **Test différentiel.** Exécuter la source dans son **runtime réel**
   (CPython+NumPy, Octave/MATLAB, `gfortran`, `clang`) et le Rust émis sur un
   corpus d'entrées : aléas germés + cas limites (0, NaN/Inf, matrices
   singulières, tableaux vides) + éventuellement property-based. Comparer sous
   la tolérance déclarée du port.
2. **Test métamorphique** quand aucun runtime de référence n'est disponible :
   vérifier des invariants (linéarité, conservation d'énergie/masse, symétrie,
   monotonie) que le port doit préserver.
3. **Rapport signé** hash-chaîné, réutilisant l'infrastructure d'audit
   existante (CCOS dans `scirust-sciagent::ccos`, chaîne SHA-256 de
   `scirust-mcp`). Chaque port acceptable produit une preuve rejouable.
4. **Gate CI.** Aucun port fusionné sans oracle vert ; la tolérance et le
   corpus font partie du livrable, pas un à-côté.

---

## 9. Feuille de route par phases

- **Phase 0 — MVP (tranche verticale la plus fine). ✅ LIVRÉE.** Sous-ensemble
  Python/NumPy → Rust déterministe std-only, **gated par oracle différentiel
  contre NumPy réel**. Objectif atteint : **le pipeline est prouvé de bout en
  bout** (front-end → SIR → lowering → oracle vert). Corpus livré (7 cas,
  200 essais chacun) : intégrateur **RK4** (scalaire), **dot**, **norme**
  euclidienne, **moyenne pondérée**, **cumsum** (boucle + tableau en sortie),
  **saxpy** (broadcast), **tanh** élémentaire. *Écart honnête vs le plan
  initial :* `np.linalg.solve` et `np.fft` ne sont **pas** encore livrés — ils
  exigent le routage vers `scirust-solvers`/`scirust-signal`, prévu en Phase 1.
- **Phase 1 — Élargir Python + router vers les noyaux vérifiés.** ✅ **en cours,
  déjà livré :** contrôle de flux `if`/`elif`/`else` + `while`, et le premier
  **routage `np.linalg.solve` → `scirust-solvers`** (résolution LU vérifiée,
  cas d'oracle compilé via cargo). ⏳ **reste :** `np.fft` → `scirust-signal`,
  tableaux 2-D généraux, fonctions multiples. Secteurs débloqués par le
  routage : robotique, finance, imagerie.
- **Phase 2 — MATLAB.** Sous-ensemble matriciel + broadcasting ; secteurs :
  aéro, automobile, contrôle.
- **Phase 3 — Fortran.** Routines numériques héritées ; secteurs : météo,
  géophysique, spatial, physique.
- **Phase 4 — C/C++.** Sous-ensemble numérique via pré-passe `c2rust`.

Chaque phase livre : contrat de sous-ensemble + corpus d'oracle + matrice des
secteurs réellement débloqués.

---

## 9-bis. État d'implémentation (mesuré, pas revendiqué)

| Brique du pipeline (§2)                     | Statut | Emplacement |
|---------------------------------------------|--------|-------------|
| Front-end Python/NumPy (lexer + parser)     | ✅ livré | `scirust-transpiler/src/front_python/` |
| Scientific IR typée (scalaire/tableau/int)  | ✅ livré | `scirust-transpiler/src/sir.rs` |
| Lowering + inférence de types/formes        | ✅ livré | `scirust-transpiler/src/lower.rs` |
| Émission Rust déterministe (ordre pinné)    | ✅ livré | `scirust-transpiler/src/emit.rs` |
| Oracle différentiel contre NumPy réel       | ✅ livré | `scirust-transpiler/examples/oracle.rs` |
| Tests unitaires (gate CI, sans Python)      | ✅ livré | `scirust-transpiler/src/lib.rs` (13 tests) |
| Contrôle de flux `if`/`elif`/`else` + comparaisons | ✅ livré (Phase 1) | `front_python/` + `sir.rs` + `emit.rs` |
| Boucles `while` (algorithmes itératifs)     | ✅ livré (Phase 1) | `front_python/` + `sir.rs` + `emit.rs` |
| Routage `np.linalg.solve` (LU) + `np.linalg.det` → `scirust-solvers` | ✅ livré (Phase 1) | `sir.rs` (`LinSolve`, `Det`, `required_crates`) + `emit.rs` |
| Routage `np.fft` → `scirust-signal`         | ⏳ Phase 1 | — |
| Tableaux 2-D généraux                       | ⏳ Phase 1 | — |
| Front-ends MATLAB / Fortran / C++           | ⏳ Phases 2-4 | — |

**Résultat de l'oracle (reproductible).**

```
$ cargo run -p scirust-transpiler --example oracle
tolerance: |Δ| ≤ 1e-7 + 1e-9·|numpy|, 200 trials/case
  ✓ rk4_step / dot / norm / weighted_mean / cumsum / saxpy / tanh_activation
  ✓ relu / clamp / sign            (if/elif/else — Phase 1)
  ✓ newton_sqrt / newton_conv      (while — Phase 1)
  ✓ linalg.solve / linalg.det      (routed to scirust-solvers, cargo-compiled — Phase 1)
  ✓ sin/cos/abs / exp / ** / ones  (full intrinsic & operator coverage)
  ORACLE GREEN — 19/19 cases match NumPy within tolerance
```

Un point d'entrée unique lance toute la suite (tests unitaires + oracle) avec
un rapport et un code de sortie non nul à la moindre divergence :

```
$ ./scripts/test_transpiler.sh
```

Vérification de non-vacuité : l'injection d'un opérateur faux dans l'émetteur
(`*` → `+`) fait passer **4/7 cas au ROUGE** — le gate mord réellement.

> **Note de réutilisation `codetrans`.** Le §10 vise `codetrans::Expr` comme
> backend d'émission. En pratique son nœud `Function` porte des paramètres
> **non typés** (`Vec<String>`), ce qui ne permet pas d'émettre des signatures
> Rust typées (`&[f64]` vs `f64`) qui *compilent*. Le MVP utilise donc un
> émetteur dédié typé ; unifier avec `codetrans` (en étendant son `Function`
> avec des types de paramètres) reste un travail ultérieur.

---

## 10. Réutilisation concrète de l'existant (points d'ancrage dans le code)

| Besoin | Réutiliser | Fichier |
|--------|-----------|---------|
| Backend d'émission Rust | `codetrans::Expr` + pretty-printer | `scirust-codetrans/src/lib.rs` (`Display for Expr`, l.249) |
| Passes d'optimisation | 20 règles (`optimization_rules`, CSE, DCE, LICM) | `scirust-codetrans/src/lib.rs` (l.1958+) |
| Vocabulaire cible | solvers, signal, estimation, core, vision… | crates `scirust-*` (§4-5) |
| Preuve / audit | CCOS + chaîne SHA-256 | `scirust-sciagent::ccos`, `scirust-mcp` |
| Pilotage agent | exposer le transpileur comme outil MCP | `scirust-mcp` |
| Déterminisme flottant | réductions à ordre pinné, fingerprint | `scirust-core` |

Un nouveau crate `scirust-transpiler` (front-ends + SIR + lowering + oracle)
se poserait **au-dessus** de ces briques, sans les dupliquer.

---

## 11. Frontière honnête — ce qui ne sera PAS livré (à court terme)

Fidèle à la doctrine du dépôt, on énonce d'emblée les non-objectifs :

- **Pas de « tout langage / tout programme ».** Sous-ensembles scientifiques
  statiquement analysables uniquement. Un `eval`, une réflexion, un
  monkeypatch Python → **refus diagnostiqué**, pas de devinette.
- **Pas de reproductibilité bit-exacte *cross-language* garantie.** L'ordre des
  opérations de NumPy/BLAS n'est pas spécifié ; on garantit (a) une **tolérance
  déclarée** source ⇄ Rust et (b) la **bit-exactitude *interne* Rust**
  (indépendante du nombre de threads, via `scirust-core`). Prétendre l'égalité
  bit-à-bit avec CPython serait malhonnête.
- **Pas de traduction de l'UB C/C++.** Comportement indéfini → signalé, jamais
  « interprété ».
- **La performance vient du routage, pas d'une magie de transpilation.** Le
  Rust émis vise d'abord correction + déterminisme ; la vitesse provient des
  noyaux SIMD/GPU `scirust-*` ciblés, mesurée, pas supposée.

---

## 12. Critères d'acceptation — « comment être sûr »

Un port est réputé livrable si et seulement si :

1. l'**oracle est vert** sur le corpus déclaré (différentiel et/ou
   métamorphique) ;
2. la **tolérance déclarée** est respectée sur tout le corpus ;
3. la **bit-exactitude interne** est vérifiée (fingerprint identique
   1/2/4/8 threads) ;
4. **zéro `unsafe` non justifié** ; aliasing tracé ;
5. un **rapport signé** hash-chaîné est produit et rejouable ;
6. le **sous-ensemble couvert** est documenté, ainsi que ce qui a été refusé.

Tant que ces six gates ne sont pas outillés, la réponse honnête à « SciRust
sait-il transpiler mon code ? » reste **« pas encore automatiquement — voici
le plan et les garanties visées »**, et non un « oui » marketing.

---

*Voir aussi : `docs/DOMAIN_ROADMAP.md` (secteurs régulés), `docs/ARCHITECTURE.md`
(architecture du runtime), `scirust-codetrans` (backend d'émission),
`scirust-mcp` (pilotage agent + audit).*
