# Changelog

Le format suit [Keep a Changelog](https://keepachangelog.com/) ;
versions sémantiques à partir de la prochaine release taguée.

## [Non publié]

### Ajouté — synergie d'écosystème (CCOS, SLHAv2)
- **Guard à garantie statistique** (`nn::guard::StatisticalGuard`) : une porte de réponse à
  **garantie de couverture sans hypothèse de distribution**, pour alimenter le `guard` de **CCOS**
  (valider/abstenir sur la sortie d'un modèle) sans seuil ad-hoc. À partir des probabilités de
  classe d'une décision, le guard forme l'**ensemble de prédiction conforme** (#21,
  `ConformalClassifier`) et en tire un verdict : une seule classe franchit `1−q̂` ⇒ **Accept** ;
  plusieurs ⇒ **Abstain** (ambigu) ; aucune ⇒ **Reject** (hors-distribution). La calibration
  conforme garantit que la vraie classe est dans l'ensemble avec proba **≥ 1−α** sur données
  échangeables, *quelle que soit la distribution* — le guard ne laisse donc prouvablement pas
  filer la bonne réponse plus d'une fraction `α` du temps. Oracle honnête : **couverture empirique
  ≥ 1−α** sur données fraîches (3-classes, déterministe) + logique de verdict (confiant→Accept,
  partagé→Abstain, plat/OOD→Reject). Les deep ensembles (#40) donnent un signal épistémique
  complémentaire pour le flag OOD.
- **Codec KV accéléré SIMD, bit-exact** (`scirust_simd::ops::dequantize_int4_into`, câblé dans
  `nn::elastic_kv_cache`) : la déquantification INT4 (`out[i]=code[i]·échelle`) passe par le kernel
  SIMD `mul_f32` ; étant **élémentaire** (pas de réduction) et un produit IEEE-754 identique par
  lane et en scalaire, le résultat est **bit-identique entre largeurs SIMD et plateformes** — le
  chemin rapide du codec KV **sans casser le déterminisme** (les réductions cosinus/attention
  restent sur le chemin déterministe). Oracle : SIMD ≡ scalaire **bit-exact** pour toute longueur
  (y compris < une lane) et une plage d'échelles.
- **Journal d'attestation hash-chaîné** (`scirust_runtime::attest`) : le pont de l'**inférence
  vérifiable** de scirust (`vinfer` #80) vers l'`event_log` de **CCOS**. Chaque `InferenceEvent`
  fige l'engagement du modèle, le hash de l'entrée et le hash de la sortie, et se chaîne au
  précédent par un **hash SHA-256** (`entréeₙ = H(entréeₙ₋₁ ‖ seq ‖ engagement ‖ entrée ‖ sortie)`)
  — exactement la forme append-only et inviolable de CCOS, donc les inférences d'un runtime scirust
  s'ingèrent dans son journal d'audit. Recalculer la chaîne re-dérive la **même tête** (replay
  déterministe) ; toute mutation ou réordonnancement la **casse**. `attest_and_record` vérifie en
  plus, *avant* d'ajouter, que la paire `(entrée, sortie)` est une inférence **authentique** du
  modèle engagé (Freivalds sur `GF(p)`, #80) — la chaîne n'atteste donc que des inférences réelles.
  Oracle honnête : la chaîne se vérifie et se rejoue (même tête) ; falsification d'un événement /
  réordonnancement **détectés** ; une inférence authentique est attestée et chaînée tandis qu'une
  sortie **falsifiée est rejetée** (journal inchangé). Complète la pile de preuve (#3, `proof`,
  DiFR #5, `vinfer` #80).
- **KV-cache compressé élastique** (`nn::elastic_kv_cache`) : la primitive déterministe
  partagée derrière **SLHAv2** (compresser le KV-cache pour faire tourner un LLM dans le cache
  du CPU plutôt que sur un GPU hors de prix) et **CCOS** (paging à mémoire bornée), bâtie sur la
  quantification et le déterminisme de scirust. Une paire clé/valeur d'attention est compressée
  en une `KvTile` par quantification **INT4 à deux niveaux** (base symétrique + **résidu** INT4 —
  le « residual tracking » de SLHAv2), chaque niveau à **échelles adaptatives par groupe**
  (`quantize_int4_grouped` : une échelle plus fine par groupe de canaux ⇒ « adaptive scaling »
  cosine-aware de SLHAv2, dans l'esprit per-canal de KVQuant #68), ce qui porte la fidélité
  **cosinus** au-delà de 0,99 tout en réduisant l'empreinte plusieurs fois par rapport au `f32`. L'`ElasticKvCache` conserve ces
  tuiles sous un **budget** optionnel et évince la plus ancienne au dépassement (soft-paging /
  mémoire élastique — l'abstraction de paging commune avec CCOS), et sert l'attention directement
  depuis les tuiles compressées en réutilisant `contiguous_attention` (#63), si bien que le seul
  écart avec un cache pleine précision est l'erreur de compression (mesurée). Oracle honnête :
  reconstruction à **fidélité cosinus** élevée (>0,95, le niveau résidu battant strictement la
  base seule) ; **attention compressée ≈ pleine** (cosinus >0,99) ; **ratio de compression** ≥3×
  vs `f32` ; cache **borné** sous budget (la plus ancienne évincée) et **bit-exact déterministe**.
  Codec exposé (`quantize_int4`/`dequantize_int4`/`KvTile`/`cosine_similarity`) pour être consommé
  par SLHAv2/CCOS. Rejoint KVQuant (#68) et PagedAttention (#63) dans la pile KV-cache.

### Corrigé
- **SIMD `portable` — bug d'alignement (résultats faux, non déterministe)** :
  `add_f32/f64_inplace`, `dot_f32/f64` et `fma_f32` (`scirust-simd::portable`)
  découpaient **chaque opérande indépendamment** via `as_simd`/`as_simd_mut`.
  Quand deux slices avaient des alignements mémoire différents (fréquent : dépend
  de l'allocation), les boucles SIMD du cœur appariaient des lanes **décalées** →
  résultats **incorrects**, de façon **non déterministe** (d'où le test
  `test_add_f32_inplace` qui échouait ~30–50 % des lancers). Réécrites avec
  `chunks_exact`, qui apparie le bloc k de chaque slice à l'identique quel que
  soit l'alignement. Ajout d'un test de régression couvrant tous les décalages
  relatifs (add/dot/fma vs référence scalaire) ; 12/12 lancers verts. Au passage,
  un `needless_return` dans `complex.rs` (chemin `portable-simd`) corrigé.

### Ajouté — campagne « faire grandir scirust »
- **Reluplex — vérification *complète* de style SMT** (`nn::ibp::reluplex_verify`/
  `reluplex_unstable_count`, Katz et al. 2017, roadmap #4) : une recherche de **satisfiabilité**
  d'un contre-exemple par **case-splitting des phases ReLU** — mais **paresseuse**, la signature de
  Reluplex : un neurone dont l'intervalle de pré-activation reste entièrement d'un côté de 0 sur la
  boîte est **stable**, donc sa phase est **forcée** (jamais scindée) ; seuls les neurones
  **instables** sont scindés, soit `2^instables` feuilles au lieu des `2^cachés` de l'énumération
  *eager* du MILP (#31). Sur chaque feuille (un patron ReLU complet) le réseau est affine et un
  contre-exemple est cherché en minimisant chaque marge sur la région du patron (le **LP 2D exact**
  partagé avec le vérificateur MILP) ; on renvoie le **premier** contre-exemple trouvé (SAT) ou
  Robust. Distinct du branch-and-bound (#26, scinde le domaine d'entrée) et du MILP (#31, énumère
  *tous* les patrons) par le **splitting paresseux des phases ReLU**. Oracle honnête : **accord avec
  MILP** sur tout un balayage de rayons (deux méthodes exactes ⇒ mêmes décisions) ; contre-exemple
  réel (marge ≤ 0, dans la boîte) ; à petit rayon, **moins de neurones scindés** que `cachés`
  (élimination par bornes) ; déterministe. Réseau (2 entrées, 1 couche). **Clôt la pile de
  vérification** (IBP, CROWN, zonotopes, DeepPoly, randomized smoothing, Lipschitz, CROWN-IBP, BaB,
  MILP, Reluplex).
- **Inférence vérifiable — argument cryptographique compact** (`scirust_runtime::vinfer`,
  ZK-based Verifiable ML, roadmap #80) : prolonge les certificats `proof` de la ré-exécution
  bit-exacte vers une **garantie de soundness succincte**. Le modèle (une couche linéaire entière
  quantifiée sur le corps premier `GF(p)`, `p = 2³¹−1`) est **engagé** par le hachage de ses poids.
  Pour vérifier une sortie batchée `Y` revendiquée pour des entrées `X`, le vérifieur exécute la
  **vérification de Freivalds** sur `GF(p)` : tirer un `r` aléatoire et tester `W·(X·r) = Y·r`.
  Calculer `W·(X·r)` coûte `O(out·in + in·b)` contre `O(out·in·b)` pour recalculer `Y = W·X`, donc
  pour un batch c'est **succinct** (sous-linéaire dans le coût de recalcul). Un `Y` faux passe avec
  proba `≤ 1/p` par défi, donc quelques défis donnent une erreur de soundness négligeable. Le défi
  `r` est dérivé par **Fiat-Shamir** d'un hachage de `(engagement, X, Y)`, donc non-interactif et
  **lié à la sortie revendiquée** (le prouveur ne peut pas adapter `Y` à un `r` connu). Oracle
  honnête : accepte une inférence correcte (déterministe) ; **soundness** — sur 1000 falsifications
  aléatoires d'une entrée de la sortie, **toutes** rejetées ; l'engagement **lie** le modèle
  (vérifier contre l'engagement d'un autre modèle échoue) ; Fiat-Shamir **lie** la sortie (la sortie
  valide d'**autres** entrées est rejetée pour `X`). Fournit la **soundness** cryptographique (la
  sortie provient prouvablement du modèle engagé), **pas** le zero-knowledge — le vérifieur détient
  les poids ; les zk-SNARK cachant les poids restent hors périmètre. Couronne la pile de preuve
  (sommation reproductible #3, certificats `proof`, DiFR #5).
- **DiFR — vérification d'inférence malgré le non-déterminisme** (`scirust_runtime::difr::difr_verify`,
  2025, roadmap #5) : les certificats [`proof`] vérifient une inférence par **ré-exécution
  bit-exacte** — ce qui ne marche que si le vérificateur reproduit l'arithmétique du prouveur à
  l'identique. Or sur un **matériel différent** (largeurs SIMD, FMA, nombre de threads) la sommation
  flottante est **non-déterministe**, donc un contrôle bit-exact rejetterait des sorties pourtant
  honnêtes. DiFR vérifie *malgré* cela : il recompute une **référence canonique** avec
  `reproducible_dot` (produits et somme accumulés en `f64`, indépendant de l'ordre) et accepte la
  sortie revendiquée ssi elle se trouve dans une **enveloppe d'erreur flottante saine** de cette
  référence. *Tout* calcul `f32` honnête — dans *n'importe quel* ordre de sommation — est
  prouvablement dans l'enveloppe (donc accepté) ; une sortie **falsifiée** au-delà est rejetée.
  L'enveloppe est la borne d'arrondi du produit scalaire `γ·Σ|termes|` propagée à travers les
  couches (la ReLU est 1-lipschitzienne, elle la transmet sans l'amplifier) et reste **minuscule**
  (quelques ppm de l'échelle d'activation), si bien que le contrôle attrape toute falsification
  signifiante. Oracle honnête : accepte un calcul `f32` dans un **ordre de sommation différent** ;
  enveloppe **saine** (1000 ordres aléatoires, tous acceptés) et **fine** (< 0,001 de l'échelle) ;
  **rejette** une falsification (au-delà de l'enveloppe, ici de quoi changer la classe prédite) ;
  déterministe. Prolonge la sommation reproductible (#3) et l'outillage de preuve d'inférence.
- **MILP — vérification *exacte*** (`nn::ibp::milp_min_margin`/`milp_verify_robustness`, Tjeng
  et al. 2019, roadmap #31) : la vérification exacte d'un réseau ReLU par la formulation MILP.
  L'observation clé : les **patrons d'activation** des ReLU sont précisément les variables
  **binaires** du MILP, et sur le domaine d'un patron fixé le réseau est **affine**. Pour un petit
  réseau (2 entrées, 1 couche cachée) on **énumère** les patrons et on résout chaque LP
  **exactement** — la marge `logitₜ − logitⱼ` y est affine, minimisée sur la boîte intersectée
  avec les demi-espaces d'activation du patron par **énumération des sommets** du polygone 2D (pas
  de simplexe fragile : robuste et exact). Le minimum global sur tous les patrons et toutes les
  classes concurrentes est donc **exact** ; `> 0` ⇒ robuste, sinon l'argmin est un **contre-exemple
  exact**. Oracle honnête : le minimum énumérée **égale la force brute** (il minore toute valeur
  d'une grille fine et la grille s'en approche), le contre-exemple est **réel** (marge ≤ 0, dans la
  boîte), et — étant exact — il est **≥ la borne inférieure (saine) de DeepPoly** partout et
  **strictement plus serré** à certains rayons ; déterministe. Distinct du branch-and-bound (#26),
  complet **à tolérance près** : MILP est exact (tranche même la frontière de mesure nulle).
- **Branch-and-bound — vérification *complète*** (`nn::ibp::verify_robustness`/`BabResult`,
  GCP-CROWN, Zhang et al. 2022, roadmap #26) : là où IBP/CROWN/DeepPoly donnent **une** borne
  *saine mais incomplète*, le branch-and-bound **décide**. Il borne les **marges** par classe
  (`logitₜ − logitⱼ`, fusionnées dans une dernière couche pour que DeepPoly suive la corrélation)
  sur la boîte d'entrée ; si toutes les bornes inférieures sont `> 0` la boîte est **prouvée
  robuste** ; sinon il sonde le **centre** de la boîte pour un **contre-exemple concret**, et à
  défaut **scinde** la boîte le long de son axe le plus large et récurse. Comme les sous-boîtes
  rétrécissent, la relaxation ReLU de DeepPoly devient exacte, si bien que la recherche **tranche**
  (jusqu'à une tolérance) — prouvant des cas qu'une borne unique ne peut pas, et renvoyant un
  contre-exemple réel quand la classe peut effectivement changer. Oracle honnête : `Robust` est
  **sain** (5000 points échantillonnés bien classés) ; le **rayon ℓ∞ certifié dépasse strictement**
  celui de DeepPoly seul (et la région supplémentaire est échantillonnée robuste) ; `Unsafe`
  renvoie un **vrai** contre-exemple (mal classé, dans la boîte) ; déterministe. Exposé dans la CLI
  `certify`. (Le branchement est sur le **domaine d'entrée** ; le split des ReLU instables et les
  plans coupants de GCP-CROWN ne sont pas implémentés.) Couronne la pile de vérification (IBP #1,
  CROWN #2, zonotopes #29, DeepPoly #28, CROWN-IBP #30).
- **DeepPoly — domaine abstrait relationnel** (`nn::ibp::deeppoly_certify`/`IbpMlp::certify_deeppoly`,
  Singh et al. 2019, roadmap #28) : un vérificateur de robustesse plus précis qu'IBP. Là où IBP
  traite chaque neurone par un simple intervalle (perdant toute corrélation), DeepPoly garde pour
  chaque neurone une **borne basse et haute affines en les entrées** du réseau et les **back-
  substitue** couche par couche. La relaxation ReLU est **asymétrique** : pour un pré-activation de
  plage `[l,u]` instable, la borne supérieure est la **corde** `z ≤ (u/(u−l))(y−l)` et la borne
  inférieure `z ≥ λy` avec `λ` choisi pour **minimiser l'aire** de la relaxation (`λ=1` si `u>−l`,
  sinon `0`). Comme les bornes restent affines, les corrélations sont préservées et le résultat est
  plus serré qu'IBP — **à n'importe quelle profondeur** (là où `crown_bounds` était limité à 2
  couches). Oracle honnête : **sain** (4000 points échantillonnés ∈ boîte certifiée, MLP 3 couches)
  + **strictement plus serré qu'IBP** sur `relu(x)+relu(−x)=|x|` sur `x∈[−1,1]` (DeepPoly donne la
  boîte **exacte** [0,1] car le `x` s'annule dans la borne supérieure, vs IBP [0,2]) + déterminisme.
  Exposé dans la CLI `certify` (à côté d'IBP, CROWN, zonotopes, smoothing). Prolonge IBP (#1) /
  CROWN (#2) / zonotopes (#29).
- **CROWN-IBP — entraînement certifié (vérifié)** (`nn::crown_ibp::CrownIbpMlp`, Zhang et al.
  2020, roadmap #30) : l'entraînement ordinaire minimise la perte aux entrées *concrètes* — un
  réseau peut les ajuster parfaitement et pourtant **changer de prédiction** sous une perturbation
  minime. CROWN-IBP entraîne au contraire sur une **borne certifiée de la perte du pire cas** sur
  une boule ℓ∞ autour de chaque entrée, rendant le réseau **prouvablement** robuste. L'idée clé :
  la **propagation par intervalles (IBP) est différentiable**. Pour une couche affine `y=x·W+b`,
  la boîte se transforme en `centre'=centre·W+b`, `rayon'=rayon·|W|` — et `|W|=relu(W)+relu(−W)`,
  donc toute la borne (y compris le `|W|` qui semblait exiger un op `abs` dédié) tourne sur la
  tape N-D ; la ReLU sur un intervalle `[l,u]` devient `[relu(l),relu(u)]`. Les **logits robustes**
  placent la vraie classe à sa borne **inférieure** et les autres à leur borne **supérieure**
  (`zₜ=cₜ−rₜ`, `z_j=c_j+r_j`) : une cross-entropy faible dessus signifie que la vraie classe gagne
  *même dans le pire cas* — le point est **certifié**. Oracle honnête : la propagation IBP sur la
  tape **coïncide** avec le vérificateur de référence `IbpMlp` (plain `f32`) et est **saine**
  (2000 points échantillonnés ∈ boîte certifiée) ; après entraînement certifié, le **rayon ℓ∞
  certifié croît** nettement (réseau robuste-entraîné vs accuracy-only, tous deux classant juste à
  100 %) + déterminisme bit-exact. Prolonge IBP (#1) / CROWN (#2) / zonotopes (#29) vers
  l'entraînement.
- **Sophia — optimiseur de 2e ordre clippé** (`nn::nd_optim::NdSophia`, Liu et al. 2023, roadmap
  #44) : Sophia met à l'échelle le momentum de chaque coordonnée par une estimation de la
  **Hessienne diagonale** et **clippe** le résultat : `θ ← θ − lr·clip(m/max(γ·h,eps),ρ)`. Les
  directions plates (petite courbure `h`) prennent un pas borné de type signe ; les directions
  courbées prennent un pas de type **Newton** `m/h` — d'où une robustesse au mauvais
  conditionnement. La Hessienne diagonale est estimée par un **estimateur de Hutchinson** avec un
  **produit Hessien-vecteur en différences finies** : avec un vecteur de signes `v∈{±1}` seedé,
  `Hv ≈ (∇L(θ+εv) − ∇L(θ))/ε` et `ĥ = v⊙Hv` (pour un quadratique c'est la Hessienne diagonale
  **exacte**, mon ancien blocage « il faut un op `abs` sur la tape » était infondé — le clipping
  se fait dans l'optimiseur en `f32`, pas sur la tape). Comme SAM, cela demande **deux** calculs
  de gradient par pas, donc l'appelant orchestre `probe` (perturbe `θ` de `εv`) puis `step`
  (restaure `θ`, applique la mise à jour) — optimiseur **de bibliothèque hors de la boucle
  `lm --opt`** (à un seul gradient). Oracle honnête : **converge sur un quadratique mal
  conditionné** (courbures 4 vs 0,25, conditionnement 16) là où le pas Newton par coordonnée
  neutralise le conditionnement + déterminisme bit-exact (probe seedé). Rejoint la famille
  d'optimiseurs (Adam, Lion, Muon, Shampoo, SOAP, Adafactor, LAMB, Adan, Prodigy, SAM, …).
- **QuIP# — incohérence Hadamard + codebook lattice E8** (`quantization::quantize_quip`/
  `nearest_e8`/`random_hadamard_transform`, Tseng et al. 2024, roadmap #64) : deux idées. (1) Le
  **traitement d'incohérence** : multiplier les poids par une **transformée de Hadamard
  randomisée** (signes ±1 seedés puis FWHT, *orthogonale*) étale les aberrants à travers les
  coordonnées et **rétrécit la plage dynamique** ; à budget de bits **égal**, les `2^bits`
  niveaux fixes résolvent alors bien mieux le gros des poids (le RTN scalaire devait, lui, étaler
  ses rares niveaux sur toute la plage pour couvrir les aberrants). (2) Le codebook **lattice
  E8** : quantifier les poids tournés par blocs de 8 vers le point le plus proche du **réseau
  E8** (`D8 ∪ (D8+½·1)`, décodeur fermé de Conway-Sloane) — le plus dense en dimension 8, avec un
  **moment quadratique** plus bas que la grille cubique à densité **égale** (gain de packing
  ~14 %). Oracle honnête : la RHT est orthogonale (round-trip exact) et **réduit la plage** d'un
  poids à aberrants ; le décodeur E8 renvoie un point **valide** du réseau (coords toutes
  entières ou toutes demi-entières, somme paire) et quantifie **mieux que la grille cubique en
  moyenne** (gain lattice mesuré sur 4000 vecteurs) ; bout-en-bout, QuIP# reconstruit **mieux que
  le RTN** scalaire à budget 2-bit sur des poids à aberrants épars + déterminisme bit-exact. (Le
  grand Hadamard global et le codebook E8P curé de QuIP# sont simplifiés ici en un Hadamard par
  bloc de 8 et le réseau E8 nu.) Complète la famille de quantification (AQLM, GPTQ, AWQ, NF4,
  SqueezeLLM, SpQR, KVQuant, LLM.int8, OmniQuant, BitNet).
- **AQLM — quantification additive multi-codebook** (`quantization::quantize_aqlm`/`AqlmResult`,
  Egiazarian et al. 2024, roadmap #70) : au lieu de quantifier chaque poids **scalairement**, AQLM
  découpe les poids en **groupes** de dimension `g` et approxime chaque groupe par la **somme**
  d'un mot de code tiré de chacun de `M` codebooks appris (de `K` mots chacun). Les codebooks sont
  initialisés par **k-means résiduel** puis affinés par **optimisation alternée** : ré-encoder
  chaque groupe (assignation résiduelle gloutonne à travers les `M` codebooks) puis ré-ajuster
  chaque codebook par moindres carrés sachant la contribution des autres (la beam search d'AQLM
  est ici simplifiée en assignation gloutonne — documenté). Comme les mots de code sont des
  **vecteurs**, la quantification additive capte la **structure inter-dimensions** que le
  round-to-nearest scalaire ignore, d'où une bien meilleure reconstruction à bas budget. Oracle
  honnête : erreur **< 0,7× RTN** scalaire à budget ~2-bit **égal** (`M·log₂K/g`) sur des poids
  structurés (groupes bâtis sur quelques directions prototypes) + round-trip exact (longueur non
  divisible, padding zéro) + déterminisme bit-exact. Rejoint la famille de quantification (GPTQ,
  AWQ, NF4, SqueezeLLM, SpQR, KVQuant, LLM.int8, OmniQuant, BitNet).
- **S5 — SSM MIMO + scan associatif parallèle** (`nn::nd_layers::s5_scan`/`s5_parallel_scan`/
  `NdS5`, Smith et al. 2023, roadmap #52) : contrairement aux SSM **SISO par canal** de S4D
  (chaque canal son propre état indépendant), S5 pilote un **unique état partagé** de dimension
  `n` avec **toutes** les entrées via une matrice `B`, et lit `m` sorties via `C` (d'où *MIMO*) :
  `hₜ=Ā⊙hₜ₋₁+xₜB`, `yₜ=hₜC`. La récurrence étant linéaire, elle se calcule par un **scan
  associatif** : l'élément `(aₜ,uₜ)` représente la carte affine `h↦aₜ⊙h+uₜ`, et ces cartes se
  composent par l'opérateur **associatif** `(a₁,u₁)∘(a₂,u₂)=(a₂⊙a₁, a₂⊙u₁+u₂)`. Un scan
  inclusif de **Hillis-Steele** (ordre de doublage `log₂ seq` fixe ⇒ **déterministe**) produit
  tous les états préfixes en parallèle. Oracle honnête : le **scan parallèle ≡ la récurrence
  séquentielle** — testé avec `aₜ` **variable dans le temps** (un vrai scan associatif, pas le
  cas trivial constant), ce qui prouve l'associativité qui licencie la parallélisation ;
  `s5_scan` sur la tape ≡ référence MIMO écrite à la main (valide le câblage `B`/`C`) ;
  **gradient check** (x, Ā, B, C) ; `NdS5` entraîne (MSE↓) + déterminisme bit-exact. Complète la
  famille espace-d'états (Mamba, Mamba-2/SSD, S4).
- **Mamba-2 / SSD — dualité espace-d'états ↔ attention** (`nn::nd_layers::ssd_dual`/`NdMamba2`,
  Dao & Gu 2024, roadmap #50) : Mamba-2 restreint la matrice d'état du SSM à une **décroissance
  scalaire** `aₜ` par pas (au lieu du `A` diagonal par canal de Mamba). Cette restriction rend
  la récurrence linéaire `Hₜ=aₜHₜ₋₁+xₜBₜᵀ` (état `d×n`), `yₜ=HₜCₜ` **exactement égale** à une
  unique forme quadratique masquée de type attention — la **dualité** : `Y=(L⊙CBᵀ)X` avec
  `L[i,j]=∏_{j<k≤i}aₖ` pour `i≥j`. Calculée sur la tape : le log-décroissance cumulé
  `cumlogᵢ=Σ_{k≤i}a_logₖ` est une **préfixe-somme** (matmul avec une matrice triangulaire de
  uns), `L=exp(cumlogᵢ−cumlogⱼ)` masquée causale, `Y=(L⊙CBᵀ)X`. `a_log=log a` est le paramètre
  (en Mamba-2 `a_logₜ=Δₜ·A`), donc **aucun op `log`** n'est requis ; le masque est appliqué
  **avant** l'`exp` (`diff⊙mask`, puis `exp`, puis `⊙mask`) pour garder l'exposant borné dans le
  triangle supérieur (évite `inf·0=NaN`) et y annuler exactement. Oracle honnête : la **forme
  duale ≡ la récurrence séquentielle** écrite à la main (c'est littéralement la dualité du
  papier) ; **gradient check** (x, B, C, a_log) ; `NdMamba2` entraîne (MSE↓) + déterminisme
  bit-exact. Rejoint Mamba/S4/RWKV/RetNet/GLA/HGRN/DeltaNet/xLSTM/Hyena.
- **FNO — opérateur neuronal de Fourier** (`nn::fno::FnoSpectralConv1d`/`NdFno`, Li et al.
  2021, roadmap #75) : un opérateur neuronal apprend une application entre **fonctions** (p.ex.
  condition initiale ↦ solution de PDE), pas entre vecteurs de taille fixe. FNO réalise
  l'intégrale de noyau **globale** dans le **domaine de Fourier** : transformer le signal
  échantillonné, garder les `modes` plus basses fréquences, multiplier chaque mode par un
  **poids complexe appris** `R_k=Ar_k+iAi_k` (matrice `width×width`, mélange de canaux), puis
  transformer en sens inverse. La DFT réelle et son inverse sont des **matrices cosinus/sinus
  fixes** : tout le transform est un matmul ordinaire (déterministe) que la tape N-D dérive
  directement — **sans FFT, sans type complexe, sans nouvel op** ; les poids par mode sont
  appliqués par un matmul **par lots** (`bmm`) sur les modes. Bloc FNO complet :
  `σ(SpectralConv(v)+W·v)`. Oracle honnête : reconstruction **exacte** d'un signal band-limité
  aux modes gardés (DFT⁻¹∘DFT, valide les matrices + l'inverse unilatéral facteur-2) ;
  **gradient check** par différences finies (signal, Ar, Ai) ; comme la dérivation est
  diagonale en Fourier (`d/dx↔×ik`), une seule conv spectrale **apprend l'opérateur de
  dérivation** `sin(ωx+φ)↦ω cos(ωx+φ)` et **généralise à une phase non vue** (MSE test <0,02,
  ajustement convexe) ; déterminisme bit-exact. Rejoint la famille calcul scientifique
  (Neural ODE, PINN, DeepONet, KAN).
- **Hyena — convolutions longues implicites + gating** (`nn::nd_layers::hyena_long_conv`/
  `NdHyena`, Poli et al. 2023, roadmap #56) : un mélangeur de tokens **sans attention**. La
  portée longue vient d'une **convolution causale** dont le filtre n'est pas stocké tap par
  tap mais **généré** par un petit MLP à partir d'un encodage positionnel fixe, puis fenêtré
  par une décroissance exponentielle apprenable `exp(−γ·t̄)` par canal — c'est ce qui permet
  des filtres **longs à peu de paramètres** (le coeur de Hyena). L'équivalent du rôle de
  l'attention (la dépendance aux données) est fourni par un **gating multiplicatif** :
  `z=x1⊙(h1*v)` puis `z=x2⊙(h2*z)` (ordre 2). La convolution causale par canal
  `y[t,c]=Σ_τ h[τ,c]·u[t−τ,c]` est exprimée sur la tape comme `Σ_τ h[τ,:]⊙(Sτ·u)` avec des
  **matrices de décalage constantes** `Sτ` (distribuer le matmul sur les taps apprenables ⇒
  différentiable en `u` et `h` sans op scatter). Oracle honnête : conv ≡ référence causale
  écrite à la main ; **gradient check** par différences finies (`u`, `h`) ; entraînement
  `NdHyena` (MSE↓) + déterminisme bit-exact. Rejoint la famille de modèles de séquence.
- **xLSTM — sLSTM scalaire + mLSTM matriciel** (`nn::nd_layers::slstm_scan`/`mlstm_scan`/
  `NdXlstm`, Beck et al. 2024, roadmap #57) : le LSTM étendu remplace les portes sigmoïdes
  de l'entrée par une **porte exponentielle** `iₜ=exp(ĩₜ)` accompagnée d'un **état
  normaliseur** `nₜ=fₜnₜ₋₁+iₜ`, la sortie étant `hₜ=oₜ⊙(cₜ/nₜ)`. Comme `cₜ/nₜ` est une
  moyenne pondérée positive de `zₜ=tanh∈(−1,1)`, la sortie reste bornée dans (−1,1) : la
  récurrence est **stable sans le stabilisateur log** (omis, c'est un pur dispositif
  numérique qui s'annule dans le ratio). `tanh` est construit à partir du seul op `sigmoid`
  via l'identité exacte `tanh(x)=2σ(2x)−1`. La variante **mLSTM** porte une mémoire
  covariance `d×d` mise à jour par produits externes `vₜᵀkₜ`, lue par requête, avec le
  dénominateur stabilisant `max(|nₜ·qₜ|,1)` reconstruit **exactement** via `|a|=relu(a)+
  relu(−a)` et `max(a,1)=relu(a−1)+1` (aucun nouvel op, garde fidèle). Oracle honnête :
  mLSTM ≡ récurrence de référence écrite à la main (dénominateur actif) ; **gradient check**
  par différences finies (sLSTM : 4 portes ; mLSTM : q,k,v,iₜ,fₜ, régime lisse) ;
  entraînement `NdXlstm` (MSE↓) + déterminisme bit-exact. Rejoint la famille de modèles de
  séquence (Mamba, S4, RWKV, RetNet, GLA, HGRN, DeltaNet).
- **OmniQuant — clipping de poids apprenable** (`quantization::omniquant_quantize`, Shao
  et al. 2024, roadmap #65) : le round-to-nearest quantifie chaque canal sur sa plage
  **complète** `[−max|w|, max|w|]` — avec des poids à queue lourde, la plupart des niveaux
  de code sont gaspillés sur de rares aberrants. OmniQuant apprend un **facteur de coupe**
  `γ∈(0,1]` par canal qui **rétrécit** la plage à `γ·max|w|`, échangeant un peu d'erreur de
  coupe sur les aberrants contre des pas bien plus fins sur le gros des poids — trouvé ici
  par une recherche déterministe sur une grille qui **inclut `γ=1`** (RTN pur). Oracle
  honnête : erreur de reconstruction **< RTN** sur poids à queue lourde (≥1 canal coupe
  réellement) + **jamais pire** que RTN (γ=1 est candidat) + déterminisme bit-exact.
  Rejoint la famille de quantification (GPTQ, AWQ, NF4, SqueezeLLM, SpQR, KVQuant, LLM.int8).
- **S4 (S4D) — espace d'états structuré diagonal** (`nn::nd_layers::s4_scan`/`NdS4`,
  Gu et al. 2022, roadmap #51) : SSM **linéaire invariant dans le temps** (contrairement
  au `selective_scan` de Mamba dont les matrices dépendent de l'entrée) — `A` diagonal,
  `B`/`C`/`Δ` sont des **paramètres fixes** ; discrétisation `Ā=exp(Δ⊙A)`, `B̄=Δ⊙B`,
  récurrence `h_t=Ā⊙h_{t−1}+B̄⊙x_t` (état `(d,n)`) déroulée sur la tape, lecture
  `y_t=Σ_n C⊙h_t`. Init **HiPPO** diagonale (S4D-Lin) `A[:,j]=−(j+1)`, `A<0` contractif.
  La couche `NdS4` ajoute projections d'entrée/sortie + skip gaté `D⊙x`. Oracle :
  **gradient check** (différences finies vs analytique sur x, a_log, B, C, log_dt) +
  entraînement (MSE↓ vers une cible) + déterminisme bit-exact. Couche de bibliothèque.
- **AI² / zonotopes — domaine abstrait pour la vérification** (`nn::ibp::Zonotope`/
  `IbpMlp::certify_zonotope`, Gehr et al. 2018, roadmap #29) : propagation par
  **zonotopes** (centre + générateurs, `{c+Σεᵢgᵢ : εᵢ∈[−1,1]}`) — les couches affines
  sont **exactes**, la ReLU est relaxée façon **DeepZ** (`y=λx+μ±μ`, `λ=u/(u−l)`,
  `μ=−λl/2`, un générateur frais par neurone instable). Les `εᵢ` partagés capturent les
  **corrélations** linéaires que les intervalles perdent. Oracle honnête : affine exacte
  (= forward intervalle) + **soundness** (des milliers de points échantillonnés dans la
  boîte d'entrée tombent dans la boîte zonotope d'un MLP ReLU 3 couches) + **plus serré
  qu'IBP sous corrélation** (réseau `relu(x)−relu(x)` ≡ 0 : zonotope `[−0,5;0,5]` vs IBP
  `[−1;1]`, les deux sains). Étend `nn::ibp` (IBP #1, CROWN #2) ; affiché dans la CLI
  `certify` à côté d'IBP et CROWN.
- **EAGLE — décodage spéculatif au niveau features** (`nn::nd_decoder::EagleHead`/
  `generate_eagle`, Li et al. 2024, roadmap #62) : là où Medusa prédit des *tokens*
  futurs, EAGLE brouillonne au niveau **feature** — une tête légère mappe
  `(feature_t, embed(token_{t+1})) → feature_{t+1}`, et la tête LM **gelée** transforme
  la feature prédite en token ; chaînée, elle donne un brouillon **autorégressif**
  vérifié par une passe (préfixe accepté + correction greedy). `NdDecoderLM` expose
  `token_embedding`/`head_logits`/`d_model` ; `EagleHead::train` ajuste la tête par MSE
  sur les features du modèle gelé. Oracle honnête : sortie **exactement = greedy** pour
  une tête **quelconque** (vérification) + déterminisme + tête **entraînée** ⇒ ≥1 bloc
  accepte >1 token (forwards < 2·n) en restant exact. Couche de bibliothèque.
- **Medusa — décodage à têtes multiples** (`nn::nd_decoder::MedusaHeads`/`generate_medusa`,
  Cai et al. 2024, roadmap #61) : accélère le décodage en attachant au modèle de base
  des **têtes supplémentaires** (tête `j` prédit le token à `+j+2` depuis l'état caché),
  qui produisent un **brouillon multi-token d'un seul forward** ; une passe de
  vérification accepte le plus long préfixe correspondant à l'argmax du modèle puis
  commet un token de correction/bonus. `NdDecoderLM` expose désormais
  `forward_hidden`/`forward_with_hidden` (état caché post-LayerNorm) ; `MedusaHeads::train`
  entraîne les têtes sur les états cachés du modèle **gelé**. Oracle honnête : sortie
  **exactement = greedy** pour des têtes **quelconques** (même aléatoires — la vérification
  garantit l'exactitude) + déterminisme + têtes **entraînées** ⇒ au moins un bloc accepte
  >1 token (forwards < 2·n) tout en restant exact. Couche de bibliothèque.
- **PagedAttention — KV-cache paginé** (`nn::paged_attention::PagedKvCache`, Kwon et al.
  / vLLM 2023, roadmap #63) : le cache clés/valeurs du décodage est découpé en **blocs**
  de taille fixe tirés d'un pool partagé, adressés indirectement par une **table de
  blocs** (comme la pagination mémoire) ⇒ quasi zéro fragmentation. `append` remplit les
  blocs à la demande, `gather_keys/values` reconstruit le cache contigu, et `attention`
  fait le produit scalaire softmax en indexant clés/valeurs **à travers la table**.
  Oracle honnête : avec des blocs **leurres** interleavés (layout physique non
  séquentiel), le gather est **bit-identique** aux vecteurs insérés et l'attention
  paginée est **bit-identique** à l'attention sur cache contigu (même ordre
  arithmétique) — la pagination est prouvée sans coût numérique ; + comptabilité des
  blocs (`⌈len/bloc⌉`) et cas vide + déterminisme. Couche de bibliothèque (nouveau module).
- **DoRA — adaptation low-rank décomposée poids** (`nn::dora::DoraLinear`, Liu et al.
  2024, roadmap #73) : PEFT qui décompose un poids gelé `W₀` en **magnitude** (vecteur
  par colonne `m`) × **direction** (normalisée), la direction étant pilotée par une
  mise à jour low-rank LoRA `BA` : `W' = m ⊙ (W₀+BA)/‖W₀+BA‖_col`. Seuls `m`, `A`, `B`
  s'entraînent. Backward de la normalisation par colonne en **forme close** (`u=V/‖V‖`,
  `∂L/∂V=(m/‖V‖)(gw−u·s)`, `∂L/∂m=s`). Oracle honnête : init `B=0, m=‖W₀‖_col` ⇒ poids
  effectif **= W₀ exactement** (l'adaptation part de la fonction pré-entraînée) +
  **gradient check** (différences finies centrales vs analytique, params génériques) +
  récupère une cible générée par DoRA (perte ÷100 par descente de gradient) +
  déterminisme bit-exact. Couche de bibliothèque (nouveau module).
- **GaLore — projection low-rank des gradients** (`nn::nd_optim::NdGalore`/
  `galore_subspace`, Zhao et al. 2024, roadmap #48) : optimiseur à **mémoire
  réduite** — pour un paramètre matriciel, le gradient `G` est projeté sur son
  propre sous-espace dominant rang-`r` `P` (top-`r` vecteurs singuliers gauches via
  `jacobi_eigenvectors`, rafraîchi tous les `update_gap` pas), Adam tourne sur le
  petit gradient projeté `PᵀG` puis l'update est remonté par `P`. Les états passent
  de `m×n` à `rank×max(m,n)` ; les vecteurs retombent sur Adam. Oracle honnête :
  `P` **orthonormal** (`PᵀP=I`) et projection **orthogonale optimale** (identité de
  Pythagore `‖G−PPᵀG‖²=‖G‖²−‖PᵀG‖²`, erreur décroissante en `r`, nulle au rang
  plein) + gradient **bas-rang reconstruit exactement** (sous-rang ⇒ résidu) +
  **convergence sur une cible bas-rang** avec état compressé `2×4` (≠ `4×4`) +
  sous-rang ne l'atteint pas + déterminisme bit-exact. Rejoint la famille
  d'optimiseurs ; CLI `lm --opt galore`.
- **YaRN — extension de contexte RoPE** (`nn::yarn`, Peng et al. 2023, roadmap #60) :
  étend le contexte utilisable d'un modèle RoPE d'un facteur `s` par interpolation
  **NTK-by-parts** — `yarn_frequencies` garde intactes les dimensions **haute
  fréquence** (`r_p>β` ⇒ ordre local préservé), interpole pleinement les **basses
  fréquences** (`r_p<α` ⇒ `θ_p→θ_p/s`), avec une rampe linéaire entre les deux
  (`θ'_p=θ_p·((1−γ)/s+γ)`). `rope_apply_freqs`/`rope_yarn` appliquent la rotation
  (convention emboîtée identique à la RoPE existante de `autodiff::nd`) ;
  `yarn_attention_scale` donne la température `0.1·ln(s)+1`. Oracle honnête :
  **propriété de position relative** `⟨rope(q,m),rope(k,n)⟩=g(m−n)` préservée malgré
  les fréquences modifiées + l'angle d'une dimension basse fréquence à la longueur
  **étendue** `s·L` revient **exactement** à sa valeur d'entraînement à `L` (alors
  que la RoPE simple explose) + bornes NTK-by-parts (haute fréquence inchangée, basse
  = `θ/s`, rampe monotone) + `scale=1` ≡ RoPE simple + déterminisme. Couche de
  bibliothèque (primitive positionnelle, pas de CLI).
- **Learn then Test (LtT)** (`nn::conformal::learn_then_test`/`hoeffding_pvalue`,
  Angelopoulos et al. 2021, roadmap #37) : contrôle **distribution-free** de
  **risques multiples arbitraires** (non emboîtés) par tests d'hypothèses. Chaque
  configuration `λ` d'une grille devient une **p-value de Hoeffding** pour
  `H₀: R(λ) > α` (`p = exp(−2n(α−R̂)₊²)`, super-uniforme sous le null), puis
  correction **familiale de Bonferroni** au niveau `δ` : on retient les `λ` avec
  `p ≤ δ/m`. Garantit que, avec proba `≥ 1−δ`, **toute** config retenue vérifie
  `R(λ) ≤ α` (FWER `≤ δ`) — **sans** hypothèse de monotonie (contrairement à RCPS
  #36). Oracle honnête : FWER vérifié **par simulation** (toutes les configs sur
  la frontière `R=α` ⇒ FWER mesuré `≤ δ`, vs sélection naïve qui échoue ~toujours)
  + puissance (les configs sûres sont retenues, les non-sûres rejetées) +
  déterminisme. Couche de bibliothèque.
- **Comptable RDP (Rényi DP)** (`dp::gaussian_rdp`/`rdp_to_dp`/`rdp_gaussian_epsilon`,
  Mironov 2017, roadmap #78) : comptabilité de budget de confidentialité par
  **Rényi-DP**, plus serrée et plus principielle que la composition `(ε,δ)` naïve.
  RDP du mécanisme gaussien `RDP(α)=α/(2σ²)` (additif en composition), conversion
  Mironov `ε=RDP(α)+ln(1/δ)/(α−1)` (le `α−1` est ce qui la rend serrée), optimisée
  sur une grille d'ordres α. Renforce le DP-SGD existant (#19). Oracle : RDP et
  conversion exactes (formes closes) + `ε` **bien en dessous** de la composition
  linéaire basique (qui paie une pénalité ~√étapes) + monotonie (plus d'étapes ⇒ ε
  plus grand ; plus de bruit ⇒ ε plus petit). Couche de bibliothèque.
- **Watermark pour LLM** (`nn::watermark`, Kirchenbauer et al. 2023, roadmap #79) :
  filigrane statistique rendant le texte généré **auditable sans accès au modèle**.
  Le token précédent seede une partition du vocabulaire en liste **verte** (fraction
  γ) / rouge ; `apply_green_bias` ajoute `δ` aux logits verts pour orienter la
  génération. Le détecteur, qui ne connaît que le seed et γ, recompte les tokens
  verts : un texte filigrané en contient bien plus que la fraction γ attendue par
  hasard, ce qu'un **test z** `(g−γn)/√(nγ(1−γ))` (`detect_z`) signale par une
  p-value minuscule, tandis que le texte naturel score `z≈0`. Tout est un hash
  déterministe de `(seed, prev, token)`. Oracle : fraction verte ≈ γ + biais
  appliqué aux seuls tokens verts + texte filigrané détecté (z≫8) vs naturel (z≈0)
  + un **mauvais seed ne détecte pas** (pas de fausse provenance) + déterminisme.
  Couche de bibliothèque.
- **DeepONet — apprentissage d'opérateurs** (`nn::deeponet::DeepONet`, Lu et al.
  2021, roadmap #76) : apprend un **opérateur** `G : u ↦ G(u)` (fonction →
  fonction) via une factorisation **branch × trunk** `G(u)(y) ≈ Σ_k b_k(u)·t_k(y)`
  — la branch encode la fonction d'entrée `u` (échantillonnée à des capteurs
  fixes), la trunk encode la position `y`. Variante **POD-DeepONet** (trunk cosinus
  **fixe** `cos(kπy)` + branch **linéaire**) ⇒ ajustement **convexe**, exact pour
  les opérateurs linéaires comme l'**antidérivée** `∫₀^y u`. Oracle : entraîné sur
  certaines fonctions, il approxime l'antidérivée sur des fonctions **non vues** à
  MSE test < 0,01 (≪ prédicteur constant) — la propriété d'apprentissage
  d'opérateurs — + déterminisme. Couche de bibliothèque.
- **Deep Ensembles** (`nn::ensemble::DeepEnsemble`, Lakshminarayanan, Pritzel &
  Blundell 2017, roadmap #40) : incertitude prédictive par **ensemble seedé**. N
  petits MLP ReLU (`1→hidden→1`) entraînés sur la tape N-D avec `NdAdam`, chacun
  seedé différemment ; `predict(x)` renvoie `(moyenne, écart-type)` — l'estimation
  ponctuelle et son **incertitude épistémique** (désaccord entre membres). Oracle :
  la MSE de la moyenne d'ensemble est ≤ la MSE moyenne des membres (Jensen) +
  l'écart-type est **bien plus grand hors-distribution** (loin de la plage
  d'entraînement) qu'en-distribution + déterminisme bit-exact. Couche de
  bibliothèque.
- **LLM.int8()** (`quantization::int8_mixed_matmul`, Dettmers et al. 2022, roadmap
  #71) : matmul mixte int8/fp32. Les activations des transformeurs ont quelques
  **colonnes de features outliers** de très grande magnitude ; les quantifier en
  int8 avec le reste gonfle l'échelle et écrase la résolution des features
  normales. LLM.int8() garde ces colonnes (et les lignes de W correspondantes) en
  **pleine précision** et quantifie le reste en **int8** :
  `X·W = X_normal·W_normal (int8) + X_outlier·W_outlier (fp32)`. Une colonne est
  outlier si un `|X[i,j]|` dépasse le seuil (défaut 6.0). Oracle : sur des
  activations à colonnes outliers, l'erreur vs fp est **< 0,5×** celle de l'int8
  simple ; sans outliers, se réduit à l'int8 pur ; déterminisme. Couche de
  bibliothèque.
- **RCPS — Risk-Controlling Prediction Sets** (`nn::conformal::hoeffding_ucb` +
  `rcps_select`, Bates et al. 2021, roadmap #36) : là où le conformal contrôle la
  *couverture*, RCPS contrôle un **risque borné quelconque** (perte dans [0,1] :
  taux de faux négatifs, non-couverture, …) avec une garantie **haute probabilité
  (PAC)**. Pour une famille de prédicteurs `C_λ` à risque non-croissant en λ, RCPS
  choisit le plus petit `λ̂` dont la **borne de concentration de Hoeffding** sur le
  risque est ≤ α (pour λ̂ et tout λ plus grand) ⇒ `R(λ̂) ≤ α` avec proba ≥ 1−δ.
  Oracle : la borne dépasse la moyenne du bon écart + sélection exacte (cas
  calculé) + sur données fraîches le risque empirique reste ≤ α (borne
  conservatrice). Couche de bibliothèque.
- **Prodigy** (`nn::nd_optim::NdProdigy` + `ProdigyConfig`, Mishchenko & Defazio
  2023, roadmap #46) : un Adam **sans learning-rate** (« parameter-free »). Il
  estime en ligne la distance `d ≈ ‖x₀ − x*‖` à la solution — via la corrélation
  globale `⟨g, x₀ − x⟩` accumulée — et l'utilise comme taux effectif, partant d'un
  `d₀ = 1e-6` minuscule qui croît jusqu'à l'échelle du problème. `d`, le numérateur
  `r` et la norme du dénominateur sont des scalaires **globaux** sur tous les
  paramètres. Oracle : `d` s'adapte à l'échelle de la distance (sans réglage de lr)
  + la perte quadratique chute fortement + déterminisme bit-exact. CLI :
  `scirust lm --opt prodigy` (8 langues).
- **KVQuant** (`quantization::kvquant_kv`, Hooper et al. 2024, roadmap #68) :
  quantification du **KV-cache** d'attention à la granularité qui épouse sa
  structure d'outliers — **clés per-canal** (les outliers des clés se concentrent
  par colonne de features) et **valeurs per-token** (par ligne). Bien plus fidèle
  qu'une échelle per-tensor unique, qu'une poignée de gros canaux de clés
  domineraient (écrasant la résolution de tous les autres). Oracle : sur des clés à
  outliers de canal, l'erreur de la sortie d'attention vs fp est **< 0,6×** celle
  de la quantif per-tensor ; le per-canal résout les petites colonnes (<0,1× erreur)
  là où le per-tensor échoue ; déterminisme. Couche de bibliothèque.
- **ALiBi — Attention with Linear Biases** (`nn::nd_layers::alibi_slopes` +
  `alibi_bias` + `NdMultiHeadAttention::with_alibi`, Press, Smith & Lewis 2022,
  roadmap #59) : remplace les positions apprises/rotatives par un **biais statique
  linéaire en distance** ajouté aux scores d'attention — pour la requête `i` et la
  clé `j ≤ i`, `−penteₕ·(i−j)`, avec des pentes par tête en suite géométrique
  `2^(−8h/H)`. Aucune position apprise ⇒ **extrapolation en longueur**. Branché dans
  `NdMultiHeadAttention` (builder `with_alibi`, inclut le masque causal). Oracle :
  pentes géométriques (ratio `2^(−8/H)`) + biais linéaire/causal/Toeplitz + poids
  softmax décroissant avec la distance (exactement `∝ exp(−pente·dist)`) + forward
  d'attention déterministe.
- **ACI — Adaptive Conformal Inference** (`nn::conformal::AdaptiveConformal`, Gibbs
  & Candès 2021, roadmap #38) : conformal **en ligne** robuste à la **dérive de
  distribution**. Le conformal statique perd silencieusement sa couverture sous
  changement de distribution ; ACI suit un niveau effectif `αₜ` et le corrige après
  chaque observation par rétroaction `αₜ₊₁ = αₜ + γ(α − errₜ)`, ce qui pilote le
  taux d'erreur long-terme vers `α` (couverture vers `1−α`) pour **tout** flux de
  scores. Avec une fenêtre glissante de scores récents, la couverture reste ≈ 1−α
  à travers les changements là où le conformal statique s'effondre. Oracle : règle
  de mise à jour de `αₜ` exacte (cas calculé) + couverture ≈ 1−α maintenue sous
  changement de variance (vs conformal statique qui chute) + déterminisme. Couche
  de bibliothèque. Complète CQR/APS/RAPS dans le pilier conformal.
- **KAN — Kolmogorov-Arnold Networks** (`nn::kan::KanLayer`, Liu et al. 2024 ;
  base RBF de FastKAN, Li 2024 ; roadmap #77) : activations **apprenables sur les
  arêtes** plutôt que sur les nœuds — `y_j = Σᵢ φᵢⱼ(xᵢ)` avec chaque `φ` une somme
  de RBF gaussiennes (grille fixe) + un terme de base `SiLU`. La sortie est
  **linéaire dans les coefficients**, donc l'ajustement est un problème de moindres
  carrés **convexe** résolu par descente de gradient déterministe. Oracle : une
  seule couche KAN ajuste la cible additive non-linéaire `sin(2x₀)+x₁²` à MSE<0,02
  — bien en dessous du meilleur modèle linéaire (qui ne peut représenter sin/carré)
  ; base RBF localisée ; déterminisme bit-exact. Couche de bibliothèque (variante
  RBF/FastKAN, pas les B-splines du papier original).
- **RWKV time-mixing (WKV)** (`nn::nd_layers::rwkv_wkv` + `NdRwkv`, Peng et al.
  2023, roadmap #53) : opérateur **WKV** — attention linéaire récurrente à
  **décroissance temporelle exponentielle par canal** `decay ∈ (0,1)` plus un
  **bonus** pour le token courant, normalisée (numérateur/dénominateur), déroulée
  en temps linéaire sur la tape. A nécessité un nouvel op autograd **`div`**
  (division élémentaire, gradient `∂a=g/b`, `∂b=−g·a/b²`, gradient-checké). La
  couche `NdRwkv` ajoute une **réception** `r=σ(W_r·x)` qui gate la sortie, avec
  decay/bonus par canal apprenables. Oracle : la récurrence sur tape **≡ la
  formule de somme pondérée explicite** + gradient check (k, v, decay, bonus) +
  entraînement (MSE↓) + déterminisme bit-exact. CLI : `scirust rwkv` (8 langues).
- **GloRo — robustesse certifiée par Lipschitz** (`nn::lipschitz`, Leino, Wang &
  Fredrikson 2021, roadmap #32) : `spectral_norm` (norme spectrale par power
  iteration déterministe), `spectral_normalize` (couche **1-Lipschitz** contrainte)
  et `GloroClassifier` (classifieur linéaire à **rayon de robustesse L2 prouvé**
  `marge/(√2·‖W‖₂)`, sans recherche ni échantillonnage ; le `√2` vient de la
  Lipschitz `≤ √2·L` de la marge `f_A−f_B`). Oracle : normes spectrales connues
  (diagonale, rectangulaire) ; norme ≈ 1 après normalisation ; rayon **sain** (la
  pire perturbation à ce rayon ne bascule pas la prédiction) **et conservateur**
  (≤ distance exacte à la frontière la plus proche) ; déterminisme. Couche de
  bibliothèque. Complète le pilier certifiable : IBP, CROWN, smoothing, GloRo.
- **Randomized Smoothing — robustesse L2 certifiée** (`nn::smoothing::SmoothedClassifier`
  + `clopper_pearson_lower` + `inv_normal_cdf`, Cohen, Rosenfeld & Kolter 2019,
  roadmap #27) : transforme tout classifieur en un classifieur **lissé** sous bruit
  gaussien `N(0,σ²I)`, avec un **rayon de robustesse L2 prouvé** `σ·Φ⁻¹(pₐ)`. La
  probabilité de la classe top `pₐ` est minorée par **Clopper-Pearson** (beta
  incomplète régularisée `betai`/`lgamma`, exacte) ; `Φ⁻¹` par l'approximation
  rationnelle d'Acklam. Oracle : pour un classifieur **demi-espace** le rayon
  certifié **égale la distance exacte à la frontière** (indépendant de σ) +
  soundness/abstention au bord + déterminisme + valeurs repères de `Φ⁻¹`/`betai`/
  Clopper-Pearson. CLI : `scirust certify` affiche désormais IBP/CROWN
  (déterministe) **et** smoothing (probabiliste).
- **SpQR — Sparse-Quantized Representation** (`quantization::SpqrOutliers`,
  Dettmers et al. 2023, roadmap #67) : l'erreur de quantification est à **queue
  lourde** — une petite fraction de poids « outliers » concentre l'essentiel de
  l'erreur. SpQR garde cette fraction (les plus grosses erreurs de quantif dense)
  en **pleine précision** (canal creux) et quantifie le reste densément, donc ~1 %
  d'outliers retire une grande part de l'erreur pour un faible surcoût mémoire.
  Oracle : sur poids gaussiens + outliers injectés, garder 1 % des poids divise
  l'erreur quadratique par > 3 ; reconstruction exacte des outliers ; déterminisme.
  Couche de bibliothèque (les scales groupés bi-niveaux du papier sont orthogonaux).
- **SqueezeLLM** (`quantization::SqueezeLlmCodebook` + `weighted_quant_error`, Kim
  et al. 2023, roadmap #66) : quantification **non-uniforme** des poids par
  **k-means pondéré par la sensibilité** (proxy de la diagonale de la Hessienne)
  — un codebook de `2^bits` centroïdes placés là où ils réduisent le plus la
  *perte*, et non là où les poids sont denses. Init déterministe (quantiles) +
  itérations de Lloyd pondérées. Oracle : erreur de quantification pondérée
  **strictement < round-to-nearest uniforme** (poids gaussiens, 3 bits, < 0,85×) +
  round-trip exact sur les valeurs du codebook + déterminisme. Couche de
  bibliothèque (la branche « sparse » outliers n'est pas modélisée).
- **APS / RAPS — ensembles de prédiction adaptatifs** (`nn::conformal::AdaptivePredictionSets`,
  Romano, Sesia & Candès 2020 ; Angelopoulos et al. 2021 ; roadmap #34/#35) :
  conformal **classification** par score cumulatif `s(x,c)` = masse de toutes les
  classes au moins aussi probables que `c`. Set `{c : s(x,c) ≤ q̂}` ⇒ couverture
  marginale sans distribution ≥ 1−α avec **taille adaptative** (entrée confiante →
  petit ensemble, ambiguë → grand). **RAPS** ajoute `λ·max(0, rang−k_reg)` au
  score (`calibrate_raps`) pour rogner les classes peu probables et produire des
  ensembles **plus petits** à couverture égale. Oracle : score cumulatif exact
  (cas main) + couverture sur données fraîches + adaptativité (facile vs ambigu) +
  RAPS < APS en taille moyenne + déterminisme. Couche de bibliothèque (comme
  `ConformalClassifier`).
- **CQR — Conformalized Quantile Regression** (`nn::conformal::ConformalQuantileRegressor`,
  Romano, Patterson & Candès 2019, roadmap #33) : conformalise un régresseur de
  **quantiles** pour produire des intervalles **adaptatifs** (hétéroscédastiques)
  à couverture garantie. Score signé `Eᵢ = max(q_lo(xᵢ)−yᵢ, yᵢ−q_hi(xᵢ))`,
  correction finie `Q` (quantile conformal des `Eᵢ`, réutilise `conformal_quantile`),
  intervalle `[q_lo(x)−Q, q_hi(x)+Q]` — largeur **variable selon x** là où le
  split-conformal symétrique est de largeur constante (`Q` peut être négatif et
  resserrer une bande trop large). Oracle : sémantique exacte du score (cas
  calculé à la main) + couverture marginale ≥ 1−α sur données fraîches +
  **adaptativité** (intervalles bien plus larges en région à fort bruit) +
  déterminisme. CLI : `scirust conformal` montre désormais split **et** CQR.
- **SAM — Sharpness-Aware Minimization** (`nn::nd_optim::NdSam` + `SamConfig`,
  Foret et al. 2021, roadmap #47) : optimiseur **à deux phases** qui minimise la
  perte du *pire cas* dans une boule de rayon ρ (biais vers les minima plats).
  `ascent` perturbe les poids vers `θ + ρ·g/‖g‖` (norme **globale** du gradient) ;
  `descent` restaure θ et fait un pas SGD avec le gradient **au point perturbé**.
  Deux gradients par pas ⇒ hors de la boucle `lm --opt` à gradient unique (couche
  de bibliothèque). Oracle : perturbation = `ρ·g/‖g‖` avec `‖ε‖ = ρ` + convergence
  sur quadratique (bande ∝ lr·ρ) + déterminisme.
- **Shampoo** (`nn::nd_optim::NdShampoo` + `ShampooConfig` + `inverse_pth_root`,
  Gupta/Koren/Singer 2018, roadmap #41) : préconditionneur **Kronecker** structuré
  — pour une matrice de poids, maintient les deux facteurs `L = E[GGᵀ]`,
  `R = E[GᵀG]` et avance par l'update préconditionné
  `W ← W − lr·L^(−1/4) G R^(−1/4)`. Les racines inverses des matrices viennent
  d'une décomposition de Jacobi (`inverse_pth_root`, réutilise
  `jacobi_eigenvectors`), cachées et rafraîchies tous les `precond_freq` pas.
  Paramètres non-matriciels : Adagrad diagonal. Oracle : `A^(−1/2)²·A ≈ I` +
  convergence sur quadratique matricielle + repli Adagrad + déterminisme. CLI :
  `scirust lm --opt shampoo` (11e valeur `--opt`).
- **Adafactor** (`nn::nd_optim::NdAdafactor` + `AdafactorConfig`, Shazeer & Stern
  2018, roadmap #42) : optimiseur à **moments du 2e ordre factorisés** — pour une
  matrice de poids, ne stocke que les sommes **ligne** et **colonne** du carré du
  gradient (`rows + cols` nombres au lieu de `rows·cols`) et reconstruit la rank-1
  `V[i,j] = R[i]·C[j]/ΣR` (mémoire sous-linéaire). Update `G/√V` **clippé en RMS** ;
  planning `β2ₜ = 1 − t^(−0.8)`. Paramètres non-matriciels : 2e moment complet
  (RMSProp). Oracle : reconstruction rang-1 **exacte** quand `G²` est rang-1 +
  convergence (bande) + chemin matriciel factorisé qui réduit `½‖W−T‖²` +
  déterminisme. CLI : `scirust lm --opt adafactor` (10e valeur `--opt`).
- **NF4** (`quantization::nf4_quantize`/`nf4_dequantize` + `NF4_LEVELS`, QLoRA,
  Dettmers et al. 2023, roadmap #74) : type 4-bit **NormalFloat** — 16 niveaux qui
  sont les **quantiles d'une normale** (échelle absmax par bloc). Optimal pour des
  poids gaussiens. Oracle : erreur de reconstruction **< int4 uniforme** sur des
  poids gaussiens (Box-Muller seedé) + round-trip exact sur les niveaux +
  déterminisme. Couche de bibliothèque.
- **BitNet b1.58** (`quantization::ternary_quantize` + `ternary_matmul`, Ma et al.
  2024, roadmap #69) : quantification **ternaire** des poids vers `{−1,0,+1}`
  (échelle absmean, ~1,58 bit/poids, ~20× plus compact) ; **matmul sans
  multiplication** (addition / soustraction / skip selon le signe). Oracle :
  `ternary_matmul` = la forme somme-de-signes **bit-exact** et = le produit
  déquantifié à la réassociation flottante près. CLI : `scirust bitnet` (en
  direct : max erreur 1,4e-6 vs déquant, 986/4096 poids nuls). Déterministe.
- **HGRN** (`nn::nd_layers::hgrn` + `NdHgrn`, Qin et al. 2023, roadmap #58) : RNN
  linéaire à intégration leaky par canal (`hₜ = fₜ⊙h_{t-1} + (1−fₜ)⊙cₜ`), porte
  d'oubli **bornée inférieurement** `f = lb + (1−lb)·σ(·)` (la borne `lb` fixe
  l'horizon mémoire minimal). Pas d'état matriciel ; déroulé sur la tape. Tests :
  match référence + gradient check (c,f) + entraînement + déterminisme. CLI :
  `scirust hgrn` (en direct : MSE 27.37 → 4.59).
- **GLA — Gated Linear Attention** (`nn::nd_layers::gated_linear_attention` +
  `NdGla`, Yang et al. 2024, roadmap #55) : attention linéaire **gatée** — porte
  d'oubli par canal **dépendante de l'entrée** `αₜ=σ(·)`
  (`S_t = diag(αₜ)·S_{t-1} + kₜᵀvₜ`, `o_t = q_t·S_t`), déroulée sur la tape.
  Tests : match d'une référence Vec + gradient check (q,k,v,α) + entraînement +
  déterminisme. CLI : `scirust gla` (en direct : MSE 27.16 → 0.0000).
- **RetNet** (`nn::nd_layers::retention` + `NdRetention`, Sun et al. 2023,
  roadmap #54) : couche de **rétention** — attention linéaire récurrente à
  décroissance `γ` (`S_t = γ·S_{t-1} + kₜᵀvₜ`, `o_t = q_t·S_t`), déroulée sur la
  tape. **Oracle de dualité** : la forme récurrente **égale** la forme parallèle
  `(QKᵀ⊙D)V` (`D_{nm}=γ^{n-m}`), testé ; + gradient check (q,k,v) + entraînement
  + déterminisme. CLI : `scirust retnet` (en direct : MSE 24.63 → 0.0002).
- **LAMB** (`nn::nd_optim::NdLamb`, You et al. 2020, roadmap #43) : Adam à
  **confiance par couche** — direction Adam `r` remise à l'échelle par
  `‖θ‖/‖r‖` par tenseur. CLI `lm --opt lamb`. Tests : convergence (bande ∝ lr,
  car la norme de pas ≈ lr·‖θ‖) + déterminisme.
- **Adan** (`nn::nd_optim::NdAdan`, Xie et al. 2022, roadmap #49) : momentum de
  **Nesterov adaptatif** — 3 EMA (gradient `m`, différences `v`, terme
  look-ahead au carré `n`) ; `θ ← (θ − η⊙(m+(1−β2)v))/(1+lr·wd)`. CLI
  `lm --opt adan`. Tests : convergence quadratique + déterminisme.
- **LoRA** (`nn::nd_layers::LoraLinear`, Hu et al. 2022, roadmap #72) : adaptation
  **low-rank** — poids de base `W` **gelé** + mise à jour `ΔW = (α/r)·A·B` ; seuls
  `A` (`in×r`) et `B` (`r×out`) sont entraînés (`r·(in+out)` paramètres au lieu de
  `in·out`). `B=0` à l'init ⇒ la couche **vaut exactement la base**. Couche de la
  tape N-D. Tests : init = base, **gradient check** sur `A` et `B`, `parameters()`
  n'expose que `A`,`B`.
- **Temperature scaling / calibration** (`nn::calibration`, Guo et al. 2017,
  roadmap #39) : `temperature_scale` (recherche golden-section sur la NLL) +
  `expected_calibration_error` + `nll`. Recalibration post-hoc des probabilités
  **sans changer l'accuracy** (l'argmax est invariant à `T>0`). Déterministe. CLI :
  `scirust calibrate` (en direct : ECE 0.29 → 0.004, −98,5 %, T=2,70). Tests : ECE
  baisse + accuracy inchangée + déterminisme.
- **Lookahead** (`nn::nd_optim::NdLookahead`, Zhang et al. 2019, roadmap #45) :
  optimiseur **wrapper** poids lents/rapides autour d'Adam — `k` pas rapides puis
  `φ ← φ + α(θ − φ) ; θ ← φ`. Déterministe. CLI : `scirust lm --opt lookahead`.
  Tests : convergence quadratique, déterminisme bit-à-bit. (1er du pool de
  candidats Tier 8-14.)
- **PINN** (`nn::pinn` : `Pinn1D`, `solve_harmonic`, Raissi et al. 2019,
  roadmap #17) : réseau **physics-informed** — la **physique est dans la loss**
  via un résidu de PDE aux points de collocation + conditions aux limites.
  Résout le problème aux limites `u'' = −u`, `u(0)=0`, `u(π/2)=1` (solution
  exacte `sin x`) ; la dérivée seconde `u''` est prise par **différences finies
  dans l'entrée** (les évaluations `u(x±h)` passent par les *mêmes* paramètres
  dans un seul graphe forward), donc le gradient par rapport aux paramètres reste
  exact (autodiff inverse) et déterministe. Vérifié contre la solution analytique
  (erreur max ≈ 0,004). CLI : `scirust pinn`.
- **Mamba** (`nn::nd_layers::selective_scan` + `NdMamba`, Gu & Dao 2023,
  roadmap #18) : **selective scan** S6 — état-espace à matrice `A` diagonale et
  paramètres `Δ, B, C` **dépendants de l'entrée** (sélectifs) ; discrétisation
  par maintien d'ordre zéro `Ā = exp(Δ·A)`, `B̄x = Δ·B·x` ; récurrence
  déterministe linéaire-temps `h_t = Ā_t ⊙ h_{t-1} + B̄x_t`, `y_t = h_t·C_t`,
  déroulée sur la tape N-D. Nouvel op autograd `NdVar::exp` (gradient-checké).
  Init S4D-réelle (`A[:,j] = −(j+1)`), saut `D⊙x`. Tests : `selective_scan` match
  une référence Vec, gradient check (x, Δ, A, B, C), couche entraîne (MSE↓) +
  déterminisme. CLI : `scirust mamba`.
- **DeltaNet** (`nn::nd_layers::delta_rule` + `NdDeltaNet`, Yang et al. 2024,
  roadmap #25) : couche d'**attention linéaire récurrente** à règle delta
  (`S_t = S_{t-1} + β_t(v_t − S_{t-1}k_t)k_tᵀ`, `o_t = S_t q_t`) — mémoire à poids
  rapides, temps linéaire, causale. La récurrence est **déroulée sur la tape N-D**
  (nouvel op autograd `NdVar::cat0` : concaténation axe 0 + backward par découpe,
  **gradient-checké**), donc les gradients sont exacts et vérifiés par différences
  finies (q, k, v, β). Tests : correspondance avec une référence Vec, gradient
  check, entraînement (MSE↓) + déterminisme bit-à-bit. CLI : `scirust deltanet`.
- **SOAP** (`nn::nd_optim::NdSoap` + `jacobi_eigenvectors`, Vyas et al. 2024,
  roadmap #24) : optimiseur qui exécute **Adam dans la base propre de Shampoo**.
  Pour chaque matrice de poids : facteurs `L = E[GGᵀ]`, `R = E[GᵀG]` (moyenne
  mobile) ; rotation du gradient dans leur base propre (`Ĝ = Q_Lᵀ G Q_R`), Adam
  dans cette base, puis rotation inverse de la mise à jour. Base propre par
  **eigensolveur de Jacobi cyclique** déterministe (`jacobi_eigenvectors`),
  rafraîchie tous les `precond_freq` pas (moments tournés dans la nouvelle base).
  Repli Adam pour les paramètres non matriciels. Déterministe. CLI :
  `scirust lm --opt soap`. Tests : Jacobi diagonalise (orthogonalité +
  reconstruction), convergence sur quadratique matricielle, déterminisme bit-à-bit.
- **AWQ** (`quantization::awq_quantize` + `awq_act_scale` + `AwqResult`, Lin et al.
  2023, roadmap #15) : quantification int8 **consciente des activations** par
  recherche d'échelle. Importance par canal d'entrée `a_j = moyenne|x_:,j|` ;
  facteurs `s_j = a_j^alpha` (normalisés à moyenne géométrique unité) appliqués
  aux poids avant la quantification int8 per-canal, l'équivalence étant préservée
  côté activations ; `alpha` choisi par **grille** sur `[0,1]` (`alpha=0` =
  round-to-nearest) en minimisant l'erreur de sortie pondérée par la calibration.
  CLI : `scirust awq [--seed N] [--samples S] [--grid G]`. Tests : protège les
  canaux saillants → erreur < round-to-nearest (`alpha>0` choisi) + déterminisme
  bit-à-bit. **Complète le volet quantification #15** (SmoothQuant + GPTQ + AWQ).
- **GPTQ** (`quantization::quantize_gptq` + `gptq_hessian`, Frantar et al. 2022,
  roadmap #15) : quantification int8 des poids par **feedback d'erreur d'ordre 2**.
  Hessienne proxy `H = XᵀX` sur des activations de calibration ; inverse par
  Cholesky (en f64, déterministe) ; pour chaque canal de sortie, quantification
  séquentielle des poids d'entrée avec propagation de l'erreur (OBQ/GPTQ, ordre
  naturel) et complément de Schur. Scale symétrique par canal de sortie. CLI :
  `scirust gptq [--seed N] [--samples S] [--damp D]`. Tests : **erreur de
  reconstruction pondérée par la calibration < round-to-nearest** (≈ −85 % sur
  données corrélées) + soundness (jamais pire) + déterminisme bit-à-bit. Complète
  le volet quantification (#15) avec SmoothQuant et l'int8 per-canal.
- **CROWN** (`nn::ibp::crown_bounds`, Zhang et al. 2018, roadmap #2) : bornes de
  sortie d'un MLP ReLU à 1 couche cachée par **relaxation linéaire** +
  back-substitution sur une boîte L∞. Relaxation par neurone : exacte pour les
  neurones stables, chorde supérieure / pente inférieure adaptative pour les
  instables. **Plus serrée qu'IBP** (prouvé par test). CLI : `scirust certify`
  affiche désormais IBP **et** CROWN côte à côte (CROWN certifie la robustesse
  là où IBP échoue). Tests : soundness (échantillonnage de la boîte) + largeur
  CROWN ≤ largeur IBP par sortie.
- **AdEMAMix** (`nn::nd_optim::NdAdEMAMix`, Pagliardini et al. 2024, roadmap #23) :
  Adam à **deux EMA** du gradient (rapide β1 + lente β3 à longue mémoire, mélangées
  par α) ; déterministe. CLI : `scirust lm --opt ademamix`. Tests : convergence
  quadratique (bande), déterminisme bit-à-bit.

### Nettoyé
- Suppression de `scirust-core/src/nn/.legacy/` (**2363 lignes** de code mort) :
  répertoire non câblé dans l'arbre de modules (dotfile, zéro référence),
  superposé par les implémentations réelles `nn::conv2d`/`batch_norm`/`layer_norm`/
  `pool`/`loss`/`transformer`. Conforme au fondamental « code sous src/ câblé et
  testé, sinon archivé ».

### Ajouté — campagne « faire grandir scirust » (suite)
- **Schedule-Free** (`nn::nd_optim::NdScheduleFree`, Defazio et al. 2024, roadmap
  #22) : optimiseur **sans planning de learning-rate** — séquence de base `z`
  (descente), moyenne de Polyak `x` (**point d'évaluation**), gradient pris en
  `y = (1−β)z + βx`. Déterministe. CLI : `scirust lm --opt schedule-free`
  (le point d'éval `x` est chargé avant la prédiction). Tests : convergence sur
  quadratique, déterminisme bit-à-bit.
- **Conformal prediction** (`nn::conformal`, Angelopoulos & Bates 2021, roadmap
  #21) : `conformal_quantile`, `ConformalRegressor`, `ConformalClassifier` —
  ensembles/intervalles de prédiction à **couverture garantie sans hypothèse de
  distribution** (`≥ 1 − α`). Tests : la couverture empirique atteint la cible
  sur des données fraîches (régression *et* classification). CLI : `scirust
  conformal [--seed N] [--alpha A]` (couverture mesurée en direct, ex. 90,8 %
  pour une cible de 90 %). CLI : 41 → 42 commandes.
- **Lot 3 recherche → fonctions** (testées, 8 gates verts ; **14 des 20** items
  de [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)) :
  - **Muon** (`nn::nd_optim`, Jordan et al. 2024) : optimiseur matriciel —
    momentum puis **orthogonalisation Newton–Schulz** (quintique, sans SVD) de
    la mise à jour des matrices 2-D ; `newton_schulz_orthogonalize` exposé.
    Déterministe. Tests : orthogonalité (déviation ‖A·Aᵀ−I‖ s'effondre), perte
    matricielle, déterminisme.
  - **Wanda** (`pruning::prune_wanda`, Sun et al. 2023) : élagage one-shot par
    `|W|·‖X‖` (poids × norme d'activation), par ligne de sortie — diffère de
    l'élagage par magnitude sur les canaux à activations aberrantes.
  - **SmoothQuant** (`quantization::smoothquant_scales`/`apply_smoothquant`,
    Xiao et al. 2022) : lissage par canal d'entrée qui migre les valeurs
    aberrantes d'activation vers les poids ; **préserve `X·W`**.
- **Lot 2 recherche → fonctions** (3 features de plus, testées, 8 gates verts ;
  **11 des 20** items de [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)) :
  - **RoPE** (`autodiff::nd`, Su et al. 2021) : op `rope` (rotation par paires,
    backward = rotation inverse) ; gradient-checkée, conservation de norme et
    **propriété de position relative** testées ; branchée via
    `NdMultiHeadAttention::with_rope`.
  - **GQA / MQA** (`nn::nd_layers`, Ainslie et al. 2023) :
    `NdMultiHeadAttention::new_gqa(num_kv_heads, …)` — têtes K/V partagées via le
    broadcast `bmm` (aucune nouvelle op) ; gradient-checkée (GQA et MQA).
  - **Neural ODE** (`nn::neural_ode`, Chen et al. 2018) : `rk4_integrate` +
    `NeuralOde` — backprop **à travers** le solveur RK4 sur la tape N-D (fusion
    solveurs + autograd). RK4 validé (`dy/dt=y → e`), gradient-check à travers
    le solveur, et la dynamique **apprend** (Adam).
- **Feuille de route recherche → fonctions** ([`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)) :
  20 papers réels traduits en fonctions concrètes, avec statut et effort. Premier
  lot **livré cette session** (testé, 8 gates verts) :
  - **IBP — bornes de sortie certifiées** (`nn::ibp`, Gowal et al. 2018) :
    propagation d'intervalles dans un MLP ReLU → boîte de sortie **prouvée** ;
    `certified_robust` transforme la borne en garantie de classe. Soundness
    testée par échantillonnage (4000 points ∈ boîte certifiée). *Le* pilier « IA
    certifiable » rendu concret.
  - **Réductions reproductibles** (`reproducible`, Demmel & Nguyen) :
    `reproducible_sum`/`_mean`/`_dot` **bit-identiques quel que soit l'ordre /
    le nombre de threads** (tri canonique + expansion exacte de Shewchuk) ;
    survit à l'annulation catastrophique.
  - **Couches LLaMA N-D** (`nn::nd_layers`) : `NdRmsNorm`, `NdSwiGLU` (+ ops
    `rmsnorm`/`sigmoid` gradient-checkées) et `NdLlamaBlock` (Pre-RMSNorm +
    attention causale + SwiGLU) — entraînables, Adam-ready.
  - **Décodage spéculatif exact** (`nn::nd_decoder`, Leviathan/Chen 2023) :
    `generate_speculative` produit **exactement** la sortie greedy de la cible
    pour n'importe quel brouillon, avec moins de forwards ; + `generate_greedy`.
  - **Optimiseurs** (`nn::nd_optim`) : **AdamW** (weight-decay découplé) et
    **Lion** (sign-momentum, déterministe).
- **Commande CLI `lm`** : entraîne un petit LM décodeur causal (tape N-D + Adam)
  sur une séquence de tokens et rapporte la courbe de perte + le rappel exact —
  `scirust lm ["t0,t1,.."] [--seed N] [--steps S] [--lr R]`. Déterministe par
  graine ; expose toute la pile N-D (embeddings, attention causale, gather,
  cross-entropy, Adam) en une commande. CLI : 39 → 40 commandes.
- **Optimiseur Adam N-D, réutilisable et déterministe** (`nn::nd_optim`) :
  `NdAdam` (Kingma & Ba) sur un ensemble ordonné de paramètres. Chaque couche
  expose `parameters() -> Vec<NdParam>` (vue `&mut` des valeurs + index du
  gradient issu de `backward`) ; la composition remonte l'arbre
  (`NdLinear`/`NdEmbedding`/`NdLayerNorm` → attention → bloc → `NdDecoderLM`),
  donc **un seul `opt.step()` met à jour tout le modèle**. Arithmétique f32 en
  ordre fixe ⇒ **bit-à-bit déterministe**. Tests : convergence sur quadratique
  (oracle), déterminisme bit-à-bit, et **le LM décodeur entraîné par Adam via
  `parameters()`** (< 10 % de perte en 150 pas vs 300 en SGD, prédictions
  exactes).
- **Modèle de langage décodeur causal bout-en-bout** (`nn::nd_decoder`) :
  `NdDecoderLM` de style GPT entièrement sur la tape N-D — embedding de tokens
  + embedding positionnel appris → N blocs transformer Pre-LN **causals** →
  LayerNorm final → tête linéaire vers les logits de vocabulaire, entraîné par
  cross-entropy au token suivant. Test phare : **le LM sur-apprend une séquence
  et la reprédit exactement** à chaque position (preuve bout-en-bout que toute
  la pile apprend) ; forward déterministe par graine. `NdEmbedding` (table
  adossée à `gather`) ajoutée comme couche réutilisable.
- **Ops N-D `gather` + `cross_entropy`** (`autodiff::nd`) : `gather(indices)`
  (lookup d'embedding `(vocab, dim) → (n, dim)`, backward scatter-add — les
  indices répétés s'accumulent, les lignes jamais vues gardent un gradient nul)
  et `cross_entropy(targets)` (softmax + NLL moyen **fusionné**, log-sum-exp
  stable, backward `(softmax − onehot)/n`). Gradient-checkées ; sanity
  `logits uniformes → ln(vocab)`.
- **Attention causale N-D** (`NdMultiHeadAttention { causal }`, propagée à
  `NdTransformerBlock`) : masque triangulaire additif (`-1e9` au-dessus de la
  diagonale) avant le softmax — aucune nouvelle op d'autograd. Test de
  **causalité** : perturber le dernier token d'entrée laisse **chaque** sortie
  antérieure bit-à-bit inchangée, tandis que la sortie perturbée bouge.
- **Bloc transformer N-D complet et entraînable** (`nn::nd_layers`) :
  `NdLinear`, `NdMultiHeadAttention`, `NdLayerNorm` (affine γ/β) et
  `NdTransformerBlock` (Pre-LN : `x + Attn(LN(x))`, `x₁ + FFN(LN(x₁))`) sur la
  tape N-D, tous **entraînables** (`sgd_step`). Tests : gradient check
  entrée/couche d'attention/LayerNorm, **un MLP N-D qui apprend** ET **un bloc
  transformer N-D complet qui apprend** (perte < 70 % de l'initiale). Ops
  N-D ajoutées : `bmm`, `softmax`, `transpose_last2`, `reshape`, `permute`,
  `layernorm` — toutes gradient-checkées.
- **`MiniLLM::generate_sampled(&str)`** : génération publique à partir d'une
  chaîne, sampling seedé sur le KV-cache, déterministe ; greedy reproduit
  `generate`.
- **Attention N-D gradient-checkée** : `autodiff::nd` exprime une **attention
  multi-tête complète** `softmax(Q·Kᵀ/√d)·V` sur `(têtes, seq, d)` (ops
  `bmm`/`transpose_last2`/`softmax`/`mul`/`add`/`sub`/`relu`/`sum`), validée
  par gradient check. La tape N-D devient le sur-ensemble capable ; la 2D
  reste le défaut par choix d'architecture (coexistence, cf. GROWTH_PLAN).
- **Sampling seedé** (`nn::sampling`) : température / top-k / top-p pilotés par
  `PcgEngine` seedé → déterministe. `MiniLLM::generate_ids_cached_sampled`
  (génération O(n) à KV-cache avec sampling). Greedy reproduit le chemin argmax.
- **BPE byte-level** (`ByteBpeTokenizer`, style GPT-2) : vocab de base = 256
  octets ⇒ **aucun OOV**, round-trip **lossless** sur tout UTF-8 (accents,
  emoji, scripts inconnus). Déterministe. Exposé en CLI via `bpe --bytes`.
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
