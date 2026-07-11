# Audit des algorithmes — SciRust

> Date : 2026-07-11 · Branche : `claude/scirust-algorithm-audit-a619qo`
> Périmètre : les ~120 crates algorithmiques du monorepo (algèbre linéaire,
> EDO/optimisation/quadrature, noyaux tenseur & SIMD, autodiff/NN/optimiseurs,
> statistiques & fonctions spéciales, signal & audio, séries temporelles &
> finance, ML classique/clustering/RL, évolutionnaire & symbolique,
> raisonnement/graphes/NLP, GPU & compilation de tenseurs, verticaux
> industriels certifiables), plus un méta-audit de l'infrastructure de test et
> une analyse d'écart de couverture face à SciPy/LAPACK/GSL/statsmodels/scikit-learn.
> **Méthode** : 14 auditeurs de domaine indépendants ont lu le code source
> (pas seulement les signatures) et produit 157 constats ; chaque constat de
> sévérité P0/P1 (46 au total) a ensuite été soumis à **deux vérificateurs
> adversariaux indépendants** chargés de le réfuter à partir du code réel —
>44 confirmés, 1 contesté (nuancé ci-dessous), 1 rejeté (nuancé ci-dessous).

---

## 1. Synthèse exécutive

**Objectif du propriétaire du projet : faire de SciRust la référence
mondiale des algorithmes scientifiques — implémentation, amélioration,
tests.** Ce document mesure l'écart entre cette ambition et l'état réel du
code, sans complaisance.

**Verdict global : un socle d'ingénierie sérieux et souvent honnête, mais qui
n'est pas encore, algorithmiquement, une référence mondiale.**

Ce qui va bien, et va au-delà de la moyenne d'un projet de cette taille :
- Les implémentations sont très majoritairement **fidèles à leurs sources
  publiées** (chaque module cite ses références — Golub & Van Loan, Hairer,
  Hansen, Mihalcea & Tarau…) et **déterministes par construction**, un choix
  d'architecture rare et cohérent avec l'ambition affichée.
- Le meilleur du dépôt (`scirust-sparse`, `scirust-gp`, `scirust-stiff`,
  `scirust-tolerance`, la voie reproductibilité de `scirust-core`) est
  **validé contre des références externes** (oracles denses indépendants,
  tables NIST/AIAG, littérature PdM/fiabilité) — c'est la pratique qui
  distingue une bibliothèque numérique sérieuse, et elle existe déjà, mais
  de façon inégale.
- L'infrastructure CI (Miri, cross-arch qemu, oracles GPU, campagne de
  certification d'arrondi sur 30 milliards d'entrées f32) est **au niveau de
  l'état de l'art** pour la vérification de déterminisme.

Ce qui empêche aujourd'hui le statut de référence :
- **5 défauts P0** : des résultats mathématiquement faux dans des cas
  d'usage courants, pas des cas exotiques (gamma incomplète fausse pour tout
  test du χ² à grand nombre de degrés de liberté ; le PPO et l'actor-critic
  du RL n'apprennent pas ou apprennent à l'envers ; la différentiation duale
  échoue sur toute puissance d'un nombre négatif ; un GEMM GPU fusionné
  renvoie un résultat faux et exposé publiquement sans aucun test).
- **41 défauts P1** confirmés : instabilités numériques classiques (seuils
  absolus non adimensionnés, annulation catastrophique, formes de Kalman non
  Joseph), bugs de robustesse (CG/BiCGSTAB qui échouent sur `b=0`), et des
  pans entiers d'algorithmes absents malgré une doc ou une API qui les
  suggère (filtrage IIR/FIR, ARIMA, valeurs propres non symétriques).
- **Un écart de couverture massif** face à SciPy/LAPACK/GSL/statsmodels/
  scikit-learn : chaque module couvre grossièrement 10 à 40 % de son
  équivalent de référence. C'est, à terme, le chantier le plus lourd.
- **Une méthodologie de test qui plafonne à l'auto-cohérence** : sur 157
  constats, l'écrasante majorité des lacunes de test ont la même signature —
  très peu de valeurs de référence externes (SciPy/R/tables publiées),
  aucun test de propriétés (proptest/quickcheck absent des ~120 crates),
  un seul target de fuzzing. C'est précisément le trou par lequel les 5 P0
  sont passés inaperçus.

### Tableau des scores par domaine

| Domaine | Score | Ce qui bloque la référence |
|---|:---:|---|
| Algèbre linéaire dense et creuse | B− | Pas d'eigen non symétrique, pas de rcond, seuils absolus, CG/BiCGSTAB cassés sur `b=0` |
| EDO, optimisation, quadrature, racines | B− | Compteur de rejets DOPRI5 faux, NaN fatal, pas de BDF/L-BFGS/Levenberg-Marquardt |
| Noyaux tenseur/SIMD, reproductibilité | B+ | Double arrondi sous-normal viole le contrat « correctement arrondi » ; GEMM très loin d'un BLAS |
| Autodiff, NN, optimiseurs | B | Comptable DP non *sound*, Lottery Ticket qui stagne, GQA factice |
| Statistiques, spéciales, multivarié, estimation | B− | Gamma incomplète fausse (P0), quantile normal détruit en queue, PCA fausse à petite échelle |
| Signal et audio | C+ | Aucun design de filtre IIR/FIR, DFT O(N²) dans tout scirust-audio, FFT puissances de 2 seulement |
| Séries temporelles, saisonnalité, finance | C+ | Backtest circulaire, seuil de cointegration erroné, pas d'ARIMA |
| ML classique, clustering, AutoML, GP, RL | C+ | PPO et actor-critic cassés (P0), GMM/EM numériquement instable, CART non pondéré |
| Évolutionnaire, symbolique, synthèse | C+ | 3 résultats symboliques mathématiquement faux (P0/P1), CMA-ES qui n'en est pas un |
| Raisonnement, SAT/CSP, graphes, NLP | C+ | TextRank et UMass faux, ni CDCL/AC-3/Dijkstra/Tarjan/HNSW |
| GPU, tenseurs, einsum, TN | C+ | GEMM fusionné faux et non testé (P0), réduction Max/Norm cassée |
| Verticaux industriels certifiables | B | Page-Hinkley unilatéral, covariance non-Joseph, RMS ISO 10816 fabriqué |
| Infrastructure de test (méta-audit) | B+ | Zéro test de propriétés, un seul fuzz target, pas de suivi de perf |
| Écart de couverture vs état de l'art | C+ | 10-40 % de couverture par module face à SciPy/LAPACK |

---

## 2. Les 5 défauts P0 — à corriger en premier

Tous confirmés par deux vérificateurs indépendants relisant le code source.

1. **`scirust-special/src/lib.rs:328`** — `regularized_gamma_p` est silencieusement
   fausse pour `a` grand : `regularized_gamma_p(1e4, 1e4) = 0.49994` contre la
   vraie valeur `0.50133` (erreur relevée par exécution). Cause : la série
   converge en O(√a) itérations près de `x≈a` mais `MAX_ITERS=300` est fixe.
   Impact direct : tout test du χ² à grand nombre de degrés de liberté est
   faux. Référence : Temme, *The asymptotic expansion of the incomplete
   gamma functions* (1979/1987) — c'est la bascule que fait `cephes igam.c`
   dans SciPy.

2. **`scirust-learning/src/rl/ppo.rs:58`** — le terme clippé du surrogate PPO
   est construit hors du graphe de différentiation, et le `min` clippé/non
   clippé utilise une comparaison stricte qui, dans la zone nominale
   `ratio ∈ [1-ε, 1+ε]`, sélectionne systématiquement la constante : **le
   gradient de la politique est nul à la première passe pour tous les
   échantillons**. Référence : Schulman et al., *PPO*, 2017, eq. 7.

3. **`scirust-rl-algo/src/lib.rs:1309`** — `ActorCriticAgent::update` calcule
   `grad_scale = td_error * log_prob` (`log_prob ≤ 0` toujours) : pour une
   action meilleure qu'attendu, la mise à jour **diminue** sa probabilité —
   signe inversé. Le correctif exact figure déjà, en commentaire, dans
   `ReinforceAgent` du même fichier (ligne 1148) mais n'a pas été appliqué ici.

4. **`scirust-symbolic/src/lib.rs:1002`** — `Dual::powf` utilise la formule
   log-dérivée `v·(other.tangent·ln(self.primal) + …)`, valide seulement pour
   base positive. `Dual::var(-3.0).powf(Dual::primal(2.0))` renvoie une
   tangente `NaN` au lieu de `-6` — toute dérivée d'une puissance entière
   d'une variable négative est fausse, un cas très courant en autodiff.

5. **`scirust-gpu/src/kernels.rs:82`** — l'indexation en mémoire partagée de
   `TILED_GEMM_WGSL` et `FUSED_GEMM_WGSL` transpose par erreur les rôles
   ligne/indice-de-sommation. Simulation exacte de la sémantique WGSL :
   erreur maximale de 15 à 33 sur des GEMM 16×16×16 et 32×32×32 aléatoires,
   pour **toutes** les tailles. `FUSED_GEMM_WGSL` est dispatché par l'API
   publique `FusedLayer::execute`, et les deux seuls tests du chemin de
   fusion se **court-circuitent volontairement** (`eprintln!("skipped")` +
   `return`) — c'est exactement pour cela que le bug a survécu.

---

## 3. Défauts P1 confirmés — par thème

### 3.1 Robustesse numérique et seuils mal dimensionnés (le motif le plus répété)

Ce motif revient dans **8 domaines différents** — c'est le problème
systémique n°1 du dépôt :

- `scirust-solvers/src/linalg/iterative.rs:104` / `bicgstab.rs:102` — CG et
  BiCGSTAB renvoient `Err(Singular)` sur `b=0` ou `x0` déjà solution (pas de
  test de convergence initial), et le seuil de breakdown `1e-13` est
  appliqué à une quantité **quadratique** : tout système avec `‖b‖ ≲ 3e-7`
  échoue à tort. GMRES et le CG creux, dans le même dépôt, font ce test
  correctement — c'est un oubli local, pas un choix.
- `scirust-sparse/src/lib.rs:53`, `lu.rs:13`, `cholesky.rs:10`, `qr.rs:221` —
  seuils de pivot **absolus** (`1e-12`, `1e-14`) : une matrice à l'échelle
  physique naturelle (Farads, micro-unités) est déclarée singulière alors
  qu'elle est parfaitement inversible.
- `scirust-multivariate/src/lib.rs:344` — même défaut dans `jacobi_eigen` :
  la PCA de données corrélées à l'échelle `1e-7` renvoie des vecteurs propres
  faux (`[1,0]` au lieu de `[0.707, 0.707]`) car la covariance `~2e-14`
  passe sous le seuil de convergence absolu.
- `scirust-estimation/src/linalg.rs:165` — même pivot absolu dans l'inverse
  utilisée par les filtres de Kalman.
- `scirust-nav/src/fusion.rs:115`, `scirust-estimation/src/kalman.rs:84` —
  mise à jour de covariance par la forme courte `P←(I−KH)P` au lieu de la
  **forme de Joseph**, qui garantit symétrie et positivité en précision
  finie (Bucy & Joseph 1968) — un défaut inacceptable dans des crates qui se
  revendiquent *certifiables*.
- `scirust-reasoning/src/lib.rs:77`, `scirust-symbolic` — formule quadratique
  classique `(-b±√Δ)/(2a)` sans la forme stable de Numerical Recipes §5.6 :
  annulation catastrophique quand `b² ≫ 4ac`.

**Recommandation transversale** : un audit systématique de tous les seuils
absolus du dépôt (`grep` sur `1e-1[0-9]` dans les crates de calcul numérique)
et leur remplacement par des critères relatifs à la norme/l'échelle des
données d'entrée, à la LAPACK.

### 3.2 Formules mathématiquement fausses (hors P0)

- **TextRank** (`scirust-nlp-advanced/src/keyword.rs:180`) normalise par le
  degré du nœud récepteur au lieu de l'émetteur — contraire à Mihalcea &
  Tarau 2004.
- **Cohérence UMass** (`topic.rs:334`) calcule un rapport de logarithmes au
  lieu du logarithme d'un rapport — signe et magnitude faux face à Mimno et
  al. 2011.
- **`prove_equal`** (`scirust-symbolic/src/lib.rs:661`) déclare `ln(x)` non
  équivalent à lui-même dès que le premier point d'échantillonnage tombe
  hors du domaine de `ln`.
- **`solve_quadratic`** (deux occurrences indépendantes,
  `scirust-symbolic/src/lib.rs:535` et `scirust-reasoning/src/lib.rs:77`) —
  l'une extrait des coefficients par échantillonnage sans vérifier le degré
  réel (`x³-2` renvoie une « racine » `2.0`), l'autre souffre de
  l'annulation catastrophique classique.
- **DP-SGD** (`scirust-core/src/dp.rs:136`) — le comptable de moments
  sous-estime le log-moment gaussien sous-échantillonné d'un facteur
  `~2(λ+1)` par rapport au Lemme 3 d'Abadi et al. 2016 : **ε est sous-déclaré,
  la garantie de confidentialité affichée est fausse** (non *sound*), et le
  test verrouille la valeur erronée au lieu de la valeur correcte.
- **Lottery Ticket** (`scirust-core/src/pruning.rs:208`) — `prune_magnitude`
  retrie systématiquement tous les poids (y compris les zéros déjà élagués),
  donc la sparsité plafonne à `p` au lieu de converger vers `1-(1-p)^k`.
- **Le RL, plus largement** : DQN sans jamais synchroniser son réseau cible
  (`scirust-learning/src/rl/deep.rs:114` — `grep target_model` ne remonte
  qu'un seul fichier, aucune méthode `update_target`), GMM/EM détruit par un
  aller-retour `exp`/`ln` (`scirust-unsupervised/src/lib.rs:786`), CART non
  pondéré par les effectifs (`scirust-automl/src/lib.rs:1225`), pivots
  Cholesky négatifs masqués en `NaN.max(1e-12)` (`lib.rs:913`).

### 3.3 Algorithmes absents malgré une doc/API qui les suggère

- **Aucun design de filtre classique** dans tout `scirust-signal`/`scirust-audio`
  (pas de Butterworth/Chebyshev/elliptique, pas de bilinéaire, pas de
  biquad/SOS, pas de `filtfilt`) — *nuance* : le module `denoise` contient
  déjà des filtres fréquentiels réels (`fft_lowpass`, `notch_filter`), donc
  le constat initial « aucune fonction de filtrage » était trop absolu ; la
  lacune réelle et confirmée est l'absence de **synthèse de filtre à pôles/zéros**
  (IIR/FIR classique), qui reste le trou n°1 du DSP.
- `scirust-audio` calcule **toutes** ses features spectrales (MFCC, centroïde,
  bande passante, rolloff, flatness, entropie) via une **DFT naïve O(N²)**
  alors que `scirust-signal`, une dépendance directe, expose une FFT
  O(N log N) jamais appelée — pour 10 s à 44,1 kHz, ~2×10¹¹ opérations.
- **CmaEs n'est pas CMA-ES** (`scirust-evo/src/lib.rs:143`) : pas de matrice
  de covariance, pas de chemins d'évolution, `σ` constant — une ES isotrope
  sous un nom trompeur, incapable de converger sur Rosenbrock.
- **Cointegration/backtest financiers** : seuil ADF codé en dur à `-2.5`
  (valeur critique MacKinnon réelle ≈ `-3.34`) sans augmentation en retards
  (`scirust-trader/src/pairs.rs:196`) ; le « backtest » du moteur de risque
  utilise la prédiction du modèle comme PnL réalisé — circulaire par
  construction (`scirust-trader/src/risk.rs:245`).
- **Réduction GPU incomplète** (`scirust-gpu/src/kernels.rs:131`) —
  `DETERMINISTIC_REDUCE_WGSL` n'implémente ni `Max` ni `Norm` (renvoie
  `sqrt(somme)` pour les deux), alors que l'énum public les déclare.
- **`einsum` panique sur dimension nulle**, **`fixed_point_gemm_q16` déborde
  silencieusement en i32** — deux défauts de robustesse confirmés côté GPU/tenseur.
- **Page-Hinkley unilatéral** (`scirust-pdm/src/change_detection.rs:167`) —
  ne détecte que les dérives à la hausse ; une dégradation typique en
  maintenance prédictive (chute d'un indice de santé) n'est **jamais**
  détectée, malgré une API qui expose `direction: -1`.
- **Sévérité ISO 10816 fabriquée** (`scirust-pdm/src/detectors.rs:184`) — un
  facteur `0.5` codé en dur sans unité convertit une amplitude FFT en une
  fausse vitesse RMS mm/s, alors qu'un module `iso10816.rs` correct existe
  et n'est pas utilisé.

*Deux constats initiaux ont été révisés après vérification adversariale :*
le manque de « tout optimiseur sous contraintes/global » était trop large —
`scirust-solvers::spg` (Spectral Projected Gradient, contraintes de boîte) et
`scirust-evo::{CmaEs, GeneticAlgorithm}` (optimisation globale) existent déjà ;
la lacune réelle et confirmée reste l'absence de **Levenberg-Marquardt** pour
les moindres carrés non linéaires (aucun `curve_fit` dans tout le dépôt).

---

## 4. Écart de couverture vs SciPy / LAPACK / GSL / statsmodels / scikit-learn

Chaque module couvre grossièrement **10 à 40 %** de son équivalent de
référence. Les manques les plus structurants, par ordre d'impact estimé :

| Domaine SciPy/LAPACK | Manque critique | Référence |
|---|---|---|
| `scipy.signal` | Filtrage IIR/FIR complet, FFT taille arbitraire (Bluestein), Welch/STFT | Oppenheim & Schafer ; Bluestein 1970 ; Welch 1967 |
| `scipy.linalg` | Eigen non symétrique (Francis QR), `expm`/`logm`, `rcond`, `lstsq` rang-déficient | Golub & Van Loan ch. 7 ; LAPACK dgeev/dgelsd |
| `scipy.optimize` | Levenberg-Marquardt/`curve_fit`, L-BFGS(-B), optimisation globale scipy-style | Moré 1978 (MINPACK) ; Byrd et al. 1995 |
| `scipy.special` | Bessel J/Y/I/K, Airy, elliptiques de Carlson, Lambert W, expint | Amos 1986 ; Abramowitz & Stegun |
| `scipy.stats` | Distributions discrètes, tests non paramétriques (Mann-Whitney, Wilcoxon, Shapiro-Wilk), KDE | Royston 1995 ; Silverman 1986 |
| `scipy.integrate` | Gauss-Kronrod/QUADPACK, BDF à ordre variable, Radau IIA | Piessens et al. 1983 ; Hairer & Wanner |
| `scipy.sparse` | Eigensolveurs (Lanczos/ARPACK), Cholesky sparse + AMD, spgemm | Lehoucq, Sorensen & Yang 1998 |
| `scipy.spatial` | KD-tree, cdist/pdist, convex hull, Delaunay (absent → DBSCAN/LOF en O(n²)) | Bentley 1975 ; Barber et al. 1996 |
| `scipy.interpolate` | B-splines, splines de lissage, interpolation 2D/ND, RBF | de Boor 1978 ; Dierckx 1993 |
| statsmodels | ARIMA/SARIMA, GARCH, tests ADF/KPSS/Johansen calibrés | Box & Jenkins ; Bollerslev 1986 |
| scikit-learn | Clustering hiérarchique/spectral/HDBSCAN, GMM covariance pleine, permutation importance | Ng, Jordan & Weiss 2002 |
| networkx/MiniSat/Z3 | Dijkstra/A*/Tarjan pondérés, CDCL, AC-3, Simplex SMT | Marques-Silva & Sakallah 1996 |

C'est la liste complète, avec chaque référence bibliographique, qui figure
dans les rapports détaillés par domaine (voir `missing_algorithms` de
l'audit brut, conservé dans l'historique de la branche pour traçabilité).

---

## 5. Infrastructure de test — le vrai levier structurel

**~4147 `#[test]`** au total, une base solide en volume. La qualité du haut
du panier (`scirust-sparse`, `scirust-gp`, `scirust-stiff`,
`scirust-tolerance`, `portable_f32`) est excellente : oracles denses
indépendants, valeurs tabulées externes (NIST/AIAG), campagnes exhaustives
(30 milliards d'entrées f32 pour les transcendantales).

Mais trois lacunes structurelles expliquent, à elles seules, pourquoi les
5 P0 et la plupart des P1 sont passés inaperçus :

1. **Aucun test de propriétés** (`proptest`/`quickcheck`) dans les ~120
   crates. C'est la pratique standard de SciPy (`hypothesis`, depuis 2021) et
   de LAPACK (ratios résiduels sur matrices générées à conditionnement
   prescrit, *LAPACK Users' Guide* ch. 7). Sans elle, on ne teste que les
   points qu'on a pensé à tester — exactement les cas `b=0`, base négative,
   dimension nulle, degré ≠ 2, qui ont produit les P0/P1.
2. **Un seul target de fuzzing** (`qsr1_from_bytes`) sur tout le dépôt ; le
   parseur ONNX non fiable et aucun noyau numérique ne sont fuzzés ; pas de
   fuzzing différentiel SIMD/scalaire/GPU.
3. **Très peu de valeurs de référence externes.** La majorité des tests sont
   de l'auto-cohérence (reconstruction, propriétés internes) plutôt que des
   comparaisons à SciPy/R/LAPACK/tables publiées — et quand une valeur
   externe existe, elle est parfois testée avec une tolérance si large
   qu'elle masque l'erreur (`p < 1e-5` alors que la valeur exacte
   `2.1e-6` est connue et documentée en commentaire, jamais assertée).

**C'est le chantier le plus rentable** : introduire `proptest` dans
`scirust-solvers`, `scirust-core` (linalg), `scirust-stats` et
`scirust-special` capturerait probablement, à lui seul, une bonne partie des
P1 restants avant qu'ils ne soient trouvés en production.

---

## 6. Plan de travail priorisé

### Chantier 0 — Corriger les 5 P0 (bloquant, cette semaine)
1. `regularized_gamma_p`/`_q` : asymptotique de Temme pour `a` grand.
2. PPO : rendre le `min` clippé/non-clippé différentiable correctement.
3. `ActorCriticAgent` : `grad_scale = td_error` (retirer le `* log_prob`).
4. `Dual::powf` : cas particulier exposant constant (`n·x^(n-1)·x'`).
5. GEMM tuilé GPU : corriger l'indexation mémoire partagée + activer enfin
   les tests de fusion (retirer les `eprintln!("skipped")`).

### Chantier 1 — Seuils absolus → relatifs (motif transversal, 1-2 semaines)
Audit systématique de tous les seuils `1e-1[0-9]` codés en dur dans
`scirust-solvers`, `scirust-sparse`, `scirust-multivariate`,
`scirust-estimation` ; remplacement par des critères relatifs à la norme
d'entrée (à la LAPACK) ; ajout du test de convergence initial manquant dans
CG/BiCGSTAB.

### Chantier 2 — Formes numériquement sûres dans les verticaux « certifiables »
Forme de Joseph pour tous les filtres de Kalman (`nav`, `estimation`),
formule quadratique stable partout, DARE par Schur plutôt que point fixe.
Ces crates revendiquent la certifiabilité : la barre de preuve doit être
maximale en premier ici.

### Chantier 3 — Fondations de test (rentabilise tout le reste)
`proptest` dans les 4 crates numériques cœur ; un deuxième target de fuzzing
(parseur ONNX) ; fuzzing différentiel SIMD/scalaire/GPU ; verrouiller les
valeurs de référence externes déjà connues mais non assertées (ex. le
`p ≈ 2.1e-6` documenté en commentaire).

### Chantier 4 — Combler les manques qui font 80 % de l'usage
Par ordre d'impact estimé sur l'utilisabilité :
1. Filtrage IIR/FIR (`scirust-signal`) — Butterworth/Chebyshev + bilinéaire + SOS.
2. Levenberg-Marquardt / `curve_fit` (`scirust-solvers`).
3. Eigensolveur non symétrique + `expm` (`scirust-solvers/linalg`).
4. ARIMA/SARIMA avec MLE Kalman (`scirust-forecast`).
5. Bessel J/Y/I/K (`scirust-special`) — débloque aussi la fenêtre de Kaiser.
6. `scirust-spatial` (KD-tree, cdist/pdist) — débloque DBSCAN/LOF en O(n log n).
7. Tests non paramétriques + distributions discrètes (`scirust-stats`).
8. CDCL/AC-3/Dijkstra pondéré/Tarjan (`scirust-neuro-symbolic`, `scirust-graph`).

### Chantier 5 — CMA-ES véritable, e-graphs, GAE pour PPO
Une fois le chantier 0 fermé : implémenter le CMA-ES complet de Hansen (le
nom est actuellement trompeur), les e-graphs pour la simplification
symbolique (explicitement dans le périmètre visé), et le calcul de GAE que
PPO reçoit actuellement de l'extérieur sans jamais le produire.

---

## 7. Note de méthode et de transparence

Sur 46 constats P0/P1 soumis à vérification adversariale (deux
vérificateurs indépendants par constat, chargés explicitement de réfuter),
**44 ont été confirmés tels quels**. Deux ont été nuancés plutôt que
simplement validés — traités en conséquence dans ce rapport (§3.3) :
- Le constat « aucune fonction de filtrage » a été jugé **trop absolu** : des
  filtres fréquentiels existent dans `scirust-signal::denoise`, mais aucune
  synthèse de filtre classique (IIR/FIR à pôles/zéros) — la reformulation
  précise remplace le constat initial.
- Le constat sur l'absence d'optimisation sous contraintes/globale a été
  **rejeté en l'état** : `spg` (contraintes de boîte) et `CmaEs`/
  `GeneticAlgorithm` (globale) existent déjà dans le dépôt ; seul le manque
  de Levenberg-Marquardt pour les moindres carrés non linéaires est retenu.

Les 111 constats P2/P3 n'ont pas été soumis à vérification adversariale
(contrainte de budget) ; ils restent des observations d'un seul auditeur de
domaine et devraient être revérifiés avant correction si un doute existe.

Les données brutes complètes (157 constats, 14 rapports de domaine, verdicts
de vérification détaillés) sont conservées dans l'historique de session
associé à cette branche.
