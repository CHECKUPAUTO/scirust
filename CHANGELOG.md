# Changelog

Le format suit [Keep a Changelog](https://keepachangelog.com/) ;
versions sémantiques à partir de la prochaine release taguée.

## [Non publié]

### Ajouté — campagne « faire grandir scirust »
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
