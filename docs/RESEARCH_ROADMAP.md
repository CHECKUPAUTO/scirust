# SciRust — Feuille de route « recherche → fonctions »

> Papers de recherche réels sélectionnés pour leur **fit avec les fondamentaux**
> de scirust (déterminisme bit-exact, certifiabilité, Rust pur testable) et
> traduits en **fonctions concrètes**. Chaque entrée : référence → fonction/
> module cible → statut → effort.
>
> Statuts : ✅ livré (testé, 8 gates verts) · 🔨 en cours · 📋 planifié.
> Effort : S (heures) · M (jours) · L (semaine) · XL (mois).
>
> Règle d'or (cf. [GROWTH_PLAN](GROWTH_PLAN.md)) : **aucune** entrée ne passe
> à ✅ sans test (gradient check pour un op, oracle/soundness pour une garantie)
> et sans les 8 gates verts. Pas de stub, pas de demi-implémentation.

## Tier 1 — Certifiable + déterministe (les différenciateurs)

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 1 | Gowal et al., *On the Effectiveness of Interval Bound Propagation* (2018) | `IbpMlp::certify(box) -> box` + `certified_robust` : intervalles propagés couche par couche ; **borne de sortie prouvée** (soundness testée) | `nn::ibp` | ✅ | S |
| 2 | Zhang et al., *CROWN* (NeurIPS 2018) ; Wang et al., *β-CROWN* (NeurIPS 2021, arXiv:2103.06624) | `crown_bounds` — bornes de sortie par relaxation linéaire (back-substitution) **plus serrées qu'IBP** ; soundness + tighter-than-IBP testés ; exposé dans `certify` (affiche IBP **et** CROWN) | `nn::ibp` | ✅ | L |
| 3 | Demmel & Nguyen, *Algorithms for Efficient Reproducible Floating-Point Summation* (ACM TOMS 2020) | `reproducible_sum`/`_mean`/`_dot` : somme **bit-identique quel que soit l'ordre / le nombre de threads** (tri canonique + expansion exacte) | `reproducible` | ✅ | M |
| 4 | Katz et al., *Reluplex* (CAV 2017, arXiv:1702.01135) ; *Marabou* (CAV 2019) | `verify-net` : vérification **complète** (SMT) d'une propriété sur un petit réseau ReLU | `scirust-neuro-symbolic` + CLI | 📋 | XL |
| 5 | *DiFR: Inference Verification Despite Nondeterminism* (2025, arXiv:2511.20621) | vérifier une trace d'inférence malgré le non-déterminisme | `scirust_runtime::proofcli` | 📋 | L |

## Tier 2 — Pile LLM N-D (gains rapides, gradient-checkables)

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 6 | Zhang & Sennrich, *Root Mean Square Layer Normalization* (NeurIPS 2019) | `NdRmsNorm` (+ op `rmsnorm`) ; `NdLlamaBlock` | `autodiff::nd`, `nn::nd_layers` | ✅ | S |
| 7 | Shazeer, *GLU Variants Improve Transformer* (2020, arXiv:2002.05202) | `NdSwiGLU` (+ op `sigmoid`/SiLU) ; `NdLlamaBlock` | `autodiff::nd`, `nn::nd_layers` | ✅ | S |
| 8 | Su et al., *RoFormer / RoPE* (2021) | op `rope` (gradient-checkée, propriété de position relative testée) + `NdMultiHeadAttention::with_rope` | `autodiff::nd`, `nn::nd_layers` | ✅ | M |
| 9 | Milakov & Gimelshein, *Online normalizer for softmax* (2018) ; Dao, *FlashAttention-2* (arXiv:2307.08691) | FlashAttention tuilée + online-softmax avec **backward** — **déjà présent** | `nn::transformer::flash_attention` | ✅ | M |
| 10 | Leviathan et al., *Speculative Decoding* (ICML 2023) ; Chen et al., *Speculative Sampling* (2023) | `generate_speculative` (variante greedy) : sortie **exactement** = greedy cible, moins de forwards. + `generate_greedy` | `nn::nd_decoder` | ✅ | M |
| 11 | Ainslie et al., *GQA* (2023) ; Shazeer, *MQA* (2019) | `NdMultiHeadAttention::new_gqa` (`num_kv_heads`, partage via broadcast `bmm` ; MQA = 1) | `nn::nd_layers` | ✅ | M |

## Tier 3 — Optimiseurs

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 12 | Loshchilov & Hutter, *Decoupled Weight Decay (AdamW)* (2017, arXiv:1711.05101) | `weight_decay` découplé dans `AdamConfig` + `NdAdam::with_lr_wd` | `nn::nd_optim` | ✅ | S |
| 13 | Chen et al., *Symbolic Discovery of Optimization Algorithms (Lion)* (2023) | `NdLion` (sign-based, mémoire moitié, déterministe) | `nn::nd_optim` | ✅ | S |
| 14 | Jordan et al., *Muon* (2024) | `NdMuon` (momentum + orthogonalisation Newton-Schulz) + `newton_schulz_orthogonalize` | `nn::nd_optim` | ✅ | M |

## Tier 4 — Quantification (thèse int8 bit-exact)

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 15 | Frantar et al., *GPTQ* (2022) ; Lin et al., *AWQ* (2023) ; Xiao et al., *SmoothQuant* (2022, arXiv:2211.10438) | **SmoothQuant** (`smoothquant_scales`/`apply_smoothquant`) + int8 per-canal + **GPTQ** (`quantize_gptq`/`gptq_hessian` : feedback d'erreur d'ordre 2 via Hessienne inverse de calibration ; CLI `gptq`) + **AWQ** (`awq_quantize`/`awq_act_scale` : scaling per-canal par recherche, conscient des activations ; CLI `awq`). Les trois testés < round-to-nearest | `quantization` | ✅ | L |

## Tier 5 — Pont calcul scientifique (fusion unique : solveurs + autograd + symbolique)

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 16 | Chen et al., *Neural ODEs* (NeurIPS 2018, arXiv:1806.07366) | `rk4_integrate` + `NeuralOde` : backprop **à travers** le solveur RK4 sur la tape N-D | `nn::neural_ode` | ✅ | M |
| 17 | Raissi, Perdikaris & Karniadakis, *PINNs* (J. Comp. Phys. 2019) | `nn::pinn` (`Pinn1D`, `solve_harmonic`) : **résidu de PDE dans la loss** — résout le problème aux limites `u''=−u`, `u(0)=0`, `u(π/2)=1` (solution `sin x`) ; `u''` par différences finies dans l'entrée (réseau partagé, grad params exact par autodiff inverse) ; vérifié vs solution analytique (erreur max ≈ 0,004) ; CLI `pinn` | `nn::pinn` | ✅ | L |

## Tier 6 — Architectures alternatives & confiance

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 18 | Gu & Dao, *Mamba* (2023, arXiv:2312.00752) | `selective_scan` + `NdMamba` — *selective scan* S6 (Δ, B, C dépendants de l'entrée ; A diagonal ; discrétisation `exp(Δ·A)`), récurrence déterministe linéaire-temps déroulée sur la tape (nouvel op `exp`) ⇒ **gradient check** ; match référence + entraînement ; CLI `mamba` | `nn::nd_layers` | ✅ | XL |
| 19 | Abadi et al., *Deep Learning with Differential Privacy (DP-SGD)* (2016) | `clip_gradients` + `add_noise` (gaussien **seedé**) + `dp_protect` + moments accountant (Rényi DP) — **déjà présent** | `dp` | ✅ | M |
| 20 | Frantar & Alistarh, *SparseGPT* (2023) ; Sun et al., *Wanda* (2023) ; Frankle & Carbin, *Lottery Ticket* (2019) | `prune_wanda` (activation-aware) + magnitude/structured/Lottery-Ticket déjà présents | `pruning` | ✅ | M |

## Tier 7 — Nouveaux papers (cycle 2, recherche du 14/06)

Trouvés lors d'une seconde recherche ; choisis pour leur fit avec les
fondamentaux (certifiable, déterministe, implémentable, testable).

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 21 | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (2021, arXiv:2107.07511) | `nn::conformal` : `conformal_quantile`, `ConformalRegressor`, `ConformalClassifier` — couverture garantie *sans hypothèse de distribution* ; tests : couverture empirique ≈ 1−α (régression + classification). CLI `scirust conformal`. | `nn::conformal` | ✅ | M |
| 22 | Defazio et al., *The Road Less Scheduled (Schedule-Free)* (2024 ; vainqueur MLCommons AlgoPerf) | `NdScheduleFree` : optimiseur **sans planning de LR** (moyenne Polyak `x`, point d'éval séparé) ; déterministe ; CLI `lm --opt schedule-free` | `nn::nd_optim` | ✅ | M |
| 23 | Pagliardini et al. (Apple), *The AdEMAMix Optimizer* (2024, arXiv:2409.03137) | `NdAdEMAMix` : **deux EMA** du gradient (rapide β1 + lente β3, mélange α) ; déterministe ; CLI `lm --opt ademamix` | `nn::nd_optim` | ✅ | M |
| 24 | Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam* (2024) | `NdSoap` — Adam dans la **base propre** de Shampoo (`L=E[GGᵀ]`, `R=E[GᵀG]` ; eigensolveur **Jacobi** déterministe `jacobi_eigenvectors`) ; convergence + déterminisme testés ; CLI `lm --opt soap` | `nn::nd_optim` | ✅ | L |
| 25 | Yang et al., *Gated Delta Networks / DeltaNet* (2024, arXiv:2412.06464) | `delta_rule` + `NdDeltaNet` — **attention linéaire récurrente** (règle delta : `S_t = S_{t-1} + β_t(v_t − S_{t-1}k_t)k_tᵀ`), temps linéaire, causale, déterministe ; déroulée sur la tape (nouvel op `cat0`) ⇒ **gradient check** ; match référence + entraînement ; CLI `deltanet` | `nn::nd_layers` | ✅ | L |

## Tier 8 — Candidats vérifiés (cycle 3, recherche du 15/06) — vérification & robustesse certifiée

> ~55 papers réels (arXiv vérifié) trouvés en recherche, traduits en fonctions
> scirust concrètes, choisis pour leur fit avec les fondamentaux (certifiable,
> déterministe, testable, Rust pur). Tous 📋 (candidats à implémenter), au même
> standard que les ✅ (test/oracle + 8 gates). Prolongent IBP/CROWN (#1-2).

| # | Papier | Fonction scirust proposée | Module | Statut | Effort |
|---|--------|---------------------------|--------|--------|--------|
| 26 | Zhang et al., *GCP-CROWN : General Cutting Planes for BaB Verification* (NeurIPS 2022, arXiv:2208.05740) | vérificateur **complet** par branch-and-bound (split des ReLU instables, bornes IBP/CROWN pour l'élagage, plans coupants) ; prolonge `verify-net` (#4) | `nn::ibp` + CLI | 📋 | XL |
| 27 | Cohen, Rosenfeld & Kolter, *Certified Robustness via Randomized Smoothing* (ICML 2019, arXiv:1902.02918) | `nn::smoothing::SmoothedClassifier::certify` : classifieur lissé, **rayon L2 prouvé** `σ·Φ⁻¹(pₐ)` via bruit gaussien seedé + borne **Clopper-Pearson** (`betai`/`lgamma`) + probit `Φ⁻¹` (Acklam) ; oracle : rayon = distance exacte au demi-espace (indép. de σ) + soundness/abstention + déterminisme ; CLI `certify` (IBP/CROWN + smoothing) | `nn::smoothing` | ✅ | M |
| 28 | Singh et al., *DeepPoly : An Abstract Domain for Certifying NN* (POPL 2019) | domaine abstrait (polyèdres à une variable) : bornes plus serrées que les intervalles ; relaxation ReLU asymétrique | `nn::ibp` | 📋 | L |
| 29 | Gehr et al., *AI² : Abstract Interpretation for NN* (IEEE S&P 2018) | `Zonotope`/`certify_zonotope` : propagation par **zonotopes** (DeepZ) — affine exact, ReLU relaxé `λx+μ±μ` (1 générateur/neurone instable) ; oracle : affine exact + **soundness** (échantillonnage) + **plus serré qu'IBP** sous corrélation (`relu(x)−relu(x)` : zono [−0,5;0,5] vs IBP [−1;1]) ; exposé dans CLI `certify` | `nn::ibp` | ✅ | L |
| 30 | Zhang et al., *CROWN-IBP : Stable & Efficient Verified Training* (ICLR 2020, arXiv:1906.06316) | **entraînement certifié** : loss bornée (IBP + CROWN) ⇒ réseau prouvablement robuste ; oracle : rayon certifié croît à l'entraînement | `nn::ibp` + `nn::nd_layers` | 📋 | L |
| 31 | Tjeng, Xiao & Tedrake, *Evaluating Robustness with MILP* (ICLR 2019, arXiv:1711.07356) | vérification **exacte** d'un petit réseau ReLU par programmation linéaire en nombres entiers (encodage big-M) ; complète #4 | `scirust-neuro-symbolic` + CLI | 📋 | L |
| 32 | Leino, Wang & Fredrikson, *Globally-Robust Neural Networks (GloRo)* (ICML 2021) | `nn::lipschitz` : `spectral_norm` (power iteration) + `spectral_normalize` (couche **1-Lipschitz**) + `GloroClassifier` (rayon L2 prouvé `marge/(√2‖W‖₂)`) ; oracle : normes spectrales connues + rayon **sain** (pire perturbation ne bascule pas) + **conservateur** (≤ distance exacte à la frontière) + déterminisme | `nn::lipschitz` | ✅ | M |

## Tier 9 — Incertitude, calibration & conformal (au-delà du split-conformal #21)

| # | Papier | Fonction scirust proposée | Module | Statut | Effort |
|---|--------|---------------------------|--------|--------|--------|
| 33 | Romano, Patterson & Candès, *Conformalized Quantile Regression (CQR)* (NeurIPS 2019, arXiv:1905.03222) | `ConformalQuantileRegressor` : conformalise un régresseur de quantiles (score `Eᵢ=max(q_lo−y, y−q_hi)`, correction finie `Q`) ⇒ intervalles **adaptatifs** `[q_lo−Q, q_hi+Q]`, couverture marginale ≥ 1−α ; oracle : sémantique exacte du score + couverture (bande) + largeur variable (région forte vs faible bruit) + déterminisme ; CLI `conformal` (split + CQR) | `nn::conformal` | ✅ | M |
| 34 | Romano, Sesia & Candès, *Classification with Valid & Adaptive Coverage (APS)* (NeurIPS 2020, arXiv:2006.02544) | `AdaptivePredictionSets` : ensembles de **classification** par score cumulatif `s(x,c)` (masse des classes ≥ probables que c) ; set `{c : s≤q̂}`, couverture marginale ≥ 1−α + **taille adaptative** (facile→petit, ambigu→grand) ; oracle : score exact + couverture + adaptativité + déterminisme | `nn::conformal` | ✅ | M |
| 35 | Angelopoulos et al., *RAPS : Regularized Adaptive Prediction Sets* (ICLR 2021, arXiv:2009.14193) | `AdaptivePredictionSets::calibrate_raps` : pénalité `λ·max(0, rang−k_reg)` ajoutée au score ⇒ ensembles **plus petits** qu'APS à couverture égale ; oracle : taille moyenne RAPS < APS avec couverture ≥ 1−α | `nn::conformal` | ✅ | M |
| 36 | Bates et al., *Risk-Controlling Prediction Sets (RCPS)* (J. ACM 2021, arXiv:2101.02703) | `hoeffding_ucb` + `rcps_select` : contrôle d'un **risque** borné (au-delà de la couverture) — plus petit λ dont la borne de concentration (Hoeffding) est ≤ α ⇒ `R(λ̂)≤α` w.p. 1−δ (PAC) ; oracle : sélection exacte + risque test ≤ α sur données fraîches | `nn::conformal` | ✅ | M |
| 37 | Angelopoulos et al., *Learn then Test* (arXiv:2110.01052) | `learn_then_test` : contrôle de **risques multiples** par p-values de Hoeffding + correction familiale (Bonferroni) ; FWER ≤ δ vérifié par simulation ; déterministe | `nn::conformal` | ✅ | M |
| 38 | Gibbs & Candès, *Adaptive Conformal Inference* (NeurIPS 2021, arXiv:2106.00170) | `AdaptiveConformal` : conformal **en ligne** — niveau αₜ adapté par rétroaction `αₜ₊₁=αₜ+γ(α−errₜ)` ⇒ couverture ≈ 1−α **sous dérive** (là où le conformal statique s'effondre) ; oracle : règle de mise à jour exacte + couverture maintenue sous changement de variance + déterminisme | `nn::conformal` | ✅ | M |
| 39 | Guo et al., *On Calibration of Modern NN (Temperature Scaling)* (ICML 2017, arXiv:1706.04599) | `nn::calibration` : `temperature_scale` (golden-section sur la NLL) + `expected_calibration_error` + `nll` ; recalibration post-hoc **sans changer l'accuracy** ; oracle : ECE baisse (testé, déterministe) ; CLI `calibrate` | `nn::calibration` | ✅ | S |
| 40 | Lakshminarayanan et al., *Deep Ensembles* (NeurIPS 2017, arXiv:1612.01474) | `nn::ensemble::DeepEnsemble` : N MLP ReLU seedés (tape + NdAdam) ; `predict→(moy, écart-type)` = estimation + **incertitude épistémique** ; oracle : MSE ensemble ≤ moy. membres (Jensen) + écart-type **≫ hors-distribution** + déterminisme | `nn` | ✅ | M |

## Tier 10 — Optimiseurs (au-delà d'Adam/Lion/Muon/SF/AdEMAMix/SOAP)

| # | Papier | Fonction scirust proposée | Module | Statut | Effort |
|---|--------|---------------------------|--------|--------|--------|
| 41 | Gupta, Koren & Singer, *Shampoo* (ICML 2018, arXiv:1802.09568) | `NdShampoo` : préconditionneur **Kronecker** (`L^{-1/4} G R^{-1/4}`, racines inverses via `inverse_pth_root`/`jacobi_eigenvectors`) ; matrices → update préconditionné, vecteurs → Adagrad diagonal ; oracle racine inverse (`A^{-1/2}²·A≈I`) + convergence + déterminisme testés ; CLI `lm --opt shampoo` | `nn::nd_optim` | ✅ | L |
| 42 | Shazeer & Stern, *Adafactor* (ICML 2018, arXiv:1804.04235) | `NdAdafactor` : moments du 2e ordre **factorisés** (sommes ligne/colonne `V[i,j]=R[i]·C[j]/ΣR`, mémoire sous-linéaire) + clipping RMS de l'update + planning β2 ; reconstruction rang-1 exacte + convergence (bande) + déterminisme testés ; CLI `lm --opt adafactor` | `nn::nd_optim` | ✅ | M |
| 43 | You et al., *LAMB* (ICLR 2020, arXiv:1904.00962) | `NdLamb` : Adam à **confiance par couche** (ratio `‖θ‖/‖r‖` par tenseur) ; convergence (bande) + déterminisme testés ; CLI `lm --opt lamb` | `nn::nd_optim` | ✅ | M |
| 44 | Liu et al., *Sophia* (arXiv:2305.14342) | `NdSophia` : 2e ordre **clippé** (Hessienne diagonale estimée, Hutchinson seedé) ; CLI `lm --opt sophia` | `nn::nd_optim` | 📋 | L |
| 45 | Zhang et al., *Lookahead* (NeurIPS 2019, arXiv:1907.08610) | `NdLookahead` : wrapper **poids lents/rapides** autour d'Adam (`k` pas rapides puis `φ←φ+α(θ−φ); θ←φ`) ; déterministe ; convergence + déterminisme testés ; CLI `lm --opt lookahead` | `nn::nd_optim` | ✅ | S |
| 46 | Mishchenko & Defazio, *Prodigy* (arXiv:2306.06101) | `NdProdigy` : Adam **sans learning-rate** — estime `d≈‖x₀−x*‖` en ligne (corrélation `⟨g,x₀−x⟩` globale) et l'utilise comme pas ; oracle : `d` s'adapte à l'échelle + perte réduite + déterminisme ; CLI `lm --opt prodigy` | `nn::nd_optim` | ✅ | M |
| 47 | Foret et al., *Sharpness-Aware Minimization (SAM)* (ICLR 2021, arXiv:2010.01412) | `NdSam` : `ascent` (perturbe vers `θ+ρ·g/‖g‖`, pire cas local) puis `descent` (restaure θ, pas SGD au gradient perturbé) ; oracle : perturbation = `ρ·g/‖g‖` (‖ε‖=ρ) + convergence (bande ∝ lr·ρ) + déterminisme ; bibliothèque (2 gradients/pas ⇒ hors boucle `lm --opt`) | `nn::nd_optim` | ✅ | M |
| 48 | Zhao et al., *GaLore* (ICML 2024, arXiv:2403.03507) | `NdGalore`/`galore_subspace` : **projection low-rank des gradients** — Adam tourne dans le sous-espace dominant `PᵀG` (top-`r` vecteurs singuliers, réutilise `jacobi_eigenvectors`) ⇒ états d'optimiseur compressés `r×max(m,n)` ; oracle : `P` orthonormal + projection orthogonale optimale (Pythagore) + gradient bas-rang reconstruit exactement + **convergence sur cible bas-rang avec état compressé** + sous-rang ⇒ résidu + déterminisme ; CLI `lm --opt galore` | `nn::nd_optim` | ✅ | M |
| 49 | Xie et al., *Adan* (arXiv:2208.06677) | `NdAdan` : momentum de **Nesterov adaptatif** (3 EMA : gradient, différences, terme look-ahead au carré) ; convergence + déterminisme testés ; CLI `lm --opt adan` | `nn::nd_optim` | ✅ | M |

## Tier 11 — Modèles de séquence efficaces (au-delà de Mamba/DeltaNet/Flash/RoPE/GQA)

| # | Papier | Fonction scirust proposée | Module | Statut | Effort |
|---|--------|---------------------------|--------|--------|--------|
| 50 | Dao & Gu, *Mamba-2 / Structured State-Space Duality* (ICML 2024, arXiv:2405.21060) | `selective_scan_ssd` : scan par **dualité** (chunké, plus rapide) ; match exact de `selective_scan` (#18) + gradient check | `nn::nd_layers` | 📋 | L |
| 51 | Gu, Goel & Ré, *S4 : Structured State Spaces* (ICLR 2022, arXiv:2111.00396) | `s4_scan`/`NdS4` : SSM diagonal **LTI** (S4D) — `Ā=exp(Δ⊙A)`, `B̄=Δ⊙B`, `h_t=Ā⊙h_{t−1}+B̄⊙x_t`, `y_t=Σ_n C⊙h_t` déroulé sur la tape ; init **HiPPO** diagonale `A[:,j]=−(j+1)` ; oracle : **gradient check** (x, a_log, B, C, log_dt) + entraînement (MSE↓) + déterminisme | `nn::nd_layers` | ✅ | L |
| 52 | Smith, Warrington & Linderman, *S5* (ICLR 2023, arXiv:2208.04933) | `NdS5` : SSM **MIMO** par scan parallèle (associatif) ; déterministe | `nn::nd_layers` | 📋 | L |
| 53 | Peng et al., *RWKV* (EMNLP Findings 2023, arXiv:2305.13048) | `rwkv_wkv` + `NdRwkv` : mélange temporel **WKV** (décroissance expo. par canal + bonus, normalisé) déroulé sur la tape (nouvel op `div`) ; oracle : récurrent ≡ formule explicite + gradient check (k,v,decay,bonus) + entraînement + déterminisme ; CLI `rwkv` | `nn::nd_layers` | ✅ | L |
| 54 | Sun et al., *RetNet : Retentive Network* (arXiv:2307.08621) | `retention` + `NdRetention` : récurrence d'attention linéaire à décroissance γ (`S_t=γS_{t-1}+kₜᵀvₜ`, `o_t=q_tS_t`) déroulée sur la tape ; **oracle : forme récurrente ≡ forme parallèle** `(QKᵀ⊙D)V` + gradient check + entraînement ; CLI `retnet` | `nn::nd_layers` | ✅ | L |
| 55 | Yang et al., *Gated Linear Attention (GLA)* (ICML 2024, arXiv:2312.06635) | `gated_linear_attention` + `NdGla` : attention linéaire **gatée** — porte d'oubli par canal **dépendante de l'entrée** `αₜ=σ(·)` (`S_t=diag(αₜ)S_{t-1}+kₜᵀvₜ`), déroulée sur la tape ; match référence + gradient check (q,k,v,α) + entraînement ; CLI `gla` | `nn::nd_layers` | ✅ | L |
| 56 | Poli et al., *Hyena* (ICML 2023, arXiv:2302.10866) | `NdHyena` : **convolutions longues implicites** + gating (alternative à l'attention) ; gradient check | `nn::nd_layers` | 📋 | L |
| 57 | Beck et al., *xLSTM* (NeurIPS 2024, arXiv:2405.04517) | `NdXlstm` : LSTM étendu (sLSTM scalaire + mLSTM matriciel), récurrence déroulée ; gradient check | `nn::nd_layers` | 📋 | L |
| 58 | Qin et al., *HGRN : Hierarchically Gated RNN* (NeurIPS 2023, arXiv:2311.04823) | `hgrn` + `NdHgrn` : RNN linéaire à intégration leaky par canal, porte d'oubli **bornée inférieurement** `f=lb+(1−lb)σ(·)` (la borne `lb` fixe l'horizon mémoire minimal, croissant par couche) ; match référence + gradient check (c,f) + entraînement ; CLI `hgrn` | `nn::nd_layers` | ✅ | M |
| 59 | Press, Smith & Lewis, *ALiBi* (ICLR 2022, arXiv:2108.12409) | `alibi_slopes` + `alibi_bias` (biais d'attention **linéaire en distance**, pentes `2^(−8h/H)`) + `NdMultiHeadAttention::with_alibi` ; oracle : pentes géométriques + biais linéaire/causal/Toeplitz + poids softmax décroissants (∝ exp(−pente·dist)) + attention déterministe | `nn::nd_layers` | ✅ | S |
| 60 | Peng et al., *YaRN* (arXiv:2309.00071) | `nn::yarn` : `yarn_frequencies`/`rope_yarn` — extension de contexte RoPE par **interpolation NTK-by-parts** (garde les hautes fréquences, interpole les basses) + température `yarn_attention_scale` ; oracle : propriété de **position relative** préservée + angle basse-fréquence ramené en distribution à `s·L` + `scale=1` ≡ RoPE simple | `nn::yarn` | ✅ | M |

## Tier 12 — Décodage & inférence efficaces (au-delà du spéculatif #10)

| # | Papier | Fonction scirust proposée | Module | Statut | Effort |
|---|--------|---------------------------|--------|--------|--------|
| 61 | Cai et al., *Medusa* (ICML 2024, arXiv:2401.10774) | `MedusaHeads`/`generate_medusa` : **têtes de décodage multiples** (tête `j` prédit le token à +`j+2` depuis l'état caché) ⇒ brouillon multi-token d'un seul forward, vérifié (préfixe accepté + correction greedy) ; oracle : sortie **exactement = greedy** pour des têtes quelconques + déterminisme + têtes entraînées ⇒ blocs acceptent >1 token (forwards < 2·n) en restant exact | `nn::nd_decoder` | ✅ | M |
| 62 | Li et al., *EAGLE* (ICML 2024, arXiv:2401.15077) | `EagleHead`/`generate_eagle` : décodage spéculatif au niveau **features** — tête `(feature, embed(token)) → feature suivant` autorégressée, mappée par la tête LM gelée ⇒ brouillon ; vérifié (préfixe + correction) ; oracle : sortie **exacte = greedy** pour une tête quelconque + déterminisme + tête entraînée (MSE sur features) ⇒ blocs acceptent >1 token (forwards < 2·n) en restant exact | `nn::nd_decoder` | ✅ | M |
| 63 | Kwon et al., *PagedAttention / vLLM* (SOSP 2023, arXiv:2309.06180) | `nn::paged_attention::PagedKvCache` : KV-cache **paginé** (blocs d'un pool, table de blocs), attention indexée **via la table** ; oracle : gather **bit-identique** sous fragmentation (blocs leurres interleavés) + `attention` paginée **bit-identique** au cache contigu + comptabilité des blocs/cas vide | `nn::paged_attention` | ✅ | M |

## Tier 13 — Quantification & compression (au-delà de GPTQ/AWQ/SmoothQuant #15)

| # | Papier | Fonction scirust proposée | Module | Statut | Effort |
|---|--------|---------------------------|--------|--------|--------|
| 64 | Tseng et al., *QuIP#* (ICML 2024, arXiv:2402.04396) | `quantize_quip` : **incohérence Hadamard** (randomisée seedée) + codebooks lattice E8 ; oracle < round-to-nearest en 2-bit | `quantization` | 📋 | L |
| 65 | Shao et al., *OmniQuant* (ICLR 2024, arXiv:2308.13137) | `omniquant_quantize` : **clipping de poids apprenable** (LWC) — facteur de coupe `γ∈(0,1]` par canal de sortie qui rétrécit la plage `γ·max\|w\|` (recherche déterministe sur grille **incluant γ=1=RTN**) ; oracle : **< RTN** sur poids à queue lourde (au moins un canal coupe) + **jamais pire** que RTN + déterminisme | `quantization` | ✅ | L |
| 66 | Kim et al., *SqueezeLLM* (arXiv:2306.07629) | `SqueezeLlmCodebook` : quantification **non-uniforme** par k-means **pondéré par la sensibilité** (proxy Hessien diag.) ; init quantile + Lloyd déterministe ; oracle : erreur pondérée **< RTN** (gaussien, 3 bits, <0,85×) + round-trip exact + déterminisme ; bibliothèque | `quantization` | ✅ | M |
| 67 | Dettmers et al., *SpQR* (arXiv:2306.03078) | `SpqrOutliers` : **sparse-quantized** — garde la fraction d'outliers (plus grosses erreurs de quantif) en fp, le reste en dense bas-bit ; oracle : erreur lourde-queue divisée (1 % d'outliers ⇒ erreur < 0,3×) + reconstruction exacte des outliers + déterminisme ; bibliothèque | `quantization` | ✅ | M |
| 68 | Hooper et al., *KVQuant* (NeurIPS 2024, arXiv:2401.18079) | `kvquant_kv` : quant du **KV-cache** — clés **per-canal** (outliers de canal), valeurs **per-token** ; oracle : erreur d'attention **< per-tensor** (<0,5×, clés à outliers de canal) + per-canal résout les petites colonnes + déterminisme | `quantization` | ✅ | M |
| 69 | Ma et al., *BitNet b1.58* (arXiv:2402.17764) | `ternary_quantize` + `ternary_matmul` : poids **ternaires {−1,0,1}** (échelle absmean, ~1,58 bit/poids) ; matmul **sans multiplication** (somme/diff/skip) ; **oracle** : = produit déquantifié (bit-exact pour la forme somme-de-signes) ; CLI `bitnet` | `quantization` | ✅ | M |
| 70 | Egiazarian et al., *AQLM : Additive Quantization* (ICML 2024, arXiv:2401.06118) | `quantize_aqlm` : **quantification additive** multi-codebook (codebooks appris) ; oracle < RTN en 2-bit | `quantization` | 📋 | L |
| 71 | Dettmers et al., *LLM.int8()* (NeurIPS 2022, arXiv:2208.07339) | `int8_mixed_matmul` : matmul mixte — colonnes de features **outliers** (>seuil) gardées en fp32, reste en **int8** ; oracle : erreur vs fp **< 0,5×** l'int8 simple (activations à outliers) + réduit à l'int8 pur sans outliers + déterminisme | `quantization` | ✅ | M |
| 72 | Hu et al., *LoRA* (ICLR 2022, arXiv:2106.09685) | `LoraLinear` : adaptation **low-rank** (`W` gelé + `ΔW = (α/r)·A·B`, seuls `A`,`B` entraînés) ; `B=0` à l'init ⇒ = base ; gradient check sur `A`,`B` ; couche de la tape N-D | `nn::nd_layers` | ✅ | M |
| 73 | Liu et al., *DoRA* (ICML 2024, arXiv:2402.09353) | `nn::dora::DoraLinear` : LoRA décomposée **magnitude/direction** `W'=m⊙(W₀+BA)/‖W₀+BA‖_col` (W₀ gelé ; m, A, B entraînés) ; backward de la normalisation par colonne en forme close ; oracle : init `B=0, m=‖W₀‖_col` ⇒ **W'=W₀ exact** + **gradient check** (différences finies vs analytique) + récupère une cible DoRA (perte ÷100) + déterminisme | `nn::dora` | ✅ | M |
| 74 | Dettmers et al., *QLoRA / NF4* (NeurIPS 2023, arXiv:2305.14314) | `nf4_quantize`/`nf4_dequantize` + `NF4_LEVELS` : type 4-bit **NormalFloat** (16 niveaux = quantiles d'une normale, échelle absmax) ; **oracle** : erreur < int4 uniforme sur poids gaussiens (+ round-trip exact + déterminisme) | `quantization` | ✅ | M |

## Tier 14 — Calcul scientifique, déterminisme & audit (au-delà de Neural ODE/PINN/reproducible)

| # | Papier | Fonction scirust proposée | Module | Statut | Effort |
|---|--------|---------------------------|--------|--------|--------|
| 75 | Li et al., *Fourier Neural Operator (FNO)* (ICLR 2021, arXiv:2010.08895) | `nn::fno` : opérateur appris dans le **domaine de Fourier** (DFT réelle déterministe + multiplication spectrale) ; résout des PDE paramétriques ; gradient check | `nn::fno` | 📋 | L |
| 76 | Lu et al., *DeepONet* (Nature Mach. Intell. 2021, arXiv:1910.03193) | `nn::deeponet::DeepONet` : apprentissage d'**opérateurs** `G(u)(y)=Σ b_k(u)·t_k(y)` (trunk cosinus fixe + branch linéaire = POD-DeepONet, convexe) ; oracle : apprend l'**antidérivée** et **généralise à des fonctions non vues** (MSE test < 0,01, ≪ baseline) + déterminisme | `nn::deeponet` | ✅ | L |
| 77 | Liu et al., *KAN : Kolmogorov-Arnold Networks* (arXiv:2404.19756) | `nn::kan::KanLayer` : activations **apprenables sur arêtes** (base RBF de FastKAN, Li 2024) `y_j=Σᵢφᵢⱼ(xᵢ)` ; sortie linéaire en coeffs ⇒ ajustement convexe (GD) ; oracle : ajuste une cible additive non-linéaire `sin(2x₀)+x₁²` (MSE<0,02, ≪ modèle linéaire) + base localisée + déterminisme ; bibliothèque | `nn::kan` | ✅ | L |
| 78 | Mironov, *Rényi Differential Privacy* (CSF 2017, arXiv:1702.07476) | `dp::gaussian_rdp`/`rdp_to_dp`/`rdp_gaussian_epsilon` : **RDP** du mécanisme gaussien `α/(2σ²)` + conversion Mironov `ε=RDP+ln(1/δ)/(α−1)` optimisée sur α ; renforce DP-SGD (#19) ; oracle : RDP/conversion exactes + **bien plus serré** que la composition basique + monotonie | `dp` | ✅ | M |
| 79 | Kirchenbauer et al., *A Watermark for LLMs* (ICML 2023, arXiv:2301.10226) | `nn::watermark` : `is_green`/`apply_green_bias`/`detect_z` — partition vert/rouge **seedée** (hash de `(seed,prev,token)`) + biais de logits + détection par **test z** `(g−γn)/√(nγ(1−γ))` **sans accès au modèle** ; oracle : fraction verte ≈ γ + biais sur tokens verts seulement + texte filigrané détecté (z≫8) vs naturel (z≈0) + mauvais seed non détecté + déterminisme | `nn::watermark` | ✅ | M |
| 80 | *ZK-based Verifiable ML* (survey arXiv:2502.18535 ; zkSNARK eval arXiv:2402.02675) | preuve d'**inférence vérifiable** : empreinte/argument compact qu'une sortie provient bien du modèle déclaré ; prolonge `proofcli` (#5) vers une garantie cryptographique | `scirust_runtime::proofcli` | 📋 | XL |

---

## Ordre d'attaque

**✅ Livré / présent** (testé + 8 gates verts) : IBP certifié (#1) · **CROWN
(#2)** · sommation reproductible (#3) · RoPE N-D (#8) · RMSNorm + SwiGLU +
`NdLlamaBlock` (#6, #7) · FlashAttention online-softmax (#9) · décodage spéculatif
exact (#10) · GQA/MQA (#11) · AdamW + Lion (#12, #13) · Muon (#14) · Neural ODE
(#16) · DP-SGD (#19) · pruning Wanda + magnitude/lottery (#20) · **SmoothQuant +
GPTQ + AWQ (#15)** · **conformal prediction (#21)** · **Schedule-Free (#22)** ·
**AdEMAMix (#23)** · **SOAP (#24)** · **DeltaNet (#25)** · **Mamba (#18)** ·
**PINN (#17)**. → **18/20 + #21 + #22 + #23 + #24 + #25**.

**Paris lourds** (planifiés, jalonnés) : SMT/Marabou (#4) · DiFR (#5).

**Pool de candidats vérifiés (Tier 8-14, #26-#80, ~55 papers, recherche du
15/06)** : prochaines implémentations, par ordre de tractabilité estimé —
*gains rapides* : Lookahead (#45), ALiBi (#59), temperature scaling (#39),
LoRA (#72) ; *modèles de séquence* (réutilisent tape + `cat0`/`exp`) : Mamba-2
(#50), GLA (#55), RetNet (#54), RWKV (#53), S4/S5 (#51/#52), HGRN (#58) ;
*optimiseurs* (réutilisent `jacobi_eigenvectors`) : Shampoo (#41), Sophia (#44),
Adafactor (#42), LAMB (#43), Adan (#49), Prodigy (#46), SAM (#47) ;
*quantification* (oracle < RTN) : QuIP# (#64), AQLM (#70), BitNet b1.58 (#69),
SqueezeLLM (#66), SpQR (#67), KVQuant (#68), NF4 (#74), LLM.int8 (#71) ;
*certifiable* (prolonge IBP/CROWN) : randomized smoothing (#27), GCP-CROWN BaB
(#26), CROWN-IBP (#30), MILP (#31), DeepPoly/AI² (#28/#29), Lipschitz (#32) ;
*conformal/incertitude* : CQR (#33), APS/RAPS (#34/#35), RCPS+LtT (#36/#37), ACI
(#38), deep ensembles (#40) ; *décodage* : Medusa (#61), EAGLE (#62),
PagedAttention (#63) ; *scientifique* : FNO (#75), DeepONet (#76), KAN (#77) ;
*audit/privacy* : Rényi-DP accountant (#78), watermark LLM (#79), preuve
d'inférence ZK (#80). Tous au même standard : test/oracle + 8 gates avant ✅.

Chaque item respecte les fondamentaux : op autograd ⇒ **gradient check** ;
garantie (borne, privacy, reproductibilité) ⇒ **test d'oracle/soundness** ;
déterminisme préservé (PCG seedé, ordre fixe) ; 8 gates verts.
