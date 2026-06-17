# LIVESTATE — scirust

> Fichier de bord partagé entre agents.
> Dernière mise à jour : 2026-06-17

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
