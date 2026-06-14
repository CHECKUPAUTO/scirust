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
| 2 | Zhang et al., *CROWN* (NeurIPS 2018) ; Wang et al., *β-CROWN* (NeurIPS 2021, arXiv:2103.06624) | bornes par relaxation linéaire (plus serrées qu'IBP) | `nn::certified` | 📋 | L |
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
| 15 | Frantar et al., *GPTQ* (2022) ; Lin et al., *AWQ* (2023) ; Xiao et al., *SmoothQuant* (2022, arXiv:2211.10438) | **SmoothQuant** (`smoothquant_scales`/`apply_smoothquant`) + int8 per-canal déjà présent ; **GPTQ/AWQ à venir** | `quantization` | 🔨 | L |

## Tier 5 — Pont calcul scientifique (fusion unique : solveurs + autograd + symbolique)

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 16 | Chen et al., *Neural ODEs* (NeurIPS 2018, arXiv:1806.07366) | `rk4_integrate` + `NeuralOde` : backprop **à travers** le solveur RK4 sur la tape N-D | `nn::neural_ode` | ✅ | M |
| 17 | Raissi, Perdikaris & Karniadakis, *PINNs* (J. Comp. Phys. 2019) | `pinn` : résidu de PDE dans la loss (besoin d'autodiff d'ordre 2) | `nn` + `symbolic` | 📋 | L |

## Tier 6 — Architectures alternatives & confiance

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 18 | Gu & Dao, *Mamba* (2023, arXiv:2312.00752) | *selective scan* (récurrence déterministe) ; modèle séquence linéaire-temps | `nn` | 📋 | XL |
| 19 | Abadi et al., *Deep Learning with Differential Privacy (DP-SGD)* (2016) | `clip_gradients` + `add_noise` (gaussien **seedé**) + `dp_protect` + moments accountant (Rényi DP) — **déjà présent** | `dp` | ✅ | M |
| 20 | Frantar & Alistarh, *SparseGPT* (2023) ; Sun et al., *Wanda* (2023) ; Frankle & Carbin, *Lottery Ticket* (2019) | `prune_wanda` (activation-aware) + magnitude/structured/Lottery-Ticket déjà présents | `pruning` | ✅ | M |

## Tier 7 — Nouveaux papers (cycle 2, recherche du 14/06)

Trouvés lors d'une seconde recherche ; choisis pour leur fit avec les
fondamentaux (certifiable, déterministe, implémentable, testable).

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 21 | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (2021, arXiv:2107.07511) | `conformal` : ensembles de prédiction **split-conformal** à couverture garantie *sans hypothèse de distribution* (quantile de calibration) ; test : couverture empirique ≈ 1−α. **Fort fit « IA certifiable »** | `nn` (nouveau) | 📋 | M |
| 22 | Defazio et al., *The Road Less Scheduled (Schedule-Free)* (2024 ; vainqueur MLCommons AlgoPerf) | `NdScheduleFree` : optimiseur **sans planning de LR** (moyenne de poids) ; déterministe | `nn::nd_optim` | 📋 | M |
| 23 | Pagliardini et al. (Apple), *The AdEMAMix Optimizer* (2024, arXiv:2409.03137) | optimiseur à **deux EMA** (mémoire de gradient longue) ; déterministe | `nn::nd_optim` | 📋 | M |
| 24 | Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam* (2024) | optimiseur préconditionné (Shampoo dans la base propre + Adam) | `nn::nd_optim` | 📋 | L |
| 25 | Yang et al., *Gated Delta Networks / DeltaNet* (2024, arXiv:2412.06464) | couche d'**attention linéaire récurrente** (règle delta), temps linéaire, déterministe ; alternative plus tractable que Mamba | `nn::nd_layers` | 📋 | L |

---

## Ordre d'attaque

**✅ Livré / présent** (testé + 8 gates verts) : IBP certifié (#1) · sommation
reproductible (#3) · RoPE N-D (#8) · RMSNorm + SwiGLU + `NdLlamaBlock` (#6, #7) ·
FlashAttention online-softmax (#9) · décodage spéculatif exact (#10) · GQA/MQA
(#11) · AdamW + Lion (#12, #13) · Muon (#14) · Neural ODE (#16) · DP-SGD (#19) ·
pruning Wanda + magnitude/lottery (#20). → **14 des 20** ; SmoothQuant (#15)
partiel.

**Ensuite** : **conformal prediction (#21, fort fit certifiable)** · Schedule-Free
(#22) · AdEMAMix (#23) · GPTQ/AWQ (#15, raffinement de la quantification).

**Paris lourds** (planifiés, jalonnés) : CROWN (#2) · SMT/Marabou (#4) ·
Mamba (#18) · DeltaNet (#25) · SOAP (#24) · PINN (#17, après l'autodiff d'ordre 2)
· DiFR (#5).

Chaque item respecte les fondamentaux : op autograd ⇒ **gradient check** ;
garantie (borne, privacy, reproductibilité) ⇒ **test d'oracle/soundness** ;
déterminisme préservé (PCG seedé, ordre fixe) ; 8 gates verts.
