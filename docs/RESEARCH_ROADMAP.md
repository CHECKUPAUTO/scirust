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
| 1 | Gowal et al., *On the Effectiveness of Interval Bound Propagation* (2018) | `certify_ibp(input_box) -> output_box` : arithmétique d'intervalles propagée couche par couche ; **borne de sortie prouvée** | `nn::certified` | 🔨 | S |
| 2 | Zhang et al., *CROWN* (NeurIPS 2018) ; Wang et al., *β-CROWN* (NeurIPS 2021, arXiv:2103.06624) | bornes par relaxation linéaire (plus serrées qu'IBP) | `nn::certified` | 📋 | L |
| 3 | Demmel & Nguyen, *Algorithms for Efficient Reproducible Floating-Point Summation* (ACM TOMS 2020) | `reproducible_sum` / accumulateur « binned » : somme **bit-identique quel que soit l'ordre / le nombre de threads** | `tensor` | 🔨 | M |
| 4 | Katz et al., *Reluplex* (CAV 2017, arXiv:1702.01135) ; *Marabou* (CAV 2019) | `verify-net` : vérification **complète** (SMT) d'une propriété sur un petit réseau ReLU | `scirust-neuro-symbolic` + CLI | 📋 | XL |
| 5 | *DiFR: Inference Verification Despite Nondeterminism* (2025, arXiv:2511.20621) | vérifier une trace d'inférence malgré le non-déterminisme | `scirust_runtime::proofcli` | 📋 | L |

## Tier 2 — Pile LLM N-D (gains rapides, gradient-checkables)

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 6 | Zhang & Sennrich, *Root Mean Square Layer Normalization* (NeurIPS 2019) | `NdRmsNorm` (+ op `rmsnorm` de la tape) | `autodiff::nd`, `nn::nd_layers` | 🔨 | S |
| 7 | Shazeer, *GLU Variants Improve Transformer* (2020, arXiv:2002.05202) | `NdSwiGLU` (+ op `sigmoid`/SiLU) | `autodiff::nd`, `nn::nd_layers` | 🔨 | S |
| 8 | Su et al., *RoFormer / RoPE* (2021) | RoPE porté en N-D | `nn::nd_layers` | 📋 | M |
| 9 | Milakov & Gimelshein, *Online normalizer for softmax* (2018) ; Dao, *FlashAttention-2* (arXiv:2307.08691) | softmax une-passe + attention tuilée `O(seq)` mémoire, déterministe | `autodiff::nd` | 📋 | M |
| 10 | Leviathan et al., *Speculative Decoding* (ICML 2023) ; Chen et al., *Speculative Sampling* (2023) | `generate_speculative` : accélération **exacte** (préserve la distribution) | `nn::nd_decoder` | 📋 | M |
| 11 | Ainslie et al., *GQA* (2023) ; Shazeer, *MQA* (2019) | `num_kv_heads` dans l'attention N-D | `nn::nd_layers` | 📋 | M |

## Tier 3 — Optimiseurs

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 12 | Loshchilov & Hutter, *Decoupled Weight Decay (AdamW)* (2017, arXiv:1711.05101) | `weight_decay` découplé dans `AdamConfig` | `nn::nd_optim` | 🔨 | S |
| 13 | Chen et al., *Symbolic Discovery of Optimization Algorithms (Lion)* (2023) | `NdLion` (sign-based, mémoire moitié) | `nn::nd_optim` | 🔨 | S |
| 14 | Jordan et al., *Muon* (2024) | `NdMuon` (orthogonalisation Newton-Schulz) | `nn::nd_optim` | 📋 | M |

## Tier 4 — Quantification (thèse int8 bit-exact)

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 15 | Frantar et al., *GPTQ* (2022) ; Lin et al., *AWQ* (2023) ; Xiao et al., *SmoothQuant* (2022, arXiv:2211.10438) | PTQ déterministe W8A8 / W4, round-trip testé | `quantization` | 📋 | L |

## Tier 5 — Pont calcul scientifique (fusion unique : solveurs + autograd + symbolique)

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 16 | Chen et al., *Neural ODEs* (NeurIPS 2018, arXiv:1806.07366) | `NeuralODE` : solveur (rk4/dopri5) + tape N-D | `nn` + `scirust-solvers` | 📋 | M |
| 17 | Raissi, Perdikaris & Karniadakis, *PINNs* (J. Comp. Phys. 2019) | `pinn` : résidu de PDE dans la loss (besoin d'autodiff d'ordre 2) | `nn` + `symbolic` | 📋 | L |

## Tier 6 — Architectures alternatives & confiance

| # | Papier | Fonction scirust | Module | Statut | Effort |
|---|--------|------------------|--------|--------|--------|
| 18 | Gu & Dao, *Mamba* (2023, arXiv:2312.00752) | *selective scan* (récurrence déterministe) ; modèle séquence linéaire-temps | `nn` | 📋 | XL |
| 19 | Abadi et al., *Deep Learning with Differential Privacy (DP-SGD)* (2016) | `dp_sgd` : clipping par échantillon + bruit gaussien **seedé** (privé + déterministe + auditable) | `dp` + `nn::nd_optim` | 📋 | M |
| 20 | Frantar & Alistarh, *SparseGPT* (2023) ; Sun et al., *Wanda* (2023) ; Frankle & Carbin, *Lottery Ticket* (2019) | élagage one-shot déterministe | `pruning` | 📋 | M |

---

## Ordre d'attaque

**Lot 1 (cette session)** — les quick wins à plus fort fit, chacun testé + gates verts :
RMSNorm, SwiGLU (#6, #7) · AdamW, Lion (#12, #13) · sommation reproductible (#3)
· IBP certifié (#1).

**Ensuite** : GQA (#11) · décodage spéculatif (#10) · Neural ODE (#16) · DP-SGD
(#19) · online softmax (#9) · RoPE N-D (#8).

**Paris lourds** (planifiés, jalonnés) : CROWN (#2) · SMT/Marabou (#4) · GPTQ/AWQ
(#15) · Mamba (#18) · PINN (#17, après l'autodiff d'ordre 2) · Muon (#14) ·
pruning (#20) · DiFR (#5).

Chaque item respecte les fondamentaux : op autograd ⇒ **gradient check** ;
garantie (borne, privacy, reproductibilité) ⇒ **test d'oracle/soundness** ;
déterminisme préservé (PCG seedé, ordre fixe) ; 8 gates verts.
