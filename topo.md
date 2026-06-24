# Topo honnête sur SciRust

> Synthèse rédigée le 2026-06-23 à partir d'une exploration réelle du dépôt
> (`/root/scirust`, branche master, 445 commits, 75 crates, ~166 k lignes Rust).

## Ce qu'est devenu SciRust

Un framework de deep learning et calcul scientifique en **Rust pur, zéro FFI**,
fondé sur une thèse forte : **déterminisme bit-exact, certifiabilité,
auditabilité**. L'ambition n'est pas de concurrencer PyTorch en perf, mais
d'offrir un framework qu'on peut lire, modifier et prouver.

**Chiffres réels (mesurés dans le repo) :**

- ~166 000 lignes de Rust, **75 crates** workspace
- **625 tests** dans `scirust-core` seul, **1718+ tests** workspace (README)
- Roadmap recherche : **80/80 papers livrés** (Katz, CROWN, Mamba, FlashAttention,
  GPTQ, AWQ, BitNet, NF4, QuIP#, PagedAttention, Medusa, EAGLE, Reluplex, DeepPoly,
  CROWN-IBP, conformal CQR/APS/RAPS/RCPS/ACI/LtT, Sophia, Shampoo, GaLore, SAM,
  Prodigy, DoRA, YaRN, xLSTM, Hyena, S5, Mamba-2/SSD, FNO, DeepONet, KAN, PINN,
  Rényi-DP, watermark, vinfer/Freivalds, DiFR…)
- 445 commits, dernière activité 2026-06-23 (trader, NLP-advanced, CI/SBOM)

## Ce qui est solide (le cœur réel)

- `scirust-core` (29k lignes) : tensor 2D, autodiff reverse-mode sur tape, couches
  NN complètes (MLP/CNN/Transformer/LSTM/FlashAttn/ViT/GNN/TT-Linear), int8
  déterministe, DP-SGD, pruning, distributed, lazy graph
- Pile **LLM N-D** : LLaMA block (RMSNorm/SwiGLU/RoPE/GQA), Mamba, RWKV, RetNet,
  GLA, HGRN, xLSTM, S4/S5, DeltaNet, Mamba-2, Hyena, décodage spéculatif/Medusa/
  EAGLE/PagedAttention
- Pile **certification** complète : IBP→CROWN→DeepPoly→zonotopes→Lipschitz→
  smoothing→CROWN-IBP→BaB→MILP→Reluplex
- Pile **quantization** : int8, NF4, GPTQ, AWQ, SmoothQuant, BitNet b1.58, QuIP#
  (E8 lattice), AQLM, KVQuant, SpQR, SqueezeLLM, OmniQuant, LLM.int8, LoRA, DoRA
- `scirust-runtime` : inférence bit-exact + certificats (`proof`, `vinfer`,
  `difr`), attestation hash-chaînée
- `scirust-simd` : kernels AVX2/SSE2/NEON (un bug d'alignement non déterministe a
  été corrigé)
- GPU wgpu (WGSL GEMM sur lavapipe, dans le tape) ; CUDA archivé/non reproductible
- Verticales industrielles : signal, opcua, mqtt, pdm, mlops, func-safety,
  estimation, nav, water, ids, metrology, reliability, bms, hvac
- Extras : `scirust-rsi` (recursive self-improvement + Claude API),
  `scirust-trader`, simulateur quantique MPS, neuro-symbolic

## Ce qui est moins glorieux (lecture d'audit)

L'audit interne (`scirust_complete_audit_report.md`) le dit lui-même, honnêtement :

- **~1/3 des crates sont réellement développées et testées ; ~2/3 sont des
  prototypes ou squelettes** (`scirust-som`, `events-*`, `edge`, `embedded`,
  `bridge`, `macros` embryonnaires)
- `scirust-gpu` : trompeur — 67 lignes réellement câblées, le reste était du code
  mort hors module tree
- `scirust-autodiff` : doublon conceptuel de `core::autodiff`
- Le tensor est **2D row-major** ; le N-D vit dans
  `scirust-tensor-core::TensorND`, **non unifié** avec le core
- CUDA non reproductible aujourd'hui (pas de runner GPU hardware)
- Perf non compétitive vs Burn/candle/PyTorch (choix assumé : lisibilité > vitesse)
- Licence **PolyForm Noncommercial** : freine sérieusement l'adoption industrielle
- Beaucoup de surfaces déclarées mais peu de benchmarks comparatifs publiés

## Axes de développement proposés

### A. Consolidation & dette technique (priorité : robustesse)

1. **Unifier le tensor** : fusionner `core::Tensor` (2D) et `TensorND` en un seul
   type N-D ; supprimer `scirust-autodiff` (doublon)
2. **Nettoyer le code mort** et squelettes : soit terminer (`som`, `events-*`,
   `edge`, `embedded`), soit extraire dans un repo `scirust-labs` pour ne garder
   que ce qui compile et se teste
3. **Stabiliser l'API publique** : semver, documentation rustdoc complète,
   `#![deny(missing_docs)]`
4. **CI sur aarch64 + Jetson** (déjà testé ponctuellement) pour tenir la promesse
   « architecture-agnostic »

### B. Performance & GPU réels

5. **Runner GPU hardware** (CUDA ou wgpu sur GPU réel) : rendre le chemin `cuda`
   reproductible ou pivoter 100% sur wgpu portable
6. **im2col et activations sur GPU** (P2.2 du `GPU.md`) : finir la résidence VRAM
7. **Kernels SIMD matmul + fused ops** comparés vs Burn/candle (publier un
   benchmark honnête)
8. **Profiling** : un profiler de tape (timeline ops/mémoire) — pas d'outil
   équivalent en pur Rust aujourd'hui

### C. Adoption & écosystème

9. **License** : ajouter une option Apache-2.0/MIT pour usage commercial (ou
   dual-license comme Burn) — sinon le projet restera marginal
10. **Publication crates.io** : `scirust-core`, `scirust-simd`,
    `scirust-runtime` comme vraies crates versionnées
11. **Import modèles HuggingFace** : loader safetensors réel + tokenizer
    (BPE/SPM) pour faire tourner un LLM réel, pas juste un toy LM
12. **Exemples bout-en-bout** : un mini-GPT préentraînable sur CPU determinist,
    un MNIST SOTA reproductible documenté

### D. Nouvelles fonctionnalités différenciantes

13. **Debugger de tape** : visualisation du graphe autograd, inspection des
    gradients, points d'arrêt — le genre d'outil que PyTorch n'a pas nativement
    et qui colle à la thèse « lisible »
14. **Quantification end-to-end** : pipeline complet float→int4→runtime certifié
    sur un vrai LLM (piste SLHAv2 mentionnée dans le CHANGELOG)
15. **Distributed multi-nœuds** : all-reduce réel (aujourd'hui c'est intra-process)
16. **Quantique complexe** : étendre le simulateur MPS aux phases S/T/Rz
    (actuel = réelles)
17. **Federated learning déterministe** : angle certifiable/explicable naturel
    pour SciRust, peu exploré
18. **Conformal pour LLM** : ensemble de génération avec garantie de coverage
    (extension de la pile conformal au cas génératif)
19. **Auto-parallelisation du tape** : scheduling déterministe multi-thread avec
    garantie bit-exact (déjà commencé, à généraliser)
20. **Bridge Rust→PyTorch** ou **bindings Python** : exposer scirust comme
    backend vérifiable pour des notebooks (inverse du pont actuel
    Rust→Python/C)

## Verdict honnête

SciRust est un projet **réel et impressionnant pour un effort individuel/petite
équipe** : le cœur est solide, la roadmap recherche est bouclée à 80/80 avec une
discipline de test/oracle rare. C'est aussi un projet **sur-déclaré** : la
périphérie est en grande partie du squelette, et la perf/GPU réel reste à prouver.
Le principal frein à l'adoption n'est pas technique, c'est la **licence
Noncommercial** + l'absence de story « faire tourner un vrai modèle ». Les axes
les plus rentables à court terme sont donc : (a) nettoyage/unification,
(b) license + publication crates.io, (c) un chemin « run un LLM HuggingFace
réel ».