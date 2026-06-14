# SciRust — Plan de croissance ambitieux

> Document stratégique. La feuille de route opérationnelle détaillée vit dans
> [`INDUSTRIAL_ROADMAP.md`](INDUSTRIAL_ROADMAP.md) ; ce document donne la
> **vision**, les **fondamentaux non négociables**, et un **phasage ambitieux**.
> La feuille de route **« recherche → fonctions »** (papers réels traduits en
> fonctions concrètes, avec statut) vit dans
> [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md).

## 1. Vision : l'IA certifiable

SciRust ne cherche **pas** à concurrencer PyTorch sur les TFLOPS. Sa proposition
de valeur, défendable et unique, est de **posséder le créneau de l'IA
certifiable, reproductible et auditable** :

- **déterminisme bit-exact mesuré** (inférence *et* entraînement multi-thread),
- **auditabilité totale** : pur Rust, zéro FFI, `Cargo.lock` + SBOM CycloneDX +
  `cargo deny`,
- **embarqué / edge** : quantization int8 bit-exacte, `no_std` partiel,
- **traçabilité forensique** : certificats de preuve d'inférence (SRT1).

Clients visés : secteurs régulés (finance, santé, sûreté, défense), edge/embarqué,
recherche reproductible, audit/conformité IA.

## 2. Fondamentaux non négociables

Toute croissance **doit** respecter ces invariants (ce sont eux qui font la
valeur du projet) :

1. **100 % du code sous `*/src/` est câblé et testé.** Sinon → `archive/`.
2. **Déterminisme.** Tout aléa via `PcgEngine` seedé ; même graine ⇒ sortie
   bit-identique. Les réductions parallèles sont à ordre fixe.
3. **Pur Rust, zéro FFI.** Auditable de bout en bout.
4. **Pas de sur-promesse.** Chaque capacité revendiquée est adossée à un test.
   *Pas de claim sans CI* (ex. : pas de claim GPU sans runner GPU).
5. **8 gates verts** : `fmt`, `clippy -D warnings`, build/test (nightly **et**
   stable), cross-check aarch64, `cargo deny`, `rustdoc` 0 warning.
6. **Honnêteté documentaire.** Le README/CHANGELOG reflètent l'état mesuré,
   jamais une aspiration.

## 3. Chantiers (objectif · acquis · jalons)

### A. Cœur tenseur N-D unifié — *le verrou architectural*
- **Objectif** : une tape autograd N-D dont la 2D devient un cas particulier
  (shape inference au-delà de `rows/cols`).
- **Acquis** : primitives de forme (`broadcast_shape`/`matmul_shape`/
  `broadcast_to`) ; autograd N-D MVP (`autodiff::nd` : add/sub/mul broadcastés,
  matmul 2D, **bmm batché**, relu, sum) — **gradient-checké**.
- **Jalons** : softmax/layernorm/transpose-axes/reductions N-D · `nd::Linear`
  puis `nd::Attention` · migration progressive des couches · à terme la 2D = un
  alias.
- **Fondamental** : chaque op **gradient-checkée** ; déterminisme préservé.

### B. Pile LLM — inférence & serving
- **Objectif** : un chemin LLM crédible, déterministe, embarquable.
- **Acquis** : attention/flash/MoE/RoPE · **KV-cache O(n) prouvé équivalent** au
  recalcul complet · **BPE déterministe** · `generate_ids` découplé du tokenizer.
- **Jalons** : **sampling seedé** (température/top-k/top-p → déterministe) ·
  tokenizer BPE-bytes de production · KV-cache câblé dans un `generate` public ·
  batching · **int8 pour l'inférence LLM** · un petit modèle **entraîné** (preuve
  bout-en-bout).
- **Fondamental** : sampling **seedé** (reproductible) ; int8 bit-exact.

### C. GPU mûr — wgpu, opt-in
- **Objectif** : accélération portable, sans trahir le déterminisme par défaut.
- **Acquis** : GEMM/Conv2d/elementwise wgpu · **résidence VRAM** (couche entière)
  · oracle CPU · testé sur Vulkan logiciel (lavapipe).
- **Jalons** : op-set complet GPU (softmax, layernorm, reductions) · résidence
  **transparente** dans la tape (`DeviceTensor` matérialisé paresseusement) ·
  **runner GPU matériel en CI** (claim de perf **seulement** alors) · fusion de
  kernels.
- **Fondamental** : oracle CPU bit-tolérant ; **aucun claim de perf sans runner**.

### D. Interopérabilité & écosystème
- **Objectif** : charger/exporter des modèles ; un « model zoo » reproductible.
- **Acquis** : ONNX export (template) + **import des poids** (round-trip bit-exact).
- **Jalons** : **ONNX protobuf réel** (modèles externes) · export de graphe fidèle
  par couche · `safetensors` · model zoo (poids + manifeste + **certificat**).
- **Fondamental** : import validé par **round-trip** ; provenance via SBOM.

### E. Entraînement certifié & distribué
- **Objectif** : étendre la garantie bit-exacte au distribué.
- **Acquis** : data-parallel **déterminisme certifié** (1/2/4/8 threads
  bit-identique) ; boucle SGD multi-pas invariante.
- **Jalons** : **multi-nœuds** à réduction à arbre fixe (déterminisme
  inter-machines) · checkpointing déterministe · **« preuve d'entraînement »**
  (certificat reproductible d'une run).
- **Fondamental** : réductions à **ordre fixe** ; certificats.

### F. Analyse de code (SOM) — précision rustc
- **Objectif** : passer de l'oracle `syn` conservateur (mode rapide) à une
  précision NLL/types résolus (mode précis).
- **Acquis** : oracle `syn` · `scirust-rustc-driver` recompile + visible en CI.
- **Jalons** : passe **MIR** ownership/NLL → format de rapport SOM · intégration
  linter (SARIF déjà livré).

### G. Outillage & confiance
- **Acquis** : CLI (40 commandes) · SBOM CycloneDX · automatisation de release ·
  certificats de preuve · `cargo deny`.
- **Jalons** : **protection de branche** · fuzzing des parsers · couverture
  mesurée · benchmarks **reproductibles** · docs exhaustives.

## 4. Phasage

| Horizon | Livrables visés |
|---|---|
| **Court terme** (semaines) | sampling seedé + KV-cache dans un `generate` public · softmax/layernorm N-D · ONNX import élargi · protection de branche + release v0.14 |
| **Moyen terme** (mois) | `nd::Linear`/`nd::Attention` + migration · lavapipe → runner GPU + résidence transparente · entraînement multi-nœuds déterministe · passe MIR SOM |
| **Long terme** | tape N-D unifiée (2D = cas particulier) · model zoo certifié · « IA certifiable » comme produit (audit/conformité) |

**État (volet 29)** : court terme *sampling seedé + KV-cache* et *softmax/
layernorm N-D* = **faits** ; moyen terme `nd::Linear`/`nd::Attention` = **faits**
et déjà dépassés — la tape N-D porte désormais un **LM décodeur causal complet**
(`nn::nd_decoder::NdDecoderLM` : embeddings tok/pos, blocs causals, tête lm,
cross-entropy) **entraînable de bout en bout** par un **optimiseur Adam N-D
déterministe** (`nn::nd_optim::NdAdam` + `parameters()` sur toutes les couches).
Toutes les ops N-D (dont `gather`, `cross_entropy`, attention causale) sont
gradient-checkées ; le LM sur-apprend et reprédit une séquence exactement.

## 5. Métrique de succès

Pas les TFLOPS — mais le **nombre de propriétés certifiables testées** :
déterminisme (inférence + entraînement), reproductibilité bout-en-bout,
auditabilité (SBOM, zéro FFI), bit-exactitude int8, certificats. Un **tableau de
bord de garanties** plutôt qu'un benchmark de débit.

## 6. Comment contribuer sans casser les fondamentaux

- Un nouvel op autograd ? → **gradient check** obligatoire.
- Une nouvelle capacité « device » (GPU/CUDA) ? → renvoyer `Unavailable` tant
  qu'il n'y a pas de runner ; **jamais** de résultat fabriqué.
- Un parseur / format ? → test de **round-trip**.
- Du parallélisme ? → réduction à **ordre fixe** + test d'invariance au nombre
  de threads.
- Toujours : 8 gates verts, et le README dit la **vérité mesurée**.
