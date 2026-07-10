# Audit de couverture — RepDL (Microsoft) vs SciRust

> Date : 2026-07-10 · Branche : `claude/repdl-scirust-audit-81gcb8`
> Objet : vérifier que SciRust couvre les fonctionnalités de
> [microsoft/RepDL](https://github.com/microsoft/RepDL) (bibliothèque
> semi-concurrente), fermer les écarts fermables, et garantir **zéro risque de
> copyright** dans la démarche.

---

## 1. Synthèse exécutive

**Verdict : couverture fonctionnelle quasi complète, dans un régime de
garanties différent et documenté.** Sur les 23 éléments d'API publique de
RepDL, SciRust en couvrait déjà 18 avant cet audit (opérations, couches,
gradients, optimiseur, exemples entraînables). Trois écarts réels et
fermables sont **fermés par cette PR** (AMSGrad, hachage SHA-256 de
tenseurs/paramètres, exp/ln par promotion `f64`). Deux éléments sont
**non applicables par conception** (conversion de modules PyTorch) ou
**couverts par composition** (réductions 4D).

**Risque copyright : nul.**
- Aucun code de RepDL n'existe dans ce dépôt (§3) ; les seules occurrences de
  « RepDL » sont des citations documentaires (positionnement scientifique,
  volet 108).
- Le présent audit a été mené sur **spécification uniquement** : surface
  d'API, README et résumé arXiv ; les trois implémentations ajoutées dérivent
  d'algorithmes publiés (Reddi et al. 2018 ; FIPS 180-4 via la crate `sha2`
  déjà en dépendance ; promotion double — technique folklorique), pas du code
  de RepDL.

Le seul axe où RepDL reste objectivement plus fort — reproductibilité f32
**inter-plates-formes** par arrondi correct — était déjà identifié et acté
comme travail futur dans `paper/RELATED_WORK.md` (volet 108). Cette PR y fait
un pas honnête (§6.3) sans sur-promettre.

---

## 2. Fiche RepDL (état au 2026-07-10)

| Champ | Valeur |
|---|---|
| Dépôt | github.com/microsoft/RepDL (≈13 commits, projet de recherche) |
| Licence | **MIT** |
| Référence | Xie, Zhang & Chen, *RepDL: Bit-level Reproducible Deep Learning Training and Inference*, arXiv:2510.09180 (2025) |
| Nature | Surcouche **PyTorch** (Python) + backends C++ (OpenMP) et CUDA |
| Promesse | Résultats **bit-identiques inter-plates-formes** (entraînement et inférence), f32 uniquement |
| Techniques | (a) ordre des opérations flottantes figé (sommations séquentielles, GEMM à ordre d'accumulation fixe, `fmaf`) ; (b) évitement des instructions non-IEEE-754 ; (c) fonctions mathématiques « correctement arrondies » par **promotion en double** (`exp`, `log`, `sqrt` calculés en f64 puis arrondis en f32) |
| Limites revendiquées | « Only a subset of functions and modules is available » ; pas de basse précision (bf16/int8) ; rapport sans évaluation chiffrée |

Surface d'API publique complète :

- `repdl.ops` : `mm` (transposes optionnelles), `div`, `sqrt`, `softmax`,
  `sum1d`, `sum2d_dim0`, `sum2d_dim1`, `sum4d_dim023`, `conv2d`,
  `conv2d_grad_input`, `conv2d_grad_kernel`, `cross_entropy`
- `repdl.func` (avec backward autograd) : `expand_as`, `mean1d`,
  `mean2d_dim0`, `mean4d_dim023`
- `repdl.nn` : `Linear`, `Conv2d`, `BatchNorm1d`, `BatchNorm2d`,
  `CrossEntropyLoss` (+ `nn.functional` correspondants)
- `repdl.optim` : `Adam` (option AMSGrad)
- `repdl.from_torch_module(m)` : conversion récursive d'un module PyTorch
- `repdl.utils` : `get_hash` / `print_hash` (SHA-256 d'un tenseur ou des
  paramètres d'un module — l'outil de vérification de la reproductibilité)

## 3. Méthodologie et propriété intellectuelle

**Sources consultées** : page du dépôt, README, arborescence, inventaires
d'API (signatures + sémantique en prose), résumé arXiv. Les algorithmes des
backends ont été caractérisés **en prose** (ordre d'accumulation, promotion
double) pour situer la classe de garantie — aucun code n'a été recopié,
traduit ni adapté.

**Constat sur le dépôt SciRust** (recherche exhaustive) :

- `grep -ri "repdl|sum2d_dim|sum4d_dim|mean4d_dim|from_torch_module"` sur tout
  l'arbre : **0 correspondance dans le code source**. Les 7 fichiers touchés
  sont tous documentaires (`README.md`, `CHANGELOG.md`, `LIVESTATE.md`,
  `paper/RELATED_WORK.md`, `paper/PAPER_PLAN.md`, `docs/INDUSTRIAL_ROADMAP.md`,
  `docs/DOSSIER_FINANCEURS.md`) et relèvent de la **citation scientifique**
  (autorisée et souhaitable — c'est le travail d'honnêteté du volet 108).
- Les implémentations SciRust préexistantes (GEMM, conv2d im2col/col2im,
  sommation Demmel–Nguyen/Shewchuk, Kahan, PCG…) sont architecturalement
  différentes de RepDL et antérieures à cet audit : **aucun risque d'œuvre
  dérivée**.

**Positions de licence** : RepDL est sous MIT — une réutilisation de code
serait *légale* moyennant conservation de la notice de copyright Microsoft,
mais elle créerait une obligation d'attribution dans un dépôt PolyForm
Noncommercial et un risque de confusion. **Politique retenue (zéro risque) :
ne jamais copier ni traduire de code RepDL ; ne réimplémenter que depuis des
specs/papiers publics.** Cette PR s'y conforme ; toute contribution future
touchant la reproductibilité devrait suivre la même règle.

## 4. Matrice de couverture, élément par élément

Statuts : ✅ couvert · ✅➕ couvert par composition · 🆕 fermé par cette PR ·
Ⓝ non applicable par conception.

| API RepDL | Équivalent SciRust | Statut | Preuve |
|---|---|---|---|
| `ops.mm` (transA/transB) | `Var::matmul` + `Op::MatMul` ; GEMM à drapeaux de transposition (interne) | ✅ | `scirust-core/src/autodiff/reverse.rs:31-43,412,868-892,1253` |
| `ops.div` | `Op::Div` / `Op::DivBroadcast` (autograd) — la division IEEE-754 est correctement arrondie par le standard | ✅ | `reverse.rs:606,610,1065,1202` |
| `ops.sqrt` | `Op::Sqrt` (autograd) — `sqrt` IEEE-754 correctement arrondie par le standard (RepDL passe par f64, résultat identique) | ✅ | `reverse.rs:620,1410` |
| `ops.softmax` | `Op::Softmax` (+ `LogSoftmax`) 2-D, et softmax dernier axe sur la tape N-D | ✅ | `reverse.rs:383,649,1648` ; `nd.rs:190` |
| `ops.sum1d` | `Op::Sum` | ✅ | `reverse.rs:639,1534` |
| `ops.sum2d_dim0/dim1` | `Op::SumAxis(axis ∈ {0,1})` | ✅ | `reverse.rs:640,1539` |
| `ops.sum4d_dim023` | composition `transpose().reshape([C, N·H·W])` + réduction axe 1 (usage réel : stats par canal de BatchNorm2d) | ✅➕ | `scirust-core/src/nn/batch_norm_2d.rs:86-113` |
| `ops.conv2d` | `Op::Conv2dForward` (+ `ConvTranspose2d`) | ✅ | `reverse.rs:724-736` ; `nn/conv2d.rs:120` |
| `ops.conv2d_grad_input` | backward `Conv2dForward` : `dcol = Wᵀ·dout` puis `col2im` | ✅ | `reverse.rs:2188-2193` |
| `ops.conv2d_grad_kernel` | backward `Conv2dForward` : `dw = dout·colᵀ` | ✅ | `reverse.rs:2184-2186` ; test `test_conv_grad.rs:11-102` |
| `ops.cross_entropy` | log-softmax stable + NLL (one-hot et indices) | ✅ | `nn/loss/cross_entropy.rs:27,63` |
| `func.expand_as` | `Op::Broadcast` (backward = réduction) sur le régime 2-D de la tape | ✅➕ | `reverse.rs:644,1620,3142` |
| `func.mean1d` | `Var::mean` / `MeanAxis` | ✅ | `reverse.rs:641,1544` |
| `func.mean2d_dim0` | `Var::mean_axis(0)` | ✅ | `reverse.rs:3156` |
| `func.mean4d_dim023` | composition (cf. `sum4d_dim023`) | ✅➕ | `batch_norm_2d.rs:101-103` |
| `nn.Linear` | `nn::Linear` (autograd, state_dict) | ✅ | `nn/linear.rs:69-137` |
| `nn.Conv2d` | `nn::Conv2d` | ✅ | `nn/conv2d.rs` |
| `nn.BatchNorm1d` | `nn::BatchNorm` (train/eval, running stats) | ✅ | `nn/batch_norm.rs:49-134` |
| `nn.BatchNorm2d` | `nn::BatchNorm2d` (train/eval, running stats) | ✅ | `nn/batch_norm_2d.rs:54-120` |
| `nn.CrossEntropyLoss` | `nn::loss::CrossEntropyLoss` (gradient vérifié = softmax − cible) | ✅ | `nn/loss/cross_entropy.rs:179-207` |
| `optim.Adam` | `autodiff::optim::Adam` (betas, eps, weight decay, bias correction) + `NdAdam`/AdamW | ✅ | `autodiff/optim.rs` ; `nn/nd_optim.rs` |
| `optim.Adam(amsgrad=True)` | **ajouté** : `Adam::with_amsgrad()` (max historique du 2ᵉ moment, bias-corrigé) + 2 tests (oracle de convergence, propriété anti-pic) | 🆕 | `autodiff/optim.rs` (cette PR) |
| `utils.get_hash`/`print_hash` | **ajouté** : `scirust_runtime::hash::{sha256_hex_f32, sha256_hex_tensor, sha256_hex_state_dict}` (encodage LE indépendant de la plate-forme, clés triées) + 5 tests | 🆕 | `scirust-runtime/src/hash.rs` (cette PR) |
| `from_torch_module` | Ⓝ SciRust n'est pas une surcouche PyTorch — équivalents : lecteur **safetensors** (poids HF/PyTorch), format SRT1 déterministe, export/import ONNX-JSON, `state_dict`/`load_state_dict` par couche | Ⓝ | `scirust-core/src/io/safetensors.rs:138` ; `scirust-runtime/src/lib.rs` ; `scirust-onnx/src/lib.rs:295` |
| Transcendantales par promotion f64 (`exp2d`, `log1d`) | **ajouté** : `reproducible::{exp_via_f64, ln_via_f64}` — même classe de technique, documentation honnête de la classe de garantie | 🆕 | `scirust-core/src/reproducible.rs` (cette PR) |

Les exemples `mnist_classifier` et `cifar10_classifier` entraînent réellement
des modèles avec ces briques (boucles forward/backward/step complètes,
critère de précision > 90 % sur MNIST) — l'équivalent du
`examples/mnist_training.py` de RepDL.

## 5. Axe déterminisme — garanties comparées

C'est ici que les deux projets diffèrent réellement (constat conforme au
positionnement déjà acté dans `paper/RELATED_WORK.md`) :

| Garantie | RepDL | SciRust |
|---|---|---|
| f32 bit-exact **inter-plates-formes** (CPU↔GPU, x86↔ARM) | ✅ (sa raison d'être ; non évalué chiffré dans son rapport) | ❌ assumé hors périmètre pour la voie f32 (`scirust-runtime/README.md:34-35`) ; **travail futur acté** |
| f32 bit-exact **intra-architecture**, invariant au nombre de threads | (implicite) | ✅ testé : fingerprint identique sur 1/2/4/8/16/64 threads, 0 divergence sur 5 120 logits (`tests/fingerprint_thread_invariance.rs`, rapport §6.2) |
| Basse précision déterministe (int8/int16/fixed-point) **cross-platform par construction** | ❌ hors périmètre | ✅ GEMM int8/int16/Q16/Q32/Zq, NEON == scalaire bit-exact (`quantization.rs:1959`), GPU == CPU bit-exact voies entières (`scirust-gpu/src/deterministic_gpu.rs`) |
| Sommation reproductible indépendante de l'ordre | ordre séquentiel figé (dépend du parcours) | ✅ plus fort : somme **correctement arrondie du multiensemble** (Demmel–Nguyen + Shewchuk), bit-identique sous permutation (`reproducible.rs`) |
| Réductions parallèles à ordre fixe | OpenMP, ordre figé | ✅ agrégation en ordre de worker/rang, testée bit-exacte (`data_parallel.rs`, `distributed.rs`) |
| Vérification par empreinte (hash) | `get_hash` SHA-256 | 🆕 `runtime::hash` (cette PR) + FNV-1a existant + chaîne d'attestation SHA-256 (`attest.rs`) |
| Vérifiabilité au-delà du hash | ❌ | ✅ Freivalds/GF(p) (`vinfer.rs`), enveloppe d'erreur DiFR (`difr.rs`) — sans équivalent RepDL |
| TCB | PyTorch + libtorch (millions de lignes C++) | 100 % Rust auditable, zéro FFI dans le chemin de calcul |

Points de vigilance connus et inchangés (déjà tracés ailleurs) : le job CI
aarch64 ne fait que `cargo check` (l'exécution ARM réelle vit sur Jetson,
hors CI) ; les réductions SIMD/GPU **flottantes** restent égales-en-tolérance
au scalaire, pas bit-exactes — seule la voie entière l'est.

## 6. Écarts fermés par cette PR

### 6.1 `Adam::with_amsgrad()` — parité `optim.Adam(amsgrad=True)`
Buffer `v_max` (max historique du 2ᵉ moment, bias-corrigé comme `v`),
implémenté d'après Reddi, Kale & Kumar (ICLR 2018). Deux tests : oracle de
convergence sur quadratique, et propriété définitoire (après un pic de
gradient, les pas AMSGrad restent < 10 % des pas Adam).

### 6.2 `scirust_runtime::hash` — parité `utils.get_hash`/`print_hash`
Empreintes SHA-256 hex de slices f32, de tenseurs (forme incluse) et de
`state_dict` complets (clés triées ⇒ indépendant de l'ordre d'insertion).
Encodage little-endian des bits IEEE-754 ⇒ empreinte identique sur toute
plate-forme pour des données bit-identiques. C'est l'outil qui permet à un
utilisateur de *constater* la reproductibilité (deux machines, même hash).

### 6.3 `reproducible::{exp_via_f64, ln_via_f64}` — parité `exp2d`/`log1d`
Même classe de technique que RepDL (promotion en double). La documentation
énonce la classe de garantie sans sur-promettre : fidèlement arrondi (et
correctement arrondi hors cas de dilemme du fabricant de tables), déterministe
sur un binaire donné, identité inter-plates-formes très probable mais non
prouvée — les transcendantales correctement arrondies *prouvées* en Rust pur
restent le travail futur acté au volet 108.

## 7. Écarts non retenus (justifiés)

- **`sum4d_dim023` / `mean4d_dim023` en op dédiée** : le besoin réel (stats
  par canal de BatchNorm2d) est couvert par composition
  (`batch_norm_2d.rs:86-113`). Une op fusionnée serait une optimisation de
  performance, pas un manque fonctionnel.
- **`expand_as` N-D général** : le régime 2-D de la tape est couvert par
  `Op::Broadcast` ; la tape N-D n'en a pas eu besoin à ce jour.
- **`from_torch_module`** : non applicable — SciRust est un framework
  autonome, pas une surcouche PyTorch ; l'import de poids externes passe par
  safetensors.
- **AMSGrad sur `NdAdam`** : non ajouté (le `NdAdam` vise AdamW pour les
  décodeurs N-D) ; à faire si un besoin transformer l'exige.

## 8. Recommandations

1. **P1 — politique IP écrite** : consigner (fait ici, §3) la règle « aucune
   copie/traduction de code RepDL ; réimplémentation sur specs publiques
   uniquement » pour tout travail futur sur la reproductibilité.
2. **P2 — brancher les nouvelles briques** : publier le hash
   `sha256_hex_state_dict` dans les pièces d'audit existantes (protocole de
   test, rapports Jetson) à côté des fingerprints FNV ; envisager
   `exp_via_f64` dans le softmax de la tape si la portabilité f32 devient un
   objectif produit.
3. **P2 — CI ARM native** : le jour où un runner aarch64 est disponible,
   exécuter (pas seulement `cargo check`) les tests d'invariance et comparer
   les empreintes x86/ARM des voies entières — transformerait « bit-exact
   cross-platform par construction » en « …et testé en CI ».
