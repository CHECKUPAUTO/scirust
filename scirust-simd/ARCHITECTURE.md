# Architecture — `scirust-simd`

Ce document cartographie le crate `scirust-simd` : un **socle BLAS + Transformer
portable**, du datacenter x86_64 à l'embarqué ARM, construit autour d'une seule
idée directrice — **un binaire, sélection du backend à l'exécution, repli
scalaire garanti**.

Il est le fruit d'un chantier incrémental (≈ 20 PR) partant de simples kernels
`add`/`mul` pour aboutir à un pipeline d'inférence **et** d'entraînement complet.

---

## 1. Le fil directeur : dispatch runtime + portabilité

Toute la performance repose sur une **couche d'abstraction unique** qui choisit,
au premier appel, le meilleur jeu d'instructions disponible sur le CPU courant,
puis le met en cache :

```
                 detect_backend()  (OnceLock, coût amorti = 1 load atomique)
                        │
     ┌──────────┬───────┴────┬──────────┬───────────┐
   AVX-512    AVX2/FMA      SSE2       NEON        Scalaire
  (x86_64)   (x86_64)     (x86_64)   (aarch64)    (partout)
```

- **x86_64** : `AVX-512F` (+ `VNNI`/`BW` selon les kernels) → `AVX2+FMA` →
  `SSE2`. Détection via `std::is_x86_feature_detected!`.
- **aarch64** : `NEON` (baseline ARMv8), plus une amorce `SVE`.
- **Partout ailleurs** : chemin scalaire, toujours correct.

Conséquence : **le même code source** tourne du serveur AVX-512 à un Jetson /
Raspberry Pi / Rockchip RK3588, sans recompilation conditionnelle par le
développeur. Le repli scalaire est la **référence de correction** contre
laquelle tous les kernels vectoriels sont testés.

Modules concernés : [`dispatch`](src/dispatch.rs), [`matrix`](src/matrix/),
[`portable`](src/portable.rs).

---

## 2. Carte des couches

Du bas (silicium) vers le haut (application) :

| Couche | Module(s) | Contenu |
|---|---|---|
| **Dispatch / backends** | `dispatch`, `matrix`, `portable` | Détection CPU, trait `SimdBackend` (saxpy/sdot/…), backends AVX-512/AVX2/SSE2/NEON/scalaire |
| **BLAS — GEMM** | `gemm` | SGEMM (`f32`) & DGEMM (`f64`) tuilés/packés, multi-thread, GEMM fusionné `act(A·B+b)` |
| **Activations** | `activations` | `exp` vectorisée (range-reduction + `scalef`) → `sigmoid`/`tanh`/`GELU`/`SiLU` |
| **Quantification** | `x86_ext` | dot int8 `u8·i8→i32` (VNNI), bf16 mixed-precision, masques `k`, NT-stores, prefetch |
| **Attention** | `attention`, `kv_cache` | naïve, **flash** (softmax en ligne), **causale**, **multi-tête**, **cache KV** |
| **Normalisations** | `norm` | RMSNorm, LayerNorm (vectorisées), RoPE |
| **Assemblage** | `transformer`, `model` | Bloc décodeur pre-norm (prefill **+** decode), modèle multi-couche + génération |
| **Entraînement** | `grad` | Backward de tous les noyaux, validés par **gradcheck** |
| **Application** | `scirust-learning::simd_nn` | `DenseLayer`/`Mlp` entraînables, optimiseur **AdamW** |

---

## 3. BLAS : le GEMM, cœur de la performance

Le produit matriciel irrigue tout le reste (projections, FFN, backward). Trois
propriétés le rendent rapide :

1. **Blocking cache** (`MC`/`KC`/`NC`) façon BLIS : les panneaux travaillés
   tiennent en L2/L1.
2. **Packing explicite** de `A` et `B` en buffers contigus → le micro-kernel les
   lit en **stride unitaire** (prefetch matériel optimal).
3. **Micro-kernel registre-bloqué 8×16** : 8 accumulateurs `zmm` gardés en
   registres sur toute la dimension `KC` ; bords gérés par masque `k`.
4. **Parallélisme** : découpe de la dimension `M` en blocs de lignes disjoints
   via `std::thread::scope` (aucune dépendance externe).

Le **même GEMM packé est porté sur NEON** (aarch64) : structure identique
(blocking `MC_N`/`KC_N`/`NC_N`, packing de `A`/`B`), tuile registre `8×8`
(16 accumulateurs `float32x4_t`, épilogue par tampon car NEON n'a pas de store
masqué). Conséquence : **tout le stack — Transformer et entraînement — passe du
scalaire au vectorisé sur l'embarqué ARM** (Jetson, Raspberry Pi, RK3588), et
non plus seulement les primitives `saxpy`/`sdot`. Le noyau est validé
bit-à-tolérance contre le scalaire **sous `qemu-aarch64`** (exécution réelle,
pas seulement compilation).

### Perfs mesurées (machine AVX-512, 4 cœurs)

| Kernel | Débit | Gain vs naïf |
|---|---:|---:|
| SGEMM 1024³, 1 thread | 56.8 GFLOP/s | ~84× |
| SGEMM 1024³, 4 threads | 110 GFLOP/s | ~163× |
| DGEMM 1024³, 4 threads | 127 GFLOP/s | — |
| Couche dense fusionnée 4096×1024×1024 (ReLU) | 53.9 GFLOP/s | ~86× |

Le **GEMM fusionné** (`sgemm_bias_act`) calcule `act(α·A·B + biais)` : `A·B` par
le GEMM tuilé (n'importe quel `k`), puis un épilogue biais+activation vectorisé
en un seul passage `O(m·n)`.

Voir [`gemm`](src/gemm.rs) et le benchmark [`examples/bench.rs`](examples/bench.rs).

---

## 4. Pipeline Transformer

Toutes les briques d'un bloc décodeur, chaînables :

```
RMSNorm → Q,K,V = proj(·)      (GEMM tuilé)
        → RoPE(Q,K)            (par tête)
        → Attention causale multi-tête
        → + proj·Wo  (résidu)
RMSNorm → FFN : SiLU(·W₁+b₁)·W₂ (GEMM fusionné + GEMM)
        → + (résidu)
```

- **Attention** ([`attention`](src/attention.rs)) : version naïve, **flash**
  (softmax en ligne, mémoire `O(d)` par requête), **causale** (triangle, ~2× moins
  de travail), **multi-tête**.
- **Cache KV** ([`kv_cache`](src/kv_cache.rs)) : décodage autoregressif
  incrémental, `O(t·d)` par token au lieu de recalculer le préfixe.
- **Bloc & modèle** ([`transformer`](src/transformer.rs),
  [`model`](src/model.rs)) : deux régimes — *prefill* (séquence entière) et
  *decode* (token par token via cache). Un invariant clé garantit leur
  cohérence : **`prefill ≡ decode`** ligne par ligne, propagé sur toute la pile.

---

## 5. Entraînement

[`grad`](src/grad.rs) fournit la **rétropropagation** de tous les noyaux :

- `linear_backward` (réutilise le GEMM tuilé), `relu`/`silu`/`gelu_backward`,
  `rmsnorm`/`layernorm_backward`, `softmax_backward`, et
  **`attention_backward`** (chaîne `dV = Pᵀ·dO`, `dScores = softmax'`, `dQ`, `dK`).

Côté application, [`scirust-learning::simd_nn`](../scirust-learning/src/simd_nn.rs) :
`DenseLayer` et `Mlp` entraînables (forward fusionné + backward chaîné), et
l'optimiseur **`AdamW`** (moments, correction de biais, weight decay découplé).

---

## 6. Philosophie de test

Chaque affirmation est vérifiée mécaniquement :

- **Correction vectorielle** : chaque kernel SIMD est comparé au repli scalaire
  (souvent bit-à-bit ou à tolérance serrée), sur toutes les longueurs (y compris
  les épilogues masqués `1..15`).
- **Gradients** : tous les backward sont validés par **différences finies
  centrées** (gradcheck) contre un forward de référence indépendant.
- **Équivalences structurelles** : `prefill ≡ decode` (bloc et pile),
  `incrémental ≡ batch` (cache KV), `AdamW < SGD` à budget égal.
- **Portabilité** : le workspace compile sous
  `RUSTFLAGS="-D warnings" cargo check --target aarch64-unknown-linux-gnu`.

---

## 7. Index des modules

| Module | Rôle |
|---|---|
| [`dispatch`](src/dispatch.rs) | Détection CPU + backends arch-spécifiques |
| [`matrix`](src/matrix/) | Trait `SimdBackend`, vues matricielles |
| [`gemm`](src/gemm.rs) | SGEMM/DGEMM tuilés, parallèles, fusionnés |
| [`activations`](src/activations.rs) | `exp`/`sigmoid`/`tanh`/`GELU`/`SiLU` vectorisés |
| [`x86_ext`](src/x86_ext.rs) | VNNI int8, bf16, masques `k`, NT-stores, prefetch |
| [`attention`](src/attention.rs) | Attention naïve/flash/causale/multi-tête |
| [`kv_cache`](src/kv_cache.rs) | Cache KV, décodage incrémental |
| [`norm`](src/norm.rs) | RMSNorm, LayerNorm, RoPE |
| [`transformer`](src/transformer.rs) | Bloc décodeur (prefill + decode) |
| [`model`](src/model.rs) | Modèle multi-bloc + génération |
| [`grad`](src/grad.rs) | Backward de tous les noyaux (gradcheck) |
| [`complex`](src/complex.rs) | Arithmétique complexe SIMD |

---

*Ce document est vivant : il accompagne l'évolution du crate. Le fil rouge reste
constant — maîtrise fine du matériel x86_64 **et** couverture de toute la grille
de plateformes, derrière une abstraction unique.*
