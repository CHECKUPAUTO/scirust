# Étude empirique « dead guards » — gardes epsilon mortes dans les bases de code numériques publiques

Date de campagne : 2026-07-10. Outil : `epsilon-audit --mine` (crate
`scirust-sigma` v0.1.0, binaire std-only, parsing lexical multi-langage).
Revue manuelle : chaque candidat émis par l'outil a été relu en contexte dans
l'arbre cloné avant classification.

## 1. Question de recherche

La classe de bug « **garde epsilon morte** » — une constante de garde f32 si
petite que la garde ne protège pas — existe-t-elle dans des bases de code
numériques réelles, et à quelle prévalence ?

Deux mécanismes de mort sont détectés (détaillés dans
`scirust-sigma/src/mine.rs`) :

- **M1 (flush)** : littéral f32 avec `0 < |v| < 1.17549435e-38`
  (`f32::MIN_POSITIVE`). Un tel littéral est **sous-normal** : sous FTZ/DAZ
  (fast-math, drivers GPU, modes CPU) il est écrasé à `0` — `x.max(g)` devient
  `x.max(0)` et la garde n'existe plus.
- **M2 (inversion)** : littéral f32 avec `0 < |v| < 2.938736e-39`
  (= `1/f32::MAX`). Même sans FTZ : si `x.max(g)` vaut `g`, alors
  `1.0/(x.max(g))` déborde en `inf`. La plage M2 est incluse dans la plage
  M1 ; les deux sont classées séparément (M2 = mécanisme le plus fort).

## 2. Méthodologie exacte

1. **Clonage** : `git clone --depth 1 <url>` dans `/tmp/mining/` (clone
   sparse `--filter=blob:none --sparse` limité aux sous-répertoires indiqués
   pour les dépôts géants). SHA de commit enregistré pour chaque dépôt.
2. **Scan** : `epsilon-audit --mine /tmp/mining/<repo> --out reports/<repo>.md`
   (binaire compilé en release depuis ce commit du dépôt SciRust). Extensions
   scannées : `.rs`, `.c`, `.h`, `.cpp`, `.hpp`, `.cc`, `.cu`, `.cuh`, `.cl`,
   `.metal`, `.wgsl`, `.glsl`, `.comp`. Exclusions : répertoires `test*/`,
   `bench*/`, `benchmark*/`, `third_party/`, `3rdparty/`, `vendor/`,
   `external/`, artefacts (`target/`, `build/`), fichiers `*_test.*` /
   `test_*.*`.
3. **Typage f32 (heuristique lexicale documentée)** :
   - Rust : suffixe `f32` ou `f32` sur la ligne → CONFIRMED-F32 ; `f64` →
     hors périmètre ; sinon UNCERTAIN (jamais compté).
   - Famille C/CUDA/OpenCL : suffixe `f`/`F` → CONFIRMED-F32 ; littéral nu sur
     une ligne contenant `float` → PROBABLE-F32 ; sinon UNCERTAIN (littéral nu
     = `double` en C — jamais compté comme finding).
   - Shaders (WGSL/GLSL/Metal/compute) : flottants f32 par défaut →
     CONFIRMED-F32 (et les GPU flushent très couramment les sous-normaux).
   - La comparaison au seuil se fait sur la valeur **arrondie en f32**
     (sémantique de matérialisation : `1.17549435e-38` arrondit exactement à
     `f32::MIN_POSITIVE` → garde licite, non capturée).
4. **Fast-math** : grep des drapeaux `-ffast-math`, `use_fast_math`
   (couvre `--use_fast_math`/`-use_fast_math`), `-funsafe-math-optimizations`
   et `ftz` (insensible à la casse) dans les fichiers de build (CMakeLists,
   `*.cmake`, Makefile*, `*.mk`, build.rs, setup.py, meson.build, BUILD,
   `*.bzl`/`*.bazel`, `*.gn`/`*.gni`) → colonne « FTZ probable ».
5. **Revue manuelle obligatoire** : chaque candidat relu en contexte
   (fichier complet dans l'arbre cloné) et classé `CONFIRMED_DEAD_GUARD` /
   `BENIGN` / `UNCERTAIN`. Un candidat n'est CONFIRMED que si : typage f32
   établi **et** usage de garde établi (`.max(`, dénominateur, `fmaxf`,
   seuil protégeant division/log/sqrt) **et** mécanisme M1 ou M2 applicable.

Les rapports par dépôt (Markdown scellés par SHA-256) sont reproductibles
bit-à-bit à arbre identique ; l'outil ne modifie jamais les dépôts clonés.

## 3. Corpus — 22 dépôts scannés, 0 échec de clone

| Dépôt | SHA | Périmètre scanné | Fichiers | LOC | Candidats | Confirmés | Incertains | FTZ probable |
|---|---|---|---:|---:|---:|---:|---:|---|
| ggml-org/llama.cpp | `8f114a9b` | racine | 1 550 | 609 678 | 0 | 0 | 0 | oui (7) |
| ggml-org/ggml | `524f974b` | racine | 1 098 | 416 806 | 0 | 0 | 0 | oui (7) |
| huggingface/candle | `31f35b14` | racine | 772 | 247 523 | 0 | 0 | 0 | oui (2) |
| tracel-ai/burn | `105b0e9b` | racine | 1 142 | 287 987 | 0 | 0 | 0 | non |
| pytorch/pytorch | `3bda7431` | `aten/src`, `c10`, `caffe2/utils/math` | 2 599 | 709 503 | 0 | 0 | 0 | oui (5) |
| tensorflow/tensorflow | `bb8ff7dc` | `tensorflow/core/kernels` | 1 565 | 309 452 | 0 | 0 | 0 | oui (3, ftz) |
| microsoft/onnxruntime | `f4aa2b44` | `onnxruntime/core` | 2 718 | 763 846 | 0 | 0 | 0 | non |
| OpenMathLib/OpenBLAS | `7c991951` | `kernel`, `lapack` | 1 279 | 419 995 | 0 | 0 | 0 | oui (1) |
| libeigen/eigen (GitLab) | `26f009db` | `Eigen/src` | 409 | 183 137 | 0 | 0 | 0 | non |
| NVIDIA/cutlass | `e6233cba` | `include` | 785 | 674 100 | 0 | 0 | 0 | non |
| rust-ndarray/ndarray | `bd3ade99` | racine | 113 | 33 147 | 12 | 0 | 0 | non |
| dimforge/nalgebra | `3320ecca` | racine | 279 | 73 273 | 0 | 0 | 0 | non |
| sarah-quinones/faer-rs | `0539947f` | racine | 163 | 120 623 | 0 | 0 | 9 | non |
| sonos/tract | `26edc98e` | racine | 818 | 207 109 | 0 | 0 | 0 | oui (1) |
| gfx-rs/wgpu | `48904f8e` | racine | 683 | 318 099 | 2 | 0 | 0 | non |
| bitshifter/glam-rs | `16e0d32f` | racine | 225 | 226 782 | 0 | 0 | 0 | non |
| Tencent/ncnn | `13b6d531` | `src` | 1 747 | 883 793 | 0 | 0 | 0 | oui (1) |
| alibaba/MNN | `785907a8` | `source` | 2 092 | 1 613 017 | 0 | 0 | 0 | non |
| apache/tvm | `67bd1ea1` | `src` | 874 | 275 047 | 0 | 0 | 0 | non |
| ggml-org/whisper.cpp | `6fc7c33b` | racine | 1 329 | 628 589 | 0 | 0 | 0 | oui (3) |
| leejet/stable-diffusion.cpp | `cc734292` | racine | 161 | 141 339 | 0 | 0 | 0 | non |
| webonnx/wonnx | `c62f5d33` | racine | 49 | 18 003 | 0 | 0 | 0 | non |
| **Total** | | | **22 450** | **9 160 848** | **14** | **0** | **9** | **9/22 dépôts** |

## 4. Revue manuelle des 14 candidats

### 4.1 ndarray@bd3ade99 — 12 candidats, tous BENIGN (contexte test)

`src/array_approx.rs:199-234` — module `#[cfg(test)] mod tests` **inline dans
`src/`** (lignes 181-182 du fichier), donc non couvert par l'exclusion de
chemins. Six assertions du type :

```rust
// Check epsilon.
assert_abs_diff_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
assert_abs_diff_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);
```

- Typage f32 : établi (suffixes `f32`). Mécanisme M2 applicable en plage.
- Usage de garde : **non établi** — `1e-40`/`1e-41` sont des tolérances
  d'assertion choisies **délibérément** sous-normales pour tester la
  sémantique des comparaisons `approx` autour de zéro. Aucune division,
  aucun `.max(`, aucun seuil de protection.
- Classification : **BENIGN** (contexte test ; valeur non utilisée comme
  garde). Idem pour les lignes 216-217 (`assert_relative_*`) et 233-234
  (`assert_ulps_*`).

### 4.2 wgpu@48904f8e — 2 candidats, tous BENIGN (constantes de test délibérées)

`naga/src/front/wgsl/parse/lexer.rs:854-856` — section `#[cfg(test)]` du
lexer WGSL :

```rust
const SMALLEST_POSITIVE_SUBNORMAL_F32: f32 = 1e-45;
const LARGEST_SUBNORMAL_F32: f32 = 1.1754942e-38;
```

- Typage f32 : établi (annotation `: f32`). Mécanismes M2/M1 applicables en
  plage.
- Usage de garde : **non établi** — constantes nommées explicitement
  `SUBNORMAL`, utilisées pour vérifier que le lexer WGSL parse correctement
  les littéraux sous-normaux. Le caractère sous-normal est l'objet même du
  test.
- Classification : **BENIGN** (contexte test ; sous-normalité intentionnelle
  et documentée par le nom).

### 4.3 faer-rs@0539947f — 9 littéraux UNCERTAIN (non comptés)

`1e-200`/`1e-250` dans des boucles de test de normes (`for factor in [...,
1e-250]`) et des helpers f128 (`eigen-bench-setup/eigen.cpp`,
`faer-ffi/quad.hpp`). Contextes f64/f128 : hors plage f32 par typage —
correctement exclus du décompte par l'heuristique.

### 4.4 Drapeaux fast-math / FTZ (colonne « FTZ probable »)

Le mécanisme M1 suppose un environnement FTZ/DAZ. La campagne confirme que
cet environnement est **réel et répandu** : 9 des 22 dépôts activent
fast-math ou FTZ dans leurs fichiers de build, notamment :

- `ggml`/`llama.cpp`/`whisper.cpp` : `-ffast-math` (CPU), `--use_fast_math`
  (CUDA), HIP/MUSA (`src/ggml-cpu/CMakeLists.txt:720`,
  `src/ggml-cuda/CMakeLists.txt:197`) ;
- `pytorch` : `-ffast-math` dans QNNPACK
  (`aten/src/ATen/native/quantized/cpu/qnnpack/buckbuild.bzl`) ;
- `tensorflow` : contrôle `ftz` explicite des kernels générés
  (`tensorflow/core/kernels/mlir_generated/build_defs.bzl:169`) ;
- `candle` : `--use_fast_math` (flash-attn, `candle-flash-attn/build.rs:141`) ;
- `OpenBLAS` (`Makefile.power:147`), `ncnn` (`src/CMakeLists.txt:379`),
  `tract` (bench).

Le *modèle de menace* (des sous-normaux flushés en production GPU/fast-math)
est donc confirmé ; c'est la *classe de bug elle-même* qui n'a pas été
observée dans le corpus.

## 5. Limitations

- **Parsing lexical, pas sémantique** : aucune inférence de types (le typage
  repose sur suffixes et mentions de type sur la ligne), aucune
  macro-expansion, aucune propagation de constantes (`f32::from_bits`,
  constantes nommées réutilisées ailleurs échappent au scan), littéraux
  hexadécimaux flottants (`0x1p-149`) non couverts.
- **Exclusion de tests incomplète par construction** : les exclusions sont
  des règles de chemins ; les modules de test *inline* (`#[cfg(test)]` dans
  `src/`) sont scannés — les 14 candidats de la campagne venaient tous de
  tels modules et ont été écartés en revue manuelle, pas par l'outil.
- **Périmètre** : les dépôts géants sont scannés sur les sous-répertoires
  cœur indiqués dans la table (clone sparse), pas sur l'arbre entier ;
  les shaders embarqués dans des chaînes de caractères (WGSL inline dans du
  Rust, comme en pratique dans wgpu/wonnx) ne sont pas scannés — le scanner
  saute les chaînes.
- **Biais de la plage** : seuls les littéraux `< f32::MIN_POSITIVE` sont
  capturés. Une garde « trop petite pour son échelle » mais normale
  (ex. `1e-30` face à des carrés de valeurs ~`1e-5`) est une classe de bug
  voisine, réelle, mais hors du critère mécanique M1/M2 — non mesurée ici.
- **Un instantané** : un commit par dépôt, à la date de campagne.

## 6. Règle de décision appliquée et verdict

Règle (fixée avant la campagne) :

- ≥ 3 `CONFIRMED_DEAD_GUARD` dans ≥ 2 dépôts distincts connus → **GO** :
  l'étude devient la section de motivation du paper, avec bug reports
  rédigés (jamais postés).
- Sinon → **NO-GO** : résultat négatif consigné honnêtement ; le paper se
  positionne sans cette section.

Chiffres : 22 dépôts scannés (seuil d'acceptation : ≥ 20 — atteint),
9 160 848 lignes, 14 candidats bruts, **0 CONFIRMED_DEAD_GUARD**,
14 BENIGN, 0 UNCERTAIN restant après revue (les 9 littéraux incertains de
faer-rs sont des contextes f64/f128, écartés par typage).

### Verdict : **NO-GO**

La classe de bug « garde epsilon morte » (au sens strict M1/M2 : littéral
f32 sous-normal utilisé comme garde) **n'a pas été observée** dans le corpus
de 22 dépôts numériques majeurs (~9,2 M LOC). Les seuls littéraux
sous-normaux f32 trouvés hors tests exclus étaient des valeurs de test
délibérées (tolérances `approx` de ndarray, constantes de lexer de naga).

Lecture honnête du résultat négatif :

1. **La prévalence dans du code mûr et largement relu est ≈ 0** sur ce
   critère mécanique. Les praticiens choisissent des gardes normales
   (`1e-6`…`1e-12` en f32 typique) — l'erreur « garde sous-normale » ne
   survit apparemment pas dans les projets de cette maturité.
2. **Le résultat ne réfute pas l'utilité du gate σ interne** : le gate
   `epsilon-audit --check` de SciRust est *préventif* (il bloque
   l'introduction d'une telle garde en CI sur la voie sanitized, où
   `sanitize_f32` écrase tout sous-normal par construction) ; la campagne
   montre que l'environnement FTZ visé est répandu (9/22 dépôts), pas que
   l'erreur est fréquente.
3. **Conséquence pour le paper (Lot 3)** : pas de section « prévalence
   mesurée de la classe de bug » ; l'argument se recentre sur le coût
   mesuré du déterminisme et l'architecture d'évidence. L'étude négative
   peut être citée en une phrase (méthode + chiffres) comme due diligence.

Aucun bug report n'est donc rédigé (la branche GO n'est pas prise), aucune
issue/PR n'a été ouverte, aucun contact extérieur n'a eu lieu ; les extraits
cités sont ≤ 3 lignes par finding (dépôts publics, usage d'analyse).
