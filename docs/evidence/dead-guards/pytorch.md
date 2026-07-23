# Rapport de minage « dead guards »

Généré par `epsilon-audit --mine` (crate `scirust-sigma`, std-only, parsing lexical).
Racine minée : `/tmp/mining/pytorch`.
Seuils : M1 (flush FTZ/DAZ) `< 1.1754943508222875e-38` = `f32::MIN_POSITIVE` ; M2 (inversion) `< 2.938736052218037e-39` = `1/f32::MAX`.
Rapport déterministe (tri stable, aucun horodatage) — reproductible bit-à-bit.

## 1. Statistiques

| Mesure | Valeur |
|---|---:|
| Fichiers scannés | 2599 |
| Lignes scannées | 709503 |
| Fichiers exclus (tests, vendor…) | 14 |
| Candidats (CONFIRMED-F32 + PROBABLE-F32) | 0 |
| — dont mécanisme M2 (inversion) | 0 |
| Littéraux sous-seuil UNCERTAIN (non comptés) | 0 |
| Littéraux sous-seuil NOT-F32 (écartés) | 0 |
| Drapeaux fast-math/FTZ (fichiers de build) | 5 |

## 2. Candidats (à revue manuelle)

Aucun candidat.

## 3. TSV (agrégation)

```tsv
file	line	lang	literal	value	mechanism	verdict	guard	extract
```

## 4. Drapeaux fast-math / FTZ

- `aten/src/ATen/native/quantized/cpu/qnnpack/buckbuild.bzl:98` — `-ffast-math`
- `aten/src/ATen/native/quantized/cpu/qnnpack/buckbuild.bzl:147` — `-ffast-math`
- `aten/src/ATen/native/quantized/cpu/qnnpack/buckbuild.bzl:196` — `-ffast-math`
- `aten/src/ATen/native/quantized/cpu/qnnpack/buckbuild.bzl:410` — `-ffast-math`
- `aten/src/ATen/native/quantized/cpu/qnnpack/buckbuild.bzl:528` — `-ffast-math`

## 5. Littéraux UNCERTAIN (0 — non comptés comme findings)

```text
```

---

Report-SHA256: `6c4208cc07c8926bacf9a2fc5940fd0dfce09d9261ec1832a59aee6ddea9dd0c`
