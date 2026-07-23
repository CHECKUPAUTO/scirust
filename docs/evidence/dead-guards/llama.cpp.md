# Rapport de minage « dead guards »

Généré par `epsilon-audit --mine` (crate `scirust-sigma`, std-only, parsing lexical).
Racine minée : `/tmp/mining/llama.cpp`.
Seuils : M1 (flush FTZ/DAZ) `< 1.1754943508222875e-38` = `f32::MIN_POSITIVE` ; M2 (inversion) `< 2.938736052218037e-39` = `1/f32::MAX`.
Rapport déterministe (tri stable, aucun horodatage) — reproductible bit-à-bit.

## 1. Statistiques

| Mesure | Valeur |
|---|---:|
| Fichiers scannés | 1550 |
| Lignes scannées | 609678 |
| Fichiers exclus (tests, vendor…) | 0 |
| Candidats (CONFIRMED-F32 + PROBABLE-F32) | 0 |
| — dont mécanisme M2 (inversion) | 0 |
| Littéraux sous-seuil UNCERTAIN (non comptés) | 0 |
| Littéraux sous-seuil NOT-F32 (écartés) | 0 |
| Drapeaux fast-math/FTZ (fichiers de build) | 7 |

## 2. Candidats (à revue manuelle)

Aucun candidat.

## 3. TSV (agrégation)

```tsv
file	line	lang	literal	value	mechanism	verdict	guard	extract
```

## 4. Drapeaux fast-math / FTZ

- `ggml/src/ggml-cpu/CMakeLists.txt:720` — `-ffast-math`
- `ggml/src/ggml-cuda/CMakeLists.txt:197` — `use_fast_math`
- `ggml/src/ggml-hip/CMakeLists.txt:133` — `-ffast-math`
- `ggml/src/ggml-hip/CMakeLists.txt:133` — `use_fast_math`
- `ggml/src/ggml-hip/CMakeLists.txt:134` — `-funsafe-math-optimizations`
- `ggml/src/ggml-hip/CMakeLists.txt:162` — `-ffast-math`
- `ggml/src/ggml-musa/CMakeLists.txt:60` — `-ffast-math`

## 5. Littéraux UNCERTAIN (0 — non comptés comme findings)

```text
```

---

Report-SHA256: `7a1db3ac121cedc9fdd6d779ea87826f86ccd3e50ddaeb931752be7f2426bdcf`
