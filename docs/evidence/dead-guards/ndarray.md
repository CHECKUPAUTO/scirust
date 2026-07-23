# Rapport de minage « dead guards »

Généré par `epsilon-audit --mine` (crate `scirust-sigma`, std-only, parsing lexical).
Racine minée : `/tmp/mining/ndarray`.
Seuils : M1 (flush FTZ/DAZ) `< 1.1754943508222875e-38` = `f32::MIN_POSITIVE` ; M2 (inversion) `< 2.938736052218037e-39` = `1/f32::MAX`.
Rapport déterministe (tri stable, aucun horodatage) — reproductible bit-à-bit.

## 1. Statistiques

| Mesure | Valeur |
|---|---:|
| Fichiers scannés | 113 |
| Lignes scannées | 33147 |
| Fichiers exclus (tests, vendor…) | 0 |
| Candidats (CONFIRMED-F32 + PROBABLE-F32) | 12 |
| — dont mécanisme M2 (inversion) | 12 |
| Littéraux sous-seuil UNCERTAIN (non comptés) | 0 |
| Littéraux sous-seuil NOT-F32 (écartés) | 0 |
| Drapeaux fast-math/FTZ (fichiers de build) | 0 |

## 2. Candidats (à revue manuelle)

| Fichier:ligne | Langage | Littéral | Mécanisme | Typage | Garde | Extrait |
|---|---|---|---|---|---|---|
| `src/array_approx.rs:199` | rust | `1e-40` | M2 | CONFIRMED-F32 | eps | `assert_abs_diff_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);` |
| `src/array_approx.rs:199` | rust | `1e-40` | M2 | CONFIRMED-F32 | eps | `assert_abs_diff_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);` |
| `src/array_approx.rs:200` | rust | `1e-40` | M2 | CONFIRMED-F32 | eps | `assert_abs_diff_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);` |
| `src/array_approx.rs:200` | rust | `1e-41` | M2 | CONFIRMED-F32 | eps | `assert_abs_diff_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);` |
| `src/array_approx.rs:216` | rust | `1e-40` | M2 | CONFIRMED-F32 | eps | `assert_relative_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);` |
| `src/array_approx.rs:216` | rust | `1e-40` | M2 | CONFIRMED-F32 | eps | `assert_relative_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);` |
| `src/array_approx.rs:217` | rust | `1e-40` | M2 | CONFIRMED-F32 | eps | `assert_relative_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);` |
| `src/array_approx.rs:217` | rust | `1e-41` | M2 | CONFIRMED-F32 | eps | `assert_relative_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);` |
| `src/array_approx.rs:233` | rust | `1e-40` | M2 | CONFIRMED-F32 | eps | `assert_ulps_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);` |
| `src/array_approx.rs:233` | rust | `1e-40` | M2 | CONFIRMED-F32 | eps | `assert_ulps_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);` |
| `src/array_approx.rs:234` | rust | `1e-40` | M2 | CONFIRMED-F32 | eps | `assert_ulps_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);` |
| `src/array_approx.rs:234` | rust | `1e-41` | M2 | CONFIRMED-F32 | eps | `assert_ulps_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);` |

## 3. TSV (agrégation)

```tsv
file	line	lang	literal	value	mechanism	verdict	guard	extract
src/array_approx.rs	199	rust	1e-40	1e-40	M2	CONFIRMED-F32	eps	assert_abs_diff_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
src/array_approx.rs	199	rust	1e-40	1e-40	M2	CONFIRMED-F32	eps	assert_abs_diff_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
src/array_approx.rs	200	rust	1e-40	1e-40	M2	CONFIRMED-F32	eps	assert_abs_diff_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);
src/array_approx.rs	200	rust	1e-41	1e-41	M2	CONFIRMED-F32	eps	assert_abs_diff_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);
src/array_approx.rs	216	rust	1e-40	1e-40	M2	CONFIRMED-F32	eps	assert_relative_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
src/array_approx.rs	216	rust	1e-40	1e-40	M2	CONFIRMED-F32	eps	assert_relative_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
src/array_approx.rs	217	rust	1e-40	1e-40	M2	CONFIRMED-F32	eps	assert_relative_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);
src/array_approx.rs	217	rust	1e-41	1e-41	M2	CONFIRMED-F32	eps	assert_relative_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);
src/array_approx.rs	233	rust	1e-40	1e-40	M2	CONFIRMED-F32	eps	assert_ulps_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
src/array_approx.rs	233	rust	1e-40	1e-40	M2	CONFIRMED-F32	eps	assert_ulps_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
src/array_approx.rs	234	rust	1e-40	1e-40	M2	CONFIRMED-F32	eps	assert_ulps_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);
src/array_approx.rs	234	rust	1e-41	1e-41	M2	CONFIRMED-F32	eps	assert_ulps_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);
```

## 4. Drapeaux fast-math / FTZ

Aucun drapeau détecté dans les fichiers de build.

## 5. Littéraux UNCERTAIN (0 — non comptés comme findings)

```text
```

---

Report-SHA256: `3a10130866d8eef3f1c09e09d1456d20f55f6a4afbe7f1df3fd5b85207a68952`
