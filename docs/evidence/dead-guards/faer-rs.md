# Rapport de minage « dead guards »

Généré par `epsilon-audit --mine` (crate `scirust-sigma`, std-only, parsing lexical).
Racine minée : `/tmp/mining/faer-rs`.
Seuils : M1 (flush FTZ/DAZ) `< 1.1754943508222875e-38` = `f32::MIN_POSITIVE` ; M2 (inversion) `< 2.938736052218037e-39` = `1/f32::MAX`.
Rapport déterministe (tri stable, aucun horodatage) — reproductible bit-à-bit.

## 1. Statistiques

| Mesure | Valeur |
|---|---:|
| Fichiers scannés | 163 |
| Lignes scannées | 120623 |
| Fichiers exclus (tests, vendor…) | 0 |
| Candidats (CONFIRMED-F32 + PROBABLE-F32) | 0 |
| — dont mécanisme M2 (inversion) | 0 |
| Littéraux sous-seuil UNCERTAIN (non comptés) | 9 |
| Littéraux sous-seuil NOT-F32 (écartés) | 0 |
| Drapeaux fast-math/FTZ (fichiers de build) | 0 |

## 2. Candidats (à revue manuelle)

Aucun candidat.

## 3. TSV (agrégation)

```tsv
file	line	lang	literal	value	mechanism	verdict	guard	extract
```

## 4. Drapeaux fast-math / FTZ

Aucun drapeau détecté dans les fichiers de build.

## 5. Littéraux UNCERTAIN (9 — non comptés comme findings)

```text
eigen-bench-setup/eigen.cpp:193  1e-200  static fx128 min() { return fx128{1e-200}; }
faer/src/linalg/reductions/norm_l1.rs:142  1e-250  for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
faer/src/linalg/reductions/norm_l2.rs:178  1e-250  for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
faer/src/linalg/reductions/norm_l2.rs:202  1e-250  for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
faer/src/linalg/reductions/norm_l2_sqr.rs:141  1e-250  for factor in [0.0, 1.0, 1e30, 1e120, 1e-30, 1e-250] {
faer/src/linalg/reductions/norm_max.rs:141  1e-250  for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
faer/src/linalg/reductions/sum.rs:92  1e-250  for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
faer/src/linalg/reductions/sum.rs:114  1e-250  for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
faer-ffi/quad.hpp:231  1e-200  return quad::f128 {1e-200};
```

---

Report-SHA256: `cab72d8b05339137091004bd113bf4ab3cbf44dbf18671ed56b0bd846ea78192`
