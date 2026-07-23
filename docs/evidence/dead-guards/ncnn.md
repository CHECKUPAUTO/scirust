# Rapport de minage « dead guards »

Généré par `epsilon-audit --mine` (crate `scirust-sigma`, std-only, parsing lexical).
Racine minée : `/tmp/mining/ncnn`.
Seuils : M1 (flush FTZ/DAZ) `< 1.1754943508222875e-38` = `f32::MIN_POSITIVE` ; M2 (inversion) `< 2.938736052218037e-39` = `1/f32::MAX`.
Rapport déterministe (tri stable, aucun horodatage) — reproductible bit-à-bit.

## 1. Statistiques

| Mesure | Valeur |
|---|---:|
| Fichiers scannés | 1747 |
| Lignes scannées | 883793 |
| Fichiers exclus (tests, vendor…) | 0 |
| Candidats (CONFIRMED-F32 + PROBABLE-F32) | 0 |
| — dont mécanisme M2 (inversion) | 0 |
| Littéraux sous-seuil UNCERTAIN (non comptés) | 0 |
| Littéraux sous-seuil NOT-F32 (écartés) | 0 |
| Drapeaux fast-math/FTZ (fichiers de build) | 1 |

## 2. Candidats (à revue manuelle)

Aucun candidat.

## 3. TSV (agrégation)

```tsv
file	line	lang	literal	value	mechanism	verdict	guard	extract
```

## 4. Drapeaux fast-math / FTZ

- `src/CMakeLists.txt:379` — `-ffast-math`

## 5. Littéraux UNCERTAIN (0 — non comptés comme findings)

```text
```

---

Report-SHA256: `c0fa5d9cca8cd0bf6e2b0a70c1702b3826430ac3dc6f7b685406c8564ed48e69`
