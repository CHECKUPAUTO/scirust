# Rapport de minage « dead guards »

Généré par `epsilon-audit --mine` (crate `scirust-sigma`, std-only, parsing lexical).
Racine minée : `/tmp/mining/nalgebra`.
Seuils : M1 (flush FTZ/DAZ) `< 1.1754943508222875e-38` = `f32::MIN_POSITIVE` ; M2 (inversion) `< 2.938736052218037e-39` = `1/f32::MAX`.
Rapport déterministe (tri stable, aucun horodatage) — reproductible bit-à-bit.

## 1. Statistiques

| Mesure | Valeur |
|---|---:|
| Fichiers scannés | 279 |
| Lignes scannées | 73273 |
| Fichiers exclus (tests, vendor…) | 0 |
| Candidats (CONFIRMED-F32 + PROBABLE-F32) | 0 |
| — dont mécanisme M2 (inversion) | 0 |
| Littéraux sous-seuil UNCERTAIN (non comptés) | 0 |
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

## 5. Littéraux UNCERTAIN (0 — non comptés comme findings)

```text
```

---

Report-SHA256: `cdff6b3e09126f3dd55473abb851884c74a73594b50990bc1593d2e16138d4a1`
