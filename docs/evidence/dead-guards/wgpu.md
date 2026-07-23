# Rapport de minage « dead guards »

Généré par `epsilon-audit --mine` (crate `scirust-sigma`, std-only, parsing lexical).
Racine minée : `/tmp/mining/wgpu`.
Seuils : M1 (flush FTZ/DAZ) `< 1.1754943508222875e-38` = `f32::MIN_POSITIVE` ; M2 (inversion) `< 2.938736052218037e-39` = `1/f32::MAX`.
Rapport déterministe (tri stable, aucun horodatage) — reproductible bit-à-bit.

## 1. Statistiques

| Mesure | Valeur |
|---|---:|
| Fichiers scannés | 683 |
| Lignes scannées | 318099 |
| Fichiers exclus (tests, vendor…) | 2 |
| Candidats (CONFIRMED-F32 + PROBABLE-F32) | 2 |
| — dont mécanisme M2 (inversion) | 1 |
| Littéraux sous-seuil UNCERTAIN (non comptés) | 0 |
| Littéraux sous-seuil NOT-F32 (écartés) | 0 |
| Drapeaux fast-math/FTZ (fichiers de build) | 0 |

## 2. Candidats (à revue manuelle)

| Fichier:ligne | Langage | Littéral | Mécanisme | Typage | Garde | Extrait |
|---|---|---|---|---|---|---|
| `naga/src/front/wgsl/parse/lexer.rs:854` | rust | `1e-45` | M2 | CONFIRMED-F32 | — | `const SMALLEST_POSITIVE_SUBNORMAL_F32: f32 = 1e-45;` |
| `naga/src/front/wgsl/parse/lexer.rs:856` | rust | `1.1754942e-38` | M1 | CONFIRMED-F32 | — | `const LARGEST_SUBNORMAL_F32: f32 = 1.1754942e-38;` |

## 3. TSV (agrégation)

```tsv
file	line	lang	literal	value	mechanism	verdict	guard	extract
naga/src/front/wgsl/parse/lexer.rs	854	rust	1e-45	1e-45	M2	CONFIRMED-F32	-	const SMALLEST_POSITIVE_SUBNORMAL_F32: f32 = 1e-45;
naga/src/front/wgsl/parse/lexer.rs	856	rust	1.1754942e-38	1.1754942e-38	M1	CONFIRMED-F32	-	const LARGEST_SUBNORMAL_F32: f32 = 1.1754942e-38;
```

## 4. Drapeaux fast-math / FTZ

Aucun drapeau détecté dans les fichiers de build.

## 5. Littéraux UNCERTAIN (0 — non comptés comme findings)

```text
```

---

Report-SHA256: `5b49cbc7ecdeb4674b4530dfa2824702250e1d7e2e165a9375a96cd3e8e75d16`
