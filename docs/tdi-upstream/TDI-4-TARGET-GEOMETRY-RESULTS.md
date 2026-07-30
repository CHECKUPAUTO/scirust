# TDI-4 — Résultats préenregistrés Target Geometry

## Identité

- Branche : `tdi-4-target-geometry`
- Gel scientifique : `367bff80467d22f0aee0b3122de9dac893a0e0b3`
- CI du gel : `29233554774`
- Environnement : Jetson AGX Thor, Linux AArch64
- Rust : `rustc 1.97.0`
- Durée : 148 secondes
- Résultat brut : `results/tdi-target-geometry.log`
- Console complète : `results/tdi4-first-complete-evaluation-console.log`
- SHA-256 résultat : `f1b67144b91f025e586cafc3a4b18b44360564055e46de2be1eecb18d04059a9`
- SHA-256 console : `11227c39c348d41a8514eed3eb8d0a48c181af8f554a660d931f1928f6f64ffc`

## Verdicts préenregistrés

    CRITÈRE PRINCIPAL TDI-4A : ÉCHOUÉ
    CRITÈRE TRANSFERT TDI-4B : ÉCHOUÉ

Ces verdicts sont définitifs pour ce protocole.

## Géométrie de la cible

Aucune récupération exacte à l’horizon 6 n’a été observée.

| Population | Systèmes | Récupérations exactes |
|---|---:|---:|
| Apprentissage largeurs 3 et 4 | 20 000 | 0 |
| Holdout largeur 3 | 5 000 | 0 |
| Holdout largeur 4 | 5 000 | 0 |
| Holdout OOD largeur 5 | 5 000 | 0 |
| Total | 35 000 | 0 |

La cible de la tête binaire est donc constante et égale à zéro.

La baseline et le challenger obtiennent tous deux un Brier score nul.
L’amélioration Brier et son intervalle bootstrap sont exactement nuls.

Cette condition empêche formellement la réussite de TDI-4A et TDI-4B.

## Résultats de la composante continue

La composante conditionnelle utilise :

\[
U=-\log_2(1-O_6).
\]

| Population | Perte baseline | Perte TDI-4 | Réduction |
|---|---:|---:|---:|
| Largeur 3 | 0.241887657 | 0.179439894 | 25.816846 % |
| Largeur 4 | 0.085792849 | 0.065318515 | 23.864849 % |
| Combiné 3 et 4 | 0.163840253 | 0.122379204 | 25.305777 % |
| Largeur 5 OOD | 0.071703443 | 0.035827770 | 50.033403 % |

## Bootstrap de la perte composite

| Population | Intervalle à 95 % |
|---|---:|
| Largeur 3 | [0.055802305, 0.069284921] |
| Largeur 4 | [0.018695561, 0.022346104] |
| Combiné 3 et 4 | [0.037685803, 0.044976080] |
| Largeur 5 OOD | [0.034570495, 0.037174978] |

Toutes les bornes inférieures de l’amélioration de perte composite sont
strictement positives.

## Largeur 3

| Métrique conditionnelle | Baseline | TDI-4 |
|---|---:|---:|
| MSE | 0.483775315 | 0.358879788 |
| R² | 0.450815004 | 0.592597247 |
| Spearman | 0.637248808 | 0.757689675 |

La MSE reconstruite diminue de 0.001712584 à 0.001497982.

La MAE reconstruite diminue de 0.011346318 à 0.010076199.

## Largeur 4

| Métrique conditionnelle | Baseline | TDI-4 |
|---|---:|---:|
| MSE | 0.171585697 | 0.130637029 |
| R² | 0.332284748 | 0.491633987 |
| Spearman | 0.565404261 | 0.696432192 |

La MSE reconstruite diminue de 0.000001282 à 0.000001099.

La MAE reconstruite diminue de 0.000479102 à 0.000426239.

## Transfert hors distribution — largeur 5

| Métrique conditionnelle | Baseline | TDI-4 |
|---|---:|---:|
| MSE | 0.143406887 | 0.071655540 |
| MAE | 0.318879657 | 0.217243811 |
| R² | -0.762078825 | 0.119549183 |
| Spearman | 0.467776309 | 0.615705212 |
| Biais | -0.284013178 | -0.146952367 |

| Métrique reconstruite | Baseline | TDI-4 |
|---|---:|---:|
| MSE | 0.000000005 | 0.000000002 |
| MAE | 0.000061450 | 0.000038676 |
| R² | -0.641389205 | 0.243959397 |
| Spearman | 0.467776309 | 0.615705212 |
| Biais | -0.000050411 | -0.000019773 |

Intervalles bootstrap OOD :

- perte composite : [0.034570495, 0.037174978]
- MSE conditionnelle : [0.069140990, 0.074349957]
- MSE reconstruite : [0.000000003, 0.000000003]
- MAE reconstruite : [0.000022019, 0.000023481]
- Brier : [0.000000000, 0.000000000]

## Analyse des critères

### TDI-4A

Conditions réussies :

- réduction composite combinée supérieure à 5 % ;
- intervalle composite combiné strictement positif ;
- intervalles composites des largeurs 3 et 4 strictement positifs ;
- amélioration ponctuelle de la MSE reconstruite ;
- amélioration ponctuelle de la MAE reconstruite ;
- Spearman conditionnel positif aux largeurs 3 et 4.

Condition échouée :

- borne inférieure de l’amélioration Brier strictement positive.

### TDI-4B

Conditions réussies :

- intervalle composite OOD strictement positif ;
- réduction de MSE reconstruite supérieure à 5 % ;
- intervalle de MSE reconstruite strictement positif ;
- Spearman challenger positif et supérieur à la baseline ;
- biais absolu challenger inférieur à celui de la baseline.

Condition échouée :

- borne inférieure de l’amélioration Brier strictement positive.

## Interprétation scientifique

Le protocole complet à deux têtes échoue formellement.

La tête de récupération exacte est dégénérée dans les populations étudiées,
car aucun exemple positif n’a été observé.

La géométrie continue du déficit est néanmoins fortement soutenue :

- amélioration aux largeurs 3 et 4 ;
- intervalles composites strictement positifs ;
- transfert hors distribution important à la largeur 5 ;
- Spearman OOD positif et amélioré ;
- passage d’un R² OOD négatif à un R² positif ;
- réduction du biais de reconstruction.

L’interprétation retenue est donc :

1. le verdict préenregistré TDI-4 reste négatif ;
2. la tête binaire n’est pas identifiable dans ce régime ;
3. la composante continue présente un signal robuste ;
4. les variables TDI apportent une information prédictive inter-largeurs ;
5. une expérience ultérieure devra préenregistrer directement l’étude de la
   géométrie continue, sur de nouvelles populations.
