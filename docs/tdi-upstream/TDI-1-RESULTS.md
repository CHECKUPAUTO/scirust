# TDI-1 — Résultats reproductibles

## Hypothèse testée

La structure prospective des futurs accessibles apporte un signal prédictif supplémentaire par rapport à l’entropie de Shannon seule pour prédire la récupération d’un système déterministe après perturbation.

## Protocole expérimental

- Réseaux déterministes finis
- Largeur : 3 bits
- États possibles par système : 8
- Entraînement : 12 000 systèmes
- Holdout indépendant : 4 000 systèmes
- Horizon d’entropie : 8
- Horizon prospectif TDI : 4
- Horizon maximal de récupération : 32
- Perturbation : inversion du bit 2
- Bootstrap apparié déterministe : 2 000 réplications

## Résultats holdout

| Modèle | Accuracy | Balanced accuracy | Brier | AUPRC |
|---|---:|---:|---:|---:|
| Entropie seule | 0.716250 | 0.500000 | 0.200117 | 0.760675 |
| Profil de retour TDI | 0.812000 | 0.706496 | 0.144390 | 0.827000 |
| Entropie + TDI | 0.814750 | 0.692721 | 0.141205 | 0.854758 |

## Gains observés

- Gain AUPRC du profil TDI sur l’entropie : `+0.066325`
- Gain AUPRC du modèle combiné : `+0.094083`
- Amélioration du score de Brier avec TDI : `+0.055728`
- Amélioration du score de Brier combiné : `+0.058912`

## Intervalles de confiance bootstrap à 95 %

| Comparaison | IC 95 % | Médiane |
|---|---:|---:|
| Gain AUPRC TDI | [0.051310, 0.080562] | 0.066138 |
| Amélioration Brier TDI | [0.050331, 0.061267] | 0.055780 |
| Gain AUPRC combiné | [0.081111, 0.106507] | 0.093957 |
| Amélioration Brier combinée | [0.053399, 0.064276] | 0.058929 |

## Critère préenregistré TDI-1

Le succès exigeait simultanément :

1. un gain AUPRC TDI observé d’au moins `0.05` ;
2. une borne inférieure de l’IC 95 % du gain AUPRC strictement positive ;
3. une borne inférieure de l’IC 95 % de l’amélioration Brier strictement positive.

```text
CRITÈRE PRÉENREGISTRÉ TDI-1 : RÉUSSI
```

## Conclusion limitée aux preuves disponibles

Sur la famille synthétique étudiée, le profil prospectif de retour TDI contient une information prédictive qui n’est pas conservée par l’entropie de Shannon seule.

L’expérience établit notamment :

- l’existence de systèmes ayant la même entropie mais des comportements de récupération différents ;
- la séparation de milliers de paires opposées par le profil TDI ;
- un gain prédictif sur un ensemble holdout indépendant ;
- des intervalles de confiance appariés strictement positifs.

## Limites

TDI-1 ne démontre pas encore :

- une loi fondamentale générale de l’information ;
- une validité biologique, physique ou quantique ;
- une supériorité sur toutes les mesures dynamiques existantes ;
- une généralisation à des systèmes continus, stochastiques ou réels ;
- une indépendance complète vis-à-vis du protocole de perturbation choisi.

## Contrôle dynamique orbital

Une baseline plus forte utilise les longueurs transitoires et les périodes des orbites de référence et perturbée.

| Modèle | Accuracy | Balanced accuracy | Brier | AUPRC |
|---|---:|---:|---:|---:|
| Baseline orbitale | 0.926250 | 0.870044 | 0.053284 | 0.984074 |
| Orbite + TDI | 0.926250 | 0.870044 | 0.053284 | 0.984074 |
| Entropie + orbite | 0.917000 | 0.860927 | 0.057089 | 0.983091 |
| Entropie + orbite + TDI | 0.917000 | 0.860927 | 0.057089 | 0.983091 |

Les gains incrémentaux de TDI contre la baseline orbitale sont exactement nuls :

- gain AUPRC : `0.000000` ;
- amélioration Brier : `0.000000` ;
- IC bootstrap à 95 % : `[0.000000, 0.000000]`.

```text
CRITÈRE ORIGINAL TDI-1 VS ENTROPIE : RÉUSSI
CONTRÔLE DE NOUVEAUTÉ VS ORBITE     : ÉCHOUÉ
```

TDI-1 réfute donc la suffisance de l’entropie scalaire seule, mais ne démontre pas encore un invariant dynamique indépendant des propriétés orbitales classiques.
