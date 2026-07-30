# TDI-3 — Résultats de l’évaluation inter-width préenregistrée

## Statut

- **Critère principal TDI-3A : ÉCHOUÉ**
- **Critère de transfert TDI-3B : ÉCHOUÉ**

Ce résultat est négatif au sens strict des critères préenregistrés. Il ne
signifie cependant pas que les variables TDI-3 sont dépourvues de signal :
leurs effets sont positifs sur certaines largeurs et négatifs ou
insuffisamment généralisables sur d’autres.

## Identité de l’expérience

- branche : `tdi-3-interwidth`
- commit scientifique gelé :
  `5c7fc5ecae57b8cfa10e6c400f9c03bbd030af4c`
- date d’exécution : 12 juillet 2026
- durée : 118 secondes
- Rust : `rustc 1.97.0`
- architecture : Linux ARM64, Jetson AGX Thor
- résultat :
  `results/tdi-interwidth-continuous.log`
- SHA-256 du résultat :
  `047a33708f691be20308fd51b9a428f98ee041ec1e5809516b5196de15ac7396`
- SHA-256 de la console :
  `633d36937f290bbe528e49111b609fff51166cfd5d40ee6cfa69992b53a12179`

L’intégrité du préenregistrement, de l’évaluateur, du code scientifique et
des dépendances vendoriées a été vérifiée avant l’exécution.

## Populations

| Population | Systèmes retenus | Systèmes exclus |
|---|---:|---:|
| entraînement width 3 | 8 000 | 42 |
| holdout width 3 | 4 000 | 24 |
| entraînement width 4 | 8 000 | 0 |
| holdout width 4 | 4 000 | 0 |
| holdout OOD width 5 | 4 000 | 0 |

Le modèle a été entraîné sur 16 000 systèmes combinant les largeurs 3 et 4.
Les critères ont été évalués sur 8 000 systèmes de holdout in-distribution
et 4 000 systèmes hors distribution de largeur 5.

## Résultats width 3

| Mesure | Baseline | Baseline + TDI-3 |
|---|---:|---:|
| MSE | 0.002080792 | 0.001789824 |
| MAE | 0.018622116 | 0.017337850 |
| R² | 0.305464087 | 0.402584700 |
| Spearman | 0.591430188 | 0.697022323 |

- réduction relative de MSE : **13.983526 %**
- amélioration MAE : **0.001284266**
- IC 95 % de l’amélioration MSE :
  **[0.000195826, 0.000396785]**
- IC 95 % de l’amélioration MAE :
  **[0.000928742, 0.001645253]**

Les variables TDI-3 apportent ici une amélioration nette, statistiquement
stable et accompagnée d’une meilleure qualité de classement.

## Résultats width 4

| Mesure | Baseline | Baseline + TDI-3 |
|---|---:|---:|
| MSE | 0.000025287 | 0.000039535 |
| MAE | 0.002883024 | 0.003628335 |
| R² | -22.139403817 | -35.176751612 |
| Spearman | 0.018318165 | 0.279302369 |

- réduction relative de MSE : **-56.342626 %**
- amélioration MAE : **-0.000745311**
- IC 95 % de l’amélioration MSE :
  **[-0.000017242, -0.000011408]**
- IC 95 % de l’amélioration MAE :
  **[-0.000870391, -0.000621827]**

Les variables TDI-3 augmentent le signal ordinal, mais détériorent
significativement les erreurs MSE et MAE. Les intervalles bootstrap sont
entièrement négatifs : il ne s’agit pas d’une fluctuation d’échantillonnage.

## Holdout combiné widths 3 et 4

| Mesure | Baseline | Baseline + TDI-3 |
|---|---:|---:|
| MSE | 0.001053039 | 0.000914679 |
| MAE | 0.010752570 | 0.010483093 |
| R² | 0.324153855 | 0.412954221 |
| Spearman | 0.490755159 | 0.566936416 |

- réduction relative de MSE : **13.139139 %**
- amélioration MAE : **0.000269478**
- IC 95 % de l’amélioration MSE :
  **[0.000091703, 0.000190158]**
- IC 95 % de l’amélioration MAE :
  **[0.000079518, 0.000467571]**

Le résultat agrégé est favorable et dépasse le seuil de 5 %. Le critère
TDI-3A échoue néanmoins, car le préenregistrement exigeait également une
amélioration MSE positive dans chaque largeur. La largeur 4 viole cette
condition de manière statistiquement nette.

## Transfert hors distribution width 5

| Mesure | Baseline | Baseline + TDI-3 |
|---|---:|---:|
| MSE | 0.000767583 | 0.000483356 |
| MAE | 0.027293619 | 0.021295341 |
| R² | -251476.348326980 | -158357.366983186 |
| Spearman | -0.403040291 | -0.217928974 |
| biais moyen | -0.027293619 | -0.021295341 |

- réduction relative de MSE : **37.028775 %**
- amélioration MAE : **0.005998278**
- IC 95 % de l’amélioration MSE :
  **[0.000279329, 0.000289017]**
- IC 95 % de l’amélioration MAE :
  **[0.005897616, 0.006096733]**

TDI-3 réduit fortement les erreurs absolues et le biais sur la largeur 5.
Cependant, le R² et la corrélation de Spearman restent négatifs. Le modèle
se rapproche donc davantage de cibles presque toutes voisines de 1, sans
apprendre correctement leur variation individuelle. Le critère TDI-3B,
qui exigeait notamment des valeurs positives de R² et de Spearman, échoue.

## Interprétation scientifique

Les résultats indiquent trois faits distincts :

1. **Signal réel en width 3.**
   TDI-3 améliore simultanément l’erreur, le classement et la variance
   expliquée.

2. **Instabilité entre largeurs 3 et 4.**
   La largeur 4 présente une cible extrêmement concentrée près de 1. Les
   variables TDI-3 améliorent Spearman mais détériorent la calibration et
   les erreurs absolues. Le signal appris n’est donc pas invariant à la
   largeur sous cette formulation.

3. **Réduction d’erreur sans généralisation structurelle en width 5.**
   Le gain MSE de 37 % est réel, mais il provient principalement d’une
   correction du biais moyen. Les valeurs négatives de R² et Spearman
   interdisent de conclure à une prédiction structurelle réussie.

## Conclusion

TDI-3 ne valide pas l’hypothèse forte d’une représentation inter-width
universelle selon les critères préenregistrés.

L’expérience fournit néanmoins une information constructive : les
caractéristiques TDI-3 contiennent un signal prédictif, mais ce signal est
sensible à la largeur et à la forte concentration de la cible près de 1.
Toute étude ultérieure devra faire l’objet d’un nouveau préenregistrement,
sans modifier rétroactivement le verdict de TDI-3.
