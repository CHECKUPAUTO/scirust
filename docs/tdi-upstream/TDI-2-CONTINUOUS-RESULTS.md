# TDI-2 — Résultats de l’évaluation continue préenregistrée

Date d’exécution : 12 juillet 2026  
Branche : `tdi-2-branching`  
Préenregistrement : `3c8f54d`  
Implémentation gelée : `ccbfc31`

## Correction technique préalable

La première tentative d’exécution a échoué avant toute production de
résultats en raison d’un dépassement intermédiaire lors de la comparaison
de deux rationnels `u128` sur le holdout de largeur 4.

La comparaison par produits croisés a été remplacée par une comparaison
exacte fondée sur les quotients successifs de l’algorithme d’Euclide.

Cette correction :

- ne change aucune population ;
- ne change aucun horizon ;
- ne change aucune caractéristique ;
- ne change aucun modèle ;
- ne change aucune métrique ;
- ne consulte aucun résultat de holdout ;
- ajoute un test avec des rationnels proches de `u128::MAX`.

## Validation logicielle

- formatage : réussi ;
- tests workspace : réussi ;
- tests `tdi-core` : 38 réussis ;
- Clippy avec `-D warnings` : réussi ;
- vérification Git : réussie ;
- exécution du benchmark : réussie.

## Population effectivement analysée

| Ensemble | Largeur | Systèmes acceptés | Exclus à l’horizon 2 |
|---|---:|---:|---:|
| Apprentissage | 3 | 12 000 | 55 |
| Holdout principal | 3 | 4 000 | 24 |
| Holdout hors distribution | 4 | 4 000 | 0 |

Les exclusions correspondent uniquement aux systèmes dont les distributions
étaient déjà exactement identiques à l’horizon d’observation 2.

## Holdout principal — largeur 3

### Baseline appariée

- MSE : `0.001816873`
- MAE : `0.018304843`
- R² : `0.393556075`
- Spearman : `0.411338780`

### Baseline + TDI-2

- MSE : `0.001579275`
- MAE : `0.017156635`
- R² : `0.472862476`
- Spearman : `0.557910624`

### Gain observé

- amélioration MSE : `0.000237598`
- réduction relative MSE : `13.077285 %`
- amélioration MAE : `0.001148208`

### Bootstrap apparié déterministe — 2 000 réplications

- IC 95 % amélioration MSE :
  `[0.000162544, 0.000315743]`
- médiane amélioration MSE :
  `0.000236356`
- IC 95 % amélioration MAE :
  `[0.000788787, 0.001495130]`
- médiane amélioration MAE :
  `0.001151804`

## Verdict principal préenregistré

Le critère principal est **réussi** :

1. réduction relative de MSE supérieure à 5 % ;
2. borne inférieure de l’IC 95 % MSE strictement positive ;
3. borne inférieure de l’IC 95 % MAE strictement positive.

Le gain de TDI-2 sur la baseline appariée est donc statistiquement robuste
sur le holdout principal de largeur 3.

## Holdout hors distribution — largeur 4

### Baseline appariée

- MSE : `0.157274396`
- MAE : `0.372795169`
- R² : `-109506.813463054`
- Spearman : `-0.507018418`

### Baseline + TDI-2

- MSE : `0.147526127`
- MAE : `0.366151366`
- R² : `-102719.239290951`
- Spearman : `-0.489819920`

### Gain observé

- amélioration MSE : `0.009748269`
- réduction relative MSE : `6.198256 %`
- amélioration MAE : `0.006643803`

Le critère minimal préenregistré du holdout hors distribution demandait
uniquement une amélioration MSE positive. Il est donc formellement réussi.

## Réserve scientifique majeure

La confirmation hors distribution ne doit pas être présentée comme une
bonne généralisation absolue.

Malgré une amélioration relative face à la baseline :

- les deux modèles ont un R² extrêmement négatif ;
- les deux corrélations de Spearman sont négatives ;
- les erreurs absolues sont très élevées ;
- un modèle entraîné en largeur 3 est manifestement mal calibré en largeur 4.

La conclusion honnête est donc :

- **succès confirmatoire sur le holdout principal de largeur 3** ;
- **gain relatif TDI-2 également observé en largeur 4** ;
- **absence de validité prédictive absolue hors distribution** ;
- **nécessité d’une étude séparée de transfert inter-largeurs**.

## Conclusion

TDI-2 apporte une information prédictive incrémentale au-delà de
l’entropie et de la topologie appariées au même horizon d’observation.

Le résultat principal est positif et soutenu par les intervalles bootstrap.
Il établit un signal prospectif incrémental dans la population de largeur 3.

Il n’établit pas encore une loi universelle indépendante de la taille du
système.
