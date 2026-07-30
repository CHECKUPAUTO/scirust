# TDI-2 — Préenregistrement de l’évaluation continue

Date de gel : 12 juillet 2026  
Branche : `tdi-2-branching`

## 1. Motivation

L’objectif binaire initial définissait une récupération comme une égalité
distributionnelle exacte à l’horizon 6.

Après exclusion des systèmes déjà récupérés à l’horizon d’observation 2,
cet objectif devient dégénéré :

- apprentissage : 1 cas positif sur 12 000 ;
- holdout : 0 cas positif sur 4 000 ;
- AUPRC non interprétable ;
- critère TDI-2 binaire échoué.

La distribution exhaustive de largeur 2 montre néanmoins 44 300 systèmes
à recouvrement final partiel sur 50 625. La variable continue de recouvrement
contient donc beaucoup plus d’information que l’événement rare
« recouvrement exactement égal à 1 ».

## 2. Question principale

Les caractéristiques prospectives conditionnées par l’intervention,
observées jusqu’à l’horizon 2, permettent-elles de prédire le recouvrement
distributionnel à l’horizon 6 au-delà d’une baseline non-TDI utilisant le
même budget temporel ?

## 3. Population

### Apprentissage

- largeur : 3 ;
- 12 000 systèmes acceptés ;
- graines à partir de 0 ;
- transitions générées par `SplitMix64` ;
- chaque état possède un ensemble non vide de successeurs ;
- choix uniforme local entre les successeurs distincts.

### Holdout principal

- largeur : 3 ;
- 4 000 systèmes acceptés ;
- graines à partir de 1 000 000 ;
- aucune utilisation pendant la conception ou l’ajustement.

### Holdout hors distribution

- largeur : 4 ;
- 4 000 systèmes acceptés ;
- graines à partir de 2 000 000 ;
- même protocole, espace d’états plus grand.

## 4. Intervention et horizons

- état de référence : état nul ;
- perturbation : inversion du bit `width - 1` ;
- action dynamique : `Noop` ;
- horizon d’observation : 2 ;
- horizon de résultat : 6.

Les systèmes dont le recouvrement vaut déjà exactement 1 à l’horizon 2
sont exclus de l’analyse principale, car leur recouvrement futur est alors
déterminé exactement par la propriété de Markov.

Aucun autre système n’est exclu.

## 5. Variable cible

La cible continue est le recouvrement exact à l’horizon 6 :

\[
O_6 = \sum_x \min(P_6(x), Q_6(x)).
\]

Elle appartient à l’intervalle `[0, 1]` et vaut `1 - TV(P_6,Q_6)`.

Les rationnels exacts sont conservés pendant la propagation. La conversion
en `f64` n’intervient que pour l’apprentissage et les métriques.

## 6. Baseline appariée

La baseline ne peut utiliser que des informations disponibles jusqu’à
l’horizon 2 :

- entropie des chemins de référence aux profondeurs 1 et 2 ;
- entropie des chemins perturbés aux profondeurs 1 et 2 ;
- nombre d’états accessibles aux profondeurs 1 et 2 ;
- nombre de chemins aux profondeurs 1 et 2 ;
- mêmes mesures pour les trajectoires perturbées.

Elle ne peut utiliser :

- aucun recouvrement entre distributions ;
- aucune donnée d’un horizon supérieur à 2 ;
- aucun attracteur ou descripteur calculé sur le futur complet ;
- aucune information provenant de la cible.

## 7. Caractéristiques TDI-2

Le modèle challenger ajoute à la baseline :

- recouvrement exact à la profondeur 1 ;
- recouvrement exact à la profondeur 2 ;
- variation `O₂ - O₁`.

Ces caractéristiques représentent la géométrie prospective de la réponse
à l’intervention, et non une propriété orbitale complète.

## 8. Modèle prédictif

Les deux modèles utilisent exactement le même algorithme :

- régression ridge déterministe ;
- standardisation calculée uniquement sur l’apprentissage ;
- terme constant non pénalisé ;
- coefficient de régularisation fixé à `λ = 1`;
- résolution déterministe des équations normales ;
- prédictions bornées dans `[0,1]`.

Aucun hyperparamètre ne sera ajusté sur les holdouts.

## 9. Métriques

### Principale

- erreur quadratique moyenne, MSE.

### Secondaires

- erreur absolue moyenne, MAE ;
- coefficient de détermination `R²` ;
- corrélation de rang de Spearman.

## 10. Incertitude

Un bootstrap apparié déterministe de 2 000 réplications mesurera :

- amélioration MSE : `MSE_baseline - MSE_TDI`;
- amélioration MAE : `MAE_baseline - MAE_TDI`.

Graine bootstrap :

`0x5444_4932_434F_4E54`

## 11. Critère principal de réussite

TDI-2 est déclaré réussi sur le holdout principal uniquement si :

1. la réduction relative observée de MSE est supérieure ou égale à 5 % ;
2. la borne inférieure de l’intervalle bootstrap à 95 % de l’amélioration
   MSE est strictement positive ;
3. la borne inférieure de l’intervalle bootstrap à 95 % de l’amélioration
   MAE est strictement positive.

Le holdout de largeur 4 est confirmatoire. Il doit présenter une amélioration
MSE positive, mais son intervalle n’est pas inclus dans le critère principal.

## 12. Interprétation négative

Le résultat sera déclaré négatif si :

- le critère principal échoue ;
- le gain disparaît face à la baseline appariée ;
- la performance provient uniquement des systèmes presque ou déjà récupérés ;
- le gain ne se reproduit pas sur le holdout hors distribution.

Aucun seuil, horizon, échantillon ou critère ne sera modifié après le premier
calcul du holdout principal.
