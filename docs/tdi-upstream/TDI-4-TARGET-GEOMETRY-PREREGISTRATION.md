# TDI-4 — Préenregistrement de l’évaluation Target Geometry

## 1. Statut

Ce document est établi avant :

- l’implémentation de l’évaluateur TDI-4 ;
- la génération des populations TDI-4 ;
- l’observation des métriques TDI-4 ;
- toute adaptation des seuils de succès.

Les résultats TDI-3 sont connus et motivent la formulation de TDI-4, mais
aucune population TDI-3 ne sera réutilisée pour entraîner ou évaluer
TDI-4.

## 2. Motivation

TDI-3 a montré :

- un signal prédictif favorable en largeur 3 ;
- une détérioration des erreurs en largeur 4 ;
- une réduction importante du biais et de la MSE en largeur 5 ;
- des valeurs négatives de R² et de Spearman en largeur 5 ;
- une concentration croissante de la cible `O₆` près de `1`.

La moyenne observée de `O₆` était notamment proche de :

- `0.9836` en largeur 3 ;
- `0.9990` en largeur 4 ;
- `0.99988` en largeur 5.

Une régression directe de `O₆` confond donc deux phénomènes :

1. l’atteinte exacte de `O₆ = 1` ;
2. la taille du déficit résiduel lorsque `O₆ < 1`.

TDI-4 teste une géométrie de cible en deux parties afin de séparer ces
phénomènes.

## 3. Hypothèses

### H4.1 — Géométrie de cible

Une représentation en deux parties de la récupération à l’horizon 6 est
plus stable entre les largeurs qu’une régression directe de `O₆`.

### H4.2 — Valeur ajoutée des variables TDI

À architecture de modèle et géométrie de cible identiques, les variables
TDI observées à l’horizon 2 améliorent la prédiction par rapport aux seules
variables de baseline.

### H4.3 — Transfert hors distribution

Le gain du challenger entraîné sur les largeurs 3 et 4 reste positif sur
une population intacte de largeur 5.

## 4. Systèmes et horizons

Pour chaque système :

- état de référence : état nul de la largeur considérée ;
- perturbation : inversion du dernier nœud ;
- action future : `Noop` ;
- horizon d’observation : `2` ;
- horizon de résultat : `6`.

Les distributions sont propagées avec l’arithmétique rationnelle exacte
fondée sur `BigUint`.

## 5. Exclusion

Un système est exclu si les distributions sont déjà exactement identiques
à l’horizon d’observation :

`O₂ = 1`.

Cette règle est identique à celle de TDI-3 et évite une fuite logique
directe de la cible.

Les systèmes tels que `O₆ = 1` sont conservés. Ils constituent précisément
la composante binaire du nouveau problème.

Aucune autre exclusion dépendant de la cible n’est autorisée.

## 6. Populations et graines

Les plages de graines sont entièrement nouvelles.

| Population | Largeur | Effectif retenu | Première graine |
|---|---:|---:|---:|
| entraînement | 3 | 10 000 | 30 000 000 |
| holdout | 3 | 5 000 | 31 000 000 |
| entraînement | 4 | 10 000 | 40 000 000 |
| holdout | 4 | 5 000 | 41 000 000 |
| holdout OOD | 5 | 5 000 | 50 000 000 |

La génération continue au-delà de la première graine jusqu’à obtention du
nombre préenregistré de systèmes retenus.

Les populations d’entraînement des largeurs 3 et 4 sont combinées.

Les holdouts ne sont jamais utilisés pour :

- choisir les variables ;
- choisir les transformations ;
- régler les seuils ;
- sélectionner le modèle ;
- ajuster `lambda`.

## 7. Variables explicatives

### 7.1 Baseline appariée

La baseline conserve exactement les 13 variables utilisées dans TDI-3 :

- entropies normalisées ;
- nombres d’états accessibles normalisés ;
- logarithmes des nombres de chemins ;
- caractéristiques de largeur.

La largeur est incluse comme variable explicative de la même façon que dans
TDI-3.

### 7.2 Challenger TDI-4

Le challenger contient les 13 variables de baseline et les trois variables
TDI déjà définies :

- `O₁` ;
- `O₂` ;
- `O₂ - O₁`.

Aucune variable supplémentaire ne peut être ajoutée après le gel.

## 8. Géométrie de cible en deux parties

Soit :

`O₆ ∈ [0, 1]`

le recouvrement exact à l’horizon 6.

### 8.1 Composante de récupération exacte

La cible binaire est :

`Z = 1` si `O₆ = 1`, sinon `Z = 0`.

Un premier modèle prédit :

`p = P(Z = 1 | X)`.

La prédiction linéaire est bornée dans `[0, 1]`.

### 8.2 Composante conditionnelle

Pour les seuls systèmes tels que `O₆ < 1`, le déficit est :

`D = 1 - O₆`.

La cible conditionnelle est :

`U = -log₂(D)`.

Une grande valeur de `U` indique un déficit très faible.

`U` est standardisé à partir de la moyenne et de l’écart-type calculés
exclusivement sur les systèmes d’entraînement non complètement récupérés.

### 8.3 Reconstruction

La prédiction finale de recouvrement est :

`Ô₆ = 1 - (1 - p̂) × 2^(-Û)`.

La valeur reconstruite est bornée dans `[0, 1]`.

La même procédure est appliquée à la baseline et au challenger.

## 9. Modèles

Les deux composantes utilisent une régression ridge déterministe.

Paramètre fixé :

`lambda = 1`.

Deux têtes sont entraînées pour chaque famille de variables :

1. tête binaire sur `Z` ;
2. tête conditionnelle sur `U`, uniquement pour les systèmes où `Z = 0`.

Aucune recherche d’hyperparamètres n’est autorisée.

L’ordre d’accumulation flottante reste déterministe.

## 10. Perte composite principale

La perte principale est :

`L = 0.5 × Brier(Z, p̂) + 0.5 × MSE(U_std, Û_std | Z = 0)`.

La seconde composante est calculée uniquement sur les observations non
complètement récupérées.

Si une population d’évaluation ne contient aucun système non complètement
récupéré, l’évaluation doit s’interrompre avec une erreur explicite. Aucun
terme nul artificiel ne sera substitué.

Une amélioration est définie par :

`ΔL = L_baseline - L_challenger`.

Une valeur positive favorise TDI-4.

## 11. Métriques rapportées

Pour chaque largeur et pour le holdout combiné seront rapportés :

### Tête binaire

- Brier score ;
- taux observé de récupération exacte ;
- moyenne de `p̂` ;
- calibration intercept ;
- calibration slope ;
- fraction de prédictions bornées à `0` ;
- fraction de prédictions bornées à `1`.

### Tête conditionnelle

- MSE sur `U` standardisé ;
- MAE sur `U` standardisé ;
- R² conditionnel ;
- Spearman conditionnel.

### Reconstruction de `O₆`

- MSE ;
- MAE ;
- R² ;
- Spearman ;
- biais moyen ;
- moyenne observée ;
- moyenne prédite ;
- calibration intercept ;
- calibration slope.

### Perte principale

- perte composite `L` ;
- amélioration absolue `ΔL` ;
- réduction relative de `L`.

## 12. Comparateur secondaire

Un comparateur secondaire reproduit, sur les nouvelles populations, la
régression ridge directe de `O₆` utilisée dans TDI-3.

Ce comparateur sert uniquement à déterminer si la géométrie en deux parties
améliore la stabilité.

Il ne peut pas remplacer la baseline principale et ne participe pas au
verdict TDI-4A ou TDI-4B.

## 13. Bootstrap

Le bootstrap est :

- apparié ;
- déterministe ;
- effectué séparément pour chaque population ;
- composé de 2 000 réplications.

Graine fixe :

`0x5444_4934_4745_4F4D`.

Des intervalles percentile à 95 % sont calculés pour :

- l’amélioration de perte composite ;
- l’amélioration du Brier score ;
- l’amélioration de la MSE conditionnelle ;
- l’amélioration de la MSE reconstruite ;
- l’amélioration de la MAE reconstruite.

## 14. Critère principal TDI-4A

TDI-4A est déclaré **RÉUSSI** seulement si toutes les conditions suivantes
sont satisfaites sur les holdouts des largeurs 3 et 4 :

1. réduction relative de la perte composite combinée supérieure ou égale à
   `5 %` ;
2. borne inférieure de l’IC 95 % de `ΔL` combiné strictement positive ;
3. borne inférieure de l’IC 95 % de l’amélioration du Brier combiné
   strictement positive ;
4. borne inférieure de l’IC 95 % de `ΔL` en largeur 3 strictement positive ;
5. borne inférieure de l’IC 95 % de `ΔL` en largeur 4 strictement positive ;
6. amélioration ponctuelle positive de la MSE reconstruite combinée ;
7. amélioration ponctuelle positive de la MAE reconstruite combinée ;
8. Spearman conditionnel positif pour le challenger dans les deux largeurs.

L’échec d’une seule condition entraîne le verdict **ÉCHOUÉ**.

## 15. Critère de transfert TDI-4B

TDI-4B est déclaré **RÉUSSI** seulement si toutes les conditions suivantes
sont satisfaites sur le holdout intact de largeur 5 :

1. borne inférieure de l’IC 95 % de `ΔL` strictement positive ;
2. réduction relative de la MSE reconstruite supérieure ou égale à `5 %` ;
3. borne inférieure de l’IC 95 % de l’amélioration de MSE reconstruite
   strictement positive ;
4. borne inférieure de l’IC 95 % de l’amélioration du Brier score
   strictement positive ;
5. Spearman conditionnel du challenger strictement positif ;
6. Spearman conditionnel du challenger supérieur ou égal à celui de la
   baseline ;
7. biais absolu reconstruit du challenger inférieur à celui de la baseline.

L’échec d’une seule condition entraîne le verdict **ÉCHOUÉ**.

## 16. Analyses secondaires

Les analyses suivantes sont rapportées mais ne modifient pas les verdicts :

- résultats séparés pour chaque tête ;
- comparaison avec la régression directe TDI-3 ;
- coefficients des modèles ;
- taux de récupération exacte par largeur ;
- distribution de `U` ;
- saturation des prédictions ;
- résultats par déciles de déficit observé.

Aucune analyse secondaire ne peut être requalifiée comme critère principal
après observation des résultats.

## 17. Politique en cas d’erreur

Une erreur d’implémentation ou une limite numérique découverte avant la
production de métriques peut être corrigée si :

- l’échec initial est conservé ;
- la cause est documentée ;
- les critères et populations ne sont pas modifiés ;
- le code corrigé est regélé avant une nouvelle exécution.

Une fois les métriques produites, aucune modification du protocole,
de l’évaluateur ou des critères n’est autorisée pour cette expérience.

## 18. Verdicts attendus dans la sortie

L’évaluateur doit produire exactement deux lignes finales :

`CRITÈRE PRINCIPAL TDI-4A : RÉUSSI|ÉCHOUÉ`

`CRITÈRE TRANSFERT TDI-4B : RÉUSSI|ÉCHOUÉ`
