# TDI-3 — Préenregistrement du transfert inter-largeurs

Date de gel : 12 juillet 2026.

## 1. Motivation

TDI-2 a établi, sur des systèmes de largeur 3, qu’un profil de
recouvrement distributionnel observé jusqu’à l’horizon 2 apporte une
information prédictive supplémentaire par rapport à une baseline
entropique et topologique appariée.

Sur le holdout de largeur 4, TDI-2 a conservé un gain relatif de MSE, mais
les prédictions absolues étaient très mal calibrées :

- R² fortement négatif ;
- corrélation de rang négative ;
- erreurs absolues élevées.

TDI-3 teste explicitement si un apprentissage multi-largeurs et des
caractéristiques normalisées permettent une généralisation inter-largeurs
sans supprimer le signal incrémental TDI.

## 2. Questions principales

### TDI-3A — Généralisation multi-largeurs connue

Sur des systèmes inédits de largeurs 3 et 4, le challenger TDI améliore-t-il
la prédiction du recouvrement à l’horizon 6 par rapport à une baseline
appariée utilisant le même horizon d’observation ?

### TDI-3B — Transfert vers une largeur jamais observée

Un modèle entraîné uniquement sur les largeurs 3 et 4 conserve-t-il :

- un gain prédictif relatif ;
- une corrélation de rang positive ;
- une validité prédictive absolue ;

sur des systèmes de largeur 5 jamais utilisés pendant l’apprentissage ?

## 3. Génération des systèmes

Les systèmes suivent le protocole de TDI-2 :

- systèmes finis à transitions branchantes ;
- ensemble non vide de successeurs pour chaque état ;
- choix uniforme entre les successeurs distincts ;
- génération déterministe par `SplitMix64` ;
- état de référence nul ;
- perturbation par inversion du bit `width - 1` ;
- action dynamique `Noop`.

Les propagations distributionnelles utilisent des rationnels exacts.
La conversion en `f64` intervient uniquement pour l’apprentissage et les
métriques.

## 4. Populations gelées

### Apprentissage

- largeur 3 : 8 000 systèmes acceptés ;
- largeur 4 : 8 000 systèmes acceptés ;
- total : 16 000 systèmes ;
- largeur 3 : graines à partir de `0` ;
- largeur 4 : graines à partir de `10_000_000`.

### Holdout largeur 3

- 4 000 systèmes acceptés ;
- graines à partir de `1_000_000`.

### Holdout largeur 4

- 4 000 systèmes acceptés ;
- graines à partir de `11_000_000`.

### Holdout hors distribution largeur 5

- 4 000 systèmes acceptés ;
- graines à partir de `20_000_000`;
- aucune donnée de largeur 5 ne peut intervenir dans la conception,
  l’apprentissage, la standardisation ou la sélection du modèle.

Les graines sont consommées séquentiellement jusqu’à obtention du nombre
de systèmes acceptés prévu.

## 5. Horizons et exclusion

- horizon d’observation : 2 ;
- horizon cible : 6 ;
- cible : recouvrement exact `O6` entre les distributions de référence et
  perturbée.

Les systèmes pour lesquels `O2 = 1` sont exclus, car leur recouvrement futur
est déjà déterminé par la propriété de Markov.

Aucune autre exclusion n’est autorisée.

Le nombre de systèmes examinés, acceptés et exclus doit être publié pour
chaque largeur.

## 6. Cible

La cible continue est :

O6 = somme sur x de min(P6(x), Q6(x)).

Elle appartient à l’intervalle `[0, 1]` et correspond à `1 - TV(P6,Q6)`.

## 7. Baseline TDI-3 appariée

La baseline ne peut utiliser que des informations disponibles jusqu’à
l’horizon 2.

Elle comprend, pour les trajectoires de référence et perturbées :

- entropies aux profondeurs 1 et 2 ;
- fractions d’états accessibles aux profondeurs 1 et 2 ;
- logarithmes des nombres de chemins aux profondeurs 1 et 2 ;
- largeur du système comme caractéristique partagée.

Normalisations gelées :

- entropie divisée par `ln(2^width)` lorsque le dénominateur est non nul ;
- nombre d’états accessibles divisé par `2^width` ;
- nombre de chemins transformé par `ln(1 + count)` ;
- largeur représentée par sa valeur entière convertie en `f64`.

La baseline ne peut utiliser :

- aucun recouvrement entre distributions ;
- aucune donnée postérieure à l’horizon 2 ;
- aucune caractéristique orbitale calculée sur le futur complet ;
- aucune information dérivée de la cible.

## 8. Caractéristiques TDI-3

Le challenger utilise toutes les caractéristiques de la baseline et ajoute :

- recouvrement exact `O1` ;
- recouvrement exact `O2` ;
- variation `O2 - O1`.

Ces trois caractéristiques sont déjà sans dimension et ne dépendent pas
directement du nombre d’états.

La largeur est disponible de manière identique pour la baseline et le
challenger.

## 9. Modèle prédictif

Les deux modèles utilisent exactement le même algorithme :

- régression ridge déterministe ;
- standardisation calculée exclusivement sur l’ensemble d’apprentissage ;
- terme constant non pénalisé ;
- coefficient de régularisation fixé à `lambda = 1` ;
- équations normales résolues de manière déterministe ;
- prédictions bornées dans `[0, 1]`.

Aucun hyperparamètre ne peut être ajusté après calcul des holdouts.

Aucune calibration post-hoc ne sera ajustée sur les holdouts.

## 10. Métriques

Pour chaque largeur et pour le holdout combiné des largeurs 3 et 4 :

- MSE ;
- MAE ;
- R² ;
- corrélation de rang de Spearman ;
- biais moyen : moyenne prédiction moins moyenne cible ;
- calibration dans le large : moyenne prédite et moyenne observée ;
- pente et intercept de calibration obtenus en régressant la cible sur la
  prédiction.

Les métriques sont calculées séparément pour la baseline et le challenger.

## 11. Incertitude

Un bootstrap apparié déterministe de 2 000 réplications est appliqué :

- au holdout largeur 3 ;
- au holdout largeur 4 ;
- au holdout combiné largeurs 3 et 4 ;
- au holdout hors distribution largeur 5.

Il mesure :

- `MSE_baseline - MSE_TDI` ;
- `MAE_baseline - MAE_TDI`.

Graine bootstrap :

`0x5444_4933_494E_5445`.

Les intervalles utilisent les quantiles empiriques 2,5 % et 97,5 %.

## 12. Critère principal TDI-3A

TDI-3A est déclaré réussi uniquement si les conditions suivantes sont
toutes satisfaites sur le holdout combiné des largeurs 3 et 4 :

1. réduction relative de MSE supérieure ou égale à 5 % ;
2. borne inférieure de l’IC 95 % de l’amélioration MSE strictement positive ;
3. borne inférieure de l’IC 95 % de l’amélioration MAE strictement positive ;
4. amélioration MSE observée strictement positive séparément en largeur 3 ;
5. amélioration MSE observée strictement positive séparément en largeur 4 ;
6. Spearman du challenger strictement positif séparément en largeurs 3 et 4 ;
7. R² du challenger strictement positif séparément en largeurs 3 et 4.

## 13. Critère de transfert TDI-3B

TDI-3B est déclaré réussi sur la largeur 5 uniquement si :

1. l’amélioration MSE observée est strictement positive ;
2. la borne inférieure de l’IC 95 % de l’amélioration MSE est strictement
   positive ;
3. la borne inférieure de l’IC 95 % de l’amélioration MAE est strictement
   positive ;
4. le R² du challenger est strictement positif ;
5. le Spearman du challenger est strictement positif ;
6. la valeur absolue du biais moyen du challenger est inférieure à celle de
   la baseline.

Une réduction relative positive accompagnée d’un R² ou d’un Spearman
négatif ne sera pas présentée comme une généralisation valide.

## 14. Analyses secondaires

Seront publiées sans modifier le verdict principal :

- résultats séparés par largeur ;
- distribution des cibles ;
- distribution des prédictions ;
- proportion de prédictions bornées à 0 ou à 1 ;
- coefficients standardisés des deux modèles ;
- résultats par déciles de la cible ;
- résultats par déciles de `O2` ;
- matrices de corrélation des caractéristiques ;
- temps d’exécution et nombre de systèmes rejetés.

## 15. Contrôles logiciels obligatoires

Avant l’évaluation finale :

- génération déterministe vérifiée par tests ;
- longueurs exactes des vecteurs de caractéristiques ;
- absence de données des holdouts dans la standardisation ;
- absence de chevauchement des plages de graines ;
- valeurs normalisées finies ;
- prédictions comprises dans `[0, 1]` ;
- bootstrap reproductible ;
- vérification du hash du présent préenregistrement dans la CI.

## 16. Interdictions après gel

Après création du hash :

- aucun changement des populations ;
- aucun changement des graines ;
- aucun changement des horizons ;
- aucun changement des caractéristiques ;
- aucun changement de lambda ;
- aucun changement des seuils de réussite ;
- aucun examen des résultats de holdout avant gel complet du code
  d’évaluation.

Toute correction de bug découverte après le premier calcul doit être
documentée, testée et accompagnée d’une nouvelle exécution complète.

## 17. Interprétation négative

Le résultat est négatif pour TDI-3A si son critère principal échoue.

Le résultat est négatif pour TDI-3B si au moins une condition du transfert
largeur 5 échoue.

Les verdicts TDI-3A et TDI-3B sont indépendants et doivent être publiés
séparément.

Un succès TDI-3A associé à un échec TDI-3B signifierait que le signal TDI
est exploitable dans une population multi-largeurs connue, sans preuve de
transfert vers une taille nouvelle.

## 18. Livrables prévus

- binaire `tdi-interwidth-continuous` ;
- script `scripts/reproduce-tdi3.sh` ;
- log déterministe dans `results/` ;
- rapport `docs/TDI-3-INTERWIDTH-RESULTS.md` ;
- hash SHA-256 du préenregistrement ;
- hash SHA-256 du résultat de référence ;
- tests unitaires et contrôles CI ;
- release Git dédiée si l’évaluation est terminée.
