# TDI-5 — Continuous Deficit Geometry

## Préenregistrement expérimental

### Statut

Ce document définit l’expérience TDI-5 avant toute implémentation de
l’évaluateur et avant toute génération des nouvelles populations.

Aucun résultat TDI-5 ne doit être produit avant :

1. le commit de ce préenregistrement ;
2. son push sur GitHub ;
3. l’implémentation séparée de l’évaluateur ;
4. le gel SHA-256 de l’évaluateur et du code scientifique ;
5. la validation complète de la CI.

---

## 1. Motivation

Les expériences antérieures ont étudié la capacité des observables TDI
précoces à prédire le recouvrement futur entre une dynamique de référence
et une dynamique perturbée.

Le recouvrement à l’horizon \(h\) est :

\[
O_h
=
\sum_{x \in \mathcal X}
\min\!\left(P_h(x),Q_h(x)\right).
\]

Il vérifie :

\[
O_h
=
1-\operatorname{TV}(P_h,Q_h).
\]

TDI-4 a utilisé une architecture à deux têtes :

\[
Z=\mathbf 1[O_6=1]
\]

et, conditionnellement à \(O_6<1\),

\[
U_6=-\log_2(1-O_6).
\]

Aucune récupération exacte \(O_6=1\) n’a été observée dans les 35 000
systèmes TDI-4. La tête binaire était donc dégénérée.

En revanche, la composante continue a montré :

- une réduction de perte combinée de 25,305777 % aux largeurs 3 et 4 ;
- une réduction de perte OOD de 50,033403 % à la largeur 5 ;
- un Spearman OOD passant de 0,467776309 à 0,615705212 ;
- un \(R^2\) OOD passant de -0,762078825 à 0,119549183.

TDI-5 étudie donc directement la géométrie continue du déficit.

---

## 2. Définitions

Le déficit de recouvrement est :

\[
D_h=1-O_h.
\]

La coordonnée logarithmique du déficit est :

\[
U_h=-\log_2(D_h)
=
-\log_2(1-O_h).
\]

L’expérience ne conserve que les systèmes vérifiant :

\[
D_h>0
\]

aux horizons cibles concernés.

Les observables TDI précoces sont :

\[
\operatorname{TDI}_2
=
\left(
O_1,\,
O_2,\,
O_2-O_1
\right).
\]

---

## 3. Question scientifique principale

Les observables TDI mesurées aux horizons 1 et 2 apportent-elles une
information prédictive supplémentaire et transférable sur la trajectoire
future du déficit, au-delà des variables structurelles et entropiques de
la baseline ?

La relation étudiée est :

\[
(O_1,O_2,O_2-O_1)
\longrightarrow
(U_3,U_4,U_5,U_6,U_8).
\]

---

## 4. Hypothèses

### Hypothèse principale

L’ajout des trois variables :

\[
O_1,\qquad O_2,\qquad O_2-O_1
\]

réduit significativement l’erreur de prédiction de \(U_6\) sur des systèmes
de largeurs 3 et 4 jamais vus pendant l’apprentissage.

### Hypothèse de transfert

Le gain se transfère à des largeurs non observées pendant l’apprentissage :

\[
w=5
\]

et :

\[
w=6.
\]

### Hypothèse de trajectoire

Le gain n’est pas limité à \(U_6\), mais reste observable sur plusieurs
horizons futurs :

\[
U_3,\ U_4,\ U_5,\ U_6,\ U_8.
\]

---

## 5. Génération des populations

Toutes les populations TDI-5 doivent être nouvelles et disjointes des
populations TDI-3 et TDI-4.

| Population | Largeur | Taille acceptée | Graine initiale |
|---|---:|---:|---:|
| apprentissage | 3 | 15 000 | 60 000 000 |
| apprentissage | 4 | 15 000 | 70 000 000 |
| holdout | 3 | 5 000 | 61 000 000 |
| holdout | 4 | 5 000 | 71 000 000 |
| OOD principal | 5 | 10 000 | 80 000 000 |
| OOD extrême | 6 | 5 000 | 90 000 000 |

Une graine désigne un candidat généré.

Un candidat rejeté consomme définitivement sa graine.

La génération continue jusqu’à atteindre le nombre exact de systèmes
acceptés pour chaque population.

Les graines finales exclusives et le nombre de candidats rejetés doivent
être consignés dans le résultat brut.

---

## 6. Horizons

Horizon d’observation :

\[
h_{\mathrm{obs}}=2.
\]

Horizons cibles :

\[
\mathcal H=\{3,4,5,6,8\}.
\]

La cible confirmatoire principale est :

\[
U_6.
\]

Les cibles \(U_3\), \(U_4\), \(U_5\) et \(U_8\) sont des analyses
secondaires préenregistrées.

---

## 7. Critères d’exclusion

Un système est exclu si l’une des conditions suivantes est satisfaite :

1. le recouvrement exact \(O_2\) vaut 1 ;
2. le déficit exact est nul à l’un des horizons cibles ;
3. une variable requise n’est pas finie après transformation ;
4. une opération exacte du moteur dynamique échoue ;
5. la génération viole les invariants structurels déjà imposés par TDI-4.

Aucune exclusion ne peut dépendre :

- des prédictions ;
- des erreurs du modèle ;
- de la valeur des variables TDI relativement à la baseline ;
- de l’amélioration observée.

---

## 8. Variables explicatives

### Baseline

La baseline conserve exactement les 13 variables structurelles utilisées
par TDI-4.

Aucune nouvelle variable structurelle ne doit être ajoutée.

### Challenger principal

Le challenger contient les mêmes 13 variables plus :

\[
O_1,
\]

\[
O_2,
\]

\[
O_2-O_1.
\]

Le challenger possède donc 16 variables.

---

## 9. Ablations secondaires

Les modèles secondaires sont :

\[
M_0
=
\text{baseline},
\]

\[
M_1
=
\text{baseline}+O_1,
\]

\[
M_2
=
\text{baseline}+O_1+O_2,
\]

\[
M_3
=
\text{baseline}+O_1+O_2+(O_2-O_1).
\]

Le modèle confirmatoire principal est \(M_3\).

Les modèles \(M_1\) et \(M_2\) ne peuvent pas remplacer \(M_3\) pour les
verdicts principaux.

---

## 10. Modèles

Pour chaque horizon cible, la baseline et le challenger utilisent une
régression ridge séparée.

La fonction objectif est :

\[
\min_{\beta_0,\beta}
\sum_{i=1}^{N}
\left(
\widetilde U_{h,i}
-
\beta_0
-
\widetilde X_i^\top\beta
\right)^2
+
\lambda\|\beta\|_2^2.
\]

La pénalisation est fixée à :

\[
\lambda=1.
\]

L’intercept n’est pas pénalisé.

La baseline et le challenger doivent utiliser :

- le même algorithme ;
- le même ordre d’accumulation ;
- la même précision numérique ;
- la même pénalisation ;
- les mêmes populations ;
- les mêmes transformations de cible.

La seule différence est l’ajout des trois variables TDI au challenger.

---

## 11. Normalisation

Pour chaque variable explicative :

\[
\widetilde X_j
=
\frac{X_j-\mu_j}{s_j}.
\]

Pour chaque horizon cible :

\[
\widetilde U_h
=
\frac{U_h-\mu_{U_h}}{s_{U_h}}.
\]

Toutes les moyennes et échelles sont apprises uniquement sur l’ensemble
d’apprentissage combiné des largeurs 3 et 4.

Elles sont ensuite figées pour :

- le holdout largeur 3 ;
- le holdout largeur 4 ;
- l’OOD largeur 5 ;
- l’OOD largeur 6.

Une échelle nulle est remplacée par 1.

---

## 12. Métrique principale

Pour \(U_6\), la métrique principale est la MSE dans l’espace standardisé :

\[
\operatorname{MSE}_{U_6}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\widetilde U_{6,i}
-
\widehat{\widetilde U}_{6,i}
\right)^2.
\]

L’amélioration absolue est :

\[
\Delta_{U_6}
=
\operatorname{MSE}_{U_6}^{\mathrm{baseline}}
-
\operatorname{MSE}_{U_6}^{\mathrm{TDI}}.
\]

La réduction relative est :

\[
G_{U_6}
=
\frac{
\operatorname{MSE}_{U_6}^{\mathrm{baseline}}
-
\operatorname{MSE}_{U_6}^{\mathrm{TDI}}
}{
\operatorname{MSE}_{U_6}^{\mathrm{baseline}}
}.
\]

---

## 13. Métriques secondaires

Pour chaque horizon et chaque population, l’évaluateur rapporte :

- MSE standardisée ;
- MAE standardisée ;
- \(R^2\) ;
- corrélation de Spearman ;
- biais moyen ;
- moyenne observée ;
- moyenne prédite ;
- pente de calibration ;
- intercept de calibration.

---

## 14. Reconstruction du recouvrement

La reconstruction est :

\[
\widehat O_h
=
1-2^{-\widehat U_h}.
\]

L’évaluateur rapporte également, dans l’espace de \(O_h\) :

- MSE ;
- MAE ;
- \(R^2\) ;
- Spearman ;
- biais moyen ;
- calibration ;
- proportion de prédictions ramenées aux bornes numériques.

Ces métriques sont secondaires.

---

## 15. Bootstrap apparié

Les différences de perte sont évaluées par bootstrap apparié.

Nombre de réplications :

\[
B=2000.
\]

Graine :

\[
\texttt{0x5444\_4935\_4344\_4745}.
\]

Chaque réplication rééchantillonne les indices des exemples avec remise.

Les prédictions baseline et challenger restent appariées pour chaque indice.

L’intervalle à 95 % utilise les quantiles empiriques 2,5 % et 97,5 %.

Aucun réentraînement n’est effectué à l’intérieur du bootstrap.

---

## 16. Critère principal TDI-5A

Le critère principal est évalué sur le holdout combiné des largeurs 3 et 4
pour la cible \(U_6\).

TDI-5A est déclaré **RÉUSSI** si toutes les conditions suivantes sont
satisfaites :

1. réduction relative combinée :

\[
G_{U_6}^{(3+4)}\geq 10\%;
\]

2. borne inférieure de l’IC bootstrap à 95 % de
   \(\Delta_{U_6}^{(3+4)}\) strictement positive ;

3. amélioration ponctuelle strictement positive en largeur 3 ;

4. amélioration ponctuelle strictement positive en largeur 4 ;

5. borne inférieure de l’IC bootstrap de largeur 3 strictement positive ;

6. borne inférieure de l’IC bootstrap de largeur 4 strictement positive ;

7. Spearman du challenger strictement supérieur à celui de la baseline sur
   le holdout combiné ;

8. Spearman du challenger strictement positif séparément aux largeurs 3 et 4 ;

9. biais absolu standardisé du challenger ne dépassant pas celui de la
   baseline de plus de 0,02 sur le holdout combiné.

Sinon :

`CRITÈRE PRINCIPAL TDI-5A : ÉCHOUÉ`

En cas de réussite :

`CRITÈRE PRINCIPAL TDI-5A : RÉUSSI`

---

## 17. Critère de transfert TDI-5B

TDI-5B concerne la largeur 5 pour la cible \(U_6\).

Il est déclaré **RÉUSSI** si toutes les conditions suivantes sont
satisfaites :

1. réduction relative :

\[
G_{U_6}^{(5)}\geq 20\%;
\]

2. borne inférieure de l’IC bootstrap à 95 % strictement positive ;

3. Spearman challenger strictement positif ;

4. Spearman challenger supérieur ou égal à celui de la baseline ;

5. \(R^2\) challenger strictement supérieur au \(R^2\) baseline ;

6. biais absolu challenger strictement inférieur au biais absolu baseline ;

7. amélioration ponctuelle strictement positive de la MSE reconstruite ;

8. amélioration ponctuelle strictement positive de la MAE reconstruite.

Sinon :

`CRITÈRE TRANSFERT TDI-5B : ÉCHOUÉ`

En cas de réussite :

`CRITÈRE TRANSFERT TDI-5B : RÉUSSI`

---

## 18. Critère de transfert extrême TDI-5C

TDI-5C concerne la largeur 6 pour la cible \(U_6\).

Il est déclaré **RÉUSSI** si toutes les conditions suivantes sont
satisfaites :

1. amélioration ponctuelle :

\[
\Delta_{U_6}^{(6)}>0;
\]

2. borne inférieure de l’IC bootstrap à 95 % strictement positive ;

3. Spearman challenger strictement positif ;

4. Spearman challenger supérieur ou égal à celui de la baseline ;

5. biais absolu challenger inférieur ou égal au biais absolu baseline ;

6. amélioration ponctuelle strictement positive de la MSE reconstruite.

Sinon :

`CRITÈRE TRANSFERT EXTRÊME TDI-5C : ÉCHOUÉ`

En cas de réussite :

`CRITÈRE TRANSFERT EXTRÊME TDI-5C : RÉUSSI`

---

## 19. Critère de trajectoire TDI-5D

Les horizons secondaires sont :

\[
\{3,4,5,8\}.
\]

Pour chaque horizon, on calcule l’amélioration combinée des largeurs 3 et 4.

TDI-5D est déclaré **RÉUSSI** si :

1. au moins trois des quatre horizons présentent une amélioration ponctuelle
   strictement positive ;

2. \(U_8\) présente une amélioration ponctuelle strictement positive ;

3. aucune cible ne présente une dégradation relative supérieure à 5 % ;

4. la moyenne arithmétique des quatre réductions relatives est strictement
   positive.

Sinon :

`CRITÈRE TRAJECTOIRE TDI-5D : ÉCHOUÉ`

En cas de réussite :

`CRITÈRE TRAJECTOIRE TDI-5D : RÉUSSI`

Ce critère est secondaire et ne remplace pas TDI-5A.

---

## 20. Comparateur direct secondaire

Un comparateur ridge prédit directement \(O_6\) avec :

- les 13 variables de baseline ;
- puis les 13 variables plus les trois variables TDI.

Ce comparateur est secondaire.

Ses résultats ne peuvent pas modifier les verdicts TDI-5A, TDI-5B,
TDI-5C ou TDI-5D.

---

## 21. Déterminisme

L’évaluateur doit garantir :

- génération déterministe par graine ;
- ordre fixe d’itération ;
- ordre fixe d’accumulation flottante ;
- absence de parallélisme dans les calculs qui modifient l’ordre des sommes ;
- bootstrap déterministe ;
- résultats textuels déterministes ;
- exécution hors ligne ;
- absence de dépendance réseau ;
- absence de sélection après observation.

---

## 22. Sortie requise

Le résultat brut doit inclure :

1. identité Git ;
2. versions Rust et Cargo ;
3. tailles des populations ;
4. nombres d’exclusions ;
5. graines finales exclusives ;
6. statistiques de \(U_h\) pour chaque horizon et population ;
7. normalisations apprises ;
8. coefficients de tous les modèles confirmatoires ;
9. résultats de \(M_0\), \(M_1\), \(M_2\) et \(M_3\) ;
10. métriques dans l’espace de \(U_h\) ;
11. métriques reconstruites dans l’espace de \(O_h\) ;
12. intervalles bootstrap ;
13. comparateur direct secondaire ;
14. les quatre lignes finales exactes :

`CRITÈRE PRINCIPAL TDI-5A : RÉUSSI|ÉCHOUÉ`

`CRITÈRE TRANSFERT TDI-5B : RÉUSSI|ÉCHOUÉ`

`CRITÈRE TRANSFERT EXTRÊME TDI-5C : RÉUSSI|ÉCHOUÉ`

`CRITÈRE TRAJECTOIRE TDI-5D : RÉUSSI|ÉCHOUÉ`

---

## 23. Interdictions avant gel

Avant le gel de l’évaluateur, il est interdit de :

- générer les populations TDI-5 complètes ;
- consulter des métriques TDI-5 ;
- modifier les critères en fonction d’un résultat intermédiaire ;
- tester plusieurs valeurs de \(\lambda\) ;
- sélectionner les graines ;
- changer la baseline ;
- retirer une largeur défavorable ;
- retirer un horizon défavorable ;
- modifier les seuils de réussite.

Les tests unitaires synthétiques et les tests de déterminisme sont autorisés,
à condition qu’ils ne génèrent pas les populations préenregistrées.

---

## 24. Ordre d’exécution

L’ordre immuable est :

1. commit du préenregistrement ;
2. push de la branche ;
3. implémentation de l’évaluateur ;
4. tests unitaires ;
5. gel SHA-256 de l’évaluateur ;
6. gel du code scientifique ;
7. commit et push du gel ;
8. validation de la CI ;
9. première exécution unique ;
10. archivage des résultats ;
11. tag immuable ;
12. PR vers `main`.
