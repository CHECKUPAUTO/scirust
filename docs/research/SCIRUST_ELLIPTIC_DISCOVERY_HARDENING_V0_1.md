# SciRust Elliptic Discovery — durcissement borné v0.1

## Statut

Ce document définit la phase 7 avant toute modification du code de production.
Il complète les phases 0 à 6 sans élargir leur périmètre : seules des courbes
jouets, générées localement dans des corps premiers bornés, sont représentables.

## Motivation vérifiable

Les revues des phases précédentes ont relevé quatre comportements qui doivent
être traités comme des défauts de sûreté expérimentale, et non comme des
optimisations facultatives :

1. le parcours de falsification continuait après le budget déclaré ;
2. la grammaire pouvait matérialiser un très grand ensemble avant d'appliquer le
   budget de candidats ;
3. certaines identités connues dépendaient de l'ordre arbitraire des arbres
   syntaxiques ;
4. le contrôle de l'automorphisme de type \(j=0\) utilisait une échelle qui
   n'était pas une racine cubique de l'unité dans le premier corps rencontré.

La validation de cette phase a également révélé un défaut d'intégration déjà
présent sur `master` : le rapport canonique utilisait une accolade ouvrante non
échappée dans une chaîne de format Rust. La phase rétablit `{{` et vérifie que
le rapport JSON commence bien par l'accolade littérale attendue.

Les observations sont traçables dans les commentaires de revue de
[#918](https://github.com/Memorithm/scirust/pull/918#discussion_r3696552709),
[#919](https://github.com/Memorithm/scirust/pull/919#discussion_r3696555831)
et [#920](https://github.com/Memorithm/scirust/pull/920#discussion_r3696556768).

## Objectif

La phase 7 rend l'exécution bornée au sens opérationnel : une limite déclarée
borne effectivement les appels à une relation et la mémoire de travail de la
génération. Elle ne prétend ni augmenter la portée mathématique du moteur, ni
produire une découverte.

## Décisions de conception

### Falsification bornée et observable

Une nouvelle primitive de falsification bornée retourne :

- le premier contre-exemple canonique éventuel ;
- le nombre exact de tuples évalués ;
- un résultat sans contre-exemple lorsque la limite est atteinte.

Le parcours s'arrête avant d'appeler la relation sur un tuple situé au-delà de
la limite. Une porte G2, G3 ou G6 ne passe que lorsque tous les tuples requis
ont été évalués sans contre-exemple. Une limite atteinte avant cette couverture
produit `InsufficientCoverage`.

L'API historique de recherche exhaustive reste disponible et délègue à cette
primitive avec une limite maximale ; elle ne change donc pas le sens des
rapports existants.

### Génération pilotée par le budget

`generate_relations` reste une génération locale et déterministe, mais son
espace de travail est désormais déterminé avant l'expansion : le nombre
d'expressions nécessaires aux égalités de points est la plus petite valeur
\(n\) telle que \(n(n+1)/2\) couvre le budget demandé. Les bornes publiques du
crate sont appliquées aussi à cette fonction afin qu'un appel direct ne puisse
pas contourner `SearchPlan`.

La sélection est une séquence canonique fixe et non une promesse de
matérialiser l'univers complet de la grammaire. Elle intercale, dans cet ordre :

1. égalités de points ;
2. prédicats d'infini ;
3. égalités du coefficient \(a\) ;
4. égalités du j-invariant.

Ainsi, les quatre variantes publiques de `Relation` sont atteignables dans une
recherche bornée. Les constantes de \(j\) incluent explicitement `0` et `1728`.
L'ordre et les plafonds font partie du comportement reproductible et sont
encodés indirectement dans les reçus de phase 6.

Comme la séquence de candidats change, la phase 7 passe les domaines canoniques
de plan, d'évaluation et de reçu à `V2`, et la version du crate à `0.2.0`.
Il n'existe aucun décodeur de reçu externe : cette séparation de domaine suffit
à empêcher la confusion d'empreintes entre les deux comportements.

### Normalisation structurelle des lois connues

La signature de négation reconnaît les deux côtés d'une égalité. La signature
d'inverse additif reconnaît les deux ordres des sommants. Ces règles ne
changent pas l'évaluation exacte : elles évitent seulement qu'une loi déjà
cataloguée soit présentée comme relation non reconnue à cause de l'ordre de
construction de l'arbre.

### Contrôle conditionnel \(j=0\)

Le contrôle négatif universel est évalué dans \(\mathbb{F}_7\) avec
\(\zeta=2\), car \(2^3 = 1 \pmod 7\) et \(2 \ne 1\). Il ignore les autres
corps et vérifie d'abord les courbes `a = 0`; son premier contre-exemple doit
donc provenir d'une courbe du même corps avec `a != 0`. Le test démontre que
le contrôle distingue la condition \(j=0\) de la simple invalidité d'une
constante d'échelle.

### Frontière de statut inchangée

`CandidateUnclassified` reste un statut final de `review_candidate`, après une
tentative de justification et une revue humaine complète. Le passage de G6 ne
peut donc pas l'attribuer seul : l'état automatisé demeure
`NeedsLiteratureReview`. Cette décision conserve la frontière établie en phase
5 entre résultat automatisé et absence de conflit après revue indépendante.

## Invariants

| Invariant | Garantie de phase 7 |
|---|---|
| Entrées | Aucune nouvelle entrée publique de clé, adresse, point SEC 1 ou réseau. |
| Exactitude | Seulement des entiers, corps premiers et opérations de groupe exacts. |
| Reproductibilité | Ordres de parcours et de génération fixes, sans horloge ni hasard caché. |
| Budget | Aucun appel de relation au-delà de la limite déclarée. |
| Mémoire | La génération ne construit jamais un univers proportionnel à une profondeur non bornée. |
| Non-nouveauté | Les identités reconnues restent `Known`; les inconnues restent soumises à revue. |
| Rust pur | Pas de `unsafe`, FFI ni nouvelle dépendance. |

## Tests de sortie

La phase est terminée lorsque :

- un prédicat vrai avec un budget de un est appelé exactement une fois ;
- une porte insuffisamment couverte s'arrête réellement au budget ;
- une demande de profondeur et de scalaire maximaux reste plafonnée par le
  budget de candidats ;
- chaque variante de `Relation` apparaît dans une séquence suffisamment
  budgetée ;
- les formes inversées de double négation et d'inverse additif sont cataloguées ;
- le contrôle \(j=0\) utilise une racine cubique non triviale et son
  contre-exemple a `p = 7` et `a != 0` ;
- les tests, Clippy, formatage et vérification MSRV applicables au workspace
  passent en CI.

## Hors périmètre

- sérialisation ou désérialisation de relations externes ;
- courbes arbitraires, secp256k1, adresses Bitcoin, clés publiques ou privées ;
- cibles blockchain, réseau, RPC, fichiers de clés ou récupération de secrets ;
- déclaration de nouveauté, preuve générale ou extension au-delà des bornes
  locales établies.
