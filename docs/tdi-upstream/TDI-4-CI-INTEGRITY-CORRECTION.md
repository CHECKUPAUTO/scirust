# TDI-4 — Correction d’intégrité CI avant évaluation

## Statut

Cette correction est intervenue avant toute exécution complète de TDI-4
et avant toute production de métrique ou de verdict TDI-4.

## Échec observé

Le gel initial TDI-4, au commit :

`30916731078e9fe04f35d30b5388d04f9ed12d65`

a échoué dans le job CI `Preregistration integrity`.

Le manifeste scientifique immuable TDI-3 signalait uniquement :

`tdi-core/src/signature.rs: FAILED`

## Cause

L’implémentation initiale de TDI-4 avait ajouté une méthode à
`tdi-core/src/signature.rs`.

Ce fichier appartient au gel scientifique TDI-3. Une expérience ultérieure
ne doit pas modifier un fichier couvert par ce manifeste historique.

## Correction

`tdi-core/src/signature.rs` a été restauré exactement dans son état gelé
par TDI-3.

Le calcul de la cible conditionnelle TDI-4 est désormais localisé dans
l’évaluateur TDI-4 :

- le numérateur exact du déficit est calculé avec `BigUint` ;
- le logarithme du numérateur et du dénominateur est déterminé à partir de
  leur longueur binaire et de leurs 53 bits significatifs supérieurs ;
- la cible reste `U = -log₂(1 - O₆)` ;
- aucun arrondi préalable de `O₆` vers `1.0` n’est utilisé.

## Invariants

Cette correction ne modifie pas :

- le préenregistrement ;
- les populations et leurs graines ;
- les horizons ;
- les variables explicatives ;
- les modèles ;
- `lambda` ;
- le bootstrap ;
- les critères TDI-4A et TDI-4B.

Aucun résultat TDI-4 n’a été observé avant cette correction.
