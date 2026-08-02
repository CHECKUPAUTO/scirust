# SciRust Elliptic Discovery — exécution et rejeu locaux v0.1

## Statut

Ce document définit la phase 6, après l’intégration des phases 0 à 5. Il ne
modifie pas le périmètre de sécurité du document de conception principal :
seules des courbes jouets générées localement peuvent être évaluées.

## Objectif

La bibliothèque sait construire les corpus, générer des relations, les
falsifier et produire des justifications. Il manque encore une frontière
d’exécution unique qui :

1. exécute un `SearchPlan` validé ;
2. résume exactement tous les candidats évalués ;
3. produit un reçu canonique avec séparation de domaine et empreinte SHA-256 ;
4. rejoue le même plan et détecte toute divergence ;
5. ne décode aucune donnée externe.

Le reçu n’est ni une preuve mathématique ni une affirmation de nouveauté. Il
atteste seulement qu’un plan local donné a produit un résultat automatisé
précis avec cette version du schéma.

## Décisions

### API de bibliothèque, sans protocole externe

La phase 6 ajoute une API Rust dans `scirust-elliptic-discovery`. Elle
n’ajoute pas de CLI généraliste, de format JSON entrant, de serveur, de réseau
ou de système de greffons. Le seul point d’entrée est un `SearchPlan` déjà
borné par le crate.

Cette décision empêche qu’un chemin de rejeu devienne implicitement un parseur
d’adresse, de clé publique, d’encodage SEC 1 ou de cible blockchain.

### Reçu canonique

Un `ExecutionReceipt` contient :

- le plan exact ;
- les empreintes ordonnées des trois corpus intégrés ;
- une empreinte ordonnée de chaque évaluation de candidat ;
- un résumé par statut autorisé ;
- le nombre de contre-exemples enregistrés.

L'encodage est binaire, big-endian, à longueurs explicites et séparé par le
domaine `SCIRUST-ELLIPTIC-DISCOVERY/EXECUTION-RECEIPT/V1` dans la définition
d'origine. La phase 7 fait évoluer le comportement de génération et passe les
domaines de plan, d'évaluation et de reçu à `V2`; voir
[`SCIRUST_ELLIPTIC_DISCOVERY_HARDENING_V0_1.md`](SCIRUST_ELLIPTIC_DISCOVERY_HARDENING_V0_1.md).
Les relations sont encodées par leur arbre syntaxique typé, jamais par `Debug`
ou `Display`.

### Rejeu

`replay_local` réexécute le plan porté par un reçu, recalcule un reçu complet
et compare les octets canoniques. Le résultat expose les empreintes attendue et
observée ainsi qu’un booléen de concordance. Il ne remplace pas le reçu
attendu, afin de conserver la divergence pour audit.

## Invariants

| Invariant | Vérification |
|---|---|
| Entrées locales seulement | L’API accepte uniquement `SearchPlan`. |
| Bornes finies | La construction de `SearchPlan` conserve toutes les limites de phase 4. |
| Exactitude | Aucun flottant n’est ajouté au chemin d’exécution ou de reçu. |
| Ordre stable | Corpus et candidats restent dans leurs ordres canoniques existants. |
| Rejeu strict | La comparaison porte sur tous les octets du reçu, pas sur un résumé partiel. |
| Non-nouveauté | Le résumé réutilise exclusivement `ClassificationStatus`. |
| Rust pur | Aucun `unsafe`, aucune FFI et aucune nouvelle dépendance. |
| Pas d’E/S cachée | L’exécution ne lit ni fichier, ni variable d’environnement, ni réseau. |

## Tests de sortie

La phase est terminée lorsque :

- deux exécutions du même plan produisent des reçus identiques ;
- deux graines distinctes produisent des empreintes distinctes ;
- le rejeu d’un reçu intact concorde ;
- le rejeu détecte une altération du reçu ;
- l’encodage couvre chaque variante de relation, de statut, de porte et de
  contre-exemple ;
- la matrice CI du workspace reste verte sur le MSRV déclaré.

## Hors périmètre

- import ou export de courbes arbitraires ;
- lecture d’adresses Bitcoin, de clés ou d’encodages de points ;
- récupération de secrets ;
- connexion à une blockchain ;
- prétention de découverte ou de preuve à partir du reçu ;
- désérialisation d’un reçu provenant d’une source non fiable.
