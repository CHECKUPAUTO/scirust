# Résultats — banc de co-changement CCOS

Protocole : `docs/CCOS_COCHANGE_PROTOCOL.md`, figé et commité **avant** toute
implémentation et avant d'avoir vu le moindre chiffre.

Corpus : `Memorithm/scirust` à `e042861`, 2 785 fichiers `.rs`.
Graphe CCOS : 76 065 nœuds, 135 012 arêtes.
Vérité terrain : 518 commits retenus sur 2 246, soit 1 841 cas ancre/cibles.
Échantillon mesuré : **220 cas**, tirés par ordre de hachage de contenu sur tout
l'historique.

## Résultat

| budget | Recall CCOS | Recall BM25 | écart | MRR CCOS | MRR BM25 | Precision CCOS | Precision BM25 |
|---|---|---|---|---|---|---|---|
| 1 024 | **37,9 %** | 16,5 % | +21,4 pt | 43,9 % | 35,5 % | 12,1 % | 34,8 % |
| 2 048 | **44,6 %** | 16,8 % | +27,8 pt | 44,2 % | 35,6 % | 10,1 % | 31,9 % |
| 8 192 | **69,2 %** | 25,4 % | +43,8 pt | 44,8 % | 41,9 % | 5,3 % | 20,3 % |

Le découpage post-hoc excluant l'arbre vendorisé `external/` (n = 218) est
identique à moins de 0,4 point sur chaque cellule.

## Verdict au regard du critère figé

Le protocole §6 exigeait, écrit avant mesure :

> le Recall@budget de CCOS dépasse celui de BM25 **aux trois budgets**, et l'écart
> au budget 2 048 est d'au moins **5 points de pourcentage**.

CCOS mène aux trois budgets ; l'écart à 2 048 est de **+27,8 points**.
**Le critère est rempli.**

C'est la première mesure établissant que CCOS fait ce qu'il annonce, contre une
vérité terrain qu'il n'a pas produite.

## Ce que CCOS perd

La précision, nettement : 10,1 % contre 31,9 % à 2 048 tokens. CCOS ratisse plus
large — il remplit la fenêtre de nœuds granulaires issus de toute la région
causale, là où BM25 rend peu de fichiers entiers et tape plus juste. À budget
fixe, une part réelle du contexte part dans des fichiers dont l'agent n'avait pas
besoin. Le protocole désignait le recall comme métrique principale *avant* la
mesure, et c'est le bon choix pour « l'agent avait-il ce qu'il fallait » — mais le
gaspillage est réel et se paie.

## Latence

| | avant | après |
|---|---|---|
| CCOS, requête `around` | 8 812 ms | **1 314 ms** |
| BM25, même corpus | 21 ms | 21 ms |

Le coût venait de `hop_distances`, dont le BFS cherchait les voisins en
parcourant **toutes** les arêtes du graphe à chaque nœud dépilé — O(V·E). Un index
d'adjacence mis en cache par version de graphe le ramène à O(V+E) : facteur 6,6,
et **toutes les métriques de qualité inchangées au dixième de point**, ce qu'un
test verrouille en recalculant les distances à l'ancienne façon.

CCOS reste 63× plus lent que BM25. Le terme quadratique est parti ; ce qui reste
n'a pas encore été attribué, et aucune promesse n'est faite avant profilage.

Cette mesure éclaire aussi le plafond résident de 5 000 nœuds par défaut : sa
fonction réelle n'était pas d'économiser la RAM mais d'empêcher la requête de
devenir impraticable. Il reste que la configuration mesurée ici — plafond levé,
corpus entier résident — n'est pas celle qui est livrée.

## Ce que ce banc ne prouve pas

Rappelé du protocole §7, et vrai après mesure comme avant :

- il mesure la **sélection de contexte**, pas la réussite d'un agent ;
- la co-modification est un **proxy** du besoin ;
- un seul dépôt, un seul langage ;
- l'ancre est un fichier, pas une intention en langage naturel ;
- 220 cas sur 1 841, soit 12 % du corpus disponible.

## Deux erreurs de mesure, consignées

Le protocole §8 impose de rapporter les exécutions écartées et pourquoi.

**Run 1 — CCOS à 0,0 % partout.** Panne du harnais, pas résultat produit.
`CcosMemory::new()` plafonne à 5 000 nœuds résidents ; le corpus en produit
76 065, et j'avais omis l'appel `ensure_resident` que la façade CLI fait avant
chaque recall. L'ancre était démotée en COLD, la fenêtre revenait vide. Corrigé
par `ensure_resident` plus la levée du plafond, pour que les deux systèmes voient
bien le même corpus comme le protocole l'exige.

**Run 2 — +12 points à 2 048, sur 25 cas.** Résultat flatteur et non fiable.
`git log` sort en ordre antichronologique : les 25 premiers cas étaient les
commits de synchronisation du code de CCOS lui-même, dense dans son propre graphe
et co-changeant parce que copié en bloc. Que BM25 y tombe à 7,5 % au lieu de
14,2 % suffisait à disqualifier le tirage. Corrigé par un tri déterministe sur le
hachage du contenu.

Les deux erreurs étaient miennes, en sens opposés, et aucune n'était visible dans
le chiffre seul. C'est la raison d'être du protocole figé : il empêche de choisir
après coup lequel des deux publier.

## Reproduire

```bash
sh scripts/ccos/cochange_cases.sh > cases.tsv
cd external/ccos-core
CASES_LIMIT=220 cargo run --release --example cochange_eval -- ../.. ../../cases.tsv
```
