# Protocole — banc de co-changement CCOS

**Statut : figé.** Ce document est écrit et commité *avant* toute implémentation et
*avant* d'avoir vu le moindre chiffre. Il existe pour que le protocole ne puisse
pas être ajusté après coup en fonction du résultat obtenu.

Toute modification ultérieure de ce fichier doit être un commit séparé,
postérieur à la publication des résultats, et dire explicitement ce qui change
et pourquoi.

---

## 1. La question

CCOS est vendu comme une **mémoire causale** : elle prétend sélectionner un
meilleur contexte de travail qu'une recherche. La question mesurable est donc :

> Partant d'un fichier sur lequel on travaille, CCOS place-t-il dans la fenêtre de
> contexte les autres fichiers qu'il faudra réellement modifier — mieux qu'une
> recherche lexicale, à budget de tokens égal ?

## 2. Pourquoi les bancs existants ne répondent pas

- `examples/pure_retrieval_vs_rag.rs` prend pour vérité terrain « les fichiers de
  dépendance que l'AST a résolus », c'est-à-dire **le graphe de CCOS lui-même**.
  Un banc dont la référence est produite par le système évalué ne peut pas
  échouer. Il mesure la cohérence interne, pas la valeur.
- `examples/recall_eval.rs` tourne sur un **corpus synthétique**, construit avec
  sa propre vérité terrain.
- `examples/beir_eval.rs` mesure de la recherche documentaire sur des résumés
  scientifiques (SciFact). Utile pour situer les retrievers, sans rapport avec la
  sélection de contexte de code.

Aucun n'a de vérité terrain **extérieure à CCOS**. C'est le trou que ce banc
comble.

## 3. Vérité terrain — l'historique git

Un commit qui modifie plusieurs fichiers `.rs` est une observation directe de
« ces fichiers devaient être touchés ensemble ». Cette information est produite
par des humains résolvant de vraies tâches, des mois avant que CCOS n'existe.
Elle ne peut pas être influencée par le système évalué.

**Sélection des cas** (déterministe, sans réglage) :

- parcourir l'historique complet de `Memorithm/scirust` (2 246 commits) ;
- retenir les commits touchant **entre 2 et 8 fichiers `.rs`** — au moins deux
  pour qu'il y ait quelque chose à prédire, au plus huit pour écarter les
  refactorings de masse et les renommages automatiques, qui ne représentent
  aucune tâche de raisonnement ;
- exclure les commits de fusion (aucun contenu propre) ;
- ne retenir que les fichiers **présents dans l'arbre courant** : un fichier
  supprimé depuis ne peut être ni recallé ni recherché, l'y attendre pénaliserait
  les deux systèmes à l'identique mais brouillerait la lecture ;
- après ce filtre, ne garder que les commits qui conservent **≥ 2 fichiers**.

**Construction d'un cas** : pour un commit retenu de fichiers `{f₁ … fₙ}`, on
produit `n` cas. Le cas `i` a pour **ancre** `fᵢ` et pour **cibles** `{f₁ … fₙ} \ {fᵢ}`.

## 4. Ce qui est comparé

Les deux systèmes reçoivent **le même corpus** (l'arbre courant), **la même
ancre**, et **le même budget de tokens**.

- **CCOS** — `recall` stratégie `around`, ancre `file:<chemin>`, sur le workspace
  `.ccos/workspace.ccos` ingéré depuis ce même arbre.
- **Témoin — BM25 lexical.** Le contenu du fichier ancre sert de requête, on
  classe tous les autres fichiers du corpus par BM25 et on remplit la fenêtre
  jusqu'au budget.

Le témoin n'est **pas** un homme de paille : BM25 est la ligne de base que la
littérature IR utilise encore aujourd'hui, et l'implémentation vient de
`ccos_core::retrieval`, donc du même code, avec le même tokenizer et la même
arithmétique déterministe. Comparer CCOS à un agent aveugle ne prouverait rien.

## 5. Métriques

Pour chaque cas, on regarde quels fichiers cibles apparaissent dans la fenêtre
rendue :

- **Recall@budget** — fraction des cibles présentes dans la fenêtre. Métrique
  principale : c'est littéralement « le contexte contenait-il ce dont j'avais
  besoin ».
- **MRR** — inverse du rang du premier fichier cible. Mesure si les cibles
  arrivent tôt.
- **Precision@budget** — fraction de la fenêtre qui était pertinente. Mesure le
  gaspillage de contexte.

Rapportées à **trois budgets** : 1 024, 2 048 et 8 192 tokens. Un avantage qui
n'existe qu'à un seul budget est un artefact de réglage, pas un résultat.

## 6. Critère de réussite — fixé maintenant

CCOS est déclaré **meilleur que la recherche** si, et seulement si :

> le Recall@budget de CCOS dépasse celui de BM25 **aux trois budgets**, et l'écart
> au budget 2 048 est d'au moins **5 points de pourcentage**.

Tout autre résultat est un échec du banc pour CCOS, et sera rapporté comme tel.
Sont explicitement des échecs :

- CCOS gagne à un ou deux budgets sur trois ;
- CCOS gagne partout mais de moins de 5 points au budget 2 048 ;
- CCOS perd.

Le seuil de 5 points est posé avant mesure. Il correspond à ce qui serait visible
sur une tâche réelle ; un écart d'un point ne justifierait pas de vendre une
architecture.

## 7. Ce que ce banc ne prouve pas

À énoncer avec les résultats, sans attendre qu'on le demande :

- Il mesure la **sélection de contexte**, pas la réussite d'un agent. Un meilleur
  contexte devrait aider ; ce banc ne le démontre pas.
- La co-modification est un **proxy** du besoin. Deux fichiers changés ensemble
  l'ont parfois été pour des raisons sans rapport (mise à jour de version,
  reformatage).
- Un seul dépôt, un seul langage. Rien ici ne se transpose à un dépôt Python ou à
  une base de dix millions de lignes sans nouvelle mesure.
- L'ancre est un fichier, pas une intention en langage naturel. Le scénario réel
  d'un agent part souvent d'une description de tâche.

## 8. Exécution

Le banc est lancé **une seule fois** après implémentation, et son résultat est
rapporté quel qu'il soit. S'il faut le relancer (bug du harnais, corpus mal
construit), la raison est consignée dans le rapport, avec le résultat de la
première exécution.
