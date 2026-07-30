# CCOS contre les RAG — comparatif agressif, famille par famille

> *Registre : offensif mais **défendable**. Chaque pique est adossée soit à un exemple runnable
> (`examples/*_crux.rs`), soit à une mesure (`docs/MEASUREMENT_*`, `docs/FIELD_CAMPAIGN_H.md`), soit à
> un argument **structurel** que le code de l'autre camp ne peut pas contredire. La règle maison tient :
> on dit aussi **où le RAG gagne** — un comparatif qui ne concède rien n'est pas cru, il est balayé.*

Voir aussi la version « honnête neutre » : [`COMPARISON_vs_rag.md`](COMPARISON_vs_rag.md). Ce document-ci
en est la **variante commerciale, assumée**.

> **⚠️ Ce qu'on ne prétend PAS — le résultat négatif, en tête, pas en note de bas de page.**
> CCOS ne bat **pas** le RAG sur son propre terrain, la *récupération*. On était partis pour le montrer ;
> on l'a testé sur données réelles ; **ça n'a pas tenu.** Sur **70 commits réels** de correction de bogues,
> un simple récupérateur lexical TF-IDF **fait jeu égal** avec la sélection causale et **la bat à budget
> serré** ; un pivot par trace de crash **perd** contre un RAG-sur-le-message-d'erreur — parce que sur du
> vrai code, les fichiers d'un correctif et ses messages d'erreur *partagent leur vocabulaire*
> (conclusion du papier, `docs/paper/ccos_regions.fr.md`). On **publie** ce résultat négatif au lieu de
> l'enterrer. Donc ce comparatif n'attaque **pas** sur le recall : il attaque sur les axes qu'un RAG ne
> sait pas *représenter* — **croyance, temps, provenance, replay, déterminisme**. Ce qui survit n'est pas
> un meilleur récupérateur, c'est un **type d'objet différent** : une mémoire de travail déterministe,
> rejouable, auditable.

---

## 0. TL;DR — verdict par famille de RAG

| Famille | Ce qu'ils vendent | Là où ils cassent | Verdict |
|---|---|---|---|
| **Naïve RAG** (chunk + dense) | « on retrouve le passage pertinent » | sur un crux construit, recall@10 **~50 %** des dépendances croisées (`rag_crux.rs`) — **mais** sur 70 vrais commits le lexical fait jeu égal, cf. bandeau ⚠️ | **Match nul sur le recall** ; CCOS gagne ailleurs (croyance, replay) |
| **Hybrid + reranker** | « dense + BM25 + cross-encoder, le meilleur recall » | recall *plus fort que nous* ; mais **zéro axe de croyance, zéro temps, zéro replay** ; le cross-encoder est un modèle → **non déterministe** | **RAG gagne le recall** ; CCOS gagne audit/déterminisme — catégories différentes |
| **GraphRAG** (Microsoft) | « graphe d'entités + résumés LLM par communauté » | ingestion **LLM = chère et non rejouable** ; graphe d'**entités**, pas de **causalité de code** ; contradictions agrégées, pas arbitrées | **CCOS gagne** sur coût, déterminisme, contradiction (pas sur le recall) |
| **Agentic RAG** | « l'agent boucle : query → retrieve → raisonne → re-query » | chaque tour est un tirage LLM → **irreproductible** ; aucune mémoire *rejouable* de ce que l'agent a cru à chaque pas | **CCOS complète** : il est la mémoire de travail rejouable que cette boucle n'a pas |
| **Mémoire d'agent** (mem0 · Zep · **TiMEM**) | « hiérarchie temporelle, consolidation, personas » | consolidation **faite par un LLM** → probabiliste, non rejouable bit-à-bit ; conçue pour le **dialogue**, pas le **code** | **Adjacent, pas frontal** — CCOS se distingue par déterminisme + domaine code |

**La phrase à retenir :**
> Tout RAG répond à *« quels documents **ressemblent** à ma requête ? »*.
> CCOS répond à *« que dois-je **croire**, d'**où** ça vient, et comment ma compréhension **a changé dans
> le temps** ? »* — **rejouable, auditable, hors-ligne.**

---

## 1. La thèse offensive

Toutes les saveurs de RAG — naïve, hybride, re-rankée, GraphRAG, agentique — sont la **même primitive** :
*récupération sans état par similarité*. On découpe le corpus, on l'embarque dans un espace vectoriel, on
renvoie les top-k plus proches d'une requête. C'est utile. Ce n'est **pas une mémoire**.

CCOS n'est **pas un meilleur RAG**. C'est la couche que la pile RAG n'a **structurellement** pas : une
**mémoire de travail causale, déterministe, rejouable, qui porte des croyances**. Les axes sur lesquels on
attaque — structure, croyance, temps, provenance, replay — ne sont pas des axes où le RAG est *faible* : ce
sont des axes qu'un RAG **ne sait pas représenter du tout**. On ne gagne pas 20 % ; on joue à un jeu que
l'adversaire ne peut pas jouer.

---

## 2. Famille par famille

### 2.1 Naïve RAG (chunk + dense) — *« la structure lui est invisible »*

Le RAG naïf embarque des chunks de 512 tokens et renvoie les plus proches. Sur un cas **construit pour
isoler le mécanisme**, `examples/rag_crux.rs` mesure qu'un RAG lexical récupère **~50 %** seulement des
dépendances croisées (recall@10), là où le graphe causal les récupère **~100 % par construction** : une
arête `use crate::config` relie deux fichiers dont le **vocabulaire ne se recoupe pas** — la similarité est
aveugle à la dépendance.

> **⚠️ Le crux n'est PAS le cas général — à ne jamais présenter comme tel.** Sur **70 commits réels** de
> correction de bogues, ce même avantage **s'évapore** : les fichiers d'un correctif et ses messages
> d'erreur partagent leur vocabulaire, donc un lexical TF-IDF fait jeu égal avec la sélection causale et
> **la bat à budget serré** (résultat négatif publié, `docs/paper/ccos_regions.fr.md`). `rag_crux` prouve
> que le mécanisme *existe* — pas qu'il domine en moyenne. Citer le crux sans le terrain = se faire
> corriger par notre propre papier.

L'attaque défendable n'est donc pas « on récupère mieux », mais : *pour un agent de codage, CCOS place la
région causale (self-bounding, sans `k` à régler) dans un budget serré et la rend **rejouable*** — la
récupération plate n'a ni l'auto-borne ni le replay, quel que soit son recall.

Pire pour un agent de codage : au même budget (2048 tokens), quand vous travaillez un fichier réel, CCOS
place ses dépendances croisées dans la fenêtre **81–100 %** du temps, contre **0–2 %** pour ouvrir
naïvement le fichier tronqué au même budget (`docs/FIELD_CAMPAIGN_H.md`). Ce n'est pas un cas limite : les
dépendances croisées sont **partout**.

### 2.2 Hybrid RAG + reranker — *« plus de recall, toujours zéro croyance »*

C'est le RAG sérieux : dense + BM25, puis un cross-encoder qui re-classe. Soyons nets — **sur le pur recall
sémantique, il gagne** (voir §3). Mais il empile trois modèles et hérite de leurs trois défauts :
**non-déterminisme** (poids non bit-stables), **coût**, **dépendances**. Et surtout, il ne répond toujours
qu'à *« ça ressemble ? »*. Donnez-lui deux sources contradictoires : il classe la **réfutée #1** si elle est
lexicalement plus proche (`examples/scirust_vs_rag_crux.rs` : precision@1 = 1/2). Il n'a **pas d'axe de
croyance** — il ne peut pas, par construction, préférer la source *autoritaire* à la source *ressemblante*.

### 2.3 GraphRAG (Microsoft) — *« un graphe, oui, mais le mauvais, et payé au LLM »*

GraphRAG a de la structure — mais un graphe d'**entités** peuplé de **résumés LLM par communauté**. Trois
prises :

1. **Ingestion chère et non rejouable.** Résumer chaque communauté au LLM coûte, et *dérive* : deux
   ingestions ne donnent pas le même graphe. CCOS ingère en **O(N)** (~2139 fichiers/s), la LSA
   incrémentale replie un batch en **O(batch)**, et le tout est **bit-exact** (`replay == live` tient).
2. **Mauvais graphe pour le code.** Un graphe d'entités n'est pas un graphe de `imports · calls · data-flow
   · Causes`. GraphRAG relie « Alice » à « Acme » ; CCOS relie `writer.rs` à `config.rs` par l'arête qui
   fait *paniquer le test*.
3. **Contradictions agrégées, pas arbitrées.** GraphRAG fond les sources en un résumé ; CCOS tient
   `support/contra → belief + tension`, avec décroissance et propagation (`examples/scirust_vs_rag_crux.rs`
   écrase la source réfutée de #1 à #5/#7 et garde l'autoritaire #1, **2/2**).

### 2.4 Agentic RAG — *« la boucle qu'aucune mémoire ne trace »*

L'agentic RAG fait boucler le modèle : interroge, récupère, raisonne, ré-interroge. C'est puissant — et
c'est **exactement le cas d'usage où l'absence de mémoire rejouable fait mal**. Chaque tour est un tirage
LLM : rejouez la session, vous obtenez un autre chemin. Quand l'agent part en vrille sur un horizon long,
vous n'avez **aucun moyen de revenir au pas exact** où sa représentation du projet s'est corrompue. C'est le
trou que CCOS remplit : `replay_to`, `recall_what_if`, et le watchpoint `missing <node>` qui **nomme le
moment précis** où la vraie cause a été évincée du budget. L'agentic RAG *produit* le désordre ; CCOS en est
le **flight recorder**.

### 2.5 Mémoire d'agent — mem0 · Zep · **TiMEM** — *« adjacent, pas frontal »*

Ici on ne parle plus de récupération de documents mais de **mémoire d'agent** — et c'est la famille la plus
proche de nous, donc celle qu'il faut traiter avec le plus de précision (pas de caricature). TiMEM
(hiérarchie temporelle L1–L5, arbre mémoire temporel, recall adaptatif à la complexité), mem0, Zep :
excellents sur leur terrain. Deux différences **structurelles**, pas cosmétiques :

- **Ils consolident au LLM.** La mémoire de TiMEM est *construite* par un modèle qui résume et abstrait les
  tours de conversation. C'est donc **probabiliste et non rejouable bit-à-bit** — par conception. CCOS
  construit sa mémoire par un graphe **déterministe**, sans LLM dans la boucle d'écriture : `replay == live`
  est prouvé octet-pour-octet (`tests/replay_equivalence_property.rs`).
- **Ils visent le dialogue, pas le code.** Leur objet est le tour de conversation, le fait utilisateur, le
  persona (support client, tutorat). Le nôtre est le fichier, la dépendance croisée, la cause racine d'un
  bug multi-fichiers. Un ingénieur d'agent de codage n'ira pas chercher TiMEM, et inversement.

**Verdict honnête :** concurrent *analogue*, pas frontal. Ils deviennent frontaux **si** ils basculent vers
le code **et** ajoutent un replay déterministe — deux virages que leur cœur LLM-based rend structurellement
coûteux.

---

## 3. L'atout qu'on sous-vend : **l'embedder déterministe**

On a un embedder sémantique **entièrement déterministe** — et on n'en parle presque jamais. C'est une
erreur de communication, parce que c'est précisément l'axe où *aucun* RAG neuronal ne peut nous suivre.

- **Ce qu'il est.** Un signal sémantique **INT4 TF-IDF** (cosine), déterministe par défaut. En option
  (`--features learned-embed`), il se distille en une **projection LSA latente** — les vecteurs singuliers
  dominants de la co-occurrence du corpus, via un balayage de Jacobi à **ordre fixe** — donc un terme qui
  ne fait que *co-occurrer* avec ceux d'un fichier le retrouve quand même (la synonymie que le TF-IDF brut
  ne voit pas). **Zéro nouvelle dépendance, entièrement déterministe** : le build par défaut reste
  byte-identique, l'invariant de replay tient.
- **La preuve qu'on ne joue pas petit bras.** Sur **BEIR**, corpus standard, métrique de la communauté IR,
  notre BM25 pur-Rust, zéro-dép, **sans stemming ni stopwords ni tuning**, fait **nDCG@10 = 0.662 sur
  SciFact** contre **0.665** pour la baseline Anserini/Lucene publiée — un écart de **0.003**
  (`docs/MEASUREMENT_beir.md`, sortie réelle d'un run). Et là où le vocabulaire diverge (query et réponse
  ne partagent **aucun** terme), le même encodeur LSA bat le même lexical **17 % → 0 % Recall@1**
  (`semantic_retrieval_crux`). On expédie **les deux signaux et une fusion RRF**, parce que *c'est la
  structure de la tâche qui décide*.
- **Le moat, dit franchement.** Un embedder transformer casserait `replay == live` (les poids ne sont pas
  bit-stables d'un build à l'autre, l'éval dépend du non-déterminisme GPU et de la dérive de version). CCOS
  fait donc un **choix délibéré** : garder un **plancher sémantique déterministe, sans dépendance,
  air-gappable**, et investir la différenciation là où le RAG ne peut structurellement pas aller —
  structure, croyance, temps, audit. Ce n'est pas « on est moins bon en sémantique » ; c'est « on est le
  **seul** dont la couche sémantique est **rejouable octet-pour-octet et hors-ligne** ».

> **Formule commerciale :** *« Notre embedding n'a pas besoin de GPU, de réseau, ni de confiance : il est
> déterministe, il tourne air-gappé, et il égale la baseline BM25 publiée sur BEIR — pendant qu'il reste
> rejouable bit-pour-bit. Aucun RAG neuronal ne peut cocher ces quatre cases à la fois. »*

---

## 4. La matrice complète

| Dimension | Naïve RAG | Hybrid + reranker | GraphRAG | Mémoire d'agent (TiMEM…) | **CCOS** |
|---|---|---|---|---|---|
| Unité de récupération | chunk 512 tok | chunk + BM25 | chunk + graphe d'entités | tour de dialogue résumé | **nœud causal typé** |
| Base de similarité | dense neuronal | dense + lexical + cross-encoder | graphe + résumés LLM | embeddings LLM | TF-IDF·INT4 + **LSA causale incrémentale** |
| Structure | ❌ plat | ❌ plat | entités / communautés | arbre temporel | **graphe causal** (imports · calls · data-flow · Causes) |
| **Contradictions** | ❌ | ❌ | agrégation floue | ❌ | ✅ **support/contra → belief + tension, décroissance, propagation** |
| Provenance / audit | faible | faible | moyenne | faible | ✅ **hash-chain + `replay == live` octet-exact** |
| Déterminisme | ❌ (dérive modèle) | ❌ | ❌ (LLM) | ❌ (LLM) | ✅ **bit-exact, sans RNG** |
| Time-travel / what-if | ❌ | ❌ | ❌ | ordre temporel, non rejouable | ✅ **`replay_to`, `recall_what_if`** |
| Dynamique temporelle | ❌ | ❌ | ❌ | ✅ (mais pas de croyance) | ✅ **décroissance (demi-vie) + tenseur temporel (« courbe de fièvre »)** |
| Coût d'ingestion | embed + index ANN | + reranker | **résumés LLM = cher** | **consolidation LLM = cher** | **O(N) graphe (~2139 f/s) + O(batch) LSA** |
| Dépendances / offline | modèle + vector DB | + reranker | + LLM | + LLM | ✅ **zéro-dép-extra, hors-ligne, air-gappable** |
| Embedder **déterministe** | ❌ | ❌ | ❌ | ❌ | ✅ **INT4 TF-IDF + LSA, BM25 = Anserini à 0.003 près** |
| *Recall sémantique pur* | ✅ fort | ✅✅ le plus fort | ✅ fort | ✅ fort | ⚠️ **moyen** (déterministe, mais BEIR-compétitif en lexical) |

---

## 5. Où le RAG gagne (sans survendre)

On concède, parce que concéder est ce qui rend le reste crédible :

- **Recall sémantique pur à l'échelle web.** Un dense RAG bien tuné (transformer + cross-encoder) récupère
  **mieux** que notre TF-IDF/LSA quand le vocabulaire diverge fortement et que l'échelle est en milliards de
  chunks. `rag_crux` le montre honnêtement : le signal lexical est réel mais **incomplet**.
- **QA documentaire généraliste.** Pour répondre à des questions sur un corpus documentaire massif, un RAG
  neuronal est le bon outil. CCOS vise la **mémoire de travail d'agent et le code**, pas le Q&A web.
- **Écosystème.** Les RAG ont des vector DB matures, des intégrations, une adoption. CCOS est un prototype
  de recherche mono-binaire Rust.

Ces concessions ne touchent **aucun** des axes du §4 marqués ✅ chez nous. Le RAG gagne le recall ; il ne
gagne ni la croyance, ni le temps, ni la provenance, ni le replay — parce qu'il ne les a pas.

---

## 6. Verdict

Un RAG répond : *« quels documents ressemblent à ma requête ? »*.
CCOS répond : *« que dois-je croire, d'où ça vient, et comment ma compréhension a changé dans le temps ? »* —
**rejouablement et auditablement**.

Pour un agent qui ne doit pas se faire tromper par une contradiction, qui doit rejouer et auditer son propre
raisonnement, et rester déterministe et hors-ligne, **CCOS gagne structurellement**. Pour du recall
sémantique brut, un dense RAG tuné récupère plus. Les deux sont **complémentaires** : CCOS est la couche de
mémoire de travail causale qui se pose **au-dessus** — ou **à la place** — de la récupération plate.

**Ligne de fond :** CCOS ne concourt pas sur le « plus proche voisin ». C'est une mémoire causale
déterministe, auditable, porteuse de croyances — et sur les axes qu'un RAG ne sait même pas représenter
(contradiction, temps, provenance, replay), la comparaison n'est pas serrée.
