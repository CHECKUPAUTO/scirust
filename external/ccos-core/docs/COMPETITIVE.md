# CCOS vs Headroom — analyse concurrentielle honnête
### Lecture du **code source réel** de `headroomlabs-ai/headroom`, pas du README.

> *Document d'information interne. Méthode : lecture du code source public de Headroom
> (branche `main`) le **2026-06-23**, via récupération des fichiers bruts
> (`raw.githubusercontent.com`). Chaque affirmation technique est ancrée sur du texte de
> code réellement lu (chemin de fichier + citation courte). Le registre est délibérément
> honnête : on distingue ce que leur **code** fait de ce que leurs **noms** ou leur
> **marketing** suggèrent, et on **corrige nos propres affirmations** quand leur source les
> dément. Headroom est activement développé (un portage Rust est en cours) : cette analyse
> est un instantané et peut dériver.*

> **⚠️ Mise à jour 2026-06-23 — CCOS a bougé depuis cette lecture.** Cette analyse a été
> écrite *avant* que CCOS ne gagne des capacités qui **réduisent l'écart** là où je donnais
> Headroom gagnant :
> - **Compression réversible + CCR** (`src/compressor.rs`) : CCOS re-encode désormais le
>   contenu (CausalCrusher/AST/Summ, déterministes) et garde les originaux récupérables via
>   l'outil MCP `ccos_retrieve` — l'équivalent déterministe de `headroom_retrieve`. L'axe
>   frugalité n'est plus *vide* côté CCOS (≈ 30–50 % mesuré sur du Rust réel, le plancher
>   déterministe de la fourchette Headroom).
> - **Embeddings INT4** (`src/embeddings.rs`) : un *embedder* sémantique déterministe
>   (TF-IDF quantifié INT4, cosine), **désormais câblé sur le chemin vif** via
>   `Recall::Semantic` (l'index est reconstruit à la volée — la mise en cache est un point
>   perf tracé, pas de correction). Recall sémantique réel, mais qui reste en deçà du RAG
>   complet de Headroom (sqlite-vec + FTS5 + mem0).
> - **Dé-obfuscation Unicode + signal d'injection** (`src/sanitizer.rs` + classifieur) : un
>   **nouvel axe** que la lecture du code de Headroom n'a pas montré chez eux — défang
>   déterministe/auditable des vecteurs Trojan-Source / zero-width / Tags, findings versés
>   dans le log hash-chaîné. Voir [`SECURITY.md`](SECURITY.md).
>
> Les corrections de fond ci-dessous (Headroom *a* une vraie mémoire ; ses compresseurs
> *sont* déterministes) restent valables. À re-vérifier des deux côtés avant tout usage
> public daté.

---

## 0. TL;DR — verdict par axe

| Axe | Qui gagne | En une ligne |
|---|---|---|
| **Frugalité / coût-tokens** | **Headroom**, nettement | Pipeline de compression *content-aware* mûr, Rust-backed, avec un vrai modèle ML. |
| **Mémoire long-terme** | **Headroom** (RAG complet) | Store vectoriel persistant (sqlite-vec + FTS5 + mem0). CCOS a un recall sémantique INT4 TF-IDF **câblé** (`Recall::Semantic`), en deçà mais réel. |
| **Dé-obfuscation Unicode / hardening de l'entrée** | **CCOS** | Défang déterministe + auditable (Trojan-Source / zero-width / Tags) + signal d'injection forensic. Non trouvé dans le code de Headroom. |
| **Mémoire de travail *rejouable, auditable, debuggable post-mortem*** | **CCOS**, seul | *Confirmé par leur source* : Headroom n'a ni log hash-chaîné, ni replay déterministe, ni event-sourcing des évictions, ni watchpoint. |

**Conclusion stratégique :** la repositionnement de CCOS vers le **« flight recorder » de la
mémoire de travail (replay déterministe + watchpoint d'éviction)** est le bon choix — c'est
le seul axe que le concurrent le plus sérieux **n'occupe pas**, et son propre code le
confirme. Mais : (1) c'est un axe **étroit** ; (2) *être unique ≠ être utile* — la valeur
reste à démontrer (objet de la Campagne J / du travail résolution).

---

## 1. Ce que Headroom EST vraiment (image corrigée)

Ce n'est **pas** « juste un compresseur de tokens ». Le code révèle quatre sous-systèmes réels.

1. **Pipeline de compression Rust-backed, structurel et déterministe.**
   `crates/headroom-core` (Rust) implémente les transforms — `smart_crusher`, `text_crusher`,
   `diff/log/search_compressor`, détection de type via `magika`, tokenizers — exposés à Python
   en PyO3 (`crates/headroom-py`, lib `_core`). Les modules Python sont des *shims* qui
   délèguent au Rust (ex. `headroom/transforms/smart_crusher.py` :
   `from headroom._core import SmartCrusher as _RustSmartCrusher`). Il existe même un crate
   `crates/headroom-parity` = **harnais d'équivalence Rust↔Python** : ils tiennent à ce que
   les deux implémentations produisent le même résultat.

2. **Une vraie mémoire RAG persistante.** `headroom/memory/` est un store hiérarchique réel :
   SQLite + extension **`sqlite-vec`** (table virtuelle `vec0`, KNN cosinus) + **FTS5/BM25** +
   un store graphe *optionnel* (`sqlite_graph`), isolé **par projet**
   (`memory/storage_router.py` : *"each workspace a physically isolated SQLite database file"*).
   Récupération par **similarité sémantique** (`embedder.embed(query)` → recherche vectorielle),
   pas par graphe causal. ADR-008 confirme : *"SQLite + sqlite-vec + FTS5"*.

3. **CCR — compression réellement réversible.** Lossy « sur le fil », mais l'original est
   récupérable. Quand un compresseur jette des données il laisse un marqueur `<<ccr:HASH …>>`
   et stocke l'original clé-**SHA-256** dans `~/.headroom/ccr_store.db` (backends sqlite /
   mémoire / redis). L'outil **`headroom_retrieve(hash)`** (`headroom/ccr/tool_injection.py`,
   `ccr/response_handler.py`) rend l'original au modèle à la demande. Round-trip **lossless
   end-to-end** (`crates/headroom-core/src/ccr/backends/sqlite.rs`, doc : *"lossy on the wire,
   lossless end-to-end"*).

4. **`learn/` — boucle d'amélioration hybride.** Scanners de transcripts et détection de
   boucles déterministes (`learn/loops.py` : *"does not call an LLM directly"*), **mais
   l'analyse cœur est LLM** (`learn/analyzer.py` : *"Session analysis via LLM"*,
   `litellm.completion(...)`). Écrit des corrections entre marqueurs dans
   **`CLAUDE.local.md`** (volontairement le fichier gitignoré, *"so we never pollute the
   team-shared CLAUDE.md"*), `AGENTS.md`, `GEMINI.md`.

---

## 2. Corrections à nos propres affirmations (honnêteté intellectuelle)

Lire leur code nous oblige à corriger deux raccourcis qu'on avait pris :

- ❌ **« Headroom n'a pas de mémoire » → FAUX.** Ils ont une mémoire RAG vectorielle robuste
  et persistante. Pitcher CCOS sur « eux n'ont pas de mémoire » serait réfutable par leur
  propre source — à ne **pas** faire.
- ❌ **« Headroom est non-déterministe » → trop large.** Leurs compresseurs structurels
  (SmartCrusher, CodeCompressor AST, TextCrusher, CacheAligner) sont **déterministes** par
  conception. Le non-déterminisme est *localisé* : (a) le modèle ML *Kompress* (télécharge
  ses poids, variance ONNX/PyTorch) et (b) l'analyseur `learn/` adossé LLM. La compression,
  elle, est largement reproductible.

---

## 3. Là où Headroom est franchement PLUS FORT

| Dimension | Headroom | CCOS |
|---|---|---|
| Compression | 6+ compresseurs *content-aware*, détection `magika`, AST tree-sitter, **un vrai modèle ML entraîné** (Kompress / ModernBERT) | rien de comparable |
| Mémoire long-terme | RAG vectoriel + BM25 + graphe (sqlite-vec / FTS5) | graphe causal de code (pas de recall sémantique) |
| Réversibilité | CCR hash-addressé, lossless end-to-end | watchpoint + replay (réponse alternative — §5) |
| Surface produit | proxy + MCP + SDK TS/Py + 6 intégrations agents + portage Rust | prototype Rust mono-binaire |
| Maturité / adoption | projet large, multi-langage, déploiement production | prototype de recherche |

Sur **l'axe frugalité/coût-tokens**, Headroom est plus mûr, plus large, plus
production-ready. Si la thèse de CCOS est « on économise mieux les tokens », **CCOS perd**.

### Détail des algorithmes de compression (lus dans `headroom/transforms/`)

| Algorithme | Mécanisme | Lossy/Lossless | Déterministe ? |
|---|---|---|---|
| **SmartCrusher** | Compacteur de tableaux JSON (JSON→CSV/markdown-KV), Rust-backed | les deux modes (`lossless_only` force le byte-recoverable) | **Oui** (transform structurel, pas de modèle) |
| **CodeCompressor** | AST via tree-sitter ; garde imports/signatures/types, coupe commentaires + corps de fonctions ; `_verify_syntax()` | Lossy | **Oui** |
| **Kompress** | Modèle ML (ModernBERT, `chopratejas/kompress-v2-base`) score chaque token keep/discard | Lossy | Déterministe *à modèle+backend fixés* ; **télécharge ses poids** (réseau) |
| **CacheAligner** | Détecteur seul : repère le contenu volatil (UUID, timestamps, JWT) déstabilisant le KV-cache ; *ne modifie pas* le prompt | Lossless / non-mutant | **Oui** |
| **TextCrusher** | Compresseur extractif de prose (BM25 + récence + saillance) ; sélectionne des phrases *verbatim* | Lossy mais verbatim | **Oui** (*"fast deterministic extractive prose compressor"*) |
| **ContentRouter** | Dispatcher : détecte le type (`magika`) → route vers le compresseur adéquat | n/a | **Oui** |

---

## 4. Là où CCOS reste réellement distinct — *confirmé par leur code*

Recherches repo-wide à **0 résultat** sur `prev_hash | merkle | event sourcing | tamper |
watchpoint | time-travel`. Concrètement :

- **Log d'événements hash-chaîné / infalsifiable → ils ne l'ont pas.** Leurs ledgers
  (`headroom/savings_ledger.py`, `headroom/storage/jsonl.py`) sont append-only sous `flock`
  mais en **lignes plates indépendantes**. Ils *hachent* (SHA-256 pour CCR, `messages_hash`
  pour le cache-keying) — mais pour **adresser/dédupliquer**, jamais pour **chaîner**.
  → « ils utilisent des hash » = vrai ; « ils ont un audit hash-chaîné » = faux.
- **Replay déterministe / time-travel des décisions de contexte → absent.** Leur « replay »
  = harnais de bench/charge (`scripts/replay_codex_ws_load.py`) ou renvoi verbatim d'octets
  côté proxy. L'état de maturation par requête (`transforms/read_maturation.py`) est
  *in-memory, par-process, non persisté* (*"Per-process, like all session state"*) — non
  rejouable.
- **Event-sourcing des évictions (« pourquoi gardé / jeté », interrogeable après coup) →
  absent.** Chez eux « eviction » = pure gestion **LRU de capacité de cache**
  (`cache/backends/base.py` : *"eviction policies… No business logic"*). Les *raisons*
  n'existent que pour des recommandations d'**expansion** (`ccr/context_tracker.py`,
  `ExpansionRecommendation.reason`), en mémoire, jamais persistées.
- **Watchpoint d'éviction → absent.** `ccr/context_tracker.py`, commentaire explicite :
  *"No watchpoint or alert support… No callbacks or monitoring for eviction events."*
- **Graphe causal de dépendances de code → ils n'en ont pas pour piloter l'assemblage.**
  Leur pertinence est IR (`relevance/bm25.py`, `embedding.py`, `hybrid.py`) ; leur dossier
  `graph/` n'est qu'`installer.py` + `watcher.py`. *Nuance honnête :* ils ont *un* store
  graphe **optionnel** pour les enregistrements mémoire (`sqlite_graph`) — ce n'est pas le
  même objet qu'un graphe de dépendances de code.

---

## 5. Pièges & insight stratégique

### Pièges de nommage (à connaître AVANT de les citer)

Trois noms induisent en erreur — les citer sans lire le code = se faire corriger :

- **`audit/`** ≠ journal d'audit runtime. C'est de l'outillage **offline** de mesure de
  trafic (`audit/__init__.py` : *"Offline traffic audits — measure opportunity sizes before
  tuning defaults"*) : il scanne des transcrits locaux pour dimensionner les défauts de
  compression. Il **n'audite aucune décision live**.
- **« breakpoint »** dans leur code = marqueur de **prompt-cache Anthropic** (`cache_control`,
  où placer le marqueur), **pas** un point d'arrêt de débogueur.
- **« replay »** = harnais de bench/charge, **pas** du time-travel de session.

### CCR vs watchpoint : deux designs pour la même peur

CCR est la **réponse alternative de Headroom à l'angoisse que le watchpoint de CCOS
adresse** : *« et si le modèle a besoin du contexte qu'on a jeté ? »*

- **Headroom** répond : *retrieval à la demande par hash* (`headroom_retrieve`).
- **CCOS** répond : *watchpoint d'éviction + replay déterministe* (`missing <node>`,
  `recall_what_if`, post-mortem).

Ce n'est **pas** « eux rien, nous tout ». Ce sont deux solutions différentes au même
problème — et c'est exactement comme ça qu'il faut le présenter, sans caricaturer l'autre.

---

## 6. Positionnement net

- L'axe revendiqué par CCOS — **mémoire de travail rejouable, auditable, debuggable
  post-mortem (flight recorder + watchpoint d'éviction)** — est, *preuves dans leur source à
  l'appui*, le seul que Headroom n'occupe pas.
- Garder l'honnêteté sur deux limites : (1) axe **étroit** ; (2) unique ≠ utile — la valeur
  se démontre, elle ne se décrète pas.
- **Phrase de contraste défendable :**
  > *« Headroom optimise **ce que le modèle voit** (compresser + récupérer) ;
  > CCOS rend **rejouable, auditable et debuggable** ce que le modèle **a vu**. »*

---

## Annexe — provenance & méthode

- **Cible :** `github.com/headroomlabs-ai/headroom`, branche `main`, lue le **2026-06-23**
  (commit non épinglé — instantané susceptible de dériver).
- **Méthode :** lecture des fichiers bruts via `raw.githubusercontent.com` + listings
  d'arborescence. La couche de récupération résume via un petit modèle : les faits porteurs
  ont été confirmés contre des citations littérales (signatures, SQL, imports, constantes).
- **Fichiers réellement lus (extrait) :** `crates/headroom-core/{Cargo.toml, src/lib.rs,
  src/transforms/mod.rs, src/ccr/mod.rs, src/ccr/backends/sqlite.rs}`,
  `crates/headroom-{py,parity,proxy}/Cargo.toml`,
  `headroom/transforms/{smart_crusher,code_compressor,kompress_compressor,cache_aligner,
  content_router,text_crusher}.py`, `headroom/compression/universal.py`,
  `headroom/ccr/{tool_injection,response_handler,context_tracker,batch_store}.py`,
  `headroom/cache/compression_store.py`, `headroom/memory/{core,models,storage_router,
  system,tracker}.py`, `headroom/memory/adapters/sqlite_vector.py`,
  `headroom/audit/{__init__,codex,maturation,reads}.py`,
  `headroom/storage/{base,sqlite,jsonl}.py`, `headroom/savings_ledger.py`,
  `headroom/learn/{scanner,analyzer,writer,loops}.py`,
  `headroom/relevance/{bm25,embedding,hybrid}.py`, `headroom/{capture,observability}/*`,
  `docs/spec/003-adrs.md` (ADR-008).
- **Limites :** lecture non exhaustive (corps de certains `.rs` et shims `diff/log/search`
  non inspectés ligne à ligne) ; les claims sur diff/log/search reposent sur le routeur +
  les déclarations de modules Rust.
