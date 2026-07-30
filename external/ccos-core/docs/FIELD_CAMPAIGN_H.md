# Campagne H — Le test décisif : bugs multi-fichiers (CCOS vs dump)

Le corpus de 16 runs a montré que les vrais bugs testés étaient **mono-fichier**, où
CCOS est à parité avec un dump du fichier. L'avantage de frugalité de CCOS est un
phénomène **multi-fichier** — non encore testé. Cette campagne le teste : des bugs où
**le `cargo test` panique dans un fichier, mais la cause est dans un *autre*.**

## La question décisive

> Quand le symptôme et la cause sont dans des fichiers différents, CCOS ramène-t-il la
> **cause** dans le contexte, en **bien moins de tokens** que dumper tout le `src/` ?

- **Baseline-symptôme** : dumper le seul fichier du panic → **rate la cause** (autre
  fichier).
- **Baseline-tout** : dumper tout `src/` → contient la cause mais **cher**.
- **CCOS** : `recall around <fichier-du-panic>` → la **région causale**, qui inclut le
  fichier-cause par l'arête `use crate::…`, frugalement. ← l'hypothèse à prouver.

## Protocole de mesure (model-free, robuste — par bug Hk)

1. `cargo new --lib hk`, écris les fichiers du bug (ci-dessous), `cd hk`.
2. `cargo test` → **capture la sortie rouge**. Note le **fichier-symptôme** (la
   localisation du panic, `src/X.rs:ligne`).
3. Ingère **tous** les fichiers `src/` dans un workspace neuf `corpus_H/Hk/` (l'agent
   les aurait lus).
4. `page_fault {output:"<sortie cargo test brute>"}` → CCOS pressurise le symptôme.
5. **CCOS** : `recall {around, anchor:"src/X.rs", budget:2048}` → enregistre :
   - `cause_in_window` : `file:src/<cause>.rs` est-il dans la fenêtre ? *(le booléen clé)*
   - `ccos_tokens` : tokens de la fenêtre.
6. **Baselines** : `all_tokens` = somme (chars/4) de tous les `src/*.rs` ;
   `symptom_tokens` = celui du seul fichier-symptôme.
7. Export : `ccos postmortem corpus_H/Hk --json > corpus_H/Hk.json`.

**Tableau visé** : `bug | sauts | cause_in_CCOS | ccos_tokens | all_tokens | ratio`.

**Critère de succès** : pour les bugs ≤ 3 sauts, la cause **est** dans la fenêtre CCOS,
à `ccos_tokens ≪ all_tokens`. Pour H4 (3 sauts) on teste la limite. Pour H5 (leurre),
CCOS inclut la **vraie** cause et **pas** le décoy.

*(Optionnel — moitié suffisante)* : envoie au LLM local la fenêtre CCOS vs le dump-tout
**au même budget**, « corrige le fichier en cause », et note par `cargo test`.

---

## Les bugs (copier-coller, prêts à injecter)

> Dans chaque crate, `src/lib.rs` déclare les `pub mod …` et porte le test. Le **panic
> surgit dans le fichier-symptôme**, la **cause est ailleurs**.

### H1 — Constante fausse, 1 saut  *(symptôme `writer.rs` → cause `config.rs`)*
```rust
// src/config.rs
pub fn buffer_size() -> usize { 0 }              // BUG : devrait être 8
// src/writer.rs
use crate::config;
pub fn render() -> u8 {
    let mut buf = vec![0u8; config::buffer_size()];
    buf[0] = 42;                                 // panic : index out of bounds (len 0)
    buf[0]
}
// src/lib.rs
pub mod config; pub mod writer;
#[cfg(test)] mod tests { #[test] fn renders() { assert_eq!(crate::writer::render(), 42); } }
```

### H2 — Off-by-one dans un helper, 1 saut  *(`access.rs` → `idx.rs`)*
```rust
// src/idx.rs
pub fn last_index(len: usize) -> usize { len }   // BUG : devrait être len - 1
// src/access.rs
use crate::idx;
pub fn last(v: &[i32]) -> i32 { v[idx::last_index(v.len())] }   // index out of bounds
// src/lib.rs
pub mod idx; pub mod access;
#[cfg(test)] mod tests { #[test] fn t() { assert_eq!(crate::access::last(&[1,2,3]), 3); } }
```

### H3 — Diviseur nul, 2 sauts  *(`engine.rs` → `math.rs` → `settings.rs`)*
```rust
// src/settings.rs
pub fn divisor() -> i64 { 0 }                    // BUG : devrait être 2
// src/math.rs
use crate::settings;
pub fn d() -> i64 { settings::divisor() }
// src/engine.rs
use crate::math;
pub fn run() -> i64 { 100 / math::d() }          // panic : divide by zero
// src/lib.rs
pub mod settings; pub mod math; pub mod engine;
#[cfg(test)] mod tests { #[test] fn t() { assert_eq!(crate::engine::run(), 50); } }
```

### H4 — Cause à 3 sauts (teste la limite de propagation)  *(`api.rs` → ctrl → repo → `store.rs`)*
```rust
// src/store.rs
pub fn capacity() -> usize { 0 }                 // BUG : devrait être 4
// src/repo.rs
use crate::store; pub fn cap() -> usize { store::capacity() }
// src/ctrl.rs
use crate::repo; pub fn limit() -> usize { repo::cap() }
// src/api.rs
use crate::ctrl;
pub fn first() -> u8 { let v = vec![0u8; ctrl::limit()]; v[0] }   // panic : index out of bounds
// src/lib.rs
pub mod store; pub mod repo; pub mod ctrl; pub mod api;
#[cfg(test)] mod tests { #[test] fn t() { assert_eq!(crate::api::first(), 0); } }
```

### H5 — Le leurre (structure vs lexical)  *(symptôme `handler.rs` → vraie cause `config.rs`, décoy `handler_helpers.rs`)*
```rust
// src/config.rs
pub fn timeout() -> u64 { 0 }                    // BUG : la vraie cause (devrait être 200)
// src/handler.rs
use crate::config;
pub fn handle() -> u64 { 1000 / config::timeout() }   // panic : divide by zero
// src/handler_helpers.rs
// DÉCOY : lexicalement proche de handler.rs, MAIS hors du chemin causal.
pub fn handle_format(s: &str) -> String { format!("handled: {s}") }
// src/lib.rs
pub mod config; pub mod handler; pub mod handler_helpers;
#[cfg(test)] mod tests { #[test] fn t() { assert_eq!(crate::handler::handle(), 5); } }
```
**Attendu (vérifié en preview)** : CCOS `around handler.rs` **atteint la vraie cause**
`config.rs` (via `use crate::config`) ✅. Mais le décoy `handler_helpers.rs` **est aussi
tiré dans la fenêtre** — car `pub mod handler_helpers;` dans `lib.rs` connecte les
modules frères en une seule région via la racine. C'est la **précision faible** connue du
paper (recall 0.97 / précision 0.06), confirmée sur du vrai code. La vraie question de H5
n'est donc pas « le décoy est-il exclu ? » (il ne l'est pas) mais : **sous budget serré
ou avec la pression d'échec, `config.rs` est-il classé au-dessus du décoy ?** (mesure le
rang/score des deux). Un baseline lexical, lui, attraperait le décoy *à la place* de la
cause — CCOS, au moins, a la cause.

---

## Preview sur fichiers-jouets (ce repo, binaire 0.3.0) — fait

Exécuté ici via le serveur MCP (workspace neuf par bug, `cargo test` réel →
`page_fault` → `recall around <symptôme>`, budget 2048). **Tous les bugs paniquent
bien dans le fichier-symptôme attendu.**

| Bug | sauts | cause dans CCOS ? | rang de la cause | ccos_tokens | all_tokens | ratio |
| --- | ----- | ----------------- | ---------------- | ----------- | ---------- | ----- |
| H1  | 1 | ✅ oui | **#1** (0.875) | 173 | 71 | 0.41× |
| H2  | 1 | ✅ oui | top (0.875) | 139 | 60 | 0.43× |
| H3  | 2 | ✅ oui | top (0.875) | 167 | 69 | 0.41× |
| H4  | 3 | ✅ oui (chaîne `api→ctrl→repo→store` entière) | présent (0.779/0.569) | 202 | 91 | 0.45× |
| H5  | 1 (+ décoy) | ✅ oui ; décoy **présent mais classé dernier** | **#1** (0.875) ; décoy 0.569 | 201 | 92 | 0.46× |

**Trois lectures, honnêtes :**

1. **Couverture — PROUVÉE à 1, 2 et 3 sauts.** La cause d'un *autre* fichier est dans
   la fenêtre à chaque fois, y compris la chaîne à 3 sauts de H4 (avec décroissance
   visible du score, mais présente). C'est exactement ce que le corpus mono-fichier
   n'avait jamais montré.
2. **Classement sous pression — meilleur que prévu.** En H5, le décoy lexical
   `handler_helpers.rs` *entre* bien dans la région (sur-connexion par la racine de
   module `pub mod`), **mais le score causal le classe bon dernier** ; la vraie cause
   `config.rs` est **#1**. Sous budget serré (200) l'ordre est
   `config.rs > handler.rs > … > handler_helpers.rs`. Un baseline lexical classerait le
   décoy *haut* (préfixe « handler »). Le score fait le tri que la topologie seule ne
   fait pas. ✅
3. **Frugalité — PAS démontrée sur des jouets (ratio 0.41–0.46×, CCOS *plus gros*).** Sur
   des fichiers de 60–90 tokens, la fenêtre duplique le source : un même fichier apparaît
   dans son nœud `file:`, ses nœuds `sym:` et ses nœuds `use:` (chacun portant **tout** le
   fichier — granularité fichier, pas symbole). Dumper `src/` est donc *moins* cher que la
   région. **La frugalité exige (a) de vrais gros fichiers** (où `all-src ≫ région`) **et
   (b) la granularité symbole** (qu'un nœud `sym:` ne porte que sa fonction). C'est le
   point qui relie directement H à l'item roadmap Q2.

➡️ **Conclusion preview** : le *quoi* (la cause multi-fichier est atteinte et bien classée)
est acquis sur des jouets ; le *combien* (frugalité) ne se mesure que sur du vrai code
volumineux. Je l'ai mesuré — section suivante.

## Réalité sur vrai code (le `src/` de CCOS, 32 fichiers, 130 287 tokens) — la correction

Le protocole H, appliqué non plus à des crates-jouets mais au **propre `src/` de CCOS**
(32 fichiers réels, vraies arêtes `use crate::`, fichiers de 700 à 17 000 tokens),
**renverse le résultat des jouets**. Le binaire 0.3.0, via MCP (ingest des 32 fichiers →
`signal_failure file:src/mcp.rs` → `recall around file:src/mcp.rs`) donne :

| profondeur | budget | symptôme dans la fenêtre ? | deps couvertes | ce que contient la fenêtre |
| ---------- | ------ | -------------------------- | -------------- | -------------------------- |
| **3 (défaut)** | 2 048 → 32 768 | ❌ non | **0/2** | 1 nœud = le **hub `memory.rs`** entier (9 747 tok) |
| 1 | 2 048 | ❌ non | 1/2 | 1 fichier entier (`external_memory.rs`) |
| 1 | **32 768** | ✅ oui | 2/2 | les 3 fichiers entiers (20 405 tok uniques) |

**Trois causes racines, chacune mesurée (plus des hypothèses) :**

1. **Granularité fichier — le blocant n°1 (Q2), désormais prouvé.** Chaque nœud
   `sym:`/`use:`/`mod:` porte **tout son fichier**. Un seul nœud (`sym:memory.rs:MemoryGraph`
   = 9 747 tok) **dépasse à lui seul un budget de 2 048**. La fenêtre ne tient donc qu'1
   nœud = 1 fichier dupliqué. La **région complète** (budget ∞) autour de *n'importe quel*
   ancre = **les 32 fichiers, ~2 000 000 tokens** pour 130 287 tokens uniques : un facteur
   **15× de duplication** (les ~6 symboles de `main.rs` portent chacun les 64 655 chars de
   `main.rs`). Tant que `sym:` ne porte pas seulement *sa* fonction, le budget est brûlé.
2. **Propagation à profondeur fixe — elle inonde les graphes denses. ✅ CORRIGÉ.**
   `signal_failure` sur **un** fichier pressurisait **518 / 1037 nœuds** (depth 3) : la
   moitié du dépôt, et le hub finissait par surclasser le symptôme. Correctif livré :
   **propagation consciente du degré** — un nœud *distribue* sa pression sur ses arêtes
   (`damp = failure_fanout / out_degree`, no-op sous `failure_fanout=6`) au lieu de la
   *répliquer*, plus un **cutoff au plancher de paging** (on arrête de relayer une pression
   qui ne peut plus rien pager). Résultat mesuré : **flood 518 → 37 nœuds (14×)** ; le
   **défaut `depth=3` atteint maintenant le symptôme + 2/2 deps en 2035 tokens** (identique
   à depth=1 — la tension profondeur/densité est **résolue**) ; le symptôme `mcp.rs` repasse
   **#1** (0.850) et le hub `memory.rs` sort de la fenêtre. La **réuse-clé** : sur une chaîne
   creuse `a→b→c→d` (degré 1), `damp=1`, donc la cause à 3 sauts `d.rs` est **toujours**
   atteinte — le degré-conscient préserve la portée profonde là où baisser la profondeur
   l'aurait perdue. (`CCOS_FAILURE_FANOUT` ajustable.)
3. **Domination du hub & bruit de composante. ✅ CORRIGÉ.** Diagnostic initial : `memory.rs`
   (utilisé par presque tout) surclassait le symptôme. La **partie hub a été réglée par #2**
   (le symptôme `mcp.rs` est repassé #1). Restait une imprécision **mesurée plus finement** :
   la région de `mcp.rs` est **31 fichiers sur 32** (le vrai graphe `use crate::` de CCOS est
   quasi entièrement connexe). Ce **n'était donc pas** un pont `dep:`/std (le clustering les
   exclut déjà) mais une **composante dense sans terme de proximité** : sans pression d'échec
   forte, la *récence* fait qu'un nœud lointain (`use:commands_demo`, ~3 sauts) **égalait** les
   vraies deps (1 saut). Correctif livré : **décroissance par proximité** dans `around`/`task`
   — score `×= proximity_decay^(distance au point d'ancrage)`, distance par BFS bidirectionnel
   **qui ne relaie pas à travers les hubs `dep:`**. Résultat : le bruit `commands_demo` (rang 3
   à 0.444) **sort** ; le top devient symptôme → ses 2 deps réelles → ses propres symboles. Le
   symptôme + 2/2 deps tiennent toujours en 2045 tokens, et la chaîne creuse garde ses 3 sauts.
   (`CCOS_PROXIMITY_DECAY` / `CCOS_PROXIMITY_HOPS` ajustables.)

**Le verdict frugalité, sans fard (diagnostic d'origine)** : les 3 fichiers pertinents =
**20 405 tokens**. CCOS exigeait un budget de **32 768** pour les livrer (taxe de
duplication) **et seulement** à `depth=1` ; au **défaut `depth=3`, il ne les atteignait sous
aucun budget testé** — il servait le hub.

> **MISE À JOUR (les trois causes corrigées).** Granularité symbole (#1) + propagation
> degré-consciente (#2) + proximité d'ancrage (#3). La même mesure donne maintenant, au
> **réglage par défaut, budget 2048** : **symptôme #1, ses 2 deps réelles, ses symboles —
> en ~2045 tokens, sans bruit de composante.** Bilan avant→après : region complète 15× → 1.2× ;
> flood 518 → 37 ; couverture deps 0/2 → 2/2 ; le nœud non pertinent `commands_demo` évincé.
> CCOS livre les bons fichiers **et dit lesquels**, frugalement, par défaut, sur du vrai code.

Ce n'était pas un échec de la machinerie (event-sourcing, time-travel, persistance, audit
fonctionnent) : c'était l'**assemblage de contexte** — le cœur de la thèse — qui ne
survivait pas à l'échelle réelle. **Les trois leviers sont livrés**, chacun avec test de
régression et chiffres avant/après ; ces items sont passés de « roadmap spéculative » à
« corrigés et mesurés ». Reste à **confirmer sur un autre dépôt** (Thor : ripgrep/bat/fd) que
le gain n'est pas propre au `src/` de CCOS.

## Confirmation indépendante — crate `syn` (44 fichiers, 191 130 tokens, **zéro code CCOS**)

Pour vérifier que le gain n'est pas propre au `src/` de CCOS, la triade a été mesurée sur
**`syn-1.0.103`** (sources vendorisées, layout `src/*.rs` plat, 141 `use crate::`), ~1.5× la
taille de CCOS. Protocole structurel : ingérer les 44 fichiers, `signal_failure(depth=3)` sur
un fichier, `recall around` ce fichier, et compter combien de ses **vraies deps `use crate::`**
entrent dans la fenêtre, à quel coût.

- **#1 granularité — confirmé.** Région complète = 111 618 tokens vs 191 130 uniques →
  **0.58×** (aucune duplication ; sur CCOS c'était 15× → 1.2×).
- **#2 flood — confirmé (borné).** `signal_failure` sur un fichier pressurise **46–122 nœuds**
  (sur ~2 000) selon la taille du fichier — borné, jamais la moitié (proportionnel au nombre
  de symboles du fichier, pas à la taille du dépôt).
- **#3 couverture + localité — confirmé, au bon budget.** À budget **8192** (3.4–4.3 % de
  l'all-src) :

  | ancre | deps atteintes | fenêtre (tok) | % all-src | fichiers-bruit |
  | ----- | -------------- | ------------- | --------- | -------------- |
  | item.rs  | **7/7** | 6 577 | 3.4 % | 0 |
  | token.rs | **7/7** | 7 039 | 3.7 % | 0 |
  | ty.rs    | **6/6** | 8 167 | 4.3 % | 2 |
  | expr.rs  | **5/5** | 8 079 | 4.2 % | 0 |

  Le **voisinage causal entier** (l'ancre + **toutes** ses vraies deps) tient en **~25–30× moins
  de tokens que dumper `syn`**, quasi sans bruit. Le linking cross-file marche, la localité
  tient.

> **Le bémol honnête que `syn` a révélé (que le `src/` plus petit de CCOS cachait) : le budget
> devait suivre la taille de l'ancre. ✅ CORRIGÉ.** Les fichiers de `syn` sont gros
> (`item.rs` ≈ 88 symboles) ; à budget 2048, le contenu propre de l'ancre remplissait la
> fenêtre et n'atteignait qu'1/7 deps. **Deux plafonds** (header ≤ 24 signatures ;
> aucun fichier > 40 % du budget ancré — `CCOS_HEADER_SYMBOLS` / `CCOS_RECALL_FILE_CAP`)
> donnent maintenant, à **budget fixe 2048** : item.rs **7/7**, token.rs **7/7**, ty.rs
> **6/6**, expr.rs **5/5** — quelle que soit la taille de l'ancre. Détail + simulation dans
> `docs/DESIGN_recall_budget.md`. (Aucun seul plafond ne suffit : 5/7 chacun, 7/7 ensemble.)

**Conclusion** : sur un dépôt **indépendant**, la triade donne une fenêtre multi-fichiers
**correcte (100 % des deps), locale (≈0 bruit) et frugale (~4 % de l'all-src)** — au budget
dimensionné à l'ancre. Le test de Thor sur encore un autre dépôt (ripgrep/bat/fd) reste utile,
mais la généralisation est déjà démontrée ici.

## Suffisance (Q7) — le contexte fait-il *résoudre* ? (Campagne J, Thor)

La couverture/frugalité est **nécessaire** ; la question qui décide la valeur de CCOS est la
**suffisance** : à budget égal, le contexte CCOS fait-il *résoudre* un bug à un LLM local
qu'un dump ne peut pas ? Protocole (Thor, `qwen3-coder:30b`, budget 2048) : 3 bugs
multi-fichiers **contrôlés** — fichier-symptôme **paddé au-delà du budget**, cause = une
constante dans un **fichier-dep** ; deux contextes à budget égal (région CCOS vs dump naïf
tronqué) ; appliquer le diff du modèle, `cargo test`. **Vérité terrain = la sortie
`cargo test`** (pas une heuristique — voir la note grader plus bas).

| Bug (synthétique) | symptôme→cause | cause dans CCOS / baseline | **CCOS** | **baseline** | fichier patché baseline |
| ----------------- | -------------- | -------------------------- | -------- | ------------ | ----------------------- |
| JM1 `buffer_size` (*devinable*) | writer→config | oui / non | ✅ 3/3 | ✅ 3/3 *(deviné)* | config.rs (deviné juste) |
| JM2 `HEADER_SIZE` | renderer→config | oui / non | ✅ 3/3 (corrige config) | ❌ **1/3** | renderer.rs (hack du symptôme) |
| JM3 `MIN_SCORE` | reader→filter | oui / non | ✅ 3/3 (corrige filter) | ❌ **0/3** | renderer.rs (mauvais fichier inventé) |

**Résultat** : sur les bugs où la *valeur* de la cause n'est pas inférable du symptôme
(JM2, JM3), **CCOS résout (3/3) là où le dump à budget égal échoue** — il patche la racine
(le fichier-cause), tandis que la baseline hacke le symptôme (JM2) ou hallucine un fichier
sans rapport (JM3), parce que le budget a tronqué la cause de son contexte. **C'est la
première preuve de suffisance** : le contexte CCOS ne fait pas que *couvrir* la cause, il
fait *résoudre*.

**Bémols honnêtes (pas de survente) :**
- **n = 2 cas décisifs**, **synthétiques** (bugs construits, pas des commits réels minés),
  **un seul modèle**. C'est une **démonstration de mécanisme**, pas encore un résultat à
  l'échelle. → relance : vrais bugs (`bat`/`ripgrep`, fix dans une dep d'un gros fichier) +
  2–3 modèles.
- **JM1 ne compte pas** : la cause est devinable depuis l'import + l'assert ; la baseline a
  deviné le bon fichier. Le gain n'existe que quand la *valeur* est **définie** dans la cause.
- **CCOS n'est pas *moins cher* ici** : 2276/2351 tok vs 2048. Le gain est la **correction**
  à budget comparable, pas la frugalité (axe séparé : la couverture).
- **Grader corrigé** : le premier `result.json` de Thor notait JM3 CCOS `resolved=False`
  alors que `cargo test` montrait **3 passed** (faux négatif d'une heuristique sur le fichier
  patché). `scripts/ccos_grade.py` note désormais la **vérité `cargo test`**. Le test
  « cause » direct (`assert_eq!(cause(), valeur_correcte)`) défait le hack du symptôme : un
  patch local au symptôme passe 2/3 mais échoue ce test-là — seul un fix racine passe 3/3.

### Round 2 — multi-modèles (3 bugs × 3 modèles, grader = `ccos_grade.py`)

| Bug | qwen3-coder:30B | DeepSeek-V2-Lite (~16B) | qwen2.5-coder:1.5B |
| --- | --------------- | ----------------------- | ------------------ |
| JR1 `MIN_SCORE` (filter) | CCOS ✅ 3/3 / base ❌ CE | CCOS ✅ 3/3 / base ❌ 0/3 | CCOS ❌ CE / base ❌ CE |
| JR2 `backoff` | CCOS ✅ 3/3 / base ❌ CE | CCOS ✅ 3/3 / base ❌ 0/3 | CCOS ❌ CE / base ❌ 1/3 |
| JR3 `BUFFER_CAPACITY` | *(bug mal conçu — écarté)* | *(idem)* | *(idem)* |

**Le résultat solide** : sur les bugs bien formés (JR1, JR2) × modèles **capables** (30B et
16B), **CCOS résout 4/4, la baseline 0/4** — et l'asymétrie de patch est totale : la baseline
patche **toujours** le fichier-symptôme (cause tronquée), CCOS **toujours** le fichier-cause.
La suffisance **tient sur deux familles de modèles**.

**Nuances honnêtes (corrections aux claims de la 1ʳᵉ synthèse) :**
- Le gain **n'est pas** « model-independent » : à **1.5B les deux échouent** (le petit modèle
  n'exploite aucun contexte ; il a même patché `lib.rs`, le mauvais fichier). C'est « robuste
  sur modèles capables », pas universel. Le gap est **maximal à 16B** (3/3 vs 0/3) puis
  **s'effondre à 1.5B** — non-monotone, un plancher de capacité existe.
- **CCOS résout 4/9 cellules** (pas « 9/9 bon fichier » : sur JR3 il patche la bonne cause mais
  ne résout pas ; à 1.5B il échoue). Le « bon fichier » ≠ « résolu ».
- **Le format annoté gêne les petits modèles** (`// sym:` lu comme du code → compile error).
  Piste : un **mode contexte brut** sans annotations pour les modèles faibles.
- **JR3 mal conçu** : un test assert la valeur *buggée* (`len()==64`), donc corriger la racine
  casse ce test. Écarté.
- **Toujours synthétique** : pas encore de vrais commits minés (DeepSeek plantait en HTTP 500).
  C'est le dernier pas — round 3 ci-dessous.

### Round 3 — vers de vrais bugs, et la trouvaille qui borne la thèse

Objectif : miner de **vrais** commits de correction (`bat`/`ripgrep`/`fd`) où la cause est dans
un fichier différent du symptôme. Un seul modèle (`qwen3-coder:30B`). Vérité = `ccos_grade.py`.

| Bug (motif réel) | CCOS | baseline | détail |
| ---------------- | ---- | -------- | ------ |
| R01 `bat` config-default | ✅ **3/3** (patche `config.rs`) | ❌ CE (`printer.rs`) | cause tronquée côté baseline — **vrai gain** |
| R02 `ripgrep` line-buffer | ❌ **CE** | ❌ 1/3 | le modèle a *droppé une ligne `use`* → CE |
| R03 `fd` walk-depth | ❌ **CE** (patche `walk.rs`, **pas** la cause `config.rs`) | ❌ CE | CCOS **avait** la cause, le modèle a réécrit le **symptôme** et cassé la compilation |

**Vérité terrain : 1 gain net sur 3** (R01). (La 1ʳᵉ synthèse de Thor annonçait R03 « 3/3 / config.rs » ;
son propre JSON dit `resolved:false, compile_error:true, patched:walk.rs` — **2ᵉ divergence
récit↔artefact**, corrigée ici.) **R03 est l'enseignement** : *même avec la cause dans le
contexte*, un modèle peut patcher le symptôme et casser le build. **Le bon contexte est
nécessaire, pas suffisant — il faut que le modèle l'exploite.**

**La trouvaille qui borne la thèse** : Thor a scanné **5709 commits** (bat+ripgrep+fd) → les
vrais bugs **multi-fichiers sont rares (~1–2 % des fixes)**. La plupart des correctifs touchent
le fichier du symptôme ; les tests sont ajoutés *avec* le fix, pas avant. Le « cas parfait »
(cause dans un fichier, test pré-existant qui échoue ailleurs, symptôme tronquable) **est
rare dans le débogage réel archivé** — R01/R02/R03 sont donc *synthétiques sur motifs réels*,
faute d'avoir pu en miner un vrai.

### Mot de la fin — Campaign J, sans fard

- **Couverture / asymétrie : acquise (100 %).** La baseline n'a jamais la cause tronquée ; CCOS
  l'a toujours. Frugalité prouvée sur CCOS, `syn`, `bat`.
- **Résolution : réelle mais conditionnelle.** Gains nets : JM2, JM3, JR1×{30B,16B}, JR2×{30B,16B},
  R01. Échecs *malgré* le contexte : R03 (mauvais fichier), R02 (import droppé), tous les 1.5B.
  → il faut un **modèle capable qui exploite** le contexte.
- **Applicabilité : étroite.** ~1–2 % des fixes réels ont cette structure.

**Conclusion honnête** : CCOS fournit de façon fiable et frugale le voisinage causal (le cœur
technique tient), et ce contexte *peut* faire résoudre ce qu'un dump à budget égal ne peut pas —
mais sur une **minorité** de bugs et **seulement si** le modèle s'en sert bien. Ce n'est pas un
triomphe ; c'est un outil à **valeur réelle sur un créneau précis**, mesuré honnêtement. La
suite utile n'est pas « plus de bugs synthétiques » mais soit (a) un **mode contexte brut** pour
les modèles faibles, soit (b) mesurer la valeur de CCOS sur l'**assemblage de contexte en tâche
générale** (où les dépendances sont omniprésentes), pas seulement sur le bug-fix-avec-test —
créneau qui, lui, s'est avéré rare.

## Valeur en tâche générale — le créneau large (model-free)

Le bug-fix multi-fichiers est rare (~1–2 %). Mais le *besoin* sous-jacent — « je travaille sur
le fichier X, ai-je ses dépendances cross-file dans mon budget ? » — est **omniprésent** (tout
fichier non trivial utilise ses deps). Mesuré sur **chaque** fichier de vrais crates
(`scripts/ccos_context_value.py`, budget 2048, sans LLM, sans signal d'échec — juste « je
travaille sur X ») : pour chaque arête de dépendance `use crate::Y`, le contenu de Y est-il dans
`recall around X` ? Comparé à ouvrir X naïvement (tronqué au budget, puis les voisins).

| crate | arêtes-dep | **CCOS** | dump naïf | gros fichiers (> budget) : CCOS / naïf |
| ----- | ---------- | -------- | --------- | -------------------------------------- |
| `syn` | 112 | **81 %** | 2 % | 79 % / 0 % |
| CCOS `src/` | 57 | **93 %** | 0 % | 92 % / 0 % |
| `serde_json` | 14 | **100 %** | 0 % | 100 % / 0 % |

**Sur 183 arêtes de dépendance réelles, CCOS place la dépendance dans une fenêtre de 2048 tokens
81–100 % du temps ; ouvrir le fichier en donne 0–2 %.** Et c'est concentré là où ça compte : sur
les **gros fichiers** (plus grands que le budget), ouvrir le fichier **tronque toutes ses deps**
(0 %), CCOS en garde **79–100 %**. C'est le créneau **large** que le test de résolution (rare)
sous-mesurait : l'avantage de couverture n'est pas réservé aux bugs multi-fichiers, c'est le
quotidien « travailler sur un fichier et avoir besoin de ses deps ».

**Bémol honnête** : ceci mesure la **couverture** (la dep est *présente*), pas la **valeur
prouvée** (la dep *aide*). La suffisance (couverture → meilleur résultat) n'a été testée que sur
le créneau étroit du bug-fix, où elle tient *si le modèle exploite le contexte*. Donc : avantage
de **couverture large et fort** (prouvé ici), **suffisance démontrée sur l'étroit**. On ne
sur-vend pas : la couverture est nécessaire et omniprésente ; reste à prouver que l'aide qu'elle
procure se généralise au-delà du bug-fix.


## Grille de résultats à remplir (vrai code, Thor)

Refais les 5 bugs **mais greffés sur de vrais fichiers volumineux** (insère le mauvais
constant dans un module réel de `ripgrep`/`bat`/`fd` ; le symptôme panique ailleurs). Là,
`all_tokens` = plusieurs milliers, et on verra si la région reste petite.

| Bug | sauts symptôme→cause | cause dans CCOS ? | ccos_tokens | all_tokens | ratio |
| --- | -------------------- | ----------------- | ----------- | ---------- | ----- |
| H1  | 1 | ? | ? | ? | ? |
| H2  | 1 | ? | ? | ? | ? |
| H3  | 2 | ? | ? | ? | ? |
| H4  | 3 | ? | ? | ? | ? |
| H5  | 1 (+ décoy) | ? (rang cause vs décoy) | ? | ? | ? |

**Lecture** : si, sur du vrai code, CCOS ramène la cause d'un autre fichier à quelques
centaines de tokens là où le dump-tout en coûte plusieurs milliers — **c'est la preuve de
frugalité que le corpus n'avait pas encore montrée.** Si le ratio reste ≤ 1 même sur de
gros fichiers, **la granularité symbole (Q2) devient l'item bloquant**, pas un détail.

Ramène `corpus_H/` complet — on le déroule ensemble.
