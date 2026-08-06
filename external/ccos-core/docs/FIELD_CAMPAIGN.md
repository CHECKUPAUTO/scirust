# Campagne de collecte de données terrain — CCOS (Thor)

Objectif : accumuler un **corpus** de `session.json` / `.oplog` riche et **orienté
questions ouvertes** — pas du volume pour le volume. Chaque campagne répond à une
question précise et produit des exports analysables. C'est ce corpus qui dira quel
item de roadmap (profondeur, granularité symbole, parseur) pince vraiment en usage
réel.

## Les questions ouvertes (le *pourquoi* de chaque run)

- **Q1 — Profondeur causale.** Le fix `depth=3` atteint-il les causes à 3 sauts ? Où
  la décroissance par saut perd-elle encore une cause profonde ?
- **Q2 — Granularité du contenu.** Quand le contenu *à granularité fichier* (un nœud
  `sym:` porte tout son fichier) fait-il vraiment mal — gros fichiers, gros budget ?
- **Q3 — Utilité réelle de `around`.** Sur de vrais bugs, la région ancrée donne-t-elle
  un contexte utile et frugal ? Le hook dogfood capte-t-il du signal exploitable ?
- **Q4 — Vraies dérives.** Sur du vrai travail, l'agent chasse-t-il le symptôme ?
  `missing` / `energy` rendent-ils la dérive lisible ?
- **Q5 — Limites du parseur.** Où le linking cross-file casse-t-il (Python/JS/Go) ?
- **Q6 — Échelle.** La timeline / le time-travel tiennent-ils sur de longues sessions
  (compaction) ?
- **Q7 — Condition suffisante.** À budget égal, le contexte CCOS aide-t-il *vraiment*
  un LLM local à résoudre un bug (vs un dump de fichiers / RAG) ?

## Protocole de collecte

- Un **workspace neuf par run** : `corpus/<camp><run>/` (dossier — supporté).
- Chaque run se termine par un export : `ccos postmortem corpus/<camp><run> --json >
  corpus/<camp><run>.json`.
- Pour les campagnes organiques (D), **hook dogfood ON** (`scripts/ccos_self_feed.py`).
- En fin de campagne : `scripts/fleet_collect.sh` ou un simple `tar` du dossier
  `corpus/` — ramène-le, on l'analyse ensemble.

---

## Campagne A — Profondeur causale & décroissance (Q1) — 10 runs

🎯 Valider `depth=3` et **cartographier** où les causes profondes se perdent.

📋 Construis des crates en chaîne de profondeur **1, 2, 3, 4, 5** (`a → b → … `, chaque
fichier `use crate::<suivant>`, le dernier = la cause). Pour chaque profondeur, fais
**2 runs** : un à `CCOS_PAGE_FAULT_DEPTH=3` (défaut), un à `=5`.
Dans chaque run : ingère la chaîne → `page_fault` sur le fichier d'entrée (symptôme) →
`recall around <entrée>` (budgets 2048 **et** 200) → export.

📊 **Donnée visée** : à quelle profondeur la cause cesse de recevoir de la pression
(`failure_relevance` de la cause par profondeur) ; et à quel budget elle est évincée
malgré sa présence dans la région. Tableau profondeur × budget × (cause pressurisée ?
dans la fenêtre ?).

---

## Campagne B — Vrais bugs, boucle de correction (Q3, Q4, Q7) — 15+ runs

🎯 Le cœur : CCOS sur de **vrais** bugs, comme un agent réel.

📋 Mine 15–20 vrais commits de correction (de `fd`, `bat`, `hyperfine`, `ripgrep`, ou
tout crate Rust que tu as). Pour chacun : `git checkout` l'arbre **avant** le fix.
Puis travaille le bug **avec CCOS** : lis les fichiers (→ `ingest`), lance
`cargo test` (→ `page_fault` sur la sortie rouge), tente un fix, re-teste — jusqu'à
vert ou abandon. Export à la fin.

📊 **Donnée visée** : la cause du bug était-elle dans le `recall around <fichier-fautif>`
(et à combien de tokens) ? L'agent a-t-il dérivé (un `energy`/`missing` post-mortem le
montre-t-il) ? Combien de tours de `page_fault` jusqu'au vert ? C'est la donnée la plus
précieuse — garde le commit-hash de chaque bug dans le nom du run.

---

## Campagne C — Sessions longues & compaction (Q6) — 3 runs

🎯 Tenue sur le long horizon.

📋 Une **grosse tâche** par run (implémenter une feature, refactorer un module,
réécrire un fichier en plusieurs passes) — vise **150–300 opérations**. Lance avec
`CCOS_OPLOG_MAX=64 CCOS_OPLOG_KEEP=16` pour déclencher la compaction plusieurs fois.
Export **tous les ~50 ops** (cron : `ccos postmortem ws --json > corpus/C<n>_t<k>.json`),
puis **tue + rouvre** une fois en milieu de session.

📊 **Donnée visée** : la timeline survit-elle à la compaction + restart ? `recall_what_if`
sur un step *retenu* vs *sous le plancher* ? L'agent dérive-t-il sur 200 pas (la fenêtre
finit-elle pleine de bruit) ? Taille du `.oplog` dans le temps (bornage effectif).

---

## Campagne D — Dogfood organique (Q3, Q4) — collecte continue

🎯 CCOS en usage **réel et non scénarisé**.

📋 Branche le hook `PostToolUse`, `CCOS_WORKSPACE=corpus/D/workspace.ccos`, et laisse
Thor faire son **travail normal** pendant une journée (ou plus). Aucun appel CCOS
explicite — tout passe par l'intercept. Export **quotidien** (cron).

📊 **Donnée visée** : ratio signal/bruit d'un graphe né du travail organique (combien de
fichiers réellement pertinents vs touchés en passant) ; le `ccos://session/context` (qui
s'ancre sur l'échec actif) injecte-t-il quelque chose d'utile au fil de l'eau ? Premiers
vrais drifts non provoqués.

---

## Campagne E — Multi-langage (Q5) — 5 runs

🎯 Cartographier honnêtement la limite du parseur (calibré Rust).

📋 Ingère de vrais fichiers **Python**, **JS/TS**, **Go**, **C** (3–6 fichiers avec
imports cross-file par langage). `recall around <fichier-entrée>` et regarde si la
chaîne d'imports remonte. Export par langage.

📊 **Donnée visée** : pour chaque langage, le cross-file linking produit-il des arêtes
`file→file` ou juste des nœuds isolés ? Là où il casse, c'est l'item parseur P1.3 qui
se justifie (ou pas) en priorité. **Résultat attendu honnête : Rust fort, le reste
fin.**

---

## Campagne F — Stress & cas pathologiques (Q2) — 6 runs

🎯 Trouver où `working_set` / `around` / le budget se dégradent.

📋 Un cas par run : (1) un **très gros fichier** (5k+ lignes) ingéré seul ; (2) un
fichier **généré** (massif, répétitif) ; (3) une **dépendance circulaire** (`a↔b`) ;
(4) des fichiers **sans aucun import** ; (5) un **monorepo** (50+ fichiers, plusieurs
sous-systèmes) ; (6) un `recall around` sur un nœud inexistant. Export chacun.

📊 **Donnée visée** : le gros fichier remplit-il/dépasse-t-il le budget à lui seul
(quand la granularité symbole deviendrait nécessaire — Q2) ? Les régions sur le monorepo
sont-elles cohérentes ou tout fusionne-t-il ? Comportement aux bords.

---

## Campagne G — Condition suffisante : le contexte aide-t-il ? (Q7) — 10 runs

🎯 La question qui compte : à **budget égal**, CCOS fait-il *résoudre* mieux/aussi bien
en moins de tokens ?

📋 Reprends 10 bugs de la campagne B. Pour chacun, construis **deux contextes au même
budget** : (a) `recall around <fichier-fautif>` (CCOS) ; (b) un dump brut du fichier
fautif + voisins lexicaux (baseline). Envoie chacun au **LLM local** (qwen/deepseek sur
le Thor) avec « corrige ce fichier », et **note par `cargo test`**. Enregistre tokens
d'entrée + résolu/non.

📊 **Donnée visée** : taux de résolution CCOS vs baseline, et **tokens jusqu'au succès**.
C'est la Phase 4 du paper, à l'échelle, sur ton matériel. (Rappel honnête : on s'attend à
la **parité de résolution** mais à une **frugalité** nette côté CCOS — c'est le seul axe
où il gagne. Si la résolution diverge, c'est une donnée majeure.)

---

## Grille de priorité

| Camp. | Question | Effort | Valeur de la donnée |
| ----- | -------- | ------ | ------------------- |
| **B** | Q3/Q4/Q7 | élevé  | 🔥 la plus précieuse (vrais bugs, vraies dérives) |
| **G** | Q7       | élevé  | 🔥 tranche « est-ce que ça aide vraiment » |
| A | Q1 | moyen | valide/borne le fix profondeur |
| D | Q3/Q4 | faible (passif) | usage réel, drifts non provoqués |
| F | Q2 | moyen | déclenche (ou non) la granularité symbole |
| C | Q6 | moyen | tenue longue durée |
| E | Q5 | faible | borne honnête du parseur |

**Ordre conseillé** : D en fond continu (passif) + A puis B puis G (les trois qui
décident la roadmap). Ramène le `corpus/` quand B a 5+ runs — on commence à dérouler.
