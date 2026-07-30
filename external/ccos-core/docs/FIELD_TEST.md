# Protocole de test terrain — CCOS sur le Jetson (Thor)

Batterie de tâches **graduées** pour valider les capacités de CCOS en conditions
réelles sur le Thor. Chaque tâche cible une nouveauté précise et monte en complexité ;
l'ensemble culmine sur un **vrai drift** à dérouler en post-mortem.

**Légende** — 🟢 *en live* = outils MCP que l'agent appelle ; 🔵 *en post-mortem* =
CLI `ccos postmortem` (commandes `diff` / `energy` / `missing`) après coup.

**Préalable**
- Dernière version de CCOS (≥ commit `9322ca6`).
- Serveur lancé en **persistant** : `["mcp", "workspace.ccos"]`, ou un dossier de
  workspace (`["mcp", "ws/"]`).
- Un workspace **neuf par tâche** (`ws1/`, `ws2/`, …) pour ne pas mélanger les
  timelines.
- `ingest` accepte l'uri avec ou sans préfixe (`src/api.rs` == `file:src/api.rs`), et
  `recall around` résout les deux formes d'ancre.

---

## Tâche 1 — Chaîne causale profonde (5 sauts) + budget

🎯 **Teste** : la portée causale cross-file sur une longue chaîne, et le recall borné
par budget.

Ingère ces 6 fichiers Rust (chaîne `api → service → repo → cache → db → config`) :

```rust
// src/config.rs
pub fn limit() -> i64 { 100 }

// src/db.rs
use crate::config;
pub fn timeout() -> i64 { config::limit() / 3 }

// src/cache.rs
use crate::db;
pub fn ttl() -> i64 { db::timeout() * 2 }

// src/repo.rs
use crate::cache;
pub fn fetch() -> i64 { cache::ttl() + 1 }

// src/service.rs
use crate::repo;
pub fn run() -> i64 { repo::fetch() }

// src/api.rs
use crate::service;
pub fn handle() -> i64 { service::run() }
```

🟢 Pour chaque fichier : `ingest {uri:"src/<nom>.rs", source:"<contenu>\n"}`.
🟢 `recall {strategy:"around", anchor:"src/api.rs", budget:4000}`
🟢 `recall {strategy:"around", anchor:"src/api.rs", budget:50}`

✅ **Succès** : à `budget:4000`, la fenêtre atteint **`file:src/config.rs` (5 sauts)** ;
à `budget:50`, `config.rs` / `db.rs` **disparaissent** (évincés). Si la chaîne complète
remonte à 4000, le linking cross-file profond fonctionne.
(Repère : `handle()` vaut 67 — `100/3=33 → *2=66 → +1=67`.)

---

## Tâche 2 — Symptôme ≠ cause (LE drift à diagnostiquer)

🎯 **Teste** : le watchpoint d'éviction `missing` et la vue `energy` — le cœur de la
valeur de CCOS.

Sur le **même workspace** (chaîne de la tâche 1 déjà ingérée) :

🟢 `page_fault` avec un panic qui pointe le **symptôme** (haut de chaîne), pas la cause :
```
thread 'main' panicked at src/api.rs:2:14:
attempt to add with overflow
```
soit : `page_fault {output:"thread 'main' panicked at src/api.rs:2:14:\nattempt to add with overflow\n", budget:50}`.
🟢 `recall {strategy:"working_set", budget:50}` → note qui est en tête.
🟢 `timeline` → relève les **numéros d'étape exacts** (avant/après le page_fault).

🔵 `ccos postmortem ws2/workspace.ccos` (ou en pipe) :
```bash
printf '%s\n' 'timeline' 'energy <avant> <après>' 'missing src/config.rs 50' 'quit' \
  | ccos postmortem ws2/workspace.ccos
```
- `energy <avant_pagefault> <après>` → la chaleur reste-t-elle sur **api.rs** ou
  atteint-elle **config.rs** ?
- `missing src/config.rs 50` → à quelle étape la **vraie cause** sort de la fenêtre.

✅ **Succès** : `missing` montre `config.rs` **évincée** sous budget serré pendant que la
pression reste sur le symptôme. C'est la signature *« l'attention a chassé le
symptôme »* — exactement ce que l'outil doit rendre visible.

🔁 **Contre-épreuve honnête** : refais un `page_fault` avec `src/config.rs:1` dans la
trace → la pression migre vers la cause. Compare les deux `energy`. (Si un jour le
symptôme est *en amont* de la cause, c'est `ccos failure … --bidirectional` qu'il faut.)

---

## Tâche 3 — Boucle page_fault sur du **vrai** `cargo test`

🎯 **Teste** : le parseur de trace sur de la sortie réelle (pas synthétique) + la boucle
compilateur-en-boucle.

Crée un vrai crate avec un bug :
```rust
// src/lib.rs
pub fn add(a: i64, b: i64) -> i64 { a - b }   // bug : devrait être a + b

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adds() { assert_eq!(add(2, 2), 4); }
}
```
🟢 Lance `cargo test`, **capture la sortie brute** (elle contient
`panicked at src/lib.rs:8:…`).
🟢 `page_fault {output:"<sortie cargo test brute>", budget:1024}`.
🟢 `recall {strategy:"around", anchor:"src/lib.rs", budget:1024}`.

✅ **Succès** : CCOS **parse le vrai chemin** `src/lib.rs:<ligne>` de la sortie,
pressurise le bon fichier, et `recall around` remonte sa région. Corrige (`a + b`),
re-`cargo test` (vert), re-`ingest src/lib.rs` → la `timeline` trace toute la boucle de
correction.

---

## Tâche 4 — Session longue + compaction + redémarrage

🎯 **Teste** : la compaction de l'op-log, la stabilité des index absolus, le replay
cross-restart.

🟢 Lance le serveur avec `CCOS_OPLOG_MAX=20 CCOS_OPLOG_KEEP=5`.
🟢 Enchaîne **50+ opérations** (ingère 30+ fichiers réels d'un repo, intercale des
`recall`).
🟢 `timeline` → tu dois voir le marqueur **« N earlier operation(s) compacted into the
baseline »**.
🟢 **Tue le serveur**, puis rouvre le même workspace.
🟢 `recall_what_if {step:<dans la queue retenue>}` → marche.
🟢 `recall_what_if {step:<sous le plancher de compaction>}` → retombe sur la baseline
(comportement documenté).
🟢 `verify` → `{"valid": true}`.

✅ **Succès** : la timeline survit au crash, l'op-log reste **borné** par la compaction,
et `t=50` reste `t=50` (index absolus stables). C'est la résilience longue-durée.

---

## Tâche 5 — Le hook automatique (dogfood) sur une vraie tâche

🎯 **Teste** : l'intercept transparent — CCOS nourri sans que l'agent y pense.

🟢 Branche `scripts/ccos_self_feed.py` en hook **PostToolUse** (cf.
`docs/SELF_ANALYSIS.md`), avec `CCOS_WORKSPACE=ws5/workspace.ccos`.
🟢 Fais bosser Thor sur une **vraie mini-tâche** : lire 5+ fichiers, éditer 2, lancer un
`cargo test` qui échoue une fois puis passe — **sans appeler aucun outil CCOS
explicitement**.

🔵 `ccos postmortem ws5/workspace.ccos --json` → le record doit contenir des `ingest`
(issus des lectures) **et** un `page_fault` (issu du test rouge), **zéro appel manuel**.

✅ **Succès** : le `.oplog` s'est rempli tout seul. C'est la preuve que le « hardware
intercept » fonctionne en conditions réelles.

---

## 🏁 Capstone — le post-mortem terrain conjoint

De la tâche la plus riche (2 ou 5), exporte le dossier terrain :
```bash
ccos postmortem ws/workspace.ccos --json > session_terrain.json
```
Ramène ce `session_terrain.json` (ou le `.oplog` brut). On le déroule ensemble avec
`energy` / `missing` / `diff` — et là on saura si, sur du vrai code, CCOS rend visible
une **vraie dérive d'attention**. C'est le test qui compte.

---

## Grille de lecture rapide

| Tâche | Nouveauté validée | Critère de succès |
| ----- | ----------------- | ----------------- |
| 1 | Recall causal cross-file profond + budget | config.rs atteint à 4000, évincé à 50 |
| 2 | `missing` (eviction watchpoint) + `energy` | la cause évincée pendant que le symptôme chauffe |
| 3 | Parseur de trace sur `cargo test` réel | le vrai `src/…:ligne` pressurisé |
| 4 | Compaction + index stables + cross-restart | timeline survit au crash, op-log borné |
| 5 | Hook PostToolUse (dogfood transparent) | `.oplog` rempli sans appel manuel |
| 🏁 | Post-mortem terrain | une vraie dérive lisible dans `energy`/`missing` |
