# CCOS Core — épreuve terrain sur scirust

Campagne d'épreuve de [CCOS Core](CCOS_MEMORY.md) (`external/ccos-core`, amont
`9c1b7d9`) comme mémoire causale de l'agent, exécutée sur ce dépôt. Objectif :
vérifier par la mesure ce que la mémoire tient et où elle casse — pas la
présenter sous son meilleur jour.

Environnement : x86_64-linux, build `--release --features llm,license`,
`ccos 0.4.0-pre`, tier community (aucune clé vendeur, Pro fail-closed).

## Verdict

Le cœur de la promesse tient et se mesure : la couverture causale sous budget,
le déterminisme bit-à-bit, la chaîne de hash et la détection d'injection se
comportent comme annoncé. **Un défaut réel a été trouvé et corrigé** : la façade
CLI `ccos memory` ne faisait pas le page-in du tier COLD, donc un `recall around`
sur une ancre démotée renvoyait une fenêtre **vide** là où le même appel via MCP
renvoyait 31 éléments.

| Épreuve | Résultat |
|---|---|
| 1. Continuité d'état (MCP) | ✅ workspace rechargé, chaîne valide, timeline cohérente |
| 2. Couverture causale sous budget | ✅ 88–100 % vs 0 % pour un dump naïf |
| 3. Page fault réel → cause inter-fichiers | ✅ la cause remonte, `causal_blame` la classe #2 |
| 4. Déterminisme / intégrité / time-travel | ✅ recall byte-identique, falsification détectée |
| 5. Robustesse adversariale | ✅ injection notée 1.0, dégénérés sans crash |
| 6. Suite de tests amont | ✅ 758 tests, 0 échec (avec le correctif) |
| — Défaut trouvé | ⚠️ page-in COLD absent de la façade CLI → **corrigé** |
| — Bug amont trouvé | ⚠️ le paging perd des arêtes causales → **corrigé en amont** (§8) |

## 1. Continuité d'état

Le serveur MCP (`.mcp.json`) recharge le workspace amorcé par le CLI : `stats`
renvoie 4 974 nœuds / 8 528 arêtes / 167 événements, `verify` → `valid: true`,
`timeline` restitue les opérations dans l'ordre. La mémoire survit donc au
changement de processus et de chemin d'accès (CLI ↔ MCP, format partagé).

## 2. Couverture causale sous budget (harnais amont, sur scirust)

Reproduction du protocole model-free de l'amont
(`external/ccos-core/scripts/ccos_context_value.py`) : pour chaque fichier avec
au moins une dépendance `use crate::`, quelle fraction de ses dépendances réelles
tient dans une fenêtre de 2 048 tokens ?

| Crate | Fichiers analysés | Arêtes de dép. | CCOS | Dump naïf |
|---|---|---|---|---|
| `scirust-core` | 10 | 21 | **100 %** | 0 % |
| `scirust-sim` | 19 | 24 | **100 %** | 0 % |
| `scirust-estimation` | 13 | 16 | **88 %** | 0 % |
| `scirust-fluids` | 8 | 8 | **100 %** | 0 % |

Sur les gros fichiers — ceux dont l'ouverture naïve tronque toute dépendance —
l'écart est de 85–100 % contre 0 %. Les chiffres tombent dans la fourchette
81–100 % annoncée par l'amont, mesurée ici sur un autre corpus que le leur.

**Le seul manque (`scirust-estimation/src/rls.rs`, 0/2)** est explicable et non
trivial : ses deux `use crate::` sont déclarés **à l'intérieur d'une fonction de
test** (`rls.rs:409-410`), pas au niveau module. Le harnais de mesure les compte
comme dépendances réelles ; le parseur AST les rattache au scope local. C'est une
divergence de convention entre l'oracle et le parseur, pas une perte causale.

## 3. Page fault réel et propagation

Une vraie trace `cargo test` (panic de `Conv2d::forward` sur un désaccord de
forme) injectée via l'outil MCP `page_fault` : la fenêtre reconstruite contient
le symptôme (`nn/nd_layers.rs`) **et** la cause inter-fichiers nommée dans la
backtrace (`tensor/tensor_nd.rs`, score 0.72), plus `error.rs`, `autodiff/nd.rs`
et `nn/rng.rs` — les dépendances directes du symptôme.

`causal_blame` sur le fichier symptôme classe `tensor_nd.rs` en tête des causes
candidates réelles (poids 1.97, juste derrière le pseudo-nœud `dep:crate`),
devant `error.rs` (1.56) et `autodiff/reverse.rs` (1.33) — l'ordre qu'un humain
donnerait en lisant la backtrace.

## 4. Déterminisme, intégrité, time-travel

- **Reconstruction déterministe.** Deux workspaces neufs construits depuis la
  même entrée (36 fichiers) produisent un `recall` **byte-identique** et des
  `stats` identiques.
- **Chaîne de hash.** Falsification d'un octet dans le payload d'un événement →
  `content tampered — hash mismatch`. Falsification d'un maillon de la chaîne →
  `broken link — prev_hash does not match the chain` **plus** la cascade sur les
  événements suivants. Les deux sont détectés, `valid: false`.
  *Nuance honnête* : une première tentative modifiant l'état **snapshot**
  reconstruit (et non le log) passait `verify` — c'est cohérent (le log est la
  source de vérité et il est revérifié), mais cela signifie que `verify` atteste
  le journal, pas la copie de travail.
- **Time-travel.** `recall_what_if(step=3)` rejoue la fenêtre telle qu'elle était
  avant le page fault : elle contient les symboles de `nd_layers` mais **pas**
  `tensor_nd.rs` — on voit littéralement la cause entrer en contexte à l'étape
  suivante. Le watchpoint `missing` du post-mortem date l'éviction (`○○○○●`) et
  chiffre le manque en tokens.

## 5. Robustesse adversariale

| Entrée | Comportement |
|---|---|
| Injection de prompt franche (exfiltration de clés) | `injection_score: 1.0`, `flagged: true` |
| Fichier sain témoin | `score: 3.6e-16`, non flaggé (pas de faux positif) |
| Obfuscation zero-width + base64 | anomalies `ZWSP` localisées par offset/codepoint |
| Octets de contrôle (NUL, 0x01…) | anomalies `Control` signalées, pas de crash |
| Fichier de 914 Ko | ingéré en 28 s, 4 994 nœuds, pas de crash |
| Fichier vide / 50 000 lignes identiques | absorbés sans erreur |

## 6. Le défaut trouvé — page-in COLD absent de la façade CLI

**Symptôme.** Sur un workspace de 300 fichiers (3 635 nœuds démotés en COLD), le
même `recall around` sur la même ancre et le même budget :

```
via MCP  (AgentSession) : items = 31, tokens = 2048
via CLI  (ccos memory)  : items =  0, tokens =    0
```

**Cause.** `MemoryProvider::recall` prend `&self` : il ne peut pas paginer. C'est
`CcosMemory::ensure_resident` (`&mut self`) qui rend le tier COLD transparent, et
sa propre documentation dit qu'« the session layer calls this before an Around
recall ». `AgentSession::recall` le fait bien (`agent_session.rs:1277`) — donc le
MCP est correct — mais la façade CLI appelait `mem.recall()` directement
(`main.rs:2862`), sans page-in. Le tier COLD n'était transparent que sur un des
deux chemins.

**Ce que ce n'était pas.** Premier diagnostic (erroné) : « le plafond de 5 000
nœuds est trop bas ». Instrumenté, le plafond fait exactement son travail — avec
un cap de 1, `page_in` ramène 3 nœuds et le re-paging en redémote 2, ce qui est
le comportement correct d'un cache borné. Le défaut était l'absence d'appel, pas
la valeur du plafond.

**Second point : un correctif tenté, puis retiré.** `CcosMemory::new()` code
`MemoryGraph::new(0.2, 5000)` en dur alors que `CCOS_MAX_RESIDENT` est documenté
comme le réglage du plafond et que `commands_runtime.rs` l'honore déjà via
`new_from_env`. La façade et le MCP ignorent donc la variable. La correction
« évidente » — faire lire l'environnement à `new()` — a été implémentée, testée,
**puis annulée** : elle n'apporte rien et casse la certification. Voir §7.

**Correctif retenu** (`external/ccos-core/src/main.rs`, un test de
non-régression) : la façade appelle `ensure_resident` avant un recall `Around`,
comme la session. Le correctif couvre ses **deux** appelants, `ccos memory` et
`ccos stdin`, qui partagent `run_op_stream`.

**Vérification.** Même workspace, même ancre, même budget :

```
CLI avant : items =  0, tokens =    0
CLI après : items = 31, tokens = 2048   ← parité avec MCP
```

Suite amont complète : **626 tests, 0 échec** (625 amont + le test ajouté).

## 7. Ce qu'une revue adversariale du correctif a donné

Le correctif ci-dessus a été soumis à une revue en fan-out (inventaire des
points d'appel, risque replay/persistance, portée du réglage, conventions
amont), puis chaque constat sérieux a été soumis à un réfutateur indépendant.
Deux constats étaient annoncés comme **bloquants**. Les deux se sont révélés
faux **à configuration égale**, et le vérifier a mis au jour un vrai bug amont.

**« Le page-in détruit la timeline » — réfuté.** Mesuré : la timeline passe de
8 à 0 opérations **aussi avec le binaire vierge et sans aucun recall**. La cause
est le conflit « deux écrivains » déjà documenté (`docs/CCOS_MEMORY.md`) : un
flux CLI mutant sur un workspace tenu par MCP fait diverger le snapshot de
l'op-log, et le garde de cohérence d'`AgentSession::open` repart du snapshot.
Le correctif n'y est pour rien.

**« Le page-in détruit des arêtes causales » — réfuté, mais révélateur.**
À plafond identique (20, persisté dans le snapshot), sur le même workspace :

| Binaire | Flux | items | arêtes |
|---|---|---|---|
| vierge | ingest seul | — | 16 → **11** |
| vierge | ingest + recall | 0 | 16 → **11** |
| corrigé | ingest seul | — | 16 → **11** |
| corrigé | ingest + recall | 3 | 16 → **12** |

La perte de 5 arêtes est **identique sans le correctif et sans recall** : elle
vient de la démotion elle-même, pas du page-in. Le correctif en préserve même
une de plus (12 vs 11), puisque le page-in ramène une arête.

**Le vrai défaut, lui, est amont et préexistant : le paging perd des arêtes.**
Démoter puis repaginer ne restaure une arête archivée que si ses deux extrémités
sont résidentes (`memory.rs`, `page_in`) ; une arête dont l'autre extrémité est
encore COLD est supprimée du seul endroit où elle existait. Cela contredit le
contrat annoncé du tier COLD (« non-destructif : le nœud et ses liens sont
conservés »). C'est indépendant de ce correctif, cela affecte aussi le chemin
MCP, et cela mérite un rapport amont séparé.

**Constat retenu, et il a tué la moitié du correctif.** Le second volet du
patch — faire lire `CCOS_MAX_RESIDENT` à `CcosMemory::new()` via `new_from_env`
— a été **annulé** après vérification. Trois raisons, toutes mesurées :

1. **Inerte là où ça compte.** `open()` n'appelle `new()` que si le fichier est
   absent, et le plafond est un champ sérialisé : un workspace existant garde le
   sien. Mesuré : 60 résidents / 0 COLD en rouvrant sous `CCOS_MAX_RESIDENT=5`,
   contre 5 / 55 pour un workspace neuf. Mes vérifications initiales
   (« cap 500 → 500 résidents ») portaient à chaque fois sur un workspace
   **neuf** — un faux positif que je n'avais pas vu.
2. **Casse la certification.** Avec `CCOS_MAX_RESIDENT=3` dans l'environnement,
   `ccos setup` tombe de `6/6 checks passed` à **`4/6 — NOT certified`**
   (`causal recall` et `failure propagation` échouent), alors que le binaire
   vierge reste à 6/6. Cela contredit frontalement le contrat
   « deterministic by construction » de `setup.rs`.
3. **Rend le replay sensible à l'environnement ambiant**, et le test associé
   était un faux-ami : son mutex ne protégeait que lui, alors qu'une centaine de
   constructeurs du même binaire de test lisent désormais la variable.

Le défaut d'origine — la façade ignore `CCOS_MAX_RESIDENT` — est donc **réel et
non corrigé**. Il est documenté à l'endroit du code, avec la raison pour laquelle
la correction évidente est pire que le mal.

## 8. Les défauts restants, corrigés en amont

Les constats laissés ouverts au §7 ont été traités dans
[`Memorithm/CCOS-Core#2`](https://github.com/Memorithm/CCOS-Core/pull/2).

**Le tier COLD n'était pas non-destructif.** `demote` archive une arête d'un
**seul** côté ; `page_in` ne la reliait que si ses deux extrémités étaient
résidentes et la jetait sinon — alors que le `ColdNode` qui la portait venait
d'être retiré, donc c'était la dernière copie. Mesuré sur un graphe de 4 nœuds
(tout démoter, tout repaginer) : **4 arêtes avant, 3 après**. Une arête dont
l'autre bout est encore COLD est désormais confiée à ce voisin, et l'entrée
d'adjacence inverse conservée.

**`ensure_resident` ne garantissait pas sa propre postcondition.** Le swap de
capacité de `page_in` ne protège que le nœud qu'il vient de restaurer : sous un
plafond trop serré, paginer un voisin pouvait réévincer l'ancre demandée.
L'ancre est maintenant restaurée en dernier. Ce n'était pas théorique — un test
amont existant s'est mis à échouer dès que le page fault a commencé à paginer la
région.

**Le débogueur post-mortem voyait tout sauf les nœuds démotés.**
`recall_what_if` : **0 élément** là où le recall vivant en rendait **31**, même
ancre et même budget. L'asymétrie était à l'envers — rejouer une op `Around`
*enregistrée* repasse par `ensure_resident`, donc le what-if fonctionnait pour
une ancre déjà consultée et échouait pour celle qui ne l'avait pas été,
c'est-à-dire exactement la question à laquelle la fonctionnalité sert à répondre.

**`page_fault` et le self-test `setup.rs`** reçoivent le même page-in, par
cohérence. Honnêtement : sur un gros workspace réel, cela ne change aucun
résultat, la propagation de panne ayant déjà ramené les nœuds utiles. Mon
observation « 26 vs 32 » mesurait en fait un écart de **scoring**, pas de
pagination — vérifié après correctif, le chiffre est inchangé. Seul le test
unitaire, qui échoue sans l'appel, prouve le défaut.

Leçon de méthode : ma propre vérification empirique initiale — « le partage
résident/COLD persisté est identique » — comparait les compteurs de nœuds et
**pas les arêtes**, donc elle était trop grossière pour voir quoi que ce soit.
C'est la comparaison à configuration égale, pas l'analyse statique seule, qui a
tranché dans les deux sens.

## Limites connues, à l'échelle du monorepo

- **Coût d'ingestion.** Les 2 055 fichiers `.rs` du dépôt (28 Mo) s'ingèrent en
  **5 min 36 s** (~6 Mo/min), snapshot de 70 Mo. Amorcer `scirust-core` seul
  prend ~3 s. Pour un usage quotidien, amorcer les crates sur lesquelles on
  travaille, pas tout le monorepo.
- **Le plafond reste à 5 000 nœuds par défaut.** À l'échelle du monorepo, la
  quasi-totalité du graphe part en COLD (52 301 nœuds démotés). Le page-in le
  rend fonctionnellement transparent, mais si l'on veut garder tout le monorepo
  résident il faut lever `CCOS_MAX_RESIDENT` (désormais effectif) — au prix de
  la RAM.
- **`verify` atteste le log**, pas le snapshot de travail (cf. §4).

## Reproduire

```bash
sh scripts/ccos/install.sh

# §2 — couverture causale
python3 external/ccos-core/scripts/ccos_context_value.py scirust-core/src \
  --budget 2048 --ccos "$(command -v ccos)"

# §4 — déterminisme : deux workspaces neufs, même entrée, comparer les sorties
# §6 — parité CLI/MCP sur une ancre démotée
CCOS_MAX_RESIDENT=500 ccos memory --path /tmp/ws.ccos < ops.jsonl
```
