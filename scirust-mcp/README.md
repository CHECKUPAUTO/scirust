# scirust-mcp

Serveur [Model Context Protocol](https://modelcontextprotocol.io) (MCP) pour
SciRust : expose les capacités de la plateforme — solveurs numériques, outils
de développement du SLM `scirust-sciagent`, et à terme la découverte d'actifs
OT/IT de `scirust-discovery` — comme des **outils MCP standard**, appelables
par n'importe quel agent : le SLM embarqué de SciRust, Claude, ChatGPT, ou un
simple script.

## Pourquoi MCP plutôt qu'un format d'appel d'outil maison

Le SLM `scirust-sciagent` avait déjà un mini-format d'appel d'outil interne
(`scirust_sciagent::agentic::{Tool, AgentRouter}`, un JSON `{"name": ...,
"params": ...}` ad hoc). Cela marche pour un seul agent développé en interne,
mais ne se généralise pas : chaque nouvel agent externe (Claude Desktop,
ChatGPT, un script d'automatisation industrielle) devrait réimplémenter son
propre parsing pour parler à SciRust.

MCP (publié par Anthropic en novembre 2024, spécification stable depuis juin
2025) est devenu le standard de facto pour ce problème : JSON-RPC 2.0,
primitives **tools** (fonctions appelables, schéma JSON d'entrée),
**resources** (données en lecture seule) et **prompts** ; découverte
dynamique (`tools/list`) plutôt que glue code codée en dur par intégration ;
transport `stdio` (sous-processus local — ce que ce crate implémente) ou
`Streamable HTTP` (distant, avec OAuth 2.1). C'est ce que Claude Desktop,
les IDE (VS Code, JetBrains), et un nombre croissant d'agents savent déjà
parler nativement. Choisir MCP signifie que connecter SciRust à *n'importe
quel* agent devient une question de configuration, pas de code.

`scirust-mcp` **réutilise** l'implémentation existante des outils de
développement du SLM (`scirust_sciagent::agentic::tools::Tool::builtins()`)
plutôt que de la dupliquer — voir `src/tools/dev.rs`. MCP est ici une couche
de *transport* supplémentaire au-dessus de capacités qui existaient déjà, pas
une réécriture.

## Profils de capacités

Le serveur démarre en profil **production**. Ce profil n'enregistre ni les
outils `dev_*` (lecture/recherche de fichiers, build, tests, git) ni le
passe-plat `scirust_cli`; une injection de prompt ne peut donc pas les utiliser
pour lire les secrets du processus ou démarrer un sous-processus.

Le profil `development` doit être activé explicitement avec
`SCIRUST_MCP_PROFILE=development`. Il est réservé à un checkout local de
confiance. Ses chemins sont canonicalisés et confinés à `SCIAGENT_ROOT` (la
racine du workspace par défaut), y compris après résolution des liens
symboliques. Les lectures, plages de lignes, sorties, arguments et temps
d'exécution sont bornés.

## Outils disponibles

| Outil | Domaine | Description |
|---|---|---|
| `dev_search`, `dev_grep`, `dev_read`, `dev_explain`, `dev_build`, `dev_test`, `dev_status` | Développement opt-in | Disponibles uniquement dans le profil `development` |
| `linalg_eigen_symmetric` | Algèbre linéaire | Décomposition en valeurs propres symétrique (Householder + QL implicite, voir `scirust-solvers`) |
| `linalg_svd` | Algèbre linéaire | SVD générale (Jacobi à un côté) |
| `linalg_gmres` | Algèbre linéaire | GMRES(m) pour systèmes non symétriques |
| `discovery_scan` | Découverte OT/IT | Sonde des cibles réseau (OPC-UA, Modbus, mDNS) via `scirust-discovery`, sous portée signée — voir `scirust-discovery/README.md` |
| `sis_verify_sif_loop` | Sûreté procédés (IEC 61511) | PFDavg total + SIL atteint d'une boucle SIF multi-sous-systèmes via `scirust-sis` |
| `sis_size_proof_test_interval` | Sûreté procédés (IEC 61511) | Intervalle de test de preuve maximal pour un PFDavg cible, par inversion numérique |
| `sim_epidemic` | Simulation (`scirust-sim`) | Épidémie SIR : R0, pic infecté et jour du pic, taux d'attaque final |
| `sim_battery_discharge` | Simulation (`scirust-sim`) | Cellule Thévenin 1-RC + thermique (plante `scirust-bms`) à courant constant : SoC, tension, température finales |
| `sim_grid_stability` | Simulation (`scirust-sim`) | Équation d'oscillation machine-réseau (plante `scirust-grid`) : synchronisme, équilibre, fréquence petit signal, transitoire |
| `scirust_cli` | Passe-plat opt-in | Disponible uniquement dans le profil `development` |

`discovery_scan` ne peut jamais s'auto-autoriser depuis la conversation : la
clé qui vérifie la signature de la portée vit côté serveur
(`SCIRUST_DISCOVERY_KEY`), jamais dans les arguments de l'appel d'outil.
Sans cette variable définie par l'opérateur, l'outil refuse tout — voir
`scirust-discovery/README.md`.

Un nouveau domaine (ex. `scirust-discovery`, un futur `scirust-pdm` exposé)
s'ajoute en implémentant `fn xxx_tools() -> Vec<McpTool>` dans
`src/tools/` et en l'enregistrant dans
[`default_registry`](src/lib.rs) — aucune autre modification requise pour
que tous les clients MCP existants le voient.

## Auditabilité

Chaque `tools/call` — succès ou échec — est ajouté à un journal hash-chaîné
SHA-256 (`src/audit.rs`, `AuditLog`), sur le même principe que
`scirust-func-safety::audit` (chaque entrée contient le hash de la
précédente, ce qui rend toute falsification après coup détectable), mais
avec un vrai SHA-256 (réutilisation de `scirust_sciagent::sha256`, du
domaine public FIPS 180-4) plutôt qu'un hash maison — pour une trace
d'intégrité d'audit, la résistance aux collisions n'est pas
négociable. Le journal stocke le **hash** des arguments et du résultat, pas
leur contenu en clair : il peut être exporté sans exposer de données
potentiellement sensibles issues d'une infrastructure cliente. Pour préserver
le format public historique, `AuditLog::export_json` produit toujours le tableau
d'entrées seul. `AuditLog::export_snapshot_json` produit une enveloppe versionnée
avec l'ancre précédant la fenêtre conservée, la tête, le prochain numéro de
séquence et les entrées ; `AuditExport::from_json` en vérifie la cohérence après
rotation.

Cette chaîne SHA-256 n'est toutefois ni une signature ni un MAC : elle ne prouve
pas l'identité du producteur, et un acteur pouvant remplacer l'export entier peut
recalculer tous ses hashes. Pour obtenir une preuve d'altération, conservez la
`head` par un canal de confiance indépendant puis utilisez
`AuditExport::validate_against_head`. Une `anchor` de confiance établit la
continuité avec le passé, mais n'authentifie pas seule les entrées suivantes.

## Utilisation

```bash
cargo run -p scirust-mcp --bin scirust-mcp
```

Le serveur lit des requêtes JSON-RPC 2.0 sur stdin (une par ligne) et écrit
les réponses sur stdout (une par ligne) — c'est le transport `stdio` du MCP,
compatible avec Claude Desktop et tout autre client MCP standard. Exemple de
configuration Claude Desktop (`claude_desktop_config.json`) :

```json
{
  "mcpServers": {
    "scirust": {
      "command": "cargo",
      "args": ["run", "--release", "-p", "scirust-mcp", "--bin", "scirust-mcp"],
      "cwd": "/chemin/vers/scirust"
    }
  }
}
```

`SCIRUST_BIN` (variable d'environnement) pointe l'outil `scirust_cli` vers un
binaire `scirust` déjà compilé (`cargo install --path scirust-cli`) plutôt
que de le reconstruire à chaque appel.

Pour une session locale de développement uniquement :

```bash
SCIRUST_MCP_PROFILE=development SCIAGENT_ROOT="$PWD" \
  cargo run -p scirust-mcp --bin scirust-mcp
```

Toute autre valeur de `SCIRUST_MCP_PROFILE` est refusée au démarrage.

## Sources

- Model Context Protocol — spécification : <https://modelcontextprotocol.io>
- Anthropic, « Introducing the Model Context Protocol », nov. 2024.
- Comparaison avec Google Agent2Agent (A2A, avril 2025) : MCP est « un agent
  utilise un outil », A2A est « un agent délègue à un autre agent » — les
  deux sont complémentaires, pas concurrents.
