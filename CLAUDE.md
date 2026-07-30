# CLAUDE.md

## Mémoire causale — CCOS Core

Ce dépôt embarque **CCOS Core** (Causal Context Operating System, vendorisé sous
`external/ccos-core/`, commit amont `9c1b7d9`) comme mémoire causale de l'agent :
graphe causal event-sourcé, log hash-chaîné, recall auto-borné, replay
déterministe et débogueur post-mortem. Doc complète : `docs/CCOS_MEMORY.md`.

- **Binaire** : `ccos` (si absent : `sh scripts/ccos/install.sh` — build release
  + install + `ccos doctor`).
- **Workspace mémoire** : `.ccos/workspace.ccos` (+ `.oplog`), non versionné.
- **Serveur MCP** : déclaré dans `.mcp.json` (`ccos mcp .ccos/workspace.ccos`) —
  outils `ingest`, `recall`, `signal_failure`, `page_fault`, `timeline`,
  `recall_what_if`; ressource `ccos://session/context`.
- **Sans MCP**, utiliser le CLI (une op JSON par ligne) :

  ```bash
  printf '%s\n' '{"op":"recall","strategy":"around","anchor":"file:scirust-core/src/lib.rs","budget":2048}' \
    | ccos memory --path .ccos/workspace.ccos
  ```

  Ops : `ingest`, `failure`, `recall` (`around`/`task`/`working_set`), `impact`,
  `causes`, `verify`, `stats`. Post-mortem : `ccos postmortem .ccos/workspace.ccos`.
- **Règle d'or** : un seul écrivain par workspace — soit le serveur MCP, soit le
  hook d'auto-alimentation (`scripts/ccos/self_feed_hook.sh`), jamais les deux.
