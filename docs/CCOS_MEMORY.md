# CCOS Core — mémoire causale du dépôt

CCOS Core (Causal Context Operating System) est installé dans ce dépôt comme
**mémoire causale de l'agent** : il cartographie les effets de bord d'une session
de code (fichiers lus/édités, échecs `cargo test`/`build`, panics) dans un graphe
causal, page ce graphe sous un budget de tokens, et journalise chaque transition
dans un log **déterministe, rejouable bit-à-bit et hash-chaîné**.

- Source vendorisée : [`external/ccos-core/`](../external/ccos-core/) —
  copie du dépôt `Memorithm/CCOS-Core`, commit amont `9c1b7d99d014841adbf62b642f233e7fa6906b57`.
- Références amont : `external/ccos-core/README.md`,
  `external/ccos-core/docs/SELF_ANALYSIS.md` (dogfooding agent),
  `external/ccos-core/docs/USAGE.md`, `external/ccos-core/docs/MEMORY_INTERFACE.md`.

## Installation

```bash
sh scripts/ccos/install.sh
# = cargo build --release --features llm,license (dans external/ccos-core)
#   + install du binaire `ccos` (/usr/local/bin, sinon ~/.cargo/bin ou ~/.local/bin)
#   + `ccos doctor`
```

La toolchain est pinnée par `external/ccos-core/rust-toolchain.toml` (1.89.0).
Le build par défaut est **tier community** (aucune crypto Pro, fail-closed) —
c'est le comportement attendu tant qu'aucune clé vendeur n'est configurée.

`scripts/ccos/ensure_built.sh` est un garde idempotent pensé pour un hook
`SessionStart` : no-op si le binaire existe, sinon lance le build en arrière-plan
(journal dans `.ccos/build.log`) sans jamais bloquer.

## État runtime

| Chemin | Rôle | Git |
|---|---|---|
| `.ccos/workspace.ccos` | snapshot mémoire (graphe + log hash-chaîné) | ignoré |
| `.ccos/workspace.ccos.oplog` | timeline cognitive (time-travel) | ignoré |
| `external/ccos-core/target/` | artefacts de build | ignoré |

Amorçage typique d'un workspace neuf (ingestion du cœur du dépôt) :

```bash
python3 - <<'EOF' | ccos memory --path .ccos/workspace.ccos > /dev/null
import json, os
for root in ("src", "scirust-core/src"):
    for dirpath, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            if f.endswith(".rs"):
                p = os.path.join(dirpath, f)
                src = open(p, encoding="utf-8", errors="replace").read()
                print(json.dumps({"op": "ingest", "uri": p, "source": src}))
EOF
```

## Deux modes d'alimentation — un seul écrivain par workspace

> Règle amont (`docs/SELF_ANALYSIS.md`) : le serveur MCP persistant et le hook
> d'auto-alimentation ne doivent **jamais** écrire le même `workspace.ccos` en
> même temps. Choisir un mode.

### Mode A — serveur MCP (actif par défaut ici)

[`.mcp.json`](../.mcp.json) déclare le serveur (consent-gated : l'hôte demande
l'approbation à l'ouverture de session) :

```json
{ "mcpServers": { "ccos": { "command": "ccos", "args": ["mcp", ".ccos/workspace.ccos"] } } }
```

L'agent obtient les outils natifs `ingest`, `recall`, `signal_failure`,
`page_fault`, `stats`, `verify`, `timeline`, `recall_what_if`, plus la ressource
`ccos://session/context` (la fenêtre de travail auto-bornée, prête à injecter).

### Mode B — auto-alimentation transparente (hook PostToolUse)

`scripts/ccos/self_feed_hook.sh` intercepte chaque effet de bord de l'agent
(lecture/édition d'un fichier source → `ingest` ; échec cargo → `page_fault`)
et nourrit la mémoire **sans aucun coût cognitif pour l'agent**. Il ne bloque
jamais (exit 0 systématique, no-op si le binaire manque).

Pour l'activer, ajouter vous-même dans `.claude/settings.json` (un agent ne peut
pas câbler ses propres hooks — action réservée à l'humain) :

```json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [ { "type": "command",
          "command": "sh \"$CLAUDE_PROJECT_DIR\"/scripts/ccos/ensure_built.sh" } ] }
    ],
    "PostToolUse": [
      { "matcher": "Read|Edit|Write|NotebookEdit|Bash",
        "hooks": [ { "type": "command", "async": true,
          "command": "sh \"$CLAUDE_PROJECT_DIR\"/scripts/ccos/self_feed_hook.sh" } ] }
    ]
  }
}
```

**Si vous activez le Mode B, désactivez le Mode A** (supprimer/désapprouver le
serveur `ccos` de `.mcp.json`) pour respecter la règle « un écrivain ».
Les deux scripts sont pipe-testés : un événement `Read` synthétique produit bien
un `Ingest(...)` visible dans `ccos postmortem`.

## Lire la mémoire, déboguer une dérive

```bash
# recall causal autour d'un fichier (fenêtre auto-bornée)
printf '%s\n' '{"op":"recall","strategy":"around","anchor":"file:scirust-core/src/lib.rs","budget":2048}' \
  | ccos memory --path .ccos/workspace.ccos

# intégrité + stats
printf '%s\n' '{"op":"verify"}' '{"op":"stats"}' | ccos memory --path .ccos/workspace.ccos

# post-mortem time-travel (timeline, diff, energy, missing <node>)
ccos postmortem .ccos/workspace.ccos

# archive analytique d'une session
ccos postmortem .ccos/workspace.ccos --json > archive_$(date +%F).json
```

Protocole de dérive (amont, §SELF_ANALYSIS) : `timeline` → `missing <cause>` →
`energy A B` → `goto K` + `recall` — pour dater précisément l'éviction de la
vraie cause hors de la fenêtre budgétée.
