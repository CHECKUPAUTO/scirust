# `ccos setup` — one-command install, agent wiring & self-test

One command turns a checkout into a wired, self-tested deployment:

```sh
sh scripts/install.sh
```

That script builds the release binary, installs it, runs `ccos doctor`, then
runs **`ccos setup --yes`** — the subject of this page. `setup` can also be run
on its own, any time, on any host the binary is already installed on:

```sh
ccos setup            # interactive: shows the plan, asks before writing
ccos setup --yes      # non-interactive (what install.sh runs)
ccos setup --dry-run  # show the plan + run the battery, write nothing
```

## What a setup pass does

| phase | what happens | writes |
|---|---|---|
| **1 · probe** | host target & build profile, license tier, SHA-256 of the running binary, `.mcp.json` wiring state, Mode B hook state, optional local-LLM reachability (egress-checked), existing workspace | nothing |
| **2 · wire** | register the MCP server (`ccos mcp workspace.ccos`) in the **project's** `.mcp.json` — an idempotent *merge* that preserves every other server and key | `.mcp.json`, with consent only |
| **3 · self-test** | the deterministic first-run battery (below) against a throwaway in-process kernel | temp files only |
| **4 · report** | seal every probe result, wiring action and check verdict into a content-hashed report | `setup_report.json` |

Re-running `setup` on an already-wired host is a no-op that re-verifies:
phase 2 reports `already wired`, the battery runs again, the report is
refreshed. Exit code is `0` only when **every** check passed.

## The first-run battery

Six checks against the real kernel — the same code paths an agent will use,
not mocks:

1. **ingest** — a two-file witness with a cross-file dependency parses into
   nodes and edges.
2. **causal recall** — the dependency is paged into a 2048-token window
   anchored on the dependent file (the core coverage promise, measured here
   rather than assumed).
3. **failure propagation** — failure pressure reaches the causes of a failing
   file.
4. **hash-chain integrity** — the tamper-evident chain over everything the
   battery just did verifies.
5. **checkpoint determinism** — checkpoint → reload reproduces the state
   fingerprint bit-for-bit on this host's filesystem.
6. **mcp handshake** — an MCP host completes `initialize`, sees the tool
   catalogue and the `ccos://setup/report` resource.

The battery is deterministic: the same build gives the same verdict.

## The verdict is code, the agent is the messenger

`setup` deliberately does **not** rely on an LLM to announce success. A model
can be instructed to report a result; it cannot be *obliged* to report it
truthfully — so the certification an operator acts on must come from
deterministic code. The contract:

- **Source of truth:** `setup_report.json` — schema `ccos.setup.report/v1`,
  per-check pass/fail, every wiring action (including every *skipped* one),
  the host probe, the binary's SHA-256, and a `report_sha256` content hash
  sealing the whole document.
- **The messenger:** any MCP agent connected to the server reads the resource
  **`ccos://setup/report`** and relays the verdict to the user. If the agent
  misstates it, the report file still says what actually happened. A missing
  report reads as an announced pointer ("run `ccos setup`"), never a protocol
  error. Path override: `$CCOS_SETUP_REPORT`.

## Consent & fail-closed rules

- Nothing outside the project directory is ever written. Wiring touches only
  the project `.mcp.json`; the report lands in the project directory.
- Writing `.mcp.json` requires consent: `--yes`, or an interactive `y` at the
  prompt. Non-interactive without `--yes` ⇒ the write is **skipped and
  recorded** as `skipped_no_consent` — an announced refusal, never silence.
- An unparseable `.mcp.json` is never rewritten (`failed`, with the parse
  error); fix or remove it by hand first.
- The optional local-LLM probe goes through the same egress allowlist as every
  network call site (`docs/SECURITY.md`): a non-loopback `OLLAMA_ENDPOINT` is
  refused unless allowlisted, and the refusal appears in the report.
- Agent-host settings (`.claude/settings.json`) are **never** edited. The
  Mode B feed hook executes commands on the agent host's behalf, so wiring it
  belongs to the operator: `ccos setup --hook` prints the snippet to paste.

## One writer per workspace (Mode A vs Mode B)

`setup` wires **Mode A** (the agent calls CCOS tools over MCP). The
alternative, **Mode B** (the transparent PostToolUse feed hook), is documented
in [`SELF_ANALYSIS.md`](SELF_ANALYSIS.md). Pick **one writer** per
`workspace.ccos` — if the hook already feeds a workspace, don't let agents
call mutating MCP tools on the same file. `setup` detects a wired hook and
prints this warning instead of guessing your intent.

## Flags

```
ccos setup [--yes] [--dry-run] [--json] [--hook]
           [--dir D] [--report F] [--workspace W]
```

| flag | effect |
|---|---|
| `--yes` / `-y` | apply the wiring without prompting |
| `--dry-run` | print the plan, run the battery, write nothing |
| `--json` | print the sealed report JSON instead of the human summary |
| `--hook` | also print the Mode B hook snippet (manual wiring) |
| `--dir D` | project directory (default `.`) |
| `--report F` | report path (default `<dir>/setup_report.json`) |
| `--workspace W` | workspace the registered server persists (default `workspace.ccos`) |

Environment: `CCOS_SETUP_REPORT` (where the MCP resource reads the report),
`OLLAMA_ENDPOINT` (probe target), `CCOS_EGRESS_ALLOW` (egress allowlist),
`CCOS_SETUP=0` (skip the setup pass in `install.sh`).
