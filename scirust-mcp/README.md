# scirust-mcp

[Model Context Protocol](https://modelcontextprotocol.io) (MCP) server for
SciRust: exposes the platform's capabilities — numerical solvers, development
tools for the `scirust-sciagent` SLM, and eventually OT/IT asset discovery from
`scirust-discovery` — as **standard MCP tools**, callable by any agent:
SciRust's embedded SLM, Claude, ChatGPT, or a simple script.

## Why MCP rather than a homegrown tool-call format

The `scirust-sciagent` SLM already had a small internal tool-call format
(`scirust_sciagent::agentic::{Tool, AgentRouter}`, an ad hoc JSON `{"name": ...,
"params": ...}`). That works for a single agent developed in-house, but does not
generalize: every new external agent (Claude Desktop, ChatGPT, an industrial
automation script) would have to reimplement its own parsing to talk to SciRust.

MCP (published by Anthropic in November 2024, stable specification since June
2025) has become the de facto standard for this problem: JSON-RPC 2.0,
**tools** primitives (callable functions, JSON input schema), **resources**
(read-only data) and **prompts**; dynamic discovery (`tools/list`) rather than
hard-coded glue code per integration; `stdio` transport (local subprocess — what
this crate implements) or `Streamable HTTP` (remote, with OAuth 2.1). This is
what Claude Desktop, IDEs (VS Code, JetBrains), and a growing number of agents
already speak natively. Choosing MCP means that connecting SciRust to *any*
agent becomes a matter of configuration, not code.

`scirust-mcp` **reuses** the existing implementation of the SLM's development
tools (`scirust_sciagent::agentic::tools::Tool::builtins()`) rather than
duplicating it — see `src/tools/dev.rs`. MCP is here an additional *transport*
layer on top of capabilities that already existed, not a rewrite.

## Capability profiles

The server starts in the **production** profile. This profile registers neither
the `dev_*` tools (file read/search, build, tests, git) nor the `scirust_cli`
pass-through; a prompt injection therefore cannot use them to read the
process's secrets or spawn a subprocess.

The `development` profile must be enabled explicitly with
`SCIRUST_MCP_PROFILE=development`. It is reserved for a trusted local checkout.
Its paths are canonicalized and confined to `SCIAGENT_ROOT` (the workspace root
by default), including after symlink resolution. Reads, line ranges, outputs,
arguments and execution times are bounded.

## Available tools

| Tool | Domain | Description |
|---|---|---|
| `dev_search`, `dev_grep`, `dev_read`, `dev_explain`, `dev_build`, `dev_test`, `dev_status` | Opt-in development | Available only in the `development` profile |
| `linalg_eigen_symmetric` | Linear algebra | Symmetric eigendecomposition (Householder + implicit QL, see `scirust-solvers`) |
| `linalg_svd` | Linear algebra | General SVD (one-sided Jacobi) |
| `linalg_gmres` | Linear algebra | GMRES(m) for non-symmetric systems |
| `discovery_scan` | OT/IT discovery | Probes network targets (OPC-UA, Modbus, mDNS) via `scirust-discovery`, under a signed scope — see `scirust-discovery/README.md` |
| `sis_verify_sif_loop` | Process safety (IEC 61511) | Total PFDavg + achieved SIL of a multi-subsystem SIF loop via `scirust-sis` |
| `sis_size_proof_test_interval` | Process safety (IEC 61511) | Maximum proof test interval for a target PFDavg, by numerical inversion |
| `sim_epidemic` | Simulation (`scirust-sim`) | SIR epidemic: R0, infected peak and day of peak, final attack rate |
| `sim_battery_discharge` | Simulation (`scirust-sim`) | Thévenin 1-RC cell + thermal (plant `scirust-bms`) at constant current: final SoC, voltage, temperature |
| `sim_grid_stability` | Simulation (`scirust-sim`) | Machine-grid swing equation (plant `scirust-grid`): synchronism, equilibrium, small-signal frequency, transient |
| `scirust_cli` | Opt-in pass-through | Available only in the `development` profile |

`discovery_scan` can never self-authorize from the conversation: the key that
verifies the scope signature lives server-side (`SCIRUST_DISCOVERY_KEY`), never
in the tool-call arguments. Without this variable set by the operator, the tool
refuses everything — see `scirust-discovery/README.md`.

A new domain (e.g. `scirust-discovery`, a future exposed `scirust-pdm`) is
added by implementing `fn xxx_tools() -> Vec<McpTool>` in `src/tools/` and
registering it in [`default_registry`](src/lib.rs) — no other change is
required for all existing MCP clients to see it.

## Auditability

Every `tools/call` — success or failure — is appended to a SHA-256 hash-chained
log (`src/audit.rs`, `AuditLog`), on the same principle as
`scirust-func-safety::audit` (each entry contains the hash of the previous one,
which makes any subsequent tampering detectable), but with a real SHA-256
(reusing `scirust_sciagent::sha256`, from the public-domain FIPS 180-4) rather
than a homegrown hash — for an audit integrity trail, collision resistance is
non-negotiable. The log stores the **hash** of the arguments and the result,
not their plaintext content: it can be exported without exposing potentially
sensitive data from a client infrastructure. To preserve the historical public
format, `AuditLog::export_json` always produces the array of entries alone.
`AuditLog::export_snapshot_json` produces a versioned envelope with the anchor
preceding the window preserved, the head, the next sequence number and the
entries; `AuditExport::from_json` checks its consistency after rotation.

This SHA-256 chain is, however, neither a signature nor a MAC: it does not prove
the identity of the producer, and an actor able to replace the entire export can
recompute all of its hashes. To obtain proof of tampering, keep the `head`
through an independent trusted channel, then use
`AuditExport::validate_against_head`. A trusted `anchor` establishes continuity
with the past, but does not by itself authenticate the following entries.

## Usage

```bash
cargo run -p scirust-mcp --bin scirust-mcp
```

The server reads JSON-RPC 2.0 requests on stdin (one per line) and writes
responses to stdout (one per line) — this is MCP's `stdio` transport,
compatible with Claude Desktop and any other standard MCP client. Example
Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "scirust": {
      "command": "cargo",
      "args": ["run", "--release", "-p", "scirust-mcp", "--bin", "scirust-mcp"],
      "cwd": "/path/to/scirust"
    }
  }
}
```

`SCIRUST_BIN` (environment variable) points the `scirust_cli` tool to an
already-compiled `scirust` binary (`cargo install --path scirust-cli`) rather
than rebuilding it on every call.

For a local development session only:

```bash
SCIRUST_MCP_PROFILE=development SCIAGENT_ROOT="$PWD" \
  cargo run -p scirust-mcp --bin scirust-mcp
```

Any other value of `SCIRUST_MCP_PROFILE` is refused at startup.

## Sources

- Model Context Protocol — specification: <https://modelcontextprotocol.io>
- Anthropic, "Introducing the Model Context Protocol", Nov. 2024.
- Comparison with Google Agent2Agent (A2A, April 2025): MCP is "an agent uses a
  tool", A2A is "an agent delegates to another agent" — the two are
  complementary, not competing.
