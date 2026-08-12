# Evolving an algorithm from your LLM (MCP server)

This guide lets a **non-technical user** ask their LLM, in natural language, to
**connect to scirust** and evolve an algorithm from examples they provide.

The mechanism is the **Model Context Protocol (MCP)**: scirust exposes a small
server (`scirust-rsi-mcp`) that your LLM client calls like a tool. The server
evolves an arithmetic program (over an input `x`) to reproduce your
*input → output* examples, under scirust's guarantees (bounded, elitist/
non-regressive, reproducible, sandboxed).

> ⚠️ A **one-time** configuration is required: an LLM can only connect to an
> external tool after declaring it once in an MCP-compatible client (Claude
> Desktop, Claude Code, etc.). A purely web-based LLM without MCP support
> cannot connect to it — see the note at the bottom.

## 1. Compile the server (once)

```sh
cargo build -p scirust-rsi --bin scirust-rsi-mcp --features mcp --release
# produced binary: target/release/scirust-rsi-mcp
```

## 2. Declare the server in your client (once)

**Claude Code** (CLI):

```sh
claude mcp add scirust -- /absolute/path/to/target/release/scirust-rsi-mcp
```

**Claude Desktop** — add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "scirust": {
      "command": "/absolute/path/to/target/release/scirust-rsi-mcp"
    }
  }
}
```

Restart the client. The `evolve_algorithm` tool is now available.

## 3. The command to paste into your LLM

Copy-paste (adapting your examples):

> Connect to the "scirust" MCP server and use the **`evolve_algorithm`** tool
> to evolve a program that reproduces these input → output examples:
>
> `1 → 2, 2 → 4, 3 → 6, 4 → 8`
>
> (in other words: double the input). Give me the evolved program, its error,
> and check that it matches every example.

The LLM will call the tool with `examples = [[1,2],[2,4],[3,6],[4,8]]` and
answer you with the program found (here `x x +`, i.e. `2·x`), its error (0),
and the verification table.

### Evolving *your* starting algorithm

If you already have an algorithm and want to improve it, give it as a starting
point — the result will **never be worse** than yours (elitist selection):

> … use `evolve_algorithm` with my examples `-1 → 2, 0 → 1, 1 → 2, 2 → 5`
> and **`seed_program` = "x x *"** as the starting point. Improve it until it
> matches the examples.

## The `evolve_algorithm` tool

| Argument | Required | Default | Role |
|---|---|---|---|
| `examples` | yes | — | `[input, output]` pairs, e.g. `[[1,2],[2,4]]` |
| `seed_program` | no | `"x"` | starting program (reverse Polish notation) |
| `max_iters` | no | `1500` | iteration cap (always bounded) |
| `samples` | no | `32` | candidates proposed per round (best-of-n) |
| `seed` | no | `0` | RNG seed → reproducible run |

The program is expressed in **reverse Polish notation** over `x`, with the
tokens `x`, numbers, and `+ - * /`. Example: `x x * 1 +` means `x² + 1`.

## What scirust guarantees (and what it does not do)

- **Bounded**: `max_iters` ⇒ evolution always terminates.
- **Non-regressive**: elitist adoption ⇒ the result is never worse than the
  starting program.
- **Reproducible**: same `seed` ⇒ same result.
- **Sandboxed**: only a fixed arithmetic interpreter is executed — no generated
  code is run, no access to the machine, no self-rewriting.

Evolution runs **locally and offline** in the server: no API key is required.
The LLM only serves to translate your request into a tool call and to explain
the result to you.

## Limitation & alternative

A web LLM without MCP support (e.g. a basic chat interface) cannot connect to
a local binary. Two options in that case:

1. Use an MCP-compatible client (Claude Desktop / Claude Code) — recommended.
2. Host this engine behind an **HTTP API** that the LLM can call (not included
   here; the core `scirust_rsi::progevo::evolve` is directly reusable for
   that).
