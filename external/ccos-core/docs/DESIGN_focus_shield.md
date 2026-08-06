# Design — the attentional shield (`ccos focus`) and its IDE client

The honest pivot the field data points to (see `FIELD_CAMPAIGN_H.md`): CCOS's **proven**
strength is *coverage* — putting a file's cross-file dependencies in a tight budget **81–100 %**
of the time vs **0–2 %** for naively opening it. Its *sufficiency* (does the context make a
model **resolve** better) is real but **narrow** (multi-file bugs, ~1–2 % of fixes) and depends
on the model using the context well. So the strongest, most honest product leans on coverage and
lets a **human** supply the reasoning CCOS can't guarantee: an **attentional shield**. When a
build/test fails, don't show the raw 50-line backtrace — show the **causal region** (the likely
root cause + its direct dependencies), hide the noise.

## Phase 0 — `ccos focus` (BUILT)

The brain is a CLI command, so the editor plugin is a thin client (and the value is provable
without any editor toolchain).

```
cargo test 2>&1 | ccos focus src          # human shield
cargo test 2>&1 | ccos focus src --json   # editor-client payload
ccos focus src --input red.txt            # from a captured file
```

It **composes already-proven pieces**: ingest the tree into an `AgentSession`, `page_fault` the
trace (parse faulting locations → inject failure pressure → recall the causal region — the whole
triad + budget balancing), then render. The only new logic is `focus_view` (pure, unit-tested):
reduce the window to one entry per file, tag the **trace's own files as the symptom** and the
top **causally-pulled file as the likely cause** — the "skip to the root" signal a backtrace
buries.

Verified end-to-end on a multi-file bug (symptom `writer.rs`, cause `config.rs`):

```
⚡ CCOS focus — 3 files in workspace → 3 in view (~874 tokens)
  panicked: …panicked at src/writer.rs:3:78
  symptom:  src/writer.rs:3
  ▸ src/writer.rs   · symptom site
  ▸ src/config.rs   ◀ likely cause (pulled in causally)
      pub fn buffer_size() -> usize { 0 }   // ROOT CAUSE: should be 8
```

The root cause is surfaced **without opening a file**, the 400 padding functions and the
backtrace are hidden. `--json` emits `{message, symptom_files, workspace_files, tokens,
entries:[{file, role, score, content}]}` for a client to render.

## Phase 1 — the IDE thin client (SCOPED, not built)

The plugin owns *only* capture + render; the brain stays in `ccos focus --json`.

- **VS Code** (~1–2 days for an MVP): a `tasks`/terminal listener (or a `cargo test` wrapper
  task) captures stderr → spawn `ccos focus <ws>/src --json` → render `entries` as a
  webview/tree panel; clicking the **cause** entry opens that file at the symbol. ~200–300 lines
  of TypeScript, no new Rust.
- **Neovim** (~1 day): a Lua autocommand on the test/quickfix output → `jobstart('ccos focus …
  --json')` → populate a floating "cause" window / the quickfix list, cause first.
- **Incremental freshness**: re-ingest a file on save so the graph tracks edits (a `ccos focus`
  that reuses a persisted `workspace.ccos` instead of re-ingesting every run — a `--workspace`
  flag — would make this O(Δ); a small follow-on).

**Effort to a usable demo: ~2–3 days of client code, zero kernel rewrite** — exactly Gemini's
"change the client connected to the MCP server", grounded.

## Why this is the right pivot (and what it deliberately avoids)

- It **uses the proven axis** (coverage 81–100 %), not the narrow one (resolution on ~1–2 %).
- The **human is the reasoner**, so it sidesteps CCOS's "necessary but not sufficient" caveat —
  the human supplies the sufficiency a weak (or even strong) model didn't (`FIELD_CAMPAIGN_H.md`
  R03). The shield doesn't have to *be right* about the cause; ranking it first and hiding noise
  is already a win.
- It is **honest forensics-adjacent without the overclaim**: it shows *what context the failure
  implicates*, it does not claim to read anyone's mind.

## Honest limits (carry these into any pitch)

- The causal graph is **structural, not semantic** (containment/`use` edges, not call/data flow)
  and the default parser is a **line-based Rust heuristic** — broad editor use wants
  `--features syn-parser` and multi-language parsing (a real, separate effort).
- `focus` keys files by the **`src/…` tail** to match `cargo`'s paths; a non-`src` layout or a
  workspace with multiple crates needs path-mapping work.
- The "likely cause" is the **top non-trace file** — a heuristic. It is right when the cause is a
  pulled-in dependency (the case that matters); on a single-file bug there is no separate cause
  and it should (and does) just show the symptom region. False positives are the UX risk to watch.
- Re-ingesting the whole tree per invocation is fine for a CLI; an editor wants the persisted-
  workspace `--workspace` follow-on for O(Δ) freshness.

## Status

Phase 0 shipped: `ccos focus` (human + `--json`), `focus_view` unit-tested, verified on a
multi-file bug, clippy/fmt clean. Phase 1 is a thin client over `--json` — the next concrete
build if we pursue the shield.
