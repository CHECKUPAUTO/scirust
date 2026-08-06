# CCOS editor clients — the attentional shield

Thin clients over **`ccos focus --json`**. When a `cargo test` run fails, instead of the
raw 50-line backtrace they show the **causal region** — the likely root cause first, then the
symptom, then the rest — and hide the noise. All the intelligence is in the `ccos` binary
(the triad + budget balancing + the `focus_view` cause/symptom tagging); the clients only
*capture the failure → call `ccos focus` → render*.

Build the binary first: `cargo build --release` (then put `target/release/ccos` on `PATH`,
or set the `ccos.binary` / `opts.ccos` path).

## The contract (write your own client against this)

```
ccos focus <src> --input <cargo-output> --workspace <ws.ccos> --json
```
emits:
```json
{
  "message": "…panicked at src/writer.rs:3:78",
  "symptom_files": ["src/lib.rs", "src/writer.rs"],
  "workspace_files": 3,
  "reparsed_files": 0,
  "tokens": 874,
  "entries": [
    { "file": "src/config.rs", "role": "cause",   "score": 0.67, "content": "pub fn buffer_size() -> usize { 0 }" },
    { "file": "src/writer.rs",  "role": "symptom", "score": 0.72, "content": "…" }
  ]
}
```
`role` is `cause` (top file pulled in *causally*, not named by the trace — the likely root),
`symptom` (a file the trace names), or `context`. `--workspace` makes re-parse **O(Δ)**
(only changed files), so calling `focus` on every save stays ~instant (~15 ms warm).

## Neovim (`editors/nvim/`)

```lua
-- lazy.nvim
{ dir = "/path/to/CCOS/editors/nvim", config = function()
    require("ccos_focus").setup({ ccos = "ccos", keymap = "<leader>cf" })
  end }
```
`:CcosFocus` (or `<leader>cf`) runs the test command, then floats the shield; `<CR>` on a file
opens it, `q`/`<Esc>` closes. Config: `ccos`, `src`, `workspace`, `budget`, `test_cmd`, `keymap`.

## VS Code (`editors/vscode/`)

```sh
cd editors/vscode && npm install && npm run compile && npm test
```
Press **F5** to launch an Extension Development Host, then run **“CCOS: Focus on failure”** from
the command palette. It runs the test command and opens a side panel, *CCOS Attentional Shield*;
clicking the **likely cause** opens that file. Settings under `ccos.*` (binary, src, workspace,
budget, testCommand).

The `--workspace` checkpoint lands in `.ccos/` at the crate root — add `.ccos/` to your
project's `.gitignore`.

## Status — honest

- **Backend** (`ccos focus`, `--json`, `--workspace`): verified end-to-end on a multi-file bug
  (the cross-file cause is surfaced and tagged), unit-tested, in the main CI gate.
- **VS Code extension**: `extension.ts` **type-checks clean against the real `@types/vscode` API**
  (`npm run compile`, tsc strict, exit 0), and the pure render logic (`src/render.ts`) **passes
  its tests** (`npm test` — verified against real `ccos focus --json` output). The glue still
  needs the editor (F5) to exercise the actual UI; only the compile + render tests run headless.
- **Neovim plugin**: **loads + registers `:CcosFocus` on real Neovim 0.9.5/arm64** (verified
  headless on the Jetson). `require("ccos_focus").render(payload)` is exposed so the float
  render can be smoke-tested headless too:
  ```sh
  nvim --headless --noplugin -u NONE -c "set rtp+=…/CCOS/editors/nvim" \
    -c "lua local m=require('ccos_focus'); m.render({workspace_files=3,tokens=9,message='x', \
        entries={{file='src/filter.rs',role='cause',score=0.7,content='pub const MIN_SCORE: f64 = 0.0;'}}, _root='/tmp'}); \
        assert(table.concat(vim.api.nvim_buf_get_lines(0,0,-1,false),'\n'):find('filter.rs'),'no render'); \
        print('PASS: render')" -c "qa"
  ```
  The interactive bits — the floating window appearing, `<CR>` opening the file — still need a
  real session. Treat both clients as MVPs to drive, not shipped extensions.

## Known limits (inherited from the kernel)

The causal graph is structural, not semantic; the default parser is a line-based Rust heuristic
(`--features syn-parser` for a real AST). `focus` keys files by the `src/…` tail to match
`cargo`'s paths — a multi-crate workspace needs path-mapping. The “likely cause” is a heuristic
(the top non-trace file): right when the cause is a pulled-in dependency, and harmless on a
single-file bug (it just shows the symptom region). See `docs/DESIGN_focus_shield.md`.
