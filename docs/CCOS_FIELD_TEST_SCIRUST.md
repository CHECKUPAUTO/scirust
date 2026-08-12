# CCOS Core — field test on scirust

Field-test campaign of [CCOS Core](CCOS_MEMORY.md) (`external/ccos-core`, upstream
`9c1b7d9`) as the agent's causal memory, run on this repository. Goal:
verify by measurement what the memory holds and where it breaks — not to
present it in its best light.

Environment: x86_64-linux, `--release --features llm,license` build,
`ccos 0.4.0-pre`, community tier (no vendor key, Pro fail-closed).

## Verdict

The core of the promise holds and measures: bounded-budget causal coverage,
bit-for-bit determinism, the hash chain, and injection detection all behave
as advertised. **One real defect was found and fixed**: the `ccos memory` CLI
facade did not page in the COLD tier, so a `recall around` on a demoted
anchor returned an **empty** window where the same call via MCP
returned 31 items.

| Trial | Result |
|---|---|
| 1. State continuity (MCP) | ✅ workspace reloaded, valid chain, coherent timeline |
| 2. Causal coverage under budget | ✅ 88–100% vs 0% for a naive dump |
| 3. Real page fault → cross-file cause | ✅ the cause surfaces, `causal_blame` ranks it #2 |
| 4. Determinism / integrity / time-travel | ✅ byte-identical recall, falsification detected |
| 5. Adversarial robustness | ✅ injection scored 1.0, degenerates without crash |
| 6. Upstream test suite | ✅ 758 tests, 0 failures (with the fix) |
| — Defect found | ⚠️ COLD page-in missing from the CLI facade → **fixed** |
| — Upstream bug found | ⚠️ paging loses causal edges → **fixed upstream** (§8) |

## 1. State continuity

The MCP server (`.mcp.json`) reloads the workspace bootstrapped by the CLI: `stats`
returns 4 974 nodes / 8 528 edges / 167 events, `verify` → `valid: true`,
`timeline` replays the operations in order. The memory therefore survives
process and access-path changes (CLI ↔ MCP, shared format).

## 2. Causal coverage under budget (upstream harness, on scirust)

Reproduction of the upstream model-free protocol
(`external/ccos-core/scripts/ccos_context_value.py`): for each file with at
least one `use crate::` dependency, what fraction of its real dependencies
fits in a 2 048-token window?

| Crate | Files analyzed | Dep. edges | CCOS | Naive dump |
|---|---|---|---|---|
| `scirust-core` | 10 | 21 | **100 %** | 0 % |
| `scirust-sim` | 19 | 24 | **100 %** | 0 % |
| `scirust-estimation` | 13 | 16 | **88 %** | 0 % |
| `scirust-fluids` | 8 | 8 | **100 %** | 0 % |

On the large files — those whose naive opening truncates every dependency —
the gap is 85–100% versus 0%. The numbers fall within the 81–100% range
announced upstream, measured here on a corpus different from theirs.

**The only shortfall (`scirust-estimation/src/rls.rs`, 0/2)** is explainable and
non-trivial: its two `use crate::` declarations are **inside a test
function** (`rls.rs:409-410`), not at module level. The measurement harness
counts them as real dependencies; the AST parser attaches them to the local
scope. This is a convention divergence between the oracle and the parser,
not a causal loss.

## 3. Real page fault and propagation

A real `cargo test` trace (a `Conv2d::forward` panic on a shape
mismatch) injected via the MCP `page_fault` tool: the rebuilt window contains
the symptom (`nn/nd_layers.rs`) **and** the cross-file cause named in the
backtrace (`tensor/tensor_nd.rs`, score 0.72), plus `error.rs`, `autodiff/nd.rs`
and `nn/rng.rs` — the symptom's direct dependencies.

`causal_blame` on the symptom file ranks `tensor_nd.rs` at the top of the
real candidate causes (weight 1.97, just behind the `dep:crate` pseudo-node),
ahead of `error.rs` (1.56) and `autodiff/reverse.rs` (1.33) — the order a
human would give when reading the backtrace.

## 4. Determinism, integrity, time-travel

- **Deterministic reconstruction.** Two fresh workspaces built from the same
  input (36 files) produce a **byte-identical** `recall` and identical `stats`.
- **Hash chain.** Falsifying one byte in an event's payload →
  `content tampered — hash mismatch`. Falsifying one chain link →
  `broken link — prev_hash does not match the chain` **plus** the cascade over
  the following events. Both are detected, `valid: false`.
  *Honest nuance*: a first attempt modifying the reconstructed **snapshot**
  state (not the log) passed `verify` — this is coherent (the log is the
  source of truth and it is re-verified), but it means `verify` attests to
  the journal, not the working copy.
- **Time-travel.** `recall_what_if(step=3)` replays the window as it was
  before the page fault: it contains the `nd_layers` symbols but **not**
  `tensor_nd.rs` — one literally sees the cause entering context at the next
  step. The post-mortem `missing` watchpoint dates the eviction (`○○○○●`) and
  quantifies the shortfall in tokens.

## 5. Adversarial robustness

| Input | Behavior |
|---|---|
| Blatant prompt injection (key exfiltration) | `injection_score: 1.0`, `flagged: true` |
| Healthy control file | `score: 3.6e-16`, not flagged (no false positive) |
| Zero-width + base64 obfuscation | `ZWSP` anomalies located by offset/codepoint |
| Control bytes (NUL, 0x01…) | `Control` anomalies reported, no crash |
| 914 KB file | ingested in 28 s, 4 994 nodes, no crash |
| Empty file / 50 000 identical lines | absorbed without error |

## 6. The defect found — COLD page-in missing from the CLI facade

**Symptom.** On a 300-file workspace (3 635 nodes demoted to COLD), the
same `recall around` on the same anchor and the same budget:

```
via MCP  (AgentSession) : items = 31, tokens = 2048
via CLI  (ccos memory)  : items =  0, tokens =    0
```

**Cause.** `MemoryProvider::recall` takes `&self`: it cannot page in. It is
`CcosMemory::ensure_resident` (`&mut self`) that makes the COLD tier transparent, and
its own documentation says that "the session layer calls this before an Around
recall". `AgentSession::recall` does it (`agent_session.rs:1277`) — so the
MCP path is correct — but the CLI facade called `mem.recall()` directly
(`main.rs:2862`), without page-in. The COLD tier was transparent on only one
of the two paths.

**What it was not.** First (wrong) diagnosis: "the 5 000-node cap is too
low". Instrumented, the cap does exactly its job — with a cap of 1,
`page_in` brings back 3 nodes and re-paging demotes 2, which is the correct
behavior of a bounded cache. The defect was the missing call, not the cap
value.

**Second point: a fix attempted, then withdrawn.** `CcosMemory::new()` hardcodes
`MemoryGraph::new(0.2, 5000)` while `CCOS_MAX_RESIDENT` is documented as the
cap setting and `commands_runtime.rs` already honors it via
`new_from_env`. Both the facade and the MCP therefore ignore the variable.
The "obvious" fix — making `new()` read the environment — was implemented,
tested, **then reverted**: it brings nothing and breaks certification. See §7.

**Fix retained** (`external/ccos-core/src/main.rs`, one
non-regression test): the facade calls `ensure_resident` before an `Around`
recall, like the session. The fix covers its **two** callers, `ccos memory` and
`ccos stdin`, which share `run_op_stream`.

**Verification.** Same workspace, same anchor, same budget:

```
CLI before : items =  0, tokens =    0
CLI after  : items = 31, tokens = 2048   ← parity with MCP
```

Full upstream suite: **626 tests, 0 failures** (625 upstream + the added test).

## 7. What an adversarial review of the fix gave

The above fix was submitted to a fan-out review (inventory of
call sites, replay/persistence risk, scope of the setting, upstream
conventions), then each serious finding was submitted to an independent
refuter. Two findings were announced as **blocking**. Both turned out to be
false **at equal configuration**, and verifying this brought a real upstream
bug to light.

**"The page-in destroys the timeline" — refuted.** Measured: the timeline drops
from 8 to 0 operations **also with the pristine binary and with no recall at
all**. The cause is the already-documented "two writers" conflict
(`docs/CCOS_MEMORY.md`): a mutating CLI stream on a workspace held by MCP
diverges the snapshot from the op-log, and `AgentSession::open`'s
consistency guard restarts from the snapshot. The fix is not involved.

**"The page-in destroys causal edges" — refuted, but revealing.**
At an identical cap (20, persisted in the snapshot), on the same workspace:

| Binary | Stream | items | edges |
|---|---|---|---|
| pristine | ingest only | — | 16 → **11** |
| pristine | ingest + recall | 0 | 16 → **11** |
| fixed | ingest only | — | 16 → **11** |
| fixed | ingest + recall | 3 | 16 → **12** |

The loss of 5 edges is **identical without the fix and without recall**: it
comes from the demotion itself, not from the page-in. The fix even preserves
one more (12 vs 11), since the page-in brings an edge back.

**The real defect, for its part, is upstream and pre-existing: paging loses
edges.** Demoting then re-paging restores an archived edge only if both its
endpoints are resident (`memory.rs`, `page_in`); an edge whose other endpoint
is still COLD is deleted from the only place it existed. This contradicts the
announced COLD-tier contract ("non-destructive: the node and its links are
kept"). It is independent of this fix, it also affects the MCP path, and it
deserves a separate upstream report.

**Finding retained, and it killed half the fix.** The second part of the
patch — making `CcosMemory::new()` read `CCOS_MAX_RESIDENT` via `new_from_env`
— was **reverted** after verification. Three reasons, all measured:

1. **Inert where it matters.** `open()` only calls `new()` if the file is
   absent, and the cap is a serialized field: an existing workspace keeps its
   own. Measured: 60 resident / 0 COLD when reopening under
   `CCOS_MAX_RESIDENT=5`, versus 5 / 55 for a fresh workspace. My initial
   checks ("cap 500 → 500 resident") were each time on a **fresh** workspace —
   a false positive I had not seen.
2. **Breaks certification.** With `CCOS_MAX_RESIDENT=3` in the environment,
   `ccos setup` drops from `6/6 checks passed` to **`4/6 — NOT certified`**
   (`causal recall` and `failure propagation` fail), while the pristine
   binary stays at 6/6. This directly contradicts the
   "deterministic by construction" contract of `setup.rs`.
3. **Makes replay sensitive to the ambient environment**, and the associated
   test was a false friend: its mutex only protected itself, while a hundred
   constructors of the same test binary now read the variable.

The original defect — the facade ignores `CCOS_MAX_RESIDENT` — is therefore **real and
uncorrected**. It is documented at the code location, with the reason why the
obvious fix is worse than the disease.

## 8. Remaining defects, fixed upstream

The findings left open in §7 were handled in
[`Memorithm/CCOS-Core#2`](https://github.com/Memorithm/CCOS-Core/pull/2).

**The COLD tier was not non-destructive.** `demote` archives an edge on
**one** side only; `page_in` only relinked it if both its endpoints were
resident and discarded it otherwise — even though the `ColdNode` carrying it
had just been removed, so it was the last copy. Measured on a 4-node graph
(demote everything, re-page everything): **4 edges before, 3 after**. An edge
whose other end is still COLD is now entrusted to that neighbor, and the
reverse adjacency entry is kept.

**`ensure_resident` did not guarantee its own postcondition.** The capacity
swap of `page_in` only protects the node it just restored: under a too-tight
cap, paging in a neighbor could re-evict the requested anchor.
The anchor is now restored last. This was not theoretical — an existing
upstream test started failing as soon as the page fault began paging in the
region.

**The post-mortem debugger saw everything except demoted nodes.**
`recall_what_if`: **0 items** where the live recall returned **31**, same
anchor and same budget. The asymmetry was inverted — replaying a *recorded*
`Around` op goes through `ensure_resident` again, so the what-if worked for
an already-consulted anchor and failed for one that had not been, i.e.
exactly the question the feature exists to answer.

**`page_fault` and the `setup.rs` self-test** receive the same page-in, for
consistency. Honestly: on a large real workspace, this changes no result,
since failure propagation had already brought back the useful nodes. My
"26 vs 32" observation actually measured a **scoring** gap, not a paging one
— verified after the fix, the figure is unchanged. Only the unit test,
which fails without the call, proves the defect.

Methodological lesson: my own initial empirical check — "the persisted
resident/COLD split is identical" — compared node counters and **not edges**,
so it was too coarse to see anything. It is the equal-configuration
comparison, not static analysis alone, that settled both directions.

## Known limits, at monorepo scale

- **Ingestion cost.** The repository's 2 055 `.rs` files (28 MB) ingest in
  **5 min 36 s** (~6 MB/min), 70 MB snapshot. Bootstrapping `scirust-core`
  alone takes ~3 s. For daily use, bootstrap the crates you work on, not the
  whole monorepo.
- **The cap remains 5 000 nodes by default.** At monorepo scale, nearly all
  of the graph goes COLD (52 301 demoted nodes). Page-in makes it functionally
  transparent, but if you want to keep the whole monorepo resident you must
  raise `CCOS_MAX_RESIDENT` (now effective) — at the cost of RAM.
- **`verify` attests to the log**, not the working snapshot (see §4).

## Reproduce

```bash
sh scripts/ccos/install.sh

# §2 — causal coverage
python3 external/ccos-core/scripts/ccos_context_value.py scirust-core/src \
  --budget 2048 --ccos "$(command -v ccos)"

# §4 — determinism: two fresh workspaces, same input, compare the outputs
# §6 — CLI/MCP parity on a demoted anchor
CCOS_MAX_RESIDENT=500 ccos memory --path /tmp/ws.ccos < ops.jsonl
```
