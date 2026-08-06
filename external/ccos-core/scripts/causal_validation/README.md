# CCOS causal-validation harness

A closed-loop harness that tests CCOS's central claim against the repository's
own Git history, **without an LLM**. It is the empirical counterpart to the
synthetic `ccos experiment` / `ccos eval` benchmarks.

> **Claim under test.** When a fault is injected at one file, CCOS's failure
> propagation pulls the *other files a real fix had to touch* into a bounded
> working set — better than a budget-blind baseline.

## Phases

| Phase | What it does | Status |
| ----- | ------------ | ------ |
| **1 — Mine & inject** | Scan `git log` for fix commits; the changed `.rs` files that already existed at the parent are the ground truth `F_target`; check out the parent in a throwaway worktree; build a CCOS snapshot; inject a fault at the highest-out-degree changed file `n_fail` (`ccos failure … --json`). | ✅ `validate.py` |
| **2 — Coverage** | For each node budget `K`, score `R_cov = \|F_target ∩ WorkingSet_K\| / \|F_target\|`; report the arithmetic and **geometric** mean over scenarios. | ✅ `validate.py` |
| **3 — Optimise weights** | Treat the 5 scoring/decay knobs (now `CCOS_W_*` / `CCOS_FAILURE_DECAY`) as hyperparameters; maximise geo-mean `R_cov` under **leave-one-commit-out** cross-validation. | ⏳ planned (wraps this module) |
| **4 — End-to-end vs RAG** | Real LLM, real patch-and-test loop; resolution rate, tokens, iterations. | ⏳ planned (needs a model — e.g. local Ollama on the Jetson) |

Phases 3–4 deliberately reuse Phase 1–2: the weights flow through the
environment, so an optimiser can wrap `validate.py` unchanged.

## Requirements

Standard-library Python 3.9+ and a built `ccos` binary. No third-party packages
(Phase 3 will add `optuna`/`scipy`, optional).

## Usage

```bash
# Validate the pipeline on a single commit (verbose, no aggregation):
python scripts/causal_validation/validate.py --dry-run

# Score a batch and write per-scenario JSONL:
python scripts/causal_validation/validate.py --limit 25 --k 20 50 100 \
    --out scripts/causal_validation/results.jsonl

# Point at a larger external repo (recommended — see caveat below):
python scripts/causal_validation/validate.py --repo /path/to/some-rust-project \
    --subdir src --limit 50
```

Key flags: `--repo` (target repo, default = this one), `--subdir` (default
`src`), `--k` (budgets), `--depth` (propagation depth), `--keywords`,
`--ccos-bin` (skip the release build), and `--w-failure …` / `--failure-decay …`
to override weights for a single run.

## How a scenario is scored

```
fix commit N ──diff──> F_target = changed .rs files present at N-1
     │
     └─ checkout N-1 (worktree) ─> ccos analyze ─> snapshot (file:<path> nodes)
            │
            └─ n_fail = changed file with max out-degree
                   └─ ccos failure snap n_fail --max-nodes K --json
                          └─ WorkingSet_K ──> R_cov = |F_target ∩ WS_K| / |F_target|
```

Robustness: every subprocess is timed out and captured; a scenario that fails to
analyze, or whose changed files aren't graph nodes, is skipped (not fatal);
worktrees are always cleaned up.

## First findings — and the improvement they drove

Run against **this** repository (a young prototype), the harness finds only a
handful of qualifying commits. The first run exposed a real limitation and the
fix for it is itself measured here — the loop earning its keep.

**Baseline (downstream-only propagation):** `R_cov ≈ 0.30`, **flat across K** —
in every scenario only the seed file is recovered (`R_cov = 1/|F_target|`). The
cause was structural: co-changed files are typically *upstream importers* of the
fault node, linked only through `dep:` hubs, whereas `propagate_failure` walked
`source → target` only, so the pressure never reached them.

**Fix:** (a) `ccos analyze` now resolves intra-crate imports into `file→file`
edges (`link_module_imports`), and (b) `ccos failure --bidirectional` propagates
pressure in both directions (`propagate_failure_bidirectional`). Re-running with
`--bidirectional`:

| K | downstream-only | bidirectional |
| --- | --- | --- |
| 20  | 0.333 | 0.278 |
| 50  | 0.333 | **0.444** |
| 100 | 0.333 | **0.611** (33% fully covered) |

So bidirectional propagation roughly **doubles coverage at a moderate/large
budget**, at the cost of **diluting** it at a very tight one (`K=20`): marking
neighbours on both sides floods a small working set and can evict a target. A
real, falsifiable trade-off — exactly what the harness is for.

### Across mature external repos (70 real bugs)

The same comparison on three established crates — `R_cov` as
*downstream-only* / **bidirectional**:

| Repo (n)         | K=20        | K=50          | K=100         |
| ---------------- | ----------- | ------------- | ------------- |
| `fd` (25)        | 0.50 / 0.28 | 0.50 / **0.92** | 0.50 / **0.96** |
| `bat` (25)       | 0.84 / 0.20 | 0.84 / **1.00** | 0.84 / **1.00** |
| `hyperfine` (20) | 0.64 / 0.19 | 0.73 / **0.85** | 0.81 / **0.92** |

Across 70 mined fix commits, at a sufficient budget (`K≥50`) CCOS recovers
**0.85–1.0 of the files a fix touched**, with a systematic tight-budget (`K=20`)
dilution (0.19–0.28).

### Is that any good? CCOS vs a lexical-RAG baseline (the honest part)

A high `R_cov` only matters if it beats the obvious baseline. The harness now also
runs **classical lexical RAG** (TF-IDF cosine over file text, queried by the fault
file) at the *same file budget*, isolating the selection rule. `R_cov` as
**CCOS / RAG**:

| Repo (n)         | K=20            | K=50            | K=100           |
| ---------------- | --------------- | --------------- | --------------- |
| `fd` (25)        | 0.28 / 0.34     | 0.92 / 0.94     | 0.96 / 0.98     |
| `bat` (25)       | 0.20 / **0.56** | 1.00 / 0.98     | 1.00 / 1.00     |
| `hyperfine` (20) | 0.20 / **0.73** | 0.87 / 0.92     | 0.97 / 0.96     |

**The honest conclusion: causal selection has no net coverage advantage over
lexical similarity here, and is worse at a tight budget.** At `K≥50` the two tie
(differences are smaller than the per-scenario std ≈ 0.2); at `K=20`, RAG is
clearly better because bidirectional propagation dilutes a small window. On real
bugs the files a fix co-touches are *lexically similar to each other*, so TF-IDF
finds them too — the synthetic premise (a cause that is lexically dissimilar from
the symptom, `experiment.rs` §8) does **not** reproduce on these repos.

So the strong-looking `0.85–1.0` is real but **not a CCOS win**: it is the
*necessary* condition (the fix's files are in the window), which a plain
retriever also satisfies.

```bash
git clone https://github.com/sharkdp/bat /tmp/bat
python scripts/causal_validation/validate.py --repo /tmp/bat \
    --ccos-bin target/release/ccos --limit 25 --bidirectional   # prints CCOS vs RAG
```

**Where CCOS might still differ (untested here):** robustness when the *query* is
degraded or absent (the §8 premise), and when the fault *symptom* (a failing test)
is lexically far from its *cause* — which needs the real Phase-1 (run the tests to
pick the seed) rather than the highest-degree-changed-file heuristic. And the
*sufficient* condition (an LLM's patch passes the tests, Phase 4) is still
untested. Until one of those shows a gap, **CCOS ≈ RAG on this metric.**
