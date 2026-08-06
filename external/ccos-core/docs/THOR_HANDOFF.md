# Thor handoff — confirm generalisation, then test sufficiency (Q7)

## Where we are

The "context assembly" core of CCOS was failing on real code; three measured fixes (the
triad) plus a budget-balancing fix landed and are validated on **CCOS's own src** *and* the
**`syn` crate** (independent):

- **#1 granularity** — no node carries a whole file (region blow-up 15× → ~1×).
- **#2 degree-aware propagation** — a hub distributes pressure, no flood (518 → 37 on CCOS).
- **#3 anchor proximity** — near neighbours outrank distant component noise.
- **budget balancing** — header cap + per-file cap, so deps fit a fixed budget regardless of
  anchor size (`syn` item.rs: 1/7 → **7/7** at budget 2048).

At a size-appropriate budget, `recall around <failing file>` returns the file **and all its
real dependencies** in ~a few % of the cost of dumping the crate. Details +
before/after numbers: `docs/FIELD_CAMPAIGN_H.md`, `docs/DESIGN_symbol_granularity.md`,
`docs/DESIGN_recall_budget.md`.

**Two questions remain — that's your run:**

1. **Generalisation.** Does this hold on a *third* independent repo (not CCOS, not `syn`)?
2. **Sufficiency (the decisive one — paper Phase 4 / Q7).** At an *equal token budget*, does
   the CCOS causal window make your local LLM **resolve** a bug better/cheaper than a naive
   file dump? Coverage and frugality are necessary; *this* is the claim that matters.

## Setup

```sh
cargo build --release            # produces ./target/release/ccos
```

Both helper scripts drive the `ccos mcp` server over stdio; they need only Python 3.

## Campaign I — generalisation (structural, model-free)

Pick **2–3 real Rust crates you have** with a mostly **flat `src/*.rs`** layout (CCOS's
cross-file linking is strongest there; sub-`mod.rs` trees are a separate known limit). Good
picks: `ripgrep`'s `grep-matcher` / `grep-regex`, `bat`'s `src`, `fd`'s `src`, or any crate
in `~/.cargo/registry/src`.

```sh
python3 scripts/ccos_campaign_probe.py <crate>/src --budget 2048 --out corpus_I/<crate>.json
python3 scripts/ccos_campaign_probe.py <crate>/src --budget 8192 --out corpus_I/<crate>-8k.json
```

It prints, per crate: the #1 blow-up factor, and per anchor the #2 flood and #3 coverage
(`deps_in_win`, `window_tok`, `% all-src`, `noise`). **Fill this grid** (one row per anchor):

| crate | anchor | #deps | deps_in_win | window_tok | % all-src | noise | blow-up |
| ----- | ------ | ----- | ----------- | ---------- | --------- | ----- | ------- |

**Expected (honest):** blow-up ~0.5–1.5×; coverage = all deps once the budget fits the
anchor's dep count (a high-fan-out file like `lib.rs` needs more budget — that *is* the
budget-scaling signal); window a few % of all-src; noise small at a tight budget. If a crate
breaks linking (deps never appear even at large budget), that's a **parser finding** — note
the crate and a file, it justifies the sub-`mod.rs` parser work.

## Campaign J — sufficiency / Q7 (the decisive one)

Mine **8–12 real bug-fix commits** (prefer ones where the fix touches a *different* file than
the failing test — multi-file bugs are where a dump misses the cause and CCOS shouldn't).
For each: `git checkout` the tree **before** the fix, confirm `cargo test` is red, note the
**failing file**.

Build the two **equal-budget** contexts:

```sh
cargo test 2>&1 | tee red.txt          # capture the red output
python3 scripts/ccos_resolution_context.py <crate>/src <failing_file>.rs \
    --budget 4096 --cargo-output red.txt --outdir corpus_J/<bug-hash>
# -> context_ccos.txt and context_baseline.txt, both ~4096 tokens
```

`context_ccos.txt` = the causal region around the failing file. `context_baseline.txt` = the
failing file + siblings dumped and truncated to the **same** budget (the naive agent).

Then, for **each** context, with the **same** prompt to your local LLM:

> "Here is the context. The test in `<failing_file>` fails. Return a unified diff that fixes
> the bug."

apply the diff, run `cargo test`, and record. **Fill this grid:**

| bug (hash) | failing file | cause file | same file? | resolved_ccos | resolved_baseline | tokens_ccos | tokens_baseline |
| ---------- | ------------ | ---------- | ---------- | ------------- | ----------------- | ----------- | --------------- |

**The honest hypothesis:** equal *resolution* on single-file bugs (the dump has the cause
too), but a **CCOS win on multi-file bugs** — where the baseline truncates away the cause
file and CCOS's region still carries it. If resolution is *equal everywhere* at equal budget,
that is a real (negative-for-the-thesis) result and we report it. If CCOS resolves multi-file
bugs the dump can't, that is the sufficiency evidence the paper needs.

## Bring back

- `corpus_I/` (the JSON files + your filled Campaign I grid).
- `corpus_J/` (per bug: the two context files, the model's two diffs, the two `cargo test`
  results, and the filled grid). Keep each bug's commit hash in the dir name.

Then `tar czf corpus_handoff.tar.gz corpus_I corpus_J` and send it — we analyse together.
Note your model id and the Jetson power mode (`scripts/jetson_repro_env.sh`) so the numbers
are reproducible.

## Campaign J round 2 — what the first run showed, and how to scale it

Round 1 (qwen3-coder:30b) gave the first **sufficiency** evidence: on multi-file bugs where
the cause *value* lives in a budget-truncated file, **CCOS resolved 3/3 where the equal-budget
dump failed** (JM2 1/3, JM3 0/3) — CCOS patched the cause file, the baseline hacked the
symptom or invented a wrong file. Single-file / guessable bugs were parity (expected). See the
"Suffisance (Q7)" table in `docs/FIELD_CAMPAIGN_H.md`. Two things to fix and scale:

### Use the ground-truth grader (don't infer resolution from the patched file)

Round 1 mis-scored a *passing* CCOS fix as unresolved. Grade from `cargo test`:

```sh
# after applying the model's diff to the crate (a git repo):
python3 scripts/ccos_grade.py <crate_dir> --bug JM3 --context ccos --tokens <n> \
    --out corpus_J/JM3/ccos_result.json
# resolved iff cargo test ran >=1 test and none failed; records patched files via git diff
```

### Verified controlled recipe (decisive, and a symptom-hack can't pass it)

A crate where the symptom file is padded past the budget and the cause is a constant in a dep,
**plus a test that asserts the cause directly** so a local hack on the symptom fails it. Note
the `i32` pad return — `u8` overflows past 255 and won't compile.

```sh
cargo new --lib jm && cd jm
printf 'pub const MIN_SCORE: f64 = 0.0;   // ROOT CAUSE: should be 0.5\npub fn is_relevant(s: f64) -> bool { s >= MIN_SCORE }\n' > src/filter.rs
printf 'use crate::filter;\npub fn keep(scores: &[f64]) -> usize { scores.iter().filter(|s| filter::is_relevant(**s)).count() }\n' > src/reader.rs
python3 -c "open('src/reader.rs','a').write(''.join('pub fn pad%d() -> i32 { let _x = %d; _x }\n'%(i,i) for i in range(400)))"
cat > src/lib.rs <<'RS'
pub mod filter; pub mod reader;
#[cfg(test)] mod tests {
    use crate::{filter, reader};
    #[test] fn keeps()    { assert_eq!(reader::keep(&[0.2, 0.6, 0.9]), 2); }   // fails if MIN_SCORE=0.0
    #[test] fn boundary() { assert!(!filter::is_relevant(0.49)); }             // fails if MIN_SCORE=0.0
    #[test] fn cause()    { assert_eq!(filter::MIN_SCORE, 0.5); }              // only a ROOT fix passes this
}
RS
git add -A && git commit -qm bug
cargo test > red.txt 2>&1                                   # RED
python3 ~/CCOS/scripts/ccos_resolution_context.py src reader.rs --budget 2048 \
    --cargo-output red.txt --ccos ~/CCOS/target/release/ccos --outdir corpus_J/JR1
# feed each context to each model -> apply -> ccos_grade.py
```

### Scale it (this is what turns a demo into a result)

- **Real bugs, not just synthetic.** Mine 5–8 real `bat`/`ripgrep`/`fd` fix-commits where the
  fix file ≠ the failing-test file and the symptom file is large. Same two-context protocol.
- **2–3 models** (e.g. qwen3-coder:30b, deepseek-coder, a smaller one) — does the CCOS win
  hold as the model shrinks? (Hypothesis: the win *grows* on weaker models, which can't
  guess the missing cause.)
- Fill, per (bug × model × context): `resolved, tests_passed/failed, patched_file, tokens`.

The decisive line stays: **does CCOS resolve multi-file bugs the equal-budget dump can't?**
Round 1 says yes on controlled bugs; round 2 says whether it holds on real ones and across
models.
