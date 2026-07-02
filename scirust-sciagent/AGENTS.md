# SCIAGENT SML — operating guide for AI agents

This document tells an autonomous agent how to **load and exploit** the
pretrained SCIAGENT small language model shipped in this crate. It is written
for machine consumption: every command is copy‑pasteable and every claim about
behaviour is exact.

## 1. What this is (and is not)

SCIAGENT is a **from‑scratch, deterministic** decoder‑only transformer
specialised on the SciRust ecosystem (SciRust + CCOS + SLHAv2 sources plus
crates.io Rust). Architecture: GQA attention, interleaved‑pair RoPE, RMSNorm,
SwiGLU, tied embeddings; a byte‑level BPE tokenizer (8192 vocab); Muon
optimizer; tape‑based reverse‑mode autodiff from `scirust-core` (no external
ML runtime, no BLAS, no GPU dependency).

The **shipped checkpoint** is the `small` config: **1.6M parameters**,
`d_model=128`, 4 layers, `max_seq_len=256`. It is a *research‑scale* model.

> **Set expectations correctly.** This model produces *plausible Rust
> syntax* — `impl` blocks, `pub fn`, generics with lifetimes, `#[cfg(...)]`
> attributes, FFI‑style constants — but **not compilable, semantically
> correct programs**. Use it for syntax‑level suggestions, tokenizer/round‑trip
> testing, determinism/attestation demos, and as a substrate for the agentic
> and conformal‑guard tooling. Do **not** wire its raw output into anything
> that assumes valid code without a compile/verify step.

## 2. Quick start (shell)

The checkpoint lives at `scirust-sciagent/checkpoints/small-20M/final`.

```bash
# Build the CLI once (native codegen recommended for the CPU forward pass).
RUSTFLAGS="-C target-cpu=native" cargo build --release -p scirust-sciagent --bin sciagent

# Greedy, with a repetition penalty (deterministic, best for "one good answer").
./target/release/sciagent --model small \
  --checkpoint scirust-sciagent/checkpoints/small-20M/final \
  --temperature 0.0 --repetition-penalty 1.3 \
  --max-tokens 60 ask "pub fn "

# Nucleus sampling (diverse, seed-reproducible).
./target/release/sciagent --model small \
  --checkpoint scirust-sciagent/checkpoints/small-20M/final \
  --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.2 --seed 3 \
  ask "impl "

# JSON output for programmatic capture.
./target/release/sciagent --model small \
  --checkpoint scirust-sciagent/checkpoints/small-20M/final \
  --temperature 0.8 --top-p 0.95 --repetition-penalty 1.3 --seed 7 --json \
  ask "let result = "
```

The binary prints a `Loading checkpoint ...` line to **stdout** before the
result. When parsing `--json`, slice from the first `{`:

```python
import json, subprocess
out = subprocess.run([...], capture_output=True, text=True).stdout
obj = json.loads(out[out.index("{"):])
print(obj["response"])
```

Subcommands: `ask <prompt>`, `chat`, `explain <path> [--lines A-B]`,
`generate <description>`, `info`.

## 3. Decoding knobs

The sampler applies, in order: **repetition penalty → temperature (or greedy)
→ top‑k → top‑p → renormalise → sample**. All stages are deterministic given
`(prompt, seed, flags)`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--temperature` | `0.0` | `0` = greedy argmax; `>0` softens the distribution |
| `--top-k` | `0` (off) | keep only the `k` most probable tokens |
| `--top-p` | `1.0` (off) | nucleus: smallest set reaching cumulative prob `p` |
| `--repetition-penalty` | `1.0` (off) | demote tokens seen in the last 64 tokens; **also works in greedy mode** |
| `--seed` | `42` | RNG seed; identical seed ⇒ identical sample |
| `--max-tokens` | `2048` | generation length cap |

**Recommended presets**

- *Deterministic best guess*: `--temperature 0.0 --repetition-penalty 1.3`.
  Greedy alone loops on this small model (`{k} {k} {k} …`); the penalty breaks it.
- *Diverse candidates*: `--temperature 0.7 --top-k 40 --top-p 0.9
  --repetition-penalty 1.2`, vary `--seed` to draw different samples.

## 4. Determinism and attestation

`replay == live`: identical `(prompt, seed, flags, checkpoint)` yields
byte‑identical output. Every inference can be recorded in a hash‑chained
attestation log (`ccos::CcosLog`) using **SHA‑256** over an unambiguous,
length‑prefixed serialization that includes the timestamp — tamper‑evident and
stable across architectures and Rust releases. Verify a chain with
`CcosLog::verify()`.

## 5. Evaluate a checkpoint

`sciagent-eval` reports deterministic held‑out cross‑entropy and perplexity
over a directory of token shards (no gradient, no shuffle):

```bash
cargo build --release -p scirust-sciagent --bin sciagent-eval
./target/release/sciagent-eval \
  --checkpoint scirust-sciagent/checkpoints/small-20M/final \
  --data-dir <dir-of-*.bin-shards> --batch-size 8 --max-batches 120
```

Build shards from any `.rs` corpus with `collect-data` (see §7).

## 6. Programmatic use (Rust)

```rust
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::generate::Generator;
use scirust_sciagent::bpe::BpeTokenizer;
use scirust_sciagent::train::checkpoint::load_checkpoint;
use std::path::Path;

let dir = Path::new("scirust-sciagent/checkpoints/small-20M/final");
let cfg = SciAgentConfig::small();          // must match the checkpoint config
let mut model = SciAgentModel::new(&cfg);
load_checkpoint(&mut model, dir).unwrap();  // loads weights (config read from meta.json)

let tok = BpeTokenizer::from_embedded().unwrap();
let prompt = tok.encode_with_special("pub fn ", true, false);

let gen = Generator::new(&cfg)
    .with_temperature(0.7)
    .with_top_k(40)
    .with_top_p(0.9)
    .with_repetition_penalty(1.2);
let ids = gen.generate(&mut model, &prompt, 60, /*seed=*/3);
println!("{}", tok.decode(&ids));
```

## 7. Reproduce / extend training

```bash
# 1. Stage a corpus of .rs files, then tokenize into shards.
./target/release/collect-data --input <dir> \
  --tokenizer scirust-sciagent/tokenizer/bpe.json \
  --output shards/ --recursive --seq-len 256

# 2. Train (gradients flow correctly; accumulation is exact).
./target/release/sciagent-train --model small --data-dir shards/ \
  --total-steps 2000 --micro-batch-size 8 --grad-accum-steps 2 \
  --warmup-steps 150 --lr 0.015 --min-lr 0.0015 --checkpoint-dir ckpt/
```

Larger configs (`350m`, `7b`) exist in `config.rs` but the CPU tape‑autodiff
forward is not practical for them without a GPU backend.

## 8. Checkpoint provenance

- **Config**: `small` (1.6M params, vocab 8192, `d_model=128`, 4 layers,
  4 heads / 2 KV heads, `d_ff=256`, `max_seq_len=256`, tied embeddings).
- **Corpus**: SciRust + CCOS + SLHAv2 sources + crates.io registry
  (~20M tokens, ~0.8 epoch, no data repetition).
- **Schedule**: 2000 Muon steps, lr 1.5e‑2 → 1.5e‑3 cosine, effective batch 16.
- **Held‑out perplexity**: ≈ 108 (generalization gap ~6%; see §9).

## 9. Measured quality

Deterministic `sciagent-eval`, 120 batches (245 760 tokens) each, batch size 8:

| Eval set | Loss (nats) | Perplexity |
|----------|------------:|-----------:|
| Training data (seen) | 4.631 | 102.6 |
| Held‑out crate `tests/`, `benches/`, `examples/` (never trained on) | 4.687 | 108.5 |

**Generalization gap ≈ 0.06 nats (~6% perplexity).** The model performs almost
identically on data it trained on and on Rust files it has never seen — it
learned the *structure* of the language rather than memorizing its corpus. (A
memorizing model shows a large gap: low seen loss, high held‑out loss.) The
absolute perplexity (~100 over real Rust, versus the 8192 uniform baseline) is
what a 1.6M‑parameter model at ~0.8 epoch reaches; scaling params/tokens is the
lever for lowering it. The single ~3.88 training‑loss figure quoted elsewhere is
the final annealed step on one batch, not a corpus average — use the numbers in
this table for comparison.

## 10. Rules for agents

1. **Always pass a checkpoint.** Without `--checkpoint`, weights are random
   and output is noise.
2. **Match the config.** Use `--model small` for this checkpoint; the config
   is also embedded in `meta.json` and read automatically on load.
3. **Never trust output as valid code.** Pipe it through `rustc`/`cargo check`
   before acting on it.
4. **Determinism is a feature.** To reproduce a result, reuse the exact
   `(prompt, seed, flags)`. To get variety, change `--seed`, not the prompt.
5. **Prefer the presets in §3.** Raw greedy loops; always add a repetition
   penalty.
