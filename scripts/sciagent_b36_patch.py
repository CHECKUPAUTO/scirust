#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scirust-sciagent/src/train/dataset.rs"
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
EXAMPLE = ROOT / "scirust-sciagent/examples/cuda_pretrain.rs"
EVAL = ROOT / "scirust-sciagent/examples/cuda_eval.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:180]!r}")
    return text.replace(old, new, count)


def replace_fn(text: str, marker: str, new_src: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing {marker!r}")
    brace = text.find("{", start)
    depth = 0
    in_str = False
    esc = False
    for i in range(brace, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[:start] + new_src.rstrip() + text[i + 1 :]
    raise SystemExit(f"unterminated {marker!r}")


DATA_HELPERS = r'''
/// Version of the deterministic train/validation window split. Persisted in exact
/// optimizer sidecars so a future split-policy change cannot silently resume a run
/// on a different data trajectory.
pub const WINDOW_SPLIT_VERSION: u32 = 1;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Deterministically partition non-overlapping `seq_len` windows across the entire
/// token stream. Validation windows are selected by hash, not by corpus position, so
/// a sorted crates.io corpus cannot turn the held-out set into an alphabetic tail.
/// Returned values are token start offsets. Validation starts are independently
/// hash-ordered so a bounded eval sample spans the whole corpus rather than its head.
pub fn distributed_window_split(
    token_len: usize,
    seq_len: usize,
    val_frac: f32,
) -> (Vec<usize>, Vec<usize>) {
    if seq_len == 0 || token_len <= seq_len {
        return (Vec::new(), Vec::new());
    }
    let n_windows = token_len.saturating_sub(1) / seq_len;
    let frac = val_frac.clamp(0.0, 0.5) as f64;
    let threshold = (frac * u64::MAX as f64) as u64;
    let mut train = Vec::with_capacity(n_windows);
    let mut val = Vec::with_capacity(((n_windows as f64) * frac) as usize + 1);
    for i in 0..n_windows {
        let h = splitmix64((i as u64) ^ 0x5641_4C5F_5350_4C49);
        if frac > 0.0 && h <= threshold {
            val.push(i * seq_len);
        } else {
            train.push(i * seq_len);
        }
    }
    // Tiny test corpora can hash to an empty side. Keep the contract usable while
    // preserving disjointness; production corpora have millions of windows.
    if train.is_empty() && !val.is_empty() {
        train.push(val.pop().expect("non-empty val"));
    }
    if frac > 0.0 && val.is_empty() && train.len() > 1 {
        val.push(train.pop().expect("non-empty train"));
    }
    val.sort_by_key(|&start| splitmix64((start / seq_len) as u64 ^ 0x4556_414C_4F52_4445));
    (train, val)
}

'''


def patch_dataset(text: str) -> str:
    if "WINDOW_SPLIT_VERSION" in text:
        raise SystemExit("dataset already B36 patched")
    marker = "/// Deterministic FNV-1a fingerprint of a token stream, hashing each u32 in\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing data helper insertion point")
    text = text[:pos] + DATA_HELPERS + text[pos:]
    insert = r'''

    #[test]
    fn distributed_split_is_deterministic_disjoint_and_spread() {
        let (train1, val1) = distributed_window_split(100_001, 100, 0.10);
        let (train2, val2) = distributed_window_split(100_001, 100, 0.10);
        assert_eq!(train1, train2);
        assert_eq!(val1, val2);
        assert!(!train1.is_empty() && !val1.is_empty());
        let train_set: std::collections::HashSet<_> = train1.iter().copied().collect();
        assert!(val1.iter().all(|v| !train_set.contains(v)));
        assert!(val1.iter().all(|v| v % 100 == 0));
        // Hash-ordering means the first bounded eval windows are not just a prefix.
        let first_ten = &val1[..10.min(val1.len())];
        assert!(first_ten.iter().copied().max().unwrap_or(0) > 20_000);
    }
'''
    end = text.rfind("\n}")
    if end < 0:
        raise SystemExit("missing dataset tests module end")
    return text[:end] + insert + text[end:]


MODEL_IMPORT = '''use crate::train::checkpoint::{CheckpointMeta, save_checkpoint};\nuse crate::train::scheduler::WarmupCosineSchedule;\n'''
MODEL_IMPORT_NEW = '''use crate::train::checkpoint::{CheckpointMeta, save_checkpoint};\nuse crate::train::dataset::{WINDOW_SPLIT_VERSION, distributed_window_split};\nuse crate::train::scheduler::WarmupCosineSchedule;\n'''

EVAL_WINDOWS_MODEL = r'''
    /// Mean CE over explicit non-overlapping window starts. Used by the distributed
    /// held-out protocol so evaluation samples the full corpus deterministically.
    pub fn eval_loss_windows(
        &self,
        tokens: &[u32],
        seq_len: usize,
        starts: &[usize],
        max_windows: usize,
    ) -> f32 {
        let s = seq_len;
        let mut total = 0.0f64;
        let mut count = 0usize;
        for &start in starts.iter().take(max_windows.max(1)) {
            if start + s >= tokens.len() {
                continue;
            }
            let inputs = &tokens[start..start + s];
            let targets = &tokens[start + 1..start + s + 1];
            let logits = self.forward_resident(inputs);
            total += self.chain.cross_entropy_loss(&logits, targets) as f64;
            count += 1;
        }
        if count == 0 { f32::NAN } else { (total / count as f64) as f32 }
    }

'''

TRAINER_EVAL_WINDOWS = r'''
    fn eval_loss_windows(
        &self,
        tokens: &[u32],
        seq_len: usize,
        starts: &[usize],
        max_windows: usize,
    ) -> f32 {
        self.model
            .eval_loss_windows(tokens, seq_len, starts, max_windows)
    }

'''


def patch_model(text: str) -> str:
    if "split_version" in text:
        raise SystemExit("model already B36 patched")
    text = must_replace(text, MODEL_IMPORT, MODEL_IMPORT_NEW)
    text = must_replace(
        text,
        '''    pub corpus_hash: Option<u64>,\n}\n''',
        '''    pub corpus_hash: Option<u64>,\n    pub split_version: Option<u32>,\n}\n''',
        1,
    )
    text = must_replace(text, '            "version": 2,\n', '            "version": 3,\n')
    text = must_replace(
        text,
        '''            "corpus_hash": cfg.corpus_hash,\n        });\n''',
        '''            "corpus_hash": cfg.corpus_hash,\n            "split_version": WINDOW_SPLIT_VERSION,\n        });\n''')
    text = must_replace(
        text,
        '''        if version != 1 && version != 2\n''',
        '''        if !(1..=3).contains(&version)\n''')
    text = must_replace(
        text,
        '''            corpus_hash: if version >= 2\n            {\n                optional_u64("corpus_hash")\n            }\n            else\n            {\n                None\n            },\n        }))\n''',
        '''            corpus_hash: if version >= 2\n            {\n                optional_u64("corpus_hash")\n            }\n            else\n            {\n                None\n            },\n            split_version: if version >= 3\n            {\n                optional_u64("split_version").map(|v| v as u32)\n            }\n            else\n            {\n                None\n            },\n        }))\n''')
    # Add plain-model explicit-window eval immediately after existing eval_loss.
    marker = "    /// Autoregressive generation from `prompt`, appending up to `max_new` tokens.\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing CudaModel eval insertion point")
    text = text[:pos] + EVAL_WINDOWS_MODEL + text[pos:]
    # Trainer wrapper before forward method.
    marker2 = "    /// Forward `tokens → logits` on the (possibly trained) bf16 model — a thin\n"
    pos2 = text.find(marker2)
    if pos2 < 0:
        raise SystemExit("missing trainer eval insertion point")
    text = text[:pos2] + TRAINER_EVAL_WINDOWS + text[pos2:]

    old_split = '''        let val_len = ((tokens.len() as f32 * cfg.val_frac.max(0.0)) as usize)\n            .min(tokens.len().saturating_sub(s + 1));\n        let (train_tokens, val_tokens): (&[u32], &[u32]) = if val_len > s + 1\n        {\n            let cut = tokens.len() - val_len;\n            (&tokens[..cut], &tokens[cut..])\n        }\n        else\n        {\n            (tokens, &[])\n        };\n        if !val_tokens.is_empty()\n        {\n            println!(\n                "held-out validation: {} tokens ({:.0}% tail)\\n",\n                val_tokens.len(),\n                cfg.val_frac * 100.0\n            );\n        }\n\n        let schedule =\n            WarmupCosineSchedule::new(cfg.base_lr, cfg.min_lr, cfg.warmup_steps, cfg.total_steps);\n        let mut step = cfg.start_step;\n        let n_windows = train_tokens.len().saturating_sub(1) / s;\n        if n_windows == 0\n        {\n            return losses;\n        }\n        let mut order: Vec<usize> = (0..n_windows).collect();\n'''
    new_split = '''        let (mut order, val_windows) = distributed_window_split(tokens.len(), s, cfg.val_frac);\n        let n_windows = order.len();\n        if n_windows == 0\n        {\n            return losses;\n        }\n        if !val_windows.is_empty()\n        {\n            println!(\n                "held-out validation: {} distributed windows ({:.1}% target, split-v{})\\n",\n                val_windows.len(),\n                cfg.val_frac * 100.0,\n                WINDOW_SPLIT_VERSION\n            );\n        }\n\n        let schedule =\n            WarmupCosineSchedule::new(cfg.base_lr, cfg.min_lr, cfg.warmup_steps, cfg.total_steps);\n        let mut step = cfg.start_step;\n'''
    text = must_replace(text, old_split, new_split)
    text = must_replace(
        text,
        '''                let start = order[wi] * s;\n                wi += 1;\n                packed_inputs.extend_from_slice(&train_tokens[start..start + s]);\n                packed_targets.extend_from_slice(&train_tokens[start + 1..start + s + 1]);\n''',
        '''                let start = order[wi];\n                wi += 1;\n                packed_inputs.extend_from_slice(&tokens[start..start + s]);\n                packed_targets.extend_from_slice(&tokens[start + 1..start + s + 1]);\n''')
    text = must_replace(text, "                && !val_tokens.is_empty()\n", "                && !val_windows.is_empty()\n")
    text = must_replace(
        text,
        '''                let val = self.eval_loss(val_tokens, s, cfg.eval_windows);\n''',
        '''                let val = self.eval_loss_windows(tokens, s, &val_windows, cfg.eval_windows);\n''',
        1,
    )
    text = must_replace(
        text,
        '''                        if !val_tokens.is_empty()\n                        {\n                            let v = self.eval_loss(val_tokens, s, cfg.eval_windows);\n''',
        '''                        if !val_windows.is_empty()\n                        {\n                            let v = self.eval_loss_windows(tokens, s, &val_windows, cfg.eval_windows);\n''')
    return text


def patch_example(text: str) -> str:
    if "WINDOW_SPLIT_VERSION" in text:
        raise SystemExit("cuda_pretrain already B36 patched")
    text = must_replace(
        text,
        '''    ShardLoader, content_hash, source_quality, token_stream_hash,\n''',
        '''    ShardLoader, WINDOW_SPLIT_VERSION, content_hash, source_quality, token_stream_hash,\n''')
    # Exact resume must refuse old split semantics unless the operator explicitly
    # declares a non-exact experiment.
    needle = '''        if let Some(v) = saved.corpus_hash\n        {\n            if v != corpus_hash\n            {\n                mismatches.push(format!(\n                    "corpus_hash saved={v:016x} current={corpus_hash:016x}"\n                ));\n            }\n        }\n'''
    add = needle + '''        match saved.split_version\n        {\n            Some(v) if v == WINDOW_SPLIT_VERSION => {},\n            Some(v) => mismatches.push(format!(\n                "split_version saved={v} current={WINDOW_SPLIT_VERSION}"\n            )),\n            None if saved.step > 0 => mismatches.push(format!(\n                "split_version saved=legacy-tail current={WINDOW_SPLIT_VERSION}"\n            )),\n            None => {},\n        }\n'''
    text = must_replace(text, needle, add)
    return text


def patch_eval(text: str) -> str:
    if "distributed_window_split" in text:
        raise SystemExit("cuda_eval already B36 patched")
    text = must_replace(
        text,
        '''//! - `SCIAGENT_SHARDS` / `SCIAGENT_TEXT` — held-out corpus for the exact val loss +\n//!   nats/char (tail `SCIAGENT_VAL_FRAC`, default 2%). Omit to report train-only.\n''',
        '''//! - `SCIAGENT_SHARDS` / `SCIAGENT_TEXT` — corpus for the same deterministic,\n//!   distributed window holdout used by pretraining (`SCIAGENT_VAL_FRAC`, default 2%).\n''')
    text = must_replace(
        text,
        '''use scirust_sciagent::train::dataset::ShardLoader;\n''',
        '''use scirust_sciagent::train::dataset::{ShardLoader, distributed_window_split};\n''')
    # Remove tail_split helper entirely.
    start = text.find("/// Tail `frac` of `tokens` as the held-out split")
    if start < 0:
        raise SystemExit("missing tail_split helper")
    end = text.find("\nfn main()", start)
    if end < 0:
        raise SystemExit("missing main after tail_split")
    text = text[:start] + text[end + 1:]

    old_val = '''    let val_tokens: Option<Vec<u32>> = if let Ok(sd) = std::env::var("SCIAGENT_SHARDS")\n    {\n        let mut loader = ShardLoader::new();\n        match loader.load_dir(&sd)\n        {\n            Ok(_) => Some(tail_split(loader.tokens(), val_frac).to_vec()),\n            Err(e) =>\n            {\n                eprintln!("could not load shards from {sd}: {e} — skipping val metrics");\n                None\n            },\n        }\n    }\n    else if let Ok(text) = std::env::var("SCIAGENT_TEXT")\n    {\n        match std::fs::read(&text)\n        {\n            Ok(b) => Some(\n                tail_split(&b.iter().map(|&x| x as u32).collect::<Vec<_>>(), val_frac).to_vec(),\n            ),\n            Err(e) =>\n            {\n                eprintln!("could not read {text}: {e} — skipping val metrics");\n                None\n            },\n        }\n    }\n    else\n    {\n        None\n    };\n\n    if let Some(val) = &val_tokens\n    {\n        let val_loss = cm.eval_loss(val, eval_seq, eval_windows);\n        // chars/token: decode the whole val stream and count Unicode scalar values.\n        // (For a BPE token, decode(&[id]) returns its text — specials/placeholders\n        // contribute 0 chars; for a byte model each token is one byte.)\n        let total_chars: usize = match &tokenizer\n        {\n            Some(tok) => val\n                .iter()\n                .map(|&t| tok.decode(&[t as usize]).chars().count())\n                .sum(),\n            None =>\n            {\n                let bytes: Vec<u8> = val.iter().map(|&t| t as u8).collect();\n                String::from_utf8_lossy(&bytes).chars().count()\n            },\n        };\n        let chars_per_token = if val.is_empty()\n        {\n            0.0\n        }\n        else\n        {\n            total_chars as f32 / val.len() as f32\n        };\n'''
    new_val = '''    let corpus_tokens: Option<Vec<u32>> = if let Ok(sd) = std::env::var("SCIAGENT_SHARDS")\n    {\n        let mut loader = ShardLoader::new();\n        match loader.load_dir(&sd)\n        {\n            Ok(_) => Some(loader.into_tokens()),\n            Err(e) =>\n            {\n                eprintln!("could not load shards from {sd}: {e} — skipping val metrics");\n                None\n            },\n        }\n    }\n    else if let Ok(text) = std::env::var("SCIAGENT_TEXT")\n    {\n        match std::fs::read(&text)\n        {\n            Ok(b) => Some(b.into_iter().map(u32::from).collect()),\n            Err(e) =>\n            {\n                eprintln!("could not read {text}: {e} — skipping val metrics");\n                None\n            },\n        }\n    }\n    else\n    {\n        None\n    };\n\n    if let Some(corpus) = &corpus_tokens\n    {\n        let (_, val_starts) = distributed_window_split(corpus.len(), eval_seq, val_frac);\n        let selected: Vec<usize> = val_starts.iter().copied().take(eval_windows.max(1)).collect();\n        let val_loss = cm.eval_loss_windows(corpus, eval_seq, &selected, eval_windows);\n        let mut metric_tokens = Vec::with_capacity(selected.len() * eval_seq);\n        for &start in &selected\n        {\n            if start + eval_seq <= corpus.len()\n            {\n                metric_tokens.extend_from_slice(&corpus[start..start + eval_seq]);\n            }\n        }\n        // chars/token is measured over the exact distributed windows used for CE.\n        let total_chars: usize = match &tokenizer\n        {\n            Some(tok) => metric_tokens\n                .iter()\n                .map(|&t| tok.decode(&[t as usize]).chars().count())\n                .sum(),\n            None =>\n            {\n                let bytes: Vec<u8> = metric_tokens.iter().map(|&t| t as u8).collect();\n                String::from_utf8_lossy(&bytes).chars().count()\n            },\n        };\n        let chars_per_token = if metric_tokens.is_empty()\n        {\n            0.0\n        }\n        else\n        {\n            total_chars as f32 / metric_tokens.len() as f32\n        };\n'''
    text = must_replace(text, old_val, new_val)
    text = must_replace(
        text,
        '''            "val   loss (held-out {:.0}% tail): {val_loss:.4} nats/token   (perplexity {:.2})",\n''',
        '''            "val   loss (distributed {:.0}% windows): {val_loss:.4} nats/token   (perplexity {:.2})",\n''')
    text = must_replace(
        text,
        '''            val.len()\n''',
        '''            metric_tokens.len()\n''',
        1,
    )
    text = must_replace(
        text,
        '''        let out = cm.generate(&prompt, max_new, &params, seed);\n''',
        '''        let out = cm.generate_cached(&prompt, max_new, &params, seed);\n''')
    return text


DATA.write_text(patch_dataset(DATA.read_text()))
MODEL.write_text(patch_model(MODEL.read_text()))
EXAMPLE.write_text(patch_example(EXAMPLE.read_text()))
EVAL.write_text(patch_eval(EVAL.read_text()))
print("patched B36 distributed validation + eval alignment")
