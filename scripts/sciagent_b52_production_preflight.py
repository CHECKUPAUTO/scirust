#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scirust-sciagent/examples/cuda_pretrain.rs"


def rep(text: str, old: str, new: str, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise SystemExit(f"expected {count}, found {actual}: {old[:180]!r}")
    return text.replace(old, new, count)


s = SRC.read_text()

# Document the new fail-closed production controls and correct the now-stale B32
# optimizer-resume statement.
old = '''//! - `SCIAGENT_SAVE=<steps>` explicitly selects a step cadence. When unset, exact
//!   recovery checkpoints default to `SCIAGENT_SAVE_HOURS=6` wall-clock hours.
//! - `SCIAGENT_MAX_TOKENS` — cap the corpus (default: **no cap**). Truncation keeps a
'''
new = '''//! - `SCIAGENT_SAVE=<steps>` explicitly selects a step cadence. When unset, exact
//!   recovery checkpoints default to `SCIAGENT_SAVE_HOURS=6` wall-clock hours.
//! - `SCIAGENT_REQUIRE_FRESH=1` — production safety gate: `SCIAGENT_CKPT` must be
//!   absent or completely empty, non-exact resume overrides are rejected, and any
//!   `SCIAGENT_MAX_TOKENS` setting is refused.
//! - `SCIAGENT_EXPECT_SHARDS`, `SCIAGENT_EXPECT_CORPUS_TOKENS`, and
//!   `SCIAGENT_EXPECT_CORPUS_FNV64` — optional fail-closed corpus identity gates.
//! - `SCIAGENT_PREFLIGHT_ONLY=1` — validate CUDA/model/corpus/fresh-checkpoint guards,
//!   print a machine-readable `SCIAGENT_PREFLIGHT_OK` line, then exit before update 1.
//! - `SCIAGENT_MAX_TOKENS` — cap the corpus (default: **no cap**). Truncation keeps a
'''
s = rep(s, old, new)

old = '''//! On start-up the newest `step_N/` in `SCIAGENT_CKPT` is loaded and training
//! resumes from it (the LR schedule continues from `meta.step`; the AdamW moments
//! restart from zero, which the warmup re-absorbs). Exit code 2 means no CUDA
//! device was found — run on the Jetson Thor.
'''
new = '''//! On normal start-up the newest `step_N/` in `SCIAGENT_CKPT` is loaded and training
//! resumes from it with the exact LR trajectory **and AdamW m/v + bias-correction
//! step** restored. Historical model-only checkpoints are the only case where moments
//! restart. Exit code 2 means no CUDA device was found — run on the Jetson Thor.
'''
s = rep(s, old, new)

# Add strict env parsing and fresh-namespace helpers beside the existing env helper.
anchor = '''fn allow_nonexact_resume() -> bool {
'''
helpers = r'''fn env_flag(key: &str) -> bool {
    matches!(
        std::env::var(key).as_deref(),
        Ok("1" | "true" | "yes" | "on")
    )
}

fn strict_env_usize(key: &str) -> Option<usize> {
    match std::env::var(key)
    {
        Ok(raw) => match raw.parse::<usize>()
        {
            Ok(value) => Some(value),
            Err(e) =>
            {
                eprintln!("invalid {key}={raw:?}: expected an unsigned integer ({e})");
                std::process::exit(1);
            },
        },
        Err(std::env::VarError::NotPresent) => None,
        Err(e) =>
        {
            eprintln!("cannot read {key}: {e}");
            std::process::exit(1);
        },
    }
}

fn strict_env_hex_u64(key: &str) -> Option<u64> {
    match std::env::var(key)
    {
        Ok(raw) =>
        {
            let value = raw.trim();
            let digits = value
                .strip_prefix("0x")
                .or_else(|| value.strip_prefix("0X"))
                .unwrap_or(value);
            match u64::from_str_radix(digits, 16)
            {
                Ok(value) => Some(value),
                Err(e) =>
                {
                    eprintln!("invalid {key}={raw:?}: expected a hexadecimal u64 ({e})");
                    std::process::exit(1);
                },
            }
        },
        Err(std::env::VarError::NotPresent) => None,
        Err(e) =>
        {
            eprintln!("cannot read {key}: {e}");
            std::process::exit(1);
        },
    }
}

fn count_bin_shards(dir: &Path) -> std::io::Result<usize> {
    let mut count = 0usize;
    for entry in std::fs::read_dir(dir)?
    {
        let entry = entry?;
        if entry.path().extension().is_some_and(|ext| ext == "bin")
        {
            count = count.checked_add(1).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "shard count overflow")
            })?;
        }
    }
    Ok(count)
}

fn require_empty_checkpoint_namespace(path: &Path) {
    match std::fs::metadata(path)
    {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
        Err(e) =>
        {
            eprintln!(
                "fresh checkpoint namespace refused: cannot inspect {}: {e}",
                path.display()
            );
            std::process::exit(1);
        },
        Ok(meta) if !meta.is_dir() =>
        {
            eprintln!(
                "fresh checkpoint namespace refused: {} exists and is not a directory",
                path.display()
            );
            std::process::exit(1);
        },
        Ok(_) => match std::fs::read_dir(path)
        {
            Err(e) =>
            {
                eprintln!(
                    "fresh checkpoint namespace refused: cannot read {}: {e}",
                    path.display()
                );
                std::process::exit(1);
            },
            Ok(mut entries) => match entries.next()
            {
                None => {},
                Some(Ok(entry)) =>
                {
                    eprintln!(
                        "fresh checkpoint namespace refused: {} is not empty (found {}). \
                         Use a new SCIAGENT_CKPT directory; do not mix semantics-v2 \
                         production weights with historical checkpoints.",
                        path.display(),
                        entry.path().display()
                    );
                    std::process::exit(1);
                },
                Some(Err(e)) =>
                {
                    eprintln!(
                        "fresh checkpoint namespace refused: cannot enumerate {}: {e}",
                        path.display()
                    );
                    std::process::exit(1);
                },
            },
        },
    }
}

'''
s = rep(s, anchor, helpers + anchor)

# Gate the checkpoint namespace before latest_checkpoint() can auto-resume anything.
old = '''fn main() {
    let ckpt_dir = std::env::var("SCIAGENT_CKPT").unwrap_or_else(|_| "checkpoints/cuda".into());

    // Config: a resumed checkpoint's own config wins; else SCIAGENT_CONFIG; else the
'''
new = '''fn main() {
    let ckpt_dir = std::env::var("SCIAGENT_CKPT").unwrap_or_else(|_| "checkpoints/cuda".into());
    let require_fresh = env_flag("SCIAGENT_REQUIRE_FRESH");
    let preflight_only = env_flag("SCIAGENT_PREFLIGHT_ONLY");
    if require_fresh
    {
        if std::env::var_os("SCIAGENT_ALLOW_NONEXACT_RESUME").is_some()
        {
            eprintln!(
                "SCIAGENT_REQUIRE_FRESH conflicts with SCIAGENT_ALLOW_NONEXACT_RESUME; \
                 fresh production runs must not enable a resume override"
            );
            std::process::exit(1);
        }
        if std::env::var_os("SCIAGENT_MAX_TOKENS").is_some()
        {
            eprintln!(
                "SCIAGENT_REQUIRE_FRESH refuses SCIAGENT_MAX_TOKENS: a production-fresh \
                 run must consume the uncapped corpus"
            );
            std::process::exit(1);
        }
        require_empty_checkpoint_namespace(Path::new(&ckpt_dir));
    }

    // Config: a resumed checkpoint's own config wins; else SCIAGENT_CONFIG; else the
'''
s = rep(s, old, new)

# Track and validate the direct .bin shard count before loading the 4.1 GB token stream.
old = '''    // Token stream: BPE shards, byte-level text, or a synthetic corpus.
    let tokens: Vec<u32> = if let Ok(dir) = std::env::var("SCIAGENT_SHARDS")
    {
        let mut loader = ShardLoader::new();
'''
new = '''    let expected_shards = strict_env_usize("SCIAGENT_EXPECT_SHARDS");
    if expected_shards.is_some() && std::env::var_os("SCIAGENT_SHARDS").is_none()
    {
        eprintln!("SCIAGENT_EXPECT_SHARDS requires SCIAGENT_SHARDS=<directory>");
        std::process::exit(1);
    }
    let mut observed_shards = None;

    // Token stream: BPE shards, byte-level text, or a synthetic corpus.
    let tokens: Vec<u32> = if let Ok(dir) = std::env::var("SCIAGENT_SHARDS")
    {
        let shard_count = match count_bin_shards(Path::new(&dir))
        {
            Ok(count) => count,
            Err(e) =>
            {
                eprintln!("failed to enumerate .bin shards in {dir}: {e}");
                std::process::exit(1);
            },
        };
        observed_shards = Some(shard_count);
        println!("BPE shard set: {shard_count} direct .bin files in {dir}");
        if let Some(expected) = expected_shards
        {
            if shard_count != expected
            {
                eprintln!(
                    "corpus shard-count mismatch: expected {expected}, observed {shard_count} in {dir}"
                );
                std::process::exit(1);
            }
        }
        let mut loader = ShardLoader::new();
'''
s = rep(s, old, new)

# Enforce token/hash identity and support a no-update production preflight exit.
old = '''    let corpus_tokens = tokens.len();
    let corpus_hash = token_stream_hash(&tokens);
    println!("corpus identity: {corpus_tokens} tokens | fnv64 {corpus_hash:016x}");
    if let Some(saved) = optimizer_resume.as_ref()
'''
new = '''    let corpus_tokens = tokens.len();
    let corpus_hash = token_stream_hash(&tokens);
    println!("corpus identity: {corpus_tokens} tokens | fnv64 {corpus_hash:016x}");
    if let Some(expected) = strict_env_usize("SCIAGENT_EXPECT_CORPUS_TOKENS")
    {
        if corpus_tokens != expected
        {
            eprintln!(
                "corpus token-count mismatch: expected {expected}, observed {corpus_tokens}"
            );
            std::process::exit(1);
        }
    }
    if let Some(expected) = strict_env_hex_u64("SCIAGENT_EXPECT_CORPUS_FNV64")
    {
        if corpus_hash != expected
        {
            eprintln!(
                "corpus hash mismatch: expected {expected:016x}, observed {corpus_hash:016x}"
            );
            std::process::exit(1);
        }
    }
    if preflight_only
    {
        println!(
            "SCIAGENT_PREFLIGHT_OK semantics={} params={} shards={} corpus_tokens={} \
             corpus_fnv64={:016x} checkpoint_dir={} fresh={}",
            SCIAGENT_MODEL_SEMANTICS_VERSION,
            params,
            observed_shards.unwrap_or(0),
            corpus_tokens,
            corpus_hash,
            ckpt_dir,
            require_fresh
        );
        println!("preflight-only requested: exiting before optimizer update 1");
        return;
    }
    if let Some(saved) = optimizer_resume.as_ref()
'''
s = rep(s, old, new)

SRC.write_text(s)
print("B52 patched cuda_pretrain production preflight guards")
