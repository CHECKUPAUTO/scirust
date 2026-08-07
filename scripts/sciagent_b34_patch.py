#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scirust-sciagent/src/train/dataset.rs"
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
EXAMPLE = ROOT / "scirust-sciagent/examples/cuda_pretrain.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:160]!r}")
    return text.replace(old, new, count)


def patch_dataset(text: str) -> str:
    if "pub fn into_tokens(self)" in text:
        raise SystemExit("dataset already B34 patched")
    text = must_replace(
        text,
        '''    pub fn from_slice(data: &[u32], seq_len: usize, vocab_size: usize) -> Self {\n        Self {\n            data: data.to_vec(),\n            position: 0,\n            seq_len,\n            vocab_size,\n        }\n    }\n''',
        '''    pub fn from_slice(data: &[u32], seq_len: usize, vocab_size: usize) -> Self {\n        Self::from_vec(data.to_vec(), seq_len, vocab_size)\n    }\n\n    /// Construct without copying an already-owned token buffer.\n    pub fn from_vec(data: Vec<u32>, seq_len: usize, vocab_size: usize) -> Self {\n        Self {\n            data,\n            position: 0,\n            seq_len,\n            vocab_size,\n        }\n    }\n''')
    text = must_replace(
        text,
        '''    pub fn load_bin<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {\n        let bytes = std::fs::read(path.as_ref())?;\n        let mut data = vec![0u32; bytes.len() / 4];\n        for (i, chunk) in bytes.as_chunks::<4>().0.iter().enumerate()\n        {\n            data[i] = u32::from_le_bytes(*chunk);\n        }\n        self.buffer = data;\n        Ok(())\n    }\n''',
        '''    pub fn load_bin<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {\n        let bytes = std::fs::read(path.as_ref())?;\n        if !bytes.len().is_multiple_of(4)\n        {\n            return Err(std::io::Error::new(\n                std::io::ErrorKind::InvalidData,\n                format!(\"{} has {} bytes, not a multiple of 4\", path.as_ref().display(), bytes.len()),\n            ));\n        }\n        let mut data = Vec::with_capacity(bytes.len() / 4);\n        for chunk in bytes.as_chunks::<4>().0\n        {\n            data.push(u32::from_le_bytes(*chunk));\n        }\n        self.buffer = data;\n        Ok(())\n    }\n''')
    text = must_replace(
        text,
        '''    pub fn load_dir<P: AsRef<Path>>(&mut self, dir: P) -> std::io::Result<()> {\n        let mut all_data = Vec::new();\n        let mut entries: Vec<_> = std::fs::read_dir(dir.as_ref())?\n            .filter_map(|e| e.ok())\n            .filter(|e| e.path().extension().is_some_and(|ext| ext == \"bin\"))\n            .collect();\n        entries.sort_by_key(|e| e.file_name());\n        for entry in &entries\n        {\n            let bytes = std::fs::read(entry.path())?;\n            for chunk in bytes.as_chunks::<4>().0\n            {\n                all_data.push(u32::from_le_bytes(*chunk));\n            }\n        }\n        self.buffer = all_data;\n        Ok(())\n    }\n''',
        '''    pub fn load_dir<P: AsRef<Path>>(&mut self, dir: P) -> std::io::Result<()> {\n        let mut entries: Vec<_> = std::fs::read_dir(dir.as_ref())?\n            .filter_map(|e| e.ok())\n            .filter(|e| e.path().extension().is_some_and(|ext| ext == \"bin\"))\n            .collect();\n        entries.sort_by_key(|e| e.file_name());\n\n        // Reserve the final token buffer once. At 1.03B tokens this avoids repeated\n        // reallocations/copies of a ~4.1 GB Vec while the shard set is loaded.\n        let mut total_bytes = 0usize;\n        for entry in &entries\n        {\n            let len = usize::try_from(entry.metadata()?.len()).map_err(|_| {\n                std::io::Error::new(std::io::ErrorKind::InvalidData, \"shard size exceeds usize\")\n            })?;\n            if !len.is_multiple_of(4)\n            {\n                return Err(std::io::Error::new(\n                    std::io::ErrorKind::InvalidData,\n                    format!(\"{} has {len} bytes, not a multiple of 4\", entry.path().display()),\n                ));\n            }\n            total_bytes = total_bytes.checked_add(len).ok_or_else(|| {\n                std::io::Error::new(std::io::ErrorKind::InvalidData, \"total shard size overflows usize\")\n            })?;\n        }\n        let mut all_data = Vec::with_capacity(total_bytes / 4);\n        for entry in &entries\n        {\n            let bytes = std::fs::read(entry.path())?;\n            for chunk in bytes.as_chunks::<4>().0\n            {\n                all_data.push(u32::from_le_bytes(*chunk));\n            }\n        }\n        self.buffer = all_data;\n        Ok(())\n    }\n''')
    text = must_replace(
        text,
        '''    pub fn tokens(&self) -> &[u32] {\n        &self.buffer\n    }\n\n    pub fn into_dataset(self, seq_len: usize, vocab_size: usize) -> PretrainDataset {\n        PretrainDataset::from_slice(&self.buffer, seq_len, vocab_size)\n    }\n''',
        '''    pub fn tokens(&self) -> &[u32] {\n        &self.buffer\n    }\n\n    /// Transfer ownership of the raw token vector without a second corpus-sized copy.\n    pub fn into_tokens(self) -> Vec<u32> {\n        self.buffer\n    }\n\n    pub fn into_dataset(self, seq_len: usize, vocab_size: usize) -> PretrainDataset {\n        PretrainDataset::from_vec(self.buffer, seq_len, vocab_size)\n    }\n''')
    marker = '''/// Heuristic corpus-quality gate: does `content` (from a file named `name`) look\n'''
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing token hash insertion point")
    helper = '''/// Deterministic FNV-1a fingerprint of a token stream, hashing each u32 in\n/// little-endian byte order. Used to fail closed when an exact optimizer/data resume\n/// points at a different corpus than the checkpoint was trained on.\npub fn token_stream_hash(tokens: &[u32]) -> u64 {\n    let mut h: u64 = 0xcbf2_9ce4_8422_2325;\n    for &token in tokens\n    {\n        for b in token.to_le_bytes()\n        {\n            h ^= b as u64;\n            h = h.wrapping_mul(0x0000_0100_0000_01b3);\n        }\n    }\n    h\n}\n\n'''
    return text[:pos] + helper + text[pos:]


def patch_model(text: str) -> str:
    if "pub corpus_hash: u64" in text:
        raise SystemExit("model already B34 patched")
    text = must_replace(
        text,
        '''    pub weight_decay: f32,\n}\n''',
        '''    pub weight_decay: f32,\n    /// Exact-data resume fields are absent on legacy B32/v1 sidecars.\n    pub seq_len: Option<usize>,\n    pub batch_size: Option<usize>,\n    pub corpus_tokens: Option<usize>,\n    pub corpus_hash: Option<u64>,\n}\n''',
        1,
    )
    text = must_replace(
        text,
        '''            \"version\": 1,\n            \"step\": self.step,\n''',
        '''            \"version\": 2,\n            \"step\": self.step,\n''',
    )
    text = must_replace(
        text,
        '''            \"weight_decay\": cfg.weight_decay,\n        });\n''',
        '''            \"weight_decay\": cfg.weight_decay,\n            \"seq_len\": cfg.seq_len,\n            \"batch_size\": cfg.batch_size,\n            \"corpus_tokens\": cfg.corpus_tokens,\n            \"corpus_hash\": cfg.corpus_hash,\n        });\n''',
    )
    text = must_replace(
        text,
        '''        if meta[\"version\"].as_u64() != Some(1)\n        {\n            return Err(format!(\n                \"unsupported optimizer checkpoint version in {}\",\n                meta_path.display()\n            ));\n        }\n''',
        '''        let version = meta[\"version\"].as_u64().unwrap_or(0);\n        if version != 1 && version != 2\n        {\n            return Err(format!(\n                \"unsupported optimizer checkpoint version {version} in {}\",\n                meta_path.display()\n            ));\n        }\n''')
    old_return = '''        Ok(Some(CudaOptimizerResume {\n            step,\n            base_lr: number(\"base_lr\")?,\n            min_lr: number(\"min_lr\")?,\n            warmup_steps: usize_field(\"warmup_steps\")?,\n            total_steps: usize_field(\"total_steps\")?,\n            betas: (beta0, beta1),\n            adam_eps: number(\"adam_eps\")?,\n            weight_decay: number(\"weight_decay\")?,\n        }))\n'''
    new_return = '''        let optional_usize = |key: &str| meta[key].as_u64().map(|x| x as usize);\n        let optional_u64 = |key: &str| meta[key].as_u64();\n        Ok(Some(CudaOptimizerResume {\n            step,\n            base_lr: number(\"base_lr\")?,\n            min_lr: number(\"min_lr\")?,\n            warmup_steps: usize_field(\"warmup_steps\")?,\n            total_steps: usize_field(\"total_steps\")?,\n            betas: (beta0, beta1),\n            adam_eps: number(\"adam_eps\")?,\n            weight_decay: number(\"weight_decay\")?,\n            seq_len: if version >= 2 { optional_usize(\"seq_len\") } else { None },\n            batch_size: if version >= 2 { optional_usize(\"batch_size\") } else { None },\n            corpus_tokens: if version >= 2 { optional_usize(\"corpus_tokens\") } else { None },\n            corpus_hash: if version >= 2 { optional_u64(\"corpus_hash\") } else { None },\n        }))\n'''
    text = must_replace(text, old_return, new_return)
    text = must_replace(
        text,
        '''        let n_windows = train_tokens.len().saturating_sub(1) / s;\n        let mut order: Vec<usize> = (0..n_windows).collect();\n        let mut epoch: u64 = 0;\n        let reshuffle = |order: &mut [usize], epoch: u64| {\n            shuffle_windows(\n                order,\n                (cfg.start_step as u64).wrapping_add(epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15)),\n            );\n        };\n        if cfg.shuffle\n        {\n            reshuffle(&mut order, epoch);\n        }\n        let mut wi = 0usize;\n''',
        '''        let n_windows = train_tokens.len().saturating_sub(1) / s;\n        if n_windows == 0\n        {\n            return losses;\n        }\n        let mut order: Vec<usize> = (0..n_windows).collect();\n        // The permutation is a pure function of the absolute epoch, never of the\n        // process invocation. Resume reconstructs both epoch and cursor from the\n        // number of sequences already consumed, so interrupted and uninterrupted\n        // runs see the exact same next windows.\n        let consumed_windows = cfg.start_step.saturating_mul(batch);\n        let mut epoch: u64 = (consumed_windows / n_windows) as u64;\n        let mut wi = consumed_windows % n_windows;\n        let reshuffle = |order: &mut [usize], epoch: u64| {\n            shuffle_windows(order, 0x5343_4941_4745_4E54u64 ^ epoch);\n        };\n        if cfg.shuffle\n        {\n            reshuffle(&mut order, epoch);\n        }\n''')
    text = must_replace(
        text,
        '''    /// Shuffle the training window order (re-shuffled deterministically each epoch).\n''',
        '''    /// Exact-resume corpus identity (B34). Zero is reserved for legacy/tests that\n    /// do not provide a production corpus fingerprint.\n    pub corpus_tokens: usize,\n    pub corpus_hash: u64,\n    /// Shuffle the training window order (re-shuffled deterministically each epoch).\n''')
    text = must_replace(
        text,
        '''            keep_last: 3,\n            shuffle: true,\n''',
        '''            keep_last: 3,\n            corpus_tokens: 0,\n            corpus_hash: 0,\n            shuffle: true,\n''')
    return text


def patch_example(text: str) -> str:
    if "SCIAGENT_ALLOW_NONEXACT_RESUME" in text:
        raise SystemExit("example already B34 patched")
    text = must_replace(
        text,
        '''use scirust_sciagent::train::dataset::{ShardLoader, content_hash, source_quality};\n''',
        '''use scirust_sciagent::train::dataset::{\n    ShardLoader, content_hash, source_quality, token_stream_hash,\n};\n''')
    text = must_replace(
        text,
        '''    let seq_len = env_usize("SCIAGENT_SEQ", 128).min(config.max_seq_len);\n    let batch_size = env_usize("SCIAGENT_BATCH", 1).max(1);\n''',
        '''    // Large production runs default to 512: the historical 128-token default\n    // rarely held a complete Rust function and contributed to the syntax wall. Exact\n    // B34 resumes inherit their saved B/T unless the operator explicitly overrides.\n    let default_seq = if config.d_model > 256 { 512 } else { 128 };\n    let explicit_seq = std::env::var("SCIAGENT_SEQ")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok());\n    let seq_len = explicit_seq\n        .or_else(|| optimizer_resume.as_ref().and_then(|s| s.seq_len))\n        .unwrap_or(default_seq)\n        .min(config.max_seq_len);\n    let explicit_batch = std::env::var("SCIAGENT_BATCH")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok());\n    let batch_size = explicit_batch\n        .or_else(|| optimizer_resume.as_ref().and_then(|s| s.batch_size))\n        .unwrap_or(1)\n        .max(1);\n''')
    old_shards = '''        let raw = loader.tokens();\n        let maxid = raw.iter().copied().max().unwrap_or(0) as usize;\n        if maxid >= config.vocab_size\n        {\n            eprintln!(\n                "shard token id {maxid} >= config vocab_size {}: these shards were tokenised for a\\n\\\n                 different vocab. Set SCIAGENT_CONFIG to the matching preset (e.g. 350m), or\\n\\\n                 re-tokenise with collect-data.",\n                config.vocab_size\n            );\n            std::process::exit(1);\n        }\n        if max_tokens < raw.len()\n        {\n            println!(\n                "streaming {} of {} tokens from BPE shards in {dir} \\\n                 (TRUNCATED to {:.1}% by SCIAGENT_MAX_TOKENS={max_tokens} — the shard walk is\\n\\\n                 alphabetical, so a truncated corpus is a *prefix*, not a sample)",\n                max_tokens,\n                raw.len(),\n                100.0 * max_tokens as f64 / raw.len() as f64\n            );\n        }\n        else\n        {\n            println!("streaming {} tokens from BPE shards in {dir}", raw.len());\n        }\n        raw.iter().take(max_tokens).copied().collect()\n'''
    new_shards = '''        let mut raw = loader.into_tokens();\n        let original_len = raw.len();\n        let maxid = raw.iter().copied().max().unwrap_or(0) as usize;\n        if maxid >= config.vocab_size\n        {\n            eprintln!(\n                "shard token id {maxid} >= config vocab_size {}: these shards were tokenised for a\\n\\\n                 different vocab. Set SCIAGENT_CONFIG to the matching preset (e.g. 350m), or\\n\\\n                 re-tokenise with collect-data.",\n                config.vocab_size\n            );\n            std::process::exit(1);\n        }\n        if max_tokens < original_len\n        {\n            println!(\n                "streaming {} of {} tokens from BPE shards in {dir} \\\n                 (TRUNCATED to {:.1}% by SCIAGENT_MAX_TOKENS={max_tokens} — the shard walk is\\n\\\n                 alphabetical, so a truncated corpus is a *prefix*, not a sample)",\n                max_tokens,\n                original_len,\n                100.0 * max_tokens as f64 / original_len as f64\n            );\n            raw.truncate(max_tokens);\n        }\n        else\n        {\n            println!("streaming {} tokens from BPE shards in {dir}", original_len);\n        }\n        raw\n'''
    text = must_replace(text, old_shards, new_shards)
    marker = '''    // Exact B32 resumes inherit the saved optimizer/LR trajectory by default. An\n'''
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing post-token insertion point")
    check = '''    let corpus_tokens = tokens.len();\n    let corpus_hash = token_stream_hash(&tokens);\n    println!(\n        "corpus identity: {corpus_tokens} tokens | fnv64 {corpus_hash:016x}"\n    );\n    if let Some(saved) = optimizer_resume.as_ref()\n    {\n        let mut mismatches = Vec::new();\n        if let Some(v) = saved.seq_len\n        {\n            if v != seq_len { mismatches.push(format!("seq_len saved={v} current={seq_len}")); }\n        }\n        if let Some(v) = saved.batch_size\n        {\n            if v != batch_size { mismatches.push(format!("batch saved={v} current={batch_size}")); }\n        }\n        if let Some(v) = saved.corpus_tokens\n        {\n            if v != corpus_tokens { mismatches.push(format!("corpus_tokens saved={v} current={corpus_tokens}")); }\n        }\n        if let Some(v) = saved.corpus_hash\n        {\n            if v != corpus_hash { mismatches.push(format!("corpus_hash saved={v:016x} current={corpus_hash:016x}")); }\n        }\n        if !mismatches.is_empty()\n        {\n            let allow = matches!(\n                std::env::var("SCIAGENT_ALLOW_NONEXACT_RESUME").as_deref(),\n                Ok("1" | "true")\n            );\n            if !allow\n            {\n                eprintln!(\n                    "exact resume refused: {}. Set SCIAGENT_ALLOW_NONEXACT_RESUME=1 only for an intentional branch experiment.",\n                    mismatches.join(", ")\n                );\n                std::process::exit(1);\n            }\n            eprintln!("WARNING: non-exact resume explicitly allowed: {}", mismatches.join(", "));\n        }\n    }\n\n'''
    text = text[:pos] + check + text[pos:]
    text = must_replace(
        text,
        '''        keep_last,\n        shuffle,\n        ..Default::default()\n''',
        '''        keep_last,\n        corpus_tokens,\n        corpus_hash,\n        shuffle,\n        ..Default::default()\n''')
    return text


DATA.write_text(patch_dataset(DATA.read_text()))
MODEL.write_text(patch_model(MODEL.read_text()))
EXAMPLE.write_text(patch_example(EXAMPLE.read_text()))
print("patched B34 exact data resume + zero-copy shard ownership")
