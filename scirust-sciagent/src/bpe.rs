use std::collections::{BTreeMap, HashMap};
use std::fs;

const SPECIAL_TOKENS: &[(&str, usize)] = &[("<pad>", 0), ("<bos>", 1), ("<eos>", 2), ("<unk>", 3)];

fn byte_to_str(b: u8) -> String {
    let single = vec![b];
    String::from_utf8(single).unwrap_or_else(|_| format!("<{b}>"))
}

/// Reversible byte → "unit char" map (GPT-2-style, simplified) for the **v2**
/// byte-level scheme. Every byte value maps to a distinct Unicode scalar, so the
/// base vocab is representable as ordinary keys AND `decode` is exactly invertible:
/// no byte can collapse to a `<NNN>` placeholder, so multibyte UTF-8 (`· — é ✅ 世`)
/// round-trips through encode → decode unchanged. Bytes `0..=127` map to
/// themselves (ASCII — so a v2 tokenizer's ASCII tokens are byte-for-byte identical
/// to the legacy `byte_to_str` representation); bytes `128..=255` map into the
/// `256..=383` block, which never collides with ASCII.
fn byte_to_unit(b: u8) -> char {
    let u = if b < 128
    {
        b as u32
    }
    else
    {
        256 + (b as u32 - 128)
    };
    char::from_u32(u).expect("byte-unit codepoint is always valid")
}

/// Inverse of [`byte_to_unit`]: a unit char back to its byte, or `None` if `c` is
/// not a unit char (e.g. a character from a legacy special-token string).
fn unit_to_byte(c: char) -> Option<u8> {
    let u = c as u32;
    if u < 128
    {
        Some(u as u8)
    }
    else if (256..384).contains(&u)
    {
        Some((u - 256 + 128) as u8)
    }
    else
    {
        None
    }
}

pub struct BpeTrainer {
    vocab_size: usize,
    min_frequency: u32,
}

impl BpeTrainer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            min_frequency: 2,
        }
    }

    pub fn min_frequency(mut self, f: u32) -> Self {
        self.min_frequency = f;
        self
    }

    pub fn train(&self, texts: &[String]) -> BpeTokenizer {
        let mut id = SPECIAL_TOKENS.len();

        // Étape 1 : construire le vocabulaire de base (bytes)
        let mut vocab: BTreeMap<String, usize> = BTreeMap::new();
        let mut rev: Vec<String> = Vec::new();
        for (tok, idx) in SPECIAL_TOKENS
        {
            vocab.insert(tok.to_string(), *idx);
            rev.push(tok.to_string());
        }

        // Base vocab: ALL 256 byte values, each as its reversible unit char (v2
        // scheme). Adding every byte — not just those the corpus happens to contain —
        // guarantees `encode` never emits `<unk>` for a byte and `decode` is fully
        // reversible, independent of corpus coverage. The order is byte-ascending, so
        // the base ids are deterministic across corpora.
        for b in 0u8..=255
        {
            let s = byte_to_unit(b).to_string();
            if !vocab.contains_key(&s)
            {
                vocab.insert(s.clone(), id);
                rev.push(s);
                id += 1;
            }
        }

        // Étape 2 : tokeniser chaque texte en séquence de bytes (unit chars)
        let mut corpus: Vec<Vec<usize>> = Vec::with_capacity(texts.len());
        for t in texts
        {
            let ids: Vec<usize> = t
                .bytes()
                .map(|b| vocab[&byte_to_unit(b).to_string()])
                .collect();
            corpus.push(ids);
        }

        // Étape 3 : itérer les merges BPE
        // On regroupe les merges par lots pour réduire le nombre de passes
        let mut merges: Vec<(usize, usize, usize)> = Vec::new();
        let merge_batch_size = std::cmp::min(2000, (self.vocab_size.saturating_sub(id)) / 4 + 1);

        while id < self.vocab_size
        {
            let mut pair_counts: HashMap<(usize, usize), u64> = HashMap::new();
            for tokens in &corpus
            {
                for w in tokens.windows(2)
                {
                    if w[0] != 0 && w[1] != 0
                    {
                        *pair_counts.entry((w[0], w[1])).or_insert(0) += 1;
                    }
                }
            }

            // Prendre les N meilleures paires
            let mut ranked: Vec<((usize, usize), u64)> = pair_counts
                .into_iter()
                .filter(|&(_, count)| count >= self.min_frequency as u64)
                .collect();
            // Deterministic order: by count DESC, ties broken by the pair ids ASC.
            // Without the tiebreak, `sort_unstable` leaves equal-count pairs in an
            // arbitrary order, so the same corpus can yield a different tokenizer
            // across runs — meaning a lost/corrupt tokenizer.json can't be
            // regenerated to stay compatible with already-tokenised shards.
            ranked.sort_unstable_by(|&(pa, ca), &(pb, cb)| cb.cmp(&ca).then(pa.cmp(&pb)));
            let batch: Vec<(usize, usize)> = ranked
                .into_iter()
                .take(merge_batch_size)
                .map(|((a, b), _)| (a, b))
                .collect();

            if batch.is_empty()
            {
                break;
            }

            for &(pa, pb) in &batch
            {
                if id >= self.vocab_size
                {
                    break;
                }
                merges.push((pa, pb, id));

                let new_token = format!("{}{}", rev[pa], rev[pb]);
                vocab.insert(new_token.clone(), id);
                rev.push(new_token);
                id += 1;
            }

            // Appliquer tous les merges du lot
            // Le premier merge du lot a pour id (vocab_size_initial + num_merges_avant_batch)
            let base_id = id - batch.len();
            let merge_map: HashMap<(usize, usize), usize> = batch
                .iter()
                .enumerate()
                .map(|(i, &(a, b))| ((a, b), base_id + i))
                .collect();

            for tokens in &mut corpus
            {
                let mut out = Vec::with_capacity(tokens.len());
                let mut i = 0;
                while i < tokens.len()
                {
                    if i + 1 < tokens.len()
                    {
                        let key = (tokens[i], tokens[i + 1]);
                        if let Some(&new_id) = merge_map.get(&key)
                        {
                            out.push(new_id);
                            i += 2;
                            continue;
                        }
                    }
                    out.push(tokens[i]);
                    i += 1;
                }
                *tokens = out;
            }

            if id.is_multiple_of(1000)
            {
                eprintln!("BPE: {}/{} tokens", id, self.vocab_size);
            }
        }

        eprintln!(
            "BPE training complete: {} tokens, {} merges",
            vocab.len(),
            merges.len()
        );

        BpeTokenizer {
            vocab,
            rev,
            merges,
            reversible: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    vocab: BTreeMap<String, usize>,
    rev: Vec<String>,
    merges: Vec<(usize, usize, usize)>,
    /// `true` for the **v2** reversible byte-level scheme (`byte_to_unit`), `false`
    /// for a legacy tokenizer (`byte_to_str` + `<NNN>` placeholders, e.g. the
    /// embedded `bpe.json`). Selects both the byte→key map in `encode` and the
    /// `decode` path, so old and new tokenizers stay bit-for-bit compatible with the
    /// data they were trained on.
    reversible: bool,
}

impl BpeTokenizer {
    pub fn new(
        vocab: BTreeMap<String, usize>,
        rev: Vec<String>,
        merges: Vec<(usize, usize, usize)>,
    ) -> Self {
        // Infer the scheme: a v2 vocab always contains every 256-byte unit char, so
        // the byte-128 unit ('Ā', U+0100) is present iff this is a v2 tokenizer.
        let reversible = vocab.contains_key(&byte_to_unit(128).to_string());
        Self {
            vocab,
            rev,
            merges,
            reversible,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn special_id(&self, name: &str) -> usize {
        *self.vocab.get(name).unwrap_or(&3)
    }

    /// The base-vocab key for a single byte under this tokenizer's scheme: the
    /// reversible unit char (v2) or the legacy `byte_to_str` (`<NNN>` for non-ASCII).
    fn byte_key(&self, b: u8) -> String {
        if self.reversible
        {
            byte_to_unit(b).to_string()
        }
        else
        {
            byte_to_str(b)
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<usize> = text
            .bytes()
            .map(|b| *self.vocab.get(&self.byte_key(b)).unwrap_or(&3))
            .collect();

        if ids.is_empty()
        {
            return ids;
        }

        // Appliquer les merges gloutonnement (optimisé avec lookup table)
        let merge_lookup: HashMap<(usize, usize), usize> = self
            .merges
            .iter()
            .map(|&(a, b, new_id)| ((a, b), new_id))
            .collect();

        if merge_lookup.is_empty()
        {
            return ids;
        }

        // Single-pass BPE merge using output buffer (avoids O(n²) remove())
        loop
        {
            let mut out = Vec::with_capacity(ids.len());
            let mut any_merged = false;
            let mut i = 0;
            while i < ids.len()
            {
                if i + 1 < ids.len() && ids[i] != 0 && ids[i + 1] != 0
                {
                    if let Some(&new_id) = merge_lookup.get(&(ids[i], ids[i + 1]))
                    {
                        out.push(new_id);
                        i += 2;
                        any_merged = true;
                        continue;
                    }
                }
                out.push(ids[i]);
                i += 1;
            }
            if !any_merged
            {
                break;
            }
            ids = out;
        }

        ids
    }

    pub fn encode_with_special(
        &self,
        text: &str,
        prepend_bos: bool,
        append_eos: bool,
    ) -> Vec<usize> {
        let mut ids = Vec::new();
        if prepend_bos
        {
            ids.push(self.special_id("<bos>"));
        }
        ids.extend(self.encode(text));
        if append_eos
        {
            ids.push(self.special_id("<eos>"));
        }
        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        if self.reversible
        {
            // v2: structural + fully reversible. Each non-special token is a string of
            // unit chars, each standing for exactly one byte; concatenate all those
            // bytes and UTF-8-decode ONCE at the end — so a multibyte char split
            // across two tokens reassembles correctly (the whole reason `<NNN>` is
            // gone). Named specials (ids `0..len`) carry no bytes and are skipped by
            // id, not by string content, so a learned merge that happens to look like
            // "<200>" is emitted as the real text it is.
            let mut bytes: Vec<u8> = Vec::new();
            for &id in ids
            {
                if id < SPECIAL_TOKENS.len()
                {
                    continue;
                }
                if let Some(s) = self.rev.get(id)
                {
                    for ch in s.chars()
                    {
                        if let Some(b) = unit_to_byte(ch)
                        {
                            bytes.push(b);
                        }
                    }
                }
            }
            return String::from_utf8_lossy(&bytes).into_owned();
        }

        // Legacy: concatenate token strings, skipping specials + `<NNN>` placeholders.
        // A blanket `starts_with('<')` also swallowed every learned merge that BEGINS
        // with a literal '<' ("<T", "< ", "<<") — which a Rust corpus is full of
        // (generics, comparisons, shifts) — silently deleting '<' from decoded code,
        // so `is_non_text_token` matches only the exact `<NNN>` byte-placeholder shape.
        let mut out = String::new();
        for &id in ids
        {
            if id < self.rev.len()
            {
                let s = &self.rev[id];
                if !Self::is_non_text_token(s)
                {
                    out.push_str(s);
                }
            }
        }
        out
    }

    /// True for tokens that carry no decodable text: the named special tokens
    /// and the `<NNN>` placeholders minted for non-UTF-8 bytes.
    fn is_non_text_token(s: &str) -> bool {
        if SPECIAL_TOKENS.iter().any(|(tok, _)| *tok == s)
        {
            return true;
        }
        // "<200>"-style byte placeholders: '<' + digits + '>'.
        s.len() >= 3
            && s.starts_with('<')
            && s.ends_with('>')
            && s[1..s.len() - 1].bytes().all(|b| b.is_ascii_digit())
    }

    #[allow(dead_code)]
    fn find_merge(&self, left: usize, right: usize) -> Option<usize> {
        for &(l, r, new_id) in &self.merges
        {
            if l == left && r == right
            {
                return Some(new_id);
            }
        }
        None
    }

    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::json!({
            // Scheme tag: v2 tokenizers decode reversibly (no `<NNN>`); its absence
            // means a legacy tokenizer, so a re-loaded file keeps its original decode.
            "version": if self.reversible { "byte_level_v2" } else { "legacy_v1" },
            "vocab": self.vocab,
            "merges": self.merges.iter().map(|(a, b, c)| format!("{a} {b} {c}")).collect::<Vec<_>>(),
        });
        fs::write(path, serde_json::to_string_pretty(&json)?)?;
        Ok(())
    }

    pub fn load_json(path: &str) -> std::io::Result<Self> {
        let s = fs::read_to_string(path)?;
        Self::from_json_str(&s)
    }

    pub fn from_embedded() -> std::io::Result<Self> {
        let bpe_json = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/tokenizer/bpe.json"));
        let s = std::str::from_utf8(bpe_json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Self::from_json_str(s)
    }

    /// Parse a tokenizer from its JSON text. The `"version"` tag selects the decode
    /// scheme: `"byte_level_v2"` → reversible byte-level; anything else (including a
    /// missing tag, e.g. the legacy embedded `bpe.json`) → legacy `<NNN>` decode.
    fn from_json_str(s: &str) -> std::io::Result<Self> {
        let json: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let vocab: BTreeMap<String, usize> = serde_json::from_value(json["vocab"].clone())?;
        let rev: Vec<String> = {
            let mut v = vec![String::new(); vocab.len()];
            for (s, &id) in &vocab
            {
                if id < v.len()
                {
                    v[id] = s.clone();
                }
            }
            v
        };
        let merges: Vec<(usize, usize, usize)> = json["merges"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let parts: Vec<&str> = m.as_str()?.split_whitespace().collect();
                        if parts.len() == 3
                        {
                            Some((
                                parts[0].parse().ok()?,
                                parts[1].parse().ok()?,
                                parts[2].parse().ok()?,
                            ))
                        }
                        else
                        {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        let reversible = json.get("version").and_then(|v| v.as_str()) == Some("byte_level_v2");
        Ok(Self {
            vocab,
            rev,
            merges,
            reversible,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_train_small() {
        let texts = vec![
            "low low low low low low".to_string(),
            "lowest lowest lowest".to_string(),
            "newer newer".to_string(),
            "wider wider".to_string(),
        ];
        let trainer = BpeTrainer::new(50).min_frequency(1);
        let tok = trainer.train(&texts);
        assert!(tok.vocab_size() > 20, "vocab_size={}", tok.vocab_size());

        let encoded = tok.encode("low");
        assert!(!encoded.is_empty());

        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "low");
    }

    #[test]
    fn bpe_encode_survives_save_load() {
        // Realistic corpus: multi-line, with the Unicode a real code+docs corpus has
        // (→ ᵀ × ✅ ⊙ √ é), which the byte-level base vocab represents as `<NNN>`
        // placeholder keys — the case a pure-ASCII test misses.
        let sample =
            "fn main() {\n    // rms → √(mean(x²)+eps) · wᵀ ✅ é ⊙\n    println!(\"hi\");\n}\n";
        let texts = vec![sample.repeat(50)];
        let trainer = BpeTrainer::new(1024).min_frequency(1);
        let tok = trainer.train(&texts);
        let before = tok.encode("fn main");
        let unk_before = before.iter().filter(|&&id| id == 3).count();
        assert!(
            unk_before < before.len(),
            "encode is all <unk> BEFORE save (train bug): {before:?}"
        );

        let path = std::env::temp_dir().join("scirust_bpe_roundtrip.json");
        let path = path.to_str().unwrap();
        tok.save_json(path).unwrap();
        let tok2 = BpeTokenizer::load_json(path).unwrap();
        assert!(
            tok2.vocab.contains_key("f"),
            "base token \"f\" lost after save/load — vocab has {} entries",
            tok2.vocab.len()
        );
        let after = tok2.encode("fn main");
        assert_eq!(
            before, after,
            "encode differs after save/load — save/load loses the vocab.\n\
             before {before:?}\nafter  {after:?}"
        );
        assert_eq!(tok2.decode(&after), "fn main", "decode after load");
    }

    #[test]
    fn bpe_training_is_deterministic() {
        let texts = vec![
            "fn a() { let x = 1; }\nfn b() { let y = 2; }\n".repeat(20),
            "// → ᵀ × ✅ comment ⊙ √\nstruct S { f: u32 }\n".repeat(20),
        ];
        let train = || BpeTrainer::new(600).min_frequency(1).train(&texts);
        let (t1, t2) = (train(), train());
        assert_eq!(t1.merges, t2.merges, "merges differ across identical runs");
        assert_eq!(
            t1.encode("fn a() { let x"),
            t2.encode("fn a() { let x"),
            "encode differs across identical runs"
        );
    }

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let trainer = BpeTrainer::new(50).min_frequency(1);
        let texts = vec![
            "hello world this is a test".to_string(),
            "hello hello world".to_string(),
        ];
        let tok = trainer.train(&texts);
        let original = "hello world";
        let ids = tok.encode(original);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_bpe_with_special() {
        let trainer = BpeTrainer::new(50).min_frequency(1);
        let texts = vec!["test data for tokenizer".to_string()];
        let tok = trainer.train(&texts);
        let ids = tok.encode_with_special("hello", true, true);
        assert_eq!(ids[0], 1); // <bos>
        assert_eq!(*ids.last().unwrap(), 2); // <eos>
    }

    #[test]
    fn test_bpe_save_load_json() {
        let trainer = BpeTrainer::new(50).min_frequency(1);
        let texts = vec!["save and load test".to_string()];
        let tok = trainer.train(&texts);
        let path = "/tmp/test_bpe_sciagent.json";
        tok.save_json(path).unwrap();
        let loaded = BpeTokenizer::load_json(path).unwrap();
        assert_eq!(tok.vocab_size(), loaded.vocab_size());
        let original = "save and load";
        assert_eq!(tok.encode(original), loaded.encode(original));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_bpe_decode_preserves_angle_brackets() {
        // Regression: decode skipped every token starting with '<', deleting the
        // '<' of generics/comparisons from decoded Rust ("fn f<T>" -> "fn f>").
        let tok = BpeTokenizer::from_embedded().unwrap();
        for src in ["Vec<T>", "fn f<T>(x: T) -> T { x }", "a < b && b << 2"]
        {
            let ids = tok.encode(src);
            assert_eq!(tok.decode(&ids), src, "roundtrip must preserve '<'");
        }
        // The named specials are still elided.
        let with_special = tok.encode_with_special("x", true, true);
        assert_eq!(tok.decode(&with_special), "x");
    }

    #[test]
    fn test_bpe_empty_input() {
        let trainer = BpeTrainer::new(50).min_frequency(1);
        let texts = vec!["test".to_string()];
        let tok = trainer.train(&texts);
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn bpe_v2_roundtrips_arbitrary_utf8() {
        // v2 tokenizers reconstruct bytes structurally, so ANY UTF-8 — multibyte
        // punctuation, accents, CJK, emoji — round-trips through encode→decode
        // exactly. This is the whole point of the reversible scheme: no `<NNN>`, and
        // multibyte chars split across two BPE tokens still reassemble.
        let corpus = "fn f() { let s = \"café — π ≈ 3.14 · ✅ 世界 🚀\"; }\n".repeat(40);
        let tok = BpeTrainer::new(800).min_frequency(1).train(&[corpus]);
        for s in [
            "café — π ≈ 3.14 · ✅ 世界 🚀",
            "fn main() { let x: Vec<u8> = vec![0xFF, 0x00]; }",
            "// rms → √(mean(x²)+eps) · wᵀ",
            "",
            "a",
        ]
        {
            let ids = tok.encode(s);
            assert_eq!(tok.decode(&ids), s, "v2 must round-trip exactly: {s:?}");
        }
    }

    #[test]
    fn bpe_v2_never_leaks_placeholder() {
        // No decoded output may contain a `<NNN>` byte placeholder — the artifact the
        // old scheme leaked for every non-ASCII byte (`<194><183>` etc. in real runs).
        let corpus = "let x = \"— · é ✅\";\n".repeat(50);
        let tok = BpeTrainer::new(400).min_frequency(1).train(&[corpus]);
        let out = tok.decode(&tok.encode("— · é ✅ literal <200> stays"));
        assert!(
            !out.contains("<194>") && !out.contains("<183>"),
            "byte placeholder leaked into decode: {out:?}"
        );
        // A literal "<200>" in the text must survive (structural decode, not skipped
        // as if it were a placeholder).
        assert!(
            out.contains("<200>"),
            "real <200> text was dropped: {out:?}"
        );
        // All 256 bytes are base tokens ⇒ encode never emits `<unk>` (id 3) for bytes.
        let unk = tok.encode("ΩΩΩ ∑∫∂ 🔥").iter().filter(|&&t| t == 3).count();
        assert_eq!(
            unk, 0,
            "v2 must have every byte in the base vocab (no <unk>)"
        );
    }

    #[test]
    fn bpe_v2_scheme_survives_save_load() {
        let tok = BpeTrainer::new(400)
            .min_frequency(1)
            .train(&["日本語 test → ok\n".repeat(30)]);
        let path = std::env::temp_dir().join("scirust_bpe_v2_roundtrip.json");
        let path = path.to_str().unwrap();
        tok.save_json(path).unwrap();
        let tok2 = BpeTokenizer::load_json(path).unwrap();
        let s = "日本語 → ok";
        assert_eq!(
            tok.encode(s),
            tok2.encode(s),
            "encode differs after save/load"
        );
        assert_eq!(
            tok2.decode(&tok2.encode(s)),
            s,
            "v2 decode must survive save/load (version tag preserved)"
        );
        let _ = std::fs::remove_file(path);
    }
}
