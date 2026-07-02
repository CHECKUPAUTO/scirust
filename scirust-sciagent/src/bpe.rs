use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;

const SPECIAL_TOKENS: &[(&str, usize)] = &[
    ("<pad>", 0),
    ("<bos>", 1),
    ("<eos>", 2),
    ("<unk>", 3),
];

fn byte_to_str(b: u8) -> String {
    let single = vec![b];
    String::from_utf8(single).unwrap_or_else(|_| format!("<{b}>"))
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
        for (tok, idx) in SPECIAL_TOKENS {
            vocab.insert(tok.to_string(), *idx);
            rev.push(tok.to_string());
        }

        // Ajouter tous les bytes uniques du corpus (byte-level BPE)
        let mut byte_set = HashSet::new();
        for t in texts {
            for b in t.bytes() {
                byte_set.insert(b);
            }
        }
        let mut byte_ids: Vec<u8> = byte_set.into_iter().collect();
        byte_ids.sort();
        for b in &byte_ids {
            let s = byte_to_str(*b);
            if !vocab.contains_key(&s) {
                vocab.insert(s.clone(), id);
                rev.push(s);
                id += 1;
            }
        }

        // Étape 2 : tokeniser chaque texte en séquence de bytes
        let mut corpus: Vec<Vec<usize>> = Vec::with_capacity(texts.len());
        for t in texts {
            let ids: Vec<usize> = t.bytes().map(|b| vocab[&byte_to_str(b)]).collect();
            corpus.push(ids);
        }

        // Étape 3 : itérer les merges BPE
        // On regroupe les merges par lots pour réduire le nombre de passes
        let mut merges: Vec<(usize, usize, usize)> = Vec::new();
        let merge_batch_size = std::cmp::min(2000, (self.vocab_size.saturating_sub(id)) / 4 + 1);

        while id < self.vocab_size {
            let mut pair_counts: HashMap<(usize, usize), u64> = HashMap::new();
            for tokens in &corpus {
                for w in tokens.windows(2) {
                    if w[0] != 0 && w[1] != 0 {
                        *pair_counts.entry((w[0], w[1])).or_insert(0) += 1;
                    }
                }
            }

            // Prendre les N meilleures paires
            let mut ranked: Vec<((usize, usize), u64)> = pair_counts
                .into_iter()
                .filter(|&(_, count)| count >= self.min_frequency as u64)
                .collect();
            ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            let batch: Vec<(usize, usize)> = ranked
                .into_iter()
                .take(merge_batch_size)
                .map(|((a, b), _)| (a, b))
                .collect();

            if batch.is_empty() {
                break;
            }

            for &(pa, pb) in &batch {
                if id >= self.vocab_size {
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

            for tokens in &mut corpus {
                let mut out = Vec::with_capacity(tokens.len());
                let mut i = 0;
                while i < tokens.len() {
                    if i + 1 < tokens.len() {
                        let key = (tokens[i], tokens[i + 1]);
                        if let Some(&new_id) = merge_map.get(&key) {
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

            if id % 1000 == 0 {
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
        }
    }
}

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    vocab: BTreeMap<String, usize>,
    rev: Vec<String>,
    merges: Vec<(usize, usize, usize)>,
}

impl BpeTokenizer {
    pub fn new(vocab: BTreeMap<String, usize>, rev: Vec<String>, merges: Vec<(usize, usize, usize)>) -> Self {
        Self { vocab, rev, merges }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn special_id(&self, name: &str) -> usize {
        *self.vocab.get(name).unwrap_or(&3)
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<usize> = text
            .bytes()
            .map(|b| *self.vocab.get(&byte_to_str(b)).unwrap_or(&3))
            .collect();

        if ids.is_empty() {
            return ids;
        }

        // Appliquer les merges gloutonnement (optimisé avec lookup table)
        let merge_lookup: HashMap<(usize, usize), usize> = self
            .merges
            .iter()
            .map(|&(a, b, new_id)| ((a, b), new_id))
            .collect();

        if merge_lookup.is_empty() {
            return ids;
        }

        // Single-pass BPE merge using output buffer (avoids O(n²) remove())
        loop {
            let mut out = Vec::with_capacity(ids.len());
            let mut any_merged = false;
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] != 0 && ids[i + 1] != 0 {
                    if let Some(&new_id) = merge_lookup.get(&(ids[i], ids[i + 1])) {
                        out.push(new_id);
                        i += 2;
                        any_merged = true;
                        continue;
                    }
                }
                out.push(ids[i]);
                i += 1;
            }
            if !any_merged {
                break;
            }
            ids = out;
        }

        ids
    }

    pub fn encode_with_special(&self, text: &str, prepend_bos: bool, append_eos: bool) -> Vec<usize> {
        let mut ids = Vec::new();
        if prepend_bos {
            ids.push(self.special_id("<bos>"));
        }
        ids.extend(self.encode(text));
        if append_eos {
            ids.push(self.special_id("<eos>"));
        }
        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id < self.rev.len() {
                let s = &self.rev[id];
                // skip special tokens
                if !s.starts_with('<') {
                    out.push_str(s);
                }
            }
        }
        out
    }

    #[allow(dead_code)]
    fn find_merge(&self, left: usize, right: usize) -> Option<usize> {
        for &(l, r, new_id) in &self.merges {
            if l == left && r == right {
                return Some(new_id);
            }
        }
        None
    }

    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::json!({
            "vocab": self.vocab,
            "merges": self.merges.iter().map(|(a, b, c)| format!("{a} {b} {c}")).collect::<Vec<_>>(),
        });
        fs::write(path, serde_json::to_string_pretty(&json)?)?;
        Ok(())
    }

    pub fn load_json(path: &str) -> std::io::Result<Self> {
        let s = fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&s)?;
        let vocab: BTreeMap<String, usize> =
            serde_json::from_value(json["vocab"].clone())?;
        let rev: Vec<String> = {
            let mut v = vec![String::new(); vocab.len()];
            for (s, &id) in &vocab {
                if id < v.len() {
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
                        if parts.len() == 3 {
                            Some((
                                parts[0].parse().ok()?,
                                parts[1].parse().ok()?,
                                parts[2].parse().ok()?,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self { vocab, rev, merges })
    }

    pub fn from_embedded() -> std::io::Result<Self> {
        let bpe_json = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/tokenizer/bpe.json"));
        let s = std::str::from_utf8(bpe_json).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        let json: serde_json::Value = serde_json::from_str(s).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        let vocab: BTreeMap<String, usize> =
            serde_json::from_value(json["vocab"].clone())?;
        let rev: Vec<String> = {
            let mut v = vec![String::new(); vocab.len()];
            for (s, &id) in &vocab {
                if id < v.len() {
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
                        if parts.len() == 3 {
                            Some((
                                parts[0].parse().ok()?,
                                parts[1].parse().ok()?,
                                parts[2].parse().ok()?,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self { vocab, rev, merges })
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
    fn test_bpe_empty_input() {
        let trainer = BpeTrainer::new(50).min_frequency(1);
        let texts = vec!["test".to_string()];
        let tok = trainer.train(&texts);
        assert!(tok.encode("").is_empty());
    }
}
