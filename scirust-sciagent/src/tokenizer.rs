use std::collections::BTreeMap;

pub struct SciAgentTokenizer {
    vocab: BTreeMap<String, usize>,
    rev: Vec<String>,
}

impl SciAgentTokenizer {
    pub fn new_char_level(texts: &[&str]) -> Self {
        let mut chars = BTreeMap::new();
        for t in texts {
            for c in t.chars() {
                *chars.entry(c).or_insert(0) += 1;
            }
        }
        let mut vocab = BTreeMap::new();
        let mut rev = Vec::new();
        vocab.insert("<pad>".to_string(), 0);
        rev.push("<pad>".to_string());
        vocab.insert("<bos>".to_string(), 1);
        rev.push("<bos>".to_string());
        vocab.insert("<eos>".to_string(), 2);
        rev.push("<eos>".to_string());
        vocab.insert("<unk>".to_string(), 3);
        rev.push("<unk>".to_string());
        for (i, c) in chars.keys().enumerate() {
            let s = c.to_string();
            vocab.insert(s.clone(), i + 4);
            rev.push(s);
        }
        Self { vocab, rev }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for c in text.chars() {
            let s = c.to_string();
            ids.push(*self.vocab.get(&s).unwrap_or(&3));
        }
        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id < self.rev.len() {
                out.push_str(&self.rev[id]);
            }
        }
        out
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
