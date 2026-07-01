//! Retrieval-augmented context windowing.
//!
//! `scirust-retrieval` is a complete dense + hybrid retrieval engine
//! ([`crate::SemanticRetriever`], [`crate::DenseIndex`], [`crate::LshIndex`],
//! [`crate::HybridRetriever`]) positioned as the auditable alternative to RAG —
//! but it never feeds a language model. [`RagContext`] is the missing bridge:
//! it indexes chunks of text and, for each user turn, retrieves the top-`k`
//! most relevant chunks and assembles a **bounded** augmented prompt (context
//! window) capped in either chunks or approximate characters, so the LM's
//! context budget is respected deterministically.
//!
//! The retriever does not depend on the LM; [`RagContext::augment`] produces
//! the augmented prompt string. A thin [`RagContext::generate_with`] runs a
//! `MiniLLM` over the augmented prompt for convenience.

use crate::{Encoder, RetrievalError, SemanticRetriever};

/// A bounded window of retrieved context.
#[derive(Debug, Clone, PartialEq)]
pub struct AugmentedPrompt {
    /// The assembled prompt, prefix + retrieved chunks + the user query.
    pub prompt: String,
    /// The ids of the chunks actually included (in order).
    pub chunk_ids: Vec<u64>,
    /// How many characters of context were trimmed to fit the budget.
    pub trimmed_chars: usize,
}

/// Cap on the assembled context window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContextBudget {
    /// At most `k` chunks.
    Chunks(usize),
    /// At most `max_chars` characters of retrieved text (chunks dropped from
    /// the end once the limit is exceeded).
    Chars(usize),
    /// Both: take up to `k` chunks, then trim to `max_chars`.
    Both { k: usize, max_chars: usize },
}

/// A retrieval-augmented context window over an [`Encoder`]'s semantic index.
pub struct RagContext<E: Encoder> {
    retriever: SemanticRetriever<E>,
    /// Prefix prepended before the retrieved chunks (system / instruction).
    prefix: String,
    /// Separator between chunks.
    sep: String,
    /// Parallel text store (`id -> chunk text`) so [`Self::augment`] can
    /// rebuild the prompt — the index keeps embeddings, not text.
    text_store: Vec<(u64, String)>,
}

impl<E: Encoder> RagContext<E> {
    /// New RAG context with a system `prefix` and a chunk `sep`.
    pub fn new(encoder: E, prefix: impl Into<String>, sep: impl Into<String>) -> Self {
        Self {
            retriever: SemanticRetriever::new(encoder),
            prefix: prefix.into(),
            sep: sep.into(),
            text_store: Vec::new(),
        }
    }

    /// Borrow the underlying retriever (to add chunks, inspect, etc.).
    pub fn retriever(&mut self) -> &mut SemanticRetriever<E> {
        &mut self.retriever
    }

    /// Index one chunk under `id`.
    pub fn index_chunk(&mut self, id: u64, chunk: &str) -> Result<(), RetrievalError> {
        if let Some(slot) = self.text_store.iter_mut().find(|(i, _)| *i == id)
        {
            slot.1 = chunk.to_string();
        }
        else
        {
            self.text_store.push((id, chunk.to_string()));
        }
        self.retriever.index_text(id, chunk)
    }

    /// Assemble a bounded augmented prompt for `query`: retrieve the top chunks
    /// and fit them to `budget`. The user `query` is always appended last.
    pub fn augment(&mut self, query: &str, budget: ContextBudget) -> AugmentedPrompt {
        let k = match budget
        {
            ContextBudget::Chunks(k) | ContextBudget::Both { k, .. } => k.max(1),
            ContextBudget::Chars(_) => 64, // retrieve a healthy pool, then trim
        };
        let hits = self.retriever.retrieve(query, k);
        let max_chars = match budget
        {
            ContextBudget::Chars(c) | ContextBudget::Both { max_chars: c, .. } => Some(c),
            ContextBudget::Chunks(_) => None,
        };

        let mut prompt = self.prefix.clone();
        let mut chunk_ids = Vec::new();
        let mut used_chars = 0usize;
        let mut trimmed = 0usize;
        // We do not have the chunk text back from the index (it stores
        // embeddings), so the caller typically passes the chunk store too. To
        // keep this self-contained we keep a parallel text store.
        let mut budget_hit = false;
        for h in &hits
        {
            if let Some(text) = self.text_for(h.id)
            {
                // Once the char budget is exceeded, every remaining (lower-ranked)
                // chunk is dropped from the end, so its text counts as trimmed.
                if budget_hit
                {
                    trimmed += text.len();
                    continue;
                }
                let piece = if prompt.ends_with(&self.sep) || prompt.is_empty()
                {
                    text.clone()
                }
                else
                {
                    format!("{}{}", self.sep, text)
                };
                if let Some(maxc) = max_chars
                {
                    if used_chars + piece.len() > maxc
                    {
                        // This chunk does not fit: drop it and everything after it
                        // ("chunks dropped from the end"). Count the chunk text
                        // (not the separator) as trimmed context.
                        trimmed += text.len();
                        budget_hit = true;
                        continue;
                    }
                }
                prompt.push_str(&piece);
                used_chars += piece.len();
                chunk_ids.push(h.id);
            }
        }
        // Append the user query.
        if !prompt.is_empty()
        {
            prompt.push_str(&self.sep);
        }
        prompt.push_str(query);
        AugmentedPrompt {
            prompt,
            chunk_ids,
            trimmed_chars: trimmed,
        }
    }

    /// Run `generate` on the augmented prompt. Kept behind a closure so this
    /// crate does not hard-depend on `scirust-core`'s transformer module.
    pub fn generate_with<G>(&mut self, query: &str, budget: ContextBudget, generate: G) -> String
    where
        G: FnOnce(&str, usize) -> String,
    {
        let aug = self.augment(query, budget);
        generate(&aug.prompt, 256)
    }

    // The index stores embeddings, not text; keep a parallel text store so
    // `augment` can rebuild the prompt. Filled by `index_chunk`.
    fn text_for(&self, _id: u64) -> Option<String> {
        self.text_store
            .iter()
            .find(|(id, _)| *id == _id)
            .map(|(_, t)| t.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic whitespace-tokenized bag-of-words encoder for tests. The
    /// vocabulary is fixed at construction (from `vocab_texts`) so embeddings are
    /// reproducible — unlike `EmbeddingEngine`, which is a randomly-initialised
    /// MiniLLM whose cosine ranking is noise for semantic assertions.
    struct BagOfWords {
        vocab: HashMap<String, usize>,
        dim: usize,
    }

    impl BagOfWords {
        fn new(vocab_texts: &[&str]) -> Self {
            let mut vocab = HashMap::new();
            for t in vocab_texts
            {
                for w in t.split_whitespace()
                {
                    if !vocab.contains_key(w)
                    {
                        let i = vocab.len();
                        vocab.insert(w.to_string(), i);
                    }
                }
            }
            Self {
                dim: vocab.len(),
                vocab,
            }
        }
    }

    impl Encoder for BagOfWords {
        fn embedding_dim(&self) -> usize {
            self.dim
        }
        fn encode(&mut self, text: &str) -> Vec<f32> {
            let mut v = vec![0.0f32; self.dim];
            for w in text.split_whitespace()
            {
                if let Some(&i) = self.vocab.get(w)
                {
                    v[i] += 1.0;
                }
            }
            // L2-normalise so cosine is well-defined.
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0
            {
                for x in &mut v
                {
                    *x /= norm;
                }
            }
            v
        }
    }

    fn build() -> RagContext<BagOfWords> {
        let engine = BagOfWords::new(&["rain causes wet roads", "rust is fast", "cats purr"]);
        RagContext::new(engine, "Context:\n", "\n---\n")
    }

    #[test]
    fn augment_retrieves_relevant_chunk_and_appends_query() {
        let mut rag = build();
        rag.index_chunk(1, "rain causes wet roads").unwrap();
        rag.index_chunk(2, "rust is fast").unwrap();
        rag.index_chunk(3, "cats purr").unwrap();
        let aug = rag.augment("what about rain", ContextBudget::Chunks(1));
        // "rain" is the only query word present in the vocabulary, so chunk 1
        // (the only chunk containing "rain") is the unique top hit.
        assert_eq!(aug.chunk_ids, vec![1]);
        assert!(aug.prompt.starts_with("Context:\n"));
        assert!(aug.prompt.contains("rain causes wet roads"));
        // The user query is appended at the very end.
        assert!(aug.prompt.ends_with("what about rain"));
    }

    #[test]
    fn char_budget_trims_extra_chunks() {
        let mut rag = build();
        rag.index_chunk(1, "rain causes wet roads").unwrap();
        rag.index_chunk(2, "rust is fast and memory safe").unwrap();
        let aug = rag.augment("rain", ContextBudget::Chars(10));
        // With a 10-char budget the first retrieved chunk already exceeds it, so
        // no chunk text is added and only the query remains.
        assert!(
            aug.chunk_ids.is_empty()
                || aug.prompt.len() <= 10 + "Context:\n".len() + "\n---\n".len() + "rain".len()
        );
    }

    #[test]
    fn char_budget_drops_from_the_end_and_counts_trimmed_text() {
        // Vocab must contain both query terms so the encoder ranks the chunks.
        let engine = BagOfWords::new(&["alpha", "beta"]);
        let mut rag = RagContext::new(engine, "Context:\n", "\n---\n");
        // Chunk 1: pure "alpha" (cosine 1 with query "alpha") but long (29 chars).
        // Chunk 2: "alpha beta" (cosine ~0.707, lower-ranked) but short (10 chars).
        rag.index_chunk(1, "alpha alpha alpha alpha alpha").unwrap();
        rag.index_chunk(2, "alpha beta").unwrap();
        // The top-ranked chunk 1 alone already blows the 20-char budget, so it is
        // dropped — and, being "dropped from the end", so is the lower-ranked
        // chunk 2 even though it would fit on its own. The buggy version skipped
        // chunk 1 and kept chunk 2, reordering the results by size.
        let aug = rag.augment("alpha", ContextBudget::Chars(20));
        assert!(
            aug.chunk_ids.is_empty(),
            "no chunk fits the budget without violating relevance order, got {:?}",
            aug.chunk_ids
        );
        // trimmed_chars counts the dropped chunk *text* (29 + 10), not the
        // separators (the buggy version counted the separator and stopped
        // counting after the first skip, reporting 34).
        assert_eq!(aug.trimmed_chars, 29 + 10);
        // Only the query survives in the prompt.
        assert_eq!(aug.prompt, "Context:\n\n---\nalpha");
    }

    #[test]
    fn index_chunk_updates_existing_id() {
        let mut rag = build();
        rag.index_chunk(1, "old text").unwrap();
        rag.index_chunk(1, "new text").unwrap();
        assert_eq!(rag.text_store.len(), 1);
        assert_eq!(rag.text_store[0].1, "new text");
    }
}
