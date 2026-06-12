//! Moteur d'embeddings basé sur MiniLLM.
//!
//! Remplace l'appel HTTP à Ollama dans synergie en utilisant
//! les états cachés du transformer local comme vecteurs d'embedding.

use crate::nn::transformer::mini_llm::{CharTokenizer, MiniLLM, MiniLLMConfig};

/// Moteur d'embeddings basé sur MiniLLM.
/// Utilise les états cachés du transformer comme vecteurs d'embedding.
pub struct EmbeddingEngine {
    llm: MiniLLM,
}

impl EmbeddingEngine {
    /// Crée un nouvel engine avec vocab et config par défaut.
    ///
    /// La config par défaut produit des embeddings 128-dim (d_model=128).
    pub fn new(vocab_texts: &[&str]) -> Self {
        let tokenizer = CharTokenizer::new(vocab_texts);
        let config = MiniLLMConfig {
            vocab_size: tokenizer.vocab_size,
            ..MiniLLMConfig::default()
        };
        let llm = MiniLLM::new(config, tokenizer);
        Self { llm }
    }

    /// Crée un engine avec config personnalisée.
    pub fn new_with_config(vocab_texts: &[&str], config: MiniLLMConfig) -> Self {
        let tokenizer = CharTokenizer::new(vocab_texts);
        let config = MiniLLMConfig {
            vocab_size: tokenizer.vocab_size,
            d_model: config.d_model,
            n_heads: config.n_heads,
            n_layers: config.n_layers,
            d_ff: config.d_ff,
            max_seq_len: config.max_seq_len,
        };
        let llm = MiniLLM::new(config, tokenizer);
        Self { llm }
    }

    /// Dimension des vecteurs d'embedding produits.
    pub fn dim(&self) -> usize {
        self.llm.config.d_model
    }

    /// Embed un texte en vecteur f32.
    ///
    /// Tokenize l'entrée, forward pass dans le MiniLLM,
    /// récupère les états cachés (d_model), et prend la
    /// moyenne sur tous les tokens → vecteur 128-dim normalisé L2.
    pub fn embed(&mut self, text: &str) -> Vec<f32> {
        let ids = self.llm.tokenizer.encode(text);
        if ids.is_empty()
        {
            // Chaîne vide: retourne un vecteur nul (non-NaN)
            return vec![0.0f32; self.llm.config.d_model];
        }
        let hidden = self.llm.forward_hidden(&ids);
        // hidden shape = (seq_len, d_model), row-major
        let seq_len = hidden.nrows();
        let d_model = hidden.ncols();
        let mut mean = vec![0.0f32; d_model];
        let len_f = seq_len as f32;
        for i in 0..seq_len
        {
            let base = i * d_model;
            #[allow(clippy::needless_range_loop)]
            for j in 0..d_model
            {
                let v = hidden.data[base + j];
                if v.is_finite()
                {
                    mean[j] += v / len_f;
                }
            }
        }
        l2_normalize(&mut mean);
        mean
    }

    /// Embed un batch de textes.
    pub fn embed_batch(&mut self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Cosine similarity entre deux vecteurs f32.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let denom = norm_a.max(f32::EPSILON) * norm_b.max(f32::EPSILON);
        dot / denom
    }
}

/// Normalisation L2 in-place.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON
    {
        let inv = 1.0 / norm;
        for x in v.iter_mut()
        {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_returns_128_dims() {
        let mut engine = EmbeddingEngine::new(&["hello world test"]);
        let vec = engine.embed("hello");
        assert_eq!(
            vec.len(),
            128,
            "embedding dimension should be 128 (default d_model)"
        );
        // Vérifie que ce n'est pas NaN
        for &v in &vec
        {
            assert!(!v.is_nan(), "NaN in embedding vector");
        }
    }

    #[test]
    fn test_similar_sentences_high_cosine() {
        let mut engine = EmbeddingEngine::new(&["hello world hi test"]);
        let v1 = engine.embed("hello world");
        let v2 = engine.embed("hi world");
        let sim = EmbeddingEngine::cosine_similarity(&v1, &v2);
        assert!(
            sim > 0.5,
            "similar sentences should have cosine > 0.5, got {}",
            sim
        );
    }

    #[test]
    fn test_embed_batch_size() {
        let mut engine = EmbeddingEngine::new(&["hello world test"]);
        let texts = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
        let results = engine.embed_batch(&texts);
        assert_eq!(results.len(), 3, "batch should return 3 embeddings");
        for (i, v) in results.iter().enumerate()
        {
            assert_eq!(v.len(), 128, "embedding {} should be 128-dim", i);
        }
    }

    #[test]
    fn test_embed_empty_string() {
        let mut engine = EmbeddingEngine::new(&["hello"]);
        let vec = engine.embed("");
        assert_eq!(
            vec.len(),
            128,
            "empty string should still return 128-dim vector"
        );
        for &v in &vec
        {
            assert!(!v.is_nan(), "NaN in empty-string embedding");
            assert!(v.is_finite(), "non-finite in empty-string embedding");
        }
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = EmbeddingEngine::cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical vectors should have similarity 1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = EmbeddingEngine::cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 0.01,
            "orthogonal vectors should have similarity ~0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_l2_normalized() {
        let mut engine = EmbeddingEngine::new(&["hello world test"]);
        let vec = engine.embed("hello world");
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "L2-normalized vector should have norm ~1.0, got {}",
            norm
        );
    }
}
