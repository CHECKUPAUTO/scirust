use crate::nlp::tokenization::Tokenizer;
use scirust_core::autodiff::reverse::Tensor;
use scirust_core::autodiff::reverse::{Tape, Var};
use scirust_core::nn::{Embedding, Linear, Module, PcgEngine};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentimentPolarity {
    Positive,
    Negative,
    Neutral,
}

#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub polarity: SentimentPolarity,
    pub confidence: f32,
    pub logits: Vec<f32>,
}

pub struct SentimentPipeline {
    pub embed: Embedding,
    pub classifier: Linear,
    pub tokenizer: Box<dyn Tokenizer>,
    pub max_seq_len: usize,
}

impl SentimentPipeline {
    pub fn new(
        tokenizer: Box<dyn Tokenizer>,
        embed_dim: usize,
        max_seq_len: usize,
        rng: &mut PcgEngine,
    ) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let embed = Embedding::new(vocab_size, embed_dim, &scirust_core::nn::KaimingNormal, rng);
        let classifier = Linear::new(
            embed_dim,
            2,
            &scirust_core::nn::KaimingNormal,
            &scirust_core::nn::Zeros,
            rng,
        );

        Self {
            embed,
            classifier,
            tokenizer,
            max_seq_len,
        }
    }

    pub fn forward<'a>(&mut self, tape: &'a Tape, text: &str) -> Var<'a> {
        let mut tokens = self.tokenizer.tokenize(text);
        let actual_len = tokens.len().min(self.max_seq_len);
        tokens.truncate(self.max_seq_len);
        let pad_id = self.tokenizer.pad_id();
        while tokens.len() < self.max_seq_len
        {
            tokens.push(pad_id);
        }

        let input_tensor = Tensor::from_vec(
            tokens.iter().map(|&t| t as f32).collect(),
            1,
            self.max_seq_len,
        );
        let input_var = tape.input(input_tensor);

        let embedded = self.embed.forward(tape, input_var);

        // Mean pooling intelligent : on ne moyenne que les tokens non-padding
        let pooled = self.mean_pool(tape, embedded, actual_len);

        self.classifier.forward(tape, pooled)
    }

    fn mean_pool<'a>(&self, tape: &'a Tape, embedded: Var<'a>, actual_len: usize) -> Var<'a> {
        let actual_len = actual_len.max(1); // Éviter division par zéro
        let inv_len = 1.0 / actual_len as f32;

        // Création d'un masque de pooling : 1/L pour les tokens réels, 0 pour le padding
        let mut mask = vec![0.0f32; self.max_seq_len];
        #[allow(clippy::needless_range_loop)]
        for i in 0..actual_len
        {
            mask[i] = inv_len;
        }

        let pool_mat = tape.input(Tensor::from_vec(mask, 1, self.max_seq_len));
        pool_mat.matmul(embedded)
    }

    pub fn predict(&mut self, text: &str) -> SentimentResult {
        let tape = Tape::new();
        let logits_var = self.forward(&tape, text);
        let logits = tape.value(logits_var.idx());

        let p0 = logits.data[0].exp();
        let p1 = logits.data[1].exp();
        let sum = p0 + p1;

        let prob0 = p0 / sum;
        let prob1 = p1 / sum;

        if prob1 > prob0
        {
            SentimentResult {
                polarity: SentimentPolarity::Positive,
                confidence: prob1,
                logits: logits.data.clone(),
            }
        }
        else
        {
            SentimentResult {
                polarity: SentimentPolarity::Negative,
                confidence: prob0,
                logits: logits.data.clone(),
            }
        }
    }

    pub fn sync(&mut self, tape: &Tape) {
        self.embed.sync(tape);
        self.classifier.sync(tape);
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = self.embed.parameter_indices();
        v.extend(self.classifier.parameter_indices());
        v
    }
}
