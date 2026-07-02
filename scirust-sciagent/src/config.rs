#[derive(Debug, Clone)]
pub struct SciAgentConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub tie_embeddings: bool,
    pub use_bias: bool,
    pub eps: f32,
}

impl SciAgentConfig {
    pub fn sciagent_7b() -> Self {
        Self {
            vocab_size: 32768,
            d_model: 4096,
            n_layers: 40,
            n_heads: 32,
            n_kv_heads: 8,
            d_ff: 11008,
            max_seq_len: 8192,
            rope_theta: 1_000_000.0,
            tie_embeddings: true,
            use_bias: false,
            eps: 1e-5,
        }
    }

    pub fn sciagent_350m() -> Self {
        Self {
            vocab_size: 32768,
            d_model: 1024,
            n_layers: 24,
            n_heads: 16,
            n_kv_heads: 4,
            d_ff: 2816,
            max_seq_len: 8192,
            rope_theta: 1_000_000.0,
            tie_embeddings: true,
            use_bias: false,
            eps: 1e-5,
        }
    }

    pub fn small() -> Self {
        Self {
            vocab_size: 8192,
            d_model: 128,
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 2,
            d_ff: 256,
            max_seq_len: 256,
            rope_theta: 10000.0,
            tie_embeddings: true,
            use_bias: false,
            eps: 1e-5,
        }
    }

    pub fn debug() -> Self {
        Self {
            vocab_size: 256,
            d_model: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            d_ff: 128,
            max_seq_len: 128,
            rope_theta: 10000.0,
            tie_embeddings: false,
            use_bias: false,
            eps: 1e-5,
        }
    }

    pub fn total_parameters(&self) -> usize {
        let embed = self.vocab_size * self.d_model;
        let d_head = self.d_model / self.n_heads;
        let kv_dim = self.n_kv_heads * d_head;
        let per_layer = self.d_model * self.d_model
            + self.d_model * kv_dim
            + self.d_model * kv_dim
            + self.d_model * self.d_model
            + self.d_model * self.d_ff
            + self.d_model * self.d_ff
            + self.d_ff * self.d_model;
        let layers = self.n_layers * per_layer;
        let final_norm = self.d_model;
        let head = if self.tie_embeddings
        {
            0
        }
        else
        {
            self.d_model * self.vocab_size
        };
        embed + layers + final_norm + head
    }
}
