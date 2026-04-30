// scirust-core/src/nn/transformer/mod.rs
//
// Le module Transformer regroupe tous les composants nécessaires pour
// construire des architectures encoder-style (BERT-like).
//
// USAGE TYPIQUE :
//
//   use scirust_core::nn::transformer::{TransformerEncoder, MultiHeadAttention};
//   use scirust_core::tensor::tensor3d::{Tensor3D, Var3D};
//
//   let mut encoder = TransformerEncoder::new(
//       n_layers, d_model, n_heads, d_ff, causal=false,
//       &init_w, &init_b, &mut rng,
//   );
//
//   let x_3d = Var3D::input_3d(&tape, x_tensor3d);
//   let h = encoder.forward_3d(&tape, x_3d);
//   // h.shape() = (batch, seq_len, d_model)
//
// SCOPE V11 :
//   ✅ Encoder seul (BERT-style)
//   ✅ Self-attention multi-tête (avec ou sans causal mask)
//   ✅ FeedForward layer (2 linears + ReLU)
//   ✅ LayerNorm pré-normalisation
//   ✅ Residual connections
//   ❌ Decoder (cross-attention) — reporté à v12
//   ❌ Position encoding intégré — laissé à l'utilisateur (positional
//      embedding learnable typiquement, voir la démo)

pub mod attention;
pub mod block;
pub mod encoder;

pub use attention::MultiHeadAttention;
pub use block::TransformerBlock;
pub use encoder::TransformerEncoder;
