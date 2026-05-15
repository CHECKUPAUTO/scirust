// scirust-core/src/nn/transformer/mod.rs
//
// Le module Transformer regroupe tous les composants necessaires pour
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

pub mod attention;
pub mod block;
pub mod decoder;
pub mod encoder;
pub mod flash_attention;
pub mod mini_llm;

pub use attention::MultiHeadAttention;
pub use block::TransformerBlock;
pub use decoder::{TransformerDecoder, TransformerDecoderBlock};
pub use encoder::TransformerEncoder;
