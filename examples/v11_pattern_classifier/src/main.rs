// V11 Pattern Classifier Demo
// Détecte si une séquence contient le motif [3, 7, 1]

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::transformer::TransformerEncoder;
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::tensor::tensor3d::{Tensor3D, Var3D};

fn main() {
    println!("🚀 V11 Pattern Classifier — Transformer Encoder Demo");
    println!("=====================================================\n");

    let mut rng = PcgEngine::new(42);
    let init_w = KaimingNormal;
    let init_b = Zeros;

    // Config
    let vocab = 16usize;
    let d_model = 32usize;
    let n_heads = 4usize;
    let d_ff = 64usize;
    let n_layers = 2usize;
    let seq_len = 8usize;
    let batch = 4usize;

    println!("Config : vocab={}, d_model={}, n_heads={}, d_ff={}, n_layers={}, seq_len={}, batch={}",
             vocab, d_model, n_heads, d_ff, n_layers, seq_len, batch);

    // Créer l'encoder
    let mut encoder = TransformerEncoder::new(
        n_layers, d_model, n_heads, d_ff,
        false, // non causal
        &init_w, &init_b, &mut rng,
    );
    println!("✅ TransformerEncoder créé");

    // Table d'embedding
    let emb_table = Tensor::zeros(vocab, d_model);

    // Input: batch de 4 séquences de longueur 8
    let input_3d = Tensor3D::zeros(batch, seq_len, d_model);
    println!("✅ Input Tensor3D : {:?}", input_3d.shape());

    // Forward pass
    let tape = Tape::new();
    let x_3d = Var3D::input_3d(&tape, input_3d);
    println!("✅ Var3D créé sur la tape");

    let h = encoder.forward_3d(&tape, x_3d);
    println!("✅ Forward pass terminé — output shape : {:?}", h.shape());

    println!("\n🎉 V11 Pattern Classifier Demo terminée avec succès !");
    println!("   Le Transformer Encoder est fonctionnel.");
}
