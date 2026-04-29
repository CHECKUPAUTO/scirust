// examples/v9_quickstart/src/main.rs
//
// SciRust quickstart — moins de 50 lignes effectives pour entraîner
// un classifieur 2 classes sur des données synthétiques.
//
// SI CE PROGRAMME COMPILE ET CONVERGE, l'installation marche.

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::nn::{
    PcgEngine, Module, Sequential, Linear, ReLU,
    KaimingNormal, Zeros,
};
use scirust_core::nn::loss::{Loss, strict::CrossEntropyLoss};

fn main() {
    println!("=== SciRust quickstart ===\n");

    let mut rng = PcgEngine::new(42);
    let mut model = Sequential::new()
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng).with_name("fc1"))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng).with_name("fc2"));

    // 4 points : 2 dans le quadrant +,+ et 2 dans -,- — class 0 vs class 1
    let x = Tensor::from_vec(
        vec![1.0, 1.0,    2.0, 2.0,    -1.0, -1.0,    -2.0, -2.0],
        4, 2);
    let y = Tensor::from_vec(
        vec![1.0, 0.0,    1.0, 0.0,     0.0, 1.0,      0.0, 1.0],
        4, 2);

    let mut opt = Adam::new(0.05);

    for epoch in 0..100 {
        let tape = Tape::new();
        let xv = tape.input(x.clone());
        let yv = tape.input(y.clone());
        let logits = model.forward(&tape, xv);
        let loss = CrossEntropyLoss.forward(logits, yv);
        loss.backward();
        opt.step(&model.parameter_indices(), &tape);
        model.sync(&tape);

        if epoch % 20 == 0 {
            println!("epoch {epoch:3}: loss = {:.4}",
                     tape.value(loss.idx()).data[0]);
        }
    }

    // Inférence
    println!("\nFinal predictions:");
    let tape = Tape::new();
    let xv = tape.input(x.clone());
    let logits = model.forward(&tape, xv);
    let probs = tape.value(logits.idx());
    let labels = ["+,+", "+,+", "-,-", "-,-"];
    for i in 0..4 {
        let row = &probs.data[i*2..(i+1)*2];
        let pred = if row[0] > row[1] { 0 } else { 1 };
        let ok = if (pred == 0 && i < 2) || (pred == 1 && i >= 2) { "✅" } else { "❌" };
        println!("  {} : pred={pred} {ok}", labels[i]);
    }

    println!("\n→ Si tu vois 4 ✅, l'installation est validée.");
    println!("→ Étape suivante : docs/MNIST.md pour le dataset réel");
}
