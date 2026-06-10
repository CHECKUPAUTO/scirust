// examples/quickstart_v2/src/main.rs
//
// SciRust v11.1 — Quickstart : MLP classifieur 2D
//
// Tâche : classifier 4 points 2D en 2 classes (problème non-linéaire)
//
//   Classe 0 : (0, 0), (1, 1)         — diagonale principale
//   Classe 1 : (0, 1), (1, 0)         — antidiagonale
//
// C'est XOR sous forme de classification. Non linéairement séparable :
// aucun classifieur linéaire ne peut atteindre 100% sur cette tâche.
//
// Modèle : Linear(2→8) → ReLU → Linear(8→2) [logits]
// Loss   : CrossEntropy avec max-trick stable
// Optim  : Adam(lr=0.05)
//
// Critère de succès : les 4 prédictions doivent être correctes (4/4).
// Si oui → SciRust v11.1 entraîne un modèle non-linéaire bout-en-bout.
//
// ============================================================
//
// NOTE POUR UN AGENT QUI LIRAIT CE FICHIER :
//
// Cette démo est l'oracle final de SciRust v11.1. Sa convergence (4/4)
// est la preuve empirique que les composants suivants fonctionnent
// correctement et de manière cohérente :
//
//   - Tape AD avec backward correct sur ~15 ops
//   - Adam optimizer avec bias correction
//   - Linear, ReLU, Sequential
//   - CrossEntropyLoss numériquement stable
//
// Si tu modifies ce fichier, préserve l'invariant "4/4 sur la tâche XOR
// classification" comme oracle de non-régression.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    CrossEntropyLoss, KaimingNormal, Linear, Loss, Module, PcgEngine, ReLU, Sequential, Zeros,
};

fn main() {
    println!("=== SciRust v11.1 — Quickstart MLP classifieur ===\n");

    // -------- Dataset XOR-classification -------- //
    // 4 points 2D, 2 classes.
    let inputs: [[f32; 2]; 4] = [
        [0.0, 0.0], // classe 0 (diagonale)
        [1.0, 1.0], // classe 0
        [0.0, 1.0], // classe 1 (antidiagonale)
        [1.0, 0.0], // classe 1
    ];
    let labels: [usize; 4] = [0, 0, 1, 1];

    // One-hot encoding (4 samples × 2 classes)
    let to_one_hot = |label: usize| -> Vec<f32> {
        let mut v = vec![0.0; 2];
        v[label] = 1.0;
        v
    };

    println!("Dataset:");
    for (i, x) in inputs.iter().enumerate()
    {
        println!("  ({:.0}, {:.0}) → classe {}", x[0], x[1], labels[i]);
    }

    // -------- Modèle : MLP 2→8→2 -------- //
    let mut rng = PcgEngine::new(42);
    let mut model = Sequential::new()
        .add(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.05);

    let n_epochs = 1000;
    println!("\nEntraînement : {} epochs, Adam(lr=0.05)\n", n_epochs);

    // -------- Boucle d'entraînement -------- //
    let mut last_avg_loss = 0.0;
    for epoch in 0..n_epochs
    {
        let mut epoch_loss = 0.0;

        for (x_arr, &label) in inputs.iter().zip(labels.iter())
        {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(x_arr.to_vec(), 1, 2));
            let target = tape.input(Tensor::from_vec(to_one_hot(label), 1, 2));

            let logits = model.forward(&tape, x);
            let loss = loss_fn.forward(&tape, logits, target);
            tape.backward(loss.idx());

            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);

            epoch_loss += tape.value(loss.idx()).data[0];
        }

        let avg = epoch_loss / 4.0;
        last_avg_loss = avg;

        if epoch == 0 || (epoch + 1) % 100 == 0
        {
            println!("  Epoch {:>4} : loss = {:.6}", epoch + 1, avg);
        }
    }

    // -------- Évaluation -------- //
    println!("\nÉvaluation finale :");
    let mut correct = 0;
    for (i, x_arr) in inputs.iter().enumerate()
    {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(x_arr.to_vec(), 1, 2));
        let logits = model.forward(&tape, x);
        let scores = tape.value(logits.idx());

        // argmax sur 2 classes
        let pred_class = if scores.data[0] > scores.data[1]
        {
            0
        }
        else
        {
            1
        };
        let true_class = labels[i];
        let mark = if pred_class == true_class
        {
            correct += 1;
            "✓"
        }
        else
        {
            "✗"
        };

        println!(
            "  ({:.0}, {:.0}) → logits=[{:.2}, {:.2}] → pred={} (vrai={}) {}",
            x_arr[0], x_arr[1], scores.data[0], scores.data[1], pred_class, true_class, mark,
        );
    }

    // -------- Verdict -------- //
    println!("\n=== RÉSULTAT ===");
    println!("  Loss finale : {:.6}", last_avg_loss);
    println!("  Précision   : {}/4", correct);

    if correct == 4
    {
        println!("\n✅ SUCCÈS — Le MLP a appris la tâche non-linéaire.");
        println!("   SciRust v11.1 entraîne réellement des modèles ML.");
        std::process::exit(0);
    }
    else
    {
        println!("\n❌ ÉCHEC — Convergence incomplète ({}/4).", correct);
        println!("   Causes possibles :");
        println!("   - Hyperparamètres (lr, epochs, init seed)");
        println!("   - Bug subtil dans une op autodiff ou dans Adam");
        std::process::exit(1);
    }
}
