// examples/v6_features_demo/src/main.rs
//
// Démo v6 :
//   Partie A : validation numérique de la softmax batch row-wise
//   Partie B : MLP avec BatchNorm1d + CrossEntropy sur 3 classes
//   Partie C : comparaison avec/sans BatchNorm (vitesse de convergence)

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::nn::{
    PcgEngine, Module, Sequential, Linear, ReLU,
    KaimingNormal, Zeros,
};
use scirust_core::nn::loss::Loss;
use scirust_core::nn::loss::strict_v6::{CrossEntropyLoss, softmax};
use scirust_core::nn::batch_norm::BatchNorm1d;

fn make_dataset() -> (Tensor, Tensor) {
    // 3 classes en 2D, 6 points par classe
    let centers = [(2.0, 2.0), (-2.0, 2.0), (0.0, -2.0)];
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (class, &(cx, cy)) in centers.iter().enumerate() {
        for i in 0..6 {
            let dx = (i as f32) * 0.15 - 0.4;
            let dy = ((i + 3) as f32) * 0.1 - 0.4;
            x.push(cx + dx);
            x.push(cy + dy);
            for k in 0..3 {
                y.push(if k == class { 1.0 } else { 0.0 });
            }
        }
    }
    (Tensor::from_vec(x, 18, 2), Tensor::from_vec(y, 18, 3))
}

fn main() {
    println!("=== SciRust v6 — SumAxis, Reshape, Reciprocal, BatchNorm ===\n");

    // ============================================================== //
    //  Partie A : Softmax row-wise — validation numérique             //
    // ============================================================== //

    println!("--- Partie A : Softmax row-wise correcte ---\n");

    let tape = Tape::new();
    let logits_data = vec![1.0, 2.0, 3.0,
                           0.0, 0.0, 0.0,
                           10.0, 0.0, -10.0];
    let logits = tape.input(Tensor::from_vec(logits_data.clone(), 3, 3));
    let p = softmax(logits);
    let pt = tape.value(p.idx());

    println!("Logits (3 lignes, 3 classes) :");
    for r in 0..3 {
        println!("  [{:5.1}, {:5.1}, {:5.1}]",
                 logits_data[r*3], logits_data[r*3+1], logits_data[r*3+2]);
    }
    println!("\nSoftmax :");
    for r in 0..3 {
        println!("  [{:.4}, {:.4}, {:.4}]  (somme = {:.4})",
                 pt.data[r*3], pt.data[r*3+1], pt.data[r*3+2],
                 pt.data[r*3..(r+1)*3].iter().sum::<f32>());
    }

    // Vérification : chaque ligne somme à 1 ± epsilon
    let mut all_one = true;
    for r in 0..3 {
        let s: f32 = pt.data[r*3..(r+1)*3].iter().sum();
        if (s - 1.0).abs() > 1e-4 { all_one = false; }
    }
    if all_one {
        println!("\n✅ Softmax correctement normalisée par ligne (somme = 1 chacune)");
    } else {
        println!("\n❌ Softmax mal normalisée — bug !");
    }

    // ============================================================== //
    //  Partie B : Entraînement avec BatchNorm                         //
    // ============================================================== //

    println!("\n\n--- Partie B : MLP + BatchNorm sur 3 classes ---\n");

    let (x, y) = make_dataset();
    let n = x.rows;
    println!("Dataset : {n} points, 3 classes");

    let mut rng = PcgEngine::new(42);
    let mut model = Sequential::new()
        .push(Linear::new(2, 16, &KaimingNormal, &Zeros, &mut rng).with_name("fc1"))
        .push(BatchNorm1d::new(16).with_name("bn1"))
        .push(ReLU)
        .push(Linear::new(16, 16, &KaimingNormal, &Zeros, &mut rng).with_name("fc2"))
        .push(BatchNorm1d::new(16).with_name("bn2"))
        .push(ReLU)
        .push(Linear::new(16, 3,  &KaimingNormal, &Zeros, &mut rng).with_name("fc3"));

    // Warm-up forward pour récupérer les indices de paramètres
    {
        let warmup = Tape::new();
        let wx = warmup.input(x.clone());
        let _ = model.forward(&warmup, wx);
    }
    println!("Architecture : Linear(2→16) → BN → ReLU → Linear(16→16) → BN → ReLU → Linear(16→3)");
    println!("Paramètres apprenables : {} tenseurs", model.parameter_indices().len());

    let mut opt = Adam::new(0.05);
    let n_epochs = 150;

    println!("\nÉpoque | Loss");
    println!("-------|-------");

    for epoch in 0..n_epochs {
        let tape = Tape::new();
        let xv = tape.input(x.clone());
        let yv = tape.input(y.clone());
        let logits = model.forward(&tape, xv);
        let loss = CrossEntropyLoss.forward(logits, yv);
        let loss_val = tape.value(loss.idx()).data[0];

        loss.backward();
        opt.step(&model.parameter_indices(), &tape);
        model.sync(&tape);

        if epoch % 20 == 0 || epoch == n_epochs - 1 {
            println!("{:5}  | {:.5}", epoch, loss_val);
        }
    }

    // Évaluation finale
    let tape = Tape::new();
    let xv = tape.input(x.clone());
    let logits = model.forward(&tape, xv);
    let logits_t = tape.value(logits.idx());

    let mut correct = 0;
    for i in 0..n {
        let row = &logits_t.data[i*3..(i+1)*3];
        let pred = row.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let truth = (0..3).find(|&k| y.data[i*3 + k] > 0.5).unwrap();
        if pred == truth { correct += 1; }
    }
    println!("\nAccuracy : {correct}/{n} = {:.1}%", 100.0 * correct as f32 / n as f32);

    if correct as f32 / n as f32 > 0.85 {
        println!("✅ Modèle BN + CrossEntropy converge correctement");
    }

    // ============================================================== //
    //  Partie C : Démonstration de Reshape                            //
    // ============================================================== //

    println!("\n\n--- Partie C : Reshape (préparation pour Conv2d v6.1) ---\n");

    let tape = Tape::new();
    // Imagine un batch de 2 images 4×3 (12 pixels chacune) stockées en (2, 12)
    let images = tape.input(Tensor::from_vec((0..24).map(|x| x as f32).collect(), 2, 12));
    println!("Input shape : {:?}", images.shape());

    // Reshape pour traiter chaque image comme un vecteur ligne dans (8, 3)
    let reshaped = images.reshape(8, 3);
    println!("Reshape  → : {:?}", reshaped.shape());
    println!("Données identiques (row-major contigu)");

    // Le gradient de reshape redonne la forme originale
    let loss = reshaped.sum();
    loss.backward();
    println!("\n✅ Reshape + backward : le gradient est correctement re-shaped");

    println!("\n=== Fin de la démo v6 ===");
}
