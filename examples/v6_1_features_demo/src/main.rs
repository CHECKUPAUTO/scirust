// examples/v6_1_features_demo/src/main.rs
//
// Démo v6.1 : MINI-CNN COMPLET en pure Rust.
//
// Architecture :
//   Input  : (B, 1·8·8)
//   Conv2d(1 → 4, kernel=3, same)  → (B, 4·8·8)
//   ReLU
//   MaxPool2d(2, 2)                 → (B, 4·4·4)
//   Conv2d(4 → 8, kernel=3, same)  → (B, 8·4·4)
//   ReLU
//   MaxPool2d(2, 2)                 → (B, 8·2·2)
//   Linear(32 → 3)                  → (B, 3)
//
// Données : 3 classes synthétiques en 8×8
//   Classe 0 : "+" (croix)
//   Classe 1 : "-" (barre horizontale)
//   Classe 2 : "O" (carré)

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::nn::{
    PcgEngine, Module, Sequential, Linear, ReLU,
    KaimingNormal, Zeros,
};
use scirust_core::nn::loss::Loss;
use scirust_core::nn::loss::strict_v6_1::CrossEntropyLossStable;
use scirust_core::nn::conv2d::Conv2d;
use scirust_core::nn::pool::MaxPool2d;
use scirust_core::nn::conv_utils::Padding;

// ---------- Génération des chiffres synthétiques 8×8 ---------- //

fn make_plus(noise: f32, rng: &mut PcgEngine) -> Vec<f32> {
    // Croix : ligne 3-4 horizontale, colonne 3-4 verticale
    let mut img = vec![0.0f32; 64];
    for j in 0..8 { img[3 * 8 + j] = 1.0; img[4 * 8 + j] = 1.0; }
    for i in 0..8 { img[i * 8 + 3] = 1.0; img[i * 8 + 4] = 1.0; }
    for v in img.iter_mut() { *v += rng.uniform(-noise, noise); }
    img
}

fn make_minus(noise: f32, rng: &mut PcgEngine) -> Vec<f32> {
    let mut img = vec![0.0f32; 64];
    for j in 0..8 { img[3 * 8 + j] = 1.0; img[4 * 8 + j] = 1.0; }
    for v in img.iter_mut() { *v += rng.uniform(-noise, noise); }
    img
}

fn make_o(noise: f32, rng: &mut PcgEngine) -> Vec<f32> {
    // Carré : bordure 6×6 dans le 8×8
    let mut img = vec![0.0f32; 64];
    for k in 1..=6 {
        img[1 * 8 + k] = 1.0;          // ligne haute
        img[6 * 8 + k] = 1.0;          // ligne basse
        img[k * 8 + 1] = 1.0;          // colonne gauche
        img[k * 8 + 6] = 1.0;          // colonne droite
    }
    for v in img.iter_mut() { *v += rng.uniform(-noise, noise); }
    img
}

fn make_dataset(n_per_class: usize, noise: f32, seed: u64)
    -> (Tensor, Tensor)
{
    let mut rng = PcgEngine::new(seed);
    let n = 3 * n_per_class;
    let mut x_data = Vec::with_capacity(n * 64);
    let mut y_data = Vec::with_capacity(n * 3);

    for class in 0..3 {
        for _ in 0..n_per_class {
            let img = match class {
                0 => make_plus(noise, &mut rng),
                1 => make_minus(noise, &mut rng),
                2 => make_o(noise, &mut rng),
                _ => unreachable!(),
            };
            x_data.extend(img);
            for k in 0..3 {
                y_data.push(if k == class { 1.0 } else { 0.0 });
            }
        }
    }
    (Tensor::from_vec(x_data, n, 64), Tensor::from_vec(y_data, n, 3))
}

fn main() {
    println!("=== SciRust v6.1 — Mini-LeNet en pur Rust ===\n");

    // ---- Dataset ---- //
    let (x_train, y_train) = make_dataset(20, 0.05, 42);
    let n = x_train.rows;
    println!("Dataset : {n} échantillons (3 classes × 20), bruit léger");

    // ---- Architecture ---- //
    let mut rng = PcgEngine::new(7);
    let mut conv1 = Conv2d::new(1, 4, 3, 1, Padding::Same,
        &KaimingNormal, Some(&Zeros), &mut rng).input_dims(8, 8).with_name("conv1");
    let mut pool1 = MaxPool2d::new(2, 2).input_shape(4, 8, 8);
    let mut conv2 = Conv2d::new(4, 8, 3, 1, Padding::Same,
        &KaimingNormal, Some(&Zeros), &mut rng).input_dims(4, 4).with_name("conv2");
    let mut pool2 = MaxPool2d::new(2, 2).input_shape(8, 4, 4);
    let mut fc    = Linear::new(8 * 2 * 2, 3, &KaimingNormal, &Zeros, &mut rng).with_name("fc");

    println!("Architecture :");
    println!("  Conv2d(1 → 4, k=3, same) → ReLU → MaxPool(2,2)");
    println!("  Conv2d(4 → 8, k=3, same) → ReLU → MaxPool(2,2)");
    println!("  Linear(32 → 3)");

    // Comme nos couches sont déclarées séparément (pas dans un Sequential),
    // on construit la fonction forward à la main.
    fn forward_model<'t>(
        tape: &'t Tape,
        x: scirust_core::autodiff::reverse::Var<'t>,
        conv1: &mut Conv2d, pool1: &mut MaxPool2d,
        conv2: &mut Conv2d, pool2: &mut MaxPool2d,
        fc:    &mut Linear,
    ) -> scirust_core::autodiff::reverse::Var<'t> {
        let h1 = conv1.forward(tape, x).relu();
        let p1 = pool1.forward(tape, h1);
        let h2 = conv2.forward(tape, p1).relu();
        let p2 = pool2.forward(tape, h2);
        fc.forward(tape, p2)
    }

    // ---- Entraînement ---- //
    let mut opt = Adam::new(0.01);
    let n_epochs = 50;

    println!("\nÉpoque | Loss   | Acc");
    println!("-------|--------|-----");

    for epoch in 0..n_epochs {
        let tape = Tape::new();
        let xv = tape.input(x_train.clone());
        let yv = tape.input(y_train.clone());

        let logits = forward_model(&tape, xv, &mut conv1, &mut pool1,
                                    &mut conv2, &mut pool2, &mut fc);
        let loss = CrossEntropyLossStable.forward(logits, yv);
        let loss_val = tape.value(loss.idx()).data[0];

        // Eval accuracy
        let logits_val = tape.value(
            forward_model(&tape, tape.input(x_train.clone()),
                         &mut conv1, &mut pool1,
                         &mut conv2, &mut pool2, &mut fc).idx()
        );
        let mut correct = 0;
        for i in 0..n {
            let row = &logits_val.data[i*3..(i+1)*3];
            let pred = row.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let truth = (0..3).find(|&k| y_train.data[i*3 + k] > 0.5).unwrap();
            if pred == truth { correct += 1; }
        }

        loss.backward();
        // Collecte les paramètres de toutes les couches
        let mut params = Vec::new();
        params.extend(conv1.parameter_indices());
        params.extend(conv2.parameter_indices());
        params.extend(fc.parameter_indices());
        opt.step(&params, &tape);
        conv1.sync(&tape); conv2.sync(&tape); fc.sync(&tape);

        if epoch % 5 == 0 || epoch == n_epochs - 1 {
            let acc = 100.0 * correct as f32 / n as f32;
            println!("{:5}  | {:.4} | {:.1}%", epoch, loss_val, acc);
        }
    }

    // ---- Évaluation finale ---- //
    let (x_test, y_test) = make_dataset(10, 0.05, 999);
    let n_test = x_test.rows;

    let tape = Tape::new();
    let xv = tape.input(x_test.clone());
    let logits = forward_model(&tape, xv, &mut conv1, &mut pool1,
                                &mut conv2, &mut pool2, &mut fc);
    let logits_val = tape.value(logits.idx());

    let mut correct = 0;
    let mut confusion = [[0usize; 3]; 3];
    for i in 0..n_test {
        let row = &logits_val.data[i*3..(i+1)*3];
        let pred = row.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let truth = (0..3).find(|&k| y_test.data[i*3 + k] > 0.5).unwrap();
        if pred == truth { correct += 1; }
        confusion[truth][pred] += 1;
    }

    let test_acc = 100.0 * correct as f32 / n_test as f32;
    println!("\n=== Résultats sur dataset de test (graine différente) ===");
    println!("Test accuracy : {correct}/{n_test} = {test_acc:.1}%");
    println!("\nMatrice de confusion (lignes = truth, colonnes = pred) :");
    println!("       +    -    O");
    let labels = ["+", "-", "O"];
    for i in 0..3 {
        print!("  {} :", labels[i]);
        for j in 0..3 {
            print!(" {:4}", confusion[i][j]);
        }
        println!();
    }

    if test_acc > 80.0 {
        println!("\n✅ Mini-LeNet converge sur les 3 classes — pari gagné");
        println!("   100 % pure Rust : pas une ligne de C++, pas de Python");
    } else {
        println!("\n⚠️  Test accuracy faible — vérifier la convergence");
    }
}
