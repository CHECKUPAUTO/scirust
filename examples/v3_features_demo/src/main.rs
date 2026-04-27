// examples/v3_features_demo/src/main.rs
//
// Démo v3 : entraînement complet d'une régression linéaire avec biais.
// Met en œuvre :
//   1. Broadcasting (ajout du biais b: (1,1) à pred: (N,1))
//   2. Optimiseur Adam
//   3. Tape rebuilt à chaque époque (pattern PyTorch)
//
// La cible est y = 2 x₀ + 3 x₁ + 0.5
// Initialisation aléatoire des paramètres → convergence visible.

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::optim::{Adam, Sgd, Optimizer};

fn main() {
    println!("=== SciRust v3 — Entraînement régression linéaire ===\n");

    // ----- Données ---------------------------------------------------- //
    // 8 échantillons, 2 features
    let x_data = vec![
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
        1.0, 2.0,
        0.5, 0.5,
        2.0, 2.0,
        3.0, 1.0,
    ];
    // y = 2 x₀ + 3 x₁ + 0.5
    let y_data: Vec<f32> = x_data.chunks(2)
        .map(|x| 2.0 * x[0] + 3.0 * x[1] + 0.5)
        .collect();

    println!("Cible : y = 2·x₀ + 3·x₁ + 0.5");
    println!("Échantillons : {}\n", y_data.len());

    // ----- Paramètres initiaux (volontairement faux) ------------------- //
    let mut w_data = vec![0.0_f32, 0.0];   // poids (2,1)
    let mut b_data = vec![0.0_f32];        // biais (1,1)

    // ----- Optimiseur ------------------------------------------------- //
    let mut opt = Adam::new(0.1);
    let n_epochs = 200;
    let n = y_data.len();

    println!("Époque | Loss   | w₀     | w₁     | b");
    println!("-------|--------|--------|--------|------");

    for epoch in 0..n_epochs {
        // ➜ Tape neuf à chaque époque (les valeurs précédentes sont mortes)
        let tape = Tape::new();

        // Inputs
        let x = tape.input(Tensor::from_vec(x_data.clone(), n, 2));
        let y_true = tape.input(Tensor::from_vec(y_data.clone(), n, 1));
        let w = tape.input(Tensor::from_vec(w_data.clone(), 2, 1));
        let b = tape.input(Tensor::from_vec(b_data.clone(), 1, 1));

        // Forward : pred = x @ w + b  (avec broadcasting du biais)
        let xw = x.matmul(w);
        let pred = xw.add_broadcast(b);  // pred:(n,1) + b:(1,1) → (n,1)

        // MSE = sum((pred - y_true)²) / n
        let diff = pred.sub(y_true);
        let sq   = diff.hadamard(diff);
        let loss = sq.sum().scale(1.0 / n as f32);

        let loss_val = tape.value(loss.idx()).data[0];

        // Backward
        loss.backward();

        // Step
        opt.step(&[w.idx(), b.idx()], &tape);

        // Récupérer les nouvelles valeurs (avant que le tape soit dropé)
        w_data = tape.value(w.idx()).data;
        b_data = tape.value(b.idx()).data;

        if epoch % 20 == 0 || epoch == n_epochs - 1 {
            println!("{:4}   | {:.4} | {:.4} | {:.4} | {:.4}",
                     epoch, loss_val, w_data[0], w_data[1], b_data[0]);
        }
    }

    println!("\n=== Résultats finaux ===");
    println!("w = [{:.3}, {:.3}]   (cible : [2.0, 3.0])", w_data[0], w_data[1]);
    println!("b = {:.3}            (cible : 0.5)", b_data[0]);

    let err_w0 = (w_data[0] - 2.0).abs();
    let err_w1 = (w_data[1] - 3.0).abs();
    let err_b  = (b_data[0] - 0.5).abs();
    println!("\nErreurs : |Δw₀|={err_w0:.4}  |Δw₁|={err_w1:.4}  |Δb|={err_b:.4}");

    if err_w0 < 0.05 && err_w1 < 0.05 && err_b < 0.05 {
        println!("✅ Convergence réussie !");
    } else {
        println!("⚠️  Convergence partielle (relancer avec plus d'époques)");
    }

    // ----- Comparaison rapide SGD vs Adam ----------------------------- //
    println!("\n=== SGD vs Adam (50 époques, mêmes données) ===");
    compare_optimizers(&x_data, &y_data, n);
}

fn compare_optimizers(x_data: &[f32], y_data: &[f32], n: usize) {
    for label in &["SGD", "Adam"] {
        let mut w = vec![0.0_f32, 0.0];
        let mut b = vec![0.0_f32];

        let mut sgd = Sgd::new(0.05).with_momentum(0.9);
        let mut adam = Adam::new(0.1);

        let mut last_loss = 0.0;
        for _ in 0..50 {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(x_data.to_vec(), n, 2));
            let y_true = tape.input(Tensor::from_vec(y_data.to_vec(), n, 1));
            let wv = tape.input(Tensor::from_vec(w.clone(), 2, 1));
            let bv = tape.input(Tensor::from_vec(b.clone(), 1, 1));

            let pred = x.matmul(wv).add_broadcast(bv);
            let diff = pred.sub(y_true);
            let loss = diff.hadamard(diff).sum().scale(1.0 / n as f32);

            last_loss = tape.value(loss.idx()).data[0];
            loss.backward();

            match *label {
                "SGD"  => sgd.step(&[wv.idx(), bv.idx()], &tape),
                "Adam" => adam.step(&[wv.idx(), bv.idx()], &tape),
                _ => unreachable!(),
            }
            w = tape.value(wv.idx()).data;
            b = tape.value(bv.idx()).data;
        }
        println!("  {label}: loss finale = {last_loss:.4}, w = [{:.3}, {:.3}], b = {:.3}",
                 w[0], w[1], b[0]);
    }
}
