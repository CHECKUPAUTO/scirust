// examples/mnist_classifier/src/main.rs
//
// MNIST classifier avec SciRust v11.1
// Architecture : Linear(784→256) → ReLU → Linear(256→10) [logits]
// Loss : CrossEntropy (multi-batch stable)
// Optim : Adam
//
// Critère de succès : > 90% accuracy sur test set (ou subset rapide).

use scirust_core::autodiff::reverse::{Tape};
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::nn::{Module, Sequential, Linear, ReLU, PcgEngine, KaimingNormal, Zeros, CrossEntropyLoss, Loss};
use scirust_core::data::{MnistDataset, DataLoader};

fn main() {
    println!("=== SciRust v11.1 — MNIST Classifier ===\n");

    // -------- Chargement MNIST -------- //
    let data_dir = std::env::var("MNIST_DIR").unwrap_or_else(|_| "/root/scirust/data/mnist".to_string());
    println!("Chargement MNIST depuis {}...", data_dir);

    let train = MnistDataset::load_idx(
        format!("{}/train-images-idx3-ubyte", data_dir),
        format!("{}/train-labels-idx1-ubyte", data_dir),
    ).expect("Échec chargement train MNIST");

    let test = MnistDataset::load_idx(
        format!("{}/t10k-images-idx3-ubyte", data_dir),
        format!("{}/t10k-labels-idx1-ubyte", data_dir),
    ).expect("Échec chargement test MNIST");

    println!("Train : {} images {}x{}", train.n, train.h, train.w);
    println!("Test  : {} images {}x{}\n", test.n, test.h, test.w);

    // Sous-échantillonnage rapide pour test (optionnel)
    let max_train = std::env::var("MNIST_MAX_TRAIN")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(train.n);
    let max_test = std::env::var("MNIST_MAX_TEST")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(test.n);

    let train_small = train.subsample(max_train);
    let test_small = test.subsample(max_test);

    println!("Utilisé : {} train, {} test\n", train_small.n_samples(), test_small.n_samples());

    // -------- Modèle : MLP 784→256→10 -------- //
    let input_dim = train.h * train.w;  // 784
    let hidden_dim = 256;
    let n_classes = 10;

    let mut rng = PcgEngine::new(42);
    let mut model = Sequential::new()
        .add(Linear::new(input_dim, hidden_dim, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(hidden_dim, n_classes, &KaimingNormal, &Zeros, &mut rng));

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.001);

    let batch_size = 64;
    let n_epochs = 5;

    println!("Architecture : Linear({}→{}) → ReLU → Linear({}→{})", input_dim, hidden_dim, hidden_dim, n_classes);
    println!("Entraînement : {} epochs, batch={}, Adam(lr=0.001)\n", n_epochs, batch_size);

    // -------- Boucle d'entraînement -------- //
    let mut train_loader = DataLoader::new(train_small, batch_size, true, 42);

    for epoch in 0..n_epochs {
        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        train_loader.shuffle_epoch(epoch as u64);

        for (x_batch, y_batch) in train_loader.iter() {
            let tape = Tape::new();
            let x = tape.input(x_batch);
            let target = tape.input(y_batch);

            let logits = model.forward(&tape, x);
            let loss = loss_fn.forward(&tape, logits, target);
            tape.backward(loss.idx());

            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);

            epoch_loss += tape.value(loss.idx()).data[0];
            n_batches += 1;
        }

        let avg_loss = epoch_loss / n_batches as f32;
        println!("  Epoch {:>2} : loss = {:.4} ({} batches)", epoch + 1, avg_loss, n_batches);
    }

    // -------- Évaluation -------- //
    println!("\nÉvaluation sur test set...");
    let mut correct = 0;
    let mut total = 0;

    let mut test_loader = DataLoader::new(test_small, batch_size, false, 42);

    for (x_batch, y_batch) in test_loader.iter() {
        let tape = Tape::new();
        let x = tape.input(x_batch);
        let logits = model.forward(&tape, x);
        let scores = tape.value(logits.idx());

        let (bs, _) = scores.shape();
        for i in 0..bs {
            let mut max_score = scores.data[i * n_classes];
            let mut pred_class = 0usize;
            for c in 1..n_classes {
                if scores.data[i * n_classes + c] > max_score {
                    max_score = scores.data[i * n_classes + c];
                    pred_class = c;
                }
            }

            let true_class = y_batch.data[i * n_classes..(i + 1) * n_classes]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if pred_class == true_class { correct += 1; }
            total += 1;
        }
    }

    let accuracy = correct as f32 / total as f32 * 100.0;
    println!("\n=== RÉSULTAT ===");
    println!("  Accuracy : {:.2}% ({}/{})", accuracy, correct, total);

    if accuracy >= 90.0 {
        println!("\n✅ SUCCÈS — SciRust v11.1 classifie MNIST correctement.");
        std::process::exit(0);
    } else if accuracy >= 85.0 {
        println!("\n⚠️  PARTIEL — {:.2}% est acceptable mais < 90%. Augmenter epochs ou lr.", accuracy);
        std::process::exit(0);
    } else {
        println!("\n❌ ÉCHEC — Convergence insuffisante. Vérifier lr, architecture, ou données.");
        std::process::exit(1);
    }
}
