// examples/cifar10_classifier/src/main.rs
//
// CIFAR-10 classifier CNN avec SciRust v11.4
// Architecture : Conv(3→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → Linear(4096→256) → ReLU → Linear(256→10)
// Loss : CrossEntropy
// Optim : Adam
//
// Critère de succès : > 60% accuracy sur test set (raisonnable pour un CNN simple CPU).

use scirust_core::autodiff::reverse::Tape;
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::nn::{Module, Sequential, Linear, ReLU, Conv2d, MaxPool2d, PcgEngine, KaimingNormal, Zeros, CrossEntropyLoss, Loss, Padding};
use scirust_core::data::{Cifar10Dataset, DataLoader};

fn main() {
    println!("=== SciRust v11.4 — CIFAR-10 Classifier (CNN) ===\n");

    let data_dir = std::env::var("CIFAR10_DIR").unwrap_or_else(|_| "/root/scirust/data/cifar-10-batches-bin".to_string());
    println!("Chargement CIFAR-10 depuis {}...", data_dir);

    let dataset = match Cifar10Dataset::load(&data_dir) {
        Ok(ds) => ds,
        Err(e) => {
            println!("Échec chargement CIFAR-10 : {}", e);
            println!("Veuillez télécharger CIFAR-10 depuis https://www.cs.toronto.edu/~kriz/cifar.html");
            println!("et extraire dans {} (ou définir CIFAR10_DIR)", data_dir);
            std::process::exit(1);
        }
    };

    println!("Train : {} images 32×32×3", dataset.n_train);
    println!("Test  : {} images 32×32×3\n", dataset.n_test);

    let max_train = std::env::var("CIFAR10_MAX_TRAIN")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(dataset.n_train);
    let max_test = std::env::var("CIFAR10_MAX_TEST")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(dataset.n_test);

    let train_ds = dataset.subsample_train(max_train);
    let test_ds = dataset.subsample_test(max_test);

    println!("Utilisé : {} train, {} test\n", train_ds.n_samples(), test_ds.n_samples());

    // -------- Modèle CNN -------- //
    let mut rng = PcgEngine::new(42);
    let mut model = Sequential::new()
        // Block 1 : Conv(3→32, 3×3, same) → ReLU → MaxPool(2×2)
        .add(Conv2d::new(3, 32, 3, 1, Padding::Same, &KaimingNormal, Some(&Zeros), &mut rng).input_dims(32, 32))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2).input_shape(32, 32, 32))
        // Block 2 : Conv(32→64, 3×3, same) → ReLU → MaxPool(2×2)
        .add(Conv2d::new(32, 64, 3, 1, Padding::Same, &KaimingNormal, Some(&Zeros), &mut rng).input_dims(16, 16))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2).input_shape(64, 16, 16))
        // Classifier : Linear(64×8×8=4096 → 256) → ReLU → Linear(256 → 10)
        .add(Linear::new(4096, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng));

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.001);

    let batch_size = 128;
    let n_epochs = 10;

    println!("Architecture : CNN simple (Conv→Pool→Conv→Pool→MLP)");
    println!("Entraînement : {} epochs, batch={}, Adam(lr=0.001)\n", n_epochs, batch_size);

    // -------- Boucle d'entraînement -------- //
    let mut train_loader = DataLoader::new(train_ds, batch_size, true, 42);

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

    let mut test_loader = DataLoader::new(test_ds, batch_size, false, 42);

    for (x_batch, y_batch) in test_loader.iter() {
        let tape = Tape::new();
        let x = tape.input(x_batch);
        let logits = model.forward(&tape, x);
        let scores = tape.value(logits.idx());

        let (bs, _) = scores.shape();
        for i in 0..bs {
            let mut max_score = scores.data[i * 10];
            let mut pred_class = 0usize;
            for c in 1..10 {
                if scores.data[i * 10 + c] > max_score {
                    max_score = scores.data[i * 10 + c];
                    pred_class = c;
                }
            }

            let true_class = y_batch.data[i * 10..(i + 1) * 10]
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

    if accuracy >= 60.0 {
        println!("\n✅ SUCCÈS — SciRust v11.4 classifie CIFAR-10 correctement.");
        std::process::exit(0);
    } else if accuracy >= 50.0 {
        println!("\n⚠️  PARTIEL — {:.2}% est acceptable mais < 60%. Augmenter epochs ou lr.", accuracy);
        std::process::exit(0);
    } else {
        println!("\n❌ ÉCHEC — Convergence insuffisante. Vérifier lr, architecture, ou données.");
        std::process::exit(1);
    }
}
