// examples/v10a_augmented_mnist/src/main.rs
//
// Démo v10-A : entraîne deux MLPs sur MNIST, l'un sans augmentation,
// l'autre avec un pipeline d'augmentation, et mesure l'écart d'accuracy.
//
// On utilise volontairement un MLP (pas un CNN) pour deux raisons :
//   1. Plus rapide à entraîner sur MNIST (~30s par expérience)
//   2. Sensible aux augmentations qui dégradent la pixel position
//      (sans CNN's translation invariance, l'aug compte plus)
//
// Pipeline d'augmentation choisi pour MNIST :
//   - RandomCrop avec padding 2 (déplacement spatial subtil)
//   - AddGaussianNoise(0.05) (robustesse à la qualité d'image)
//
// On NE flippe PAS horizontalement parce que les chiffres MNIST ne sont
// pas symétriques (un 3 flippé n'est plus un 3).
//
// Sur MNIST l'écart attendu est petit (~0.3-0.5%) — sur CIFAR il serait
// de ~3-5%. C'est documenté dans la sortie.

use std::time::Instant;

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::optim::{Adam, Optimizer, apply_schedule};
use scirust_core::autodiff::scheduler::{CosineAnnealing, LrSchedule};
use scirust_core::data::{DataLoader, Dataset, InMemoryDataset};
use scirust_core::data::augment::{
    AugmentedDataset, Compose, RandomCrop, AddGaussianNoise, Normalize,
};
use scirust_core::data::mnist::MnistDataset;
use scirust_core::nn::{
    PcgEngine, Module, Sequential, Linear, ReLU, Dropout,
    KaimingNormal, Zeros,
};
use scirust_core::nn::loss::{Loss, strict::CrossEntropyLoss};

const N_EPOCHS: usize = 5;
const BATCH:    usize = 64;
const LR_INIT:  f32   = 0.001;
const LR_MIN:   f32   = 0.0001;

// ================================================================== //
//  Architecture commune                                                //
// ================================================================== //

fn build_model(rng: &mut PcgEngine) -> Sequential {
    Sequential::new()
        .push(Linear::new(784, 256, &KaimingNormal, &Zeros, rng).with_name("fc1"))
        .push(ReLU)
        .push(Dropout::new(0.2, rng.next_u32() as u64))
        .push(Linear::new(256, 128, &KaimingNormal, &Zeros, rng).with_name("fc2"))
        .push(ReLU)
        .push(Linear::new(128, 10, &KaimingNormal, &Zeros, rng).with_name("fc3"))
}

// ================================================================== //
//  Boucle d'entraînement générique                                    //
// ================================================================== //

fn train(
    label:        &str,
    train_ds:     impl Dataset + 'static,
    test_ds:      &InMemoryDataset,
    seed:         u64,
) -> f32 {
    let mut rng = PcgEngine::new(seed);
    let mut model = build_model(&mut rng);
    let mut opt = Adam::new(LR_INIT);

    let n_train = train_ds.len();
    let total_steps = (n_train / BATCH) * N_EPOCHS;
    let scheduler = CosineAnnealing::new(LR_INIT, LR_MIN, total_steps);

    let mut loader = DataLoader::new(train_ds, BATCH, true, seed);
    let mut step = 0;

    println!("\n[{label}] Entraînement {N_EPOCHS} epochs, {n_train} samples/epoch");
    let t_start = Instant::now();

    for epoch in 0..N_EPOCHS {
        loader.shuffle_epoch(epoch as u64);
        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        for (x_batch, y_batch) in loader.iter() {
            apply_schedule(&scheduler, &mut opt, step);

            let tape = Tape::new();
            let xv = tape.input(x_batch);
            let yv = tape.input(y_batch);

            let logits = model.forward(&tape, xv);
            let loss = CrossEntropyLoss.forward(logits, yv);
            let loss_idx = loss.idx();
            loss.backward();

            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);

            epoch_loss += tape.value(loss_idx).data[0];
            n_batches += 1;
            step += 1;
        }

        let avg = epoch_loss / n_batches as f32;
        let acc = evaluate(&mut model, test_ds);
        println!("  epoch {} : loss = {:.4}  test_acc = {:.2}%  lr = {:.5}",
                 epoch, avg, acc * 100.0, opt.lr());
    }

    let elapsed = t_start.elapsed().as_secs_f32();
    let final_acc = evaluate(&mut model, test_ds);
    println!("[{label}] terminé en {elapsed:.1}s — accuracy finale = {:.2}%",
             final_acc * 100.0);
    final_acc
}

// ================================================================== //
//  Évaluation                                                         //
// ================================================================== //

fn evaluate(model: &mut Sequential, test_ds: &InMemoryDataset) -> f32 {
    // Eval mode (désactive Dropout)
    model.train(false);

    let n = test_ds.len();
    let batch = 256;
    let mut correct = 0;

    let mut i = 0;
    while i < n {
        let end = (i + batch).min(n);
        let bs = end - i;
        // Construire un batch
        let mut x_data = Vec::with_capacity(bs * 784);
        let mut y_labels = Vec::with_capacity(bs);
        for j in i..end {
            let (x, y) = test_ds.get(j);
            x_data.extend_from_slice(&x.data);
            // y est one-hot : trouver l'index du 1
            let label = y.iter().position(|&v| v > 0.5).unwrap();
            y_labels.push(label);
        }
        let x_batch = Tensor::from_vec(x_data, bs, 784);

        let tape = Tape::new();
        let xv = tape.input(x_batch);
        let logits = model.forward(&tape, xv);
        let probs = tape.value(logits.idx());

        for (k, &true_label) in y_labels.iter().enumerate() {
            let row = &probs.data[k * 10..(k + 1) * 10];
            let pred = row.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap();
            if pred == true_label { correct += 1; }
        }

        i = end;
    }

    model.train(true);
    correct as f32 / n as f32
}

// ================================================================== //
//  Main                                                                //
// ================================================================== //

fn main() {
    println!("=== SciRust v10-A — Augmented MNIST training ===");
    println!("Comparaison avec/sans augmentation pipeline\n");

    let mnist_path = std::env::args().nth(1)
        .unwrap_or_else(|| "./mnist".to_string());

    println!("Chargement MNIST depuis {mnist_path}/...");
    let train_raw = match MnistDataset::load_idx(
        &format!("{mnist_path}/train-images-idx3-ubyte"),
        &format!("{mnist_path}/train-labels-idx1-ubyte"),
    ) {
        Ok(ds) => ds.into_in_memory(),
        Err(e) => {
            eprintln!("❌ Impossible de charger MNIST : {e}");
            eprintln!("   Télécharge le dataset depuis :");
            eprintln!("   https://storage.googleapis.com/cvdf-datasets/mnist/");
            eprintln!("   Voir docs/MNIST.md pour les instructions détaillées.");
            std::process::exit(1);
        }
    };
    let test_raw = match MnistDataset::load_idx(
        &format!("{mnist_path}/t10k-images-idx3-ubyte"),
        &format!("{mnist_path}/t10k-labels-idx1-ubyte"),
    ) {
        Ok(ds) => ds.into_in_memory(),
        Err(e) => { eprintln!("❌ test set : {e}"); std::process::exit(1); }
    };

    // Pour accélérer la démo, on prend un sous-ensemble du train set.
    // Sur le full train set (60k), l'écart d'augmentation est plus mesurable
    // mais l'expérience prend ~5min. Sur 10k, ça prend ~1min et l'écart
    // est encore visible.
    let train_subset = train_raw.subsample(10_000, 42);
    println!("✅ Train: {} samples (sous-échantillonné), Test: {} samples\n",
             train_subset.len(), test_raw.len());

    // ---- Expérience 1 : sans augmentation ---- //
    let acc_baseline = train(
        "BASELINE (sans aug)",
        train_subset.clone(),
        &test_raw,
        100,
    );

    // ---- Expérience 2 : avec augmentation ---- //
    let aug_pipeline = Compose::new()
        .add(RandomCrop::new(28, 28, 2))             // déplacement subtil
        .add(AddGaussianNoise::new(0.05))            // robustesse au bruit
        .add(Normalize::mnist());                    // normalisation finale

    let aug_train = AugmentedDataset::from_pipeline(
        train_subset, aug_pipeline, 1, 28, 28,
    ).with_seed(7);

    let acc_augmented = train(
        "AUGMENTÉ",
        aug_train,
        &test_raw,
        100,   // même seed pour init weights identiques
    );

    // ---- Comparaison finale ---- //
    println!("\n=== RÉSULTATS ===");
    println!("  Sans augmentation : {:.2}%", acc_baseline  * 100.0);
    println!("  Avec augmentation : {:.2}%", acc_augmented * 100.0);
    let delta = (acc_augmented - acc_baseline) * 100.0;
    if delta > 0.0 {
        println!("  Δ = +{delta:.2} pts");
        if delta > 0.3 {
            println!("\n✅ L'augmentation aide (gain > 0.3 pt sur MNIST)");
        } else {
            println!("\n⚠️  Gain mineur. Sur MNIST l'écart est faible — sur CIFAR-10 il");
            println!("   serait typiquement 3-5 pts. Le pipeline marche ; il faudrait");
            println!("   un dataset plus complexe pour mieux le mettre en valeur.");
        }
    } else {
        println!("  Δ = {delta:.2} pts");
        println!("\n⚠️  L'augmentation n'a pas aidé sur ce run.");
        println!("   Causes possibles : sous-échantillon trop petit, hyperparamètres,");
        println!("   variance entre runs. Re-essayer avec full train set + seed différent.");
    }
}
