// examples/v7a_mnist_demo/src/main.rs
//
// Démo v7-A : MNIST réel, MLP, data parallelism.
//
// USAGE :
//
//   1. Télécharger MNIST (4 fichiers IDX) :
//        curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
//        curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
//        curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
//        curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
//        gunzip *.gz
//
//   2. Lancer :
//        cargo run --package v7a_mnist_demo --release -- ./mnist
//
//   Le mode --quick (par défaut si pas de fichiers MNIST) entraîne sur des
//   données synthétiques pour valider la mécanique sans télécharger MNIST.

use std::env;
use std::path::Path;
use std::time::Instant;

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::data::{DataLoader, Dataset, InMemoryDataset};
use scirust_core::data::mnist::MnistDataset;
use scirust_core::nn::{
    PcgEngine, Module, Sequential, Linear, ReLU,
    KaimingNormal, Zeros,
};
use scirust_core::nn::loss::Loss;
use scirust_core::nn::loss::strict_v6_1::CrossEntropyLossStable;
use scirust_core::nn::parallel::{Grads, ParallelStep, parallel_step};

// ================================================================== //
//  Construction du modèle                                              //
// ================================================================== //

fn build_model(rng: &mut PcgEngine) -> Sequential {
    Sequential::new()
        .push(Linear::new(28 * 28, 256, &KaimingNormal, &Zeros, rng).with_name("fc1"))
        .push(ReLU)
        .push(Linear::new(256, 128, &KaimingNormal, &Zeros, rng).with_name("fc2"))
        .push(ReLU)
        .push(Linear::new(128, 10, &KaimingNormal, &Zeros, rng).with_name("fc3"))
}

// ================================================================== //
//  Stepper pour data parallelism                                       //
// ================================================================== //

struct MnistStepper {
    model: Sequential,
}

impl ParallelStep for MnistStepper {
    fn step(&self, x: Tensor, y: Tensor) -> (f32, Grads) {
        // Chaque worker construit sa Tape, son propre clone de modèle
        let tape = Tape::new();
        let mut model = self.model.clone();  // requiert Clone sur Sequential
        let xv = tape.input(x);
        let yv = tape.input(y);
        let logits = model.forward(&tape, xv);
        let loss = CrossEntropyLossStable.forward(logits, yv);
        let loss_val = tape.value(loss.idx()).data[0];
        loss.backward();

        // Collecte les gradients dans l'ordre canonique du state_dict
        let mut grads = Vec::new();
        let state = model.state_dict();
        let indices = model.parameter_indices();
        // state_dict et parameter_indices sont dans le même ordre par construction
        for ((name, _), idx) in state.iter().zip(indices.iter()) {
            grads.push((name.clone(), tape.grad(*idx)));
        }
        (loss_val, grads)
    }

    fn box_clone(&self) -> Box<dyn ParallelStep> {
        Box::new(MnistStepper { model: self.model.clone() })
    }
}

// ================================================================== //
//  Application des gradients agrégés sur le modèle master              //
// ================================================================== //

fn apply_to_model(
    model: &mut Sequential,
    grads: &Grads,
    optimizer: &mut Adam,
) {
    // Astuce : on construit une Tape "fake" pour réutiliser l'API de l'optimizer.
    // Chaque param est poussé sur la tape avec son gradient prédéfini.
    let tape = Tape::new();
    let state = model.state_dict();

    let mut indices_for_step = Vec::with_capacity(state.len());
    for ((name, value), (g_name, g_tensor)) in state.iter().zip(grads.iter()) {
        assert_eq!(name, g_name, "ordre params/grads incohérent");
        let v = tape.input(value.clone());
        // Injecte le grad calculé directement
        tape.set_grad(v.idx(), g_tensor.clone());
        indices_for_step.push(v.idx());
    }

    optimizer.step(&indices_for_step, &tape);

    // Reconstruit le state_dict mis à jour et l'applique au modèle
    use std::collections::HashMap;
    let updated: HashMap<String, Tensor> = state.iter().zip(indices_for_step.iter())
        .map(|((name, _), idx)| (name.clone(), tape.value(*idx)))
        .collect();
    model.load_state_dict(&updated);
}

// ================================================================== //
//  Boucle d'entraînement                                               //
// ================================================================== //

fn train(
    train_ds: InMemoryDataset,
    test_ds:  InMemoryDataset,
    n_epochs: usize,
    batch_size: usize,
    n_workers: usize,
    lr: f32,
) {
    let mut rng = PcgEngine::new(42);
    let mut model = build_model(&mut rng);

    // Warm-up forward pour initialiser parameter_indices
    {
        let warmup_tape = Tape::new();
        let warmup_x = warmup_tape.input(Tensor::zeros(1, 28*28));
        let _ = model.forward(&warmup_tape, warmup_x);
    }
    let n_params = model.parameter_indices().len();
    println!("Modèle : 784 → 256 → 128 → 10 ({n_params} tenseurs apprenables)");

    let mut optimizer = Adam::new(lr);
    let mut loader = DataLoader::new(train_ds, batch_size, true, 42);
    let n_batches = loader.n_batches();

    println!("Training : {n_epochs} epochs × {n_batches} batches de {batch_size}");
    println!("Workers parallèles : {n_workers}\n");

    let total_start = Instant::now();

    for epoch in 0..n_epochs {
        loader.shuffle_epoch(epoch as u64);
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut count = 0;

        for (x_batch, y_batch) in loader.iter() {
            let stepper = MnistStepper { model: model.clone() };
            let (loss, grads) = parallel_step(&stepper, x_batch, y_batch, n_workers);
            apply_to_model(&mut model, &grads, &mut optimizer);
            epoch_loss += loss;
            count += 1;
        }

        let mean_loss = epoch_loss / count as f32;
        let elapsed = epoch_start.elapsed().as_secs_f32();
        let test_acc = evaluate(&mut model, &test_ds);
        println!("Epoch {epoch:2}/{n_epochs} | loss = {mean_loss:.4} | \
                  test acc = {test_acc:.2}% | {elapsed:.1}s");
    }

    let total = total_start.elapsed().as_secs_f32();
    println!("\nTotal training time : {total:.1}s");
}

// ================================================================== //
//  Évaluation                                                          //
// ================================================================== //

fn evaluate(model: &mut Sequential, ds: &InMemoryDataset) -> f32 {
    // On évalue par chunks de 256 pour gérer la mémoire
    let chunk = 256;
    let n = ds.len();
    let mut correct = 0;
    let mut idx = 0;

    while idx < n {
        let end = (idx + chunk).min(n);
        let actual = end - idx;
        let mut x_buf = Vec::with_capacity(actual * ds.x_features());
        let mut y_truth = Vec::with_capacity(actual);
        for i in idx..end {
            let (x, y) = ds.get(i);
            x_buf.extend(x.data);
            // Trouve la classe true depuis le one-hot
            let truth = y.data.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            y_truth.push(truth);
        }
        let x_tensor = Tensor::from_vec(x_buf, actual, ds.x_features());

        let tape = Tape::new();
        let xv = tape.input(x_tensor);
        let logits = model.forward(&tape, xv);
        let logits_t = tape.value(logits.idx());

        for i in 0..actual {
            let row = &logits_t.data[i * 10..(i + 1) * 10];
            let pred = row.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == y_truth[i] { correct += 1; }
        }
        idx = end;
    }
    100.0 * correct as f32 / n as f32
}

// ================================================================== //
//  Datasets synthétiques de fallback                                   //
// ================================================================== //

fn make_synthetic_mnist(n_per_class: usize, seed: u64) -> InMemoryDataset {
    let mut rng = PcgEngine::new(seed);
    let n_classes = 10;
    let n = n_per_class * n_classes;
    let mut x_data = Vec::with_capacity(n * 784);
    let mut y_data = vec![0.0f32; n * 10];

    // Pour chaque classe, on génère une "signature" : un patch carré centré
    // à une position dépendant de la classe.
    for class in 0..n_classes {
        for i in 0..n_per_class {
            let mut img = vec![0.05f32; 784];   // bruit faible partout
            // Patch carré 4×4 à une position class-dépendante
            let row_start = (class / 5) * 8 + 6;
            let col_start = (class % 5) * 5 + 4;
            for dy in 0..4 {
                for dx in 0..4 {
                    let r = row_start + dy;
                    let c = col_start + dx;
                    if r < 28 && c < 28 {
                        img[r * 28 + c] = 0.95 + rng.uniform(-0.05, 0.05);
                    }
                }
            }
            // Ajout de bruit
            for v in img.iter_mut() { *v += rng.uniform(-0.02, 0.02); }
            x_data.extend(img);

            let sample_idx = class * n_per_class + i;
            y_data[sample_idx * 10 + class] = 1.0;
            let _ = sample_idx;
        }
    }
    InMemoryDataset::new(x_data, y_data, 784, 10)
}

// ================================================================== //
//  Main                                                                //
// ================================================================== //

fn main() {
    println!("=== SciRust v7-A — MNIST training (data parallel) ===\n");

    let args: Vec<String> = env::args().collect();
    let mnist_dir = if args.len() > 1 { Some(args[1].clone()) } else { None };

    let (train_ds, test_ds) = match mnist_dir.as_ref()
        .and_then(|d| try_load_mnist(d).ok())
    {
        Some((tr, te)) => {
            println!("✅ MNIST réel chargé : {} train, {} test", tr.len(), te.len());
            (tr, te)
        }
        None => {
            println!("⚠️  MNIST non trouvé (ou chemin non fourni)");
            println!("   Fallback sur dataset synthétique 10 classes");
            println!("   Pour MNIST réel : cargo run --release -- /path/to/mnist\n");
            (
                make_synthetic_mnist(50, 42),    // 500 train
                make_synthetic_mnist(20, 999),   // 200 test
            )
        }
    };

    let n_workers = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(2);

    train(train_ds, test_ds, 5, 64, n_workers, 0.001);
}

fn try_load_mnist(dir: &str) -> std::io::Result<(InMemoryDataset, InMemoryDataset)> {
    let dir = Path::new(dir);
    let train_imgs = dir.join("train-images-idx3-ubyte");
    let train_lbls = dir.join("train-labels-idx1-ubyte");
    let test_imgs  = dir.join("t10k-images-idx3-ubyte");
    let test_lbls  = dir.join("t10k-labels-idx1-ubyte");

    let train = MnistDataset::load_idx(&train_imgs, &train_lbls)?;
    let test  = MnistDataset::load_idx(&test_imgs,  &test_lbls)?;

    Ok((train.into_in_memory(), test.into_in_memory()))
}
