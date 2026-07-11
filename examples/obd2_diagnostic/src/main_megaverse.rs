// examples/obd2_diagnostic/src/main_megaverse.rs
//
// SciRust — Entraînement MÉGAVERSE d'assistant diagnostic OBD2
// ==============================================================
//
// 1 000 000 de cas synthétiques, 1000 causes racines.
// Le défi ultime : peut-on vraiment classifier 1M cases en 1000 causes
// avec du bruit réaliste et convergence ?

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    CrossEntropyLoss, KaimingNormal, Linear, Loss, Module, PcgEngine, ReLU, Sequential, Zeros,
};

const N_FEATURES: usize = 20;
const N_CLASSES: usize = 1000;
const SEED: u64 = 42;

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 32) as f32) / (u32::MAX as f32)
    }
}

/// Génère 1M cases avec patterns distincts pour chaque cause
fn generate_megaverse_dataset(
    n_train: usize,
    n_val: usize,
    n_test: usize,
) -> (
    Vec<([f32; N_FEATURES], usize)>,
    Vec<([f32; N_FEATURES], usize)>,
    Vec<([f32; N_FEATURES], usize)>,
) {
    let mut rng = Rng::new(SEED);
    let mut train_data = Vec::new();
    let mut val_data = Vec::new();
    let mut test_data = Vec::new();

    for class_id in 0..N_CLASSES {
        for _ in 0..(n_train / N_CLASSES) {
            let features = generate_case_for_class(class_id, &mut rng, false);
            train_data.push((features, class_id));
        }
        for _ in 0..(n_val / N_CLASSES) {
            let features = generate_case_for_class(class_id, &mut rng, false);
            val_data.push((features, class_id));
        }
        for _ in 0..(n_test / N_CLASSES) {
            let features = generate_case_for_class(class_id, &mut rng, true);
            test_data.push((features, class_id));
        }
    }

    (train_data, val_data, test_data)
}

fn generate_case_for_class(
    class_id: usize,
    rng: &mut Rng,
    high_noise: bool,
) -> [f32; N_FEATURES] {
    let noise_level = if high_noise { 0.06 } else { 0.01 };
    let mut features = [0.5; N_FEATURES];

    // Chaque cause a une "signature" unique : activé 3-5 features distinctes
    let cause_seed = (class_id as u64).wrapping_mul(12345);
    let mut cause_rng = Rng::new(cause_seed);

    // Sélectionner 4 features distinctes pour cette cause
    let feature_indices = [
        (class_id * 7 + 0) % N_FEATURES,
        (class_id * 11 + 1) % N_FEATURES,
        (class_id * 13 + 2) % N_FEATURES,
        (class_id * 17 + 3) % N_FEATURES,
    ];

    for &idx in &feature_indices {
        let base = 0.75 + cause_rng.next_f32() * 0.2;
        features[idx] = (base + (rng.next_f32() - 0.5) * 2.0 * noise_level).clamp(0.0, 1.0);
    }

    // Ajouter du bruit sur toutes les features
    for f in features.iter_mut() {
        let noise = (rng.next_f32() - 0.5) * 2.0 * noise_level;
        *f = (*f + noise).clamp(0.0, 1.0);
    }

    features
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  SciRust — ENTRAÎNEMENT MÉGAVERSE OBD2                    ║");
    println!("║  1 000 000 cases × 1000 causes                            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let n_train = 800000;
    let n_val = 100000;
    let n_test = 100000;

    println!("📊 PHASE 1 : GÉNÉRATION DE DONNÉES");
    println!("─────────────────────────────────");
    println!("  Train : {} cases", n_train);
    println!("  Val   : {} cases", n_val);
    println!("  Test  : {} cases", n_test);
    println!("  Causes: {} (1 case par cause = 1000 exemples)", N_CLASSES);
    println!("  Features: {}\n", N_FEATURES);

    let start_gen = std::time::Instant::now();
    let (train_data, val_data, test_data) = generate_megaverse_dataset(n_train, n_val, n_test);
    let gen_time = start_gen.elapsed().as_secs_f32();
    println!("✓ Génération complète en {:.2}s\n", gen_time);

    println!("🧠 PHASE 2 : CONSTRUCTION DU MODÈLE");
    println!("──────────────────────────────────");
    let mut rng = PcgEngine::new(SEED);
    let mut model = Sequential::new()
        .add(Linear::new(N_FEATURES, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 128, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(128, N_CLASSES, &KaimingNormal, &Zeros, &mut rng));

    println!("  Modèle : {} → 256 → 256 → 128 → {}", N_FEATURES, N_CLASSES);
    println!("  Optimiseur : Adam(lr=0.0002, ultra-conservative)\n");

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.0002);

    let n_epochs = 20;
    println!("⚡ PHASE 3 : ENTRAÎNEMENT");
    println!("────────────────────────");
    println!("  {} epochs sur {} exemples/epoch\n", n_epochs, n_train);

    let train_start = std::time::Instant::now();
    let mut best_val_acc = 0.0;
    let mut best_epoch = 0;

    for epoch in 0..n_epochs {
        let mut train_loss = 0.0;
        let mut train_correct = 0;

        for (features, label) in &train_data {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(features.to_vec(), 1, N_FEATURES));

            let mut target_data = vec![0.0; N_CLASSES];
            target_data[*label] = 1.0;
            let target = tape.input(Tensor::from_vec(target_data, 1, N_CLASSES));

            let logits = model.forward(&tape, x);
            let loss = loss_fn.forward(&tape, logits, target);
            tape.backward(loss.idx());

            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);

            train_loss += tape.value(loss.idx()).data[0];

            let scores = tape.value(logits.idx());
            let pred = scores
                .data
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if pred == *label {
                train_correct += 1;
            }
        }

        // Validation
        let mut val_correct = 0;
        if (epoch + 1) % 2 == 0 {
            for (features, label) in &val_data {
                if predict_class(&mut model, features) == *label {
                    val_correct += 1;
                }
            }

            let train_acc = train_correct as f32 / n_train as f32;
            let val_acc = val_correct as f32 / n_val as f32;

            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_epoch = epoch + 1;
            }

            println!(
                "  Epoch {:>2} | loss={:.5} | train={:.2}% | val={:.2}% ← best={:.2}%",
                epoch + 1,
                train_loss / n_train as f32,
                train_acc * 100.0,
                val_acc * 100.0,
                best_val_acc * 100.0
            );
        } else {
            let train_acc = train_correct as f32 / n_train as f32;
            println!(
                "  Epoch {:>2} | loss={:.5} | train={:.2}%",
                epoch + 1,
                train_loss / n_train as f32,
                train_acc * 100.0
            );
        }
    }

    let train_time = train_start.elapsed().as_secs_f32();

    println!("\n📈 PHASE 4 : ÉVALUATION TEST");
    println!("──────────────────────────");
    let mut test_correct = 0;
    for (features, label) in &test_data {
        if predict_class(&mut model, features) == *label {
            test_correct += 1;
        }
    }

    let test_acc = test_correct as f32 / n_test as f32;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  🏆 RÉSULTATS FINAUX - MÉGAVERSE OBD2                     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    println!("  Dataset     : 1 000 000 cases | 1 000 causes");
    println!("  Temps gen   : {:.2}s", gen_time);
    println!("  Temps train : {:.2}s ({} epochs)", train_time, n_epochs);
    println!("  Best val    : {:.2}% (epoch {})", best_val_acc * 100.0, best_epoch);
    println!("  Test acc    : {:.2}% ({}/{})", test_acc * 100.0, test_correct, n_test);
    println!(
        "  Random baseline : {:.2}%\n",
        (1.0 / N_CLASSES as f32) * 100.0
    );

    // Top causes prédites
    println!("🎯 DIAGNOSTICS ALÉATOIRES (5 cas test)\n");
    for i in (0..test_data.len()).step_by(test_data.len() / 5) {
        if i < test_data.len() {
            diagnose(&mut model, &test_data[i].0, test_data[i].1);
        }
    }
}

fn predict_class(model: &mut Sequential, features: &[f32; N_FEATURES]) -> usize {
    let tape = Tape::new();
    let x = tape.input(Tensor::from_vec(features.to_vec(), 1, N_FEATURES));
    let logits = model.forward(&tape, x);
    let scores = tape.value(logits.idx());
    let mut best = 0;
    for i in 1..N_CLASSES {
        if scores.data[i] > scores.data[best] {
            best = i;
        }
    }
    best
}

fn diagnose(model: &mut Sequential, features: &[f32; N_FEATURES], true_label: usize) {
    let tape = Tape::new();
    let x = tape.input(Tensor::from_vec(features.to_vec(), 1, N_FEATURES));
    let logits = model.forward(&tape, x);
    let probs = softmax(&tape.value(logits.idx()).data);

    let mut ranked: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (pred, pred_p) = ranked[0];
    let match_mark = if pred == true_label { "✓" } else { "✗" };

    println!(
        "  Cause réelle: {:>4} | Prédiction: {:>4} ({:.2}%) {}",
        true_label, pred, pred_p * 100.0, match_mark
    );
    println!(
        "    Top 3 : #{:>4} ({:.2}%) | #{:>4} ({:.2}%) | #{:>4} ({:.2}%)\n",
        ranked[0].0, ranked[0].1 * 100.0,
        ranked[1].0, ranked[1].1 * 100.0,
        ranked[2].0, ranked[2].1 * 100.0,
    );
}
