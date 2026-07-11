// examples/obd2_diagnostic/src/main_ultra.rs
//
// SciRust — Entraînement ULTRA-MASSIF d'assistant diagnostic OBD2
// ================================================================
//
// 100 000+ cas synthétiques, 10 causes, modèle profond.
// Défi : peut-on vraiment converger sur 100K cases synthétiques avec du bruit ?
// Réponse : oui, avec un modèle plus profond et un learning rate adapté.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    CrossEntropyLoss, KaimingNormal, Linear, Loss, Module, PcgEngine, ReLU, Sequential, Zeros,
};

const N_FEATURES: usize = 10;
const N_CLASSES: usize = 10;
const SEED: u64 = 42;

const CAUSES: [&str; N_CLASSES] = [
    "Prise d'air / fuite depression",
    "Capteur MAF encrassé",
    "Système d'allumage defectueux",
    "Convertisseur catalytique",
    "Fuite circuit EVAP",
    "Thermostat moteur",
    "Capteur O2 / sonde lambda",
    "Injecteur carburant",
    "Pompe carburant / pression",
    "Turbo / suralimentation",
];

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
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 32) as f32) / (u32::MAX as f32)
    }
}

/// Génère 100K + cas avec bruit réaliste
fn generate_ultra_dataset(
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

    for class_id in 0..N_CLASSES
    {
        for _ in 0..(n_train / N_CLASSES)
        {
            let features = generate_case_for_class(class_id, &mut rng, false);
            train_data.push((features, class_id));
        }
        for _ in 0..(n_val / N_CLASSES)
        {
            let features = generate_case_for_class(class_id, &mut rng, false);
            val_data.push((features, class_id));
        }
        for _ in 0..(n_test / N_CLASSES)
        {
            let features = generate_case_for_class(class_id, &mut rng, true);
            test_data.push((features, class_id));
        }
    }

    (train_data, val_data, test_data)
}

fn generate_case_for_class(class_id: usize, rng: &mut Rng, high_noise: bool) -> [f32; N_FEATURES] {
    let noise_level = if high_noise { 0.08 } else { 0.02 };
    let mut features = [0.5; N_FEATURES];

    match class_id
    {
        0 =>
        {
            features[0] = 0.85 + rng.next_f32() * 0.08;
            features[1] = 0.15 + rng.next_f32() * 0.08;
            features[4] = 0.80 + rng.next_f32() * 0.08;
            features[5] = 0.75 + rng.next_f32() * 0.08;
            features[8] = 0.20 + rng.next_f32() * 0.08;
        },
        1 =>
        {
            features[0] = 0.80 + rng.next_f32() * 0.08;
            features[2] = 0.80 + rng.next_f32() * 0.08;
            features[4] = 0.75 + rng.next_f32() * 0.08;
            features[5] = 0.30 + rng.next_f32() * 0.08;
        },
        2 =>
        {
            features[1] = 0.85 + rng.next_f32() * 0.08;
            features[4] = 0.48 + rng.next_f32() * 0.08;
            features[5] = 0.80 + rng.next_f32() * 0.08;
        },
        3 =>
        {
            features[3] = 0.90 + rng.next_f32() * 0.08;
            features[4] = 0.50 + rng.next_f32() * 0.08;
            features[0] = 0.20 + rng.next_f32() * 0.08;
        },
        4 =>
        {
            features[6] = 0.90 + rng.next_f32() * 0.08;
            features[4] = 0.50 + rng.next_f32() * 0.08;
        },
        5 =>
        {
            features[7] = 0.85 + rng.next_f32() * 0.08;
            features[4] = 0.50 + rng.next_f32() * 0.08;
            features[0] = 0.25 + rng.next_f32() * 0.08;
        },
        6 =>
        {
            features[8] = 0.15 + rng.next_f32() * 0.08;
            features[0] = 0.65 + rng.next_f32() * 0.08;
        },
        7 =>
        {
            features[1] = 0.50 + rng.next_f32() * 0.08;
            features[4] = 0.40 + rng.next_f32() * 0.08;
        },
        8 =>
        {
            features[9] = 0.85 + rng.next_f32() * 0.08;
            features[0] = 0.75 + rng.next_f32() * 0.08;
        },
        9 =>
        {
            features[4] = 0.65 + rng.next_f32() * 0.08;
            features[2] = 0.20 + rng.next_f32() * 0.08;
        },
        _ =>
        {},
    }

    for f in features.iter_mut()
    {
        let noise = (rng.next_f32() - 0.5) * 2.0 * noise_level;
        *f = (*f + noise).clamp(0.0, 1.0);
    }

    features
}

fn main() {
    println!("=== SciRust — Entraînement ULTRA-MASSIF OBD2 (100K) ===\n");

    let n_train = 80000;
    let n_val = 10000;
    let n_test = 10000;

    println!(
        "Génération : {} train + {} val + {} test (10 causes)\n",
        n_train, n_val, n_test
    );
    let start_gen = std::time::Instant::now();
    let (train_data, val_data, test_data) = generate_ultra_dataset(n_train, n_val, n_test);
    let gen_time = start_gen.elapsed().as_secs_f32();
    println!("Génération en {:.2}s\n", gen_time);

    // Modèle très profond : 10 -> 128 -> 64 -> 32 -> 10
    let mut rng = PcgEngine::new(SEED);
    let mut model = Sequential::new()
        .add(Linear::new(
            N_FEATURES,
            128,
            &KaimingNormal,
            &Zeros,
            &mut rng,
        ))
        .add(ReLU::new())
        .add(Linear::new(128, 64, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(64, 32, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(32, N_CLASSES, &KaimingNormal, &Zeros, &mut rng));

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.0005); // lr encore plus bas pour 100K

    let n_epochs = 30;
    println!("Modèle : 10 -> 128 -> 64 -> 32 -> 10 (très profond)\n");
    println!("Entraînement : {} epochs, Adam(lr=0.0005)\n", n_epochs);

    let train_start = std::time::Instant::now();
    let mut best_val_acc = 0.0;
    let mut best_epoch = 0;

    for epoch in 0..n_epochs
    {
        let mut train_loss = 0.0;
        let mut train_correct = 0;

        for (features, label) in &train_data
        {
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
            if pred == *label
            {
                train_correct += 1;
            }
        }

        // Validation (toutes les 2 epochs pour accélérer)
        let mut val_correct = 0;
        let mut val_loss = 0.0;
        if (epoch + 1) % 2 == 0
        {
            for (features, label) in &val_data
            {
                let tape = Tape::new();
                let x = tape.input(Tensor::from_vec(features.to_vec(), 1, N_FEATURES));
                let mut target_data = vec![0.0; N_CLASSES];
                target_data[*label] = 1.0;
                let target = tape.input(Tensor::from_vec(target_data, 1, N_CLASSES));

                let logits = model.forward(&tape, x);
                let loss = loss_fn.forward(&tape, logits, target);
                tape.backward(loss.idx());

                val_loss += tape.value(loss.idx()).data[0];

                let scores = tape.value(logits.idx());
                let pred = scores
                    .data
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                if pred == *label
                {
                    val_correct += 1;
                }
            }

            let val_acc = val_correct as f32 / n_val as f32;
            if val_acc > best_val_acc
            {
                best_val_acc = val_acc;
                best_epoch = epoch + 1;
            }

            let train_acc = train_correct as f32 / n_train as f32;
            println!(
                "Epoch {:>2} | loss={:.4} | train_acc={:.2}% | val_acc={:.2}%",
                epoch + 1,
                train_loss / n_train as f32,
                train_acc * 100.0,
                val_acc * 100.0
            );
        }
        else
        {
            let train_acc = train_correct as f32 / n_train as f32;
            println!(
                "Epoch {:>2} | loss={:.4} | train_acc={:.2}%",
                epoch + 1,
                train_loss / n_train as f32,
                train_acc * 100.0
            );
        }
    }

    let train_time = train_start.elapsed().as_secs_f32();

    // Test final
    println!("\nÉvaluation test final...");
    let mut test_correct = 0;
    for (features, label) in &test_data
    {
        if predict_class(&mut model, features) == *label
        {
            test_correct += 1;
        }
    }

    let test_acc = test_correct as f32 / n_test as f32;

    println!("\n=== 🚀 RÉSULTATS ULTRA-MASSIFS ===");
    println!("Temps génération data : {:.2}s", gen_time);
    println!(
        "Temps entraînement    : {:.2}s ({} epochs)",
        train_time, n_epochs
    );
    println!(
        "Meilleure val_acc     : {:.2}% (epoch {})",
        best_val_acc * 100.0,
        best_epoch
    );
    println!(
        "Test accuracy         : {:.2}% ({}/{})",
        test_acc * 100.0,
        test_correct,
        n_test
    );

    println!("\n=== DIAGNOSTICS DE CAS RÉELS (test set) ===");
    for i in 0..8
    {
        if i < test_data.len()
        {
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
    for i in 1..N_CLASSES
    {
        if scores.data[i] > scores.data[best]
        {
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
        "\n  Réelle: {} | Prédiction: {} ({:.1}%) {}",
        CAUSES[true_label],
        CAUSES[pred],
        pred_p * 100.0,
        match_mark
    );
    println!(
        "    Top 3 : {:.1}% {} | {:.1}% {} | {:.1}% {}",
        ranked[0].1 * 100.0,
        CAUSES[ranked[0].0],
        ranked[1].1 * 100.0,
        CAUSES[ranked[1].0],
        ranked[2].1 * 100.0,
        CAUSES[ranked[2].0],
    );
}
