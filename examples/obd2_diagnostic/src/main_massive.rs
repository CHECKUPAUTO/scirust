// examples/obd2_diagnostic/src/main_massive.rs
//
// SciRust — Entraînement MASSIF d'assistant diagnostic OBD2
// ===========================================================
//
// Version production-grade : 10 causes racines, 10 000+ cas synthétiques
// avec bruit réaliste, validation/test sets, métriques de performance.
//
// Données générées algorithmiquement selon les patterns d'atelier réels :
// pas de CSV externe, tout est reproducible (graine fixe).

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    CrossEntropyLoss, KaimingNormal, Linear, Loss, Module, PcgEngine, ReLU, Sequential, Zeros,
};

const N_FEATURES: usize = 10;
const N_CLASSES: usize = 10;
const SEED: u64 = 42;

type Dataset = (
    Vec<([f32; N_FEATURES], usize)>,
    Vec<([f32; N_FEATURES], usize)>,
    Vec<([f32; N_FEATURES], usize)>,
);

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

/// Générateur pseudo-aléatoire LCG pour reproductibilité
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

/// Génère N_TRAIN + N_VAL + N_TEST cas d'entraînement avec bruit réaliste
fn generate_massive_dataset(n_train: usize, n_val: usize, n_test: usize) -> Dataset {
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
            let features = generate_case_for_class(class_id, &mut rng, true); // bruit plus élevé
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
            // Prise d'air : pauvre + trim élevé + MAF NORMAL + ralenti instable
            features[0] = 0.85 + rng.next_f32() * 0.08; // mélange pauvre fort
            features[1] = 0.15 + rng.next_f32() * 0.08; // MAF normal
            features[4] = 0.80 + rng.next_f32() * 0.08; // trim très élevé
            features[5] = 0.75 + rng.next_f32() * 0.08; // ralenti instable
            features[8] = 0.20 + rng.next_f32() * 0.08; // pression normal
        },
        1 =>
        {
            // MAF encrassé : pauvre + MAF BAS (clé !) + trim élevé
            features[0] = 0.80 + rng.next_f32() * 0.08; // mélange pauvre
            features[2] = 0.80 + rng.next_f32() * 0.08; // débit d'air BAS (distinctif)
            features[4] = 0.75 + rng.next_f32() * 0.08; // trim moyen-élevé
            features[5] = 0.30 + rng.next_f32() * 0.08; // ralenti stable
        },
        2 =>
        {
            // Allumage : ratés + trim NORMAL + ralenti instable
            features[1] = 0.85 + rng.next_f32() * 0.08; // ratés d'allumage forts
            features[4] = 0.48 + rng.next_f32() * 0.08; // trim normal
            features[5] = 0.80 + rng.next_f32() * 0.08; // ralenti instable
        },
        3 =>
        {
            // Catalyseur : code cata + tout le reste normal
            features[3] = 0.90 + rng.next_f32() * 0.08; // code cata fort
            features[4] = 0.50 + rng.next_f32() * 0.08; // trim normal
            features[0] = 0.20 + rng.next_f32() * 0.08; // pas de pauvre
        },
        4 =>
        {
            // EVAP : code EVAP + tout normal
            features[6] = 0.90 + rng.next_f32() * 0.08; // code EVAP fort
            features[4] = 0.50 + rng.next_f32() * 0.08;
        },
        5 =>
        {
            // Thermostat : temp anormale + tout normal
            features[7] = 0.85 + rng.next_f32() * 0.08; // temp moteur anormale
            features[4] = 0.50 + rng.next_f32() * 0.08;
            features[0] = 0.25 + rng.next_f32() * 0.08;
        },
        6 =>
        {
            // O2 / Lambda : sonde défectueuse + peut sembler pauvre
            features[8] = 0.15 + rng.next_f32() * 0.08; // sonde défectueuse (clé)
            features[0] = 0.65 + rng.next_f32() * 0.08; // peut sembler pauvre
        },
        7 =>
        {
            // Injecteur : ratés légers + trim bas
            features[1] = 0.50 + rng.next_f32() * 0.08; // ratés légers
            features[4] = 0.40 + rng.next_f32() * 0.08; // trim bas
        },
        8 =>
        {
            // Pompe carburant : pression basse + pauvre
            features[9] = 0.85 + rng.next_f32() * 0.08; // pression basse (clé)
            features[0] = 0.75 + rng.next_f32() * 0.08; // mélange pauvre
        },
        9 =>
        {
            // Turbo : boost anormal + débit d'air élevé
            features[4] = 0.65 + rng.next_f32() * 0.08; // boost anormal
            features[2] = 0.20 + rng.next_f32() * 0.08; // débit d'air élevé (pas bas)
        },
        _ =>
        {},
    }

    // Ajouter du bruit petit
    for f in features.iter_mut()
    {
        let noise = (rng.next_f32() - 0.5) * 2.0 * noise_level;
        *f = (*f + noise).clamp(0.0, 1.0);
    }

    features
}

fn main() {
    println!("=== SciRust — Entraînement MASSIF OBD2 ===\n");

    let n_train = 8000;
    let n_val = 1000;
    let n_test = 1000;

    println!(
        "Génération : {} train + {} val + {} test (10 causes)\n",
        n_train, n_val, n_test
    );
    let (train_data, val_data, test_data) = generate_massive_dataset(n_train, n_val, n_test);

    // Modèle plus profond : 10 -> 64 -> 32 -> 10
    let mut rng = PcgEngine::new(SEED);
    let mut model = Sequential::new()
        .add(Linear::new(
            N_FEATURES,
            64,
            &KaimingNormal,
            &Zeros,
            &mut rng,
        ))
        .add(ReLU::new())
        .add(Linear::new(64, 32, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(32, N_CLASSES, &KaimingNormal, &Zeros, &mut rng));

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.001); // lr plus bas pour stabilité

    let n_epochs = 50;
    println!("Modèle : 10 -> 64 -> 32 -> 10\n");
    println!(
        "Entraînement : {} epochs, Adam(lr=0.001, batch_size={})\n",
        n_epochs, n_train
    );

    // ---- Boucle d'entraînement ----
    let mut best_val_acc = 0.0;
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

        // Évaluation validation
        let mut val_correct = 0;
        for (features, label) in &val_data
        {
            if predict_class(&mut model, features) == *label
            {
                val_correct += 1;
            }
        }

        let train_acc = train_correct as f32 / n_train as f32;
        let val_acc = val_correct as f32 / n_val as f32;
        if val_acc > best_val_acc
        {
            best_val_acc = val_acc;
        }

        if (epoch + 1) % 5 == 0
        {
            println!(
                "Epoch {:>2} | loss={:.4} | train_acc={:.2}% | val_acc={:.2}%",
                epoch + 1,
                train_loss / n_train as f32,
                train_acc * 100.0,
                val_acc * 100.0
            );
        }
    }

    // ---- Évaluation test final ----
    let mut test_correct = 0;
    for (features, label) in &test_data
    {
        if predict_class(&mut model, features) == *label
        {
            test_correct += 1;
        }
    }

    let test_acc = test_correct as f32 / n_test as f32;
    println!("\n=== RÉSULTATS FINAUX ===");
    println!("Meilleure val_acc : {:.2}%", best_val_acc * 100.0);
    println!(
        "Test accuracy     : {:.2}% ({}/{})",
        test_acc * 100.0,
        test_correct,
        n_test
    );

    // ---- Cas de test réel ----
    println!("\n=== DIAGNOSTIC DE CAS RÉELS (test set) ===");
    for i in 0..5
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
        "\nCause réelle : {} | Prédiction : {} ({:.1}%) {}",
        CAUSES[true_label],
        CAUSES[pred],
        pred_p * 100.0,
        match_mark
    );
    println!("  Top 3 hypothèses :");
    for (i, (cause_idx, p)) in ranked.iter().take(3).enumerate()
    {
        println!("    {}. {:.1}% — {}", i + 1, p * 100.0, CAUSES[*cause_idx]);
    }
}
