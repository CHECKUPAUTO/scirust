// examples/obd2_diagnostic/src/main_megaverse.rs
//
// SciRust — Entraînement MÉGAVERSE d'assistant diagnostic OBD2
// ==============================================================
//
// 1 000 000 de cas synthétiques, 1000 causes racines.
//
// Version 2 — corrige trois défauts fatals de la v1 :
//
//  1. SIGNATURES EN COLLISION : les indices `(class_id*7)%20, …` étaient
//     périodiques modulo 20 → seulement 20 signatures distinctes pour
//     1000 causes (50 causes partageaient les mêmes features actives).
//     Désormais : 8 features tirées par Fisher-Yates partiel par cause,
//     chacune avec une polarité haute/basse (~32 M de motifs possibles),
//     l'unicité des 1000 signatures est vérifiée à la génération.
//
//  2. AUCUN MÉLANGE : le flux trié par classe (800 cas de la cause 0,
//     puis 800 de la cause 1…) provoquait un oubli catastrophique — le
//     modèle ne prédisait que la dernière classe vue (~0.1 % en val).
//     Désormais : Fisher-Yates sur l'ordre des exemples à chaque epoch.
//
//  3. UN TAPE PAR EXEMPLE : 800 000 graphes d'autodiff par epoch
//     rendaient l'epoch interminable (~9 h). Désormais : mini-batches
//     de 256 via le support multi-batch natif (matmul batché +
//     CrossEntropy à labels entiers) → 3 125 graphes/epoch.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::io::safetensors::save_state_dict;
use scirust_core::nn::{
    CrossEntropyLoss, KaimingNormal, Linear, Module, PcgEngine, ReLU, Sequential, Zeros,
};
use std::collections::HashSet;

const N_FEATURES: usize = 20;
const N_CLASSES: usize = 1000;
const SIG_FEATURES: usize = 8;
const SEED: u64 = 42;
const BATCH: usize = 256;
const EVAL_BATCH: usize = 500;

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
        // Avance d'un pas pour disperser les graines proches.
        let mut r = Rng {
            state: seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1),
        };
        r.next_u32();
        r
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

/// Signature d'une cause : 8 capteurs anormaux (haut ou bas), le reste à 0.5.
struct CauseSignature {
    indices: [usize; SIG_FEATURES],
    values: [f32; SIG_FEATURES],
}

/// Construit 1000 signatures garanties distinctes (features + polarités).
fn build_signatures() -> Vec<CauseSignature> {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut sigs = Vec::with_capacity(N_CLASSES);

    for class_id in 0..N_CLASSES
    {
        let mut salt = 0u64;
        loop
        {
            let mut rng = Rng::new((class_id as u64 + 1) ^ (salt << 32));

            // Fisher-Yates partiel : 8 indices distincts parmi 20
            let mut pool: [usize; N_FEATURES] = [0; N_FEATURES];
            for (i, p) in pool.iter_mut().enumerate()
            {
                *p = i;
            }
            for k in 0..SIG_FEATURES
            {
                let j = k + rng.next_usize(N_FEATURES - k);
                pool.swap(k, j);
            }

            let mut indices = [0usize; SIG_FEATURES];
            indices.copy_from_slice(&pool[..SIG_FEATURES]);
            indices.sort_unstable();

            let mut values = [0.0f32; SIG_FEATURES];
            let mut key = 0u64;
            for k in 0..SIG_FEATURES
            {
                let high = rng.next_f32() < 0.5;
                values[k] = if high
                {
                    0.80 + 0.15 * rng.next_f32() // capteur anormalement haut
                }
                else
                {
                    0.05 + 0.15 * rng.next_f32() // capteur anormalement bas
                };
                key = key
                    .wrapping_mul(41)
                    .wrapping_add((indices[k] * 2 + high as usize) as u64);
            }

            // Collision de motif (features + polarités) : on re-tire.
            if seen.insert(key)
            {
                sigs.push(CauseSignature { indices, values });
                break;
            }
            salt += 1;
        }
    }
    sigs
}

fn generate_case(sig: &CauseSignature, rng: &mut Rng, noise: f32) -> [f32; N_FEATURES] {
    let mut features = [0.5f32; N_FEATURES]; // relevés "normaux" normalisés
    for k in 0..SIG_FEATURES
    {
        features[sig.indices[k]] = sig.values[k];
    }
    for f in features.iter_mut()
    {
        *f = (*f + (rng.next_f32() - 0.5) * 2.0 * noise).clamp(0.0, 1.0);
    }
    features
}

/// Évaluation batchée : renvoie la précision sur les indices donnés.
fn evaluate(
    model: &mut Sequential,
    xs: &[[f32; N_FEATURES]],
    ys: &[usize],
    indices: &[usize],
) -> f32 {
    let mut correct = 0usize;
    for chunk in indices.chunks(EVAL_BATCH)
    {
        let mut xdata = Vec::with_capacity(chunk.len() * N_FEATURES);
        for &i in chunk
        {
            xdata.extend_from_slice(&xs[i]);
        }
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(xdata, chunk.len(), N_FEATURES));
        let logits = model.forward(&tape, x);
        let scores = tape.value(logits.idx());
        for (r, &i) in chunk.iter().enumerate()
        {
            let row = &scores.data[r * N_CLASSES..(r + 1) * N_CLASSES];
            let mut best = 0usize;
            for c in 1..N_CLASSES
            {
                if row[c] > row[best]
                {
                    best = c;
                }
            }
            if best == ys[i]
            {
                correct += 1;
            }
        }
    }
    correct as f32 / indices.len() as f32
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
        true_label,
        pred,
        pred_p * 100.0,
        match_mark
    );
    println!(
        "    Top 3 : #{:>4} ({:.2}%) | #{:>4} ({:.2}%) | #{:>4} ({:.2}%)\n",
        ranked[0].0,
        ranked[0].1 * 100.0,
        ranked[1].0,
        ranked[1].1 * 100.0,
        ranked[2].0,
        ranked[2].1 * 100.0,
    );
}

fn main() {
    let n_epochs: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(8);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  SciRust — ENTRAÎNEMENT MÉGAVERSE OBD2 (v2)               ║");
    println!(
        "║  1 000 000 cases × 1000 causes — mini-batch {}            ║",
        BATCH
    );
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let n_train = 800_000usize;
    let n_val = 100_000usize;
    let n_test = 100_000usize;

    println!("📊 PHASE 1 : GÉNÉRATION DE DONNÉES");
    println!("─────────────────────────────────");
    println!("  Train : {} cases (bruit ±0.03)", n_train);
    println!("  Val   : {} cases (bruit ±0.03)", n_val);
    println!("  Test  : {} cases (bruit ±0.05, plus dur)", n_test);
    println!("  Causes: {} | Features: {}", N_CLASSES, N_FEATURES);

    let start_gen = std::time::Instant::now();
    let sigs = build_signatures();
    println!("  Signatures uniques : {} / {}", sigs.len(), N_CLASSES);

    let mut rng = Rng::new(SEED);
    let mut train_x: Vec<[f32; N_FEATURES]> = Vec::with_capacity(n_train);
    let mut train_y: Vec<usize> = Vec::with_capacity(n_train);
    let mut val_x: Vec<[f32; N_FEATURES]> = Vec::with_capacity(n_val);
    let mut val_y: Vec<usize> = Vec::with_capacity(n_val);
    let mut test_x: Vec<[f32; N_FEATURES]> = Vec::with_capacity(n_test);
    let mut test_y: Vec<usize> = Vec::with_capacity(n_test);

    for (class_id, sig) in sigs.iter().enumerate()
    {
        for _ in 0..(n_train / N_CLASSES)
        {
            train_x.push(generate_case(sig, &mut rng, 0.03));
            train_y.push(class_id);
        }
        for _ in 0..(n_val / N_CLASSES)
        {
            val_x.push(generate_case(sig, &mut rng, 0.03));
            val_y.push(class_id);
        }
        for _ in 0..(n_test / N_CLASSES)
        {
            test_x.push(generate_case(sig, &mut rng, 0.05));
            test_y.push(class_id);
        }
    }
    let gen_time = start_gen.elapsed().as_secs_f32();
    println!("✓ Génération complète en {:.2}s\n", gen_time);

    println!("🧠 PHASE 2 : CONSTRUCTION DU MODÈLE");
    println!("──────────────────────────────────");
    let mut init_rng = PcgEngine::new(SEED);
    let mut model = Sequential::new()
        .add(Linear::new(
            N_FEATURES,
            256,
            &KaimingNormal,
            &Zeros,
            &mut init_rng,
        ))
        .add(ReLU::new())
        .add(Linear::new(256, 128, &KaimingNormal, &Zeros, &mut init_rng))
        .add(ReLU::new())
        .add(Linear::new(
            128,
            N_CLASSES,
            &KaimingNormal,
            &Zeros,
            &mut init_rng,
        ));

    let n_params = N_FEATURES * 256 + 256 + 256 * 128 + 128 + 128 * N_CLASSES + N_CLASSES;
    println!(
        "  Modèle : {} → 256 → 128 → {} ({} paramètres)",
        N_FEATURES, N_CLASSES, n_params
    );
    println!("  Optimiseur : Adam(lr=0.001) | batch {}\n", BATCH);

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.001);

    let n_batches = n_train / BATCH;
    println!("⚡ PHASE 3 : ENTRAÎNEMENT");
    println!("────────────────────────");
    println!(
        "  {} epochs × {} mini-batches (shuffle par epoch)\n",
        n_epochs, n_batches
    );

    // Sous-ensemble de validation stratifié (1 cas sur 10 → 10 par cause)
    let val_subset: Vec<usize> = (0..n_val).step_by(10).collect();

    let train_start = std::time::Instant::now();
    let mut shuffle_rng = Rng::new(SEED ^ 0xDEADBEEF);
    let mut order: Vec<usize> = (0..n_train).collect();
    let mut best_val_acc = 0.0f32;
    let mut best_epoch = 0usize;

    for epoch in 0..n_epochs
    {
        let epoch_start = std::time::Instant::now();

        // Fisher-Yates : indispensable, les données sont triées par classe
        for i in (1..n_train).rev()
        {
            let j = shuffle_rng.next_usize(i + 1);
            order.swap(i, j);
        }

        let mut train_loss = 0.0f32;
        let mut train_correct = 0usize;

        for b in 0..n_batches
        {
            let mut xdata = Vec::with_capacity(BATCH * N_FEATURES);
            let mut ydata = Vec::with_capacity(BATCH);
            for &idx in &order[b * BATCH..(b + 1) * BATCH]
            {
                xdata.extend_from_slice(&train_x[idx]);
                ydata.push(train_y[idx] as f32);
            }

            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(xdata, BATCH, N_FEATURES));
            let logits = model.forward(&tape, x);
            let targets = Tensor::from_vec(ydata, BATCH, 1);
            let loss = loss_fn.forward_with_indices(&tape, logits, &targets);
            tape.backward(loss.idx());

            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);

            train_loss += tape.value(loss.idx()).data[0];

            let scores = tape.value(logits.idx());
            for (r, &idx) in order[b * BATCH..(b + 1) * BATCH].iter().enumerate()
            {
                let row = &scores.data[r * N_CLASSES..(r + 1) * N_CLASSES];
                let mut best = 0usize;
                for c in 1..N_CLASSES
                {
                    if row[c] > row[best]
                    {
                        best = c;
                    }
                }
                if best == train_y[idx]
                {
                    train_correct += 1;
                }
            }

            if (b + 1) % 500 == 0
            {
                println!(
                    "    … batch {:>4}/{} | loss={:.4} | {:.0}s écoulées",
                    b + 1,
                    n_batches,
                    train_loss / (b + 1) as f32,
                    epoch_start.elapsed().as_secs_f32()
                );
            }
        }

        let val_acc = evaluate(&mut model, &val_x, &val_y, &val_subset);
        if val_acc > best_val_acc
        {
            best_val_acc = val_acc;
            best_epoch = epoch + 1;
        }

        println!(
            "  Epoch {:>2} | loss={:.4} | train={:.2}% | val={:.2}% (best {:.2}%) | {:.0}s",
            epoch + 1,
            train_loss / n_batches as f32,
            train_correct as f32 / (n_batches * BATCH) as f32 * 100.0,
            val_acc * 100.0,
            best_val_acc * 100.0,
            epoch_start.elapsed().as_secs_f32()
        );
    }

    let train_time = train_start.elapsed().as_secs_f32();

    println!("\n📈 PHASE 4 : ÉVALUATION TEST (100 000 cas, bruit renforcé)");
    println!("──────────────────────────────────────────────────────");
    let all_test: Vec<usize> = (0..n_test).collect();
    let test_acc = evaluate(&mut model, &test_x, &test_y, &all_test);
    let test_correct = (test_acc * n_test as f32).round() as usize;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  🏆 RÉSULTATS FINAUX - MÉGAVERSE OBD2 (v2)                ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    println!("  Dataset     : 1 000 000 cases | 1 000 causes");
    println!("  Temps gen   : {:.2}s", gen_time);
    println!("  Temps train : {:.2}s ({} epochs)", train_time, n_epochs);
    println!(
        "  Best val    : {:.2}% (epoch {})",
        best_val_acc * 100.0,
        best_epoch
    );
    println!(
        "  Test acc    : {:.2}% ({}/{})",
        test_acc * 100.0,
        test_correct,
        n_test
    );
    println!(
        "  Random baseline : {:.2}%\n",
        (1.0 / N_CLASSES as f32) * 100.0
    );

    println!("🎯 DIAGNOSTICS ALÉATOIRES (5 cas test)\n");
    for i in (0..n_test).step_by(n_test / 5)
    {
        diagnose(&mut model, &test_x[i], test_y[i]);
    }

    println!("💾 SAUVEGARDE DES POIDS (safetensors)");
    println!("────────────────────────────────────");
    let models_dir = ["models", "examples/obd2_diagnostic/models"]
        .into_iter()
        .find(|d| std::path::Path::new(d).exists())
        .unwrap_or("models");
    std::fs::create_dir_all(models_dir).ok();
    let weights_path = format!("{models_dir}/obd2_megaverse.safetensors");

    let sd = model.state_dict();
    let mut meta: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    meta.insert("model".into(), "obd2_megaverse".into());
    meta.insert(
        "arch".into(),
        format!("{N_FEATURES}-256-128-{N_CLASSES} ReLU"),
    );
    meta.insert("n_classes".into(), N_CLASSES.to_string());
    meta.insert("n_features".into(), N_FEATURES.to_string());
    meta.insert("seed".into(), SEED.to_string());
    meta.insert("test_acc".into(), format!("{:.4}", test_acc));
    save_state_dict(&weights_path, &sd, Some(meta)).expect("échec sauvegarde poids");
    let size = std::fs::metadata(&weights_path)
        .map(|m| m.len())
        .unwrap_or(0);
    println!("  Poids → {weights_path} ({:.1} Ko)", size as f32 / 1024.0);
}
