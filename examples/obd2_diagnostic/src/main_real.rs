// examples/obd2_diagnostic/src/main_real.rs
//
// SciRust — Diagnostic OBD2 sur DONNÉES RÉELLES d'atelier
// =========================================================
//
// Données : télémétrie réelle d'une Opel Corsa 1.2 (2012) captée via
// adaptateur ELM327 + python-OBD (dataset Hugging Face
// `PedroCuisinier2025/OBD2_panel_opel_2012`, licence CC-BY-4.0).
// L'échantillon committé (`data/opel_corsa_telemetry.csv`, 43 139 relevés)
// couvre 7 segments de conduite avec 10 capteurs + les fuel trims.
//
// Principe du diagnostic : le modèle apprend la relation SAINE entre
// l'état moteur (RPM, MAF, charge, sondes O2, températures…) et la
// correction carburant long terme (LONG_FUEL_TRIM_1). En atelier, un
// écart important entre le trim OBSERVÉ et le trim PRÉDIT signale une
// anomalie du mélange : prise d'air (trim réel anormalement haut),
// capteur MAF incohérent, etc. — la logique P0171 du premier exemple,
// cette fois apprise sur de vrais relevés.
//
// Le modèle entraîné est sauvegardé au format safetensors avec les
// statistiques de normalisation en métadonnées : le fichier est
// auto-suffisant pour une future API de diagnostic.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::io::safetensors::{load_state_dict, save_state_dict};
use scirust_core::nn::{
    KaimingNormal, Linear, Loss, Module, MseLoss, PcgEngine, ReLU, Sequential, Zeros,
};
use std::collections::HashMap;

const FEATURES: [&str; 10] = [
    "RPM",
    "SPEED",
    "THROTTLE_POS",
    "MAF",
    "COOLANT_TEMP",
    "INTAKE_TEMP",
    "O2_B1S1",
    "ENGINE_LOAD",
    "INTAKE_PRESSURE",
    "O2_B1S2",
];
const TARGET: &str = "LONG_FUEL_TRIM_1";
const N_FEATURES: usize = 10;
const SEED: u64 = 42;
const BATCH: usize = 256;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng {
            state: seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1),
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

struct Sample {
    features: [f32; N_FEATURES],
    trim: f32,
    segment: u32,
}

/// Cherche un chemin relatif depuis la racine du workspace ou l'exemple.
fn resolve(rel: &str) -> String {
    for c in [rel.to_string(), format!("examples/obd2_diagnostic/{rel}")]
    {
        if std::path::Path::new(&c).exists()
        {
            return c;
        }
    }
    rel.to_string()
}

fn load_csv(path: &str) -> Vec<Sample> {
    let content =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("impossible de lire {path}: {e}"));
    let mut lines = content.lines();
    let header: Vec<&str> = lines.next().expect("CSV vide").split(',').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("colonne manquante: {name}"))
    };
    let feat_idx: Vec<usize> = FEATURES.iter().map(|f| col(f)).collect();
    let target_idx = col(TARGET);
    let seg_idx = col("segment_id");

    let mut samples = Vec::new();
    for line in lines
    {
        let fields: Vec<&str> = line.split(',').collect();
        let mut features = [0.0f32; N_FEATURES];
        let mut ok = true;
        for (k, &i) in feat_idx.iter().enumerate()
        {
            match fields[i].parse::<f32>()
            {
                Ok(v) => features[k] = v,
                Err(_) =>
                {
                    ok = false;
                    break;
                },
            }
        }
        let trim = fields[target_idx].parse::<f32>();
        let segment = fields[seg_idx].parse::<u32>();
        if let (true, Ok(trim), Ok(segment)) = (ok, trim, segment)
        {
            samples.push(Sample {
                features,
                trim,
                segment,
            });
        }
    }
    samples
}

struct Normalizer {
    mean: [f32; N_FEATURES],
    std: [f32; N_FEATURES],
    t_mean: f32,
    t_std: f32,
}

impl Normalizer {
    fn fit(data: &[&Sample]) -> Self {
        let n = data.len() as f32;
        let mut mean = [0.0f32; N_FEATURES];
        let mut std = [0.0f32; N_FEATURES];
        let mut t_mean = 0.0;
        let mut t_std = 0.0;
        for s in data
        {
            for k in 0..N_FEATURES
            {
                mean[k] += s.features[k];
            }
            t_mean += s.trim;
        }
        for m in mean.iter_mut()
        {
            *m /= n;
        }
        t_mean /= n;
        for s in data
        {
            for k in 0..N_FEATURES
            {
                let d = s.features[k] - mean[k];
                std[k] += d * d;
            }
            let d = s.trim - t_mean;
            t_std += d * d;
        }
        for v in std.iter_mut()
        {
            *v = (*v / n).sqrt().max(1e-6);
        }
        t_std = (t_std / n).sqrt().max(1e-6);
        Normalizer {
            mean,
            std,
            t_mean,
            t_std,
        }
    }

    fn norm_features(&self, f: &[f32; N_FEATURES]) -> [f32; N_FEATURES] {
        let mut out = [0.0f32; N_FEATURES];
        for k in 0..N_FEATURES
        {
            out[k] = (f[k] - self.mean[k]) / self.std[k];
        }
        out
    }

    fn norm_trim(&self, t: f32) -> f32 {
        (t - self.t_mean) / self.t_std
    }

    fn denorm_trim(&self, z: f32) -> f32 {
        z * self.t_std + self.t_mean
    }
}

/// Prédit le trim (en % brut) pour un lot d'échantillons.
fn predict_batch(model: &mut Sequential, norm: &Normalizer, xs: &[[f32; N_FEATURES]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(xs.len());
    for chunk in xs.chunks(500)
    {
        let mut xdata = Vec::with_capacity(chunk.len() * N_FEATURES);
        for f in chunk
        {
            xdata.extend_from_slice(&norm.norm_features(f));
        }
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(xdata, chunk.len(), N_FEATURES));
        let pred = model.forward(&tape, x);
        let vals = tape.value(pred.idx());
        for r in 0..chunk.len()
        {
            out.push(norm.denorm_trim(vals.data[r]));
        }
    }
    out
}

fn mae(model: &mut Sequential, norm: &Normalizer, data: &[&Sample]) -> f32 {
    let xs: Vec<[f32; N_FEATURES]> = data.iter().map(|s| s.features).collect();
    let preds = predict_batch(model, norm, &xs);
    let sum: f32 = data
        .iter()
        .zip(&preds)
        .map(|(s, p)| (s.trim - p).abs())
        .sum();
    sum / data.len() as f32
}

fn main() {
    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| resolve("data/opel_corsa_telemetry.csv"));
    let n_epochs: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(40);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  SciRust — DIAGNOSTIC OBD2 SUR DONNÉES RÉELLES            ║");
    println!("║  Opel Corsa 1.2 (2012), télémétrie ELM327 (CC-BY-4.0)     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("📊 PHASE 1 : CHARGEMENT DES DONNÉES RÉELLES");
    println!("──────────────────────────────────────────");
    let samples = load_csv(&csv_path);
    println!("  Fichier : {csv_path}");
    println!("  Relevés : {}", samples.len());

    // Split par segment de conduite (chronologique, sans fuite temporelle)
    let train: Vec<&Sample> = samples.iter().filter(|s| s.segment <= 9).collect();
    let val: Vec<&Sample> = samples.iter().filter(|s| s.segment == 10).collect();
    let test: Vec<&Sample> = samples.iter().filter(|s| s.segment >= 11).collect();
    println!("  Train : {} (segments 6-9)", train.len());
    println!("  Val   : {} (segment 10)", val.len());
    println!("  Test  : {} (segments 11-12)", test.len());
    println!("  Cible : {TARGET} (correction carburant long terme, %)\n");

    let norm = Normalizer::fit(&train);
    println!(
        "  Trim moyen (train) : {:.2}% | écart-type : {:.2}%\n",
        norm.t_mean, norm.t_std
    );

    println!("🧠 PHASE 2 : MODÈLE DE RÉFÉRENCE « MOTEUR SAIN »");
    println!("───────────────────────────────────────────────");
    let mut init_rng = PcgEngine::new(SEED);
    let mut model = Sequential::new()
        .add(Linear::new(
            N_FEATURES,
            64,
            &KaimingNormal,
            &Zeros,
            &mut init_rng,
        ))
        .add(ReLU::new())
        .add(Linear::new(64, 32, &KaimingNormal, &Zeros, &mut init_rng))
        .add(ReLU::new())
        .add(Linear::new(32, 1, &KaimingNormal, &Zeros, &mut init_rng));
    println!(
        "  Modèle : {} capteurs → 64 → 32 → 1 (trim prédit)",
        N_FEATURES
    );
    println!(
        "  Optimiseur : Adam(lr=0.001) | batch {} | {} epochs\n",
        BATCH, n_epochs
    );

    let loss_fn = MseLoss::new();
    let mut opt = Adam::new(0.001);

    println!("⚡ PHASE 3 : ENTRAÎNEMENT");
    println!("────────────────────────");
    let train_start = std::time::Instant::now();
    let mut shuffle_rng = Rng::new(SEED ^ 0xDEADBEEF);
    let mut order: Vec<usize> = (0..train.len()).collect();
    let n_batches = train.len() / BATCH;
    let mut best_val_mae = f32::INFINITY;

    for epoch in 0..n_epochs
    {
        for i in (1..order.len()).rev()
        {
            let j = shuffle_rng.next_usize(i + 1);
            order.swap(i, j);
        }

        let mut train_loss = 0.0f32;
        for b in 0..n_batches
        {
            let mut xdata = Vec::with_capacity(BATCH * N_FEATURES);
            let mut ydata = Vec::with_capacity(BATCH);
            for &idx in &order[b * BATCH..(b + 1) * BATCH]
            {
                xdata.extend_from_slice(&norm.norm_features(&train[idx].features));
                ydata.push(norm.norm_trim(train[idx].trim));
            }
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(xdata, BATCH, N_FEATURES));
            let y = tape.input(Tensor::from_vec(ydata, BATCH, 1));
            let pred = model.forward(&tape, x);
            let loss = loss_fn.forward(&tape, pred, y);
            tape.backward(loss.idx());
            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);
            train_loss += tape.value(loss.idx()).data[0];
        }

        if (epoch + 1) % 5 == 0 || epoch == 0
        {
            let val_mae = mae(&mut model, &norm, &val);
            if val_mae < best_val_mae
            {
                best_val_mae = val_mae;
            }
            println!(
                "  Epoch {:>3} | mse(z)={:.4} | MAE val={:.3}% trim",
                epoch + 1,
                train_loss / n_batches as f32,
                val_mae
            );
        }
    }
    let train_time = train_start.elapsed().as_secs_f32();

    println!("\n📈 PHASE 4 : ÉVALUATION SUR SEGMENTS JAMAIS VUS");
    println!("──────────────────────────────────────────────");
    let baseline: f32 = test
        .iter()
        .map(|s| (s.trim - norm.t_mean).abs())
        .sum::<f32>()
        / test.len() as f32;
    let test_mae = mae(&mut model, &norm, &test);
    println!(
        "  MAE baseline (prédire la moyenne) : {:.3}% trim",
        baseline
    );
    println!(
        "  MAE modèle                        : {:.3}% trim",
        test_mae
    );
    println!("  Temps d'entraînement              : {:.1}s\n", train_time);

    // Seuil d'anomalie : percentile 99 des résidus absolus en validation
    let val_xs: Vec<[f32; N_FEATURES]> = val.iter().map(|s| s.features).collect();
    let val_preds = predict_batch(&mut model, &norm, &val_xs);
    let mut residuals: Vec<f32> = val
        .iter()
        .zip(&val_preds)
        .map(|(s, p)| (s.trim - p).abs())
        .collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = residuals[residuals.len() * 99 / 100];
    println!("  Seuil d'anomalie (p99 val) : ±{:.2}% trim\n", threshold);

    println!("🔧 PHASE 5 : DÉMO DIAGNOSTIC — PANNES SIMULÉES SUR RELEVÉS RÉELS");
    println!("────────────────────────────────────────────────────────────────");
    println!("  (relevés test réels, défaut injecté artificiellement)\n");
    let mut demo_rng = Rng::new(SEED ^ 0xCAFE);
    for case in 0..3
    {
        let s = test[demo_rng.next_usize(test.len())];
        let pred_healthy = predict_batch(&mut model, &norm, &[s.features])[0];

        match case
        {
            0 =>
            {
                // Relevé sain, tel quel
                let residual = (s.trim - pred_healthy).abs();
                let verdict = if residual > threshold
                {
                    "⚠ ANOMALIE"
                }
                else
                {
                    "✓ sain"
                };
                println!(
                    "  Cas réel sain      | trim observé {:+.2}% | prédit {:+.2}% | résidu {:.2}% → {}",
                    s.trim, pred_healthy, residual, verdict
                );
            },
            1 =>
            {
                // Prise d'air simulée : air non mesuré → l'ECU sur-corrige (+14 %)
                let observed = s.trim + 14.0;
                let residual = (observed - pred_healthy).abs();
                let verdict = if residual > threshold
                {
                    "⚠ ANOMALIE (mélange pauvre — prise d'air ?)"
                }
                else
                {
                    "✓ sain"
                };
                println!(
                    "  Prise d'air simulée| trim observé {:+.2}% | prédit {:+.2}% | résidu {:.2}% → {}",
                    observed, pred_healthy, residual, verdict
                );
            },
            _ =>
            {
                // MAF sous-lisant simulé : capteur -35 % → incohérence prédite/observée
                let mut faulty = s.features;
                faulty[3] *= 0.65; // MAF
                let pred_faulty = predict_batch(&mut model, &norm, &[faulty])[0];
                let residual = (s.trim - pred_faulty).abs();
                let verdict = if residual > threshold
                {
                    "⚠ ANOMALIE (incohérence MAF ?)"
                }
                else
                {
                    "✓ sain"
                };
                println!(
                    "  MAF -35% simulé    | trim observé {:+.2}% | prédit {:+.2}% | résidu {:.2}% → {}",
                    s.trim, pred_faulty, residual, verdict
                );
            },
        }
    }

    println!("\n💾 PHASE 6 : SAUVEGARDE DES POIDS (safetensors)");
    println!("──────────────────────────────────────────────");
    let models_dir = resolve("models");
    std::fs::create_dir_all(&models_dir).ok();
    let weights_path = format!("{models_dir}/obd2_real_fueltrim.safetensors");

    let sd = model.state_dict();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("model".into(), "obd2_real_fueltrim".into());
    meta.insert("arch".into(), format!("{N_FEATURES}-64-32-1 ReLU"));
    meta.insert("target".into(), TARGET.into());
    meta.insert("features".into(), FEATURES.join(","));
    meta.insert(
        "feature_mean".into(),
        norm.mean
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    meta.insert(
        "feature_std".into(),
        norm.std
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    meta.insert("target_mean".into(), norm.t_mean.to_string());
    meta.insert("target_std".into(), norm.t_std.to_string());
    meta.insert("anomaly_threshold_pct".into(), threshold.to_string());
    meta.insert(
        "dataset".into(),
        "PedroCuisinier2025/OBD2_panel_opel_2012 (Hugging Face, CC-BY-4.0)".into(),
    );
    save_state_dict(&weights_path, &sd, Some(meta)).expect("échec sauvegarde poids");
    let size = std::fs::metadata(&weights_path)
        .map(|m| m.len())
        .unwrap_or(0);
    println!("  Poids + normalisation → {weights_path} ({size} octets)");

    // Preuve de round-trip : recharger dans un modèle vierge → prédictions identiques
    let (loaded_sd, loaded_meta) = load_state_dict(&weights_path).expect("échec lecture poids");
    let mut rng2 = PcgEngine::new(999); // init différente : les poids chargés doivent l'écraser
    let mut model2 = Sequential::new()
        .add(Linear::new(
            N_FEATURES,
            64,
            &KaimingNormal,
            &Zeros,
            &mut rng2,
        ))
        .add(ReLU::new())
        .add(Linear::new(64, 32, &KaimingNormal, &Zeros, &mut rng2))
        .add(ReLU::new())
        .add(Linear::new(32, 1, &KaimingNormal, &Zeros, &mut rng2));
    model2
        .load_state_dict(&loaded_sd)
        .expect("échec chargement state dict");

    let check_xs: Vec<[f32; N_FEATURES]> = test.iter().take(200).map(|s| s.features).collect();
    let p1 = predict_batch(&mut model, &norm, &check_xs);
    let p2 = predict_batch(&mut model2, &norm, &check_xs);
    let max_diff = p1
        .iter()
        .zip(&p2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  Vérification round-trip : écart max sur 200 prédictions = {:.2e} {}",
        max_diff,
        if max_diff < 1e-5 { "✓" } else { "✗ ÉCHEC" }
    );
    println!(
        "  Métadonnées embarquées : {} clés (features, normalisation, seuil, source)",
        loaded_meta.len()
    );

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  🏆 RÉSUMÉ — MODÈLE RÉEL PRÊT POUR L'API                  ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!(
        "  Données réelles : {} relevés Opel Corsa (7 trajets)",
        samples.len()
    );
    println!(
        "  MAE test        : {:.3}% trim (baseline {:.3}%)",
        test_mae, baseline
    );
    println!("  Poids           : {weights_path}");
    println!("  Le fichier contient poids + normalisation + seuil d'anomalie :");
    println!("  une future API n'a besoin QUE de ce fichier pour diagnostiquer.");
}
