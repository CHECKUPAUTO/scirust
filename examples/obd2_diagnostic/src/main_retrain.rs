// examples/obd2_diagnostic/src/main_retrain.rs
//
// SciRust — Ré-entraînement OBD2 à partir du feedback atelier
// ==============================================================
//
// Ferme la boucle promise par `obd2_api` (`POST /feedback`) : les cas
// confirmés en atelier, archivés en JSONL par l'API, redeviennent des
// exemples d'entraînement pour affiner le modèle « données réelles ».
//
// Principe : on entraîne DEUX fois le même modèle (même architecture,
// mêmes hyperparamètres, même split de test jamais touché) — une fois
// sur le CSV seul (baseline), une fois CSV + feedback (augmenté) — et on
// compare les deux MAE sur les segments de test. Ça répond honnêtement à
// la question « le feedback aide-t-il vraiment ? » plutôt que de
// présumer que plus de données égale toujours mieux.
//
// Seuls les cas de feedback avec les 10 capteurs ET le trim confirmé
// sont exploitables pour CETTE régression (le champ texte libre
// `cause_confirmee` n'est pas une cible numérique ; il attend un futur
// classifieur de causes entraîné sur historique d'atelier labellisé).

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

#[derive(Clone)]
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

/// Extrait `"key": <nombre>` n'importe où sur la ligne — les noms de
/// capteurs ne collisionnent pas avec les autres clés du JSONL de feedback
/// (`horodatage`, `cause_confirmee`, `notes`), donc pas besoin d'isoler
/// l'objet `"cas": {...}` : une recherche directe suffit.
fn line_number(line: &str, key: &str) -> Option<f32> {
    let pat = format!("\"{key}\"");
    let start = line.find(&pat)? + pat.len();
    let rest = line[start..].trim_start().strip_prefix(':')?.trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Charge les cas de feedback exploitables (10 capteurs + trim confirmé).
/// Fichier absent ou vide → aucun cas, pas une erreur (premier run avant
/// toute confirmation d'atelier).
fn load_feedback(path: &str) -> Vec<Sample> {
    let Ok(content) = std::fs::read_to_string(path)
    else
    {
        return Vec::new();
    };
    let mut samples = Vec::new();
    let mut skipped = 0usize;
    for line in content.lines()
    {
        if line.trim().is_empty()
        {
            continue;
        }
        let mut features = [0.0f32; N_FEATURES];
        let mut ok = true;
        for (k, name) in FEATURES.iter().enumerate()
        {
            match line_number(line, name)
            {
                Some(v) => features[k] = v,
                None =>
                {
                    ok = false;
                    break;
                },
            }
        }
        let trim = line_number(line, TARGET);
        match (ok, trim)
        {
            (true, Some(trim)) =>
            {
                // segment=0 : toujours inclus côté train (filtre <= 9),
                // jamais en val (== 10) ni test (>= 11) — le feedback ne
                // contamine jamais l'évaluation sur segments jamais vus.
                samples.push(Sample {
                    features,
                    trim,
                    segment: 0,
                });
            },
            _ => skipped += 1,
        }
    }
    if skipped > 0
    {
        println!("  ⚠ {skipped} cas de feedback ignorés (capteurs ou trim manquants)");
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
            for (m, f) in mean.iter_mut().zip(&s.features)
            {
                *m += f;
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

struct TrainResult {
    model: Sequential,
    norm: Normalizer,
    test_mae: f32,
    threshold: f32,
}

/// Entraîne le modèle `10 → 64 → 32 → 1` sur `train`, évalue sur `val`/`test`.
/// Factorisé pour entraîner deux fois de suite (baseline vs augmenté) avec
/// exactement les mêmes hyperparamètres — seule la composition de `train` change.
fn train_once(
    train: &[&Sample],
    val: &[&Sample],
    test: &[&Sample],
    n_epochs: usize,
) -> TrainResult {
    let norm = Normalizer::fit(train);

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

    let loss_fn = MseLoss::new();
    let mut opt = Adam::new(0.001);
    let mut shuffle_rng = Rng::new(SEED ^ 0xDEADBEEF);
    let mut order: Vec<usize> = (0..train.len()).collect();
    let n_batches = train.len() / BATCH;

    for _epoch in 0..n_epochs
    {
        for i in (1..order.len()).rev()
        {
            let j = shuffle_rng.next_usize(i + 1);
            order.swap(i, j);
        }
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
        }
    }

    let val_xs: Vec<[f32; N_FEATURES]> = val.iter().map(|s| s.features).collect();
    let val_preds = predict_batch(&mut model, &norm, &val_xs);
    let mut residuals: Vec<f32> = val
        .iter()
        .zip(&val_preds)
        .map(|(s, p)| (s.trim - p).abs())
        .collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = residuals[residuals.len() * 99 / 100];

    let test_mae = mae(&mut model, &norm, test);
    TrainResult {
        model,
        norm,
        test_mae,
        threshold,
    }
}

fn main() {
    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| resolve("data/opel_corsa_telemetry.csv"));
    let feedback_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| resolve("data/feedback.jsonl"));
    let n_epochs: usize = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(40);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  SciRust — RÉ-ENTRAÎNEMENT OBD2 DEPUIS LE FEEDBACK ATELIER ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("📊 PHASE 1 : CHARGEMENT");
    println!("──────────────────────");
    let base_samples = load_csv(&csv_path);
    let feedback_samples = load_feedback(&feedback_path);
    println!(
        "  CSV atelier : {} relevés ({csv_path})",
        base_samples.len()
    );
    println!(
        "  Feedback    : {} cas exploitables ({feedback_path})",
        feedback_samples.len()
    );

    let base_train: Vec<&Sample> = base_samples.iter().filter(|s| s.segment <= 9).collect();
    let val: Vec<&Sample> = base_samples.iter().filter(|s| s.segment == 10).collect();
    let test: Vec<&Sample> = base_samples.iter().filter(|s| s.segment >= 11).collect();

    let mut augmented_train: Vec<&Sample> = base_train.clone();
    augmented_train.extend(feedback_samples.iter());

    println!("  Train baseline : {} (CSV seul)", base_train.len());
    println!(
        "  Train augmenté : {} (CSV + feedback)",
        augmented_train.len()
    );
    println!("  Val   : {} (segment 10, jamais touché)", val.len());
    println!("  Test  : {} (segments 11-12, jamais touché)\n", test.len());

    println!("🧠 PHASE 2 : ENTRAÎNEMENT BASELINE (sans feedback)");
    println!("──────────────────────────────────────────────────");
    let baseline = train_once(&base_train, &val, &test, n_epochs);
    println!("  MAE test (baseline) : {:.3}% trim\n", baseline.test_mae);

    println!("🧠 PHASE 3 : ENTRAÎNEMENT AUGMENTÉ (CSV + feedback)");
    println!("───────────────────────────────────────────────────");
    let augmented = train_once(&augmented_train, &val, &test, n_epochs);
    println!("  MAE test (augmenté) : {:.3}% trim\n", augmented.test_mae);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  🏆 COMPARAISON                                            ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    let delta = augmented.test_mae - baseline.test_mae;
    let verdict = if feedback_samples.is_empty()
    {
        "aucun cas de feedback disponible — le modèle augmenté est identique à la baseline"
    }
    else if delta < -0.01
    {
        "le feedback AMÉLIORE la précision sur segments jamais vus"
    }
    else if delta > 0.01
    {
        "le feedback DÉGRADE légèrement la précision — à surveiller (cas peu représentatifs ?)"
    }
    else
    {
        "effet négligeable sur ce volume de feedback"
    };
    println!("  MAE baseline : {:.3}%", baseline.test_mae);
    println!(
        "  MAE augmenté : {:.3}% ({:+.3} pts)",
        augmented.test_mae, delta
    );
    println!("  → {verdict}\n");

    println!("💾 PHASE 4 : SAUVEGARDE DU MODÈLE AUGMENTÉ");
    println!("───────────────────────────────────────────");
    let models_dir = resolve("models");
    std::fs::create_dir_all(&models_dir).ok();
    let weights_path = format!("{models_dir}/obd2_real_fueltrim.safetensors");

    let mut model = augmented.model;
    let norm = augmented.norm;
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
    meta.insert(
        "anomaly_threshold_pct".into(),
        augmented.threshold.to_string(),
    );
    meta.insert(
        "dataset".into(),
        "PedroCuisinier2025/OBD2_panel_opel_2012 (Hugging Face, CC-BY-4.0) + feedback atelier"
            .into(),
    );
    meta.insert(
        "feedback_cases_used".into(),
        feedback_samples.len().to_string(),
    );
    meta.insert(
        "baseline_mae_no_feedback_pct".into(),
        format!("{:.4}", baseline.test_mae),
    );
    save_state_dict(&weights_path, &sd, Some(meta)).expect("échec sauvegarde poids");
    println!("  Poids → {weights_path}");

    // Round-trip : recharger dans un modèle vierge → prédictions identiques.
    let (loaded_sd, _) = load_state_dict(&weights_path).expect("échec lecture poids");
    let mut rng2 = PcgEngine::new(999);
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
}
