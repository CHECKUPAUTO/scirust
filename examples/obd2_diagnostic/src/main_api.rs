// examples/obd2_diagnostic/src/main_api.rs
//
// SciRust — API HTTP de diagnostic OBD2
// =======================================
//
// API HTTP **sans aucune dépendance externe** : serveur `std::net::TcpListener`,
// parsing JSON minimal fait main. Charge deux modèles safetensors auto-suffisants
// (poids + métadonnées embarquées : features, normalisation, seuil, architecture) :
//
//   - `obd2_real_fueltrim`  : régression sur données réelles (Opel Corsa),
//     détecte les anomalies de mélange air/carburant par résidu.
//   - `obd2_megaverse`      : classifieur 1000 causes (démo de passage à
//     l'échelle sur données synthétiques, cf. README).
//
// Endpoints :
//   GET  /health              → état du service, modèles chargés
//   GET  /model                → métadonnées fueltrim (features, seuil…)
//   GET  /model/megaverse      → métadonnées megaverse (classes, précision…)
//   POST /diagnose             → relevés capteurs → trim prédit, résidu, verdict
//   POST /diagnose/megaverse   → 20 features brutes → top-3 causes prédites
//   POST /trip/start           → démarre un trajet → trip_id
//   POST /trip/{id}/reading    → ajoute un relevé au trajet → résidu glissant
//   GET  /trip/{id}/status     → stats du trajet sans ajouter de relevé
//   POST /feedback             → cas confirmé en atelier → archivé (JSONL) pour
//                                 un futur ré-entraînement
//
// Pourquoi le suivi de trajet ? Un relevé isolé peut manquer un défaut discret
// (ex. capteur MAF -35% : résidu ~2% sous le seuil ~9%, cf. limite documentée
// dans le README). Le bruit indépendant par relevé s'annule dans une moyenne
// (facteur 1/√n), donc un biais SYSTÉMATIQUE persistant — même minime — devient
// statistiquement détectable sur plusieurs relevés là où il ne dépasse jamais
// individuellement le seuil de repérage instantané.
//
// Lancement :
//   cargo run -p obd2_diagnostic --release --bin obd2_api            # port 8080
//   cargo run -p obd2_diagnostic --release --bin obd2_api -- 9090    # port choisi
//
// Exemple :
//   curl -s localhost:8080/diagnose -d '{"RPM":1898,"SPEED":39,
//     "THROTTLE_POS":23.5,"MAF":2.66,"COOLANT_TEMP":93,"INTAKE_TEMP":27,
//     "O2_B1S1":0.625,"ENGINE_LOAD":5.5,"INTAKE_PRESSURE":26,
//     "O2_B1S2":0.055,"LONG_FUEL_TRIM_1":17.97}'

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::io::safetensors::load_state_dict;
use scirust_core::nn::{Linear, Module, PcgEngine, ReLU, Sequential, Zeros};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

/// Reconstruit un MLP Linear/ReLU depuis les shapes du state dict
/// (clés Sequential : "{idx}.weight" / "{idx}.bias").
fn rebuild_mlp(sd: &HashMap<String, Tensor>) -> Sequential {
    let mut linear_idx: Vec<usize> = sd
        .keys()
        .filter_map(|k| k.strip_suffix(".weight")?.parse().ok())
        .collect();
    linear_idx.sort_unstable();

    let mut rng = PcgEngine::new(0); // écrasé par load_state_dict
    let mut model = Sequential::new();
    for (n, idx) in linear_idx.iter().enumerate()
    {
        let w = &sd[&format!("{idx}.weight")];
        let (in_f, out_f) = w.shape();
        model = model.add(Linear::new(in_f, out_f, &Zeros, &Zeros, &mut rng));
        if n + 1 < linear_idx.len()
        {
            model = model.add(ReLU::new());
        }
    }
    model
        .load_state_dict(sd)
        .expect("state dict incompatible avec l'architecture reconstruite");
    model
}

/// Trouve un fichier soit relatif au cwd, soit relatif à `examples/obd2_diagnostic/`.
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

/// Même logique que `resolve` mais pour un dossier (utilisé pour `data/`,
/// qui n'existe pas forcément encore côté fichier `feedback.jsonl`).
fn resolve_dir(rel_dir: &str) -> String {
    for c in [
        rel_dir.to_string(),
        format!("examples/obd2_diagnostic/{rel_dir}"),
    ]
    {
        if std::path::Path::new(&c).is_dir()
        {
            return c;
        }
    }
    rel_dir.to_string()
}

fn parse_csv_f32(meta: &HashMap<String, String>, key: &str) -> Vec<f32> {
    meta.get(key)
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_default()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|z| (z - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// Modèle 1 : régression trim carburant sur données réelles
// ─────────────────────────────────────────────────────────────────────────

struct FuelTrimService {
    model: Sequential,
    features: Vec<String>,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    target: String,
    target_mean: f32,
    target_std: f32,
    threshold: f32,
    arch: String,
    dataset: String,
}

impl FuelTrimService {
    fn load(path: &str) -> Self {
        let (sd, meta) =
            load_state_dict(path).unwrap_or_else(|e| panic!("impossible de charger {path}: {e}"));
        let model = rebuild_mlp(&sd);

        let features: Vec<String> = meta
            .get("features")
            .map(|s| s.split(',').map(str::to_string).collect())
            .unwrap_or_default();
        let feature_mean = parse_csv_f32(&meta, "feature_mean");
        let feature_std = parse_csv_f32(&meta, "feature_std");
        assert_eq!(
            features.len(),
            feature_mean.len(),
            "métadonnées incohérentes"
        );
        assert_eq!(
            features.len(),
            feature_std.len(),
            "métadonnées incohérentes"
        );

        FuelTrimService {
            model,
            features,
            feature_mean,
            feature_std,
            target: meta.get("target").cloned().unwrap_or_default(),
            target_mean: meta
                .get("target_mean")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            target_std: meta
                .get("target_std")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            threshold: meta
                .get("anomaly_threshold_pct")
                .and_then(|v| v.parse().ok())
                .unwrap_or(10.0),
            arch: meta.get("arch").cloned().unwrap_or_default(),
            dataset: meta.get("dataset").cloned().unwrap_or_default(),
        }
    }

    /// Prédit le trim (%) attendu pour un moteur SAIN dans cet état.
    fn predict_trim(&mut self, sensors: &[f32]) -> f32 {
        let normed: Vec<f32> = sensors
            .iter()
            .zip(self.feature_mean.iter().zip(&self.feature_std))
            .map(|(v, (m, s))| (v - m) / s)
            .collect();
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(normed, 1, self.features.len()));
        let pred = self.model.forward(&tape, x);
        tape.value(pred.idx()).data[0] * self.target_std + self.target_mean
    }

    /// Lit les capteurs + la cible observée dans le JSON, renvoie
    /// (observé, prédit, résidu) — factorisé pour /diagnose et /trip/*.
    fn compute_residual(&mut self, body: &str) -> Result<(f32, f32, f32), String> {
        let mut sensors = Vec::with_capacity(self.features.len());
        for f in self.features.clone()
        {
            match json_number(body, &f)
            {
                Some(v) => sensors.push(v),
                None => return Err(format!("champ capteur manquant ou invalide: {f}")),
            }
        }
        let observed = json_number(body, &self.target)
            .ok_or_else(|| format!("champ manquant: {} (trim observé)", self.target))?;
        let predicted = self.predict_trim(&sensors);
        Ok((observed, predicted, observed - predicted))
    }

    fn diagnose(&mut self, body: &str) -> Result<String, String> {
        let (observed, predicted, residual) = self.compute_residual(body)?;
        let anomaly = residual.abs() > self.threshold;

        let (verdict, interpretation) = if !anomaly
        {
            (
                "sain",
                "Le trim observé est cohérent avec l'état moteur : mélange air/carburant normal.",
            )
        }
        else if residual > 0.0
        {
            (
                "anomalie_melange_pauvre",
                "Trim observé anormalement HAUT vs l'état moteur : l'ECU sur-corrige un manque \
                 de carburant. Suspects : prise d'air / fuite de dépression, MAF sous-évaluant, \
                 pression carburant faible, injecteurs encrassés (logique P0171).",
            )
        }
        else
        {
            (
                "anomalie_melange_riche",
                "Trim observé anormalement BAS vs l'état moteur : l'ECU retire du carburant. \
                 Suspects : injecteur fuyard, MAF sur-évaluant, sonde O2 défaillante, \
                 régulateur de pression (logique P0172).",
            )
        };

        Ok(format!(
            "{{\"trim_observe_pct\":{observed:.2},\"trim_predit_pct\":{predicted:.2},\
             \"residu_pct\":{residual:.2},\"seuil_pct\":{:.2},\"anomalie\":{anomaly},\
             \"verdict\":\"{verdict}\",\"interpretation\":\"{interpretation}\"}}",
            self.threshold
        ))
    }

    fn model_info(&self) -> String {
        let feats = self
            .features
            .iter()
            .map(|f| format!("\"{f}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"arch\":\"{}\",\"features\":[{feats}],\"target\":\"{}\",\
             \"seuil_anomalie_pct\":{:.2},\"dataset\":\"{}\"}}",
            self.arch, self.target, self.threshold, self.dataset
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Modèle 2 : classifieur mégaverse (1000 causes, démo synthétique)
// ─────────────────────────────────────────────────────────────────────────

struct MegaverseService {
    model: Sequential,
    n_features: usize,
    n_classes: usize,
    arch: String,
    seed: String,
    test_acc: f32,
}

impl MegaverseService {
    fn load(path: &str) -> Self {
        let (sd, meta) =
            load_state_dict(path).unwrap_or_else(|e| panic!("impossible de charger {path}: {e}"));
        let model = rebuild_mlp(&sd);
        MegaverseService {
            model,
            n_features: meta
                .get("n_features")
                .and_then(|v| v.parse().ok())
                .unwrap_or(20),
            n_classes: meta
                .get("n_classes")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000),
            arch: meta.get("arch").cloned().unwrap_or_default(),
            seed: meta.get("seed").cloned().unwrap_or_default(),
            test_acc: meta
                .get("test_acc")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
        }
    }

    fn diagnose(&mut self, body: &str) -> Result<String, String> {
        let features = json_float_array(body, "features").ok_or_else(|| {
            format!(
                "champ manquant ou invalide: features (array de {} nombres)",
                self.n_features
            )
        })?;
        if features.len() != self.n_features
        {
            return Err(format!(
                "features attend {} valeurs, {} reçues",
                self.n_features,
                features.len()
            ));
        }

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(features, 1, self.n_features));
        let logits = self.model.forward(&tape, x);
        let probs = softmax(&tape.value(logits.idx()).data);

        let mut ranked: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top3 = &ranked[..3.min(ranked.len())];
        let top3_json = top3
            .iter()
            .map(|(c, p)| format!("{{\"cause_id\":{c},\"probabilite\":{p:.4}}}"))
            .collect::<Vec<_>>()
            .join(",");

        let mut extra = String::new();
        if let Some(true_label) = json_number(body, "vraie_cause").map(|v| v as usize)
        {
            let correct = top3[0].0 == true_label;
            extra = format!(",\"vraie_cause\":{true_label},\"correct\":{correct}");
        }
        Ok(format!("{{\"top3\":[{top3_json}]{extra}}}"))
    }

    fn model_info(&self) -> String {
        format!(
            "{{\"arch\":\"{}\",\"n_features\":{},\"n_classes\":{},\
             \"seed\":\"{}\",\"test_acc\":{:.4}}}",
            self.arch, self.n_features, self.n_classes, self.seed, self.test_acc
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Suivi de trajet : résidu glissant (mono-thread, pas de verrou nécessaire)
// ─────────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct TripState {
    n_readings: usize,
    sum_residual: f32,
}

struct TripStore {
    trips: HashMap<u64, TripState>,
    next_id: u64,
}

impl TripStore {
    fn new() -> Self {
        TripStore {
            trips: HashMap::new(),
            next_id: 1,
        }
    }

    fn start(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.trips.insert(id, TripState::default());
        id
    }

    fn add_reading(&mut self, id: u64, residual: f32) -> Result<(), String> {
        let state = self
            .trips
            .get_mut(&id)
            .ok_or_else(|| format!("trajet inconnu: {id} (POST /trip/start d'abord)"))?;
        state.n_readings += 1;
        state.sum_residual += residual;
        Ok(())
    }

    /// Verdict basé sur la moyenne du résidu, avec un seuil resserré en
    /// 1/√n : le bruit par relevé s'annule dans la moyenne, un biais
    /// persistant reste visible même sous le seuil de repérage instantané.
    fn status(&self, id: u64, single_reading_threshold: f32) -> Result<String, String> {
        let state = self
            .trips
            .get(&id)
            .ok_or_else(|| format!("trajet inconnu: {id}"))?;
        if state.n_readings == 0
        {
            return Ok(format!(
                "{{\"trip_id\":{id},\"n_releves\":0,\"residu_moyen_pct\":0.0,\"anomalie\":false}}"
            ));
        }
        let n = state.n_readings as f32;
        let mean_residual = state.sum_residual / n;
        let effective_threshold = (single_reading_threshold / n.sqrt()).max(1.0);
        let anomaly = mean_residual.abs() > effective_threshold;
        Ok(format!(
            "{{\"trip_id\":{id},\"n_releves\":{},\"residu_moyen_pct\":{mean_residual:.2},\
             \"seuil_effectif_pct\":{effective_threshold:.2},\"anomalie\":{anomaly}}}",
            state.n_readings
        ))
    }
}

fn trip_reading(
    fueltrim: &mut FuelTrimService,
    trips: &mut TripStore,
    id_str: &str,
    body: &str,
) -> Result<String, String> {
    let id: u64 = id_str.parse().map_err(|_| "trip_id invalide".to_string())?;
    let (_, _, residual) = fueltrim.compute_residual(body)?;
    trips.add_reading(id, residual)?;
    trips.status(id, fueltrim.threshold)
}

// ─────────────────────────────────────────────────────────────────────────
// Feedback atelier : archivage JSONL pour un futur ré-entraînement
// ─────────────────────────────────────────────────────────────────────────

fn append_feedback(path: &str, body: &str) -> Result<usize, String> {
    json_string(body, "cause_confirmee").ok_or_else(|| {
        "champ manquant: cause_confirmee (diagnostic confirmé en atelier)".to_string()
    })?;

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body_flat = body.replace(['\n', '\r'], " ");
    let line = format!("{{\"horodatage\":{ts},\"cas\":{body_flat}}}\n");

    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("écriture impossible ({path}): {e}"))?;
    f.write_all(line.as_bytes())
        .map_err(|e| format!("écriture impossible ({path}): {e}"))?;

    let total = std::fs::read_to_string(path)
        .map(|s| s.lines().count())
        .unwrap_or(1);
    Ok(total)
}

// ─────────────────────────────────────────────────────────────────────────
// Parsing JSON minimal (pas de dépendance externe)
// ─────────────────────────────────────────────────────────────────────────

/// Extrait la valeur numérique de `"key": <nombre>` dans un JSON plat.
fn json_number(body: &str, key: &str) -> Option<f32> {
    let pat = format!("\"{key}\"");
    let start = body.find(&pat)? + pat.len();
    let rest = body[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Extrait le tableau `"key": [n1, n2, …]`. Suppose des nombres simples
/// (pas de tableaux imbriqués).
fn json_float_array(body: &str, key: &str) -> Option<Vec<f32>> {
    let pat = format!("\"{key}\"");
    let start = body.find(&pat)? + pat.len();
    let rest = body[start..].trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('[')?;
    let end = rest.find(']')?;
    rest[..end]
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .ok()
}

/// Extrait la valeur `"key": "texte"` — ne gère pas les guillemets échappés,
/// suffisant pour les champs de diagnostic (pas de texte libre complexe attendu).
fn json_string(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let start = body.find(&pat)? + pat.len();
    let rest = body[start..].trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

// ─────────────────────────────────────────────────────────────────────────
// Serveur HTTP
// ─────────────────────────────────────────────────────────────────────────

struct ApiState {
    fueltrim: FuelTrimService,
    megaverse: MegaverseService,
    trips: TripStore,
    feedback_path: String,
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
}

fn handle(stream: &mut TcpStream, state: &mut ApiState) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    // Lit jusqu'à la fin des en-têtes, puis le corps selon Content-Length.
    let (head_end, content_len) = loop
    {
        match stream.read(&mut tmp)
        {
            Ok(0) => return,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(_) => return,
        }
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n")
        {
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            let cl = head
                .lines()
                .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                .and_then(|l| l.split(':').nth(1)?.trim().parse::<usize>().ok())
                .unwrap_or(0);
            break (pos + 4, cl);
        }
        if buf.len() > 65536
        {
            return; // requête déraisonnable
        }
    };
    while buf.len() < head_end + content_len
    {
        match stream.read(&mut tmp)
        {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(_) => return,
        }
    }

    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let body =
        String::from_utf8_lossy(&buf[head_end..head_end + content_len.min(buf.len() - head_end)])
            .to_string();
    let request_line = head.lines().next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let path_only = path.split('?').next().unwrap_or("");
    let segments: Vec<&str> = path_only
        .trim_start_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();

    let (status, json): (&str, String) = match (method, segments.as_slice())
    {
        ("GET", ["health"]) => (
            "200 OK",
            format!(
                "{{\"status\":\"ok\",\"models\":[{{\"name\":\"obd2_real_fueltrim\",\"arch\":\"{}\"}},\
                 {{\"name\":\"obd2_megaverse\",\"arch\":\"{}\"}}]}}",
                state.fueltrim.arch, state.megaverse.arch
            ),
        ),
        ("GET", ["model"]) => ("200 OK", state.fueltrim.model_info()),
        ("GET", ["model", "megaverse"]) => ("200 OK", state.megaverse.model_info()),
        ("POST", ["diagnose"]) => match state.fueltrim.diagnose(&body)
        {
            Ok(j) => ("200 OK", j),
            Err(e) => ("400 Bad Request", format!("{{\"erreur\":\"{e}\"}}")),
        },
        ("POST", ["diagnose", "megaverse"]) => match state.megaverse.diagnose(&body)
        {
            Ok(j) => ("200 OK", j),
            Err(e) => ("400 Bad Request", format!("{{\"erreur\":\"{e}\"}}")),
        },
        ("POST", ["trip", "start"]) =>
        {
            let id = state.trips.start();
            ("200 OK", format!("{{\"trip_id\":{id}}}"))
        },
        ("POST", ["trip", id_str, "reading"]) =>
        {
            match trip_reading(&mut state.fueltrim, &mut state.trips, id_str, &body)
            {
                Ok(j) => ("200 OK", j),
                Err(e) => ("400 Bad Request", format!("{{\"erreur\":\"{e}\"}}")),
            }
        },
        ("GET", ["trip", id_str, "status"]) => match id_str.parse::<u64>()
        {
            Ok(id) => match state.trips.status(id, state.fueltrim.threshold)
            {
                Ok(j) => ("200 OK", j),
                Err(e) => ("404 Not Found", format!("{{\"erreur\":\"{e}\"}}")),
            },
            Err(_) => (
                "400 Bad Request",
                "{\"erreur\":\"trip_id invalide\"}".to_string(),
            ),
        },
        ("POST", ["feedback"]) => match append_feedback(&state.feedback_path, &body)
        {
            Ok(total) => (
                "200 OK",
                format!("{{\"stored\":true,\"total_feedback\":{total}}}"),
            ),
            Err(e) => ("400 Bad Request", format!("{{\"erreur\":\"{e}\"}}")),
        },
        _ => (
            "404 Not Found",
            "{\"erreur\":\"routes: GET /health, GET /model, GET /model/megaverse, \
             POST /diagnose, POST /diagnose/megaverse, POST /trip/start, \
             POST /trip/{id}/reading, GET /trip/{id}/status, POST /feedback\"}"
                .to_string(),
        ),
    };
    respond(stream, status, "application/json", &json);
}

fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(8080);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  SciRust — API DE DIAGNOSTIC OBD2                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    let fueltrim_path = resolve("models/obd2_real_fueltrim.safetensors");
    let mut fueltrim = FuelTrimService::load(&fueltrim_path);
    println!("  Modèle 1 : {} ({})", fueltrim_path, fueltrim.arch);
    println!("             features: {}", fueltrim.features.join(", "));
    println!(
        "             cible: {} | seuil anomalie ±{:.2}%",
        fueltrim.target, fueltrim.threshold
    );

    let megaverse_path = resolve("models/obd2_megaverse.safetensors");
    let megaverse = MegaverseService::load(&megaverse_path);
    println!(
        "  Modèle 2 : {} ({}, {} classes, {:.2}% test acc)",
        megaverse_path,
        megaverse.arch,
        megaverse.n_classes,
        megaverse.test_acc * 100.0
    );

    // Auto-test au démarrage : un relevé sain réel doit être classé sain.
    let healthy = [
        1898.0, 39.0, 23.53, 2.66, 93.0, 27.0, 0.625, 5.49, 26.0, 0.055,
    ];
    let pred = fueltrim.predict_trim(&healthy);
    println!("  Auto-test : trim prédit {pred:.2}% pour un relevé réel sain ✓\n");

    let feedback_path = format!("{}/feedback.jsonl", resolve_dir("data"));
    let mut state = ApiState {
        fueltrim,
        megaverse,
        trips: TripStore::new(),
        feedback_path,
    };

    let addr = format!("127.0.0.1:{port}");
    let listener =
        TcpListener::bind(&addr).unwrap_or_else(|e| panic!("bind {addr} impossible: {e}"));
    println!("🚀 En écoute sur http://{addr}");
    println!(
        "   GET /health | GET /model[/megaverse] | POST /diagnose[/megaverse]\n\
         \x20  POST /trip/start | POST /trip/{{id}}/reading | GET /trip/{{id}}/status\n\
         \x20  POST /feedback\n"
    );

    for mut s in listener.incoming().flatten()
    {
        handle(&mut s, &mut state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Parsing JSON minimal ────────────────────────────────────────────

    #[test]
    fn json_number_parses_int_and_float() {
        let body = r#"{"RPM":1898,"trim":-3.5}"#;
        assert_eq!(json_number(body, "RPM"), Some(1898.0));
        assert_eq!(json_number(body, "trim"), Some(-3.5));
    }

    #[test]
    fn json_number_missing_key_is_none() {
        assert_eq!(json_number(r#"{"RPM":1898}"#, "SPEED"), None);
    }

    #[test]
    fn json_number_rejects_non_numeric_value() {
        assert_eq!(json_number(r#"{"RPM":"abc"}"#, "RPM"), None);
    }

    #[test]
    fn json_float_array_parses() {
        let body = r#"{"features":[0.1, 0.2,0.30,-1]}"#;
        assert_eq!(
            json_float_array(body, "features"),
            Some(vec![0.1, 0.2, 0.30, -1.0])
        );
    }

    #[test]
    fn json_float_array_missing_key_is_none() {
        assert_eq!(json_float_array(r#"{"other":[1,2]}"#, "features"), None);
    }

    #[test]
    fn json_string_parses() {
        let body = r#"{"cause_confirmee":"prise_air","notes":"joint durci"}"#;
        assert_eq!(
            json_string(body, "cause_confirmee"),
            Some("prise_air".to_string())
        );
        assert_eq!(json_string(body, "notes"), Some("joint durci".to_string()));
    }

    #[test]
    fn json_string_missing_key_is_none() {
        assert_eq!(json_string(r#"{"RPM":1898}"#, "cause_confirmee"), None);
    }

    // ── TripStore : résidu glissant ──────────────────────────────────────
    // Reproduit exactement le scénario vérifié manuellement (README) :
    // un biais constant de +3% (sous le seuil ponctuel 8.85%) reste
    // non détecté pendant 8 relevés, puis bascule au 9e relevé quand
    // 8.85/√9 = 2.95 passe sous 3.00.

    #[test]
    fn trip_threshold_shrinks_and_flags_persistent_bias() {
        let mut trips = TripStore::new();
        let id = trips.start();
        let threshold = 8.85f32;

        for n in 1..=8
        {
            trips.add_reading(id, 3.0).unwrap();
            let status = trips.status(id, threshold).unwrap();
            assert!(
                status.contains("\"anomalie\":false"),
                "relevé {n}: devrait rester sain, eu {status}"
            );
        }

        trips.add_reading(id, 3.0).unwrap(); // 9e relevé
        let status = trips.status(id, threshold).unwrap();
        assert!(
            status.contains("\"anomalie\":true"),
            "relevé 9: devrait basculer en anomalie, eu {status}"
        );
    }

    #[test]
    fn trip_zero_bias_never_flags() {
        let mut trips = TripStore::new();
        let id = trips.start();
        for _ in 0..50
        {
            trips.add_reading(id, 0.0).unwrap();
        }
        let status = trips.status(id, 8.85).unwrap();
        assert!(status.contains("\"anomalie\":false"));
    }

    #[test]
    fn trip_unknown_id_is_error() {
        let trips = TripStore::new();
        assert!(trips.status(999, 8.85).is_err());
    }

    #[test]
    fn trip_add_reading_unknown_id_is_error() {
        let mut trips = TripStore::new();
        assert!(trips.add_reading(999, 1.0).is_err());
    }

    // ── FuelTrimService : contre les poids réels committés ──────────────

    fn load_fueltrim_for_test() -> FuelTrimService {
        let path = resolve("models/obd2_real_fueltrim.safetensors");
        FuelTrimService::load(&path)
    }

    #[test]
    fn fueltrim_healthy_reading_is_sain() {
        let mut svc = load_fueltrim_for_test();
        let body = r#"{"RPM":1898,"SPEED":39,"THROTTLE_POS":23.53,"MAF":2.66,
            "COOLANT_TEMP":93,"INTAKE_TEMP":27,"O2_B1S1":0.625,"ENGINE_LOAD":5.49,
            "INTAKE_PRESSURE":26,"O2_B1S2":0.055,"LONG_FUEL_TRIM_1":17.97}"#;
        let json = svc.diagnose(body).expect("diagnose should succeed");
        assert!(json.contains("\"verdict\":\"sain\""), "eu: {json}");
    }

    #[test]
    fn fueltrim_lean_bias_flags_pauvre() {
        let mut svc = load_fueltrim_for_test();
        // Même relevé, trim gonflé de +14% (symptôme prise d'air).
        let body = r#"{"RPM":1898,"SPEED":39,"THROTTLE_POS":23.53,"MAF":2.66,
            "COOLANT_TEMP":93,"INTAKE_TEMP":27,"O2_B1S1":0.625,"ENGINE_LOAD":5.49,
            "INTAKE_PRESSURE":26,"O2_B1S2":0.055,"LONG_FUEL_TRIM_1":31.97}"#;
        let json = svc.diagnose(body).expect("diagnose should succeed");
        assert!(
            json.contains("\"verdict\":\"anomalie_melange_pauvre\""),
            "eu: {json}"
        );
    }

    #[test]
    fn fueltrim_missing_sensor_is_error() {
        let mut svc = load_fueltrim_for_test();
        let err = svc.diagnose(r#"{"RPM":1898}"#).unwrap_err();
        assert!(
            err.contains("SPEED") || err.contains("capteur"),
            "eu: {err}"
        );
    }

    // ── MegaverseService : contre une signature de classe connue ────────
    // Features exactes de la cause 200 (générées par build_signatures()
    // dans main_megaverse.rs, seed 42, sans bruit) — même vecteur que
    // celui vérifié manuellement via curl pendant le développement du PR.
    const CLASS_200_SIGNATURE: [f32; 20] = [
        0.5000, 0.5000, 0.0622, 0.8148, 0.5000, 0.5000, 0.0592, 0.0633, 0.5000, 0.5000, 0.5000,
        0.9370, 0.5000, 0.9065, 0.5000, 0.5000, 0.5000, 0.9021, 0.5000, 0.1791,
    ];

    fn load_megaverse_for_test() -> MegaverseService {
        let path = resolve("models/obd2_megaverse.safetensors");
        MegaverseService::load(&path)
    }

    #[test]
    fn megaverse_predicts_known_class_200() {
        let mut svc = load_megaverse_for_test();
        let features_json = CLASS_200_SIGNATURE
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let body = format!("{{\"features\":[{features_json}],\"vraie_cause\":200}}");
        let json = svc.diagnose(&body).expect("diagnose should succeed");
        assert!(json.contains("\"cause_id\":200"), "eu: {json}");
        assert!(json.contains("\"correct\":true"), "eu: {json}");
    }

    #[test]
    fn megaverse_wrong_feature_count_is_error() {
        let mut svc = load_megaverse_for_test();
        let err = svc.diagnose(r#"{"features":[0.1,0.2]}"#).unwrap_err();
        assert!(err.contains("20"), "eu: {err}");
    }

    #[test]
    fn megaverse_missing_features_is_error() {
        let mut svc = load_megaverse_for_test();
        assert!(svc.diagnose(r#"{"other":1}"#).is_err());
    }
}
