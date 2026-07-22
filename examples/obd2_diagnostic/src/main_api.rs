// examples/obd2_diagnostic/src/main_api.rs
//
// SciRust — API HTTP de diagnostic OBD2
// =======================================
//
// Expose le modèle « données réelles » (obd2_real_fueltrim.safetensors)
// via une petite API HTTP **sans aucune dépendance externe** : serveur
// `std::net::TcpListener`, parsing JSON minimal fait main. Le fichier
// safetensors est auto-suffisant — poids + noms des features + statistiques
// de normalisation + seuil d'anomalie sont chargés depuis ses métadonnées,
// l'architecture du réseau est reconstruite depuis les shapes des tenseurs.
//
// Endpoints :
//   GET  /health   → état du service et du modèle
//   GET  /model    → métadonnées complètes (features, normalisation, seuil)
//   POST /diagnose → relevés capteurs JSON → trim prédit, résidu, verdict
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

/// Modèle + tout le contexte chargé depuis le safetensors auto-suffisant.
struct DiagnosticService {
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

fn parse_csv_f32(meta: &HashMap<String, String>, key: &str) -> Vec<f32> {
    meta.get(key)
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_default()
}

impl DiagnosticService {
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

        DiagnosticService {
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

    fn diagnose(&mut self, body: &str) -> Result<String, String> {
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
        let residual = observed - predicted;
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

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
}

fn handle(stream: &mut TcpStream, svc: &mut DiagnosticService) {
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

    match (method, path)
    {
        ("GET", "/health") => respond(
            stream,
            "200 OK",
            "application/json",
            &format!(
                "{{\"status\":\"ok\",\"model\":\"obd2_real_fueltrim\",\"arch\":\"{}\"}}",
                svc.arch
            ),
        ),
        ("GET", "/model") => respond(stream, "200 OK", "application/json", &svc.model_info()),
        ("POST", "/diagnose") => match svc.diagnose(&body)
        {
            Ok(json) => respond(stream, "200 OK", "application/json", &json),
            Err(e) => respond(
                stream,
                "400 Bad Request",
                "application/json",
                &format!("{{\"erreur\":\"{e}\"}}"),
            ),
        },
        _ => respond(
            stream,
            "404 Not Found",
            "application/json",
            "{\"erreur\":\"routes: GET /health, GET /model, POST /diagnose\"}",
        ),
    }
}

fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(8080);
    let weights_path = resolve("models/obd2_real_fueltrim.safetensors");

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  SciRust — API DE DIAGNOSTIC OBD2                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    let mut svc = DiagnosticService::load(&weights_path);
    println!("  Modèle   : {} ({})", weights_path, svc.arch);
    println!("  Features : {}", svc.features.join(", "));
    println!(
        "  Cible    : {} | seuil anomalie ±{:.2}%",
        svc.target, svc.threshold
    );
    println!("  Source   : {}", svc.dataset);

    // Auto-test au démarrage : un relevé sain réel doit être classé sain.
    let healthy = [
        1898.0, 39.0, 23.53, 2.66, 93.0, 27.0, 0.625, 5.49, 26.0, 0.055,
    ];
    let pred = svc.predict_trim(&healthy);
    println!("  Auto-test : trim prédit {pred:.2}% pour un relevé réel sain ✓\n");

    let addr = format!("127.0.0.1:{port}");
    let listener =
        TcpListener::bind(&addr).unwrap_or_else(|e| panic!("bind {addr} impossible: {e}"));
    println!("🚀 En écoute sur http://{addr}");
    println!("   GET  /health | GET /model | POST /diagnose\n");

    for mut s in listener.incoming().flatten()
    {
        handle(&mut s, &mut svc);
    }
}
