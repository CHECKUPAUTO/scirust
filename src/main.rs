//! # OpenClaw-U Autonomous Core (bundled experimental binary)
//!
//! Noyau d'évolution asynchrone orienté bootstrap. **Ce binaire est une démo
//! expérimentale d'agent autonome** livrée avec le dépôt ; il n'est **pas** un
//! composant du framework de deep learning SciRust (voir `src/lib.rs` pour la
//! bibliothèque `scirust`). Il n'est requis ni pour construire ni pour utiliser
//! le framework.
//!
//! ## Hardening (audit `AUDIT_COMPLET.md`, finding S2)
//!
//! La persistance (`state.json`) est **intégrité-protégée** par un tag
//! HMAC-SHA256 dérivé d'une clé d'au moins 32 octets passée via
//! `OPENCLAW_U_STATE_KEY`. Sans clé valide, la démo fonctionne sans lire ni
//! écrire d'état persistant. Un `state.json` dont le MAC ne
//! vérifie pas (fichier forgé, édité, ou issu d'une autre machine/clé) est
//! **rejeté** et l'agent repart d'un état vierge (fail-safe).
//!
//! La **mutation autonome** (écriture de fichiers source générés + appel
//! `cargo check`) est **désactivée par défaut** : elle n'a lieu que si la
//! variable d'environnement `OPENCLAW_UNSAFE_MUTATE=1` est positionnée. Sans
//! elle, l'agent s'exécute en lecture/heartbeat purs — il ne mute ni le code
//! ni le système de build. Les fichiers générés sont écrits sous
//! `target/openclaw-u/` (jamais dans `src/`), pour ne pas polluer l'arbre
//! source du framework.
//!
//! Usage : `cargo run --features openclaw --bin openclaw-u`
//!         `OPENCLAW_UNSAFE_MUTATE=1 cargo run --features openclaw --bin openclaw-u`

use chrono::Local;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::path::Path;
use std::process::Command;
use std::time::Duration;
use tokio::io::AsyncBufReadExt;
use tokio::sync::mpsc;
use tokio::time::interval;

const STATE_FILE: &str = "state.json";
const EVOLUTION_LOG: &str = "evolution.log";
/// Generated source files are written here (never into `src/`), so the demo
/// cannot pollute the framework's source tree.
const UPGRADE_DIR: &str = "target/openclaw-u";
/// Env var that must be set to `1` to enable autonomous code mutation + the
/// `cargo check` validation step. Off by default (defense-in-depth).
const MUTATE_ENV: &str = "OPENCLAW_UNSAFE_MUTATE";
/// Env var carrying the HMAC key for `state.json`.
const STATE_KEY_ENV: &str = "OPENCLAW_U_STATE_KEY";
const HEARTBEAT_SECS: u64 = 15; // Accélération du cycle pour le développement
const MAX_HISTORY: usize = 50;

fn state_key() -> Option<Vec<u8>> {
    std::env::var(STATE_KEY_ENV)
        .ok()
        .map(|value| value.into_bytes())
        .filter(|value| value.len() >= 32)
}

/// HMAC-SHA256 (RFC 2104) over `msg` keyed by the resolved state key.
fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK
    {
        let h = Sha256::digest(key);
        k[..32].copy_from_slice(&h);
    }
    else
    {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK
    {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(msg);
    let inner_h = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner_h);
    let mut out = [0u8; 32];
    out.copy_from_slice(&outer.finalize());
    out
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    use std::fmt::Write;
    for b in bytes
    {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Constant-time byte comparison (length-checked).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len()
    {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// On-disk envelope: the state JSON plus its HMAC tag over the canonical
/// `state` field. A tampered or foreign file fails verification.
#[derive(Serialize, Deserialize)]
struct StateEnvelope {
    state: CoreState,
    mac: String,
}

impl StateEnvelope {
    fn seal(state: &CoreState) -> Result<Self, String> {
        let key =
            state_key().ok_or_else(|| format!("{STATE_KEY_ENV} must contain at least 32 bytes"))?;
        let json = serde_json::to_string(state)
            .map_err(|error| format!("cannot serialize OpenClaw state: {error}"))?;
        let mac = to_hex(&hmac_sha256(&key, json.as_bytes()));
        Ok(Self {
            state: state.clone(),
            mac,
        })
    }

    fn verify(&self) -> bool {
        let Some(key) = state_key()
        else
        {
            return false;
        };
        let json = match serde_json::to_string(&self.state)
        {
            Ok(s) => s,
            Err(_) => return false,
        };
        let expected = to_hex(&hmac_sha256(&key, json.as_bytes()));
        let stored = match hex_decode(&self.mac)
        {
            Some(b) => b,
            None => return false,
        };
        let want = match hex_decode(&expected)
        {
            Some(b) => b,
            None => return false,
        };
        ct_eq(&stored, &want)
    }
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0
    {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len()
    {
        let hi = (b[i] as char).to_digit(16)?;
        let lo = (b[i + 1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    Some(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoreState {
    energy: f32,
    goals: Vec<String>,
    history: VecDeque<String>,
    version: String,
    created_at: String,
    last_heartbeat: String,
    task_count: u64,
    upgrade_attempts: u64,
    upgrade_successes: u64,
}

impl CoreState {
    fn birth() -> Self {
        let now = Local::now().to_rfc3339();
        Self {
            energy: 50.0,
            goals: Vec::new(),
            history: VecDeque::new(),
            version: "0.3.0-scirust-bootstrap".to_string(),
            created_at: now.clone(),
            last_heartbeat: now,
            task_count: 0,
            upgrade_attempts: 0,
            upgrade_successes: 0,
        }
    }

    fn log(&mut self, entry: &str) {
        let line = format!("[{}] {}", Local::now().format("%H:%M:%S"), entry);
        if self.history.len() >= MAX_HISTORY
        {
            self.history.pop_front();
        }
        self.history.push_back(line);
    }

    fn touch(&mut self) {
        self.last_heartbeat = Local::now().to_rfc3339();
    }
}

struct GoalEngine;

impl GoalEngine {
    fn reflect(&self, _state: &CoreState) -> String {
        let scirust_tasks = [
            "Générer la structure de Tenseur alignée pour AVX2/NEON",
            "Forger les abstractions de Traits pour le backend SIMD CPU",
            "Échafauder le module d'Attention et de graphe de calcul",
            "Vérifier l'intégrité de la compilation du pipeline tensoriel",
        ];
        let mut rng = rand::thread_rng();
        scirust_tasks
            .choose(&mut rng)
            .unwrap_or(&"Maintien du noyau")
            .to_string()
    }

    fn ensure_goal(&self, state: &mut CoreState) {
        if state.goals.is_empty()
        {
            let goal = self.reflect(state);
            state.log(&format!(
                "[GoalEngine] Nouveau jalon SciRust assigné : {}",
                goal
            ));
            state.goals.push(goal);
            state.task_count += 1;
        }
    }
}

#[derive(Debug)]
enum Event {
    UserMessage(String),
    #[allow(dead_code)]
    SystemAlert(String),
    Shutdown,
}

// Persistance asynchrone non-bloquante (Correction du goulot d'étranglement).
// Le `state.json` est integrity-protégé par un tag HMAC-SHA256 (envelope). Un
// fichier forgé, édité à la main, ou produit sous une autre clé est rejeté :
// l'agent repart d'un état vierge (fail-safe) plutôt que de charger un état
// non authentifié qui pourrait piloter la génération de code.
async fn load_state() -> CoreState {
    if state_key().is_none()
    {
        eprintln!("[openclaw-u] {STATE_KEY_ENV} absent ou trop courte : persistance désactivée.");
        return CoreState::birth();
    }
    if Path::new(STATE_FILE).exists()
    {
        if let Ok(raw) = tokio::fs::read_to_string(STATE_FILE).await
        {
            match serde_json::from_str::<StateEnvelope>(&raw)
            {
                Ok(env) if env.verify() => return env.state,
                Ok(_) =>
                {
                    eprintln!(
                        "[openclaw-u] state.json rejeté : MAC invalide \
                         (fichier forgé/édité/clé différente). Reprise vierge."
                    );
                },
                Err(_) =>
                {
                    eprintln!("[openclaw-u] state.json illisible ou non signé. Reprise vierge.");
                },
            }
        }
    }
    CoreState::birth()
}

async fn save_state(state: &CoreState) {
    let env = match StateEnvelope::seal(state)
    {
        Ok(envelope) => envelope,
        Err(_) => return,
    };
    let json = match serde_json::to_string_pretty(&env)
    {
        Ok(json) => json,
        Err(error) =>
        {
            eprintln!("[openclaw-u] impossible de sérialiser l'état signé : {error}");
            return;
        },
    };
    if let Err(error) = tokio::fs::write(STATE_FILE, json).await
    {
        eprintln!("[openclaw-u] impossible d'écrire {STATE_FILE} : {error}");
    }
}

/// Whether autonomous code mutation (file writes + `cargo check`) is enabled.
/// Requires `OPENCLAW_UNSAFE_MUTATE=1`. Off by default — the agent then only
/// runs heartbeat/reading and never touches the source tree or the build.
fn mutation_enabled() -> bool {
    std::env::var(MUTATE_ENV)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v == "yes")
        .unwrap_or(false)
}

async fn log_evolution(entry: &str) {
    let line = format!("[{}] {}\n", Local::now().to_rfc3339(), entry);
    if let Ok(mut file) = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(EVOLUTION_LOG)
        .await
    {
        use tokio::io::AsyncWriteExt;
        let _ = file.write_all(line.as_bytes()).await;
    }
}

// Génération de code réel et structuré selon l'état d'avancement de l'agent.
// Hardening : ne s'exécute que si `OPENCLAW_UNSAFE_MUTATE=1` (mutation_enabled).
// Les fichiers générés vont sous `target/openclaw-u/` (jamais dans `src/`), et
// le `cargo check` de validation tourne dans ce même répertoire isolé.
async fn propose_upgrade(state: &mut CoreState) {
    if !mutation_enabled()
    {
        // Mutation désactivée par défaut — l'agent ne touche ni au code ni au build.
        return;
    }
    state.upgrade_attempts += 1;
    let stage = state.upgrade_successes % 3;

    // Répertoire isolé pour les artefacts générés (jamais l'arbre source).
    if let Err(e) = tokio::fs::create_dir_all(UPGRADE_DIR).await
    {
        state.log(&format!(
            "[Evolution] Impossible de créer {UPGRADE_DIR} : {e}"
        ));
        return;
    }

    let (file_name, code) = match stage
    {
        0 => (
            "tensor.rs",
            r#"//! SciRust Tensor Core - Alignement mémoire strict pour SIMD
#[repr(align(32))]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        // Allocation brute initialisée à zéro
        Self {
            data: vec![0.0; size],
            shape,
            strides: vec![1; size],
        }
    }
}
"#,
        ),
        1 => (
            "simd_backend.rs",
            r#"//! SciRust SIMD Abstraction Layer (AVX2 / NEON)
pub trait SimdKernel {
    fn fma_vector(a: &[f32], b: &[f32], c: &mut [f32]);
}

pub struct CpuBackend;

impl SimdKernel {
    #[inline(always)]
    pub fn fma_naive(a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..a.len() {
            c[i] += a[i] * b[i];
        }
    }
}
"#,
        ),
        _ => (
            "upgrade_patch.rs",
            "// Intégration globale SciRust opérationnelle.\npub fn status() -> &'static str { \"SciRust V1 Core online\" }\n",
        ),
    };
    let target_file = format!("{UPGRADE_DIR}/{file_name}");

    // Écritures asynchrones pour préserver l'exécuteur Tokio
    if let Err(e) = tokio::fs::write(&target_file, code).await
    {
        state.log(&format!(
            "[Evolution] Erreur écriture {} : {}",
            target_file, e
        ));
        return;
    }

    // Validation par compilation non-bloquante, dans le répertoire isolé. On
    // compile les fichiers générés comme une cible autonome (rustc) plutôt que
    // d'invoquer `cargo check` à la racine du workspace, pour ne jamais risquer
    // de valider/charger du code dans le framework SciRust lui-même.
    let target_file_clone = target_file.clone();
    let metadata_file = format!("{target_file}.rmeta");
    let metadata_file_clone = metadata_file.clone();
    let result = tokio::task::spawn_blocking(move || {
        Command::new("rustc")
            .args([
                "--edition",
                "2021",
                "--crate-type",
                "lib",
                "--emit=metadata",
                "-o",
            ])
            .arg(&metadata_file_clone)
            .arg(&target_file_clone)
            .output()
    })
    .await;

    match result
    {
        Ok(Ok(output)) =>
        {
            if output.status.success()
            {
                state.upgrade_successes += 1;
                state.energy += 5.0;
                let msg = format!("Succès Mutation : {} compilé avec succès.", target_file);
                log_evolution(&msg).await;
                state.log(&format!("[Evolution] {}", msg));
            }
            else
            {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let first_err = stderr.lines().next().unwrap_or("Erreur de syntaxe Rust");
                state.energy -= 2.0;
                log_evolution(&format!(
                    "Échec de mutation sur {} : {}",
                    target_file, first_err
                ))
                .await;
                state.log(&format!(
                    "[Evolution] Rejet sur {} : {}",
                    target_file, first_err
                ));
            }
        },
        _ =>
        {
            state.log("[Evolution] Panique ou erreur lors du processus rustc");
        },
    }
    let _ = tokio::fs::remove_file(metadata_file).await;
}

async fn heartbeat(state: &mut CoreState, engine: &GoalEngine) {
    state.touch();
    state.energy -= 0.5;
    engine.ensure_goal(state);

    println!(
        "[Heartbeat v{}] ⚡{:.1} | Attribué: {} | Mutations Réussies: {}/{}",
        state.version,
        state.energy,
        state.goals.last().unwrap_or(&"Aucun".to_string()),
        state.upgrade_successes,
        state.upgrade_attempts
    );
}

async fn external_listener(tx: mpsc::Sender<Event>) {
    let stdin = tokio::io::stdin();
    let reader = tokio::io::BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Ok(Some(line)) = lines.next_line().await
    {
        let trimmed = line.trim();
        if trimmed.is_empty()
        {
            continue;
        }
        let event = if trimmed.eq_ignore_ascii_case("exit")
            || trimmed.eq_ignore_ascii_case("shutdown")
        {
            Event::Shutdown
        }
        else
        {
            Event::UserMessage(trimmed.to_string())
        };
        if tx.send(event).await.is_err()
        {
            break;
        }
    }
}

#[tokio::main]
async fn main() {
    let mut state = load_state().await;
    let engine = GoalEngine;
    let (tx, mut rx) = mpsc::channel::<Event>(32);

    tokio::spawn(external_listener(tx));
    let mut heartbeat_interval = interval(Duration::from_secs(HEARTBEAT_SECS));

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       OpenClaw-U v0.3.0 — SciRust Bootstrapper           ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    loop
    {
        tokio::select! {
            _ = heartbeat_interval.tick() => {
                heartbeat(&mut state, &engine).await;

                // Mutation active déclenchée par l'intentionnalité de l'agent
                if state.energy > 5.0 {
                    propose_upgrade(&mut state).await;
                }
                save_state(&state).await;
            }
            Some(event) = rx.recv() => {
                match event {
                    Event::UserMessage(msg) => {
                        println!("[Agent] Message enregistré : '{}'. Focus maintenu sur SciRust.", msg);
                        state.log(&format!("Opérateur: {}", msg));
                    }
                    Event::SystemAlert(alert) => {
                        state.log(&format!("ALERTE: {}", alert));
                    }
                    Event::Shutdown => {
                        println!("[Agent] Sauvegarde finale et mise en veille...");
                        break;
                    }
                }
                save_state(&state).await;
            }
        }
    }
    save_state(&state).await;
}
