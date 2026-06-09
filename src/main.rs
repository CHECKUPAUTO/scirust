//! # OpenClaw-U Autonomous Core (bundled experimental binary)
//!
//! Noyau d'évolution asynchrone orienté bootstrap. **Ce binaire est une démo
//! expérimentale d'agent autonome** livrée avec le dépôt ; il n'est **pas** un
//! composant du framework de deep learning SciRust (voir `src/lib.rs` pour la
//! bibliothèque `scirust`). Il n'est requis ni pour construire ni pour utiliser
//! le framework.
//!
//! Usage : `cargo run --bin openclaw-u`

use chrono::Local;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use std::process::Command;
use std::time::Duration;
use tokio::io::AsyncBufReadExt;
use tokio::sync::mpsc;
use tokio::time::interval;

const STATE_FILE: &str = "state.json";
const EVOLUTION_LOG: &str = "evolution.log";
const HEARTBEAT_SECS: u64 = 15; // Accélération du cycle pour le développement
const MAX_HISTORY: usize = 50;

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
        if self.history.len() >= MAX_HISTORY {
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
    fn reflect(&self, state: &CoreState) -> String {
        let scirust_tasks = [
            "Générer la structure de Tenseur alignée pour AVX2/NEON",
            "Forger les abstractions de Traits pour le backend SIMD CPU",
            "Échafauder le module d'Attention et de graphe de calcul",
            "Vérifier l'intégrité de la compilation du pipeline tensoriel",
        ];
        let mut rng = rand::thread_rng();
        scirust_tasks.choose(&mut rng).unwrap_or(&"Maintien du noyau").to_string()
    }

    fn ensure_goal(&self, state: &mut CoreState) {
        if state.goals.is_empty() {
            let goal = self.reflect(state);
            state.log(&format!("[GoalEngine] Nouveau jalon SciRust assigné : {}", goal));
            state.goals.push(goal);
            state.task_count += 1;
        }
    }
}

#[derive(Debug)]
enum Event {
    UserMessage(String),
    SystemAlert(String),
    Shutdown,
}

// Persistance asynchrone non-bloquante (Correction du goulot d'étranglement)
async fn load_state() -> CoreState {
    if Path::new(STATE_FILE).exists() {
        if let Ok(raw) = tokio::fs::read_to_string(STATE_FILE).await {
            if let Ok(state) = serde_json::from_str::<CoreState>(&raw) {
                return state;
            }
        }
    }
    CoreState::birth()
}

async fn save_state(state: &CoreState) {
    if let Ok(json) = serde_json::to_string_pretty(state) {
        let _ = tokio::fs::write(STATE_FILE, json).await;
    }
}

async fn log_evolution(entry: &str) {
    let line = format!("[{}] {}\n", Local::now().to_rfc3339(), entry);
    if let Ok(mut file) = tokio::fs::OpenOptions::new().create(true).append(true).open(EVOLUTION_LOG).await {
        use tokio::io::AsyncWriteExt;
        let _ = file.write_all(line.as_bytes()).await;
    }
}

// Génération de code réel et structuré selon l'état d'avancement de l'agent
async fn propose_upgrade(state: &mut CoreState) {
    state.upgrade_attempts += 1;
    let stage = state.upgrade_successes % 3;

    let (target_file, code) = match stage {
        0 => (
            "src/tensor.rs",
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
            "src/simd_backend.rs",
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
            "src/upgrade_patch.rs",
            "// Intégration globale SciRust opérationnelle.\npub fn status() -> &'static str { \"SciRust V1 Core online\" }\n",
        ),
    };

    // Écritures asynchrones pour préserver l'exécuteur Tokio
    if let Err(e) = tokio::fs::write(target_file, code).await {
        state.log(&format!("[Evolution] Erreur écriture {} : {}", target_file, e));
        return;
    }

    // Validation par compilation non-bloquante
    let result = tokio::task::spawn_blocking(|| {
        Command::new("cargo").args(["check"]).output()
    }).await;

    match result {
        Ok(Ok(output)) => {
            if output.status.success() {
                state.upgrade_successes += 1;
                state.energy += 5.0;
                let msg = format!("Succès Mutation : {} compilé avec succès.", target_file);
                log_evolution(&msg).await;
                state.log(&format!("[Evolution] {}", msg));
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let first_err = stderr.lines().next().unwrap_or("Erreur de syntaxe Rust");
                state.energy -= 2.0;
                log_evolution(&format!("Échec de mutation sur {} : {}", target_file, first_err)).await;
                state.log(&format!("[Evolution] Rejet sur {} : {}", target_file, first_err));
            }
        }
        _ => { state.log("[Evolution] Panique ou erreur lors du processus cargo check"); }
    }
}

async fn heartbeat(state: &mut CoreState, engine: &GoalEngine) {
    state.touch();
    state.energy -= 0.5;
    engine.ensure_goal(state);

    println!(
        "[Heartbeat v{}] ⚡{:.1} | Attribué: {} | Mutations Réussies: {}/{}",
        state.version, state.energy, state.goals.last().unwrap_or(&"Aucun".to_string()),
        state.upgrade_successes, state.upgrade_attempts
    );
}

async fn external_listener(tx: mpsc::Sender<Event>) {
    let stdin = tokio::io::stdin();
    let reader = tokio::io::BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let event = if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("shutdown") {
            Event::Shutdown
        } else {
            Event::UserMessage(trimmed.to_string())
        };
        if tx.send(event).await.is_err() { break; }
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

    loop {
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
