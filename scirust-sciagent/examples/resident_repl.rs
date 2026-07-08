//! **Resident REPL** — an interactive on-device generation loop. The model is
//! mirrored into VRAM **once** and stays resident across turns (no per-prompt
//! weight upload), so each line you type is prompt → KV-cached sampled
//! continuation on the fully-resident GPU path. Sampling knobs are adjustable
//! live, without restarting.
//!
//! Reads prompts from stdin (one per line); works interactively or piped
//! (`printf 'a\nb\n' | cargo run … --example resident_repl`). EOF (Ctrl-D) or
//! `:q` exits.
//!
//! Runtime commands (start a line with `:`):
//! - `:q` / `:quit` — exit
//! - `:temp <f>` — temperature (0 = greedy)   · `:topk <n>` · `:topp <f>`
//! - `:rep <f>` — repetition penalty          · `:new <n>` — max new tokens
//! - `:seed <n>` — RNG seed                    · `:show` — print current settings
//! - `:help` — list commands
//!
//! Startup env mirrors `resident_generate`: `SCIAGENT_CKPT` (checkpoint dir; else
//! a fresh byte demo model), and the same `SCIAGENT_TEMP/TOP_K/TOP_P/REP_PENALTY/
//! SEED/MAX_NEW` initial defaults. Exit code 2 means no GPU adapter.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::generate::SamplingParams;
use scirust_sciagent::gpu::ResidentModel;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{latest_checkpoint, load_checkpoint, read_meta};

fn byte_config() -> SciAgentConfig {
    SciAgentConfig {
        vocab_size: 256,
        d_model: 256,
        n_layers: 6,
        n_heads: 8,
        n_kv_heads: 2,
        d_ff: 512,
        max_seq_len: 512,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Parse a prompt line into token ids: raw bytes for a byte-level model, else
/// comma-separated in-vocab ids.
fn parse_prompt(line: &str, byte_level: bool, vocab: usize) -> Vec<u32> {
    if byte_level
    {
        line.bytes().map(u32::from).collect()
    }
    else
    {
        line.split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .filter(|&t| (t as usize) < vocab)
            .collect()
    }
}

fn show_settings(p: &SamplingParams, max_new: usize, seed: u64) {
    println!(
        "  T {} · top_k {} · top_p {} · rep {} · max_new {max_new} · seed {seed}",
        p.temperature, p.top_k, p.top_p, p.repetition_penalty
    );
}

fn main() {
    // Load the base model once — it becomes resident for the whole session.
    let mut model;
    if let Ok(ckpt) = std::env::var("SCIAGENT_CKPT")
    {
        let dir = latest_checkpoint(Path::new(&ckpt)).unwrap_or_else(|| ckpt.clone().into());
        let meta = read_meta(&dir).unwrap_or_else(|e| panic!("read meta {}: {e}", dir.display()));
        model = SciAgentModel::new(&meta.config);
        load_checkpoint(&mut model, &dir).unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));
        println!(
            "model loaded from {} (vocab {})",
            dir.display(),
            meta.config.vocab_size
        );
    }
    else
    {
        model = SciAgentModel::new(&byte_config());
        println!("no SCIAGENT_CKPT — fresh byte model (demo; set it for real generation)");
    }
    let config = model.config.clone();
    assert!(
        config.tie_embeddings,
        "resident REPL requires a tied-embedding model"
    );

    let Some(rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    let byte_level = config.vocab_size >= 256;
    println!(
        "resident REPL on: {} — model stays in VRAM across turns",
        rm.adapter_name()
    );

    // Mutable session state (adjustable live via `:` commands).
    let mut params = SamplingParams {
        temperature: env_f32("SCIAGENT_TEMP", 0.8),
        top_k: env_usize("SCIAGENT_TOP_K", 0),
        top_p: env_f32("SCIAGENT_TOP_P", 0.95),
        repetition_penalty: env_f32("SCIAGENT_REP_PENALTY", 1.1),
        repetition_window: env_usize("SCIAGENT_REP_WINDOW", 64),
    };
    let mut max_new = env_usize("SCIAGENT_MAX_NEW", 128);
    let mut seed: u64 = std::env::var("SCIAGENT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    print!("type a prompt (`:help` for commands, Ctrl-D to quit)\n> ");
    let _ = io::stdout().flush();

    let stdin = io::stdin();
    for line in stdin.lock().lines()
    {
        let Ok(line) = line
        else
        {
            break;
        };
        let trimmed = line.trim_end_matches(['\r', '\n']);

        // Runtime commands.
        if let Some(cmd) = trimmed.strip_prefix(':')
        {
            let mut it = cmd.split_whitespace();
            let head = it.next().unwrap_or("");
            let arg = it.next().unwrap_or("");
            match head
            {
                "q" | "quit" | "exit" => break,
                "help" =>
                {
                    println!(
                        "  :temp <f>  :topk <n>  :topp <f>  :rep <f>  :new <n>  :seed <n>\n  \
                         :show  :q"
                    );
                },
                "show" => show_settings(&params, max_new, seed),
                "temp" =>
                {
                    if let Ok(v) = arg.parse()
                    {
                        params.temperature = v;
                    }
                },
                "topk" =>
                {
                    if let Ok(v) = arg.parse()
                    {
                        params.top_k = v;
                    }
                },
                "topp" =>
                {
                    if let Ok(v) = arg.parse()
                    {
                        params.top_p = v;
                    }
                },
                "rep" =>
                {
                    if let Ok(v) = arg.parse()
                    {
                        params.repetition_penalty = v;
                    }
                },
                "new" =>
                {
                    if let Ok(v) = arg.parse()
                    {
                        max_new = v;
                    }
                },
                "seed" =>
                {
                    if let Ok(v) = arg.parse()
                    {
                        seed = v;
                    }
                },
                other => println!("  unknown command `:{other}` (`:help`)"),
            }
            print!("> ");
            let _ = io::stdout().flush();
            continue;
        }

        let prompt = parse_prompt(trimmed, byte_level, config.vocab_size);
        if prompt.is_empty()
        {
            print!("> ");
            let _ = io::stdout().flush();
            continue;
        }
        // Keep prompt + new within the trained RoPE range.
        let budget = config.max_seq_len.saturating_sub(prompt.len());
        let n = max_new.min(budget);

        let t = Instant::now();
        let out = rm.generate_sampled(&prompt, n, &params, seed);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let generated = &out[prompt.len()..];

        if byte_level
        {
            let bytes: Vec<u8> = generated.iter().map(|&t| t as u8).collect();
            println!("{}", String::from_utf8_lossy(&bytes));
        }
        else
        {
            println!("{generated:?}");
        }
        let tps = if ms > 0.0
        {
            generated.len() as f64 / (ms / 1e3)
        }
        else
        {
            0.0
        };
        eprintln!(
            "[{} tokens · {:.0} ms · {:.1} tok/s]",
            generated.len(),
            ms,
            tps
        );

        print!("> ");
        let _ = io::stdout().flush();
    }
    println!("\nbye");
}
