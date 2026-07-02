use std::path::PathBuf;

use clap::{Parser, Subcommand};
use scirust_core::autodiff::reverse::Tape;
use scirust_sciagent::bpe::BpeTokenizer;
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::model::SciAgentModel;

#[derive(Parser)]
#[command(
    name = "sciagent",
    about = "SCIAGENT — determinist SLM for scirust ecosystem"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[arg(global = true, long, default_value = "350m")]
    model: String,

    #[arg(global = true, long, default_value_t = 42)]
    seed: u64,

    #[arg(global = true, long, default_value_t = 2048)]
    max_tokens: usize,

    #[arg(global = true, long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(global = true, long)]
    json: bool,

    #[arg(global = true, long)]
    checkpoint: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    Ask {
        prompt: String,
    },
    Chat,
    Explain {
        path: PathBuf,
        #[arg(long)]
        lines: Option<String>,
    },
    Generate {
        description: String,
    },
    Info,
}

fn build_model(cli: &Cli) -> SciAgentModel {
    if let Some(ref ckpt) = cli.checkpoint
    {
        println!("Loading checkpoint from {:?} ...", ckpt);
        let meta_path = ckpt.join("meta.json");
        let config = if let Ok(meta_str) = std::fs::read_to_string(&meta_path)
        {
            if let Ok(meta_json) = serde_json::from_str::<serde_json::Value>(&meta_str)
            {
                meta_json
                    .get("config")
                    .and_then(|cfg| {
                        Some(SciAgentConfig {
                            vocab_size: cfg["vocab_size"].as_u64()? as usize,
                            d_model: cfg["d_model"].as_u64()? as usize,
                            n_layers: cfg["n_layers"].as_u64()? as usize,
                            n_heads: cfg["n_heads"].as_u64()? as usize,
                            n_kv_heads: cfg["n_kv_heads"].as_u64()? as usize,
                            d_ff: cfg["d_ff"].as_u64()? as usize,
                            max_seq_len: cfg["max_seq_len"].as_u64()? as usize,
                            rope_theta: cfg["rope_theta"].as_f64()? as f32,
                            tie_embeddings: cfg["tie_embeddings"].as_bool()?,
                            use_bias: cfg["use_bias"].as_bool().unwrap_or(false),
                            eps: cfg["eps"].as_f64().unwrap_or(1e-5) as f32,
                        })
                    })
                    .unwrap_or_else(|| get_config(&cli.model))
            }
            else
            {
                eprintln!("Warning: cannot parse meta.json, using CLI config");
                get_config(&cli.model)
            }
        }
        else
        {
            eprintln!("Warning: no meta.json found, using CLI config");
            get_config(&cli.model)
        };
        let mut m = SciAgentModel::new(&config);
        let _ = scirust_sciagent::train::checkpoint::load_checkpoint(&mut m, ckpt);
        m
    }
    else
    {
        SciAgentModel::new(&get_config(&cli.model))
    }
}

fn main() {
    let cli = Cli::parse();
    let mut model = build_model(&cli);

    match &cli.command
    {
        Command::Ask { prompt } => cmd_ask(&mut model, prompt, &cli),
        Command::Chat => cmd_chat(&mut model, &cli),
        Command::Explain { path, lines } => cmd_explain(path, lines.as_deref(), &cli),
        Command::Generate { description } => cmd_generate(&mut model, description, &cli),
        Command::Info => cmd_info(&model.config, &cli),
    }
}

fn get_config(model_name: &str) -> SciAgentConfig {
    match model_name
    {
        "debug" => SciAgentConfig::debug(),
        "small" | "Small" => SciAgentConfig::small(),
        "350m" | "350M" => SciAgentConfig::sciagent_350m(),
        "7b" | "7B" => SciAgentConfig::sciagent_7b(),
        _ =>
        {
            eprintln!("Unknown model '{model_name}', using 350M");
            SciAgentConfig::sciagent_350m()
        },
    }
}

fn cmd_ask(model: &mut SciAgentModel, prompt: &str, cli: &Cli) {
    let vocab = model.config.vocab_size;
    let tokens = tokenize_with_vocab(prompt, vocab);
    let tape = Tape::new();
    let _ = model.forward(&tape, &tokens, tokens.len());

    let gen = scirust_sciagent::generate::Generator::new(&model.config);
    let result = gen.generate(model, &tokens, cli.max_tokens, cli.seed);
    let text = detokenize_with_vocab(&result, vocab);

    if cli.json
    {
        let output = serde_json::json!({
            "prompt": prompt,
            "response": text,
            "tokens": result.len(),
            "seed": cli.seed,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    }
    else
    {
        println!("{text}");
    }
}

fn cmd_chat(model: &mut SciAgentModel, cli: &Cli) {
    let vocab = model.config.vocab_size;
    let max_seq = model.config.max_seq_len;
    println!("SCIAGENT chat (Ctrl+D to exit)");
    let mut history: Vec<usize> = Vec::new();
    let gen = scirust_sciagent::generate::Generator::new(&model.config);

    loop
    {
        use std::io::{self, BufRead};
        let stdin = io::stdin();
        print!("> ");
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let mut line = String::new();
        match stdin.lock().read_line(&mut line)
        {
            Ok(0) | Err(_) => break,
            Ok(_) =>
            {},
        }
        let line = line.trim();
        if line.is_empty()
        {
            continue;
        }

        let tokens = tokenize_with_vocab(line, vocab);
        history.extend(&tokens);
        let ctx = if history.len() > max_seq
        {
            &history[history.len() - max_seq..]
        }
        else
        {
            &history
        };

        let result = gen.generate(model, ctx, cli.max_tokens.min(512), cli.seed);
        let text = detokenize_with_vocab(&result, vocab);
        println!("{text}");
        history.push(result.last().copied().unwrap_or(0));
    }
}

fn cmd_explain(path: &PathBuf, lines: Option<&str>, cli: &Cli) {
    let content = match std::fs::read_to_string(path)
    {
        Ok(c) => c,
        Err(e) =>
        {
            eprintln!("Cannot read {:?}: {e}", path);
            return;
        },
    };

    let excerpt = match lines
    {
        Some(range) =>
        {
            let parts: Vec<&str> = range.splitn(2, '-').collect();
            let start: usize = parts[0].parse().unwrap_or(1);
            let end: usize = parts
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(start + 30);
            content
                .lines()
                .skip(start.saturating_sub(1))
                .take(end - start + 1)
                .collect::<Vec<_>>()
                .join("\n")
        },
        None => content.chars().take(2000).collect::<String>(),
    };

    let vocab = scirust_sciagent::config::SciAgentConfig::debug().vocab_size;
    let prompt = format!("Explain this code:\n```rust\n{excerpt}\n```");
    cmd_ask(&mut model_placeholder(vocab), &prompt, cli);
}

fn model_placeholder(vocab_size: usize) -> SciAgentModel {
    let mut cfg = SciAgentConfig::debug();
    cfg.vocab_size = vocab_size;
    SciAgentModel::new(&cfg)
}

fn cmd_generate(model: &mut SciAgentModel, description: &str, cli: &Cli) {
    let prompt = format!("Write Rust code for: {description}");
    cmd_ask(model, &prompt, cli);
}

fn cmd_info(config: &SciAgentConfig, _cli: &Cli) {
    println!("=== SCIAGENT Model Info ===");
    println!("Name: scirust-sciagent");
    println!("Architecture: GQA + SwiGLU + RoPE + RMSNorm");
    println!("Vocab size: {}", config.vocab_size);
    println!("d_model: {}", config.d_model);
    println!("n_layers: {}", config.n_layers);
    println!(
        "n_heads: {} ({} KV heads)",
        config.n_heads, config.n_kv_heads
    );
    println!("d_ff: {}", config.d_ff);
    println!("max_seq_len: {}", config.max_seq_len);
    println!(
        "Total parameters: {}",
        fmt_params(config.total_parameters())
    );
    println!("Tie embeddings: {}", config.tie_embeddings);
}

fn tokenize_with_vocab(text: &str, vocab_size: usize) -> Vec<usize> {
    if let Ok(tok) = BpeTokenizer::from_embedded()
    {
        if tok.vocab_size() <= vocab_size
        {
            tok.encode_with_special(text, true, false)
        }
        else
        {
            text.bytes().map(|b| b as usize).collect()
        }
    }
    else
    {
        text.bytes().map(|b| b as usize).collect()
    }
}

fn detokenize_with_vocab(ids: &[usize], vocab_size: usize) -> String {
    if let Ok(tok) = BpeTokenizer::from_embedded()
    {
        if tok.vocab_size() <= vocab_size
        {
            tok.decode(ids)
        }
        else
        {
            ids.iter()
                .filter_map(|&id| char::from_u32(id as u32))
                .collect()
        }
    }
    else
    {
        ids.iter()
            .filter_map(|&id| char::from_u32(id as u32))
            .collect()
    }
}

fn fmt_params(n: usize) -> String {
    if n >= 1_000_000_000
    {
        format!("{:.1}B", n as f64 / 1e9)
    }
    else if n >= 1_000_000
    {
        format!("{:.1}M", n as f64 / 1e6)
    }
    else
    {
        format!("{n}")
    }
}
