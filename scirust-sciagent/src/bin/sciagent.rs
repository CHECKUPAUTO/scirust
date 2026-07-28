use std::path::PathBuf;

use clap::{Parser, Subcommand};
use scirust_core::autodiff::reverse::Tape;
use scirust_sciagent::bpe::BpeTokenizer;
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{load_checkpoint, read_meta};

type CliResult<T> = Result<T, (i32, String)>;

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

    #[arg(global = true, long, default_value_t = 0)]
    top_k: usize,

    #[arg(global = true, long, default_value_t = 1.0)]
    top_p: f32,

    #[arg(global = true, long, default_value_t = 1.0)]
    repetition_penalty: f32,

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

fn report_cli_error((code, message): (i32, String)) -> ! {
    eprintln!("error: {message}");
    std::process::exit(code);
}

/// Build a model only from a valid, explicitly supplied checkpoint.
///
/// A freshly initialized model contains random weights. All inference paths
/// must fail closed when the checkpoint is missing or invalid.
fn build_model(cli: &Cli) -> CliResult<SciAgentModel> {
    let checkpoint = cli.checkpoint.as_ref().ok_or_else(|| {
        (
            2,
            String::from(
                "missing required `--checkpoint PATH`; refusing to run with random weights",
            ),
        )
    })?;

    eprintln!("Loading checkpoint from {:?} ...", checkpoint);

    let meta = read_meta(checkpoint).map_err(|error| {
        (
            1,
            format!(
                "cannot read checkpoint metadata from `{}`: {error}",
                checkpoint.display()
            ),
        )
    })?;

    let mut model = SciAgentModel::new(&meta.config);
    load_checkpoint(&mut model, checkpoint).map_err(|error| {
        (
            1,
            format!(
                "cannot load checkpoint from `{}`: {error}",
                checkpoint.display()
            ),
        )
    })?;

    Ok(model)
}

/// Resolve `info` configuration without allocating random model weights.
fn info_config(cli: &Cli) -> CliResult<SciAgentConfig> {
    if let Some(checkpoint) = cli.checkpoint.as_ref()
    {
        read_meta(checkpoint)
            .map(|meta| meta.config)
            .map_err(|error| {
                (
                    1,
                    format!(
                        "cannot read checkpoint metadata from `{}`: {error}",
                        checkpoint.display()
                    ),
                )
            })
    }
    else
    {
        Ok(get_config(&cli.model))
    }
}

fn main() {
    let cli = Cli::parse();

    if matches!(&cli.command, Command::Info)
    {
        let config = info_config(&cli).unwrap_or_else(|error| report_cli_error(error));
        cmd_info(&config, &cli);
        return;
    }

    let mut model = build_model(&cli).unwrap_or_else(|error| report_cli_error(error));

    match &cli.command
    {
        Command::Ask { prompt } => cmd_ask(&mut model, prompt, &cli),
        Command::Chat => cmd_chat(&mut model, &cli),
        Command::Explain { path, lines } => cmd_explain(&mut model, path, lines.as_deref(), &cli),
        Command::Generate { description } => cmd_generate(&mut model, description, &cli),
        Command::Info => unreachable!("info returns before model construction"),
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

    let gen = scirust_sciagent::generate::Generator::new(&model.config)
        .with_temperature(cli.temperature)
        .with_top_k(cli.top_k)
        .with_top_p(cli.top_p)
        .with_repetition_penalty(cli.repetition_penalty);
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
    let gen = scirust_sciagent::generate::Generator::new(&model.config)
        .with_temperature(cli.temperature)
        .with_top_k(cli.top_k)
        .with_top_p(cli.top_p)
        .with_repetition_penalty(cli.repetition_penalty);

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

fn cmd_explain(model: &mut SciAgentModel, path: &PathBuf, lines: Option<&str>, cli: &Cli) {
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
            let (start, end) = match parse_line_range(range)
            {
                Ok(bounds) => bounds,
                Err(error) =>
                {
                    eprintln!("Invalid --lines value `{range}`: {error}");
                    return;
                },
            };
            content
                .lines()
                .skip(start.saturating_sub(1))
                .take(end - start + 1)
                .collect::<Vec<_>>()
                .join("\n")
        },
        None => content.chars().take(2000).collect::<String>(),
    };

    let prompt = format!("Explain this code:\n```rust\n{excerpt}\n```");
    cmd_ask(model, &prompt, cli);
}

fn parse_line_range(range: &str) -> Result<(usize, usize), &'static str> {
    let (start, end) = match range.split_once('-')
    {
        Some((start, end)) =>
        {
            if start.is_empty() || end.is_empty() || end.contains('-')
            {
                return Err("expected START-END with two positive line numbers");
            }
            let start = start
                .parse::<usize>()
                .map_err(|_| "START is not a positive line number")?;
            let end = end
                .parse::<usize>()
                .map_err(|_| "END is not a positive line number")?;
            (start, end)
        },
        None =>
        {
            let start = range
                .parse::<usize>()
                .map_err(|_| "expected a positive line number or START-END")?;
            let end = start
                .checked_add(30)
                .ok_or("line range overflows this platform")?;
            (start, end)
        },
    };

    if start == 0 || end == 0
    {
        return Err("line numbers are one-based and must be greater than zero");
    }
    if end < start
    {
        return Err("END must be greater than or equal to START");
    }
    Ok((start, end))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cli(checkpoint: Option<PathBuf>) -> Cli {
        Cli {
            command: Command::Ask {
                prompt: String::from("fn main() {}"),
            },
            model: String::from("debug"),
            seed: 42,
            max_tokens: 16,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            json: false,
            checkpoint,
        }
    }

    #[test]
    fn line_ranges_are_validated() {
        assert_eq!(parse_line_range("4-9"), Ok((4, 9)));
        assert_eq!(parse_line_range("4"), Ok((4, 34)));
        assert!(parse_line_range("0-2").is_err());
        assert!(parse_line_range("9-4").is_err());
        assert!(parse_line_range("bad").is_err());
        assert!(parse_line_range("1-2-3").is_err());
    }

    #[test]
    fn inference_requires_an_explicit_checkpoint() {
        let error = match build_model(&test_cli(None))
        {
            Ok(_) => panic!("randomly initialized model must not be accepted"),
            Err(error) => error,
        };
        assert_eq!(error.0, 2);
        assert!(error.1.contains("--checkpoint PATH"));
        assert!(error.1.contains("random weights"));
    }

    #[test]
    fn invalid_checkpoint_fails_closed() {
        let cli = test_cli(Some(PathBuf::from("/path/that/does/not/exist")));
        let error = match build_model(&cli)
        {
            Ok(_) => panic!("invalid checkpoint must not produce a model"),
            Err(error) => error,
        };
        assert_eq!(error.0, 1);
        assert!(error.1.contains("cannot read checkpoint metadata"));
    }

    #[test]
    fn info_config_does_not_allocate_model_weights() {
        let config = info_config(&test_cli(None)).expect("info config should resolve");
        assert_eq!(config.d_model, SciAgentConfig::debug().d_model);
    }
}
