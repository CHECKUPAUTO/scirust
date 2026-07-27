//! SCIAGENT SLM: deterministic small language model specialised for Rust + agentic.
//! Subcommands: `ask`, `chat`, `explain`, `generate`, `info`, `attest`, `quantize`.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use scirust_sciagent::bpe::BpeTokenizer;
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::quantize::QuantizedSciAgent;
use scirust_sciagent::train::checkpoint::{load_checkpoint, read_meta};

type CliResult<T> = Result<T, (u8, String)>;

fn flag_str(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn flag_u64(args: &[String], name: &str, default: u64) -> u64 {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn model_name(args: &[String]) -> String {
    flag_str(args, "--model").unwrap_or_else(|| String::from("debug"))
}

fn report_cli_error((code, message): (u8, String)) -> u8 {
    eprintln!("error: {message}");
    code
}

/// Load configuration and weights from an explicit checkpoint.
///
/// A newly constructed model contains random weights. Inference must therefore
/// fail closed when the checkpoint is missing, unreadable, or invalid.
fn load_required_model(args: &[String]) -> CliResult<SciAgentModel> {
    let path = flag_str(args, "--checkpoint")
        .map(PathBuf::from)
        .ok_or_else(|| {
            (
                2,
                String::from(
                    "missing required `--checkpoint PATH`; refusing to run with random weights",
                ),
            )
        })?;

    let meta = read_meta(&path).map_err(|error| {
        (
            1,
            format!(
                "cannot read checkpoint metadata from `{}`: {error}",
                path.display()
            ),
        )
    })?;

    let mut model = SciAgentModel::new(&meta.config);
    load_checkpoint(&mut model, &path).map_err(|error| {
        (
            1,
            format!("cannot load checkpoint from `{}`: {error}", path.display()),
        )
    })?;

    Ok(model)
}

fn get_config(name: &str) -> SciAgentConfig {
    match name
    {
        "debug" => SciAgentConfig::debug(),
        "350m" | "350M" => SciAgentConfig::sciagent_350m(),
        "7b" | "7B" => SciAgentConfig::sciagent_7b(),
        _ =>
        {
            eprintln!("Unknown model '{name}', using debug");
            SciAgentConfig::debug()
        },
    }
}

fn tokenize(text: &str) -> Vec<usize> {
    if let Ok(tok) = BpeTokenizer::from_embedded()
    {
        tok.encode_with_special(text, true, false)
    }
    else
    {
        text.chars().map(|c| (c as usize) % 32768).collect()
    }
}

fn detokenize(ids: &[usize]) -> String {
    if let Ok(tok) = BpeTokenizer::from_embedded()
    {
        tok.decode(ids)
    }
    else
    {
        ids.iter()
            .map(|&id| char::from_u32((id % 32768) as u32).unwrap_or('?'))
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

fn cmd_ask(model: &mut SciAgentModel, prompt: &str, max_tokens: usize) {
    let tokens = tokenize(prompt);
    let gen = scirust_sciagent::generate::Generator::new(&model.config);
    let result = gen.generate(model, &tokens, max_tokens, 42);
    let text = detokenize(&result);
    println!("{text}");
}

fn run_ask(args: &[String]) -> u8 {
    let prompt = flag_str(args, "--prompt")
        .or_else(|| args.first().cloned())
        .unwrap_or_default();
    if prompt.is_empty()
    {
        eprintln!("usage: scirust sciagent ask <prompt> --checkpoint PATH");
        return 2;
    }
    let mut model = match load_required_model(args)
    {
        Ok(model) => model,
        Err(error) => return report_cli_error(error),
    };
    cmd_ask(&mut model, &prompt, 512);
    0
}

fn run_chat(args: &[String]) -> u8 {
    let mut model = match load_required_model(args)
    {
        Ok(model) => model,
        Err(error) => return report_cli_error(error),
    };
    let gen = scirust_sciagent::generate::Generator::new(&model.config);

    println!("SCIAGENT chat (Ctrl+D to exit)");
    let mut history: Vec<usize> = Vec::new();
    let stdin = io::stdin();
    loop
    {
        print!("> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line)
        {
            Ok(0) | Err(_) => break,
            Ok(_) =>
            {},
        }
        let line = line.trim().to_string();
        if line.is_empty()
        {
            continue;
        }
        let tokens = tokenize(&line);
        history.extend(&tokens);
        let max_seq = model.config.max_seq_len;
        let ctx = if history.len() > max_seq
        {
            &history[history.len() - max_seq..]
        }
        else
        {
            &history
        };
        let result = gen.generate(&mut model, ctx, 256, 42);
        let text = detokenize(&result);
        println!("{text}");
        history.push(result.last().copied().unwrap_or(0));
    }
    0
}

fn run_explain(args: &[String]) -> u8 {
    let path = flag_str(args, "--path")
        .or_else(|| args.first().cloned())
        .map(PathBuf::from);
    let path = match path
    {
        Some(p) => p,
        None =>
        {
            eprintln!("usage: scirust sciagent explain <path> [--lines N-M]");
            return 2;
        },
    };
    let content = match std::fs::read_to_string(&path)
    {
        Ok(c) => c,
        Err(e) =>
        {
            eprintln!("Cannot read {:?}: {e}", path);
            return 1;
        },
    };
    let excerpt = match flag_str(args, "--lines")
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
    let prompt = format!("Explain this code:\n```rust\n{excerpt}\n```");
    let mut model = match load_required_model(args)
    {
        Ok(model) => model,
        Err(error) => return report_cli_error(error),
    };
    cmd_ask(&mut model, &prompt, 512);
    0
}

fn run_generate(args: &[String]) -> u8 {
    let desc = flag_str(args, "--desc")
        .or_else(|| args.first().cloned())
        .unwrap_or_default();
    if desc.is_empty()
    {
        eprintln!("usage: scirust sciagent generate <description>");
        return 2;
    }
    let mut model = match load_required_model(args)
    {
        Ok(model) => model,
        Err(error) => return report_cli_error(error),
    };
    cmd_ask(&mut model, &format!("Write Rust code for: {desc}"), 512);
    0
}

fn run_info(args: &[String]) -> u8 {
    let cfg = match flag_str(args, "--checkpoint").map(PathBuf::from)
    {
        Some(path) => match read_meta(&path)
        {
            Ok(meta) => meta.config,
            Err(error) =>
            {
                return report_cli_error((
                    1,
                    format!(
                        "cannot read checkpoint metadata from `{}`: {error}",
                        path.display()
                    ),
                ));
            },
        },
        None => get_config(&model_name(args)),
    };

    println!("=== SCIAGENT Model Info ===");
    println!("Name: scirust-sciagent");
    println!("Architecture: GQA + SwiGLU + RoPE + RMSNorm");
    println!("Vocab size: {}", cfg.vocab_size);
    println!("d_model: {}", cfg.d_model);
    println!("n_layers: {}", cfg.n_layers);
    println!("n_heads: {} ({} KV heads)", cfg.n_heads, cfg.n_kv_heads);
    println!("d_ff: {}", cfg.d_ff);
    println!("max_seq_len: {}", cfg.max_seq_len);
    println!("Total parameters: {}", fmt_params(cfg.total_parameters()));
    println!("Tie embeddings: {}", cfg.tie_embeddings);
    0
}

fn run_attest(args: &[String]) -> u8 {
    let model = match load_required_model(args)
    {
        Ok(model) => model,
        Err(error) => return report_cli_error(error),
    };
    let cfg = model.config.clone();
    let mut inf = scirust_sciagent::inference::SciAgentInference::new(model, &cfg);
    let prompt = vec![4usize, 5, 6];
    let _out = inf.generate(&prompt, 10);
    let chain = inf.attestation.current_chain_hash();
    let ok = inf.attestation.verify();
    println!("CCOS Attestation Log");
    println!("  entries: {}", inf.attestation.len());
    println!("  chain head: {chain}");
    println!("  chain valid: {ok}");
    println!("  jsonl:");
    println!("{}", inf.attestation.to_json_lines());
    0
}

fn run_quantize(args: &[String]) -> u8 {
    let model = match load_required_model(args)
    {
        Ok(model) => model,
        Err(error) => return report_cli_error(error),
    };
    let cfg = model.config.clone();
    let group_size = flag_u64(args, "--group", 32) as usize;
    let quantized = QuantizedSciAgent::from_model(&model, group_size);

    println!("INT4 Quantization — SCIAGENT");
    println!(
        "  model: {:?} (params: {})",
        (cfg.d_model, cfg.n_layers),
        fmt_params(cfg.total_parameters())
    );
    println!("  group size: {group_size}");
    println!(
        "  original: {} MB",
        quantized.estimate_original_bytes() as f64 / 1_048_576.0
    );
    println!(
        "  compressed: {} MB",
        quantized.total_compressed_bytes() as f64 / 1_048_576.0
    );
    println!("  ratio: {:.2}×", quantized.compression_ratio());

    if let Some(out) = flag_str(args, "--output")
    {
        let path = PathBuf::from(&out);
        match quantized.save_bin(&path)
        {
            Ok(_) => println!("  saved to: {out}"),
            Err(e) => eprintln!("  error: {e}"),
        }
    }
    0
}

/// Dispatch `sciagent` subcommands.
pub fn run(args: &[String]) -> u8 {
    let sub = args.first().map(String::as_str).unwrap_or("info");
    let rest = if args.len() > 1 { &args[1..] } else { &[] };
    match sub
    {
        "ask" => run_ask(rest),
        "chat" => run_chat(rest),
        "explain" => run_explain(rest),
        "generate" => run_generate(rest),
        "info" => run_info(rest),
        "attest" => run_attest(rest),
        "quantize" => run_quantize(rest),
        _ =>
        {
            eprintln!("unknown sciagent subcommand: `{sub}`");
            eprintln!(
                "usage: scirust sciagent <ask|chat|explain|generate|info|attest|quantize> [args]"
            );
            2
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_requires_an_explicit_checkpoint() {
        let error = match load_required_model(&[])
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
        let args = vec![
            String::from("--checkpoint"),
            String::from("/path/that/does/not/exist"),
        ];
        let error = match load_required_model(&args)
        {
            Ok(_) => panic!("invalid checkpoint must not produce a model"),
            Err(error) => error,
        };
        assert_eq!(error.0, 1);
        assert!(error.1.contains("cannot read checkpoint metadata"));
    }

    #[test]
    fn info_does_not_require_model_weights() {
        assert_eq!(run_info(&[]), 0);
    }
}
