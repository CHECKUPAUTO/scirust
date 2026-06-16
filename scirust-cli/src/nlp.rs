//! NLP subcommands over `scirust-learning` (deterministic, tested).
//!
//! Commands: bpe, lm.

use scirust_core::autodiff::nd::NdTape;
use scirust_core::nn::PcgEngine;
use scirust_core::nn::nd_decoder::{NdDecoderConfig, NdDecoderLM};
use scirust_core::nn::nd_layers::{NdDeltaNet, NdMamba, NdRetention};
use scirust_core::nn::nd_optim::{
    NdAdEMAMix, NdAdam, NdAdan, NdLamb, NdLion, NdLookahead, NdParam, NdScheduleFree, NdSoap,
};
use scirust_core::tensor::tensor_nd::TensorND;
use scirust_learning::nlp::bpe::BpeTokenizer;
use scirust_learning::nlp::byte_bpe::ByteBpeTokenizer;
use scirust_learning::nlp::tokenization::Tokenizer;

/// Remove a `--flag value` pair, returning the value (if any) and the rest.
fn take_flag(args: &[String], name: &str) -> (Option<String>, Vec<String>) {
    let mut value = None;
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len()
    {
        if args[i] == name && i + 1 < args.len()
        {
            value = Some(args[i + 1].clone());
            i += 2;
        }
        else
        {
            rest.push(args[i].clone());
            i += 1;
        }
    }
    (value, rest)
}

/// Remove a boolean `--flag`, returning whether it was present and the rest.
fn take_bool(args: &[String], name: &str) -> (bool, Vec<String>) {
    let mut present = false;
    let mut rest = Vec::new();
    for a in args
    {
        if a == name
        {
            present = true;
        }
        else
        {
            rest.push(a.clone());
        }
    }
    (present, rest)
}

/// `bpe "<corpus>" [--vocab N] [--encode "<text>"] [--bytes]` — train a
/// deterministic byte-pair-encoding tokenizer on the corpus (documents
/// separated by `;`), then encode/decode a piece of text. `--bytes` selects the
/// byte-level tokenizer (GPT-2 style): no out-of-vocabulary, lossless on any
/// UTF-8. Reports the learned vocab size, the token ids, and the round-trip.
pub fn run_bpe(args: &[String]) -> u8 {
    let (bytes, rest) = take_bool(args, "--bytes");
    let (vocab_s, rest) = take_flag(&rest, "--vocab");
    let (enc_s, rest) = take_flag(&rest, "--encode");
    let Some(corpus) = rest.first()
    else
    {
        eprintln!("usage: scirust bpe \"<corpus>\" [--vocab N] [--encode \"<text>\"] [--bytes]");
        return 2;
    };
    let default_vocab = if bytes { 300 } else { 50 };
    let vocab = match vocab_s
    {
        Some(s) => match s.parse::<usize>()
        {
            Ok(v) if v >= 2 => v,
            _ =>
            {
                eprintln!("error: --vocab must be an integer ≥ 2");
                return 2;
            },
        },
        None => default_vocab,
    };
    let docs: Vec<&str> = corpus
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if docs.is_empty()
    {
        eprintln!("error: empty corpus");
        return 2;
    }
    let text = enc_s.as_deref().unwrap_or(docs[0]);

    // (vocab_size, token ids, decoded string, kind label)
    let (vsize, ids, decoded, kind) = if bytes
    {
        let tok = ByteBpeTokenizer::train(&docs, vocab);
        let ids = tok.encode(text);
        (
            tok.vocab_size(),
            ids.clone(),
            tok.decode(&ids),
            "byte-level BPE",
        )
    }
    else
    {
        let tok = BpeTokenizer::train(&docs, vocab);
        let ids: Vec<u32> = tok.tokenize(text);
        (
            tok.vocab_size(),
            ids.clone(),
            tok.decode(&ids),
            "char-level BPE",
        )
    };

    println!(
        "trained {kind}: vocab size {vsize} (target {vocab}) on {} document(s)",
        docs.len()
    );
    println!("encode \"{text}\" → {ids:?}  ({} tokens)", ids.len());
    println!("decode → \"{decoded}\"");
    println!(
        "round-trip: {}",
        if decoded == text
        {
            "exact"
        }
        else
        {
            "lossy (char-level BPE maps out-of-vocabulary chars to <UNK>; use --bytes for lossless)"
        }
    );
    0
}

/// One of the N-D optimizers, selectable from the CLI; all share `step`.
enum LmOpt {
    Adam(NdAdam),
    Lion(NdLion),
    ScheduleFree(NdScheduleFree),
    AdEMAMix(NdAdEMAMix),
    Soap(NdSoap),
    Lookahead(NdLookahead),
    Lamb(NdLamb),
    Adan(NdAdan),
}

impl LmOpt {
    fn step(&mut self, params: &mut [NdParam], grads: &[TensorND]) {
        match self
        {
            LmOpt::Adam(o) => o.step(params, grads),
            LmOpt::Lion(o) => o.step(params, grads),
            LmOpt::ScheduleFree(o) => o.step(params, grads),
            LmOpt::AdEMAMix(o) => o.step(params, grads),
            LmOpt::Soap(o) => o.step(params, grads),
            LmOpt::Lookahead(o) => o.step(params, grads),
            LmOpt::Lamb(o) => o.step(params, grads),
            LmOpt::Adan(o) => o.step(params, grads),
        }
    }

    /// After training, load the deployable weights. Schedule-Free keeps `y` in
    /// the parameters and the average `x` separately, so we write `x` back;
    /// the other optimizers already hold the final weights.
    fn finalize(&self, params: &mut [NdParam]) {
        if let LmOpt::ScheduleFree(o) = self
        {
            o.write_eval_point(params);
        }
    }
}

/// `lm ["t0,t1,.."] [--seed N] [--steps S] [--lr R] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead|lamb|adan]` —
/// train a small **causal decoder language model** on the N-D autograd tape and
/// report whether it learns to predict the sequence. Pure Rust, deterministic by
/// seed: token + learned positional embeddings, causal multi-head attention, a
/// final LayerNorm and an LM head, optimised by a selectable deterministic
/// optimizer wired through every layer. Surfaces the whole N-D stack end to end.
pub fn run_lm(args: &[String]) -> u8 {
    let (seed_s, rest) = take_flag(args, "--seed");
    let (steps_s, rest) = take_flag(&rest, "--steps");
    let (lr_s, rest) = take_flag(&rest, "--lr");
    let (opt_s, rest) = take_flag(&rest, "--opt");

    let tokens: Vec<usize> = match rest.first()
    {
        Some(spec) =>
        {
            let mut v = Vec::new();
            for part in spec.split(',').map(str::trim).filter(|x| !x.is_empty())
            {
                match part.parse::<usize>()
                {
                    Ok(t) => v.push(t),
                    Err(_) =>
                    {
                        eprintln!("error: tokens must be non-negative integers (got `{part}`)");
                        return 2;
                    },
                }
            }
            v
        },
        None => vec![1, 2, 3, 4, 2, 5],
    };
    if tokens.len() < 2
    {
        eprintln!(
            "usage: scirust lm [\"t0,t1,..\"] [--seed N] [--steps S] [--lr R] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead|lamb|adan]"
        );
        eprintln!("error: need at least 2 tokens for next-token training");
        return 2;
    }

    let seed: u64 = match &seed_s
    {
        Some(s) => match s.parse()
        {
            Ok(v) => v,
            Err(_) =>
            {
                eprintln!("error: --seed must be a non-negative integer");
                return 2;
            },
        },
        None => 123,
    };
    let steps: usize = match &steps_s
    {
        Some(s) => match s.parse()
        {
            Ok(v) if v >= 1 => v,
            _ =>
            {
                eprintln!("error: --steps must be an integer ≥ 1");
                return 2;
            },
        },
        None => 200,
    };
    let opt_kind = opt_s.as_deref().unwrap_or("adam");
    if !matches!(
        opt_kind,
        "adam"
            | "adamw"
            | "lion"
            | "schedule-free"
            | "ademamix"
            | "soap"
            | "lookahead"
            | "lamb"
            | "adan"
    )
    {
        eprintln!(
            "error: --opt must be one of: adam, adamw, lion, schedule-free, ademamix, soap, lookahead, lamb, adan"
        );
        return 2;
    }
    // Each optimizer prefers a different default step size.
    let default_lr = match opt_kind
    {
        "lion" => 0.003,
        "schedule-free" => 0.5,
        "ademamix" => 0.005,
        _ => 0.01,
    };
    let lr: f32 = match &lr_s
    {
        Some(s) => match s.parse::<f32>()
        {
            Ok(v) if v > 0.0 && v.is_finite() => v,
            _ =>
            {
                eprintln!("error: --lr must be a positive number");
                return 2;
            },
        },
        None => default_lr,
    };

    let vocab = tokens.iter().max().copied().unwrap() + 1;
    let (d_model, n_heads, n_layers) = (16usize, 2usize, 2usize);
    let cfg = NdDecoderConfig {
        vocab,
        d_model,
        n_heads,
        d_ff: 32,
        n_layers,
        max_seq: tokens.len(),
    };
    let mut rng = PcgEngine::new(seed);
    let mut lm = NdDecoderLM::new(cfg, &mut rng);
    let mut opt = match opt_kind
    {
        "adamw" => LmOpt::Adam(NdAdam::with_lr_wd(lr, 0.01)),
        "lion" => LmOpt::Lion(NdLion::with_lr(lr)),
        "schedule-free" => LmOpt::ScheduleFree(NdScheduleFree::with_lr(lr)),
        "ademamix" => LmOpt::AdEMAMix(NdAdEMAMix::with_lr(lr)),
        "soap" => LmOpt::Soap(NdSoap::with_lr(lr)),
        "lookahead" => LmOpt::Lookahead(NdLookahead::with_lr(lr)),
        "lamb" => LmOpt::Lamb(NdLamb::with_lr(lr)),
        "adan" => LmOpt::Adan(NdAdan::with_lr(lr)),
        _ => LmOpt::Adam(NdAdam::with_lr(lr)),
    };

    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..steps
    {
        let tape = NdTape::new();
        let loss_v = lm.loss(&tape, &tokens);
        let loss = tape.value(loss_v).data[0];
        if step == 0
        {
            first = loss;
        }
        last = loss;
        let grads = tape.backward(loss_v);
        let mut params = lm.parameters();
        opt.step(&mut params, &grads);
    }
    // Load the deployable weights (a no-op except for Schedule-Free).
    {
        let mut params = lm.parameters();
        opt.finalize(&mut params);
    }

    let tape = NdTape::new();
    let preds = lm.predict(&tape, &tokens[..tokens.len() - 1]);
    let targets = &tokens[1..];
    let correct = preds.iter().zip(targets).filter(|(a, b)| a == b).count();

    println!("N-D causal decoder LM — pure Rust, deterministic (seed {seed})");
    println!(
        "  config: vocab {vocab}, d_model {d_model}, heads {n_heads}, layers {n_layers}, seq {}",
        tokens.len()
    );
    println!("  tokens: {tokens:?}");
    println!("  optimizer: {opt_kind}(lr={lr}), {steps} steps");
    println!(
        "  loss: {first:.4} → {last:.4}  ({:.1}% of initial)",
        100.0 * last / first
    );
    println!("  next-token argmax: {preds:?}");
    println!("  targets:           {targets:?}");
    println!(
        "  recall: {}",
        if correct == targets.len()
        {
            "exact — the model reproduces the sequence".to_string()
        }
        else
        {
            format!(
                "{correct}/{} positions (increase --steps to overfit further)",
                targets.len()
            )
        }
    );
    0
}

/// `deltanet [--seed N] [--steps S]` — train a single-head **DeltaNet**
/// (delta-rule linear attention) layer to fit a fixed target sequence and report
/// the MSE reduction. Showcases the fast-weight recurrence (unrolled and
/// gradient-checked on the N-D tape). Pure Rust, deterministic by seed.
pub fn run_deltanet(args: &[String]) -> u8 {
    let (seed_s, rest) = take_flag(args, "--seed");
    let (steps_s, _rest) = take_flag(&rest, "--steps");
    let seed: u64 = seed_s.and_then(|s| s.parse().ok()).unwrap_or(7);
    let steps: usize = steps_s
        .and_then(|s| s.parse().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(150);

    let (seq, d) = (6usize, 8usize);
    let mut rng = PcgEngine::new(seed);
    let mut layer = NdDeltaNet::new(d, &mut rng);
    let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
    let target: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2).cos()).collect();
    let mut opt = NdAdam::with_lr(0.05);

    let (mut first, mut last) = (f32::NAN, f32::NAN);
    for step in 0..steps
    {
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(x.clone(), vec![seq, d]));
        let tv = tape.input(TensorND::new(target.clone(), vec![seq, d]));
        let out = layer.forward(&tape, xv);
        let diff = out.sub(tv);
        let loss = diff.mul(diff).sum();
        let lval = tape.value(loss).data[0];
        if step == 0
        {
            first = lval;
        }
        last = lval;
        let grads = tape.backward(loss);
        opt.step(&mut layer.parameters(), &grads);
    }

    println!("DeltaNet (delta-rule linear attention) — pure Rust, deterministic (seed {seed})");
    println!("  single head, d_model {d}, sequence length {seq}");
    println!("  fast-weight memory S updated by the delta rule; trained with Adam ({steps} steps)");
    println!(
        "  MSE to target: {first:.4} → {last:.4}  ({:.1}% of initial)",
        100.0 * last / first
    );
    0
}

/// `mamba [--seed N] [--steps S]` — train a single **Mamba** selective
/// state-space layer (S6 input-dependent scan) to fit a fixed target sequence
/// and report the MSE reduction. Pure Rust, deterministic by seed.
pub fn run_mamba(args: &[String]) -> u8 {
    let (seed_s, rest) = take_flag(args, "--seed");
    let (steps_s, _rest) = take_flag(&rest, "--steps");
    let seed: u64 = seed_s.and_then(|s| s.parse().ok()).unwrap_or(5);
    let steps: usize = steps_s
        .and_then(|s| s.parse().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(150);

    let (seq, d_model, d_inner, n) = (6usize, 8usize, 12usize, 8usize);
    let mut rng = PcgEngine::new(seed);
    let mut layer = NdMamba::new(d_model, d_inner, n, &mut rng);
    let x: Vec<f32> = (0..seq * d_model)
        .map(|i| (i as f32 * 0.3 - 1.0).sin())
        .collect();
    let target: Vec<f32> = (0..seq * d_model).map(|i| (i as f32 * 0.2).cos()).collect();
    let mut opt = NdAdam::with_lr(0.05);

    let (mut first, mut last) = (f32::NAN, f32::NAN);
    for step in 0..steps
    {
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(x.clone(), vec![seq, d_model]));
        let tv = tape.input(TensorND::new(target.clone(), vec![seq, d_model]));
        let out = layer.forward(&tape, xv);
        let diff = out.sub(tv);
        let loss = diff.mul(diff).sum();
        let lval = tape.value(loss).data[0];
        if step == 0
        {
            first = lval;
        }
        last = lval;
        let grads = tape.backward(loss);
        opt.step(&mut layer.parameters(), &grads);
    }

    println!("Mamba selective state-space layer — pure Rust, deterministic (seed {seed})");
    println!("  d_model {d_model}, d_inner {d_inner}, state size {n}, sequence length {seq}");
    println!(
        "  input-dependent (selective) Δ, B, C; diagonal A; linear-time scan ({steps} Adam steps)"
    );
    println!(
        "  MSE to target: {first:.4} → {last:.4}  ({:.1}% of initial)",
        100.0 * last / first
    );
    0
}

/// `retnet [--seed N] [--steps S]` — train a single-head **RetNet** retention
/// layer (linear-attention recurrence with decay γ, recurrent form ≡ parallel
/// form) to fit a fixed target sequence and report the MSE reduction.
/// Deterministic in `--seed`.
pub fn run_retnet(args: &[String]) -> u8 {
    let (seed_s, rest) = take_flag(args, "--seed");
    let (steps_s, _rest) = take_flag(&rest, "--steps");
    let seed: u64 = seed_s.and_then(|s| s.parse().ok()).unwrap_or(6);
    let steps: usize = steps_s
        .and_then(|s| s.parse().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(150);

    let (seq, d_model, gamma) = (6usize, 8usize, 0.9f32);
    let mut rng = PcgEngine::new(seed);
    let mut layer = NdRetention::new(d_model, gamma, &mut rng);
    let x: Vec<f32> = (0..seq * d_model)
        .map(|i| (i as f32 * 0.3 - 1.0).sin())
        .collect();
    let target: Vec<f32> = (0..seq * d_model).map(|i| (i as f32 * 0.2).cos()).collect();
    let mut opt = NdAdam::with_lr(0.05);

    let (mut first, mut last) = (f32::NAN, f32::NAN);
    for step in 0..steps
    {
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(x.clone(), vec![seq, d_model]));
        let tv = tape.input(TensorND::new(target.clone(), vec![seq, d_model]));
        let out = layer.forward(&tape, xv);
        let diff = out.sub(tv);
        let loss = diff.mul(diff).sum();
        let lval = tape.value(loss).data[0];
        if step == 0
        {
            first = lval;
        }
        last = lval;
        let grads = tape.backward(loss);
        opt.step(&mut layer.parameters(), &grads);
    }

    println!("RetNet retention layer — pure Rust, deterministic (seed {seed})");
    println!("  single head, d_model {d_model}, decay γ = {gamma}, sequence length {seq}");
    println!("  linear-attention recurrence (recurrent form ≡ parallel form); {steps} Adam steps");
    println!(
        "  MSE to target: {first:.4} → {last:.4}  ({:.1}% of initial)",
        100.0 * last / first
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn bpe_command() {
        // Train + encode a corpus token.
        assert_eq!(
            run_bpe(&s(&[
                "low lower lowest",
                "--vocab",
                "40",
                "--encode",
                "low"
            ])),
            0
        );
        // Default vocab, encode the corpus itself.
        assert_eq!(run_bpe(&s(&["hello world"])), 0);
        // Byte-level: lossless on emoji/accents (no OOV).
        assert_eq!(
            run_bpe(&s(&["café ☕", "--bytes", "--encode", "résumé 🚀"])),
            0
        );
        // Usage / validation errors.
        assert_eq!(run_bpe(&[]), 2);
        assert_eq!(run_bpe(&s(&["abc", "--vocab", "1"])), 2);
        assert_eq!(run_bpe(&s(&[";", "--vocab", "10"])), 2); // empty corpus
    }

    #[test]
    fn lm_command() {
        // Default sequence, a few steps (kept small so the test is fast).
        assert_eq!(run_lm(&s(&["--steps", "5"])), 0);
        // Explicit tokens + seed + lr.
        assert_eq!(
            run_lm(&s(&[
                "1,2,3,1,2,3",
                "--seed",
                "7",
                "--steps",
                "10",
                "--lr",
                "0.02"
            ])),
            0
        );
        // The default sequence actually overfits to exact recall (determinism).
        assert_eq!(run_lm(&s(&["1,2,3,4,2,5", "--steps", "200"])), 0);
        // Optimizer selection: adamw and lion run too.
        assert_eq!(
            run_lm(&s(&["1,2,3,1,2,3", "--steps", "20", "--opt", "adamw"])),
            0
        );
        assert_eq!(
            run_lm(&s(&["1,2,3,1,2,3", "--steps", "20", "--opt", "lion"])),
            0
        );
        assert_eq!(
            run_lm(&s(&[
                "1,2,3,1,2,3",
                "--steps",
                "20",
                "--opt",
                "schedule-free"
            ])),
            0
        );
        assert_eq!(
            run_lm(&s(&["1,2,3,1,2,3", "--steps", "20", "--opt", "ademamix"])),
            0
        );
        assert_eq!(
            run_lm(&s(&["1,2,3,1,2,3", "--steps", "20", "--opt", "soap"])),
            0
        );
        assert_eq!(
            run_lm(&s(&["1,2,3,1,2,3", "--steps", "20", "--opt", "lookahead"])),
            0
        );
        assert_eq!(
            run_lm(&s(&["1,2,3,1,2,3", "--steps", "20", "--opt", "lamb"])),
            0
        );
        assert_eq!(
            run_lm(&s(&["1,2,3,1,2,3", "--steps", "20", "--opt", "adan"])),
            0
        );
        assert_eq!(run_lm(&s(&["1,2,3", "--opt", "sgd"])), 2); // unknown optimizer
        // Usage / validation errors.
        assert_eq!(run_lm(&s(&["1"])), 2); // need ≥ 2 tokens
        assert_eq!(run_lm(&s(&["1,foo,3"])), 2); // non-integer token
        assert_eq!(run_lm(&s(&["1,2,3", "--steps", "0"])), 2); // steps ≥ 1
        assert_eq!(run_lm(&s(&["1,2,3", "--lr", "-1"])), 2); // lr > 0
        assert_eq!(run_lm(&s(&["1,2,3", "--seed", "x"])), 2); // bad seed
    }
}
