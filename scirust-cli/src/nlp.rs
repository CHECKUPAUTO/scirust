//! NLP subcommands over `scirust-learning` (deterministic, tested).
//!
//! Commands: bpe.

use scirust_learning::nlp::bpe::BpeTokenizer;
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

/// `bpe "<corpus>" [--vocab N] [--encode "<text>"]` — train a deterministic
/// byte-pair-encoding tokenizer on the corpus (documents separated by `;`),
/// then encode/decode a piece of text. Reports the learned vocab size, the
/// token ids, and the round-trip.
pub fn run_bpe(args: &[String]) -> u8 {
    let (vocab_s, rest) = take_flag(args, "--vocab");
    let (enc_s, rest) = take_flag(&rest, "--encode");
    let Some(corpus) = rest.first()
    else
    {
        eprintln!("usage: scirust bpe \"<corpus>\" [--vocab N] [--encode \"<text>\"]");
        return 2;
    };
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
        None => 50,
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

    let tok = BpeTokenizer::train(&docs, vocab);
    let text = enc_s.as_deref().unwrap_or(docs[0]);
    let ids = tok.tokenize(text);
    let decoded = tok.decode(&ids);

    println!(
        "trained BPE: vocab size {} (target {vocab}) on {} document(s)",
        tok.vocab_size(),
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
            "lossy (out-of-vocabulary characters mapped to <UNK>)"
        }
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
        // Usage / validation errors.
        assert_eq!(run_bpe(&[]), 2);
        assert_eq!(run_bpe(&s(&["abc", "--vocab", "1"])), 2);
        assert_eq!(run_bpe(&s(&[";", "--vocab", "10"])), 2); // empty corpus
    }
}
