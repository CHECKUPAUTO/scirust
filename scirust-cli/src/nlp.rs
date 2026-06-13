//! NLP subcommands over `scirust-learning` (deterministic, tested).
//!
//! Commands: bpe.

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
}
