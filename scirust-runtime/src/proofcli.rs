//! File-to-file driver for inference proof certificates (`SCIRUST-PROOF-1`).
//!
//! Shared by the standalone `scirust-verify` binary and the unified
//! `scirust verify` subcommand so the emit/verify logic lives in exactly
//! one place (no duplication). Returns a process exit code: 0 = success /
//! MATCH, 1 = verification MISMATCH, 2 = usage or I/O error.

use crate::proof::{ProofBundle, verify_file};

const DEFAULT_BATCH: usize = 2;
const DEFAULT_SEEDS: [u64; 3] = [1, 2, 3];

/// Dispatch `emit` / `verify`. `args` excludes the program name.
pub fn run(args: &[String]) -> u8 {
    match args.first().map(String::as_str)
    {
        Some("emit") => emit(&args[1..]),
        Some("verify") => verify(&args[1..]),
        _ => usage(),
    }
}

fn usage() -> u8 {
    eprintln!(
        "usage:\n  scirust verify emit   <model.qsr1> <out.proof> [batch] [seeds...]\n  scirust verify verify <bundle.proof> <model.qsr1>"
    );
    2
}

fn emit(args: &[String]) -> u8 {
    let (model_path, out_path) = match (args.first(), args.get(1))
    {
        (Some(m), Some(o)) => (m, o),
        _ => return usage(),
    };
    let batch = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BATCH);
    let seeds: Vec<u64> = if args.len() > 3
    {
        match args[3..].iter().map(|s| s.parse::<u64>()).collect()
        {
            Ok(v) => v,
            Err(_) =>
            {
                eprintln!("error: seeds must be u64");
                return 2;
            },
        }
    }
    else
    {
        DEFAULT_SEEDS.to_vec()
    };

    let artifact = match std::fs::read(model_path)
    {
        Ok(b) => b,
        Err(e) =>
        {
            eprintln!("error: cannot read {model_path}: {e}");
            return 2;
        },
    };

    let bundle = ProofBundle::build(&artifact, batch, &seeds);
    let text = bundle.to_canonical();
    if let Err(e) = std::fs::write(out_path, &text)
    {
        eprintln!("error: cannot write {out_path}: {e}");
        return 2;
    }
    println!(
        "proof bundle written: {out_path} ({} vectors, bundle_sha256={})",
        bundle.vectors.len(),
        bundle.bundle_digest()
    );
    0
}

fn verify(args: &[String]) -> u8 {
    let (bundle_path, model_path) = match (args.first(), args.get(1))
    {
        (Some(b), Some(m)) => (b, m),
        _ => return usage(),
    };
    let text = match std::fs::read_to_string(bundle_path)
    {
        Ok(t) => t,
        Err(e) =>
        {
            eprintln!("error: cannot read {bundle_path}: {e}");
            return 2;
        },
    };
    let artifact = match std::fs::read(model_path)
    {
        Ok(b) => b,
        Err(e) =>
        {
            eprintln!("error: cannot read {model_path}: {e}");
            return 2;
        },
    };

    let checks = verify_file(&text, &artifact);
    let mut all = true;
    for (label, ok) in &checks
    {
        println!("  [{}] {label}", if *ok { "PASS" } else { "FAIL" });
        all &= *ok;
    }
    if all
    {
        println!("VERDICT: MATCH — l'artefact reproduit le certificat bit pour bit.");
        0
    }
    else
    {
        println!("VERDICT: MISMATCH — au moins un contrôle a échoué.");
        1
    }
}
