//! `scirust-verify` — émission et vérification de certificats d'inférence.
//!
//! Productise `scirust_runtime::proof` (format canonique `SCIRUST-PROOF-1`)
//! en outil fichier-à-fichier utilisable par un tiers, sans écrire de Rust :
//!
//! ```text
//! scirust-verify emit   <model.qsr1> <out.proof> [batch] [seeds...]
//! scirust-verify verify <bundle.proof> <model.qsr1>
//! ```
//!
//! `emit` exécute le modèle sur des entrées canoniques seedées et scelle :
//! sha256 de l'artefact, certificat de ressources (RAM scratch, MACs,
//! couches, dim sortie), et pour chaque seed l'empreinte FNV 64-bit + le
//! sha256 des sorties — après avoir prouvé que les chemins std et no_std
//! produisent des bits identiques.
//!
//! `verify` re-dérive TOUT depuis les octets (rien n'est cru sur parole)
//! et sort avec le code 0 si chaque contrôle passe, 1 sinon — utilisable
//! en CI ou en audit externe.

use std::process::ExitCode;

use scirust_runtime::proof::{ProofBundle, verify_file};

const DEFAULT_BATCH: usize = 2;
const DEFAULT_SEEDS: [u64; 3] = [1, 2, 3];

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  scirust-verify emit   <model.qsr1> <out.proof> [batch] [seeds...]\n  scirust-verify verify <bundle.proof> <model.qsr1>"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str)
    {
        Some("emit") => emit(&args[1..]),
        Some("verify") => verify(&args[1..]),
        _ => usage(),
    }
}

fn emit(args: &[String]) -> ExitCode {
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
                return ExitCode::from(2);
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
            return ExitCode::from(2);
        },
    };

    let bundle = ProofBundle::build(&artifact, batch, &seeds);
    let text = bundle.to_canonical();
    if let Err(e) = std::fs::write(out_path, &text)
    {
        eprintln!("error: cannot write {out_path}: {e}");
        return ExitCode::from(2);
    }
    println!(
        "proof bundle written: {out_path} ({} vectors, bundle_sha256={})",
        bundle.vectors.len(),
        bundle.bundle_digest()
    );
    ExitCode::SUCCESS
}

fn verify(args: &[String]) -> ExitCode {
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
            return ExitCode::from(2);
        },
    };
    let artifact = match std::fs::read(model_path)
    {
        Ok(b) => b,
        Err(e) =>
        {
            eprintln!("error: cannot read {model_path}: {e}");
            return ExitCode::from(2);
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
        ExitCode::SUCCESS
    }
    else
    {
        println!("VERDICT: MISMATCH — au moins un contrôle a échoué.");
        ExitCode::FAILURE
    }
}
