//! `prov` — offline signer and public verifier for SciRust-emitted artifacts.
//!
//! ```text
//! prov sign   --seed <hex64> [--leaf <n>] [--height <h>] [--in-place] <file>
//! prov sign   --seed-file <path> [--leaf <n>] [--height <h>] [--in-place] <file>
//! prov verify [--root <hex64>] <file>
//! ```
//!
//! `sign` is the **offline vendor step** — it needs the secret master seed and
//! must run on a trusted host (an HSM export, never a shipped binary). By default
//! it writes the signed artifact to stdout; `--in-place` rewrites `<file>`.
//!
//! `verify` ships to anyone: it needs only the public root (defaults to the pinned
//! [`scirust_provenance::EMIT_ROOT_HEX`]). Exit code `0` means verified, `1` means
//! the banner is missing/malformed/forged, `2` means a usage error.

use scirust_license::hashsig::{self, MerkleSigner};
use scirust_provenance::{Verdict, emit_root, sign_artifact, verify_artifact};
use std::process::ExitCode;

const USAGE: &str = "\
prov — SciRust codegen provenance tool

USAGE:
  prov sign   --seed <hex64> | --seed-file <path> [--leaf <n>] [--height <h>] [--in-place] <file>
  prov verify [--root <hex64>] <file>

SIGN (offline, needs the secret seed):
  --seed <hex64>     32-byte master seed as 64 hex chars
  --seed-file <path> read the seed (hex, whitespace-trimmed) from a file
  --leaf <n>         one-time leaf index (default 0); never reuse for two artifacts
  --height <h>       Merkle height, 1..=20 (default 10)
  --in-place         rewrite <file> instead of printing to stdout

VERIFY (public, needs only the root):
  --root <hex64>     trusted 32-byte Merkle root (default: pinned EMIT_ROOT)

Exit: 0 verified · 1 not verified · 2 usage error";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str)
    {
        Some("sign") => cmd_sign(&args[1..]),
        Some("verify") => cmd_verify(&args[1..]),
        Some("-h") | Some("--help") | None =>
        {
            println!("{USAGE}");
            ExitCode::SUCCESS
        },
        Some(other) =>
        {
            eprintln!("prov: unknown subcommand `{other}`\n\n{USAGE}");
            ExitCode::from(2)
        },
    }
}

/// A single `--flag value` / positional parser — no external dependency, keeping
/// the tool auditable in the SciRust spirit.
struct Args {
    seed: Option<String>,
    seed_file: Option<String>,
    root: Option<String>,
    leaf: u32,
    height: u32,
    in_place: bool,
    file: Option<String>,
}

fn parse(rest: &[String]) -> Result<Args, String> {
    let mut a = Args {
        seed: None,
        seed_file: None,
        root: None,
        leaf: 0,
        height: 10,
        in_place: false,
        file: None,
    };
    let mut it = rest.iter();
    while let Some(tok) = it.next()
    {
        match tok.as_str()
        {
            "--seed" => a.seed = Some(next(&mut it, "--seed")?),
            "--seed-file" => a.seed_file = Some(next(&mut it, "--seed-file")?),
            "--root" => a.root = Some(next(&mut it, "--root")?),
            "--leaf" =>
            {
                a.leaf = next(&mut it, "--leaf")?
                    .parse()
                    .map_err(|_| "--leaf must be a u32".to_string())?
            },
            "--height" =>
            {
                a.height = next(&mut it, "--height")?
                    .parse()
                    .map_err(|_| "--height must be a u32".to_string())?
            },
            "--in-place" => a.in_place = true,
            other if other.starts_with("--") =>
            {
                return Err(format!("unknown flag `{other}`"));
            },
            positional =>
            {
                if a.file.is_some()
                {
                    return Err("expected exactly one <file>".to_string());
                }
                a.file = Some(positional.to_string());
            },
        }
    }
    Ok(a)
}

fn next(it: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<String, String> {
    it.next()
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn seed_bytes(a: &Args) -> Result<[u8; 32], String> {
    let hex = if let Some(h) = &a.seed
    {
        h.trim().to_string()
    }
    else if let Some(path) = &a.seed_file
    {
        std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read seed file {path}: {e}"))?
            .trim()
            .to_string()
    }
    else
    {
        return Err("sign needs --seed <hex64> or --seed-file <path>".to_string());
    };
    let bytes = hashsig::hex_decode(&hex).ok_or("seed is not valid hex")?;
    if bytes.len() != 32
    {
        return Err(format!("seed must be 32 bytes (got {})", bytes.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn cmd_sign(rest: &[String]) -> ExitCode {
    let a = match parse(rest)
    {
        Ok(a) => a,
        Err(e) => return usage_err(&e),
    };
    let Some(file) = a.file.clone()
    else
    {
        return usage_err("sign needs a <file>");
    };
    let seed = match seed_bytes(&a)
    {
        Ok(s) => s,
        Err(e) => return usage_err(&e),
    };
    let src = match std::fs::read_to_string(&file)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("prov: cannot read {file}: {e}");
            return ExitCode::from(2);
        },
    };
    let signer = MerkleSigner::from_seed(&seed, a.height);
    if a.leaf as usize >= signer.capacity()
    {
        return usage_err(&format!(
            "--leaf {} out of range for height {} (capacity {})",
            a.leaf,
            a.height,
            signer.capacity()
        ));
    }
    let signed = sign_artifact(&src, &signer, a.leaf);

    if a.in_place
    {
        if let Err(e) = std::fs::write(&file, &signed)
        {
            eprintln!("prov: cannot write {file}: {e}");
            return ExitCode::from(2);
        }
        eprintln!(
            "prov: signed {file} in place (root {}, leaf {})",
            hashsig::hex_encode(&signer.root()[..4]),
            a.leaf
        );
    }
    else
    {
        print!("{signed}");
    }
    ExitCode::SUCCESS
}

fn cmd_verify(rest: &[String]) -> ExitCode {
    let a = match parse(rest)
    {
        Ok(a) => a,
        Err(e) => return usage_err(&e),
    };
    let Some(file) = a.file.clone()
    else
    {
        return usage_err("verify needs a <file>");
    };
    let root = match &a.root
    {
        None => emit_root(),
        Some(hex) => match hashsig::hex_decode(hex.trim())
        {
            Some(b) if b.len() == 32 =>
            {
                let mut r = [0u8; 32];
                r.copy_from_slice(&b);
                r
            },
            _ => return usage_err("--root must be 64 hex chars (32 bytes)"),
        },
    };
    let src = match std::fs::read_to_string(&file)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("prov: cannot read {file}: {e}");
            return ExitCode::from(2);
        },
    };
    match verify_artifact(&src, &root)
    {
        Verdict::Verified { leaf } =>
        {
            println!(
                "VERIFIED  {file}  (root {}, leaf {leaf})",
                hashsig::hex_encode(&root[..4])
            );
            ExitCode::SUCCESS
        },
        Verdict::NoBanner =>
        {
            println!("NO-BANNER {file}  (no srl-emit provenance mark found)");
            ExitCode::from(1)
        },
        Verdict::MalformedBanner =>
        {
            println!("MALFORMED {file}  (banner present but signature unreadable)");
            ExitCode::from(1)
        },
        Verdict::Forged =>
        {
            println!("FORGED    {file}  (banner does not verify against this root)");
            ExitCode::from(1)
        },
    }
}

fn usage_err(msg: &str) -> ExitCode {
    eprintln!("prov: {msg}\n\n{USAGE}");
    ExitCode::from(2)
}
