//! **Vendor-side** offline license tool — requires `--features license`.
//!
//! The engine only ever *verifies* (against the baked-in [`ccos_core::license::LICENSE_PUBLIC_KEY`]); this
//! is the private-key side, never shipped to customers. Two subcommands:
//!
//! - `keygen` — generate an ed25519 keypair. Paste the printed public key into `LICENSE_PUBLIC_KEY`
//!   (`src/license.rs`) and keep the secret seed somewhere safe; it signs every license.
//! - `sign` — sign a token for a licensee, optionally with an expiry in days (omit for perpetual).
//!
//! ```text
//! cargo run --features license --example license_sign -- keygen
//! CCOS_LICENSE_SIGNING_SEED=<64-hex-seed> \
//!   cargo run --features license --example license_sign -- sign --licensee "Acme Corp" --days 365
//! ```
//! The emitted token goes in `$CCOS_LICENSE` or the license file; `ccos license` then reports Pro.

use ccos_core::license::sign_token;
use std::fmt::Write as _;

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

fn now_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str).unwrap_or("help") {
        "keygen" => {
            // The 32-byte seed IS the private key; its ed25519 public key is what gets embedded.
            use rand::RngCore;
            let mut seed = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut seed);
            let sk = ed25519_dalek::SigningKey::from_bytes(&seed);
            let pk = sk.verifying_key().to_bytes();
            println!("// Paste into src/license.rs as the embedded vendor key:");
            print!("pub const LICENSE_PUBLIC_KEY: [u8; 32] = [");
            for (i, b) in pk.iter().enumerate() {
                if i % 12 == 0 {
                    print!("\n    ");
                }
                print!("{b}, ");
            }
            println!("\n];\n");
            println!("# KEEP SECRET — signing seed (export as CCOS_LICENSE_SIGNING_SEED to sign):");
            println!("{}", hex(&seed));
        }
        "sign" => {
            let Some(seed) = std::env::var("CCOS_LICENSE_SIGNING_SEED")
                .ok()
                .and_then(|s| parse_hex32(&s))
            else {
                eprintln!(
                    "set CCOS_LICENSE_SIGNING_SEED to the 64-hex signing seed (from `keygen`)"
                );
                std::process::exit(2);
            };
            let mut licensee = "unnamed".to_string();
            let mut days: Option<u64> = None;
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--licensee" => {
                        i += 1;
                        if let Some(v) = args.get(i) {
                            licensee = v.clone();
                        }
                    }
                    "--days" => {
                        i += 1;
                        days = args.get(i).and_then(|v| v.parse().ok());
                    }
                    other => eprintln!("ignoring unknown flag '{other}'"),
                }
                i += 1;
            }
            let exp = days.map(|d| now_secs() + d * 86_400);
            println!("{}", sign_token(&seed, &licensee, exp));
        }
        _ => {
            eprintln!("usage: license_sign keygen | sign --licensee NAME [--days N]");
            std::process::exit(2);
        }
    }
}
