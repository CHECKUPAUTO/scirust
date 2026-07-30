//! **Vendor-side** offline *post-quantum* license tool — requires `--features license-pq`.
//!
//! The engine only ever *verifies* (against the baked-in [`ccos_core::license::LICENSE_SLH_DSA_PUBLIC_KEY`]);
//! this is the private-key side, never shipped to customers. It uses **SLH-DSA** (NIST FIPS 205,
//! formerly SPHINCS+), SLH-DSA-SHAKE-128s: a 32-byte public key and a 7,856-byte (~10.5 KB base64url)
//! post-quantum signature. Two subcommands:
//!
//! - `keygen` — generate an SLH-DSA keypair from a random 48-byte seed. Paste the printed public key
//!   into `LICENSE_SLH_DSA_PUBLIC_KEY` (`src/license.rs`) and keep the 64-byte secret key somewhere
//!   safe; it signs every post-quantum license.
//! - `sign` — sign a `slhdsa.`-tagged token for a licensee, optionally with an expiry in days (omit
//!   for perpetual). Signing is deterministic, so a given secret key + payload always yield the same
//!   token.
//!
//! ```text
//! cargo run --features license-pq --example license_sign_pq -- keygen
//! CCOS_LICENSE_PQ_SIGNING_SEED=<128-hex-secret-key> \
//!   cargo run --features license-pq --example license_sign_pq -- sign --licensee "Acme Corp" --days 365
//! ```
//! The emitted token goes in the license file (preferred for the ~10.5 KB size) or `$CCOS_LICENSE`;
//! `ccos doctor` / `ccos license` then report Pro with the `slh-dsa` verifier.
//!
//! **Note:** the `lattice-slh-dsa` crate is pure Rust but **not independently audited** — see
//! `docs/DEPLOYMENT.md` §4b before trusting it to gate production features.

use ccos_core::license::sign_token_slhdsa;
use std::fmt::Write as _;

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Parse 128 hex chars into the 64-byte SLH-DSA secret key.
fn parse_hex64(s: &str) -> Option<[u8; 64]> {
    let s = s.trim();
    if s.len() != 128 {
        return None;
    }
    let mut out = [0u8; 64];
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
            // keygen_seed derives (pk, sk) deterministically from a 48-byte seed (3·n for 128s).
            // We draw the seed from rand (already a CCOS dependency) — the crate's own getrandom
            // RNG is off (default-features = false), so we supply the entropy ourselves.
            use rand::RngCore;
            let mut seed = [0u8; 48];
            rand::thread_rng().fill_bytes(&mut seed);
            let (pk, sk) = slh_dsa::keygen_seed(slh_dsa::params::SLH_DSA_SHAKE_128S, &seed);
            assert_eq!(pk.len(), 32);
            assert_eq!(sk.len(), 64);
            println!(
                "// Paste into src/license.rs as the embedded post-quantum vendor key (SLH-DSA-SHAKE-128s):"
            );
            print!("pub const LICENSE_SLH_DSA_PUBLIC_KEY: [u8; 32] = [");
            for (i, b) in pk.iter().enumerate() {
                if i % 12 == 0 {
                    print!("\n    ");
                }
                print!("{b}, ");
            }
            println!("\n];\n");
            println!(
                "# KEEP SECRET — 64-byte SLH-DSA secret key (export as \
                 CCOS_LICENSE_PQ_SIGNING_SEED to sign):"
            );
            println!("{}", hex(&sk));
        }
        "sign" => {
            let Some(sk) = std::env::var("CCOS_LICENSE_PQ_SIGNING_SEED")
                .ok()
                .and_then(|s| parse_hex64(&s))
            else {
                eprintln!(
                    "set CCOS_LICENSE_PQ_SIGNING_SEED to the 128-hex secret key (from `keygen`)"
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
            println!("{}", sign_token_slhdsa(&sk, &licensee, exp));
        }
        _ => {
            eprintln!("usage: license_sign_pq keygen | sign --licensee NAME [--days N]");
            std::process::exit(2);
        }
    }
}
