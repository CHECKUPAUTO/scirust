use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

const KEY_FILE_ENV: &str = "CCOS_LICENSE_PUBLIC_KEYS_FILE";
const MAX_KEYS_PER_SCHEME: usize = 8;

fn main() {
    println!("cargo:rerun-if-env-changed={KEY_FILE_ENV}");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join("license_build_keys.rs");

    let generated = match env::var_os(KEY_FILE_ENV) {
        None => generate("none", &[], &[]),
        Some(path) => {
            let path = PathBuf::from(path);
            println!("cargo:rerun-if-changed={}", path.display());
            let text = fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!(
                    "{KEY_FILE_ENV} points to unreadable file {}: {e}",
                    path.display()
                )
            });
            let (profile, ed25519, slh_dsa) = parse_key_file(&text);
            if profile == "test" && env::var("PROFILE").as_deref() == Ok("release") {
                panic!("test licensing keys are forbidden in release builds");
            }
            generate(&profile, &ed25519, &slh_dsa)
        }
    };
    fs::write(out, generated).expect("write generated licensing keyring");
}

type Key = (String, [u8; 32]);

fn parse_key_file(text: &str) -> (String, Vec<Key>, Vec<Key>) {
    if text.len() > 64 * 1024 {
        panic!("licensing public-key file exceeds 64 KiB");
    }
    let mut profile = None;
    let mut ed25519 = Vec::new();
    let mut slh_dsa = Vec::new();
    let mut seen = BTreeSet::new();

    for (index, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.as_slice() {
            ["profile", value] if *value == "production" || *value == "test" => {
                if profile.replace((*value).to_string()).is_some() {
                    panic!("duplicate licensing key profile on line {}", index + 1);
                }
            }
            [scheme @ ("ed25519" | "slh-dsa-shake-128s"), kid, hex] => {
                validate_kid(kid, index + 1);
                if !seen.insert(((*scheme).to_string(), (*kid).to_string())) {
                    panic!("duplicate licensing key {scheme}/{kid}");
                }
                let key = decode_hex_key(hex, index + 1);
                if *scheme == "ed25519" {
                    ed25519.push(((*kid).to_string(), key));
                } else {
                    slh_dsa.push(((*kid).to_string(), key));
                }
            }
            _ => panic!(
                "malformed licensing key file line {} (expected `profile production|test` or `SCHEME KID 64_HEX`) ",
                index + 1
            ),
        }
    }
    let profile = profile.expect("licensing key file is missing `profile production|test`");
    if ed25519.is_empty() && slh_dsa.is_empty() {
        panic!("configured licensing key file contains no public keys");
    }
    if ed25519.len() > MAX_KEYS_PER_SCHEME || slh_dsa.len() > MAX_KEYS_PER_SCHEME {
        panic!("licensing keyring exceeds {MAX_KEYS_PER_SCHEME} keys per scheme");
    }
    (profile, ed25519, slh_dsa)
}

fn validate_kid(kid: &str, line: usize) {
    if kid.is_empty()
        || kid.len() > 64
        || !kid
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    {
        panic!("invalid licensing key id on line {line}");
    }
}

fn decode_hex_key(hex: &str, line: usize) -> [u8; 32] {
    if hex.len() != 64 {
        panic!("public key on line {line} is not 32-byte lowercase hex");
    }
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        let pair = &hex[index * 2..index * 2 + 2];
        if !pair
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            panic!("public key on line {line} is not lowercase hex");
        }
        *byte = u8::from_str_radix(pair, 16).expect("validated hex pair");
    }
    if out == [0u8; 32] {
        panic!("all-zero licensing public keys are forbidden");
    }
    out
}

fn generate(profile: &str, ed25519: &[Key], slh_dsa: &[Key]) -> String {
    let mut out = String::new();
    writeln!(out, "pub const LICENSE_BUILD_PROFILE: &str = {profile:?};").unwrap();
    write_keys(&mut out, "EMBEDDED_ED25519_KEYS", ed25519);
    write_keys(&mut out, "EMBEDDED_SLH_DSA_KEYS", slh_dsa);
    writeln!(
        out,
        "pub const EMBEDDED_ED25519_PRIMARY: [u8; 32] = {:?};",
        ed25519.first().map(|(_, key)| *key).unwrap_or([0u8; 32])
    )
    .unwrap();
    writeln!(
        out,
        "pub const EMBEDDED_SLH_DSA_PRIMARY: [u8; 32] = {:?};",
        slh_dsa.first().map(|(_, key)| *key).unwrap_or([0u8; 32])
    )
    .unwrap();
    out
}

fn write_keys(out: &mut String, name: &str, keys: &[Key]) {
    writeln!(out, "pub const {name}: &[(&str, [u8; 32])] = &[").unwrap();
    for (kid, key) in keys {
        writeln!(out, "    ({kid:?}, {key:?}),").unwrap();
    }
    writeln!(out, "];").unwrap();
}
