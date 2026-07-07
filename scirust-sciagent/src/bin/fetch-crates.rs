//! Fetch top Rust crates from crates.io, extract `.rs` files, and shard them for
//! SCIAGENT training.  Uses the crates.io API for metadata and system `tar` for
//! extraction (no extra Rust dep for archives).
//!
//! Usage:
//!   cargo run --bin fetch-crates -- --count 50 --output ./data/raw
//!   cargo run --bin collect-data -- --input ./data/raw -o ./data/shards --tokenizer ./tokenizer/bpe.json --recursive

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use clap::Parser;
use serde::Deserialize;
use sha2::{Digest, Sha256};

// ── crates.io API response shapes ──────────────────────────────────────

#[derive(Deserialize)]
struct ApiResponse {
    crates: Vec<CrateSummary>,
}

#[derive(Deserialize)]
struct CrateSummary {
    id: String,
    #[serde(default)]
    max_version: String,
    #[serde(default)]
    downloads: u64,
}

#[derive(Deserialize)]
struct CrateDetail {
    #[serde(default)]
    versions: Vec<CrateVersion>,
}

#[derive(Deserialize, Clone)]
struct CrateVersion {
    num: String,
    /// SHA-256 of the published `.crate` tarball, as reported by crates.io.
    /// Verifying the download against this is what makes the supply-chain
    /// fetch tamper-evident (audit `AUDIT_COMPLET.md`, finding S3).
    #[serde(default)]
    checksum: String,
}

// ── CLI ────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "fetch-crates",
    about = "Download top Rust crates from crates.io for SCIAGENT training"
)]
struct Args {
    #[arg(short, long, default_value_t = 50)]
    count: usize,

    #[arg(short, long, default_value = "./data/raw")]
    output: PathBuf,

    #[arg(long, default_value = "30")]
    delay_ms: u64,

    #[arg(long)]
    skip_extract: bool,

    #[arg(long, default_value = "3")]
    max_concurrent: usize,

    #[arg(long)]
    resume: bool,

    /// Path to a file listing specific crates (one per line) instead of fetching top crates.
    #[arg(long)]
    crate_list: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.output).expect("Cannot create output dir");
    let out = &args.output;

    let crates: Vec<CrateSummary> = if let Some(list_path) = &args.crate_list
    {
        let content = fs::read_to_string(list_path).expect("Cannot read crate list");
        content
            .lines()
            .map(|line| CrateSummary {
                id: line.trim().to_string(),
                max_version: String::new(),
                downloads: 0,
            })
            .filter(|c| !c.id.is_empty())
            .collect()
    }
    else
    {
        fetch_top_crates(args.count)
    };

    eprintln!("Fetching {} crates...", crates.len());

    let mut fetched = 0usize;
    let mut skipped = 0usize;

    for c in &crates
    {
        if out.join(format!("{}-done", c.id)).exists() && args.resume
        {
            skipped += 1;
            continue;
        }

        if let Err(e) = fetch_and_extract(c, out, args.skip_extract)
        {
            eprintln!("  SKIP {}: {e}", c.id);
        }
        else
        {
            let _ = fs::File::create(out.join(format!("{}-done", c.id)));
            fetched += 1;
        }

        if args.delay_ms > 0
        {
            std::thread::sleep(Duration::from_millis(args.delay_ms));
        }
    }

    eprintln!(
        "Done. Fetched {fetched} crates (skipped {skipped}) into {:?}",
        out.canonicalize().unwrap_or_else(|_| out.clone())
    );
    eprintln!(
        "Next: cargo run --bin collect-data -- --input {:?} -o ./data/shards --tokenizer ./tokenizer/bpe.json --recursive",
        out
    );
}

fn fetch_top_crates(count: usize) -> Vec<CrateSummary> {
    let url = format!(
        "https://crates.io/api/v1/crates?page=1&per_page={}&sort=downloads",
        count.min(100)
    );
    let resp = ureq::get(&url)
        .set("User-Agent", "scirust-sciagent/0.1 (data-collection)")
        .call()
        .unwrap_or_else(|e| {
            eprintln!("Error fetching crate list: {e}");
            std::process::exit(1);
        });
    let api: ApiResponse = resp.into_json().unwrap_or_else(|e| {
        eprintln!("Error parsing crate list JSON: {e}");
        std::process::exit(1);
    });
    // If we need more than 100, fetch additional pages
    if count > 100
    {
        let mut all = api.crates;
        let pages = count.div_ceil(100);
        for page in 2..=pages
        {
            let url =
                format!("https://crates.io/api/v1/crates?page={page}&per_page=100&sort=downloads");
            if let Ok(resp) = ureq::get(&url)
                .set("User-Agent", "scirust-sciagent/0.1 (data-collection)")
                .call()
            {
                if let Ok(api) = resp.into_json::<ApiResponse>()
                {
                    all.extend(api.crates);
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        all.truncate(count);
        all
    }
    else
    {
        api.crates
    }
}

fn fetch_and_extract(
    krate: &CrateSummary,
    out: &Path,
    skip_extract: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Resolve the version AND its published checksum from the crates.io
    //    detail endpoint. The checksum is the SHA-256 of the .crate tarball
    //    as published; verifying the download against it makes the fetch
    //    tamper-evident (audit `AUDIT_COMPLET.md`, finding S3).
    let detail_url = format!("https://crates.io/api/v1/crates/{}", krate.id);
    let resp = ureq::get(&detail_url)
        .set("User-Agent", "scirust-sciagent/0.1 (data-collection)")
        .call()?;
    let detail: CrateDetail = resp.into_json()?;
    let chosen = if krate.max_version.is_empty()
    {
        detail.versions.first().cloned().ok_or("no versions")?
    }
    else
    {
        detail
            .versions
            .iter()
            .find(|v| v.num == krate.max_version)
            .cloned()
            .or_else(|| detail.versions.first().cloned())
            .ok_or("no versions")?
    };
    let version = chosen.num.clone();
    let expected_checksum = chosen.checksum.clone();

    // 2. Download tarball via crates.io API (follows redirect to static.crates.io)
    let dl_url = format!(
        "https://crates.io/api/v1/crates/{name}/{version}/download",
        name = krate.id,
    );
    let tarball_name = format!("{}-{}.tar.gz", krate.id, version);
    let tarball_path = out.join(&tarball_name);

    if !tarball_path.exists()
    {
        eprintln!(
            "  DL {} v{} ({} downloads)",
            krate.id, version, krate.downloads
        );
        let resp = ureq::get(&dl_url)
            .set("User-Agent", "scirust-sciagent/0.1 (data-collection)")
            .timeout(Duration::from_secs(120))
            .call()?;

        let mut body = Vec::new();
        resp.into_reader().read_to_end(&mut body)?;

        // Supply-chain integrity: verify the SHA-256 of the downloaded bytes
        // against the checksum published by crates.io for this version. A
        // mismatch (MITM, corrupted mirror, substituted tarball) aborts the
        // fetch and discards the bad file rather than extracting it.
        if !expected_checksum.is_empty()
        {
            let got = to_hex(&Sha256::digest(&body));
            if !got.eq_ignore_ascii_case(&expected_checksum)
            {
                eprintln!(
                    "  REJECT {} v{}: checksum mismatch (expected {}, got {})",
                    krate.id, version, expected_checksum, got
                );
                let _ = fs::remove_file(&tarball_path);
                return Err(format!(
                    "checksum mismatch for {} v{}: expected {}, got {}",
                    krate.id, version, expected_checksum, got
                )
                .into());
            }
        }
        else
        {
            eprintln!(
                "  WARN {} v{}: no published checksum to verify against",
                krate.id, version
            );
        }

        let mut f = fs::File::create(&tarball_path)?;
        f.write_all(&body)?;
    }
    else
    {
        eprintln!("  CACHED {} v{} (tarball exists)", krate.id, version);
    }

    // 3. Extract .rs files
    if !skip_extract
    {
        let extract_dir = out.join(&krate.id);
        if !extract_dir.exists()
        {
            fs::create_dir_all(&extract_dir)?;
            let status = Command::new("tar")
                .args([
                    "xzf",
                    &tarball_path.to_string_lossy(),
                    "-C",
                    &extract_dir.to_string_lossy(),
                    "--strip-components=1",
                ])
                .status()?;
            if !status.success()
            {
                eprintln!("  tar extraction failed for {}", krate.id);
            }
            // Defense-in-depth against path traversal in a malicious tarball:
            // every extracted entry must canonicalize to a path inside
            // `extract_dir`. Anything escaping is removed and the crate is
            // rejected (audit finding S3).
            ensure_contained(&extract_dir, &extract_dir)?;
        }

        // Collect .rs files into a flat "all" directory
        let all_dir = out.join("all");
        fs::create_dir_all(&all_dir)?;
        collect_rs_files(&extract_dir, &all_dir, &krate.id)?;
    }

    Ok(())
}

/// Recursively verify that every path under `root` stays inside `base` (after
/// canonicalization). Removes and rejects any entry that escapes via `..` or
/// an absolute/symlinked path — a defense against tar path traversal.
fn ensure_contained(base: &Path, root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let base_canon = base.canonicalize().unwrap_or_else(|_| base.to_path_buf());
    for entry in fs::read_dir(root)?
    {
        let entry = entry?;
        let path = entry.path();
        // Skip the entry if it cannot be canonicalized (e.g. broken symlink);
        // a malicious symlink targeting outside `base` is removed.
        match path.canonicalize()
        {
            Ok(canon) =>
            {
                if !canon.starts_with(&base_canon)
                {
                    eprintln!("  REJECT path escaping extract dir: {:?}", path);
                    if path.is_dir()
                    {
                        let _ = fs::remove_dir_all(&path);
                    }
                    else
                    {
                        let _ = fs::remove_file(&path);
                    }
                    return Err(format!("extracted path escapes base dir: {:?}", path).into());
                }
            },
            Err(_) =>
            {
                // Broken/suspicious symlink — remove it.
                let _ = fs::remove_file(&path);
            },
        }
        if path.is_dir()
        {
            ensure_contained(base, &path)?;
        }
    }
    Ok(())
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    use std::fmt::Write;
    for b in bytes
    {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn collect_rs_files(
    dir: &Path,
    out: &Path,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !dir.is_dir()
    {
        return Ok(());
    }
    let mut file_count = 0u32;
    collect_rs_recursive(dir, out, prefix, &mut file_count)?;
    if file_count > 0
    {
        eprintln!("    -> {file_count} .rs files");
    }
    Ok(())
}

fn collect_rs_recursive(
    dir: &Path,
    out: &Path,
    prefix: &str,
    count: &mut u32,
) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir()
        {
            collect_rs_recursive(&path, out, prefix, count)?;
        }
        else if path.extension().is_some_and(|e| e == "rs")
        {
            *count += 1;
            let name = format!("{prefix}_{count}.rs");
            // Symlink to avoid duplicating disk space
            let link = out.join(&name);
            if !link.exists()
            {
                let _ = std::os::unix::fs::symlink(&path, &link);
            }
        }
    }
    Ok(())
}
