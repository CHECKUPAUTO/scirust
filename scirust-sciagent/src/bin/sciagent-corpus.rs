//! Inspect and migrate legacy SCIAGENT training corpora.
//!
//! A raw crate corpus is generated data, not source.  The legacy location
//! `data/crates_raw` sits inside the checkout and makes recursive search tools
//! traverse millions of downloaded files.  This utility moves that directory
//! atomically to the configured external corpus location and rebuilds the
//! generated `all/` symlink aggregate before the move.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};
use scirust_sciagent::corpus_paths;

#[derive(Parser)]
#[command(
    name = "sciagent-corpus",
    about = "Inspect and safely migrate generated SCIAGENT training corpora"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the external default locations without creating them.
    Location,
    /// Move `data/crates_raw` out of this checkout after an explicit --apply.
    Migrate {
        /// External destination. Defaults to SCIRUST_SCIAGENT_CORPUS_DIR or
        /// the platform data directory.
        #[arg(long, value_name = "DIR")]
        destination: Option<PathBuf>,

        /// Perform the migration. Without this flag the command is a dry run.
        #[arg(long)]
        apply: bool,
    },
}

#[derive(Default)]
struct Inventory {
    directories: usize,
    files: usize,
    symlinks: usize,
    broken_symlinks: usize,
}

#[derive(Default)]
struct AggregateReport {
    previous_links: usize,
    rebuilt_links: usize,
}

struct MigrationReport {
    source: PathBuf,
    destination: PathBuf,
    inventory: Inventory,
    aggregate: Option<AggregateReport>,
}

fn main() {
    if let Err(error) = run(Args::parse())
    {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> io::Result<()> {
    match args.command
    {
        Command::Location =>
        {
            println!(
                "corpus={}",
                corpus_paths::resolve_external_corpus_dir(None)?.display()
            );
            println!(
                "shards={}",
                corpus_paths::resolve_external_shards_dir(None)?.display()
            );
            Ok(())
        },
        Command::Migrate { destination, apply } =>
        {
            let workspace = corpus_paths::workspace_root()?;
            let source = workspace.join("data").join("crates_raw");
            let destination = corpus_paths::resolve_external_corpus_dir(destination)?;
            let report = migrate_legacy_corpus(&workspace, &source, &destination, apply)?;

            println!("legacy corpus: {}", report.source.display());
            println!("destination: {}", report.destination.display());
            println!(
                "inventory: {} directories, {} files, {} symlinks, {} broken symlinks",
                report.inventory.directories,
                report.inventory.files,
                report.inventory.symlinks,
                report.inventory.broken_symlinks
            );

            if let Some(aggregate) = report.aggregate
            {
                println!(
                    "aggregate rebuilt: {} legacy links replaced by {} deterministic links",
                    aggregate.previous_links, aggregate.rebuilt_links
                );
                println!("migration complete: the source checkout no longer contains the corpus");
            }
            else
            {
                println!("dry run only: no data was changed; rerun with --apply to migrate");
            }
            Ok(())
        },
    }
}

fn migrate_legacy_corpus(
    workspace: &Path,
    source: &Path,
    destination: &Path,
    apply: bool,
) -> io::Result<MigrationReport> {
    let workspace = fs::canonicalize(workspace)?;
    let source = fs::canonicalize(source).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("legacy corpus not found at {}: {error}", source.display()),
        )
    })?;

    if !source.starts_with(&workspace)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "refusing to migrate a source outside the SciRust checkout: {}",
                source.display()
            ),
        ));
    }
    if !source.is_dir()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("legacy corpus is not a directory: {}", source.display()),
        ));
    }

    let destination = corpus_paths::validate_external_directory_against(destination, &workspace)?;
    if destination.exists() || fs::symlink_metadata(&destination).is_ok()
    {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!(
                "refusing to overwrite existing external corpus destination: {}",
                destination.display()
            ),
        ));
    }

    let inventory = inspect_tree(&source)?;
    let aggregate = if apply
    {
        let parent = destination.parent().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("destination has no parent: {}", destination.display()),
            )
        })?;
        fs::create_dir_all(parent)?;
        let aggregate = rebuild_aggregate(&source)?;

        fs::rename(&source, &destination).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "atomic move from {} to {} failed; source data was left in place: {error}",
                    source.display(),
                    destination.display()
                ),
            )
        })?;
        Some(aggregate)
    }
    else
    {
        None
    };

    Ok(MigrationReport {
        source,
        destination,
        inventory,
        aggregate,
    })
}

fn inspect_tree(root: &Path) -> io::Result<Inventory> {
    let mut inventory = Inventory::default();
    inspect_directory(root, &mut inventory)?;
    Ok(inventory)
}

fn inspect_directory(directory: &Path, inventory: &mut Inventory) -> io::Result<()> {
    for entry in sorted_entries(directory)?
    {
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir()
        {
            inventory.directories += 1;
            inspect_directory(&path, inventory)?;
        }
        else if file_type.is_symlink()
        {
            inventory.symlinks += 1;
            if fs::metadata(&path).is_err()
            {
                inventory.broken_symlinks += 1;
            }
        }
        else
        {
            inventory.files += 1;
        }
    }
    Ok(())
}

/// Rebuild `all/` from the extracted crate directories.  The old aggregate is
/// accepted only if it contains symlinks exclusively; it is then replaced by a
/// deterministic aggregate.  No downloaded crate source or tarball is removed.
fn rebuild_aggregate(corpus: &Path) -> io::Result<AggregateReport> {
    let aggregate = corpus.join("all");
    let previous_links = match fs::symlink_metadata(&aggregate)
    {
        Ok(metadata) =>
        {
            if !metadata.file_type().is_dir()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("aggregate is not a directory: {}", aggregate.display()),
                ));
            }
            ensure_generated_aggregate(&aggregate)?
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error),
    };

    let staging = unique_aggregate_path(corpus, "rebuild");
    fs::create_dir(&staging)?;
    let rebuilt_links = match populate_aggregate(corpus, &staging)
    {
        Ok(count) => count,
        Err(error) =>
        {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        },
    };

    if aggregate.exists()
    {
        let backup = unique_aggregate_path(corpus, "backup");
        fs::rename(&aggregate, &backup)?;
        if let Err(error) = fs::rename(&staging, &aggregate)
        {
            let restore = fs::rename(&backup, &aggregate);
            let message = match restore
            {
                Ok(()) => format!(
                    "could not install rebuilt aggregate; original aggregate restored: {error}"
                ),
                Err(restore_error) => format!(
                    "could not install rebuilt aggregate ({error}) and could not restore original aggregate ({restore_error})"
                ),
            };
            return Err(io::Error::new(error.kind(), message));
        }
        // Preflight proved the backup contains only generated symlinks.  Their
        // targets are deterministically reconstructed in the new aggregate.
        fs::remove_dir_all(&backup)?;
    }
    else
    {
        fs::rename(&staging, &aggregate)?;
    }

    Ok(AggregateReport {
        previous_links,
        rebuilt_links,
    })
}

fn ensure_generated_aggregate(aggregate: &Path) -> io::Result<usize> {
    let mut links = 0;
    for entry in sorted_entries(aggregate)?
    {
        if !entry.file_type()?.is_symlink()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "refusing to replace non-symlink aggregate entry: {}",
                    entry.path().display()
                ),
            ));
        }
        links += 1;
    }
    Ok(links)
}

fn populate_aggregate(corpus: &Path, aggregate: &Path) -> io::Result<usize> {
    let mut link_count = 0;
    for entry in sorted_entries(corpus)?
    {
        let name = entry.file_name();
        if name == "all" || name.to_string_lossy().starts_with(".all-")
        {
            continue;
        }
        if entry.file_type()?.is_dir()
        {
            let prefix = name.to_string_lossy();
            let mut crate_file_count = 0;
            collect_crate_sources(
                corpus,
                &entry.path(),
                aggregate,
                &prefix,
                &mut crate_file_count,
            )?;
            link_count += crate_file_count;
        }
    }
    Ok(link_count)
}

fn collect_crate_sources(
    corpus: &Path,
    directory: &Path,
    aggregate: &Path,
    prefix: &str,
    file_count: &mut usize,
) -> io::Result<()> {
    for entry in sorted_entries(directory)?
    {
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir()
        {
            collect_crate_sources(corpus, &path, aggregate, prefix, file_count)?;
        }
        else if file_type.is_file() && path.extension().is_some_and(|extension| extension == "rs")
        {
            *file_count += 1;
            let source_relative = path.strip_prefix(corpus).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("source escapes corpus root: {}", path.display()),
                )
            })?;
            let target = PathBuf::from("..").join(source_relative);
            let link = aggregate.join(format!("{prefix}_{file_count}.rs"));
            create_file_symlink(&target, &link)?;
        }
    }
    Ok(())
}

fn create_file_symlink(target: &Path, link: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(target, link)
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = (target, link);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "SCIAGENT corpus symlinks are unsupported on this platform",
        ))
    }
}

fn sorted_entries(directory: &Path) -> io::Result<Vec<fs::DirEntry>> {
    let mut entries = fs::read_dir(directory)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    Ok(entries)
}

fn unique_aggregate_path(corpus: &Path, purpose: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos();
    corpus.join(format!(".all-{purpose}-{}-{nonce}", std::process::id()))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn temporary_directory(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "scirust-sciagent-corpus-{test_name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("create temporary directory");
        directory
    }

    #[test]
    fn migration_rebuilds_legacy_links_and_removes_the_checkout_corpus() {
        let root = temporary_directory("migration");
        let workspace = root.join("workspace");
        let corpus = workspace.join("data/crates_raw");
        let source = corpus.join("example/src");
        let aggregate = corpus.join("all");
        let destination = root.join("external/crates_raw");

        fs::create_dir_all(&source).expect("create source directory");
        fs::create_dir_all(&aggregate).expect("create aggregate directory");
        fs::write(source.join("z.rs"), "pub fn z() {}").expect("write z.rs");
        fs::write(source.join("a.rs"), "pub fn a() {}").expect("write a.rs");
        std::os::unix::fs::symlink(
            "data/crates_raw/example/src/a.rs",
            aggregate.join("example_1.rs"),
        )
        .expect("create deliberately broken legacy link");
        std::os::unix::fs::symlink("missing.rs", aggregate.join("stale.rs"))
            .expect("create stale legacy link");

        let report = migrate_legacy_corpus(&workspace, &corpus, &destination, true)
            .expect("migrate legacy corpus");

        assert_eq!(report.inventory.broken_symlinks, 2);
        assert!(!corpus.exists());
        assert_eq!(
            fs::read_link(destination.join("all/example_1.rs")).expect("read repaired link"),
            PathBuf::from("../example/src/a.rs")
        );
        assert_eq!(
            fs::read_to_string(destination.join("all/example_1.rs")).expect("follow repaired link"),
            "pub fn a() {}"
        );
        assert!(fs::symlink_metadata(destination.join("all/stale.rs")).is_err());
        assert_eq!(report.aggregate.expect("aggregate report").rebuilt_links, 2);

        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    fn migration_is_a_dry_run_without_apply() {
        let root = temporary_directory("dry-run");
        let workspace = root.join("workspace");
        let corpus = workspace.join("data/crates_raw");
        let destination = root.join("external/crates_raw");
        fs::create_dir_all(&corpus).expect("create corpus directory");

        let report = migrate_legacy_corpus(&workspace, &corpus, &destination, false)
            .expect("inspect legacy corpus");

        assert!(corpus.exists());
        assert!(!destination.exists());
        assert!(report.aggregate.is_none());

        fs::remove_dir_all(root).expect("remove temporary directory");
    }
}
