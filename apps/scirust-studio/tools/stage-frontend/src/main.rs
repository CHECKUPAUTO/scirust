//! Copy the Dioxus web build into `dist/`, the directory Tauri bundles.
//!
//! `tauri.conf.json` points `frontendDist` at `../dist`. The Dioxus CLI
//! writes its output somewhere under `target/dx/…`, and exactly where has
//! changed between CLI versions. Rather than hard-code a path that will
//! quietly stop matching — producing a bundle with a stale or empty frontend
//! and no error anywhere — this searches for the built bundle and verifies
//! it before staging it.
//!
//! Its contract is **"`dist/` holds a real bundle when this exits zero"**,
//! not "a copy just happened" — because it runs as Tauri's
//! `beforeBuildCommand`, and on a packaging machine the bundle arrives as a
//! CI artifact unpacked straight into `dist/`, with no `target/dx` anywhere.
//! An already-staged bundle is therefore verified and kept.
//!
//! What it refuses to do:
//!
//! - **Stage something that is not a frontend.** A directory without an
//!   `index.html` and a `.wasm` module is not a bundle; copying it would
//!   produce an application that opens a blank window.
//! - **Leave the previous build behind.** When there is a fresh build,
//!   `dist/` is emptied first, so a file the new build no longer produces
//!   cannot survive into the next bundle.
//! - **Invent a bundle.** Neither a build nor a staged bundle is a hard
//!   failure naming the command that produces one.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use sha2::{Digest, Sha256};

fn main() -> ExitCode {
    match run()
    {
        Ok(report) =>
        {
            println!("{report}");
            ExitCode::SUCCESS
        },
        Err(message) =>
        {
            eprintln!("stage-frontend: {message}");
            ExitCode::FAILURE
        },
    }
}

fn run() -> Result<String, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h")
    {
        return Ok(usage());
    }

    let profile = option(&args, "--profile").unwrap_or_else(|| "release".to_string());
    let app_root = app_root()?;
    let destination = app_root.join("dist");
    let dx_root = app_root.join("target").join("dx");

    let explicit = option(&args, "--source").map(PathBuf::from);
    match plan(explicit, &dx_root, &destination, &profile)?
    {
        Plan::KeepStaged => describe(&destination, None),
        Plan::Copy(source) =>
        {
            clear_dist(&destination)?;
            let copied = copy_tree(&source, &destination)?;
            describe(&destination, Some((&source, copied)))
        },
    }
}

/// What staging should do.
#[derive(Debug, PartialEq)]
enum Plan {
    /// Copy this bundle over `dist/`.
    Copy(PathBuf),
    /// `dist/` already holds a verified bundle; leave it alone.
    KeepStaged,
}

/// Decide what to do, or explain why nothing can be done.
///
/// The "keep staged" case is not a convenience. This runs as Tauri's
/// `beforeBuildCommand`, and on a packaging machine the bundle legitimately
/// arrives as a CI artifact unpacked straight into `dist/`: there is no
/// `target/dx` there and never will be. So the question this tool answers is
/// **"does `dist/` hold a real bundle"**, not "has a build just happened".
///
/// What it still refuses is the case that matters: neither a fresh build nor
/// an already-staged one. That is the path to an application that opens a
/// blank window, and it fails here with the command that fixes it.
fn plan(
    explicit: Option<PathBuf>,
    dx_root: &Path,
    destination: &Path,
    profile: &str,
) -> Result<Plan, String> {
    let source = match explicit
    {
        Some(path) => Some(path),
        None => find_bundle(dx_root, profile),
    };

    let Some(source) = source
    else
    {
        return match bundle_problems(destination).as_slice()
        {
            [] => Ok(Plan::KeepStaged),
            problems => Err(format!(
                "there is no Dioxus build under {}, and {} does not already \
                 hold a usable bundle:\n  - {}\n\
                 Build it first:\n    dx build --{profile} --platform web",
                dx_root.display(),
                destination.display(),
                problems.join("\n  - ")
            )),
        };
    };

    let problems = bundle_problems(&source);
    if problems.is_empty()
    {
        Ok(Plan::Copy(source))
    }
    else
    {
        Err(format!(
            "{} is not a usable web bundle:\n  - {}",
            source.display(),
            problems.join("\n  - ")
        ))
    }
}

/// Report what `dist/` now holds.
fn describe(destination: &Path, staged: Option<(&Path, usize)>) -> Result<String, String> {
    let index = destination.join("index.html");
    let digest = sha256_of(&index)?;
    let wasm_bytes: u64 = wasm_files(destination)
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum();

    let provenance = match staged
    {
        Some((source, copied)) =>
        {
            format!("source      {}\nfiles       {copied}\n", source.display())
        },
        None => "source      already staged (no local Dioxus build to copy)\n".to_string(),
    };

    Ok(format!(
        "{provenance}\
         destination {}\n\
         wasm        {wasm_bytes} bytes\n\
         index sha256 {digest}",
        destination.display()
    ))
}

fn usage() -> String {
    "usage: stage-frontend [--profile release|debug] [--source <dir>]\n\
     \n\
     Copies the Dioxus web build into apps/scirust-studio/dist, which is what\n\
     tauri.conf.json bundles. Refuses to stage a directory that is not a real\n\
     bundle. Never downloads anything."
        .to_string()
}

fn option(args: &[String], name: &str) -> Option<String> {
    let index = args.iter().position(|a| a == name)?;
    args.get(index + 1).cloned()
}

/// Why a directory is not a web bundle, empty when it is one.
///
/// Both conditions matter and fail differently: without `index.html` the
/// WebView has no document to load, and without a `.wasm` module it loads a
/// document that renders nothing. Reporting them separately means the
/// message says which half of the build went wrong.
fn bundle_problems(dir: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    if !dir.is_dir()
    {
        problems.push(format!("{} is not a directory", dir.display()));
        return problems;
    }
    if !dir.join("index.html").is_file()
    {
        problems.push("there is no index.html".to_string());
    }
    if wasm_files(dir).is_empty()
    {
        problems.push("there is no .wasm module anywhere inside it".to_string());
    }
    problems
}

/// Every `.wasm` file in a tree.
fn wasm_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    walk(dir, &mut |path| {
        if path.extension().is_some_and(|e| e == "wasm")
        {
            found.push(path.to_path_buf());
        }
    });
    found
}

/// Depth-first walk, ignoring anything unreadable.
fn walk(dir: &Path, visit: &mut impl FnMut(&Path)) {
    let Ok(entries) = std::fs::read_dir(dir)
    else
    {
        return;
    };
    for entry in entries.flatten()
    {
        let path = entry.path();
        if path.is_dir()
        {
            walk(&path, visit);
        }
        else
        {
            visit(&path);
        }
    }
}

/// Find the newest directory under `root` that looks like a web bundle for
/// this profile.
///
/// Searching rather than hard-coding, because the Dioxus CLI's output layout
/// is its own business and has changed across versions. The profile has to
/// appear in the path so a debug build is never staged into a release
/// bundle.
fn find_bundle(root: &Path, profile: &str) -> Option<PathBuf> {
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    collect_bundles(root, profile, &mut candidates);
    candidates.sort_by_key(|(time, _)| *time);
    candidates.pop().map(|(_, path)| path)
}

fn collect_bundles(dir: &Path, profile: &str, found: &mut Vec<(std::time::SystemTime, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir)
    else
    {
        return;
    };
    for entry in entries.flatten()
    {
        let path = entry.path();
        if !path.is_dir()
        {
            continue;
        }
        let index = path.join("index.html");
        let in_profile = path
            .components()
            .any(|c| c.as_os_str().to_string_lossy() == profile);
        if in_profile && index.is_file() && bundle_problems(&path).is_empty()
        {
            let time = std::fs::metadata(&index)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            found.push((time, path));
            // A bundle does not contain another bundle; do not descend.
            continue;
        }
        collect_bundles(&path, profile, found);
    }
}

/// Empty `dist/`, keeping the placeholder that keeps it in git.
fn clear_dist(dist: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dist).map_err(|e| format!("cannot create {}: {e}", dist.display()))?;
    let entries =
        std::fs::read_dir(dist).map_err(|e| format!("cannot read {}: {e}", dist.display()))?;
    for entry in entries.flatten()
    {
        let path = entry.path();
        if path.file_name().is_some_and(|n| n == ".gitkeep")
        {
            continue;
        }
        let result = if path.is_dir()
        {
            std::fs::remove_dir_all(&path)
        }
        else
        {
            std::fs::remove_file(&path)
        };
        result.map_err(|e| format!("cannot remove {}: {e}", path.display()))?;
    }
    Ok(())
}

/// Copy a tree, returning how many files were written.
fn copy_tree(source: &Path, destination: &Path) -> Result<usize, String> {
    std::fs::create_dir_all(destination)
        .map_err(|e| format!("cannot create {}: {e}", destination.display()))?;
    let mut count = 0;
    let entries =
        std::fs::read_dir(source).map_err(|e| format!("cannot read {}: {e}", source.display()))?;
    for entry in entries.flatten()
    {
        let from = entry.path();
        let Some(name) = from.file_name()
        else
        {
            continue;
        };
        let to = destination.join(name);
        if from.is_dir()
        {
            count += copy_tree(&from, &to)?;
        }
        else
        {
            std::fs::copy(&from, &to)
                .map_err(|e| format!("cannot copy {} to {}: {e}", from.display(), to.display()))?;
            count += 1;
        }
    }
    Ok(count)
}

/// This file lives at `apps/scirust-studio/tools/stage-frontend/src`, so the
/// application root is three levels up from the crate directory.
fn app_root() -> Result<PathBuf, String> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            format!(
                "cannot locate the application root from {}",
                manifest.display()
            )
        })
}

fn sha256_of(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest
    {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory that cleans up after itself.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let path = std::env::temp_dir().join(format!(
                "stage-frontend-{}-{name}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Scratch(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn write(&self, relative: &str, contents: &str) -> PathBuf {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, contents).unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The failure this tool exists to prevent: a directory that looks
    /// plausible but would bundle into a blank window.
    #[test]
    fn a_directory_without_a_wasm_module_is_refused() {
        let scratch = Scratch::new("no-wasm");
        scratch.write("public/index.html", "<html></html>");
        let problems = bundle_problems(&scratch.path().join("public"));
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains(".wasm"), "{problems:?}");
    }

    #[test]
    fn a_directory_without_an_index_is_refused() {
        let scratch = Scratch::new("no-index");
        scratch.write("public/assets/app.wasm", "\0asm");
        let problems = bundle_problems(&scratch.path().join("public"));
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("index.html"), "{problems:?}");
    }

    #[test]
    fn a_missing_directory_is_refused_by_name() {
        let problems = bundle_problems(Path::new("/definitely/not/here"));
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("/definitely/not/here"));
    }

    #[test]
    fn a_real_bundle_is_accepted() {
        let scratch = Scratch::new("real");
        scratch.write("public/index.html", "<html></html>");
        scratch.write("public/assets/app_bg.wasm", "\0asm");
        assert!(bundle_problems(&scratch.path().join("public")).is_empty());
    }

    /// The search must find the bundle wherever the CLI put it, and must not
    /// pick a debug build when a release one was asked for.
    #[test]
    fn the_search_finds_a_nested_bundle_for_the_right_profile() {
        let scratch = Scratch::new("search");
        scratch.write(
            "dx/scirust-studio-ui/release/web/public/index.html",
            "<html>",
        );
        scratch.write("dx/scirust-studio-ui/release/web/public/a.wasm", "\0asm");
        scratch.write("dx/scirust-studio-ui/debug/web/public/index.html", "<html>");
        scratch.write("dx/scirust-studio-ui/debug/web/public/a.wasm", "\0asm");

        let release = find_bundle(&scratch.path().join("dx"), "release").expect("a release bundle");
        assert!(release.ends_with("release/web/public"), "{release:?}");

        let debug = find_bundle(&scratch.path().join("dx"), "debug").expect("a debug bundle");
        assert!(debug.ends_with("debug/web/public"), "{debug:?}");
    }

    /// The packaging case: the bundle came from a CI artifact, so there is
    /// no `target/dx` and there never will be. Staging must accept it.
    #[test]
    fn an_already_staged_bundle_is_kept_when_there_is_no_build() {
        let scratch = Scratch::new("staged");
        scratch.write("dist/index.html", "<html>");
        scratch.write("dist/assets/app.wasm", "\0asm");
        let dx = scratch.path().join("target/dx");

        let decision = plan(None, &dx, &scratch.path().join("dist"), "release");
        assert_eq!(decision, Ok(Plan::KeepStaged));
    }

    /// ...but only if it is genuinely a bundle. A `dist/` holding an index
    /// and no module is the blank-window failure, and it must not pass.
    #[test]
    fn a_half_staged_bundle_is_refused_not_kept() {
        let scratch = Scratch::new("half-staged");
        scratch.write("dist/index.html", "<html>");
        let dx = scratch.path().join("target/dx");

        let error = plan(None, &dx, &scratch.path().join("dist"), "release")
            .expect_err("a bundle with no module must be refused");
        assert!(error.contains(".wasm"), "{error}");
        assert!(error.contains("dx build"), "{error}");
    }

    #[test]
    fn nothing_staged_and_nothing_built_names_the_command_that_fixes_it() {
        let scratch = Scratch::new("nothing");
        let error = plan(
            None,
            &scratch.path().join("target/dx"),
            &scratch.path().join("dist"),
            "release",
        )
        .expect_err("neither source is an error");
        assert!(
            error.contains("dx build --release --platform web"),
            "{error}"
        );
    }

    /// A fresh build wins over whatever happens to be staged, so a rebuild
    /// is never silently ignored.
    #[test]
    fn a_fresh_build_is_preferred_over_an_already_staged_bundle() {
        let scratch = Scratch::new("fresh");
        scratch.write("dist/index.html", "<html>old");
        scratch.write("dist/old.wasm", "\0asm");
        scratch.write("target/dx/app/release/web/public/index.html", "<html>new");
        scratch.write("target/dx/app/release/web/public/new.wasm", "\0asm");

        let decision = plan(
            None,
            &scratch.path().join("target/dx"),
            &scratch.path().join("dist"),
            "release",
        )
        .expect("a plan");
        match decision
        {
            Plan::Copy(source) => assert!(source.ends_with("release/web/public"), "{source:?}"),
            other => panic!("a fresh build must be staged, got {other:?}"),
        }
    }

    #[test]
    fn an_explicit_source_that_is_not_a_bundle_is_refused() {
        let scratch = Scratch::new("explicit");
        scratch.write("elsewhere/index.html", "<html>");
        let error = plan(
            Some(scratch.path().join("elsewhere")),
            &scratch.path().join("target/dx"),
            &scratch.path().join("dist"),
            "release",
        )
        .expect_err("an explicit source is still verified");
        assert!(error.contains(".wasm"), "{error}");
    }

    #[test]
    fn the_search_reports_nothing_when_there_is_no_build() {
        let scratch = Scratch::new("empty");
        assert!(find_bundle(scratch.path(), "release").is_none());
    }

    /// Staging must not leave a previous build's files behind: a file the
    /// new build no longer produces would otherwise still be bundled.
    #[test]
    fn staging_clears_the_destination_but_keeps_the_placeholder() {
        let scratch = Scratch::new("clear");
        let dist = scratch.path().join("dist");
        std::fs::create_dir_all(dist.join("old")).unwrap();
        std::fs::write(dist.join("old/stale.js"), "old").unwrap();
        std::fs::write(dist.join("stale.html"), "old").unwrap();
        std::fs::write(dist.join(".gitkeep"), "keep").unwrap();

        clear_dist(&dist).unwrap();

        assert!(dist.join(".gitkeep").is_file(), "the placeholder survives");
        assert!(!dist.join("stale.html").exists());
        assert!(!dist.join("old").exists());
    }

    #[test]
    fn copying_reproduces_the_whole_tree() {
        let scratch = Scratch::new("copy");
        scratch.write("src/index.html", "<html>");
        scratch.write("src/assets/app.wasm", "\0asm");
        scratch.write("src/assets/deep/style.css", "body{}");

        let destination = scratch.path().join("out");
        let count = copy_tree(&scratch.path().join("src"), &destination).unwrap();

        assert_eq!(count, 3);
        assert!(destination.join("index.html").is_file());
        assert!(destination.join("assets/app.wasm").is_file());
        assert_eq!(
            std::fs::read_to_string(destination.join("assets/deep/style.css")).unwrap(),
            "body{}"
        );
    }

    #[test]
    fn the_application_root_is_the_one_tauri_bundles_from() {
        let root = app_root().expect("a root");
        assert!(root.join("src-tauri").join("tauri.conf.json").is_file());
        assert!(root.join("Dioxus.toml").is_file());
    }

    #[test]
    fn hashing_a_known_input_is_stable() {
        let scratch = Scratch::new("hash");
        let path = scratch.write("index.html", "abc");
        assert_eq!(
            sha256_of(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
