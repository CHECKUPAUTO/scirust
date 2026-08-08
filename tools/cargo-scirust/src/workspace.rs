use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{AppError, AppResult};

#[derive(Debug, Clone)]
pub struct PackageInfo {
    pub id: String,
    pub name: String,
    pub dir: PathBuf,
    pub features: BTreeMap<String, Vec<String>>,
}

#[derive(Debug)]
pub struct Workspace {
    pub root: PathBuf,
    pub packages: Vec<PackageInfo>,
    reverse_deps: HashMap<String, BTreeSet<String>>,
}

#[derive(Debug, Clone)]
pub struct Impact {
    pub changed_files: Vec<PathBuf>,
    pub direct: Vec<String>,
    pub affected: Vec<String>,
    pub global_change: bool,
    pub base: String,
    pub head: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    workspace_root: String,
    workspace_members: Vec<String>,
    packages: Vec<CargoPackage>,
}

#[derive(Debug, Deserialize)]
struct CargoPackage {
    id: String,
    name: String,
    manifest_path: String,
    dependencies: Vec<CargoDependency>,
    #[serde(default)]
    features: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct CargoDependency {
    path: Option<String>,
}

impl Workspace {
    pub fn load() -> AppResult<Self> {
        let cwd = env::current_dir().map_err(AppError::io)?;
        let first = metadata_from(&cwd, None)?;

        // When invoked while developing this standalone tool, the nearest
        // workspace is tools/cargo-scirust itself. Detect that case and locate
        // the real SciRust workspace above it.
        let metadata = if first.packages.len() == 1 && first.packages[0].name == "cargo-scirust" {
            let root_manifest = find_scirust_manifest(&cwd).ok_or_else(|| {
                AppError::message(
                    "could not locate the SciRust root Cargo.toml from this directory",
                )
            })?;
            metadata_from(&cwd, Some(&root_manifest))?
        } else {
            first
        };

        Self::from_metadata(metadata)
    }

    fn from_metadata(metadata: CargoMetadata) -> AppResult<Self> {
        let root = normalize_existing(Path::new(&metadata.workspace_root));
        let workspace_ids: BTreeSet<&str> = metadata
            .workspace_members
            .iter()
            .map(String::as_str)
            .collect();

        let mut packages = Vec::new();
        for package in &metadata.packages {
            if !workspace_ids.contains(package.id.as_str()) {
                continue;
            }
            let manifest = normalize_existing(Path::new(&package.manifest_path));
            let dir = manifest
                .parent()
                .ok_or_else(|| {
                    AppError::message("Cargo metadata returned a manifest without a parent")
                })?
                .to_path_buf();
            packages.push(PackageInfo {
                id: package.id.clone(),
                name: package.name.clone(),
                dir,
                features: package.features.clone(),
            });
        }

        packages.sort_by(|a, b| a.name.cmp(&b.name));

        let mut dir_to_id = HashMap::new();
        for package in &packages {
            dir_to_id.insert(package.dir.clone(), package.id.clone());
        }

        let mut reverse_deps: HashMap<String, BTreeSet<String>> = HashMap::new();
        for package in &metadata.packages {
            if !workspace_ids.contains(package.id.as_str()) {
                continue;
            }
            for dependency in &package.dependencies {
                let Some(path) = dependency.path.as_deref() else {
                    continue;
                };
                let dep_dir = normalize_existing(Path::new(path));
                if let Some(dep_id) = dir_to_id.get(&dep_dir) {
                    reverse_deps
                        .entry(dep_id.clone())
                        .or_default()
                        .insert(package.id.clone());
                }
            }
        }

        Ok(Self {
            root,
            packages,
            reverse_deps,
        })
    }

    pub fn package(&self, name: &str) -> Option<&PackageInfo> {
        self.packages.iter().find(|package| package.name == name)
    }

    pub fn package_names(&self) -> Vec<String> {
        self.packages.iter().map(|p| p.name.clone()).collect()
    }

    pub fn impact(&self, base: Option<&str>, head: Option<&str>) -> AppResult<Impact> {
        let base = match base {
            Some(value) => value.to_string(),
            None => self.infer_base()?,
        };

        let changed_files = self.changed_files(&base, head)?;
        let global_change = changed_files.iter().any(|path| is_global_path(path));

        if global_change {
            let names = self.package_names();
            return Ok(Impact {
                changed_files,
                direct: names.clone(),
                affected: names,
                global_change: true,
                base,
                head: head.map(str::to_owned),
            });
        }

        let mut direct_ids = BTreeSet::new();
        for changed in &changed_files {
            if let Some(package) = self.owner_of(changed) {
                direct_ids.insert(package.id.clone());
            }
        }

        let mut affected_ids = direct_ids.clone();
        let mut queue: VecDeque<String> = direct_ids.iter().cloned().collect();
        while let Some(id) = queue.pop_front() {
            if let Some(dependents) = self.reverse_deps.get(&id) {
                for dependent in dependents {
                    if affected_ids.insert(dependent.clone()) {
                        queue.push_back(dependent.clone());
                    }
                }
            }
        }

        let mut direct = self.names_for_ids(&direct_ids);
        let mut affected = self.names_for_ids(&affected_ids);
        direct.sort();
        affected.sort();

        Ok(Impact {
            changed_files,
            direct,
            affected,
            global_change: false,
            base,
            head: head.map(str::to_owned),
        })
    }

    fn names_for_ids(&self, ids: &BTreeSet<String>) -> Vec<String> {
        self.packages
            .iter()
            .filter(|package| ids.contains(&package.id))
            .map(|package| package.name.clone())
            .collect()
    }

    fn owner_of(&self, changed: &Path) -> Option<&PackageInfo> {
        let absolute = self.root.join(changed);
        self.packages
            .iter()
            .filter(|package| absolute.starts_with(&package.dir))
            .filter(|package| package.dir != self.root || root_package_owns(changed))
            .max_by_key(|package| package.dir.components().count())
    }

    fn infer_base(&self) -> AppResult<String> {
        if let Ok(base_ref) = env::var("GITHUB_BASE_REF") {
            if !base_ref.trim().is_empty() {
                for candidate in [format!("origin/{base_ref}"), base_ref] {
                    if let Some(base) = self.git_merge_base(&candidate) {
                        return Ok(base);
                    }
                }
            }
        }

        for candidate in ["origin/master", "master", "origin/main", "main"] {
            if let Some(base) = self.git_merge_base(candidate) {
                return Ok(base);
            }
        }

        if self.git_ref_exists("HEAD^") {
            Ok("HEAD^".to_string())
        } else {
            Ok("HEAD".to_string())
        }
    }

    fn git_merge_base(&self, reference: &str) -> Option<String> {
        if !self.git_ref_exists(reference) {
            return None;
        }
        let output = Command::new("git")
            .current_dir(&self.root)
            .args(["merge-base", "HEAD", reference])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let base = String::from_utf8_lossy(&output.stdout).trim().to_string();
        (!base.is_empty()).then_some(base)
    }

    fn git_ref_exists(&self, reference: &str) -> bool {
        Command::new("git")
            .current_dir(&self.root)
            .args(["rev-parse", "--verify", "--quiet", reference])
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn changed_files(&self, base: &str, head: Option<&str>) -> AppResult<Vec<PathBuf>> {
        let range = head
            .map(|head| format!("{base}...{head}"))
            .unwrap_or_else(|| base.to_string());
        let output = Command::new("git")
            .current_dir(&self.root)
            .args(["diff", "--name-only", "--relative", &range])
            .output()
            .map_err(AppError::io)?;
        if !output.status.success() {
            return Err(AppError::command(
                "git diff",
                output.status.code(),
                &output.stderr,
            ));
        }

        let mut paths = BTreeSet::new();
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if !line.trim().is_empty() {
                paths.insert(PathBuf::from(line));
            }
        }

        if head.is_none() {
            let output = Command::new("git")
                .current_dir(&self.root)
                .args(["ls-files", "--others", "--exclude-standard"])
                .output()
                .map_err(AppError::io)?;
            if !output.status.success() {
                return Err(AppError::command(
                    "git ls-files",
                    output.status.code(),
                    &output.stderr,
                ));
            }
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if !line.trim().is_empty() {
                    paths.insert(PathBuf::from(line));
                }
            }
        }

        Ok(paths.into_iter().collect())
    }
}

fn metadata_from(cwd: &Path, manifest: Option<&Path>) -> AppResult<CargoMetadata> {
    let mut command = Command::new("cargo");
    command
        .current_dir(cwd)
        .args(["metadata", "--format-version", "1", "--no-deps"]);
    if let Some(manifest) = manifest {
        command.arg("--manifest-path").arg(manifest);
    }
    let output = command.output().map_err(AppError::io)?;
    if !output.status.success() {
        return Err(AppError::command(
            "cargo metadata",
            output.status.code(),
            &output.stderr,
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|err| AppError::message(format!("invalid cargo metadata JSON: {err}")))
}

fn find_scirust_manifest(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let manifest = ancestor.join("Cargo.toml");
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        if text.contains("\"scirust-core\"") && text.contains("[workspace]") {
            return Some(manifest);
        }
    }
    None
}

fn normalize_existing(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn root_package_owns(path: &Path) -> bool {
    let mut components = path.components();
    match components
        .next()
        .and_then(|component| component.as_os_str().to_str())
    {
        Some("src" | "tests" | "benches" | "examples") => true,
        _ => matches!(path.file_name().and_then(OsStr::to_str), Some("build.rs")),
    }
}

fn is_global_path(path: &Path) -> bool {
    let text = path.to_string_lossy().replace('\\', "/");
    matches!(
        text.as_str(),
        "Cargo.toml"
            | "Cargo.lock"
            | "rust-toolchain"
            | "rust-toolchain.toml"
            | ".cargo/config"
            | ".cargo/config.toml"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_files_are_detected_conservatively() {
        assert!(is_global_path(Path::new("Cargo.toml")));
        assert!(is_global_path(Path::new("Cargo.lock")));
        assert!(is_global_path(Path::new(".cargo/config.toml")));
        assert!(!is_global_path(Path::new("docs/guide.md")));
    }

    #[test]
    fn root_package_ownership_does_not_swallow_nested_crates() {
        assert!(root_package_owns(Path::new("src/lib.rs")));
        assert!(root_package_owns(Path::new("tests/api.rs")));
        assert!(!root_package_owns(Path::new("scirust-core/src/lib.rs")));
        assert!(!root_package_owns(Path::new("README.md")));
    }
}
