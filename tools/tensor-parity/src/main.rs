use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Serialize)]
struct Baseline {
    version: String,
    commit: String,
    collected: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct Tolerance {
    atol: f64,
    rtol: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct Operator {
    name: String,
    torch: String,
    family: String,
    #[serde(default)]
    impls: Vec<String>,
    #[serde(default)]
    dtypes: Vec<String>,
    #[serde(default)]
    autograd: bool,
    tolerance: Tolerance,
    status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    notes: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Registry {
    profile: String,
    baseline: Baseline,
    #[serde(rename = "operator")]
    operators: Vec<Operator>,
}

#[derive(Debug, Serialize)]
struct Summary {
    total: usize,
    parity: usize,
    experimental: usize,
    missing: usize,
}

#[derive(Debug, Serialize)]
struct CoverageOutput {
    profile: String,
    baseline: Baseline,
    generated: bool,
    summary: Summary,
    families: BTreeMap<String, FamilySummary>,
    operators: Vec<Operator>,
}

#[derive(Debug, Serialize)]
struct FamilySummary {
    total: usize,
    parity: usize,
    experimental: usize,
    missing: usize,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("cannot canonicalize repo root from CARGO_MANIFEST_DIR")
}

fn main() {
    let root = repo_root();
    let registry_path = root.join("tensor-operators.toml");
    let registry: Registry = {
        let txt = fs::read_to_string(&registry_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", registry_path.display()));
        toml::from_str(&txt)
            .unwrap_or_else(|e| panic!("invalid TOML {}: {e}", registry_path.display()))
    };

    let mut operators: Vec<Operator> = registry.operators;
    operators.sort_by(|a, b| a.name.cmp(&b.name));

    let mut families: BTreeMap<String, FamilySummary> = BTreeMap::new();
    let mut summary = Summary {
        total: 0,
        parity: 0,
        experimental: 0,
        missing: 0,
    };
    for op in &operators
    {
        summary.total += 1;
        match op.status.as_str()
        {
            "parity" => summary.parity += 1,
            "experimental" => summary.experimental += 1,
            "missing" => summary.missing += 1,
            other => panic!("invalid status {other:?} for operator {}", op.name),
        }
        let f = families.entry(op.family.clone()).or_insert(FamilySummary {
            total: 0,
            parity: 0,
            experimental: 0,
            missing: 0,
        });
        f.total += 1;
        match op.status.as_str()
        {
            "parity" => f.parity += 1,
            "experimental" => f.experimental += 1,
            _ => f.missing += 1,
        }
    }

    let output = CoverageOutput {
        profile: registry.profile.clone(),
        baseline: Baseline {
            version: registry.baseline.version.clone(),
            commit: registry.baseline.commit.clone(),
            collected: registry.baseline.collected.clone(),
        },
        generated: true,
        summary,
        families,
        operators,
    };

    let artifacts = root.join("artifacts");
    fs::create_dir_all(&artifacts).expect("cannot create artifacts dir");
    let json_path = artifacts.join("tensor-coverage.json");
    let json_txt = serde_json::to_string_pretty(&output).expect("serialize coverage json") + "\n";
    fs::write(&json_path, json_txt)
        .unwrap_or_else(|e| panic!("write {}: {e}", json_path.display()));

    let csv_path = artifacts.join("tensor-coverage.csv");
    let mut csv = String::from("operator,family,status,impls,autograd,dtypes,atol,rtol,device\n");
    for op in &output.operators
    {
        let mut devices: Vec<&str> = Vec::new();
        if op.impls.iter().any(|i| i != "gpu" && i != "cuda")
        {
            devices.push("cpu");
        }
        if op.impls.iter().any(|i| i == "gpu")
        {
            devices.push("gpu");
        }
        if op.impls.iter().any(|i| i == "cuda")
        {
            devices.push("cuda");
        }
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            csv_escape(&op.name),
            csv_escape(&op.family),
            op.status,
            op.impls.join("|"),
            op.autograd,
            op.dtypes.join("|"),
            op.tolerance.atol,
            op.tolerance.rtol,
            devices.join("|"),
        ));
    }
    fs::write(&csv_path, csv).unwrap_or_else(|e| panic!("write {}: {e}", csv_path.display()));

    println!(
        "profile={} baseline={}@{} total={} parity={} experimental={} missing={}",
        output.profile,
        output.baseline.version,
        &output.baseline.commit[..8.min(output.baseline.commit.len())],
        output.summary.total,
        output.summary.parity,
        output.summary.experimental,
        output.summary.missing
    );
    println!("wrote {}", json_path.display());
    println!("wrote {}", csv_path.display());
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n')
    {
        format!("\"{}\"", s.replace('"', "\"\""))
    }
    else
    {
        s.to_string()
    }
}

#[allow(dead_code)]
fn _path_exists(_p: &Path) -> bool {
    false
}
