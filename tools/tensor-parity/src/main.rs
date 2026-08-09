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
struct Verified {
    harness: String,
    fixtures: String,
    on: String,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    verified_impls: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    verified: Option<Verified>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reference_impl: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reference_verified: Option<Verified>,
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
    reference_parity: usize,
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
    reference_parity: usize,
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

fn validate_production_harness(root: &Path, op_name: &str, verified: &Verified) {
    let relative = Path::new(&verified.harness);

    if relative.is_absolute()
        || relative
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        panic!(
            "operator {} has unsafe production harness path {:?}",
            op_name, verified.harness
        );
    }

    let tests_root = Path::new("scirust-core").join("tests");
    if !relative.starts_with(&tests_root)
        || relative.extension().and_then(|v| v.to_str()) != Some("rs")
    {
        panic!(
            "operator {} production proof must use a Rust integration test under scirust-core/tests, got {:?}",
            op_name, verified.harness
        );
    }

    let harness_path = root.join(relative);
    let canonical = harness_path.canonicalize().unwrap_or_else(|error| {
        panic!(
            "operator {} production harness {} does not exist: {error}",
            op_name,
            harness_path.display()
        )
    });

    if !canonical.starts_with(root)
    {
        panic!(
            "operator {} production harness escapes repository root: {}",
            op_name,
            canonical.display()
        );
    }

    let source = fs::read_to_string(&canonical).unwrap_or_else(|error| {
        panic!(
            "operator {} cannot read production harness {}: {error}",
            op_name,
            canonical.display()
        )
    });

    // Ignore ordinary // comments so documentation such as
    // "does not use the reference parity module" does not trip the guard.
    // Executable Rust still must not import/call the independent reference
    // implementation.
    let active_source = source
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");

    for forbidden in [
        "scirust_core::tensor::parity",
        "::tensor::parity",
        "tensor::parity",
        "parity::",
    ]
    {
        if active_source.contains(forbidden)
        {
            panic!(
                "operator {} production harness {:?} references forbidden reference code token {:?}",
                op_name, verified.harness, forbidden
            );
        }
    }

    let fixture_relative = Path::new(&verified.fixtures);
    if fixture_relative.is_absolute()
        || fixture_relative
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        panic!(
            "operator {} has unsafe production fixture path {:?}",
            op_name, verified.fixtures
        );
    }

    let fixture_path = root.join(fixture_relative);
    if !fixture_path.exists()
    {
        panic!(
            "operator {} production fixture path does not exist: {}",
            op_name,
            fixture_path.display()
        );
    }

    // A proof that is merely present in the repository but never executed in
    // Tensor Parity CI is not accepted as production evidence.
    let test_name = relative
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_else(|| {
            panic!(
                "operator {} has invalid production harness filename {:?}",
                op_name, verified.harness
            )
        });

    let workflow_path = root.join(".github/workflows/tensor-parity-validation.yml");
    let workflow = fs::read_to_string(&workflow_path).unwrap_or_else(|error| {
        panic!(
            "cannot read Tensor Parity workflow {}: {error}",
            workflow_path.display()
        )
    });

    let invocation = format!("--test {test_name}");
    if !workflow.contains(&invocation)
    {
        panic!(
            "operator {} production harness {:?} is not explicitly executed by Tensor Parity CI ({:?} missing)",
            op_name, verified.harness, invocation
        );
    }
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
        reference_parity: 0,
        experimental: 0,
        missing: 0,
    };
    for op in &operators
    {
        if op.reference_verified.is_some() != op.reference_impl.is_some()
        {
            panic!(
                "operator {} must set reference_impl and reference_verified together",
                op.name
            );
        }

        if let Some(reference_impl) = op.reference_impl.as_deref()
        {
            if reference_impl != "scirust_core::tensor::parity"
            {
                panic!(
                    "operator {} has unsupported reference_impl {:?}",
                    op.name, reference_impl
                );
            }
        }

        let has_production_proof = !op.verified_impls.is_empty();

        if has_production_proof != op.verified.is_some()
        {
            panic!(
                "operator {} must set verified_impls and verified together",
                op.name
            );
        }

        {
            let mut seen = std::collections::BTreeSet::new();
            for verified_impl in &op.verified_impls
            {
                if !seen.insert(verified_impl)
                {
                    panic!(
                        "operator {} repeats verified implementation {:?}",
                        op.name, verified_impl
                    );
                }
            }
        }

        for verified_impl in &op.verified_impls
        {
            if !op.impls.contains(verified_impl)
            {
                panic!(
                    "operator {} verifies impl {:?} which is absent from impls {:?}",
                    op.name, verified_impl, op.impls
                );
            }
        }

        if let Some(verified) = &op.verified
        {
            if verified.harness == "scirust-core/tests/parity_differential.rs"
            {
                panic!(
                    "operator {} uses the reference harness as production proof",
                    op.name
                );
            }

            validate_production_harness(&root, &op.name, verified);
        }

        match op.status.as_str()
        {
            "parity" =>
            {
                if op.impls.is_empty() || !has_production_proof
                {
                    panic!(
                        "operator {} claims production parity without implementations/proof",
                        op.name
                    );
                }
                if op.verified_impls.len() != op.impls.len()
                    || op.impls.iter().any(|id| !op.verified_impls.contains(id))
                {
                    panic!(
                        "operator {} claims full production parity but verified_impls {:?} does not cover impls {:?}",
                        op.name, op.verified_impls, op.impls
                    );
                }
            },
            "experimental" =>
            {
                if op.impls.is_empty()
                {
                    panic!(
                        "operator {} is experimental but declares no implementation",
                        op.name
                    );
                }
                if has_production_proof
                    && op.verified_impls.len() == op.impls.len()
                    && op.impls.iter().all(|id| op.verified_impls.contains(id))
                {
                    panic!(
                        "operator {} has every implementation directly verified but remains experimental",
                        op.name
                    );
                }
            },
            "missing" =>
            {
                if !op.impls.is_empty() || has_production_proof
                {
                    panic!(
                        "operator {} is missing but declares implementations/proof",
                        op.name
                    );
                }
            },
            other =>
            {
                panic!("operator {} has invalid status {:?}", op.name, other);
            },
        }

        if op.reference_verified.is_some()
        {
            summary.reference_parity += 1;
        }

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
            reference_parity: 0,
            experimental: 0,
            missing: 0,
        });
        if op.reference_verified.is_some()
        {
            f.reference_parity += 1;
        }
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
    let mut csv = String::from(
        "operator,family,status,impls,verified_impls,reference_parity,reference_impl,autograd,dtypes,atol,rtol,device\n",
    );
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
            "{},{},{},{},{},{},{},{},{},{},{},{}\n",
            csv_escape(&op.name),
            csv_escape(&op.family),
            op.status,
            op.impls.join("|"),
            op.verified_impls.join("|"),
            op.reference_verified.is_some(),
            csv_escape(op.reference_impl.as_deref().unwrap_or("")),
            op.autograd,
            op.dtypes.join("|"),
            op.tolerance.atol,
            op.tolerance.rtol,
            devices.join("|"),
        ));
    }
    fs::write(&csv_path, csv).unwrap_or_else(|e| panic!("write {}: {e}", csv_path.display()));

    println!(
        "profile={} baseline={}@{} total={} production_parity={} reference_parity={} experimental={} missing={}",
        output.profile,
        output.baseline.version,
        &output.baseline.commit[..8.min(output.baseline.commit.len())],
        output.summary.total,
        output.summary.parity,
        output.summary.reference_parity,
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
