//! Deterministic ingestion of RSI engineering trajectories into SciAgent examples.
//!
//! P7.2 keeps the RSI v3 trajectory wire format as the immutable provenance
//! source while producing a SciAgent-native dataset view. Split assignment is
//! grouped by frozen `task_spec_id`, so all trajectories for one held-out task
//! stay out of training. Deduplication happens before splitting and rejected
//! trajectories remain first-class negative examples.

use crate::sha256::sha256_hex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fmt;

pub const RSI_ENGINEERING_TRAJECTORY_SCHEMA_VERSION: u64 = 3;
pub const SCIAGENT_ENGINEERING_DATASET_MANIFEST_VERSION: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineeringDatasetSplit {
    Train,
    Eval,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineeringIngestConfig {
    pub split_salt: String,
    pub split_modulus: u32,
    pub eval_buckets: u32,
}

impl Default for EngineeringIngestConfig {
    fn default() -> Self {
        Self {
            split_salt: "scirust-sciagent-engineering-v1".to_string(),
            split_modulus: 10,
            eval_buckets: 2,
        }
    }
}

impl EngineeringIngestConfig {
    pub fn validate(&self) -> Result<(), EngineeringTrajectoryIngestError> {
        if self.split_salt.trim().is_empty() || self.split_salt.trim() != self.split_salt {
            return Err(EngineeringTrajectoryIngestError::InvalidConfig(
                "split_salt must be non-empty and canonical".to_string(),
            ));
        }
        if self.split_modulus < 2 {
            return Err(EngineeringTrajectoryIngestError::InvalidConfig(
                "split_modulus must be at least 2".to_string(),
            ));
        }
        if self.eval_buckets == 0 || self.eval_buckets >= self.split_modulus {
            return Err(EngineeringTrajectoryIngestError::InvalidConfig(
                "eval_buckets must be in 1..split_modulus".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringDatasetExample {
    pub example_id: String,
    pub semantic_state_id: String,
    pub source_sha256: String,
    pub task_spec_id: String,
    pub parent_state_id: String,
    pub patch_set_identity: String,
    pub split: EngineeringDatasetSplit,
    pub accepted: bool,
    pub prompt: Value,
    pub target: Value,
    pub provenance: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineeringDatasetManifest {
    pub manifest_version: u64,
    pub source_schema_version: u64,
    pub split_salt: String,
    pub split_modulus: u32,
    pub eval_buckets: u32,
    pub source_records: usize,
    pub unique_examples: usize,
    pub duplicate_records: usize,
    pub train_examples: usize,
    pub eval_examples: usize,
    pub rejected_examples: usize,
    pub dataset_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringDataset {
    pub train: Vec<EngineeringDatasetExample>,
    pub eval: Vec<EngineeringDatasetExample>,
    pub manifest: EngineeringDatasetManifest,
}

impl EngineeringDataset {
    pub fn ingest_json_records<I, S>(
        records: I,
        config: EngineeringIngestConfig,
    ) -> Result<Self, EngineeringTrajectoryIngestError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        config.validate()?;
        let mut by_semantic_state: BTreeMap<String, (String, EngineeringDatasetExample)> =
            BTreeMap::new();
        let mut source_records = 0usize;
        let mut duplicate_records = 0usize;

        for record in records {
            source_records = source_records.checked_add(1).ok_or_else(|| {
                EngineeringTrajectoryIngestError::InvalidRecord(
                    "source record count overflow".to_string(),
                )
            })?;
            let root: Value = serde_json::from_str(record.as_ref())
                .map_err(EngineeringTrajectoryIngestError::Json)?;
            let example = example_from_trajectory(root, &config)?;
            match by_semantic_state.get(&example.semantic_state_id) {
                Some((existing_source, _)) if existing_source == &example.source_sha256 => {
                    duplicate_records = duplicate_records.checked_add(1).ok_or_else(|| {
                        EngineeringTrajectoryIngestError::InvalidRecord(
                            "duplicate count overflow".to_string(),
                        )
                    })?;
                }
                Some((existing_source, _)) => {
                    return Err(EngineeringTrajectoryIngestError::ConflictingDuplicate {
                        semantic_state_id: example.semantic_state_id,
                        first_source_sha256: existing_source.clone(),
                        second_source_sha256: example.source_sha256,
                    });
                }
                None => {
                    by_semantic_state.insert(
                        example.semantic_state_id.clone(),
                        (example.source_sha256.clone(), example),
                    );
                }
            }
        }

        if source_records == 0 {
            return Err(EngineeringTrajectoryIngestError::EmptyDataset);
        }

        let mut train = Vec::new();
        let mut eval = Vec::new();
        let mut rejected_examples = 0usize;
        for (_, (_, example)) in by_semantic_state {
            if !example.accepted {
                rejected_examples += 1;
            }
            match example.split {
                EngineeringDatasetSplit::Train => train.push(example),
                EngineeringDatasetSplit::Eval => eval.push(example),
            }
        }
        train.sort_by(|left, right| left.example_id.cmp(&right.example_id));
        eval.sort_by(|left, right| left.example_id.cmp(&right.example_id));

        let dataset_sha256 = dataset_fingerprint(&train, &eval)?;
        let manifest = EngineeringDatasetManifest {
            manifest_version: SCIAGENT_ENGINEERING_DATASET_MANIFEST_VERSION,
            source_schema_version: RSI_ENGINEERING_TRAJECTORY_SCHEMA_VERSION,
            split_salt: config.split_salt,
            split_modulus: config.split_modulus,
            eval_buckets: config.eval_buckets,
            source_records,
            unique_examples: train.len() + eval.len(),
            duplicate_records,
            train_examples: train.len(),
            eval_examples: eval.len(),
            rejected_examples,
            dataset_sha256,
        };

        Ok(Self {
            train,
            eval,
            manifest,
        })
    }

    pub fn manifest_json(&self) -> Result<String, EngineeringTrajectoryIngestError> {
        serde_json::to_string(&self.manifest).map_err(EngineeringTrajectoryIngestError::Json)
    }
}

#[derive(Debug)]
pub enum EngineeringTrajectoryIngestError {
    Json(serde_json::Error),
    EmptyDataset,
    InvalidConfig(String),
    InvalidRecord(String),
    UnsupportedSchema(u64),
    ConflictingDuplicate {
        semantic_state_id: String,
        first_source_sha256: String,
        second_source_sha256: String,
    },
}

impl fmt::Display for EngineeringTrajectoryIngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "engineering trajectory JSON: {error}"),
            Self::EmptyDataset => write!(f, "engineering trajectory dataset is empty"),
            Self::InvalidConfig(message) => write!(f, "engineering ingest config: {message}"),
            Self::InvalidRecord(message) => write!(f, "engineering trajectory record: {message}"),
            Self::UnsupportedSchema(version) => {
                write!(f, "unsupported engineering trajectory schema version: {version}")
            }
            Self::ConflictingDuplicate {
                semantic_state_id,
                first_source_sha256,
                second_source_sha256,
            } => write!(
                f,
                "conflicting duplicate {semantic_state_id}: {first_source_sha256} != {second_source_sha256}"
            ),
        }
    }
}

impl std::error::Error for EngineeringTrajectoryIngestError {}

fn example_from_trajectory(
    root: Value,
    config: &EngineeringIngestConfig,
) -> Result<EngineeringDatasetExample, EngineeringTrajectoryIngestError> {
    let object = root.as_object().ok_or_else(|| {
        EngineeringTrajectoryIngestError::InvalidRecord("root must be a JSON object".to_string())
    })?;
    let schema_version = required_u64(object, "schema_version")?;
    if schema_version != RSI_ENGINEERING_TRAJECTORY_SCHEMA_VERSION {
        return Err(EngineeringTrajectoryIngestError::UnsupportedSchema(
            schema_version,
        ));
    }

    let task_spec_id = required_identity(object, "task_spec_id")?;
    let parent_state_id = required_identity(object, "parent_state_id")?;
    let patch_set = required_object_value(object, "patch_set")?;
    let patch_set_object = patch_set.as_object().ok_or_else(|| {
        EngineeringTrajectoryIngestError::InvalidRecord("patch_set must be an object".to_string())
    })?;
    let patch_set_identity = required_hex_identity(patch_set_object, "identity", &[64])?;
    let operations = patch_set_object
        .get("operations")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            EngineeringTrajectoryIngestError::InvalidRecord(
                "patch_set.operations must be an array".to_string(),
            )
        })?;
    if operations.is_empty() {
        return Err(EngineeringTrajectoryIngestError::InvalidRecord(
            "patch_set.operations must not be empty".to_string(),
        ));
    }

    let compatibility = required_object_value(object, "compatibility")?;
    let proposer = required_object_value(object, "proposer")?;
    let admissibility = required_object_value(object, "admissibility")?;
    let evidence = required_array_value(object, "compiler_test_device_evidence")?;
    if evidence.as_array().is_some_and(Vec::is_empty) {
        return Err(EngineeringTrajectoryIngestError::InvalidRecord(
            "compiler_test_device_evidence must not be empty".to_string(),
        ));
    }
    let benchmarks = required_array_value(object, "benchmarks")?;
    let later_verdicts = required_array_value(object, "later_verdicts")?;
    let verdict = required_string(object, "verdict")?;
    let accepted = match verdict.as_str() {
        "accepted" => true,
        "rejected" => false,
        other => {
            return Err(EngineeringTrajectoryIngestError::InvalidRecord(format!(
                "invalid verdict: {other}"
            )));
        }
    };
    let verdict_reason = required_string(object, "verdict_reason")?;
    if verdict_reason.trim().is_empty() {
        return Err(EngineeringTrajectoryIngestError::InvalidRecord(
            "verdict_reason must not be empty".to_string(),
        ));
    }
    if accepted && !all_hard_gates_pass(admissibility.as_object().ok_or_else(|| {
        EngineeringTrajectoryIngestError::InvalidRecord(
            "admissibility must be an object".to_string(),
        )
    })?)? {
        return Err(EngineeringTrajectoryIngestError::InvalidRecord(
            "accepted trajectory has a non-pass hard gate".to_string(),
        ));
    }

    let canonical_source = canonical_json(&root)?;
    let source_sha256 = sha256_hex(canonical_source.as_bytes());
    let semantic_state_id = sha256_hex(
        format!("{task_spec_id}\n{parent_state_id}\n{patch_set_identity}").as_bytes(),
    );
    let example_id = sha256_hex(format!("{semantic_state_id}\n{source_sha256}").as_bytes());
    let split = split_for_task(&task_spec_id, config);

    // Prompt contains only immutable problem/context/provenance inputs. Outcome,
    // PatchSet and evidence are supervision targets and therefore cannot leak
    // from held-out tasks into training through this representation.
    let prompt = json!({
        "schema_version": schema_version,
        "task_spec_id": task_spec_id,
        "parent_state_id": parent_state_id,
        "compatibility": compatibility,
        "proposer": proposer,
    });
    let target = json!({
        "patch_set": patch_set,
        "compiler_test_device_evidence": evidence,
        "admissibility": admissibility,
        "benchmarks": benchmarks,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "later_verdicts": later_verdicts,
    });
    let provenance = json!({
        "source_schema_version": schema_version,
        "source_sha256": source_sha256,
        "semantic_state_id": semantic_state_id,
        "task_spec_id": task_spec_id,
        "parent_state_id": parent_state_id,
        "patch_set_identity": patch_set_identity,
    });

    Ok(EngineeringDatasetExample {
        example_id,
        semantic_state_id,
        source_sha256,
        task_spec_id,
        parent_state_id,
        patch_set_identity,
        split,
        accepted,
        prompt,
        target,
        provenance,
    })
}

fn split_for_task(task_spec_id: &str, config: &EngineeringIngestConfig) -> EngineeringDatasetSplit {
    let digest = crate::sha256::sha256(format!("{}\n{task_spec_id}", config.split_salt).as_bytes());
    let bucket = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]])
        % config.split_modulus;
    if bucket < config.eval_buckets {
        EngineeringDatasetSplit::Eval
    } else {
        EngineeringDatasetSplit::Train
    }
}

fn dataset_fingerprint(
    train: &[EngineeringDatasetExample],
    eval: &[EngineeringDatasetExample],
) -> Result<String, EngineeringTrajectoryIngestError> {
    let payload = json!({
        "manifest_version": SCIAGENT_ENGINEERING_DATASET_MANIFEST_VERSION,
        "source_schema_version": RSI_ENGINEERING_TRAJECTORY_SCHEMA_VERSION,
        "train": train,
        "eval": eval,
    });
    let canonical = canonical_json(&payload)?;
    Ok(sha256_hex(canonical.as_bytes()))
}

fn canonical_json(value: &Value) -> Result<String, EngineeringTrajectoryIngestError> {
    let canonical = canonical_value(value);
    serde_json::to_string(&canonical).map_err(EngineeringTrajectoryIngestError::Json)
}

fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let sorted: BTreeMap<&String, &Value> = object.iter().collect();
            let mut canonical = Map::new();
            for (key, value) in sorted {
                canonical.insert(key.clone(), canonical_value(value));
            }
            Value::Object(canonical)
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical_value).collect()),
        _ => value.clone(),
    }
}

fn all_hard_gates_pass(
    admissibility: &Map<String, Value>,
) -> Result<bool, EngineeringTrajectoryIngestError> {
    const GATES: [&str; 7] = [
        "build",
        "required_tests",
        "numerical_parity",
        "provenance",
        "deterministic_contract",
        "resource_budget",
        "policy_checks",
    ];
    for gate in GATES {
        let status = required_string(admissibility, gate)?;
        if !matches!(status.as_str(), "pass" | "fail" | "unknown") {
            return Err(EngineeringTrajectoryIngestError::InvalidRecord(format!(
                "invalid hard-gate status for {gate}: {status}"
            )));
        }
        if status != "pass" {
            return Ok(false);
        }
    }
    Ok(true)
}

fn required_string(
    object: &Map<String, Value>,
    field: &str,
) -> Result<String, EngineeringTrajectoryIngestError> {
    let value = object.get(field).and_then(Value::as_str).ok_or_else(|| {
        EngineeringTrajectoryIngestError::InvalidRecord(format!("missing string field {field}"))
    })?;
    if value.trim().is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(EngineeringTrajectoryIngestError::InvalidRecord(format!(
            "invalid string field {field}"
        )));
    }
    Ok(value.to_string())
}

fn required_identity(
    object: &Map<String, Value>,
    field: &str,
) -> Result<String, EngineeringTrajectoryIngestError> {
    required_hex_identity(object, field, &[40, 64])
}

fn required_hex_identity(
    object: &Map<String, Value>,
    field: &str,
    lengths: &[usize],
) -> Result<String, EngineeringTrajectoryIngestError> {
    let value = required_string(object, field)?;
    if !lengths.contains(&value.len()) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(EngineeringTrajectoryIngestError::InvalidRecord(format!(
            "invalid immutable identity field {field}"
        )));
    }
    Ok(value)
}

fn required_u64(
    object: &Map<String, Value>,
    field: &str,
) -> Result<u64, EngineeringTrajectoryIngestError> {
    object.get(field).and_then(Value::as_u64).ok_or_else(|| {
        EngineeringTrajectoryIngestError::InvalidRecord(format!("missing integer field {field}"))
    })
}

fn required_object_value(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Value, EngineeringTrajectoryIngestError> {
    let value = object.get(field).cloned().ok_or_else(|| {
        EngineeringTrajectoryIngestError::InvalidRecord(format!("missing object field {field}"))
    })?;
    if !value.is_object() {
        return Err(EngineeringTrajectoryIngestError::InvalidRecord(format!(
            "field {field} must be an object"
        )));
    }
    Ok(value)
}

fn required_array_value(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Value, EngineeringTrajectoryIngestError> {
    let value = object.get(field).cloned().ok_or_else(|| {
        EngineeringTrajectoryIngestError::InvalidRecord(format!("missing array field {field}"))
    })?;
    if !value.is_array() {
        return Err(EngineeringTrajectoryIngestError::InvalidRecord(format!(
            "field {field} must be an array"
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trajectory(task_byte: char, parent_byte: char, patch_byte: char, accepted: bool) -> String {
        let task = task_byte.to_string().repeat(64);
        let parent = parent_byte.to_string().repeat(64);
        let patch = patch_byte.to_string().repeat(64);
        let gate = if accepted { "pass" } else { "fail" };
        json!({
            "schema_version": 3,
            "task_spec_id": task,
            "compatibility": {
                "revisions": [{"repository":"Memorithm/scirust","revision":"8c051f664fb82465215569d94fa640bf8d0328f3","role":"target"}],
                "toolchain":"1.89.0",
                "feature_contract":"default"
            },
            "parent_state_id": parent,
            "patch_set": {
                "identity": patch,
                "operations": [{"kind":"create","path":"src/new.rs","content":"pub fn x() {}"}]
            },
            "proposer": {"provider":"fixture","model":"fixture-v1","configuration_id":"deterministic"},
            "compiler_test_device_evidence": ["cargo test: pass"],
            "admissibility": {
                "build": gate,
                "required_tests": gate,
                "numerical_parity": gate,
                "provenance": gate,
                "deterministic_contract": gate,
                "resource_budget": gate,
                "policy_checks": gate
            },
            "benchmarks": [],
            "verdict": if accepted { "accepted" } else { "rejected" },
            "verdict_reason": if accepted { "all frozen gates passed" } else { "frozen gate failed" },
            "later_verdicts": []
        })
        .to_string()
    }

    #[test]
    fn same_task_never_crosses_train_eval_boundary() {
        let config = EngineeringIngestConfig::default();
        let first = trajectory('a', 'b', 'c', true);
        let second = trajectory('a', 'd', 'e', false);
        let dataset = EngineeringDataset::ingest_json_records([first, second], config).unwrap();
        assert!(dataset.train.is_empty() || dataset.eval.is_empty());
        assert_eq!(dataset.manifest.unique_examples, 2);
        assert_eq!(dataset.manifest.rejected_examples, 1);
    }

    #[test]
    fn duplicate_record_is_counted_once() {
        let record = trajectory('a', 'b', 'c', true);
        let dataset = EngineeringDataset::ingest_json_records(
            [record.clone(), record],
            EngineeringIngestConfig::default(),
        )
        .unwrap();
        assert_eq!(dataset.manifest.source_records, 2);
        assert_eq!(dataset.manifest.unique_examples, 1);
        assert_eq!(dataset.manifest.duplicate_records, 1);
    }

    #[test]
    fn conflicting_semantic_duplicate_fails_closed() {
        let first = trajectory('a', 'b', 'c', true);
        let mut second: Value = serde_json::from_str(&first).unwrap();
        second["verdict_reason"] = Value::String("different reviewed outcome".to_string());
        let error = EngineeringDataset::ingest_json_records(
            [first, second.to_string()],
            EngineeringIngestConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            EngineeringTrajectoryIngestError::ConflictingDuplicate { .. }
        ));
    }

    #[test]
    fn accepted_record_with_failed_gate_is_rejected() {
        let mut record: Value = serde_json::from_str(&trajectory('a', 'b', 'c', true)).unwrap();
        record["admissibility"]["numerical_parity"] = Value::String("fail".to_string());
        let error = EngineeringDataset::ingest_json_records(
            [record.to_string()],
            EngineeringIngestConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            EngineeringTrajectoryIngestError::InvalidRecord(_)
        ));
    }

    #[test]
    fn manifest_and_split_are_byte_stable() {
        let records = [
            trajectory('a', 'b', 'c', true),
            trajectory('d', 'e', 'f', false),
        ];
        let first = EngineeringDataset::ingest_json_records(
            records.clone(),
            EngineeringIngestConfig::default(),
        )
        .unwrap();
        let second = EngineeringDataset::ingest_json_records(
            records.into_iter().rev(),
            EngineeringIngestConfig::default(),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.manifest_json().unwrap(), second.manifest_json().unwrap());
    }

    #[test]
    fn prompt_excludes_supervision_fields() {
        let dataset = EngineeringDataset::ingest_json_records(
            [trajectory('a', 'b', 'c', false)],
            EngineeringIngestConfig::default(),
        )
        .unwrap();
        let example = dataset
            .train
            .first()
            .or_else(|| dataset.eval.first())
            .unwrap();
        assert!(example.prompt.get("patch_set").is_none());
        assert!(example.prompt.get("verdict").is_none());
        assert!(example.prompt.get("compiler_test_device_evidence").is_none());
        assert_eq!(example.target["verdict"], "rejected");
    }
}
