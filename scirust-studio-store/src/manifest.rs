//! What is recorded alongside every stored run.

use serde::{Deserialize, Serialize};

/// The schema version of [`RunManifest`] itself.
///
/// Independent of the scenario schema version and of
/// `scirust_studio_runtime::RESULT_SCHEMA_VERSION`: the manifest describes
/// *storage*, and can gain fields without the result contract changing.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// How a stored run ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RunStatus {
    /// The run finished and produced a result.
    Completed,
    /// The run was cancelled before producing a result.
    Cancelled,
    /// The run failed.
    Failed {
        /// Why it failed, already formatted.
        message: String,
    },
}

impl RunStatus {
    /// A short, stable word for this status — for tables and filters.
    pub fn label(&self) -> &'static str {
        match self
        {
            RunStatus::Completed => "completed",
            RunStatus::Cancelled => "cancelled",
            RunStatus::Failed { .. } => "failed",
        }
    }

    /// Whether a `result.json` exists for a run in this state.
    pub fn has_result(&self) -> bool {
        matches!(self, RunStatus::Completed)
    }
}

/// The build and platform a run was produced on.
///
/// Deliberately limited to what the standard library can report truthfully
/// without a platform crate: the compilation target. It is **not** a
/// hardware fingerprint — there is no CPU model, microcode revision, or
/// core count here, because nothing in this workspace can obtain those
/// portably today, and a field that is right on Linux and invented on
/// Windows is worse than an absent one. What is here is enough to answer
/// the question determinism actually turns on: "was this the same target?"
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentFingerprint {
    /// `std::env::consts::OS`, e.g. `"linux"`, `"windows"`, `"macos"`.
    pub os: String,
    /// `std::env::consts::ARCH`, e.g. `"x86_64"`, `"aarch64"`.
    pub arch: String,
    /// `std::env::consts::FAMILY`, e.g. `"unix"`, `"windows"`.
    pub family: String,
    /// Pointer width in bits.
    pub pointer_width: u32,
}

impl EnvironmentFingerprint {
    /// The fingerprint of the currently-running build.
    pub fn current() -> Self {
        EnvironmentFingerprint {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            family: std::env::consts::FAMILY.to_string(),
            pointer_width: usize::BITS,
        }
    }

    /// Whether another fingerprint describes the same compilation target.
    pub fn same_target_as(&self, other: &EnvironmentFingerprint) -> bool {
        self.os == other.os && self.arch == other.arch && self.pointer_width == other.pointer_width
    }
}

/// Everything recorded about one stored run except the scenario and the
/// result themselves, which live beside it as their own files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    /// [`MANIFEST_SCHEMA_VERSION`] at the time this run was written.
    pub manifest_schema_version: u32,
    /// The run's identifier, matching its directory name.
    pub run_id: String,
    /// The capability that was run.
    pub capability_id: String,
    /// SHA-256 of the exact scenario bytes stored in `scenario.scirust.toml`,
    /// lowercase hex.
    pub scenario_sha256: String,
    /// SHA-256 of the exact bytes stored in `result.json`, lowercase hex.
    /// Absent for a run that produced no result.
    pub result_sha256: Option<String>,
    /// How the run ended.
    #[serde(flatten)]
    pub status: RunStatus,
    /// When the run started, RFC 3339 UTC.
    pub started_at_rfc3339: String,
    /// When the run settled, RFC 3339 UTC.
    pub finished_at_rfc3339: String,
    /// The build and target the run was produced on.
    pub environment: EnvironmentFingerprint,
    /// `scirust_studio_runtime::RESULT_SCHEMA_VERSION` at write time.
    pub result_schema_version: u32,
    /// The version of the crate that wrote this manifest.
    pub store_version: String,
}

impl RunManifest {
    /// Serialize to pretty-printed JSON.
    ///
    /// Serialization lives in the crate that owns the type — the same rule
    /// `scirust_studio_runtime::RunResult::to_json_pretty` follows — so a
    /// caller such as `scirust-cli` does not need its own `serde_json`
    /// dependency just to offer `--format json`.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Serialize a list of manifests to pretty-printed JSON.
pub fn manifests_to_json(manifests: &[RunManifest]) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(manifests)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_round_trip_through_json() {
        for status in [
            RunStatus::Completed,
            RunStatus::Cancelled,
            RunStatus::Failed {
                message: "boom".to_string(),
            },
        ]
        {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: RunStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
    }

    #[test]
    fn only_a_completed_run_has_a_result() {
        assert!(RunStatus::Completed.has_result());
        assert!(!RunStatus::Cancelled.has_result());
        assert!(
            !RunStatus::Failed {
                message: String::new()
            }
            .has_result()
        );
    }

    #[test]
    fn the_current_fingerprint_is_populated() {
        let fp = EnvironmentFingerprint::current();
        assert!(!fp.os.is_empty());
        assert!(!fp.arch.is_empty());
        assert!(fp.pointer_width >= 16);
        assert!(fp.same_target_as(&EnvironmentFingerprint::current()));
    }

    #[test]
    fn a_different_architecture_is_not_the_same_target() {
        let a = EnvironmentFingerprint::current();
        let mut b = a.clone();
        b.arch = "some-other-arch".to_string();
        assert!(!a.same_target_as(&b));
    }

    #[test]
    fn the_status_is_flattened_into_the_manifest_object() {
        let manifest = RunManifest {
            manifest_schema_version: MANIFEST_SCHEMA_VERSION,
            run_id: "r".to_string(),
            capability_id: "c".to_string(),
            scenario_sha256: "aa".to_string(),
            result_sha256: None,
            status: RunStatus::Cancelled,
            started_at_rfc3339: "2026-01-01T00:00:00Z".to_string(),
            finished_at_rfc3339: "2026-01-01T00:00:01Z".to_string(),
            environment: EnvironmentFingerprint::current(),
            result_schema_version: 1,
            store_version: "0.1.0".to_string(),
        };
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains(r#""status":"cancelled""#), "{json}");
        let decoded: RunManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, manifest);
    }
}
