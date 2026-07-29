//! The on-disk run store.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_studio_runtime::{
    RESULT_SCHEMA_VERSION, RESULT_SCHEMA_VERSION_V1, RunResult, RunResultV1, XAxisMeaning,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::StoreError;
use crate::manifest::{EnvironmentFingerprint, MANIFEST_SCHEMA_VERSION, RunManifest, RunStatus};

/// The subdirectory of the store root that holds runs.
const RUNS_DIR: &str = "runs";
/// Prefix marking a run directory that has not been finalized yet.
///
/// A directory still carrying this prefix means the process writing it did
/// not finish — see [`RunStore::interrupted`].
const PENDING_PREFIX: &str = ".partial-";

const MANIFEST_FILE: &str = "manifest.json";
const SCENARIO_FILE: &str = "scenario.scirust.toml";
const RESULT_FILE: &str = "result.json";

/// Lowercase-hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest
    {
        use std::fmt::Write;
        // Writing into a String cannot fail; the Result is discarded
        // deliberately rather than unwrapped to keep this infallible.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// An immutable store of completed runs.
///
/// Nothing here can modify a finalized run: there is no update, no
/// overwrite, and no "re-save" operation. A run is assembled in a
/// [`PendingRun`] under a `.partial-` directory and becomes visible only
/// when that directory is renamed into place — a single atomic step on
/// every platform this targets. A reader therefore never observes a
/// half-written run, and a writer that dies leaves evidence rather than
/// corruption.
#[derive(Debug, Clone)]
pub struct RunStore {
    root: PathBuf,
}

impl RunStore {
    /// Open (creating if needed) a store rooted at `root`.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let root = root.into();
        let runs = root.join(RUNS_DIR);
        fs::create_dir_all(&runs).map_err(|e| StoreError::io(&runs, e))?;
        Ok(RunStore { root })
    }

    /// The store's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn runs_dir(&self) -> PathBuf {
        self.root.join(RUNS_DIR)
    }

    /// The directory a finalized run lives in.
    pub fn run_dir(&self, run_id: &str) -> PathBuf {
        self.runs_dir().join(run_id)
    }

    /// Start recording a run.
    ///
    /// The scenario text is written immediately, so a run that is later
    /// interrupted still records exactly what was being attempted.
    pub fn begin(
        &self,
        capability_id: &str,
        scenario_toml: &str,
    ) -> Result<PendingRun, StoreError> {
        let started_at = chrono::Utc::now();
        let scenario_sha256 = sha256_hex(scenario_toml.as_bytes());
        let run_id = self.allocate_run_id(&started_at, &scenario_sha256)?;

        let pending_dir = self.runs_dir().join(format!("{PENDING_PREFIX}{run_id}"));
        fs::create_dir(&pending_dir).map_err(|e| StoreError::io(&pending_dir, e))?;

        let scenario_path = pending_dir.join(SCENARIO_FILE);
        fs::write(&scenario_path, scenario_toml).map_err(|e| StoreError::io(&scenario_path, e))?;

        Ok(PendingRun {
            store: self.clone(),
            pending_dir,
            run_id,
            capability_id: capability_id.to_string(),
            scenario_sha256,
            started_at_rfc3339: started_at.to_rfc3339(),
        })
    }

    /// Build a run id that is time-sortable, content-tied, and unused.
    ///
    /// The timestamp puts runs in chronological order in any directory
    /// listing; the scenario-hash prefix makes it obvious at a glance when
    /// two runs came from identical input. Neither is trusted for
    /// uniqueness on its own — the same scenario really can be launched
    /// twice inside one second — so a numeric suffix is appended until the
    /// name is free.
    fn allocate_run_id(
        &self,
        started_at: &chrono::DateTime<chrono::Utc>,
        scenario_sha256: &str,
    ) -> Result<String, StoreError> {
        let stamp = started_at.format("%Y%m%dT%H%M%SZ");
        let short_hash = &scenario_sha256[..16];
        let base = format!("{stamp}-{short_hash}");

        for attempt in 0..1000
        {
            let candidate = if attempt == 0
            {
                base.clone()
            }
            else
            {
                format!("{base}-{attempt}")
            };
            let final_dir = self.runs_dir().join(&candidate);
            let pending_dir = self.runs_dir().join(format!("{PENDING_PREFIX}{candidate}"));
            if !final_dir.exists() && !pending_dir.exists()
            {
                return Ok(candidate);
            }
        }
        Err(StoreError::RunIdExhausted { base })
    }

    /// Every finalized run's manifest, oldest first.
    ///
    /// Run ids begin with a fixed-width UTC timestamp, so sorting the
    /// directory names sorts by start time.
    pub fn list(&self) -> Result<Vec<RunManifest>, StoreError> {
        let mut ids = self.finalized_run_ids()?;
        ids.sort();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids
        {
            out.push(self.load_manifest(&id)?);
        }
        Ok(out)
    }

    fn finalized_run_ids(&self) -> Result<Vec<String>, StoreError> {
        let dir = self.runs_dir();
        let entries = fs::read_dir(&dir).map_err(|e| StoreError::io(&dir, e))?;
        let mut ids = Vec::new();
        for entry in entries
        {
            let entry = entry.map_err(|e| StoreError::io(&dir, e))?;
            if !entry.path().is_dir()
            {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with(PENDING_PREFIX)
            {
                ids.push(name);
            }
        }
        Ok(ids)
    }

    /// Load one run's manifest.
    pub fn load_manifest(&self, run_id: &str) -> Result<RunManifest, StoreError> {
        let path = self.run_dir(run_id).join(MANIFEST_FILE);
        let text = fs::read_to_string(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound
            {
                StoreError::NoSuchRun {
                    run_id: run_id.to_string(),
                }
            }
            else
            {
                StoreError::io(&path, e)
            }
        })?;
        serde_json::from_str(&text).map_err(|e| StoreError::Corrupt {
            path: path.clone(),
            reason: e.to_string(),
        })
    }

    /// Load the exact scenario text a run was given.
    pub fn load_scenario(&self, run_id: &str) -> Result<String, StoreError> {
        let path = self.run_dir(run_id).join(SCENARIO_FILE);
        fs::read_to_string(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound
            {
                StoreError::NoSuchRun {
                    run_id: run_id.to_string(),
                }
            }
            else
            {
                StoreError::io(&path, e)
            }
        })
    }

    /// Load a run's result.
    ///
    /// Fails with [`StoreError::NoResult`] for a run that was cancelled or
    /// failed — those genuinely have no result, and returning an empty one
    /// would be a lie the caller could not detect.
    pub fn load_result(&self, run_id: &str) -> Result<LoadedRunResult, StoreError> {
        let manifest = self.load_manifest(run_id)?;
        if !manifest.status.has_result()
        {
            return Err(StoreError::NoResult {
                run_id: run_id.to_string(),
                status: manifest.status.label().to_string(),
            });
        }
        let path = self.run_dir(run_id).join(RESULT_FILE);
        let text = fs::read_to_string(&path).map_err(|e| StoreError::io(&path, e))?;

        // Dispatch on the version recorded *inside the file*, not on the
        // manifest's copy. A manifest and its result are written together,
        // but the file is what is actually being decoded, and trusting the
        // other one would turn a corrupted pair into a silent mis-parse.
        let probe: VersionProbe = serde_json::from_str(&text).map_err(|e| StoreError::Corrupt {
            path: path.clone(),
            reason: e.to_string(),
        })?;

        let corrupt = |e: serde_json::Error| StoreError::Corrupt {
            path: path.clone(),
            reason: e.to_string(),
        };
        match probe.schema_version
        {
            RESULT_SCHEMA_VERSION_V1 => serde_json::from_str(&text)
                .map(LoadedRunResult::V1)
                .map_err(corrupt),
            v if v == RESULT_SCHEMA_VERSION => serde_json::from_str(&text)
                .map(LoadedRunResult::V2)
                .map_err(corrupt),
            other => Err(StoreError::UnsupportedResultSchema {
                run_id: run_id.to_string(),
                found: other,
                supported: vec![RESULT_SCHEMA_VERSION_V1, RESULT_SCHEMA_VERSION],
            }),
        }
    }

    /// Load a run's result, requiring the current schema version.
    ///
    /// For callers that genuinely cannot work with a legacy result — a chart
    /// that must plot against physical coordinates, for instance. Returns
    /// [`StoreError::UnsupportedResultSchema`] for a v1 run rather than
    /// fabricating the coordinates v1 does not contain.
    pub fn load_result_v2(&self, run_id: &str) -> Result<RunResult, StoreError> {
        match self.load_result(run_id)?
        {
            LoadedRunResult::V2(result) => Ok(*result),
            LoadedRunResult::V1(_) => Err(StoreError::UnsupportedResultSchema {
                run_id: run_id.to_string(),
                found: RESULT_SCHEMA_VERSION_V1,
                supported: vec![RESULT_SCHEMA_VERSION],
            }),
        }
    }

    /// Recompute a run's stored hashes and check them against its manifest.
    ///
    /// This is what makes "immutable" checkable rather than merely intended:
    /// a run whose scenario or result was edited after the fact fails here,
    /// naming the file that no longer matches.
    pub fn verify(&self, run_id: &str) -> Result<(), StoreError> {
        let manifest = self.load_manifest(run_id)?;

        let scenario = self.load_scenario(run_id)?;
        let actual = sha256_hex(scenario.as_bytes());
        if actual != manifest.scenario_sha256
        {
            return Err(StoreError::HashMismatch {
                run_id: run_id.to_string(),
                file: SCENARIO_FILE.to_string(),
                expected: manifest.scenario_sha256.clone(),
                actual,
            });
        }

        if let Some(expected) = &manifest.result_sha256
        {
            let path = self.run_dir(run_id).join(RESULT_FILE);
            let bytes = fs::read(&path).map_err(|e| StoreError::io(&path, e))?;
            let actual = sha256_hex(&bytes);
            if actual != *expected
            {
                return Err(StoreError::HashMismatch {
                    run_id: run_id.to_string(),
                    file: RESULT_FILE.to_string(),
                    expected: expected.clone(),
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Runs whose writer never finished — a crash, a kill, or a power loss
    /// between [`RunStore::begin`] and the finalizing step.
    ///
    /// These are reported, never silently deleted: a leftover directory is
    /// the only evidence that a run was attempted at all, and deciding what
    /// to do with it belongs to whoever is looking at the store.
    pub fn interrupted(&self) -> Result<Vec<InterruptedRun>, StoreError> {
        let dir = self.runs_dir();
        let entries = fs::read_dir(&dir).map_err(|e| StoreError::io(&dir, e))?;
        let mut out = Vec::new();
        for entry in entries
        {
            let entry = entry.map_err(|e| StoreError::io(&dir, e))?;
            let path = entry.path();
            if !path.is_dir()
            {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(run_id) = name.strip_prefix(PENDING_PREFIX)
            else
            {
                continue;
            };
            let scenario = fs::read_to_string(path.join(SCENARIO_FILE)).ok();
            out.push(InterruptedRun {
                run_id: run_id.to_string(),
                path: path.clone(),
                scenario_toml: scenario,
            });
        }
        out.sort_by(|a, b| a.run_id.cmp(&b.run_id));
        Ok(out)
    }

    /// Delete one interrupted run's leftover directory.
    ///
    /// Refuses to touch anything that is not a `.partial-` directory, so
    /// this can never remove a finalized run.
    pub fn discard_interrupted(&self, run_id: &str) -> Result<(), StoreError> {
        let path = self.runs_dir().join(format!("{PENDING_PREFIX}{run_id}"));
        if !path.is_dir()
        {
            return Err(StoreError::NoSuchRun {
                run_id: run_id.to_string(),
            });
        }
        fs::remove_dir_all(&path).map_err(|e| StoreError::io(&path, e))
    }
}

/// A run being recorded. Becomes visible in the store only when one of the
/// finalizing methods succeeds.
#[derive(Debug)]
pub struct PendingRun {
    store: RunStore,
    pending_dir: PathBuf,
    run_id: String,
    capability_id: String,
    scenario_sha256: String,
    started_at_rfc3339: String,
}

impl PendingRun {
    /// The id this run will have once finalized.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// The directory currently holding the partially-written run.
    pub fn pending_dir(&self) -> &Path {
        &self.pending_dir
    }

    /// Record a successful run and make it visible.
    pub fn complete(self, result: &RunResult) -> Result<String, StoreError> {
        let json =
            serde_json::to_string_pretty(result).map_err(|e| StoreError::Encode(e.to_string()))?;
        let path = self.pending_dir.join(RESULT_FILE);
        fs::write(&path, &json).map_err(|e| StoreError::io(&path, e))?;
        let result_sha256 = sha256_hex(json.as_bytes());
        self.finalize(RunStatus::Completed, Some(result_sha256))
    }

    /// Record that the run was cancelled, and make that record visible.
    pub fn cancel(self) -> Result<String, StoreError> {
        self.finalize(RunStatus::Cancelled, None)
    }

    /// Record that the run failed, and make that record visible.
    pub fn fail(self, message: &str) -> Result<String, StoreError> {
        self.finalize(
            RunStatus::Failed {
                message: message.to_string(),
            },
            None,
        )
    }

    fn finalize(
        self,
        status: RunStatus,
        result_sha256: Option<String>,
    ) -> Result<String, StoreError> {
        let manifest = RunManifest {
            manifest_schema_version: MANIFEST_SCHEMA_VERSION,
            run_id: self.run_id.clone(),
            capability_id: self.capability_id.clone(),
            scenario_sha256: self.scenario_sha256.clone(),
            result_sha256,
            status,
            started_at_rfc3339: self.started_at_rfc3339.clone(),
            finished_at_rfc3339: chrono::Utc::now().to_rfc3339(),
            environment: EnvironmentFingerprint::current(),
            result_schema_version: RESULT_SCHEMA_VERSION,
            store_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| StoreError::Encode(e.to_string()))?;
        let manifest_path = self.pending_dir.join(MANIFEST_FILE);
        fs::write(&manifest_path, json).map_err(|e| StoreError::io(&manifest_path, e))?;

        // The single step that publishes the run. Everything above wrote
        // into a directory no reader looks in; nothing below can leave a
        // partially-visible run behind.
        let final_dir = self.store.run_dir(&self.run_id);
        fs::rename(&self.pending_dir, &final_dir).map_err(|e| StoreError::io(&final_dir, e))?;
        Ok(self.run_id)
    }
}

/// A run whose writing process never finished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterruptedRun {
    /// The id the run would have had.
    pub run_id: String,
    /// The leftover directory.
    pub path: PathBuf,
    /// The scenario that was being attempted, if it had been written.
    pub scenario_toml: Option<String>,
}

/// Just enough of a stored result to learn which schema it uses.
#[derive(Deserialize)]
struct VersionProbe {
    schema_version: u32,
}

/// A stored result, tagged with the schema version it was written under.
///
/// Existing stores predate result schema v2, and a run store is immutable —
/// so upgrading a stored file in place is not an option: it would invalidate
/// the hash recorded for it and destroy exactly the property the store
/// exists to provide. Legacy runs are therefore read as what they are, and
/// callers are made to acknowledge the difference rather than having a
/// plausible-looking time axis invented for them.
///
/// Boxed variants: a `RunResult` carrying long time-courses is far larger
/// than the enum's discriminant, and an unboxed variant would make every
/// value of this type that size.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadedRunResult {
    /// A result stored before axes carried their coordinates.
    V1(Box<RunResultV1>),
    /// A current result, with exact axis coordinates.
    V2(Box<RunResult>),
}

impl LoadedRunResult {
    /// The schema version this result was stored under.
    pub fn schema_version(&self) -> u32 {
        match self
        {
            LoadedRunResult::V1(_) => RESULT_SCHEMA_VERSION_V1,
            LoadedRunResult::V2(_) => RESULT_SCHEMA_VERSION,
        }
    }

    /// What a consumer may truthfully claim the horizontal axis represents.
    ///
    /// The whole point of routing this through one method: a caller asks the
    /// same question of either version and cannot forget that v1 has no
    /// coordinates.
    pub fn x_axis_meaning(&self) -> XAxisMeaning {
        match self
        {
            LoadedRunResult::V1(r) => r.x_axis_meaning(),
            LoadedRunResult::V2(_) => XAxisMeaning::PhysicalCoordinates,
        }
    }

    /// The current-schema result, if this is one.
    pub fn as_v2(&self) -> Option<&RunResult> {
        match self
        {
            LoadedRunResult::V2(r) => Some(r),
            LoadedRunResult::V1(_) => None,
        }
    }

    /// The legacy result, if this is one.
    pub fn as_v1(&self) -> Option<&RunResultV1> {
        match self
        {
            LoadedRunResult::V1(r) => Some(r),
            LoadedRunResult::V2(_) => None,
        }
    }

    /// The capability that produced this result, whichever version it is.
    pub fn capability_id(&self) -> &str {
        match self
        {
            LoadedRunResult::V1(r) => &r.capability_id,
            LoadedRunResult::V2(r) => &r.capability_id,
        }
    }

    /// The run summary, whichever version it is.
    pub fn summary(&self) -> &scirust_studio_runtime::RunSummary {
        match self
        {
            LoadedRunResult::V1(r) => &r.summary,
            LoadedRunResult::V2(r) => &r.summary,
        }
    }

    /// The metrics, which both versions record identically.
    pub fn metrics(&self) -> &[scirust_studio_runtime::Metric] {
        match self
        {
            LoadedRunResult::V1(r) => &r.metrics,
            LoadedRunResult::V2(r) => &r.metrics,
        }
    }

    /// The scientific verification checks, which both versions record
    /// identically.
    pub fn verifications(&self) -> &[scirust_studio_runtime::VerificationResult] {
        match self
        {
            LoadedRunResult::V1(r) => &r.verifications,
            LoadedRunResult::V2(r) => &r.verifications,
        }
    }

    /// The warnings, which both versions record identically.
    pub fn warnings(&self) -> &[scirust_studio_runtime::RunWarning] {
        match self
        {
            LoadedRunResult::V1(r) => &r.warnings,
            LoadedRunResult::V2(r) => &r.warnings,
        }
    }

    /// Provenance, which both versions record identically.
    pub fn provenance(&self) -> &scirust_studio_runtime::RunProvenance {
        match self
        {
            LoadedRunResult::V1(r) => &r.provenance,
            LoadedRunResult::V2(r) => &r.provenance,
        }
    }

    /// How many series this result carries.
    pub fn series_len(&self) -> usize {
        match self
        {
            LoadedRunResult::V1(r) => r.series.len(),
            LoadedRunResult::V2(r) => r.series.len(),
        }
    }
}
