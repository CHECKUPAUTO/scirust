//! [`Manifest`] — the authorable front end to a [`Plan`].
//!
//! Everything else in this crate takes a `Plan` as given. Something has to
//! *produce* one, and a `Plan` is not a thing a person writes by hand: it is a
//! validated DAG of [`Stage`]s, each carrying content addresses. RFC-0002 §08
//! §1 names this the scheduler's other half —
//! `resolve(&manifest, &graph) -> PlanGraph` — and this module is that
//! resolution, minus the graph half (see "What this deliberately does not do").
//!
//! A study manifest is TOML:
//!
//! ```toml
//! [study]
//! name = "decay-sweep"
//! seed = 42                        # every stage is reproducible; see below
//!
//! [[stage]]
//! id      = "coarse"
//! plugin  = "ode-rk4"
//! version = "1.0.0"
//! pin     = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
//! config  = "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
//!
//! [[stage]]
//! id      = "fine"
//! plugin  = "ode-rk4"
//! version = "1.0.0"
//! pin     = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
//! config  = "fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9"
//! consumes = ["coarse"]            # run after `coarse` *and* read its outputs
//! seed     = 7                     # overrides the study seed for this stage
//! ```
//!
//! `needs` is the ordering-only sibling of `consumes`: use it when a stage
//! must run after another without reading what it produced, so a study that
//! only wants sequencing does not acquire provenance parents it never read.
//!
//! ## Why it names things by content address
//!
//! `pin` is the plugin's content hash and `config` is the configuration's
//! content address, written out rather than resolved from a friendly name.
//! That is not an ergonomic oversight — it is the same pinning
//! [`Dispatch`](crate::Dispatch) enforces at bind time, moved to where a human
//! can see it. A manifest that named its plugin only by version would silently
//! start running different code the day the registry held a different build;
//! written this way, the drift is visible in the diff of the study file, and
//! [`Dispatch`] refuses the run if the two disagree. Lockfiles work the same
//! way for the same reason.
//!
//! ## Three properties this resolution guarantees
//!
//! * **Pure and total.** Resolution reads nothing but the text — no clock, no
//!   environment, no filesystem, no registry. The same manifest therefore
//!   produces the same [`Plan`], hence the same
//!   [`CacheKey`](crate::CacheKey)s, on every machine. A resolver that
//!   consulted the environment would silently break memoization across a team.
//! * **Unknown keys are errors.** A misspelled `seeed = 7` does not quietly
//!   leave the stage on the study seed and memoize the result as if it were
//!   the requested one — it fails. Silently ignoring input that was meant to
//!   change the computation is the same failure mode content addressing exists
//!   to prevent.
//! * **Graph validation is not duplicated.** Duplicate stage ids, dependencies
//!   on stages that do not exist, and cycles are [`Plan::new`]'s job and stay
//!   there; this module surfaces them as [`ManifestError::Plan`] rather than
//!   re-implementing (and eventually disagreeing with) those checks.
//!
//! ## Seeds are required, not defaulted to zero
//!
//! [`Stage::seed`] is documented as mandatory — *"a stage you cannot reseed is
//! not reproducible"*. So `[study] seed` is required rather than defaulting:
//! an author who has not thought about the seed should be told, not handed a
//! silent `0` that then travels into every cache key as though it had been
//! chosen. A stage may override it; if it does not, it inherits the study's.
//!
//! ## What this deliberately does not do
//!
//! * **No symbolic input resolution.** RFC-0002's signature takes the
//!   knowledge `graph` too, so a stage could name its inputs by *what they
//!   are* ("the claims contradicting H₁") rather than by object id. That needs
//!   a query language this crate does not have and should not invent alone;
//!   inputs here are content addresses. The manifest is honest about being a
//!   lockfile-shaped artifact, not a query.
//! * **No plugin discovery.** Resolution does not consult a
//!   [`Registry`](sos_registry::Registry) — it produces a `Plan`, and binding
//!   that plan to code stays [`Dispatch`]'s job (Invariant VIII). A manifest
//!   naming a plugin nobody registered resolves fine and fails at run time,
//!   with the error naming the stage.
//! * **No scheduling policy.** Order comes from [`Plan::schedule`], which is
//!   already deterministic. A manifest cannot request a different order.

use serde::Deserialize;
use sos_core::{Digest, ObjectId, SemVer};
use thiserror::Error;

use crate::descriptor::StageDescriptor;
use crate::error::WorkflowError;
use crate::plan::{Plan, Stage, StageId};

/// Errors from parsing or resolving a study manifest.
///
/// Every variant names the stage it came from where one applies: a study file
/// with twenty stages is unhelpful to debug from a bare "invalid version".
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ManifestError {
    /// The text is not valid TOML, or does not have the shape a manifest
    /// needs (missing key, wrong type, unknown key, malformed digest or
    /// object id — the content-addressed fields validate on the way in).
    #[error("manifest is not a valid study: {0}")]
    Toml(String),
    /// A stage's `version` is not `major.minor.patch`.
    #[error("stage {stage}: `{version}` is not a major.minor.patch version")]
    InvalidVersion {
        /// The offending stage's id.
        stage: String,
        /// The version text that failed to parse.
        version: String,
    },
    /// The manifest declares no stages. An empty plan is almost certainly a
    /// mistake — an empty `[[stage]]` list runs nothing and reports success.
    #[error("study `{0}` declares no stages")]
    NoStages(String),
    /// The stages parsed, but do not form a valid DAG (duplicate ids, a
    /// dependency on a stage that does not exist, or a cycle).
    #[error("manifest does not resolve to a valid plan: {0}")]
    Plan(#[from] WorkflowError),
}

/// A parsed study manifest: the study's own metadata plus its stages.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Study-level metadata.
    pub study: Study,
    /// The stages, in declaration order. Order does not affect the resolved
    /// plan — [`Plan::schedule`] is deterministic by dependency and id — but
    /// it is preserved so errors can name a position an author recognizes.
    #[serde(default, rename = "stage")]
    pub stages: Vec<StageSpec>,
}

/// Study-level metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Study {
    /// A human-readable name for the study. Recorded for humans; it does not
    /// enter any cache key, so renaming a study never invalidates its results.
    pub name: String,
    /// The default seed for stages that do not set their own. Required —
    /// see the module docs on why this is not defaulted to zero.
    pub seed: u64,
}

/// One stage as written in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageSpec {
    /// The stage's id, unique within the study.
    pub id: String,
    /// The plugin name this stage invokes.
    pub plugin: String,
    /// The plugin version, as `major.minor.patch`.
    pub version: String,
    /// The plugin's content hash — the exact build this study was written
    /// against. [`Dispatch`](crate::Dispatch) refuses the run if the registry
    /// now holds something else under the same name and version.
    pub pin: Digest,
    /// The content address of this stage's configuration.
    pub config: Digest,
    /// Object ids this stage consumes. Defaults to none.
    #[serde(default)]
    pub inputs: Vec<ObjectId>,
    /// Stage ids that must run before this one. Defaults to none.
    ///
    /// **Ordering only.** `needs` says "run after", not "read the output of";
    /// for dataflow use [`consumes`](Self::consumes), which implies ordering
    /// as well. Keeping the two separate means a study that genuinely only
    /// wants sequencing does not acquire provenance parents it never read.
    #[serde(default)]
    pub needs: Vec<String>,
    /// Stage ids whose **outputs this stage reads**. Defaults to none.
    ///
    /// A consumed stage's output object ids become this stage's inputs, which
    /// is the only way a manifest can express dataflow: `inputs` takes literal
    /// object ids, and an upstream stage's ids do not exist until it has run.
    /// Consuming implies `needs`, so an id here need not be repeated there.
    #[serde(default)]
    pub consumes: Vec<String>,
    /// This stage's seed. Defaults to the study's.
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Manifest {
    /// Parse a study manifest from TOML text.
    ///
    /// # Errors
    /// [`ManifestError::Toml`] if the text is not valid TOML or does not match
    /// the manifest shape — including a malformed `pin`/`config` digest or
    /// `inputs` entry, which are validated by their own parsers on the way in.
    pub fn parse(text: &str) -> Result<Self, ManifestError> {
        toml::from_str(text).map_err(|e| ManifestError::Toml(e.to_string()))
    }

    /// Resolve this manifest into a validated [`Plan`].
    ///
    /// # Errors
    /// [`ManifestError::InvalidVersion`] for a malformed plugin version,
    /// [`ManifestError::NoStages`] for an empty study, and
    /// [`ManifestError::Plan`] if the stages do not form a DAG.
    pub fn resolve(&self) -> Result<Plan, ManifestError> {
        if self.stages.is_empty()
        {
            return Err(ManifestError::NoStages(self.study.name.clone()));
        }
        let stages = self
            .stages
            .iter()
            .map(|s| self.resolve_stage(s))
            .collect::<Result<Vec<Stage>, ManifestError>>()?;
        Ok(Plan::new(stages)?)
    }

    fn resolve_stage(&self, spec: &StageSpec) -> Result<Stage, ManifestError> {
        let version =
            spec.version
                .parse::<SemVer>()
                .map_err(|_| ManifestError::InvalidVersion {
                    stage: spec.id.clone(),
                    version: spec.version.clone(),
                })?;
        Ok(Stage::consuming(
            StageId::new(spec.id.clone()),
            StageDescriptor::new(spec.plugin.clone(), version, spec.pin),
            spec.inputs.clone(),
            spec.config,
            spec.seed.unwrap_or(self.study.seed),
            spec.needs.iter().cloned().map(StageId::new).collect(),
            spec.consumes.iter().cloned().map(StageId::new).collect(),
        ))
    }
}

/// Parse and resolve a study manifest in one step.
///
/// # Errors
/// See [`Manifest::parse`] and [`Manifest::resolve`].
pub fn resolve_manifest(text: &str) -> Result<Plan, ManifestError> {
    Manifest::parse(text)?.resolve()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIN: &str = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
    const CFG_A: &str = "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae";
    const CFG_B: &str = "fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9";

    fn two_stage_manifest() -> String {
        format!(
            r#"
[study]
name = "decay-sweep"
seed = 42

[[stage]]
id      = "coarse"
plugin  = "ode-rk4"
version = "1.0.0"
pin     = "{PIN}"
config  = "{CFG_A}"

[[stage]]
id      = "fine"
plugin  = "ode-rk4"
version = "1.2.3"
pin     = "{PIN}"
config  = "{CFG_B}"
needs   = ["coarse"]
seed    = 7
"#
        )
    }

    #[test]
    fn resolves_a_two_stage_study() {
        let plan = resolve_manifest(&two_stage_manifest()).unwrap();
        let schedule = plan.schedule();
        assert_eq!(
            schedule,
            vec![StageId::new("coarse"), StageId::new("fine")],
            "dependencies must order the schedule"
        );

        let fine = plan
            .stages()
            .iter()
            .find(|s| s.id == StageId::new("fine"))
            .unwrap();
        assert_eq!(fine.descriptor.name, "ode-rk4");
        assert_eq!(fine.descriptor.version, SemVer::new(1, 2, 3));
        assert_eq!(fine.descriptor.content_hash, Digest::from_hex(PIN).unwrap());
        assert_eq!(fine.config_hash, Digest::from_hex(CFG_B).unwrap());
        assert_eq!(fine.deps, vec![StageId::new("coarse")]);
    }

    #[test]
    fn a_stage_without_a_seed_inherits_the_studys() {
        let plan = resolve_manifest(&two_stage_manifest()).unwrap();
        let seed = |id: &str| {
            plan.stages()
                .iter()
                .find(|s| s.id == StageId::new(id))
                .unwrap()
                .seed
        };
        assert_eq!(seed("coarse"), 42, "inherited from the study");
        assert_eq!(seed("fine"), 7, "the stage's own override wins");
    }

    #[test]
    fn resolution_is_pure_so_the_same_text_always_gives_the_same_plan() {
        // The property memoization rests on: if resolution varied by machine
        // or by run, two people running the same study would miss each
        // other's cache entirely.
        let a = resolve_manifest(&two_stage_manifest()).unwrap();
        let b = resolve_manifest(&two_stage_manifest()).unwrap();
        assert_eq!(a.stages(), b.stages());
    }

    #[test]
    fn declaration_order_does_not_change_the_resolved_plan() {
        let forward = resolve_manifest(&two_stage_manifest()).unwrap();
        let reversed_text = {
            let m = Manifest::parse(&two_stage_manifest()).unwrap();
            let mut m = m;
            m.stages.reverse();
            m
        };
        let reversed = reversed_text.resolve().unwrap();
        assert_eq!(forward.schedule(), reversed.schedule());
    }

    #[test]
    fn a_misspelled_key_is_an_error_not_a_silent_default() {
        // The whole point of `deny_unknown_fields`: `seeed` would otherwise
        // leave the stage on the study seed and memoize that result as though
        // the author had asked for it.
        let text = format!(
            r#"
[study]
name = "s"
seed = 1

[[stage]]
id      = "a"
plugin  = "p"
version = "1.0.0"
pin     = "{PIN}"
config  = "{CFG_A}"
seeed   = 9
"#
        );
        assert!(matches!(
            resolve_manifest(&text),
            Err(ManifestError::Toml(_))
        ));
    }

    #[test]
    fn a_missing_study_seed_is_an_error() {
        let text = format!(
            r#"
[study]
name = "s"

[[stage]]
id      = "a"
plugin  = "p"
version = "1.0.0"
pin     = "{PIN}"
config  = "{CFG_A}"
"#
        );
        assert!(matches!(
            resolve_manifest(&text),
            Err(ManifestError::Toml(_))
        ));
    }

    #[test]
    fn a_malformed_digest_is_rejected_at_parse_time() {
        for bad in ["not-hex", "", &"a".repeat(63), &"zz".repeat(32)]
        {
            let text = format!(
                r#"
[study]
name = "s"
seed = 1

[[stage]]
id      = "a"
plugin  = "p"
version = "1.0.0"
pin     = "{bad}"
config  = "{CFG_A}"
"#
            );
            assert!(
                matches!(resolve_manifest(&text), Err(ManifestError::Toml(_))),
                "pin {bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn a_malformed_version_names_the_stage_it_came_from() {
        let text = format!(
            r#"
[study]
name = "s"
seed = 1

[[stage]]
id      = "the-bad-one"
plugin  = "p"
version = "1.0"
pin     = "{PIN}"
config  = "{CFG_A}"
"#
        );
        let err = resolve_manifest(&text).unwrap_err();
        assert!(
            matches!(err, ManifestError::InvalidVersion { .. }),
            "{err:?}"
        );
        assert!(err.to_string().contains("the-bad-one"));
    }

    #[test]
    fn an_empty_study_is_an_error_not_a_plan_that_runs_nothing() {
        let text = r#"
[study]
name = "empty"
seed = 1
"#;
        assert!(matches!(
            resolve_manifest(text),
            Err(ManifestError::NoStages(name)) if name == "empty"
        ));
    }

    #[test]
    fn graph_errors_come_from_plan_new_rather_than_being_re_implemented() {
        // Duplicate ids...
        let dup = format!(
            r#"
[study]
name = "s"
seed = 1

[[stage]]
id      = "a"
plugin  = "p"
version = "1.0.0"
pin     = "{PIN}"
config  = "{CFG_A}"

[[stage]]
id      = "a"
plugin  = "p"
version = "1.0.0"
pin     = "{PIN}"
config  = "{CFG_B}"
"#
        );
        assert!(matches!(
            resolve_manifest(&dup),
            Err(ManifestError::Plan(WorkflowError::DuplicateStage(_)))
        ));

        // ...and a dependency on a stage nobody declared.
        let missing = format!(
            r#"
[study]
name = "s"
seed = 1

[[stage]]
id      = "a"
plugin  = "p"
version = "1.0.0"
pin     = "{PIN}"
config  = "{CFG_A}"
needs   = ["ghost"]
"#
        );
        assert!(matches!(
            resolve_manifest(&missing),
            Err(ManifestError::Plan(WorkflowError::MissingDependency { .. }))
        ));
    }

    #[test]
    fn a_cycle_is_refused() {
        let text = format!(
            r#"
[study]
name = "s"
seed = 1

[[stage]]
id      = "a"
plugin  = "p"
version = "1.0.0"
pin     = "{PIN}"
config  = "{CFG_A}"
needs   = ["b"]

[[stage]]
id      = "b"
plugin  = "p"
version = "1.0.0"
pin     = "{PIN}"
config  = "{CFG_B}"
needs   = ["a"]
"#
        );
        assert!(matches!(
            resolve_manifest(&text),
            Err(ManifestError::Plan(WorkflowError::Cycle))
        ));
    }

    #[test]
    fn inputs_parse_with_or_without_the_sos1_prefix() {
        let id = ObjectId::compute(sos_core::HashAlgo::Sha256, b"t", b"upstream");
        let text = format!(
            r#"
[study]
name = "s"
seed = 1

[[stage]]
id      = "a"
plugin  = "p"
version = "1.0.0"
pin     = "{PIN}"
config  = "{CFG_A}"
inputs  = ["{id}", "{bare}"]
"#,
            bare = id.digest().to_hex()
        );
        let plan = resolve_manifest(&text).unwrap();
        // `Stage::new` sorts and dedups inputs, so the two spellings of the
        // same id collapse to one.
        assert_eq!(plan.stages()[0].inputs, vec![id]);
    }

    #[test]
    fn never_panics_on_adversarial_text() {
        // Manifests are untrusted input — the CI fuzz job names them
        // explicitly. Parsing must always return, never panic.
        let cases = [
            "",
            "\0",
            "[",
            "[study]",
            "[[stage]]",
            "study = 1",
            "[study]\nname = 1\nseed = \"x\"",
            "[study]\nname = \"s\"\nseed = -1",
            "[study]\nname = \"s\"\nseed = 18446744073709551616",
            &"[[stage]]\n".repeat(500),
            &format!("[study]\nname = \"{}\"\nseed = 0", "x".repeat(10_000)),
            "\u{feff}[study]",
            "[study]\r\nname = \"s\"\r\nseed = 0\r\n",
        ];
        for case in cases
        {
            let _ = resolve_manifest(case);
        }
    }
}
