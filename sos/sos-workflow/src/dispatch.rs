//! [`Dispatch`] — the registry-mediated [`StageExecutor`] that binds a stage to
//! the implementation that runs it.
//!
//! [`run_plan`](crate::run_plan) schedules stages and memoizes their results,
//! but it deliberately never decides *what code a stage runs* — it takes a
//! [`StageExecutor`] and calls it. That leaves a seam: a [`Stage`] names its
//! plugin by [`StageDescriptor`](crate::StageDescriptor) (name + version +
//! content hash), and something has to turn that name into running code.
//!
//! `Dispatch` is that something, and it is where three guarantees the RFC
//! promises stop being aspirational:
//!
//! * **Drift is detected, not assumed away.** A stage pins the exact
//!   `content_hash` of the plugin it was written against. Binding goes through
//!   [`Registry::resolve_pinned`], so if the registry now holds a *different*
//!   implementation under that name and version, the run **fails** with
//!   [`RegistryError::Drift`] rather than silently computing something else and
//!   calling it the same result (RFC-0002 §10 §1).
//! * **Least privilege is enforced, refusing by default.** Every binding is
//!   [`authorize`]d against the study's [`Grant`]. A plugin that needs the GPU,
//!   the network, or the right to cause effects does not get them because it
//!   asked — it gets them because the study granted them.
//! * **A missing implementation is an error, not a no-op.** A descriptor that
//!   resolves but has no registered handler is
//!   [`WorkflowError::NoHandler`](crate::WorkflowError::NoHandler), never a
//!   quietly-empty stage output.
//!
//! `Dispatch` is itself a [`StageExecutor`], so it composes: it drops into
//! [`run_plan`](crate::run_plan) exactly where a hand-written executor would,
//! and the memoization, scheduling, and ledger above it are unchanged. A
//! cache-*hit* stage is never bound at all — the scheduler does not call the
//! executor — so binding cost and capability checks fall only on work that
//! actually runs.
//!
//! ## What this is not
//!
//! It resolves *names to code*; it does not decide what a stage's code should
//! be. The handlers themselves are supplied by the engine crates and backend
//! adapters (Invariant VIII): `sos-scirust` for computation, `sos-ccos` for
//! cognition. Turning a TOML study manifest into a [`Plan`](crate::Plan) is a
//! separate, still-deferred frontend concern.

use std::collections::BTreeMap;

use sos_core::ObjectId;
use sos_registry::{Grant, Registry, VersionReq, authorize};

use crate::engine::StageExecutor;
use crate::error::{Result, WorkflowError};
use crate::plan::Stage;

/// A [`StageExecutor`] that binds each stage to a registered implementation,
/// enforcing content-hash pinning and capability authorization on the way.
///
/// Register one handler per plugin **name**; the version and content hash come
/// from the [`Registry`] and the [`Stage`] respectively, and must agree.
///
/// ## Example
///
/// ```
/// use sos_core::{HashAlgo, ObjectId, SemVer};
/// use sos_registry::{Capability, Grant, PluginDescriptor, Registry, Role};
/// use sos_workflow::{
///     Dispatch, Stage, StageDescriptor, StageExecutor, StageId, WorkflowError,
/// };
///
/// // A handler that produces one deterministic output.
/// struct Simulate;
/// impl StageExecutor for Simulate {
///     fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
///         Ok(vec![ObjectId::compute(HashAlgo::default(), b"out", stage.id.0.as_bytes())])
///     }
/// }
///
/// let hash = HashAlgo::default().hash(b"plugin", b"sim-1.0.0");
/// let mut registry = Registry::new();
/// registry.register(
///     PluginDescriptor::new("sim", SemVer::new(1, 0, 0), hash, Role::Simulation)
///         .needs(Capability::Effectful),
/// );
///
/// let stage = Stage::new(
///     StageId::new("run"),
///     StageDescriptor::new("sim", SemVer::new(1, 0, 0), hash),
///     vec![], hash, 0, vec![],
/// );
///
/// // A study that grants the effect capability can bind and run it...
/// let grant = Grant::new().allow(Capability::Effectful);
/// let mut dispatch = Dispatch::new(&registry, grant);
/// dispatch.register("sim", Box::new(Simulate));
/// assert_eq!(dispatch.run(&stage).unwrap().len(), 1);
///
/// // ...but one that grants nothing is refused, by default.
/// let mut ungranted = Dispatch::new(&registry, Grant::new());
/// ungranted.register("sim", Box::new(Simulate));
/// assert!(matches!(ungranted.run(&stage), Err(WorkflowError::Unbound { .. })));
/// ```
pub struct Dispatch<'a> {
    registry: &'a Registry,
    grant: Grant,
    handlers: BTreeMap<String, Box<dyn StageExecutor + 'a>>,
}

impl<'a> Dispatch<'a> {
    /// A dispatcher over `registry`, running plugins under `grant`.
    ///
    /// [`Grant::new`] grants nothing, so a dispatcher built with it runs only
    /// plugins that declare no capabilities at all — the least-privilege
    /// default.
    #[must_use]
    pub fn new(registry: &'a Registry, grant: Grant) -> Self {
        Self {
            registry,
            grant,
            handlers: BTreeMap::new(),
        }
    }

    /// Register the code that runs plugin `name`.
    ///
    /// Registering the same name twice replaces the handler; the registry, not
    /// this map, is what pins *which version and content* that name resolves to.
    pub fn register(&mut self, name: impl Into<String>, handler: Box<dyn StageExecutor + 'a>) {
        self.handlers.insert(name.into(), handler);
    }

    /// The plugin names this dispatcher can run, sorted.
    #[must_use]
    pub fn handled(&self) -> Vec<&str> {
        self.handlers.keys().map(String::as_str).collect()
    }

    /// Whether a handler is registered for `name`.
    #[must_use]
    pub fn handles(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }
}

impl StageExecutor for Dispatch<'_> {
    /// Bind `stage` to its implementation and run it.
    ///
    /// Binding resolves the stage's plugin at **exactly** the version it names,
    /// pinned to the content hash it recorded, then authorizes it against the
    /// grant. Only then is the handler called.
    ///
    /// # Errors
    /// [`WorkflowError::Unbound`] if the plugin is unregistered, has no
    /// satisfying version, **drifted** from the pinned content hash, or was
    /// denied for missing capabilities; [`WorkflowError::NoHandler`] if it
    /// resolved but no handler was registered to run it; whatever the handler
    /// itself returns otherwise.
    fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>> {
        let name = stage.descriptor.name.clone();

        // Resolve at exactly the pinned version, and verify the registered
        // implementation is byte-for-byte the one this stage was written
        // against — this is the drift check.
        let descriptor = self
            .registry
            .resolve_pinned(
                &name,
                &VersionReq::Exact(stage.descriptor.version),
                &stage.descriptor.content_hash,
            )
            .map_err(|source| WorkflowError::Unbound {
                stage: stage.id.clone(),
                source,
            })?;

        // Least privilege: refuse unless the study granted everything it needs.
        authorize(descriptor, &self.grant).map_err(|source| WorkflowError::Unbound {
            stage: stage.id.clone(),
            source,
        })?;

        let handler = self
            .handlers
            .get_mut(&name)
            .ok_or_else(|| WorkflowError::NoHandler {
                stage: stage.id.clone(),
                plugin: name,
            })?;
        handler.run(stage)
    }
}

impl core::fmt::Debug for Dispatch<'_> {
    /// Handlers are trait objects and cannot be shown; their names can.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Dispatch")
            .field("grant", &self.grant)
            .field("handlers", &self.handled())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use sos_core::{Digest, HashAlgo, SemVer};
    use sos_registry::{Capability, PluginDescriptor, RegistryError, Role};

    use super::*;
    use crate::descriptor::StageDescriptor;
    use crate::plan::StageId;

    /// A handler that records how many times it ran.
    struct Counting {
        ran: usize,
    }

    impl StageExecutor for Counting {
        fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>> {
            self.ran += 1;
            Ok(vec![ObjectId::compute(
                HashAlgo::default(),
                b"out",
                stage.id.0.as_bytes(),
            )])
        }
    }

    /// A handler that always fails.
    struct Failing;

    impl StageExecutor for Failing {
        fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>> {
            Err(WorkflowError::StageFailed {
                stage: stage.id.clone(),
                reason: "backend exploded".into(),
            })
        }
    }

    fn hash(tag: &[u8]) -> Digest {
        HashAlgo::default().hash(b"test", tag)
    }

    fn stage_named(id: &str, plugin: &str, version: SemVer, content: Digest) -> Stage {
        Stage::new(
            StageId::new(id),
            StageDescriptor::new(plugin, version, content),
            vec![],
            hash(b"cfg"),
            0,
            vec![],
        )
    }

    fn registry_with(plugin: &str, version: SemVer, content: Digest) -> Registry {
        let mut r = Registry::new();
        r.register(PluginDescriptor::new(
            plugin,
            version,
            content,
            Role::Simulation,
        ));
        r
    }

    #[test]
    fn binds_and_runs_a_registered_plugin() {
        let content = hash(b"sim-v1");
        let registry = registry_with("sim", SemVer::new(1, 0, 0), content);
        let mut dispatch = Dispatch::new(&registry, Grant::new());
        dispatch.register("sim", Box::new(Counting { ran: 0 }));

        let stage = stage_named("a", "sim", SemVer::new(1, 0, 0), content);
        let out = dispatch.run(&stage).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn a_drifted_implementation_is_refused() {
        // The registry holds a *different* build than the stage pins.
        let registry = registry_with("sim", SemVer::new(1, 0, 0), hash(b"rebuilt"));
        let mut dispatch = Dispatch::new(&registry, Grant::new());
        dispatch.register("sim", Box::new(Counting { ran: 0 }));

        let stage = stage_named("a", "sim", SemVer::new(1, 0, 0), hash(b"as-written"));
        let err = dispatch.run(&stage).unwrap_err();
        assert!(matches!(
            err,
            WorkflowError::Unbound {
                source: RegistryError::Drift { .. },
                ..
            }
        ));
    }

    #[test]
    fn an_unregistered_plugin_is_refused() {
        let registry = Registry::new();
        let mut dispatch = Dispatch::new(&registry, Grant::new());
        let stage = stage_named("a", "ghost", SemVer::new(1, 0, 0), hash(b"x"));
        assert!(matches!(
            dispatch.run(&stage).unwrap_err(),
            WorkflowError::Unbound {
                source: RegistryError::NotFound(_),
                ..
            }
        ));
    }

    #[test]
    fn a_version_mismatch_is_refused() {
        let content = hash(b"sim-v1");
        let registry = registry_with("sim", SemVer::new(1, 0, 0), content);
        let mut dispatch = Dispatch::new(&registry, Grant::new());
        dispatch.register("sim", Box::new(Counting { ran: 0 }));

        // The stage wants 2.0.0; only 1.0.0 is registered.
        let stage = stage_named("a", "sim", SemVer::new(2, 0, 0), content);
        assert!(matches!(
            dispatch.run(&stage).unwrap_err(),
            WorkflowError::Unbound {
                source: RegistryError::VersionUnsatisfied { .. },
                ..
            }
        ));
    }

    #[test]
    fn capabilities_are_refused_by_default_and_allowed_when_granted() {
        let content = hash(b"gpu-sim");
        let mut registry = Registry::new();
        registry.register(
            PluginDescriptor::new("sim", SemVer::new(1, 0, 0), content, Role::Simulation)
                .needs(Capability::Gpu),
        );
        let stage = stage_named("a", "sim", SemVer::new(1, 0, 0), content);

        // Refuse by default: the grant is empty.
        let mut ungranted = Dispatch::new(&registry, Grant::new());
        ungranted.register("sim", Box::new(Counting { ran: 0 }));
        let err = ungranted.run(&stage).unwrap_err();
        match err
        {
            WorkflowError::Unbound {
                source: RegistryError::Denied { missing, .. },
                ..
            } => assert_eq!(missing, vec![Capability::Gpu]),
            other => panic!("expected a capability denial, got {other:?}"),
        }

        // Granted: the same stage binds and runs.
        let mut granted = Dispatch::new(&registry, Grant::new().allow(Capability::Gpu));
        granted.register("sim", Box::new(Counting { ran: 0 }));
        assert!(granted.run(&stage).is_ok());
    }

    #[test]
    fn a_resolved_plugin_with_no_handler_is_an_error_not_an_empty_output() {
        let content = hash(b"sim-v1");
        let registry = registry_with("sim", SemVer::new(1, 0, 0), content);
        // Registry knows the plugin; nothing is registered to *run* it.
        let mut dispatch = Dispatch::new(&registry, Grant::new());

        let stage = stage_named("a", "sim", SemVer::new(1, 0, 0), content);
        assert!(matches!(
            dispatch.run(&stage).unwrap_err(),
            WorkflowError::NoHandler { .. }
        ));
    }

    #[test]
    fn a_handler_failure_propagates_unchanged() {
        let content = hash(b"sim-v1");
        let registry = registry_with("sim", SemVer::new(1, 0, 0), content);
        let mut dispatch = Dispatch::new(&registry, Grant::new());
        dispatch.register("sim", Box::new(Failing));

        let stage = stage_named("a", "sim", SemVer::new(1, 0, 0), content);
        assert!(matches!(
            dispatch.run(&stage).unwrap_err(),
            WorkflowError::StageFailed { .. }
        ));
    }

    #[test]
    fn handlers_are_introspectable_and_replaceable() {
        let registry = Registry::new();
        let mut dispatch = Dispatch::new(&registry, Grant::new());
        assert!(dispatch.handled().is_empty());

        dispatch.register("a", Box::new(Counting { ran: 0 }));
        dispatch.register("b", Box::new(Counting { ran: 0 }));
        assert_eq!(dispatch.handled(), vec!["a", "b"]); // sorted
        assert!(dispatch.handles("a"));
        assert!(!dispatch.handles("z"));

        // Re-registering replaces rather than duplicating.
        dispatch.register("a", Box::new(Failing));
        assert_eq!(dispatch.handled(), vec!["a", "b"]);
    }

    #[test]
    fn different_plugins_route_to_their_own_handlers() {
        let (c1, c2) = (hash(b"one"), hash(b"two"));
        let mut registry = Registry::new();
        registry.register(PluginDescriptor::new(
            "good",
            SemVer::new(1, 0, 0),
            c1,
            Role::Simulation,
        ));
        registry.register(PluginDescriptor::new(
            "bad",
            SemVer::new(1, 0, 0),
            c2,
            Role::Simulation,
        ));

        let mut dispatch = Dispatch::new(&registry, Grant::new());
        dispatch.register("good", Box::new(Counting { ran: 0 }));
        dispatch.register("bad", Box::new(Failing));

        assert!(
            dispatch
                .run(&stage_named("a", "good", SemVer::new(1, 0, 0), c1))
                .is_ok()
        );
        assert!(
            dispatch
                .run(&stage_named("b", "bad", SemVer::new(1, 0, 0), c2))
                .is_err()
        );
    }
}
