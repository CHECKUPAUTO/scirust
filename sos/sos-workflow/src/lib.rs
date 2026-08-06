//! # `sos-workflow` — the SOS Workflow Engine (deterministic core)
//!
//! The Workflow Engine is the **scheduler** of the scientific OS — a *build
//! system whose artifact is knowledge* (RFC-0002 §08, SDE §04). A workflow is an
//! immutable DAG of stages; the engine decides what must run, runs only that,
//! memoizes the rest by content address, and records the schedule it took.
//!
//! This crate is the pure, backend-agnostic core. It provides:
//!
//! * [`Plan`] — an immutable, validated [`Stage`] DAG with a **deterministic**
//!   topological [schedule](Plan::schedule) (ties broken by [`StageId`]).
//!   A stage's edges come in two kinds, deliberately kept apart:
//!   [`deps`](Stage::deps) is *ordering* ("run after"), while
//!   [`consumes`](Stage::consumes) is *dataflow* ("read the outputs of", which
//!   implies ordering). Without the second there was no way to feed one stage
//!   into the next at all — [`inputs`](Stage::inputs) takes literal
//!   [`ObjectId`](sos_core::ObjectId)s and an upstream stage's ids do not
//!   exist until it has run — so every output recorded zero provenance
//!   parents and a plan produced a graph with no edges between its own nodes.
//!   [`run_plan`] resolves consumed outputs into a stage's inputs **before**
//!   computing its [`CacheKey`], so a downstream stage necessarily misses the
//!   cache when its upstream produced something different.
//! * [`CacheKey`] — the content address of a stage invocation:
//!   `hash(descriptor ⊕ inputs ⊕ config ⊕ seed ⊕ env)`. The one mechanism that
//!   gives **both** reproducibility and incremental compute.
//! * [`run_plan`] — the memoized driver: cache-hit ⇒ reuse (nothing runs),
//!   cache-miss ⇒ execute via a [`StageExecutor`], everything recorded in a
//!   [`RunLedger`]. Re-running an unchanged plan against a warm [`Memo`] is all
//!   cache hits — provably identical and nearly free (and the same property makes
//!   a crashed run resumable).
//! * [`RunLedger`] — the immutable, content-addressed record of *how* the plan
//!   ran: control flow is data too.
//! * [`Dispatch`] — the registry-mediated [`StageExecutor`] that **binds** a
//!   stage to the code that runs it: resolving its plugin pinned to the content
//!   hash the stage recorded (so a drifted implementation **fails** rather than
//!   silently computing something else), authorizing it against the study's
//!   capability `Grant` (refusing by default), then delegating. It is itself a
//!   `StageExecutor`, so it composes into [`run_plan`] with the memoization and
//!   ledger above it unchanged.
//!
//! ## What is deliberately *not* here yet
//!
//! The scheduler is engine-agnostic — it sees [`StageDescriptor`]s and
//! [`ObjectId`](sos_core::ObjectId)s, not "curiosity" vs "reasoning." The stage
//! *logic* (running a sweep, a derivation, a simulation) is supplied by the
//! engine crates and backend adapters through the [`StageExecutor`] trait
//! (Invariant VIII); this crate binds and schedules that logic but never
//! implements it. [`Manifest`] now supplies the other half of RFC-0002 §08
//! §1's `resolve(&manifest, &graph)` — a TOML study resolves to a validated
//! `Plan` — but only the manifest half: naming inputs symbolically through the
//! knowledge graph needs a query language this crate will not invent alone, so
//! a stage's inputs are content addresses. Still deferred, with **no stub**:
//! that graph half, and information-theoretic stopping rules (`sos-planner` /
//! statistics). The pieces here are the deterministic heart —
//! cache keys, scheduling, memoization, binding, resolution, ledger — fully implemented and
//! tested.
//!
//! ## Example — memoization makes an unchanged re-run free
//!
//! ```
//! use sos_core::{HashAlgo, ObjectId, SemVer};
//! use sos_workflow::{
//!     run_plan, MemoTable, Plan, Stage, StageDescriptor, StageExecutor, StageId, WorkflowError,
//! };
//!
//! // A trivial executor: each stage produces one deterministic output id, and we
//! // count how many times a stage actually ran.
//! struct Counting { ran: usize }
//! impl StageExecutor for Counting {
//!     fn run(&mut self, stage: &Stage) -> Result<Vec<ObjectId>, WorkflowError> {
//!         self.ran += 1;
//!         Ok(vec![ObjectId::compute(HashAlgo::default(), b"out", stage.id.0.as_bytes())])
//!     }
//! }
//!
//! let d = HashAlgo::default().hash(b"x", b"y"); // stand-in config/plugin digest
//! let mk = |id: &str, deps: Vec<StageId>| Stage::new(
//!     StageId::new(id),
//!     StageDescriptor::new(id, SemVer::new(1, 0, 0), d),
//!     vec![], d, 0, deps,
//! );
//! let plan = Plan::new(vec![mk("a", vec![]), mk("b", vec![StageId::new("a")])]).unwrap();
//!
//! let env = HashAlgo::default().hash(b"env", b"linux-x86_64");
//! let mut memo = MemoTable::new();
//! let mut exec = Counting { ran: 0 };
//!
//! // First run: both stages execute.
//! let first = run_plan(&plan, &env, &mut memo, &mut exec).unwrap();
//! assert_eq!(first.ran_count(), 2);
//! assert_eq!(exec.ran, 2);
//!
//! // Second run against the warm memo: nothing runs — all cache hits, same outputs.
//! let second = run_plan(&plan, &env, &mut memo, &mut exec).unwrap();
//! assert_eq!(second.cache_hit_count(), 2);
//! assert_eq!(exec.ran, 2); // the executor was not called again
//! assert_eq!(first.steps[0].outputs, second.steps[0].outputs);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod cache;
pub mod descriptor;
pub mod dispatch;
pub mod engine;
pub mod error;
pub mod ledger;
pub mod manifest;
pub mod plan;

pub use cache::CacheKey;
pub use descriptor::StageDescriptor;
pub use dispatch::Dispatch;
pub use engine::{Memo, MemoTable, StageExecutor, run_plan};
pub use error::{Result, WorkflowError};
pub use ledger::{LedgerStep, RunLedger, StepOutcome};
pub use manifest::{Manifest, ManifestError, StageSpec, Study, resolve_manifest};
pub use plan::{Plan, Stage, StageId};
