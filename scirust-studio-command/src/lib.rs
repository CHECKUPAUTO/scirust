//! `scirust-studio-command` — typed command descriptors, a command registry,
//! and the `SRST-*` error catalogue that SciRust Studio's CLI and (later)
//! its desktop application are meant to share, so command names, help text,
//! and argument definitions are written once.
//!
//! This is Phase 1 of the SciRust Studio effort (see
//! `docs/studio/adr/0000-scope-and-sequencing.md`): it defines the shared
//! vocabulary. It does **not** yet refactor `scirust-cli`'s existing ~55
//! hand-dispatched commands onto this registry — that is Phase 4, deliberately
//! deferred so those commands' passing regression tests are not put at risk
//! before the registry has proven itself on new commands first.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod registry;

pub use error::{CatalogedError, ErrorCode, ErrorFamily};
pub use registry::{
    ArgumentDescriptor, CommandDescriptor, CommandExample, CommandId, CommandRegistry,
    RegistryError, SafetyClass,
};
