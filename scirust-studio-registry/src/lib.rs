//! `scirust-studio-registry` — machine-readable capability descriptors and
//! the registry that catalogues them.
//!
//! This crate owns *descriptions* of capabilities (what parameters they
//! take, what units, what solvers, what they verify) as typed, static data.
//! It does not execute anything — that is `scirust-studio-runtime`, which
//! implements one [`CapabilityAdapter`](https://docs.rs/scirust-studio-runtime)
//! per capability and exposes each one's [`CapabilityDescriptor`] through
//! `descriptor()`. Nothing in this crate can construct a descriptor from a
//! bare id or name, which is how "never advertise a capability without a
//! tested executable adapter" is enforced structurally rather than by
//! convention.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod descriptor;
mod registry;

pub use descriptor::{
    BackendKind, CapabilityCategory, CapabilityDescriptor, CapabilityId, CapabilityMaturity,
    Cardinality, DeterminismClass, FieldDescriptor, OutputDescriptor, PrecisionKind,
    SolverDescriptor, VerificationCheckDescriptor, VerificationDescriptor,
};
pub use registry::{CapabilityRegistry, RegistryError};
