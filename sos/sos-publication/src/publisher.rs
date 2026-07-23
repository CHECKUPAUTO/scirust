//! The [`Publish`] trait and the stateless [`Publisher`] that implements it.

use sos_core::ObjectId;
use std::collections::BTreeMap;

use crate::error::Result;
use crate::key::{FigureKey, TableKey};
use crate::policy::SupportPolicy;
use crate::publication::Publication;
use crate::render::{Artifact, Format};
use crate::source::PublicationObjectSource;
use crate::verify::{ExhibitReport, VerificationReport};

/// The Publication Engine's surface (RFC-0002 §10.5): render a publication and
/// verify it against the object graph.
pub trait Publish {
    /// Render `publication` to `format`.
    ///
    /// # Errors
    /// See [`crate::render::render`].
    fn render(&self, publication: &Publication, format: Format) -> Result<Artifact>;

    /// Verify `publication` against `source` under `policy`.
    ///
    /// # Errors
    /// See [`crate::verify::verify`].
    fn verify<S, P>(
        &self,
        publication: &Publication,
        source: &S,
        policy: &P,
    ) -> Result<VerificationReport>
    where
        S: PublicationObjectSource + ?Sized,
        P: SupportPolicy;
}

/// The default publisher — a stateless renderer/verifier over the deterministic
/// core.
#[derive(Debug, Clone, Copy, Default)]
pub struct Publisher;

impl Publisher {
    /// Construct the publisher.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Run the executable-exhibit drift check (see
    /// [`crate::verify::verify_exhibits`]).
    #[must_use]
    pub fn verify_exhibits(
        &self,
        publication: &Publication,
        figures: &BTreeMap<FigureKey, ObjectId>,
        tables: &BTreeMap<TableKey, ObjectId>,
    ) -> ExhibitReport {
        crate::verify::verify_exhibits(publication, figures, tables)
    }
}

impl Publish for Publisher {
    fn render(&self, publication: &Publication, format: Format) -> Result<Artifact> {
        crate::render::render(publication, format)
    }

    fn verify<S, P>(
        &self,
        publication: &Publication,
        source: &S,
        policy: &P,
    ) -> Result<VerificationReport>
    where
        S: PublicationObjectSource + ?Sized,
        P: SupportPolicy,
    {
        crate::verify::verify(publication, source, policy)
    }
}
