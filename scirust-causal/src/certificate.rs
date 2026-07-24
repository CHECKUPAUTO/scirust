//! [`CausalCertificate`]: the program's mandated shape for reporting any
//! causal claim — "under assumptions A, using evidence E, property Q is
//! {status}[, estimated by M as `estimate` ± `uncertainty`], with sensitivity
//! S and unresolved alternatives R."
//!
//! This phase defines the type and its coherence rule only. No phase in this
//! crate yet *produces* a certificate with `status = Identifiable` and a real
//! estimate — that is later work (effect identification, estimation). What
//! exists here is the structural guarantee that when that work lands, it
//! cannot silently attach a point estimate to a claim it has not certified
//! identifiable: [`CausalCertificateBuilder::finalize`] is the only path to a
//! [`CausalCertificate`], and it enforces that rule unconditionally.

use crate::assumptions::CausalAssumption;
use crate::error::CausalError;
use crate::fingerprint::sha256_hex;

/// Whether — and how — a causal property was identified from the stated
/// assumptions and evidence. Every variant is a legitimate, first-class
/// outcome; `Identifiable` is not privileged over the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IdentifiabilityStatus {
    /// The queried property is a function of the observed distribution(s)
    /// under the stated assumptions.
    Identifiable,
    /// Provably not a function of the observed distribution(s) under the
    /// stated assumptions (e.g. an unblocked backdoor path with no available
    /// adjustment set).
    NotIdentifiable,
    /// Only a Markov-equivalence-class representative is available; the
    /// specific directed claim is not determined by the evidence.
    EquivalenceClassOnly,
    /// Neither identifiability nor non-identifiability could be established
    /// (e.g. the check that would decide it has not been run, or did not
    /// terminate). Never rounded up to `Identifiable`.
    Inconclusive,
}

/// A reproducible statement of a causal claim's status. See the module docs
/// for the shape this encodes. Every field is read-only outside this crate —
/// the only way to build one is [`CausalCertificate::builder`] followed by
/// [`CausalCertificateBuilder::finalize`], which is where the coherence rule
/// and the fingerprint are enforced together.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CausalCertificate {
    query: String,
    status: IdentifiabilityStatus,
    assumptions_used: Vec<CausalAssumption>,
    evidence_summary: String,
    method: Option<String>,
    estimate: Option<f64>,
    uncertainty: Option<f64>,
    sensitivity_note: Option<String>,
    unresolved_alternatives: Vec<String>,
    fingerprint: String,
}

impl CausalCertificate {
    /// Starts a builder for a certificate about `query`, with the given
    /// `status`, `assumptions_used`, and `evidence_summary` (a human-readable
    /// description of what data/environments back this claim).
    pub fn builder(
        query: impl Into<String>,
        status: IdentifiabilityStatus,
        assumptions_used: Vec<CausalAssumption>,
        evidence_summary: impl Into<String>,
    ) -> CausalCertificateBuilder {
        CausalCertificateBuilder {
            query: query.into(),
            status,
            assumptions_used,
            evidence_summary: evidence_summary.into(),
            method: None,
            estimate: None,
            uncertainty: None,
            sensitivity_note: None,
            unresolved_alternatives: Vec::new(),
        }
    }

    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }

    #[must_use]
    pub fn status(&self) -> IdentifiabilityStatus {
        self.status
    }

    #[must_use]
    pub fn assumptions_used(&self) -> &[CausalAssumption] {
        &self.assumptions_used
    }

    #[must_use]
    pub fn evidence_summary(&self) -> &str {
        &self.evidence_summary
    }

    #[must_use]
    pub fn method(&self) -> Option<&str> {
        self.method.as_deref()
    }

    #[must_use]
    pub fn estimate(&self) -> Option<f64> {
        self.estimate
    }

    #[must_use]
    pub fn uncertainty(&self) -> Option<f64> {
        self.uncertainty
    }

    #[must_use]
    pub fn sensitivity_note(&self) -> Option<&str> {
        self.sensitivity_note.as_deref()
    }

    #[must_use]
    pub fn unresolved_alternatives(&self) -> &[String] {
        &self.unresolved_alternatives
    }

    /// The deterministic fingerprint of this certificate's content (see
    /// [`CausalCertificateBuilder::finalize`]).
    #[must_use]
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

/// Builder for [`CausalCertificate`] — accumulate optional fields, then
/// [`CausalCertificateBuilder::finalize`].
#[derive(Debug, Clone, PartialEq)]
pub struct CausalCertificateBuilder {
    query: String,
    status: IdentifiabilityStatus,
    assumptions_used: Vec<CausalAssumption>,
    evidence_summary: String,
    method: Option<String>,
    estimate: Option<f64>,
    uncertainty: Option<f64>,
    sensitivity_note: Option<String>,
    unresolved_alternatives: Vec<String>,
}

impl CausalCertificateBuilder {
    /// Attaches a point estimate and its uncertainty, produced by `method`.
    /// Only meaningful — and only accepted by [`Self::finalize`] — when the
    /// certificate's status is [`IdentifiabilityStatus::Identifiable`].
    #[must_use]
    pub fn with_estimate(
        mut self,
        method: impl Into<String>,
        estimate: f64,
        uncertainty: f64,
    ) -> Self {
        self.method = Some(method.into());
        self.estimate = Some(estimate);
        self.uncertainty = Some(uncertainty);
        self
    }

    #[must_use]
    pub fn with_sensitivity(mut self, note: impl Into<String>) -> Self {
        self.sensitivity_note = Some(note.into());
        self
    }

    #[must_use]
    pub fn with_unresolved_alternative(mut self, alternative: impl Into<String>) -> Self {
        self.unresolved_alternatives.push(alternative.into());
        self
    }

    /// Validates coherence and computes the deterministic fingerprint,
    /// producing the finished [`CausalCertificate`].
    ///
    /// `assumptions_used` and `unresolved_alternatives` are sorted and
    /// deduplicated first — these are semantically *sets*, so the fingerprint
    /// (and the stored certificate) do not depend on the order the builder
    /// happened to receive them in.
    ///
    /// # Errors
    ///
    /// [`CausalError::InvalidContract`] if: `query` is empty; `status` is not
    /// [`IdentifiabilityStatus::Identifiable`] but an estimate is present (the
    /// one rule this type exists to make impossible to violate); or `estimate`
    /// / `uncertainty` is non-finite, or `uncertainty` is negative.
    pub fn finalize(mut self) -> Result<CausalCertificate, CausalError> {
        if self.query.trim().is_empty()
        {
            return Err(CausalError::InvalidContract {
                detail: "certificate query must not be empty",
            });
        }
        if self.status != IdentifiabilityStatus::Identifiable && self.estimate.is_some()
        {
            return Err(CausalError::InvalidContract {
                detail: "only Identifiable may carry a numeric estimate",
            });
        }
        if let Some(v) = self.estimate
        {
            if !v.is_finite()
            {
                return Err(CausalError::InvalidContract {
                    detail: "certificate estimate must be finite",
                });
            }
        }
        if let Some(v) = self.uncertainty
        {
            if !v.is_finite() || v < 0.0
            {
                return Err(CausalError::InvalidContract {
                    detail: "certificate uncertainty must be finite and non-negative",
                });
            }
        }

        self.assumptions_used.sort();
        self.assumptions_used.dedup();
        self.unresolved_alternatives.sort();
        self.unresolved_alternatives.dedup();

        let pre_image = CertificatePreImage {
            query: &self.query,
            status: self.status,
            assumptions_used: &self.assumptions_used,
            evidence_summary: &self.evidence_summary,
            method: &self.method,
            estimate: self.estimate,
            uncertainty: self.uncertainty,
            sensitivity_note: &self.sensitivity_note,
            unresolved_alternatives: &self.unresolved_alternatives,
        };
        let canonical = serde_json::to_vec(&pre_image)
            .expect("CertificatePreImage serialization is infallible");
        let fingerprint = sha256_hex(&canonical);

        Ok(CausalCertificate {
            query: self.query,
            status: self.status,
            assumptions_used: self.assumptions_used,
            evidence_summary: self.evidence_summary,
            method: self.method,
            estimate: self.estimate,
            uncertainty: self.uncertainty,
            sensitivity_note: self.sensitivity_note,
            unresolved_alternatives: self.unresolved_alternatives,
            fingerprint,
        })
    }
}

/// The byte-for-byte canonical content a [`CausalCertificate`]'s fingerprint
/// commits to: every semantic field, and nothing else (notably not the
/// fingerprint field itself, which would be self-referential).
///
/// Reproducibility holds under the same caveat already documented at the
/// crate root: a fixed implementation, build, and execution environment.
#[derive(serde::Serialize)]
struct CertificatePreImage<'a> {
    query: &'a str,
    status: IdentifiabilityStatus,
    assumptions_used: &'a [CausalAssumption],
    evidence_summary: &'a str,
    method: &'a Option<String>,
    estimate: Option<f64>,
    uncertainty: Option<f64>,
    sensitivity_note: &'a Option<String>,
    unresolved_alternatives: &'a [String],
}
