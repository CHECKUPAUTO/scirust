//! Correlated enterprise audit trail.
//!
//! Binds the security chain into one verifiable record: tenant, user/agent,
//! session, [`ApprovalRequestId`], `call_id`, tool, decision, sandbox,
//! execution digest and artifact digest are recorded in one event, and each
//! event is chained to the previous one with SHA-256 so the trail is
//! tamper-evident. The bounded projection stays in RAM; the digest chain is
//! the authoritative correlation key.

use super::approval_request::ApprovalRequestId;
use super::enterprise::TenantId;
use crate::sha256::sha256_hex;

/// SHA-256 chain hash of the empty audit trail.
pub const ENTERPRISE_AUDIT_GENESIS: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// One correlated enterprise audit event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnterpriseAuditEvent {
    pub sequence: u64,
    pub tenant: TenantId,
    pub subject: String,
    pub session_id: String,
    pub request_id: Option<ApprovalRequestId>,
    pub call_id: Option<String>,
    pub tool: Option<String>,
    pub decision: Option<String>,
    pub sandbox: Option<String>,
    pub execution_digest: Option<String>,
    pub artifact_digest: Option<String>,
    pub prev_hash: String,
    pub chain_hash: String,
}

impl EnterpriseAuditEvent {
    /// Build one event and compute its chain hash over a length-prefixed
    /// canonical serialization of every field.
    pub fn new(
        sequence: u64,
        tenant: TenantId,
        subject: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Self {
        let mut event = Self {
            sequence,
            tenant,
            subject: subject.into(),
            session_id: session_id.into(),
            request_id: None,
            call_id: None,
            tool: None,
            decision: None,
            sandbox: None,
            execution_digest: None,
            artifact_digest: None,
            prev_hash: ENTERPRISE_AUDIT_GENESIS.to_string(),
            chain_hash: String::new(),
        };
        event.chain_hash = event.compute_chain_hash();
        event
    }

    /// Correlate an approval resolution.
    pub fn with_approval(mut self, request_id: ApprovalRequestId, decision: &str) -> Self {
        self.request_id = Some(request_id);
        self.decision = Some(decision.to_string());
        self
    }

    /// Correlate a tool execution.
    pub fn with_execution(
        mut self,
        call_id: impl Into<String>,
        tool: impl Into<String>,
        sandbox: &str,
        execution_digest: impl Into<String>,
    ) -> Self {
        self.call_id = Some(call_id.into());
        self.tool = Some(tool.into());
        self.sandbox = Some(sandbox.to_string());
        self.execution_digest = Some(execution_digest.into());
        self
    }

    /// Link the produced artifact (provenance digest).
    pub fn with_artifact(mut self, artifact_digest: impl Into<String>) -> Self {
        self.artifact_digest = Some(artifact_digest.into());
        self
    }

    /// Canonical bytes for chaining: length-prefixed fields in a fixed order.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(512);
        push_field(&mut bytes, &self.sequence.to_le_bytes());
        push_field(&mut bytes, self.tenant.as_str().as_bytes());
        push_field(&mut bytes, self.subject.as_bytes());
        push_field(&mut bytes, self.session_id.as_bytes());
        push_field(
            &mut bytes,
            self.request_id
                .as_ref()
                .map(|id| id.as_str().as_bytes())
                .unwrap_or_default(),
        );
        push_field(
            &mut bytes,
            self.call_id.as_deref().unwrap_or_default().as_bytes(),
        );
        push_field(
            &mut bytes,
            self.tool.as_deref().unwrap_or_default().as_bytes(),
        );
        push_field(
            &mut bytes,
            self.decision.as_deref().unwrap_or_default().as_bytes(),
        );
        push_field(
            &mut bytes,
            self.sandbox.as_deref().unwrap_or_default().as_bytes(),
        );
        push_field(
            &mut bytes,
            self.execution_digest
                .as_deref()
                .unwrap_or_default()
                .as_bytes(),
        );
        push_field(
            &mut bytes,
            self.artifact_digest
                .as_deref()
                .unwrap_or_default()
                .as_bytes(),
        );
        push_field(&mut bytes, self.prev_hash.as_bytes());
        bytes
    }

    fn compute_chain_hash(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }

    fn verify_chain_hash(&self) -> bool {
        self.chain_hash == self.compute_chain_hash()
    }
}

fn push_field(bytes: &mut Vec<u8>, field: &[u8]) {
    bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());
    bytes.extend_from_slice(field);
}

/// Append-only correlated audit trail with tamper-evident chaining.
#[derive(Debug, Default)]
pub struct EnterpriseAuditTrail {
    events: std::sync::Mutex<Vec<EnterpriseAuditEvent>>,
}

impl EnterpriseAuditTrail {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one event, chaining it to the previous hash. A failed append
    /// (lock poisoned) fails closed and records nothing.
    pub fn append(&self, mut event: EnterpriseAuditEvent) -> Result<u64, String> {
        let mut events = self
            .events
            .lock()
            .map_err(|_| "enterprise audit trail is unavailable".to_string())?;
        let sequence = events.len() as u64;
        if event.sequence != sequence
        {
            return Err(format!(
                "enterprise audit sequence mismatch: expected {sequence}, got {}",
                event.sequence
            ));
        }
        event.prev_hash = events
            .last()
            .map(|last| last.chain_hash.clone())
            .unwrap_or_else(|| ENTERPRISE_AUDIT_GENESIS.to_string());
        event.chain_hash = event.compute_chain_hash();
        events.push(event);
        Ok(sequence)
    }

    /// Verify the whole chain from the genesis hash. Any tampering or gap
    /// fails verification.
    pub fn verify(&self) -> Result<(), String> {
        let events = self
            .events
            .lock()
            .map_err(|_| "enterprise audit trail is unavailable".to_string())?;
        let mut expected = ENTERPRISE_AUDIT_GENESIS.to_string();
        for (index, event) in events.iter().enumerate()
        {
            if event.sequence != index as u64
            {
                return Err(format!("enterprise audit sequence gap at {index}"));
            }
            if event.prev_hash != expected || !event.verify_chain_hash()
            {
                return Err(format!("enterprise audit chain broken at {index}"));
            }
            expected = event.chain_hash.clone();
        }
        Ok(())
    }

    /// Bounded RAM projection; the chain hashes are the authoritative link.
    pub fn events(&self) -> Result<Vec<EnterpriseAuditEvent>, String> {
        Ok(self
            .events
            .lock()
            .map_err(|_| "enterprise audit trail is unavailable".to_string())?
            .clone())
    }

    pub fn len(&self) -> Result<usize, String> {
        Ok(self
            .events
            .lock()
            .map_err(|_| "enterprise audit trail is unavailable".to_string())?
            .len())
    }

    pub fn is_empty(&self) -> Result<bool, String> {
        Ok(self.len()? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tenant() -> TenantId {
        TenantId::parse("acme").unwrap()
    }

    fn base(sequence: u64) -> EnterpriseAuditEvent {
        EnterpriseAuditEvent::new(sequence, tenant(), "alice", "session-1")
    }

    #[test]
    fn append_chains_and_verifies() {
        let trail = EnterpriseAuditTrail::new();
        trail.append(base(0)).unwrap();
        trail.append(base(1)).unwrap();
        trail.append(base(2)).unwrap();
        assert_eq!(trail.len().unwrap(), 3);
        trail.verify().unwrap();
    }

    #[test]
    fn tampering_breaks_the_chain() {
        let trail = EnterpriseAuditTrail::new();
        trail
            .append(base(0).with_approval(ApprovalRequestId::generate(), "allowed-once"))
            .unwrap();
        trail.append(base(1)).unwrap();
        // Tamper with the first event's decision after the fact.
        {
            let mut events = trail.events.lock().unwrap();
            events[0].decision = Some("allowed-persistent".to_string());
        }
        let error = trail.verify().expect_err("tampering must break the chain");
        assert!(error.contains("chain broken"), "{error}");
    }

    #[test]
    fn sequence_mismatch_is_rejected() {
        let trail = EnterpriseAuditTrail::new();
        trail.append(base(0)).unwrap();
        let error = trail
            .append(base(5))
            .expect_err("sequence mismatch must be rejected");
        assert!(error.contains("sequence mismatch"), "{error}");
        assert_eq!(trail.len().unwrap(), 1);
    }

    #[test]
    fn full_correlation_record() {
        let trail = EnterpriseAuditTrail::new();
        let request_id = ApprovalRequestId::generate();
        trail
            .append(
                base(0)
                    .with_approval(request_id.clone(), "allowed-once")
                    .with_execution("call-42", "build", "workspace-write", "exec-digest-1")
                    .with_artifact("artifact-digest-1"),
            )
            .unwrap();
        let events = trail.events().unwrap();
        assert_eq!(events[0].request_id, Some(request_id));
        assert_eq!(events[0].call_id.as_deref(), Some("call-42"));
        assert_eq!(events[0].tool.as_deref(), Some("build"));
        assert_eq!(events[0].sandbox.as_deref(), Some("workspace-write"));
        assert_eq!(
            events[0].artifact_digest.as_deref(),
            Some("artifact-digest-1")
        );
        trail.verify().unwrap();
    }

    #[test]
    fn empty_trail_verifies() {
        let trail = EnterpriseAuditTrail::new();
        trail.verify().unwrap();
        assert_eq!(trail.len().unwrap(), 0);
    }

    #[test]
    fn correlated_event_survives_replay() {
        let trail = EnterpriseAuditTrail::new();
        trail
            .append(base(0).with_execution("c1", "read", "read-only", "e1"))
            .unwrap();
        trail
            .append(base(1).with_execution("c2", "grep", "read-only", "e2"))
            .unwrap();
        // Replay = clone the projection and verify; the chain hashes must
        // match the original order.
        let events = trail.events().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].prev_hash, events[0].chain_hash);
        trail.verify().unwrap();
    }
}
