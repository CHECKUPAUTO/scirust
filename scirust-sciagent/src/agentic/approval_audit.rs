use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Approval channel whose lifecycle is being observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalChannel {
    Tool,
    Sandbox,
}

/// Lifecycle phase for one correlated approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalLifecycle {
    Requested,
    Resolved,
}

/// Closed resolution vocabulary shared by supervision clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalResolution {
    AllowedOnce,
    AllowedSession,
    Rejected,
    Cancelled,
    Unavailable,
}

/// One machine-readable approval lifecycle event.
///
/// `call_id` is the correlation key. Sandbox events additionally carry the
/// requested permission and justification. Tool events may carry a subject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalAuditEvent {
    pub sequence: u64,
    pub channel: ApprovalChannel,
    pub lifecycle: ApprovalLifecycle,
    pub call_id: String,
    pub tool: String,
    pub subject: Option<String>,
    pub requested_permission: Option<String>,
    pub justification: Option<String>,
    pub resolution: Option<ApprovalResolution>,
}

impl ApprovalAuditEvent {
    pub fn tool_requested(call_id: &str, tool: &str, subject: &str) -> Self {
        Self {
            sequence: 0,
            channel: ApprovalChannel::Tool,
            lifecycle: ApprovalLifecycle::Requested,
            call_id: call_id.to_string(),
            tool: tool.to_string(),
            subject: (!subject.is_empty()).then(|| subject.to_string()),
            requested_permission: None,
            justification: None,
            resolution: None,
        }
    }

    pub fn tool_resolved(
        call_id: &str,
        tool: &str,
        subject: &str,
        resolution: ApprovalResolution,
    ) -> Self {
        let mut event = Self::tool_requested(call_id, tool, subject);
        event.lifecycle = ApprovalLifecycle::Resolved;
        event.resolution = Some(resolution);
        event
    }

    pub fn sandbox_requested(
        call_id: &str,
        tool: &str,
        requested_permission: &str,
        justification: &str,
    ) -> Self {
        Self {
            sequence: 0,
            channel: ApprovalChannel::Sandbox,
            lifecycle: ApprovalLifecycle::Requested,
            call_id: call_id.to_string(),
            tool: tool.to_string(),
            subject: None,
            requested_permission: Some(requested_permission.to_string()),
            justification: Some(justification.to_string()),
            resolution: None,
        }
    }

    pub fn sandbox_resolved(
        call_id: &str,
        tool: &str,
        requested_permission: &str,
        justification: &str,
        resolution: ApprovalResolution,
    ) -> Self {
        let mut event = Self::sandbox_requested(
            call_id,
            tool,
            requested_permission,
            justification,
        );
        event.lifecycle = ApprovalLifecycle::Resolved;
        event.resolution = Some(resolution);
        event
    }
}

/// Observation seam for approval lifecycle events.
///
/// When a sink is explicitly installed, recording failures fail closed: an
/// approval is not allowed to proceed without the configured audit trail.
pub trait ApprovalAuditSink: Send + Sync {
    fn record(&self, event: ApprovalAuditEvent) -> Result<(), String>;
}

/// Bounded, process-local journal suitable for a supervision UI.
pub struct InMemoryApprovalAudit {
    capacity: usize,
    next_sequence: AtomicU64,
    events: Mutex<VecDeque<ApprovalAuditEvent>>,
}

impl InMemoryApprovalAudit {
    pub fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("approval audit capacity must be greater than zero".to_string());
        }
        Ok(Self {
            capacity,
            next_sequence: AtomicU64::new(1),
            events: Mutex::new(VecDeque::with_capacity(capacity)),
        })
    }

    pub fn snapshot(&self) -> Result<Vec<ApprovalAuditEvent>, String> {
        Ok(self
            .events
            .lock()
            .map_err(|_| "approval audit journal is unavailable".to_string())?
            .iter()
            .cloned()
            .collect())
    }

    pub fn clear(&self) -> Result<(), String> {
        self.events
            .lock()
            .map_err(|_| "approval audit journal is unavailable".to_string())?
            .clear();
        Ok(())
    }
}

impl ApprovalAuditSink for InMemoryApprovalAudit {
    fn record(&self, mut event: ApprovalAuditEvent) -> Result<(), String> {
        event.sequence = self.next_sequence.fetch_add(1, Ordering::SeqCst);
        let mut events = self
            .events
            .lock()
            .map_err(|_| "approval audit journal is unavailable".to_string())?;
        if events.len() == self.capacity {
            events.pop_front();
        }
        events.push_back(event);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_journal_is_ordered_and_evicts_oldest() {
        let audit = InMemoryApprovalAudit::new(2).unwrap();
        audit
            .record(ApprovalAuditEvent::tool_requested("c1", "build", "core"))
            .unwrap();
        audit
            .record(ApprovalAuditEvent::tool_resolved(
                "c1",
                "build",
                "core",
                ApprovalResolution::AllowedOnce,
            ))
            .unwrap();
        audit
            .record(ApprovalAuditEvent::sandbox_requested(
                "c2",
                "test",
                "danger-full-access",
                "needs external fixture",
            ))
            .unwrap();
        let events = audit.snapshot().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence, 2);
        assert_eq!(events[1].sequence, 3);
        assert_eq!(events[1].call_id, "c2");
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(InMemoryApprovalAudit::new(0).is_err());
    }

    #[test]
    fn clear_removes_all_events() {
        let audit = InMemoryApprovalAudit::new(4).unwrap();
        audit
            .record(ApprovalAuditEvent::tool_requested("c1", "build", ""))
            .unwrap();
        audit.clear().unwrap();
        assert!(audit.snapshot().unwrap().is_empty());
    }
}
