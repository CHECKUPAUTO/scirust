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
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// Emission seam for automatic runtime auditing.
///
/// Sinks own sequencing: callers describe one correlated execution and the
/// sink assigns the next sequence number and chains it. A sink that cannot
/// durably record an execution must fail closed at its call site.
pub trait EnterpriseAuditSink: Send + Sync {
    // The correlation record IS ten fields wide; splitting it into structs
    // would hide which combinations are legal from the type system's most
    // verbose but honest expression.
    #[allow(clippy::too_many_arguments)]
    fn record_execution(
        &self,
        tenant: &TenantId,
        subject: &str,
        session_id: &str,
        request_id: Option<&ApprovalRequestId>,
        call_id: &str,
        tool: &str,
        sandbox: &str,
        decision: &str,
        output_digest: &str,
    ) -> Result<(), String>;
}

impl EnterpriseAuditTrail {
    fn append_chained(&self, mut event: EnterpriseAuditEvent) -> Result<u64, String> {
        let mut events = self
            .events
            .lock()
            .map_err(|_| "enterprise audit trail is unavailable".to_string())?;
        let sequence = events.len() as u64;
        event.sequence = sequence;
        event.prev_hash = events
            .last()
            .map(|last| last.chain_hash.clone())
            .unwrap_or_else(|| ENTERPRISE_AUDIT_GENESIS.to_string());
        event.chain_hash = event.compute_chain_hash();
        events.push(event);
        Ok(sequence)
    }
}

impl EnterpriseAuditSink for EnterpriseAuditTrail {
    #[allow(clippy::too_many_arguments)]
    fn record_execution(
        &self,
        tenant: &TenantId,
        subject: &str,
        session_id: &str,
        request_id: Option<&ApprovalRequestId>,
        call_id: &str,
        tool: &str,
        sandbox: &str,
        decision: &str,
        output_digest: &str,
    ) -> Result<(), String> {
        let sequence = self.len()? as u64;
        let mut event = EnterpriseAuditEvent::new(sequence, tenant.clone(), subject, session_id);
        if let Some(request_id) = request_id
        {
            event = event.with_approval(request_id.clone(), decision);
        }
        event = event.with_execution(call_id, tool, sandbox, output_digest);
        // The closed decision vocabulary rides on `decision` even without an
        // approval correlation so every executed call is auditable alone.
        event.decision = Some(decision.to_string());
        event.chain_hash = String::new();
        self.append_chained(event)?;
        Ok(())
    }
}

/// Durable counterpart of [`EnterpriseAuditTrail`]: one JSON line per event,
/// SHA-256 chained exactly like the in-memory trail, verified on every read.
/// A corrupt log fails closed — replay surfaces the corruption and appends
/// are refused until the operator replaces the file.
#[derive(Debug)]
pub struct FileEnterpriseAuditTrail {
    path: std::path::PathBuf,
    transaction: std::sync::Mutex<()>,
    max_segment_bytes: Option<u64>,
}

// Advisory whole-log writer lock, stable across segment rotations. It lives
// on `<path>.lock` — never on a data segment — because rotation renames
// segments and flock follows inodes. Raw syscall like the rest of this
// crate; on platforms without it the in-process transaction mutex still
// guards every operation and the single-writer-per-path caveat is
// documented instead of pretended away.
#[cfg(unix)]
unsafe extern "C" {
    fn syscall(number: std::os::raw::c_long, ...) -> std::os::raw::c_long;
}

#[cfg(unix)]
struct LogFileLock<'a> {
    file: std::fs::File,
    _marker: std::marker::PhantomData<&'a ()>,
}

#[cfg(not(unix))]
struct LogFileLock<'a> {
    file: std::fs::File,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> LogFileLock<'a> {
    fn acquire(path: &std::path::Path) -> Result<Self, String> {
        let lock_path = lock_path_for(path);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .read(true)
            .open(&lock_path)
            .map_err(|error| {
                format!(
                    "enterprise audit lock {} cannot be opened: {error}",
                    lock_path.display()
                )
            })?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            const LOCK_EX: i32 = 2;
            let status =
                unsafe { syscall(SYS_FLOCK, file.as_raw_fd(), LOCK_EX as std::os::raw::c_uint) };
            if status != 0
            {
                return Err(format!(
                    "enterprise audit lock {} cannot be acquired: {}",
                    lock_path.display(),
                    std::io::Error::last_os_error()
                ));
            }
        }
        Ok(Self {
            file,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<'a> Drop for LogFileLock<'a> {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            unsafe {
                let _ = syscall(
                    SYS_FLOCK,
                    self.file.as_raw_fd(),
                    LOCK_UN as std::os::raw::c_uint,
                );
            }
        }
    }
}

// Syscall numbers diverge between the x86_64 historical table and the
// asm-generic table used by aarch64 (the Jetson Thor target).
#[cfg(all(unix, target_arch = "x86_64"))]
const SYS_FLOCK: std::os::raw::c_long = 73;
#[cfg(all(unix, target_arch = "aarch64"))]
const SYS_FLOCK: std::os::raw::c_long = 32;
#[cfg(unix)]
const LOCK_UN: i32 = 8;

fn next_rotation_index(rotated: &[std::path::PathBuf]) -> u64 {
    rotated
        .iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_string_lossy().into_owned();
            name.rsplit(".r").next()?.parse::<u64>().ok()
        })
        .max()
        .unwrap_or(0)
        + 1
}

fn rotated_path_for(path: &std::path::Path, index: u64) -> std::path::PathBuf {
    let mut owned = path.as_os_str().to_owned();
    owned.push(format!(".r{index:06}"));
    std::path::PathBuf::from(owned)
}

fn lock_path_for(path: &std::path::Path) -> std::path::PathBuf {
    let mut owned = path.as_os_str().to_owned();
    owned.push(".lock");
    std::path::PathBuf::from(owned)
}

impl FileEnterpriseAuditTrail {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            path: path.into(),
            transaction: std::sync::Mutex::new(()),
            max_segment_bytes: None,
        }
    }

    /// Rotate the active segment once it reaches `max_segment_bytes`, so a
    /// single JSONL chain cannot grow without bound.
    ///
    /// Every rotated segment is an independent SHA-256 chain starting at the
    /// genesis hash while event sequences keep counting globally, so replay
    /// order stays total and tamper evidence is per-segment.
    pub fn with_rotation(mut self, max_segment_bytes: u64) -> Self {
        if max_segment_bytes > 0
        {
            self.max_segment_bytes = Some(max_segment_bytes);
        }
        self
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    fn lock_transaction(&self) -> Result<std::sync::MutexGuard<'_, ()>, String> {
        self.transaction.lock().map_err(|_| {
            format!(
                "enterprise audit log {} transaction lock is unavailable",
                self.path.display()
            )
        })
    }

    /// Rotated segments in deterministic replay order (oldest first).
    fn rotated_segments(&self) -> Result<Vec<std::path::PathBuf>, String> {
        let directory = match self.path.parent()
        {
            Some(parent) if parent.as_os_str().is_empty() => std::path::PathBuf::from("."),
            Some(parent) => parent.to_path_buf(),
            None => std::path::PathBuf::from("."),
        };
        let prefix = format!("{}.r", self.file_label()?);
        let mut numbered: Vec<(u64, std::path::PathBuf)> = Vec::new();
        let entries = std::fs::read_dir(&directory).map_err(|error| {
            format!(
                "enterprise audit dir {} unreadable: {error}",
                directory.display()
            )
        })?;
        for entry in entries
        {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(suffix) = name.strip_prefix(&prefix)
            {
                let index: u64 = suffix.parse().map_err(|_| {
                    format!("enterprise audit segment {} has a malformed index", name)
                })?;
                numbered.push((index, entry.path()));
            }
        }
        numbered.sort_by_key(|(index, _)| *index);
        // Duplicate indexes would make replay order ambiguous — refuse.
        for window in numbered.windows(2)
        {
            if window[0].0 == window[1].0
            {
                return Err(format!(
                    "enterprise audit has duplicate segment index {}",
                    window[0].0
                ));
            }
        }
        Ok(numbered.into_iter().map(|(_, path)| path).collect())
    }

    fn file_label(&self) -> Result<String, String> {
        self.path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .ok_or_else(|| "enterprise audit path has no file name".to_string())
    }

    /// Load every segment (rotated oldest-first, then active), verifying
    /// each segment's independent genesis-rooted chain and the global
    /// sequence continuity across segments.
    ///
    /// Returns the full event stream plus the active segment's tail hash so
    /// an append can continue exactly where this segment ends.
    fn load_unlocked(&self) -> Result<(Vec<EnterpriseAuditEvent>, Option<String>), String> {
        let mut paths = self.rotated_segments()?;
        paths.push(self.path.clone());
        let mut events = Vec::new();
        let mut active_tail: Option<String> = None;
        for path in &paths
        {
            let file = match std::fs::File::open(path)
            {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound =>
                {
                    if *path == self.path
                    {
                        continue;
                    }
                    return Err(format!(
                        "enterprise audit segment {} disappeared mid-replay",
                        path.display()
                    ));
                },
                Err(error) =>
                {
                    return Err(format!(
                        "enterprise audit log {} cannot be opened: {error}",
                        path.display()
                    ));
                },
            };
            // Every segment roots its own chain at genesis; sequences keep
            // counting globally across rotations.
            let mut expected_prev = ENTERPRISE_AUDIT_GENESIS.to_string();
            for (index, line) in std::io::BufRead::lines(std::io::BufReader::new(file)).enumerate()
            {
                let line = line.map_err(|error| {
                    format!(
                        "enterprise audit log {} is unreadable at line {}: {error}",
                        path.display(),
                        index + 1
                    )
                })?;
                if line.trim().is_empty()
                {
                    continue;
                }
                let event: EnterpriseAuditEvent =
                    serde_json::from_str(line.trim()).map_err(|error| {
                        format!(
                            "enterprise audit log {} is corrupt at line {}: {error}",
                            path.display(),
                            index + 1
                        )
                    })?;
                if event.sequence != events.len() as u64
                {
                    return Err(format!(
                        "enterprise audit log {} has a sequence gap at line {}",
                        path.display(),
                        index + 1
                    ));
                }
                if event.prev_hash != expected_prev || !event.verify_chain_hash()
                {
                    return Err(format!(
                        "enterprise audit log {} failed chain verification at line {}",
                        path.display(),
                        index + 1
                    ));
                }
                expected_prev = event.chain_hash.clone();
                events.push(event);
            }
            active_tail = if expected_prev == ENTERPRISE_AUDIT_GENESIS && *path == self.path
            {
                None
            }
            else
            {
                Some(expected_prev)
            };
        }
        Ok((events, active_tail))
    }

    /// Replay the full durable trail in order. Corruption fails closed.
    pub fn replay(&self) -> Result<Vec<EnterpriseAuditEvent>, String> {
        let _transaction = self.lock_transaction()?;
        let _writer_lock = LogFileLock::acquire(&self.path)?;
        Ok(self.load_unlocked()?.0)
    }

    /// Verify every segment chain and the global sequence continuity.
    pub fn verify(&self) -> Result<(), String> {
        let _transaction = self.lock_transaction()?;
        let _writer_lock = LogFileLock::acquire(&self.path)?;
        self.load_unlocked()?;
        Ok(())
    }
}

impl EnterpriseAuditSink for FileEnterpriseAuditTrail {
    #[allow(clippy::too_many_arguments)]
    fn record_execution(
        &self,
        tenant: &TenantId,
        subject: &str,
        session_id: &str,
        request_id: Option<&ApprovalRequestId>,
        call_id: &str,
        tool: &str,
        sandbox: &str,
        decision: &str,
        output_digest: &str,
    ) -> Result<(), String> {
        // Atomic for concurrent calls sharing this sink instance, and the
        // advisory lock extends that serialization across processes.
        let _transaction = self.lock_transaction()?;
        let _writer_lock = LogFileLock::acquire(&self.path)?;
        let (events, active_tail) = self.load_unlocked()?;

        // Rotate before appending when the active segment reached its cap:
        // the fresh segment roots a brand-new genesis chain while global
        // sequences keep counting (event.sequence = events.len()).
        let active_size = std::fs::metadata(&self.path)
            .map(|meta| meta.len())
            .unwrap_or(0);
        let rotate = matches!(self.max_segment_bytes, Some(limit) if active_size >= limit);
        let prev_hash = if rotate
        {
            let rotated =
                rotated_path_for(&self.path, next_rotation_index(&self.rotated_segments()?));
            std::fs::rename(&self.path, &rotated).map_err(|error| {
                format!(
                    "enterprise audit rotation {} -> {} failed: {error}",
                    self.path.display(),
                    rotated.display()
                )
            })?;
            ENTERPRISE_AUDIT_GENESIS.to_string()
        }
        else
        {
            active_tail.unwrap_or_else(|| ENTERPRISE_AUDIT_GENESIS.to_string())
        };

        let mut event =
            EnterpriseAuditEvent::new(events.len() as u64, tenant.clone(), subject, session_id);
        if let Some(request_id) = request_id
        {
            event = event.with_approval(request_id.clone(), decision);
        }
        event.decision = Some(decision.to_string());
        event = event.with_execution(call_id, tool, sandbox, output_digest);
        event.prev_hash = prev_hash;
        event.chain_hash = event.compute_chain_hash();

        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|error| {
                format!(
                    "enterprise audit log {} cannot be opened for append: {error}",
                    self.path.display()
                )
            })?;
        let line = serde_json::to_string(&event)
            .map_err(|error| format!("enterprise audit event cannot serialize: {error}"))?;
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.flush())
            .map_err(|error| {
                format!(
                    "enterprise audit log {} append failed: {error}",
                    self.path.display()
                )
            })?;
        Ok(())
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

    // -- Durable enterprise audit ------------------------------------------

    use std::path::PathBuf;

    /// Isolated per-test directory UNDER THE CRATE TARGET, not `/tmp`: the
    /// audit chain must stay verifiable even on machines whose tmpfs is
    /// unstable, and parallel tests never share a parent they could clean
    /// out from under each other.
    fn temp_log() -> PathBuf {
        let run_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/test-tmp")
            .join(format!(
                "enterprise-audit-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ));
        std::fs::create_dir_all(&run_dir).expect("test-tmp must be creatable");
        run_dir.join("audit.jsonl")
    }

    fn record_two(sink: &dyn EnterpriseAuditSink, shared: &ApprovalRequestId) {
        sink.record_execution(
            &tenant(),
            "alice",
            "session-1",
            None,
            "call-1",
            "build",
            "workspace-write",
            "executed",
            "digest-1",
        )
        .unwrap();
        sink.record_execution(
            &tenant(),
            "alice",
            "session-1",
            Some(shared),
            "call-2",
            "test",
            "workspace-write",
            "executed",
            "digest-2",
        )
        .unwrap();
    }

    #[test]
    fn durable_append_replay_round_trip() {
        let path = temp_log();
        let store = FileEnterpriseAuditTrail::new(&path);
        record_two(&store, &ApprovalRequestId::generate());
        let events = store.replay().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].tool.as_deref(), Some("build"));
        assert!(events[1].request_id.is_some());
        assert_eq!(events[1].decision.as_deref(), Some("executed"));
        assert_eq!(
            events[1].artifact_digest.as_deref(),
            None,
            "no artifact was declared"
        );
        store.verify().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn restart_replays_exact_trail_and_continues_the_chain() {
        let path = temp_log();
        {
            let store = FileEnterpriseAuditTrail::new(&path);
            record_two(&store, &ApprovalRequestId::generate());
            let first = store.replay().unwrap();
            drop(store);
            // A brand-new instance over the same file continues seamlessly.
            let reopened = FileEnterpriseAuditTrail::new(&path);
            let replayed = reopened.replay().unwrap();
            assert_eq!(replayed, first);
            reopened
                .record_execution(
                    &tenant(),
                    "alice",
                    "session-1",
                    None,
                    "call-3",
                    "status",
                    "read-only",
                    "executed",
                    "digest-3",
                )
                .unwrap();
            let grown = reopened.replay().unwrap();
            assert_eq!(grown.len(), 3);
            assert_eq!(grown[2].prev_hash, replayed[1].chain_hash);
            reopened.verify().unwrap();
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn torn_tail_fails_closed() {
        let path = temp_log();
        {
            let store = FileEnterpriseAuditTrail::new(&path);
            record_two(&store, &ApprovalRequestId::generate());
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let cut = content.len() - 12;
        std::fs::write(&path, &content[..cut]).unwrap();
        let store = FileEnterpriseAuditTrail::new(&path);
        let error = store.replay().expect_err("torn tail must fail closed");
        assert!(error.contains("corrupt"), "{error}");
        // Appending onto a corrupt log is refused too.
        assert!(
            store
                .record_execution(
                    &tenant(),
                    "alice",
                    "session-1",
                    None,
                    "call-x",
                    "build",
                    "workspace-write",
                    "executed",
                    "d"
                )
                .is_err()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tampered_field_breaks_the_chain() {
        let path = temp_log();
        {
            let store = FileEnterpriseAuditTrail::new(&path);
            record_two(&store, &ApprovalRequestId::generate());
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let tampered = content.replace("call-1", "evil");
        std::fs::write(&path, tampered).unwrap();
        let store = FileEnterpriseAuditTrail::new(&path);
        let error = store.replay().expect_err("tampering must fail closed");
        assert!(error.contains("chain verification"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn failed_append_does_not_record_authority() {
        let path = temp_log();
        let store = FileEnterpriseAuditTrail::new(&path);
        record_two(&store, &ApprovalRequestId::generate());
        // A directory cannot be appended to as a log file.
        let dir = temp_log();
        std::fs::create_dir_all(&dir).unwrap();
        let broken = FileEnterpriseAuditTrail::new(&dir);
        assert!(
            broken
                .record_execution(
                    &tenant(),
                    "alice",
                    "session-1",
                    None,
                    "call-y",
                    "build",
                    "workspace-write",
                    "executed",
                    "d"
                )
                .is_err()
        );
        assert_eq!(store.replay().unwrap().len(), 2);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_log_replays_empty() {
        let path = temp_log();
        let store = FileEnterpriseAuditTrail::new(&path);
        assert!(store.replay().unwrap().is_empty());
        store.verify().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn durable_replay_revalidates_tenant_id() {
        let path = temp_log();
        let valid = serde_json::to_string(&base(0)).unwrap();
        let invalid = valid.replace("\"acme\"", "\"bad/tenant\"");
        std::fs::write(&path, format!("{invalid}\n")).unwrap();
        let store = FileEnterpriseAuditTrail::new(&path);
        let error = store.replay().expect_err("invalid tenant must fail");
        assert!(error.contains("invalid tenant id"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn shared_file_sink_serializes_concurrent_appends() {
        const WORKERS: usize = 16;
        let path = temp_log();
        let store = std::sync::Arc::new(FileEnterpriseAuditTrail::new(&path));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(WORKERS));
        let mut handles = Vec::with_capacity(WORKERS);
        for index in 0..WORKERS
        {
            let store = std::sync::Arc::clone(&store);
            let barrier = std::sync::Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                store
                    .record_execution(
                        &TenantId::parse("acme").unwrap(),
                        "worker",
                        "session-concurrent",
                        None,
                        &format!("call-{index}"),
                        "build",
                        "workspace-write",
                        "executed",
                        &format!("digest-{index}"),
                    )
                    .unwrap();
            }));
        }
        for handle in handles
        {
            handle.join().unwrap();
        }
        let events = store.replay().unwrap();
        assert_eq!(events.len(), WORKERS);
        for (index, event) in events.iter().enumerate()
        {
            assert_eq!(event.sequence, index as u64);
            let expected = if index == 0
            {
                ENTERPRISE_AUDIT_GENESIS
            }
            else
            {
                events[index - 1].chain_hash.as_str()
            };
            assert_eq!(event.prev_hash, expected);
        }
        store.verify().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn memory_and_file_sinks_produce_identical_chains() {
        let path = temp_log();
        let memory = EnterpriseAuditTrail::new();
        let durable = FileEnterpriseAuditTrail::new(&path);
        let shared = ApprovalRequestId::generate();
        record_two(&memory, &shared);
        record_two(&durable, &shared);
        let mem_events = memory.events().unwrap();
        let file_events = durable.replay().unwrap();
        assert_eq!(mem_events, file_events, "chain hashes must be identical");
        let _ = std::fs::remove_file(&path);
    }

    // -- Segment rotation ----------------------------------------------------

    fn record_n(sink: &dyn EnterpriseAuditSink, count: usize, tag: &str) {
        for index in 0..count
        {
            sink.record_execution(
                &tenant(),
                "rotator",
                "session-rotation",
                None,
                &format!("call-{tag}-{index}"),
                "build",
                "workspace-write",
                "executed",
                &format!("digest-{tag}-{index}"),
            )
            .unwrap();
        }
    }

    #[test]
    fn rotation_produces_independent_verified_segments() {
        let path = temp_log();
        let store = FileEnterpriseAuditTrail::new(&path).with_rotation(1);
        // Every append after the first rotates: threshold 1 byte is always
        // exceeded by the previous line.
        record_n(&store, 5, "r");

        let rotated = store.rotated_segments().unwrap();
        assert_eq!(rotated.len(), 4, "four rotations for five appends");
        let events = store.replay().unwrap();
        assert_eq!(events.len(), 5);
        for (index, event) in events.iter().enumerate()
        {
            assert_eq!(event.sequence, index as u64, "global continuity");
        }
        assert_eq!(
            events[3].prev_hash, ENTERPRISE_AUDIT_GENESIS,
            "a fresh segment roots a new chain"
        );
        store.verify().unwrap();

        // A reopened store continues the LAST segment and keeps replay order.
        let reopened = FileEnterpriseAuditTrail::new(&path);
        record_n(&reopened, 1, "after-restart");
        let events = reopened.replay().unwrap();
        assert_eq!(events.len(), 6);
        assert_eq!(events[5].call_id.as_deref(), Some("call-after-restart-0"));
        reopened.verify().unwrap();
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn no_rotation_keeps_a_single_segment() {
        let path = temp_log();
        let store = FileEnterpriseAuditTrail::new(&path).with_rotation(u64::MAX);
        record_n(&store, 6, "n");
        assert!(store.rotated_segments().unwrap().is_empty());
        assert_eq!(store.replay().unwrap().len(), 6);
        store.verify().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tampered_rotated_segment_is_detected() {
        let path = temp_log();
        let store = FileEnterpriseAuditTrail::new(&path).with_rotation(1);
        record_n(&store, 4, "t");
        let first_rotated = store.rotated_segments().unwrap()[0].clone();
        let content = std::fs::read_to_string(&first_rotated).unwrap();
        std::fs::write(&first_rotated, content.replace("build", "evil")).unwrap();
        let error = store
            .replay()
            .expect_err("tampering a rotated segment must fail closed");
        assert!(error.contains("chain verification"), "{error}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn separate_instances_serialize_through_the_advisory_lock() {
        const WRITERS: usize = 8;
        const PER_WRITER: usize = 12;
        let path = temp_log();
        let path = std::sync::Arc::new(path);
        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::new();
        for writer in 0..WRITERS
        {
            let path = std::sync::Arc::clone(&path);
            handles.push(std::thread::spawn(move || {
                // Each thread owns its own instance over the SAME path:
                // serialization can only come from the advisory lock.
                let store = FileEnterpriseAuditTrail::new((*path).clone()).with_rotation(256);
                for index in 0..PER_WRITER
                {
                    store
                        .record_execution(
                            &TenantId::parse("acme").unwrap(),
                            &format!("writer-{writer}"),
                            "session-concurrent",
                            None,
                            &format!("call-{writer}-{index}"),
                            "build",
                            "workspace-write",
                            "executed",
                            "d",
                        )
                        .unwrap();
                }
            }));
        }
        for handle in handles
        {
            handle.join().unwrap();
        }
        let verifier = FileEnterpriseAuditTrail::new(&*path);
        let events = verifier.replay().unwrap();
        assert_eq!(events.len(), WRITERS * PER_WRITER);
        for (index, event) in events.iter().enumerate()
        {
            assert_eq!(event.sequence, index as u64);
        }
        verifier.verify().unwrap();
        let keep = std::env::var("KEEP_SEGMENTS").is_ok();
        if !keep
        {
            std::fs::remove_dir_all(path.parent().unwrap()).ok();
        }
    }

    #[test]
    fn sink_sequence_gaps_are_impossible_from_the_seam() {
        let trail = EnterpriseAuditTrail::new();
        trail
            .record_execution(
                &tenant(),
                "bob",
                "s",
                None,
                "c",
                "search",
                "read-only",
                "executed",
                "d",
            )
            .unwrap();
        let events = trail.events().unwrap();
        assert_eq!(events[0].sequence, 0);
        assert_eq!(events[0].subject, "bob");
        trail.verify().unwrap();
    }
}
