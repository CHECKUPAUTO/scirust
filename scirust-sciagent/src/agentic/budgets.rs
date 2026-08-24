//! Resource budgets and explicit egress policy for CCOS Enterprise.
//!
//! Filesystem sandboxing (Bubblewrap/Landlock) is NOT network isolation.
//! Egress is a separate axis: [`EgressPolicy`] defaults to deny-all and any
//! claim that a policy isolates network access while the enforcement backend
//! is unavailable fails closed. Resource limits are checked for
//! satisfiability BEFORE execution: a limit that cannot be enforced denies
//! the call instead of running unbounded.

use serde::{Deserialize, Serialize};

fn default_enforce() -> bool {
    true
}

/// One egress target: hostname or CIDR plus the allowed ports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EgressTarget {
    pub host: String,
    pub ports: Vec<u16>,
}

impl EgressTarget {
    pub fn new(host: impl Into<String>, ports: Vec<u16>) -> Self {
        Self {
            host: host.into(),
            ports,
        }
    }

    /// Whether a (host, port) pair is covered by this target. Host matching
    /// is exact or suffix (domain suffix with a leading dot).
    pub fn covers(&self, host: &str, port: u16) -> bool {
        let host_ok = self.host == host
            || (host.len() > self.host.len()
                && host.ends_with(&self.host)
                && host.as_bytes()[host.len() - self.host.len() - 1] == b'.');
        host_ok && (self.ports.is_empty() || self.ports.contains(&port))
    }
}

/// Explicit network egress policy. Default deny-all: any egress attempt not
/// covered by an allow target is refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EgressPolicy {
    /// True when the enforcement backend is present and authoritative.
    /// When false, claiming isolation is impossible: egress fails closed.
    /// Deserialization defaults to `true` so a wire payload that omits the
    /// field keeps the fail-closed deny-all posture instead of silently
    /// disabling enforcement.
    #[serde(default = "default_enforce")]
    pub enforce: bool,
    #[serde(default)]
    pub allow: Vec<EgressTarget>,
}

impl EgressPolicy {
    pub fn deny_all() -> Self {
        Self {
            enforce: true,
            allow: Vec::new(),
        }
    }

    pub fn with_targets(allow: Vec<EgressTarget>) -> Self {
        Self {
            enforce: true,
            allow,
        }
    }

    /// Without an enforcement backend the policy cannot be honored. Failing
    /// closed here is the only safe choice.
    pub fn without_backend(self) -> Self {
        Self {
            enforce: false,
            allow: self.allow,
        }
    }

    /// Authorize one egress attempt. `None` backend => deny; uncovered
    /// target => deny.
    pub fn authorize(&self, host: &str, port: u16) -> Result<(), String> {
        if !self.enforce
        {
            return Err(
                "egress policy cannot be enforced (no backend); refusing network access"
                    .to_string(),
            );
        }
        if self.allow.iter().any(|target| target.covers(host, port))
        {
            Ok(())
        }
        else
        {
            Err(format!(
                "egress policy denies {host}:{port} (not in allow list)"
            ))
        }
    }
}

/// Resource limits for one execution. `None` means no limit is set — the
/// enforcement layer must still be able to verify it can run without limits.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub wall_time_seconds: Option<u64>,
    pub max_memory_bytes: Option<u64>,
    pub max_cpus: Option<u32>,
    pub max_processes: Option<u32>,
    pub max_file_size_bytes: Option<u64>,
    pub gpu_memory_bytes: Option<u64>,
}

impl ResourceLimits {
    /// Enforceability check: a limit can only be honored when the enforcing
    /// backend is present. This seam lets a runtime declare what it can
    /// actually enforce; an unenforceable limit fails closed at check time.
    pub fn check_enforceable(
        &self,
        backend: &dyn ResourceBackend,
        egress: &EgressPolicy,
    ) -> Result<(), String> {
        if let Some(bytes) = self.max_memory_bytes
        {
            if !backend.supports_memory_limit(bytes)
            {
                return Err(format!(
                    "memory limit {bytes} bytes cannot be enforced by the active backend"
                ));
            }
        }
        if let Some(cpus) = self.max_cpus
        {
            if !backend.supports_cpu_limit(cpus)
            {
                return Err(format!(
                    "cpu limit {cpus} cannot be enforced by the active backend"
                ));
            }
        }
        if let Some(seconds) = self.wall_time_seconds
        {
            if !backend.supports_wall_time(seconds)
            {
                return Err(format!(
                    "wall-time limit {seconds}s cannot be enforced by the active backend"
                ));
            }
        }
        if let Some(processes) = self.max_processes
        {
            if !backend.supports_process_limit(processes)
            {
                return Err(format!(
                    "process limit {processes} cannot be enforced by the active backend"
                ));
            }
        }
        if let Some(bytes) = self.max_file_size_bytes
        {
            if !backend.supports_file_size_limit(bytes)
            {
                return Err(format!(
                    "file size limit {bytes} cannot be enforced by the active backend"
                ));
            }
        }
        if let Some(bytes) = self.gpu_memory_bytes
        {
            if !backend.supports_gpu_memory_limit(bytes)
            {
                return Err(format!(
                    "gpu memory limit {bytes} cannot be enforced by the active backend"
                ));
            }
        }
        // Network is a separate axis: even with no resource limits, an
        // unenforceable egress policy refuses execution.
        if !egress.enforce
        {
            return Err(
                "egress policy cannot be enforced (no backend); refusing execution".to_string(),
            );
        }
        Ok(())
    }
}

/// Capabilities of the enforcement backend (cgroups, rlimits, sandbox, ...).
pub trait ResourceBackend: Send + Sync {
    fn supports_memory_limit(&self, _bytes: u64) -> bool {
        false
    }
    fn supports_cpu_limit(&self, _cpus: u32) -> bool {
        false
    }
    fn supports_wall_time(&self, _seconds: u64) -> bool {
        false
    }
    fn supports_process_limit(&self, _processes: u32) -> bool {
        false
    }
    fn supports_file_size_limit(&self, _bytes: u64) -> bool {
        false
    }
    fn supports_gpu_memory_limit(&self, _bytes: u64) -> bool {
        false
    }
    /// Whether the backend can enforce a non-empty host/port allow-list.
    /// Kernel-level backends in this workspace can only deny whole address
    /// families (deny-all); per-host filtering needs a network namespace with
    /// a proxy or an nftables/ebpf setup, so the safe default is `false`.
    fn supports_egress_allow_list(&self) -> bool {
        false
    }
    /// Whether the backend can enforce deny-all network egress (blocking
    /// INET socket creation kernel-side). The safe default is `false`: a
    /// host with no enforcement capability must refuse governed execution
    /// instead of silently allowing network access.
    fn supports_egress_deny_all(&self) -> bool {
        false
    }
}

/// Minimal backend with no enforcement capability: every limit fails closed.
#[derive(Debug, Default)]
pub struct NoResourceBackend;

impl ResourceBackend for NoResourceBackend {}

/// Full backend that can enforce every limit.
#[derive(Debug, Default)]
pub struct FullResourceBackend;

impl ResourceBackend for FullResourceBackend {
    fn supports_memory_limit(&self, _bytes: u64) -> bool {
        true
    }
    fn supports_cpu_limit(&self, _cpus: u32) -> bool {
        true
    }
    fn supports_wall_time(&self, _seconds: u64) -> bool {
        true
    }
    fn supports_process_limit(&self, _processes: u32) -> bool {
        true
    }
    fn supports_file_size_limit(&self, _bytes: u64) -> bool {
        true
    }
    fn supports_gpu_memory_limit(&self, _bytes: u64) -> bool {
        true
    }
    fn supports_egress_allow_list(&self) -> bool {
        true
    }
    fn supports_egress_deny_all(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deny_all_refuses_every_egress() {
        let policy = EgressPolicy::deny_all();
        assert!(policy.authorize("example.com", 443).is_err());
        assert!(policy.authorize("10.0.0.1", 22).is_err());
    }

    #[test]
    fn allow_list_matches_host_and_ports() {
        let policy = EgressPolicy::with_targets(vec![EgressTarget::new("example.com", vec![443])]);
        assert!(policy.authorize("example.com", 443).is_ok());
        assert!(policy.authorize("example.com", 80).is_err());
        assert!(policy.authorize("other.com", 443).is_err());
    }

    #[test]
    fn domain_suffix_matches_subdomains() {
        let policy = EgressPolicy::with_targets(vec![EgressTarget::new("example.com", vec![])]);
        assert!(policy.authorize("api.example.com", 8443).is_ok());
        // Bare suffix without a dot separator is NOT a subdomain match.
        assert!(policy.authorize("notexample.com", 443).is_err());
    }

    #[test]
    fn without_backend_fails_closed() {
        let policy = EgressPolicy::with_targets(vec![EgressTarget::new("example.com", vec![443])])
            .without_backend();
        assert!(policy.authorize("example.com", 443).is_err());
    }

    #[test]
    fn no_backend_refuses_any_limit() {
        let limits = ResourceLimits {
            wall_time_seconds: Some(60),
            ..ResourceLimits::default()
        };
        let error = limits
            .check_enforceable(&NoResourceBackend, &EgressPolicy::deny_all())
            .expect_err("no backend must fail closed");
        assert!(error.contains("cannot be enforced"), "{error}");
    }

    #[test]
    fn full_backend_accepts_all_limits() {
        let limits = ResourceLimits {
            wall_time_seconds: Some(60),
            max_memory_bytes: Some(1 << 30),
            max_cpus: Some(4),
            max_processes: Some(16),
            max_file_size_bytes: Some(1 << 20),
            gpu_memory_bytes: Some(1 << 29),
        };
        assert!(
            limits
                .check_enforceable(&FullResourceBackend, &EgressPolicy::deny_all())
                .is_ok()
        );
    }

    #[test]
    fn unenforceable_egress_refuses_execution() {
        let limits = ResourceLimits::default();
        let error = limits
            .check_enforceable(
                &FullResourceBackend,
                &EgressPolicy::deny_all().without_backend(),
            )
            .expect_err("missing egress backend must refuse execution");
        assert!(error.contains("egress"), "{error}");
    }

    #[test]
    fn partial_backend_rejects_specific_limit() {
        struct CpuOnly;
        impl ResourceBackend for CpuOnly {
            fn supports_cpu_limit(&self, _cpus: u32) -> bool {
                true
            }
        }
        let limits = ResourceLimits {
            max_cpus: Some(2),
            max_processes: Some(4),
            ..ResourceLimits::default()
        };
        let error = limits
            .check_enforceable(&CpuOnly, &EgressPolicy::deny_all())
            .expect_err("process limit cannot be enforced by cpu-only backend");
        assert!(error.contains("process limit"), "{error}");
    }

    #[test]
    fn wire_payload_defaults_to_enforced_deny_all() {
        let egress: EgressPolicy = serde_json::from_str("{}").expect("empty payload must parse");
        assert!(egress.enforce, "missing enforce flag must default to true");
        assert!(egress.allow.is_empty());
        assert!(egress.authorize("example.com", 443).is_err());
    }

    #[test]
    fn limits_and_egress_round_trip_through_json() {
        let limits = ResourceLimits {
            wall_time_seconds: Some(12),
            max_memory_bytes: Some(1 << 20),
            ..ResourceLimits::default()
        };
        let egress = EgressPolicy::with_targets(vec![EgressTarget::new("example.com", vec![443])]);
        let limits_json = serde_json::to_string(&limits).unwrap();
        let egress_json = serde_json::to_string(&egress).unwrap();
        assert_eq!(
            serde_json::from_str::<ResourceLimits>(&limits_json).unwrap(),
            limits
        );
        assert_eq!(
            serde_json::from_str::<EgressPolicy>(&egress_json).unwrap(),
            egress
        );
    }

    #[test]
    fn explicit_false_enforce_flag_is_preserved() {
        let egress: EgressPolicy =
            serde_json::from_str(r#"{"enforce": false}"#).expect("payload must parse");
        assert!(!egress.enforce);
        assert!(egress.authorize("example.com", 443).is_err());
    }
}
