//! Strong type for one approval question/resolution pair.
//!
//! An [`ApprovalRequestId`] is independent from `call_id`, `session_id`,
//! audit sequence numbers and tool names: every NEW approval question gets a
//! fresh id, and the Requested/Resolved pair of that question SHARES the id.
//! `call_id` remains the correlation to the actual tool invocation.
//!
//! The id is a validated 32-hex-character string generated locally by the
//! runtime. An untrusted answerer can never supply an id: only
//! [`ApprovalRequestId::generate`] and [`ApprovalRequestId::parse`] create
//! values, and parsing validates the exact format.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Number of hex characters in one id (128 bits of entropy).
pub const APPROVAL_REQUEST_ID_CHARS: usize = 32;

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Fresh, validated identifier for one approval question.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ApprovalRequestId(String);

impl ApprovalRequestId {
    /// Generate a fresh id: 128 bits encoded as 32 lowercase hex characters —
    /// 64 bits from the platform RNG plus the process-local counter in the
    /// remaining 64 bits, so uniqueness holds even if the RNG is not.
    pub fn generate() -> Self {
        let random = random_u64();
        let counter = ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(format!("{:016x}{:016x}", random, counter))
    }

    /// Validate an untrusted wire value. Anything that is not exactly 32
    /// lowercase hex characters is rejected.
    pub fn parse(value: &str) -> Result<Self, ApprovalRequestIdError> {
        if value.len() != APPROVAL_REQUEST_ID_CHARS
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(ApprovalRequestIdError::Invalid(value.to_string()));
        }
        Ok(Self(value.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_valid(&self) -> bool {
        self.0.len() == APPROVAL_REQUEST_ID_CHARS
            && self
                .0
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    }
}

impl fmt::Display for ApprovalRequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn random_u64() -> u64 {
    // rand is already a scirust-sciagent dependency; thread_rng is
    // cryptographically seeded by the OS. This is not a secret value, but a
    // fresh nonce per question, so a CSPRNG keeps the id unguessable by an
    // answerer that wants to pre-register a resolution.
    use rand::Rng;
    rand::thread_rng().gen()
}

/// Validation failure for an untrusted approval request id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalRequestIdError {
    Invalid(String),
}

impl fmt::Display for ApprovalRequestIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Invalid(value) => write!(
                f,
                "invalid approval request id {value:?}: expected exactly 32 lowercase hex characters"
            ),
        }
    }
}

impl std::error::Error for ApprovalRequestIdError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_ids_are_unique_and_valid() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000
        {
            let id = ApprovalRequestId::generate();
            assert!(id.is_valid());
            assert_eq!(id.as_str().len(), APPROVAL_REQUEST_ID_CHARS);
            assert!(seen.insert(id.clone()), "duplicate id generated");
        }
    }

    #[test]
    fn parse_accepts_exact_lowercase_hex() {
        let value = "0123456789abcdef0123456789abcdef";
        let id = ApprovalRequestId::parse(value).unwrap();
        assert_eq!(id.as_str(), value);
    }

    #[test]
    fn parse_rejects_uppercase_wrong_length_and_symbols() {
        for value in [
            "0123456789ABCDEF0123456789ABCDEF",  // uppercase
            "0123456789abcdef0123456789abcde",   // 31 chars
            "0123456789abcdef0123456789abcdef0", // 33 chars
            "0123456789abcdef0123456789abcdegx", // non-hex
            " 0123456789abcdef0123456789abcdef", // leading space
        ]
        {
            assert!(ApprovalRequestId::parse(value).is_err(), "{value}");
        }
    }

    #[test]
    fn serialization_round_trip() {
        let id = ApprovalRequestId::generate();
        let json = serde_json::to_string(&id).unwrap();
        let back: ApprovalRequestId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn ids_are_independent_of_call_ids() {
        // Two questions about the same call still get different request ids.
        let a = ApprovalRequestId::generate();
        let b = ApprovalRequestId::generate();
        assert_ne!(a, b);
    }

    #[test]
    fn concurrent_generation_has_no_collisions() {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    let mut local = Vec::new();
                    for _ in 0..250
                    {
                        local.push(ApprovalRequestId::generate());
                    }
                    local
                })
            })
            .collect();
        let mut all = std::collections::HashSet::new();
        for handle in handles
        {
            for id in handle.join().unwrap()
            {
                assert!(all.insert(id), "concurrent collision");
            }
        }
        assert_eq!(all.len(), 8 * 250);
    }
}
