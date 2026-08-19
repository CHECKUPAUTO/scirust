//! Scoped secret capability management for CCOS Enterprise.
//!
//! Secrets are never injected wholesale into child processes or agent
//! contexts. A [`SecretHandle`] is an opaque id plus a description; the value
//! lives only inside a [`SecretStore`] and is handed out only through
//! [`SecretStore::resolve`], which requires an explicit, auditable
//! [`SecretGrant`] for the acting subject. Handles are printable and
//! serializable; values are not. Revocation is handled by removing the grant.

use std::collections::HashMap;

/// Strict secret identifier: 1..=64 ASCII alphanumeric plus `-` and `_`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SecretId(String);

impl SecretId {
    pub fn parse(value: &str) -> Result<Self, String> {
        let ok = !value.is_empty()
            && value.len() <= 64
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
        if ok
        {
            Ok(Self(value.to_string()))
        }
        else
        {
            Err(format!(
                "invalid secret id {value:?}: expected 1..=64 ASCII alphanumeric, '-' or '_'"
            ))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SecretId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque reference to one stored secret. Safe to log and to pass across
/// process boundaries: it never carries the value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SecretHandle {
    pub id: SecretId,
    pub description: String,
}

/// Explicit grant of one secret handle to one subject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretGrant {
    pub subject: String,
    pub handle: SecretHandle,
}

/// In-memory secret store. Values are retained only here; the public API
/// never returns a value without a matching grant.
#[derive(Debug, Default)]
pub struct SecretStore {
    secrets: HashMap<SecretId, (String, String)>, // id -> (value, description)
    grants: Vec<SecretGrant>,
}

impl SecretStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a secret and return its opaque handle. The caller keeps the
    /// value; the store keeps the authoritative copy.
    pub fn register(
        &mut self,
        id: SecretId,
        value: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<SecretHandle, String> {
        if self.secrets.contains_key(&id)
        {
            return Err(format!("secret {} is already registered", id));
        }
        let description = description.into();
        self.secrets
            .insert(id.clone(), (value.into(), description.clone()));
        Ok(SecretHandle { id, description })
    }

    /// Grant a handle to a subject. Grants are explicit and auditable.
    pub fn grant(
        &mut self,
        subject: impl Into<String>,
        handle: &SecretHandle,
    ) -> Result<(), String> {
        if !self.secrets.contains_key(&handle.id)
        {
            return Err(format!("cannot grant unknown secret {}", handle.id));
        }
        self.grants.push(SecretGrant {
            subject: subject.into(),
            handle: handle.clone(),
        });
        Ok(())
    }

    /// Revoke every grant of a handle. The secret stays registered but no
    /// subject can resolve it anymore.
    pub fn revoke(&mut self, handle: &SecretHandle) {
        self.grants.retain(|grant| grant.handle.id != handle.id);
    }

    /// Resolve the value for a subject. Returns an error when the subject
    /// holds no grant for this handle — the value is never disclosed
    /// otherwise.
    pub fn resolve(&self, subject: &str, handle: &SecretHandle) -> Result<String, String> {
        let granted = self
            .grants
            .iter()
            .any(|grant| grant.subject == subject && grant.handle.id == handle.id);
        if !granted
        {
            return Err(format!(
                "subject {subject:?} has no grant for secret {}",
                handle.id
            ));
        }
        self.secrets
            .get(&handle.id)
            .map(|(value, _)| value.clone())
            .ok_or_else(|| format!("secret {} is not registered", handle.id))
    }

    /// All grants (audit view). Handles only, never values.
    pub fn grants(&self) -> &[SecretGrant] {
        &self.grants
    }

    /// Registered secret descriptions (audit view). Values never leave the
    /// store through this API.
    pub fn descriptions(&self) -> Vec<(SecretId, String)> {
        self.secrets
            .iter()
            .map(|(id, (_, description))| (id.clone(), description.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api_key_id() -> SecretId {
        SecretId::parse("api-key").unwrap()
    }

    #[test]
    fn id_validation_rejects_malformed() {
        assert!(SecretId::parse("api-key").is_ok());
        assert!(SecretId::parse("").is_err());
        assert!(SecretId::parse("has space").is_err());
        assert!(SecretId::parse("has/slash").is_err());
        assert!(SecretId::parse(&"x".repeat(65)).is_err());
    }

    #[test]
    fn resolve_requires_explicit_grant() {
        let mut store = SecretStore::new();
        let handle = store
            .register(api_key_id(), "s3cr3t", "api key for production")
            .unwrap();
        let error = store
            .resolve("alice", &handle)
            .expect_err("no grant must refuse resolution");
        assert!(error.contains("no grant"), "{error}");
    }

    #[test]
    fn grant_enables_resolution_for_that_subject_only() {
        let mut store = SecretStore::new();
        let handle = store.register(api_key_id(), "s3cr3t", "api key").unwrap();
        store.grant("alice", &handle).unwrap();
        assert_eq!(store.resolve("alice", &handle).unwrap(), "s3cr3t");
        let error = store.resolve("bob", &handle).expect_err("bob has no grant");
        assert!(error.contains("no grant"), "{error}");
    }

    #[test]
    fn grant_of_unknown_secret_is_refused() {
        let mut store = SecretStore::new();
        let ghost = SecretHandle {
            id: SecretId::parse("ghost").unwrap(),
            description: "never registered".to_string(),
        };
        assert!(store.grant("alice", &ghost).is_err());
    }

    #[test]
    fn revoke_removes_access() {
        let mut store = SecretStore::new();
        let handle = store.register(api_key_id(), "s3cr3t", "api key").unwrap();
        store.grant("alice", &handle).unwrap();
        assert!(store.resolve("alice", &handle).is_ok());
        store.revoke(&handle);
        assert!(store.resolve("alice", &handle).is_err());
    }

    #[test]
    fn duplicate_registration_is_refused() {
        let mut store = SecretStore::new();
        store.register(api_key_id(), "one", "first").unwrap();
        assert!(store.register(api_key_id(), "two", "second").is_err());
    }

    #[test]
    fn audit_views_never_contain_values() {
        let mut store = SecretStore::new();
        let handle = store
            .register(api_key_id(), "s3cr3t-value", "api key")
            .unwrap();
        store.grant("alice", &handle).unwrap();
        for (id, description) in store.descriptions()
        {
            assert_eq!(id, api_key_id());
            assert_eq!(description, "api key");
        }
        assert_eq!(store.grants().len(), 1);
        assert_eq!(store.grants()[0].subject, "alice");
    }

    #[test]
    fn handle_display_never_leaks_value() {
        let handle = SecretHandle {
            id: api_key_id(),
            description: "api key".to_string(),
        };
        let rendered = format!("{handle:?}");
        assert!(!rendered.contains("s3cr3t"));
        assert!(rendered.contains("api-key"));
    }
}
