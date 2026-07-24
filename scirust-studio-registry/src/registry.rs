//! The registry that catalogues [`CapabilityDescriptor`]s.

use crate::descriptor::{CapabilityCategory, CapabilityDescriptor, CapabilityId};

/// An error registering a capability. The registry is left unchanged on
/// error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryError {
    /// Another capability already registered this exact id.
    DuplicateId(CapabilityId),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            RegistryError::DuplicateId(id) =>
            {
                write!(f, "capability id `{id}` is already registered")
            },
        }
    }
}

impl std::error::Error for RegistryError {}

/// A registry of [`CapabilityDescriptor`]s, always kept sorted by
/// [`CapabilityId`] so every read path (`iter`, `to_text`, `to_json`) is
/// deterministic regardless of registration order.
///
/// This type does not, by itself, guarantee "never advertise a capability
/// without a tested executable adapter" — that guarantee comes from *who is
/// allowed to construct a descriptor to register*: `scirust-studio-runtime`
/// only exposes descriptors through its `CapabilityAdapter::descriptor()`
/// trait method, each implemented by a real, tested adapter. This registry
/// has no path to accept a bare id or name.
#[derive(Debug, Clone, Default)]
pub struct CapabilityRegistry {
    entries: Vec<&'static CapabilityDescriptor>,
}

impl CapabilityRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        CapabilityRegistry::default()
    }

    /// Register a capability descriptor. Rejects a duplicate id, leaving the
    /// registry unchanged.
    pub fn register(
        &mut self,
        descriptor: &'static CapabilityDescriptor,
    ) -> Result<(), RegistryError> {
        if self.entries.iter().any(|d| d.id == descriptor.id)
        {
            return Err(RegistryError::DuplicateId(descriptor.id));
        }
        self.entries.push(descriptor);
        self.entries.sort_by_key(|d| d.id);
        Ok(())
    }

    /// Every registered capability, in deterministic (id-sorted) order.
    pub fn iter(&self) -> impl Iterator<Item = &'static CapabilityDescriptor> + '_ {
        self.entries.iter().copied()
    }

    /// Number of registered capabilities.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry has no capabilities.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a capability by its exact id string.
    pub fn find(&self, id: &str) -> Option<&'static CapabilityDescriptor> {
        self.entries.iter().copied().find(|d| d.id.0 == id)
    }

    /// Every capability in a given category, in deterministic order.
    pub fn by_category(&self, category: CapabilityCategory) -> Vec<&'static CapabilityDescriptor> {
        self.entries
            .iter()
            .copied()
            .filter(|d| d.category == category)
            .collect()
    }

    /// A deterministic, human-readable text catalogue: one line per
    /// capability, id-sorted, `<id> — <summary>`.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for d in &self.entries
        {
            out.push_str(d.id.0);
            out.push_str(" — ");
            out.push_str(d.summary);
            out.push('\n');
        }
        out
    }

    /// A deterministic JSON catalogue: an array of full descriptors,
    /// id-sorted. Stable field names and ordering, suitable for tests and
    /// future GUI consumption.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{
        BackendKind, CapabilityMaturity, DeterminismClass, PrecisionKind, VerificationDescriptor,
    };

    #[test]
    fn registers_and_finds_by_id() {
        static A: CapabilityDescriptor = CapabilityDescriptor {
            id: CapabilityId("a"),
            display_name: "A",
            category: CapabilityCategory::Mechanics,
            source_crate: "scirust-sim",
            summary: "s",
            maturity: CapabilityMaturity::Stable,
            determinism: DeterminismClass::StrictSameBinarySameTarget,
            supported_backends: &[BackendKind::Cpu],
            supported_precisions: &[PrecisionKind::F64],
            supported_solvers: &[],
            parameters: &[],
            initial_state: &[],
            outputs: &[],
            verification: VerificationDescriptor { checks: &[] },
        };
        let mut reg = CapabilityRegistry::new();
        reg.register(&A).unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.find("a").unwrap().id, CapabilityId("a"));
        assert!(reg.find("nonexistent").is_none());
    }

    #[test]
    fn rejects_duplicate_id() {
        static A: CapabilityDescriptor = CapabilityDescriptor {
            id: CapabilityId("dup"),
            display_name: "A",
            category: CapabilityCategory::Mechanics,
            source_crate: "scirust-sim",
            summary: "s",
            maturity: CapabilityMaturity::Stable,
            determinism: DeterminismClass::StrictSameBinarySameTarget,
            supported_backends: &[BackendKind::Cpu],
            supported_precisions: &[PrecisionKind::F64],
            supported_solvers: &[],
            parameters: &[],
            initial_state: &[],
            outputs: &[],
            verification: VerificationDescriptor { checks: &[] },
        };
        let mut reg = CapabilityRegistry::new();
        reg.register(&A).unwrap();
        let err = reg.register(&A).unwrap_err();
        assert_eq!(err, RegistryError::DuplicateId(CapabilityId("dup")));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn iteration_and_text_catalogue_are_sorted_by_id_regardless_of_registration_order() {
        static ZEBRA: CapabilityDescriptor = CapabilityDescriptor {
            id: CapabilityId("zebra"),
            display_name: "Zebra",
            category: CapabilityCategory::Mechanics,
            source_crate: "scirust-sim",
            summary: "z summary",
            maturity: CapabilityMaturity::Stable,
            determinism: DeterminismClass::StrictSameBinarySameTarget,
            supported_backends: &[BackendKind::Cpu],
            supported_precisions: &[PrecisionKind::F64],
            supported_solvers: &[],
            parameters: &[],
            initial_state: &[],
            outputs: &[],
            verification: VerificationDescriptor { checks: &[] },
        };
        static ALPHA: CapabilityDescriptor = CapabilityDescriptor {
            id: CapabilityId("alpha"),
            display_name: "Alpha",
            category: CapabilityCategory::Orbital,
            source_crate: "scirust-sim",
            summary: "a summary",
            maturity: CapabilityMaturity::Stable,
            determinism: DeterminismClass::StrictSameBinarySameTarget,
            supported_backends: &[BackendKind::Cpu],
            supported_precisions: &[PrecisionKind::F64],
            supported_solvers: &[],
            parameters: &[],
            initial_state: &[],
            outputs: &[],
            verification: VerificationDescriptor { checks: &[] },
        };
        let mut reg = CapabilityRegistry::new();
        reg.register(&ZEBRA).unwrap();
        reg.register(&ALPHA).unwrap();
        let ids: Vec<&str> = reg.iter().map(|d| d.id.0).collect();
        assert_eq!(ids, vec!["alpha", "zebra"]);
        let text = reg.to_text();
        assert!(text.find("alpha").unwrap() < text.find("zebra").unwrap());
    }

    #[test]
    fn by_category_filters_correctly() {
        let mut reg = CapabilityRegistry::new();
        static M: CapabilityDescriptor = CapabilityDescriptor {
            id: CapabilityId("m"),
            display_name: "M",
            category: CapabilityCategory::Mechanics,
            source_crate: "scirust-sim",
            summary: "s",
            maturity: CapabilityMaturity::Stable,
            determinism: DeterminismClass::StrictSameBinarySameTarget,
            supported_backends: &[BackendKind::Cpu],
            supported_precisions: &[PrecisionKind::F64],
            supported_solvers: &[],
            parameters: &[],
            initial_state: &[],
            outputs: &[],
            verification: VerificationDescriptor { checks: &[] },
        };
        static O: CapabilityDescriptor = CapabilityDescriptor {
            id: CapabilityId("o"),
            display_name: "O",
            category: CapabilityCategory::Orbital,
            source_crate: "scirust-sim",
            summary: "s",
            maturity: CapabilityMaturity::Stable,
            determinism: DeterminismClass::StrictSameBinarySameTarget,
            supported_backends: &[BackendKind::Cpu],
            supported_precisions: &[PrecisionKind::F64],
            supported_solvers: &[],
            parameters: &[],
            initial_state: &[],
            outputs: &[],
            verification: VerificationDescriptor { checks: &[] },
        };
        reg.register(&M).unwrap();
        reg.register(&O).unwrap();
        let mechanics = reg.by_category(CapabilityCategory::Mechanics);
        assert_eq!(mechanics.len(), 1);
        assert_eq!(mechanics[0].id, CapabilityId("m"));
    }

    #[test]
    fn json_catalogue_is_valid_and_contains_every_capability() {
        let mut reg = CapabilityRegistry::new();
        static A: CapabilityDescriptor = CapabilityDescriptor {
            id: CapabilityId("json.a"),
            display_name: "A",
            category: CapabilityCategory::Mechanics,
            source_crate: "scirust-sim",
            summary: "s",
            maturity: CapabilityMaturity::Stable,
            determinism: DeterminismClass::StrictSameBinarySameTarget,
            supported_backends: &[BackendKind::Cpu],
            supported_precisions: &[PrecisionKind::F64],
            supported_solvers: &[],
            parameters: &[],
            initial_state: &[],
            outputs: &[],
            verification: VerificationDescriptor { checks: &[] },
        };
        reg.register(&A).unwrap();
        let json = reg.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 1);
        assert_eq!(parsed[0]["id"], "json.a");
    }
}
