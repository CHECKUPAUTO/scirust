//! One module per capability adapter, plus the bootstrap that wires them
//! into a [`CapabilityRegistry`] and makes them dispatchable by id.

mod rlc;
mod robertson;
mod sir;
mod spring_mass_damper;
mod two_body;

pub use rlc::RlcAdapter;
pub use robertson::RobertsonAdapter;
pub use sir::SirAdapter;
pub use spring_mass_damper::SpringMassDamperAdapter;
pub use two_body::TwoBodyAdapter;

use scirust_studio_registry::CapabilityRegistry;

use crate::adapter::CapabilityAdapter;

/// Every adapter this crate implements, in a fixed order.
///
/// This is the single place a new capability is wired in: [`build_registry`]
/// and [`find_adapter`] both derive from this list, so a capability cannot
/// end up in the catalogue without being executable, or executable without
/// being catalogued — see the bidirectional consistency test in
/// `scirust-cli`'s integration tests.
pub fn all_adapters() -> Vec<Box<dyn CapabilityAdapter>> {
    vec![
        Box::new(SpringMassDamperAdapter),
        Box::new(SirAdapter),
        Box::new(TwoBodyAdapter),
        Box::new(RlcAdapter),
        Box::new(RobertsonAdapter),
    ]
}

/// Build a [`CapabilityRegistry`] containing every adapter's descriptor.
pub fn build_registry() -> CapabilityRegistry {
    let mut registry = CapabilityRegistry::new();
    for adapter in all_adapters()
    {
        registry
            .register(adapter.descriptor())
            .expect("all_adapters() must not contain two adapters with the same capability id");
    }
    registry
}

/// Find the adapter for a capability id, if any.
pub fn find_adapter(id: &str) -> Option<Box<dyn CapabilityAdapter>> {
    all_adapters()
        .into_iter()
        .find(|a| a.descriptor().id.0 == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_registry_contains_every_adapter() {
        let registry = build_registry();
        let adapters = all_adapters();
        assert_eq!(registry.len(), adapters.len());
        for adapter in &adapters
        {
            assert!(registry.find(adapter.descriptor().id.0).is_some());
        }
    }

    #[test]
    fn find_adapter_matches_the_registry() {
        let registry = build_registry();
        for descriptor in registry.iter()
        {
            assert!(
                find_adapter(descriptor.id.0).is_some(),
                "no adapter for {}",
                descriptor.id
            );
        }
    }

    #[test]
    fn find_adapter_rejects_unknown_id() {
        assert!(find_adapter("no.such.capability").is_none());
    }
}
