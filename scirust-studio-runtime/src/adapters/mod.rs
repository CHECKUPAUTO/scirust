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

/// Every capability's shipped tutorial scenario, compiled into the binary,
/// paired with the file it comes from under `docs/studio/tutorials/`.
///
/// These are the exact files each adapter's own test suite executes, so a
/// caller offering "load the example for this capability" hands the user
/// something that is known to run — not a snippet in a doc that drifted.
const TUTORIAL_SCENARIOS: &[(&str, &str, &str)] = &[
    (
        "sim.mechanics.spring_mass_damper",
        "spring_mass_damper.scirust.toml",
        include_str!("../../../docs/studio/tutorials/spring_mass_damper.scirust.toml"),
    ),
    (
        "sim.epidemiology.sir",
        "sir_epidemic.scirust.toml",
        include_str!("../../../docs/studio/tutorials/sir_epidemic.scirust.toml"),
    ),
    (
        "sim.orbital.two_body",
        "two_body_orbit.scirust.toml",
        include_str!("../../../docs/studio/tutorials/two_body_orbit.scirust.toml"),
    ),
    (
        "sim.electrical.rlc",
        "rlc_circuit.scirust.toml",
        include_str!("../../../docs/studio/tutorials/rlc_circuit.scirust.toml"),
    ),
    (
        "sim.chemistry.robertson",
        "robertson_stiff.scirust.toml",
        include_str!("../../../docs/studio/tutorials/robertson_stiff.scirust.toml"),
    ),
];

/// The shipped tutorial scenario text for a capability id, if it has one.
pub fn tutorial_scenario_for(capability_id: &str) -> Option<&'static str> {
    TUTORIAL_SCENARIOS
        .iter()
        .find(|(id, _, _)| *id == capability_id)
        .map(|(_, _, text)| *text)
}

/// The file name a capability's tutorial scenario ships under, if it has
/// one — for callers that want to write the example out for the user.
pub fn tutorial_file_name_for(capability_id: &str) -> Option<&'static str> {
    TUTORIAL_SCENARIOS
        .iter()
        .find(|(id, _, _)| *id == capability_id)
        .map(|(_, name, _)| *name)
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

    /// Every capability ships an example that is known to work — and the
    /// table has no entries for capabilities that no longer exist.
    #[test]
    fn every_capability_ships_a_tutorial_and_every_tutorial_has_a_capability() {
        for adapter in all_adapters()
        {
            let id = adapter.descriptor().id.0;
            assert!(
                tutorial_scenario_for(id).is_some(),
                "{id} ships no tutorial scenario"
            );
            assert!(tutorial_file_name_for(id).is_some(), "{id}");
        }
        for (id, _, _) in TUTORIAL_SCENARIOS
        {
            assert!(
                find_adapter(id).is_some(),
                "tutorial listed for unknown capability {id}"
            );
        }
    }

    /// Each shipped tutorial must parse and validate against the very
    /// capability it claims to demonstrate.
    #[test]
    fn every_tutorial_validates_against_its_own_capability() {
        for (id, file, toml) in TUTORIAL_SCENARIOS
        {
            let scenario = scirust_studio_schema::parse_toml(toml)
                .unwrap_or_else(|e| panic!("{file} does not parse: {e}"));
            assert_eq!(
                scenario.capability.id, *id,
                "{file} declares a different capability"
            );
            let adapter = find_adapter(id).expect("checked above");
            adapter
                .validate(&scenario)
                .unwrap_or_else(|e| panic!("{file} does not validate: {e}"));
        }
    }

    /// Every adapter must refuse to start when handed an already-cancelled
    /// control, and must say so both in its return value and on its event
    /// stream. This is the one cancellation guarantee that holds for *all*
    /// capabilities, including the adaptive stiff solver whose third-party
    /// step loop offers no per-step callback to interrupt (see
    /// `docs/studio/adr/0003-worker-process-and-ipc.md` on why the worker's
    /// process-level cancellation is what covers that case mid-run).
    #[test]
    fn every_adapter_honours_a_pre_cancelled_control() {
        use crate::adapter::ExecutionError;
        use crate::control::ExecutionControl;
        use crate::sink::{CollectingEventSink, RunEvent};

        for adapter in all_adapters()
        {
            let id = adapter.descriptor().id.0;
            let toml = tutorial_scenario_for(id)
                .unwrap_or_else(|| panic!("no shipped tutorial scenario for {id}"));
            let scenario = scirust_studio_schema::parse_toml(toml)
                .unwrap_or_else(|e| panic!("{id} tutorial does not parse: {e}"));
            let validated = adapter
                .validate(&scenario)
                .unwrap_or_else(|e| panic!("{id} tutorial does not validate: {e}"));

            let control = ExecutionControl::new();
            control.cancel();
            let mut sink = CollectingEventSink::new();
            let err = adapter
                .execute(&validated, &control, &mut sink)
                .expect_err("a pre-cancelled control must not produce a result");

            assert_eq!(err, ExecutionError::Cancelled, "{id}");
            assert!(
                sink.events().contains(&RunEvent::Cancelled),
                "{id} did not emit Cancelled: {:?}",
                sink.events()
            );
        }
    }
}
