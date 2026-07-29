//! One module per capability adapter, plus the bootstrap that wires them
//! into a [`CapabilityRegistry`] and makes them dispatchable by id.

mod double_pendulum;
mod logistic_growth;
mod lotka_volterra;
mod ornstein_uhlenbeck;
mod pendulum;
mod rlc;
mod robertson;
mod sir;
mod spring_mass_damper;
mod two_body;

pub use double_pendulum::DoublePendulumAdapter;
pub use logistic_growth::LogisticGrowthAdapter;
pub use lotka_volterra::LotkaVolterraAdapter;
pub use ornstein_uhlenbeck::OrnsteinUhlenbeckAdapter;
pub use pendulum::PendulumAdapter;
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
        Box::new(LotkaVolterraAdapter),
        Box::new(LogisticGrowthAdapter),
        Box::new(PendulumAdapter),
        Box::new(DoublePendulumAdapter),
        Box::new(OrnsteinUhlenbeckAdapter),
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
    (
        "sim.ecology.lotka_volterra",
        "lotka_volterra.scirust.toml",
        include_str!("../../../docs/studio/tutorials/lotka_volterra.scirust.toml"),
    ),
    (
        "sim.ecology.logistic_growth",
        "logistic_growth.scirust.toml",
        include_str!("../../../docs/studio/tutorials/logistic_growth.scirust.toml"),
    ),
    (
        "sim.mechanics.pendulum",
        "pendulum.scirust.toml",
        include_str!("../../../docs/studio/tutorials/pendulum.scirust.toml"),
    ),
    (
        "sim.mechanics.double_pendulum",
        "double_pendulum.scirust.toml",
        include_str!("../../../docs/studio/tutorials/double_pendulum.scirust.toml"),
    ),
    (
        "sim.stochastic.ornstein_uhlenbeck",
        "ornstein_uhlenbeck.scirust.toml",
        include_str!("../../../docs/studio/tutorials/ornstein_uhlenbeck.scirust.toml"),
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

    /// Every capability must emit result schema v2 with real coordinates:
    /// an axis that carries values, series bound to it by id and aligned
    /// with it, and a summary that agrees with the axis exactly.
    #[test]
    fn every_adapter_emits_exact_axis_coordinates() {
        use crate::control::ExecutionControl;
        use crate::result::{AxisMonotonicity, RESULT_SCHEMA_VERSION, TIME_AXIS_ID};
        use crate::sink::NullEventSink;

        for adapter in all_adapters()
        {
            let id = adapter.descriptor().id.0;
            let toml = tutorial_scenario_for(id).expect("tutorial");
            let scenario = scirust_studio_schema::parse_toml(toml).expect("parses");
            let validated = adapter.validate(&scenario).expect("validates");
            let result = adapter
                .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
                .unwrap_or_else(|e| panic!("{id} failed to execute: {e}"));

            assert_eq!(result.schema_version, RESULT_SCHEMA_VERSION, "{id}");

            let axis = result
                .time_axis()
                .unwrap_or_else(|| panic!("{id} has no `{TIME_AXIS_ID}` axis"));
            assert!(
                !axis.values.is_empty(),
                "{id}: the axis must carry its coordinates"
            );
            assert_eq!(
                axis.monotonicity,
                AxisMonotonicity::StrictlyIncreasing,
                "{id}: a forward integration's time axis is strictly increasing"
            );

            // Exact, not approximate: the summary must be the same numbers
            // the axis holds, not a recomputation of them.
            assert_eq!(result.summary.t_start, axis.values[0], "{id}");
            assert_eq!(result.summary.t_end, *axis.values.last().unwrap(), "{id}");
            assert_eq!(
                result.summary.steps,
                axis.values.len() - 1,
                "{id}: steps must match the coordinates recorded"
            );

            assert!(!result.series.is_empty(), "{id}");
            for series in &result.series
            {
                assert_eq!(series.axis_id, TIME_AXIS_ID, "{id}/{}", series.id);
                assert_eq!(
                    series.values.len(),
                    axis.values.len(),
                    "{id}/{}: series and axis lengths must agree",
                    series.id
                );
            }

            // The adapters call this themselves before returning; asserting
            // it here too means a future adapter that forgets is caught.
            crate::result::validate_result(&result)
                .unwrap_or_else(|d| panic!("{id}: {}", crate::result::describe_defects(&d)));
        }
    }

    /// The reason schema v2 exists, stated as a test: Robertson's stiff
    /// solver chooses its own steps, so its coordinates are *not* uniformly
    /// spaced and could never have been reconstructed from a start, an end
    /// and a count. A regression that resampled onto a linear grid would
    /// pass every other test in this crate and fail this one.
    #[test]
    fn the_adaptive_capability_records_genuinely_non_uniform_coordinates() {
        use crate::control::ExecutionControl;
        use crate::sink::NullEventSink;

        let id = "sim.chemistry.robertson";
        let adapter = find_adapter(id).expect("registered");
        let scenario =
            scirust_studio_schema::parse_toml(tutorial_scenario_for(id).expect("tutorial"))
                .expect("parses");
        let validated = adapter.validate(&scenario).expect("validates");
        let result = adapter
            .execute(&validated, &ExecutionControl::new(), &mut NullEventSink)
            .expect("executes");

        let t = &result.time_axis().expect("time axis").values;
        assert!(t.len() > 10, "expected a multi-step run, got {}", t.len());

        let gaps: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
        let smallest = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
        let largest = gaps.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            largest > smallest * 10.0,
            "Robertson's steps should vary by at least an order of magnitude, \
             but ranged only {smallest:e}..{largest:e} — this looks like a \
             regenerated uniform grid, not the solver's own timestamps"
        );

        // And a linear reconstruction really would have been wrong.
        let uniform_gap = (t.last().unwrap() - t[0]) / (t.len() - 1) as f64;
        let worst = gaps
            .iter()
            .map(|g| (g - uniform_gap).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst > uniform_gap * 0.5,
            "a uniform axis would have been a materially different picture"
        );
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
