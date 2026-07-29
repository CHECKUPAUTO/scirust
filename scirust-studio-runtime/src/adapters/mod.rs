//! One module per capability adapter, plus the bootstrap that wires them
//! into a [`CapabilityRegistry`] and makes them dispatchable by id.

mod apd;
mod battery;
mod consecutive_reactions;
mod double_pendulum;
mod grid;
mod heat_rod;
mod hvac_zone;
mod laser;
mod logistic_growth;
mod lotka_volterra;
mod oral_dose;
mod ornstein_uhlenbeck;
mod pendulum;
mod photodiode;
mod projectile;
mod rc_circuit;
mod reversible_reaction;
mod rigid_body;
mod rlc;
mod robertson;
mod seir;
mod sir;
mod spring_mass_damper;
mod two_body;
mod two_compartment_iv;
mod van_der_pol;

pub use apd::AvalanchePhotodiodeAdapter;
pub use battery::BatteryAdapter;
pub use consecutive_reactions::ConsecutiveReactionsAdapter;
pub use double_pendulum::DoublePendulumAdapter;
pub use grid::SwingEquationAdapter;
pub use heat_rod::HeatRodAdapter;
pub use hvac_zone::HvacZoneAdapter;
pub use laser::SemiconductorLaserAdapter;
pub use logistic_growth::LogisticGrowthAdapter;
pub use lotka_volterra::LotkaVolterraAdapter;
pub use oral_dose::OralOneCompartmentAdapter;
pub use ornstein_uhlenbeck::OrnsteinUhlenbeckAdapter;
pub use pendulum::PendulumAdapter;
pub use photodiode::PhotodiodeAdapter;
pub use projectile::ProjectileAdapter;
pub use rc_circuit::RcCircuitAdapter;
pub use reversible_reaction::ReversibleReactionAdapter;
pub use rigid_body::RigidBodyAdapter;
pub use rlc::RlcAdapter;
pub use robertson::RobertsonAdapter;
pub use seir::SeirAdapter;
pub use sir::SirAdapter;
pub use spring_mass_damper::SpringMassDamperAdapter;
pub use two_body::TwoBodyAdapter;
pub use two_compartment_iv::TwoCompartmentIvAdapter;
pub use van_der_pol::VanDerPolAdapter;

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
        Box::new(HeatRodAdapter),
        Box::new(PhotodiodeAdapter),
        Box::new(HvacZoneAdapter),
        Box::new(RigidBodyAdapter),
        Box::new(OralOneCompartmentAdapter),
        Box::new(BatteryAdapter),
        Box::new(SemiconductorLaserAdapter),
        Box::new(SwingEquationAdapter),
        Box::new(AvalanchePhotodiodeAdapter),
        Box::new(SeirAdapter),
        Box::new(ConsecutiveReactionsAdapter),
        Box::new(ReversibleReactionAdapter),
        Box::new(RcCircuitAdapter),
        Box::new(ProjectileAdapter),
        Box::new(VanDerPolAdapter),
        Box::new(TwoCompartmentIvAdapter),
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
    (
        "sim.thermal.heat_rod_1d",
        "heat_rod.scirust.toml",
        include_str!("../../../docs/studio/tutorials/heat_rod.scirust.toml"),
    ),
    (
        "sim.pharmacology.oral_one_compartment",
        "oral_dose.scirust.toml",
        include_str!("../../../docs/studio/tutorials/oral_dose.scirust.toml"),
    ),
    (
        "sim.mechanics.rigid_body",
        "rigid_body.scirust.toml",
        include_str!("../../../docs/studio/tutorials/rigid_body.scirust.toml"),
    ),
    (
        "sim.thermal.hvac_zone",
        "hvac_zone.scirust.toml",
        include_str!("../../../docs/studio/tutorials/hvac_zone.scirust.toml"),
    ),
    (
        "sim.optoelectronics.photodiode",
        "photodiode.scirust.toml",
        include_str!("../../../docs/studio/tutorials/photodiode.scirust.toml"),
    ),
    (
        "sim.energy.battery_thevenin",
        "battery_discharge.scirust.toml",
        include_str!("../../../docs/studio/tutorials/battery_discharge.scirust.toml"),
    ),
    (
        "sim.optoelectronics.semiconductor_laser",
        "semiconductor_laser.scirust.toml",
        include_str!("../../../docs/studio/tutorials/semiconductor_laser.scirust.toml"),
    ),
    (
        "sim.power.swing_equation",
        "swing_equation.scirust.toml",
        include_str!("../../../docs/studio/tutorials/swing_equation.scirust.toml"),
    ),
    (
        "sim.optoelectronics.avalanche_photodiode",
        "avalanche_photodiode.scirust.toml",
        include_str!("../../../docs/studio/tutorials/avalanche_photodiode.scirust.toml"),
    ),
    (
        "sim.epidemiology.seir",
        "seir_epidemic.scirust.toml",
        include_str!("../../../docs/studio/tutorials/seir_epidemic.scirust.toml"),
    ),
    (
        "sim.chemistry.consecutive_reactions",
        "consecutive_reactions.scirust.toml",
        include_str!("../../../docs/studio/tutorials/consecutive_reactions.scirust.toml"),
    ),
    (
        "sim.chemistry.reversible_reaction",
        "reversible_reaction.scirust.toml",
        include_str!("../../../docs/studio/tutorials/reversible_reaction.scirust.toml"),
    ),
    (
        "sim.electrical.rc",
        "rc_circuit.scirust.toml",
        include_str!("../../../docs/studio/tutorials/rc_circuit.scirust.toml"),
    ),
    (
        "sim.mechanics.projectile",
        "projectile.scirust.toml",
        include_str!("../../../docs/studio/tutorials/projectile.scirust.toml"),
    ),
    (
        "sim.electrical.van_der_pol",
        "van_der_pol.scirust.toml",
        include_str!("../../../docs/studio/tutorials/van_der_pol.scirust.toml"),
    ),
    (
        "sim.pharmacology.two_compartment_iv",
        "two_compartment_iv.scirust.toml",
        include_str!("../../../docs/studio/tutorials/two_compartment_iv.scirust.toml"),
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

    /// Every adapter must answer for `experiment.replicates`, and the answer
    /// must be the one its determinism class implies.
    ///
    /// This is driven by [`all_adapters`] rather than by a list, so an
    /// eleventh capability that forgets to call `resolve_replicates` fails
    /// here instead of silently ignoring the field. The previous phase landed
    /// a break precisely because one construction site was not reached by
    /// anything that compiled it; a registry-driven assertion is the cheap
    /// version of that lesson.
    #[test]
    fn every_capability_answers_for_replicates_according_to_its_class() {
        for adapter in all_adapters()
        {
            let descriptor = adapter.descriptor();
            let id = descriptor.id.0;
            let source =
                tutorial_scenario_for(id).unwrap_or_else(|| panic!("{id} ships no tutorial"));
            let mut scenario =
                scirust_studio_schema::parse_toml(source).unwrap_or_else(|e| panic!("{id}: {e}"));

            // One realisation must mean the same thing stated as omitted.
            scenario.experiment.replicates = Some(1);
            assert!(
                adapter.validate(&scenario).is_ok(),
                "{id} rejected an explicit `replicates = 1`, which is what every scenario \
                 without the field already asks for"
            );

            scenario.experiment.replicates = Some(4);
            let outcome = adapter.validate(&scenario);
            if descriptor.determinism.draws_a_sample()
            {
                assert!(
                    outcome.is_ok(),
                    "{id} draws a sample but refused an ensemble of 4: {:?}",
                    outcome.err()
                );
            }
            else
            {
                let report = outcome.err().unwrap_or_else(|| {
                    panic!(
                        "{id} accepted `replicates = 4` although its class is {:?}; it would \
                         have run the same computation four times and called it a distribution",
                        descriptor.determinism
                    )
                });
                assert!(
                    report
                        .errors
                        .iter()
                        .any(|e| e.code == crate::validate_support::CODE_REPLICATES_UNSUPPORTED),
                    "{id} rejected the ensemble but not with the code that explains why: {:?}",
                    report.errors.iter().map(|e| e.code).collect::<Vec<_>>()
                );
            }

            // Zero is refused everywhere: it asks for a run with no result.
            scenario.experiment.replicates = Some(0);
            assert!(
                adapter.validate(&scenario).is_err(),
                "{id} accepted `replicates = 0`"
            );
        }
    }

    /// A capability may not emit a verification the catalogue does not list.
    ///
    /// The catalogue is what a user reads to decide whether a capability's
    /// claims are worth anything, so a check that appears only in the output
    /// is a claim made after the fact. Nothing enforced this before; the
    /// ensemble checks are the first additions since, and they would have
    /// gone undeclared without it.
    ///
    /// The converse — every declared check appears in every run — is
    /// deliberately not asserted here: some checks apply only to some
    /// scenarios (the ensemble ones need more than one replicate), and
    /// demanding they appear anyway would push adapters into emitting
    /// `NotApplicable` filler.
    #[test]
    fn no_adapter_emits_a_verification_its_catalogue_entry_does_not_declare() {
        for adapter in all_adapters()
        {
            let descriptor = adapter.descriptor();
            let id = descriptor.id.0;
            let declared: Vec<&str> = descriptor
                .verification
                .checks
                .iter()
                .map(|c| c.id)
                .collect();

            let source = tutorial_scenario_for(id).expect("every capability ships a tutorial");
            let scenario = scirust_studio_schema::parse_toml(source).expect("the tutorial parses");
            let validated = adapter.validate(&scenario).expect("the tutorial validates");
            let result = adapter
                .execute(
                    &validated,
                    &crate::control::ExecutionControl::new(),
                    &mut crate::sink::NullEventSink,
                )
                .unwrap_or_else(|e| panic!("{id} failed to execute its own tutorial: {e}"));

            for emitted in &result.verifications
            {
                assert!(
                    declared.contains(&emitted.id.as_str()),
                    "{id} emitted the check `{}`, which its catalogue entry does not declare; \
                     a reader deciding whether to trust this capability would never see it",
                    emitted.id
                );
            }
        }
    }

    /// `backend.precision` must mean something.
    ///
    /// The schema accepted `"f32"` and every adapter computed in `f64`
    /// regardless, because nothing compared the scenario's request against the
    /// descriptor. A run then recorded a stated precision it did not have —
    /// the same shape of defect as `experiment.seed` before Phase 3B-2, and
    /// worse, because a silently-upgraded precision looks like the user got
    /// what they asked for.
    ///
    /// Driven by each descriptor rather than by today's answer: when a
    /// capability grows an `f32` path, this follows it without being edited.
    #[test]
    fn every_capability_honours_the_precision_it_declares() {
        for adapter in all_adapters()
        {
            let descriptor = adapter.descriptor();
            let id = descriptor.id.0;
            let source = tutorial_scenario_for(id).expect("every capability ships a tutorial");
            let mut scenario =
                scirust_studio_schema::parse_toml(source).expect("the tutorial parses");

            for (name, kind) in [
                ("f64", scirust_studio_registry::PrecisionKind::F64),
                ("f32", scirust_studio_registry::PrecisionKind::F32),
            ]
            {
                scenario.backend.precision = name.to_string();
                let outcome = adapter.validate(&scenario);
                if descriptor.supported_precisions.contains(&kind)
                {
                    assert!(
                        outcome.is_ok(),
                        "{id} declares {name} but refused it: {:?}",
                        outcome.err()
                    );
                }
                else
                {
                    let report = outcome.err().unwrap_or_else(|| {
                        panic!(
                            "{id} accepted `precision = \"{name}\"` although it does not declare \
                             it; the run would have been computed at a different precision from \
                             the one recorded"
                        )
                    });
                    assert!(
                        report.errors.iter().any(|e| {
                            e.code == crate::validate_support::CODE_UNSUPPORTED_PRECISION
                        }),
                        "{id} refused {name} but not with the code that explains why: {:?}",
                        report.errors.iter().map(|e| e.code).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    /// The same for `backend.kind`. It refuses nothing today — every
    /// capability declares `Cpu` and the schema accepts only `"cpu"` — and
    /// exists so the first capability that is not CPU-only fails loudly
    /// rather than running somewhere it never claimed to.
    #[test]
    fn every_capability_honours_the_backend_it_declares() {
        for adapter in all_adapters()
        {
            let descriptor = adapter.descriptor();
            let id = descriptor.id.0;
            let source = tutorial_scenario_for(id).expect("every capability ships a tutorial");
            let mut scenario =
                scirust_studio_schema::parse_toml(source).expect("the tutorial parses");

            scenario.backend.kind = "cpu".to_string();
            let outcome = adapter.validate(&scenario);
            if descriptor
                .supported_backends
                .contains(&scirust_studio_registry::BackendKind::Cpu)
            {
                assert!(outcome.is_ok(), "{id} declares Cpu but refused it");
            }
            else
            {
                let report = outcome
                    .err()
                    .unwrap_or_else(|| panic!("{id} accepted a backend it does not declare"));
                assert!(
                    report
                        .errors
                        .iter()
                        .any(|e| e.code == crate::validate_support::CODE_UNSUPPORTED_BACKEND)
                );
            }
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
        use scirust_studio_registry::RunDomain;

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

            // Which axis to demand is the capability's own declaration, not
            // an assumption about all of them. `RunDomain::ParameterSweep`
            // exists because a receiver analysis genuinely has no time in it;
            // weakening this to "some axis" for everybody would have let a
            // time-integrating capability quietly stop emitting `t`.
            let domain = adapter.descriptor().domain;
            if domain == RunDomain::Time
            {
                assert!(
                    result.time_axis().is_some(),
                    "{id} declares RunDomain::Time but emitted no `{TIME_AXIS_ID}` axis"
                );
                assert_eq!(
                    result.summary.axis_id, TIME_AXIS_ID,
                    "{id}: a time-domain run's summary must describe its time axis"
                );
            }
            else
            {
                assert!(
                    result.time_axis().is_none(),
                    "{id} declares RunDomain::{domain:?} but emitted a `{TIME_AXIS_ID}` axis,                      which a reader would take as a trajectory"
                );
            }

            let axis = result.summary_axis().unwrap_or_else(|| {
                panic!(
                    "{id}: summary names axis `{}`, which this result does not have",
                    result.summary.axis_id
                )
            });
            assert!(
                !axis.values.is_empty(),
                "{id}: the axis must carry its coordinates"
            );
            assert_eq!(
                axis.monotonicity,
                AxisMonotonicity::StrictlyIncreasing,
                "{id}: the independent variable only ever goes forward"
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
                // Not necessarily the *time* axis. `sim.thermal.heat_rod_1d`
                // is the first capability with two, and plots two of its
                // series against position — this used to assert `t` and had
                // to be generalised rather than special-cased, because "every
                // series names an axis this result has, and matches its
                // length" is the property that was always meant.
                let named = result
                    .axes
                    .iter()
                    .find(|a| a.id == series.axis_id)
                    .unwrap_or_else(|| {
                        panic!("{id}/{}: names absent axis `{}`", series.id, series.axis_id)
                    });
                assert_eq!(
                    series.values.len(),
                    named.values.len(),
                    "{id}/{}: series and axis lengths must agree",
                    series.id
                );
            }

            // Whatever else it has, at least one series must be plotted
            // against the run's own independent variable: a capability that
            // sweeps something and shows nothing over it has lost the run's
            // shape.
            assert!(
                result.series.iter().any(|s| s.axis_id == axis.id),
                "{id}: no series is plotted against `{}`",
                axis.id
            );

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
