//! Model identity across the store — the property `sos-scirust`'s unit tests
//! cannot show on their own.
//!
//! [`sos_scirust::model`]'s own tests stop at the canonical bytes. What
//! actually matters is what those bytes buy once a result is sealed into a
//! content-addressed store and handed to `sos-repro`: a stored trajectory that
//! **names the model that produced it**, verifiably, rather than by a
//! descriptor someone asserted.
//!
//! The contrast with [`sos_scirust::ode::Rk4OdeSimulator`] is the point, and
//! it is exercised here rather than argued: two closure-based simulators
//! carrying the same descriptor and the same config produce results that
//! disagree about the physics while agreeing about their identity. The
//! catalogue backend cannot get into that state.

use sos_core::canonical::Canonical;
use sos_core::{Author, DeterminismLevel, HashAlgo, Object, ObjectId, SemVer};
use sos_scirust::model::{CatalogSimulator, ModelKind, ModelRun, ModelSpec, ModeledTrajectoryBody};
use sos_scirust::ode::{OdeConfig, Rk4OdeSimulator};
use sos_simulation::{SimDescriptor, Simulate};
use sos_store::{MemoryStore, ObjectStore, TypedStore};

fn logistic(rate: f64, capacity: f64) -> ModelRun {
    ModelRun::new(
        ModelSpec::new(ModelKind::LogisticGrowth, [rate, capacity]),
        [1.0],
        0.0,
        4.0,
        0.001,
    )
}

/// Seal a run's observation into `store`, returning its id.
fn seal(store: &mut MemoryStore, config: &ModelRun) -> ObjectId {
    let observation = CatalogSimulator::new().run(config, 0).unwrap();
    let level = observation.level();
    let object = Object::builder(ModeledTrajectoryBody::from_observation(observation))
        .level(level)
        .author(Author::engine("test"))
        .seal();
    store.put_object(&object).unwrap()
}

#[test]
fn a_stored_trajectory_can_be_asked_which_model_produced_it() {
    // Not "which descriptor was claimed" — which model, with which
    // parameters, recoverable from the object itself.
    let mut store = MemoryStore::new();
    let config = logistic(0.8, 100.0);
    let id = seal(&mut store, &config);

    let stored: Object<ModeledTrajectoryBody> = store.get_object(id).unwrap().unwrap();
    assert_eq!(stored.body.model, config.model);
    assert_eq!(stored.body.model.kind, ModelKind::LogisticGrowth);
    assert_eq!(stored.body.model.params, vec![0.8, 100.0]);
    assert_eq!(stored.level, DeterminismLevel::L3);
}

#[test]
fn two_models_cannot_share_a_stored_identity() {
    // Same integrator, same initial state, same span, same step: only the
    // model differs. The store must see two distinct objects.
    let mut store = MemoryStore::new();
    let growth = seal(&mut store, &logistic(0.8, 100.0));
    let cooling = seal(
        &mut store,
        &ModelRun::new(
            ModelSpec::new(ModelKind::NewtonCooling, [0.8, 100.0]),
            [1.0],
            0.0,
            4.0,
            0.001,
        ),
    );
    assert_ne!(growth, cooling);
    assert_eq!(store.object_ids().len(), 2);
}

#[test]
fn the_same_model_run_twice_lands_on_the_same_stored_id() {
    // The memoization property, over a real store: `L3` plus a config whose
    // address covers the model means a re-run is recognisably the same run.
    let mut store = MemoryStore::new();
    let first = seal(&mut store, &logistic(0.8, 100.0));
    let second = seal(&mut store, &logistic(0.8, 100.0));
    assert_eq!(first, second);
    assert_eq!(store.object_ids().len(), 1, "a re-run is not a new object");
}

#[test]
fn a_parameter_change_is_a_different_experiment() {
    let mut store = MemoryStore::new();
    let a = seal(&mut store, &logistic(0.8, 100.0));
    let b = seal(&mut store, &logistic(0.800_000_1, 100.0));
    assert_ne!(a, b, "a rate change must not reuse a cached result");
}

#[test]
fn a_closure_backed_config_address_does_not_determine_the_physics() {
    // The failure this module exists to remove, demonstrated rather than
    // asserted. Both simulators claim the same descriptor and are handed a
    // byte-identical `OdeConfig`, so a manifest referencing that config's
    // digest is under-determined: what actually runs depends on which closure
    // the handler was built with.
    let descriptor = SimDescriptor::new("physics/decay", SemVer::new(1, 0, 0));
    let slow = Rk4OdeSimulator::new(descriptor.clone(), |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = -0.1 * y[0];
    });
    let fast = Rk4OdeSimulator::new(descriptor.clone(), |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = -5.0 * y[0];
    });

    let config = OdeConfig::new(0.0, 2.0, vec![1.0], 0.001);
    let address = |c: &OdeConfig| HashAlgo::Sha256.hash(b"config", &c.canonical_bytes());
    // One address...
    assert_eq!(address(&config), address(&config));
    assert_eq!(slow.descriptor().name, fast.descriptor().name);

    // ...two different experiments.
    let a = slow.run(&config, 0).unwrap().output;
    let b = fast.run(&config, 0).unwrap().output;
    assert_ne!(
        a.last().unwrap().1,
        b.last().unwrap().1,
        "the same address ran two different physical systems"
    );

    // The catalogue closes it: the model is inside the address, so two
    // different systems cannot present the same one.
    let growth = logistic(0.8, 100.0);
    let mut cooling = growth.clone();
    cooling.model.kind = ModelKind::NewtonCooling;
    assert_ne!(growth.canonical_bytes(), cooling.canonical_bytes());
}

#[test]
fn a_catalogued_model_survives_a_store_round_trip_bit_for_bit() {
    // Content addressing only works if the body reloads unchanged — the
    // regression `sos-scirust`'s ODE stage handler hit when trajectory floats
    // went through JSON numbers. Here the canonical encoding is the exact
    // decimal text, and the store verifies the id on the way out.
    let mut store = MemoryStore::new();
    let config = ModelRun::new(
        ModelSpec::new(ModelKind::LotkaVolterra, [1.1, 0.4, 0.1, 0.4]),
        [10.0, 5.0],
        0.0,
        10.0,
        0.001,
    );
    let id = seal(&mut store, &config);
    let stored: Object<ModeledTrajectoryBody> = store.get_object(id).unwrap().unwrap();

    let rerun = CatalogSimulator::new().run(&config, 0).unwrap().output;
    assert!(rerun.trajectory.len() > 10_000, "a substantial trajectory");
    for (i, ((t0, y0), (t1, y1))) in rerun
        .trajectory
        .iter()
        .zip(&stored.body.trajectory)
        .enumerate()
    {
        assert_eq!(t0.to_bits(), t1.to_bits(), "t differs at sample {i}");
        for (k, (a, b)) in y0.iter().zip(y1).enumerate()
        {
            assert_eq!(a.to_bits(), b.to_bits(), "y[{k}] differs at sample {i}");
        }
    }
}
