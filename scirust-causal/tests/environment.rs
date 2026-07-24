use scirust_causal::{CausalError, Environment, Intervention, InterventionKind};

#[test]
fn observational_environment_is_observational() {
    let env = Environment::observational("baseline").unwrap();
    assert!(env.is_observational());
    assert_eq!(env.id, "baseline");
    assert!(env.interventions.is_empty());
}

#[test]
fn environment_with_interventions_is_not_observational() {
    let iv = Intervention::new(0, InterventionKind::Atomic { value: 1.0 }).unwrap();
    let env = Environment::new("site_b", vec![iv]).unwrap();
    assert!(!env.is_observational());
}

#[test]
fn rejects_empty_id() {
    assert!(matches!(
        Environment::observational(""),
        Err(CausalError::InvalidContract { .. })
    ));
    assert!(matches!(
        Environment::observational("   "),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn rejects_two_interventions_on_the_same_target() {
    let iv1 = Intervention::new(0, InterventionKind::Atomic { value: 1.0 }).unwrap();
    let iv2 = Intervention::new(0, InterventionKind::Shift { delta: 0.5 }).unwrap();
    assert!(matches!(
        Environment::new("conflict", vec![iv1, iv2]),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn accepts_simultaneous_interventions_on_distinct_targets() {
    let iv1 = Intervention::new(0, InterventionKind::Atomic { value: 1.0 }).unwrap();
    let iv2 = Intervention::new(1, InterventionKind::Shift { delta: 0.5 }).unwrap();
    let env = Environment::new("multi", vec![iv1, iv2]).unwrap();
    assert_eq!(env.interventions.len(), 2);
}

#[test]
fn json_round_trips() {
    let iv = Intervention::new(0, InterventionKind::Atomic { value: 2.0 }).unwrap();
    let env = Environment::new("site_c", vec![iv]).unwrap();
    let json = serde_json::to_string(&env).unwrap();
    let back: Environment = serde_json::from_str(&json).unwrap();
    assert_eq!(env, back);
}
