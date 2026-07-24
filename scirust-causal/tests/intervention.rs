use scirust_causal::{CausalError, Intervention, InterventionKind};

#[test]
fn accepts_atomic_with_finite_value() {
    let iv = Intervention::new(0, InterventionKind::Atomic { value: 1.5 }).unwrap();
    assert_eq!(iv.target, 0);
    assert!(matches!(iv.kind, InterventionKind::Atomic { value } if value == 1.5));
}

#[test]
fn rejects_non_finite_atomic_value() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
    {
        assert!(matches!(
            Intervention::new(0, InterventionKind::Atomic { value: bad }),
            Err(CausalError::InvalidContract { .. })
        ));
    }
}

#[test]
fn rejects_non_finite_shift_delta() {
    assert!(matches!(
        Intervention::new(0, InterventionKind::Shift { delta: f64::NAN }),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn accepts_finite_shift() {
    assert!(Intervention::new(1, InterventionKind::Shift { delta: -2.0 }).is_ok());
}

#[test]
fn rejects_empty_mechanism_change_description() {
    assert!(matches!(
        Intervention::new(
            0,
            InterventionKind::MechanismChange {
                description: "  ".to_string()
            }
        ),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn accepts_nonempty_mechanism_change_description() {
    assert!(
        Intervention::new(
            0,
            InterventionKind::MechanismChange {
                description: "sensor recalibrated".to_string()
            }
        )
        .is_ok()
    );
}

#[test]
fn unspecified_is_always_accepted() {
    assert!(Intervention::new(0, InterventionKind::Unspecified).is_ok());
}

#[test]
fn json_round_trips() {
    let iv = Intervention::new(2, InterventionKind::Atomic { value: 0.5 }).unwrap();
    let json = serde_json::to_string(&iv).unwrap();
    let back: Intervention = serde_json::from_str(&json).unwrap();
    assert_eq!(iv, back);
}
