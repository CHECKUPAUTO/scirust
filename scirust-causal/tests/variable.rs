use scirust_causal::{
    CausalError, CausalVariable, VariableKind, VariableRole, validate_variable_set,
};

#[test]
fn accepts_a_well_formed_variable() {
    let v =
        CausalVariable::new(0, "smoking", VariableRole::Treatment, VariableKind::Binary).unwrap();
    assert_eq!(v.index, 0);
    assert_eq!(v.name, "smoking");
    assert_eq!(v.role, VariableRole::Treatment);
    assert_eq!(v.kind, VariableKind::Binary);
}

#[test]
fn rejects_empty_or_blank_name() {
    assert!(matches!(
        CausalVariable::new(0, "", VariableRole::Outcome, VariableKind::Continuous),
        Err(CausalError::InvalidContract { .. })
    ));
    assert!(matches!(
        CausalVariable::new(0, "   ", VariableRole::Outcome, VariableKind::Continuous),
        Err(CausalError::InvalidContract { .. })
    ));
}

fn sample_set() -> Vec<CausalVariable> {
    vec![
        CausalVariable::new(0, "x", VariableRole::Treatment, VariableKind::Continuous).unwrap(),
        CausalVariable::new(1, "y", VariableRole::Outcome, VariableKind::Continuous).unwrap(),
        CausalVariable::new(2, "z", VariableRole::Covariate, VariableKind::Discrete).unwrap(),
    ]
}

#[test]
fn accepts_a_well_formed_variable_set() {
    assert!(validate_variable_set(&sample_set()).is_ok());
}

#[test]
fn rejects_out_of_range_index() {
    let mut vars = sample_set();
    vars[2].index = 7;
    assert!(matches!(
        validate_variable_set(&vars),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn rejects_duplicate_index() {
    let mut vars = sample_set();
    vars[2].index = 0;
    assert!(matches!(
        validate_variable_set(&vars),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn rejects_duplicate_name() {
    let mut vars = sample_set();
    vars[2].name = "x".to_string();
    assert!(matches!(
        validate_variable_set(&vars),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn empty_set_is_valid() {
    assert!(validate_variable_set(&[]).is_ok());
}

#[test]
fn json_round_trips() {
    let v = CausalVariable::new(
        3,
        "confounder",
        VariableRole::Confounder,
        VariableKind::Binary,
    )
    .unwrap();
    let json = serde_json::to_string(&v).unwrap();
    let back: CausalVariable = serde_json::from_str(&json).unwrap();
    assert_eq!(v, back);
}
