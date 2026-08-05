use super::data::{CandidateRecord, RUNTIME_FEATURE_NAMES, RUNTIME_FEATURES};
use scirust_causal::{
    CausalDataset, CausalVariable, Environment, InvarianceConfig, Intervention,
    InterventionKind, SampleBlock, VariableKind, VariableRole, invariant_causal_prediction,
};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct AcceptedCausalSet {
    pub predictors: Vec<String>,
    pub p_value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CausalDiagnostic {
    pub method: String,
    pub environments: Vec<String>,
    pub samples_per_environment: usize,
    pub target: String,
    pub predictors_considered: Vec<String>,
    pub outcome: String,
    pub causal_predictors: Vec<String>,
    pub accepted_sets: Vec<AcceptedCausalSet>,
    pub sets_tested: usize,
    pub warnings: Vec<String>,
    pub assumptions: Vec<String>,
    pub error: Option<String>,
}

fn variable(
    index: usize,
    name: &str,
    role: VariableRole,
) -> Result<CausalVariable, String> {
    CausalVariable::new(index, name, role, VariableKind::Continuous)
        .map_err(|error| error.to_string())
}

pub fn run_causal_diagnostic(
    rows: &[CandidateRecord],
    training_indices: &[usize],
) -> Result<CausalDiagnostic, String> {
    if training_indices.len() < 8 {
        return Err("causal diagnostic requires at least eight training interventions".to_string());
    }

    let treatment_index = RUNTIME_FEATURES;
    let outcome_index = RUNTIME_FEATURES + 1;
    let mut variables = Vec::with_capacity(RUNTIME_FEATURES + 2);
    for (index, name) in RUNTIME_FEATURE_NAMES.iter().enumerate() {
        variables.push(variable(index, name, VariableRole::Covariate)?);
    }
    variables.push(variable(
        treatment_index,
        "single_skip_intervention",
        VariableRole::Treatment,
    )?);
    variables.push(variable(
        outcome_index,
        "strict_trajectory_unsafe",
        VariableRole::Outcome,
    )?);

    let columns = variables.len();
    let mut baseline_data = Vec::with_capacity(training_indices.len() * columns);
    let mut intervention_data = Vec::with_capacity(training_indices.len() * columns);
    for &row_index in training_indices {
        let row = &rows[row_index];
        baseline_data.extend_from_slice(&row.raw_features);
        baseline_data.push(0.0);
        baseline_data.push(0.0);

        intervention_data.extend_from_slice(&row.raw_features);
        intervention_data.push(1.0);
        intervention_data.push(if row.strict_unsafe { 1.0 } else { 0.0 });
    }

    let baseline_environment = Environment::observational("always_refresh")
        .map_err(|error| error.to_string())?;
    let intervention = Intervention::new(
        treatment_index,
        InterventionKind::Atomic { value: 1.0 },
    )
    .map_err(|error| error.to_string())?;
    let single_skip_environment = Environment::new("single_skip", vec![intervention])
        .map_err(|error| error.to_string())?;
    let baseline_block = SampleBlock::new(
        baseline_environment,
        training_indices.len(),
        columns,
        baseline_data,
    )
    .map_err(|error| error.to_string())?;
    let intervention_block = SampleBlock::new(
        single_skip_environment,
        training_indices.len(),
        columns,
        intervention_data,
    )
    .map_err(|error| error.to_string())?;
    let dataset = CausalDataset::new(
        variables.clone(),
        vec![baseline_block, intervention_block],
        "Dream single-skip branches on GSM8K train; paired baseline and atomic skip intervention",
    )
    .map_err(|error| error.to_string())?;

    let predictors: Vec<usize> = (0..=treatment_index).collect();
    let config = InvarianceConfig::new(0.05)
        .map_err(|error| error.to_string())?
        .with_max_predictor_set_size(2);
    let result = invariant_causal_prediction(&dataset, outcome_index, &predictors, &config);
    match result {
        Ok(result) => {
            let names = |indices: &[usize]| {
                indices
                    .iter()
                    .map(|index| variables[*index].name.clone())
                    .collect::<Vec<_>>()
            };
            Ok(CausalDiagnostic {
                method: "scirust-causal invariant causal prediction".to_string(),
                environments: result.environments,
                samples_per_environment: training_indices.len(),
                target: variables[outcome_index].name.clone(),
                predictors_considered: names(&predictors),
                outcome: format!("{:?}", result.outcome),
                causal_predictors: names(&result.causal_predictors),
                accepted_sets: result
                    .accepted_sets
                    .into_iter()
                    .map(|set| AcceptedCausalSet {
                        predictors: names(&set.predictors),
                        p_value: set.p_value,
                    })
                    .collect(),
                sets_tested: result.sets_tested,
                warnings: result.warnings,
                assumptions: vec![
                    "linear conditional mechanism".to_string(),
                    "invariance across baseline and single-skip environments".to_string(),
                    "no direct intervention on the unsafe outcome".to_string(),
                    "adequate sample size".to_string(),
                ],
                error: None,
            })
        }
        Err(error) => Ok(CausalDiagnostic {
            method: "scirust-causal invariant causal prediction".to_string(),
            environments: vec!["always_refresh".to_string(), "single_skip".to_string()],
            samples_per_environment: training_indices.len(),
            target: variables[outcome_index].name.clone(),
            predictors_considered: predictors
                .iter()
                .map(|index| variables[*index].name.clone())
                .collect(),
            outcome: "DiagnosticUnavailable".to_string(),
            causal_predictors: Vec::new(),
            accepted_sets: Vec::new(),
            sets_tested: 0,
            warnings: vec![
                "The causal diagnostic is advisory and never overrides the fail-closed policy gate."
                    .to_string(),
            ],
            assumptions: vec![
                "binary trajectory outcomes are represented as continuous 0/1 values".to_string(),
                "the ICP linearity assumption may be violated".to_string(),
            ],
            error: Some(error.to_string()),
        }),
    }
}
