use super::data::{CandidateRecord, RUNTIME_FEATURE_NAMES, RUNTIME_FEATURES};
use scirust_causal::{
    CausalDataset, CausalVariable, Environment, Intervention, InterventionKind, InvarianceConfig,
    SampleBlock, VariableKind, VariableRole, invariant_causal_prediction,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

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
    pub environment_sample_counts: BTreeMap<String, usize>,
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

fn variable(index: usize, name: &str, role: VariableRole) -> Result<CausalVariable, String> {
    CausalVariable::new(index, name, role, VariableKind::Continuous)
        .map_err(|error| error.to_string())
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>()
    {
        (*message).to_string()
    }
    else if let Some(message) = payload.downcast_ref::<String>()
    {
        message.clone()
    }
    else
    {
        "non-string panic payload".to_string()
    }
}

#[allow(clippy::too_many_arguments)]
fn unavailable_diagnostic(
    variables: &[CausalVariable],
    outcome_index: usize,
    predictors: &[usize],
    environments: Vec<String>,
    environment_sample_counts: BTreeMap<String, usize>,
    error: String,
    warning: String,
) -> CausalDiagnostic {
    CausalDiagnostic {
        method: "scirust-causal invariant causal prediction across atomic skip locations"
            .to_string(),
        environments,
        samples_per_environment: environment_sample_counts
            .values()
            .copied()
            .min()
            .unwrap_or(0),
        environment_sample_counts,
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
            warning,
            "The causal diagnostic is advisory and never overrides the fail-closed policy gate."
                .to_string(),
        ],
        assumptions: vec![
            "binary trajectory outcomes are represented as continuous 0/1 values".to_string(),
            "the ICP linearity assumption may be violated".to_string(),
            "candidate ordinal identifies the controlled atomic skip location".to_string(),
        ],
        error: Some(error),
    }
}

pub fn run_causal_diagnostic(
    rows: &[CandidateRecord],
    training_indices: &[usize],
) -> Result<CausalDiagnostic, String> {
    if training_indices.len() < 8
    {
        return Err("causal diagnostic requires at least eight training interventions".to_string());
    }

    // The former diagnostic compared single-skip outcomes with a synthetic
    // always-refresh environment whose unsafe label was structurally zero.
    // Constant residual groups can make a Welch statistic undefined and, more
    // importantly, do not represent multiple environments of the same target
    // mechanism. Phase 8 instead compares the controlled atomic intervention
    // locations (candidate ordinals) using intervention rows only.
    let ordinal_index = RUNTIME_FEATURES - 1;
    let outcome_index = RUNTIME_FEATURES;
    let mut variables = Vec::with_capacity(RUNTIME_FEATURES + 1);
    for (index, name) in RUNTIME_FEATURE_NAMES.iter().enumerate()
    {
        let role = if index == ordinal_index
        {
            VariableRole::Treatment
        }
        else
        {
            VariableRole::Covariate
        };
        variables.push(variable(index, name, role)?);
    }
    variables.push(variable(
        outcome_index,
        "strict_trajectory_unsafe",
        VariableRole::Outcome,
    )?);

    let mut rows_by_ordinal: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &row_index in training_indices
    {
        rows_by_ordinal
            .entry(rows[row_index].ordinal)
            .or_default()
            .push(row_index);
    }
    rows_by_ordinal.retain(|_, indices| indices.len() >= 2);
    if rows_by_ordinal.len() < 2
    {
        return Err(
            "causal diagnostic requires at least two skip-location environments with two samples"
                .to_string(),
        );
    }

    let columns = variables.len();
    let mut blocks = Vec::with_capacity(rows_by_ordinal.len());
    let mut environment_sample_counts = BTreeMap::new();
    for (ordinal, indices) in &rows_by_ordinal
    {
        let environment_id = format!("single_skip_candidate_{ordinal}");
        let intervention_value = rows[indices[0]].raw_features[ordinal_index];
        let intervention = Intervention::new(
            ordinal_index,
            InterventionKind::Atomic {
                value: intervention_value,
            },
        )
        .map_err(|error| error.to_string())?;
        let environment = Environment::new(&environment_id, vec![intervention])
            .map_err(|error| error.to_string())?;
        let mut data = Vec::with_capacity(indices.len() * columns);
        for &row_index in indices
        {
            let row = &rows[row_index];
            data.extend_from_slice(&row.raw_features);
            data.push(if row.strict_unsafe { 1.0 } else { 0.0 });
        }
        blocks.push(
            SampleBlock::new(environment, indices.len(), columns, data)
                .map_err(|error| error.to_string())?,
        );
        environment_sample_counts.insert(environment_id, indices.len());
    }

    let dataset = CausalDataset::new(
        variables.clone(),
        blocks,
        "Dream atomic single-skip branches on GSM8K train, separated by controlled intervention location",
    )
    .map_err(|error| error.to_string())?;
    let predictors: Vec<usize> = (0..RUNTIME_FEATURES).collect();
    let config = InvarianceConfig::new(0.05)
        .map_err(|error| error.to_string())?
        .with_max_predictor_set_size(2);
    let environments: Vec<String> = environment_sample_counts.keys().cloned().collect();
    let result = catch_unwind(AssertUnwindSafe(|| {
        invariant_causal_prediction(&dataset, outcome_index, &predictors, &config)
    }));

    match result
    {
        Ok(Ok(result)) =>
        {
            let names = |indices: &[usize]| {
                indices
                    .iter()
                    .map(|index| variables[*index].name.clone())
                    .collect::<Vec<_>>()
            };
            Ok(CausalDiagnostic {
                method: "scirust-causal invariant causal prediction across atomic skip locations"
                    .to_string(),
                environments: result.environments,
                samples_per_environment: environment_sample_counts
                    .values()
                    .copied()
                    .min()
                    .unwrap_or(0),
                environment_sample_counts,
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
                    "invariance across controlled atomic skip locations".to_string(),
                    "no direct intervention on the unsafe outcome".to_string(),
                    "adequate samples in each candidate-ordinal environment".to_string(),
                ],
                error: None,
            })
        },
        Ok(Err(error)) => Ok(unavailable_diagnostic(
            &variables,
            outcome_index,
            &predictors,
            environments,
            environment_sample_counts,
            error.to_string(),
            "ICP returned a structured error for the highly imbalanced strict trajectory label."
                .to_string(),
        )),
        Err(payload) => Ok(unavailable_diagnostic(
            &variables,
            outcome_index,
            &predictors,
            environments,
            environment_sample_counts,
            panic_message(payload),
            "A degenerate residual comparison was trapped; policy discovery continues fail-closed."
                .to_string(),
        )),
    }
}
