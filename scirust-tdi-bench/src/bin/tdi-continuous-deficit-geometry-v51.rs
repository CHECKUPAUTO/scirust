use scirust_tdi::{
    Action, ExactRatio, State, TableSystem, analyze_branching_recovery, explore,
    uniform_branching_path_entropy_bits,
};

const OBSERVATION_HORIZON: usize = 2;

const TARGET_HORIZONS: [usize; 5] = [3, 4, 5, 6, 8];
const TARGET_HORIZON_COUNT: usize = TARGET_HORIZONS.len();
const PRIMARY_HORIZON: usize = 6;
const PRIMARY_HORIZON_INDEX: usize = 3;

const TRAIN_WIDTH_3: u8 = 3;
const TRAIN_WIDTH_4: u8 = 4;
const OOD_WIDTH_5: u8 = 5;
const OOD_WIDTH_6: u8 = 6;

const TRAIN_WIDTH_3_SYSTEMS: usize = 15_000;
const TRAIN_WIDTH_4_SYSTEMS: usize = 15_000;
const HOLDOUT_WIDTH_3_SYSTEMS: usize = 5_000;
const HOLDOUT_WIDTH_4_SYSTEMS: usize = 5_000;
const OOD_WIDTH_5_SYSTEMS: usize = 10_000;
const OOD_WIDTH_6_SYSTEMS: usize = 5_000;

const TRAIN_WIDTH_3_SEED_OFFSET: u64 = 60_000_000;
const HOLDOUT_WIDTH_3_SEED_OFFSET: u64 = 61_000_000;
const TRAIN_WIDTH_4_SEED_OFFSET: u64 = 70_000_000;
const HOLDOUT_WIDTH_4_SEED_OFFSET: u64 = 71_000_000;
const OOD_WIDTH_5_SEED_OFFSET: u64 = 80_000_000;
const OOD_WIDTH_6_SEED_OFFSET: u64 = 90_000_000;

const BASELINE_FEATURE_COUNT: usize = 13;
const TDI_FEATURE_COUNT: usize = 3;

const M0_FEATURE_COUNT: usize = BASELINE_FEATURE_COUNT;
const M1_FEATURE_COUNT: usize = BASELINE_FEATURE_COUNT + 1;
const M2_FEATURE_COUNT: usize = BASELINE_FEATURE_COUNT + 2;
const M3_FEATURE_COUNT: usize = BASELINE_FEATURE_COUNT + TDI_FEATURE_COUNT;

const MODEL_LAYOUT_COUNT: usize = 4;

const RIDGE_LAMBDA: f64 = 1.0;
const BOOTSTRAP_REPLICATES: usize = 2_000;
const BOOTSTRAP_SEED: u64 = 0x5444_4935_4344_4745;

const MAX_SUPPORTED_WIDTH: u8 = 6;
const WIDTH_3_ATTEMPT_MULTIPLIER: usize = 64;
const WIDTH_4_ATTEMPT_MULTIPLIER: usize = 96;
const WIDTH_5_ATTEMPT_MULTIPLIER: usize = 128;
const WIDTH_6_ATTEMPT_MULTIPLIER: usize = 256;
const WIDTH_3_NO_PROGRESS_LIMIT: usize = 25_000;
const WIDTH_4_NO_PROGRESS_LIMIT: usize = 50_000;
const WIDTH_5_NO_PROGRESS_LIMIT: usize = 75_000;
const WIDTH_6_NO_PROGRESS_LIMIT: usize = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
enum FeatureLayout {
    M0,
    M1,
    M2,
    M3,
}

impl FeatureLayout {
    const ALL: [Self; MODEL_LAYOUT_COUNT] = [Self::M0, Self::M1, Self::M2, Self::M3];

    const fn label(self) -> &'static str {
        match self
        {
            Self::M0 => "M0 — BASELINE",
            Self::M1 => "M1 — BASELINE + O1",
            Self::M2 => "M2 — BASELINE + O1 + O2",
            Self::M3 => "M3 — BASELINE + O1 + O2 + ΔO",
        }
    }

    const fn feature_count(self) -> usize {
        match self
        {
            Self::M0 => M0_FEATURE_COUNT,
            Self::M1 => M1_FEATURE_COUNT,
            Self::M2 => M2_FEATURE_COUNT,
            Self::M3 => M3_FEATURE_COUNT,
        }
    }
}

#[derive(Clone, Debug)]
struct Record {
    baseline: [f64; BASELINE_FEATURE_COUNT],
    tdi: [f64; TDI_FEATURE_COUNT],
    overlaps: [f64; TARGET_HORIZON_COUNT],
    targets_u: [f64; TARGET_HORIZON_COUNT],
}

#[derive(Clone, Debug)]
struct RidgeModel {
    means: Vec<f64>,
    scales: Vec<f64>,
    coefficients: Vec<f64>,
}

#[derive(Clone, Debug)]
struct HorizonModels {
    models: Vec<RidgeModel>,
}

impl HorizonModels {
    fn get(&self, horizon_index: usize, layout: FeatureLayout) -> &RidgeModel {
        let index = horizon_index * MODEL_LAYOUT_COUNT + layout as usize;

        &self.models[index]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Metrics {
    mse: f64,
    mae: f64,
    r_squared: f64,
    spearman: f64,
    bias: f64,
    observed_mean: f64,
    predicted_mean: f64,
    calibration_intercept: f64,
    calibration_slope: f64,
    zero_fraction: f64,
    one_fraction: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ConfidenceInterval {
    lower: f64,
    median: f64,
    upper: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttemptContext {
    width: u8,
    seed: u64,
    attempt_index: usize,
}

impl AttemptContext {
    const fn new(width: u8, seed: u64, attempt_index: usize) -> Self {
        Self {
            width,
            seed,
            attempt_index,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailureCategory {
    Arithmetic,
    Cardinality,
    Structural,
    DynamicAnalysis,
    UnsupportedWidth,
    SeedRange,
    AttemptBudget,
    NoProgress,
    InvalidConfiguration,
}

impl std::fmt::Display for FailureCategory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self
        {
            Self::Arithmetic => "arithmetic",
            Self::Cardinality => "cardinality",
            Self::Structural => "structural",
            Self::DynamicAnalysis => "dynamic-analysis",
            Self::UnsupportedWidth => "unsupported-width",
            Self::SeedRange => "seed-range",
            Self::AttemptBudget => "attempt-budget",
            Self::NoProgress => "no-progress",
            Self::InvalidConfiguration => "invalid-configuration",
        };

        formatter.write_str(label)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EvaluationError {
    context: AttemptContext,
    category: FailureCategory,
    message: String,
}

impl EvaluationError {
    fn new(context: AttemptContext, category: FailureCategory, message: impl Into<String>) -> Self {
        Self {
            context,
            category,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} failure at width {}, seed {}, attempt {}: {}",
            self.category,
            self.context.width,
            self.context.seed,
            self.context.attempt_index,
            self.message
        )
    }
}

impl std::error::Error for EvaluationError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cardinality {
    Exact(u128),
    TooLarge { width: u8, exponent: u128 },
    Invalid { width: u8, reason: &'static str },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RejectionReason {
    ObservationFullyRecovered,
    InvalidObservationGeometry,
    TargetFullyRecovered { horizon: usize },
    InvalidTargetGeometry { horizon: usize },
    InvalidTransformedTarget { horizon: usize },
    NonFiniteFeature,
}

#[derive(Clone, Debug)]
enum CandidateOutcome {
    Accepted(Record),
    Rejected(RejectionReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GenerationLimits {
    max_attempts: usize,
    no_progress_limit: usize,
}

#[derive(Clone, Debug)]
struct GenerationReport {
    records: Vec<Record>,
    next_seed: u64,
    excluded: usize,
    attempts: usize,
    limits: GenerationLimits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TerminationDiagnostic {
    context: AttemptContext,
    category: FailureCategory,
    accepted: usize,
    excluded: usize,
    target_count: usize,
    limits: GenerationLimits,
    message: String,
}

impl TerminationDiagnostic {
    fn new(
        context: AttemptContext,
        category: FailureCategory,
        accepted: usize,
        excluded: usize,
        target_count: usize,
        limits: GenerationLimits,
        message: impl Into<String>,
    ) -> Self {
        Self {
            context,
            category,
            accepted,
            excluded,
            target_count,
            limits,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for TerminationDiagnostic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} termination at width {}, seed {}, attempt {}: {}; accepted={}, excluded={}, target={}, max_attempts={}, no_progress_limit={}",
            self.category,
            self.context.width,
            self.context.seed,
            self.context.attempt_index,
            self.message,
            self.accepted,
            self.excluded,
            self.target_count,
            self.limits.max_attempts,
            self.limits.no_progress_limit
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GenerationError {
    Evaluation(EvaluationError),
    AttemptBudgetExhausted(TerminationDiagnostic),
    NoProgress(TerminationDiagnostic),
}

impl std::fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::Evaluation(error) => error.fmt(formatter),
            Self::AttemptBudgetExhausted(diagnostic) | Self::NoProgress(diagnostic) =>
            {
                diagnostic.fmt(formatter)
            },
        }
    }
}

impl std::error::Error for GenerationError {}

#[derive(Clone, Copy, Debug)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix64(self.state)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

impl RidgeModel {
    fn predict_linear(&self, features: &[f64]) -> f64 {
        assert_eq!(features.len(), self.means.len());
        assert_eq!(features.len(), self.scales.len());
        assert_eq!(self.coefficients.len(), features.len() + 1);

        features
            .iter()
            .zip(&self.means)
            .zip(&self.scales)
            .zip(self.coefficients.iter().skip(1))
            .fold(
                self.coefficients[0],
                |accumulator, (((value, mean), scale), coefficient)| {
                    accumulator + coefficient * ((value - mean) / scale)
                },
            )
    }

    fn predict_probability(&self, features: &[f64]) -> f64 {
        self.predict_linear(features).clamp(0.0, 1.0)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);

    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

    mixed ^ (mixed >> 31)
}

fn state_count_cardinality(width: u8) -> Cardinality {
    let shift = u32::from(width);

    1_u128
        .checked_shl(shift)
        .map(Cardinality::Exact)
        .unwrap_or(Cardinality::TooLarge {
            width,
            exponent: u128::from(shift),
        })
}

fn successor_set_space_cardinality(width: u8) -> Cardinality {
    let states = match state_count_cardinality(width)
    {
        Cardinality::Exact(states) => states,
        other => return other,
    };

    let Ok(shift) = u32::try_from(states)
    else
    {
        return Cardinality::TooLarge {
            width,
            exponent: states,
        };
    };

    1_u128
        .checked_shl(shift)
        .map(Cardinality::Exact)
        .unwrap_or(Cardinality::TooLarge {
            width,
            exponent: states,
        })
}

fn generation_successor_set_space_cardinality(width: u8) -> Cardinality {
    if width > MAX_SUPPORTED_WIDTH
    {
        Cardinality::Invalid {
            width,
            reason: "width is unsupported by the u64 successor-mask evaluator",
        }
    }
    else
    {
        successor_set_space_cardinality(width)
    }
}

fn state_count(context: AttemptContext) -> Result<usize, EvaluationError> {
    if context.width > MAX_SUPPORTED_WIDTH
    {
        return Err(EvaluationError::new(
            context,
            FailureCategory::UnsupportedWidth,
            format!(
                "width {} exceeds maximum supported width {MAX_SUPPORTED_WIDTH}",
                context.width
            ),
        ));
    }

    match state_count_cardinality(context.width)
    {
        Cardinality::Exact(value) => usize::try_from(value).map_err(|_| {
            EvaluationError::new(
                context,
                FailureCategory::Cardinality,
                format!("state count {value} cannot be represented as usize"),
            )
        }),
        Cardinality::TooLarge { exponent, .. } => Err(EvaluationError::new(
            context,
            FailureCategory::Cardinality,
            format!("state count 2^{exponent} exceeds exact evaluator range"),
        )),
        Cardinality::Invalid { reason, .. } => Err(EvaluationError::new(
            context,
            FailureCategory::UnsupportedWidth,
            reason,
        )),
    }
}

fn nonempty_successor_set_count(context: AttemptContext) -> Result<u64, EvaluationError> {
    match generation_successor_set_space_cardinality(context.width)
    {
        Cardinality::Exact(space_count) =>
        {
            let nonempty_count = space_count.checked_sub(1).ok_or_else(|| {
                EvaluationError::new(
                    context,
                    FailureCategory::Arithmetic,
                    "successor-mask space underflow when removing empty mask",
                )
            })?;

            u64::try_from(nonempty_count).map_err(|_| {
                EvaluationError::new(
                    context,
                    FailureCategory::Cardinality,
                    format!(
                        "non-empty successor-mask count {nonempty_count} cannot be represented as u64"
                    ),
                )
            })
        },
        Cardinality::TooLarge { exponent, .. } => Err(EvaluationError::new(
            context,
            FailureCategory::Cardinality,
            format!("successor-mask space 2^{exponent} exceeds u128 exact range"),
        )),
        Cardinality::Invalid { reason, .. } => Err(EvaluationError::new(
            context,
            FailureCategory::UnsupportedWidth,
            reason,
        )),
    }
}

fn generate_successor_masks(context: AttemptContext) -> Result<Vec<u64>, EvaluationError> {
    let states = state_count(context)?;
    let mask_count = nonempty_successor_set_count(context)?;

    let mut masks = vec![0_u64; states];
    let mut generator = context.seed;

    for mask in &mut masks
    {
        generator = splitmix64(generator);
        *mask = generator % mask_count + 1;
    }

    Ok(masks)
}

fn build_system(context: AttemptContext, masks: &[u64]) -> Result<TableSystem, EvaluationError> {
    let states = state_count(context)?;

    if masks.len() != states
    {
        return Err(EvaluationError::new(
            context,
            FailureCategory::Structural,
            format!(
                "expected {states} successor masks, received {}",
                masks.len()
            ),
        ));
    }

    let mut system = TableSystem::new(context.width).map_err(|error| {
        EvaluationError::new(
            context,
            FailureCategory::Structural,
            format!("cannot create branching system: {error:?}"),
        )
    })?;

    for (source_bits, &mask) in masks.iter().enumerate()
    {
        let source = State::new(source_bits as u64, context.width).map_err(|error| {
            EvaluationError::new(
                context,
                FailureCategory::Structural,
                format!("cannot create source state {source_bits}: {error:?}"),
            )
        })?;

        let mut successors = Vec::new();

        for target in 0..states
        {
            let shift = u32::try_from(target).map_err(|_| {
                EvaluationError::new(
                    context,
                    FailureCategory::Arithmetic,
                    format!("successor target index {target} cannot be shifted"),
                )
            })?;

            let bit = 1_u64.checked_shl(shift).ok_or_else(|| {
                EvaluationError::new(
                    context,
                    FailureCategory::Arithmetic,
                    format!("successor target index {target} exceeds u64 mask width"),
                )
            })?;

            if mask & bit != 0
            {
                successors.push(State::new(target as u64, context.width).map_err(|error| {
                    EvaluationError::new(
                        context,
                        FailureCategory::Structural,
                        format!("cannot create target state {target}: {error:?}"),
                    )
                })?);
            }
        }

        system
            .insert(source, Action::Noop, successors)
            .map_err(|error| {
                EvaluationError::new(
                    context,
                    FailureCategory::Structural,
                    format!(
                        "cannot insert branching transition for state \
                     {source_bits}: {error:?}"
                    ),
                )
            })?;
    }

    Ok(system)
}

fn entropy_profile(
    context: AttemptContext,
    system: &TableSystem,
    initial: State,
) -> Result<[f64; OBSERVATION_HORIZON], EvaluationError> {
    let mut profile = [0.0_f64; OBSERVATION_HORIZON];

    for depth in 1..=OBSERVATION_HORIZON
    {
        profile[depth - 1] =
            uniform_branching_path_entropy_bits(system, initial, Action::Noop, depth).map_err(
                |error| {
                    EvaluationError::new(
                        context,
                        FailureCategory::DynamicAnalysis,
                        format!("branching entropy failed at depth {depth}: {error:?}"),
                    )
                },
            )?;
    }

    Ok(profile)
}

fn topology_profile(
    context: AttemptContext,
    system: &TableSystem,
    initial: State,
) -> Result<([f64; OBSERVATION_HORIZON], [f64; OBSERVATION_HORIZON]), EvaluationError> {
    let actions = [Action::Noop; OBSERVATION_HORIZON];

    let report = explore(system, initial, &actions).map_err(|error| {
        EvaluationError::new(
            context,
            FailureCategory::DynamicAnalysis,
            format!("branching exploration failed: {error:?}"),
        )
    })?;

    let mut reachable = [0.0_f64; OBSERVATION_HORIZON];
    let mut paths = [0.0_f64; OBSERVATION_HORIZON];

    for depth in 1..=OBSERVATION_HORIZON
    {
        reachable[depth - 1] = report.reachable_count(depth).ok_or_else(|| {
            EvaluationError::new(
                context,
                FailureCategory::Structural,
                format!("missing reachable layer {depth}"),
            )
        })? as f64;

        paths[depth - 1] = report.path_count(depth).ok_or_else(|| {
            EvaluationError::new(
                context,
                FailureCategory::Structural,
                format!("missing path-count layer {depth}"),
            )
        })? as f64;
    }

    Ok((reachable, paths))
}

fn ratio_value(ratio: &ExactRatio) -> f64 {
    ratio.as_f64()
}

fn target_horizon_index(horizon: usize) -> Option<usize> {
    TARGET_HORIZONS
        .iter()
        .position(|&candidate| candidate == horizon)
}

fn primary_horizon_index() -> usize {
    let index =
        target_horizon_index(PRIMARY_HORIZON).expect("primary horizon belongs to target horizons");

    debug_assert_eq!(index, PRIMARY_HORIZON_INDEX);

    index
}

fn feature_layout(record: &Record, layout: FeatureLayout) -> Vec<f64> {
    let mut features = Vec::with_capacity(layout.feature_count());
    features.extend_from_slice(&record.baseline);

    match layout
    {
        FeatureLayout::M0 =>
        {},
        FeatureLayout::M1 =>
        {
            features.push(record.tdi[0]);
        },
        FeatureLayout::M2 =>
        {
            features.push(record.tdi[0]);
            features.push(record.tdi[1]);
        },
        FeatureLayout::M3 =>
        {
            features.extend_from_slice(&record.tdi);
        },
    }

    debug_assert_eq!(features.len(), layout.feature_count());

    features
}

fn target_values(records: &[Record], horizon_index: usize) -> Vec<f64> {
    records
        .iter()
        .map(|record| record.targets_u[horizon_index])
        .collect()
}

fn overlap_values(records: &[Record], horizon_index: usize) -> Vec<f64> {
    records
        .iter()
        .map(|record| record.overlaps[horizon_index])
        .collect()
}

fn biguint_log2_from_u64_digits(digits: &[u64]) -> Result<f64, String> {
    let top = digits
        .last()
        .copied()
        .ok_or_else(|| "cannot calculate log2 of zero".to_owned())?;

    if top == 0
    {
        return Err("invalid leading zero BigUint limb".to_owned());
    }

    let top_bits = 64_usize - top.leading_zeros() as usize;
    let bit_length = (digits.len() - 1) * 64 + top_bits;

    let combined = if digits.len() >= 2
    {
        (u128::from(top) << 64) | u128::from(digits[digits.len() - 2])
    }
    else
    {
        u128::from(top)
    };

    let combined_bits = if digits.len() >= 2
    {
        top_bits + 64
    }
    else
    {
        top_bits
    };

    let shift = combined_bits.saturating_sub(53);
    let significant = (combined >> shift) as u64;
    let significant_bits = combined_bits - shift;

    let mantissa = significant as f64 / 2.0_f64.powi((significant_bits - 1) as i32);

    if !mantissa.is_finite() || !(1.0..2.0).contains(&mantissa)
    {
        return Err("invalid normalized BigUint mantissa".to_owned());
    }

    let logarithm = (bit_length - 1) as f64 + mantissa.log2();

    if !logarithm.is_finite()
    {
        return Err("non-finite BigUint logarithm".to_owned());
    }

    Ok(logarithm)
}

fn exact_overlap_deficit_u(ratio: &ExactRatio) -> Result<f64, String> {
    if ratio.numerator() >= ratio.denominator()
    {
        return Err("conditional overlap must be strictly below one".to_owned());
    }

    let deficit_numerator = ratio.denominator() - ratio.numerator();

    let numerator_log2 = biguint_log2_from_u64_digits(&deficit_numerator.to_u64_digits())?;

    let denominator_log2 = biguint_log2_from_u64_digits(&ratio.denominator().to_u64_digits())?;

    let transformed = denominator_log2 - numerator_log2;

    if !transformed.is_finite() || transformed < 0.0
    {
        return Err(format!(
            "invalid conditional target geometry: {transformed}"
        ));
    }

    Ok(transformed)
}

fn normalized_entropy(entropy_bits: f64, context: AttemptContext) -> Result<f64, EvaluationError> {
    let states = state_count(context)? as f64;
    let denominator = states.ln();

    if !denominator.is_finite() || denominator <= 0.0
    {
        return Err(EvaluationError::new(
            context,
            FailureCategory::Arithmetic,
            format!("invalid entropy normalizer for width {}", context.width),
        ));
    }

    let normalized = entropy_bits * std::f64::consts::LN_2 / denominator;

    if !normalized.is_finite()
    {
        return Err(EvaluationError::new(
            context,
            FailureCategory::Arithmetic,
            "non-finite normalized entropy",
        ));
    }

    Ok(normalized)
}

fn normalized_reachable(reachable: f64, context: AttemptContext) -> Result<f64, EvaluationError> {
    let states = state_count(context)? as f64;
    let normalized = reachable / states;

    if !normalized.is_finite()
    {
        return Err(EvaluationError::new(
            context,
            FailureCategory::Arithmetic,
            format!("non-finite reachable fraction for width {}", context.width),
        ));
    }

    Ok(normalized)
}

fn transformed_path_count(
    path_count: f64,
    context: AttemptContext,
) -> Result<f64, EvaluationError> {
    let transformed = path_count.ln_1p();

    if !transformed.is_finite()
    {
        return Err(EvaluationError::new(
            context,
            FailureCategory::Arithmetic,
            "non-finite transformed path count",
        ));
    }

    Ok(transformed)
}

fn analyze_seed(context: AttemptContext) -> Result<CandidateOutcome, EvaluationError> {
    let masks = generate_successor_masks(context)?;
    let system = build_system(context, &masks)?;

    let reference = State::new(0, context.width).map_err(|error| {
        EvaluationError::new(
            context,
            FailureCategory::Structural,
            format!("cannot create reference state: {error:?}"),
        )
    })?;

    let perturbation_node = context.width.checked_sub(1).ok_or_else(|| {
        EvaluationError::new(
            context,
            FailureCategory::Structural,
            "width zero cannot define the width-1 perturbation node",
        )
    })?;

    let perturbation = Action::Flip {
        node: perturbation_node,
    };

    let perturbed = perturbation.apply(reference).map_err(|error| {
        EvaluationError::new(
            context,
            FailureCategory::Structural,
            format!("cannot apply perturbation: {error:?}"),
        )
    })?;

    let reference_entropy = entropy_profile(context, &system, reference)?;
    let perturbed_entropy = entropy_profile(context, &system, perturbed)?;

    let (reference_reachable, reference_paths) = topology_profile(context, &system, reference)?;

    let (perturbed_reachable, perturbed_paths) = topology_profile(context, &system, perturbed)?;

    let observation = analyze_branching_recovery(
        &system,
        reference,
        perturbation,
        Action::Noop,
        OBSERVATION_HORIZON,
    )
    .map_err(|error| {
        EvaluationError::new(
            context,
            FailureCategory::DynamicAnalysis,
            format!(
                "observation recovery analysis failed for width \
             {}, seed {}: {error:?}",
                context.width, context.seed
            ),
        )
    })?;

    // Critère d’exclusion préenregistré : O2 = 1.
    if observation.fully_recovered()
    {
        return Ok(CandidateOutcome::Rejected(
            RejectionReason::ObservationFullyRecovered,
        ));
    }

    let observation_overlaps = observation.overlap_profile();

    if observation_overlaps.len() != OBSERVATION_HORIZON
    {
        return Err(EvaluationError::new(
            context,
            FailureCategory::Structural,
            format!(
                "expected {OBSERVATION_HORIZON} observation overlaps, \
             received {}",
                observation_overlaps.len()
            ),
        ));
    }

    let first_overlap = ratio_value(&observation_overlaps[0]);
    let second_overlap = ratio_value(&observation_overlaps[1]);

    if !first_overlap.is_finite()
        || !second_overlap.is_finite()
        || !(0.0..=1.0).contains(&first_overlap)
        || !(0.0..1.0).contains(&second_overlap)
    {
        return Ok(CandidateOutcome::Rejected(
            RejectionReason::InvalidObservationGeometry,
        ));
    }

    let mut overlaps = [0.0_f64; TARGET_HORIZON_COUNT];
    let mut targets_u = [0.0_f64; TARGET_HORIZON_COUNT];

    for (horizon_index, &horizon) in TARGET_HORIZONS.iter().enumerate()
    {
        let outcome =
            analyze_branching_recovery(&system, reference, perturbation, Action::Noop, horizon)
                .map_err(|error| {
                    EvaluationError::new(
                        context,
                        FailureCategory::DynamicAnalysis,
                        format!(
                            "target recovery analysis failed at horizon {horizon} \
                 for width {}, seed {}: {error:?}",
                            context.width, context.seed
                        ),
                    )
                })?;

        // Critère d’exclusion préenregistré :
        // déficit exact nul à un horizon cible.
        if outcome.fully_recovered()
        {
            return Ok(CandidateOutcome::Rejected(
                RejectionReason::TargetFullyRecovered { horizon },
            ));
        }

        let overlap_ratio = outcome.final_overlap().ok_or_else(|| {
            EvaluationError::new(
                context,
                FailureCategory::Structural,
                format!(
                    "target horizon {horizon} produced no overlap \
                     for width {}, seed {}",
                    context.width, context.seed
                ),
            )
        })?;

        let overlap = ratio_value(&overlap_ratio);

        if !overlap.is_finite() || !(0.0..1.0).contains(&overlap)
        {
            return Ok(CandidateOutcome::Rejected(
                RejectionReason::InvalidTargetGeometry { horizon },
            ));
        }

        let target_u = exact_overlap_deficit_u(&overlap_ratio).map_err(|error| {
            EvaluationError::new(
                context,
                FailureCategory::Arithmetic,
                format!(
                    "cannot calculate U_{horizon} for width {width}, \
                     seed {seed}: {error}",
                    width = context.width,
                    seed = context.seed
                ),
            )
        })?;

        if !target_u.is_finite() || target_u < 0.0
        {
            return Ok(CandidateOutcome::Rejected(
                RejectionReason::InvalidTransformedTarget { horizon },
            ));
        }

        overlaps[horizon_index] = overlap;
        targets_u[horizon_index] = target_u;
    }

    let baseline = [
        normalized_entropy(reference_entropy[0], context)?,
        normalized_entropy(reference_entropy[1], context)?,
        normalized_entropy(perturbed_entropy[0], context)?,
        normalized_entropy(perturbed_entropy[1], context)?,
        normalized_reachable(reference_reachable[0], context)?,
        normalized_reachable(reference_reachable[1], context)?,
        transformed_path_count(reference_paths[0], context)?,
        transformed_path_count(reference_paths[1], context)?,
        normalized_reachable(perturbed_reachable[0], context)?,
        normalized_reachable(perturbed_reachable[1], context)?,
        transformed_path_count(perturbed_paths[0], context)?,
        transformed_path_count(perturbed_paths[1], context)?,
        f64::from(context.width),
    ];

    let tdi = [
        first_overlap,
        second_overlap,
        second_overlap - first_overlap,
    ];

    if baseline.iter().chain(&tdi).any(|value| !value.is_finite())
    {
        return Ok(CandidateOutcome::Rejected(
            RejectionReason::NonFiniteFeature,
        ));
    }

    Ok(CandidateOutcome::Accepted(Record {
        baseline,
        tdi,
        overlaps,
        targets_u,
    }))
}

fn generate_records(
    width: u8,
    start_seed: u64,
    count: usize,
) -> Result<GenerationReport, GenerationError> {
    let limits = preregistered_generation_limits(width, start_seed, count)
        .map_err(GenerationError::Evaluation)?;

    generate_records_with_limits(width, start_seed, count, limits)
}

fn preregistered_generation_limits(
    width: u8,
    start_seed: u64,
    count: usize,
) -> Result<GenerationLimits, EvaluationError> {
    let context = AttemptContext::new(width, start_seed, 0);

    if count == 0
    {
        return Err(EvaluationError::new(
            context,
            FailureCategory::InvalidConfiguration,
            "record target must be positive",
        ));
    }

    let (attempt_multiplier, no_progress_limit) = match width
    {
        TRAIN_WIDTH_3 => (WIDTH_3_ATTEMPT_MULTIPLIER, WIDTH_3_NO_PROGRESS_LIMIT),
        TRAIN_WIDTH_4 => (WIDTH_4_ATTEMPT_MULTIPLIER, WIDTH_4_NO_PROGRESS_LIMIT),
        OOD_WIDTH_5 => (WIDTH_5_ATTEMPT_MULTIPLIER, WIDTH_5_NO_PROGRESS_LIMIT),
        OOD_WIDTH_6 => (WIDTH_6_ATTEMPT_MULTIPLIER, WIDTH_6_NO_PROGRESS_LIMIT),
        _ =>
        {
            return Err(EvaluationError::new(
                context,
                FailureCategory::UnsupportedWidth,
                format!("width {width} is not part of the TDI-5.1 preregistered populations"),
            ));
        },
    };

    let max_attempts = count.checked_mul(attempt_multiplier).ok_or_else(|| {
        EvaluationError::new(
            context,
            FailureCategory::Arithmetic,
            format!(
                "attempt budget overflow for target {count} and multiplier {attempt_multiplier}"
            ),
        )
    })?;

    Ok(GenerationLimits {
        max_attempts,
        no_progress_limit,
    })
}

fn generate_records_with_limits(
    width: u8,
    start_seed: u64,
    count: usize,
    limits: GenerationLimits,
) -> Result<GenerationReport, GenerationError> {
    generate_records_with_analyzer(width, start_seed, count, limits, analyze_seed)
}

fn seed_for_attempt(
    width: u8,
    start_seed: u64,
    attempt_index: usize,
) -> Result<u64, EvaluationError> {
    let attempt_offset = u64::try_from(attempt_index).map_err(|_| {
        EvaluationError::new(
            AttemptContext::new(width, start_seed, attempt_index),
            FailureCategory::SeedRange,
            format!("attempt index {attempt_index} cannot be represented as u64"),
        )
    })?;

    start_seed.checked_add(attempt_offset).ok_or_else(|| {
        EvaluationError::new(
            AttemptContext::new(width, start_seed, attempt_index),
            FailureCategory::SeedRange,
            format!("seed range overflow from start seed {start_seed} at attempt {attempt_index}"),
        )
    })
}

fn generate_records_with_analyzer<F>(
    width: u8,
    start_seed: u64,
    count: usize,
    limits: GenerationLimits,
    mut analyzer: F,
) -> Result<GenerationReport, GenerationError>
where
    F: FnMut(AttemptContext) -> Result<CandidateOutcome, EvaluationError>,
{
    if limits.max_attempts == 0 || limits.no_progress_limit == 0
    {
        return Err(GenerationError::Evaluation(EvaluationError::new(
            AttemptContext::new(width, start_seed, 0),
            FailureCategory::InvalidConfiguration,
            "generation limits must be positive",
        )));
    }

    if count == 0
    {
        return Err(GenerationError::Evaluation(EvaluationError::new(
            AttemptContext::new(width, start_seed, 0),
            FailureCategory::InvalidConfiguration,
            "record target must be positive",
        )));
    }

    let mut records = Vec::with_capacity(count);
    let mut excluded = 0_usize;
    let mut attempts = 0_usize;
    let mut attempts_without_progress = 0_usize;

    while records.len() < count
    {
        if attempts >= limits.max_attempts
        {
            let seed = seed_for_attempt(width, start_seed, attempts)
                .map_err(GenerationError::Evaluation)?;
            let diagnostic = TerminationDiagnostic::new(
                AttemptContext::new(width, seed, attempts),
                FailureCategory::AttemptBudget,
                records.len(),
                excluded,
                count,
                limits,
                "target record count remained unattainable before the deterministic attempt budget",
            );

            return Err(GenerationError::AttemptBudgetExhausted(diagnostic));
        }

        let seed =
            seed_for_attempt(width, start_seed, attempts).map_err(GenerationError::Evaluation)?;
        let context = AttemptContext::new(width, seed, attempts);

        match analyzer(context).map_err(GenerationError::Evaluation)?
        {
            CandidateOutcome::Accepted(record) =>
            {
                records.push(record);
                attempts_without_progress = 0;
            },
            CandidateOutcome::Rejected(reason) =>
            {
                let _deterministic_rejection = reason;
                excluded += 1;
                attempts_without_progress += 1;
            },
        }

        attempts += 1;

        if records.len() < count && attempts_without_progress >= limits.no_progress_limit
        {
            let diagnostic = TerminationDiagnostic::new(
                context,
                FailureCategory::NoProgress,
                records.len(),
                excluded,
                count,
                limits,
                format!(
                    "no accepted record observed for {attempts_without_progress} consecutive attempts"
                ),
            );

            return Err(GenerationError::NoProgress(diagnostic));
        }
    }

    let next_seed =
        seed_for_attempt(width, start_seed, attempts).map_err(GenerationError::Evaluation)?;

    Ok(GenerationReport {
        records,
        next_seed,
        excluded,
        attempts,
        limits,
    })
}

fn model_features(record: &Record, layout: FeatureLayout) -> Vec<f64> {
    feature_layout(record, layout)
}

fn feature_matrix<F>(records: &[Record], feature_fn: F) -> Vec<Vec<f64>>
where
    F: Fn(&Record) -> Vec<f64>,
{
    records.iter().map(feature_fn).collect()
}

fn fit_ridge(features: &[Vec<f64>], targets: &[f64]) -> Result<RidgeModel, String> {
    if features.is_empty()
    {
        return Err("cannot fit ridge regression on an empty dataset".to_owned());
    }

    if features.len() != targets.len()
    {
        return Err(format!(
            "feature/target length mismatch: {} versus {}",
            features.len(),
            targets.len()
        ));
    }

    let feature_count = features[0].len();

    if feature_count == 0
    {
        return Err("ridge regression requires at least one feature".to_owned());
    }

    if features.iter().any(|row| row.len() != feature_count)
    {
        return Err("inconsistent feature-vector lengths".to_owned());
    }

    let sample_count = features.len() as f64;
    let mut means = vec![0.0_f64; feature_count];

    for row in features
    {
        for (mean, value) in means.iter_mut().zip(row)
        {
            *mean += value;
        }
    }

    for mean in &mut means
    {
        *mean /= sample_count;
    }

    let mut scales = vec![0.0_f64; feature_count];

    for row in features
    {
        for ((scale, value), mean) in scales.iter_mut().zip(row).zip(&means)
        {
            let difference = value - mean;
            *scale += difference * difference;
        }
    }

    for scale in &mut scales
    {
        *scale = (*scale / sample_count).sqrt();

        if !scale.is_finite() || *scale <= 1.0e-12
        {
            *scale = 1.0;
        }
    }

    let dimension = feature_count + 1;
    let mut normal = vec![vec![0.0_f64; dimension]; dimension];
    let mut right_hand_side = vec![0.0_f64; dimension];

    for (row, &target) in features.iter().zip(targets)
    {
        let mut standardized = Vec::with_capacity(dimension);
        standardized.push(1.0);

        standardized.extend(
            row.iter()
                .zip(&means)
                .zip(&scales)
                .map(|((value, mean), scale)| (value - mean) / scale),
        );

        for (left_index, &left_value) in standardized.iter().enumerate()
        {
            right_hand_side[left_index] += left_value * target;

            for (right_index, &right_value) in standardized.iter().enumerate()
            {
                normal[left_index][right_index] += left_value * right_value;
            }
        }
    }

    for (index, row) in normal.iter_mut().enumerate().skip(1)
    {
        row[index] += RIDGE_LAMBDA;
    }

    let coefficients = solve_linear_system(normal, right_hand_side)?;

    Ok(RidgeModel {
        means,
        scales,
        coefficients,
    })
}

fn fit_horizon_models(
    records: &[Record],
    target_scalers: &[TargetScaler; TARGET_HORIZON_COUNT],
) -> Result<HorizonModels, String> {
    let mut models = Vec::with_capacity(TARGET_HORIZON_COUNT * MODEL_LAYOUT_COUNT);

    for (horizon_index, scaler) in target_scalers.iter().copied().enumerate()
    {
        let raw_targets = target_values(records, horizon_index);

        let standardized_targets = raw_targets
            .iter()
            .map(|&value| scaler.standardize(value))
            .collect::<Vec<_>>();

        for layout in FeatureLayout::ALL
        {
            let matrix = feature_matrix(records, |record| model_features(record, layout));

            models.push(fit_ridge(&matrix, &standardized_targets)?);
        }
    }

    Ok(HorizonModels { models })
}

fn solve_linear_system(
    mut matrix: Vec<Vec<f64>>,
    mut right_hand_side: Vec<f64>,
) -> Result<Vec<f64>, String> {
    let dimension = matrix.len();

    if dimension == 0 || right_hand_side.len() != dimension
    {
        return Err("invalid linear-system dimensions".to_owned());
    }

    if matrix.iter().any(|row| row.len() != dimension)
    {
        return Err("linear-system matrix is not square".to_owned());
    }

    for column in 0..dimension
    {
        let pivot_row = (column..dimension)
            .max_by(|&left, &right| {
                matrix[left][column]
                    .abs()
                    .total_cmp(&matrix[right][column].abs())
            })
            .ok_or_else(|| "missing pivot row".to_owned())?;

        let pivot_value = matrix[pivot_row][column];

        if !pivot_value.is_finite() || pivot_value.abs() <= 1.0e-12
        {
            return Err(format!(
                "singular or ill-conditioned normal matrix at column {column}"
            ));
        }

        if pivot_row != column
        {
            matrix.swap(pivot_row, column);
            right_hand_side.swap(pivot_row, column);
        }

        let pivot_values = matrix[column].clone();
        let pivot_denominator = pivot_values[column];
        let pivot_right_hand_side = right_hand_side[column];

        for (row_index, row_values) in matrix.iter_mut().enumerate().skip(column + 1)
        {
            let factor = row_values[column] / pivot_denominator;

            row_values[column] = 0.0;

            for (value, pivot_value) in row_values.iter_mut().zip(&pivot_values).skip(column + 1)
            {
                *value -= factor * pivot_value;
            }

            right_hand_side[row_index] -= factor * pivot_right_hand_side;
        }
    }

    let mut solution = vec![0.0_f64; dimension];

    for row in (0..dimension).rev()
    {
        let trailing_sum = matrix[row]
            .iter()
            .enumerate()
            .skip(row + 1)
            .map(|(column, coefficient)| coefficient * solution[column])
            .sum::<f64>();

        solution[row] = (right_hand_side[row] - trailing_sum) / matrix[row][row];

        if !solution[row].is_finite()
        {
            return Err(format!("non-finite linear-system solution at row {row}"));
        }
    }

    Ok(solution)
}

fn predictions(model: &RidgeModel, features: &[Vec<f64>]) -> Vec<f64> {
    features
        .iter()
        .map(|row| model.predict_probability(row))
        .collect()
}

fn calculate_metrics(targets: &[f64], predicted: &[f64]) -> Metrics {
    assert_eq!(targets.len(), predicted.len());
    assert!(!targets.is_empty());

    let sample_count = targets.len() as f64;
    let observed_mean = targets.iter().sum::<f64>() / sample_count;
    let predicted_mean = predicted.iter().sum::<f64>() / sample_count;

    let mut squared_error = 0.0_f64;
    let mut absolute_error = 0.0_f64;
    let mut total_variance = 0.0_f64;
    let mut calibration_covariance = 0.0_f64;
    let mut prediction_variance = 0.0_f64;
    let mut zero_count = 0_usize;
    let mut one_count = 0_usize;

    for (&target, &prediction) in targets.iter().zip(predicted)
    {
        let residual = target - prediction;
        squared_error += residual * residual;
        absolute_error += residual.abs();

        let centered_target = target - observed_mean;
        let centered_prediction = prediction - predicted_mean;

        total_variance += centered_target * centered_target;
        calibration_covariance += centered_prediction * centered_target;
        prediction_variance += centered_prediction * centered_prediction;

        if prediction == 0.0
        {
            zero_count += 1;
        }

        if prediction == 1.0
        {
            one_count += 1;
        }
    }

    let r_squared = if total_variance <= 1.0e-15
    {
        0.0
    }
    else
    {
        1.0 - squared_error / total_variance
    };

    let calibration_slope = if prediction_variance <= 1.0e-15
    {
        0.0
    }
    else
    {
        calibration_covariance / prediction_variance
    };

    let calibration_intercept = observed_mean - calibration_slope * predicted_mean;

    Metrics {
        mse: squared_error / sample_count,
        mae: absolute_error / sample_count,
        r_squared,
        spearman: spearman_correlation(targets, predicted),
        bias: predicted_mean - observed_mean,
        observed_mean,
        predicted_mean,
        calibration_intercept,
        calibration_slope,
        zero_fraction: zero_count as f64 / sample_count,
        one_fraction: one_count as f64 / sample_count,
    }
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();

    indices.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then_with(|| left.cmp(&right))
    });

    let mut ranks = vec![0.0_f64; values.len()];
    let mut start = 0_usize;

    while start < indices.len()
    {
        let mut end = start + 1;

        while end < indices.len()
            && values[indices[start]].total_cmp(&values[indices[end]]) == std::cmp::Ordering::Equal
        {
            end += 1;
        }

        let average_rank = (start + 1 + end) as f64 / 2.0;

        for &index in &indices[start..end]
        {
            ranks[index] = average_rank;
        }

        start = end;
    }

    ranks
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());

    let count = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / count;
    let right_mean = right.iter().sum::<f64>() / count;

    let mut covariance = 0.0_f64;
    let mut left_variance = 0.0_f64;
    let mut right_variance = 0.0_f64;

    for (&left_value, &right_value) in left.iter().zip(right)
    {
        let centered_left = left_value - left_mean;
        let centered_right = right_value - right_mean;

        covariance += centered_left * centered_right;
        left_variance += centered_left * centered_left;
        right_variance += centered_right * centered_right;
    }

    let denominator = (left_variance * right_variance).sqrt();

    if denominator <= 1.0e-15
    {
        0.0
    }
    else
    {
        covariance / denominator
    }
}

fn spearman_correlation(left: &[f64], right: &[f64]) -> f64 {
    let left_ranks = average_ranks(left);
    let right_ranks = average_ranks(right);

    pearson_correlation(&left_ranks, &right_ranks)
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;

    if lower == upper
    {
        sorted[lower]
    }
    else
    {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn confidence_interval(mut values: Vec<f64>) -> ConfidenceInterval {
    values.sort_by(f64::total_cmp);

    ConfidenceInterval {
        lower: percentile(&values, 0.025),
        median: percentile(&values, 0.500),
        upper: percentile(&values, 0.975),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TargetScaler {
    mean: f64,
    scale: f64,
}

impl TargetScaler {
    fn fit(records: &[Record], horizon_index: usize) -> Result<Self, String> {
        let values = records
            .iter()
            .map(|record| record.targets_u[horizon_index])
            .collect::<Vec<_>>();

        if values.is_empty()
        {
            return Err("training population contains no target values".to_owned());
        }

        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;

        let variance = values
            .iter()
            .map(|value| {
                let difference = value - mean;
                difference * difference
            })
            .sum::<f64>()
            / count;

        let scale = variance.sqrt();

        if !mean.is_finite() || !scale.is_finite()
        {
            return Err("target has invalid training geometry".to_owned());
        }

        let scale = if scale <= 1.0e-12 { 1.0 } else { scale };

        Ok(Self { mean, scale })
    }

    fn standardize(self, value: f64) -> f64 {
        (value - self.mean) / self.scale
    }

    fn unstandardize(self, value: f64) -> f64 {
        self.mean + self.scale * value
    }
}

fn fit_target_scalers(records: &[Record]) -> Result<[TargetScaler; TARGET_HORIZON_COUNT], String> {
    let mut scalers = Vec::with_capacity(TARGET_HORIZON_COUNT);

    for horizon_index in 0..TARGET_HORIZON_COUNT
    {
        scalers.push(TargetScaler::fit(records, horizon_index)?);
    }

    scalers.try_into().map_err(|values: Vec<TargetScaler>| {
        format!(
            "expected {TARGET_HORIZON_COUNT} target scalers, received {}",
            values.len()
        )
    })
}

fn print_model(label: &str, model: &RidgeModel) {
    println!();
    println!("{label}");
    println!("  intercept : {:.12}", model.coefficients[0]);

    for index in 0..model.means.len()
    {
        println!(
            "  feature {index:02} | moyenne={:.12} | \
             échelle={:.12} | coefficient={:.12}",
            model.means[index],
            model.scales[index],
            model.coefficients[index + 1],
        );
    }
}

fn print_interval(label: &str, interval: ConfidenceInterval) {
    println!(
        "{label}: [{:.9}, {:.9}] (médiane {:.9})",
        interval.lower, interval.upper, interval.median
    );
}

fn ensure_seed_ranges(ranges: &[(u64, u64, &str)]) -> Result<(), String> {
    for pair in ranges.windows(2)
    {
        let (_, previous_end, previous_label) = pair[0];
        let (next_start, _, next_label) = pair[1];

        if previous_end > next_start
        {
            return Err(format!(
                "seed ranges overlap: {previous_label} ends at \
                 {previous_end}, {next_label} starts at {next_start}"
            ));
        }
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct Tdi5PredictionSet {
    standardized: Vec<f64>,
    reconstructed_overlap: Vec<f64>,
    clipped_overlap_count: usize,
}

#[derive(Clone, Debug)]
struct Tdi5LayoutEvaluation {
    layout: FeatureLayout,
    standardized: Metrics,
    reconstructed: Metrics,
    predictions: Tdi5PredictionSet,
}

#[derive(Clone, Copy, Debug)]
struct Tdi5BootstrapIntervals {
    standardized_mse: ConfidenceInterval,
    reconstructed_mse: ConfidenceInterval,
    reconstructed_mae: ConfidenceInterval,
}

fn tdi5_relative_reduction(baseline: f64, challenger: f64) -> f64 {
    if !baseline.is_finite() || !challenger.is_finite() || baseline.abs() <= 1.0e-15
    {
        0.0
    }
    else
    {
        (baseline - challenger) / baseline
    }
}

fn tdi5_reconstruct_overlap(target_u: f64) -> (f64, bool) {
    let raw = 1.0 - 2.0_f64.powf(-target_u);

    if !raw.is_finite()
    {
        return (0.0, true);
    }

    let clipped = raw.clamp(0.0, 1.0);

    (clipped, clipped != raw)
}

fn tdi5_predict(
    records: &[Record],
    horizon_index: usize,
    layout: FeatureLayout,
    model: &RidgeModel,
    scaler: TargetScaler,
) -> Result<Tdi5PredictionSet, String> {
    let mut standardized = Vec::with_capacity(records.len());
    let mut reconstructed_overlap = Vec::with_capacity(records.len());

    let mut clipped_overlap_count = 0_usize;

    for record in records
    {
        let features = feature_layout(record, layout);
        let prediction = model.predict_linear(&features);

        if !prediction.is_finite()
        {
            return Err(format!(
                "non-finite standardized prediction for {} at horizon {}",
                layout.label(),
                TARGET_HORIZONS[horizon_index],
            ));
        }

        let target_u = scaler.unstandardize(prediction);

        if !target_u.is_finite()
        {
            return Err(format!(
                "non-finite unstandardized prediction for {} at horizon {}",
                layout.label(),
                TARGET_HORIZONS[horizon_index],
            ));
        }

        let (overlap, clipped) = tdi5_reconstruct_overlap(target_u);

        clipped_overlap_count += usize::from(clipped);
        standardized.push(prediction);
        reconstructed_overlap.push(overlap);
    }

    Ok(Tdi5PredictionSet {
        standardized,
        reconstructed_overlap,
        clipped_overlap_count,
    })
}

fn tdi5_evaluate_horizon(
    records: &[Record],
    horizon_index: usize,
    models: &HorizonModels,
    scalers: &[TargetScaler; TARGET_HORIZON_COUNT],
) -> Result<Vec<Tdi5LayoutEvaluation>, String> {
    if records.is_empty()
    {
        return Err("cannot evaluate an empty population".to_owned());
    }

    let scaler = scalers[horizon_index];

    let standardized_targets = records
        .iter()
        .map(|record| scaler.standardize(record.targets_u[horizon_index]))
        .collect::<Vec<_>>();

    let overlap_targets = overlap_values(records, horizon_index);

    let mut evaluations = Vec::with_capacity(MODEL_LAYOUT_COUNT);

    for layout in FeatureLayout::ALL
    {
        let predictions = tdi5_predict(
            records,
            horizon_index,
            layout,
            models.get(horizon_index, layout),
            scaler,
        )?;

        let standardized = calculate_metrics(&standardized_targets, &predictions.standardized);

        let reconstructed = calculate_metrics(&overlap_targets, &predictions.reconstructed_overlap);

        evaluations.push(Tdi5LayoutEvaluation {
            layout,
            standardized,
            reconstructed,
            predictions,
        });
    }

    Ok(evaluations)
}

fn tdi5_layout_evaluation(
    evaluations: &[Tdi5LayoutEvaluation],
    layout: FeatureLayout,
) -> &Tdi5LayoutEvaluation {
    evaluations
        .iter()
        .find(|evaluation| evaluation.layout == layout)
        .expect("all preregistered layouts are evaluated")
}

fn tdi5_paired_bootstrap(
    records: &[Record],
    horizon_index: usize,
    scaler: TargetScaler,
    baseline: &Tdi5PredictionSet,
    challenger: &Tdi5PredictionSet,
) -> Result<Tdi5BootstrapIntervals, String> {
    let count = records.len();

    if count == 0
        || baseline.standardized.len() != count
        || challenger.standardized.len() != count
        || baseline.reconstructed_overlap.len() != count
        || challenger.reconstructed_overlap.len() != count
    {
        return Err("invalid paired-bootstrap dimensions".to_owned());
    }

    let mut generator = DeterministicRng::new(BOOTSTRAP_SEED);

    let mut standardized_mse = Vec::with_capacity(BOOTSTRAP_REPLICATES);

    let mut reconstructed_mse = Vec::with_capacity(BOOTSTRAP_REPLICATES);

    let mut reconstructed_mae = Vec::with_capacity(BOOTSTRAP_REPLICATES);

    for _ in 0..BOOTSTRAP_REPLICATES
    {
        let mut baseline_standardized_squared = 0.0;
        let mut challenger_standardized_squared = 0.0;

        let mut baseline_overlap_squared = 0.0;
        let mut challenger_overlap_squared = 0.0;

        let mut baseline_overlap_absolute = 0.0;
        let mut challenger_overlap_absolute = 0.0;

        for _ in 0..count
        {
            let index = generator.index(count);
            let record = &records[index];

            let standardized_target = scaler.standardize(record.targets_u[horizon_index]);

            let baseline_standardized_residual = standardized_target - baseline.standardized[index];

            let challenger_standardized_residual =
                standardized_target - challenger.standardized[index];

            baseline_standardized_squared +=
                baseline_standardized_residual * baseline_standardized_residual;

            challenger_standardized_squared +=
                challenger_standardized_residual * challenger_standardized_residual;

            let overlap_target = record.overlaps[horizon_index];

            let baseline_overlap_residual = overlap_target - baseline.reconstructed_overlap[index];

            let challenger_overlap_residual =
                overlap_target - challenger.reconstructed_overlap[index];

            baseline_overlap_squared += baseline_overlap_residual * baseline_overlap_residual;

            challenger_overlap_squared += challenger_overlap_residual * challenger_overlap_residual;

            baseline_overlap_absolute += baseline_overlap_residual.abs();

            challenger_overlap_absolute += challenger_overlap_residual.abs();
        }

        let denominator = count as f64;

        standardized_mse.push(
            baseline_standardized_squared / denominator
                - challenger_standardized_squared / denominator,
        );

        reconstructed_mse.push(
            baseline_overlap_squared / denominator - challenger_overlap_squared / denominator,
        );

        reconstructed_mae.push(
            baseline_overlap_absolute / denominator - challenger_overlap_absolute / denominator,
        );
    }

    Ok(Tdi5BootstrapIntervals {
        standardized_mse: confidence_interval(standardized_mse),
        reconstructed_mse: confidence_interval(reconstructed_mse),
        reconstructed_mae: confidence_interval(reconstructed_mae),
    })
}

fn tdi5_print_bootstrap_intervals(label: &str, intervals: Tdi5BootstrapIntervals) {
    println!();
    println!("{label}");

    print_interval(
        "  IC 95 % amélioration MSE U6 standardisée",
        intervals.standardized_mse,
    );

    print_interval(
        "  IC 95 % amélioration MSE O6 reconstruite",
        intervals.reconstructed_mse,
    );

    print_interval(
        "  IC 95 % amélioration MAE O6 reconstruite",
        intervals.reconstructed_mae,
    );
}

fn tdi5_print_metrics(label: &str, metrics: Metrics) {
    println!("{label}");
    println!("  MSE                    : {:.12}", metrics.mse);
    println!("  MAE                    : {:.12}", metrics.mae);
    println!("  R²                     : {:.12}", metrics.r_squared);
    println!("  Spearman               : {:.12}", metrics.spearman);
    println!("  biais                  : {:.12}", metrics.bias);
    println!("  moyenne observée       : {:.12}", metrics.observed_mean);
    println!("  moyenne prédite        : {:.12}", metrics.predicted_mean);
    println!(
        "  calibration intercept  : {:.12}",
        metrics.calibration_intercept
    );
    println!(
        "  calibration pente      : {:.12}",
        metrics.calibration_slope
    );
    println!("  fraction borne basse   : {:.12}", metrics.zero_fraction);
    println!("  fraction borne haute   : {:.12}", metrics.one_fraction);
}

fn tdi5_print_evaluations(
    population_label: &str,
    horizon_index: usize,
    evaluations: &[Tdi5LayoutEvaluation],
) {
    println!();
    println!(
        "=== {population_label} — U_{} ===",
        TARGET_HORIZONS[horizon_index]
    );

    for evaluation in evaluations
    {
        println!();
        println!("{}", evaluation.layout.label());

        tdi5_print_metrics("  espace U standardisé", evaluation.standardized);

        tdi5_print_metrics("  espace O reconstruit", evaluation.reconstructed);

        println!(
            "  prédictions O ramenées aux bornes : {} / {}",
            evaluation.predictions.clipped_overlap_count,
            evaluation.predictions.reconstructed_overlap.len(),
        );
    }

    let baseline = tdi5_layout_evaluation(evaluations, FeatureLayout::M0);

    let challenger = tdi5_layout_evaluation(evaluations, FeatureLayout::M3);

    println!(
        "  réduction relative MSE U M0→M3 : {:.9} %",
        tdi5_relative_reduction(baseline.standardized.mse, challenger.standardized.mse,) * 100.0
    );

    println!(
        "  amélioration MSE O M0→M3       : {:.12}",
        baseline.reconstructed.mse - challenger.reconstructed.mse
    );

    println!(
        "  amélioration MAE O M0→M3       : {:.12}",
        baseline.reconstructed.mae - challenger.reconstructed.mae
    );
}

fn tdi5_print_population_geometry(label: &str, records: &[Record]) {
    println!();
    println!("=== GÉOMÉTRIE — {label} ===");
    println!("systèmes : {}", records.len());

    for (horizon_index, &horizon) in TARGET_HORIZONS.iter().enumerate()
    {
        let values = target_values(records, horizon_index);
        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;

        let variance = values
            .iter()
            .map(|value| {
                let difference = value - mean;
                difference * difference
            })
            .sum::<f64>()
            / count;

        let minimum = values
            .iter()
            .copied()
            .min_by(f64::total_cmp)
            .expect("non-empty population");

        let maximum = values
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .expect("non-empty population");

        println!(
            "  U_{horizon} | moyenne={mean:.12} | \
             écart-type={:.12} | min={minimum:.12} | max={maximum:.12}",
            variance.sqrt()
        );
    }
}

fn tdi5_print_models(models: &HorizonModels, scalers: &[TargetScaler; TARGET_HORIZON_COUNT]) {
    println!();
    println!("=== NORMALISATIONS ET MODÈLES ===");

    for (horizon_index, &horizon) in TARGET_HORIZONS.iter().enumerate()
    {
        let scaler = scalers[horizon_index];

        println!();
        println!(
            "U_{horizon} | moyenne cible={:.12} | échelle cible={:.12}",
            scaler.mean, scaler.scale,
        );

        for layout in FeatureLayout::ALL
        {
            print_model(
                &format!("U_{horizon} — {}", layout.label()),
                models.get(horizon_index, layout),
            );
        }
    }
}

fn tdi5_command_output(program: &str, arguments: &[&str]) -> String {
    std::process::Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| "indisponible".to_owned())
}

fn tdi5_fit_direct_models(training: &[Record]) -> Result<(RidgeModel, RidgeModel), String> {
    let horizon_index = primary_horizon_index();
    let targets = overlap_values(training, horizon_index);

    let baseline = feature_matrix(training, |record| feature_layout(record, FeatureLayout::M0));

    let challenger = feature_matrix(training, |record| feature_layout(record, FeatureLayout::M3));

    Ok((
        fit_ridge(&baseline, &targets)?,
        fit_ridge(&challenger, &targets)?,
    ))
}

fn tdi5_print_direct_comparator(
    label: &str,
    records: &[Record],
    baseline_model: &RidgeModel,
    challenger_model: &RidgeModel,
) {
    let horizon_index = primary_horizon_index();
    let targets = overlap_values(records, horizon_index);

    let baseline_features =
        feature_matrix(records, |record| feature_layout(record, FeatureLayout::M0));

    let challenger_features =
        feature_matrix(records, |record| feature_layout(record, FeatureLayout::M3));

    let baseline_predictions = predictions(baseline_model, &baseline_features);

    let challenger_predictions = predictions(challenger_model, &challenger_features);

    let baseline = calculate_metrics(&targets, &baseline_predictions);

    let challenger = calculate_metrics(&targets, &challenger_predictions);

    println!();
    println!("=== COMPARATEUR DIRECT O6 — {label} ===");

    tdi5_print_metrics("M0 direct", baseline);
    tdi5_print_metrics("M3 direct", challenger);

    println!(
        "  réduction relative MSE directe : {:.9} %",
        tdi5_relative_reduction(baseline.mse, challenger.mse,) * 100.0
    );
}

fn run_termination_smoke() -> Result<(), String> {
    println!("=== TDI-5.1 TERMINATION SMOKE ===");

    let width_6_space = successor_set_space_cardinality(OOD_WIDTH_6);

    if width_6_space != Cardinality::Exact(18_446_744_073_709_551_616_u128)
    {
        return Err(format!("unexpected width-6 cardinality: {width_6_space:?}"));
    }

    let limits = GenerationLimits {
        max_attempts: 64,
        no_progress_limit: 64,
    };

    let report = generate_records_with_limits(TRAIN_WIDTH_3, TRAIN_WIDTH_3_SEED_OFFSET, 1, limits)
        .map_err(|error| error.to_string())?;

    println!("width 6 successor-set space : 18446744073709551616");
    println!(
        "width 3 smoke accepted       : {} in {} attempts",
        report.records.len(),
        report.attempts
    );
    println!("bounded smoke result         : PASS");

    Ok(())
}

fn main() -> Result<(), String> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();

    match arguments.as_slice()
    {
        [] => run_full_experiment(),
        [flag] if flag == "--termination-smoke" => run_termination_smoke(),
        _ => Err("usage: tdi-continuous-deficit-geometry-v51 [--termination-smoke]".to_owned()),
    }
}

fn run_full_experiment() -> Result<(), String> {
    println!("Generating preregistered TDI-5.1 width-3 training systems...");

    let GenerationReport {
        records: training_width_3,
        next_seed: training_width_3_next_seed,
        excluded: training_width_3_excluded,
        attempts: training_width_3_attempts,
        limits: training_width_3_limits,
    } = generate_records(
        TRAIN_WIDTH_3,
        TRAIN_WIDTH_3_SEED_OFFSET,
        TRAIN_WIDTH_3_SYSTEMS,
    )
    .map_err(|error| error.to_string())?;

    println!("Generating untouched TDI-5.1 width-3 holdout systems...");

    let GenerationReport {
        records: holdout_width_3,
        next_seed: holdout_width_3_next_seed,
        excluded: holdout_width_3_excluded,
        attempts: holdout_width_3_attempts,
        limits: holdout_width_3_limits,
    } = generate_records(
        TRAIN_WIDTH_3,
        HOLDOUT_WIDTH_3_SEED_OFFSET,
        HOLDOUT_WIDTH_3_SYSTEMS,
    )
    .map_err(|error| error.to_string())?;

    println!("Generating preregistered TDI-5.1 width-4 training systems...");

    let GenerationReport {
        records: training_width_4,
        next_seed: training_width_4_next_seed,
        excluded: training_width_4_excluded,
        attempts: training_width_4_attempts,
        limits: training_width_4_limits,
    } = generate_records(
        TRAIN_WIDTH_4,
        TRAIN_WIDTH_4_SEED_OFFSET,
        TRAIN_WIDTH_4_SYSTEMS,
    )
    .map_err(|error| error.to_string())?;

    println!("Generating untouched TDI-5.1 width-4 holdout systems...");

    let GenerationReport {
        records: holdout_width_4,
        next_seed: holdout_width_4_next_seed,
        excluded: holdout_width_4_excluded,
        attempts: holdout_width_4_attempts,
        limits: holdout_width_4_limits,
    } = generate_records(
        TRAIN_WIDTH_4,
        HOLDOUT_WIDTH_4_SEED_OFFSET,
        HOLDOUT_WIDTH_4_SYSTEMS,
    )
    .map_err(|error| error.to_string())?;

    println!("Generating untouched TDI-5.1 width-5 OOD systems...");

    let GenerationReport {
        records: holdout_width_5,
        next_seed: holdout_width_5_next_seed,
        excluded: holdout_width_5_excluded,
        attempts: holdout_width_5_attempts,
        limits: holdout_width_5_limits,
    } = generate_records(OOD_WIDTH_5, OOD_WIDTH_5_SEED_OFFSET, OOD_WIDTH_5_SYSTEMS)
        .map_err(|error| error.to_string())?;

    println!("Generating untouched TDI-5.1 width-6 extreme OOD systems...");

    let GenerationReport {
        records: holdout_width_6,
        next_seed: holdout_width_6_next_seed,
        excluded: holdout_width_6_excluded,
        attempts: holdout_width_6_attempts,
        limits: holdout_width_6_limits,
    } = generate_records(OOD_WIDTH_6, OOD_WIDTH_6_SEED_OFFSET, OOD_WIDTH_6_SYSTEMS)
        .map_err(|error| error.to_string())?;

    ensure_seed_ranges(&[
        (
            TRAIN_WIDTH_3_SEED_OFFSET,
            training_width_3_next_seed,
            "train w3",
        ),
        (
            HOLDOUT_WIDTH_3_SEED_OFFSET,
            holdout_width_3_next_seed,
            "holdout w3",
        ),
        (
            TRAIN_WIDTH_4_SEED_OFFSET,
            training_width_4_next_seed,
            "train w4",
        ),
        (
            HOLDOUT_WIDTH_4_SEED_OFFSET,
            holdout_width_4_next_seed,
            "holdout w4",
        ),
        (OOD_WIDTH_5_SEED_OFFSET, holdout_width_5_next_seed, "OOD w5"),
        (OOD_WIDTH_6_SEED_OFFSET, holdout_width_6_next_seed, "OOD w6"),
    ])?;

    let mut training = training_width_3.clone();
    training.extend(training_width_4.iter().cloned());

    let mut holdout_combined = holdout_width_3.clone();
    holdout_combined.extend(holdout_width_4.iter().cloned());

    let target_scalers = fit_target_scalers(&training)?;
    let models = fit_horizon_models(&training, &target_scalers)?;

    println!();
    println!("=== IDENTITÉ TDI-5.1 ===");
    println!(
        "git HEAD : {}",
        tdi5_command_output("git", &["rev-parse", "HEAD"])
    );
    println!(
        "rustc    : {}",
        tdi5_command_output("rustc", &["--version"])
    );
    println!(
        "cargo    : {}",
        tdi5_command_output("cargo", &["--version"])
    );
    println!("observation horizon : {OBSERVATION_HORIZON}");
    println!("target horizons     : {:?}", TARGET_HORIZONS);
    println!("primary horizon     : {PRIMARY_HORIZON}");
    println!("ridge lambda        : {RIDGE_LAMBDA}");
    println!("bootstrap replicates: {BOOTSTRAP_REPLICATES}");
    println!("bootstrap seed      : 0x{BOOTSTRAP_SEED:016X}");
    println!("max supported width : {MAX_SUPPORTED_WIDTH}");
    println!(
        "width 6 successor-set space : {}",
        match successor_set_space_cardinality(OOD_WIDTH_6)
        {
            Cardinality::Exact(value) => value.to_string(),
            other => format!("{other:?}"),
        }
    );

    println!();
    println!("=== POPULATIONS, GRAINES ET BORNES ===");

    for (label, accepted, excluded, attempts, limits, initial_seed, final_seed) in [
        (
            "train w3",
            training_width_3.len(),
            training_width_3_excluded,
            training_width_3_attempts,
            training_width_3_limits,
            TRAIN_WIDTH_3_SEED_OFFSET,
            training_width_3_next_seed,
        ),
        (
            "holdout w3",
            holdout_width_3.len(),
            holdout_width_3_excluded,
            holdout_width_3_attempts,
            holdout_width_3_limits,
            HOLDOUT_WIDTH_3_SEED_OFFSET,
            holdout_width_3_next_seed,
        ),
        (
            "train w4",
            training_width_4.len(),
            training_width_4_excluded,
            training_width_4_attempts,
            training_width_4_limits,
            TRAIN_WIDTH_4_SEED_OFFSET,
            training_width_4_next_seed,
        ),
        (
            "holdout w4",
            holdout_width_4.len(),
            holdout_width_4_excluded,
            holdout_width_4_attempts,
            holdout_width_4_limits,
            HOLDOUT_WIDTH_4_SEED_OFFSET,
            holdout_width_4_next_seed,
        ),
        (
            "OOD w5",
            holdout_width_5.len(),
            holdout_width_5_excluded,
            holdout_width_5_attempts,
            holdout_width_5_limits,
            OOD_WIDTH_5_SEED_OFFSET,
            holdout_width_5_next_seed,
        ),
        (
            "OOD w6",
            holdout_width_6.len(),
            holdout_width_6_excluded,
            holdout_width_6_attempts,
            holdout_width_6_limits,
            OOD_WIDTH_6_SEED_OFFSET,
            holdout_width_6_next_seed,
        ),
    ]
    {
        println!(
            "{label:12} | acceptés={accepted} | exclus={excluded} | tentatives={attempts} | \
             max_tentatives={} | seuil_sans_progrès={} | graine initiale={initial_seed} | \
             finale exclusive={final_seed}",
            limits.max_attempts, limits.no_progress_limit
        );
    }

    let populations: [(&str, &[Record]); 8] = [
        ("train combiné w3+w4", &training),
        ("holdout w3", &holdout_width_3),
        ("holdout w4", &holdout_width_4),
        ("holdout combiné w3+w4", &holdout_combined),
        ("OOD w5", &holdout_width_5),
        ("OOD extrême w6", &holdout_width_6),
        ("train w3", &training_width_3),
        ("train w4", &training_width_4),
    ];

    for &(label, records) in &populations
    {
        tdi5_print_population_geometry(label, records);
    }

    tdi5_print_models(&models, &target_scalers);

    let evaluation_populations: [(&str, &[Record]); 5] = [
        ("holdout w3", &holdout_width_3),
        ("holdout w4", &holdout_width_4),
        ("holdout combiné w3+w4", &holdout_combined),
        ("OOD w5", &holdout_width_5),
        ("OOD extrême w6", &holdout_width_6),
    ];

    for &(population_label, records) in &evaluation_populations
    {
        for horizon_index in 0..TARGET_HORIZON_COUNT
        {
            let evaluations =
                tdi5_evaluate_horizon(records, horizon_index, &models, &target_scalers)?;

            tdi5_print_evaluations(population_label, horizon_index, &evaluations);
        }
    }

    let primary_index = primary_horizon_index();
    let primary_scaler = target_scalers[primary_index];

    let combined_primary =
        tdi5_evaluate_horizon(&holdout_combined, primary_index, &models, &target_scalers)?;

    let width_3_primary =
        tdi5_evaluate_horizon(&holdout_width_3, primary_index, &models, &target_scalers)?;

    let width_4_primary =
        tdi5_evaluate_horizon(&holdout_width_4, primary_index, &models, &target_scalers)?;

    let width_5_primary =
        tdi5_evaluate_horizon(&holdout_width_5, primary_index, &models, &target_scalers)?;

    let width_6_primary =
        tdi5_evaluate_horizon(&holdout_width_6, primary_index, &models, &target_scalers)?;

    let combined_m0 = tdi5_layout_evaluation(&combined_primary, FeatureLayout::M0);

    let combined_m3 = tdi5_layout_evaluation(&combined_primary, FeatureLayout::M3);

    let width_3_m0 = tdi5_layout_evaluation(&width_3_primary, FeatureLayout::M0);

    let width_3_m3 = tdi5_layout_evaluation(&width_3_primary, FeatureLayout::M3);

    let width_4_m0 = tdi5_layout_evaluation(&width_4_primary, FeatureLayout::M0);

    let width_4_m3 = tdi5_layout_evaluation(&width_4_primary, FeatureLayout::M3);

    let width_5_m0 = tdi5_layout_evaluation(&width_5_primary, FeatureLayout::M0);

    let width_5_m3 = tdi5_layout_evaluation(&width_5_primary, FeatureLayout::M3);

    let width_6_m0 = tdi5_layout_evaluation(&width_6_primary, FeatureLayout::M0);

    let width_6_m3 = tdi5_layout_evaluation(&width_6_primary, FeatureLayout::M3);

    let combined_bootstrap = tdi5_paired_bootstrap(
        &holdout_combined,
        primary_index,
        primary_scaler,
        &combined_m0.predictions,
        &combined_m3.predictions,
    )?;

    let width_3_bootstrap = tdi5_paired_bootstrap(
        &holdout_width_3,
        primary_index,
        primary_scaler,
        &width_3_m0.predictions,
        &width_3_m3.predictions,
    )?;

    let width_4_bootstrap = tdi5_paired_bootstrap(
        &holdout_width_4,
        primary_index,
        primary_scaler,
        &width_4_m0.predictions,
        &width_4_m3.predictions,
    )?;

    let width_5_bootstrap = tdi5_paired_bootstrap(
        &holdout_width_5,
        primary_index,
        primary_scaler,
        &width_5_m0.predictions,
        &width_5_m3.predictions,
    )?;

    let width_6_bootstrap = tdi5_paired_bootstrap(
        &holdout_width_6,
        primary_index,
        primary_scaler,
        &width_6_m0.predictions,
        &width_6_m3.predictions,
    )?;

    println!();
    println!("=== INTERVALLES BOOTSTRAP U6 ===");

    for (label, intervals) in [
        ("holdout combiné w3+w4", combined_bootstrap),
        ("holdout w3", width_3_bootstrap),
        ("holdout w4", width_4_bootstrap),
        ("OOD principal w5", width_5_bootstrap),
        ("OOD extrême w6", width_6_bootstrap),
    ]
    {
        tdi5_print_bootstrap_intervals(label, intervals);
    }

    let criterion_a =
        tdi5_relative_reduction(combined_m0.standardized.mse, combined_m3.standardized.mse) >= 0.10
            && combined_bootstrap.standardized_mse.lower > 0.0
            && width_3_m0.standardized.mse - width_3_m3.standardized.mse > 0.0
            && width_4_m0.standardized.mse - width_4_m3.standardized.mse > 0.0
            && width_3_bootstrap.standardized_mse.lower > 0.0
            && width_4_bootstrap.standardized_mse.lower > 0.0
            && combined_m3.standardized.spearman > combined_m0.standardized.spearman
            && width_3_m3.standardized.spearman > 0.0
            && width_4_m3.standardized.spearman > 0.0
            && combined_m3.standardized.bias.abs() <= combined_m0.standardized.bias.abs() + 0.02;

    let criterion_b =
        tdi5_relative_reduction(width_5_m0.standardized.mse, width_5_m3.standardized.mse) >= 0.20
            && width_5_bootstrap.standardized_mse.lower > 0.0
            && width_5_m3.standardized.spearman > 0.0
            && width_5_m3.standardized.spearman >= width_5_m0.standardized.spearman
            && width_5_m3.standardized.r_squared > width_5_m0.standardized.r_squared
            && width_5_m3.standardized.bias.abs() < width_5_m0.standardized.bias.abs()
            && width_5_m0.reconstructed.mse - width_5_m3.reconstructed.mse > 0.0
            && width_5_m0.reconstructed.mae - width_5_m3.reconstructed.mae > 0.0;

    let criterion_c = width_6_m0.standardized.mse - width_6_m3.standardized.mse > 0.0
        && width_6_bootstrap.standardized_mse.lower > 0.0
        && width_6_m3.standardized.spearman > 0.0
        && width_6_m3.standardized.spearman >= width_6_m0.standardized.spearman
        && width_6_m3.standardized.bias.abs() <= width_6_m0.standardized.bias.abs()
        && width_6_m0.reconstructed.mse - width_6_m3.reconstructed.mse > 0.0;

    let secondary_horizons = [0_usize, 1, 2, 4];
    let mut positive_count = 0_usize;
    let mut reductions = Vec::with_capacity(4);
    let mut u8_positive = false;

    println!();
    println!("=== TRAJECTOIRE SECONDAIRE ===");

    for horizon_index in secondary_horizons
    {
        let evaluations =
            tdi5_evaluate_horizon(&holdout_combined, horizon_index, &models, &target_scalers)?;

        let baseline = tdi5_layout_evaluation(&evaluations, FeatureLayout::M0);

        let challenger = tdi5_layout_evaluation(&evaluations, FeatureLayout::M3);

        let delta = baseline.standardized.mse - challenger.standardized.mse;

        let reduction =
            tdi5_relative_reduction(baseline.standardized.mse, challenger.standardized.mse);

        positive_count += usize::from(delta > 0.0);
        reductions.push(reduction);

        if TARGET_HORIZONS[horizon_index] == 8
        {
            u8_positive = delta > 0.0;
        }

        println!(
            "U_{} | Δ MSE={delta:.12} | réduction={:.9} %",
            TARGET_HORIZONS[horizon_index],
            reduction * 100.0,
        );
    }

    let average_reduction = reductions.iter().sum::<f64>() / reductions.len() as f64;

    let criterion_d = positive_count >= 3
        && u8_positive
        && reductions.iter().all(|reduction| *reduction >= -0.05)
        && average_reduction > 0.0;

    let (direct_baseline_model, direct_challenger_model) = tdi5_fit_direct_models(&training)?;

    println!();
    println!("=== MODÈLES DU COMPARATEUR DIRECT O6 ===");

    print_model("comparateur direct M0", &direct_baseline_model);

    print_model("comparateur direct M3", &direct_challenger_model);

    for &(label, records) in &evaluation_populations
    {
        tdi5_print_direct_comparator(
            label,
            records,
            &direct_baseline_model,
            &direct_challenger_model,
        );
    }

    println!();
    println!(
        "CRITÈRE PRINCIPAL TDI-5.1A : {}",
        if criterion_a { "RÉUSSI" } else { "ÉCHOUÉ" }
    );
    println!(
        "CRITÈRE TRANSFERT TDI-5.1B : {}",
        if criterion_b { "RÉUSSI" } else { "ÉCHOUÉ" }
    );
    println!(
        "CRITÈRE TRANSFERT EXTRÊME TDI-5.1C : {}",
        if criterion_c { "RÉUSSI" } else { "ÉCHOUÉ" }
    );
    println!(
        "CRITÈRE TRAJECTOIRE TDI-5.1D : {}",
        if criterion_d { "RÉUSSI" } else { "ÉCHOUÉ" }
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BASELINE_FEATURE_COUNT, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, ConfidenceInterval,
        DeterministicRng, FeatureLayout, HOLDOUT_WIDTH_3_SEED_OFFSET, HOLDOUT_WIDTH_3_SYSTEMS,
        HOLDOUT_WIDTH_4_SEED_OFFSET, HOLDOUT_WIDTH_4_SYSTEMS, Metrics, OOD_WIDTH_5_SEED_OFFSET,
        OOD_WIDTH_5_SYSTEMS, OOD_WIDTH_6_SEED_OFFSET, OOD_WIDTH_6_SYSTEMS, PRIMARY_HORIZON,
        RIDGE_LAMBDA, Record, TARGET_HORIZON_COUNT, TARGET_HORIZONS, TDI_FEATURE_COUNT,
        TRAIN_WIDTH_3, TRAIN_WIDTH_3_SEED_OFFSET, TRAIN_WIDTH_3_SYSTEMS, TRAIN_WIDTH_4_SEED_OFFSET,
        TRAIN_WIDTH_4_SYSTEMS, TargetScaler, average_ranks, calculate_metrics, confidence_interval,
        primary_horizon_index, splitmix64,
    };

    #[test]
    fn deterministic_rng_is_reproducible() {
        let mut first = DeterministicRng::new(BOOTSTRAP_SEED);
        let mut second = DeterministicRng::new(BOOTSTRAP_SEED);

        for _ in 0..100
        {
            assert_eq!(first.next_u64(), second.next_u64());
        }
    }

    #[test]
    fn splitmix_is_deterministic() {
        assert_eq!(splitmix64(42), splitmix64(42));
        assert_ne!(splitmix64(42), splitmix64(43));
    }

    #[test]
    fn exact_successor_cardinalities_are_represented_through_width_6() {
        let expected_state_counts = [1_u128, 2, 4, 8, 16, 32, 64];
        let expected_successor_spaces = [
            2_u128,
            4,
            16,
            256,
            65_536,
            4_294_967_296,
            18_446_744_073_709_551_616,
        ];

        for (width, (&state_count, &successor_space)) in expected_state_counts
            .iter()
            .zip(&expected_successor_spaces)
            .enumerate()
        {
            let width = u8::try_from(width).expect("test width fits u8");

            assert_eq!(
                super::state_count_cardinality(width),
                super::Cardinality::Exact(state_count)
            );
            assert_eq!(
                super::successor_set_space_cardinality(width),
                super::Cardinality::Exact(successor_space)
            );
        }
    }

    #[test]
    fn width_6_successor_space_is_exact_u128() {
        assert_eq!(
            super::successor_set_space_cardinality(6),
            super::Cardinality::Exact(18_446_744_073_709_551_616_u128)
        );

        let context = super::AttemptContext::new(6, 90_000_000, 0);

        assert_eq!(
            super::nonempty_successor_set_count(context).expect("width 6 non-empty masks fit u64"),
            u64::MAX
        );
    }

    #[test]
    fn unsupported_widths_return_typed_errors() {
        assert_eq!(
            super::successor_set_space_cardinality(7),
            super::Cardinality::TooLarge {
                width: 7,
                exponent: 128,
            }
        );
        assert_eq!(
            super::generation_successor_set_space_cardinality(7),
            super::Cardinality::Invalid {
                width: 7,
                reason: "width is unsupported by the u64 successor-mask evaluator",
            }
        );

        let context = super::AttemptContext::new(7, 0, 0);
        let error =
            super::nonempty_successor_set_count(context).expect_err("width 7 is unsupported");

        assert_eq!(error.context, context);
        assert_eq!(error.category, super::FailureCategory::UnsupportedWidth);
    }

    #[test]
    fn arithmetic_and_structural_errors_keep_attempt_context() {
        let context = super::AttemptContext::new(3, 123, 4);

        let arithmetic = super::normalized_entropy(f64::INFINITY, context)
            .expect_err("non-finite normalized entropy is arithmetic failure");

        assert_eq!(arithmetic.context, context);
        assert_eq!(arithmetic.category, super::FailureCategory::Arithmetic);

        let structural =
            super::build_system(context, &[]).expect_err("wrong mask count is structural failure");

        assert_eq!(structural.context, context);
        assert_eq!(structural.category, super::FailureCategory::Structural);
    }

    #[test]
    fn evaluator_errors_are_not_converted_to_rejections() {
        let limits = super::GenerationLimits {
            max_attempts: 4,
            no_progress_limit: 4,
        };

        let error = super::generate_records_with_analyzer(
            TRAIN_WIDTH_3,
            TRAIN_WIDTH_3_SEED_OFFSET,
            1,
            limits,
            |context| {
                Err(super::EvaluationError::new(
                    context,
                    super::FailureCategory::Arithmetic,
                    "forced arithmetic failure",
                ))
            },
        )
        .expect_err("evaluator failure must propagate");

        match error
        {
            super::GenerationError::Evaluation(error) =>
            {
                assert_eq!(error.context.width, TRAIN_WIDTH_3);
                assert_eq!(error.context.seed, TRAIN_WIDTH_3_SEED_OFFSET);
                assert_eq!(error.context.attempt_index, 0);
                assert_eq!(error.category, super::FailureCategory::Arithmetic);
            },
            other => panic!("unexpected generation error: {other:?}"),
        }
    }

    #[test]
    fn attempt_budget_exhaustion_is_deterministic() {
        let limits = super::GenerationLimits {
            max_attempts: 2,
            no_progress_limit: 10,
        };

        let error = super::generate_records_with_analyzer(
            TRAIN_WIDTH_3,
            TRAIN_WIDTH_3_SEED_OFFSET,
            1,
            limits,
            |_context| {
                Ok(super::CandidateOutcome::Rejected(
                    super::RejectionReason::InvalidObservationGeometry,
                ))
            },
        )
        .expect_err("budget must be exhausted deterministically");

        match error
        {
            super::GenerationError::AttemptBudgetExhausted(diagnostic) =>
            {
                assert_eq!(diagnostic.context.width, TRAIN_WIDTH_3);
                assert_eq!(diagnostic.context.seed, TRAIN_WIDTH_3_SEED_OFFSET + 2);
                assert_eq!(diagnostic.context.attempt_index, 2);
                assert_eq!(diagnostic.accepted, 0);
                assert_eq!(diagnostic.excluded, 2);
                assert_eq!(diagnostic.limits, limits);
            },
            other => panic!("unexpected generation error: {other:?}"),
        }
    }

    #[test]
    fn no_progress_termination_is_deterministic() {
        let limits = super::GenerationLimits {
            max_attempts: 10,
            no_progress_limit: 3,
        };

        let error = super::generate_records_with_analyzer(
            TRAIN_WIDTH_3,
            TRAIN_WIDTH_3_SEED_OFFSET,
            1,
            limits,
            |_context| {
                Ok(super::CandidateOutcome::Rejected(
                    super::RejectionReason::InvalidObservationGeometry,
                ))
            },
        )
        .expect_err("no-progress threshold must terminate deterministically");

        match error
        {
            super::GenerationError::NoProgress(diagnostic) =>
            {
                assert_eq!(diagnostic.context.width, TRAIN_WIDTH_3);
                assert_eq!(diagnostic.context.seed, TRAIN_WIDTH_3_SEED_OFFSET + 2);
                assert_eq!(diagnostic.context.attempt_index, 2);
                assert_eq!(diagnostic.accepted, 0);
                assert_eq!(diagnostic.excluded, 3);
                assert_eq!(diagnostic.limits, limits);
            },
            other => panic!("unexpected generation error: {other:?}"),
        }
    }

    #[test]
    fn small_record_generation_succeeds_with_bounded_limits() {
        let limits = super::GenerationLimits {
            max_attempts: 256,
            no_progress_limit: 256,
        };

        let report = super::generate_records_with_limits(
            TRAIN_WIDTH_3,
            TRAIN_WIDTH_3_SEED_OFFSET,
            1,
            limits,
        )
        .expect("width 3 should produce a small accepted record under bounded limits");

        assert_eq!(report.records.len(), 1);
        assert!(report.attempts <= limits.max_attempts);
        assert_eq!(
            report.next_seed,
            TRAIN_WIDTH_3_SEED_OFFSET + u64::try_from(report.attempts).expect("attempts fit u64")
        );
    }

    // Repository-layout-bound integrity check: it shells out to `sha256sum`
    // over the *TDI* project's own preregistration artifacts (its CI workflow,
    // `docs/TDI-5-*`, `scripts/reproduce-tdi5.sh`, sibling `tdi-bench/...`
    // paths). Those files are not part of the SciRust workspace, so the check
    // cannot resolve them after re-homing; ignored here rather than deleted so
    // the frozen preregistration record stays visible. The in-code frozen
    // constants (layout / horizons / features / constants / population) are
    // still checked by the tests below.
    #[test]
    #[ignore = "hashes TDI-repo preregistration files absent from the SciRust workspace"]
    fn frozen_tdi5_protocol_hashes_are_unchanged() {
        let expected_hashes = [
            (
                ".github/workflows/tdi5-ci.yml",
                "90b1d45625c8a13bc5dd14d6e98107a6ff9d85cca912b2e883d366a6ad9eed2c",
            ),
            (
                "docs/TDI-5-CONTINUOUS-DEFICIT-GEOMETRY-EVALUATOR.sha256",
                "708dbeab3ad3c509702d1b0f9eb749eb7f11e2be2424d15908560cafda09829a",
            ),
            (
                "docs/TDI-5-CONTINUOUS-DEFICIT-GEOMETRY-PREREGISTRATION.md",
                "8481d730ae5c47506284f67ce7d75586abb1412f617fbd88029106d7c26986ef",
            ),
            (
                "docs/TDI-5-CONTINUOUS-DEFICIT-GEOMETRY-PREREGISTRATION.sha256",
                "a674b1cbe9faa19e97bfd790e086b21bd3b4ac904394cce945b2796de6321b9a",
            ),
            (
                "docs/TDI-5-SCIENTIFIC-CODE.sha256",
                "a644945cb283af2b168d6f783b5caf9d7694887d386ace2b8f549eb1963c98d0",
            ),
            (
                "scripts/reproduce-tdi5.sh",
                "d06cde6a604f7cba9754c848881a999e680968c56afbf995a52f64ca3eb09a8d",
            ),
            (
                "tdi-bench/src/bin/tdi-continuous-deficit-geometry.rs",
                "3bfd370944fd48b1c9ef0bd106de51143bbd37c3419fe254d12baee34619532a",
            ),
        ];

        let repository_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("tdi-bench has workspace parent");

        for (relative_path, expected_hash) in expected_hashes
        {
            let path = repository_root.join(relative_path);
            let output = std::process::Command::new("sha256sum")
                .arg(&path)
                .output()
                .expect("sha256sum is available");

            assert!(
                output.status.success(),
                "sha256sum failed for {relative_path}: {}",
                String::from_utf8_lossy(&output.stderr)
            );

            let stdout = String::from_utf8_lossy(&output.stdout);
            let actual_hash = stdout
                .split_whitespace()
                .next()
                .expect("sha256sum prints hash");

            assert_eq!(actual_hash, expected_hash, "{relative_path} changed");
        }
    }

    #[test]
    fn exact_deficit_geometry_is_correct() {
        let ratio = scirust_tdi::ExactRatio::new(7, 8).expect("valid ratio");

        let transformed =
            super::exact_overlap_deficit_u(&ratio).expect("valid conditional geometry");

        assert!((transformed - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn biguint_logarithm_supports_more_than_128_bits() {
        let digits = [0_u64, 0_u64, 1_u64];

        let logarithm =
            super::biguint_log2_from_u64_digits(&digits).expect("large integer logarithm");

        assert!((logarithm - 128.0).abs() < 1.0e-12);
    }

    #[test]
    fn target_scaler_round_trips() {
        let records = [
            Record {
                baseline: [0.0; BASELINE_FEATURE_COUNT],
                tdi: [0.0; TDI_FEATURE_COUNT],
                overlaps: [0.5; TARGET_HORIZON_COUNT],
                targets_u: [1.0; TARGET_HORIZON_COUNT],
            },
            Record {
                baseline: [0.0; BASELINE_FEATURE_COUNT],
                tdi: [0.0; TDI_FEATURE_COUNT],
                overlaps: [0.75; TARGET_HORIZON_COUNT],
                targets_u: [2.0; TARGET_HORIZON_COUNT],
            },
        ];

        let scaler = TargetScaler::fit(&records, primary_horizon_index()).expect("valid scaler");
        let value = 1.75;

        assert!((scaler.unstandardize(scaler.standardize(value)) - value).abs() < 1.0e-12);
    }

    #[test]
    fn reconstruction_respects_unit_interval() {
        assert_eq!(super::tdi5_reconstruct_overlap(-1000.0), (0.0, true));

        assert_eq!(super::tdi5_reconstruct_overlap(0.0), (0.0, false));

        let (reconstructed, clipped) = super::tdi5_reconstruct_overlap(3.0);

        assert!(!clipped);
        assert!((0.0..=1.0).contains(&reconstructed));
        assert!((reconstructed - 0.875).abs() < 1.0e-12);
    }
    #[test]
    fn identity_metrics_are_exact() {
        let values = [0.1, 0.3, 0.6, 0.9];

        assert_eq!(
            calculate_metrics(&values, &values),
            Metrics {
                mse: 0.0,
                mae: 0.0,
                r_squared: 1.0,
                spearman: 1.0,
                bias: 0.0,
                observed_mean: 0.475,
                predicted_mean: 0.475,
                calibration_intercept: 0.0,
                calibration_slope: 1.0,
                zero_fraction: 0.0,
                one_fraction: 0.0,
            }
        );
    }

    #[test]
    fn ranks_handle_ties() {
        assert_eq!(
            average_ranks(&[3.0, 1.0, 1.0, 2.0]),
            vec![4.0, 1.5, 1.5, 3.0]
        );
    }

    #[test]
    fn confidence_interval_is_ordered() {
        let interval = confidence_interval(vec![3.0, 1.0, 4.0, 2.0]);

        assert!(interval.lower <= interval.median);
        assert!(interval.median <= interval.upper);

        let _ = ConfidenceInterval {
            lower: interval.lower,
            median: interval.median,
            upper: interval.upper,
        };
    }

    #[test]
    fn preregistered_layout_is_fixed() {
        assert_eq!(BASELINE_FEATURE_COUNT, 13);
        assert_eq!(TDI_FEATURE_COUNT, 3);
        assert_eq!(RIDGE_LAMBDA, 1.0);
        assert_eq!(BOOTSTRAP_SEED, 0x5444_4935_4344_4745);
    }

    #[test]
    fn tdi5_target_horizons_are_frozen() {
        assert_eq!(TARGET_HORIZONS, [3, 4, 5, 6, 8]);
    }

    #[test]
    fn tdi5_primary_horizon_is_six() {
        assert_eq!(PRIMARY_HORIZON, 6);
        assert_eq!(primary_horizon_index(), 3);
        assert_eq!(TARGET_HORIZONS[primary_horizon_index()], PRIMARY_HORIZON);
    }

    #[test]
    fn tdi5_feature_layouts_are_frozen() {
        assert_eq!(FeatureLayout::M0.feature_count(), 13);
        assert_eq!(FeatureLayout::M1.feature_count(), 14);
        assert_eq!(FeatureLayout::M2.feature_count(), 15);
        assert_eq!(FeatureLayout::M3.feature_count(), 16);
    }

    #[test]
    fn tdi5_constants_are_frozen() {
        assert_eq!(RIDGE_LAMBDA, 1.0);
        assert_eq!(BOOTSTRAP_REPLICATES, 2_000);
        assert_eq!(BOOTSTRAP_SEED, 0x5444_4935_4344_4745);
    }

    #[test]
    fn tdi5_population_contract_is_frozen() {
        assert_eq!(TRAIN_WIDTH_3_SYSTEMS, 15_000);
        assert_eq!(TRAIN_WIDTH_4_SYSTEMS, 15_000);
        assert_eq!(HOLDOUT_WIDTH_3_SYSTEMS, 5_000);
        assert_eq!(HOLDOUT_WIDTH_4_SYSTEMS, 5_000);
        assert_eq!(OOD_WIDTH_5_SYSTEMS, 10_000);
        assert_eq!(OOD_WIDTH_6_SYSTEMS, 5_000);

        assert_eq!(TRAIN_WIDTH_3_SEED_OFFSET, 60_000_000);
        assert_eq!(HOLDOUT_WIDTH_3_SEED_OFFSET, 61_000_000);
        assert_eq!(TRAIN_WIDTH_4_SEED_OFFSET, 70_000_000);
        assert_eq!(HOLDOUT_WIDTH_4_SEED_OFFSET, 71_000_000);
        assert_eq!(OOD_WIDTH_5_SEED_OFFSET, 80_000_000);
        assert_eq!(OOD_WIDTH_6_SEED_OFFSET, 90_000_000);
    }

    #[test]
    fn target_scaler_uses_unit_scale_for_constant_targets() {
        let record = Record {
            baseline: [0.0; BASELINE_FEATURE_COUNT],
            tdi: [0.0; TDI_FEATURE_COUNT],
            overlaps: [0.5; TARGET_HORIZON_COUNT],
            targets_u: [2.0; TARGET_HORIZON_COUNT],
        };

        let records = [record.clone(), record];

        let scaler = TargetScaler::fit(&records, primary_horizon_index())
            .expect("constant target must remain valid");

        assert_eq!(scaler.mean, 2.0);
        assert_eq!(scaler.scale, 1.0);
    }
}
