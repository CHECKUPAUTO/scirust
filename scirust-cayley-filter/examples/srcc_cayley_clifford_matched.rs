use scirust_cayley_filter::{
    CayleyProjector, NoiseSubspaceProjector, SEDENION_DIMENSION, Sedenion, SplitCliffordProjector,
    analyze_matrix, left_multiplication_matrix, sedenion_mul, squared_norm,
};
use scirust_srcc::{LinearMap16, SrccConfig, SrccProjector};

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const ANALYSIS_TOLERANCE: f64 = 1.0e-12;
const DISTORTION_WEIGHT: f64 = 10.0;
const ENERGY_FLOOR: f64 = 1.0e-30;
const FAMILY_COUNT: usize = 3;
const TEST_CASES: usize = 24;

#[derive(Clone, Copy, Debug)]
struct Family {
    multiplier: Sedenion,
    left_indices: (usize, usize),
    left_sign: i8,
    right_indices: (usize, usize),
    right_sign: i8,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    signal: Sedenion,
    noise: Sedenion,
}

#[derive(Clone, Copy, Debug)]
struct Score {
    loss: f64,
    residual_noise_ratio: f64,
    signal_distortion_ratio: f64,
}

fn two_term_vector(first: usize, second: usize, sign: i8) -> Sedenion {
    let mut value = ZERO;
    value[first] = 1.0;
    value[second] = f64::from(sign);
    value
}

fn discover_families() -> Vec<Family> {
    let mut families = Vec::new();

    for left_first in 1..SEDENION_DIMENSION
    {
        for left_second in (left_first + 1)..SEDENION_DIMENSION
        {
            for left_sign in [-1_i8, 1_i8]
            {
                let multiplier = two_term_vector(left_first, left_second, left_sign);

                for right_first in 1..SEDENION_DIMENSION
                {
                    for right_second in (right_first + 1)..SEDENION_DIMENSION
                    {
                        for right_sign in [-1_i8, 1_i8]
                        {
                            let witness = two_term_vector(right_first, right_second, right_sign);

                            if sedenion_mul(multiplier, witness)
                                .iter()
                                .any(|value| *value != 0.0)
                            {
                                continue;
                            }

                            let repeated_left = families.iter().any(|family: &Family| {
                                family.left_indices == (left_first, left_second)
                            });

                            let repeated_right = families.iter().any(|family: &Family| {
                                family.right_indices == (right_first, right_second)
                            });

                            if repeated_left || repeated_right
                            {
                                continue;
                            }

                            families.push(Family {
                                multiplier,
                                left_indices: (left_first, left_second),
                                left_sign,
                                right_indices: (right_first, right_second),
                                right_sign,
                            });

                            if families.len() == FAMILY_COUNT
                            {
                                return families;
                            }
                        }
                    }
                }
            }
        }
    }

    families
}

fn deterministic_vector(family: usize, sample: usize) -> Sedenion {
    core::array::from_fn(|coordinate| {
        let integer = ((family + 3) * 47
            + (sample + 7) * 37
            + (coordinate + 11) * 19
            + (family + 1) * (sample + 3) * (coordinate + 5) * 7)
            % 67;

        (integer as f64 - 33.0) / 10.0
    })
}

fn mixed_noise(basis: &[Sedenion], family: usize, sample: usize) -> Sedenion {
    let mut noise = ZERO;

    for (direction_index, direction) in basis.iter().enumerate()
    {
        let integer = ((family + 2) * 23
            + (sample + 3) * 17
            + (direction_index + 5) * 13
            + (family + 1) * (sample + 2) * (direction_index + 3) * 5)
            % 17;

        let centered = integer as f64 - 8.0;
        let coefficient = if centered == 0.0
        {
            0.25
        }
        else
        {
            centered / 5.0
        };

        for coordinate in 0..SEDENION_DIMENSION
        {
            noise[coordinate] += coefficient * direction[coordinate];
        }
    }

    noise
}

fn make_cases(oracle: &NoiseSubspaceProjector, family: usize, count: usize) -> Vec<Case> {
    (0..count)
        .map(|sample| {
            let raw = deterministic_vector(family, sample);

            Case {
                signal: oracle.apply(&raw),
                noise: mixed_noise(oracle.orthonormal_basis(), family, sample),
            }
        })
        .collect()
}

fn rank_one_transport(source: &Sedenion, target: &Sedenion, sign: f64) -> LinearMap16 {
    core::array::from_fn(|row| core::array::from_fn(|column| sign * target[row] * source[column]))
}

fn build_matched_srcc(basis: &[Sedenion]) -> Result<SrccProjector, String> {
    if basis.len() < 2
    {
        return Err("matched SRCC requires at least two basis directions".into());
    }

    let seed = basis[0];
    let mut transports = Vec::with_capacity(2 * (basis.len() - 1));

    for target in &basis[1..]
    {
        transports.push(rank_one_transport(&seed, target, 1.0));
        transports.push(rank_one_transport(&seed, target, -1.0));
    }

    let config = SrccConfig {
        novelty_threshold: 1.0e-10,
        resonance_threshold: 1.0 - 1.0e-12,
        minimum_support: 2,
        maximum_dimension: basis.len(),
        maximum_rounds: basis.len(),
        energy_floor: ENERGY_FLOOR,
    };

    SrccProjector::build(&[seed], &transports, config).map_err(|error| error.to_string())
}

fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (a, b)| {
        let difference = a - b;
        sum + difference * difference
    })
}

fn evaluate(cases: &[Case], mut apply: impl FnMut(&Sedenion) -> Sedenion) -> Score {
    let mut residual_noise_ratio = 0.0;
    let mut signal_distortion_ratio = 0.0;

    for case in cases
    {
        let filtered_signal = apply(&case.signal);
        let filtered_noise = apply(&case.noise);

        residual_noise_ratio +=
            squared_norm(&filtered_noise) / squared_norm(&case.noise).max(ENERGY_FLOOR);

        signal_distortion_ratio += squared_distance(&case.signal, &filtered_signal)
            / squared_norm(&case.signal).max(ENERGY_FLOOR);
    }

    let inverse_count = 1.0 / cases.len() as f64;
    let residual_noise_ratio = residual_noise_ratio * inverse_count;
    let signal_distortion_ratio = signal_distortion_ratio * inverse_count;

    Score {
        loss: residual_noise_ratio + DISTORTION_WEIGHT * signal_distortion_ratio,
        residual_noise_ratio,
        signal_distortion_ratio,
    }
}

fn maximum_gap(
    family: usize,
    mut left: impl FnMut(&Sedenion) -> Sedenion,
    mut right: impl FnMut(&Sedenion) -> Sedenion,
) -> f64 {
    let mut maximum = 0.0_f64;

    for index in 0..SEDENION_DIMENSION
    {
        let mut basis = ZERO;
        basis[index] = 1.0;

        maximum = maximum.max(squared_distance(&left(&basis), &right(&basis)));
    }

    for sample in 0..32
    {
        let probe = deterministic_vector(family, sample + 100);

        maximum = maximum.max(squared_distance(&left(&probe), &right(&probe)));
    }

    maximum.sqrt()
}

fn print_score(family: usize, dimension: usize, method: &str, score: Score) {
    println!(
        "{family},{dimension},{method},{:.17e},{:.17e},{:.17e}",
        score.loss, score.residual_noise_ratio, score.signal_distortion_ratio,
    );
}

fn sign_symbol(sign: i8) -> &'static str {
    if sign > 0 { "+" } else { "-" }
}

fn main() -> Result<(), String> {
    let families = discover_families();

    if families.len() != FAMILY_COUNT
    {
        return Err(format!(
            "expected {FAMILY_COUNT} independent families, found {}",
            families.len(),
        ));
    }

    println!(
        "family,rejected_dimension,method,test_loss,residual_noise_ratio,signal_distortion_ratio"
    );

    for (family_index, family) in families.iter().enumerate()
    {
        let matrix = left_multiplication_matrix(family.multiplier);

        let analysis =
            analyze_matrix(&matrix, ANALYSIS_TOLERANCE).map_err(|error| error.to_string())?;

        let oracle = NoiseSubspaceProjector::new(analysis.kernel_basis(), ANALYSIS_TOLERANCE)
            .map_err(|error| error.to_string())?;

        let dimension = oracle.dimension();

        if dimension != 4
        {
            return Err(format!(
                "family {family_index}: expected nullity 4, found {dimension}",
            ));
        }

        let cayley = CayleyProjector::new(family.multiplier, ANALYSIS_TOLERANCE)
            .map_err(|error| error.to_string())?;

        let clifford = SplitCliffordProjector::from_orthonormal_basis(oracle.orthonormal_basis())
            .map_err(|error| format!("invalid Clifford basis: {error:?}"))?;

        let srcc = build_matched_srcc(oracle.orthonormal_basis())?;

        if cayley.rejected_dimension() != dimension
            || clifford.rejected_dimension() != dimension
            || srcc.rejected_dimension() != dimension
        {
            return Err(format!(
                "family {family_index}: rejected dimensions are not matched",
            ));
        }

        let cases = make_cases(&oracle, family_index, TEST_CASES);

        let identity_score = evaluate(&cases, |input| *input);
        let oracle_score = evaluate(&cases, |input| oracle.apply(input));
        let cayley_score = evaluate(&cases, |input| cayley.apply(input));
        let clifford_score = evaluate(&cases, |input| clifford.apply(input));
        let srcc_score = evaluate(&cases, |input| srcc.apply(input));

        print_score(family_index, dimension, "identity", identity_score);
        print_score(
            family_index,
            dimension,
            "real_subspace_oracle",
            oracle_score,
        );
        print_score(family_index, dimension, "matched_cayley", cayley_score);
        print_score(family_index, dimension, "matched_clifford", clifford_score);
        print_score(family_index, dimension, "matched_srcc", srcc_score);

        let cayley_gap = maximum_gap(
            family_index,
            |input| oracle.apply(input),
            |input| cayley.apply(input),
        );

        let clifford_gap = maximum_gap(
            family_index,
            |input| oracle.apply(input),
            |input| clifford.apply(input),
        );

        let srcc_gap = maximum_gap(
            family_index,
            |input| oracle.apply(input),
            |input| srcc.apply(input),
        );

        println!(
            "# family={family_index} generator=e{}{}e{} witness=e{}{}e{} rounds={} accepted_per_round={:?} certificates={} cayley_gap={:.17e} clifford_gap={:.17e} srcc_gap={:.17e}",
            family.left_indices.0,
            sign_symbol(family.left_sign),
            family.left_indices.1,
            family.right_indices.0,
            sign_symbol(family.right_sign),
            family.right_indices.1,
            srcc.closure().rounds(),
            srcc.closure().accepted_per_round(),
            srcc.closure().certificates().len(),
            cayley_gap,
            clifford_gap,
            srcc_gap,
        );

        if identity_score.loss < 0.999_999_999_999
        {
            return Err(format!("family {family_index}: invalid identity baseline",));
        }

        if oracle_score.loss > 1.0e-20
            || cayley_score.loss > 1.0e-20
            || clifford_score.loss > 1.0e-20
            || srcc_score.loss > 1.0e-20
        {
            return Err(format!(
                "family {family_index}: a matched projector failed the exact oracle",
            ));
        }

        if cayley_gap > 1.0e-10 || clifford_gap > 1.0e-10 || srcc_gap > 1.0e-10
        {
            return Err(format!(
                "family {family_index}: projector equivalence tolerance exceeded",
            ));
        }
    }

    Ok(())
}
