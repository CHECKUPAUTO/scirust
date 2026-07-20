use scirust_cayley_filter::{
    CayleyProjector, IdentityFilter, MultiplierCase, NoiseSubspaceProjector, SEDENION_DIMENSION,
    Sedenion, SoftCayleyFilter, analyze_matrix, left_multiplication_matrix, sedenion_mul,
    select_multiplier_train_dev, squared_norm,
};
use scirust_solvers::Tolerance;

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const ANALYSIS_TOLERANCE: f64 = 1.0e-12;
const RELATIVE_SCALE: f64 = 5.0e-2;
const DISTORTION_WEIGHT: f64 = 10.0;
const ENERGY_FLOOR: f64 = 1.0e-30;
const FAMILY_COUNT: usize = 3;
const TOP_K: usize = 8;

#[derive(Clone, Copy, Debug)]
struct Family {
    multiplier: Sedenion,
    witness: Sedenion,
    left_indices: (usize, usize),
    left_sign: i8,
    right_indices: (usize, usize),
    right_sign: i8,
}

#[derive(Clone, Copy, Debug)]
struct Score {
    loss: f64,
    noise_ratio: f64,
    distortion_ratio: f64,
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
                                witness,
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

fn deterministic_vector(family: usize, split: usize, sample: usize) -> Sedenion {
    core::array::from_fn(|coordinate| {
        let integer = ((family + 3) * 47
            + (split + 5) * 43
            + (sample + 7) * 37
            + (coordinate + 11) * 19
            + (family + 1) * (split + 2) * (sample + 3) * (coordinate + 5) * 7)
            % 67;

        (integer as f64 - 33.0) / 10.0
    })
}

fn mixed_noise(basis: &[Sedenion], split: usize, sample: usize) -> Sedenion {
    if split == 0 && sample < basis.len()
    {
        let amplitude = 0.75 + 0.2 * sample as f64;

        return core::array::from_fn(|coordinate| amplitude * basis[sample][coordinate]);
    }

    let mut noise = ZERO;

    for (direction_index, direction) in basis.iter().enumerate()
    {
        let integer = ((split + 2) * 23
            + (sample + 3) * 17
            + (direction_index + 5) * 13
            + (split + 1) * (sample + 2) * (direction_index + 3) * 5)
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

fn make_split(
    oracle: &NoiseSubspaceProjector,
    family: usize,
    split: usize,
    count: usize,
) -> Vec<MultiplierCase> {
    (0..count)
        .map(|sample| {
            let raw = deterministic_vector(family, split, sample);

            let signal = oracle.apply(&raw);
            let noise = mixed_noise(oracle.orthonormal_basis(), split, sample);

            MultiplierCase::new(signal, noise)
        })
        .collect()
}

fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (a, b)| {
        let difference = a - b;
        sum + difference * difference
    })
}

fn evaluate(cases: &[MultiplierCase], mut apply: impl FnMut(&Sedenion) -> Sedenion) -> Score {
    let mut noise_ratio = 0.0;
    let mut distortion_ratio = 0.0;

    for case in cases
    {
        let filtered_signal = apply(&case.signal);
        let filtered_noise = apply(&case.noise);

        noise_ratio += squared_norm(&filtered_noise) / squared_norm(&case.noise).max(ENERGY_FLOOR);

        distortion_ratio += squared_distance(&case.signal, &filtered_signal)
            / squared_norm(&case.signal).max(ENERGY_FLOOR);
    }

    let count = cases.len() as f64;
    let noise_ratio = noise_ratio / count;
    let distortion_ratio = distortion_ratio / count;

    Score {
        loss: noise_ratio + DISTORTION_WEIGHT * distortion_ratio,
        noise_ratio,
        distortion_ratio,
    }
}

fn print_score(family: usize, dimension: usize, method: &str, score: Score) {
    println!(
        "{family},{dimension},{method},{},{},{}",
        score.loss, score.noise_ratio, score.distortion_ratio,
    );
}

fn main() -> Result<(), String> {
    let families = discover_families();

    if families.len() != FAMILY_COUNT
    {
        return Err(format!(
            "expected {FAMILY_COUNT} families, found {}",
            families.len(),
        ));
    }

    println!("family,kernel_dimension,method,test_loss,noise_ratio,distortion_ratio");

    for (family_index, family) in families.iter().enumerate()
    {
        let matrix = left_multiplication_matrix(family.multiplier);

        let analysis =
            analyze_matrix(&matrix, ANALYSIS_TOLERANCE).map_err(|error| error.to_string())?;

        let oracle = NoiseSubspaceProjector::new(analysis.kernel_basis(), ANALYSIS_TOLERANCE)
            .map_err(|error| error.to_string())?;

        if oracle
            .apply(&family.witness)
            .iter()
            .any(|value| value.abs() > 1.0e-10)
        {
            return Err(format!("family {family_index}: invalid kernel witness"));
        }

        let train = make_split(&oracle, family_index, 0, oracle.dimension() * 2);

        let dev = make_split(&oracle, family_index, 1, 6);

        let test = make_split(&oracle, family_index, 2, 8);

        let selection = select_multiplier_train_dev(
            &train,
            &dev,
            TOP_K,
            RELATIVE_SCALE,
            DISTORTION_WEIGHT,
            0.05,
            Tolerance::new(1.0e-5, 1.0e-3, 4_000),
        )?;

        let selected = &selection.selected;

        let soft = SoftCayleyFilter::new(selected.multiplier, RELATIVE_SCALE)
            .map_err(|error| error.to_string())?;

        let hard = CayleyProjector::new(selected.multiplier, RELATIVE_SCALE)
            .map_err(|error| error.to_string())?;

        let identity_score = evaluate(&test, |input| IdentityFilter.apply(input));

        let oracle_score = evaluate(&test, |input| oracle.apply(input));

        let soft_score = evaluate(&test, |input| soft.apply(input));

        let hard_score = evaluate(&test, |input| hard.apply(input));

        print_score(family_index, oracle.dimension(), "identity", identity_score);

        print_score(
            family_index,
            oracle.dimension(),
            "real_subspace_oracle",
            oracle_score,
        );

        print_score(
            family_index,
            oracle.dimension(),
            "global_cayley_soft",
            soft_score,
        );

        print_score(
            family_index,
            oracle.dimension(),
            "global_cayley_hard",
            hard_score,
        );

        println!(
            "# family={family_index} generator=e{}{}e{} witness=e{}{}e{} selected_seed=e{}{}e{} seed_rank={} refinement_used={} train_loss={} dev_loss={} candidates={}",
            family.left_indices.0,
            if family.left_sign > 0 { "+" } else { "-" },
            family.left_indices.1,
            family.right_indices.0,
            if family.right_sign > 0 { "+" } else { "-" },
            family.right_indices.1,
            selected.seed_first_index,
            if selected.seed_second_sign > 0
            {
                "+"
            }
            else
            {
                "-"
            },
            selected.seed_second_index,
            selected.seed_rank,
            selected.refinement_used,
            selected.train_score.loss,
            selected.dev_score.loss,
            selection.candidates.len(),
        );

        if hard_score.loss >= identity_score.loss
        {
            return Err(format!(
                "family {family_index}: global Cayley selection did not beat identity on test"
            ));
        }
    }

    Ok(())
}
