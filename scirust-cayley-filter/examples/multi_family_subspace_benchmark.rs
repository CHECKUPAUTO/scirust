use scirust_cayley_filter::{
    CayleyProjector, IdentityFilter, MultiplierCase, NoiseSubspaceProjector, SEDENION_DIMENSION,
    Sedenion, SoftCayleyFilter, analyze_matrix, left_multiplication_matrix, optimize_multiplier,
    sedenion_mul, squared_norm,
};
use scirust_solvers::Tolerance;

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const ANALYSIS_TOLERANCE: f64 = 1.0e-12;
const RELATIVE_SCALE: f64 = 5.0e-2;
const DISTORTION_WEIGHT: f64 = 10.0;
const ENERGY_FLOOR: f64 = 1.0e-30;
const FAMILY_COUNT: usize = 3;

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

fn deterministic_vector(family_index: usize, sample: usize) -> Sedenion {
    core::array::from_fn(|coordinate| {
        let integer = ((family_index + 3) * 43
            + (sample + 5) * 37
            + (coordinate + 7) * 19
            + (family_index + 1) * (sample + 2) * (coordinate + 3) * 11)
            % 61;

        (integer as f64 - 30.0) / 10.0
    })
}

fn training_noise(basis: &[Sedenion], sample: usize) -> Sedenion {
    let amplitude = 0.75 + 0.15 * sample as f64;

    core::array::from_fn(|coordinate| amplitude * basis[sample][coordinate])
}

fn held_out_noise(basis: &[Sedenion], sample: usize) -> Sedenion {
    let mut noise = ZERO;

    for (direction_index, direction) in basis.iter().enumerate()
    {
        let integer = ((sample + 2) * 17
            + (direction_index + 3) * 13
            + (sample + 1) * (direction_index + 2) * 7)
            % 11;

        let centered = integer as f64 - 5.0;
        let coefficient = if centered == 0.0
        {
            0.25
        }
        else
        {
            centered / 4.0
        };

        for coordinate in 0..SEDENION_DIMENSION
        {
            noise[coordinate] += coefficient * direction[coordinate];
        }
    }

    noise
}

fn make_cases(oracle: &NoiseSubspaceProjector, family_index: usize) -> Vec<MultiplierCase> {
    let basis = oracle.orthonormal_basis();
    let train_count = basis.len();
    let held_out_count = basis.len().max(4);

    let mut cases = Vec::with_capacity(train_count + held_out_count);

    for sample in 0..train_count
    {
        let raw = deterministic_vector(family_index, sample);
        let signal = oracle.apply(&raw);
        let noise = training_noise(basis, sample);

        cases.push(MultiplierCase::new(signal, noise));
    }

    for sample in 0..held_out_count
    {
        let raw = deterministic_vector(family_index, train_count + sample);

        let signal = oracle.apply(&raw);
        let noise = held_out_noise(basis, sample);

        cases.push(MultiplierCase::new(signal, noise));
    }

    cases
}

fn local_initial(family: &Family, family_index: usize) -> Sedenion {
    let mut initial = family.multiplier;

    for (coordinate, value) in initial.iter_mut().enumerate()
    {
        let integer = ((family_index + 2) * (coordinate + 3) * 17) % 13;

        *value += 0.005 * (integer as f64 - 6.0);
    }

    initial
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

    println!("family,kernel_dimension,method,held_out_loss,noise_ratio,distortion_ratio");

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
            return Err(format!(
                "family {family_index}: witness is not in oracle subspace"
            ));
        }

        let cases = make_cases(&oracle, family_index);
        let train_count = oracle.dimension();
        let (train, held_out) = cases.split_at(train_count);

        let initial = local_initial(family, family_index);

        let optimized = optimize_multiplier(
            train,
            initial,
            RELATIVE_SCALE,
            DISTORTION_WEIGHT,
            0.05,
            Tolerance::new(1.0e-5, 1.0e-3, 8_000),
        )?;

        let learned_soft = SoftCayleyFilter::new(optimized.multiplier, RELATIVE_SCALE)
            .map_err(|error| error.to_string())?;

        let learned_hard = CayleyProjector::new(optimized.multiplier, RELATIVE_SCALE)
            .map_err(|error| error.to_string())?;

        let identity_score = evaluate(held_out, |input| IdentityFilter.apply(input));

        let oracle_score = evaluate(held_out, |input| oracle.apply(input));

        let soft_score = evaluate(held_out, |input| learned_soft.apply(input));

        let hard_score = evaluate(held_out, |input| learned_hard.apply(input));

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
            "cayley_soft_learned",
            soft_score,
        );

        print_score(
            family_index,
            oracle.dimension(),
            "cayley_hard_learned",
            hard_score,
        );

        println!(
            "# family={family_index} left=e{}{}e{} witness=e{}{}e{} iterations={} residual={}",
            family.left_indices.0,
            if family.left_sign > 0 { "+" } else { "-" },
            family.left_indices.1,
            family.right_indices.0,
            if family.right_sign > 0 { "+" } else { "-" },
            family.right_indices.1,
            optimized.iterations,
            optimized.residual,
        );

        if soft_score.loss >= identity_score.loss
        {
            return Err(format!(
                "family {family_index}: learned soft Cayley filter did not beat identity"
            ));
        }
    }

    Ok(())
}
