use scirust_cayley_filter::{
    IdentityFilter, MultiplierCase, NoiseDirectionProjector, SEDENION_DIMENSION, Sedenion,
    SoftCayleyFilter, optimize_multiplier, sedenion_mul, squared_norm,
};
use scirust_solvers::Tolerance;

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const RELATIVE_SCALE: f64 = 5.0e-2;
const DISTORTION_WEIGHT: f64 = 10.0;
const ENERGY_FLOOR: f64 = 1.0e-30;
const FAMILY_COUNT: usize = 3;

#[derive(Clone, Copy, Debug)]
struct Family {
    multiplier: Sedenion,
    noise_direction: Sedenion,
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
                            let noise_direction =
                                two_term_vector(right_first, right_second, right_sign);

                            let product = sedenion_mul(multiplier, noise_direction);

                            if !product.iter().all(|value| *value == 0.0)
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
                                noise_direction,
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

fn make_cases(family: &Family, family_index: usize) -> Result<Vec<MultiplierCase>, String> {
    let oracle =
        NoiseDirectionProjector::new(family.noise_direction).map_err(|error| error.to_string())?;

    Ok((0..8)
        .map(|sample| {
            let raw = deterministic_vector(family_index, sample);
            let signal = oracle.apply(&raw);
            let amplitude = 0.5 + 0.15 * sample as f64;

            let noise = core::array::from_fn(|index| amplitude * family.noise_direction[index]);

            MultiplierCase::new(signal, noise)
        })
        .collect())
}

fn local_initial(family: &Family, family_index: usize) -> Sedenion {
    let mut initial = family.multiplier;

    for (coordinate, value) in initial.iter_mut().enumerate()
    {
        let perturbation = (((family_index + 2) * (coordinate + 3) * 17) % 13) as f64 - 6.0;

        *value += 0.005 * perturbation;
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

fn print_score(family: usize, method: &str, score: Score) {
    println!(
        "{family},{method},{},{},{}",
        score.loss, score.noise_ratio, score.distortion_ratio,
    );
}

fn main() -> Result<(), String> {
    let families = discover_families();

    if families.len() != FAMILY_COUNT
    {
        return Err(format!(
            "expected {FAMILY_COUNT} distinct families, found {}",
            families.len(),
        ));
    }

    println!("family,method,held_out_loss,noise_ratio,distortion_ratio");

    for (family_index, family) in families.iter().enumerate()
    {
        let cases = make_cases(family, family_index)?;
        let (train, held_out) = cases.split_at(4);

        let initial = local_initial(family, family_index);

        let optimized = optimize_multiplier(
            train,
            initial,
            RELATIVE_SCALE,
            DISTORTION_WEIGHT,
            0.05,
            Tolerance::new(1.0e-5, 1.0e-3, 8_000),
        )?;

        let learned = SoftCayleyFilter::new(optimized.multiplier, RELATIVE_SCALE)
            .map_err(|error| error.to_string())?;

        let oracle = NoiseDirectionProjector::new(family.noise_direction)
            .map_err(|error| error.to_string())?;

        let identity_score = evaluate(held_out, |input| IdentityFilter.apply(input));

        let oracle_score = evaluate(held_out, |input| oracle.apply(input));

        let learned_score = evaluate(held_out, |input| learned.apply(input));

        print_score(family_index, "identity", identity_score);
        print_score(family_index, "orthogonal_oracle", oracle_score);
        print_score(family_index, "cayley_learned_local", learned_score);

        println!(
            "# family={family_index} left=e{}{}e{} right=e{}{}e{} iterations={} residual={}",
            family.left_indices.0,
            if family.left_sign > 0 { "+" } else { "-" },
            family.left_indices.1,
            family.right_indices.0,
            if family.right_sign > 0 { "+" } else { "-" },
            family.right_indices.1,
            optimized.iterations,
            optimized.residual,
        );

        if learned_score.loss >= identity_score.loss
        {
            return Err(format!(
                "family {family_index}: learned Cayley filter did not beat identity"
            ));
        }
    }

    Ok(())
}
