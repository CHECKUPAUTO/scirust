use scirust_cayley_filter::{
    CayleyProjector, MultiplierCase, SEDENION_DIMENSION, Sedenion, optimize_multiplier,
    score_multiplier,
};
use scirust_solvers::Tolerance;

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const RELATIVE_SCALE: f64 = 5.0e-2;
const DISTORTION_WEIGHT: f64 = 10.0;

fn deterministic_vector(sample: usize) -> Sedenion {
    core::array::from_fn(|coordinate| {
        let value =
            ((sample + 1) * 37 + (coordinate + 3) * 19 + (sample + 5) * (coordinate + 1) * 7) % 41;

        (value as f64 - 20.0) / 10.0
    })
}

fn synthetic_case(oracle: &CayleyProjector, sample: usize) -> MultiplierCase {
    let observation = deterministic_vector(sample);
    let signal = oracle.apply(&observation);

    let noise = core::array::from_fn(|index| observation[index] - signal[index]);

    MultiplierCase::new(signal, noise)
}

fn main() -> Result<(), String> {
    let mut oracle_multiplier = ZERO;
    oracle_multiplier[1] = 1.0;
    oracle_multiplier[10] = 1.0;

    let oracle =
        CayleyProjector::new(oracle_multiplier, 1.0e-12).map_err(|error| error.to_string())?;

    let cases: Vec<MultiplierCase> = (0..8)
        .map(|sample| synthetic_case(&oracle, sample))
        .collect();

    let (train, held_out) = cases.split_at(4);

    let mut initial = ZERO;
    initial[1] = 1.0;
    initial[2] = 0.05;
    initial[7] = -0.03;
    initial[10] = 0.80;

    let initial_train = score_multiplier(train, &initial, RELATIVE_SCALE, DISTORTION_WEIGHT)?;

    let initial_held_out = score_multiplier(held_out, &initial, RELATIVE_SCALE, DISTORTION_WEIGHT)?;

    let optimized = optimize_multiplier(
        train,
        initial,
        RELATIVE_SCALE,
        DISTORTION_WEIGHT,
        0.10,
        Tolerance::new(1.0e-10, 1.0e-8, 8_000),
    )?;

    let optimized_held_out = score_multiplier(
        held_out,
        &optimized.multiplier,
        RELATIVE_SCALE,
        DISTORTION_WEIGHT,
    )?;

    let oracle_held_out = score_multiplier(
        held_out,
        &oracle_multiplier,
        RELATIVE_SCALE,
        DISTORTION_WEIGHT,
    )?;

    println!("stage,dataset,loss,noise_ratio,distortion_ratio,effective_rejected_dimension");

    println!(
        "initial,train,{},{},{},{}",
        initial_train.loss,
        initial_train.mean_noise_ratio,
        initial_train.mean_distortion_ratio,
        initial_train.rejected_dimension,
    );

    println!(
        "optimized,train,{},{},{},{}",
        optimized.score.loss,
        optimized.score.mean_noise_ratio,
        optimized.score.mean_distortion_ratio,
        optimized.score.rejected_dimension,
    );

    println!(
        "initial,held_out,{},{},{},{}",
        initial_held_out.loss,
        initial_held_out.mean_noise_ratio,
        initial_held_out.mean_distortion_ratio,
        initial_held_out.rejected_dimension,
    );

    println!(
        "optimized,held_out,{},{},{},{}",
        optimized_held_out.loss,
        optimized_held_out.mean_noise_ratio,
        optimized_held_out.mean_distortion_ratio,
        optimized_held_out.rejected_dimension,
    );

    println!(
        "oracle,held_out,{},{},{},{}",
        oracle_held_out.loss,
        oracle_held_out.mean_noise_ratio,
        oracle_held_out.mean_distortion_ratio,
        oracle_held_out.rejected_dimension,
    );

    println!("iterations,{}", optimized.iterations);
    println!("residual,{}", optimized.residual);
    println!("multiplier,{:?}", optimized.multiplier);

    if optimized.score.loss >= initial_train.loss
    {
        return Err("training loss did not improve".into());
    }

    if optimized_held_out.loss >= initial_held_out.loss
    {
        return Err("held-out loss did not improve".into());
    }

    Ok(())
}
