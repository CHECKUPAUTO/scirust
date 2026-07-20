use scirust_cayley_filter::{
    MultiplierCase, SEDENION_DIMENSION, Sedenion, basis_vector, optimize_multiplier,
    score_multiplier,
};
use scirust_solvers::Tolerance;

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const RELATIVE_SCALE: f64 = 5.0e-2;
const DISTORTION_WEIGHT: f64 = 10.0;

fn main() -> Result<(), String> {
    let signal = basis_vector(0).expect("e0 exists");

    let mut noise = ZERO;
    noise[4] = 1.0;
    noise[15] = -1.0;

    let cases = [MultiplierCase::new(signal, noise)];

    // Perturbation déterministe autour de e1 + e10.
    let mut initial = ZERO;
    initial[1] = 1.0;
    initial[2] = 0.05;
    initial[7] = -0.03;
    initial[10] = 0.80;

    let initial_score = score_multiplier(&cases, &initial, RELATIVE_SCALE, DISTORTION_WEIGHT)?;

    let optimized = optimize_multiplier(
        &cases,
        initial,
        RELATIVE_SCALE,
        DISTORTION_WEIGHT,
        0.10,
        Tolerance::new(1.0e-10, 1.0e-8, 5_000),
    )?;

    println!("stage,loss,noise_ratio,distortion_ratio,effective_rejected_dimension");
    println!(
        "initial,{},{},{},{}",
        initial_score.loss,
        initial_score.mean_noise_ratio,
        initial_score.mean_distortion_ratio,
        initial_score.rejected_dimension,
    );
    println!(
        "optimized,{},{},{},{}",
        optimized.score.loss,
        optimized.score.mean_noise_ratio,
        optimized.score.mean_distortion_ratio,
        optimized.score.rejected_dimension,
    );

    println!("iterations,{}", optimized.iterations);
    println!("residual,{}", optimized.residual);
    println!("multiplier,{:?}", optimized.multiplier);

    if optimized.score.loss >= initial_score.loss
    {
        return Err(format!(
            "optimization did not improve the loss: {} -> {}",
            initial_score.loss, optimized.score.loss,
        ));
    }

    Ok(())
}
