//! Deterministic comparison on the exact zero-divisor construction.

use scirust_cayley_filter::baseline::{IdentityFilter, NoiseDirectionProjector};
use scirust_cayley_filter::filter::{CayleyFilter, FilterEvaluation};
use scirust_cayley_filter::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector};

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];

fn main() {
    let signal = basis_vector(0).expect("e0 exists");

    let mut noise = ZERO;
    noise[4] = 1.0;
    noise[15] = -1.0;

    let mut multiplier = ZERO;
    multiplier[1] = 1.0;
    multiplier[10] = 1.0;

    let identity = IdentityFilter.evaluate(&signal, &noise);

    let cayley = CayleyFilter::new(multiplier).evaluate(&signal, &noise);

    let projection = NoiseDirectionProjector::new(noise)
        .expect("valid exact noise direction")
        .evaluate(&signal, &noise);

    println!(
        "filter,signal_distortion_energy,output_noise_energy,\
noise_attenuation_db,input_snr_db,output_snr_db"
    );

    print_row("identity", &identity);
    print_row("cayley", &cayley);
    print_row("orthogonal_projection", &projection);

    assert_eq!(cayley.metrics().output_noise_energy, 0.0);
    assert_eq!(projection.metrics().output_noise_energy, 0.0);

    assert_eq!(projection.metrics().signal_distortion_energy, 0.0);

    assert!(
        cayley.metrics().signal_distortion_energy > projection.metrics().signal_distortion_energy
    );
}

fn print_row(name: &str, evaluation: &FilterEvaluation) {
    let metrics = evaluation.metrics();

    println!(
        "{name},{},{},{},{},{}",
        metrics.signal_distortion_energy,
        metrics.output_noise_energy,
        display_metric(metrics.noise_attenuation_db),
        display_metric(metrics.input_snr_db),
        display_metric(metrics.output_snr_db),
    );
}

fn display_metric(value: Option<f64>) -> String {
    match value
    {
        Some(number) => number.to_string(),
        None => "undefined".to_owned(),
    }
}
