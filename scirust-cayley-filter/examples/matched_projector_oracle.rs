use scirust_cayley_filter::{
    MultiplierCase, SEDENION_DIMENSION, Sedenion, basis_vector, rank_hard_zero_divisor_projectors,
    rank_zero_divisor_matched_nullity_four_clifford_projectors, score_cayley_projector,
    score_clifford_projector,
};

const TOLERANCE: f64 = 1.0e-12;
const DISTORTION_WEIGHT: f64 = 10.0;

fn case(scale: f64) -> MultiplierCase {
    let signal = basis_vector(0).unwrap();
    let mut noise: Sedenion = [0.0; SEDENION_DIMENSION];
    noise[4] = scale;
    noise[15] = -scale;
    MultiplierCase::new(signal, noise)
}

fn main() {
    let train = [case(0.5), case(1.0), case(2.0)];
    let dev = [case(0.75), case(1.5)];
    let test = [case(1.25), case(2.5)];

    let cayley =
        rank_hard_zero_divisor_projectors(&train, DISTORTION_WEIGHT, TOLERANCE, TOLERANCE).unwrap();

    let clifford = rank_zero_divisor_matched_nullity_four_clifford_projectors(
        &train,
        DISTORTION_WEIGHT,
        TOLERANCE,
    )
    .unwrap();

    let c = &cayley[0];
    let k = &clifford[0];

    let c_dev = score_cayley_projector(&dev, &c.projector, DISTORTION_WEIGHT).unwrap();
    let c_test = score_cayley_projector(&test, &c.projector, DISTORTION_WEIGHT).unwrap();
    let k_dev = score_clifford_projector(&dev, &k.projector, DISTORTION_WEIGHT).unwrap();
    let k_test = score_clifford_projector(&test, &k.projector, DISTORTION_WEIGHT).unwrap();

    println!("family,candidates,i,j,sign,train_loss,dev_loss,test_loss,nullity");
    println!(
        "cayley,{},{},{},{},{},{},{},{}",
        cayley.len(),
        c.first_index,
        c.second_index,
        c.second_sign,
        c.score.loss,
        c_dev.loss,
        c_test.loss,
        c.projector.rejected_dimension(),
    );
    println!(
        "clifford,{},{},{},{},{},{},{},{}",
        clifford.len(),
        k.first_index,
        k.second_index,
        k.second_sign,
        k.score.loss,
        k_dev.loss,
        k_test.loss,
        k.projector.rejected_dimension(),
    );

    assert_eq!(cayley.len(), 84);
    assert_eq!(clifford.len(), 84);
    assert!(c.score.loss < 1.0e-20);
}
