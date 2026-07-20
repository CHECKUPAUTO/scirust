use scirust_cayley_filter::{
    MultiplierCase, SEDENION_DIMENSION, Sedenion, analyze_matrix, basis_vector,
    fit_clifford_noise_subspace, left_multiplication_matrix, rank_hard_zero_divisor_projectors,
    rank_zero_divisor_matched_nullity_four_clifford_projectors, score_cayley_projector,
    score_clifford_projector, squared_norm,
};

const TOLERANCE: f64 = 1.0e-12;
const WEIGHT: f64 = 10.0;

fn normalize(vector: Sedenion) -> Sedenion {
    let norm = squared_norm(&vector).sqrt();
    vector.map(|value| value / norm)
}

fn combine(basis: &[Sedenion], coefficients: [f64; 4]) -> Sedenion {
    core::array::from_fn(|index| (0..4).map(|k| coefficients[k] * basis[k][index]).sum())
}

fn cases(basis: &[Sedenion], rows: &[[f64; 4]]) -> Vec<MultiplierCase> {
    let signal = basis_vector(0).unwrap();
    rows.iter()
        .map(|row| MultiplierCase::new(signal, combine(basis, *row)))
        .collect()
}

fn main() {
    let mut multiplier = [0.0; SEDENION_DIMENSION];
    multiplier[1] = 1.0;
    multiplier[10] = 1.0;

    let matrix = left_multiplication_matrix(multiplier);
    let analysis = analyze_matrix(&matrix, TOLERANCE).unwrap();
    assert_eq!(analysis.nullity(), 4);

    let kernel: Vec<_> = analysis
        .kernel_basis()
        .iter()
        .copied()
        .map(normalize)
        .collect();

    let train = cases(
        &kernel,
        &[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    );

    let dev = cases(
        &kernel,
        &[
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, -1.0],
        ],
    );

    let test = cases(
        &kernel,
        &[
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [2.0, 1.0, -1.0, 0.5],
            [-0.5, 1.5, 0.25, -1.0],
        ],
    );

    let cayley = rank_hard_zero_divisor_projectors(&train, WEIGHT, TOLERANCE, TOLERANCE).unwrap();

    let clifford =
        rank_zero_divisor_matched_nullity_four_clifford_projectors(&train, WEIGHT, TOLERANCE)
            .unwrap();

    let c = &cayley[0];
    let k = &clifford[0];
    let learned = fit_clifford_noise_subspace(&train, 4, TOLERANCE).unwrap();

    let c_dev = score_cayley_projector(&dev, &c.projector, WEIGHT).unwrap();
    let c_test = score_cayley_projector(&test, &c.projector, WEIGHT).unwrap();
    let k_dev = score_clifford_projector(&dev, &k.projector, WEIGHT).unwrap();
    let k_test = score_clifford_projector(&test, &k.projector, WEIGHT).unwrap();
    let l_train = score_clifford_projector(&train, &learned, WEIGHT).unwrap();
    let l_dev = score_clifford_projector(&dev, &learned, WEIGHT).unwrap();
    let l_test = score_clifford_projector(&test, &learned, WEIGHT).unwrap();

    println!("family,i,j,sign,train_loss,dev_loss,test_loss,nullity");
    println!(
        "cayley,{},{},{},{},{},{},{}",
        c.first_index,
        c.second_index,
        c.second_sign,
        c.score.loss,
        c_dev.loss,
        c_test.loss,
        c.projector.rejected_dimension(),
    );
    println!(
        "clifford,{},{},{},{},{},{},{}",
        k.first_index,
        k.second_index,
        k.second_sign,
        k.score.loss,
        k_dev.loss,
        k_test.loss,
        k.projector.rejected_dimension(),
    );

    println!(
        "learned_clifford,-,-,-,{},{},{},{}",
        l_train.loss,
        l_dev.loss,
        l_test.loss,
        learned.rejected_dimension(),
    );

    assert_eq!(cayley.len(), 84);
    assert_eq!(clifford.len(), 84);
    assert!(c.score.loss < 1.0e-20);
    assert!(c_dev.loss < 1.0e-20);
    assert!(c_test.loss < 1.0e-20);

    assert!(k.score.loss > 0.5);
    assert!(k_dev.loss > 0.5);
    assert!(k_test.loss > 0.5);

    assert!(l_train.loss < 1.0e-20);
    assert!(l_dev.loss < 1.0e-20);
    assert!(l_test.loss < 1.0e-20);
}
