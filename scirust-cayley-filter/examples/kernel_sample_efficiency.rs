use scirust_cayley_filter::{
    MultiplierCase, SEDENION_DIMENSION, Sedenion, analyze_matrix, basis_vector,
    fit_clifford_noise_subspace, left_multiplication_matrix, rank_hard_zero_divisor_projectors,
    score_cayley_projector, score_clifford_projector, squared_norm,
};

const TOLERANCE: f64 = 1.0e-12;
const WEIGHT: f64 = 10.0;

fn normalize(vector: Sedenion) -> Sedenion {
    let norm = squared_norm(&vector).sqrt();
    vector.map(|value| value / norm)
}

fn cases(basis: &[Sedenion]) -> Vec<MultiplierCase> {
    let signal = basis_vector(0).unwrap();

    basis
        .iter()
        .copied()
        .map(|noise| MultiplierCase::new(signal, noise))
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

    let test = cases(&kernel);

    println!(
        "observed,cayley_i,cayley_j,cayley_sign,\
cayley_train,cayley_test,clifford_dim,\
clifford_train,clifford_test"
    );

    for observed in 1..=4
    {
        let train = cases(&kernel[..observed]);

        let cayley =
            rank_hard_zero_divisor_projectors(&train, WEIGHT, TOLERANCE, TOLERANCE).unwrap();

        let selected = &cayley[0];
        let cayley_test = score_cayley_projector(&test, &selected.projector, WEIGHT).unwrap();

        let clifford = fit_clifford_noise_subspace(&train, 4, TOLERANCE).unwrap();

        let clifford_train = score_clifford_projector(&train, &clifford, WEIGHT).unwrap();
        let clifford_test = score_clifford_projector(&test, &clifford, WEIGHT).unwrap();

        println!(
            "{observed},{},{},{},{},{},{},{},{}",
            selected.first_index,
            selected.second_index,
            selected.second_sign,
            selected.score.loss,
            cayley_test.loss,
            clifford.rejected_dimension(),
            clifford_train.loss,
            clifford_test.loss,
        );

        let expected_clifford_test = (4 - observed) as f64 / 4.0;

        assert_eq!(
            (
                selected.first_index,
                selected.second_index,
                selected.second_sign,
            ),
            (1, 10, 1),
        );
        assert!(selected.score.loss < 1.0e-20);
        assert!(cayley_test.loss < 1.0e-20);

        assert_eq!(clifford.rejected_dimension(), observed);
        assert!(clifford_train.loss < 1.0e-20);
        assert!((clifford_test.loss - expected_clifford_test).abs() < 1.0e-12);
    }
}
