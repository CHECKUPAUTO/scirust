use scirust_cayley_filter::{
    CayleyProjector, MultiplierCase, NoiseSubspaceProjector, SEDENION_DIMENSION, Sedenion,
    analyze_matrix, basis_vector, fit_clifford_noise_subspace, left_multiplication_matrix,
    rank_hard_zero_divisor_projectors, score_cayley_projector, score_clifford_projector,
    squared_norm,
};

const TOLERANCE: f64 = 1.0e-12;
const WEIGHT: f64 = 10.0;

fn normalize(vector: Sedenion) -> Sedenion {
    let norm = squared_norm(&vector).sqrt();
    vector.map(|value| value / norm)
}

fn main() {
    let mut multiplier = [0.0; SEDENION_DIMENSION];
    multiplier[1] = 1.0;
    multiplier[10] = 1.0;

    let matrix = left_multiplication_matrix(multiplier);
    let analysis = analyze_matrix(&matrix, TOLERANCE).unwrap();

    let subspace = NoiseSubspaceProjector::new(analysis.kernel_basis(), TOLERANCE).unwrap();
    let kernel = subspace.orthonormal_basis();

    assert_eq!(kernel.len(), 4);

    let truth = CayleyProjector::new(multiplier, TOLERANCE).unwrap();

    let off_kernel = (1..SEDENION_DIMENSION)
        .map(|index| basis_vector(index).unwrap())
        .map(|axis| truth.apply(&axis))
        .find(|vector| squared_norm(vector) > 1.0e-12)
        .map(normalize)
        .unwrap();

    for direction in kernel
    {
        let dot = direction
            .iter()
            .zip(off_kernel)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        assert!(dot.abs() < 1.0e-12);
    }

    let signal = basis_vector(0).unwrap();
    let test: Vec<_> = kernel
        .iter()
        .copied()
        .map(|noise| MultiplierCase::new(signal, noise))
        .collect();

    println!(
        "epsilon,cayley_i,cayley_j,cayley_sign,\
cayley_train,cayley_test,clifford_train,clifford_test"
    );

    for epsilon in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    {
        let observed = normalize(core::array::from_fn(|index| {
            kernel[0][index] + epsilon * off_kernel[index]
        }));

        let train = [MultiplierCase::new(signal, observed)];

        let cayley =
            rank_hard_zero_divisor_projectors(&train, WEIGHT, TOLERANCE, TOLERANCE).unwrap();

        let selected = &cayley[0];
        let cayley_test = score_cayley_projector(&test, &selected.projector, WEIGHT).unwrap();

        let clifford = fit_clifford_noise_subspace(&train, 4, TOLERANCE).unwrap();

        let clifford_train = score_clifford_projector(&train, &clifford, WEIGHT).unwrap();
        let clifford_test = score_clifford_projector(&test, &clifford, WEIGHT).unwrap();

        println!(
            "{epsilon},{},{},{},{},{},{},{}",
            selected.first_index,
            selected.second_index,
            selected.second_sign,
            selected.score.loss,
            cayley_test.loss,
            clifford_train.loss,
            clifford_test.loss,
        );

        assert!(clifford_train.loss < 1.0e-20);

        let expected = 1.0 - 1.0 / (4.0 * (1.0 + epsilon * epsilon));

        assert!((clifford_test.loss - expected).abs() < 1.0e-12);

        assert!(selected.score.loss.is_finite());
        assert!(cayley_test.loss.is_finite());
        assert!(cayley_test.loss <= 1.0 + 1.0e-12);
    }
}
