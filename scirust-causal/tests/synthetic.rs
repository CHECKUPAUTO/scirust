use scirust_causal::{
    CausalError, SyntheticDataConfig, TriangularCubicFlow, generate_causal_samples,
    generate_noise_matrix,
};

// ─── Same seed gives same output ────────────────────────────────────────────

#[test]
fn same_seed_same_output() {
    let config = SyntheticDataConfig::new(42, 3, 5).unwrap();
    let m1 = generate_noise_matrix(&config);
    let m2 = generate_noise_matrix(&config);
    assert_eq!(m1.data(), m2.data());
}

// ─── Different seeds produce different values ───────────────────────────────

#[test]
fn different_seeds_diverge() {
    let c1 = SyntheticDataConfig::new(1, 3, 5).unwrap();
    let c2 = SyntheticDataConfig::new(2, 3, 5).unwrap();
    let m1 = generate_noise_matrix(&c1);
    let m2 = generate_noise_matrix(&c2);
    assert_ne!(m1.data(), m2.data());
}

// ─── Generated samples pass through forward flow ────────────────────────────

#[test]
fn forward_recovers_noise() {
    let flow = TriangularCubicFlow::from_row_major(2, vec![0.0, 0.0, 0.5, 0.0]).unwrap();
    let config = SyntheticDataConfig::new(42, 2, 3).unwrap();
    let noise = generate_noise_matrix(&config);

    let samples = generate_causal_samples(&flow, &config).unwrap();

    for row in 0..3
    {
        let sample_row: Vec<f64> = (0..2).map(|c| samples[(row, c)]).collect();
        let recovered = flow.forward(&sample_row).unwrap();
        let noise_row: Vec<f64> = (0..2).map(|c| noise[(row, c)]).collect();

        for (r, n) in recovered.iter().zip(&noise_row)
        {
            let diff = (r - n).abs();
            assert!(diff < 1.0e-10, "recovered {r} != noise {n}, diff={diff}");
        }
    }
}

// ─── Config validation ──────────────────────────────────────────────────────

#[test]
fn rejects_zero_dimension() {
    assert!(matches!(
        SyntheticDataConfig::new(42, 0, 5),
        Err(CausalError::ZeroDimension)
    ));
}

#[test]
fn rejects_zero_sample_count() {
    assert!(matches!(
        SyntheticDataConfig::new(42, 3, 0),
        Err(CausalError::ZeroSamples)
    ));
}

// ─── Finiteness of generated data ───────────────────────────────────────────

#[test]
fn generated_noise_is_finite() {
    let config = SyntheticDataConfig::new(7, 4, 10).unwrap();
    let noise = generate_noise_matrix(&config);
    for &v in noise.data()
    {
        assert!(v.is_finite());
    }
}

#[test]
fn generated_samples_are_finite() {
    let flow = TriangularCubicFlow::from_row_major(2, vec![0.0, 0.0, 0.5, 0.0]).unwrap();
    let config = SyntheticDataConfig::new(42, 2, 5).unwrap();
    let samples = generate_causal_samples(&flow, &config).unwrap();
    for &v in samples.data()
    {
        assert!(v.is_finite());
    }
}
