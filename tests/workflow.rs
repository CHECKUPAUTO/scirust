use scirust_core::matrix::backend::best_backend;
use scirust_learning::{linear_regression, polynomial_fit};

#[test]
fn test_full_workflow() {
    // 1. Data Prep
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0]; // y = 2x + 1

    // 2. Linear Regression (Learning)
    let (slope, intercept) = linear_regression(&x, &y);
    assert!((slope - 2.0).abs() < 1e-10);
    assert!((intercept - 1.0).abs() < 1e-10);

    // 3. Polynomial Fit (Learning)
    let coeffs = polynomial_fit(&x, &y, 1);
    assert!((coeffs[0] - 1.0).abs() < 1e-10);
    assert!((coeffs[1] - 2.0).abs() < 1e-10);

    // 4. Matrix operations (Core)
    let backend = best_backend();
    let a = vec![
        vec![2.0, 1.0],
        vec![1.0, 2.0],
    ];
    let mut l = a.clone();
    backend.cholesky_f64(&mut l).unwrap();

    // L * L^T should be A
    // L = [[sqrt(2), 0], [1/sqrt(2), sqrt(1.5)]]
    let s2 = 2.0f64.sqrt();
    assert!((l[0][0] - s2).abs() < 1e-10);
    assert_eq!(l[0][1], 0.0);
}
