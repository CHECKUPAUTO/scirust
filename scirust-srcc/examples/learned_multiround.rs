use scirust_srcc::{
    SrccConfig, SrccTransportSample, basis_vector, fit_srcc_projector, squared_norm,
};

fn main() {
    let e1 = basis_vector(1).unwrap();
    let e2 = basis_vector(2).unwrap();
    let e3 = basis_vector(3).unwrap();
    let e8 = basis_vector(8).unwrap();

    let minus_e2 = e2.map(|value| -value);
    let minus_e3 = e3.map(|value| -value);

    /*
     * Deterministic interleaving:
     *
     * view 0:
     *   e1 ->  e2
     *   e2 ->  e3
     *
     * view 1:
     *   e1 -> -e2
     *   e2 -> -e3
     *
     * SRCC uses sign-invariant directional consensus.
     */
    let samples = [
        SrccTransportSample::new(e1, e2),
        SrccTransportSample::new(e1, minus_e2),
        SrccTransportSample::new(e2, e3),
        SrccTransportSample::new(e2, minus_e3),
    ];

    let result = fit_srcc_projector(&[e1], &samples, 2, SrccConfig::default()).unwrap();

    let closure = result.projector.closure();

    println!("dimension={}", closure.dimension());
    println!("rounds={}", closure.rounds());
    println!("accepted_per_round={:?}", closure.accepted_per_round(),);

    for certificate in closure.certificates()
    {
        println!(
            "round={},basis_index={},support={},transports={:?},minimum_alignment={}",
            certificate.round,
            certificate.basis_index,
            certificate.support,
            certificate.transport_indices,
            certificate.minimum_alignment,
        );
    }

    let residual_e1 = squared_norm(&result.projector.apply(&e1));

    let residual_e2 = squared_norm(&result.projector.apply(&e2));

    let residual_e3 = squared_norm(&result.projector.apply(&e3));

    let preserved_e8 = result.projector.apply(&e8);

    println!("residual_e1={residual_e1}");
    println!("residual_e2={residual_e2}");
    println!("residual_e3={residual_e3}");
    println!(
        "preservation_error_e8={}",
        preserved_e8
            .iter()
            .zip(e8)
            .map(|(left, right)| {
                let difference = left - right;
                difference * difference
            })
            .sum::<f64>(),
    );

    assert_eq!(closure.dimension(), 3);
    assert_eq!(closure.rounds(), 2);
    assert_eq!(closure.accepted_per_round(), &[1, 1]);
    assert_eq!(closure.certificates().len(), 2);

    assert!(residual_e1 < 1.0e-24);
    assert!(residual_e2 < 1.0e-24);
    assert!(residual_e3 < 1.0e-24);
    assert_eq!(preserved_e8, e8);
}
