use scirust_causal::{
    CausalError, GraphExtractionConfig, OptimizerConfig, SyntheticDataConfig, TriangularCubicFlow,
    extract_causal_dag, generate_causal_samples, optimize_causal,
};
use scirust_solvers::Matrix;

#[test]
fn end_to_end_causal_discovery() -> Result<(), CausalError> {
    // 1. Define a tiny known DAG with fixed weights.
    // A = [[0, 0, 0],
    //      [0.5, 0, 0],
    //      [0, 0.3, 0]]
    // This is a chain: 0 -> 1 -> 2
    let true_weights = vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0];
    let dim = 3;
    let flow = TriangularCubicFlow::from_row_major(dim, true_weights).unwrap();

    // 2. Generate deterministic synthetic data.
    let config = SyntheticDataConfig::new(42, dim, 20).unwrap();
    let samples = generate_causal_samples(&flow, &config)?;

    // 3. Initialize interactions to zero.
    let initial = Matrix::zeros(dim, dim);

    // 4. Optimize.
    let opt_config = OptimizerConfig::new(50, 5, 1.0e-5, 1.0e-8).unwrap();
    let result = optimize_causal(&samples, &initial, 0.01, 1.0e-8, 0.0, 1.0, &opt_config)?;

    // 5. Verify all outputs are finite and coherent.
    assert!(result.objective.is_finite());
    assert!(result.gradient_norm.is_finite());
    assert!(result.acyclicity.is_finite());
    assert!(result.outer_iterations > 0);

    for i in 0..dim
    {
        for j in 0..dim
        {
            assert!(
                result.interactions[(i, j)].is_finite(),
                "non-finite coefficient at ({i}, {j})"
            );
        }
    }

    // 6. Extracted graph should be acyclic.
    let graph_config = GraphExtractionConfig::new(0.1).unwrap();
    let dag = extract_causal_dag(&result.interactions, &graph_config)?;

    // 7. The DAG should have a valid topological order.
    let topo = dag.topo_order();
    assert!(topo.is_ok(), "extracted graph is not a DAG");

    // 8. Check termination is coherent.
    match result.termination
    {
        scirust_causal::TerminationReason::Converged
        | scirust_causal::TerminationReason::MaxOuterIterations =>
        {},
        _ =>
        {
            panic!("unexpected termination: {:?}", result.termination);
        },
    }

    Ok(())
}
