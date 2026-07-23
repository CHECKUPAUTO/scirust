use scirust_causal::{CausalError, GraphExtractionConfig, extract_causal_dag};
use scirust_solvers::Matrix;

// ─── Empty-edge graph ───────────────────────────────────────────────────────

#[test]
fn empty_edge_graph() {
    let config = GraphExtractionConfig::new(0.1).unwrap();
    let interactions = Matrix::zeros(3, 3);
    let dag = extract_causal_dag(&interactions, &config).unwrap();
    assert_eq!(dag.n_nodes(), 3);
    assert_eq!(dag.n_edges(), 0);
}

// ─── One edge with orientation oracle ───────────────────────────────────────

#[test]
fn one_edge_oracle() {
    // A[1,0] = 0.5 means j=0 influences i=1, i.e., 0 -> 1
    let config = GraphExtractionConfig::new(0.1).unwrap();
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.5, 0.0]);
    let dag = extract_causal_dag(&interactions, &config).unwrap();
    assert_eq!(dag.n_nodes(), 2);
    assert_eq!(dag.n_edges(), 1);
    assert_eq!(dag.children(0), &[1_usize]);
    assert_eq!(dag.parents(1), &[0_usize]);
}

// ─── Chain ──────────────────────────────────────────────────────────────────

#[test]
fn chain_dag() {
    // 0 -> 1 -> 2
    // A[1,0] = 0.5, A[2,1] = 0.3
    let config = GraphExtractionConfig::new(0.1).unwrap();
    let interactions =
        Matrix::from_row_major(3, 3, vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0]);
    let dag = extract_causal_dag(&interactions, &config).unwrap();
    assert_eq!(dag.n_edges(), 2);
    assert_eq!(dag.children(0), &[1_usize]);
    assert_eq!(dag.children(1), &[2_usize]);
}

// ─── Fork ───────────────────────────────────────────────────────────────────

#[test]
fn fork_dag() {
    // 1 -> 0, 2 -> 0
    let config = GraphExtractionConfig::new(0.1).unwrap();
    let interactions =
        Matrix::from_row_major(3, 3, vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0]);
    let dag = extract_causal_dag(&interactions, &config).unwrap();
    assert_eq!(dag.n_edges(), 2);
}

// ─── Collider ───────────────────────────────────────────────────────────────

#[test]
fn collider_dag() {
    // 0 -> 2, 1 -> 2
    let config = GraphExtractionConfig::new(0.1).unwrap();
    let interactions =
        Matrix::from_row_major(3, 3, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.0]);
    let dag = extract_causal_dag(&interactions, &config).unwrap();
    assert_eq!(dag.n_edges(), 2);
    assert_eq!(dag.parents(2).len(), 2);
    assert!(dag.parents(2).contains(&0));
    assert!(dag.parents(2).contains(&1));
}

// ─── Threshold boundary ─────────────────────────────────────────────────────

#[test]
fn threshold_boundary() {
    let config_exact = GraphExtractionConfig::new(0.5).unwrap();
    let config_above = GraphExtractionConfig::new(0.49).unwrap();
    let config_below = GraphExtractionConfig::new(0.51).unwrap();

    let interactions = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.5, 0.0]);

    let dag_exact = extract_causal_dag(&interactions, &config_exact).unwrap();
    assert_eq!(dag_exact.n_edges(), 0);

    let dag_above = extract_causal_dag(&interactions, &config_above).unwrap();
    assert_eq!(dag_above.n_edges(), 1);

    let dag_below = extract_causal_dag(&interactions, &config_below).unwrap();
    assert_eq!(dag_below.n_edges(), 0);
}

// ─── Cycle rejection ────────────────────────────────────────────────────────

#[test]
fn rejects_cycle() {
    // 0 -> 1, 1 -> 0 creates a cycle
    let config = GraphExtractionConfig::new(0.1).unwrap();
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, 0.5, 0.5, 0.0]);

    let result = extract_causal_dag(&interactions, &config);
    assert!(matches!(result, Err(CausalError::CyclicGraph)));
}

// ─── Config validation ──────────────────────────────────────────────────────

#[test]
fn rejects_non_finite_threshold() {
    assert!(GraphExtractionConfig::new(f64::NAN).is_err());
    assert!(GraphExtractionConfig::new(f64::INFINITY).is_err());
}

#[test]
fn rejects_negative_threshold() {
    assert!(GraphExtractionConfig::new(-0.1).is_err());
}

// ─── Non-square rejection ───────────────────────────────────────────────────

#[test]
fn rejects_non_square() {
    let config = GraphExtractionConfig::new(0.1).unwrap();
    let interactions = Matrix::zeros(2, 3);
    assert!(matches!(
        extract_causal_dag(&interactions, &config),
        Err(CausalError::NotSquare { .. })
    ));
}

// ─── Non-finite coefficient rejection ───────────────────────────────────────

#[test]
fn rejects_non_finite_coefficient() {
    let config = GraphExtractionConfig::new(0.1).unwrap();
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, f64::NAN, 0.5, 0.0]);
    assert!(matches!(
        extract_causal_dag(&interactions, &config),
        Err(CausalError::NonFiniteWeight { .. })
    ));
}

// ─── Deterministic node and edge order ──────────────────────────────────────

#[test]
fn deterministic_order() {
    let config = GraphExtractionConfig::new(0.1).unwrap();
    let interactions =
        Matrix::from_row_major(3, 3, vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0]);
    let dag1 = extract_causal_dag(&interactions, &config).unwrap();
    let dag2 = extract_causal_dag(&interactions, &config).unwrap();

    assert_eq!(dag1.n_edges(), dag2.n_edges());
    for i in 0..2
    {
        assert_eq!(dag1.children(i), dag2.children(i));
    }
}
