use scirust_neuro_symbolic::core::Reasoner;
use scirust_neuro_symbolic::logic::DatalogEngine;
use scirust_neuro_symbolic::neural::DifferentiableLogicLayer;
use scirust_core::autodiff::reverse::Tensor;

#[test]
fn test_hybrid_workflow() {
    // 1. Symbolic part: Datalog
    let mut dl = DatalogEngine::new();
    dl.add_fact("is_bird", vec!["tweety"]);
    assert!(dl.query("is_bird", vec!["tweety"]));

    // 2. Neural part: Differentiable logic
    let layer = DifferentiableLogicLayer::new("TestLayer");
    let input = Tensor::from_vec(vec![0.8], 1, 1);
    let output = layer.fuzzy_and(&input, &input);
    assert!((output.data[0] - 0.64).abs() < 1e-6);

    assert_eq!(dl.name(), "DatalogEngine");
    assert_eq!(layer.name(), "TestLayer");
}
