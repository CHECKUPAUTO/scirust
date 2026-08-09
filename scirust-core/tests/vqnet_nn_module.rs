use scirust_core::autodiff::optim::{Optimizer, Sgd};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::init::Zeros;
use scirust_core::nn::linear::Linear;
use scirust_core::nn::module::Module;
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::sequential::Sequential;
use scirust_core::vqnet::{
    ParameterInitializer, QuantumModule, RotationAxis, VariationalCircuit, VariationalCircuitBuilder,
};
use std::collections::HashMap;

fn one_qubit_circuit() -> VariationalCircuit {
    let mut builder = VariationalCircuitBuilder::new(1).unwrap();
    builder.angle_encoding(RotationAxis::Y, &[0]).unwrap();
    builder.hardware_efficient_ansatz(1).unwrap();
    builder.measure_all_z().unwrap();
    builder.build().unwrap()
}

fn deterministic_hybrid_model() -> Sequential {
    let mut rng = PcgEngine::new(7);
    let mut encoder = Linear::new(2, 1, &Zeros, &Zeros, &mut rng);
    encoder.weight = Tensor::from_vec(vec![1.0, 0.5], 2, 1);
    encoder.bias = Tensor::from_vec(vec![0.1], 1, 1);

    let quantum = QuantumModule::new(
        one_qubit_circuit(),
        ParameterInitializer::Constant(0.3),
    )
    .unwrap();

    let mut readout = Linear::new(1, 1, &Zeros, &Zeros, &mut rng);
    readout.weight = Tensor::from_vec(vec![1.0], 1, 1);
    readout.bias = Tensor::from_vec(vec![0.0], 1, 1);

    Sequential::new().add(encoder).add(quantum).add(readout)
}

#[test]
fn quantum_module_composes_inside_classical_sequential() {
    let mut model = deterministic_hybrid_model();
    let tape = Tape::new();
    let input = tape.input(Tensor::from_vec(vec![0.2, -0.4], 1, 2));
    let output = model.forward(&tape, input);

    assert_eq!(output.shape(), (1, 1));
    assert_eq!(model.parameter_indices().len(), 5);

    output.sum().backward();
    let quantum_gradient = tape.grad(model.parameter_indices()[2]);
    assert_eq!(quantum_gradient.shape(), (1, 2));
    assert!(quantum_gradient.data[0].abs() > 1.0e-5);
    assert!(quantum_gradient.data[1].abs() <= 2.0e-6);

    let input_gradient = tape.grad(input.idx());
    assert_eq!(input_gradient.shape(), (1, 2));
    assert!(input_gradient.data.iter().any(|value| value.abs() > 1.0e-5));
}

#[test]
fn standard_optimizer_and_sequential_sync_update_quantum_state() {
    let mut model = deterministic_hybrid_model();
    let tape = Tape::new();
    let input = tape.input(Tensor::from_vec(vec![0.2, -0.4], 1, 2));
    let output = model.forward(&tape, input);
    output.sum().backward();

    let before = model.state_dict()["1.parameters"].data.clone();
    let mut optimizer = Sgd::new(0.05);
    optimizer.step(&model.parameter_indices(), &tape);
    model.sync(&tape);
    let after = model.state_dict()["1.parameters"].data.clone();

    assert_ne!(after[0], before[0]);
    assert!((after[1] - before[1]).abs() <= 2.0e-6);
}

#[test]
fn quantum_module_state_dict_round_trips() {
    let source = QuantumModule::new(
        one_qubit_circuit(),
        ParameterInitializer::Constant(0.37),
    )
    .unwrap();
    let state = source.state_dict();
    assert_eq!(state["parameters"].shape(), (1, 2));

    let mut target = QuantumModule::new(one_qubit_circuit(), ParameterInitializer::Zeros).unwrap();
    target.load_state_dict(&state).unwrap();

    assert_eq!(source.parameters(), target.parameters());
    assert!(target.parameter_indices().is_empty());

    let source_tape = Tape::new();
    let source_input = source_tape.input(Tensor::from_vec(vec![0.1], 1, 1));
    let source_output = source.forward(&source_tape, source_input).unwrap().output();

    let target_tape = Tape::new();
    let target_input = target_tape.input(Tensor::from_vec(vec![0.1], 1, 1));
    let target_output = target.forward(&target_tape, target_input).unwrap().output();

    assert_eq!(
        source_tape.value(source_output.idx()).data,
        target_tape.value(target_output.idx()).data
    );
}

#[test]
fn quantum_module_state_dict_rejects_wrong_shape() {
    let mut module = QuantumModule::new(one_qubit_circuit(), ParameterInitializer::Zeros).unwrap();
    let wrong = HashMap::from([(
        "parameters".to_string(),
        Tensor::from_vec(vec![0.1], 1, 1),
    )]);

    assert!(module.load_state_dict(&wrong).is_err());
    assert_eq!(module.parameters().values(), &[0.0, 0.0]);
}

#[test]
fn module_try_forward_surfaces_quantum_shape_error() {
    let mut module = QuantumModule::new(one_qubit_circuit(), ParameterInitializer::Zeros).unwrap();
    let tape = Tape::new();
    let wrong_features = tape.input(Tensor::from_vec(vec![0.1, 0.2], 1, 2));

    let error = Module::try_forward(&mut module, &tape, wrong_features).unwrap_err();
    assert_eq!(error.code(), "E_CONFIG");
    assert!(module.parameter_indices().is_empty());
}
