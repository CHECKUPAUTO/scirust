//! Deterministic full-batch four-class hybrid quantum-classical classifier.
//!
//! Two classical features pass through a trainable 2 x 2 encoder. One
//! batched quantum node then evaluates two ordered observables using two
//! trainable quantum parameters shared by all four samples. The observable
//! pair is decoded directly as a two-bit class code.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::quantum::{Circuit, Observable, Operation, Parameter, ParameterId, QuantumLayer};

const SAMPLE_COUNT: usize = 4;
const FEATURE_COUNT: usize = 2;
const OBSERVABLE_COUNT: usize = 2;

const INITIAL_ENCODER: [f32; 4] = [1.20, 0.25, -0.15, 2.70];
const INITIAL_QUANTUM_PARAMETERS: [f32; 2] = [0.35, -0.28];
const LEARNING_RATE: f32 = 0.05;
const EPOCHS: usize = 300;
const REPORT_INTERVAL: usize = 50;

const FINAL_LOSS_THRESHOLD: f32 = 1.0e-4;
const LOSS_RATIO_THRESHOLD: f32 = 1.0e-3;

const CODEWORDS: [[f32; OBSERVABLE_COUNT]; SAMPLE_COUNT] =
    [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]];

#[derive(Debug, Clone, PartialEq)]
struct TrainingResult {
    initial_loss: f32,
    final_loss: f32,
    encoder: Vec<f32>,
    quantum_parameters: Vec<f32>,
    outputs: Vec<f32>,
    predictions: Vec<usize>,
    accuracy: usize,
    checkpoints: Vec<(usize, f32)>,
}

#[derive(Debug)]
struct Evaluation {
    loss: f32,
    outputs: Vec<f32>,
}

fn layer() -> QuantumLayer {
    let encoded_0 = ParameterId(0);
    let encoded_1 = ParameterId(1);
    let theta_0 = ParameterId(2);
    let theta_1 = ParameterId(3);

    let mut circuit = Circuit::new(2).expect("two qubits are valid");
    circuit
        .push(Operation::Ry {
            target: 0,
            parameter: Parameter::Symbol(encoded_0),
        })
        .expect("valid first encoded rotation")
        .push(Operation::Ry {
            target: 1,
            parameter: Parameter::Symbol(encoded_1),
        })
        .expect("valid second encoded rotation")
        .push(Operation::Cnot {
            control: 0,
            target: 1,
        })
        .expect("valid entangling gate")
        .push(Operation::Ry {
            target: 0,
            parameter: Parameter::Symbol(theta_0),
        })
        .expect("valid first trainable rotation")
        .push(Operation::Ry {
            target: 1,
            parameter: Parameter::Symbol(theta_1),
        })
        .expect("valid second trainable rotation");

    QuantumLayer::new_multi(
        circuit,
        vec![Observable::z(0), Observable::z(1)],
        vec![encoded_0, encoded_1],
        vec![theta_0, theta_1],
    )
    .expect("complete ordered parameter mapping")
}

fn input_tensor() -> Tensor {
    Tensor::from_vec(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        SAMPLE_COUNT,
        FEATURE_COUNT,
    )
}

fn target_tensor() -> Tensor {
    Tensor::from_vec(
        CODEWORDS.iter().flatten().copied().collect(),
        SAMPLE_COUNT,
        OBSERVABLE_COUNT,
    )
}

fn evaluate(encoder: &[f32], quantum_parameters: &[f32]) -> Evaluation {
    let tape = Tape::new();
    // Keep this input order identical to the training graph so parameter
    // indices remain stable across fresh tapes.
    let inputs_var = tape.input(input_tensor());
    let encoder_var = tape.input(Tensor::from_vec(encoder.to_vec(), 2, 2));
    let quantum_parameters_var = tape.input(Tensor::from_vec(quantum_parameters.to_vec(), 1, 2));
    let targets_var = tape.input(target_tensor());

    let encoded = inputs_var.matmul(encoder_var);
    let outputs = layer()
        .forward_batch(encoded, quantum_parameters_var)
        .expect("batched quantum forward");
    let residual = outputs.sub(targets_var);
    let squared = residual.mul(residual);
    let loss = squared.sum().scale(1.0 / 8.0);
    assert_eq!(loss.shape(), (1, 1));

    Evaluation {
        loss: tape.value(loss.idx()).data[0],
        outputs: tape.value(outputs.idx()).data,
    }
}

fn nearest_class(output: &[f32]) -> usize {
    assert_eq!(output.len(), OBSERVABLE_COUNT);
    let mut best_class = 0;
    let mut best_distance = f32::INFINITY;
    for (class, codeword) in CODEWORDS.iter().enumerate()
    {
        let distance = output
            .iter()
            .zip(codeword)
            .map(|(&actual, &target)| {
                let difference = actual - target;
                difference * difference
            })
            .sum::<f32>();
        if distance < best_distance
        {
            best_distance = distance;
            best_class = class;
        }
    }
    best_class
}

fn decode(outputs: &[f32]) -> Vec<usize> {
    let (rows, remainder) = outputs.as_chunks::<OBSERVABLE_COUNT>();
    assert!(remainder.is_empty());
    rows.iter().map(|row| nearest_class(row)).collect()
}

fn accuracy(predictions: &[usize]) -> usize {
    predictions
        .iter()
        .enumerate()
        .filter(|(sample, predicted)| *sample == **predicted)
        .count()
}

fn should_report(epoch: usize) -> bool {
    epoch == 0 || epoch.is_multiple_of(REPORT_INTERVAL) || epoch + 1 == EPOCHS
}

fn train() -> TrainingResult {
    let mut encoder = INITIAL_ENCODER.to_vec();
    let mut quantum_parameters = INITIAL_QUANTUM_PARAMETERS.to_vec();
    let mut optimizer = Adam::new(LEARNING_RATE);
    let mut initial_loss = None;
    let mut checkpoints = Vec::new();

    for epoch in 0..EPOCHS
    {
        let tape = Tape::new();
        // Inputs are deliberately created in the same order on every fresh
        // tape. Adam therefore sees the same parameter indices every epoch.
        let inputs_var = tape.input(input_tensor());
        let encoder_var = tape.input(Tensor::from_vec(encoder, 2, 2));
        let quantum_parameters_var = tape.input(Tensor::from_vec(quantum_parameters, 1, 2));
        let targets_var = tape.input(target_tensor());

        let encoded = inputs_var.matmul(encoder_var);
        let outputs = layer()
            .forward_batch(encoded, quantum_parameters_var)
            .expect("batched quantum forward");
        let residual = outputs.sub(targets_var);
        let squared = residual.mul(residual);
        let loss = squared.sum().scale(1.0 / 8.0);
        assert_eq!(loss.shape(), (1, 1));

        let loss_value = tape.value(loss.idx()).data[0];
        if epoch == 0
        {
            initial_loss = Some(loss_value);
            let initial_predictions = decode(&tape.value(outputs.idx()).data);
            assert!(
                accuracy(&initial_predictions) < SAMPLE_COUNT,
                "fixed initialization must not already classify every sample; \
                 predictions={initial_predictions:?}, outputs={:?}",
                tape.value(outputs.idx()).data
            );
        }
        if should_report(epoch)
        {
            checkpoints.push((epoch, loss_value));
        }

        loss.backward();
        optimizer.step(&[encoder_var.idx(), quantum_parameters_var.idx()], &tape);
        encoder = tape.value(encoder_var.idx()).data;
        quantum_parameters = tape.value(quantum_parameters_var.idx()).data;
    }

    let evaluation = evaluate(&encoder, &quantum_parameters);
    let predictions = decode(&evaluation.outputs);
    let correct = accuracy(&predictions);
    TrainingResult {
        initial_loss: initial_loss.expect("at least one epoch"),
        final_loss: evaluation.loss,
        encoder,
        quantum_parameters,
        outputs: evaluation.outputs,
        predictions,
        accuracy: correct,
        checkpoints,
    }
}

fn assert_training_invariants(result: &TrainingResult) {
    assert!(result.initial_loss.is_finite());
    assert!(result.final_loss.is_finite());
    assert!(result.encoder.iter().all(|value| value.is_finite()));
    assert!(
        result
            .quantum_parameters
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.outputs.iter().all(|value| value.is_finite()));
    assert!(result.checkpoints.iter().all(|(_, loss)| loss.is_finite()));

    let loss_ratio = result.final_loss / result.initial_loss;
    assert!(loss_ratio.is_finite());
    assert!(result.final_loss < result.initial_loss);
    assert!(result.final_loss < FINAL_LOSS_THRESHOLD);
    assert!(loss_ratio < LOSS_RATIO_THRESHOLD);
    assert_eq!(result.accuracy, SAMPLE_COUNT);
}

fn main() {
    let result = train();
    assert_training_invariants(&result);

    for &(epoch, loss) in &result.checkpoints
    {
        println!("epoch={epoch:03} loss={loss:.9}");
    }
    println!("initial_loss={:.9}", result.initial_loss);
    println!("final_loss={:.9}", result.final_loss);
    println!("loss_ratio={:.9}", result.final_loss / result.initial_loss);
    println!(
        "encoder=[[{:.7}, {:.7}], [{:.7}, {:.7}]]",
        result.encoder[0], result.encoder[1], result.encoder[2], result.encoder[3]
    );
    println!(
        "quantum_parameters=[{:.7}, {:.7}]",
        result.quantum_parameters[0], result.quantum_parameters[1]
    );
    println!(
        "outputs=[[{:.7}, {:.7}], [{:.7}, {:.7}], [{:.7}, {:.7}], [{:.7}, {:.7}]]",
        result.outputs[0],
        result.outputs[1],
        result.outputs[2],
        result.outputs[3],
        result.outputs[4],
        result.outputs[5],
        result.outputs[6],
        result.outputs[7]
    );
    println!("predictions={:?}", result.predictions);
    println!("accuracy={}/{}", result.accuracy, SAMPLE_COUNT);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Five f32 gates plus expectation accumulation leave errors near machine
    // precision at the analytic oracle; this allows modest rounding headroom.
    const ORACLE_TOLERANCE: f32 = 2.0e-6;
    const NONZERO_GRADIENT_NORM: f32 = 1.0e-4;

    #[test]
    fn oracle_circuit_matches_hand_derived_codewords() {
        let tape = Tape::new();
        let inputs_var = tape.input(input_tensor());
        let encoder_var = tape.input(Tensor::from_vec(
            vec![core::f32::consts::PI, 0.0, 0.0, core::f32::consts::PI],
            2,
            2,
        ));
        let quantum_parameters_var = tape.input(Tensor::from_vec(vec![0.0, 0.0], 1, 2));
        let encoded = inputs_var.matmul(encoder_var);
        let outputs = layer()
            .forward_batch(encoded, quantum_parameters_var)
            .expect("batched quantum oracle forward");

        assert_eq!(encoded.shape(), (4, 2));
        assert_eq!(outputs.shape(), (4, 2));

        let inputs = input_tensor();
        let actual = tape.value(outputs.idx());
        let targets = target_tensor();
        let mut maximum_error = 0.0_f32;
        for sample in 0..SAMPLE_COUNT
        {
            let x_0 = inputs.data[sample * FEATURE_COUNT];
            let x_1 = inputs.data[sample * FEATURE_COUNT + 1];
            let expected = [
                (core::f32::consts::PI * x_0).cos(),
                (core::f32::consts::PI * x_0).cos() * (core::f32::consts::PI * x_1).cos(),
            ];
            for observable in 0..OBSERVABLE_COUNT
            {
                let index = sample * OBSERVABLE_COUNT + observable;
                assert_eq!(expected[observable], targets.data[index]);
                let error = (actual.data[index] - expected[observable]).abs();
                maximum_error = maximum_error.max(error);
                assert!(
                    error <= ORACLE_TOLERANCE,
                    "sample {sample}, observable {observable}: error {error}"
                );
            }
        }
        println!("oracle_max_absolute_error={maximum_error:.9e}");
    }

    #[test]
    fn full_batch_graph_has_required_shapes() {
        let tape = Tape::new();
        let inputs_var = tape.input(input_tensor());
        let encoder_var = tape.input(Tensor::from_vec(INITIAL_ENCODER.to_vec(), 2, 2));
        let quantum_parameters_var =
            tape.input(Tensor::from_vec(INITIAL_QUANTUM_PARAMETERS.to_vec(), 1, 2));
        let targets_var = tape.input(target_tensor());

        let encoded = inputs_var.matmul(encoder_var);
        let outputs = layer()
            .forward_batch(encoded, quantum_parameters_var)
            .expect("one full-batch quantum forward");
        let residual = outputs.sub(targets_var);
        let squared = residual.mul(residual);
        let loss = squared.sum().scale(1.0 / 8.0);

        assert_eq!(inputs_var.shape(), (4, 2));
        assert_eq!(encoder_var.shape(), (2, 2));
        assert_eq!(encoded.shape(), (4, 2));
        assert_eq!(quantum_parameters_var.shape(), (1, 2));
        assert_eq!(outputs.shape(), (4, 2));
        assert_eq!(loss.shape(), (1, 1));
    }

    #[test]
    fn gradients_reach_encoder_and_shared_quantum_parameters() {
        let tape = Tape::new();
        let inputs_var = tape.input(input_tensor());
        let encoder_var = tape.input(Tensor::from_vec(INITIAL_ENCODER.to_vec(), 2, 2));
        let quantum_parameters_var =
            tape.input(Tensor::from_vec(INITIAL_QUANTUM_PARAMETERS.to_vec(), 1, 2));
        let targets_var = tape.input(target_tensor());

        let encoded = inputs_var.matmul(encoder_var);
        let outputs = layer()
            .forward_batch(encoded, quantum_parameters_var)
            .expect("batched quantum gradient forward");
        let residual = outputs.sub(targets_var);
        let squared = residual.mul(residual);
        let loss = squared.sum().scale(1.0 / 8.0);
        loss.backward();

        let encoder_gradient = tape.grad(encoder_var.idx());
        let quantum_gradient = tape.grad(quantum_parameters_var.idx());
        assert_eq!(encoder_gradient.shape(), (2, 2));
        assert_eq!(quantum_gradient.shape(), (1, 2));
        assert!(encoder_gradient.data.iter().all(|value| value.is_finite()));
        assert!(quantum_gradient.data.iter().all(|value| value.is_finite()));

        let encoder_norm = encoder_gradient
            .data
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let quantum_norm = quantum_gradient
            .data
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!(encoder_norm > NONZERO_GRADIENT_NORM);
        assert!(quantum_norm > NONZERO_GRADIENT_NORM);
        println!("encoder_gradient_norm={encoder_norm:.9e}");
        println!("quantum_parameter_gradient_norm={quantum_norm:.9e}");
    }

    #[test]
    fn fixed_training_schedule_converges() {
        let result = train();
        assert_training_invariants(&result);
        println!("initial_loss={:.9e}", result.initial_loss);
        println!("final_loss={:.9e}", result.final_loss);
        println!("loss_ratio={:.9e}", result.final_loss / result.initial_loss);
        println!("accuracy={}/{}", result.accuracy, SAMPLE_COUNT);
    }

    #[test]
    fn complete_training_is_exactly_deterministic() {
        let first = train();
        let second = train();

        assert_eq!(first.initial_loss, second.initial_loss);
        assert_eq!(first.final_loss, second.final_loss);
        assert_eq!(first.encoder, second.encoder);
        assert_eq!(first.quantum_parameters, second.quantum_parameters);
        assert_eq!(first.outputs, second.outputs);
        assert_eq!(first.predictions, second.predictions);
        assert_eq!(first.accuracy, second.accuracy);
        println!("exact_determinism=true");
    }
}
