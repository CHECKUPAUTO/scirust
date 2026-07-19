//! Deterministic two-sample hybrid quantum-classical binary classifier.
//!
//! The classical scalar `scale` angle-encodes `x`; the quantum layer adds a
//! trainable `Ry` angle and returns `<Z>`. Both parameters are updated by the
//! existing SciRust tape `Sgd` optimizer using the parameter-shift backward.

use scirust_core::autodiff::optim::{Optimizer, Sgd};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::quantum::{Circuit, Observable, Operation, Parameter, ParameterId, QuantumLayer};

fn layer() -> QuantumLayer {
    let encoded = ParameterId(0);
    let trainable = ParameterId(1);
    let mut circuit = Circuit::new(1).expect("one qubit is valid");
    circuit
        .push(Operation::Ry {
            target: 0,
            parameter: Parameter::Symbol(encoded),
        })
        .expect("valid encoded rotation")
        .push(Operation::Ry {
            target: 0,
            parameter: Parameter::Symbol(trainable),
        })
        .expect("valid trainable rotation");
    QuantumLayer::new(circuit, Observable::z(0), vec![encoded], vec![trainable])
        .expect("complete parameter mapping")
}

fn main() {
    // A tiny deterministic binary dataset: |0> should predict +1 and the
    // pi-encoded sample should predict -1.
    let dataset = [(0.0_f32, 1.0_f32), (core::f32::consts::PI, -1.0_f32)];
    let mut scale = 0.5_f32;
    let mut angle = 0.1_f32;
    let mut optimizer = Sgd::new(0.15);
    let mut first_loss = None;
    let mut final_loss = 0.0_f32;

    for epoch in 0..120
    {
        let mut epoch_loss = 0.0;
        for &(x, target) in &dataset
        {
            let tape = Tape::new();
            let x_var = tape.input(Tensor::from_vec(vec![x], 1, 1));
            let scale_var = tape.input(Tensor::from_vec(vec![scale], 1, 1));
            let angle_var = tape.input(Tensor::from_vec(vec![angle], 1, 1));
            let target_var = tape.input(Tensor::from_vec(vec![target], 1, 1));
            let encoded = x_var.matmul(scale_var);
            let prediction = layer()
                .forward(encoded, angle_var)
                .expect("quantum forward");
            let residual = prediction.sub(target_var);
            let loss = residual.mul(residual);
            epoch_loss += tape.value(loss.idx()).data[0];
            loss.backward();
            optimizer.step(&[scale_var.idx(), angle_var.idx()], &tape);
            scale = tape.value(scale_var.idx()).data[0];
            angle = tape.value(angle_var.idx()).data[0];
        }
        let loss_value = epoch_loss / dataset.len() as f32;
        first_loss.get_or_insert(loss_value);
        final_loss = loss_value;
        if epoch % 30 == 0 || epoch == 119
        {
            println!("epoch={epoch:03} loss={loss_value:.7} scale={scale:.5} angle={angle:.5}");
        }
    }

    assert!(final_loss < first_loss.expect("at least one epoch") * 0.01);
}
