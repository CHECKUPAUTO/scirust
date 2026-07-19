//! Deterministic, single-sample hybrid quantum-classical regression/classifier core.
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
    let x = 0.7_f32;
    let target = x.cos();
    let mut scale = 0.4_f32;
    let mut angle = 0.2_f32;
    let mut optimizer = Sgd::new(0.25);
    let mut first_loss = None;
    let mut final_loss = 0.0_f32;

    for epoch in 0..80
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
        let loss_value = tape.value(loss.idx()).data[0];
        first_loss.get_or_insert(loss_value);
        final_loss = loss_value;
        loss.backward();
        optimizer.step(&[scale_var.idx(), angle_var.idx()], &tape);
        scale = tape.value(scale_var.idx()).data[0];
        angle = tape.value(angle_var.idx()).data[0];
        if epoch % 20 == 0 || epoch == 79
        {
            println!("epoch={epoch:02} loss={loss_value:.7} scale={scale:.5} angle={angle:.5}");
        }
    }

    assert!(final_loss < first_loss.expect("at least one epoch") * 0.01);
}
