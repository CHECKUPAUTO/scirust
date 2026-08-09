//! Minimal training orchestration for VQNet-like hybrid models.
//!
//! This module deliberately reuses SciRust's existing [`Module`], [`Loss`],
//! reverse-mode [`Tape`], and tape [`Optimizer`] contracts. It does not define a
//! second optimizer or loss hierarchy. A fresh tape is created for every step,
//! model state is synchronized after the optimizer update, and the exact ordered
//! parameter-index layout is pinned after the first step so stateful tape
//! optimizers cannot silently associate moments with different nodes later.

use crate::autodiff::optim::Optimizer;
use crate::autodiff::reverse::{Tape, Tensor};
use crate::error::{Result, SciRustError};
use crate::nn::loss::Loss;
use crate::nn::module::Module;

/// Observable result of one completed optimization step.
#[derive(Debug, Clone)]
pub struct TrainStepReport {
    loss: f32,
    prediction: Tensor,
    parameter_indices: Vec<usize>,
}

impl TrainStepReport {
    /// Scalar loss value evaluated before the parameter update.
    #[must_use]
    pub const fn loss(&self) -> f32 {
        self.loss
    }

    /// Model prediction evaluated before the parameter update.
    #[must_use]
    pub const fn prediction(&self) -> &Tensor {
        &self.prediction
    }

    /// Exact ordered trainable tape-node layout used for this step.
    #[must_use]
    pub fn parameter_indices(&self) -> &[usize] {
        &self.parameter_indices
    }
}

/// Stateful training session wrapping one existing SciRust tape optimizer.
///
/// The first successful step records the model's exact ordered
/// `parameter_indices()`. Every later step must reproduce that layout. This is
/// important for tape optimizers such as Adam whose moment maps are keyed by
/// tape node index: a dynamic graph that changes parameter-node identity is
/// rejected instead of silently reusing optimizer state for another parameter.
pub struct TrainingSession<O> {
    optimizer: O,
    parameter_layout: Option<Vec<usize>>,
    completed_steps: u64,
}

impl<O> TrainingSession<O>
where
    O: Optimizer,
{
    /// Creates a session around an existing SciRust tape optimizer.
    pub const fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            parameter_layout: None,
            completed_steps: 0,
        }
    }

    /// Number of successfully completed optimization steps.
    #[must_use]
    pub const fn completed_steps(&self) -> u64 {
        self.completed_steps
    }

    /// Parameter-node layout pinned by the first successful step.
    #[must_use]
    pub fn parameter_layout(&self) -> Option<&[usize]> {
        self.parameter_layout.as_deref()
    }

    /// Shared access to the underlying SciRust optimizer.
    #[must_use]
    pub const fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Mutable access for scheduler or hyperparameter control.
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Consumes the session and returns the underlying optimizer.
    pub fn into_optimizer(self) -> O {
        self.optimizer
    }

    /// Executes one full fresh-tape training step.
    ///
    /// Sequence:
    ///
    /// 1. validate finite input/target tensors;
    /// 2. create a fresh reverse-mode tape;
    /// 3. execute `Module::try_forward`;
    /// 4. require prediction and target shapes to match;
    /// 5. evaluate the existing [`Loss`] and require a finite scalar;
    /// 6. run backward and validate finite parameter gradients;
    /// 7. pin or verify exact parameter-node layout;
    /// 8. call the existing [`Optimizer::step`];
    /// 9. reject non-finite updated parameter tensors;
    /// 10. persist them through [`Module::sync`].
    pub fn train_step<M, L>(
        &mut self,
        model: &mut M,
        loss: &L,
        input: Tensor,
        target: Tensor,
    ) -> Result<TrainStepReport>
    where
        M: Module,
        L: Loss,
    {
        validate_training_tensor(&input, "training input")?;
        validate_training_tensor(&target, "training target")?;

        let tape = Tape::new();
        let input_var = tape.input(input);
        let prediction = model.try_forward(&tape, input_var)?;
        let prediction_shape = prediction.shape();
        if prediction_shape != target.shape()
        {
            return Err(SciRustError::ShapeMismatch {
                op: "VQNet training prediction/target",
                expected: prediction_shape,
                got: target.shape(),
            });
        }

        let target_var = tape.input(target);
        let loss_var = loss.forward(&tape, prediction, target_var);
        if loss_var.shape() != (1, 1)
        {
            return Err(SciRustError::ShapeMismatch {
                op: "VQNet training loss",
                expected: (1, 1),
                got: loss_var.shape(),
            });
        }

        let loss_value = tape.value(loss_var.idx()).data[0];
        if !loss_value.is_finite()
        {
            return Err(SciRustError::InvalidConfig(
                "VQNet training loss must be finite".to_string(),
            ));
        }
        let prediction_value = tape.value(prediction.idx());
        validate_training_tensor(&prediction_value, "training prediction")?;

        tape.backward(loss_var.idx());
        let parameter_indices = model.parameter_indices();
        self.validate_parameter_layout(&parameter_indices)?;
        for &index in &parameter_indices
        {
            let gradient = tape.grad(index);
            validate_training_tensor(&gradient, "training parameter gradient")?;
        }

        self.optimizer.step(&parameter_indices, &tape);
        for &index in &parameter_indices
        {
            let updated = tape.value(index);
            validate_training_tensor(&updated, "optimizer-updated parameter")?;
        }
        model.sync(&tape);

        if self.parameter_layout.is_none()
        {
            self.parameter_layout = Some(parameter_indices.clone());
        }
        self.completed_steps = self.completed_steps.checked_add(1).ok_or_else(|| {
            SciRustError::InvalidConfig("VQNet training step counter overflow".to_string())
        })?;

        Ok(TrainStepReport {
            loss: loss_value,
            prediction: prediction_value,
            parameter_indices,
        })
    }

    fn validate_parameter_layout(&self, actual: &[usize]) -> Result<()> {
        if let Some(expected) = &self.parameter_layout
            && expected.as_slice() != actual
        {
            return Err(SciRustError::InvalidConfig(format!(
                "VQNet training parameter layout changed across fresh tapes: expected {:?}, got {:?}",
                expected, actual
            )));
        }
        Ok(())
    }
}

fn validate_training_tensor(tensor: &Tensor, what: &'static str) -> Result<()> {
    tensor.validate()?;
    if tensor.data.iter().all(|value| value.is_finite())
    {
        Ok(())
    }
    else
    {
        Err(SciRustError::InvalidConfig(format!(
            "VQNet {what} must contain only finite values"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::optim::Sgd;
    use crate::autodiff::reverse::Var;
    use crate::nn::loss::MseLoss;
    use crate::quantum::Observable;
    use crate::vqnet::{
        EntanglementTopology, EntanglingGate, Hamiltonian, HamiltonianTerm,
        ParameterInitializer, QuantumModule, RotationAxis, VariationalCircuitBuilder,
    };

    fn one_qubit_module() -> QuantumModule {
        let mut builder = VariationalCircuitBuilder::new(1).unwrap();
        builder.angle_encoding(RotationAxis::Y, &[0]).unwrap();
        builder
            .variational_ansatz(
                1,
                &[RotationAxis::Y],
                EntanglementTopology::None,
                EntanglingGate::Cnot,
            )
            .unwrap();
        builder.measure_all_z().unwrap();
        QuantumModule::new(builder.build().unwrap(), ParameterInitializer::Constant(0.2)).unwrap()
    }

    #[test]
    fn training_session_reuses_existing_module_loss_and_optimizer_contracts() {
        let mut model = one_qubit_module();
        let loss = MseLoss::new();
        let mut session = TrainingSession::new(Sgd::new(0.05));
        let input = Tensor::from_vec(vec![0.2, -0.4], 2, 1);
        let target = Tensor::from_vec(vec![0.0, 0.0], 2, 1);

        let first = session
            .train_step(&mut model, &loss, input.clone(), target.clone())
            .unwrap();
        let second = session
            .train_step(&mut model, &loss, input, target)
            .unwrap();

        assert_eq!(session.completed_steps(), 2);
        assert_eq!(first.parameter_indices(), second.parameter_indices());
        assert_eq!(session.parameter_layout(), Some(first.parameter_indices()));
        assert_eq!(first.prediction().shape(), (2, 1));
        assert!(first.loss().is_finite());
        assert!(second.loss().is_finite());
        assert_ne!(model.parameters().values()[0], 0.2);
    }

    struct DriftingModule {
        calls: usize,
        parameter: Tensor,
        last_parameter: Option<usize>,
    }

    impl DriftingModule {
        fn new() -> Self {
            Self {
                calls: 0,
                parameter: Tensor::from_vec(vec![0.5], 1, 1),
                last_parameter: None,
            }
        }
    }

    impl Module for DriftingModule {
        fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
            self.calls += 1;
            if self.calls > 1
            {
                let _padding = tape.input(Tensor::from_vec(vec![0.0], 1, 1));
            }
            let parameter = tape.input(self.parameter.clone());
            self.last_parameter = Some(parameter.idx());
            input.try_hadamard(parameter).unwrap()
        }

        fn parameter_indices(&self) -> Vec<usize> {
            self.last_parameter.into_iter().collect()
        }

        fn sync(&mut self, tape: &Tape) {
            if let Some(index) = self.last_parameter
            {
                self.parameter = tape.value(index);
            }
        }
    }

    #[test]
    fn training_session_rejects_parameter_node_drift_before_optimizer_step() {
        let mut model = DriftingModule::new();
        let loss = MseLoss::new();
        let mut session = TrainingSession::new(Sgd::new(0.1));
        let input = Tensor::from_vec(vec![1.0], 1, 1);
        let target = Tensor::from_vec(vec![0.0], 1, 1);

        session
            .train_step(&mut model, &loss, input.clone(), target.clone())
            .unwrap();
        let before = model.parameter.data.clone();
        let error = session
            .train_step(&mut model, &loss, input, target)
            .unwrap_err();

        assert!(error.to_string().contains("parameter layout changed"));
        assert_eq!(session.completed_steps(), 1);
        assert_eq!(model.parameter.data, before);
    }

    #[test]
    fn training_session_rejects_non_finite_input_before_model_execution() {
        let mut model = one_qubit_module();
        let loss = MseLoss::new();
        let mut session = TrainingSession::new(Sgd::new(0.05));
        let error = session
            .train_step(
                &mut model,
                &loss,
                Tensor::from_vec(vec![f32::NAN], 1, 1),
                Tensor::from_vec(vec![0.0], 1, 1),
            )
            .unwrap_err();

        assert!(error.to_string().contains("training input"));
        assert_eq!(session.completed_steps(), 0);
    }

    #[test]
    fn hamiltonian_readout_remains_compatible_with_training_session() {
        let mut builder = VariationalCircuitBuilder::new(1).unwrap();
        builder.angle_encoding(RotationAxis::Y, &[0]).unwrap();
        builder
            .variational_ansatz(
                1,
                &[RotationAxis::Y],
                EntanglementTopology::None,
                EntanglingGate::Cnot,
            )
            .unwrap();
        builder.measure_all_z().unwrap();
        let circuit = builder.build().unwrap();
        let hamiltonian = Hamiltonian::new(vec![
            HamiltonianTerm::new(0.5, Observable::z(0)).unwrap(),
        ])
        .unwrap();
        let readout = circuit.hamiltonian_readout(&[hamiltonian]).unwrap();
        let quantum = QuantumModule::new(circuit, ParameterInitializer::Constant(0.2)).unwrap();
        let mut model = crate::nn::sequential::Sequential::new()
            .add(quantum)
            .add(readout);
        let mut session = TrainingSession::new(Sgd::new(0.05));

        let report = session
            .train_step(
                &mut model,
                &MseLoss::new(),
                Tensor::from_vec(vec![0.3], 1, 1),
                Tensor::from_vec(vec![0.0], 1, 1),
            )
            .unwrap();

        assert_eq!(report.prediction().shape(), (1, 1));
        assert_eq!(report.parameter_indices().len(), 1);
        assert!(report.loss().is_finite());
    }
}
