//! Training-session integration with SciRust's existing deterministic data loader.
//!
//! No VQNet-specific dataset or loader hierarchy is introduced. This extension
//! selects the requested loader epoch and delegates its already-batched tensor
//! iterator to [`TrainingSession::train_epoch`].

use super::{EpochReport, TrainingSession};
use crate::autodiff::optim::Optimizer;
use crate::data::{DataLoader, Dataset};
use crate::error::Result;
use crate::nn::loss::Loss;
use crate::nn::module::Module;

impl<O> TrainingSession<O>
where
    O: Optimizer,
{
    /// Trains one explicit epoch from SciRust's existing [`DataLoader`].
    ///
    /// `shuffle_epoch(epoch)` is invoked exactly once before consuming
    /// `loader.iter()`. Batching and sample order therefore remain authoritative
    /// in the core data layer, while fresh-tape execution, finite-value checks,
    /// parameter-layout guards, optimization, synchronization, and reporting all
    /// remain authoritative in [`TrainingSession::train_epoch`].
    ///
    /// With the epoch-addressable shuffle contract, `(dataset, seed, epoch)` is
    /// sufficient to reproduce the same batch order after a resumed run without
    /// replaying previous epochs.
    pub fn train_loader_epoch<M, L, D>(
        &mut self,
        model: &mut M,
        loss: &L,
        loader: &mut DataLoader<D>,
        epoch: u64,
    ) -> Result<EpochReport>
    where
        M: Module,
        L: Loss,
        D: Dataset,
    {
        loader.shuffle_epoch(epoch);
        self.train_epoch(model, loss, loader.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::optim::Sgd;
    use crate::data::InMemoryDataset;
    use crate::nn::loss::MseLoss;
    use crate::vqnet::{
        EntanglementTopology, EntanglingGate, ParameterInitializer, QuantumModule, RotationAxis,
        VariationalCircuitBuilder,
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
        QuantumModule::new(
            builder.build().unwrap(),
            ParameterInitializer::Constant(0.2),
        )
        .unwrap()
    }

    fn scalar_dataset() -> InMemoryDataset {
        InMemoryDataset::new(
            vec![0.1, -0.2, 0.35, -0.45],
            vec![0.0, 0.1, -0.1, 0.2],
            1,
            1,
        )
    }

    #[test]
    fn loader_epoch_reuses_core_batching_and_is_reproducible() {
        let mut first_model = one_qubit_module();
        let mut second_model = one_qubit_module();
        let mut first_session = TrainingSession::new(Sgd::new(0.03));
        let mut second_session = TrainingSession::new(Sgd::new(0.03));
        let mut first_loader = DataLoader::new(scalar_dataset(), 2, true, 0x51a7);
        let mut second_loader = DataLoader::new(scalar_dataset(), 2, true, 0x51a7);

        let first = first_session
            .train_loader_epoch(&mut first_model, &MseLoss::new(), &mut first_loader, 5)
            .unwrap();
        let second = second_session
            .train_loader_epoch(&mut second_model, &MseLoss::new(), &mut second_loader, 5)
            .unwrap();

        assert_eq!(first.steps(), 2);
        assert_eq!(second.steps(), 2);
        assert_eq!(first_session.completed_steps(), 2);
        assert_eq!(second_session.completed_steps(), 2);
        assert_eq!(first.mean_loss().to_bits(), second.mean_loss().to_bits());
        assert_eq!(first.last_loss().to_bits(), second.last_loss().to_bits());
        assert_eq!(first.last_prediction().data, second.last_prediction().data);
        assert_eq!(first_model.parameters(), second_model.parameters());
    }

    #[test]
    fn empty_core_loader_surfaces_existing_empty_epoch_error() {
        let dataset = InMemoryDataset::new(Vec::new(), Vec::new(), 1, 1);
        let mut loader = DataLoader::new(dataset, 2, false, 7);
        let mut model = one_qubit_module();
        let mut session = TrainingSession::new(Sgd::new(0.03));

        let error = session
            .train_loader_epoch(&mut model, &MseLoss::new(), &mut loader, 0)
            .unwrap_err();
        assert!(error.to_string().contains("at least one batch"));
        assert_eq!(session.completed_steps(), 0);
    }
}
