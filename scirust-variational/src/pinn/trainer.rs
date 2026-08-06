use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::nn::nd_layers::NdLinear;
use scirust_core::nn::nd_optim::{NdAdam, NdParam};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::Result;
use crate::learning::trainer::{HasParameters, TrainingConfig, TrainingMetrics};
use crate::pinn::collocation::CollocationPoints;
use crate::pinn::conditions::ConditionConfig;
use crate::util::nd_tanh;

pub struct PinnNet {
    pub layers: Vec<NdLinear>,
    pub ndim: usize,
    pub hidden_dim: usize,
}

impl PinnNet {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        rng: &mut PcgEngine,
    ) -> Self {
        let layers = vec![
            NdLinear::new(input_dim, hidden_dim, rng),
            NdLinear::new(hidden_dim, hidden_dim, rng),
            NdLinear::new(hidden_dim, output_dim, rng),
        ];
        Self {
            layers,
            ndim: input_dim,
            hidden_dim,
        }
    }

    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let h = nd_tanh(tape, self.layers[0].forward(tape, x));
        let h = nd_tanh(tape, self.layers[1].forward(tape, h));
        self.layers[2].forward(tape, h)
    }
}

impl HasParameters for PinnNet {
    fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = Vec::new();
        for layer in &mut self.layers
        {
            params.extend(layer.parameters());
        }
        params
    }
}

pub struct ResidualFn {
    pub residual: Box<dyn for<'a> Fn(&'a NdTape, &'a mut PinnNet, NdVar<'a>) -> NdVar<'a>>,
}

pub struct PinnTrainer {
    pub config: TrainingConfig,
}

impl PinnTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    pub fn train<F>(
        &self,
        model: &mut PinnNet,
        interior_points: &CollocationPoints,
        interior_residual: F,
        bc_config: &ConditionConfig,
    ) -> Result<TrainingMetrics>
    where
        F: for<'a> Fn(&'a NdTape, &'a mut PinnNet, NdVar<'a>) -> NdVar<'a>,
    {
        let mut opt = NdAdam::with_lr(self.config.learning_rate);
        let _rng = PcgEngine::new(self.config.seed);
        let mut metrics = TrainingMetrics::new();
        let _ndim = interior_points.ndim;

        for epoch in 0..self.config.num_epochs
        {
            let tape = NdTape::new();
            let (x_flat, x_shape) = interior_points.to_batched_tensor();
            let xv = tape.input(TensorND::new(x_flat, x_shape));

            let mut bc_loss_terms = Vec::new();
            for condition in &bc_config.conditions
            {
                let (c_flat, c_shape) = {
                    let flat: Vec<f32> = condition
                        .points
                        .iter()
                        .flat_map(|p| p.iter())
                        .copied()
                        .collect();
                    let shape = vec![condition.points.len(), condition.points[0].len()];
                    (flat, shape)
                };

                let cv = tape.input(TensorND::new(c_flat, c_shape));
                let u_bc = model.forward(&tape, cv);
                let targets = condition.evaluate_targets();
                let t_tensor = TensorND::new(targets, vec![condition.points.len(), 1]);
                let tv = tape.input(t_tensor);

                let diff = u_bc.sub(tv);
                let bc_loss = diff.mul(diff).sum();
                let w = tape.input(TensorND::new(vec![condition.weight], vec![1, 1]));
                bc_loss_terms.push(bc_loss.mul(w));
            }

            let u_pred = model.forward(&tape, xv);
            let interior_loss = interior_residual(&tape, model, u_pred);

            let total_loss = if bc_loss_terms.is_empty()
            {
                interior_loss
            }
            else
            {
                let mut loss = interior_loss;
                for term in bc_loss_terms
                {
                    loss = loss.add(term);
                }
                loss
            };

            let loss_val = tape.value(total_loss).data[0];

            let grads = tape.backward(total_loss);
            let mut params = model.parameters();
            opt.step(&mut params, &grads);

            if epoch % 10 == 0 || epoch == self.config.num_epochs - 1
            {
                metrics.record(epoch, loss_val, f32::NAN, 0.0);
            }
        }

        Ok(metrics)
    }

    pub fn solve_poisson(
        &self,
        model: &mut PinnNet,
        interior: &CollocationPoints,
        source_fn: impl Fn(f32) -> f32,
        bc_config: &ConditionConfig,
    ) -> Result<TrainingMetrics> {
        self.train(
            model,
            interior,
            |tape, net, u| {
                let eps = 1e-3;
                let x = interior.to_flat();

                let mut lap_terms = Vec::new();
                for i in 0..x.len()
                {
                    let mut xp = x.clone();
                    xp[i] += eps;
                    let xp_var = tape.input(TensorND::new(xp, vec![x.len(), 1]));
                    let up = net.forward(tape, xp_var);

                    let mut xm = x.clone();
                    xm[i] -= eps;
                    let xm_var = tape.input(TensorND::new(xm, vec![x.len(), 1]));
                    let um = net.forward(tape, xm_var);

                    let u_center = u;

                    let eps_sq = tape.input(TensorND::new(vec![eps * eps], vec![1, 1]));
                    let two = tape.input(TensorND::new(vec![2.0], vec![1, 1]));
                    let d2u = up.add(um).sub(u_center.mul(two)).div(eps_sq);
                    lap_terms.push(d2u);
                }

                let mut laplacian = tape.input(TensorND::zeros(&[1, 1]));
                for term in lap_terms
                {
                    laplacian = laplacian.add(term);
                }

                let source_vals: Vec<f32> = x.iter().map(|&xi| source_fn(xi)).collect();
                let source = tape.input(TensorND::new(source_vals, vec![x.len(), 1]));

                laplacian.add(source)
            },
            bc_config,
        )
    }
}
