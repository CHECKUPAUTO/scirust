use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::nn::nd_layers::NdLinear;
use scirust_core::nn::nd_optim::NdParam;
use scirust_core::nn::rng::PcgEngine;
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::{Result, VariationalError};
use crate::learning::trainer::HasParameters;
use crate::util::nd_tanh;

pub struct LagrangianNetwork {
    pub layers: Vec<NdLinear>,
    pub ndim: usize,
    pub hidden_dim: usize,
}

impl LagrangianNetwork {
    pub fn new(ndim: usize, hidden_dim: usize, rng: &mut PcgEngine) -> Self {
        let in_dim = 2 * ndim + 1;
        let layers = vec![
            NdLinear::new(in_dim, hidden_dim, rng),
            NdLinear::new(hidden_dim, hidden_dim, rng),
            NdLinear::new(hidden_dim, 1, rng),
        ];
        Self {
            layers,
            ndim,
            hidden_dim,
        }
    }

    pub fn forward<'t>(
        &mut self,
        tape: &'t NdTape,
        q: &[NdVar],
        dq: &[NdVar],
        t: Option<NdVar>,
    ) -> NdVar<'t> {
        let batch_ndim = q[0].shape();
        let batch_size = if batch_ndim.len() >= 2
        {
            batch_ndim[0]
        }
        else
        {
            1
        };

        let q_data = tape.value(q[0]);
        let dq_data = tape.value(dq[0]);
        let mut cat_data = Vec::new();
        for i in 0..batch_size
        {
            let base = i * self.ndim;
            for j in 0..self.ndim
            {
                cat_data.push(q_data.data[base + j]);
            }
            for j in 0..self.ndim
            {
                cat_data.push(dq_data.data[base + j]);
            }
            match t
            {
                Some(tv) =>
                {
                    let td = tape.value(tv);
                    cat_data.push(td.data[i.min(td.data.len().saturating_sub(1))]);
                },
                None => cat_data.push(0.0),
            }
        }
        let inputs = tape.input(TensorND::new(cat_data, vec![batch_size, 2 * self.ndim + 1]));

        let h = nd_tanh(tape, self.layers[0].forward(tape, inputs));
        let h = nd_tanh(tape, self.layers[1].forward(tape, h));
        self.layers[2].forward(tape, h)
    }

    pub fn acceleration_from_state(&mut self, q: &[f32], dq: &[f32], t: f32) -> Result<Vec<f32>> {
        let n = self.ndim;
        let tape = NdTape::new();
        let qv = tape.input(TensorND::new(q.to_vec(), vec![1, n]));
        let dqv = tape.input(TensorND::new(dq.to_vec(), vec![1, n]));
        let tv = tape.input(TensorND::new(vec![t], vec![1, 1]));
        let q_arr = [qv];
        let dq_arr = [dqv];
        let L = self.forward(&tape, &q_arr, &dq_arr, Some(tv));
        let grads = tape.backward(L);
        let dL_dq: Vec<f32> = grads[0].data[..n].to_vec();
        let dL_ddq: Vec<f32> = grads[1].data[..n].to_vec();

        for &v in dL_dq.iter().chain(dL_ddq.iter())
        {
            if !v.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "LNN gradient",
                    value: v,
                });
            }
        }

        let eps = 1e-4;
        let hessian = {
            let mut closure = |dq_pert: &[f32]| {
                let tape2 = NdTape::new();
                let qv2 = tape2.input(TensorND::new(q.to_vec(), vec![1, n]));
                let dqv2 = tape2.input(TensorND::new(dq_pert.to_vec(), vec![1, n]));
                let tv2 = tape2.input(TensorND::new(vec![t], vec![1, 1]));
                let q2_arr = [qv2];
                let dq2_arr = [dqv2];
                let L2 = self.forward(&tape2, &q2_arr, &dq2_arr, Some(tv2));
                let g2 = tape2.backward(L2);
                g2[1].data[..n].to_vec()
            };
            crate::util::finite_difference_hessian(&mut closure, dq, eps, n)
        };

        for i in 0..n
        {
            if !hessian[i][i].is_finite()
            {
                return Err(VariationalError::SingularVelocityHessian {
                    condition_number: f32::INFINITY,
                    tolerance: 1e-8,
                });
            }
        }

        let acc = crate::util::solve_linear_system(&hessian, &dL_dq, n).map_err(|_| {
            VariationalError::LinearSolveFailure {
                details: "LNN acceleration solve failed".into(),
            }
        })?;

        Ok(acc)
    }

    pub fn eval_lagrangian(&mut self, q: &[f32], dq: &[f32], t: f32) -> f32 {
        let tape = NdTape::new();
        let qv = tape.input(TensorND::new(q.to_vec(), vec![1, self.ndim]));
        let dqv = tape.input(TensorND::new(dq.to_vec(), vec![1, self.ndim]));
        let tv = tape.input(TensorND::new(vec![t], vec![1, 1]));
        let q_arr = [qv];
        let dq_arr = [dqv];
        let L = self.forward(&tape, &q_arr, &dq_arr, Some(tv));
        tape.value(L).data[0]
    }
}

impl HasParameters for LagrangianNetwork {
    fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = Vec::new();
        for layer in &mut self.layers
        {
            params.extend(layer.parameters());
        }
        params
    }
}
