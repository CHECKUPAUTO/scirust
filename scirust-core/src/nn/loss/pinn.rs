use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::Module;

/// Evaluator for Physics-Informed Neural Networks (PINN).
/// Integrates physical laws (PDEs) into the loss function by computing
/// higher-order derivatives of the model output with respect to its inputs.
pub struct PinnLossEvaluator<'a, M: Module> {
    pub model: &'a mut M,
    pub alpha: f32, // example physical constant (e.g., thermal diffusivity)
}

impl<'a, M: Module> PinnLossEvaluator<'a, M> {
    pub fn new(model: &'a mut M, alpha: f32) -> Self {
        Self { model, alpha }
    }

    /// Computes the PDE residual for the 1-D heat equation
    /// `du/dt - alpha * d^2u/dx^2 = 0` using **finite differences** on the input
    /// coordinates. The perturbed forward passes are recorded on `tape`, so the
    /// residual stays differentiable w.r.t. the model parameters (it can be
    /// backpropagated to train the network). This is the standard numerical-PINN
    /// formulation: SciRust's numeric reverse-mode tape does not provide analytic
    /// higher-order autodiff, so the input derivatives are taken numerically while
    /// the parameter gradients remain exact via the tape.
    ///
    /// `coords`: (batch, 2) where column 0 is x and column 1 is t.
    pub fn compute_heat_residual<'t>(&mut self, tape: &'t Tape, coords: Var<'t>) -> Var<'t> {
        let h = 1e-3_f32;
        let inv_2h = 1.0 / (2.0 * h);
        let inv_h2 = 1.0 / (h * h);

        // Raw coordinate values, used to build perturbed (constant) inputs.
        let cv = tape.value(coords.idx());
        let x_plus = tape.input(perturb_col(&cv, 0, h));
        let x_minus = tape.input(perturb_col(&cv, 0, -h));
        let t_plus = tape.input(perturb_col(&cv, 1, h));
        let t_minus = tape.input(perturb_col(&cv, 1, -h));

        // Forward passes — all recorded on the tape, hence differentiable w.r.t. params.
        let u = self.model.forward(tape, coords);
        let u_xp = self.model.forward(tape, x_plus);
        let u_xm = self.model.forward(tape, x_minus);
        let u_tp = self.model.forward(tape, t_plus);
        let u_tm = self.model.forward(tape, t_minus);

        // d2u/dx2 ≈ (u(x+h) - 2 u(x) + u(x-h)) / h^2
        let d2u_dx2 = u_xp.add(u_xm).sub(u.scale(2.0)).scale(inv_h2);
        // du/dt ≈ (u(t+h) - u(t-h)) / (2h)
        let du_dt = u_tp.sub(u_tm).scale(inv_2h);

        // Residual R = du/dt - alpha * d2u/dx2
        du_dt.sub(d2u_dx2.scale(self.alpha))
    }

    /// Combined loss: data_loss + lambda * pde_residual_loss
    pub fn total_loss<'t>(
        &mut self,
        tape: &'t Tape,
        coords: Var<'t>,
        targets: Var<'t>,
        lambda: f32,
    ) -> Var<'t> {
        let u = self.model.forward(tape, coords);

        // Data loss (MSE)
        let diff = u.try_sub(targets).unwrap();
        let data_loss = diff.try_hadamard(diff).unwrap().mean_axis(0).sum();

        let residual = self.compute_heat_residual(tape, coords);
        let pde_loss = residual.try_hadamard(residual).unwrap().mean_axis(0).sum();

        let lambda_var = tape.input(Tensor::from_vec(vec![lambda], 1, 1));
        data_loss
            .try_add(pde_loss.try_mul_broadcast(lambda_var).unwrap())
            .unwrap()
    }
}

/// Returns a copy of `t` (row-major, shape `rows x cols`) with `h` added to
/// every entry of column `col`. Used to build the perturbed coordinate inputs
/// for finite-difference PDE residuals.
fn perturb_col(t: &Tensor, col: usize, h: f32) -> Tensor {
    let mut data = t.data.clone();
    let cols = t.cols;
    let mut i = col;
    while i < data.len()
    {
        data[i] += h;
        i += cols;
    }
    Tensor::from_vec(data, t.rows, t.cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{KaimingNormal, Linear, PcgEngine, Sequential, Zeros};

    #[test]
    fn heat_residual_is_not_identically_zero() {
        // Regression: `tape_grad` used to return zeros, so the heat residual was
        // identically 0 for every input/model. The finite-difference residual
        // must instead reflect the model's actual derivatives.
        let mut rng = PcgEngine::new(7);
        let mut model = Sequential::new().add(Linear::new(2, 1, &KaimingNormal, &Zeros, &mut rng));
        let mut pinn = PinnLossEvaluator::new(&mut model, 0.1);

        let tape = Tape::new();
        // (x, t) coordinate batch: 3 rows, 2 cols.
        let coords = tape.input(Tensor::from_vec(vec![0.1, 0.2, 0.5, 0.7, 0.9, 0.3], 3, 2));
        let residual = pinn.compute_heat_residual(&tape, coords);
        let r = tape.value(residual.idx());

        assert!(
            r.data.iter().all(|v| v.is_finite()),
            "residual must be finite"
        );
        assert!(
            r.data.iter().any(|&v| v.abs() > 1e-6),
            "residual must reflect the model's t-derivative (was identically zero before the fix)"
        );
    }
}
