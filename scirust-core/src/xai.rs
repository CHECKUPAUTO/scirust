// scirust-core/src/xai.rs
// Explainable AI (XAI) Integrated Gradients Engine

use crate::autodiff::reverse::{Tape, Tensor, Var};

/// Computes Integrated Gradients for a model function.
/// Integrated Gradients maps feature attribution by path integral of gradients.
pub fn integrated_gradients<F>(
    input: &Tensor,
    baseline: &Tensor,
    steps: usize,
    model_fn: F,
) -> Tensor
where
    F: for<'t> Fn(&Var<'t>) -> Var<'t>,
{
    let mut total_gradients = Tensor::zeros(input.rows, input.cols);
    let diff = input.sub(baseline);
    let step_size = 1.0 / steps as f32;

    for i in 0..=steps
    {
        let alpha = i as f32 * step_size;
        let tape = Tape::new();

        // Interpolated input: baseline + alpha * (input - baseline)
        let interpolated_data = baseline.add(&diff.scale(alpha));
        let x = tape.input(interpolated_data);
        let x_idx = x.idx();

        let output = model_fn(&x);
        output.backward();

        let grad = tape.grad(x_idx);
        total_gradients.add_assign(&grad);
    }

    // Average gradients and multiply by (input - baseline)
    let avg_gradients = total_gradients.scale(1.0 / (steps + 1) as f32);
    avg_gradients.hadamard(&diff)
}
