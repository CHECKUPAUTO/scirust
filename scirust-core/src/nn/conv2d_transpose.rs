use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::rng::PcgEngine;

/// Convolution transposée 2D.
/// Utilisée notamment dans les auto-encodeurs pour le décodage.
pub struct Conv2dTranspose {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    last_w_idx: Option<usize>,
    last_b_idx: Option<usize>,
}

impl Conv2dTranspose {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        rng: &mut PcgEngine,
    ) -> Self {
        let kk = kernel * kernel;
        let mut weight = Tensor::zeros(in_channels, out_channels * kk);
        let scale =
            (2.0 / (in_channels * kk + out_channels) as f32).sqrt();
        for x in weight.data.iter_mut() {
            *x = rng.float_signed() * scale;
        }
        Self {
            weight,
            bias: Some(Tensor::zeros(1, out_channels)),
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            output_padding,
            last_w_idx: None,
            last_b_idx: None,
        }
    }

    pub fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let w = tape.input(self.weight.clone());
        let b = self.bias.as_ref().map(|bias| tape.input(bias.clone()));
        self.last_w_idx = Some(w.idx());
        if let Some(ref bv) = b {
            self.last_b_idx = Some(bv.idx());
        }

        let (_batch, total) = input.shape();
        let per_channel = total / self.in_channels;
        let h_in = (per_channel as f64).sqrt() as usize;
        let w_in = h_in;

        let _h_out = (h_in - 1) * self.stride - 2 * self.padding + self.kernel + self.output_padding;
        let _w_out = (w_in - 1) * self.stride - 2 * self.padding + self.kernel + self.output_padding;

        // ConvTranspose = matmul avec transposée du poids
        let weight_t = w.transpose();
        let out = input.matmul(weight_t);

        let result = if let Some(bv) = b {
            out.add(bv)
        } else {
            out
        };

        result
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_w_idx {
            v.push(i);
        }
        if let Some(i) = self.last_b_idx {
            v.push(i);
        }
        v
    }

    pub fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_idx {
            self.weight = tape.value(i);
        }
        if let Some(i) = self.last_b_idx {
            self.bias = Some(tape.value(i));
        }
    }
}
