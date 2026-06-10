// scirust-core/src/nn/conv2d_transpose.rs
//
// Conv2dTranspose — convolution transposée 2D avec autograd.
// Utilise conv2d_transpose_forward sur la tape autograd (implémentée
// comme une somme imbriquée correspondant à la multiplication par la
// transposée de la matrice de convolution, i.e. matmul avec padding
// décalé).

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::conv_utils::Padding;
use crate::nn::init::Initializer;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use std::collections::HashMap;

pub struct Conv2dTranspose {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_c: usize,
    pub out_c: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: Padding,
    pub output_padding: usize,
    last_w_idx: Option<usize>,
    last_b_idx: Option<usize>,
    pub name: String,
    cached_h: Option<usize>,
    cached_w: Option<usize>,
    cached_batch: Option<usize>,
}

impl Conv2dTranspose {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new<W: Initializer, B: Initializer>(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        padding: Padding,
        output_padding: usize,
        weight_init: &W,
        bias_init: Option<&B>,
        rng: &mut PcgEngine,
    ) -> crate::error::Result<Self> {
        if in_c == 0 || out_c == 0
        {
            crate::bail!("Conv2dTranspose: in_c={in_c} out_c={out_c}, doivent être > 0");
        }
        if kernel == 0 || stride == 0
        {
            crate::bail!("Conv2dTranspose: kernel={kernel} stride={stride}, doivent être > 0");
        }

        // Poids stockés (in_c, out_c * K*K) — ce sont les poids de la
        // convolution directe qui sera transposée lors du forward.
        let kk = kernel * kernel;
        let mut weight = Tensor::zeros(in_c, out_c * kk);
        weight_init.fill(&mut weight, in_c * kk, out_c, rng);

        let bias = bias_init.map(|init| {
            let mut b = Tensor::zeros(1, out_c);
            init.fill(&mut b, 1, out_c, rng);
            b
        });

        Ok(Self {
            weight,
            bias,
            in_c,
            out_c,
            kernel,
            stride,
            padding,
            output_padding,
            last_w_idx: None,
            last_b_idx: None,
            name: format!("conv2d_transpose_{in_c}_{out_c}_{kernel}"),
            cached_h: None,
            cached_w: None,
            cached_batch: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new<W: Initializer, B: Initializer>(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        padding: Padding,
        output_padding: usize,
        weight_init: &W,
        bias_init: Option<&B>,
        rng: &mut PcgEngine,
    ) -> Self {
        Self::try_new(
            in_c,
            out_c,
            kernel,
            stride,
            padding,
            output_padding,
            weight_init,
            bias_init,
            rng,
        )
        .expect("Conv2dTranspose::new failed — utilise try_new")
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    #[must_use]
    pub fn input_dims(mut self, h: usize, w: usize) -> Self {
        self.cached_h = Some(h);
        self.cached_w = Some(w);
        self
    }

    fn pad(&self) -> usize {
        match self.padding
        {
            Padding::Valid => 0,
            Padding::Same => (self.kernel - 1) / 2,
        }
    }

    pub fn h_out(&self, h: usize) -> usize {
        let p = self.pad();
        (h - 1) * self.stride + self.kernel - 2 * p + self.output_padding
    }

    pub fn w_out(&self, w: usize) -> usize {
        let p = self.pad();
        (w - 1) * self.stride + self.kernel - 2 * p + self.output_padding
    }
}

impl Module for Conv2dTranspose {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (b, total_features) = input.shape();
        let (h, w) = match (self.cached_h, self.cached_w)
        {
            (Some(h), Some(w)) => (h, w),
            _ =>
            {
                let per_channel = total_features / self.in_c;
                let side = (per_channel as f64).sqrt() as usize;
                assert_eq!(
                    side * side,
                    per_channel,
                    "Conv2dTranspose: utiliser .input_dims(h, w) pour des images non carrées"
                );
                (side, side)
            },
        };
        self.cached_h = Some(h);
        self.cached_w = Some(w);
        self.cached_batch = Some(b);

        let weight_v = tape.input(self.weight.clone());
        let bias_v = self.bias.as_ref().map(|t| tape.input(t.clone()));
        self.last_w_idx = Some(weight_v.idx());
        self.last_b_idx = bias_v.as_ref().map(|v| v.idx());
        let p = self.pad();

        input.try_conv2d_transpose_forward(
            weight_v,
            bias_v,
            b,
            self.in_c,
            h,
            w,
            self.out_c,
            self.kernel,
            self.stride,
            p,
            self.output_padding,
        )
        .unwrap()
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_w_idx
        {
            v.push(i);
        }
        if let Some(i) = self.last_b_idx
        {
            v.push(i);
        }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_idx
        {
            self.weight = tape.value(i);
        }
        if let Some(i) = self.last_b_idx
        {
            self.bias = Some(tape.value(i));
        }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.weight", self.name), self.weight.clone());
        if let Some(b) = &self.bias
        {
            map.insert(format!("{}.bias", self.name), b.clone());
        }
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        let w = sd
            .get(&format!("{}.weight", self.name))
            .ok_or_else(|| format!("missing key: {}.weight", self.name))?;
        let kk = self.kernel * self.kernel;
        if w.shape() != (self.in_c, self.out_c * kk)
        {
            crate::bail!(
                "weight shape mismatch: expected {:?}, got {:?}",
                (self.in_c, self.out_c * kk),
                w.shape()
            );
        }
        self.weight = w.clone();
        if let Some(b) = sd.get(&format!("{}.bias", self.name))
        {
            if b.shape() != (1, self.out_c)
            {
                crate::bail!("bias shape mismatch");
            }
            self.bias = Some(b.clone());
        }
        Ok(())
    }
}

impl Clone for Conv2dTranspose {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            in_c: self.in_c,
            out_c: self.out_c,
            kernel: self.kernel,
            stride: self.stride,
            padding: self.padding,
            output_padding: self.output_padding,
            last_w_idx: None,
            last_b_idx: None,
            name: self.name.clone(),
            cached_h: self.cached_h,
            cached_w: self.cached_w,
            cached_batch: self.cached_batch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};

    #[test]
    fn try_new_validates_zero_channels() {
        let mut rng = PcgEngine::new(1);
        let r = Conv2dTranspose::try_new(
            0,
            4,
            3,
            1,
            Padding::Valid,
            0,
            &KaimingNormal,
            Some(&Zeros),
            &mut rng,
        );
        assert!(r.is_err());
    }

    #[test]
    fn try_new_validates_zero_kernel() {
        let mut rng = PcgEngine::new(1);
        let r = Conv2dTranspose::try_new(
            1,
            4,
            0,
            1,
            Padding::Valid,
            0,
            &KaimingNormal,
            Some(&Zeros),
            &mut rng,
        );
        assert!(r.is_err());
    }

    #[test]
    fn forward_shape_valid_no_stride() {
        let mut rng = PcgEngine::new(1);
        let mut conv = Conv2dTranspose::new(
            1,
            2,
            3,
            1,
            Padding::Valid,
            0,
            &KaimingNormal,
            Some(&Zeros),
            &mut rng,
        )
        .input_dims(5, 5);
        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(1, 25));
        let y = conv.forward(&tape, x);
        // h_out = (5-1)*1 + 3 - 0 + 0 = 7
        // out = (1, 2 * 7 * 7) = (1, 98)
        assert_eq!(y.shape(), (1, 98));
    }

    #[test]
    fn forward_shape_stride2() {
        let mut rng = PcgEngine::new(1);
        let mut conv = Conv2dTranspose::new(
            1,
            1,
            3,
            2,
            Padding::Valid,
            0,
            &KaimingNormal,
            Some(&Zeros),
            &mut rng,
        )
        .input_dims(4, 4);
        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(1, 16));
        let y = conv.forward(&tape, x);
        // h_out = (4-1)*2 + 3 - 0 + 0 = 9
        assert_eq!(y.shape(), (1, 81));
    }

    #[test]
    fn forward_with_output_padding() {
        let mut rng = PcgEngine::new(1);
        let mut conv = Conv2dTranspose::new(
            1,
            1,
            3,
            2,
            Padding::Same,
            1,
            &KaimingNormal,
            Some(&Zeros),
            &mut rng,
        )
        .input_dims(4, 4);
        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(1, 16));
        let y = conv.forward(&tape, x);
        // pad = 1 for Same with kernel 3
        // h_out = (4-1)*2 + 3 - 2*1 + 1 = 8
        assert_eq!(y.shape(), (1, 64));
    }

    #[test]
    fn clone_preserves_params() {
        let mut rng = PcgEngine::new(1);
        let conv1 = Conv2dTranspose::new(
            1,
            4,
            3,
            1,
            Padding::Same,
            0,
            &KaimingNormal,
            Some(&Zeros),
            &mut rng,
        );
        let conv2 = conv1.clone();
        assert_eq!(conv2.in_c, conv1.in_c);
        assert_eq!(conv2.out_c, conv1.out_c);
        assert_eq!(conv2.last_w_idx, None);
    }

    #[test]
    fn gradient_flows_backward() {
        let mut rng = PcgEngine::new(1);
        let mut conv = Conv2dTranspose::new(
            1,
            2,
            3,
            1,
            Padding::Valid,
            0,
            &KaimingNormal,
            Some(&Zeros),
            &mut rng,
        )
        .input_dims(3, 3);
        let tape = Tape::new();
        let x = tape.input(Tensor::ones(1, 9));
        let y = conv.forward(&tape, x);
        let loss = y.sum();
        loss.backward();
        let dx = tape.grad(0);
        assert_eq!(dx.shape(), (1, 9));
        // Simple check: gradients are non-zero
        let norm = dx.data.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0);
    }
}
