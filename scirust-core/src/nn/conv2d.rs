// scirust-core/src/nn/conv2d.rs
//
// Couche Conv2d adossée à Op::Conv2dForward (op composite avec
// recompute d'im2col au backward — pas d'inflation mémoire).
//
// Conventions utilisateur :
//   Input  shape : (B, in_C·H·W) — flat row-major
//   Output shape : (B, out_C·H_out·W_out)
//
// Pour reconstruire des images depuis l'output, l'utilisateur fait :
//   output.view_as(B, out_C, H_out, W_out)  — pas requis pour le forward suivant,
//   mais utile pour la visualisation / debug.

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;
use crate::nn::init::Initializer;
use crate::nn::rng::PcgEngine;
use crate::nn::conv_utils::{ConvConfig, Padding};

pub struct Conv2d {
    pub weight:    Tensor,           // (out_C, in_C·K·K)
    pub bias:      Option<Tensor>,   // (1, out_C) ou None
    pub in_c:      usize,
    pub out_c:     usize,
    pub kernel:    usize,
    pub stride:    usize,
    pub padding:   Padding,
    last_w_idx:    Option<usize>,
    last_b_idx:    Option<usize>,
    pub name:      String,

    // Dimensions de l'input attendu — fixées au premier forward
    // (parce que H et W ne sont pas connus à la construction).
    cached_h:      Option<usize>,
    cached_w:      Option<usize>,
    cached_batch:  Option<usize>,
}

impl Conv2d {
    pub fn new<W: Initializer, B: Initializer>(
        in_c:        usize,
        out_c:       usize,
        kernel:      usize,
        stride:      usize,
        padding:     Padding,
        weight_init: &W,
        bias_init:   Option<&B>,
        rng:         &mut PcgEngine,
    ) -> Self {
        let kk = kernel * kernel;
        let mut weight = Tensor::zeros(out_c, in_c * kk);
        weight_init.fill(&mut weight, rng);

        let bias = bias_init.map(|init| {
            let mut b = Tensor::zeros(1, out_c);
            init.fill(&mut b, rng);
            b
        });

        Self {
            weight, bias,
            in_c, out_c, kernel, stride, padding,
            last_w_idx: None, last_b_idx: None,
            name: format!("conv2d_{in_c}_{out_c}_{kernel}"),
            cached_h: None, cached_w: None, cached_batch: None,
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    /// Configure les dimensions H, W de l'input. À appeler avant le premier
    /// forward, ou laisser inférer depuis input.shape() (cf. forward).
    pub fn input_dims(mut self, h: usize, w: usize) -> Self {
        self.cached_h = Some(h);
        self.cached_w = Some(w);
        self
    }
}

impl Module for Conv2d {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (b, total_features) = input.shape();
        // Si H/W non préconfigurés, on suppose une image carrée et on
        // dérive H=W depuis le total.
        let (h, w) = match (self.cached_h, self.cached_w) {
            (Some(h), Some(w)) => (h, w),
            _ => {
                let per_channel = total_features / self.in_c;
                let side = (per_channel as f64).sqrt() as usize;
                assert_eq!(side * side, per_channel,
                    "Conv2d: impossible d'inférer H=W depuis (B={b}, total={total_features}). \
                     Utilisez .input_dims(h, w) pour préciser.");
                (side, side)
            }
        };
        self.cached_h = Some(h);
        self.cached_w = Some(w);
        self.cached_batch = Some(b);

        let cfg = ConvConfig {
            batch: b, in_c: self.in_c, h, w,
            kernel: self.kernel, stride: self.stride,
            padding: self.padding, out_c: self.out_c,
        };
        cfg.check().expect("ConvConfig invalide");

        let weight_v = tape.input(self.weight.clone());
        let bias_v   = self.bias.as_ref().map(|t| tape.input(t.clone()));
        self.last_w_idx = Some(weight_v.idx());
        self.last_b_idx = bias_v.as_ref().map(|v| v.idx());

        input.conv2d_forward(weight_v, bias_v, cfg)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_w_idx { v.push(i); }
        if let Some(i) = self.last_b_idx { v.push(i); }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_idx { self.weight = tape.value(i); }
        if let Some(i) = self.last_b_idx {
            self.bias = Some(tape.value(i));
        }
    }

    fn state_dict(&self) -> Vec<(String, Tensor)> {
        let mut v = vec![(format!("{}.weight", self.name), self.weight.clone())];
        if let Some(b) = &self.bias {
            v.push((format!("{}.bias", self.name), b.clone()));
        }
        v
    }

    fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>) -> usize {
        let mut loaded = 0;
        if let Some(t) = dict.get(&format!("{}.weight", self.name)) {
            self.weight = t.clone(); loaded += 1;
        }
        if let Some(t) = dict.get(&format!("{}.bias", self.name)) {
            self.bias = Some(t.clone()); loaded += 1;
        }
        loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};

    #[test]
    fn conv2d_output_shape() {
        let mut rng = PcgEngine::new(1);
        let mut conv = Conv2d::new(
            3, 16, 3, 1, Padding::Same,
            &KaimingNormal, Some(&Zeros), &mut rng,
        ).input_dims(8, 8);

        let tape = Tape::new();
        // Batch = 4, 3 canaux, image 8×8 → 192 features
        let x = tape.input(Tensor::zeros(4, 3 * 8 * 8));
        let y = conv.forward(&tape, x);
        // Output : (4, 16 · 8 · 8) avec same padding et stride 1
        assert_eq!(y.shape(), (4, 16 * 8 * 8));
    }

    #[test]
    fn conv2d_valid_padding_shrinks() {
        let mut rng = PcgEngine::new(1);
        let mut conv = Conv2d::new(
            1, 4, 3, 1, Padding::Valid,
            &KaimingNormal, Some(&Zeros), &mut rng,
        ).input_dims(8, 8);

        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(2, 64));
        let y = conv.forward(&tape, x);
        // (8 - 3)/1 + 1 = 6
        assert_eq!(y.shape(), (2, 4 * 6 * 6));
    }

    #[test]
    fn conv2d_parameter_count() {
        let mut rng = PcgEngine::new(1);
        let mut conv = Conv2d::new(
            3, 16, 3, 1, Padding::Same,
            &KaimingNormal, Some(&Zeros), &mut rng,
        ).input_dims(8, 8);

        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(1, 192));
        let _ = conv.forward(&tape, x);
        // weight + bias = 2 paramètres
        assert_eq!(conv.parameter_indices().len(), 2);
    }
}
