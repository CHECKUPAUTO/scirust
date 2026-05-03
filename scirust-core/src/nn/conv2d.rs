// scirust-core/src/nn/conv2d.rs
//
// Conv2d CPU — implémentation via conv2d_forward sur la tape autograd.

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;
use crate::nn::init::Initializer;
use crate::nn::rng::PcgEngine;
use crate::nn::conv_utils::{ConvConfig, Padding};
use crate::error::{Result, SciRustError};

pub struct Conv2d {
    pub weight:    Tensor,
    pub bias:      Option<Tensor>,
    pub in_c:      usize,
    pub out_c:     usize,
    pub kernel:    usize,
    pub stride:    usize,
    pub padding:   Padding,
    last_w_idx:    Option<usize>,
    last_b_idx:    Option<usize>,
    pub name:      String,
    cached_h:      Option<usize>,
    cached_w:      Option<usize>,
    cached_batch:  Option<usize>,
}

impl Conv2d {
    pub fn try_new<W: Initializer, B: Initializer>(
        in_c:        usize,
        out_c:       usize,
        kernel:      usize,
        stride:      usize,
        padding:     Padding,
        weight_init: &W,
        bias_init:   Option<&B>,
        rng:         &mut PcgEngine,
    ) -> Result<Self> {
        if in_c == 0 || out_c == 0 {
            return Err(SciRustError::InvalidConfig(
                format!("Conv2d: in_c={in_c} out_c={out_c}, doivent être > 0")));
        }
        if kernel == 0 || stride == 0 {
            return Err(SciRustError::InvalidConfig(
                format!("Conv2d: kernel={kernel} stride={stride}, doivent être > 0")));
        }

        let kk = kernel * kernel;
        let mut weight = Tensor::zeros(out_c, in_c * kk);
        weight_init.fill(&mut weight, in_c * kk, out_c, rng);

        let bias = bias_init.map(|init| {
            let mut b = Tensor::zeros(1, out_c);
            init.fill(&mut b, 1, out_c, rng);
            b
        });

        Ok(Self {
            weight, bias, in_c, out_c, kernel, stride, padding,
            last_w_idx: None, last_b_idx: None,
            name: format!("conv2d_{in_c}_{out_c}_{kernel}"),
            cached_h: None, cached_w: None, cached_batch: None,
        })
    }

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
        Self::try_new(in_c, out_c, kernel, stride, padding,
                      weight_init, bias_init, rng)
            .expect("Conv2d::new failed — utilise try_new pour gérer l'erreur")
    }

    pub fn with_name(mut self, name: &str) -> Self { self.name = name.into(); self }

    pub fn input_dims(mut self, h: usize, w: usize) -> Self {
        self.cached_h = Some(h); self.cached_w = Some(w); self
    }
}

impl Module for Conv2d {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (b, total_features) = input.shape();
        let (h, w) = match (self.cached_h, self.cached_w) {
            (Some(h), Some(w)) => (h, w),
            _ => {
                let per_channel = total_features / self.in_c;
                let side = (per_channel as f64).sqrt() as usize;
                assert_eq!(side * side, per_channel,
                    "Conv2d: utiliser .input_dims(h, w) pour des images non carrées");
                (side, side)
            }
        };
        self.cached_h = Some(h); self.cached_w = Some(w); self.cached_batch = Some(b);

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

        input.conv2d_forward(weight_v, bias_v, b, self.in_c, h, w, self.out_c, self.kernel, self.stride, cfg.pad())
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_w_idx { v.push(i); }
        if let Some(i) = self.last_b_idx { v.push(i); }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_idx { self.weight = tape.value(i); }
        if let Some(i) = self.last_b_idx { self.bias = Some(tape.value(i)); }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.weight", self.name), self.weight.clone());
        if let Some(b) = &self.bias {
            map.insert(format!("{}.bias", self.name), b.clone());
        }
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> std::result::Result<(), String> {
        let w = sd.get(&format!("{}.weight", self.name))
            .ok_or_else(|| format!("missing key: {}.weight", self.name))?;
        let kk = self.kernel * self.kernel;
        if w.shape() != (self.out_c, self.in_c * kk) {
            return Err(format!("weight shape mismatch: expected {:?}, got {:?}",
                               (self.out_c, self.in_c * kk), w.shape()));
        }
        self.weight = w.clone();
        if let Some(b) = sd.get(&format!("{}.bias", self.name)) {
            if b.shape() != (1, self.out_c) {
                return Err(format!("bias shape mismatch"));
            }
            self.bias = Some(b.clone());
        }
        Ok(())
    }
}

impl Clone for Conv2d {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias:   self.bias.clone(),
            in_c: self.in_c, out_c: self.out_c,
            kernel: self.kernel, stride: self.stride,
            padding: self.padding,
            last_w_idx: None, last_b_idx: None,
            name: self.name.clone(),
            cached_h: self.cached_h, cached_w: self.cached_w,
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
        let r = Conv2d::try_new(0, 4, 3, 1, Padding::Same,
            &KaimingNormal, Some(&Zeros), &mut rng);
        assert!(matches!(r, Err(SciRustError::InvalidConfig(_))));
    }

    #[test]
    fn try_new_validates_zero_kernel() {
        let mut rng = PcgEngine::new(1);
        let r = Conv2d::try_new(1, 4, 0, 1, Padding::Same,
            &KaimingNormal, Some(&Zeros), &mut rng);
        assert!(matches!(r, Err(SciRustError::InvalidConfig(_))));
    }

    #[test]
    fn forward_cpu_works() {
        let mut rng = PcgEngine::new(1);
        let mut conv = Conv2d::new(1, 2, 3, 1, Padding::Valid,
            &KaimingNormal, Some(&Zeros), &mut rng).input_dims(5, 5);
        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(1, 25));
        let y = conv.forward(&tape, x);
        assert_eq!(y.shape(), (1, 18));   // (5-3+1)² × 2 = 18
    }

    #[test]
    fn clone_preserves_params() {
        let mut rng = PcgEngine::new(1);
        let conv1 = Conv2d::new(1, 4, 3, 1, Padding::Same,
            &KaimingNormal, Some(&Zeros), &mut rng);
        let conv2 = conv1.clone();
        assert_eq!(conv2.in_c, conv1.in_c);
        assert_eq!(conv2.out_c, conv1.out_c);
        assert_eq!(conv2.last_w_idx, None);
    }
}
