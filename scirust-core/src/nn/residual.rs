// scirust-core/src/nn/residual.rs
//
// ResidualBlock — bloc ResNet de base (deux conv 3×3 + shortcut).
//
// Architecture :
//   main = Conv2d → BN → ReLU → Conv2d → BN
//   out  = ReLU(main + shortcut)
//
// Shortcut :
//   - identity si in_c == out_c et stride == 1
//   - sinon Conv2d 1×1 + BN

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::batch_norm_2d::BatchNorm2d;
use crate::nn::conv_utils::Padding;
use crate::nn::conv2d::Conv2d;
use crate::nn::init::Initializer;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use std::collections::HashMap;

pub struct ResidualBlock {
    pub in_c: usize,
    pub out_c: usize,
    pub stride: usize,
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub conv2: Conv2d,
    pub bn2: BatchNorm2d,
    pub shortcut_conv: Option<Conv2d>,
    pub shortcut_bn: Option<BatchNorm2d>,
    pub name: String,
}

impl ResidualBlock {
    pub fn new<W: Initializer, B: Initializer>(
        in_c: usize,
        out_c: usize,
        stride: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let conv1 = Conv2d::new(
            in_c,
            out_c,
            3,
            stride,
            Padding::Same,
            w_init,
            Some(b_init),
            rng,
        );
        let bn1 = BatchNorm2d::new(out_c).with_name("rb_bn1");
        let conv2 = Conv2d::new(out_c, out_c, 3, 1, Padding::Same, w_init, Some(b_init), rng);
        let bn2 = BatchNorm2d::new(out_c).with_name("rb_bn2");

        let (shortcut_conv, shortcut_bn) = if in_c != out_c || stride != 1
        {
            let sc = Conv2d::new(
                in_c,
                out_c,
                1,
                stride,
                Padding::Valid,
                w_init,
                Some(b_init),
                rng,
            );
            let sb = BatchNorm2d::new(out_c).with_name("rb_sc_bn");
            (Some(sc), Some(sb))
        }
        else
        {
            (None, None)
        };

        Self {
            in_c,
            out_c,
            stride,
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut_conv,
            shortcut_bn,
            name: format!("resblock_{in_c}_{out_c}_s{stride}"),
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    #[must_use]
    pub fn input_dims(mut self, h: usize, w: usize) -> Self {
        self.conv1 = self.conv1.input_dims(h, w);
        // conv2 reçoit la sortie de conv1 ; on laisse conv2 auto-inférer
        // sauf si stride=1 (mêmes dims)
        if self.stride == 1
        {
            self.conv2 = self.conv2.input_dims(h, w);
        }
        if let Some(ref mut sc) = self.shortcut_conv
        {
            self.shortcut_conv = Some(sc.clone().input_dims(h, w));
        }
        self
    }
}

impl Module for ResidualBlock {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        // Chemin principal
        let h1 = self.conv1.forward(tape, input);
        let h1 = self.bn1.forward(tape, h1);
        let h1 = h1.relu();
        let h2 = self.conv2.forward(tape, h1);
        let h2 = self.bn2.forward(tape, h2);

        // Shortcut
        let shortcut = if let Some(ref mut sc) = self.shortcut_conv
        {
            let s = sc.forward(tape, input);
            self.shortcut_bn.as_mut().unwrap().forward(tape, s)
        }
        else
        {
            input
        };

        // Residual + ReLU final
        let out = h2.try_add(shortcut).unwrap();
        out.relu()
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.conv1.parameter_indices());
        v.extend(self.bn1.parameter_indices());
        v.extend(self.conv2.parameter_indices());
        v.extend(self.bn2.parameter_indices());
        if let Some(ref sc) = self.shortcut_conv
        {
            v.extend(sc.parameter_indices());
        }
        if let Some(ref sb) = self.shortcut_bn
        {
            v.extend(sb.parameter_indices());
        }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        self.conv1.sync(tape);
        self.bn1.sync(tape);
        self.conv2.sync(tape);
        self.bn2.sync(tape);
        if let Some(ref mut sc) = self.shortcut_conv
        {
            sc.sync(tape);
        }
        if let Some(ref mut sb) = self.shortcut_bn
        {
            sb.sync(tape);
        }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        let _p = &self.name;
        for (k, v) in self.conv1.state_dict()
        {
            map.insert(format!("{_p}.{k}"), v);
        }
        for (k, v) in self.bn1.state_dict()
        {
            map.insert(format!("{_p}.{k}"), v);
        }
        for (k, v) in self.conv2.state_dict()
        {
            map.insert(format!("{_p}.{k}"), v);
        }
        for (k, v) in self.bn2.state_dict()
        {
            map.insert(format!("{_p}.{k}"), v);
        }
        if let Some(ref sc) = self.shortcut_conv
        {
            for (k, v) in sc.state_dict()
            {
                map.insert(format!("{_p}.shortcut.{k}"), v);
            }
        }
        if let Some(ref sb) = self.shortcut_bn
        {
            for (k, v) in sb.state_dict()
            {
                map.insert(format!("{_p}.shortcut_bn.{k}"), v);
            }
        }
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        self.conv1.load_state_dict(sd)?;
        self.bn1.load_state_dict(sd)?;
        self.conv2.load_state_dict(sd)?;
        self.bn2.load_state_dict(sd)?;
        if let Some(ref mut sc) = self.shortcut_conv
        {
            sc.load_state_dict(sd)?;
        }
        if let Some(ref mut sb) = self.shortcut_bn
        {
            sb.load_state_dict(sd)?;
        }
        Ok(())
    }
}

impl Clone for ResidualBlock {
    fn clone(&self) -> Self {
        Self {
            in_c: self.in_c,
            out_c: self.out_c,
            stride: self.stride,
            conv1: self.conv1.clone(),
            bn1: self.bn1.clone(),
            conv2: self.conv2.clone(),
            bn2: self.bn2.clone(),
            shortcut_conv: self.shortcut_conv.clone(),
            shortcut_bn: self.shortcut_bn.clone(),
            name: self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};

    #[test]
    fn resblock_identity_passthrough() {
        // stride=1, in_c=out_c → shortcut identity
        let mut rng = PcgEngine::new(0);
        let mut block = ResidualBlock::new(1, 1, 1, &KaimingNormal, &Zeros, &mut rng);
        // Zéro les poids pour que main path = 0 → out = ReLU(x)
        block.conv1.weight = Tensor::zeros(1, 9);
        block.conv2.weight = Tensor::zeros(1, 9);
        // Pas de shortcut → out = ReLU(x)

        let tape = Tape::new();
        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x = tape.input(Tensor::from_vec(x_data.clone(), 1, 9));
        let y = block.forward(&tape, x);
        let yt = tape.value(y.idx());
        // ReLU preserve les valeurs positives
        assert_eq!(yt.data, x_data);
    }

    #[test]
    fn resblock_stride2_changes_shape() {
        let mut rng = PcgEngine::new(0);
        let mut block =
            ResidualBlock::new(1, 2, 2, &KaimingNormal, &Zeros, &mut rng).input_dims(4, 4);
        let tape = Tape::new();
        // (batch=1, 1*4*4=16)
        let x = tape.input(Tensor::from_vec(vec![1.0; 16], 1, 16));
        let y = block.forward(&tape, x);
        // stride=2, Same padding : h_out = w_out = 4/2 = 2
        // out_c=2 → total_features = 2*2*2 = 8
        assert_eq!(y.shape(), (1, 8));
    }

    #[test]
    fn resblock_gradient_flows() {
        let mut rng = PcgEngine::new(0);
        let mut block = ResidualBlock::new(1, 1, 1, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 9], 1, 9));
        let y = block.forward(&tape, x);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x.idx());
        let max_abs: f32 = g.data.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-6, "gradient is zero in ResidualBlock");
    }

    #[test]
    fn resblock_state_dict_contains_all_keys() {
        let mut rng = PcgEngine::new(0);
        let block = ResidualBlock::new(2, 4, 2, &KaimingNormal, &Zeros, &mut rng);
        let sd = block.state_dict();
        // conv1, bn1, conv2, bn2, shortcut_conv, shortcut_bn
        assert!(
            sd.keys().any(|k| k.contains("conv2d")),
            "state_dict missing conv keys: {:?}",
            sd.keys()
        );
        assert!(
            sd.keys().any(|k| k.contains("rb_bn")),
            "state_dict missing bn keys: {:?}",
            sd.keys()
        );
    }
}
