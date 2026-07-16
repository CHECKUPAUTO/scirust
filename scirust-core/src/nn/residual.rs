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
use crate::error::Result;
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
        self.try_forward(tape, input).unwrap()
    }

    fn try_forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Result<Var<'t>> {
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
            // Invariant de construction : shortcut_conv et shortcut_bn sont
            // toujours appariés (Some ensemble), l'unwrap est donc sûr.
            self.shortcut_bn.as_mut().unwrap().forward(tape, s)
        }
        else
        {
            input
        };

        // Residual + ReLU final
        let out = h2.try_add(shortcut)?;
        Ok(out.relu())
    }

    fn train(&mut self, on: bool) {
        // Propage à TOUS les enfants (convs incluses, défensivement — un futur
        // Conv2d à état de mode serait couvert sans retoucher ce composite).
        self.conv1.train(on);
        self.bn1.train(on);
        self.conv2.train(on);
        self.bn2.train(on);
        if let Some(ref mut sc) = self.shortcut_conv
        {
            sc.train(on);
        }
        if let Some(ref mut sb) = self.shortcut_bn
        {
            sb.train(on);
        }
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
        // Each child gets a DISTINCT component tag ("conv1"/"conv2"/…). Without
        // it, conv1 and conv2 share the same internal Conv2d name whenever
        // in_c == out_c, and one silently overwrites the other in the map.
        let mut map = HashMap::new();
        let p = &self.name;
        let extend =
            |map: &mut HashMap<String, Tensor>, comp: &str, child: HashMap<String, Tensor>| {
                for (k, v) in child
                {
                    map.insert(format!("{p}.{comp}.{k}"), v);
                }
            };
        extend(&mut map, "conv1", self.conv1.state_dict());
        extend(&mut map, "bn1", self.bn1.state_dict());
        extend(&mut map, "conv2", self.conv2.state_dict());
        extend(&mut map, "bn2", self.bn2.state_dict());
        if let Some(ref sc) = self.shortcut_conv
        {
            extend(&mut map, "shortcut_conv", sc.state_dict());
        }
        if let Some(ref sb) = self.shortcut_bn
        {
            extend(&mut map, "shortcut_bn", sb.state_dict());
        }
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        // Mirror state_dict: strip the "{name}.{component}." prefix and hand each
        // child the sub-dict keyed by its OWN keys (matching what it emitted).
        let p = self.name.clone();
        let sub = |comp: &str| -> HashMap<String, Tensor> {
            let prefix = format!("{p}.{comp}.");
            sd.iter()
                .filter_map(|(k, v)| {
                    k.strip_prefix(prefix.as_str())
                        .map(|rest| (rest.to_string(), v.clone()))
                })
                .collect()
        };
        self.conv1.load_state_dict(&sub("conv1"))?;
        self.bn1.load_state_dict(&sub("bn1"))?;
        self.conv2.load_state_dict(&sub("conv2"))?;
        self.bn2.load_state_dict(&sub("bn2"))?;
        if let Some(ref mut sc) = self.shortcut_conv
        {
            sc.load_state_dict(&sub("shortcut_conv"))?;
        }
        if let Some(ref mut sb) = self.shortcut_bn
        {
            sb.load_state_dict(&sub("shortcut_bn"))?;
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
    fn train_false_propagates_to_child_batch_norms() {
        let mut rng = PcgEngine::new(0);
        // in_c != out_c → le shortcut Conv+BN existe aussi.
        let mut block = ResidualBlock::new(1, 2, 2, &KaimingNormal, &Zeros, &mut rng);
        assert!(block.bn1.training && block.bn2.training);

        block.train(false);
        assert!(!block.bn1.training, "bn1 doit passer en eval");
        assert!(!block.bn2.training, "bn2 doit passer en eval");
        assert!(
            !block.shortcut_bn.as_ref().unwrap().training,
            "shortcut_bn doit passer en eval"
        );

        block.train(true);
        assert!(block.bn1.training && block.bn2.training);
        assert!(block.shortcut_bn.as_ref().unwrap().training);
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

    // Regression for the two checkpointing bugs: (1) load previously always
    // returned Err because state_dict prefixed keys the children looked up
    // unprefixed; (2) with in_c == out_c, conv1 and conv2 shared the same
    // internal Conv2d name, so one overwrote the other in the dict. This uses
    // in_c == out_c == 1 (the collision case) with DISTINCT conv1/conv2 weights
    // and asserts a full save -> load -> forward round trip.
    #[test]
    fn resblock_state_dict_round_trip_with_channel_collision() {
        let mut rng = PcgEngine::new(7);
        let mut a = ResidualBlock::new(1, 1, 1, &KaimingNormal, &Zeros, &mut rng);
        // Distinct, non-trivial conv weights so a collision would be observable.
        a.conv1.weight = Tensor::from_vec((0..9).map(|i| i as f32 * 0.1).collect(), 1, 9);
        a.conv2.weight = Tensor::from_vec((0..9).map(|i| -(i as f32) * 0.2 - 1.0).collect(), 1, 9);
        a.bn1.gamma = Tensor::from_vec(vec![1.3], 1, 1);
        a.bn2.beta = Tensor::from_vec(vec![0.4], 1, 1);

        let sd = a.state_dict();
        let mut b = ResidualBlock::new(1, 1, 1, &KaimingNormal, &Zeros, &mut rng);
        b.load_state_dict(&sd)
            .expect("load_state_dict must succeed");

        // Both conv weights must survive the round trip (this is where the
        // name collision silently dropped conv1).
        assert_eq!(
            b.conv1.weight.data, a.conv1.weight.data,
            "conv1 weights lost"
        );
        assert_eq!(
            b.conv2.weight.data, a.conv2.weight.data,
            "conv2 weights lost"
        );
        assert_eq!(b.bn1.gamma.data, a.bn1.gamma.data, "bn1 gamma lost");
        assert_eq!(b.bn2.beta.data, a.bn2.beta.data, "bn2 beta lost");

        // And the forward outputs must agree.
        let x = vec![0.5f32, -1.0, 2.0, 3.0, -0.5, 1.5, 0.2, -2.0, 1.0];
        let fwd = |blk: &mut ResidualBlock| {
            let tape = Tape::new();
            let xi = tape.input(Tensor::from_vec(x.clone(), 1, 9));
            tape.value(blk.forward(&tape, xi).idx()).data
        };
        assert_eq!(fwd(&mut a), fwd(&mut b), "forward differs after round trip");
    }
}
