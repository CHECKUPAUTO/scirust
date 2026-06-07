use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::batch_norm_2d::BatchNorm2d;
use crate::nn::conv_utils::Padding;
use crate::nn::conv2d::Conv2d;
use crate::nn::init::Initializer;
use crate::nn::module::Module;
use crate::nn::residual::ResidualBlock;
use crate::nn::rng::PcgEngine;
use std::collections::HashMap;

/// ResNet architecture implementation.
pub struct ResNet {
    pub conv1: Conv2d,
    pub bn1: BatchNorm2d,
    pub layers: Vec<Vec<ResidualBlock>>,
    pub fc: crate::nn::linear::Linear,
    pub out_channels: usize,
    pub name: String,
}

impl ResNet {
    pub fn new<W: Initializer, B: Initializer>(
        block_counts: &[usize],
        num_classes: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let conv1 = Conv2d::new(3, 64, 7, 2, Padding::Same, w_init, Some(b_init), rng);
        let bn1 = BatchNorm2d::new(64);

        let mut layers = Vec::new();
        let mut in_c = 64;
        let mut out_c = 64;

        for (i, &count) in block_counts.iter().enumerate() {
            let stride = if i == 0 { 1 } else { 2 };
            let mut layer = Vec::new();
            layer.push(ResidualBlock::new(in_c, out_c, stride, w_init, b_init, rng));
            for _ in 1..count {
                layer.push(ResidualBlock::new(out_c, out_c, 1, w_init, b_init, rng));
            }
            layers.push(layer);
            in_c = out_c;
            if i < block_counts.len() - 1 {
                out_c *= 2;
            }
        }

        let fc = crate::nn::linear::Linear::new(in_c, num_classes, w_init, b_init, rng);

        Self {
            conv1,
            bn1,
            layers,
            fc,
            out_channels: in_c,
            name: "resnet".into(),
        }
    }

    pub fn resnet18<W: Initializer, B: Initializer>(num_classes: usize, w_init: &W, b_init: &B, rng: &mut PcgEngine) -> Self {
        Self::new(&[2, 2, 2, 2], num_classes, w_init, b_init, rng)
    }

    pub fn resnet34<W: Initializer, B: Initializer>(num_classes: usize, w_init: &W, b_init: &B, rng: &mut PcgEngine) -> Self {
        Self::new(&[3, 4, 6, 3], num_classes, w_init, b_init, rng)
    }
}

impl Module for ResNet {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let mut x = self.conv1.forward(tape, input);
        x = self.bn1.forward(tape, x);
        x = x.relu();

        for layer in &mut self.layers {
            for block in layer {
                x = block.forward(tape, x);
            }
        }

        // Global Average Pooling (GAP)
        let (batch, total_features) = x.shape();
        let spatial_dim = total_features / self.out_channels;

        let mut gap_parts = Vec::with_capacity(batch);
        for b in 0..batch {
            let sample = x.slice_rows(b, 1);
            // Average across spatial dimension for each channel
            // In flattened format (batch, C*H*W), we need to average groups of size spatial_dim
            let mut channels = Vec::with_capacity(self.out_channels);
            for c in 0..self.out_channels {
                let start = c * spatial_dim;
                let channel_avg = sample.slice_cols(start, spatial_dim).sum().scale(1.0 / spatial_dim as f32);
                channels.push(channel_avg);
            }
            // Workaround: Use concat_rows then transpose if needed,
            // but since we want (1, out_channels), we can just concat_rows if we reshape?
            // Actually concat_rows returns (N, cols). If rows are (1, 1), it returns (N, 1).
            // We want (1, out_channels).
            use crate::autodiff::reverse::concat_rows;
            let channel_vec = concat_rows(tape, &channels); // (out_channels, 1)
            gap_parts.push(channel_vec.transpose_2d()); // (1, out_channels)
        }

        use crate::autodiff::reverse::concat_rows;
        let pooled = concat_rows(tape, &gap_parts);

        self.fc.forward(tape, pooled)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.conv1.parameter_indices());
        v.extend(self.bn1.parameter_indices());
        for layer in &self.layers {
            for block in layer {
                v.extend(block.parameter_indices());
            }
        }
        v.extend(self.fc.parameter_indices());
        v
    }

    fn sync(&mut self, tape: &Tape) {
        self.conv1.sync(tape);
        self.bn1.sync(tape);
        for layer in &mut self.layers {
            for block in layer {
                block.sync(tape);
            }
        }
        self.fc.sync(tape);
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        for (k, v) in self.conv1.state_dict() { map.insert(format!("conv1.{}", k), v); }
        for (k, v) in self.bn1.state_dict() { map.insert(format!("bn1.{}", k), v); }
        for (i, layer) in self.layers.iter().enumerate() {
            for (j, block) in layer.iter().enumerate() {
                for (k, v) in block.state_dict() {
                    map.insert(format!("layer{}.{}.{}", i, j, k), v);
                }
            }
        }
        for (k, v) in self.fc.state_dict() { map.insert(format!("fc.{}", k), v); }
        map
    }
}
