use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::init::Initializer;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use crate::nn::transformer::TransformerEncoder;

/// Vision Transformer (ViT) implementation.
pub struct ViT {
    pub patch_embed: PatchEmbedding,
    pub encoder: TransformerEncoder,
    pub head: Linear,
    pub d_model: usize,
    pub name: String,
}

pub struct PatchEmbedding {
    pub projection: crate::nn::conv2d::Conv2d,
}

impl PatchEmbedding {
    #[allow(clippy::too_many_arguments)]
    pub fn new<W: Initializer, B: Initializer>(
        in_channels: usize,
        embed_dim: usize,
        patch_size: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let projection = crate::nn::conv2d::Conv2d::new(
            in_channels,
            embed_dim,
            patch_size,
            patch_size,
            crate::nn::conv_utils::Padding::Valid,
            w_init,
            Some(b_init),
            rng,
        );
        Self { projection }
    }
}

impl ViT {
    #[allow(clippy::too_many_arguments)]
    pub fn new<W: Initializer, B: Initializer>(
        _img_size: usize,
        patch_size: usize,
        in_channels: usize,
        num_classes: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let patch_embed =
            PatchEmbedding::new(in_channels, d_model, patch_size, w_init, b_init, rng);
        let encoder =
            TransformerEncoder::new(n_layers, d_model, n_heads, d_ff, false, w_init, b_init, rng);
        let head = Linear::new(d_model, num_classes, w_init, b_init, rng);

        Self {
            patch_embed,
            encoder,
            head,
            d_model,
            name: "vit".into(),
        }
    }
}

impl Module for ViT {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        use crate::tensor::tensor3d::Var3D;
        let patches = self.patch_embed.projection.forward(tape, input);
        let (batch, total_feat) = patches.shape();
        let seq_len = total_feat / self.d_model;

        // The encoder expects its 2D var as (batch*seq_len, d_model) (one row per
        // token), so reshape the flat (batch, seq_len*d_model) patch embedding.
        let tokens = patches.reshape(&[batch * seq_len, self.d_model]);
        let x_3d = Var3D::from_var(tokens, batch, seq_len, self.d_model);
        let encoded = self.encoder.forward_3d(tape, x_3d); // var: (batch*seq_len, d_model)

        // Mean-pool over the sequence to (batch, d_model) before the head. This is
        // what makes the classifier actually depend on the image; the previous
        // seq_len>1 path fed a zeros tensor into the head (image-independent, with
        // no gradient to patch_embed/encoder). Pooling is permutation-invariant,
        // so it is robust to the patch flattening order.
        let pooled = if seq_len <= 1
        {
            encoded.var
        }
        else
        {
            // Constant pooling matrix P (batch, batch*seq_len): row b averages the
            // seq_len token rows [b*seq_len, (b+1)*seq_len). P @ encoded = seq mean.
            let mut p = Tensor::zeros(batch, batch * seq_len);
            let inv = 1.0 / seq_len as f32;
            for b in 0..batch
            {
                for s in 0..seq_len
                {
                    p.data[b * (batch * seq_len) + b * seq_len + s] = inv;
                }
            }
            tape.input(p).matmul(encoded.var)
        };
        self.head.forward(tape, pooled)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.patch_embed.projection.parameter_indices());
        v.extend(self.encoder.parameter_indices());
        v.extend(self.head.parameter_indices());
        v
    }

    fn sync(&mut self, tape: &Tape) {
        self.patch_embed.projection.sync(tape);
        self.encoder.sync(tape);
        self.head.sync(tape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};

    #[test]
    fn vit_output_depends_on_image_for_multiple_patches() {
        // 4x4 single-channel image, patch 2 -> 4 patches (seq_len = 4 > 1).
        let mut rng = PcgEngine::new(1);
        let mut vit = ViT::new(4, 2, 1, 3, 8, 2, 1, 16, &KaimingNormal, &Zeros, &mut rng);

        let tape = Tape::new();
        let img_a = tape.input(Tensor::from_vec((0..16).map(|i| i as f32 * 0.1).collect(), 1, 16));
        let out_a = vit.forward(&tape, img_a);
        assert_eq!(tape.value(out_a.idx()).shape(), (1, 3));
        let logits_a = tape.value(out_a.idx()).data.clone();

        let tape2 = Tape::new();
        let img_b =
            tape2.input(Tensor::from_vec((0..16).map(|i| (16 - i) as f32 * 0.1).collect(), 1, 16));
        let out_b = vit.forward(&tape2, img_b);
        let logits_b = tape2.value(out_b.idx()).data.clone();

        // Regression: distinct images must yield distinct logits (pre-fix the
        // multi-patch path fed zeros into the head, so every image was identical).
        let differ = logits_a
            .iter()
            .zip(&logits_b)
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differ, "ViT output must depend on the image content");

        // Gradients must flow (no zeros dead-end): backward does not panic.
        let loss = out_a.sum();
        tape.backward(loss.idx());
    }
}
