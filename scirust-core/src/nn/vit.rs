use crate::autodiff::reverse::{Tape, Var};
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
    pub fn new<W: Initializer, B: Initializer>(
        in_channels: usize,
        embed_dim: usize,
        patch_size: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let projection = crate::nn::conv2d::Conv2d::new(
            in_channels, embed_dim, patch_size, patch_size,
            crate::nn::conv_utils::Padding::Valid, w_init, Some(b_init), rng
        );
        Self { projection }
    }
}

impl ViT {
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
        let patch_embed = PatchEmbedding::new(in_channels, d_model, patch_size, w_init, b_init, rng);
        let encoder = TransformerEncoder::new(
            n_layers, d_model, n_heads, d_ff, false, w_init, b_init, rng
        );
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
        // 1. Patch projection: (batch, channels*h*w) -> (batch, embed_dim * h' * w')
        let patches = self.patch_embed.projection.forward(tape, input);
        let (batch, total_feat) = patches.shape();
        let seq_len = total_feat / self.d_model;

        // Reshape to (batch * seq_len, d_model) for transformer compatibility
        let patches_reshaped = patches.reshape(&[batch * seq_len, self.d_model]);

        // 2. Transformer Encoder
        use crate::tensor::tensor3d::Var3D;
        let x_3d = Var3D::from_var(patches_reshaped, batch, seq_len, self.d_model);
        let encoded = self.encoder.forward_3d(tape, x_3d);

        // 3. Global Mean Pooling across sequence dimension
        // encoded.var has shape (batch * seq_len, d_model)
        // We aggregate to (batch, d_model)
        let mut pooled_batch = Vec::with_capacity(batch);
        for b in 0..batch {
            let sample_seq = encoded.var.slice_rows(b * seq_len, seq_len);
            // Mean of all tokens in the sequence for this sample
            let pooled_sample = sample_seq.mean_axis(0); // Assuming mean_axis(0) averages rows
            pooled_batch.push(pooled_sample);
        }

        use crate::autodiff::reverse::concat_rows;
        let pooled = concat_rows(tape, &pooled_batch);

        // 4. Classification head
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
