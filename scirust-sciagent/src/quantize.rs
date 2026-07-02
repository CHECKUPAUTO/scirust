use std::fs;
use std::path::Path;

use scirust_core::autodiff::reverse::Tensor;
use scirust_core::nn::elastic_kv_cache::{
    dequantize_int4_grouped, quantize_int4_grouped,
};

use crate::model::SciAgentModel;

/// INT4 quantized version of a SciAgentModel layer weight.
#[derive(Clone, Debug)]
pub struct QuantizedWeight {
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
    pub group_size: usize,
    pub shape: (usize, usize),
}

impl QuantizedWeight {
    pub fn quantize(w: &Tensor, group_size: usize) -> Self {
        let gs = group_size.clamp(1, w.data.len().max(1));
        let (codes, scales) = quantize_int4_grouped(&w.data, gs);
        Self {
            codes,
            scales,
            group_size: gs,
            shape: (w.rows, w.cols),
        }
    }

    pub fn dequantize(&self) -> Tensor {
        let data = dequantize_int4_grouped(&self.codes, &self.scales, self.group_size);
        Tensor::from_vec(data, self.shape.0, self.shape.1)
    }

    pub fn compressed_bytes(&self) -> usize {
        let nibbles = self.codes.len();
        let scale_bytes = self.scales.len() * 4;
        nibbles.div_ceil(2) + scale_bytes
    }

    pub fn compression_ratio(&self) -> f32 {
        let original = self.shape.0 * self.shape.1 * 4;
        original as f32 / self.compressed_bytes() as f32
    }
}

/// A fully INT4-quantized model.
pub struct QuantizedSciAgent {
    pub embed_weight: QuantizedWeight,
    pub layers: Vec<QuantizedBlock>,
    pub final_norm_weight: QuantizedWeight,
    pub lm_head_weight: Option<QuantizedWeight>,
}

pub struct QuantizedBlock {
    pub attn_q: QuantizedWeight,
    pub attn_k: QuantizedWeight,
    pub attn_v: QuantizedWeight,
    pub attn_o: QuantizedWeight,
    pub ffn_gate: QuantizedWeight,
    pub ffn_up: QuantizedWeight,
    pub ffn_down: QuantizedWeight,
    pub rms_attn: QuantizedWeight,
    pub rms_ffn: QuantizedWeight,
}

impl QuantizedSciAgent {
    pub fn from_model(model: &SciAgentModel, group_size: usize) -> Self {
        let sd = model.state_dict();
        let g = group_size;

        let layers: Vec<QuantizedBlock> = (0..model.config.n_layers)
            .map(|i| {
                let prefix = format!("sciagent.layer{i}");
                QuantizedBlock {
                    attn_q: QuantizedWeight::quantize(
                        &sd[&format!("{prefix}.attn.wq.weight")], g,
                    ),
                    attn_k: QuantizedWeight::quantize(
                        &sd[&format!("{prefix}.attn.wk.weight")], g,
                    ),
                    attn_v: QuantizedWeight::quantize(
                        &sd[&format!("{prefix}.attn.wv.weight")], g,
                    ),
                    attn_o: QuantizedWeight::quantize(
                        &sd[&format!("{prefix}.attn.wo.weight")], g,
                    ),
                    ffn_gate: QuantizedWeight::quantize(
                        &sd[&format!("{prefix}.ffn.gate.weight")], g,
                    ),
                    ffn_up: QuantizedWeight::quantize(
                        &sd[&format!("{prefix}.ffn.up.weight")], g,
                    ),
                    ffn_down: QuantizedWeight::quantize(
                        &sd[&format!("{prefix}.ffn.down.weight")], g,
                    ),
                    rms_attn: QuantizedWeight::quantize(
                        &sd[&format!("{prefix}.rms_attn/weight")], g,
                    ),
                    rms_ffn: QuantizedWeight::quantize(
                        &sd[&format!("{prefix}.rms_ffn/weight")], g,
                    ),
                }
            })
            .collect();

        let embed_weight = QuantizedWeight::quantize(&sd["sciagent.embed.weight"], g);
        let final_norm_weight = QuantizedWeight::quantize(&sd["sciagent.rms_final/weight"], g);
        let lm_head_weight = sd.get("weight")
            .map(|w| QuantizedWeight::quantize(w, g));

        Self {
            embed_weight,
            layers,
            final_norm_weight,
            lm_head_weight,
        }
    }

    pub fn save_bin(&self, path: &Path) -> std::io::Result<()> {
        let mut buf = Vec::new();
        self.serialize(&mut buf);
        fs::write(path, buf)
    }

    fn serialize(&self, buf: &mut Vec<u8>) {
        self.embed_weight.serialize(buf);
        buf.extend(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers {
            layer.serialize(buf);
        }
        self.final_norm_weight.serialize(buf);
        buf.push(self.lm_head_weight.is_some() as u8);
        if let Some(ref w) = self.lm_head_weight {
            w.serialize(buf);
        }
    }

    pub fn total_compressed_bytes(&self) -> usize {
        let mut total = self.embed_weight.compressed_bytes();
        for layer in &self.layers {
            total += layer.attn_q.compressed_bytes();
            total += layer.attn_k.compressed_bytes();
            total += layer.attn_v.compressed_bytes();
            total += layer.attn_o.compressed_bytes();
            total += layer.ffn_gate.compressed_bytes();
            total += layer.ffn_up.compressed_bytes();
            total += layer.ffn_down.compressed_bytes();
            total += layer.rms_attn.compressed_bytes();
            total += layer.rms_ffn.compressed_bytes();
        }
        total += self.final_norm_weight.compressed_bytes();
        if let Some(ref w) = self.lm_head_weight {
            total += w.compressed_bytes();
        }
        total
    }

    pub fn compression_ratio(&self) -> f32 {
        let original = self.estimate_original_bytes();
        original as f32 / self.total_compressed_bytes() as f32
    }

    pub fn estimate_original_bytes(&self) -> usize {
        let param_count = self.embed_weight.shape.0 * self.embed_weight.shape.1
            + self.final_norm_weight.shape.0 * self.final_norm_weight.shape.1;
        let per_layer = self.layers.iter().map(|l| {
            l.attn_q.shape.0 * l.attn_q.shape.1
                + l.attn_k.shape.0 * l.attn_k.shape.1
                + l.attn_v.shape.0 * l.attn_v.shape.1
                + l.attn_o.shape.0 * l.attn_o.shape.1
                + l.ffn_gate.shape.0 * l.ffn_gate.shape.1
                + l.ffn_up.shape.0 * l.ffn_up.shape.1
                + l.ffn_down.shape.0 * l.ffn_down.shape.1
                + l.rms_attn.shape.0 * l.rms_attn.shape.1
                + l.rms_ffn.shape.0 * l.rms_ffn.shape.1
        }).sum::<usize>();
        let lm_head = self.lm_head_weight.as_ref()
            .map(|w| w.shape.0 * w.shape.1).unwrap_or(0);
        (param_count + per_layer + lm_head) * 4
    }
}

impl QuantizedWeight {
    fn serialize(&self, buf: &mut Vec<u8>) {
        buf.extend(&(self.shape.0 as u32).to_le_bytes());
        buf.extend(&(self.shape.1 as u32).to_le_bytes());
        buf.extend(&(self.group_size as u32).to_le_bytes());
        buf.extend(&(self.codes.len() as u32).to_le_bytes());
        for &c in &self.codes {
            buf.push(c as u8);
        }
        buf.extend(&(self.scales.len() as u32).to_le_bytes());
        for &s in &self.scales {
            buf.extend(&s.to_le_bytes());
        }
    }
}

impl QuantizedBlock {
    fn serialize(&self, buf: &mut Vec<u8>) {
        self.attn_q.serialize(buf);
        self.attn_k.serialize(buf);
        self.attn_v.serialize(buf);
        self.attn_o.serialize(buf);
        self.ffn_gate.serialize(buf);
        self.ffn_up.serialize(buf);
        self.ffn_down.serialize(buf);
        self.rms_attn.serialize(buf);
        self.rms_ffn.serialize(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SciAgentConfig;
    use crate::model::SciAgentModel;

    #[test]
    fn test_quantize_roundtrip() {
        let cfg = SciAgentConfig::debug();
        let model = SciAgentModel::new(&cfg);
        let quantized = QuantizedSciAgent::from_model(&model, 32);

        assert!(quantized.compression_ratio() > 2.0,
            "INT4 should compress at least 2x, got {:.1}x", quantized.compression_ratio());
    }

    #[test]
    fn test_quantize_weight_roundtrip() {
        let data = Tensor::from_vec(
            (0..64).map(|i| ((i % 14) - 7) as f32).collect(), 8, 8,
        );
        let qw = QuantizedWeight::quantize(&data, 4);
        let reconstructed = qw.dequantize();

        for i in 0..data.data.len() {
            let diff = (data.data[i] - reconstructed.data[i]).abs();
            assert!(diff < 2.0, "INT4 roundtrip error too large at {i}: {diff}");
        }
    }

    #[test]
    fn test_quantize_save_bin() {
        let cfg = SciAgentConfig::debug();
        let model = SciAgentModel::new(&cfg);
        let quantized = QuantizedSciAgent::from_model(&model, 32);

        let path = Path::new("/tmp/sciagent_q4_test.bin");
        quantized.save_bin(path).unwrap();
        assert!(path.exists());
        let _ = fs::remove_file(path);
    }
}
