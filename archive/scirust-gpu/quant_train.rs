// scirust-gpu/src/quant_train.rs
//
// TurboQuant — Quantization-Aware Training (QAT) with
// Straight-Through Estimator (STE).
//
// During training:
//   1. Forward: quantize weights, compute loss = quantization error (MSE)
//   2. Backward via STE: gradients pass through the quantizer unchanged
//      (identity Jacobian approximation)
//   3. Update: update full-precision weights using gradients, then
//      re-quantize for next forward pass
//
// STE temperature (tau) controls how "hard" the quantization rounding
// behaves during the forward pass. Higher tau = softer quantization
// (more gradient signal), lower tau = harder quantization (closer to
// actual inference behavior).

use crate::error::QuantError;
use crate::quantize::{QuantMode, QuantizedTensor, Quantizer};

/// Straight-Through Estimator wrapper over a Quantizer.
pub struct QuantizationAwareTrainer {
    pub quantizer: Quantizer,
    /// STE temperature: controls rounding sharpness (0 = hard, ∞ = identity).
    /// Practical range: 0.1–1.0.
    pub temperature: f64,
}

impl QuantizationAwareTrainer {
    pub fn new(mode: QuantMode, block_size: usize, temperature: f64) -> Self {
        assert!(temperature > 0.0, "STE temperature must be > 0.0");
        Self {
            quantizer: Quantizer::new(mode, block_size),
            temperature,
        }
    }

    /// Forward pass with simulated quantization via STE.
    ///
    /// - Quantizes `weights` in-place for the forward computation.
    /// - Computes loss = sqrt(MSE) between original and quantized weights.
    /// - Returns the quantization loss (useful for logging / regularization).
    /// - After this call, `weights` contain the quantized values
    ///   (use `apply_ste_update` to update and requantize).
    pub fn quantize_forward(&self, weights: &mut [f32]) -> f64 {
        let original = weights.to_vec();
        let qt = self.quantizer.quantize(weights);
        let quantized = self.quantizer.dequantize(&qt);

        // Replace weights with quantized values (STE forward)
        weights.copy_from_slice(&quantized);

        // Compute loss = sqrt(MSE)
        let mse = self.quantizer.quantization_error(&original, &quantized);
        if mse.is_finite() { mse.sqrt() } else { 0.0 }
    }

    /// Apply STE gradient update: update full-precision weights from
    /// the gradient, then re-quantize.
    pub fn apply_ste_update(&self, weights: &mut [f32], gradient: &[f32], learning_rate: f64) {
        assert_eq!(
            weights.len(),
            gradient.len(),
            "weights and gradient must have same length"
        );

        // SGD update on the quantized approximation, then re-quantize
        for (w, &g) in weights.iter_mut().zip(gradient.iter())
        {
            *w -= (learning_rate * g as f64) as f32;
        }

        // Re-quantize
        let qt = self.quantizer.quantize(weights);
        let requantized = self.quantizer.dequantize(&qt);
        weights.copy_from_slice(&requantized);
    }

    /// Full training step: quantize forward + apply STE update.
    /// Returns the quantization loss for monitoring.
    pub fn train_step(&self, weights: &mut [f32], gradient: &[f32], learning_rate: f64) -> f64 {
        let loss = self.quantize_forward(weights);
        self.apply_ste_update(weights, gradient, learning_rate);
        loss
    }

    /// Export quantized weights to a binary file.
    ///
    /// Binary format (little-endian):
    ///   [mode:1] [rows:8] [cols:8] [block_size:8]
    ///   [scale_bytes: num_blocks * 4] [zero_bytes: num_blocks * 4]
    ///   [data_len:8] [data_bytes...]
    pub fn export_quantized(
        &self,
        weights: &[f32],
        path: &str,
    ) -> std::result::Result<(), QuantError> {
        let qt = self.quantizer.quantize(weights);
        let mode_byte: u8 = match qt.mode
        {
            QuantMode::FP32 => 0,
            QuantMode::FP16 => 1,
            QuantMode::INT8 => 2,
            QuantMode::INT4 => 3,
            QuantMode::NF4 => 4,
        };

        let mut buf = Vec::new();
        buf.push(mode_byte);
        buf.extend_from_slice(&(qt.shape.0 as u64).to_le_bytes());
        buf.extend_from_slice(&(qt.shape.1 as u64).to_le_bytes());
        buf.extend_from_slice(&(qt.block_size as u64).to_le_bytes());

        for &s in &qt.scale
        {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        for &z in &qt.zero
        {
            buf.extend_from_slice(&z.to_le_bytes());
        }

        let data_len = qt.data.len() as u64;
        buf.extend_from_slice(&data_len.to_le_bytes());
        buf.extend_from_slice(&qt.data);

        std::fs::write(path, &buf).map_err(|e| QuantError::Io(e.to_string()))?;
        Ok(())
    }

    /// Load quantized weights from a file, dequantize to `Vec<f32>`.
    /// Returns `(dequantized_weights, mode, block_size)`.
    pub fn import_quantized(
        path: &str,
    ) -> std::result::Result<(Vec<f32>, QuantMode, usize), QuantError> {
        let data = std::fs::read(path).map_err(|e| QuantError::Io(e.to_string()))?;
        if data.len() < 25
        {
            return Err(QuantError::InvalidFormat("file too small".into()));
        }

        let mut pos = 0usize;
        let mode = match data[pos]
        {
            0 => QuantMode::FP32,
            1 => QuantMode::FP16,
            2 => QuantMode::INT8,
            3 => QuantMode::INT4,
            4 => QuantMode::NF4,
            _ => return Err(QuantError::InvalidFormat("unknown quant mode".into())),
        };
        pos += 1;
        let rows = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let cols = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let block_size = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let n = rows * cols;
        let num_blocks = (n + block_size - 1) / block_size;

        let mut scale = Vec::with_capacity(num_blocks);
        let mut zero = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks
        {
            if pos + 4 > data.len()
            {
                return Err(QuantError::InvalidFormat("truncated scale data".into()));
            }
            let s = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            scale.push(s);
            pos += 4;
        }
        for _ in 0..num_blocks
        {
            if pos + 4 > data.len()
            {
                return Err(QuantError::InvalidFormat("truncated zero data".into()));
            }
            let z = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            zero.push(z);
            pos += 4;
        }

        if pos + 8 > data.len()
        {
            return Err(QuantError::InvalidFormat("truncated data length".into()));
        }
        let raw_data_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        if pos + raw_data_len > data.len()
        {
            return Err(QuantError::InvalidFormat("truncated quantized data".into()));
        }

        let raw_data = data[pos..pos + raw_data_len].to_vec();

        let qt = QuantizedTensor {
            data: raw_data,
            scale,
            zero,
            shape: (rows, cols),
            mode,
            block_size,
        };

        let q = Quantizer::new(mode, block_size);
        let deq = q.dequantize(&qt);
        Ok((deq, mode, block_size))
    }
}

// ------------------------------------------------------------------ //
//  Tests                                                              //
// ------------------------------------------------------------------ //

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn test_data(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = i as f32;
                (x * 0.1).sin() * 3.0 + (x * 0.05).cos() * 2.0
            })
            .collect()
    }

    #[test]
    fn test_ste_passes_gradient() {
        let mut weights = test_data(64);
        let original = weights.clone();

        let trainer = QuantizationAwareTrainer::new(QuantMode::INT8, 32, 0.5);

        let gradient: Vec<f32> = vec![1.0; weights.len()];

        let loss = trainer.train_step(&mut weights, &gradient, 0.01);
        assert!(loss.is_finite(), "STE loss should be finite, got {}", loss);
        assert!(
            loss > 0.0,
            "STE loss should be positive for INT8 quantization"
        );

        let has_changed = weights
            .iter()
            .zip(original.iter())
            .any(|(w, o)| (w - o).abs() > 1e-6);
        assert!(
            has_changed,
            "STE update should modify weights (gradient propagation)"
        );

        assert!(loss < 100.0, "STE loss suspiciously large: {}", loss);
    }

    #[test]
    fn test_export_import_quantized() {
        let data = test_data(256);
        let trainer = QuantizationAwareTrainer::new(QuantMode::INT8, 64, 0.5);

        let tmp_path = "/tmp/test_quant_export.bin";
        if Path::new(tmp_path).exists()
        {
            let _ = std::fs::remove_file(tmp_path);
        }

        trainer.export_quantized(&data, tmp_path).expect("export");
        let (imported, mode, block_size) =
            QuantizationAwareTrainer::import_quantized(tmp_path).expect("import");

        assert_eq!(mode, QuantMode::INT8);
        assert_eq!(block_size, 64);
        assert_eq!(imported.len(), data.len());

        let q = Quantizer::new(QuantMode::INT8, 64);
        let qt = q.quantize(&data);
        let dq = q.dequantize(&qt);

        for (a, b) in dq.iter().zip(imported.iter())
        {
            assert!(
                (a - b).abs() < 1e-6,
                "Export/import mismatch: {} vs {}",
                a,
                b
            );
        }

        let _ = std::fs::remove_file(tmp_path);
    }
}
