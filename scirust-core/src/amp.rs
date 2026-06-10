//! Automatic Mixed Precision (AMP) training wrapper.
//!
//! Provides a `MixedPrecisionEnv` that automatically casts model weights
//! and activations between f32 and f16/BF16 for memory-efficient training.
//!
//! The pattern:
//! - Weights stored in f32 (master copy).
//! - Forward pass uses f16/BF16 for compute.
//! - Loss is scaled up to avoid underflow.
//! - Gradients are unscaled before the optimizer step.
//!
//! # Example
//!
//! ```ignore
//! use scirust_core::amp::MixedPrecisionEnv;
//!
//! let mut amp = MixedPrecisionEnv::new(MixedPrecisionKind::BF16, LossScale::Dynamic);
//! // ... in training loop:
//! let (loss, grads) = amp.forward_backward(&model, &data, &labels)?;
//! amp.optimizer_step(&mut opt, &grads)?;
//! ```

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::Module;

/// Precision kind for mixed-precision training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MixedPrecisionKind {
    /// IEEE 754 half-precision (5-bit exponent, 10-bit mantissa).
    FP16,
    /// Brain floating point (8-bit exponent, 7-bit mantissa).
    BF16,
}

/// Loss scaling strategy.
#[derive(Debug, Clone, Copy)]
pub enum LossScale {
    /// Fixed scaling factor.
    Fixed(f32),
    /// Dynamic scaling: increase on no-overflow, decrease on overflow.
    Dynamic,
}

/// Automatic Mixed Precision environment.
#[derive(Debug, Clone)]
pub struct MixedPrecisionEnv {
    pub kind: MixedPrecisionKind,
    pub loss_scale: LossScale,
    /// Current loss scale value (effective for Dynamic mode).
    pub current_scale: f32,
    /// Consecutive steps without overflow (for Dynamic mode).
    growth_steps: u32,
    /// Growth interval before increasing scale.
    growth_interval: u32,
    /// Scale factor when growing.
    growth_factor: f32,
    /// Scale factor when backing off after overflow.
    backoff_factor: f32,
    /// Maximum allowed loss scale.
    max_scale: f32,
}

impl MixedPrecisionEnv {
    /// Create a new AMP environment.
    pub fn new(kind: MixedPrecisionKind, loss_scale: LossScale) -> Self {
        let init_scale = match loss_scale {
            LossScale::Fixed(s) => s,
            LossScale::Dynamic => 2.0f32.powi(16), // 65536
        };

        Self {
            kind,
            loss_scale,
            current_scale: init_scale,
            growth_steps: 0,
            growth_interval: 2000,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            max_scale: 2.0f32.powi(24), // 16.7M
        }
    }

    /// Cast f32 tensor to the target precision.
    pub fn cast_to(&self, data: &[f32]) -> Vec<f32> {
        match self.kind {
            MixedPrecisionKind::FP16 => {
                data.iter().map(|&x| fp32_to_fp16(x)).collect()
            }
            MixedPrecisionKind::BF16 => {
                data.iter().map(|&x| fp32_to_bf16(x)).collect()
            }
        }
    }

    /// Cast from target precision back to f32.
    pub fn cast_from(&self, data: &[f32]) -> Vec<f32> {
        match self.kind {
            MixedPrecisionKind::FP16 => {
                data.iter().map(|&x| fp16_to_fp32(x)).collect()
            }
            MixedPrecisionKind::BF16 => {
                data.iter().map(|&x| bf16_to_fp32(x)).collect()
            }
        }
    }

    /// Scale loss up for stable gradient computation.
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.current_scale
    }

    /// Unscale gradients after backward pass.
    pub fn unscale_gradients(&self, grads: &[f32]) -> Vec<f32> {
        let inv_scale = 1.0 / self.current_scale;
        grads.iter().map(|&g| g * inv_scale).collect()
    }

    /// Check if gradients contain NaN/Inf (overflow detection).
    pub fn has_overflow(grads: &[f32]) -> bool {
        grads.iter().any(|&g| g.is_nan() || g.is_infinite())
    }

    /// Update loss scale after a training step.
    pub fn update_scale(&mut self, had_overflow: bool) {
        match self.loss_scale {
            LossScale::Fixed(_) => {} // No change
            LossScale::Dynamic => {
                if had_overflow {
                    self.current_scale *= self.backoff_factor;
                    self.growth_steps = 0;
                } else {
                    self.growth_steps += 1;
                    if self.growth_steps >= self.growth_interval {
                        self.current_scale =
                            (self.current_scale * self.growth_factor).min(self.max_scale);
                        self.growth_steps = 0;
                    }
                }
            }
        }
    }
}

// ── Precision conversion utilities ──

/// Simulate FP32 → FP16 conversion (truncate mantissa).
#[inline]
pub fn fp32_to_fp16(x: f32) -> f32 {
    let bits = x.to_bits();
    // FP16 has 10-bit mantissa vs FP32's 23-bit
    // Truncate bottom 13 bits of mantissa
    let truncated = bits & 0xFFFF_E000;
    f32::from_bits(truncated)
}

/// Simulate FP16 → FP32 conversion (identity after truncation).
#[inline]
pub fn fp16_to_fp32(x: f32) -> f32 {
    x // Already f32, the truncation happened at cast_to
}

/// Simulate FP32 → BF16 conversion (keep exponent, truncate mantissa to 7 bits).
#[inline]
pub fn fp32_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    // BF16: keep upper 16 bits (sign + 8-bit exponent + 7-bit mantissa)
    let truncated = bits & 0xFFFF_0000;
    f32::from_bits(truncated)
}

/// Simulate BF16 → FP32 conversion.
#[inline]
pub fn bf16_to_fp32(x: f32) -> f32 {
    x // Already f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_conversion() {
        let x = 1.23456789f32;
        let half = fp32_to_fp16(x);
        let back = fp16_to_fp32(half);
        // Should lose precision
        assert!((back - x).abs() > 1e-6);
    }

    #[test]
    fn test_bf16_conversion() {
        let x = 1.23456789f32;
        let half = fp32_to_bf16(x);
        let back = bf16_to_fp32(half);
        // BF16 has even less mantissa than FP16
        assert!((back - x).abs() > 1e-5);
    }

    #[test]
    fn test_overflow_detection() {
        assert!(MixedPrecisionEnv::has_overflow(&[f32::NAN, 1.0]));
        assert!(MixedPrecisionEnv::has_overflow(&[1.0, f32::INFINITY]));
        assert!(!MixedPrecisionEnv::has_overflow(&[1.0, 2.0]));
    }

    #[test]
    fn test_dynamic_scale_growth() {
        let mut amp = MixedPrecisionEnv::new(MixedPrecisionKind::BF16, LossScale::Dynamic);
        let initial = amp.current_scale;

        // No overflow for growth_interval steps
        for _ in 0..amp.growth_interval {
            amp.update_scale(false);
        }
        assert!(amp.current_scale > initial);

        // Overflow should reduce scale
        amp.update_scale(true);
        assert!(amp.current_scale < initial * amp.growth_factor);
    }

    #[test]
    fn test_fixed_scale_stability() {
        let mut amp = MixedPrecisionEnv::new(MixedPrecisionKind::FP16, LossScale::Fixed(1024.0));
        let initial = amp.current_scale;
        amp.update_scale(false);
        amp.update_scale(true);
        assert_eq!(amp.current_scale, initial);
    }

    #[test]
    fn test_loss_scaling() {
        let amp = MixedPrecisionEnv::new(MixedPrecisionKind::BF16, LossScale::Fixed(8.0));
        let loss = 0.125;
        let scaled = amp.scale_loss(loss);
        assert_eq!(scaled, 1.0);

        let grads = vec![0.5, 1.0, 2.0];
        let unscaled = amp.unscale_gradients(&grads);
        assert!((unscaled[0] - 0.0625).abs() < 1e-6);
    }
}
