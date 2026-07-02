//! First-order training-memory planner for SCIAGENT configs.
//!
//! Training a large transformer on a single device is gated by memory, and the
//! dominant, easily-missed term is the **activation** memory of reverse-mode
//! autodiff — specifically the quadratic attention score matrix
//! `batch × heads × seq × seq`, which explodes at long sequence length. This
//! module turns "will config X fit on a Y-GB device?" into an exact,
//! testable calculation so a Jetson Thor (or any target) run can be sized
//! before a single kernel is written.
//!
//! The numbers are a first-order estimate (constant factors for the linear
//! activation terms are approximate), but the asymptotics are exact and that
//! is what decides feasibility: without flash-attention the score matrix alone
//! is `L · B · H · S²` elements, and without activation checkpointing every
//! layer's activations are held at once.

use crate::config::SciAgentConfig;

const GIB: f64 = (1u64 << 30) as f64;

/// Precision of the stored tensors, in bytes per element.
#[derive(Clone, Copy, Debug)]
pub struct Precision {
    /// Weights storage (e.g. 4 = fp32, 2 = bf16).
    pub weight: usize,
    /// Gradient accumulation (usually fp32 master = 4).
    pub grad: usize,
    /// Optimizer state per parameter. Muon keeps ONE momentum buffer, so this
    /// is `1 × 4` for fp32 momentum (Adam would be `2 × 4`).
    pub opt: usize,
    /// Activation storage on the tape (matches compute dtype).
    pub act: usize,
}

impl Precision {
    pub fn fp32() -> Self {
        Self {
            weight: 4,
            grad: 4,
            opt: 4,
            act: 4,
        }
    }

    /// bf16 compute with an fp32 optimizer momentum; weights/activations bf16.
    pub fn mixed_bf16() -> Self {
        Self {
            weight: 2,
            grad: 4,
            opt: 4,
            act: 2,
        }
    }
}

/// A memory breakdown, all fields in bytes.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub params: u64,
    pub grad: u64,
    pub optimizer: u64,
    pub activations: u64,
}

impl Budget {
    pub fn total(&self) -> u64 {
        self.params + self.grad + self.optimizer + self.activations
    }

    pub fn fits(&self, ceiling_bytes: u64) -> bool {
        self.total() <= ceiling_bytes
    }

    pub fn total_gib(&self) -> f64 {
        self.total() as f64 / GIB
    }
}

/// Estimate the peak training memory for one optimizer step.
///
/// * `flash` — attention that never materializes the `S×S` score matrix
///   (streaming softmax). Removes the quadratic activation term.
/// * `checkpointing` — store only per-layer boundary activations and recompute
///   the rest during backward. Cuts the linear activation term from `O(L)` to
///   `O(1)` layers (plus the boundary tensors).
pub fn estimate(
    config: &SciAgentConfig,
    seq_len: usize,
    batch: usize,
    prec: Precision,
    flash: bool,
    checkpointing: bool,
) -> Budget {
    let p = config.total_parameters() as u64;
    let params = p * prec.weight as u64;
    let grad = p * prec.grad as u64;
    let optimizer = p * prec.opt as u64;

    let b = batch as u64;
    let s = seq_len as u64;
    let d = config.d_model as u64;
    let h = config.n_heads as u64;
    let f = config.d_ff as u64;
    let v = config.vocab_size as u64;
    let d_head = d / h;
    let kv_dim = config.n_kv_heads as u64 * d_head;
    let l = config.n_layers as u64;

    // Per-layer linear activations (elements): QKV + RoPE + context + out proj
    // + two RMSNorms (~8·D), the KV projections (~2·kv_dim), and SwiGLU's four
    // F-sized buffers + down proj (~4·F). Constants are approximate.
    let lin_per_layer = b * s * (8 * d + 2 * kv_dim + 4 * f);
    // Quadratic attention: scores + softmax probs, per head. Gone with flash.
    let quad_per_layer = if flash { b * s * d } else { 2 * b * h * s * s };
    // Output logits (B·S·V) — large at vocab 32768 — plus embedding lookup.
    let head_acts = b * s * v + b * s * d;

    let act_elems = if checkpointing
    {
        // Boundary tensors for every layer + one layer recomputed at peak.
        l * (b * s * d) + (lin_per_layer + quad_per_layer) + head_acts
    }
    else
    {
        l * (lin_per_layer + quad_per_layer) + head_acts
    };
    let activations = act_elems * prec.act as u64;

    Budget {
        params,
        grad,
        optimizer,
        activations,
    }
}

/// Largest power-of-two-ish sequence length whose training step fits under
/// `ceiling_bytes`, scanning a standard ladder. Returns `None` if even the
/// shortest rung overflows.
pub fn max_seq_len_that_fits(
    config: &SciAgentConfig,
    batch: usize,
    prec: Precision,
    flash: bool,
    checkpointing: bool,
    ceiling_bytes: u64,
) -> Option<usize> {
    let ladder = [256usize, 512, 1024, 2048, 4096, 8192];
    ladder
        .iter()
        .rev()
        .copied()
        .find(|&s| estimate(config, s, batch, prec, flash, checkpointing).fits(ceiling_bytes))
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_config_is_tiny() {
        let cfg = SciAgentConfig::small();
        let b = estimate(&cfg, 256, 8, Precision::fp32(), false, false);
        // The shipped small run: comfortably under a gigabyte.
        assert!(
            b.total_gib() < 1.0,
            "small should be < 1 GiB, got {:.3}",
            b.total_gib()
        );
    }

    #[test]
    fn naive_350m_at_8k_blows_past_a_thor() {
        // The whole point: 350M at seq 8192 with no flash, no checkpointing
        // does NOT fit in 128 GB — the S² score matrix dominates.
        let cfg = SciAgentConfig::sciagent_350m();
        let naive = estimate(&cfg, 8192, 1, Precision::fp32(), false, false);
        assert!(
            naive.total_gib() > 128.0,
            "naive 350M@8k should exceed 128 GiB, got {:.1}",
            naive.total_gib()
        );
        // Flash attention removes the quadratic term and changes the picture
        // by orders of magnitude.
        let flash = estimate(&cfg, 8192, 1, Precision::fp32(), true, false);
        assert!(
            flash.total() < naive.total() / 4,
            "flash must cut activations dramatically: {:.1} -> {:.1} GiB",
            naive.total_gib(),
            flash.total_gib()
        );
    }

    #[test]
    fn flash_plus_checkpointing_makes_350m_trainable_on_a_thor() {
        // With flash + activation checkpointing + mixed precision + a shorter
        // sequence, a single 128 GB Thor can hold a 350M training step.
        let cfg = SciAgentConfig::sciagent_350m();
        let thor = 128 * (1u64 << 30);
        let b = estimate(&cfg, 2048, 1, Precision::mixed_bf16(), true, true);
        assert!(
            b.fits(thor),
            "350M@2k flash+ckpt+bf16 should fit 128 GiB, got {:.1}",
            b.total_gib()
        );
        let s = max_seq_len_that_fits(&cfg, 1, Precision::mixed_bf16(), true, true, thor);
        assert!(s.is_some(), "some seq len must fit the Thor");
    }
}
