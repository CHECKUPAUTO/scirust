//! GPU acceleration for the SCIAGENT model (feature `gpu`).
//!
//! The whole model runs on `scirust_core`'s reverse-mode [`Tape`]. Rather than
//! reimplement the forward pass on the device, this module attaches
//! `scirust_gpu`'s validated [`WgpuEngine`] — the tape's GEMM hook — and flips
//! the tape into GPU-matmul mode with [`Tape::set_prefer_gpu_matmul`]. Every
//! plain `matmul` / `try_matmul` the model issues then runs its forward **and**
//! backward on the GPU:
//!
//! - the q/k/v/o projections and the SwiGLU gate/up/down (all `Linear`),
//! - RoPE's pair-rotation `x·W` GEMM,
//! - the per-head attention scores `Q·Kᵀ` and the `·V` re-weighting,
//! - the tied LM head `h·Eᵀ`.
//!
//! The autodiff graph and every non-GEMM op (softmax, RMSNorm, RoPE trig,
//! residual adds, the causal mask) are untouched and stay on the CPU. GEMMs are
//! the dominant FLOPs of a transformer, so routing just them is the pragmatic
//! first integration — no new kernels, and the exact same math the CPU path was
//! already validated against, brick by brick.
//!
//! GPU GEMM accumulates in a different order than the CPU BLAS, so results are
//! **not** bit-identical; they agree within a small relative tolerance. See
//! `tests/gpu_parity.rs` and `examples/gpu_forward_parity.rs`, which check a full
//! model forward + backward against the CPU on a real adapter (e.g. the Jetson
//! Thor's Blackwell).

use scirust_core::autodiff::reverse::Tape;

pub use scirust_gpu::WgpuEngine;

/// Attach a freshly-acquired [`WgpuEngine`] to `tape` and switch it into
/// GPU-matmul mode.
///
/// Returns the adapter name on success, or `None` when no GPU adapter is
/// available (no Vulkan/Metal/DX12 driver). On `None` the tape is left
/// untouched and stays CPU-only, so a caller can fall back transparently:
///
/// ```no_run
/// # use scirust_core::autodiff::reverse::Tape;
/// let tape = Tape::new();
/// match scirust_sciagent::gpu::attach_gpu(&tape) {
///     Some(name) => println!("training on {name}"),
///     None => println!("no GPU, staying on CPU"),
/// }
/// // ... model.forward(&tape, ids, seq_len) now runs its GEMMs on the device.
/// ```
pub fn attach_gpu(tape: &Tape) -> Option<String> {
    let engine = WgpuEngine::new()?;
    let name = engine.adapter_name().to_string();
    tape.set_gpu_engine(engine);
    tape.set_prefer_gpu_matmul(true);
    Some(name)
}
