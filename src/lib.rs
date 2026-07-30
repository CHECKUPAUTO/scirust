//! # SciRust
//!
//! Umbrella crate for the **SciRust** pure-Rust deep-learning and
//! scientific-computing framework. The implementations live in the member
//! crates; this facade re-exports them under a single `scirust::*` entry point
//! so the package matches its description as a framework rather than a single
//! binary.
//!
//! ```
//! use scirust::core::autodiff::reverse::Tensor;
//!
//! let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//! assert_eq!(t.rows, 2);
//! assert_eq!(t.cols, 2);
//! ```
//!
//! ## Optional: the canonical tensor facade
//!
//! `tensor_canonical` is a compile-time-planned tensor pipeline — build a
//! program, prepare it once, run it many times. It is **off by default**; see
//! its module documentation for the two features that turn it on.
//!
//! ## Note on the bundled binary
//!
//! The repository also ships an **experimental autonomous-agent demo**,
//! `openclaw-u` (`src/main.rs`). It is unrelated to the framework, is not
//! required to build or use it, and is kept as a separate, clearly-named binary.

pub use scirust_core as core;
pub use scirust_learning as learning;
pub use scirust_rsi as rsi;
pub use scirust_simd as simd;
pub use scirust_solvers as solvers;
pub use scirust_symbolic as symbolic;

/// One-import entry point: `use scirust::prelude::*;` brings the tensor,
/// autodiff, neural-network, error, and symbolic essentials into scope — the
/// exact symbols the README quickstart uses — so a first program needs a single
/// `use`.
///
/// ```
/// use scirust::prelude::*;
///
/// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
/// assert_eq!(t.shape(), (2, 2));
/// ```
pub mod prelude {
    pub use scirust_core::prelude::*;
}

#[cfg(feature = "tensor-canonical")]
pub mod tensor_canonical {
    //! The canonical tensor facade: plan a computation once, run it many times.
    //!
    //! **Optional.** Enable one of two features:
    //!
    //! ```toml
    //! # The facade, with a backend you supply yourself.
    //! scirust = { version = "0.14", features = ["tensor-canonical"] }
    //!
    //! # The facade plus the deterministic CPU adapter, ready to run.
    //! scirust = { version = "0.14", features = ["tensor-canonical-cpu"] }
    //! ```
    //!
    //! Neither is in `default`, so a plain `scirust` dependency compiles exactly as
    //! it did before this module existed. Neither activates wgpu or CUDA, and
    //! neither adds an external crate to the lockfile: every crate behind them is a
    //! pure-Rust workspace member.
    //!
    //! # Which feature
    //!
    //! `tensor-canonical` gives you the whole pipeline but no backend — you pass
    //! your own [`ComputeBackend`] implementation to [`ReferencePlanRuntime::new`].
    //! The choice of device stays yours.
    //!
    //! `tensor-canonical-cpu` additionally re-exports `CpuComputeAdapter`, the
    //! deterministic CPU Reference interpreter, so a program runs immediately:
    //!
    //! ```ignore
    //! let runtime = ReferencePlanRuntime::new(CpuComputeAdapter::new());
    //! ```
    //!
    //! That adapter lives in a crate called `scirust-gpu` for historical reasons.
    //! The name is misleading and the feature is named after what it actually
    //! provides: its default build carries no GPU code at all.
    //!
    //! # Usage
    //!
    //! ```ignore
    //! use scirust::tensor_canonical::{
    //!     CanonicalInputs, CanonicalProgram, CpuComputeAdapter, ReferencePlanRuntime, TensorND,
    //! };
    //!
    //! let mut program = CanonicalProgram::new();
    //! let x = program.input("x", &[2, 2])?;
    //! let bias = program.constant(TensorND::try_new(vec![1.0; 4], vec![2, 2])?)?;
    //! let biased = program.add(x, bias)?;
    //! let y = program.relu(biased)?;
    //! program.set_outputs([y])?;
    //!
    //! let session = program.prepare(ReferencePlanRuntime::new(CpuComputeAdapter::new()))?;
    //!
    //! let values = TensorND::try_new(vec![-2.0, 0.0, 1.0, 3.0], vec![2, 2])?;
    //! let mut inputs = CanonicalInputs::new();
    //! inputs.bind(x, &values);
    //!
    //! let outputs = session.execute(&inputs)?;
    //! ```
    //!
    //! A runnable version is `examples/canonical_tensor_cpu.rs`, built with
    //! `cargo run --features tensor-canonical-cpu --example canonical_tensor_cpu`.
    //!
    //! # What it does and does not do
    //!
    //! * **`f32` only.** Values are [`TensorND`] — dense row-major host memory, no
    //!   device field, no dtype field.
    //! * **No broadcasting.** Binary operations require identical shapes, because
    //!   the canonical IR compares whole tensor types.
    //! * **Eight operations**: `add`, `sub`, `mul`, `div`, `relu`, `scale`,
    //!   `reshape`, `permute`. `exp`, `log` and `matmul` are deliberately absent —
    //!   the layers below reject them, so offering them would advertise a
    //!   capability that does not exist.
    //! * **Prepare once, execute many times.** [`CanonicalProgram::prepare`]
    //!   consumes the program and compiles every kernel; execution recompiles
    //!   nothing and keeps no state between runs.
    //! * **No autograd, no training, no session serialisation.** This is an
    //!   inference-shaped execution path, not a learning framework. For gradients,
    //!   use the eager [`crate::core`] tape instead.
    //! * **Backend implementers** need more than [`ComputeBackend`] — the trait's
    //!   associated types and `KernelModule`, `DeviceCapabilities`, `BufferBinding`,
    //!   `LaunchConfig` and `MemorySpace` come from `scirust-compute`, which such a
    //!   caller should depend on directly.
    //!
    //! # A name worth knowing about
    //!
    //! [`TensorND`] here is `scirust_tensor_core::TensorND`, backed by a
    //! `Vec<f32>`. It is **a different type** from `scirust::core::tensor::TensorND`,
    //! which is backed by an `Arc<[f32]>` and belongs to the eager stack. They share
    //! a name and nothing else, which is why neither appears in [`crate::prelude`].

    pub use scirust_compute::ComputeBackend;
    pub use scirust_tensor_core::TensorND;
    pub use scirust_tensor_runtime::{
        CanonicalBuildError, CanonicalExecutionError, CanonicalInput, CanonicalInputSpec,
        CanonicalInputs, CanonicalOutputSpec, CanonicalOutputs, CanonicalPreparationError,
        CanonicalProgram, CanonicalSession, CanonicalValue, ReferencePlanRuntime,
    };

    #[cfg(feature = "tensor-canonical-cpu")]
    pub use scirust_gpu::CpuComputeAdapter;
}
