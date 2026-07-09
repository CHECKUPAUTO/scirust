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
