pub mod nn;
pub mod io;
pub use scirust_autodiff::*;
pub use scirust_macros::autodiff;
pub use scirust_simd::*;
pub use scirust_gpu::dispatch;

pub mod matrix {
    pub mod view;
    pub mod backend;
}

pub mod autodiff { pub mod optim; }

pub mod lazy;

pub mod data;
