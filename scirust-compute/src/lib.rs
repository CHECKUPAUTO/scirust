//! Backend-neutral compute contracts for SciRust.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod backend;
mod binding;
mod capabilities;
mod device;
mod dtype;
mod error;
mod ids;
mod kernel;
mod launch;
mod memory;
mod shape;
mod strides;
mod tensor;

pub use backend::ComputeBackend;
pub use binding::{BufferAccess, BufferBinding};
pub use capabilities::DeviceCapabilities;
pub use device::{DeviceId, DeviceKind};
pub use dtype::DType;
pub use error::{ComputeError, ComputeResult};
pub use ids::{BufferId, EventId, KernelId, StreamId};
pub use kernel::{KernelFormat, KernelModule};
pub use launch::LaunchConfig;
pub use memory::{Layout, MemorySpace};
pub use shape::Shape;
pub use strides::Strides;
pub use tensor::TensorSpec;
