//! Backends implementing [`crate::traits::MemoryProvider`]. Each backend fully
//! owns its representation; nothing here leaks a backend-specific type upward.

pub mod slha_adapter;
