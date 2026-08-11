//! # SciRust Arena — Allocateurs déterministes
//!
//! Ce module fournit des allocateurs par arène pour le calcul scientifique
//! haute performance. L'objectif est de remplacer les allocations dynamiques
//! répétées par des allocations en temps constant O(1) pour réduire la
//! variabilité de latence dans les boucles critiques.
//!
//! ## Les allocateurs
//!
//! 1. [`PinnedArena`] — allocation par bump pointer, minimum 128-byte aligned
//! 2. [`Slab`] — allocation par slab pour les états séquentiels (Mamba cells)
//! 3. [`AlignedVec`] — Vec avec alignement garanti (utilitaire)
//! 4. [`with_thread_arena`] — réutilisation scoped d'une arène locale au thread
//!
//! ## Exemple d'utilisation
//!
//! ```
//! use scirust_arena::with_thread_arena;
//!
//! with_thread_arena(1 << 20, |arena| {
//!     let x = arena.alloc_slice_fill::<f32>(768, 0.0).unwrap();
//!     assert_eq!(x.len(), 768);
//! }).unwrap();
//! // L'arène du thread est reset automatiquement et son backing storage retenu.
//! ```

mod aligned;
mod allocator;
mod slab;
#[cfg(test)]
mod tests;
mod thread_local_pool;

pub use aligned::AlignedVec;
pub use allocator::{ArenaError, PinnedArena};
pub use slab::{Slab, SlabHandle};
pub use thread_local_pool::{ThreadLocalArenaError, thread_arena_capacity, with_thread_arena};

// Re-export the maximum alignment constant
pub const ALIGNMENT: usize = 128;

/// Minimum alignment chosen by SciRust arena allocations.
///
/// This is a framework policy, not a claim about the cache-line or SIMD width of
/// every supported architecture; those hardware properties are discovered by
/// the compute/SIMD capability layers.
pub const MIN_ALIGN_BYTES: usize = 128;

/// Utility: check if a pointer is aligned to MIN_ALIGN_BYTES.
#[inline]
pub fn is_aligned<T>(ptr: *const T) -> bool {
    ptr as usize & (MIN_ALIGN_BYTES - 1) == 0
}

/// Utility: align an address up to MIN_ALIGN_BYTES.
#[inline]
pub fn align_up(address: usize) -> usize {
    address
        .checked_add(MIN_ALIGN_BYTES - 1)
        .map(|value| value & !(MIN_ALIGN_BYTES - 1))
        .expect("align_up: address overflows usize")
}
