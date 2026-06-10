//! # SciRust Arena — Allocateurs déterministes
//!
//! Ce module fournit des allocateurs par arène (arena) pour le calcul scientifique
//! haute performance. L'objectif est de remplacer les allocations dynamiques
//! par des allocations en temps constant O(1) pour éliminer la variabilité de
//! latence dans les boucles critiques (inférence SSM/Mamba, trading).
//!
//! ## Les 3 allocateurs
//!
//! 1. [`PinnedArena`] — allocation par bump pointer, 128-byte aligned
//! 2. [`Slab`] — allocation par slab pour les états séquentiels (Mamba cells)
//! 3. [`AlignedVec`] — Vec avec alignement garanti (utilitaire)
//!
//! ## Exemple d'utilisation
//!
//! ```
//! use scirust_arena::PinnedArena;
//!
//! let mut arena = PinnedArena::new(1 << 20); // 1 MB
//!
//! // O(1) allocations
//! let x = arena.alloc_slice_fill::<f32>(768, 0.0).unwrap();
//! let y = arena.alloc_slice_fill::<f32>(768, 0.0).unwrap();
//!
//! // Reset — toutes les allocations deviennent invalides en O(1)
//! arena.reset();
//! ```

#![cfg_attr(feature = "nightly", feature(ptr_metadata))]

mod allocator;
mod aligned;
mod slab;
#[cfg(test)]
mod tests;

pub use allocator::{ArenaError, PinnedArena};
pub use aligned::AlignedVec;
pub use slab::{Slab, SlabHandle};

// Re-export the maximum alignment constant
pub const ALIGNMENT: usize = 128;

/// Size in bytes of the minimum alignment for all SciRust allocations.
/// This matches the L1 cache line size and SIMD vector width on all target platforms.
pub const MIN_ALIGN_BYTES: usize = 128;

/// Utility: check if a pointer is aligned to MIN_ALIGN_BYTES.
#[inline]
pub fn is_aligned<T>(ptr: *const T) -> bool {
    ptr as usize & (MIN_ALIGN_BYTES - 1) == 0
}

/// Utility: align an address up to MIN_ALIGN_BYTES.
#[inline]
pub fn align_up(address: usize) -> usize {
    (address + MIN_ALIGN_BYTES - 1) & !(MIN_ALIGN_BYTES - 1)
}
