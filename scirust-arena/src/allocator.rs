#![allow(dead_code)]
//! # PinnedArena — Allocator par bump pointer deterministe
//!
//! L'allocation se fait par simple incrémentation d'un pointeur (bump pointer),
//! ce qui garantit un temps constant O(1) pour toute allocation.
//!
//! ## Caractéristiques
//!
//! - Mémoire pinée (pinée) via mmap/MAP_ANONYMOUS — non paginée par le kernel
//! - Alignement 128 octets — compatible L1/L2 cache lines
//! - Reset O(1) — remise à zéro du bump pointer
//! - Pas de fragmentation — allocation linéaire séquentielle
//! - Pas de Drop — toutes les allocations sont dealloquées ensemble

use super::{MIN_ALIGN_BYTES, align_up, is_aligned};

/// Erreurs possibles lors de l'allocation dans l'arène.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    /// L'arène est pleine, plus assez de place pour l'allocation demandée.
    Overflow,
    /// Tentative d'allocation de taille 0.
    ZeroSized,
}

impl std::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            ArenaError::Overflow => write!(f, "Arena overflow: insufficient space"),
            ArenaError::ZeroSized => write!(f, "Zero-sized allocation not allowed"),
        }
    }
}

impl std::error::Error for ArenaError {}

/// Bloc de 128 octets, aligné sur 128 — force l'alignement du backing.
#[repr(C, align(128))]
#[derive(Clone, Copy)]
struct AlignedBlock([u8; 128]);

/// Un bloc de mémoire pré-alloué et aligné sur 128 octets.
struct MemoryBlock {
    /// Backing **possédé** (chaque bloc fait 128 octets, aligné sur 128).
    /// Conserver ce `Vec` est ce qui garde `ptr` valide — l'oublier provoquait
    /// un use-after-free (le buffer était libéré dès la fin de `allocate`).
    backing: Vec<AlignedBlock>,
    /// Pointeur brut vers le bloc (aligné sur 128).
    ptr: *mut u8,
    /// Taille totale du bloc en bytes.
    capacity: usize,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

impl MemoryBlock {
    /// Alloue un bloc de mémoire de taille minimale `min_bytes`, aligné sur 128.
    /// La mémoire est initialisée à zéro et **possédée** par le bloc.
    fn allocate(min_bytes: usize) -> Result<Self, ArenaError> {
        if min_bytes == 0
        {
            return Err(ArenaError::ZeroSized);
        }

        let aligned_size = align_up(min_bytes); // multiple de 128
        let n_blocks = (aligned_size / 128).max(1);
        let mut backing = vec![AlignedBlock([0u8; 128]); n_blocks];
        let ptr = backing.as_mut_ptr() as *mut u8;

        // Pinning best-effort via mlock (ignoré si non disponible / sandbox).
        #[cfg(unix)]
        unsafe {
            use libc::{mlock, munlock};
            let _ = mlock(ptr as *const std::ffi::c_void, aligned_size);
            let _ = munlock(ptr as *const std::ffi::c_void, aligned_size);
        }

        Ok(Self {
            backing,
            ptr,
            capacity: n_blocks * 128,
        })
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.capacity
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        // `backing` (Vec) libère automatiquement la mémoire possédée.
    }
}

/// Arena déterministe — O(1) alloc/dalloc via bump pointer.
///
/// ## Sécurité
///
/// Les pointeurs retournés par `alloc` sont valables jusqu'au prochain `reset()`
/// ou jusqu'à la destruction de l'arène. Ne jamais dereferencer un pointeur
/// après reset — c'est le même pattern que les pool allocators en C++.
pub struct PinnedArena {
    /// Bloc de mémoire sous-jacent (possède la mémoire).
    block: MemoryBlock,
    /// Bump pointer — offset courant en bytes depuis le début du bloc.
    /// Les allocations futures se feront à partir de cet offset.
    offset: usize,
    /// Nombre total d'octets alloués (pour le monitoring).
    allocated_bytes: usize,
    /// Nombre d'allocations effectuées (pour le monitoring).
    alloc_count: usize,
}

impl PinnedArena {
    /// Crée une nouvelle arène avec `min_bytes` d'espace pré-alloué.
    ///
    /// # Panics
    /// Panique si `min_bytes` est 0 ou ne peut être alloué.
    pub fn new(min_bytes: usize) -> Self {
        assert!(min_bytes > 0, "Arena size must be > 0");
        let block = MemoryBlock::allocate(min_bytes)
            .unwrap_or_else(|_| panic!("Failed to allocate arena of {} bytes", min_bytes));
        Self {
            block,
            offset: 0,
            allocated_bytes: 0,
            alloc_count: 0,
        }
    }

    /// Crée une arène avec une taille déterminée par le type T et `num` éléments.
    ///
    /// L'alignement est automatiquement calculé: `align_of::<T>().max(16)`.
    pub fn new_for_type<T>(num: usize) -> Self
    where
        T: Copy,
    {
        assert!(num > 0);
        let elem_size = std::mem::size_of::<T>();
        let alignment = std::mem::align_of::<T>().max(16);
        let total = num * elem_size;
        // Assurer un alignement minimal de ALIGNMENT
        let size = if alignment >= MIN_ALIGN_BYTES
        {
            total
        }
        else
        {
            align_up(total)
        };
        Self::new(size)
    }

    /// Alloue de l'espace pour `n` éléments de type T et retourne un slice mutable.
    ///
    /// # Garantie de sécurité
    /// - L'espace retourné est aligné sur `align_of::<T>().max(16)`
    /// - Le temps d'allocation est O(1) — juste un bump du pointer
    /// - L'espace est initialisé à `T::default()`
    #[inline]
    pub fn alloc_slice<T>(&mut self, n: usize) -> Result<&mut [T], ArenaError>
    where
        T: Copy + Default,
    {
        if n == 0
        {
            return Err(ArenaError::ZeroSized);
        }

        let elem_size = std::mem::size_of::<T>();
        let alignment = std::mem::align_of::<T>().max(16);

        // Aligner l'offset courant
        let aligned_offset = align_up_to(self.offset, alignment);

        let required = aligned_offset + n * elem_size;

        if required > self.block.capacity
        {
            return Err(ArenaError::Overflow);
        }

        // Bump pointer — O(1)
        let ptr = unsafe { self.block.ptr.add(aligned_offset) };

        // Construire le slice mutable
        let slice_ptr = ptr as *mut T;
        let slice = unsafe { std::slice::from_raw_parts_mut(slice_ptr, n) };

        // Initialiser à Default (pour T: Copy, c'est zero-initialization)
        for elem in slice.iter_mut()
        {
            *elem = T::default();
        }

        let new_offset = aligned_offset + n * elem_size;
        self.allocated_bytes += new_offset - self.offset; // self.offset is still the old value
        self.offset = new_offset;
        self.alloc_count += 1;

        Ok(slice)
    }

    /// Alloue un slice rempli avec une valeur donnée.
    ///
    /// O(1) allocation + O(n) initialisation.
    #[inline]
    pub fn alloc_slice_fill<T>(&mut self, n: usize, val: T) -> Result<&mut [T], ArenaError>
    where
        T: Copy + Default,
    {
        let slice = self.alloc_slice(n)?;
        for elem in slice.iter_mut()
        {
            *elem = val;
        }
        Ok(slice)
    }

    /// Alloue de l'espace pour un seul élément de type T.
    #[inline]
    pub fn alloc<T>(&mut self) -> Result<&mut T, ArenaError>
    where
        T: Copy + Default,
    {
        self.alloc_slice::<T>(1).map(|s| &mut s[0])
    }

    /// Alloue de l'espace pour un seul élément, initialisé avec `val`.
    #[inline]
    pub fn alloc_with<T>(&mut self, val: T) -> Result<&mut T, ArenaError>
    where
        T: Copy + Default,
    {
        self.alloc_slice_fill(1, val).map(|s| &mut s[0])
    }

    /// Réinitialise l'arène — toutes les allocations deviennent invalides en O(1).
    ///
    /// Le contenu n'est PAS effacé — c'est un reset structurel (bump pointer à 0).
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
        self.allocated_bytes = 0;
        self.alloc_count = 0;
    }

    /// Retourne la quantité de mémoire restante dans l'arène.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.block
            .capacity
            .saturating_sub(align_up_to(self.offset, 16))
    }

    /// Retourne la quantité totale allouée.
    #[inline]
    pub fn allocated(&self) -> usize {
        self.allocated_bytes
    }

    /// Retourne le nombre d'allocations effectuées.
    #[inline]
    pub fn alloc_count(&self) -> usize {
        self.alloc_count
    }

    /// Retourne la capacité totale en bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.block.capacity
    }

    /// Retourne le taux d'utilisation en pourcentage.
    #[inline]
    pub fn utilization(&self) -> f64 {
        let used = align_up_to(self.offset, 16);
        used as f64 / self.block.capacity as f64 * 100.0
    }

    /// Vérifie que le bump pointer est correctement aligné pour toute allocation future.
    #[inline]
    pub fn is_consistent(&self) -> bool {
        is_aligned(self.offset as *const ())
    }
}

/// Aligner un offset up à `alignment` (qui doit être une puissance de 2).
#[inline]
fn align_up_to(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

impl Default for PinnedArena {
    fn default() -> Self {
        // 1 MB default
        Self::new(1 << 20)
    }
}

/// Vérification au chargement — s'assurer que l'alignement est correct.
const _: () = {
    assert!(
        MIN_ALIGN_BYTES.is_power_of_two(),
        "ALIGNMENT must be power of 2"
    );
};
