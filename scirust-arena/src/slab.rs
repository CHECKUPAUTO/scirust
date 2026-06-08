//! # Slab — Allocation par slab pour les états séquentiels SSM
//!
//! Le Slab est un allocateur optimisé pour stocker des états de cellules
//! Mamba/SSM qui ont une taille fixe mais doivent être alloués/désalloués
//! dynamiquement selon la longueur de la séquence.

use super::MIN_ALIGN_BYTES;

/// Handle vers une entrée du slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlabHandle {
    index: u32,
    version: u32,
}

impl SlabHandle {
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    #[inline]
    pub fn version(&self) -> u32 {
        self.version
    }
}

struct SlabEntry {
    state: EntryState,
    version: u32,
}

enum EntryState {
    Occupied { size: usize },
    Free,
}

/// Slab — stockage de taille fixe pour les états SSM.
///
/// ## Usage
///
/// ```
/// use scirust_arena::Slab;
///
/// let mut slab: Slab<f32, 768> = Slab::new(10);
/// let h = slab.alloc().unwrap();
/// slab.free(h);
/// ```
pub struct Slab<T: Copy, const N: usize> {
    entries: Vec<SlabEntry>,
    free_list: Vec<u32>,
    next_version: u32,
    count: usize,
    memory: Vec<u8>,
    slot_size_bytes: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Copy, const N: usize> Slab<T, N> {
    /// Crée un nouveau slab avec `capacity` slots de `N` éléments de type T.
    pub fn new(capacity: usize) -> Self {
        let elem_bytes = std::mem::size_of::<T>();
        let slot_bytes = N * elem_bytes;
        let aligned_slot = slot_bytes.max(MIN_ALIGN_BYTES);
        let total_bytes = capacity * aligned_slot;

        let memory = vec![0u8; total_bytes];
        let mut free_list = Vec::with_capacity(capacity);
        for i in 0..capacity {
            free_list.push(i as u32);
        }

        Self {
            entries: (0..capacity)
                .map(|_| SlabEntry {
                    state: EntryState::Free,
                    version: 0,
                })
                .collect(),
            free_list,
            next_version: 1,
            count: 0,
            memory,
            slot_size_bytes: aligned_slot,
            _marker: std::marker::PhantomData,
        }
    }

    /// Alloue une entrée.
    #[inline]
    pub fn alloc(&mut self) -> Option<SlabHandle> {
        let index = self.free_list.pop()?;

        let version = self.next_version;
        self.next_version += 1;

        self.entries[index as usize].state = EntryState::Occupied {
            size: self.slot_size_bytes,
        };
        self.entries[index as usize].version = version;
        self.count += 1;

        Some(SlabHandle { index, version })
    }

    /// Libère une entrée.
    #[inline]
    pub fn free(&mut self, handle: SlabHandle) {
        let entry = &mut self.entries[handle.index as usize];
        if let EntryState::Free = entry.state {
            return;
        }
        entry.state = EntryState::Free;
        entry.version = 0;
        self.free_list.push(handle.index);
        self.count = self.count.saturating_sub(1);
    }

    #[inline]
    pub fn is_valid(&self, handle: SlabHandle) -> bool {
        let entry = &self.entries[handle.index as usize];
        matches!(entry.state, EntryState::Occupied { .. })
            && entry.version == handle.version
    }

    /// Retourne un slice mutable vers les données d'une entrée.
    #[inline]
    pub fn data_slice(&mut self, handle: SlabHandle) -> Option<&mut [T]> {
        if !self.is_valid(handle) {
            return None;
        }
        let base = self.memory.as_mut_ptr() as usize;
        let slot_base = base + handle.index as usize * self.slot_size_bytes;
        let ptr = slot_base as *mut T;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, N) })
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Réinitialise tout le slab.
    pub fn reset(&mut self) {
        for entry in &mut self.entries {
            entry.state = EntryState::Free;
            entry.version = 0;
        }
        self.free_list.clear();
        for i in 0..self.entries.len() {
            self.free_list.push(i as u32);
        }
        self.next_version = 1;
        self.count = 0;
    }
}
