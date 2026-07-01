//! # Slab — Allocation par slab pour les états séquentiels SSM
//!
//! Le Slab est un allocateur optimisé pour stocker des états de cellules
//! Mamba/SSM qui ont une taille fixe mais doivent être alloués/désalloués
//! dynamiquement selon la longueur de la séquence.

use super::MIN_ALIGN_BYTES;

/// Bloc de 128 octets, aligné sur 128 — force l'alignement du buffer sous-jacent.
///
/// Un `Vec<u8>` n'est aligné que sur 1 octet ; en backant `memory` avec un
/// `Vec<AlignBlock>` on garantit que le pointeur de base est réellement aligné
/// sur 128 octets. Comme `slot_size_bytes` est toujours un multiple de
/// `MIN_ALIGN_BYTES`, chaque slot commence donc à une adresse alignée sur 128,
/// donc alignée pour tout `T` dont l'alignement divise 128.
#[repr(C, align(128))]
#[derive(Clone, Copy)]
struct AlignBlock([u8; MIN_ALIGN_BYTES]);

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
    Occupied {
        #[allow(dead_code)]
        size: usize,
    },
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
    memory: Vec<AlignBlock>,
    slot_size_bytes: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Copy, const N: usize> Slab<T, N> {
    /// Crée un nouveau slab avec `capacity` slots de `N` éléments de type T.
    pub fn new(capacity: usize) -> Self {
        let elem_bytes = std::mem::size_of::<T>();
        let slot_bytes = N * elem_bytes;
        // Chaque slot est padé à un multiple de MIN_ALIGN_BYTES pour que son
        // début reste aligné une fois le buffer de base aligné sur 128 octets.
        let aligned_slot = super::align_up(slot_bytes.max(MIN_ALIGN_BYTES));
        let total_bytes = capacity * aligned_slot;

        // Backing aligné sur 128 : `AlignBlock` fait MIN_ALIGN_BYTES octets et
        // est aligné sur MIN_ALIGN_BYTES, donc le pointeur de base l'est aussi.
        let n_blocks = total_bytes.div_ceil(MIN_ALIGN_BYTES);
        let memory = vec![AlignBlock([0u8; MIN_ALIGN_BYTES]); n_blocks];
        let mut free_list = Vec::with_capacity(capacity);
        for i in 0..capacity
        {
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
        if let EntryState::Free = entry.state
        {
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
        matches!(entry.state, EntryState::Occupied { .. }) && entry.version == handle.version
    }

    /// Retourne un slice mutable vers les données d'une entrée.
    #[inline]
    pub fn data_slice(&mut self, handle: SlabHandle) -> Option<&mut [T]> {
        if !self.is_valid(handle)
        {
            return None;
        }
        let base = self.memory.as_mut_ptr() as *mut u8;
        // `base` est aligné sur 128 (backing `AlignBlock`) et l'offset est un
        // multiple de 128 (`slot_size_bytes`), donc `ptr` est aligné pour tout
        // `T` dont l'alignement divise 128. On l'affirme en debug.
        let ptr = unsafe { base.add(handle.index as usize * self.slot_size_bytes) } as *mut T;
        debug_assert!(
            (ptr as usize).is_multiple_of(std::mem::align_of::<T>()),
            "Slab::data_slice produced a misaligned pointer"
        );
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
        for entry in &mut self.entries
        {
            entry.state = EntryState::Free;
            entry.version = 0;
        }
        self.free_list.clear();
        for i in 0..self.entries.len()
        {
            self.free_list.push(i as u32);
        }
        self.next_version = 1;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Régression : `data_slice` doit renvoyer un pointeur correctement aligné
    /// pour `T`. Auparavant, `memory` était un `Vec<u8>` (alignement 1), donc le
    /// pointeur de base — et donc chaque slot — pouvait être sous-aligné, ce qui
    /// rend `from_raw_parts_mut` UB. Le backing `AlignBlock` (aligné sur 128)
    /// garantit désormais l'alignement de tous les slots.
    #[test]
    fn data_slice_is_aligned_for_every_slot() {
        // N tel que N * size_of::<f32>() n'est pas un multiple de 128, pour
        // exercer aussi le padding par slot.
        let mut slab: Slab<f32, 33> = Slab::new(16);
        let mut handles = Vec::new();
        while let Some(h) = slab.alloc()
        {
            handles.push(h);
        }
        assert_eq!(handles.len(), 16);

        for h in &handles
        {
            let slice = slab.data_slice(*h).expect("valid handle");
            let addr = slice.as_ptr() as usize;
            assert_eq!(
                addr % std::mem::align_of::<f32>(),
                0,
                "slot pointer not aligned for T"
            );
            assert_eq!(
                addr % MIN_ALIGN_BYTES,
                0,
                "slot pointer not aligned to MIN_ALIGN_BYTES"
            );
            assert_eq!(slice.len(), 33);
        }
    }

    /// Régression : le backing lui-même doit être aligné sur 128 octets.
    #[test]
    fn backing_base_is_128_aligned() {
        let slab: Slab<f64, 100> = Slab::new(4);
        let base = slab.memory.as_ptr() as usize;
        assert_eq!(base % MIN_ALIGN_BYTES, 0, "backing base not 128-aligned");
    }

    /// Les données écrites via `data_slice` doivent survivre indépendamment par
    /// slot (pas de recouvrement dû à un mauvais calcul d'offset après padding).
    #[test]
    fn slots_do_not_overlap_after_padding() {
        let mut slab: Slab<u32, 40> = Slab::new(8);
        let handles: Vec<_> = (0..8).map(|_| slab.alloc().unwrap()).collect();

        for (i, h) in handles.iter().enumerate()
        {
            let slice = slab.data_slice(*h).unwrap();
            for elem in slice.iter_mut()
            {
                *elem = i as u32;
            }
        }

        for (i, h) in handles.iter().enumerate()
        {
            let slice = slab.data_slice(*h).unwrap();
            assert!(slice.iter().all(|&v| v == i as u32), "slot {i} corrupted");
        }
    }
}
