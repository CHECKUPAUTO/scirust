//! # Slab — fixed-size typed storage for sequential SSM state
//!
//! Slots contain actual `[T; N]` values. This avoids manufacturing `&mut [T]`
//! over zeroed bytes, which is invalid for many perfectly legal `Copy` types.

/// A slot is aligned to a cache line; an over-aligned T raises the effective
/// alignment automatically because it is a field of this struct.
#[repr(C, align(128))]
struct AlignedSlot<T, const N: usize>([T; N]);

/// Handle to a slab entry. Versions prevent stale handles from accessing a
/// slot that has since been recycled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlabHandle {
    index: u32,
    version: u64,
}

impl SlabHandle {
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }
}

struct SlabEntry {
    occupied: bool,
    version: u64,
}

/// Typed slab with `capacity` fixed-size slots of `N` elements.
pub struct Slab<T: Copy + Default, const N: usize> {
    entries: Vec<SlabEntry>,
    free_list: Vec<u32>,
    next_version: u64,
    count: usize,
    memory: Vec<AlignedSlot<T, N>>,
}

impl<T: Copy + Default, const N: usize> Slab<T, N> {
    pub fn new(capacity: usize) -> Self {
        assert!(
            u32::try_from(capacity).is_ok(),
            "Slab capacity exceeds the u32 handle index range"
        );

        let memory = (0..capacity)
            .map(|_| AlignedSlot([T::default(); N]))
            .collect();
        let entries = (0..capacity)
            .map(|_| SlabEntry {
                occupied: false,
                version: 0,
            })
            .collect();
        let free_list = (0..capacity as u32).collect();

        Self {
            entries,
            free_list,
            next_version: 1,
            count: 0,
            memory,
        }
    }

    /// Allocates one slot. Version exhaustion is reported as `None` instead of
    /// wrapping and making an ancient stale handle valid again.
    #[inline]
    pub fn alloc(&mut self) -> Option<SlabHandle> {
        let version = self.next_version;
        self.next_version = self.next_version.checked_add(1)?;
        let index = self.free_list.pop()?;
        let entry = &mut self.entries[index as usize];
        entry.occupied = true;
        entry.version = version;
        self.count += 1;
        Some(SlabHandle { index, version })
    }

    /// Frees a currently valid handle. Out-of-range, stale and already-freed
    /// handles are harmless no-ops.
    #[inline]
    pub fn free(&mut self, handle: SlabHandle) {
        let Some(entry) = self.entries.get_mut(handle.index as usize)
        else
        {
            return;
        };
        if !entry.occupied || entry.version != handle.version
        {
            return;
        }
        entry.occupied = false;
        entry.version = 0;
        self.free_list.push(handle.index);
        self.count -= 1;
    }

    #[inline]
    pub fn is_valid(&self, handle: SlabHandle) -> bool {
        self.entries
            .get(handle.index as usize)
            .is_some_and(|entry| entry.occupied && entry.version == handle.version)
    }

    #[inline]
    pub fn data_slice(&mut self, handle: SlabHandle) -> Option<&mut [T]> {
        if !self.is_valid(handle)
        {
            return None;
        }
        Some(&mut self.memory[handle.index as usize].0)
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Invalidates every outstanding handle while preserving the monotonic
    /// version counter so pre-reset handles can never become valid again.
    pub fn reset(&mut self) {
        for entry in &mut self.entries
        {
            entry.occupied = false;
            entry.version = 0;
        }
        self.free_list.clear();
        self.free_list.extend(0..self.entries.len() as u32);
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MIN_ALIGN_BYTES;

    #[repr(align(256))]
    #[derive(Clone, Copy, Default)]
    struct OverAligned(u32);

    #[test]
    fn data_slice_is_aligned_for_every_slot() {
        let mut slab: Slab<f32, 33> = Slab::new(16);
        let handles: Vec<_> = (0..16).map(|_| slab.alloc().unwrap()).collect();
        for handle in handles
        {
            let addr = slab.data_slice(handle).unwrap().as_ptr() as usize;
            assert_eq!(addr % MIN_ALIGN_BYTES, 0);
        }
    }

    #[test]
    fn supports_types_aligned_beyond_cache_line() {
        let mut slab: Slab<OverAligned, 3> = Slab::new(2);
        let handle = slab.alloc().unwrap();
        let slot = slab.data_slice(handle).unwrap();
        assert_eq!((slot.as_ptr() as usize) % 256, 0);
        assert_eq!(slot[0].0, 0);
    }

    #[test]
    fn stale_handle_cannot_free_recycled_slot() {
        let mut slab: Slab<u32, 4> = Slab::new(1);
        let stale = slab.alloc().unwrap();
        slab.free(stale);
        let current = slab.alloc().unwrap();
        slab.free(stale);
        assert!(slab.is_valid(current));
        assert_eq!(slab.count(), 1);
    }

    #[test]
    fn reset_never_revalidates_old_handle() {
        let mut slab: Slab<u32, 4> = Slab::new(1);
        let old = slab.alloc().unwrap();
        slab.reset();
        let current = slab.alloc().unwrap();
        assert_ne!(old.version(), current.version());
        assert!(!slab.is_valid(old));
        assert!(slab.is_valid(current));
    }

    #[test]
    fn slots_do_not_overlap() {
        let mut slab: Slab<u32, 40> = Slab::new(8);
        let handles: Vec<_> = (0..8).map(|_| slab.alloc().unwrap()).collect();
        for (i, handle) in handles.iter().enumerate()
        {
            slab.data_slice(*handle).unwrap().fill(i as u32);
        }
        for (i, handle) in handles.iter().enumerate()
        {
            assert!(
                slab.data_slice(*handle)
                    .unwrap()
                    .iter()
                    .all(|&value| value == i as u32)
            );
        }
    }
}
