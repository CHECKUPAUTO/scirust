//! # AlignedVec — typed storage with a guaranteed base alignment
//!
//! Unlike the former type-erased byte buffer, the element type is part of the
//! type. Safe accessors therefore cannot reinterpret zeroed bytes as a type for
//! which that bit pattern is invalid.

use super::MIN_ALIGN_BYTES;
use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::marker::PhantomData;
use std::ptr::NonNull;

/// A fixed-length vector whose base pointer is aligned to at least 128 bytes.
///
/// Elements are initialized with [`Default`] by [`Self::new`]. The `Copy`
/// bound reflects the arena crate's no-destructor use case and makes teardown
/// deterministic.
pub struct AlignedVec<T: Copy + Default> {
    ptr: NonNull<T>,
    len: usize,
    byte_len: usize,
    layout: Layout,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy + Default + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Default + Sync> Sync for AlignedVec<T> {}

impl<T: Copy + Default> std::fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedVec")
            .field("len", &self.len)
            .field("byte_len", &self.byte_len)
            .field("alignment", &self.layout.align())
            .finish_non_exhaustive()
    }
}

impl<T: Copy + Default> AlignedVec<T> {
    /// Creates `len` valid elements initialized with `T::default()`.
    pub fn new(len: usize) -> Self {
        Self::new_fill(len, T::default())
    }

    /// Creates `len` elements initialized with `val`.
    pub fn new_fill(len: usize, val: T) -> Self {
        let byte_len = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("AlignedVec::new: len * size_of::<T>() overflows usize");
        let alignment = std::mem::align_of::<T>().max(MIN_ALIGN_BYTES);
        let layout = Layout::from_size_align(byte_len, alignment)
            .expect("AlignedVec::new: requested allocation is too large");

        // The global allocator must not be called with a zero-sized layout.
        // Use a non-null, correctly aligned dangling pointer for empty/ZST
        // slices; no element will be read from or written to it.
        let ptr = if byte_len == 0
        {
            // SAFETY: `alignment` is non-zero and aligned for `T`. The pointer
            // is only used to construct zero-byte slices.
            unsafe { NonNull::new_unchecked(alignment as *mut T) }
        }
        else
        {
            let raw = unsafe { alloc(layout) } as *mut T;
            let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));
            for i in 0..len
            {
                // SAFETY: the allocation holds `len` properly aligned T values,
                // and each slot is written exactly once before `Self` escapes.
                unsafe { ptr.as_ptr().add(i).write(val) };
            }
            ptr
        };

        Self {
            ptr,
            len,
            byte_len,
            layout,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: construction initialized all `len` elements as T.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `&mut self` guarantees exclusive access to the allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn fill(&mut self, val: T) {
        self.as_mut_slice().fill(val);
    }

    /// Raw byte pointer, aligned to [`Self::alignment`].
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast()
    }

    /// Mutable raw byte pointer, aligned to [`Self::alignment`].
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr().cast()
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.layout.align()
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Number of bytes occupied by the elements (excluding allocator padding).
    #[inline]
    pub fn len_bytes(&self) -> usize {
        self.byte_len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn is_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize).is_multiple_of(MIN_ALIGN_BYTES)
    }
}

impl<T: Copy + Default> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.byte_len != 0
        {
            // T is Copy, so it has no destructor. Only release the allocation.
            unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) };
        }
    }
}

impl<T: Copy + Default> From<AlignedVec<T>> for Vec<T> {
    fn from(vec: AlignedVec<T>) -> Self {
        vec.as_slice().to_vec()
    }
}

impl<T: Copy + Default> From<Vec<T>> for AlignedVec<T> {
    fn from(v: Vec<T>) -> Self {
        let mut aligned = AlignedVec::new(v.len());
        aligned.as_mut_slice().copy_from_slice(&v);
        aligned
    }
}

/// Extension for converting a `Vec<T>` into aligned typed storage.
#[allow(dead_code)]
pub trait ToAligned<T: Copy + Default>: Sized {
    fn to_aligned(self) -> AlignedVec<T>;
}

impl<T: Copy + Default> ToAligned<T> for Vec<T> {
    fn to_aligned(self) -> AlignedVec<T> {
        AlignedVec::from(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(align(256))]
    #[derive(Clone, Copy, Default)]
    struct OverAligned(u8);

    #[test]
    fn honors_alignment_larger_than_cache_line() {
        let values = AlignedVec::<OverAligned>::new(2);
        assert_eq!(values.alignment(), 256);
        assert_eq!((values.as_ptr() as usize) % 256, 0);
        assert_eq!(values.as_slice()[0].0, 0);
    }

    #[test]
    fn empty_buffer_still_has_an_aligned_non_null_pointer() {
        let values = AlignedVec::<u32>::new(0);
        assert!(values.is_empty());
        assert!(!values.as_ptr().is_null());
        assert!(values.is_aligned());
    }
}
