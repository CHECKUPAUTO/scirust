//! Typed pinned host memory for zero-copy accelerator transfers.
//!
//! The element type is retained by the buffer and pool. Safe APIs can no
//! longer reinterpret zero-filled bytes as an unrelated or invalid Rust type.

use std::alloc::{Layout, alloc, dealloc};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinError {
    AllocationFailed,
    LockFailed,
    UnsupportedPlatform,
    SizeTooLarge(usize),
    WrongPool,
}

impl std::fmt::Display for PinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            PinError::AllocationFailed => write!(f, "memory allocation failed"),
            PinError::LockFailed => write!(f, "memory locking failed"),
            PinError::UnsupportedPlatform => write!(f, "platform does not support memory locking"),
            PinError::SizeTooLarge(size) => write!(f, "size {size} bytes is invalid or too large"),
            PinError::WrongPool => write!(f, "buffer does not belong to this pool"),
        }
    }
}

impl std::error::Error for PinError {}

/// Fixed-length, aligned and best-effort pinned storage of `T` values.
pub struct PinnedBuffer<T: Copy + Default> {
    ptr: NonNull<T>,
    len: usize,
    len_bytes: usize,
    layout: Layout,
    pinned: bool,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy + Default + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Copy + Default + Sync> Sync for PinnedBuffer<T> {}

impl<T: Copy + Default> std::fmt::Debug for PinnedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("len", &self.len)
            .field("len_bytes", &self.len_bytes)
            .field("alignment", &self.layout.align())
            .field("pinned", &self.pinned)
            .finish_non_exhaustive()
    }
}

impl<T: Copy + Default> PinnedBuffer<T> {
    /// Allocates and initializes `len` elements with `T::default()`.
    pub fn new(len: usize) -> Result<Self, PinError> {
        if len == 0 || std::mem::size_of::<T>() == 0
        {
            return Err(PinError::SizeTooLarge(0));
        }
        let len_bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(PinError::SizeTooLarge(usize::MAX))?;
        let alignment = std::mem::align_of::<T>().max(128);
        let layout = Layout::from_size_align(len_bytes, alignment)
            .map_err(|_| PinError::SizeTooLarge(len_bytes))?;

        // Evaluate user-provided Default before allocating, so a panicking
        // implementation cannot leak a partially initialized allocation.
        let value = T::default();
        let raw = unsafe { alloc(layout) } as *mut T;
        let ptr = NonNull::new(raw).ok_or(PinError::AllocationFailed)?;
        for i in 0..len
        {
            // SAFETY: layout holds `len` aligned T slots. Each is initialized
            // exactly once before the buffer becomes observable.
            unsafe { ptr.as_ptr().add(i).write(value) };
        }

        let pinned = platform::try_pin(ptr.as_ptr().cast(), len_bytes);
        Ok(Self {
            ptr,
            len,
            len_bytes,
            layout,
            pinned,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: all `len` elements were initialized during construction.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `&mut self` provides exclusive access to this allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    #[inline]
    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    #[inline]
    pub fn is_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize).is_multiple_of(128)
    }
}

impl<T: Copy + Default> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if self.pinned
        {
            platform::unpin(self.ptr.as_ptr().cast(), self.len_bytes);
        }
        // T is Copy and therefore needs no drop glue.
        unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) };
    }
}

mod platform {
    #[cfg(unix)]
    pub(super) fn try_pin(ptr: *mut u8, len: usize) -> bool {
        unsafe { libc::mlock(ptr.cast(), len) == 0 }
    }

    #[cfg(unix)]
    pub(super) fn unpin(ptr: *mut u8, len: usize) {
        unsafe {
            let _ = libc::munlock(ptr.cast(), len);
        }
    }

    #[cfg(windows)]
    mod windows {
        use std::ffi::c_void;

        #[link(name = "kernel32")]
        unsafe extern "system" {
            pub(super) fn VirtualLock(address: *mut c_void, size: usize) -> i32;
            pub(super) fn VirtualUnlock(address: *mut c_void, size: usize) -> i32;
        }
    }

    #[cfg(windows)]
    pub(super) fn try_pin(ptr: *mut u8, len: usize) -> bool {
        unsafe { windows::VirtualLock(ptr.cast(), len) != 0 }
    }

    #[cfg(windows)]
    pub(super) fn unpin(ptr: *mut u8, len: usize) {
        unsafe {
            let _ = windows::VirtualUnlock(ptr.cast(), len);
        }
    }

    #[cfg(not(any(unix, windows)))]
    pub(super) fn try_pin(_: *mut u8, _: usize) -> bool {
        false
    }

    #[cfg(not(any(unix, windows)))]
    pub(super) fn unpin(_: *mut u8, _: usize) {}
}

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// Pool of typed pinned buffers. Borrowing transfers ownership of one buffer;
/// releasing consumes it and returns it to the originating pool.
pub struct PinnedPool<T: Copy + Default> {
    id: u64,
    buffers: Vec<Option<PinnedBuffer<T>>>,
    free_indices: Vec<usize>,
}

impl<T: Copy + Default> PinnedPool<T> {
    pub fn new(capacity: usize, elem_count: usize) -> Result<Self, PinError> {
        let mut buffers = Vec::with_capacity(capacity);
        let mut free_indices = Vec::with_capacity(capacity);
        for index in 0..capacity
        {
            buffers.push(Some(PinnedBuffer::new(elem_count)?));
            free_indices.push(index);
        }
        Ok(Self {
            id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
            buffers,
            free_indices,
        })
    }

    pub fn borrow(&mut self) -> Result<PooledBuffer<T>, PinError> {
        let index = self.free_indices.pop().ok_or(PinError::AllocationFailed)?;
        let buffer = self.buffers[index]
            .take()
            .ok_or(PinError::AllocationFailed)?;
        Ok(PooledBuffer {
            buffer: Some(buffer),
            index,
            pool_id: self.id,
        })
    }

    pub fn release(&mut self, mut borrowed: PooledBuffer<T>) -> Result<(), PinError> {
        if borrowed.pool_id != self.id
            || borrowed.index >= self.buffers.len()
            || self.buffers[borrowed.index].is_some()
        {
            return Err(PinError::WrongPool);
        }
        let buffer = borrowed.buffer.take().ok_or(PinError::WrongPool)?;
        self.buffers[borrowed.index] = Some(buffer);
        self.free_indices.push(borrowed.index);
        Ok(())
    }

    #[inline]
    pub fn available(&self) -> usize {
        self.free_indices.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffers.len()
    }
}

/// An owned buffer temporarily removed from a [`PinnedPool`]. If it is dropped
/// without being released, its allocation is safely freed and the pool loses
/// that capacity rather than retaining a dangling pointer.
pub struct PooledBuffer<T: Copy + Default> {
    buffer: Option<PinnedBuffer<T>>,
    index: usize,
    pool_id: u64,
}

impl<T: Copy + Default> PooledBuffer<T> {
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.as_ref().map_or(0, PinnedBuffer::len)
    }

    #[inline]
    pub fn len_bytes(&self) -> usize {
        self.buffer.as_ref().map_or(0, PinnedBuffer::len_bytes)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.buffer
            .as_ref()
            .expect("pooled buffer released")
            .as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buffer
            .as_mut()
            .expect("pooled buffer released")
            .as_mut_slice()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    Cpu,
    Pinned,
    GpuUnified,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_buffer_is_initialized_and_writable() {
        let mut buffer = PinnedBuffer::<f32>::new(4).unwrap();
        assert_eq!(buffer.as_slice(), &[0.0; 4]);
        buffer.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buffer.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(buffer.is_aligned());
    }

    #[test]
    fn allocation_size_overflow_is_rejected() {
        let result = PinnedBuffer::<u64>::new(usize::MAX);
        assert!(matches!(result, Err(PinError::SizeTooLarge(_))));
    }

    #[test]
    fn borrow_release_cycle_reuses_buffers() {
        let mut pool = PinnedPool::<f32>::new(2, 8).unwrap();
        let mut first = pool.borrow().unwrap();
        let second = pool.borrow().unwrap();
        assert_eq!(pool.available(), 0);
        first.as_mut_slice()[0] = 7.0;
        pool.release(first).unwrap();
        pool.release(second).unwrap();
        assert_eq!(pool.available(), 2);
        let reused_a = pool.borrow().unwrap();
        let reused_b = pool.borrow().unwrap();
        assert!(reused_a.as_slice().contains(&7.0) || reused_b.as_slice().contains(&7.0));
    }

    #[test]
    fn buffer_from_another_pool_is_rejected() {
        let mut a = PinnedPool::<u32>::new(1, 4).unwrap();
        let mut b = PinnedPool::<u32>::new(1, 4).unwrap();
        let borrowed = a.borrow().unwrap();
        assert_eq!(b.release(borrowed), Err(PinError::WrongPool));
    }
}
