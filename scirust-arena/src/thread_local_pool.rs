//! Lock-free thread-local scratch arena reuse.
//!
//! Each operating-system/Rayon worker thread lazily owns one [`PinnedArena`]. A
//! scope resets the arena before use and again on scope exit, including unwinding
//! through a panic. The higher-ranked closure signature keeps references borrowed
//! from the arena from escaping the scope.

use crate::PinnedArena;
use std::cell::RefCell;

thread_local! {
    static THREAD_ARENA: RefCell<Option<PinnedArena>> = const { RefCell::new(None) };
}

/// Error returned before entering a thread-local arena scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadLocalArenaError {
    ZeroCapacity,
    ReentrantScope,
}

impl core::fmt::Display for ThreadLocalArenaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::ZeroCapacity => write!(f, "thread-local arena capacity must be non-zero"),
            Self::ReentrantScope => write!(f, "thread-local arena scope is already borrowed"),
        }
    }
}

impl std::error::Error for ThreadLocalArenaError {}

struct ResetOnDrop<'a> {
    arena: &'a mut PinnedArena,
}

impl core::ops::Deref for ResetOnDrop<'_> {
    type Target = PinnedArena;

    fn deref(&self) -> &Self::Target {
        self.arena
    }
}

impl core::ops::DerefMut for ResetOnDrop<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.arena
    }
}

impl Drop for ResetOnDrop<'_> {
    fn drop(&mut self) {
        self.arena.reset();
    }
}

/// Run `f` with this thread's reusable scratch arena.
///
/// The arena is allocated lazily. If an existing arena is smaller than
/// `min_capacity_bytes`, it is replaced before entering the scope; otherwise the
/// same backing allocation is reused. No lock is shared between threads.
///
/// The `for<'arena>` closure bound means the return type `R` cannot contain a
/// reference tied to the temporary arena borrow, preventing an allocation from
/// escaping and later being used after reset.
pub fn with_thread_arena<R, F>(
    min_capacity_bytes: usize,
    f: F,
) -> Result<R, ThreadLocalArenaError>
where
    F: for<'arena> FnOnce(&'arena mut PinnedArena) -> R,
{
    if min_capacity_bytes == 0
    {
        return Err(ThreadLocalArenaError::ZeroCapacity);
    }

    THREAD_ARENA.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|_| ThreadLocalArenaError::ReentrantScope)?;

        let needs_replacement = slot
            .as_ref()
            .is_none_or(|arena| arena.capacity() < min_capacity_bytes);
        if needs_replacement
        {
            *slot = Some(PinnedArena::new(min_capacity_bytes));
        }

        let arena = slot.as_mut().expect("thread-local arena initialized");
        arena.reset();
        let mut guard = ResetOnDrop { arena };
        Ok(f(&mut guard))
    })
}

/// Current thread's retained arena capacity, or zero before first use.
pub fn thread_arena_capacity() -> Result<usize, ThreadLocalArenaError> {
    THREAD_ARENA.with(|slot| {
        let slot = slot
            .try_borrow()
            .map_err(|_| ThreadLocalArenaError::ReentrantScope)?;
        Ok(slot.as_ref().map_or(0, PinnedArena::capacity))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_scopes_reuse_capacity_and_start_reset() {
        let first_capacity = with_thread_arena(4096, |arena| {
            assert_eq!(arena.allocated(), 0);
            let values = arena.alloc_slice_fill::<f32>(128, 2.0).unwrap();
            assert!(values.iter().all(|&value| value == 2.0));
            assert!(arena.allocated() > 0);
            arena.capacity()
        })
        .unwrap();

        let second_capacity = with_thread_arena(1024, |arena| {
            assert_eq!(arena.allocated(), 0);
            assert_eq!(arena.alloc_count(), 0);
            arena.capacity()
        })
        .unwrap();

        assert_eq!(first_capacity, second_capacity);
    }

    #[test]
    fn scope_grows_only_when_required() {
        let small = with_thread_arena(1024, |arena| arena.capacity()).unwrap();
        let large = with_thread_arena(small + 1024, |arena| arena.capacity()).unwrap();
        assert!(large >= small + 1024);
        let reused = with_thread_arena(1024, |arena| arena.capacity()).unwrap();
        assert_eq!(large, reused);
    }

    #[test]
    fn panic_still_resets_arena() {
        let result = std::panic::catch_unwind(|| {
            let _ = with_thread_arena(4096, |arena| {
                let _scratch = arena.alloc_slice_fill::<u64>(16, 7).unwrap();
                panic!("intentional test panic");
            });
        });
        assert!(result.is_err());

        with_thread_arena(1024, |arena| {
            assert_eq!(arena.allocated(), 0);
            assert_eq!(arena.alloc_count(), 0);
        })
        .unwrap();
    }

    #[test]
    fn nested_scope_is_rejected_instead_of_aliasing() {
        with_thread_arena(1024, |_arena| {
            assert_eq!(
                with_thread_arena(1024, |_| ()),
                Err(ThreadLocalArenaError::ReentrantScope)
            );
        })
        .unwrap();
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert_eq!(
            with_thread_arena(0, |_| ()),
            Err(ThreadLocalArenaError::ZeroCapacity)
        );
    }
}
