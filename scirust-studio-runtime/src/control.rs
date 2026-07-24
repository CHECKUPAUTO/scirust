//! Execution control: the cooperative-cancellation hook every
//! [`crate::CapabilityAdapter::execute`] call receives.
//!
//! Phase 2A's adapters call `simulate`/`simulate_rosenbrock` etc. as a
//! single blocking call with no progress callback, so the only cancellation
//! point available *today* is before that call starts — checked by every
//! adapter in this crate. Fine-grained mid-run cancellation (chunking the
//! integration and checking between chunks, or a real worker process that
//! can be killed) is Phase 2B's job; this type's shape does not need to
//! change to support that later; only what calls `is_cancelled()`, and how
//! often, will.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A cancellation flag shared between whoever is driving an execution and
/// the adapter running it.
#[derive(Debug, Clone, Default)]
pub struct ExecutionControl {
    cancelled: Arc<AtomicBool>,
}

impl ExecutionControl {
    /// A fresh, not-yet-cancelled control.
    pub fn new() -> Self {
        ExecutionControl::default()
    }

    /// Request cancellation. Idempotent.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_uncancelled() {
        assert!(!ExecutionControl::new().is_cancelled());
    }

    #[test]
    fn cancel_is_visible_through_a_clone() {
        let control = ExecutionControl::new();
        let clone = control.clone();
        control.cancel();
        assert!(clone.is_cancelled());
    }

    #[test]
    fn cancel_is_idempotent() {
        let control = ExecutionControl::new();
        control.cancel();
        control.cancel();
        assert!(control.is_cancelled());
    }
}
