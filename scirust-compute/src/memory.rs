/// Memory domain containing a compute buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MemorySpace {
    Host,
    HostPinned,
    Device,
    Unified,
}

/// Physical tensor layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Layout {
    ContiguousRowMajor,
    Strided,
}

/// Host-page preference requested by a prepared kernel/runtime.
///
/// These values describe intent only. In particular,
/// [`TransparentHugePageHint`](Self::TransparentHugePageHint) is not a guarantee
/// that the operating system will back a region with any particular huge-page
/// size; the allocator/backend must report what it actually obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HostPagePolicy {
    Default,
    TransparentHugePageHint,
    AvoidTransparentHugePages,
}

/// Allocation-independent policy for host scratch/model buffers.
///
/// The contract is deliberately separate from an allocator implementation so
/// Linux THP, explicit huge pages, pinned memory, NUMA placement or embedded
/// allocators can satisfy the same request without leaking OS-specific syscalls
/// into kernel APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostMemoryPolicy {
    alignment_bytes: usize,
    page_policy: HostPagePolicy,
    lock_resident: bool,
}

impl HostMemoryPolicy {
    /// Construct a validated policy. Alignment must be a non-zero power of two.
    pub const fn new(
        alignment_bytes: usize,
        page_policy: HostPagePolicy,
        lock_resident: bool,
    ) -> Result<Self, MemoryPolicyError> {
        if alignment_bytes == 0 || !alignment_bytes.is_power_of_two()
        {
            return Err(MemoryPolicyError::InvalidAlignment(alignment_bytes));
        }
        Ok(Self {
            alignment_bytes,
            page_policy,
            lock_resident,
        })
    }

    /// Conservative host policy with natural byte alignment and no page hint.
    pub const fn default_host() -> Self {
        Self {
            alignment_bytes: 1,
            page_policy: HostPagePolicy::Default,
            lock_resident: false,
        }
    }

    /// Typical cache-line/SIMD-friendly host policy without making ISA claims.
    pub const fn aligned_64() -> Self {
        Self {
            alignment_bytes: 64,
            page_policy: HostPagePolicy::Default,
            lock_resident: false,
        }
    }

    pub const fn alignment_bytes(self) -> usize {
        self.alignment_bytes
    }

    pub const fn page_policy(self) -> HostPagePolicy {
        self.page_policy
    }

    pub const fn lock_resident(self) -> bool {
        self.lock_resident
    }

    /// Return a copy with a different page preference.
    pub const fn with_page_policy(mut self, page_policy: HostPagePolicy) -> Self {
        self.page_policy = page_policy;
        self
    }

    /// Return a copy requesting or releasing resident-memory locking.
    pub const fn with_lock_resident(mut self, lock_resident: bool) -> Self {
        self.lock_resident = lock_resident;
        self
    }
}

impl Default for HostMemoryPolicy {
    fn default() -> Self {
        Self::default_host()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPolicyError {
    InvalidAlignment(usize),
}

impl core::fmt::Display for MemoryPolicyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self
        {
            Self::InvalidAlignment(alignment) => write!(
                f,
                "host memory alignment must be a non-zero power of two, got {alignment}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MemoryPolicyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_spaces_are_distinct() {
        assert_ne!(MemorySpace::Host, MemorySpace::Device);
        assert_ne!(MemorySpace::HostPinned, MemorySpace::Unified);
    }

    #[test]
    fn layouts_are_distinct() {
        assert_ne!(Layout::ContiguousRowMajor, Layout::Strided);
    }

    #[test]
    fn host_policy_validates_alignment() {
        assert!(HostMemoryPolicy::new(64, HostPagePolicy::Default, false).is_ok());
        assert_eq!(
            HostMemoryPolicy::new(48, HostPagePolicy::Default, false),
            Err(MemoryPolicyError::InvalidAlignment(48))
        );
        assert_eq!(
            HostMemoryPolicy::new(0, HostPagePolicy::Default, false),
            Err(MemoryPolicyError::InvalidAlignment(0))
        );
    }

    #[test]
    fn thp_is_encoded_as_a_hint_not_a_guarantee() {
        let policy = HostMemoryPolicy::aligned_64()
            .with_page_policy(HostPagePolicy::TransparentHugePageHint);
        assert_eq!(policy.alignment_bytes(), 64);
        assert_eq!(
            policy.page_policy(),
            HostPagePolicy::TransparentHugePageHint
        );
        assert!(!policy.lock_resident());
    }
}
