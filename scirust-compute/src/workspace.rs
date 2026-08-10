//! Backend-neutral contracts for caller-owned reusable kernel workspaces.
//!
//! A workspace separates one-time memory provisioning from repeated kernel
//! execution. Kernels can publish the storage they require through
//! [`WorkspaceSpec`] and then validate a caller-owned buffer before entering a
//! hot path. The contract itself performs no allocation and remains available in
//! `no_std` builds.

use core::fmt;

/// Storage requirements published by a kernel or prepared execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkspaceSpec {
    size_bytes: usize,
    align_bytes: usize,
}

impl WorkspaceSpec {
    /// Creates a workspace requirement.
    ///
    /// `align_bytes` must be a non-zero power of two.
    pub const fn new(size_bytes: usize, align_bytes: usize) -> Self {
        assert!(align_bytes != 0, "workspace alignment must be non-zero");
        assert!(
            align_bytes.is_power_of_two(),
            "workspace alignment must be a power of two"
        );
        Self {
            size_bytes,
            align_bytes,
        }
    }

    /// A workspace requirement that needs no scratch storage.
    pub const fn empty() -> Self {
        Self {
            size_bytes: 0,
            align_bytes: 1,
        }
    }

    /// Required usable byte count.
    pub const fn size_bytes(self) -> usize {
        self.size_bytes
    }

    /// Required base-address alignment in bytes.
    pub const fn align_bytes(self) -> usize {
        self.align_bytes
    }

    /// Returns whether this requirement needs any storage.
    pub const fn is_empty(self) -> bool {
        self.size_bytes == 0
    }

    /// Combines two independent workspace requirements into one sequential
    /// layout, including the padding required before the second region.
    pub const fn then(self, next: Self) -> Option<Self> {
        let align = if self.align_bytes > next.align_bytes
        {
            self.align_bytes
        }
        else
        {
            next.align_bytes
        };
        let second_offset = match align_up(self.size_bytes, next.align_bytes)
        {
            Some(offset) => offset,
            None => return None,
        };
        let size_bytes = match second_offset.checked_add(next.size_bytes)
        {
            Some(size) => size,
            None => return None,
        };
        Some(Self {
            size_bytes,
            align_bytes: align,
        })
    }
}

/// Validation error returned before a kernel touches a caller-owned workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceError {
    /// The provided slice is smaller than the kernel's requirement.
    TooSmall {
        required_bytes: usize,
        provided_bytes: usize,
    },
    /// The base address does not satisfy the requested alignment.
    Misaligned {
        required_alignment: usize,
        address_remainder: usize,
    },
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self
        {
            Self::TooSmall {
                required_bytes,
                provided_bytes,
            } => write!(
                f,
                "workspace too small: requires {required_bytes} bytes, got {provided_bytes}"
            ),
            Self::Misaligned {
                required_alignment,
                address_remainder,
            } => write!(
                f,
                "workspace base is not aligned to {required_alignment} bytes (remainder {address_remainder})"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WorkspaceError {}

/// A validated, caller-owned scratch buffer for one kernel invocation.
///
/// Construction checks size and alignment once. Kernels can then reuse the
/// returned mutable byte slice without allocating temporary storage.
pub struct KernelWorkspace<'a> {
    bytes: &'a mut [u8],
}

impl<'a> KernelWorkspace<'a> {
    /// Validates `bytes` against `spec` and returns a workspace restricted to
    /// the usable prefix required by the kernel.
    pub fn new(bytes: &'a mut [u8], spec: WorkspaceSpec) -> Result<Self, WorkspaceError> {
        if bytes.len() < spec.size_bytes
        {
            return Err(WorkspaceError::TooSmall {
                required_bytes: spec.size_bytes,
                provided_bytes: bytes.len(),
            });
        }
        if spec.size_bytes != 0
        {
            let remainder = (bytes.as_ptr() as usize) & (spec.align_bytes - 1);
            if remainder != 0
            {
                return Err(WorkspaceError::Misaligned {
                    required_alignment: spec.align_bytes,
                    address_remainder: remainder,
                });
            }
        }
        Ok(Self {
            bytes: &mut bytes[..spec.size_bytes],
        })
    }

    /// Returns the validated scratch storage.
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes
    }

    /// Returns the validated scratch storage mutably.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.bytes
    }

    /// Number of usable bytes in this validated workspace.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether this workspace contains no usable bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

const fn align_up(value: usize, alignment: usize) -> Option<usize> {
    let mask = alignment - 1;
    match value.checked_add(mask)
    {
        Some(sum) => Some(sum & !mask),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(align(64))]
    struct Aligned64<const N: usize>([u8; N]);

    #[test]
    fn workspace_accepts_aligned_reusable_storage() {
        let spec = WorkspaceSpec::new(128, 64);
        let mut storage = Aligned64([0_u8; 256]);
        let mut workspace = KernelWorkspace::new(&mut storage.0, spec).expect("valid workspace");
        assert_eq!(workspace.len(), 128);
        workspace.as_bytes_mut()[127] = 7;
        assert_eq!(workspace.as_bytes()[127], 7);
    }

    #[test]
    fn workspace_rejects_short_storage() {
        let spec = WorkspaceSpec::new(128, 1);
        let mut storage = [0_u8; 64];
        assert_eq!(
            KernelWorkspace::new(&mut storage, spec).err(),
            Some(WorkspaceError::TooSmall {
                required_bytes: 128,
                provided_bytes: 64,
            })
        );
    }

    #[test]
    fn sequential_specs_include_alignment_padding() {
        let left = WorkspaceSpec::new(65, 1);
        let right = WorkspaceSpec::new(64, 64);
        let combined = left.then(right).expect("workspace layout fits usize");
        assert_eq!(combined.align_bytes(), 64);
        assert_eq!(combined.size_bytes(), 192);
    }

    #[test]
    fn empty_workspace_accepts_empty_slice() {
        let mut storage = [];
        let workspace =
            KernelWorkspace::new(&mut storage, WorkspaceSpec::empty()).expect("empty workspace");
        assert!(workspace.is_empty());
    }
}
