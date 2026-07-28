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
}
