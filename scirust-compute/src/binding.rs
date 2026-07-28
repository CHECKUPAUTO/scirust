#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Debug)]
pub struct BufferBinding<'a, B> {
    pub slot: u32,
    pub buffer: &'a B,
    pub offset_bytes: usize,
    pub length_bytes: usize,
    pub access: BufferAccess,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_preserves_range_and_access() {
        let buffer = [0u8; 16];
        let binding = BufferBinding {
            slot: 2,
            buffer: &buffer,
            offset_bytes: 4,
            length_bytes: 8,
            access: BufferAccess::ReadOnly,
        };

        assert_eq!(binding.slot, 2);
        assert_eq!(binding.offset_bytes, 4);
        assert_eq!(binding.length_bytes, 8);
        assert_eq!(binding.access, BufferAccess::ReadOnly);
    }
}
