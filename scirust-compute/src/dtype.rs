/// Scalar type stored in a compute buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    Bf16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl DType {
    /// Storage width of one scalar value.
    pub const fn size_bytes(self) -> usize {
        match self
        {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn scalar_widths_are_explicit() {
        assert_eq!(DType::Bool.size_bytes(), 1);
        assert_eq!(DType::Bf16.size_bytes(), 2);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F64.size_bytes(), 8);
    }
}
