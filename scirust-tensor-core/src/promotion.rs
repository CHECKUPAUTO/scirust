use std::fmt;

use scirust_compute::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromotionError {
    pub lhs: DType,
    pub rhs: DType,
}

impl fmt::Display for PromotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "no safe implicit promotion for {:?} and {:?}",
            self.lhs, self.rhs
        )
    }
}

impl std::error::Error for PromotionError {}

/// Deterministic SciRust dtype-promotion rule.
///
/// The rule is intentionally conservative: it never silently promotes an `i64`
/// or `u64` to a floating type because that can lose integer precision. F16 and
/// BF16 meet at F32. Signed/unsigned integer mixtures choose the smallest signed
/// type that can represent both domains, or fail when no such built-in type
/// exists (`u64` mixed with any signed integer).
pub fn promote_types(lhs: DType, rhs: DType) -> Result<DType, PromotionError> {
    use DType::*;

    if lhs == rhs
    {
        return Ok(lhs);
    }

    if lhs == Bool || rhs == Bool
    {
        return Err(PromotionError { lhs, rhs });
    }

    let result = match (lhs, rhs)
    {
        (F64, F64) => F64,
        (F64, other) | (other, F64) if !matches!(other, U64 | I64) => F64,

        (F32, other) | (other, F32)
            if matches!(other, F16 | Bf16 | U8 | I8 | U16 | I16 | U32 | I32) =>
        {
            F32
        },
        (F16, Bf16) | (Bf16, F16) => F32,
        (F16, other) | (other, F16) if matches!(other, U8 | I8 | U16 | I16) => F16,
        (Bf16, other) | (other, Bf16) if matches!(other, U8 | I8) => Bf16,

        (U8, U16) | (U16, U8) => U16,
        (U8, U32) | (U32, U8) | (U16, U32) | (U32, U16) => U32,
        (U8, U64) | (U64, U8) | (U16, U64) | (U64, U16) | (U32, U64) | (U64, U32) => U64,

        (I8, I16) | (I16, I8) => I16,
        (I8, I32) | (I32, I8) | (I16, I32) | (I32, I16) => I32,
        (I8, I64) | (I64, I8) | (I16, I64) | (I64, I16) | (I32, I64) | (I64, I32) => I64,

        (U8, I8) | (I8, U8) => I16,
        (U8, I16) | (I16, U8) => I16,
        (U16, I8) | (I8, U16) | (U16, I16) | (I16, U16) => I32,
        (U8, I32) | (I32, U8) | (U16, I32) | (I32, U16) => I32,
        (U32, I8) | (I8, U32) | (U32, I16) | (I16, U32) | (U32, I32) | (I32, U32) => I64,
        (U8, I64) | (I64, U8) | (U16, I64) | (I64, U16) | (U32, I64) | (I64, U32) => I64,

        _ => return Err(PromotionError { lhs, rhs }),
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_low_precision_float_promotes_to_f32() {
        assert_eq!(promote_types(DType::F16, DType::Bf16), Ok(DType::F32));
    }

    #[test]
    fn signed_unsigned_promotion_preserves_domain() {
        assert_eq!(promote_types(DType::U16, DType::I16), Ok(DType::I32));
        assert_eq!(promote_types(DType::U32, DType::I32), Ok(DType::I64));
    }

    #[test]
    fn dangerous_integer_float_and_u64_signed_mixes_are_explicit() {
        assert!(promote_types(DType::I64, DType::F32).is_err());
        assert!(promote_types(DType::U64, DType::I64).is_err());
    }
}
