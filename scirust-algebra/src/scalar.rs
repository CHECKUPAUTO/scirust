use crate::core::{Field, Ring};

impl Ring for f64 {
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn add(&self, rhs: &Self) -> Self { *self + *rhs }
    #[inline] fn neg(&self) -> Self { -*self }
    #[inline] fn mul(&self, rhs: &Self) -> Self { *self * *rhs }
}

impl Field for f64 {
    #[inline]
    fn reciprocal(&self) -> Option<Self> {
        if *self == 0.0 { None } else { Some(1.0 / *self) }
    }
}
