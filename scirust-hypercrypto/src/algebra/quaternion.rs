//! Four-component **associative** quaternion `Quat<W>` over `Z/2^k`, used only
//! as Control D (spec Phase-1 controls). Quaternions are associative, so this is
//! a comparison baseline for structural probes. Any observed difference between
//! quaternion and octonion structure must NOT be read as a security benefit of
//! non-associativity (per the mission).
//!
//! Basis `(1, i, j, k)` with `i² = j² = k² = -1`, `ij = k`, `jk = i`, `ki = j`.

use super::word::Word;

/// A quaternion: `c[0]` scalar, `c[1..4]` the units `i, j, k`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Quat<W: Word> {
    /// Coefficients `(1, i, j, k)`.
    pub c: [W; 4],
}

#[allow(clippy::should_implement_trait)]
impl<W: Word> Quat<W> {
    /// Zero.
    pub fn zero() -> Self {
        Quat { c: [W::ZERO; 4] }
    }
    /// Build from four `u64`s (masked).
    pub fn from_u64s(v: [u64; 4]) -> Self {
        Quat {
            c: std::array::from_fn(|i| W::from_u64(v[i])),
        }
    }
    /// Coefficients as `u64`s.
    pub fn to_u64s(self) -> [u64; 4] {
        std::array::from_fn(|i| self.c[i].to_u64())
    }
    /// Component-wise wrapping add.
    pub fn add(self, o: Self) -> Self {
        Quat {
            c: std::array::from_fn(|i| self.c[i].wadd(o.c[i])),
        }
    }
    /// Bitwise XOR.
    pub fn xor(self, o: Self) -> Self {
        Quat {
            c: std::array::from_fn(|i| self.c[i].xor(o.c[i])),
        }
    }
    /// Hamilton product (associative). Written out explicitly (no table).
    pub fn mul(self, o: Self) -> Self {
        let a = self.c;
        let b = o.c;
        // (a0,a1,a2,a3) * (b0,b1,b2,b3)
        let s = |x: W, y: W| x.wmul(y);
        let c0 = s(a[0], b[0])
            .wsub(s(a[1], b[1]))
            .wsub(s(a[2], b[2]))
            .wsub(s(a[3], b[3]));
        let c1 = s(a[0], b[1])
            .wadd(s(a[1], b[0]))
            .wadd(s(a[2], b[3]))
            .wsub(s(a[3], b[2]));
        let c2 = s(a[0], b[2])
            .wsub(s(a[1], b[3]))
            .wadd(s(a[2], b[0]))
            .wadd(s(a[3], b[1]));
        let c3 = s(a[0], b[3])
            .wadd(s(a[1], b[2]))
            .wsub(s(a[2], b[1]))
            .wadd(s(a[3], b[0]));
        Quat {
            c: [c0, c1, c2, c3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::W8;

    #[test]
    fn associative_and_units() {
        let i = Quat::<W8>::from_u64s([0, 1, 0, 0]);
        let j = Quat::<W8>::from_u64s([0, 0, 1, 0]);
        let k = Quat::<W8>::from_u64s([0, 0, 0, 1]);
        // ij = k
        assert_eq!(i.mul(j), k);
        // i(jk) = (ij)k : associativity holds for quaternions
        assert_eq!(i.mul(j.mul(k)), i.mul(j).mul(k));
        // i^2 = -1
        assert_eq!(i.mul(i).to_u64s(), [255, 0, 0, 0]);
    }
}
