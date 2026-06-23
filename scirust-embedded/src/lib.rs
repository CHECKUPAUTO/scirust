#![cfg_attr(not(test), no_std)]

/// Virgule fixe Q16.16 pour le déterminisme sur microcontrôleurs sans FPU.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Q16_16(pub i32);

impl Q16_16 {
    pub const FRAC_BITS: u32 = 16;

    #[inline(always)]
    pub fn from_f32(f: f32) -> Self {
        Self((f * 65536.0 + (if f >= 0.0 { 0.5 } else { -0.5 })) as i32)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / 65536.0
    }

    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        let res = (self.0 as i64 * other.0 as i64) >> 16;
        Self(res as i32)
    }
}

/// Tenseur statique sur la pile via micro-génériques.
#[derive(Debug, Clone, Copy)]
pub struct StaticTensor<T, const R: usize, const C: usize> {
    pub data: [[T; C]; R],
}

impl<T: Copy + Default, const R: usize, const C: usize> StaticTensor<T, R, C> {
    /// Initialise un tenseur.
    pub fn new(val: T) -> Self {
        Self {
            data: [[val; C]; R],
        }
    }
}

/// Couche Linéaire (Dense) totalement alloc-free.
pub struct StaticLinear<const IN: usize, const OUT: usize> {
    pub weight: StaticTensor<Q16_16, OUT, IN>,
    pub bias: StaticTensor<Q16_16, 1, OUT>,
}

impl<const IN: usize, const OUT: usize> StaticLinear<IN, OUT> {
    /// Forward pass déterministe en virgule fixe.
    pub fn forward<const B: usize>(
        &self,
        input: &StaticTensor<Q16_16, B, IN>,
        output: &mut StaticTensor<Q16_16, B, OUT>,
    ) {
        for b in 0..B
        {
            for j in 0..OUT
            {
                let mut acc = self.bias.data[0][j];
                for i in 0..IN
                {
                    acc = acc.add(input.data[b][i].mul(self.weight.data[j][i]));
                }
                output.data[b][j] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q16_roundtrips_and_arithmetic() {
        let a = Q16_16::from_f32(1.5);
        let b = Q16_16::from_f32(2.0);
        assert!((a.to_f32() - 1.5).abs() < 1e-3);
        assert!((a.add(b).to_f32() - 3.5).abs() < 1e-3);
        assert!((a.mul(b).to_f32() - 3.0).abs() < 1e-3);
    }

    #[test]
    fn static_linear_forward_is_deterministic_fixed_point() {
        let weight = StaticTensor::<Q16_16, 2, 2> {
            data: [
                [Q16_16::from_f32(1.0), Q16_16::from_f32(0.0)],
                [Q16_16::from_f32(0.0), Q16_16::from_f32(2.0)],
            ],
        };
        let bias = StaticTensor::<Q16_16, 1, 2> {
            data: [[Q16_16::from_f32(0.5), Q16_16::from_f32(-1.0)]],
        };
        let layer = StaticLinear::<2, 2> { weight, bias };
        let input = StaticTensor::<Q16_16, 1, 2> {
            data: [[Q16_16::from_f32(3.0), Q16_16::from_f32(4.0)]],
        };
        let mut out = StaticTensor::<Q16_16, 1, 2>::new(Q16_16(0));
        layer.forward(&input, &mut out);
        // y0 = 0.5 + 3*1 + 4*0 = 3.5 ; y1 = -1 + 3*0 + 4*2 = 7
        assert!((out.data[0][0].to_f32() - 3.5).abs() < 1e-2);
        assert!((out.data[0][1].to_f32() - 7.0).abs() < 1e-2);
    }
}
