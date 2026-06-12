#![no_std]

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
