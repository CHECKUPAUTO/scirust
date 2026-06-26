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

    // ---- Q16.16 raw-integer oracle tests -------------------------------------
    //
    // Q16.16 stores `value * 2^16` in an i32, so 1.0 == 65536, 0.5 == 32768.
    // Asserting on the raw `.0` field pins the *exact* fixed-point encoding,
    // not a float approximation, leaving no tolerance to hide a regression.

    #[test]
    fn frac_bits_is_sixteen() {
        assert_eq!(Q16_16::FRAC_BITS, 16);
        // The scale factor implied by FRAC_BITS must match the literals used in
        // from_f32/to_f32 (65536.0 == 2^16).
        assert_eq!(1i32 << Q16_16::FRAC_BITS, 65536);
    }

    #[test]
    fn from_f32_encodes_exact_grid_points() {
        // Integers and halves are exactly representable: f * 2^16.
        assert_eq!(Q16_16::from_f32(0.0).0, 0);
        assert_eq!(Q16_16::from_f32(1.0).0, 65536);
        assert_eq!(Q16_16::from_f32(-1.0).0, -65536);
        assert_eq!(Q16_16::from_f32(0.5).0, 32768);
        assert_eq!(Q16_16::from_f32(-0.5).0, -32768);
        assert_eq!(Q16_16::from_f32(2.0).0, 131072);
        assert_eq!(Q16_16::from_f32(-2.0).0, -131072);
    }

    #[test]
    fn from_f32_rounds_half_away_from_zero() {
        // 0.1 * 65536 = 6553.6  -> round to 6554 (nearest).
        assert_eq!(Q16_16::from_f32(0.1).0, 6554);
        // Sign symmetry: -0.1 * 65536 = -6553.6 -> -6554.
        assert_eq!(Q16_16::from_f32(-0.1).0, -6554);
        // Exact half-way point at the LSB: (1.5 / 2^16) * 2^16 = 1.5.
        // Round-half-away-from-zero must push |x| up, giving magnitude 2.
        assert_eq!(Q16_16::from_f32(1.5 / 65536.0).0, 2);
        assert_eq!(Q16_16::from_f32(-1.5 / 65536.0).0, -2);
    }

    #[test]
    fn to_f32_is_inverse_of_raw_scale() {
        assert_eq!(Q16_16(65536).to_f32(), 1.0);
        assert_eq!(Q16_16(-32768).to_f32(), -0.5);
        assert_eq!(Q16_16(0).to_f32(), 0.0);
        // Round-trip an exact grid value with no error at all.
        assert_eq!(Q16_16::from_f32(0.25).to_f32(), 0.25);
    }

    #[test]
    fn default_and_eq_behave() {
        // Derived Default must be the additive identity (raw 0).
        assert_eq!(Q16_16::default(), Q16_16(0));
        assert_eq!(Q16_16::default().to_f32(), 0.0);
        // Derived Eq compares the underlying fixed-point bits.
        assert_eq!(Q16_16::from_f32(1.0), Q16_16(65536));
        assert_ne!(Q16_16::from_f32(1.0), Q16_16::from_f32(1.0000305)); // differs by 2 LSB
    }

    #[test]
    fn add_is_exact_and_saturates() {
        // Same-scale addition is plain integer addition.
        assert_eq!(Q16_16::from_f32(1.5).add(Q16_16::from_f32(2.0)).0, 229376); // 3.5 * 2^16
        assert_eq!(Q16_16::from_f32(-1.0).add(Q16_16::from_f32(0.25)).0, -49152); // -0.75 * 2^16
        // Saturating, not wrapping, at both ends.
        assert_eq!(Q16_16(i32::MAX).add(Q16_16(1)), Q16_16(i32::MAX));
        assert_eq!(Q16_16(i32::MIN).add(Q16_16(-1)), Q16_16(i32::MIN));
        assert_eq!(Q16_16(i32::MAX).add(Q16_16(i32::MAX)), Q16_16(i32::MAX));
    }

    #[test]
    fn mul_scales_back_by_frac_bits() {
        // (a*b) >> 16 keeps the result in Q16.16.
        // 0.5 * 0.5 = 0.25 -> 0.25 * 2^16 = 16384.
        assert_eq!(Q16_16::from_f32(0.5).mul(Q16_16::from_f32(0.5)).0, 16384);
        // 2.0 * 3.0 = 6.0 -> 6.0 * 2^16 = 393216.
        assert_eq!(Q16_16::from_f32(2.0).mul(Q16_16::from_f32(3.0)).0, 393216);
        // Multiplying by 1.0 is the identity on the raw value.
        assert_eq!(Q16_16(12345).mul(Q16_16::from_f32(1.0)).0, 12345);
        // Multiplying by 0 yields 0.
        assert_eq!(Q16_16(98765).mul(Q16_16(0)).0, 0);
    }

    #[test]
    fn mul_truncates_toward_negative_infinity() {
        // The product uses an arithmetic right shift, so rounding is toward -inf
        // and is therefore asymmetric across sign. 0.1 ~ 6554 raw.
        // 6554 * 6554 = 42954916 ; >> 16 = 655 (0.1 * 0.1 ~ 0.01).
        assert_eq!(Q16_16::from_f32(0.1).mul(Q16_16::from_f32(0.1)).0, 655);
        // 6554 * -6554 = -42954916 ; arithmetic >> 16 = -656 (floor, not -655).
        assert_eq!(Q16_16::from_f32(-0.1).mul(Q16_16::from_f32(0.1)).0, -656);
        // No precision is lost when the low 16 product bits are already zero.
        // 1.5 * 2.0 = 3.0 -> 196608, exact.
        assert_eq!(Q16_16::from_f32(1.5).mul(Q16_16::from_f32(2.0)).0, 196608);
    }

    // ---- StaticTensor --------------------------------------------------------

    #[test]
    fn static_tensor_new_fills_every_cell() {
        let t = StaticTensor::<Q16_16, 3, 4>::new(Q16_16::from_f32(2.5));
        for row in 0..3
        {
            for col in 0..4
            {
                assert_eq!(t.data[row][col].0, 163840); // 2.5 * 2^16
            }
        }
        // A zero-initialised tensor is all raw-zero.
        let z = StaticTensor::<Q16_16, 2, 2>::new(Q16_16(0));
        assert_eq!(z.data, [[Q16_16(0); 2]; 2]);
    }

    // ---- StaticLinear::forward ----------------------------------------------

    #[test]
    fn forward_multi_batch_rectangular_exact() {
        // OUT=2, IN=3. weight[j][i] is the coefficient of input i for output j.
        let weight = StaticTensor::<Q16_16, 2, 3> {
            data: [
                [
                    Q16_16::from_f32(1.0),
                    Q16_16::from_f32(2.0),
                    Q16_16::from_f32(0.5),
                ],
                [
                    Q16_16::from_f32(-1.0),
                    Q16_16::from_f32(0.0),
                    Q16_16::from_f32(3.0),
                ],
            ],
        };
        let bias = StaticTensor::<Q16_16, 1, 2> {
            data: [[Q16_16::from_f32(0.5), Q16_16::from_f32(-2.0)]],
        };
        let layer = StaticLinear::<3, 2> { weight, bias };
        let input = StaticTensor::<Q16_16, 2, 3> {
            data: [
                [
                    Q16_16::from_f32(2.0),
                    Q16_16::from_f32(1.0),
                    Q16_16::from_f32(4.0),
                ],
                [
                    Q16_16::from_f32(0.0),
                    Q16_16::from_f32(-3.0),
                    Q16_16::from_f32(2.0),
                ],
            ],
        };
        let mut out = StaticTensor::<Q16_16, 2, 2>::new(Q16_16(0));
        layer.forward(&input, &mut out);

        // All operands lie on the Q16.16 grid and every product is exact, so the
        // accumulator is exact and we assert the raw i32 directly.
        // b0,o0 = 0.5 + (2*1 + 1*2 + 4*0.5)        = 0.5 + 6  =  6.5 -> 425984
        // b0,o1 = -2.0 + (2*-1 + 1*0 + 4*3)        = -2 + 10  =  8.0 -> 524288
        // b1,o0 = 0.5 + (0*1 + -3*2 + 2*0.5)       = 0.5 - 5  = -4.5 -> -294912
        // b1,o1 = -2.0 + (0*-1 + -3*0 + 2*3)       = -2 + 6   =  4.0 -> 262144
        assert_eq!(out.data[0][0].0, 425984);
        assert_eq!(out.data[0][1].0, 524288);
        assert_eq!(out.data[1][0].0, -294912);
        assert_eq!(out.data[1][1].0, 262144);
    }

    #[test]
    fn forward_uses_bias_only_when_in_is_zero() {
        // With IN == 0 the dot-product loop runs zero times, so each output is
        // exactly its bias term. This pins that the bias is the accumulator seed.
        let weight = StaticTensor::<Q16_16, 2, 0> { data: [[], []] };
        let bias = StaticTensor::<Q16_16, 1, 2> {
            data: [[Q16_16::from_f32(0.75), Q16_16::from_f32(-3.0)]],
        };
        let layer = StaticLinear::<0, 2> { weight, bias };
        let input = StaticTensor::<Q16_16, 1, 0> { data: [[]] };
        let mut out = StaticTensor::<Q16_16, 1, 2>::new(Q16_16(123));
        layer.forward(&input, &mut out);
        assert_eq!(out.data[0][0].0, 49152); // 0.75 * 2^16
        assert_eq!(out.data[0][1].0, -196608); // -3.0 * 2^16
    }

    #[test]
    fn forward_distinguishes_weight_row_from_column() {
        // A non-symmetric weight catches any [i][j] vs [j][i] indexing swap:
        // out_j must mix inputs through row j of `weight`, not column j.
        // weight rows: o0 = [0, 1], o1 = [10, 0].
        let weight = StaticTensor::<Q16_16, 2, 2> {
            data: [
                [Q16_16::from_f32(0.0), Q16_16::from_f32(1.0)],
                [Q16_16::from_f32(10.0), Q16_16::from_f32(0.0)],
            ],
        };
        let bias = StaticTensor::<Q16_16, 1, 2>::new(Q16_16(0));
        let layer = StaticLinear::<2, 2> { weight, bias };
        let input = StaticTensor::<Q16_16, 1, 2> {
            data: [[Q16_16::from_f32(3.0), Q16_16::from_f32(7.0)]],
        };
        let mut out = StaticTensor::<Q16_16, 1, 2>::new(Q16_16(0));
        layer.forward(&input, &mut out);
        // o0 = 0*3 + 1*7 = 7.0 ; o1 = 10*3 + 0*7 = 30.0.
        // A transposed read would instead give o0 = 0*3 + 10*7 = 70 -> caught.
        assert_eq!(out.data[0][0].0, 458752); // 7.0  * 2^16
        assert_eq!(out.data[0][1].0, 1966080); // 30.0 * 2^16
    }
}
