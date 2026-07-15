//! Cross-build reproducibility & tamper anchor for the bit-exact element-wise
//! kernels.
//!
//! `dequantize_int4_into`, `mul_f32` and `add_f32` are element-wise (no
//! reduction), so an IEEE-754 multiply/add is identical per lane and scalar and
//! their output is **bit-identical across SIMD widths, platforms and builds**.
//! The in-crate test `dequantize_int4_simd_matches_scalar_bit_exact` proves
//! SIMD == scalar *on the current build*, but it would still pass if a change
//! perturbed BOTH paths equally (e.g. a low-bit "watermark" folded into the
//! shared kernel). This file closes that gap: it pins the exact output bits to a
//! **checked-in reference vector**, so any deviation — a genuine determinism
//! regression, or a deliberate mark injected into a path the framework documents
//! as bit-reproducible — fails loudly in CI.
//!
//! Inputs are reconstructed from pinned `f32` bit patterns so the vectors are
//! exact and hermetic; only the kernel arithmetic is under test. If this test
//! ever fails, do NOT silently update the golden bits — first establish WHY the
//! output moved. These paths are a documented reproducibility contract.

use scirust_simd::ops;

fn f32s(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn bits_of(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

/// The full symmetric INT4 code range, `-8..=7`.
fn int4_codes() -> Vec<i8> {
    (-8i8..=7).collect()
}

#[test]
fn dequantize_int4_bits_are_pinned() {
    // out[i] = codes[i] as f32 * scale, for the whole -8..=7 range at two scales.
    const DEQ_INT4_S0429: [u32; 16] = [
        0xbeafb7e9, 0xbe99c0ec, 0xbe83c9ef, 0xbe5ba5e3, 0xbe2fb7e9, 0xbe03c9ef, 0xbdafb7e9,
        0xbd2fb7e9, 0x00000000, 0x3d2fb7e9, 0x3dafb7e9, 0x3e03c9ef, 0x3e2fb7e9, 0x3e5ba5e3,
        0x3e83c9ef, 0x3e99c0ec,
    ];
    const DEQ_INT4_S010: [u32; 16] = [
        0xbf4ccccd, 0xbf333333, 0xbf19999a, 0xbf000000, 0xbecccccd, 0xbe99999a, 0xbe4ccccd,
        0xbdcccccd, 0x00000000, 0x3dcccccd, 0x3e4ccccd, 0x3e99999a, 0x3ecccccd, 0x3f000000,
        0x3f19999a, 0x3f333333,
    ];

    let codes = int4_codes();
    for (scale, golden) in [(0.0429f32, &DEQ_INT4_S0429), (0.1f32, &DEQ_INT4_S010)]
    {
        let mut out = vec![0.0f32; codes.len()];
        ops::dequantize_int4_into(&codes, scale, &mut out);
        assert_eq!(
            bits_of(&out),
            golden.to_vec(),
            "dequantize_int4_into output bits drifted at scale {scale}. This path is a \
             documented cross-platform bit-exact contract — investigate the cause \
             (a determinism regression or an injected mark) before touching the golden bits."
        );
    }
}

#[test]
fn elementwise_mul_add_bits_are_pinned() {
    // Hermetic fixed inputs (reconstructed from bits, so `sqrt`/parsing can't drift them).
    let a = f32s(&[
        0x3dcccccd, 0x3e4ccccd, 0x3e99999a, 0x3ecccccd, 0x3f000000, 0x3f19999a, 0x3f333333,
        0x3f4ccccd, 0x3f666667, 0x3f800000, 0x3f8ccccd, 0x3f99999a,
    ]);
    let b = f32s(&[
        0x3e99999a, 0x3ed93924, 0x3f050581, 0x3f19999a, 0x3f2bbae3, 0x3f3c1eee, 0x3f4b3197,
        0x3f593924, 0x3f666667, 0x3f72dce9, 0x3f7eb780, 0x3f850581,
    ]);

    const MUL_F32: [u32; 12] = [
        0x3cf5c290, 0x3dadc750, 0x3e1fa035, 0x3e75c290, 0x3eabbae3, 0x3ee1beb8, 0x3f0e3c50,
        0x3f2dc750, 0x3f4f5c2a, 0x3f72dce9, 0x3f8c1820, 0x3f9fa035,
    ];
    const ADD_F32: [u32; 12] = [
        0x3ecccccd, 0x3f1fcfc5, 0x3f51d24e, 0x3f800000, 0x3f95dd72, 0x3faadc44, 0x3fbf3265,
        0x3fd302f8, 0x3fe66667, 0x3ff96e74, 0x40061446, 0x400f4f8e,
    ];

    let mut mul = vec![0.0f32; a.len()];
    ops::mul_f32(&a, &b, &mut mul);
    assert_eq!(
        bits_of(&mul),
        MUL_F32.to_vec(),
        "mul_f32 output bits drifted — element-wise f32 multiply is a bit-exact contract."
    );

    let mut add = vec![0.0f32; a.len()];
    ops::add_f32(&a, &b, &mut add);
    assert_eq!(
        bits_of(&add),
        ADD_F32.to_vec(),
        "add_f32 output bits drifted — element-wise f32 add is a bit-exact contract."
    );
}
