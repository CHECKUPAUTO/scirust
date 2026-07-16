#![no_main]
//! Fuzz target for the `TensorND` typed-error op surface.
//!
//! Since the error unification, every fallible `TensorND` op (`reshape`,
//! `transpose`, `slice_axis`, `broadcast_to`, `flatten_from`, `to_tensor_2d`)
//! returns `crate::error::Result` instead of panicking. The contract under
//! test: for any *shape-valid* tensor (data length == shape product — the one
//! precondition `TensorND::new` asserts) and ANY bytes-derived arguments —
//! including invalid ranks, out-of-bounds axes, duplicate permutation entries,
//! inverted slice ranges and incompatible broadcast targets — every call
//! returns `Ok` or a typed `Err`, and NEVER panics (no out-of-bounds index,
//! no arithmetic overflow, no unbounded allocation).
//!
//! Shapes are bounded (rank ≤ 5, dims ≤ 8, total numel ≤ 4096) so each run
//! stays fast; zero-sized dims are deliberately allowed (numel == 0 is a
//! favourite edge case for stride math).

use libfuzzer_sys::fuzz_target;
use scirust_core::tensor::TensorND;

/// Byte cursor: yields 0 once the input is exhausted so every input, however
/// short, still drives a full (if degenerate) op sequence.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn u8(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    /// An arbitrary f32 bit pattern (NaN/Inf included — the shape ops only
    /// move data, they must not care what the payload is).
    fn f32_bits(&mut self) -> f32 {
        f32::from_le_bytes([self.u8(), self.u8(), self.u8(), self.u8()])
    }

    /// A bytes-derived shape: rank ≤ 5, dims 0..=8.
    fn shape(&mut self) -> Vec<usize> {
        let rank = (self.u8() % 6) as usize; // 0..=5 (rank 0 = scalar)
        (0..rank).map(|_| (self.u8() % 9) as usize).collect()
    }
}

const MAX_NUMEL: usize = 4096;

fuzz_target!(|data: &[u8]| {
    let mut c = Cursor::new(data);

    // ---- Build the starting tensor (shape-valid by construction). ----
    let shape = c.shape();
    let numel: usize = shape.iter().product();
    if numel > MAX_NUMEL
    {
        return; // keep runs fast; rank ≤ 5 / dims ≤ 8 caps this at 32768 anyway
    }
    let values: Vec<f32> = (0..numel).map(|_| c.f32_bits()).collect();
    let mut cur = TensorND::new(values, shape);

    // `zeros` takes the same untrusted shape path as `new`.
    let _ = TensorND::zeros(&c.shape());

    // ---- Bytes-driven op sequence; successful results are chained. ----
    for _ in 0..12
    {
        let out = match c.u8() % 6 {
            0 => cur.reshape(&c.shape()),
            1 => {
                // Alternate between a genuinely valid permutation (deep
                // transpose path) and raw arbitrary axes (typed-error path).
                let axes: Vec<usize> = if c.u8() & 1 == 0
                {
                    let mut perm: Vec<usize> = (0..cur.ndim()).collect();
                    // Fisher–Yates from fuzz bytes.
                    for i in (1..perm.len()).rev()
                    {
                        perm.swap(i, (c.u8() as usize) % (i + 1));
                    }
                    perm
                }
                else
                {
                    let len = (c.u8() % 7) as usize;
                    (0..len).map(|_| (c.u8() % 7) as usize).collect()
                };
                cur.transpose(&axes)
            }
            2 => {
                let axis = (c.u8() % 6) as usize;
                let start = (c.u8() % 10) as usize;
                let end = (c.u8() % 10) as usize;
                cur.slice_axis(axis, start, end)
            }
            3 => cur.broadcast_to(&c.shape()),
            4 => cur.flatten_from((c.u8() % 7) as usize),
            _ => {
                let _ = cur.to_tensor_2d(); // rank-checked conversion
                continue;
            }
        };
        // The contract: Ok or typed Err — the `match` above not panicking IS
        // the assertion. Chain successful (bounded) results to compose ops.
        if let Ok(t) = out
        {
            if t.numel() <= MAX_NUMEL
            {
                cur = t;
            }
        }
    }
});
