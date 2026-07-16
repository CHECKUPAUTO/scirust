#![no_main]
//! Fuzz target for the reverse-mode autodiff tape.
//!
//! Builds a small random-but-VALID 2-D graph from fuzz bytes — 1..=3 input
//! leaves (shapes ≤ 8×8, finite values in [-4, 4]), then a bytes-driven
//! sequence of ≤ 12 shape-compatible ops (add/sub/hadamard on matching
//! shapes, scale, relu, tanh, sigmoid, exp, matmul when the inner dims
//! align), a final `sum`, and `.backward()`.
//!
//! Contract under test (the fuzz analogue of the hand-written gradient
//! underflow tests): the forward pass, `backward()` and every `grad()` read
//! never panic, AND all input gradients are finite — no NaN/Inf — for finite
//! bounded inputs. Magnitudes are kept representable by construction: a
//! conservative per-node value bound is tracked and the running value is
//! squashed through `tanh` whenever the bound exceeds 4 (in particular `exp`
//! is only ever applied to values with |x| ≤ 4, so it cannot overflow), which
//! keeps every local backward factor — and hence any 12-op product of them —
//! far below f32::MAX.

use libfuzzer_sys::fuzz_target;
use scirust_core::autodiff::reverse::{Tape, Tensor};

/// Byte cursor: yields 0 once the input is exhausted.
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

    /// A finite f32 uniformly derived from two bytes, mapped into [lo, hi].
    fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        let raw = u16::from_le_bytes([self.u8(), self.u8()]);
        lo + (raw as f32 / u16::MAX as f32) * (hi - lo)
    }
}

/// Value-bound cap: whenever the running node's conservative |value| bound
/// exceeds this, squash through tanh (bound becomes 1). Keeps exp inputs and
/// matmul operands small so no forward value or backward factor can overflow.
const BOUND_CAP: f64 = 4.0;

fuzz_target!(|data: &[u8]| {
    let mut c = Cursor::new(data);
    let tape = Tape::new();

    // ---- 1..=3 input leaves, shapes ≤ 8×8, finite values in [-4, 4]. ----
    let n_inputs = 1 + (c.u8() % 3) as usize;
    let mut leaves = Vec::with_capacity(n_inputs);
    let mut leaf_shapes = Vec::with_capacity(n_inputs);
    for _ in 0..n_inputs
    {
        let rows = 1 + (c.u8() % 8) as usize;
        let cols = 1 + (c.u8() % 8) as usize;
        let vals: Vec<f32> = (0..rows * cols).map(|_| c.f32_in(-4.0, 4.0)).collect();
        leaves.push(tape.input(Tensor::from_vec(vals, rows, cols)));
        leaf_shapes.push((rows, cols));
    }

    let mut cur = leaves[0];
    let mut cur_shape = leaf_shapes[0];
    let mut bound: f64 = BOUND_CAP; // |leaf values| ≤ 4

    // ---- Bytes-driven op sequence: only shape-compatible ops are taken. ----
    let n_ops = (c.u8() % 13) as usize; // 0..=12
    for _ in 0..n_ops
    {
        match c.u8() % 9 {
            op @ (0 | 1 | 2) => {
                // add / sub / hadamard with a same-shaped leaf (never the
                // current node itself — the graph stays a simple chain with
                // side edges from the leaves).
                let pick = (c.u8() as usize) % n_inputs;
                let mate = (0..n_inputs)
                    .map(|k| (pick + k) % n_inputs)
                    .find(|&i| leaf_shapes[i] == cur_shape && leaves[i].idx() != cur.idx());
                if let Some(i) = mate
                {
                    cur = match op {
                        0 => cur.try_add(leaves[i]),
                        1 => cur.try_sub(leaves[i]),
                        _ => cur.try_hadamard(leaves[i]),
                    }
                    .expect("same-tape, same-shape op must be Ok");
                    bound = if op == 2 { bound * BOUND_CAP } else { bound + BOUND_CAP };
                }
            }
            3 => {
                cur = cur.scale(c.f32_in(-2.0, 2.0));
                bound *= 2.0;
            }
            4 => cur = cur.relu(), // bound unchanged
            5 => {
                cur = cur.tanh();
                bound = 1.0;
            }
            6 => {
                cur = cur.sigmoid();
                bound = 1.0;
            }
            7 => {
                // bound ≤ BOUND_CAP here (see squash below), so exp ≤ e^4.
                cur = cur.exp();
                bound = bound.exp();
            }
            _ => {
                // matmul with a leaf whose row count matches our column count.
                let pick = (c.u8() as usize) % n_inputs;
                let mate = (0..n_inputs)
                    .map(|k| (pick + k) % n_inputs)
                    .find(|&i| leaf_shapes[i].0 == cur_shape.1);
                if let Some(i) = mate
                {
                    cur = cur
                        .try_matmul(leaves[i])
                        .expect("inner dims align, matmul must be Ok");
                    cur_shape = (cur_shape.0, leaf_shapes[i].1);
                    // inner dim ≤ 8, leaf entries ≤ 4.
                    bound *= 8.0 * BOUND_CAP;
                }
            }
        }
        if bound > BOUND_CAP
        {
            cur = cur.tanh();
            bound = 1.0;
        }
    }

    // ---- Reduce, differentiate, and check every input gradient. ----
    let loss = cur.sum();
    loss.backward();

    let loss_val = tape.grad(loss.idx()); // seed — sanity: exists and is 1×1
    assert_eq!(loss_val.data.len(), 1, "sum() must be scalar");

    for (i, leaf) in leaves.iter().enumerate()
    {
        let g = tape.grad(leaf.idx());
        assert_eq!(
            g.data.len(),
            leaf_shapes[i].0 * leaf_shapes[i].1,
            "gradient shape must match input {i}"
        );
        for (j, v) in g.data.iter().enumerate()
        {
            assert!(
                v.is_finite(),
                "non-finite gradient {v} at input {i}, element {j}"
            );
        }
    }
});
