//! Counts heap allocations during `Tape::backward` on a deep many-node graph.
//! Allocation counts are deterministic (no machine-load noise), so this is a
//! drift-free way to measure the grad-reset allocation win.

use scirust_core::autodiff::reverse::{Tape, Tensor};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

struct Counter;
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed)
        {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding the caller's valid layout to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p`/`l` are the pointer and layout the caller got from `alloc`.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL: Counter = Counter;

fn main() {
    let depth = 300usize;
    let (r, c) = (16usize, 16usize);

    let tape = Tape::new();
    let mut v = tape.input(Tensor::from_vec(vec![0.5f32; r * c], r, c));
    for _ in 0..depth
    {
        v = v.scale(1.0001); // one Scale node each; grad flows through all
    }
    let loss = v.sum();

    // Warmup (populate grad slots, JIT-free but establishes steady state).
    loss.backward();

    let passes = 20usize;
    COUNTING.store(true, Ordering::Relaxed);
    let start = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..passes
    {
        loss.backward();
    }
    let total = ALLOCS.load(Ordering::Relaxed) - start;
    COUNTING.store(false, Ordering::Relaxed);

    let nodes = depth + 2; // input + `depth` scales + sum
    println!(
        "graph: {nodes} nodes ({r}x{c}) | {} allocs over {passes} backward passes = {:.1} allocs/pass",
        total,
        total as f64 / passes as f64
    );

    // Wall-clock (counting disabled; the atomic load is identical in both
    // builds so the A/B delta stays fair). Min of several batches to cut noise.
    let mut best = f64::INFINITY;
    for _ in 0..30
    {
        let t = std::time::Instant::now();
        for _ in 0..passes
        {
            loss.backward();
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / passes as f64;
        if us < best
        {
            best = us;
        }
    }
    println!("backward: {best:.1} µs/pass (min of 30 batches)");
}
