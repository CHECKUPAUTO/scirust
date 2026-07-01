//! In-process data-parallel collectives — a real, deterministic **all-reduce**,
//! **broadcast**, and **barrier** across worker *threads* that share one group
//! handle.
//!
//! # Scope (honest)
//!
//! This is the single-machine, multi-**thread** reduction used by data-parallel
//! training: workers are threads, not processes, and they rendezvous through a
//! shared [`WorkerGroup`] rather than over a socket. A multi-host TCP/RDMA
//! transport is deliberately out of scope here — this module does not pretend to
//! do inter-process networking.
//!
//! # Determinism
//!
//! [`all_reduce`] sums each worker's contribution in **ascending rank order**,
//! so the averaged gradient is bit-identical regardless of thread scheduling —
//! the same fixed-order discipline as the rest of the workspace.

use std::collections::HashMap;
use std::sync::{Arc, Barrier, Mutex};

/// Per-key, rank-indexed gradient contributions for one all-reduce round.
type ContribSlots = HashMap<String, Vec<Option<Vec<f32>>>>;
/// Per-key broadcast payloads published by rank 0.
type BcastMap = HashMap<String, Vec<f32>>;

/// Shared state for one in-process worker group. One handle is cloned into each
/// worker thread (see [`DistributedContext::group`]).
#[derive(Clone)]
pub struct WorkerGroup {
    world_size: usize,
    barrier: Arc<Barrier>,
    /// Per-key, rank-indexed contributions for the current all-reduce round.
    slots: Arc<Mutex<ContribSlots>>,
    /// Per-key broadcast payloads (published by rank 0).
    bcast: Arc<Mutex<BcastMap>>,
}

impl WorkerGroup {
    fn new(world_size: usize) -> Self {
        assert!(world_size >= 1, "world_size must be >= 1");
        Self {
            world_size,
            barrier: Arc::new(Barrier::new(world_size)),
            slots: Arc::new(Mutex::new(HashMap::new())),
            bcast: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// Per-worker context: this worker's `rank`, the `world_size`, and — for a real
/// multi-worker run — the shared [`WorkerGroup`].
#[derive(Clone)]
pub struct DistributedContext {
    /// Rank of this worker (`0 .. world_size-1`).
    pub rank: usize,
    /// Total number of workers.
    pub world_size: usize,
    group: Option<WorkerGroup>,
}

impl DistributedContext {
    /// A single-worker context — every collective is a no-op.
    pub fn single() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            group: None,
        }
    }

    /// Build `world_size` contexts that share one in-process group. Hand
    /// `contexts[r]` to the thread that acts as rank `r`.
    pub fn group(world_size: usize) -> Vec<DistributedContext> {
        let g = WorkerGroup::new(world_size);
        (0..world_size)
            .map(|rank| DistributedContext {
                rank,
                world_size,
                group: Some(g.clone()),
            })
            .collect()
    }

    /// Whether this is a multi-worker run.
    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }
}

/// All-reduce (mean): on return, every worker's `gradients` hold the element-wise
/// average across all workers. Contributions are summed in ascending rank order,
/// so the result is bit-identical run to run. Every worker must call this with
/// the **same set of keys**, each of the same length. Single-worker mode is a
/// no-op.
pub fn all_reduce(
    ctx: &DistributedContext,
    gradients: &mut HashMap<String, Vec<f32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if ctx.world_size <= 1
    {
        return Ok(());
    }
    let g = ctx.group.as_ref().ok_or_else(|| {
        "multi-worker all_reduce needs a context from DistributedContext::group".to_string()
    })?;

    // Phase 1 — publish this rank's contribution for every key.
    {
        let mut slots = g.slots.lock().unwrap();
        for (k, v) in gradients.iter()
        {
            let entry = slots
                .entry(k.clone())
                .or_insert_with(|| vec![None; g.world_size]);
            entry[ctx.rank] = Some(v.clone());
        }
    }
    g.barrier.wait();

    // Phase 2 — every worker reduces in ascending rank order and writes the mean.
    {
        let slots = g.slots.lock().unwrap();
        let inv = 1.0f32 / g.world_size as f32;
        for (k, v) in gradients.iter_mut()
        {
            let contribs = slots
                .get(k)
                .ok_or_else(|| "workers disagree on the gradient keys".to_string())?;
            let mut acc = vec![0.0f32; v.len()];
            for slot in contribs.iter()
            {
                let c = slot
                    .as_ref()
                    .ok_or_else(|| "a rank did not contribute this key".to_string())?;
                if c.len() != acc.len()
                {
                    return Err(format!(
                        "workers disagree on the length of gradient key {k:?} \
                         (this rank has {}, another contributed {})",
                        acc.len(),
                        c.len()
                    )
                    .into());
                }
                for (a, &x) in acc.iter_mut().zip(c)
                {
                    *a += x;
                }
            }
            for (dst, &a) in v.iter_mut().zip(&acc)
            {
                *dst = a * inv;
            }
        }
    }
    g.barrier.wait();

    // Phase 3 — rank 0 clears the round once everyone has finished reading.
    if ctx.rank == 0
    {
        g.slots.lock().unwrap().clear();
    }
    g.barrier.wait();
    Ok(())
}

/// Block until every worker in the group has arrived.
pub fn barrier(ctx: &DistributedContext) -> Result<(), Box<dyn std::error::Error>> {
    if ctx.world_size <= 1
    {
        return Ok(());
    }
    let g = ctx
        .group
        .as_ref()
        .ok_or_else(|| "multi-worker barrier needs a group context".to_string())?;
    g.barrier.wait();
    Ok(())
}

/// Broadcast a vector from rank 0 to every worker. Single-worker mode is a no-op.
pub fn broadcast(
    ctx: &DistributedContext,
    key: &str,
    value: &mut Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    if ctx.world_size <= 1
    {
        return Ok(());
    }
    let g = ctx
        .group
        .as_ref()
        .ok_or_else(|| "multi-worker broadcast needs a group context".to_string())?;
    if ctx.rank == 0
    {
        g.bcast
            .lock()
            .unwrap()
            .insert(key.to_string(), value.clone());
    }
    g.barrier.wait();
    if ctx.rank != 0
    {
        let b = g.bcast.lock().unwrap();
        *value = b
            .get(key)
            .ok_or_else(|| "rank 0 did not broadcast this key".to_string())?
            .clone();
    }
    g.barrier.wait();
    if ctx.rank == 0
    {
        g.bcast.lock().unwrap().clear();
    }
    g.barrier.wait();
    Ok(())
}

/// Broadcast a single `f32` from rank 0 to every worker.
pub fn broadcast_f32(
    ctx: &DistributedContext,
    value: &mut f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut v = vec![*value];
    broadcast(ctx, "__scalar", &mut v)?;
    *value = v[0];
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn single_worker_is_a_noop() {
        let ctx = DistributedContext::single();
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), vec![1.0, 2.0, 3.0]);
        all_reduce(&ctx, &mut grads).unwrap();
        assert_eq!(grads["w"], vec![1.0, 2.0, 3.0]);
        barrier(&ctx).unwrap();
        let mut x = 7.0f32;
        broadcast_f32(&ctx, &mut x).unwrap();
        assert_eq!(x, 7.0);
    }

    #[test]
    fn all_reduce_averages_across_threads_bit_exactly() {
        let world = 4;
        let ctxs = DistributedContext::group(world);
        let handles: Vec<_> = ctxs
            .into_iter()
            .map(|ctx| {
                thread::spawn(move || {
                    // rank r contributes [r+1, 2*(r+1)] under key "g".
                    let r = ctx.rank as f32;
                    let mut grads = HashMap::new();
                    grads.insert("g".to_string(), vec![r + 1.0, 2.0 * (r + 1.0)]);
                    all_reduce(&ctx, &mut grads).unwrap();
                    grads["g"].clone()
                })
            })
            .collect();
        let results: Vec<Vec<f32>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // mean of [1,2,3,4] = 2.5 ; mean of [2,4,6,8] = 5.0 — every worker agrees.
        for r in &results
        {
            assert_eq!(r, &vec![2.5, 5.0]);
        }
    }

    #[test]
    fn two_rounds_reuse_the_group() {
        let ctxs = DistributedContext::group(2);
        let handles: Vec<_> = ctxs
            .into_iter()
            .map(|ctx| {
                thread::spawn(move || {
                    let r = ctx.rank as f32;
                    let mut out = Vec::new();
                    for round in 0..2
                    {
                        let mut g = HashMap::new();
                        g.insert("g".to_string(), vec![r + round as f32]);
                        all_reduce(&ctx, &mut g).unwrap();
                        out.push(g["g"][0]);
                    }
                    out
                })
            })
            .collect();
        for h in handles
        {
            // round 0: mean(0,1)=0.5 ; round 1: mean(1,2)=1.5
            assert_eq!(h.join().unwrap(), vec![0.5, 1.5]);
        }
    }

    #[test]
    fn all_reduce_rejects_mismatched_lengths_instead_of_wrong_mean() {
        // Both ranks contribute the *same* key "g" but with different lengths.
        // The old code sized the accumulator to the local vector and used a
        // truncating `zip`, silently dropping/zero-filling the extra elements and
        // returning a wrong mean. Both ranks must instead observe an error.
        let ctxs = DistributedContext::group(2);
        let handles: Vec<_> = ctxs
            .into_iter()
            .map(|ctx| {
                thread::spawn(move || {
                    let mut grads = HashMap::new();
                    let v = if ctx.rank == 0 {
                        vec![1.0f32, 2.0]
                    } else {
                        vec![10.0f32, 20.0, 30.0]
                    };
                    grads.insert("g".to_string(), v);
                    all_reduce(&ctx, &mut grads).is_err()
                })
            })
            .collect();
        // Every worker (not just the shorter one) must see the mismatch as an
        // error rather than a silently corrupted average.
        for h in handles
        {
            assert!(h.join().unwrap(), "mismatched lengths must be an error");
        }
    }

    #[test]
    fn broadcast_sends_rank0_to_all() {
        let ctxs = DistributedContext::group(3);
        let handles: Vec<_> = ctxs
            .into_iter()
            .map(|ctx| {
                thread::spawn(move || {
                    let mut v = if ctx.rank == 0 { 42.0f32 } else { 0.0 };
                    broadcast_f32(&ctx, &mut v).unwrap();
                    v
                })
            })
            .collect();
        for h in handles
        {
            assert_eq!(h.join().unwrap(), 42.0);
        }
    }

    #[test]
    fn group_assigns_distinct_ranks() {
        let ctxs = DistributedContext::group(4);
        assert_eq!(ctxs.len(), 4);
        for (i, c) in ctxs.iter().enumerate()
        {
            assert_eq!(c.rank, i);
            assert_eq!(c.world_size, 4);
            assert!(c.is_distributed());
        }
    }
}
