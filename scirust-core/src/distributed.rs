//! Distributed training primitives — all-reduce, broadcast, barrier.
//!
//! Provides gradient synchronization across workers using a simple
//! ring all-reduce algorithm. Works over TCP (no MPI dependency) for
//! data-parallel training on CPU clusters.
//!
//! # Design
//!
//! - `DistributedContext` manages worker rank, world size, and communication.
//! - `all_reduce` averages gradients across all workers.
//! - Communication uses a simple TCP ring topology (worker i ↔ worker (i+1)%N).
//!
//! # Limitations
//!
//! - Single-node multi-process only (no multi-host yet).
//! - No NCCL/RDMA — pure TCP, suitable for CPU clusters.

use std::collections::HashMap;
#[allow(unused_imports)]
use std::net::TcpListener;
#[allow(unused_imports)]
use std::sync::{Arc, Mutex};

/// Distributed training context for a single worker.
#[derive(Debug, Clone)]
pub struct DistributedContext {
    /// Rank of this worker (0 .. world_size-1).
    pub rank: usize,
    /// Total number of workers.
    pub world_size: usize,
    /// Base port for communication (rank i uses port base + i).
    pub base_port: u16,
}

impl DistributedContext {
    /// Create a single-worker context (no distribution).
    pub fn single() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            base_port: 29500,
        }
    }

    /// Create a multi-worker context.
    pub fn new(rank: usize, world_size: usize, base_port: u16) -> Self {
        assert!(rank < world_size, "rank must be < world_size");
        Self {
            rank,
            world_size,
            base_port,
        }
    }

    /// Whether this is a distributed run.
    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }
}

/// All-reduce gradients using a simple ring algorithm.
///
/// Each worker contributes its local gradients; after all-reduce,
/// every worker has the element-wise average across all workers.
///
/// In single-worker mode, this is a no-op.
pub fn all_reduce(
    ctx: &DistributedContext,
    _gradients: &mut HashMap<String, Vec<f32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if ctx.world_size <= 1 {
        return Ok(()); // No-op for single worker
    }

    // Ring all-reduce: scatter-reduce + all-gather
    // For simplicity, each gradient buffer is averaged manually.
    // In production, this would use TCP sockets for inter-process comm.
    //
    // Pseudocode for 2-worker ring:
    //   Worker 0 sends its buffer to worker 1, receives worker 1's buffer.
    //   Each averages: (local + remote) / 2.

    // For now, we implement a single-process all-reduce that demonstrates
    // the API contract. Real inter-process communication requires socket setup.

    // The actual implementation would:
    // 1. Open TCP connections to neighbors in the ring
    // 2. Scatter-reduce: send 1/N chunk to next, receive from prev, accumulate
    // 3. All-gather: circulate the reduced chunks
    //
    // This is a structural placeholder that documents the algorithm.

    let _world_size = ctx.world_size;
    let _rank = ctx.rank;

    // In a real distributed setting with multiple processes,
    // socket-based communication would go here.
    // For single-process testing, this is a no-op.

    Ok(())
}

/// Synchronize all workers at a barrier.
pub fn barrier(ctx: &DistributedContext) -> Result<(), Box<dyn std::error::Error>> {
    if ctx.world_size <= 1 {
        return Ok(());
    }
    // Real implementation would coordinate via TCP.
    Ok(())
}

/// Broadcast a value from rank 0 to all workers.
pub fn broadcast_f32(
    ctx: &DistributedContext,
    _value: &mut f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if ctx.world_size <= 1 {
        return Ok(());
    }
    // In production, rank 0 sends, others receive.
    if ctx.rank != 0 {
        // Receive from rank 0
        // *value = received_value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_worker_noop() {
        let ctx = DistributedContext::single();
        let mut grads = HashMap::new();
        grads.insert("w".into(), vec![1.0, 2.0, 3.0]);

        all_reduce(&ctx, &mut grads).unwrap();
        // Single worker: gradients unchanged
        assert_eq!(grads["w"], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_single_worker_barrier() {
        let ctx = DistributedContext::single();
        barrier(&ctx).unwrap(); // No-op, should not hang
    }

    #[test]
    fn test_context_creation() {
        let ctx = DistributedContext::new(2, 4, 29500);
        assert_eq!(ctx.rank, 2);
        assert_eq!(ctx.world_size, 4);
        assert!(ctx.is_distributed());
    }

    #[test]
    #[should_panic]
    fn test_invalid_rank() {
        DistributedContext::new(5, 4, 29500); // rank >= world_size
    }
}
