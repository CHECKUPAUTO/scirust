// scirust-core/src/autodiff/data_parallel.rs
// Phase 4: Data Parallelism Engine — GradientAggregator & DataParallelTrainer

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::parallel::ParallelTape;

/// Aggregates gradients from multiple workers via reduce-sum / reduce-mean.
pub struct GradientAggregator;

impl GradientAggregator {
    /// Element-wise sum across workers.
    ///
    /// `grads` is a slice of per-worker gradient vectors (each of length N).
    /// Returns a single vector of length N where `result[j] = sum_i grads[i][j]`.
    pub fn reduce_sum(grads: &[Vec<f64>]) -> Vec<f64> {
        if grads.is_empty()
        {
            return Vec::new();
        }
        let n = grads[0].len();
        let mut result = vec![0.0; n];
        for worker_grads in grads
        {
            debug_assert_eq!(worker_grads.len(), n, "reduce_sum: length mismatch");
            for (j, &v) in worker_grads.iter().enumerate()
            {
                result[j] += v;
            }
        }
        result
    }

    /// Element-wise mean across workers.
    ///
    /// `grads` is a slice of per-worker gradient vectors (each of length N).
    /// Returns a single vector of length N where `result[j] = mean_i grads[i][j]`.
    pub fn reduce_mean(grads: &[Vec<f64>]) -> Vec<f64> {
        if grads.is_empty()
        {
            return Vec::new();
        }
        let n_workers = grads.len() as f64;
        let mut result = Self::reduce_sum(grads);
        for v in &mut result
        {
            *v /= n_workers;
        }
        result
    }
}

/// Manages a set of [`ParallelTape`]s — one per worker — for data-parallel
/// training.
///
/// # Example
///
/// ```ignore
/// let mut trainer = DataParallelTrainer::new(2);
/// let avg_grads = trainer.train_batch(|tape, worker| {
///     // build graph on `tape`, run backward, return gradients
///     vec![0.5, 1.0]
/// });
/// ```
pub struct DataParallelTrainer {
    n_workers: usize,
    tapes: Vec<ParallelTape>,
}

impl DataParallelTrainer {
    pub fn new(n_workers: usize) -> Self {
        let tapes = (0..n_workers).map(|_| ParallelTape::new()).collect();
        Self { n_workers, tapes }
    }

    /// Run `batch_fn` on every worker tape, collect the returned gradient
    /// vectors and produce their element-wise mean.
    ///
    /// The `batch_fn` receives a reference to the worker's tape and the
    /// worker index (0 .. n_workers-1).  It should build a computation
    /// graph on the tape, run [`ParallelTape::backward`], and return the
    /// gradient vector(s) of interest (typically as a flat `Vec<f64>`).
    pub fn train_batch<F>(&mut self, batch_fn: F) -> Vec<f64>
    where
        F: Fn(&ParallelTape, usize) -> Vec<f64>,
    {
        let mut all_grads: Vec<Vec<f64>> = Vec::with_capacity(self.n_workers);
        for i in 0..self.n_workers
        {
            let grads = batch_fn(&self.tapes[i], i);
            all_grads.push(grads);
        }
        GradientAggregator::reduce_mean(&all_grads)
    }

    /// Run the workers across `n_threads` OS threads and return the
    /// **fixed-order** mean of their gradients — a *certified-deterministic*
    /// parallel batch.
    ///
    /// Threads pull worker indices from a shared atomic counter (work
    /// stealing), but each worker's result is written to its own
    /// worker-indexed slot and the reduction always sums worker `0, 1, …,
    /// n-1` in that order. Floating-point addition is not associative, so a
    /// naive "accumulate as threads finish" reduction would depend on the
    /// scheduler; this one does not. Consequently the result is **bit-identical
    /// for any `n_threads`** and identical to the sequential [`Self::train_batch`].
    ///
    /// `batch_fn` must be `Sync` (shared across threads) and deterministic in
    /// `(tape, worker)`.
    pub fn train_batch_threaded<F>(&self, n_threads: usize, batch_fn: F) -> Vec<f64>
    where
        F: Fn(&ParallelTape, usize) -> Vec<f64> + Sync,
    {
        let n = self.n_workers;
        if n == 0
        {
            return Vec::new();
        }
        let n_threads = n_threads.clamp(1, n);
        let slots: Vec<Mutex<Option<Vec<f64>>>> = (0..n).map(|_| Mutex::new(None)).collect();
        let next = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            for _ in 0..n_threads
            {
                scope.spawn(|| {
                    loop
                    {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n
                        {
                            break;
                        }
                        let g = batch_fn(&self.tapes[i], i);
                        *slots[i].lock().expect("data-parallel slot poisoned") = Some(g);
                    }
                });
            }
        });

        let all: Vec<Vec<f64>> = slots
            .into_iter()
            .map(|m| {
                m.into_inner()
                    .expect("data-parallel slot poisoned")
                    .expect("data-parallel worker did not run")
            })
            .collect();
        GradientAggregator::reduce_mean(&all)
    }

    /// Return a reference to the worker's tape.
    pub fn tape(&self, worker: usize) -> &ParallelTape {
        &self.tapes[worker]
    }

    /// Return a mutable reference to the worker's tape.
    pub fn tape_mut(&mut self, worker: usize) -> &mut ParallelTape {
        &mut self.tapes[worker]
    }

    /// Number of workers.
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }
}

// ================================================================== //
//  Tests                                                             //
// ================================================================== //

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Node, Op, SavedData, Tensor};

    // ------------------------------------------------------------------ //
    //  Aggregator tests                                                   //
    // ------------------------------------------------------------------ //

    #[test]
    fn test_reduce_sum_single_worker() {
        let w = vec![1.0, 2.0, 3.0];
        let result = GradientAggregator::reduce_sum(&[w]);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_reduce_sum_two_workers() {
        let w1 = vec![1.0, 2.0, 3.0];
        let w2 = vec![4.0, 5.0, 6.0];
        let result = GradientAggregator::reduce_sum(&[w1, w2]);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_reduce_sum_empty() {
        let result = GradientAggregator::reduce_sum(&[] as &[Vec<f64>]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_reduce_mean_one_worker() {
        let w = vec![2.0, 4.0];
        let result = GradientAggregator::reduce_mean(&[w]);
        assert_eq!(result, vec![2.0, 4.0]);
    }

    #[test]
    fn test_reduce_mean_two_workers() {
        let w1 = vec![2.0, 4.0];
        let w2 = vec![4.0, 6.0];
        let result = GradientAggregator::reduce_mean(&[w1, w2]);
        assert_eq!(result, vec![3.0, 5.0]);
    }

    #[test]
    fn test_reduce_mean_three_workers() {
        let w1 = vec![1.0, 2.0];
        let w2 = vec![3.0, 4.0];
        let w3 = vec![5.0, 6.0];
        let result = GradientAggregator::reduce_mean(&[w1, w2, w3]);
        assert!((result[0] - 3.0).abs() < 1e-12);
        assert!((result[1] - 4.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------ //
    //  DataParallelTrainer tests                                          //
    // ------------------------------------------------------------------ //

    #[test]
    fn test_trainer_one_worker_parity_with_sequential() {
        // Sequential: f(x) = x * 2, df/dx = 2
        let seq_tape = crate::autodiff::reverse::Tape::new();
        let sx = seq_tape.input(Tensor::from_vec(vec![3.0], 1, 1));
        let sx_idx = sx.idx();
        let sy = sx.scale(2.0);
        sy.backward();
        let seq_grad: f64 = seq_tape.grad(sx_idx).sum() as f64;

        // Data-parallel with 1 worker: should match sequential
        let mut trainer = DataParallelTrainer::new(1);
        let avg_grads = trainer.train_batch(|tape, _worker| {
            let x = tape.alloc_node(Node {
                op: Op::Input,
                shape: (1, 1),
                saved: SavedData::None,
            });
            let y = tape.alloc_node(Node {
                op: Op::Scale {
                    input: x,
                    scalar: 2.0,
                },
                shape: (1, 1),
                saved: SavedData::None,
            });
            tape.set_value(x, &[3.0]);
            tape.set_value(y, &[6.0]);
            tape.backward(y);
            vec![tape.grad(x)]
        });

        assert_eq!(avg_grads.len(), 1);
        assert!(
            (avg_grads[0] - seq_grad).abs() < 1e-5,
            "seq_grad={} dp_grad={}",
            seq_grad,
            avg_grads[0]
        );
    }

    #[test]
    fn test_trainer_two_workers_gradient_consistency() {
        // Both workers compute f(x) = x * 3 for different x values:
        //   worker 0: x = 2.0  => df/dx = 3
        //   worker 1: x = 5.0  => df/dx = 3
        // mean gradient should be 3.0
        let mut trainer = DataParallelTrainer::new(2);
        let avg_grads = trainer.train_batch(|tape, worker| {
            let x_val = if worker == 0 { 2.0f32 } else { 5.0 };
            let x = tape.alloc_node(Node {
                op: Op::Input,
                shape: (1, 1),
                saved: SavedData::None,
            });
            let y = tape.alloc_node(Node {
                op: Op::Scale {
                    input: x,
                    scalar: 3.0,
                },
                shape: (1, 1),
                saved: SavedData::None,
            });
            tape.set_value(x, &[x_val]);
            tape.set_value(y, &[x_val * 3.0]);
            tape.backward(y);
            vec![tape.grad(x)]
        });

        assert_eq!(avg_grads.len(), 1);
        assert!(
            (avg_grads[0] - 3.0).abs() < 1e-5,
            "expected mean grad = 3.0, got {}",
            avg_grads[0]
        );
    }

    #[test]
    fn test_trainer_two_workers_equals_sequential_mean() {
        // Sequential: compute f(x) = x*2 for x=3 and x=7, then take mean
        // f(x) = x*2, df/dx = 2 regardless of x
        // For x in {3, 7}: grads are {2, 2}, mean = 2

        let mut trainer = DataParallelTrainer::new(2);
        let avg_grads = trainer.train_batch(|tape, worker| {
            let x_val = if worker == 0 { 3.0f32 } else { 7.0 };
            let x = tape.alloc_node(Node {
                op: Op::Input,
                shape: (1, 1),
                saved: SavedData::None,
            });
            let y = tape.alloc_node(Node {
                op: Op::Scale {
                    input: x,
                    scalar: 2.0,
                },
                shape: (1, 1),
                saved: SavedData::None,
            });
            tape.set_value(x, &[x_val]);
            tape.set_value(y, &[x_val * 2.0]);
            tape.backward(y);
            vec![tape.grad(x)]
        });

        assert!((avg_grads[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_trainer_multi_parameter() {
        // Each worker has 2 parameters (e.g. a Scale with 2-element tensor)
        // Worker 0: x=[1,2], y=x*3 => grads=[3,3]
        // Worker 1: x=[4,5], y=x*3 => grads=[3,3]
        // Mean: [3,3]
        let mut trainer = DataParallelTrainer::new(2);
        let avg_grads = trainer.train_batch(|tape, worker| {
            let x_vals: Vec<f32> = if worker == 0
            {
                vec![1.0, 2.0]
            }
            else
            {
                vec![4.0, 5.0]
            };
            let x = tape.alloc_node(Node {
                op: Op::Input,
                shape: (1, 2),
                saved: SavedData::None,
            });
            let y = tape.alloc_node(Node {
                op: Op::Scale {
                    input: x,
                    scalar: 3.0,
                },
                shape: (1, 2),
                saved: SavedData::None,
            });
            let y_vals: Vec<f32> = x_vals.iter().map(|v| v * 3.0).collect();
            tape.set_value(x, &x_vals);
            tape.set_value(y, &y_vals);
            tape.backward(y);
            // return grad of each element as separate parameters
            let gx = tape.grad(x);
            // gx is the scalar sum of the full tensor gradient
            // For Scale, each element gets gradient 3.0, so sum = 6.0
            vec![gx]
        });

        assert_eq!(avg_grads.len(), 1);
        // Each worker: grad = 3+3 = 6.0, mean = 6.0
        assert!(
            (avg_grads[0] - 6.0).abs() < 1e-5,
            "expected 6.0, got {}",
            avg_grads[0]
        );
    }

    // ------------------------------------------------------------------ //
    //  Certified determinism: thread count must not change the result    //
    // ------------------------------------------------------------------ //

    /// The aggregated gradient is bit-identical for 1/2/4/8 OS threads and
    /// equal to the sequential path. Per-worker contributions are deliberately
    /// order-sensitive (element 0 mixes ±1e16 with small values), so a
    /// scheduler-dependent reduction order *would* perturb the low bits — the
    /// fixed worker-order reduction does not.
    #[test]
    fn train_batch_threaded_is_thread_count_invariant() {
        let bf = |_tape: &ParallelTape, w: usize| -> Vec<f64> {
            let e0 = match w % 4
            {
                0 => 1e16,
                1 => 1.0,
                2 => -1e16,
                _ => 3.0,
            };
            vec![e0, (w as f64 + 1.0).recip(), (w as f64).sin()]
        };
        let run = |threads: usize| DataParallelTrainer::new(8).train_batch_threaded(threads, bf);
        let r1 = run(1);
        assert_eq!(r1, run(2), "2 threads differ from 1");
        assert_eq!(r1, run(4), "4 threads differ from 1");
        assert_eq!(r1, run(8), "8 threads differ from 1");
        // …and identical to the sequential reduction.
        let seq = DataParallelTrainer::new(8).train_batch(bf);
        assert_eq!(r1, seq, "threaded differs from sequential");
    }

    /// Same guarantee with **real autograd**: each worker builds a graph on its
    /// `ParallelTape` and runs the actual backward; the aggregate is
    /// bit-identical across 1/2/4 threads.
    #[test]
    fn parallel_tape_training_is_deterministic_across_threads() {
        let bf = |tape: &ParallelTape, w: usize| -> Vec<f64> {
            let x = tape.alloc_node(Node {
                op: Op::Input,
                shape: (1, 3),
                saved: SavedData::None,
            });
            let y = tape.alloc_node(Node {
                op: Op::Scale {
                    input: x,
                    scalar: 2.0,
                },
                shape: (1, 3),
                saved: SavedData::None,
            });
            let xv: Vec<f32> = (0..3).map(|j| ((w * 3 + j) as f32).sin()).collect();
            let yv: Vec<f32> = xv.iter().map(|v| v * 2.0).collect();
            tape.set_value(x, &xv);
            tape.set_value(y, &yv);
            tape.backward(y);
            vec![tape.grad(x)]
        };
        let run = |threads: usize| DataParallelTrainer::new(4).train_batch_threaded(threads, bf);
        let r1 = run(1);
        assert_eq!(r1, run(2), "2 threads differ from 1");
        assert_eq!(r1, run(4), "4 threads differ from 1");
    }

    /// A **full multi-step SGD run** is bit-identical across thread counts:
    /// each step's aggregated gradient is thread-count-invariant and the
    /// optimizer update is deterministic, so the whole trajectory composes.
    /// Each worker trains a shared linear model `y = x·W` on its own data shard
    /// with an MSE loss, using the real reverse-mode autograd tape.
    #[test]
    fn multi_step_training_is_thread_count_invariant() {
        use crate::autodiff::reverse::Tape;

        fn train(threads: usize) -> Vec<f32> {
            let (in_dim, out_dim, n_workers, steps, lr) = (3usize, 2usize, 4usize, 8usize, 0.05f32);
            let mut w: Vec<f32> = (0..in_dim * out_dim)
                .map(|i| (i as f32 * 0.1).sin())
                .collect();
            for _ in 0..steps
            {
                let trainer = DataParallelTrainer::new(n_workers);
                let w_ref = &w;
                let grads = trainer.train_batch_threaded(threads, |_t, worker| {
                    // Per-worker data shard, deterministic from the worker id.
                    let x: Vec<f32> = (0..in_dim)
                        .map(|j| (((worker * in_dim + j) as f32) * 0.3).cos())
                        .collect();
                    let target: Vec<f32> = (0..out_dim)
                        .map(|j| (((worker + j) as f32) * 0.2).sin())
                        .collect();
                    let tape = Tape::new();
                    let xv = tape.input(Tensor::from_vec(x, 1, in_dim));
                    let wv = tape.input(Tensor::from_vec(w_ref.clone(), in_dim, out_dim));
                    let tv = tape.input(Tensor::from_vec(target, 1, out_dim));
                    let y = xv.matmul(wv);
                    let loss = y.sub(tv).pow(2.0).sum();
                    tape.backward(loss.idx());
                    tape.grad(wv.idx()).data.iter().map(|&v| v as f64).collect()
                });
                for (wj, &g) in w.iter_mut().zip(grads.iter())
                {
                    *wj -= lr * g as f32;
                }
            }
            w
        }

        let w1 = train(1);
        assert_eq!(w1, train(2), "2 threads diverge over the run");
        assert_eq!(w1, train(4), "4 threads diverge over the run");
        // Sanity: the run actually moved the weights off their init.
        let w0: Vec<f32> = (0..6).map(|i| (i as f32 * 0.1).sin()).collect();
        assert_ne!(w1, w0, "training did not update the weights");
    }
}
