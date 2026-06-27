//! Contrastive fine-tuning (InfoNCE with in-batch negatives).
//!
//! A trainable [`ProjectionHead`] is learned on top of base embeddings so that a
//! query and its true positive land close in the projected space while every
//! other document in the batch is pushed away. This is the step that makes the
//! encoder *competitive* with dense retrieval models.
//!
//! For a batch of `N` `(query, positive)` pairs we project both sides through the
//! **same** head, form the `N×N` cosine-similarity matrix
//! `S = cosine(Wq, Wp) / τ`, and minimise the cross-entropy of each row against
//! its diagonal index — i.e. every other column in the row is an *in-batch
//! negative*. The gradient flows through the new
//! [`scirust_core::autodiff::reverse::Var::l2_normalize`] primitive, so the whole
//! objective is exact reverse-mode autodiff.
//!
//! Determinism: weights are seeded ([`scirust_core::nn::PcgEngine`]), all
//! reductions are fixed-order `f32`, and training is single-threaded — so a run
//! is bit-reproducible from one machine to the next.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor, Var};
use scirust_core::nn::{CrossEntropyLoss, PcgEngine};

use crate::Encoder;

/// A learnable linear projection `y = x·W + b` trained contrastively.
pub struct ProjectionHead {
    weight: Tensor, // (dim_in, dim_out), row-major
    bias: Tensor,   // (1, dim_out)
    dim_in: usize,
    dim_out: usize,
}

impl ProjectionHead {
    /// New head with weights seeded deterministically from `seed`
    /// (`U[-scale, scale]`, `scale = 1/√dim_in`) and a zero bias.
    pub fn new(dim_in: usize, dim_out: usize, seed: u64) -> Self {
        let mut rng = PcgEngine::new(seed);
        let scale = (1.0 / dim_in as f32).sqrt();
        let mut weight = Tensor::zeros(dim_in, dim_out);
        for x in weight.data.iter_mut()
        {
            *x = rng.float_signed() * scale;
        }
        Self {
            weight,
            bias: Tensor::zeros(1, dim_out),
            dim_in,
            dim_out,
        }
    }

    /// Input dimension (the base-embedding width).
    pub fn dim_in(&self) -> usize {
        self.dim_in
    }

    /// Output (projected) dimension.
    pub fn dim_out(&self) -> usize {
        self.dim_out
    }

    /// Project one vector at inference: `y = v·W + b`, summed in index order so
    /// the result is bit-reproducible.
    pub fn project(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.dim_in, "project: input dim mismatch");
        let mut out = vec![0.0f32; self.dim_out];
        for (j, o) in out.iter_mut().enumerate()
        {
            let mut acc = self.bias.data[j];
            for (i, &vi) in v.iter().enumerate()
            {
                acc += vi * self.weight.data[i * self.dim_out + j];
            }
            *o = acc;
        }
        out
    }
}

/// Hyperparameters for contrastive training.
#[derive(Debug, Clone, Copy)]
pub struct ContrastiveConfig {
    /// Number of full-batch epochs.
    pub epochs: usize,
    /// Adam learning rate.
    pub lr: f32,
    /// Softmax temperature `τ` (scales the cosine similarities).
    pub temperature: f32,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            epochs: 400,
            lr: 0.05,
            temperature: 0.1,
        }
    }
}

fn flatten(rows: &[Vec<f32>], dim: usize) -> Tensor {
    let mut data = Vec::with_capacity(rows.len() * dim);
    for r in rows
    {
        assert_eq!(r.len(), dim, "contrastive: row dim mismatch");
        data.extend_from_slice(r);
    }
    Tensor::from_vec(data, rows.len(), dim)
}

/// Build the InfoNCE loss on `tape`; returns `(loss, weight_idx, bias_idx)`. The
/// shared projection weights are registered exactly **once** so the gradient
/// accumulates on a single parameter node (registering per side would split it).
fn forward_infonce<'t>(
    head: &ProjectionHead,
    tape: &'t Tape,
    queries: &[Vec<f32>],
    positives: &[Vec<f32>],
    temperature: f32,
) -> (Var<'t>, usize, usize) {
    let n = queries.len();
    let qt = tape.input(flatten(queries, head.dim_in));
    let pt = tape.input(flatten(positives, head.dim_in));
    let w = tape.input(head.weight.clone());
    let b = tape.input(head.bias.clone());

    let wq = qt.try_matmul(w).unwrap().try_add_bias(b).unwrap();
    let wp = pt.try_matmul(w).unwrap().try_add_bias(b).unwrap();

    // N×N cosine similarities, temperature-scaled. Diagonal = positive pairs,
    // off-diagonal = in-batch negatives.
    let scores = wq.cosine_sim_matrix(wp).scale(1.0 / temperature);
    let targets = Tensor::from_vec((0..n).map(|i| i as f32).collect(), n, 1);
    let loss = CrossEntropyLoss::new().forward_with_indices(tape, scores, &targets);
    (loss, w.idx(), b.idx())
}

/// InfoNCE loss for the current head on a batch, without updating it.
pub fn infonce_loss(
    head: &ProjectionHead,
    queries: &[Vec<f32>],
    positives: &[Vec<f32>],
    temperature: f32,
) -> f32 {
    let tape = Tape::new();
    let (loss, _, _) = forward_infonce(head, &tape, queries, positives, temperature);
    tape.value(loss.idx()).data[0]
}

/// Train `head` to minimise the InfoNCE loss over the `(query, positive)` batch.
/// Returns the per-epoch loss (before each update), so the caller can confirm
/// convergence. Full-batch, deterministic.
pub fn train(
    head: &mut ProjectionHead,
    queries: &[Vec<f32>],
    positives: &[Vec<f32>],
    cfg: ContrastiveConfig,
) -> Vec<f32> {
    assert_eq!(
        queries.len(),
        positives.len(),
        "train: queries and positives must pair up one-to-one"
    );
    let mut opt = Adam::new(cfg.lr);
    let mut losses = Vec::with_capacity(cfg.epochs);
    for _ in 0..cfg.epochs
    {
        let tape = Tape::new();
        let (loss, w_idx, b_idx) =
            forward_infonce(head, &tape, queries, positives, cfg.temperature);
        losses.push(tape.value(loss.idx()).data[0]);
        tape.backward(loss.idx());
        opt.step(&[w_idx, b_idx], &tape);
        head.weight = tape.value(w_idx);
        head.bias = tape.value(b_idx);
    }
    losses
}

/// A base [`Encoder`] composed with a trained [`ProjectionHead`]: it encodes text
/// with the base model and then applies the contrastively-learned projection.
/// Drop it into a [`crate::SemanticRetriever`] to retrieve in the fine-tuned
/// space.
pub struct ProjectedEncoder<E: Encoder> {
    base: E,
    head: ProjectionHead,
}

impl<E: Encoder> ProjectedEncoder<E> {
    /// Compose `base` with `head`. Panics if the base embedding width does not
    /// match the head's input dimension.
    pub fn new(base: E, head: ProjectionHead) -> Self {
        assert_eq!(
            base.embedding_dim(),
            head.dim_in(),
            "ProjectedEncoder: base dim {} must equal head input dim {}",
            base.embedding_dim(),
            head.dim_in()
        );
        Self { base, head }
    }
}

impl<E: Encoder> Encoder for ProjectedEncoder<E> {
    fn embedding_dim(&self) -> usize {
        self.head.dim_out()
    }

    fn encode(&mut self, text: &str) -> Vec<f32> {
        let base = self.base.encode(text);
        self.head.project(&base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::cosine;

    #[test]
    fn infonce_loss_matches_hand_value_for_identity_projection() {
        // Identity head, orthonormal q=p=[[1,0],[0,1]], τ=1.
        // Cosine matrix = I, so each row i has logit 1 on the diagonal and 0
        // elsewhere → CE = -log(e/(e+1)) = 0.313262 per row, mean = 0.313262.
        let head = ProjectionHead {
            weight: Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
            bias: Tensor::zeros(1, 2),
            dim_in: 2,
            dim_out: 2,
        };
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let loss = infonce_loss(&head, &q, &q, 1.0);
        assert!((loss - 0.313_262).abs() < 1e-4, "InfoNCE loss {loss}");
    }

    // Recall@1 of the projected head: for each query, is its true positive the
    // nearest (by cosine) among all positives?
    fn recall_at_1(head: &ProjectionHead, queries: &[Vec<f32>], positives: &[Vec<f32>]) -> f64 {
        let proj_p: Vec<Vec<f32>> = positives.iter().map(|p| head.project(p)).collect();
        let mut correct = 0usize;
        for (i, q) in queries.iter().enumerate()
        {
            let pq = head.project(q);
            let mut best_j = 0usize;
            let mut best = f32::NEG_INFINITY;
            for (j, pj) in proj_p.iter().enumerate()
            {
                let c = cosine(&pq, pj);
                if c > best
                {
                    best = c;
                    best_j = j;
                }
            }
            if best_j == i
            {
                correct += 1;
            }
        }
        correct as f64 / queries.len() as f64
    }

    #[test]
    fn training_reduces_loss_and_aligns_cross_view_pairs() {
        // "Two views" task: query_i = [e_i ; 0], positive_i = [0 ; e_i]. The two
        // views live in DISJOINT input subspaces, so the raw cosine between any
        // query and any positive is 0 — there is no signal until a projection
        // learns to map both views of class i to the same direction. This makes
        // the test a real proof that contrastive training works, not a tautology.
        let n = 5;
        let dim_in = 2 * n;
        let mut queries = Vec::new();
        let mut positives = Vec::new();
        for i in 0..n
        {
            let mut q = vec![0.0f32; dim_in];
            q[i] = 1.0;
            let mut p = vec![0.0f32; dim_in];
            p[n + i] = 1.0;
            queries.push(q);
            positives.push(p);
        }

        let mut head = ProjectionHead::new(dim_in, 8, 7);
        let losses = train(
            &mut head,
            &queries,
            &positives,
            ContrastiveConfig {
                epochs: 500,
                lr: 0.05,
                temperature: 0.1,
            },
        );

        // Initial loss is near log(N) (no usable signal); training drives it down.
        assert!(
            losses[0] > 1.0,
            "initial loss {} should be ~log(5)",
            losses[0]
        );
        let last = *losses.last().unwrap();
        assert!(
            last < 0.4 * losses[0],
            "loss did not converge: {} -> {}",
            losses[0],
            last
        );
        // The learned space puts each query nearest its true positive.
        assert_eq!(
            recall_at_1(&head, &queries, &positives),
            1.0,
            "Recall@1 after training"
        );
    }

    #[test]
    fn project_applies_the_learned_linear_map() {
        // W = [[2,0],[0,3]], b = [1,1]; project([1,1]) = [1*2+1, 1*3+1] = [3, 4].
        let head = ProjectionHead {
            weight: Tensor::from_vec(vec![2.0, 0.0, 0.0, 3.0], 2, 2),
            bias: Tensor::from_vec(vec![1.0, 1.0], 1, 2),
            dim_in: 2,
            dim_out: 2,
        };
        assert_eq!(head.project(&[1.0, 1.0]), vec![3.0, 4.0]);
    }
}
