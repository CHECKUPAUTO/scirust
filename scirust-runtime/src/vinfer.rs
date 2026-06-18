//! **Verifiable inference** — a compact cryptographic argument that an output came
//! from the **declared** model (ZK-based Verifiable ML, survey arXiv:2502.18535;
//! zkSNARK eval arXiv:2402.02675). Extends the [`proof`](crate::proof) certificates
//! from bit-exact re-execution to a **succinct soundness** guarantee.
//!
//! The model (a quantized integer linear layer over the prime field `GF(p)`,
//! `p = 2³¹ − 1`) is **committed** by hashing its weights. To verify a claimed
//! batched output `Y` for inputs `X`, the verifier runs **Freivalds' check** over
//! `GF(p)`: draw a random `r` and test `W·(X·r) == Y·r`. Computing `W·(X·r)` costs
//! `O(out·in + in·batch)` versus `O(out·in·batch)` to recompute `Y = W·X`, so for a
//! batch it is **succinct** (sub-linear in the recompute cost). A wrong `Y` passes
//! with probability `≤ 1/p` per challenge, so a few challenges give negligible
//! soundness error. The challenge `r` is derived by **Fiat-Shamir** from a hash of
//! `(commitment, X, Y)`, so it is non-interactive and bound to the claimed output
//! (the prover cannot tailor `Y` to a known `r`).
//!
//! This provides cryptographic **soundness** (the output provably comes from the
//! committed model), **not** zero-knowledge — the verifier holds the weights. Full
//! weight-hiding zk-SNARKs are out of scope.

use sha2::{Digest, Sha256};

/// Field modulus `2³¹ − 1` (a Mersenne prime); products of two residues fit `i64`.
pub const P: i64 = 2_147_483_647;

#[inline]
fn modp(x: i64) -> i64 {
    let r = x % P;
    if r < 0 { r + P } else { r }
}

#[inline]
fn mulp(a: i64, b: i64) -> i64 {
    modp(a * b)
}

/// A committed integer linear layer `Y = W·X` over `GF(p)`. `w` is row-major
/// `(out, in)` with entries in `[0, p)`.
#[derive(Clone)]
pub struct VModel {
    /// Row-major `(out, in)` weights in `[0, p)`.
    pub w: Vec<i64>,
    /// Output dimension.
    pub out: usize,
    /// Input dimension.
    pub inn: usize,
}

impl VModel {
    /// New model, reducing weights into `[0, p)`.
    pub fn new(w: Vec<i64>, out: usize, inn: usize) -> Self {
        assert_eq!(w.len(), out * inn, "VModel: weight size");
        Self {
            w: w.into_iter().map(modp).collect(),
            out,
            inn,
        }
    }

    /// SHA-256 commitment to the weights (binds the declared model).
    pub fn commit(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update((self.out as u64).to_le_bytes());
        h.update((self.inn as u64).to_le_bytes());
        for &v in &self.w
        {
            h.update(v.to_le_bytes());
        }
        h.finalize().into()
    }

    /// Batched inference `Y = W·X` over `GF(p)`. `x` is row-major `(in, batch)`;
    /// returns row-major `(out, batch)`.
    pub fn infer(&self, x: &[i64], batch: usize) -> Vec<i64> {
        assert_eq!(x.len(), self.inn * batch, "VModel::infer: input size");
        let mut y = vec![0i64; self.out * batch];
        for o in 0..self.out
        {
            for j in 0..batch
            {
                let mut acc = 0i64;
                for i in 0..self.inn
                {
                    acc = modp(acc + mulp(self.w[o * self.inn + i], modp(x[i * batch + j])));
                }
                y[o * batch + j] = acc;
            }
        }
        y
    }
}

/// Derive the Fiat-Shamir challenge vector `r ∈ GF(p)^batch` from a hash of the
/// commitment, the inputs, the claimed outputs, and the challenge index — so `r`
/// is pseudo-random and **bound to the claimed output**.
fn challenge(commitment: &[u8; 32], x: &[i64], y: &[i64], idx: u32, batch: usize) -> Vec<i64> {
    let mut r = Vec::with_capacity(batch);
    let mut counter = 0u32;
    while r.len() < batch
    {
        let mut h = Sha256::new();
        h.update(commitment);
        h.update(idx.to_le_bytes());
        h.update(counter.to_le_bytes());
        for &v in x
        {
            h.update(v.to_le_bytes());
        }
        for &v in y
        {
            h.update(v.to_le_bytes());
        }
        let digest: [u8; 32] = h.finalize().into();
        for chunk in digest.chunks_exact(4)
        {
            if r.len() == batch
            {
                break;
            }
            let u = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            r.push((u as i64) % P);
        }
        counter += 1;
    }
    r
}

/// Verify that the claimed batched output `y` (`(out, batch)`) is `model·x` and that
/// the model matches `commitment`, by Freivalds' check over `GF(p)` with
/// `num_challenges` Fiat-Shamir challenges. Sound: a wrong `y` passes with
/// probability `≤ (1/p)^num_challenges`. Deterministic.
pub fn verify_inference(
    model: &VModel,
    x: &[i64],
    batch: usize,
    y: &[i64],
    commitment: &[u8; 32],
    num_challenges: u32,
) -> bool {
    // 1. The model must match its commitment (binding).
    if &model.commit() != commitment
    {
        return false;
    }
    assert_eq!(y.len(), model.out * batch, "verify_inference: output size");
    let xr_reduced: Vec<i64> = x.iter().map(|&v| modp(v)).collect();
    // 2. Freivalds' check W·(X·r) == Y·r for each challenge.
    for idx in 0..num_challenges
    {
        let r = challenge(commitment, x, y, idx, batch);
        // X·r  : (in,)
        let mut xr = vec![0i64; model.inn];
        for (i, xri) in xr.iter_mut().enumerate()
        {
            let mut acc = 0i64;
            for j in 0..batch
            {
                acc = modp(acc + mulp(xr_reduced[i * batch + j], r[j]));
            }
            *xri = acc;
        }
        // W·(X·r) and Y·r : (out,)
        for o in 0..model.out
        {
            let mut wxr = 0i64;
            for (i, &xri) in xr.iter().enumerate()
            {
                wxr = modp(wxr + mulp(model.w[o * model.inn + i], xri));
            }
            let mut yr = 0i64;
            for j in 0..batch
            {
                yr = modp(yr + mulp(modp(y[o * batch + j]), r[j]));
            }
            if wxr != yr
            {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::nn::PcgEngine;

    fn rand_model(out: usize, inn: usize, rng: &mut PcgEngine) -> VModel {
        let w: Vec<i64> = (0..out * inn)
            .map(|_| (rng.next_u32() as i64) % P)
            .collect();
        VModel::new(w, out, inn)
    }

    fn rand_x(inn: usize, batch: usize, rng: &mut PcgEngine) -> Vec<i64> {
        (0..inn * batch)
            .map(|_| (rng.next_u32() as i64) % P)
            .collect()
    }

    /// A correct inference under the committed model verifies.
    #[test]
    fn vinfer_accepts_correct_inference() {
        let mut rng = PcgEngine::new(1);
        let (out, inn, batch) = (5, 7, 4);
        let model = rand_model(out, inn, &mut rng);
        let x = rand_x(inn, batch, &mut rng);
        let y = model.infer(&x, batch);
        let c = model.commit();
        assert!(verify_inference(&model, &x, batch, &y, &c, 3));
        // Deterministic.
        assert!(verify_inference(&model, &x, batch, &y, &c, 3));
    }

    /// **Soundness**: across many random single-entry tampers of the output, Freivalds
    /// rejects *every* one (the random projection catches the discrepancy).
    #[test]
    fn vinfer_rejects_tampering_soundly() {
        let mut rng = PcgEngine::new(2);
        let (out, inn, batch) = (4, 6, 5);
        let model = rand_model(out, inn, &mut rng);
        let x = rand_x(inn, batch, &mut rng);
        let y = model.infer(&x, batch);
        let c = model.commit();
        for _ in 0..1000
        {
            let mut tampered = y.clone();
            let pos = (rng.next_u32() as usize) % tampered.len();
            let delta = 1 + (rng.next_u32() as i64) % (P - 1);
            tampered[pos] = modp(tampered[pos] + delta); // a genuine change
            assert!(
                !verify_inference(&model, &x, batch, &tampered, &c, 2),
                "tampered output accepted"
            );
        }
    }

    /// The commitment **binds** the model: verifying against a commitment that does
    /// not match the supplied weights fails immediately.
    #[test]
    fn vinfer_rejects_wrong_model_commitment() {
        let mut rng = PcgEngine::new(3);
        let (out, inn, batch) = (3, 4, 2);
        let model = rand_model(out, inn, &mut rng);
        let other = rand_model(out, inn, &mut rng);
        let x = rand_x(inn, batch, &mut rng);
        let y = model.infer(&x, batch);
        // Correct weights but a commitment for a *different* model → rejected.
        assert!(!verify_inference(&model, &x, batch, &y, &other.commit(), 2));
        assert!(verify_inference(&model, &x, batch, &y, &model.commit(), 2));
    }

    /// Fiat-Shamir binds the challenge to the claimed output: a substituted output
    /// (a different valid inference, for *other* inputs) does not verify for `x`.
    #[test]
    fn vinfer_rejects_substituted_output() {
        let mut rng = PcgEngine::new(4);
        let (out, inn, batch) = (4, 5, 3);
        let model = rand_model(out, inn, &mut rng);
        let x = rand_x(inn, batch, &mut rng);
        let x2 = rand_x(inn, batch, &mut rng);
        let y2 = model.infer(&x2, batch); // a valid output, but for x2 ≠ x
        let c = model.commit();
        assert!(!verify_inference(&model, &x, batch, &y2, &c, 3));
    }
}
