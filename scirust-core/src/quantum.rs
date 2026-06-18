//! **MPS / Tensor-Train quantum-circuit simulator** — represent an `n`-qubit state
//! as a chain of rank-3 tensors so that, as long as the entanglement stays moderate,
//! the cost is `O(n · χ³)` instead of the `2ⁿ` of a dense state-vector (`χ` is the
//! bond dimension that bounds the entanglement across each cut).
//!
//! A state `|ψ⟩ = Σ A^{s₁}_{1,a₁} A^{s₂}_{a₁,a₂} … A^{sₙ}_{a_{n-1},1} |s₁…sₙ⟩` is a
//! [`Mps`] of [`MpsNode`]s. A one-qubit gate contracts a `2×2` into the physical
//! index in place; a **two-qubit** gate on adjacent qubits (1) contracts the two
//! nodes into a `θ` tensor, (2) **applies** the `4×4` gate, (3) reshapes to a matrix
//! and runs a **truncated SVD** (scirust's in-house [`truncated_svd`], pure Rust —
//! no FFI), keeping at most `χ` singular values to cap the bond dimension.
//!
//! Real `f32` amplitudes (real gates: `H`, `X`, `Z`, `CNOT`, `CZ`, `Ry`, …);
//! complex amplitudes (phase/`S`/`T`/`Rz`) are future work. Deterministic; validated
//! by exact agreement with a dense state-vector simulator. The same contraction +
//! truncated-SVD machinery is the Tensor-Train weight compression in
//! [`crate::tn::tt_decompose`] / [`crate::nn::tt_linear`].

use crate::tn::ops::svd::truncated_svd;

/// A rank-3 MPS tensor `A[l, p, r]` (left bond × physical × right bond) stored as a
/// flat row-major `Vec<f32>` of length `dl · dp · dr`.
#[derive(Debug, Clone)]
pub struct MpsNode {
    /// Flat data, length `dl · dp · dr`.
    pub data: Vec<f32>,
    /// Left virtual (bond) dimension.
    pub dl: usize,
    /// Physical dimension (2 for a qubit).
    pub dp: usize,
    /// Right virtual (bond) dimension.
    pub dr: usize,
}

impl MpsNode {
    /// New node; `data` must have length `dl · dp · dr`.
    pub fn new(dl: usize, dp: usize, dr: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), dl * dp * dr, "MpsNode: dimension mismatch");
        Self { data, dl, dp, dr }
    }

    /// Element `A[l, p, r]`.
    #[inline(always)]
    pub fn get(&self, l: usize, p: usize, r: usize) -> f32 {
        self.data[l * (self.dp * self.dr) + p * self.dr + r]
    }
}

/// Standard real one- and two-qubit gates (row-major flattened matrices).
pub mod gates {
    use std::f32::consts::FRAC_1_SQRT_2;

    /// Hadamard.
    pub const H: [f32; 4] = [FRAC_1_SQRT_2, FRAC_1_SQRT_2, FRAC_1_SQRT_2, -FRAC_1_SQRT_2];
    /// Pauli-X (NOT).
    pub const X: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
    /// Pauli-Z.
    pub const Z: [f32; 4] = [1.0, 0.0, 0.0, -1.0];
    /// Identity.
    pub const I: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    /// CNOT (control = first qubit, target = second), `(q1q2)×(p1p2)` row-major.
    pub const CNOT: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, // 00 -> 00
        0.0, 1.0, 0.0, 0.0, // 01 -> 01
        0.0, 0.0, 0.0, 1.0, // 11 -> from 10
        0.0, 0.0, 1.0, 0.0, // 10 -> from 11
    ];
    /// Controlled-Z (`diag(1,1,1,−1)`).
    pub const CZ: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, -1.0,
    ];

    /// `Ry(θ)` rotation (real): `[[cos θ/2, −sin θ/2], [sin θ/2, cos θ/2]]`.
    pub fn ry(theta: f32) -> [f32; 4] {
        let (c, s) = ((theta * 0.5).cos(), (theta * 0.5).sin());
        [c, -s, s, c]
    }
}

/// An `n`-qubit quantum state as a Matrix Product State.
#[derive(Debug, Clone)]
pub struct Mps {
    nodes: Vec<MpsNode>,
}

impl Mps {
    /// The computational-basis state `|0…0⟩` (every bond dimension 1).
    pub fn zero(n_qubits: usize) -> Self {
        assert!(n_qubits >= 1, "Mps: need at least one qubit");
        let nodes = (0..n_qubits)
            .map(|_| MpsNode::new(1, 2, 1, vec![1.0, 0.0])) // |0⟩
            .collect();
        Self { nodes }
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.nodes.len()
    }

    /// The largest bond dimension across all cuts (the entanglement proxy / cost).
    pub fn max_bond(&self) -> usize {
        self.nodes.iter().map(|nd| nd.dr).max().unwrap_or(1)
    }

    /// Total number of stored `f32` amplitudes across all nodes — the MPS memory
    /// footprint, to compare against the `2ⁿ` of a dense state-vector.
    pub fn storage(&self) -> usize {
        self.nodes.iter().map(|nd| nd.data.len()).sum()
    }

    /// Apply a one-qubit gate (`2×2` row-major) to qubit `q` in place. No SVD needed.
    pub fn apply_1qubit_gate(&mut self, q: usize, gate: &[f32; 4]) {
        let node = &self.nodes[q];
        let (dl, dr) = (node.dl, node.dr);
        let mut out = vec![0.0f32; dl * 2 * dr];
        for l in 0..dl
        {
            for r in 0..dr
            {
                let (a0, a1) = (node.get(l, 0, r), node.get(l, 1, r));
                out[l * (2 * dr) + r] = gate[0] * a0 + gate[1] * a1; // physical 0
                out[l * (2 * dr) + dr + r] = gate[2] * a0 + gate[3] * a1; // physical 1
            }
        }
        self.nodes[q] = MpsNode::new(dl, 2, dr, out);
    }

    /// Apply a two-qubit gate (`4×4` row-major, output `(q1q2)` × input `(p1p2)`) to
    /// **adjacent** qubits `q` and `q+1`, truncating the bond to at most `max_chi`.
    pub fn apply_2qubit_gate(&mut self, q: usize, gate: &[f32; 16], max_chi: usize) {
        let (a, b) = (&self.nodes[q], &self.nodes[q + 1]);
        assert_eq!(a.dr, b.dl, "apply_2qubit_gate: bond mismatch");
        let (dl, mid, dr) = (a.dl, a.dr, b.dr);
        let (rows, cols) = (dl * 2, 2 * dr);

        // θ[l, q1, q2, r] = Σ_{p1,p2} gate[(q1q2),(p1p2)] · Σ_v A[l,p1,v] B[v,p2,r],
        // laid out as a (rows × cols) row-major matrix.
        let mut theta = vec![0.0f32; rows * cols];
        for l in 0..dl
        {
            for p1 in 0..2
            {
                for p2 in 0..2
                {
                    for r in 0..dr
                    {
                        let mut c = 0.0f32;
                        for v in 0..mid
                        {
                            c += a.get(l, p1, v) * b.get(v, p2, r);
                        }
                        for q1 in 0..2
                        {
                            for q2 in 0..2
                            {
                                let g = gate[(q1 * 2 + q2) * 4 + (p1 * 2 + p2)];
                                theta[l * (4 * dr) + q1 * (2 * dr) + q2 * dr + r] += g * c;
                            }
                        }
                    }
                }
            }
        }

        // Truncated SVD: θ = U diag(s) Vᵀ, keep ≤ max_chi (and drop negligible σ).
        let svd = truncated_svd(&theta, rows, cols, max_chi, 1e-7);
        let k = svd.rank;
        // Split √s symmetrically into the two new nodes.
        let mut left = vec![0.0f32; dl * 2 * k]; // (dl, 2, k)
        for l in 0..dl
        {
            for q1 in 0..2
            {
                for j in 0..k
                {
                    left[l * (2 * k) + q1 * k + j] = svd.u[(l * 2 + q1) * k + j] * svd.s[j].sqrt();
                }
            }
        }
        let mut right = vec![0.0f32; k * 2 * dr]; // (k, 2, dr)
        for j in 0..k
        {
            let sq = svd.s[j].sqrt();
            for q2 in 0..2
            {
                for r in 0..dr
                {
                    right[j * (2 * dr) + q2 * dr + r] = sq * svd.vt[j * cols + (q2 * dr + r)];
                }
            }
        }
        self.nodes[q] = MpsNode::new(dl, 2, k, left);
        self.nodes[q + 1] = MpsNode::new(k, 2, dr, right);
    }

    /// Amplitude `⟨s|ψ⟩` for a computational basis state given by `bits[i] ∈ {0,1}`
    /// (qubit `i`'s value), by contracting the chain left to right.
    pub fn amplitude(&self, bits: &[usize]) -> f32 {
        assert_eq!(
            bits.len(),
            self.nodes.len(),
            "amplitude: wrong number of bits"
        );
        let mut v = vec![1.0f32]; // left boundary (dim 1)
        for (i, node) in self.nodes.iter().enumerate()
        {
            let s = bits[i];
            let mut vn = vec![0.0f32; node.dr];
            for (l, &vl) in v.iter().enumerate()
            {
                if vl != 0.0
                {
                    for (r, vnr) in vn.iter_mut().enumerate()
                    {
                        *vnr += vl * node.get(l, s, r);
                    }
                }
            }
            v = vn;
        }
        v[0] // right boundary (dim 1)
    }

    /// The full `2ⁿ` dense state-vector (qubit `i` = bit `i`, LSB-first). For small
    /// `n` only — this is the exponential representation the MPS avoids.
    pub fn to_statevector(&self) -> Vec<f32> {
        let n = self.nodes.len();
        let dim = 1usize << n;
        let mut sv = vec![0.0f32; dim];
        let mut bits = vec![0usize; n];
        for (idx, amp) in sv.iter_mut().enumerate()
        {
            for (i, b) in bits.iter_mut().enumerate()
            {
                *b = (idx >> i) & 1;
            }
            *amp = self.amplitude(&bits);
        }
        sv
    }

    /// The squared norm `⟨ψ|ψ⟩` (1 for a normalised state).
    pub fn norm_sqr(&self) -> f32 {
        self.to_statevector().iter().map(|&a| a * a).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::gates::*;
    use super::*;
    use crate::nn::PcgEngine;

    // --- Dense reference simulator (ground truth) ---------------------------------

    fn dense_1q(state: &mut [f32], q: usize, g: &[f32; 4]) {
        let step = 1usize << q;
        let mut base = 0usize;
        while base < state.len()
        {
            if base & step == 0
            {
                let (a0, a1) = (state[base], state[base | step]);
                state[base] = g[0] * a0 + g[1] * a1;
                state[base | step] = g[2] * a0 + g[3] * a1;
            }
            base += 1;
        }
    }

    fn dense_2q(state: &mut [f32], q: usize, g: &[f32; 16]) {
        let (s1, s2) = (1usize << q, 1usize << (q + 1));
        let mut base = 0usize;
        while base < state.len()
        {
            if base & s1 == 0 && base & s2 == 0
            {
                let inp = [
                    state[base],           // p1=0,p2=0
                    state[base | s2],      // p1=0,p2=1
                    state[base | s1],      // p1=1,p2=0
                    state[base | s1 | s2], // p1=1,p2=1
                ];
                let out: Vec<f32> = (0..4)
                    .map(|k| (0..4).map(|j| g[k * 4 + j] * inp[j]).sum())
                    .collect();
                state[base] = out[0];
                state[base | s2] = out[1];
                state[base | s1] = out[2];
                state[base | s1 | s2] = out[3];
            }
            base += 1;
        }
    }

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// **Bell state**: `H(0)`, `CNOT(0,1)` gives `(|00⟩ + |11⟩)/√2`, with bond
    /// dimension 2 (maximally entangled across the cut).
    #[test]
    fn bell_state_exact() {
        let mut mps = Mps::zero(2);
        mps.apply_1qubit_gate(0, &H);
        mps.apply_2qubit_gate(0, &CNOT, 8);
        let sv = mps.to_statevector();
        let inv = std::f32::consts::FRAC_1_SQRT_2;
        let want = [inv, 0.0, 0.0, inv]; // |00⟩=idx0, |11⟩=idx3 (LSB-first)
        assert!(l2(&sv, &want) < 1e-5, "bell state {sv:?}");
        assert_eq!(mps.max_bond(), 2, "Bell should need bond 2");
        assert!((mps.norm_sqr() - 1.0).abs() < 1e-5);
    }

    /// **GHZ**: `H(0)`, `CNOT(0,1)`, `CNOT(1,2)` gives `(|000⟩ + |111⟩)/√2`.
    #[test]
    fn ghz_state_exact() {
        let mut mps = Mps::zero(3);
        mps.apply_1qubit_gate(0, &H);
        mps.apply_2qubit_gate(0, &CNOT, 8);
        mps.apply_2qubit_gate(1, &CNOT, 8);
        let sv = mps.to_statevector();
        let inv = std::f32::consts::FRAC_1_SQRT_2;
        assert!((sv[0b000] - inv).abs() < 1e-5 && (sv[0b111] - inv).abs() < 1e-5);
        let others: f32 = (0..8)
            .filter(|&i| i != 0 && i != 7)
            .map(|i| sv[i].abs())
            .sum();
        assert!(others < 1e-5, "GHZ leakage {others}");
    }

    /// **Ground truth**: the MPS (with a generous bond cap) reproduces a dense
    /// state-vector simulator **exactly** for a random circuit of H / Ry / CNOT gates.
    #[test]
    fn matches_dense_random_circuit() {
        let n = 5usize;
        let mut rng = PcgEngine::new(7);
        let mut mps = Mps::zero(n);
        let mut dense = vec![0.0f32; 1 << n];
        dense[0] = 1.0;
        for _ in 0..40
        {
            if rng.next_u32() % 2 == 0
            {
                let q = (rng.next_u32() as usize) % n;
                let g = if rng.next_u32() % 2 == 0
                {
                    H
                }
                else
                {
                    ry(rng.float_signed() * 3.0)
                };
                mps.apply_1qubit_gate(q, &g);
                dense_1q(&mut dense, q, &g);
            }
            else
            {
                let q = (rng.next_u32() as usize) % (n - 1);
                let g = if rng.next_u32() % 2 == 0 { CNOT } else { CZ };
                mps.apply_2qubit_gate(q, &g, 1 << n); // no truncation
                dense_2q(&mut dense, q, &g);
            }
        }
        assert!(
            l2(&mps.to_statevector(), &dense) < 1e-3,
            "MPS diverged from dense sim"
        );
        // Determinism.
        let mut mps2 = Mps::zero(n);
        let mut rng2 = PcgEngine::new(7);
        for _ in 0..40
        {
            if rng2.next_u32() % 2 == 0
            {
                let q = (rng2.next_u32() as usize) % n;
                let g = if rng2.next_u32() % 2 == 0
                {
                    H
                }
                else
                {
                    ry(rng2.float_signed() * 3.0)
                };
                mps2.apply_1qubit_gate(q, &g);
            }
            else
            {
                let q = (rng2.next_u32() as usize) % (n - 1);
                let g = if rng2.next_u32() % 2 == 0 { CNOT } else { CZ };
                mps2.apply_2qubit_gate(q, &g, 1 << n);
            }
        }
        assert_eq!(
            mps.to_statevector()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            mps2.to_statevector()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }

    /// A product state keeps bond dimension 1; entangling grows it; **truncation** to
    /// a smaller bond keeps a high-fidelity approximation (and `χ ≥ needed` is exact).
    #[test]
    fn bond_dimension_and_truncation() {
        // Product state: independent H on each qubit ⇒ no entanglement ⇒ bond 1.
        let mut prod = Mps::zero(4);
        for q in 0..4
        {
            prod.apply_1qubit_gate(q, &H);
        }
        assert_eq!(prod.max_bond(), 1, "product state must stay at bond 1");

        // Entangle, then compare full vs hard-truncated bond.
        let mut full = Mps::zero(4);
        for q in 0..4
        {
            full.apply_1qubit_gate(q, &H);
        }
        full.apply_2qubit_gate(0, &CZ, 8);
        full.apply_2qubit_gate(1, &CZ, 8);
        full.apply_2qubit_gate(2, &CZ, 8);
        let full_sv = full.to_statevector();

        // χ ≥ needed reproduces it exactly; a fidelity proxy stays high under heavy cap.
        let mut capped = Mps::zero(4);
        for q in 0..4
        {
            capped.apply_1qubit_gate(q, &H);
        }
        capped.apply_2qubit_gate(0, &CZ, 1);
        capped.apply_2qubit_gate(1, &CZ, 1);
        capped.apply_2qubit_gate(2, &CZ, 1);
        let cap_sv = capped.to_statevector();
        let dot: f32 = full_sv.iter().zip(&cap_sv).map(|(&a, &b)| a * b).sum();
        let nb: f32 = cap_sv.iter().map(|&b| b * b).sum::<f32>().sqrt();
        let fidelity = dot.abs() / nb.max(1e-12);
        // Bond-1 truncation is an approximation, but a sound one (fidelity well above
        // chance for a 16-dim space, ~0.25).
        assert!(fidelity > 0.5, "truncated fidelity too low: {fidelity}");
        assert!(capped.max_bond() == 1, "cap not respected");
    }
}
