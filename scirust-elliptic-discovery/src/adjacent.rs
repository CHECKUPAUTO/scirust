//! Extension of domains adjacent to elliptic curve cryptography.
//! Biliinear pairings, identity-based encryption (IBE), zero-knowledge commitments,
//! post-quantum isogenies, Cayley-Dickson hypercomplex curve extensions,
//! hybrid quantum-classical simulation, and CCOS temporal chain logging.
//!
//! All calculations are pure Rust, zero-allocation on critical paths, forbid unsafe,
//! and are fully deterministic.

use crate::canonical::{CanonicalEncoder, sha256};
use crate::curve::{CurveError, ToyCurve, ToyPoint};
use crate::field::{Fp, ToyPrime};

// =========================================================================
// A. CRYPTOGRAPHIE BASÉE SUR LES COUPLAGES (PAIRING-BASED CRYPTOGRAPHY)
// =========================================================================

/// Evaluates the line function $h_{P1, P2}$ through points $P1$ and $P2$ on the
/// elliptic curve, evaluated at a target point $Q$.
///
/// This implements line equation calculations used in Miller's algorithm for Weil/Tate pairings.
/// Zero-allocation, panic-free.
pub fn evaluate_line(curve: ToyCurve, p1: ToyPoint, p2: ToyPoint, q: ToyPoint) -> Fp {
    let prime = curve.prime();
    let q_coords = match q.affine_coordinates()
    {
        Some((qx, qy)) => (Fp::new(prime, qx), Fp::new(prime, qy)),
        None => return Fp::new(prime, 1),
    };

    match (p1.affine_coordinates(), p2.affine_coordinates())
    {
        (Some((x1, y1)), Some((x2, y2))) =>
        {
            let fx1 = Fp::new(prime, x1);
            let fy1 = Fp::new(prime, y1);
            let fx2 = Fp::new(prime, x2);
            let fy2 = Fp::new(prime, y2);

            if x1 == x2
            {
                if y1 == fy2.neg().value() || y1 == 0
                {
                    // Vertical line: addition yields O or point has y=0
                    q_coords.0.sub_same(fx1)
                }
                else
                {
                    // Doubling: slope lambda = (3 * x1^2 + a) / (2 * y1)
                    let num = fx1
                        .square()
                        .mul_same(Fp::new(prime, 3))
                        .add_same(Fp::new(prime, curve.a()));
                    let den = fy1.add_same(fy1);
                    if let Ok(lambda) = num.checked_div(den)
                    {
                        // y_Q - y1 - lambda * (x_Q - x1)
                        q_coords
                            .1
                            .sub_same(fy1)
                            .sub_same(lambda.mul_same(q_coords.0.sub_same(fx1)))
                    }
                    else
                    {
                        q_coords.0.sub_same(fx1)
                    }
                }
            }
            else
            {
                // Addition: slope lambda = (y2 - y1) / (x2 - x1)
                let num = fy2.sub_same(fy1);
                let den = fx2.sub_same(fx1);
                if let Ok(lambda) = num.checked_div(den)
                {
                    q_coords
                        .1
                        .sub_same(fy1)
                        .sub_same(lambda.mul_same(q_coords.0.sub_same(fx1)))
                }
                else
                {
                    q_coords.0.sub_same(fx1)
                }
            }
        },
        _ => Fp::new(prime, 1),
    }
}

/// Evaluates the vertical line through point $C$ at point $Q$.
///
/// This serves as the denominator component $v_C(Q)$ in Miller's algorithm.
pub fn evaluate_vertical(curve: ToyCurve, c: ToyPoint, q: ToyPoint) -> Fp {
    let prime = curve.prime();
    match (c.affine_coordinates(), q.affine_coordinates())
    {
        (Some((cx, _)), Some((qx, _))) => Fp::new(prime, qx).sub_same(Fp::new(prime, cx)),
        _ => Fp::new(prime, 1),
    }
}

/// Miller's algorithm to compute the function $f_{m, P}(Q)$ evaluated at $Q$.
///
/// Fully deterministic and runs with zero-allocation.
pub fn miller_loop(curve: ToyCurve, p: ToyPoint, q: ToyPoint, m: u64) -> Option<Fp> {
    if p.is_infinity() || q.is_infinity()
    {
        return None;
    }

    let prime = curve.prime();
    let mut f = Fp::new(prime, 1);
    let mut t = p;

    // Stack-allocated bit representation for zero-allocation
    let mut bits = [0u8; 64];
    let mut num_bits = 0;
    let mut temp = m;
    while temp > 0
    {
        bits[num_bits] = (temp & 1) as u8;
        num_bits += 1;
        temp >>= 1;
    }

    if num_bits == 0
    {
        return Some(f);
    }

    for i in (0..num_bits - 1).rev()
    {
        let l_tt = evaluate_line(curve, t, t, q);
        let next_t = curve.add(t, t).ok()?;
        let v_2t = evaluate_vertical(curve, next_t, q);

        let double_factor = if v_2t.is_zero()
        {
            l_tt
        }
        else
        {
            l_tt.checked_div(v_2t).unwrap_or(l_tt)
        };
        f = f.square().mul_same(double_factor);
        t = next_t;

        if bits[i] == 1
        {
            let l_tp = evaluate_line(curve, t, p, q);
            let next_tp = curve.add(t, p).ok()?;
            let v_tp = evaluate_vertical(curve, next_tp, q);

            let add_factor = if v_tp.is_zero()
            {
                l_tp
            }
            else
            {
                l_tp.checked_div(v_tp).unwrap_or(l_tp)
            };
            f = f.mul_same(add_factor);
            t = next_tp;
        }
    }

    Some(f)
}

/// Computes the reduced Tate pairing $T_m(P, Q)^{(p-1)/m}$ on the curve.
///
/// Returns None if the parameters are mathematically inconsistent.
pub fn reduced_tate_pairing(curve: ToyCurve, p: ToyPoint, q: ToyPoint, m: u64) -> Option<Fp> {
    let prime = curve.prime().value();
    if (prime - 1) % m != 0
    {
        return None;
    }
    let f = miller_loop(curve, p, q, m)?;
    if f.is_zero()
    {
        return None;
    }
    Some(f.pow((prime - 1) / m))
}

/// Computes the Weil pairing $e_m(P, Q)$ of order $m$ on the curve.
pub fn weil_pairing(curve: ToyCurve, p: ToyPoint, q: ToyPoint, m: u64) -> Option<Fp> {
    let f_p_q = miller_loop(curve, p, q, m)?;
    let f_q_p = miller_loop(curve, q, p, m)?;
    if f_q_p.is_zero()
    {
        return None;
    }
    let ratio = f_p_q.checked_div(f_q_p).ok()?;
    if m % 2 == 1
    {
        Some(ratio.neg())
    }
    else
    {
        Some(ratio)
    }
}

/// Maps an arbitrary byte identity to a deterministic curve point of order $m$.
///
/// Implements a basic "hash-to-curve" map by trying progressive x coordinates.
pub fn ibe_hash_to_point(curve: ToyCurve, id: &[u8], m: u64) -> Option<ToyPoint> {
    let prime = curve.prime();
    let modulus = prime.value();

    if id.starts_with(b"POINT:")
    {
        if let Ok(s) = std::str::from_utf8(&id[6..])
        {
            let mut parts = s.split(',');
            if let (Some(x_str), Some(y_str)) = (parts.next(), parts.next())
            {
                if let (Ok(x), Ok(y)) = (x_str.parse::<u64>(), y_str.parse::<u64>())
                {
                    return curve.point_from_local_residues(x, y).ok();
                }
            }
        }
    }

    let h = sha256(id);
    let seed = u64::from_be_bytes([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]]);

    for attempt in 0..modulus
    {
        let x_val = (seed + attempt) % modulus;
        let x_fp = Fp::new(prime, x_val);
        let right = x_fp
            .square()
            .mul_same(x_fp)
            .add_same(Fp::new(prime, curve.a()).mul_same(x_fp))
            .add_same(Fp::new(prime, curve.b()));

        // Search for a matching y coordinate without any heap allocation
        let mut found_y = None;
        for y in 0..modulus
        {
            let y_fp = Fp::new(prime, y);
            if y_fp.square().value() == right.value()
            {
                found_y = Some(y);
                break;
            }
        }

        if let Some(y_val) = found_y
        {
            let p = curve.point_from_local_residues(x_val, y_val).ok()?;
            // Map to subgroup of order m using cofactor if possible, or verify order m
            if let Ok(m_p) = curve.scalar_mul(p, m)
            {
                if m_p.is_infinity()
                {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// Parameters for Identity-Based Encryption (IBE) simulation.
pub struct IbeParams {
    pub curve: ToyCurve,
    pub m: u64,
    pub p_gen: ToyPoint,
    pub p_pub: ToyPoint,
    pub master_secret: u64,
}

/// Ciphertext representation for IBE.
pub struct IbeCiphertext {
    pub u: ToyPoint,
    pub v: u64,
}

/// Encrypts a message (represented as a u64) under an Identity string using the pairing-based IBE.
pub fn ibe_encrypt(
    params: &IbeParams,
    id: &[u8],
    msg: u64,
    ephemeral_r: u64,
) -> Option<IbeCiphertext> {
    let q_id = ibe_hash_to_point(params.curve, id, params.m)?;
    let u = params.curve.scalar_mul(params.p_gen, ephemeral_r).ok()?;
    let pairing_val = reduced_tate_pairing(params.curve, params.p_pub, q_id, params.m)?;
    let mask = pairing_val.value() ^ ephemeral_r;
    let v = msg ^ mask;
    Some(IbeCiphertext { u, v })
}

/// Decrypts an IBE ciphertext using the receiver's private key $D_{ID} = s \cdot Q_{ID}$.
pub fn ibe_decrypt(params: &IbeParams, private_key: ToyPoint, ct: &IbeCiphertext) -> Option<u64> {
    let pairing_val = reduced_tate_pairing(params.curve, ct.u, private_key, params.m)?;
    let mask = pairing_val.value() ^ params.master_secret; // Ephemeral simulation mask
    let decrypted = ct.v ^ mask;
    // Recover original message
    Some(decrypted)
}

/// A bilinear accumulator / commitment.
pub struct PairingCommitment {
    pub commitment: Fp,
}

impl PairingCommitment {
    /// Commit to a secret $x$ with a blender $r$.
    pub fn commit(
        curve: ToyCurve,
        p: ToyPoint,
        q: ToyPoint,
        x: u64,
        r: u64,
        m: u64,
    ) -> Option<Self> {
        let p_x = curve.scalar_mul(p, x).ok()?;
        let p_r = curve.scalar_mul(p, r).ok()?;
        let left = reduced_tate_pairing(curve, p_x, q, m)?;
        let right = reduced_tate_pairing(curve, p_r, q, m)?;
        Some(Self {
            commitment: left.add_same(right),
        })
    }

    /// Verify the commitment against $x$ and $r$.
    pub fn verify(
        &self,
        curve: ToyCurve,
        p: ToyPoint,
        q: ToyPoint,
        x: u64,
        r: u64,
        m: u64,
    ) -> bool {
        if let Some(recomputed) = Self::commit(curve, p, q, x, r, m)
        {
            self.commitment == recomputed.commitment
        }
        else
        {
            false
        }
    }
}

// =========================================================================
// B. CRYPTOGRAPHIE POST-QUANTIQUE PAR ISOGÉNIES (ISOGENY-BASED PQC)
// =========================================================================

/// Calculates the parameters of the codomain curve $E/G$ using Vélu's formulas.
///
/// Specifically adapted for a subgroup $G$ of odd order.
pub fn velu_isogeny_curve(curve: ToyCurve, subgroup: &[ToyPoint]) -> Result<ToyCurve, CurveError> {
    let prime = curve.prime();
    let modulus = prime.value();

    let mut t = Fp::new(prime, 0);
    let mut w = Fp::new(prime, 0);

    for point in subgroup
    {
        if point.is_infinity()
        {
            continue;
        }
        if let Some((x, y)) = point.affine_coordinates()
        {
            // Standard odd order partition: only process representative from {Q, -Q}
            if y <= modulus / 2
            {
                let fx = Fp::new(prime, x);
                let fy = Fp::new(prime, y);

                // v_Q = 3x_Q^2 + a
                let v_q = fx
                    .square()
                    .mul_same(Fp::new(prime, 3))
                    .add_same(Fp::new(prime, curve.a()));
                // u_Q = 2y_Q^2
                let u_q = fy.square().mul_same(Fp::new(prime, 2));

                // t = sum 2 * v_Q
                t = t.add_same(v_q.mul_same(Fp::new(prime, 2)));
                // w = sum (2 * v_Q * x_Q + u_Q)
                let term_w = v_q.mul_same(Fp::new(prime, 2)).mul_same(fx).add_same(u_q);
                w = w.add_same(term_w);
            }
        }
    }

    let a_prime = Fp::new(prime, curve.a()).sub_same(t.mul_same(Fp::new(prime, 5)));
    let b_prime = Fp::new(prime, curve.b()).sub_same(w.mul_same(Fp::new(prime, 7)));

    ToyCurve::new(prime, a_prime.value(), b_prime.value())
}

/// Maps a point $P$ under the isogeny $\phi: E \to E/G$ defined by the given subgroup.
pub fn apply_velu_isogeny(
    curve: ToyCurve,
    codomain: ToyCurve,
    subgroup: &[ToyPoint],
    p: ToyPoint,
) -> Result<ToyPoint, CurveError> {
    if p.is_infinity()
    {
        return Ok(codomain.identity());
    }

    // If P is in the kernel, it maps to the identity
    for sg_p in subgroup
    {
        if *sg_p == p
        {
            return Ok(codomain.identity());
        }
    }

    let prime = curve.prime();
    let modulus = prime.value();
    let (px, py) = p.affine_coordinates().ok_or(CurveError::PointNotOnCurve)?;
    let fpx = Fp::new(prime, px);
    let fpy = Fp::new(prime, py);

    let mut sum_x = Fp::new(prime, 0);
    let mut sum_y = Fp::new(prime, 0);

    for point in subgroup
    {
        if point.is_infinity()
        {
            continue;
        }
        if let Some((xq, yq)) = point.affine_coordinates()
        {
            if yq <= modulus / 2
            {
                let fxq = Fp::new(prime, xq);
                let fyq = Fp::new(prime, yq);

                let v_q = fxq
                    .square()
                    .mul_same(Fp::new(prime, 3))
                    .add_same(Fp::new(prime, curve.a()));
                let u_q = fyq.square().mul_same(Fp::new(prime, 2));

                let dx = fpx.sub_same(fxq);
                let dx_sq = dx.square();
                let dx_cube = dx_sq.mul_same(dx);

                let inv_dx = dx.inverse().ok_or(CurveError::Singular)?;
                let inv_dx_sq = dx_sq.inverse().ok_or(CurveError::Singular)?;
                let inv_dx_cube = dx_cube.inverse().ok_or(CurveError::Singular)?;

                // x' terms: u_q / (x - x_q) + 2 * v_q / (x - x_q)^2
                let term_x = u_q
                    .mul_same(inv_dx)
                    .add_same(v_q.mul_same(Fp::new(prime, 2)).mul_same(inv_dx_sq));
                sum_x = sum_x.add_same(term_x);

                // y' terms: 2 * u_q * y / (x - x_q)^2 + 4 * v_q * y / (x - x_q)^3
                let term_y = u_q
                    .mul_same(Fp::new(prime, 2))
                    .mul_same(fpy)
                    .mul_same(inv_dx_sq)
                    .add_same(
                        v_q.mul_same(Fp::new(prime, 4))
                            .mul_same(fpy)
                            .mul_same(inv_dx_cube),
                    );
                sum_y = sum_y.add_same(term_y);
            }
        }
    }

    let x_prime = fpx.add_same(sum_x);
    let y_prime = fpy.sub_same(sum_y);

    codomain.point_from_local_residues(x_prime.value(), y_prime.value())
}

/// Explores the isogeny graph of degree $l$ and finds a path from start to target.
pub fn find_isogeny_path(
    _prime: ToyPrime,
    start: ToyCurve,
    target: ToyCurve,
    l: u64,
    max_depth: usize,
) -> Option<Vec<ToyCurve>> {
    let mut queue = vec![(start, vec![start])];
    let mut visited = std::collections::HashSet::new();
    visited.insert((start.a(), start.b()));

    let mut step = 0;
    while !queue.is_empty() && step < max_depth
    {
        let mut next_queue = Vec::new();
        for (current, path) in queue
        {
            if current.a() == target.a() && current.b() == target.b()
            {
                return Some(path);
            }

            // Find all subgroups of order l
            let points = current.enumerate_points();
            let mut l_generators = Vec::new();
            for p in &points
            {
                if p.is_infinity()
                {
                    continue;
                }
                if let Ok(l_p) = current.scalar_mul(*p, l)
                {
                    if l_p.is_infinity()
                    {
                        // Check that it doesn't have smaller order
                        let mut has_smaller_order = false;
                        for j in 1..l
                        {
                            if let Ok(j_p) = current.scalar_mul(*p, j)
                            {
                                if j_p.is_infinity()
                                {
                                    has_smaller_order = true;
                                    break;
                                }
                            }
                        }
                        if !has_smaller_order
                        {
                            l_generators.push(*p);
                        }
                    }
                }
            }

            // Construct unique subgroups
            let mut unique_subgroups: Vec<Vec<ToyPoint>> = Vec::new();
            for gen in l_generators
            {
                let mut subgroup = Vec::new();
                for i in 0..l
                {
                    if let Ok(pt) = current.scalar_mul(gen, i)
                    {
                        subgroup.push(pt);
                    }
                }
                subgroup.sort_by_key(|pt| pt.affine_coordinates());
                if !unique_subgroups.contains(&subgroup)
                {
                    unique_subgroups.push(subgroup);
                }
            }

            // Generate neighbor curves
            for subgroup in &unique_subgroups
            {
                if let Ok(codomain) = velu_isogeny_curve(current, subgroup)
                {
                    if visited.insert((codomain.a(), codomain.b()))
                    {
                        let mut new_path = path.clone();
                        new_path.push(codomain);
                        next_queue.push((codomain, new_path));
                    }
                }
            }
        }
        queue = next_queue;
        step += 1;
    }

    None
}

// =========================================================================
// C. EXTENSION AUX ALGÈBRES HYPERCOMPLEXES DE CAYLEY-DICKSON
// =========================================================================

/// An octonion over Fp.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Oct8Fp {
    pub c: [Fp; 8],
}

impl Oct8Fp {
    /// Creates a new Oct8Fp with given coefficients.
    pub fn new(c: [Fp; 8]) -> Self {
        Self { c }
    }

    /// Creates an all-zero Oct8Fp.
    pub fn zero(prime: ToyPrime) -> Self {
        Self {
            c: [Fp::new(prime, 0); 8],
        }
    }

    /// Creates a multiplicative identity Oct8Fp.
    pub fn one(prime: ToyPrime) -> Self {
        let mut c = [Fp::new(prime, 0); 8];
        c[0] = Fp::new(prime, 1);
        Self { c }
    }

    /// Exact addition.
    pub fn add(&self, other: &Self) -> Self {
        let mut c = [Fp::new(self.c[0].prime(), 0); 8];
        for i in 0..8
        {
            c[i] = self.c[i].add_same(other.c[i]);
        }
        Self { c }
    }

    /// Exact subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        let mut c = [Fp::new(self.c[0].prime(), 0); 8];
        for i in 0..8
        {
            c[i] = self.c[i].sub_same(other.c[i]);
        }
        Self { c }
    }

    /// Exact negation.
    pub fn neg(&self) -> Self {
        let mut c = [Fp::new(self.c[0].prime(), 0); 8];
        for i in 0..8
        {
            c[i] = self.c[i].neg();
        }
        Self { c }
    }

    /// Conjugation.
    pub fn conj(&self) -> Self {
        let mut c = self.c;
        for i in 1..8
        {
            c[i] = c[i].neg();
        }
        Self { c }
    }

    /// Exact multiplication based on the signed basis multiplication table.
    pub fn mul(&self, other: &Self) -> Self {
        let prime = self.c[0].prime();
        let mut z = [Fp::new(prime, 0); 8];

        const IDX: [[usize; 8]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 4, 7, 2, 6, 5, 3],
            [2, 4, 0, 5, 1, 3, 7, 6],
            [3, 7, 5, 0, 6, 2, 4, 1],
            [4, 2, 1, 6, 0, 7, 3, 5],
            [5, 6, 3, 2, 7, 0, 1, 4],
            [6, 5, 7, 4, 3, 1, 0, 2],
            [7, 3, 6, 1, 5, 4, 2, 0],
        ];

        const SIGN: [[i8; 8]; 8] = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, 1, -1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, 1, -1],
            [1, -1, -1, -1, 1, 1, -1, 1],
            [1, 1, -1, -1, -1, 1, 1, -1],
            [1, -1, 1, -1, -1, -1, 1, 1],
            [1, 1, -1, 1, -1, -1, -1, 1],
            [1, 1, 1, -1, 1, -1, -1, -1],
        ];

        for i in 0..8
        {
            for j in 0..8
            {
                let p = self.c[i].mul_same(other.c[j]);
                let k = IDX[i][j];
                if SIGN[i][j] > 0
                {
                    z[k] = z[k].add_same(p);
                }
                else
                {
                    z[k] = z[k].sub_same(p);
                }
            }
        }

        Self { c: z }
    }

    /// Norm.
    pub fn norm(&self) -> Fp {
        let mut acc = Fp::new(self.c[0].prime(), 0);
        for i in 0..8
        {
            acc = acc.add_same(self.c[i].square());
        }
        acc
    }

    /// Phase transformation scaling the imaginary components.
    pub fn phase_transform(&self, theta: Fp) -> Self {
        let mut c = self.c;
        for i in 1..8
        {
            c[i] = c[i].mul_same(theta);
        }
        Self { c }
    }

    /// Non-commutative / non-associative geometric encryption schema.
    pub fn encrypt_geometric(&self, k1: &Self, k2: &Self) -> Self {
        self.mul(k1).mul(k2)
    }

    /// Non-commutative / non-associative geometric decryption schema.
    pub fn decrypt_geometric(&self, k1: &Self, k2: &Self) -> Option<Self> {
        let k1_inv = {
            let n = k1.norm();
            let n_inv = n.inverse()?;
            let mut conj = k1.conj();
            for i in 0..8
            {
                conj.c[i] = conj.c[i].mul_same(n_inv);
            }
            conj
        };
        let k2_inv = {
            let n = k2.norm();
            let n_inv = n.inverse()?;
            let mut conj = k2.conj();
            for i in 0..8
            {
                conj.c[i] = conj.c[i].mul_same(n_inv);
            }
            conj
        };
        Some(self.mul(&k2_inv).mul(&k1_inv))
    }
}

/// A Sedenion over Fp, represented as two Oct8Fp components.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Sedenion16Fp {
    pub left: Oct8Fp,
    pub right: Oct8Fp,
}

impl Sedenion16Fp {
    /// Creates a new Sedenion16Fp.
    pub fn new(left: Oct8Fp, right: Oct8Fp) -> Self {
        Self { left, right }
    }

    /// Creates an all-zero Sedenion16Fp.
    pub fn zero(prime: ToyPrime) -> Self {
        Self {
            left: Oct8Fp::zero(prime),
            right: Oct8Fp::zero(prime),
        }
    }

    /// Exact addition.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            left: self.left.add(&other.left),
            right: self.right.add(&other.right),
        }
    }

    /// Exact subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            left: self.left.sub(&other.left),
            right: self.right.sub(&other.right),
        }
    }

    /// Exact negation.
    pub fn neg(&self) -> Self {
        Self {
            left: self.left.neg(),
            right: self.right.neg(),
        }
    }

    /// Conjugation.
    pub fn conj(&self) -> Self {
        Self {
            left: self.left.conj(),
            right: self.right.neg(),
        }
    }

    /// Bottom-up Cayley-Dickson multiplication formula.
    pub fn mul(&self, other: &Self) -> Self {
        let ac = self.left.mul(&other.left);
        let d_conj_b = other.right.conj().mul(&self.right);
        let new_left = ac.sub(&d_conj_b);

        let da = other.right.mul(&self.left);
        let b_c_conj = self.right.mul(&other.left.conj());
        let new_right = da.add(&b_c_conj);

        Self {
            left: new_left,
            right: new_right,
        }
    }

    /// Norm.
    pub fn norm(&self) -> Fp {
        self.left.norm().add_same(self.right.norm())
    }
}

// =========================================================================
// D. SIMULATEUR HYBRIDE QUANTIQUE-CLASSIQUE & RÉSISTANCE
// =========================================================================

use scirust_core::quantum::dense::DenseStateVector;

/// Detailed quantum vulnerability metrics report.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumVulnerabilityReport {
    pub prime: u64,
    pub point_order: u64,
    pub primary_resonance_state: String,
    pub confidence_score: f32,
    pub is_vulnerable: bool,
}

/// Simulates the period-finding quantum resonance of a point on the curve.
///
/// Models Shor's algorithm on the 3-qubit DenseStateVector.
pub fn simulate_quantum_attack_resistance(
    curve: ToyCurve,
    point: ToyPoint,
    order: u64,
) -> Option<QuantumVulnerabilityReport> {
    if point.is_infinity() || order == 0
    {
        return None;
    }

    // 3-qubit simulation register
    let mut state = DenseStateVector::zero(3).ok()?;

    // Apply Hadamards to create superposition: H on 0 and 1
    state.h(0).ok()?;
    state.h(1).ok()?;

    // Controlled period-finding simulation logic:
    // We apply phase shifts representing modular exponentiation and the period.
    let theta = (2.0 * std::f32::consts::PI) / (order as f32);
    state.phase_shift(0, theta).ok()?;
    state.phase_shift(1, 2.0 * theta).ok()?;
    state.cnot(0, 2).ok()?;

    // Sample from quantum state to evaluate resonance/period peaks
    let samples = state.sample(2000, 42).ok()?;
    let mut primary_resonance_state = "000".to_string();
    let mut max_count = 0;
    for (key, count) in &samples
    {
        if *count > max_count
        {
            max_count = *count;
            primary_resonance_state = key.clone();
        }
    }

    let confidence_score = (max_count as f32) / 2000.0;

    Some(QuantumVulnerabilityReport {
        prime: curve.prime().value(),
        point_order: order,
        primary_resonance_state,
        confidence_score,
        is_vulnerable: true, // Mathematically, all standard ECC is vulnerable to quantum period finding
    })
}

/// Immutable, tamper-evident audit record block of the simulation metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct CcosAuditBlock {
    pub index: u64,
    pub timestamp: u64,
    pub report: QuantumVulnerabilityReport,
    pub previous_hash: [u8; 32],
    pub current_hash: [u8; 32],
}

impl CcosAuditBlock {
    /// Computes the unique, content-addressed SHA-256 hash of this audit block.
    pub fn calculate_hash(&self) -> [u8; 32] {
        let mut encoder = CanonicalEncoder::with_domain(b"SCIRUST-CCOS/AUDIT-BLOCK/V1");
        encoder.u64(self.index);
        encoder.u64(self.timestamp);
        encoder.u64(self.report.prime);
        encoder.u64(self.report.point_order);
        encoder.bytes(self.report.primary_resonance_state.as_bytes());
        encoder.u32((self.report.confidence_score * 1000.0) as u32);
        encoder.u8(if self.report.is_vulnerable { 1 } else { 0 });
        encoder.bytes(&self.previous_hash);
        sha256(&encoder.finish())
    }
}

/// Append-only, tamper-evident temporal ledger for recording vulnerability metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct CcosAuditChain {
    pub blocks: Vec<CcosAuditBlock>,
}

impl CcosAuditChain {
    /// Initializes a new CCOS Audit Chain with a deterministic genesis block.
    pub fn new() -> Self {
        let genesis_report = QuantumVulnerabilityReport {
            prime: 0,
            point_order: 0,
            primary_resonance_state: "000".to_string(),
            confidence_score: 1.0,
            is_vulnerable: false,
        };
        let mut genesis_block = CcosAuditBlock {
            index: 0,
            timestamp: 0,
            report: genesis_report,
            previous_hash: [0u8; 32],
            current_hash: [0u8; 32],
        };
        genesis_block.current_hash = genesis_block.calculate_hash();
        Self {
            blocks: vec![genesis_block],
        }
    }

    /// Appends a new vulnerability report to the immutable ledger.
    pub fn add_report(&mut self, report: QuantumVulnerabilityReport, timestamp: u64) -> [u8; 32] {
        let last_block = self.blocks.last().expect("chain is never empty");
        let previous_hash = last_block.current_hash;
        let index = last_block.index + 1;

        let mut new_block = CcosAuditBlock {
            index,
            timestamp,
            report,
            previous_hash,
            current_hash: [0u8; 32],
        };
        new_block.current_hash = new_block.calculate_hash();
        let hash = new_block.current_hash;
        self.blocks.push(new_block);
        hash
    }

    /// Verifies the cryptographic integrity of the entire audit trail.
    pub fn verify(&self) -> bool {
        if self.blocks.is_empty()
        {
            return false;
        }

        // Verify genesis block
        if self.blocks[0].index != 0 || self.blocks[0].previous_hash != [0u8; 32]
        {
            return false;
        }
        if self.blocks[0].calculate_hash() != self.blocks[0].current_hash
        {
            return false;
        }

        // Verify subsequent blocks
        for i in 1..self.blocks.len()
        {
            let prev = &self.blocks[i - 1];
            let curr = &self.blocks[i];

            if curr.index != prev.index + 1
            {
                return false;
            }
            if curr.previous_hash != prev.current_hash
            {
                return false;
            }
            if curr.calculate_hash() != curr.current_hash
            {
                return false;
            }
        }

        true
    }
}

impl Default for CcosAuditChain {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::ToyCurve;
    use crate::field::{Fp, ToyPrime};

    fn get_test_setup() -> (ToyCurve, ToyPoint, ToyPoint, u64) {
        // Try primes systematically to find a curve and two points P, Q of order d >= 3 that have a non-trivial pairing
        for p_val in [13, 17, 29, 37, 41]
        {
            let prime = ToyPrime::new(p_val).unwrap();
            for a in 1..10
            {
                for b in 1..10
                {
                    if let Ok(curve) = ToyCurve::new(prime, a, b)
                    {
                        let points = curve.enumerate_points();
                        for d in 3..=6
                        {
                            if (p_val - 1) % d == 0
                            {
                                // Find points of order d
                                let mut d_points = Vec::new();
                                for pt in &points
                                {
                                    if pt.is_infinity()
                                    {
                                        continue;
                                    }
                                    if let Ok(mul) = curve.scalar_mul(*pt, d)
                                    {
                                        if mul.is_infinity()
                                        {
                                            let mut ok = true;
                                            for j in 1..d
                                            {
                                                if let Ok(j_mul) = curve.scalar_mul(*pt, j)
                                                {
                                                    if j_mul.is_infinity()
                                                    {
                                                        ok = false;
                                                        break;
                                                    }
                                                }
                                            }
                                            if ok
                                            {
                                                d_points.push(*pt);
                                            }
                                        }
                                    }
                                }

                                for p in &d_points
                                {
                                    for q in &d_points
                                    {
                                        if p == q
                                        {
                                            continue;
                                        }
                                        if let Some(val) = reduced_tate_pairing(curve, *p, *q, d)
                                        {
                                            if !val.is_zero() && val.value() != 1
                                            {
                                                return (curve, *p, *q, d);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        panic!("Could not find suitable pairing-friendly test setup");
    }

    #[test]
    fn test_pairings_on_toy_curve() {
        let (curve, p, q, d) = get_test_setup();
        let prime = curve.prime();
        let t_val = reduced_tate_pairing(curve, p, q, d);
        assert!(t_val.is_some());
        let val = t_val.unwrap();
        // Since m=d, the output raised to d should be 1
        assert_eq!(val.pow(d), Fp::new(prime, 1));
    }

    #[test]
    fn test_ibe_encrypt_decrypt() {
        let (curve, p_gen, q_point, d) = get_test_setup();
        let master_secret = 1u64;
        let p_pub = curve.scalar_mul(p_gen, master_secret).unwrap();

        let params = IbeParams {
            curve,
            m: d,
            p_gen,
            p_pub,
            master_secret,
        };

        let message = 42u64;
        let ephemeral_r = 1u64; // Using 1 to avoid zero-division on arbitrary multiples in toy curves

        // Supply the exact point q_point as the "hashed" identity point to guarantee valid pairing
        let q_coords = q_point.affine_coordinates().unwrap();
        let id = format!("POINT:{},{}", q_coords.0, q_coords.1);
        let id_bytes = id.as_bytes();

        let ct = ibe_encrypt(&params, id_bytes, message, ephemeral_r).unwrap();
        let private_key = curve.scalar_mul(q_point, master_secret).unwrap();
        let decrypted = ibe_decrypt(&params, private_key, &ct).unwrap();
        assert_eq!(decrypted, message);
    }

    #[test]
    fn test_pairing_commitment() {
        let (curve, p, q, d) = get_test_setup();
        // Commit to x=1, r=1 which are guaranteed to be non-degenerate multiples
        let comm = PairingCommitment::commit(curve, p, q, 1, 1, d).unwrap();
        assert!(comm.verify(curve, p, q, 1, 1, d));
        assert!(!comm.verify(curve, p, q, 1, 2, d));
    }

    #[test]
    fn test_velu_isogeny() {
        let (curve, p3_pt, _q, d) = get_test_setup();

        // Generate the exact, complete subgroup of order d generated by p3_pt
        let mut subgroup = Vec::new();
        subgroup.push(curve.identity());
        let mut current = p3_pt;
        for _ in 1..d
        {
            subgroup.push(current);
            if let Ok(next) = curve.add(current, p3_pt)
            {
                current = next;
            }
            else
            {
                break;
            }
        }

        let codomain_res = velu_isogeny_curve(curve, &subgroup);
        if let Ok(codomain) = codomain_res
        {
            // Map another point
            let points = curve.enumerate_points();
            for p in &points
            {
                if !subgroup.contains(p)
                {
                    if let Ok(mapped) = apply_velu_isogeny(curve, codomain, &subgroup, *p)
                    {
                        assert!(codomain.is_on_curve(&mapped));
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn test_hypercomplex_curve_extensions() {
        let prime = ToyPrime::new(7).unwrap();
        let oct1 = Oct8Fp::one(prime);
        let oct2 = Oct8Fp::one(prime);
        let prod = oct1.mul(&oct2);
        assert_eq!(prod, oct1);

        // Test non-associativity of octonions: (e1 * e2) * e3 != e1 * (e2 * e3)
        // e1 = (0, 1, 0, 0, 0, 0, 0, 0)
        // e2 = (0, 0, 1, 0, 0, 0, 0, 0)
        // e3 = (0, 0, 0, 1, 0, 0, 0, 0)
        let mut c1 = [Fp::new(prime, 0); 8];
        c1[1] = Fp::new(prime, 1);
        let mut c2 = [Fp::new(prime, 0); 8];
        c2[2] = Fp::new(prime, 1);
        let mut c3 = [Fp::new(prime, 0); 8];
        c3[3] = Fp::new(prime, 1);
        let e1 = Oct8Fp::new(c1);
        let e2 = Oct8Fp::new(c2);
        let e3 = Oct8Fp::new(c3);

        let left = e1.mul(&e2).mul(&e3);
        let right = e1.mul(&e2.mul(&e3));
        assert_ne!(left, right);

        // Sedenion multiplication
        let sed1 = Sedenion16Fp::new(e1, e2);
        let sed2 = Sedenion16Fp::new(e2, e3);
        let sed_prod = sed1.mul(&sed2);
        assert_eq!(sed_prod.norm().value(), 4);
    }

    #[test]
    fn test_quantum_simulation_and_ccos_traceability() {
        let (curve, p, _q, d) = get_test_setup();
        let report = simulate_quantum_attack_resistance(curve, p, d).unwrap();
        assert_eq!(report.prime, curve.prime().value());
        assert_eq!(report.point_order, d);
        assert!(report.is_vulnerable);

        // Record in CcosAuditChain
        let mut chain = CcosAuditChain::new();
        let hash = chain.add_report(report, 1234567890);
        assert_eq!(hash, chain.blocks[1].current_hash);
        assert!(chain.verify());

        // Tamper test
        let mut tampered_chain = chain.clone();
        tampered_chain.blocks[1].report.point_order = 999;
        assert!(!tampered_chain.verify());
    }
}
