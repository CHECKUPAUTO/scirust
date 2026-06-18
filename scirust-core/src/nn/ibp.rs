//! **Interval Bound Propagation** — certified output bounds for ReLU MLPs
//! (Gowal et al., *On the Effectiveness of Interval Bound Propagation for
//! Certified Robustness*, 2018).
//!
//! Given a *box* of inputs `[lo, hi]` (e.g. an L∞ ball around a point), IBP
//! propagates an interval through each layer to a **provable** box on the
//! outputs: every concrete input in the box yields an output inside the
//! certified interval. This is scirust's "IA certifiable" thesis made testable —
//! the [`certified_robust`] check turns the bound into a guarantee that the
//! predicted class cannot change anywhere in the box.
//!
//! Rules (sound, and *exact* for the affine layer):
//! - **Affine** `x·W + b`: centre/radius form — `c = b + cₓ·W`,
//!   `r = rₓ·|W|`, output `[c−r, c+r]` (with `cₓ = (lo+hi)/2`,
//!   `rₓ = (hi−lo)/2`).
//! - **ReLU**: monotone, so `[relu(lo), relu(hi)]`.
//!
//! The bound is validated in tests by **soundness sampling**: thousands of
//! concrete points drawn from the box all land inside the certified interval.

use crate::nn::nd_layers::NdLinear;

/// An axis-aligned box `[lo, hi]` over a vector of activations.
#[derive(Clone, Debug)]
pub struct Interval {
    /// Per-coordinate lower bounds.
    pub lo: Vec<f32>,
    /// Per-coordinate upper bounds.
    pub hi: Vec<f32>,
}

impl Interval {
    /// A degenerate box at a single point (`lo == hi == x`).
    pub fn point(x: &[f32]) -> Self {
        Self {
            lo: x.to_vec(),
            hi: x.to_vec(),
        }
    }

    /// An L∞ ball of radius `eps` around `x`.
    pub fn around(x: &[f32], eps: f32) -> Self {
        Self {
            lo: x.iter().map(|&v| v - eps).collect(),
            hi: x.iter().map(|&v| v + eps).collect(),
        }
    }

    /// The widest coordinate radius `(hi − lo)/2` — 0 for a point.
    pub fn max_radius(&self) -> f32 {
        self.lo
            .iter()
            .zip(&self.hi)
            .map(|(&l, &h)| 0.5 * (h - l))
            .fold(0.0, f32::max)
    }
}

/// An affine layer `y = x·W + b` (W row-major `(in, out)`, b `(out)`) with both
/// an interval rule and a concrete forward, for IBP certification.
pub struct IbpLinear {
    w: Vec<f32>,
    b: Vec<f32>,
    in_f: usize,
    out_f: usize,
}

impl IbpLinear {
    /// New layer from raw parts (`w` is `in*out` row-major, `b` is `out`).
    pub fn new(w: Vec<f32>, b: Vec<f32>, in_f: usize, out_f: usize) -> Self {
        assert_eq!(w.len(), in_f * out_f, "IbpLinear: weight size");
        assert_eq!(b.len(), out_f, "IbpLinear: bias size");
        Self { w, b, in_f, out_f }
    }

    /// Build from a trained [`NdLinear`] (weight `(in,out)`, bias `(1,out)`), so
    /// a network trained on the N-D tape can be certified directly.
    pub fn from_nd_linear(lin: &NdLinear) -> Self {
        let shape = &lin.weight().shape;
        Self::new(
            lin.weight().data.clone(),
            lin.bias().data.clone(),
            shape[0],
            shape[1],
        )
    }

    /// Concrete forward `y = x·W + b`.
    pub fn forward_point(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_f, "IbpLinear: input size");
        let mut y = vec![0.0f32; self.out_f];
        for (o, yo) in y.iter_mut().enumerate()
        {
            let mut acc = self.b[o];
            for (i, &xi) in x.iter().enumerate()
            {
                acc += xi * self.w[i * self.out_f + o];
            }
            *yo = acc;
        }
        y
    }

    /// Interval forward (centre/radius); exact for the affine map.
    pub fn forward_interval(&self, x: &Interval) -> Interval {
        assert_eq!(x.lo.len(), self.in_f, "IbpLinear: input size");
        let mut lo = vec![0.0f32; self.out_f];
        let mut hi = vec![0.0f32; self.out_f];
        for o in 0..self.out_f
        {
            let mut c = self.b[o];
            let mut r = 0.0f32;
            for i in 0..self.in_f
            {
                let w = self.w[i * self.out_f + o];
                c += w * 0.5 * (x.lo[i] + x.hi[i]);
                r += w.abs() * 0.5 * (x.hi[i] - x.lo[i]);
            }
            lo[o] = c - r;
            hi[o] = c + r;
        }
        Interval { lo, hi }
    }
}

/// ReLU on an interval: monotone, so bounds map elementwise.
pub fn relu_interval(x: &Interval) -> Interval {
    Interval {
        lo: x.lo.iter().map(|v| v.max(0.0)).collect(),
        hi: x.hi.iter().map(|v| v.max(0.0)).collect(),
    }
}

/// A ReLU MLP: ReLU is applied after every affine layer **except the last**.
pub struct IbpMlp {
    layers: Vec<IbpLinear>,
}

impl IbpMlp {
    /// New MLP from its affine layers (in order).
    pub fn new(layers: Vec<IbpLinear>) -> Self {
        assert!(!layers.is_empty(), "IbpMlp: needs at least one layer");
        Self { layers }
    }

    /// Concrete forward over the whole network.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut cur = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate()
        {
            cur = layer.forward_point(&cur);
            if i + 1 < self.layers.len()
            {
                for v in cur.iter_mut()
                {
                    *v = v.max(0.0);
                }
            }
        }
        cur
    }

    /// Certified output box for an input box: every input in `input` produces an
    /// output inside the returned [`Interval`] (sound by construction).
    pub fn certify(&self, input: &Interval) -> Interval {
        let mut cur = input.clone();
        for (i, layer) in self.layers.iter().enumerate()
        {
            cur = layer.forward_interval(&cur);
            if i + 1 < self.layers.len()
            {
                cur = relu_interval(&cur);
            }
        }
        cur
    }

    /// Certified output box via **zonotope** propagation (AI² / DeepZ): sound, and
    /// usually **tighter than IBP** because the generators track inter-neuron
    /// correlations that plain intervals discard.
    pub fn certify_zonotope(&self, input: &Interval) -> Interval {
        let mut z = Zonotope::from_interval(input);
        for (i, layer) in self.layers.iter().enumerate()
        {
            z = layer.forward_zonotope(&z);
            if i + 1 < self.layers.len()
            {
                z = relu_zonotope(&z);
            }
        }
        z.interval()
    }

    /// Certified output box via **DeepPoly** (Singh et al., POPL 2019): the
    /// **relational polyhedral** abstract domain. Usually **tighter than IBP** (and
    /// applicable at any depth, unlike the 2-layer [`crown_bounds`]) because it keeps
    /// each neuron's lower/upper bound as an **affine function of the inputs** and
    /// back-substitutes, so correlations like `relu(x) + relu(−x) = |x|` are tracked.
    pub fn certify_deeppoly(&self, input: &Interval) -> Interval {
        deeppoly_certify(&self.layers, input)
    }
}

/// An affine bound `coeffs · input + cst` over the input box.
type Aff = (Vec<f32>, f32);

/// Concretize an affine bound over the box `[lo, hi]`: its maximum (`maximize`) or
/// minimum picks, per coordinate, the box end that extremises `cⱼ·inputⱼ`.
fn dp_concretize(aff: &Aff, lo: &[f32], hi: &[f32], maximize: bool) -> f32 {
    let mut v = aff.1;
    for (i, &c) in aff.0.iter().enumerate()
    {
        let pick = if (c >= 0.0) == maximize { hi[i] } else { lo[i] };
        v += c * pick;
    }
    v
}

/// **DeepPoly** certified output box (Singh et al., POPL 2019). Each neuron keeps a
/// lower and upper bound that is **affine in the network inputs** (eager
/// back-substitution): an affine layer composes the bounds (choosing the previous
/// layer's lower/upper bound per the weight sign), and a ReLU on a neuron with
/// concrete range `[l, u]` is relaxed by the chord upper bound
/// `z ≤ (u/(u−l))·(y − l)` and the **area-minimising** lower bound `z ≥ λ·y` with
/// `λ = 1` if `u > −l` else `0`. Sound, and tighter than IBP because the affine
/// relations track inter-neuron correlations.
pub fn deeppoly_certify(layers: &[IbpLinear], input: &Interval) -> Interval {
    assert!(!layers.is_empty(), "deeppoly: empty network");
    let d_in = input.lo.len();
    // Input neurons: lower = upper = identity (coordinate selector).
    let mut lower: Vec<Aff> = (0..d_in)
        .map(|i| {
            let mut c = vec![0.0f32; d_in];
            c[i] = 1.0;
            (c, 0.0)
        })
        .collect();
    let mut upper: Vec<Aff> = lower.clone();
    for (li, layer) in layers.iter().enumerate()
    {
        let (inf, outf) = (layer.in_f, layer.out_f);
        let mut new_lower: Vec<Aff> = Vec::with_capacity(outf);
        let mut new_upper: Vec<Aff> = Vec::with_capacity(outf);
        for o in 0..outf
        {
            let (mut lc, mut uc) = (vec![0.0f32; d_in], vec![0.0f32; d_in]);
            let (mut lcon, mut ucon) = (layer.b[o], layer.b[o]);
            for i in 0..inf
            {
                let w = layer.w[i * outf + o];
                // Positive weight keeps the orientation, negative flips it.
                let (up_src, lo_src) = if w >= 0.0
                {
                    (&upper[i], &lower[i])
                }
                else
                {
                    (&lower[i], &upper[i])
                };
                for k in 0..d_in
                {
                    uc[k] += w * up_src.0[k];
                    lc[k] += w * lo_src.0[k];
                }
                ucon += w * up_src.1;
                lcon += w * lo_src.1;
            }
            new_lower.push((lc, lcon));
            new_upper.push((uc, ucon));
        }
        lower = new_lower;
        upper = new_upper;
        // ReLU after every layer except the last.
        if li + 1 < layers.len()
        {
            for o in 0..outf
            {
                let l = dp_concretize(&lower[o], &input.lo, &input.hi, false);
                let u = dp_concretize(&upper[o], &input.lo, &input.hi, true);
                if l >= 0.0
                {
                    // Stable-active: identity (keep the bounds).
                }
                else if u <= 0.0
                {
                    // Stable-inactive: zero.
                    lower[o] = (vec![0.0f32; d_in], 0.0);
                    upper[o] = (vec![0.0f32; d_in], 0.0);
                }
                else
                {
                    // Unstable: chord upper bound, area-minimising lower bound.
                    let slope = u / (u - l);
                    for c in upper[o].0.iter_mut()
                    {
                        *c *= slope;
                    }
                    upper[o].1 = slope * (upper[o].1 - l);
                    let lam = if u > -l { 1.0f32 } else { 0.0f32 };
                    for c in lower[o].0.iter_mut()
                    {
                        *c *= lam;
                    }
                    lower[o].1 *= lam;
                }
            }
        }
    }
    let outf = layers.last().unwrap().out_f;
    let lo = (0..outf)
        .map(|o| dp_concretize(&lower[o], &input.lo, &input.hi, false))
        .collect();
    let hi = (0..outf)
        .map(|o| dp_concretize(&upper[o], &input.lo, &input.hi, true))
        .collect();
    Interval { lo, hi }
}

/// A **zonotope** over `n` dimensions: a center `c` plus `m` generator vectors,
/// concretizing to `{ c + Σ εᵢ gᵢ : εᵢ ∈ [−1, 1] }`. Affine maps transform it
/// **exactly**; ReLU is over-approximated (DeepZ). The shared `εᵢ` let it track
/// linear **correlations** between coordinates that an [`Interval`] cannot — the
/// abstract domain behind AI² (Gehr et al., IEEE S&P 2018).
#[derive(Clone, Debug)]
pub struct Zonotope {
    /// Center, length `n`.
    pub center: Vec<f32>,
    /// Generator vectors, each length `n`.
    pub generators: Vec<Vec<f32>>,
}

impl Zonotope {
    /// The zonotope exactly representing an input box: one generator per dimension
    /// of nonzero radius.
    pub fn from_interval(iv: &Interval) -> Self {
        let n = iv.lo.len();
        let center: Vec<f32> = (0..n).map(|i| 0.5 * (iv.lo[i] + iv.hi[i])).collect();
        let mut generators = Vec::new();
        for i in 0..n
        {
            let r = 0.5 * (iv.hi[i] - iv.lo[i]);
            if r != 0.0
            {
                let mut g = vec![0.0f32; n];
                g[i] = r;
                generators.push(g);
            }
        }
        Self { center, generators }
    }

    /// Per-dimension radius `Σᵢ |gᵢ|` (total generator spread).
    pub fn radii(&self) -> Vec<f32> {
        let n = self.center.len();
        let mut r = vec![0.0f32; n];
        for g in &self.generators
        {
            for (i, ri) in r.iter_mut().enumerate()
            {
                *ri += g[i].abs();
            }
        }
        r
    }

    /// The tightest enclosing box (interval concretization).
    pub fn interval(&self) -> Interval {
        let r = self.radii();
        let n = self.center.len();
        Interval {
            lo: (0..n).map(|i| self.center[i] - r[i]).collect(),
            hi: (0..n).map(|i| self.center[i] + r[i]).collect(),
        }
    }
}

impl IbpLinear {
    /// Affine forward of a zonotope — **exact**: `c → c·W + b`, each generator
    /// `g → g·W` (generators are unaffected by the bias).
    pub fn forward_zonotope(&self, z: &Zonotope) -> Zonotope {
        assert_eq!(z.center.len(), self.in_f, "forward_zonotope: input size");
        let map = |v: &[f32], bias: bool| -> Vec<f32> {
            let mut out = vec![0.0f32; self.out_f];
            for (o, oo) in out.iter_mut().enumerate()
            {
                let mut acc = if bias { self.b[o] } else { 0.0 };
                for (i, &vi) in v.iter().enumerate()
                {
                    acc += vi * self.w[i * self.out_f + o];
                }
                *oo = acc;
            }
            out
        };
        Zonotope {
            center: map(&z.center, true),
            generators: z.generators.iter().map(|g| map(g, false)).collect(),
        }
    }
}

/// ReLU on a zonotope (**DeepZ**, Singh et al. 2018): per neuron with bounds
/// `[l, u]` (from the zonotope) — if `l ≥ 0` keep it, if `u ≤ 0` zero it, else apply
/// the minimal-area relaxation `y = λx + μ` with one fresh noise term of magnitude
/// `μ`, where `λ = u/(u−l)` and `μ = −λl/2`. Sound: it encloses `max(0, x)` for every
/// `x ∈ [l, u]`, while the shared old generators (scaled by `λ`) preserve correlation.
pub fn relu_zonotope(z: &Zonotope) -> Zonotope {
    let n = z.center.len();
    let iv = z.interval();
    let mut new_center = vec![0.0f32; n];
    let mut lambda = vec![0.0f32; n];
    let mut mu = vec![0.0f32; n]; // fresh-generator magnitude (0 ⇒ stable neuron)
    for j in 0..n
    {
        let (l, u) = (iv.lo[j], iv.hi[j]);
        if l >= 0.0
        {
            lambda[j] = 1.0;
            new_center[j] = z.center[j];
        }
        else if u <= 0.0
        {
            new_center[j] = 0.0; // lambda[j] stays 0
        }
        else
        {
            let lam = u / (u - l);
            lambda[j] = lam;
            mu[j] = -lam * l * 0.5; // > 0
            new_center[j] = lam * z.center[j] + mu[j];
        }
    }
    // Scale existing generators by λ per dimension (keeps correlations).
    let mut generators: Vec<Vec<f32>> = z
        .generators
        .iter()
        .map(|g| (0..n).map(|j| lambda[j] * g[j]).collect())
        .collect();
    // One fresh generator per unstable neuron.
    for (j, &m) in mu.iter().enumerate()
    {
        if m != 0.0
        {
            let mut g = vec![0.0f32; n];
            g[j] = m;
            generators.push(g);
        }
    }
    Zonotope {
        center: new_center,
        generators,
    }
}

/// Given a certified output box, is class `target` **provably** the argmax over
/// the *whole* input box? True iff `target`'s lower bound exceeds every other
/// class's upper bound — the standard IBP robustness certificate.
pub fn certified_robust(out: &Interval, target: usize) -> bool {
    let t_lo = out.lo[target];
    out.hi
        .iter()
        .enumerate()
        .all(|(j, &hj)| j == target || t_lo > hj)
}

/// **CROWN** certified bounds (Zhang et al. 2018) for a one-hidden-layer ReLU
/// network `y = L2(relu(L1(x)))` over an input box. Unlike IBP — which takes the
/// interval at *every* layer — CROWN keeps a **linear** lower/upper bound of the
/// output in terms of the input through the ReLU relaxation, and only intervalises
/// at the very end. That back-substitution makes the certified box **tighter than
/// (or equal to) IBP**.
///
/// ReLU relaxation per hidden neuron with pre-activation bounds `[l, u]`:
/// stable neurons are exact (`relu = z` or `0`); for an unstable neuron the upper
/// bound is the chord through `(l,0)`–`(u,u)` and the lower bound is `α·z` with
/// the area-minimising `α ∈ {0,1}`.
pub fn crown_bounds(l1: &IbpLinear, l2: &IbpLinear, input: &Interval) -> Interval {
    assert_eq!(l1.out_f, l2.in_f, "crown: layer dims must chain");
    assert_eq!(input.lo.len(), l1.in_f, "crown: input size");
    let (in_f, hid, out_f) = (l1.in_f, l1.out_f, l2.out_f);

    // Pre-activation bounds of the hidden layer (interval is exact for affine).
    let pre = l1.forward_interval(input);

    // ReLU relaxation: relu(z) ≥ sl·z + il and relu(z) ≤ su·z + iu.
    // `il` (lower intercept) stays 0 for every relaxation case, so it needs no `mut`.
    let (mut sl, il) = (vec![0f32; hid], vec![0f32; hid]);
    let (mut su, mut iu) = (vec![0f32; hid], vec![0f32; hid]);
    for h in 0..hid
    {
        let (zlo, zhi) = (pre.lo[h], pre.hi[h]);
        if zlo >= 0.0
        {
            sl[h] = 1.0;
            su[h] = 1.0;
        }
        else if zhi <= 0.0
        {
            // stays zero
        }
        else
        {
            let s = zhi / (zhi - zlo);
            su[h] = s;
            iu[h] = -s * zlo; // chord through (zlo,0)–(zhi,zhi)
            sl[h] = if zhi >= -zlo { 1.0 } else { 0.0 }; // area-minimising α
        }
    }

    let c: Vec<f32> = input
        .lo
        .iter()
        .zip(&input.hi)
        .map(|(&a, &b)| 0.5 * (a + b))
        .collect();
    let r: Vec<f32> = input
        .lo
        .iter()
        .zip(&input.hi)
        .map(|(&a, &b)| 0.5 * (b - a))
        .collect();

    let mut lo = vec![0f32; out_f];
    let mut hi = vec![0f32; out_f];
    for o in 0..out_f
    {
        // Build a linear bound a + cᵀx, then intervalise over the box.
        // `upper = false` → a *lower* bound on y[o]; `true` → an upper bound.
        let bound = |upper: bool| -> f32 {
            let mut konst = l2.b[o];
            let mut coef = vec![0f32; in_f];
            for h in 0..hid
            {
                let w2 = l2.w[h * out_f + o];
                // For a lower bound: w2≥0 picks the relu lower relaxation, w2<0 the
                // upper; for an upper bound, the opposite.
                let pick_lower = (w2 >= 0.0) ^ upper;
                let (s, ic) = if pick_lower
                {
                    (sl[h], il[h])
                }
                else
                {
                    (su[h], iu[h])
                };
                let cz = w2 * s;
                konst += w2 * ic + cz * l1.b[h];
                if cz != 0.0
                {
                    for (i, ci) in coef.iter_mut().enumerate()
                    {
                        *ci += cz * l1.w[i * hid + h];
                    }
                }
            }
            // min (upper=false) or max (upper=true) of a + cᵀx over the box.
            let mut y = konst;
            for i in 0..in_f
            {
                y += coef[i] * c[i] + if upper { coef[i].abs() } else { -coef[i].abs() } * r[i];
            }
            y
        };
        lo[o] = bound(false);
        hi[o] = bound(true);
    }
    Interval { lo, hi }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::nd::NdTape;
    use crate::nn::PcgEngine;
    use crate::tensor::tensor_nd::TensorND;

    /// A point in the box: `x_i = lo_i + u·(hi_i − lo_i)`, `u ∈ [0,1)`.
    fn sample(b: &Interval, rng: &mut PcgEngine) -> Vec<f32> {
        b.lo.iter()
            .zip(&b.hi)
            .map(|(&l, &h)| l + rng.float() * (h - l))
            .collect()
    }

    /// IbpLinear's concrete forward matches the real `NdLinear` tape forward, so
    /// the certifier reasons about the *same* function the network computes.
    #[test]
    fn ibp_matches_nd_linear_forward() {
        let mut rng = PcgEngine::new(1);
        let mut lin = NdLinear::new(4, 3, &mut rng);
        let ibp = IbpLinear::from_nd_linear(&lin); // immutable borrow ends here
        let x = [0.3f32, -0.7, 0.1, 0.9];

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.to_vec(), vec![1, 4]));
        let y_tape = t.value(lin.forward(&t, xv)).data;
        let y_ibp = ibp.forward_point(&x);
        for (a, b) in y_tape.iter().zip(&y_ibp)
        {
            assert!((a - b).abs() < 1e-5, "tape {a} vs ibp {b}");
        }
    }

    /// **Soundness**: thousands of concrete points sampled from the input box all
    /// land inside the certified output interval, for a 3-layer ReLU MLP built
    /// from real `NdLinear` weights.
    #[test]
    fn ibp_mlp_soundness() {
        let mut rng = PcgEngine::new(5);
        let l1 = NdLinear::new(4, 8, &mut rng);
        let l2 = NdLinear::new(8, 8, &mut rng);
        let l3 = NdLinear::new(8, 3, &mut rng);
        let mlp = IbpMlp::new(vec![
            IbpLinear::from_nd_linear(&l1),
            IbpLinear::from_nd_linear(&l2),
            IbpLinear::from_nd_linear(&l3),
        ]);

        let centre = [0.2f32, -0.5, 0.7, -0.1];
        let box_in = Interval::around(&centre, 0.1);
        let cert = mlp.certify(&box_in);

        let mut srng = PcgEngine::new(123);
        for _ in 0..4000
        {
            let x = sample(&box_in, &mut srng);
            let y = mlp.forward(&x);
            for (k, &yk) in y.iter().enumerate()
            {
                assert!(
                    yk >= cert.lo[k] - 1e-4 && yk <= cert.hi[k] + 1e-4,
                    "soundness violated at out {k}: {yk} not in [{}, {}]",
                    cert.lo[k],
                    cert.hi[k]
                );
            }
        }
    }

    /// The affine interval is **exact**: for a single affine layer the certified
    /// box endpoints are attained at the box corners (here, the sampled extrema
    /// reach the bounds up to sampling error).
    #[test]
    fn ibp_affine_is_tight() {
        let ibp = IbpLinear::new(vec![1.0, -2.0, 0.5, 3.0], vec![0.0, 0.0], 2, 2);
        let b = Interval::around(&[0.0, 0.0], 1.0);
        let cert = ibp.forward_interval(&b);
        // out0 = x0 + 0.5 x1 → range [-1.5, 1.5]; out1 = -2 x0 + 3 x1 → [-5, 5].
        assert!((cert.lo[0] + 1.5).abs() < 1e-6 && (cert.hi[0] - 1.5).abs() < 1e-6);
        assert!((cert.lo[1] + 5.0).abs() < 1e-6 && (cert.hi[1] - 5.0).abs() < 1e-6);
    }

    /// A small enough box certifies the prediction; a large box does not — the
    /// certificate is honest in both directions.
    #[test]
    fn ibp_robustness_certificate() {
        // Two classes: class 0 favours large x0, class 1 large x1.
        let mlp = IbpMlp::new(vec![IbpLinear::new(
            vec![2.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0],
            2,
            2,
        )]);
        let centre = [1.0f32, 0.0]; // clearly class 0
        assert_eq!(
            mlp.forward(&centre)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
            0
        );

        let tight = mlp.certify(&Interval::around(&centre, 0.3));
        assert!(
            certified_robust(&tight, 0),
            "small box should certify class 0"
        );

        let loose = mlp.certify(&Interval::around(&centre, 1.5));
        assert!(
            !certified_robust(&loose, 0),
            "large box must not falsely certify"
        );
    }

    /// CROWN is **sound** (every sampled output lands in its certified box) and
    /// **tighter than IBP** (each output's certified width is ≤ the IBP width)
    /// for a one-hidden-layer ReLU network.
    #[test]
    fn crown_is_sound_and_tighter_than_ibp() {
        let mut rng = PcgEngine::new(4);
        let l1 = NdLinear::new(4, 8, &mut rng);
        let l2 = NdLinear::new(8, 3, &mut rng);
        let il1 = IbpLinear::from_nd_linear(&l1);
        let il2 = IbpLinear::from_nd_linear(&l2);

        let centre = [0.2f32, -0.5, 0.7, -0.1];
        let box_in = Interval::around(&centre, 0.15);
        let crown = crown_bounds(&il1, &il2, &box_in);

        // IBP bounds for the same network (consumes the layers).
        let mlp = IbpMlp::new(vec![il1, il2]);
        let ibp = mlp.certify(&box_in);

        // Tighter (or equal): CROWN width ≤ IBP width per output.
        for o in 0..3
        {
            let cw = crown.hi[o] - crown.lo[o];
            let iw = ibp.hi[o] - ibp.lo[o];
            assert!(
                cw <= iw + 1e-5,
                "CROWN not tighter at {o}: crown {cw} vs ibp {iw}"
            );
        }

        // Sound: sample the box; every output is inside the CROWN box.
        let mut srng = PcgEngine::new(99);
        for _ in 0..4000
        {
            let x: Vec<f32> = box_in
                .lo
                .iter()
                .zip(&box_in.hi)
                .map(|(&l, &h)| l + srng.float() * (h - l))
                .collect();
            let y = mlp.forward(&x);
            for (o, &yo) in y.iter().enumerate()
            {
                assert!(
                    yo >= crown.lo[o] - 1e-4 && yo <= crown.hi[o] + 1e-4,
                    "CROWN unsound at out {o}: {yo} not in [{}, {}]",
                    crown.lo[o],
                    crown.hi[o]
                );
            }
        }
    }

    /// The zonotope affine map is **exact**: its interval concretization equals the
    /// (exact) interval affine forward.
    #[test]
    fn zonotope_affine_is_exact() {
        let mut rng = PcgEngine::new(2);
        let lin = IbpLinear::from_nd_linear(&NdLinear::new(4, 3, &mut rng));
        let box_in = Interval::around(&[0.1, -0.4, 0.6, 0.2], 0.15);
        let zi = lin
            .forward_zonotope(&Zonotope::from_interval(&box_in))
            .interval();
        let ii = lin.forward_interval(&box_in);
        for k in 0..3
        {
            assert!((zi.lo[k] - ii.lo[k]).abs() < 1e-5, "lo mismatch at {k}");
            assert!((zi.hi[k] - ii.hi[k]).abs() < 1e-5, "hi mismatch at {k}");
        }
    }

    /// **Soundness of the zonotope abstraction**: thousands of concrete points from
    /// the input box land inside the certified zonotope box, for a 3-layer ReLU MLP.
    #[test]
    fn zonotope_mlp_soundness() {
        let mut rng = PcgEngine::new(8);
        let mlp = IbpMlp::new(vec![
            IbpLinear::from_nd_linear(&NdLinear::new(4, 8, &mut rng)),
            IbpLinear::from_nd_linear(&NdLinear::new(8, 6, &mut rng)),
            IbpLinear::from_nd_linear(&NdLinear::new(6, 3, &mut rng)),
        ]);
        let box_in = Interval::around(&[0.2, -0.5, 0.7, -0.1], 0.1);
        let cert = mlp.certify_zonotope(&box_in);

        let mut srng = PcgEngine::new(321);
        for _ in 0..4000
        {
            let x = sample(&box_in, &mut srng);
            let y = mlp.forward(&x);
            for (k, &yk) in y.iter().enumerate()
            {
                assert!(
                    yk >= cert.lo[k] - 1e-4 && yk <= cert.hi[k] + 1e-4,
                    "zonotope unsound at out {k}: {yk} not in [{}, {}]",
                    cert.lo[k],
                    cert.hi[k]
                );
            }
        }
    }

    /// **Zonotopes beat IBP when correlations matter.** The network computes
    /// `relu(x) − relu(x)` (always 0): IBP loses the correlation and reports
    /// `[−1, 1]`, while the zonotope keeps the shared noise symbol and reports a
    /// **strictly tighter** box — both sound (they contain 0).
    #[test]
    fn zonotope_tighter_than_ibp_under_correlation() {
        // x ∈ [-1,1]; L1: x ↦ (x, x); L2: (a,b) ↦ a − b.
        let l1 = IbpLinear::new(vec![1.0, 1.0], vec![0.0, 0.0], 1, 2);
        let l2 = IbpLinear::new(vec![1.0, -1.0], vec![0.0], 2, 1);
        let mlp = IbpMlp::new(vec![l1, l2]);
        let box_in = Interval {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let ibp = mlp.certify(&box_in);
        let zono = mlp.certify_zonotope(&box_in);

        let ibp_w = ibp.hi[0] - ibp.lo[0];
        let zono_w = zono.hi[0] - zono.lo[0];
        assert!(
            zono_w < ibp_w - 1e-4,
            "zonotope {zono_w} not tighter than IBP {ibp_w}"
        );
        // Both sound: the true output (0) is contained.
        assert!(
            zono.lo[0] <= 0.0 && zono.hi[0] >= 0.0,
            "zonotope excludes truth"
        );
        assert!(ibp.lo[0] <= 0.0 && ibp.hi[0] >= 0.0);
    }

    /// **DeepPoly is sound**: 4000 sampled inputs of a 3-layer ReLU MLP land inside
    /// the certified box (and the result is deterministic).
    #[test]
    fn deeppoly_mlp_soundness() {
        let mut rng = PcgEngine::new(8);
        let mlp = IbpMlp::new(vec![
            IbpLinear::from_nd_linear(&NdLinear::new(4, 8, &mut rng)),
            IbpLinear::from_nd_linear(&NdLinear::new(8, 6, &mut rng)),
            IbpLinear::from_nd_linear(&NdLinear::new(6, 3, &mut rng)),
        ]);
        let box_in = Interval::around(&[0.2, -0.5, 0.7, -0.1], 0.1);
        let cert = mlp.certify_deeppoly(&box_in);
        assert_eq!(cert.lo, mlp.certify_deeppoly(&box_in).lo); // deterministic

        let mut srng = PcgEngine::new(321);
        for _ in 0..4000
        {
            let x = sample(&box_in, &mut srng);
            let y = mlp.forward(&x);
            for (k, &yk) in y.iter().enumerate()
            {
                assert!(
                    yk >= cert.lo[k] - 1e-4 && yk <= cert.hi[k] + 1e-4,
                    "deeppoly unsound at out {k}: {yk} not in [{}, {}]",
                    cert.lo[k],
                    cert.hi[k]
                );
            }
        }
    }

    /// **DeepPoly beats IBP via relational bounds.** The network computes
    /// `relu(x) + relu(−x) = |x|` over `x ∈ [−1, 1]`. IBP treats the two ReLUs
    /// independently and reports `[0, 2]`; DeepPoly keeps each bound affine in `x`,
    /// so the `x` cancels in the upper bound and it reports the **exact** `[0, 1]`.
    #[test]
    fn deeppoly_tighter_than_ibp_on_abs() {
        // L1: x ↦ (x, −x); L2: (a, b) ↦ a + b.
        let l1 = IbpLinear::new(vec![1.0, -1.0], vec![0.0, 0.0], 1, 2);
        let l2 = IbpLinear::new(vec![1.0, 1.0], vec![0.0], 2, 1);
        let mlp = IbpMlp::new(vec![l1, l2]);
        let box_in = Interval {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let ibp = mlp.certify(&box_in);
        let dp = mlp.certify_deeppoly(&box_in);

        assert!(
            (dp.hi[0] - dp.lo[0]) < (ibp.hi[0] - ibp.lo[0]) - 1e-4,
            "deeppoly [{}, {}] not tighter than IBP [{}, {}]",
            dp.lo[0],
            dp.hi[0],
            ibp.lo[0],
            ibp.hi[0]
        );
        // Sound and in fact exact: |x| over [−1,1] is [0, 1].
        assert!(
            dp.lo[0] <= 1e-5 && (dp.hi[0] - 1.0).abs() < 1e-5,
            "deeppoly box {dp:?}"
        );
        assert!(ibp.lo[0] <= 0.0 && ibp.hi[0] >= 1.0); // IBP sound but loose ([0,2])
    }
}
