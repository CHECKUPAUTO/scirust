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
}
