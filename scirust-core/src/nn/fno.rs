//! **Fourier Neural Operator (FNO)** — Li et al., *Fourier Neural Operator for
//! Parametric Partial Differential Equations*, ICLR 2021 (arXiv:2010.08895).
//!
//! A neural *operator* learns a map between **functions** (e.g. an initial
//! condition ↦ a PDE solution) rather than between fixed-size vectors. FNO does
//! the global kernel integral in the **Fourier domain**: transform the (sampled)
//! input function, keep the lowest `modes` frequencies, multiply each by a
//! learnable **complex weight**, and transform back.
//!
//! ```text
//! v' = IDFT( R ⊙ DFT(v) )           (spectral convolution, low-pass parameterised)
//! u_{ℓ+1} = σ( SpectralConv(u_ℓ) + W·u_ℓ )   (one FNO block: global + local)
//! ```
//!
//! The real DFT and its inverse are **fixed cosine/sine matrices**, so the whole
//! transform is an ordinary (deterministic) matmul that the N-D autograd tape
//! differentiates directly — no FFT, no complex-number type, no new op. The
//! per-mode complex weights are applied with a **batched** matmul ([`NdVar::bmm`])
//! over the kept modes, mixing the `width` channels. Because differentiation is
//! diagonal in Fourier space (`d/dx ↔ ×ik`), a single spectral convolution can
//! *exactly* represent the derivative operator — a property the tests exploit to
//! show FNO learns an operator and **generalises** to unseen inputs.
//!
//! Pure, deterministic `f32`; gradient-checked.

use crate::autodiff::nd::{NdTape, NdVar};
use crate::nn::nd_layers::NdLinear;
use crate::nn::nd_optim::NdParam;
use crate::nn::rng::PcgEngine;
use crate::tensor::tensor_nd::TensorND;
use std::f32::consts::TAU;

/// Forward truncated real-DFT basis (the first `modes` frequencies) for a length-`n`
/// grid: `Cfwd[k,j] = cos(2πkj/n)`, `Sfwd[k,j] = −sin(2πkj/n)` (the `−i·sin` of
/// `e^{−2πikj/n}`), each row-major `(modes, n)`. With a signal `v` of shape
/// `(n, width)`, `Cfwd·v` and `Sfwd·v` are the real/imaginary spectra at those modes.
fn dft_forward(n: usize, modes: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cf = vec![0f32; modes * n];
    let mut sf = vec![0f32; modes * n];
    for k in 0..modes
    {
        for j in 0..n
        {
            let a = TAU * (k as f32) * (j as f32) / n as f32;
            cf[k * n + j] = a.cos();
            sf[k * n + j] = -a.sin();
        }
    }
    (cf, sf)
}

/// Inverse real-DFT basis reconstructing a length-`n` **real** signal from its first
/// `modes` frequencies, with the conjugate-symmetry factor `f₀=1, f_{k≥1}=2`:
/// `ICos[j,k] = f_k·cos(2πkj/n)/n`, `ISin[j,k] = f_k·sin(2πkj/n)/n`, each `(n, modes)`.
/// Then `v' = ICos·Outre − ISin·Outim`. Exact for signals band-limited to the kept
/// modes (the factor 2 folds in the symmetric negative frequencies).
fn dft_inverse(n: usize, modes: usize) -> (Vec<f32>, Vec<f32>) {
    let mut ic = vec![0f32; n * modes];
    let mut is = vec![0f32; n * modes];
    for j in 0..n
    {
        for k in 0..modes
        {
            let f = if k == 0 { 1.0 } else { 2.0 };
            let a = TAU * (k as f32) * (j as f32) / n as f32;
            ic[j * modes + k] = f * a.cos() / n as f32;
            is[j * modes + k] = f * a.sin() / n as f32;
        }
    }
    (ic, is)
}

/// **1-D spectral convolution** — the heart of an FNO layer. Keeps the lowest
/// `modes` Fourier frequencies of a `(n, width)` signal and multiplies each by a
/// learnable per-mode **complex** weight matrix `R_k = Ar_k + i·Ai_k`
/// (`width × width`, channel-mixing), then inverts the transform:
///
/// ```text
/// V = DFT(v) ;  Out_k = R_k · V_k ;  v' = IDFT(Out)
/// ```
///
/// All of `DFT`, the per-mode complex matmul (via [`NdVar::bmm`] over the modes),
/// and `IDFT` run on the N-D tape, so gradients w.r.t. both the signal and the
/// weights are exact. Deterministic; gradient-checked.
pub struct FnoSpectralConv1d {
    ar: TensorND, // (modes, width, width) — real part of per-mode weights
    ai: TensorND, // (modes, width, width) — imaginary part
    n: usize,
    modes: usize,
    width: usize,
    ar_idx: Option<usize>,
    ai_idx: Option<usize>,
}

impl FnoSpectralConv1d {
    /// New spectral conv over a length-`n` grid keeping `modes` frequencies, with
    /// `width` channels. Weights are seeded with the FNO `1/width` scaling.
    pub fn new(n: usize, modes: usize, width: usize, rng: &mut PcgEngine) -> Self {
        assert!(modes <= n, "modes must not exceed the grid length");
        let scale = 1.0 / width as f32;
        let mut ar = vec![0f32; modes * width * width];
        let mut ai = vec![0f32; modes * width * width];
        for x in ar.iter_mut()
        {
            *x = rng.float_signed() * scale;
        }
        for x in ai.iter_mut()
        {
            *x = rng.float_signed() * scale;
        }
        Self {
            ar: TensorND::new(ar, vec![modes, width, width]),
            ai: TensorND::new(ai, vec![modes, width, width]),
            n,
            modes,
            width,
            ar_idx: None,
            ai_idx: None,
        }
    }

    /// Forward over a `(n, width)` signal; returns `(n, width)`.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, v: NdVar<'t>) -> NdVar<'t> {
        let (m, w) = (self.modes, self.width);
        let (cf, sf) = dft_forward(self.n, m);
        let cfv = tape.input(TensorND::new(cf, vec![m, self.n]));
        let sfv = tape.input(TensorND::new(sf, vec![m, self.n]));
        let vre = cfv.matmul(v).reshape(&[m, w, 1]); // Re V  (modes,width,1)
        let vim = sfv.matmul(v).reshape(&[m, w, 1]); // Im V
        let arv = tape.input(self.ar.clone());
        self.ar_idx = Some(arv.idx());
        let aiv = tape.input(self.ai.clone());
        self.ai_idx = Some(aiv.idx());
        // Complex multiply per mode: Out = (Ar + iAi)(Vre + iVim).
        let outre = arv.bmm(vre).sub(aiv.bmm(vim)).reshape(&[m, w]);
        let outim = arv.bmm(vim).add(aiv.bmm(vre)).reshape(&[m, w]);
        let (ic, is) = dft_inverse(self.n, m);
        let icv = tape.input(TensorND::new(ic, vec![self.n, m]));
        let isv = tape.input(TensorND::new(is, vec![self.n, m]));
        icv.matmul(outre).sub(isv.matmul(outim)) // (n, width)
    }

    /// Trainable parameters (the real and imaginary per-mode weights).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let (ar_idx, ai_idx) = (self.ar_idx, self.ai_idx);
        let mut params = Vec::new();
        if let Some(i) = ar_idx
        {
            params.push(NdParam {
                value: &mut self.ar,
                grad_idx: i,
            });
        }
        if let Some(i) = ai_idx
        {
            params.push(NdParam {
                value: &mut self.ai,
                grad_idx: i,
            });
        }
        params
    }
}

/// **FNO block** — one Fourier-operator layer: lift the `in_ch` input channels to a
/// `width`-dimensional channel space, run a global [`FnoSpectralConv1d`] in parallel
/// with a **local** pointwise linear `W`, sum them, apply a ReLU non-linearity, and
/// project to `out_ch`:
///
/// ```text
/// v = P·x ;  y = σ( SpectralConv(v) + W·v ) ;  out = Q·y
/// ```
///
/// Deterministic; trainable through the N-D tape. `(n, in_ch) → (n, out_ch)`.
pub struct NdFno {
    lift: NdLinear,              // in_ch → width
    spectral: FnoSpectralConv1d, // global Fourier kernel
    local: NdLinear,             // width → width (pointwise)
    proj: NdLinear,              // width → out_ch
}

impl NdFno {
    /// New FNO block over a length-`n` grid.
    pub fn new(
        n: usize,
        in_ch: usize,
        out_ch: usize,
        width: usize,
        modes: usize,
        rng: &mut PcgEngine,
    ) -> Self {
        Self {
            lift: NdLinear::new(in_ch, width, rng),
            spectral: FnoSpectralConv1d::new(n, modes, width, rng),
            local: NdLinear::new(width, width, rng),
            proj: NdLinear::new(width, out_ch, rng),
        }
    }

    /// Forward over a `(n, in_ch)` sampled function; returns `(n, out_ch)`.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let v = self.lift.forward(tape, x); // (n, width)
        let spec = self.spectral.forward(tape, v); // global
        let loc = self.local.forward(tape, v); // local
        let y = spec.add(loc).relu(); // σ(global + local)
        self.proj.forward(tape, y) // (n, out_ch)
    }

    /// Trainable parameters (lift, spectral weights, local linear, projection).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.lift.parameters();
        params.extend(self.spectral.parameters());
        params.extend(self.local.parameters());
        params.extend(self.proj.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::nd_optim::NdAdam;

    fn mse<'t>(pred: NdVar<'t>, target: NdVar<'t>) -> NdVar<'t> {
        let diff = pred.sub(target);
        diff.mul(diff).sum()
    }

    /// With **identity** per-mode weights the spectral conv is `IDFT∘DFT`, which
    /// reconstructs any signal **band-limited to the kept modes** exactly — this
    /// validates the forward/inverse DFT matrices and the factor-2 one-sided inverse.
    #[test]
    fn fno_spectral_conv_reconstructs_bandlimited() {
        let (n, modes) = (16usize, 5usize);
        let mut rng = PcgEngine::new(1);
        let mut conv = FnoSpectralConv1d::new(n, modes, 1, &mut rng);
        // Identity weight per mode (width 1 ⇒ Ar_k = 1, Ai_k = 0).
        conv.ar = TensorND::new(vec![1.0f32; modes], vec![modes, 1, 1]);
        conv.ai = TensorND::new(vec![0.0f32; modes], vec![modes, 1, 1]);
        // Band-limited signal: only frequencies 0..modes present.
        let v: Vec<f32> = (0..n)
            .map(|j| {
                let x = TAU * j as f32 / n as f32;
                1.0 + (x).cos() + 0.5 * (2.0 * x).sin() - 0.3 * (3.0 * x).cos()
            })
            .collect();
        let t = NdTape::new();
        let vv = t.input(TensorND::new(v.clone(), vec![n, 1]));
        let got = t.value(conv.forward(&t, vv));
        for (g, w) in got.data.iter().zip(&v)
        {
            assert!((g - w).abs() < 1e-4, "reconstruction: got {g}, want {w}");
        }
    }

    /// `FnoSpectralConv1d` gradients (w.r.t. the signal and both weight parts) match
    /// finite differences — the spectral convolution is linear and smooth.
    #[test]
    fn fno_spectral_conv_gradient_check() {
        let (n, modes, width) = (6usize, 3usize, 2usize);
        let mut rng = PcgEngine::new(2);
        let mut conv = FnoSpectralConv1d::new(n, modes, width, &mut rng);
        let v: Vec<f32> = (0..n * width)
            .map(|i| (i as f32 * 0.3 - 0.5).sin())
            .collect();
        let ar0 = conv.ar.data.clone();
        let ai0 = conv.ai.data.clone();

        let loss_of = |vv: &[f32], aar: &[f32], aai: &[f32]| -> f32 {
            let mut rng = PcgEngine::new(2);
            let mut c = FnoSpectralConv1d::new(n, modes, width, &mut rng);
            c.ar = TensorND::new(aar.to_vec(), vec![modes, width, width]);
            c.ai = TensorND::new(aai.to_vec(), vec![modes, width, width]);
            let t = NdTape::new();
            let x = t.input(TensorND::new(vv.to_vec(), vec![n, width]));
            let y = c.forward(&t, x);
            t.value(y.mul(y).sum()).data[0]
        };
        let t = NdTape::new();
        let vv = t.input(TensorND::new(v.clone(), vec![n, width]));
        let y = conv.forward(&t, vv);
        let grads = t.backward(y.mul(y).sum());
        let gv = grads[vv.idx()].clone();
        let gar = grads[conv.ar_idx.unwrap()].clone();
        let gai = grads[conv.ai_idx.unwrap()].clone();
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[i]).abs() < 3e-2,
                    "fno grad {i}: numeric {num}, analytic {}",
                    analytic.data[i]
                );
            }
        };
        check(&gv, &v, &|p| loss_of(p, &ar0, &ai0));
        check(&gar, &ar0, &|p| loss_of(&v, p, &ai0));
        check(&gai, &ai0, &|p| loss_of(&v, &ar0, p));
    }

    /// **Operator learning**: a single spectral conv trained to map `sin(ωx+φ)` to
    /// its derivative `ω·cos(ωx+φ)` (a *function-to-function* map). Because
    /// differentiation is diagonal in Fourier space (`×ik`), the operator is exactly
    /// representable and input-independent — so a model fit on a few phases
    /// **generalises** to an unseen phase. Deterministic across identical runs.
    #[test]
    fn fno_spectral_conv_learns_derivative_and_generalizes() {
        let (n, modes) = (16usize, 4usize);
        // sin(ω x + φ) on the grid x_j = 2π j / n, and its x-derivative ω cos(ω x + φ).
        let sample = |omega: f32, phi: f32| -> (Vec<f32>, Vec<f32>) {
            let inp: Vec<f32> = (0..n)
                .map(|j| (omega * TAU * j as f32 / n as f32 + phi).sin())
                .collect();
            let der: Vec<f32> = (0..n)
                .map(|j| omega * (omega * TAU * j as f32 / n as f32 + phi).cos())
                .collect();
            (inp, der)
        };
        let train: Vec<(Vec<f32>, Vec<f32>)> = [1.0, 2.0, 3.0]
            .iter()
            .flat_map(|&w| [0.0f32, 0.7].iter().map(move |&p| (w, p)))
            .map(|(w, p)| sample(w, p))
            .collect();
        let test: Vec<(Vec<f32>, Vec<f32>)> =
            [1.0, 2.0, 3.0].iter().map(|&w| sample(w, 1.5)).collect();

        // One example per tape per step (a fresh tape each step keeps each
        // parameter a single node so its gradient is exact), cycling the training
        // set — SGD on this convex least-squares problem converges to the operator.
        let run = || -> f32 {
            let mut rng = PcgEngine::new(5);
            let mut conv = FnoSpectralConv1d::new(n, modes, 1, &mut rng);
            let mut opt = NdAdam::with_lr(0.05);
            for step in 0..900
            {
                let (inp, der) = &train[step % train.len()];
                let tape = NdTape::new();
                let x = tape.input(TensorND::new(inp.clone(), vec![n, 1]));
                let y = tape.input(TensorND::new(der.clone(), vec![n, 1]));
                let loss = mse(conv.forward(&tape, x), y);
                let grads = tape.backward(loss);
                opt.step(&mut conv.parameters(), &grads);
            }
            // Held-out (unseen phase) test error.
            let mut terr = 0f32;
            for (inp, der) in &test
            {
                let tape = NdTape::new();
                let x = tape.input(TensorND::new(inp.clone(), vec![n, 1]));
                let pred = tape.value(conv.forward(&tape, x));
                for (p, d) in pred.data.iter().zip(der)
                {
                    terr += (p - d) * (p - d);
                }
            }
            terr / (test.len() * n) as f32
        };
        let test_mse = run();
        assert!(
            test_mse < 0.02,
            "FNO did not learn the derivative operator: test MSE {test_mse}"
        );
        // Determinism: identical test error across runs.
        let test_mse2 = run();
        assert_eq!(test_mse.to_bits(), test_mse2.to_bits());
    }

    /// The full `NdFno` block trains (MSE↓ to a target) and is bit-for-bit
    /// deterministic across identical runs.
    #[test]
    fn nd_fno_trains_and_is_deterministic() {
        let n = 8usize;
        let run = || -> (f32, f32) {
            let mut rng = PcgEngine::new(3);
            let mut layer = NdFno::new(n, 1, 1, 4, 4, &mut rng);
            let x: Vec<f32> = (0..n).map(|j| (j as f32 * 0.5 - 1.0).sin()).collect();
            let target: Vec<f32> = (0..n).map(|j| (j as f32 * 0.3).cos()).collect();
            let mut opt = NdAdam::with_lr(0.03);
            let (mut first, mut last) = (0f32, 0f32);
            for step in 0..200
            {
                let tape = NdTape::new();
                let xv = tape.input(TensorND::new(x.clone(), vec![n, 1]));
                let tv = tape.input(TensorND::new(target.clone(), vec![n, 1]));
                let out = layer.forward(&tape, xv);
                let loss = mse(out, tv);
                let lval = tape.value(loss).data[0];
                if step == 0
                {
                    first = lval;
                }
                last = lval;
                let grads = tape.backward(loss);
                opt.step(&mut layer.parameters(), &grads);
            }
            (first, last)
        };
        let (first, last) = run();
        assert!(
            last < first * 0.6,
            "FNO block did not learn: {first} -> {last}"
        );
        let (first2, last2) = run();
        assert_eq!(first.to_bits(), first2.to_bits());
        assert_eq!(last.to_bits(), last2.to_bits());
    }
}
