use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::conv_utils::Padding;
use crate::nn::conv2d::Conv2d;
use crate::nn::init::Initializer;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use std::collections::HashMap;

/// Connectionist Temporal Classification (CTC) loss. The blank symbol is the
/// last vocabulary index (`vocab_size - 1`).
pub struct CTCLoss;

impl CTCLoss {
    /// Negative log-likelihood of the target label sequence under CTC, computed
    /// by the exact forward-backward algorithm (log-space α/β), with the correct
    /// gradient w.r.t. `logits`.
    ///
    /// `logits`: `(T, vocab_size)` pre-softmax scores. `targets`: a length-`S`
    /// row of label ids (as `f32`, no blanks). The returned scalar is
    /// differentiable w.r.t. `logits`: it carries the true CTC gradient via a
    /// value-preserving surrogate `⟨logits, G⟩ + (loss − ⟨logits, G⟩)` whose
    /// value equals the loss and whose gradient equals `G = ∂loss/∂logits`.
    pub fn forward<'t>(&self, tape: &'t Tape, logits: Var<'t>, targets: Var<'t>) -> Var<'t> {
        let (t_steps, vocab) = logits.shape();
        let blank = vocab - 1;

        let logits_val = tape.value(logits.idx());
        let log_y = log_softmax_rows(&logits_val.data, t_steps, vocab);

        let tv = tape.value(targets.idx());
        let target: Vec<usize> = tv
            .data
            .iter()
            .map(|&x| (x as usize).min(vocab - 1))
            .collect();

        let (loss, grad) = ctc_forward_backward(&log_y, &target, blank, t_steps, vocab);

        // Surrogate: value == loss, d/d logits == grad.
        let g_const = tape.input(Tensor::from_vec(grad.clone(), t_steps, vocab));
        let inner_val: f32 = logits_val.data.iter().zip(&grad).map(|(a, b)| a * b).sum();
        let correction = tape.input(Tensor::from_vec(vec![loss - inner_val], 1, 1));
        logits.hadamard(g_const).sum().add(correction)
    }
}

/// Row-wise numerically stable log-softmax of a `rows×cols` row-major buffer.
fn log_softmax_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows
    {
        let row = &x[r * cols..(r + 1) * cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = row.iter().map(|&v| (v - m).exp()).sum();
        let ld = denom.ln() + m;
        for c in 0..cols
        {
            out[r * cols + c] = row[c] - ld;
        }
    }
    out
}

#[inline]
fn logsumexp2(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY
    {
        return b;
    }
    if b == f32::NEG_INFINITY
    {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Exact CTC loss (`−ln p(target|x)`) and gradient w.r.t. the pre-softmax logits,
/// via the standard forward-backward recursion in log space. `log_y` is `(T,V)`
/// row-major log-softmax; `target` are label ids (no blanks); `blank` is the
/// blank index. α_t(s) and β_t(s) both include the emission at time t.
fn ctc_forward_backward(
    log_y: &[f32],
    target: &[usize],
    blank: usize,
    t_steps: usize,
    vocab: usize,
) -> (f32, Vec<f32>) {
    const NEG: f32 = f32::NEG_INFINITY;
    let ly = |t: usize, k: usize| log_y[t * vocab + k];

    // Extended label l' = [blank, c1, blank, c2, ..., blank], length U = 2S+1.
    let s = target.len();
    let mut ext = Vec::with_capacity(2 * s + 1);
    ext.push(blank);
    for &c in target
    {
        ext.push(c);
        ext.push(blank);
    }
    let u = ext.len();

    // Minimum time steps to emit the labels (extra step per adjacent repeat).
    let mut min_needed = s;
    for i in 1..s
    {
        if target[i] == target[i - 1]
        {
            min_needed += 1;
        }
    }
    if t_steps == 0 || (s > 0 && t_steps < min_needed.max(1))
    {
        return (f32::INFINITY, vec![0.0f32; t_steps * vocab]);
    }

    // Forward α.
    let mut la = vec![NEG; t_steps * u];
    la[0] = ly(0, ext[0]);
    if u > 1
    {
        la[1] = ly(0, ext[1]);
    }
    for t in 1..t_steps
    {
        for si in 0..u
        {
            let mut acc = la[(t - 1) * u + si];
            if si >= 1
            {
                acc = logsumexp2(acc, la[(t - 1) * u + si - 1]);
            }
            if si >= 2 && ext[si] != blank && ext[si] != ext[si - 2]
            {
                acc = logsumexp2(acc, la[(t - 1) * u + si - 2]);
            }
            la[t * u + si] = acc + ly(t, ext[si]);
        }
    }

    // Backward β.
    let mut lb = vec![NEG; t_steps * u];
    lb[(t_steps - 1) * u + (u - 1)] = ly(t_steps - 1, ext[u - 1]);
    if u >= 2
    {
        lb[(t_steps - 1) * u + (u - 2)] = ly(t_steps - 1, ext[u - 2]);
    }
    for t in (0..t_steps - 1).rev()
    {
        for si in 0..u
        {
            let mut acc = lb[(t + 1) * u + si];
            if si + 1 < u
            {
                acc = logsumexp2(acc, lb[(t + 1) * u + si + 1]);
            }
            if si + 2 < u && ext[si] != blank && ext[si] != ext[si + 2]
            {
                acc = logsumexp2(acc, lb[(t + 1) * u + si + 2]);
            }
            lb[t * u + si] = acc + ly(t, ext[si]);
        }
    }

    // ln p from the two valid final states.
    let last = (t_steps - 1) * u;
    let ln_p = if u >= 2
    {
        logsumexp2(la[last + u - 1], la[last + u - 2])
    }
    else
    {
        la[last + u - 1]
    };
    if !ln_p.is_finite()
    {
        return (f32::INFINITY, vec![0.0f32; t_steps * vocab]);
    }

    // g_tk = y_tk − (1/p) Σ_{s: ext[s]=k} α_t(s) β_t(s) / y_{t,k}.
    let mut grad = vec![0.0f32; t_steps * vocab];
    for t in 0..t_steps
    {
        let mut acc = vec![NEG; vocab];
        for si in 0..u
        {
            let k = ext[si];
            acc[k] = logsumexp2(acc[k], la[t * u + si] + lb[t * u + si]);
        }
        for k in 0..vocab
        {
            let yk = ly(t, k).exp();
            let posterior = if acc[k] == NEG
            {
                0.0
            }
            else
            {
                // acc holds log(α β), which double-counts y_{t,k}; divide once.
                (acc[k] - ly(t, k) - ln_p).exp()
            };
            grad[t * vocab + k] = yk - posterior;
        }
    }

    (-ln_p, grad)
}

/// Basic Audio Encoder (CNN-based).
pub struct AudioEncoder {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
}

impl AudioEncoder {
    pub fn new<W: Initializer, B: Initializer>(
        in_channels: usize,
        hidden_channels: usize,
        out_channels: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let conv1 = Conv2d::new(
            in_channels,
            hidden_channels,
            3,
            2,
            Padding::Same,
            w_init,
            Some(b_init),
            rng,
        );
        let conv2 = Conv2d::new(
            hidden_channels,
            out_channels,
            3,
            2,
            Padding::Same,
            w_init,
            Some(b_init),
            rng,
        );
        Self { conv1, conv2 }
    }
}

impl Module for AudioEncoder {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let x = self.conv1.forward(tape, input);
        let x = x.relu();
        self.conv2.forward(tape, x).relu()
    }

    fn train(&mut self, on: bool) {
        self.conv1.train(on);
        self.conv2.train(on);
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.conv1.parameter_indices());
        v.extend(self.conv2.parameter_indices());
        v
    }

    fn sync(&mut self, tape: &Tape) {
        self.conv1.sync(tape);
        self.conv2.sync(tape);
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        // The convs' default names are not globally unique (both could collide
        // for some channel configs), so prefix manually like TransformerBlock
        // does for its unnamed FFN linears.
        let mut map = HashMap::new();
        map.insert("conv1.weight".to_string(), self.conv1.weight.clone());
        if let Some(b) = &self.conv1.bias
        {
            map.insert("conv1.bias".to_string(), b.clone());
        }
        map.insert("conv2.weight".to_string(), self.conv2.weight.clone());
        if let Some(b) = &self.conv2.bias
        {
            map.insert("conv2.bias".to_string(), b.clone());
        }
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        load_conv(&mut self.conv1, sd, "conv1")?;
        load_conv(&mut self.conv2, sd, "conv2")?;
        Ok(())
    }
}

/// Load `{prefix}.weight` / `{prefix}.bias` into a sub-Conv2d, erroring on
/// missing keys or shape mismatches (same convention as `Conv2d`).
fn load_conv(
    conv: &mut Conv2d,
    sd: &HashMap<String, Tensor>,
    prefix: &str,
) -> crate::error::Result<()> {
    let w = sd
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| format!("missing key: {prefix}.weight"))?;
    let kk = conv.kernel * conv.kernel;
    if w.shape() != (conv.out_c, conv.in_c * kk)
    {
        crate::bail!(
            "{prefix}.weight shape mismatch: expected {:?}, got {:?}",
            (conv.out_c, conv.in_c * kk),
            w.shape()
        );
    }
    conv.weight = w.clone();
    if conv.bias.is_some()
    {
        let b = sd
            .get(&format!("{prefix}.bias"))
            .ok_or_else(|| format!("missing key: {prefix}.bias"))?;
        if b.shape() != (1, conv.out_c)
        {
            crate::bail!(
                "{prefix}.bias shape mismatch: expected {:?}, got {:?}",
                (1, conv.out_c),
                b.shape()
            );
        }
        conv.bias = Some(b.clone());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax_row(x: &[f32]) -> Vec<f32> {
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    }

    #[test]
    fn ctc_single_step_value() {
        // T=1, target=[0], V=3 (blank=2): only the path "0" collapses to [0],
        // so loss = -ln softmax(logits)[0].
        let logits = vec![0.5f32, -0.2, 0.1];
        let tape = Tape::new();
        let lg = tape.input(Tensor::from_vec(logits.clone(), 1, 3));
        let tg = tape.input(Tensor::from_vec(vec![0.0], 1, 1));
        let loss = CTCLoss.forward(&tape, lg, tg);
        let expected = -softmax_row(&logits)[0].ln();
        assert!((tape.value(loss.idx()).data[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn ctc_value_matches_bruteforce() {
        // T=3, V=2 (blank=1), target=[0]. Enumerate all 2^3 paths, sum the
        // probability of those collapsing to [0], compare loss = -ln p.
        let logits = vec![0.3f32, -0.1, -0.4, 0.2, 0.6, 0.1];
        let (t, v, blank) = (3usize, 2usize, 1usize);
        let target = [0usize];
        let sm: Vec<Vec<f32>> = (0..t)
            .map(|i| softmax_row(&logits[i * v..(i + 1) * v]))
            .collect();
        let collapse = |path: &[usize]| -> Vec<usize> {
            let mut out = Vec::new();
            let mut prev = usize::MAX;
            for &p in path
            {
                if p != prev && p != blank
                {
                    out.push(p);
                }
                prev = p;
            }
            out
        };
        let mut p = 0.0f32;
        for code in 0..v.pow(t as u32)
        {
            let path: Vec<usize> = (0..t).map(|i| (code / v.pow(i as u32)) % v).collect();
            if collapse(&path) == target
            {
                p += path
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| sm[i][c])
                    .product::<f32>();
            }
        }
        let expected = -p.ln();
        let tape = Tape::new();
        let lg = tape.input(Tensor::from_vec(logits, t, v));
        let tg = tape.input(Tensor::from_vec(vec![0.0], 1, 1));
        let loss = CTCLoss.forward(&tape, lg, tg);
        assert!(
            (tape.value(loss.idx()).data[0] - expected).abs() < 1e-4,
            "loss {} vs brute-force {expected}",
            tape.value(loss.idx()).data[0]
        );
    }

    #[test]
    fn ctc_gradient_matches_finite_differences() {
        let (t, v) = (4usize, 3usize); // blank = 2
        let data: Vec<f32> = vec![
            0.2, -0.5, 0.1, //
            -0.3, 0.4, 0.0, //
            0.1, 0.1, -0.2, //
            0.5, -0.1, 0.3,
        ];
        let target = vec![0.0f32, 1.0];
        let loss_of = |d: &[f32]| -> f32 {
            let tp = Tape::new();
            let lg = tp.input(Tensor::from_vec(d.to_vec(), t, v));
            let tg = tp.input(Tensor::from_vec(target.clone(), 1, target.len()));
            let l = CTCLoss.forward(&tp, lg, tg);
            tp.value(l.idx()).data[0]
        };
        let tape = Tape::new();
        let lg = tape.input(Tensor::from_vec(data.clone(), t, v));
        let tg = tape.input(Tensor::from_vec(target.clone(), 1, target.len()));
        let loss = CTCLoss.forward(&tape, lg, tg);
        loss.backward();
        let ana = tape.grad(lg.idx()).data.clone();
        let h = 1e-3f32;
        for i in 0..data.len()
        {
            let mut pd = data.clone();
            pd[i] += h;
            let mut md = data.clone();
            md[i] -= h;
            let num = (loss_of(&pd) - loss_of(&md)) / (2.0 * h);
            assert!(
                (ana[i] - num).abs() <= 2e-2 + 1e-2 * num.abs(),
                "grad elem {i}: analytic {} vs numerical {num}",
                ana[i]
            );
        }
    }

    #[test]
    fn audio_encoder_state_dict_round_trip() {
        use crate::nn::init::{KaimingNormal, Zeros};

        let mut rng = PcgEngine::new(7);
        let mut enc1 = AudioEncoder::new(1, 2, 3, &KaimingNormal, &Zeros, &mut rng);
        // Make the biases non-trivial so the round trip is meaningful.
        enc1.conv1.bias = Some(Tensor::from_vec(vec![0.5, -0.5], 1, 2));
        enc1.conv2.bias = Some(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));

        let sd = enc1.state_dict();
        assert_eq!(sd.len(), 4);

        let mut rng2 = PcgEngine::new(99);
        let mut enc2 = AudioEncoder::new(1, 2, 3, &Zeros, &Zeros, &mut rng2);
        // Missing keys must be an error, not a silent skip.
        assert!(enc2.load_state_dict(&HashMap::new()).is_err());
        enc2.load_state_dict(&sd).unwrap();

        // Compare every parameter tensor through the tape values.
        let tape1 = Tape::new();
        let tape2 = Tape::new();
        let x = Tensor::from_vec(vec![0.1; 16], 1, 16); // 4x4 mono spectrogram
        let _ = enc1.forward(&tape1, tape1.input(x.clone()));
        let _ = enc2.forward(&tape2, tape2.input(x));
        let idx1 = enc1.parameter_indices();
        let idx2 = enc2.parameter_indices();
        assert_eq!(idx1.len(), 4);
        assert_eq!(idx1.len(), idx2.len());
        for (a, b) in idx1.iter().zip(&idx2)
        {
            assert_eq!(tape1.value(*a).data, tape2.value(*b).data);
        }
    }
}
