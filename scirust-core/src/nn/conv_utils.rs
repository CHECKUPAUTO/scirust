// scirust-core/src/nn/conv_utils.rs
//
// im2col / col2im — fonctions de réorganisation pour les convolutions.
//
// CONVENTIONS DE LAYOUT (toutes en row-major contigu) :
//
//   Input  4D logique : (B, C, H, W)
//   Stockage 2D       : (B, C·H·W)
//
//   Filtre logique    : (out_C, in_C, K, K)
//   Stockage 2D       : (out_C, in_C·K·K)
//
//   im2col output     : (in_C·K·K, B·H_out·W_out)

use crate::autodiff::reverse::Tensor;
use crate::error::Result;

#[derive(Clone, Copy, Debug)]
pub enum Padding {
    Valid,
    Same,
}

#[derive(Clone, Copy, Debug)]
pub struct ConvConfig {
    pub batch: usize,
    pub in_c: usize,
    pub h: usize,
    pub w: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: Padding,
    pub out_c: usize,
}

impl ConvConfig {
    pub fn pad(&self) -> usize {
        match self.padding
        {
            Padding::Valid => 0,
            Padding::Same => (self.kernel - 1) / 2,
        }
    }

    pub fn h_out(&self) -> usize {
        (self.h + 2 * self.pad() - self.kernel) / self.stride + 1
    }

    pub fn w_out(&self) -> usize {
        (self.w + 2 * self.pad() - self.kernel) / self.stride + 1
    }

    pub fn check(&self) -> Result<()> {
        if self.kernel == 0
        {
            crate::bail!("kernel doit être > 0");
        }
        if self.stride == 0
        {
            crate::bail!("stride doit être > 0");
        }
        let pad = self.pad();
        if self.h + 2 * pad < self.kernel
        {
            crate::bail!(
                "input trop petit pour ce kernel : H={}, K={}, pad={}",
                self.h,
                self.kernel,
                pad
            );
        }
        if self.w + 2 * pad < self.kernel
        {
            crate::bail!("input trop petit en largeur");
        }
        Ok(())
    }
}

pub fn im2col(input: &Tensor, cfg: &ConvConfig) -> Tensor {
    cfg.check().expect("ConvConfig invalide");

    let (b, c, h, w, k, s) = (cfg.batch, cfg.in_c, cfg.h, cfg.w, cfg.kernel, cfg.stride);
    let pad = cfg.pad();
    let h_out = cfg.h_out();
    let w_out = cfg.w_out();

    let chw = c * h * w;
    let kk = k * k;
    let ckk = c * kk;
    let n_cols = b * h_out * w_out;

    assert_eq!(
        input.rows, b,
        "im2col: rows attendu = B = {b}, got {}",
        input.rows
    );
    assert_eq!(
        input.cols, chw,
        "im2col: cols attendu = C·H·W = {chw}, got {}",
        input.cols
    );

    let mut out = Tensor::zeros(ckk, n_cols);

    for c_idx in 0..c
    {
        for kh in 0..k
        {
            for kw in 0..k
            {
                let row = (c_idx * k + kh) * k + kw;
                for bi in 0..b
                {
                    for ho in 0..h_out
                    {
                        let h_in = ho * s + kh;
                        for wo in 0..w_out
                        {
                            let w_in = wo * s + kw;
                            let col = bi * h_out * w_out + ho * w_out + wo;
                            let in_h_signed = h_in as isize - pad as isize;
                            let in_w_signed = w_in as isize - pad as isize;
                            if in_h_signed >= 0
                                && in_h_signed < h as isize
                                && in_w_signed >= 0
                                && in_w_signed < w as isize
                            {
                                let in_h = in_h_signed as usize;
                                let in_w = in_w_signed as usize;
                                let src_idx = bi * chw + c_idx * h * w + in_h * w + in_w;
                                out.data[row * n_cols + col] = input.data[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

pub fn col2im(cols: &Tensor, cfg: &ConvConfig) -> Tensor {
    cfg.check().expect("ConvConfig invalide");

    let (b, c, h, w, k, s) = (cfg.batch, cfg.in_c, cfg.h, cfg.w, cfg.kernel, cfg.stride);
    let pad = cfg.pad();
    let h_out = cfg.h_out();
    let w_out = cfg.w_out();

    let chw = c * h * w;
    let kk = k * k;
    let ckk = c * kk;
    let n_cols = b * h_out * w_out;

    assert_eq!(cols.rows, ckk, "col2im: rows attendu = C·K·K = {ckk}");
    assert_eq!(
        cols.cols, n_cols,
        "col2im: cols attendu = B·H_out·W_out = {n_cols}"
    );

    let mut out = Tensor::zeros(b, chw);

    for c_idx in 0..c
    {
        for kh in 0..k
        {
            for kw in 0..k
            {
                let row = (c_idx * k + kh) * k + kw;
                for bi in 0..b
                {
                    for ho in 0..h_out
                    {
                        let h_in = ho * s + kh;
                        for wo in 0..w_out
                        {
                            let w_in = wo * s + kw;
                            let col = bi * h_out * w_out + ho * w_out + wo;
                            let in_h_signed = h_in as isize - pad as isize;
                            let in_w_signed = w_in as isize - pad as isize;
                            if in_h_signed >= 0
                                && in_h_signed < h as isize
                                && in_w_signed >= 0
                                && in_w_signed < w as isize
                            {
                                let in_h = in_h_signed as usize;
                                let in_w = in_w_signed as usize;
                                let dst_idx = bi * chw + c_idx * h * w + in_h * w + in_w;
                                out.data[dst_idx] += cols.data[row * n_cols + col];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// im2col avec pad explicite (usize), pour usage interne par conv2d_forward.
pub fn im2col_raw(
    input: &Tensor,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    pad: usize,
) -> Tensor {
    let h_out = (h + 2 * pad - k) / s + 1;
    let w_out = (w + 2 * pad - k) / s + 1;
    let chw = c * h * w;
    let n_cols = b * h_out * w_out;
    let mut out = Tensor::zeros(c * k * k, n_cols);
    let fill = |row: usize, orow: &mut [f32]| {
        let c_idx = row / (k * k);
        let rem = row % (k * k);
        let kh = rem / k;
        let kw = rem % k;
        for bi in 0..b
        {
            for ho in 0..h_out
            {
                for wo in 0..w_out
                {
                    let col = bi * h_out * w_out + ho * w_out + wo;
                    let ih = (ho * s + kh) as isize - pad as isize;
                    let iw = (wo * s + kw) as isize - pad as isize;
                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize
                    {
                        let src = bi * chw + c_idx * h * w + ih as usize * w + iw as usize;
                        orow[col] = input.data[src];
                    }
                }
            }
        }
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        out.data
            .par_chunks_mut(n_cols)
            .enumerate()
            .for_each(|(row, orow)| fill(row, orow));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for row in 0..(c * k * k)
        {
            let st = row * n_cols;
            fill(row, &mut out.data[st..st + n_cols]);
        }
    }
    out
}

/// col2im avec pad explicite (usize). Accumule les contributions chevauchantes.
pub fn col2im_raw(
    cols: &Tensor,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    pad: usize,
) -> Tensor {
    let h_out = (h + 2 * pad - k) / s + 1;
    let w_out = (w + 2 * pad - k) / s + 1;
    let chw = c * h * w;
    let n_cols = b * h_out * w_out;
    let mut out = Tensor::zeros(b, chw);
    let accum = |bi: usize, oimg: &mut [f32]| {
        for c_idx in 0..c
        {
            for kh in 0..k
            {
                for kw in 0..k
                {
                    let row = (c_idx * k + kh) * k + kw;
                    for ho in 0..h_out
                    {
                        for wo in 0..w_out
                        {
                            let col = bi * h_out * w_out + ho * w_out + wo;
                            let ih = (ho * s + kh) as isize - pad as isize;
                            let iw = (wo * s + kw) as isize - pad as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize
                            {
                                let dst = c_idx * h * w + ih as usize * w + iw as usize;
                                oimg[dst] += cols.data[row * n_cols + col];
                            }
                        }
                    }
                }
            }
        }
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        out.data
            .par_chunks_mut(chw)
            .enumerate()
            .for_each(|(bi, oimg)| accum(bi, oimg));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for bi in 0..b
        {
            let st = bi * chw;
            accum(bi, &mut out.data[st..st + chw]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_simple() -> ConvConfig {
        ConvConfig {
            batch: 1,
            in_c: 1,
            h: 4,
            w: 4,
            kernel: 3,
            stride: 1,
            padding: Padding::Valid,
            out_c: 1,
        }
    }

    #[test]
    fn h_out_w_out_arithmetic() {
        let cfg = config_simple();
        assert_eq!(cfg.h_out(), 2);
        assert_eq!(cfg.w_out(), 2);

        let cfg_same = ConvConfig {
            padding: Padding::Same,
            ..cfg
        };
        assert_eq!(cfg_same.h_out(), 4);
        assert_eq!(cfg_same.w_out(), 4);
    }

    #[test]
    fn im2col_identity_kernel() {
        let cfg = ConvConfig {
            batch: 1,
            in_c: 2,
            h: 2,
            w: 2,
            kernel: 1,
            stride: 1,
            padding: Padding::Valid,
            out_c: 1,
        };
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Tensor::from_vec(data.clone(), 1, 8);
        let cols = im2col(&input, &cfg);
        assert_eq!(cols.shape(), (2, 4));
        assert_eq!(&cols.data[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&cols.data[4..8], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn im2col_3x3_on_4x4() {
        let cfg = config_simple();
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let input = Tensor::from_vec(data.clone(), 1, 16);
        let cols = im2col(&input, &cfg);
        assert_eq!(cols.shape(), (9, 4));
        let col0: Vec<f32> = (0..9).map(|r| cols.data[r * 4]).collect();
        assert_eq!(col0, vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0]);
        let col3: Vec<f32> = (0..9).map(|r| cols.data[r * 4 + 3]).collect();
        assert_eq!(
            col3,
            vec![6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
        );
    }

    #[test]
    fn im2col_same_padding_zeros_at_border() {
        let cfg = ConvConfig {
            batch: 1,
            in_c: 1,
            h: 3,
            w: 3,
            kernel: 3,
            stride: 1,
            padding: Padding::Same,
            out_c: 1,
        };
        let data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let input = Tensor::from_vec(data, 1, 9);
        let cols = im2col(&input, &cfg);
        assert_eq!(cols.shape(), (9, 9));
        let col0: Vec<f32> = (0..9).map(|r| cols.data[r * 9]).collect();
        assert_eq!(col0, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0]);
    }

    #[test]
    fn col2im_identity_kernel_round_trip() {
        let cfg = ConvConfig {
            batch: 2,
            in_c: 3,
            h: 2,
            w: 2,
            kernel: 1,
            stride: 1,
            padding: Padding::Valid,
            out_c: 1,
        };
        let data: Vec<f32> = (1..=24).map(|x| x as f32 * 0.5).collect();
        let input = Tensor::from_vec(data.clone(), 2, 12);
        let cols = im2col(&input, &cfg);
        let restored = col2im(&cols, &cfg);
        assert_eq!(restored.shape(), input.shape());
        assert_eq!(restored.data, input.data);
    }

    #[test]
    fn col2im_accumulates_overlapping_contributions() {
        let cfg = ConvConfig {
            batch: 1,
            in_c: 1,
            h: 3,
            w: 3,
            kernel: 2,
            stride: 1,
            padding: Padding::Valid,
            out_c: 1,
        };
        let cols = Tensor::from_vec(vec![1.0; 16], 4, 4);
        let restored = col2im(&cols, &cfg);
        assert_eq!(restored.shape(), (1, 9));
        assert_eq!(restored.data[0], 1.0);
        assert_eq!(restored.data[4], 4.0);
        assert_eq!(restored.data[1], 2.0);
    }

    #[test]
    fn im2col_batch_dimension_correct() {
        let cfg = ConvConfig {
            batch: 2,
            in_c: 1,
            h: 2,
            w: 2,
            kernel: 2,
            stride: 1,
            padding: Padding::Valid,
            out_c: 1,
        };
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], 2, 4);
        let cols = im2col(&input, &cfg);
        assert_eq!(cols.shape(), (4, 2));
        let c0: Vec<f32> = (0..4).map(|r| cols.data[r * 2]).collect();
        assert_eq!(c0, vec![1.0, 2.0, 3.0, 4.0]);
        let c1: Vec<f32> = (0..4).map(|r| cols.data[r * 2 + 1]).collect();
        assert_eq!(c1, vec![10.0, 20.0, 30.0, 40.0]);
    }
}
