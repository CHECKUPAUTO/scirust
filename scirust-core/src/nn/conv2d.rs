// scirust-core/src/nn/conv2d.rs
//
// Conv2d layer implemented via im2col + GEMM.
//
// Shapes:
//   - input   : (in_channels, H, W)
//   - weight  : (out_channels, in_channels, K, K)
//   - bias    : (out_channels,)  — optional
//   - output  : (out_channels, O, O)
//       where O = (H + 2*pad - kernel_size) / stride + 1

use ndarray::{s, Array1, Array3, Array4};

use super::conv_utils::{col2im, im2col};

/// 2D convolution layer.
pub struct Conv2d {
    pub weight: Array4<f64>,               // (out_channels, in_channels, k, k)
    pub bias:   Option<Array1<f64>>,       // (out_channels,) or None
    pub kernel_size: usize,
    pub stride:  usize,
    pub pad:     usize,
}

impl Conv2d {
    /// Create a new `Conv2d` layer with no bias by default.
    ///
    /// Weight is initialized with Kaiming-normal-like scaling:
    /// `N(0, sqrt(2.0 / (in_channels * K * K)))`.
    pub fn new(
        in_channels:  usize,
        out_channels: usize,
        kernel_size:  usize,
        stride:       usize,
        pad:          usize,
    ) -> Self {
        // Kaiming-like scale for ReLU activations
        let scale = (2.0 / (in_channels * kernel_size * kernel_size) as f64).sqrt();
        let weight = Array4::from_shape_simple_fn(
            (out_channels, in_channels, kernel_size, kernel_size),
            || {
                // Box-Muller approximation
                let u1: f64 = fast_rand();
                let u2: f64 = fast_rand();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z * scale
            },
        );

        Conv2d {
            weight,
            bias: None,
            kernel_size,
            stride,
            pad,
        }
    }

    /// Create a new `Conv2d` with a bias term.
    pub fn new_with_bias(
        in_channels:  usize,
        out_channels: usize,
        kernel_size:  usize,
        stride:       usize,
        pad:          usize,
    ) -> Self {
        let mut conv = Self::new(in_channels, out_channels, kernel_size, stride, pad);
        conv.bias = Some(Array1::zeros(out_channels));
        conv
    }

    /// Return the output spatial size `O` for a given input spatial size.
    pub fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        let h_pad = h + 2 * self.pad;
        let w_pad = w + 2 * self.pad;
        let out_h = (h_pad - self.kernel_size) / self.stride + 1;
        let out_w = (w_pad - self.kernel_size) / self.stride + 1;
        (out_h, out_w)
    }

    /// Forward pass: im2col -> matmul -> reshape.
    ///
    /// # Shapes
    /// - `input` : `(in_channels, H, W)`
    /// - Returns : `(out_channels, O, O)`
    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let (in_c, h, w) = input.dim();
        assert_eq!(
            in_c,
            self.weight.shape()[1],
            "Conv2d: input channels mismatch"
        );

        let (out_c, _, k, _) = self.weight.dim();
        let (out_h, out_w) = self.output_size(h, w);

        // im2col: (K*K*C, O*O)
        let x_col = im2col(input, self.kernel_size, self.stride, self.pad);

        // Flatten weight: (out_c, K*K*C)
        let w_flat = self
            .weight
            .clone()
            .into_shape_with_order((out_c, k * k * in_c))
            .unwrap();

        // y = w_flat @ x_col  => (out_c, O*O)
        let y = w_flat.dot(&x_col);

        // Reshape to (out_c, O, O)
        let mut output = y.into_shape_with_order((out_c, out_h, out_w)).unwrap();

        // Add bias (broadcast across spatial dimensions)
        if let Some(ref bias) = self.bias {
            for oc in 0..out_c {
                let mut slice = output.slice_mut(s![oc, .., ..]);
                slice.mapv_inplace(|v| v + bias[oc]);
            }
        }

        output
    }

    /// Backward pass: compute gradients w.r.t. input, weight, and bias.
    ///
    /// # Arguments
    /// - `input`       : the input to the forward pass `(in_c, H, W)`
    /// - `grad_output` : upstream gradient `(out_c, O, O)`
    ///
    /// # Returns
    /// `(grad_input, grad_weight, grad_bias)`
    /// - `grad_input`  : `(in_c, H, W)`
    /// - `grad_weight` : `(out_c, in_c, K, K)`
    /// - `grad_bias`   : `Some(out_c,)` if bias exists, else `None`
    pub fn backward(
        &self,
        input: &Array3<f64>,
        grad_output: &Array3<f64>,
    ) -> (Array3<f64>, Array4<f64>, Option<Array1<f64>>) {
        let (in_c, h, w) = input.dim();
        let (out_c, _, k, _) = self.weight.dim();
        let (out_h, out_w) = self.output_size(h, w);

        assert_eq!(grad_output.shape(), &[out_c, out_h, out_w]);

        // --- Flatten tensors ---
        // x_col: (K*K*C, O*O)
        let x_col = im2col(input, self.kernel_size, self.stride, self.pad);
        // w_flat: (out_c, K*K*C)
        let w_flat = self
            .weight
            .clone()
            .into_shape_with_order((out_c, k * k * in_c))
            .unwrap();
        // dy: (out_c, O*O)
        let dy = grad_output
            .clone()
            .into_shape_with_order((out_c, out_h * out_w))
            .unwrap();

        // --- Gradient w.r.t. weight: dW = dy @ X_col^T ---
        // w_flat shape: (out_c, K*K*C)
        // dy   shape:   (out_c, O*O)
        // x_col shape:  (K*K*C, O*O)
        // dW_flat = dy @ x_col^T  => (out_c, K*K*C) @ (O*O, K*K*C)^T ... wait
        // dW_flat = dy @ x_col.T  => (out_c, O*O) @ (O*O, K*K*C) = (out_c, K*K*C)
        let x_col_t = x_col.t().to_owned();
        let dw_flat = dy.dot(&x_col_t);
        let grad_weight = dw_flat
            .into_shape_with_order((out_c, in_c, k, k))
            .unwrap();

        // --- Gradient w.r.t. input: dX_col = W^T @ dy  then col2im ---
        // w_flat.T: (K*K*C, out_c)
        // dy:       (out_c, O*O)
        // dx_col:   (K*K*C, O*O)
        let w_flat_t = w_flat.t().to_owned();
        let dx_col = w_flat_t.dot(&dy);

        let grad_input = col2im(
            &dx_col,
            (in_c, h, w),
            self.kernel_size,
            self.stride,
            self.pad,
        );

        // --- Gradient w.r.t. bias: sum over spatial dimensions ---
        let grad_bias = self.bias.as_ref().map(|_| {
            let mut db = Array1::zeros(out_c);
            for oc in 0..out_c {
                let mut sum = 0.0;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        sum += grad_output[[oc, oh, ow]];
                    }
                }
                db[oc] = sum;
            }
            db
        });

        (grad_input, grad_weight, grad_bias)
    }
}

/// Quick-and-dirty pseudo-random f64 in (0, 1) for weight initialization.
fn fast_rand() -> f64 {
    // Simple LCG — used only for initialization, not cryptographic.
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(42) };
    }
    SEED.with(|seed| {
        let s = seed.get();
        let s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed.set(s);
        // Upper 53 bits of the 64-bit state → f64 in (0, 1)
        ((s >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed convolution of a 2x2 image with a 2x2 kernel:
    ///
    /// input (1 channel):
    ///   [[1, 2],
    ///    [3, 4]]
    ///
    /// weight (1 out_ch, 1 in_ch, 2x2):
    ///   [[[1, 0],
    ///     [0, 1]]]
    ///
    /// stride=1, pad=0 → O = (2-2)/1+1 = 1
    /// output: 1*1 + 2*0 + 3*0 + 4*1 = 5  →  [[[5.0]]]
    #[test]
    fn conv2d_forward_hand_computed() {
        let mut conv = Conv2d::new(1, 1, 2, 1, 0);
        // Set weight manually: [[[1, 0], [0, 1]]]
        conv.weight = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        let input = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = conv.forward(&input);

        assert_eq!(output.shape(), &[1, 1, 1]);
        assert!((output[[0, 0, 0]] - 5.0).abs() < 1e-12);
    }

    /// Verify forward shape for a larger configuration.
    #[test]
    fn conv2d_forward_shape() {
        let conv = Conv2d::new(3, 6, 3, 1, 1);
        let input = Array3::<f64>::ones((3, 7, 7));
        let output = conv.forward(&input);
        // O = (7 + 2*1 - 3) / 1 + 1 = 7
        assert_eq!(output.shape(), &[6, 7, 7]);
    }

    /// Verify that stride=2 halves the spatial size.
    #[test]
    fn conv2d_stride_two() {
        let conv = Conv2d::new(1, 2, 3, 2, 0);
        let input = Array3::<f64>::ones((1, 6, 6));
        let output = conv.forward(&input);
        // O = (6 - 3) / 2 + 1 = 2 (integer division)
        assert_eq!(output.shape(), &[2, 2, 2]);
    }

    /// Verify bias addition.
    #[test]
    fn conv2d_with_bias() {
        let mut conv = Conv2d::new_with_bias(1, 2, 2, 1, 0);
        // Zero out weights so only bias matters.
        conv.weight = Array4::zeros((2, 1, 2, 2));
        conv.bias = Some(Array1::from_vec(vec![10.0, 20.0]));

        let input = Array3::from_shape_vec((1, 3, 3), (0..9).map(|x| x as f64).collect()).unwrap();
        let output = conv.forward(&input);
        // O = (3 - 2) / 1 + 1 = 2 → (2, 2, 2)
        assert_eq!(output.shape(), &[2, 2, 2]);
        // Channel 0 should be all 10.0
        for oh in 0..2 {
            for ow in 0..2 {
                assert!((output[[0, oh, ow]] - 10.0).abs() < 1e-12);
            }
        }
        // Channel 1 should be all 20.0
        for oh in 0..2 {
            for ow in 0..2 {
                assert!((output[[1, oh, ow]] - 20.0).abs() < 1e-12);
            }
        }
    }

    /// Verify backward pass: check shapes and that weight gradient
    /// is non-zero when weight is non-zero.
    #[test]
    fn conv2d_backward_shape_and_nonzero() {
        let mut conv = Conv2d::new(1, 1, 2, 1, 0);
        conv.weight = Array4::from_shape_vec((1, 1, 2, 2), vec![0.5, 0.3, -0.2, 0.1]).unwrap();

        let input = Array3::from_shape_vec((1, 3, 3), (0..9).map(|x| x as f64).collect()).unwrap();
        let _output = conv.forward(&input); // (1, 2, 2)

        // Dummy gradient: all ones
        let grad_output = Array3::<f64>::ones((1, 2, 2));
        let (grad_input, grad_weight, grad_bias) = conv.backward(&input, &grad_output);

        assert_eq!(grad_input.shape(), &[1, 3, 3]);
        assert_eq!(grad_weight.shape(), &[1, 1, 2, 2]);
        assert!(grad_bias.is_none());

        // Weight gradient should be non-zero
        let max_gw = grad_weight.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_gw > 1e-12, "weight gradient is zero");

        // Input gradient should be non-zero
        let max_gi = grad_input.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_gi > 1e-12, "input gradient is zero");
    }

    /// Check the weight gradient against a manual two-patch computation.
    /// input: (1, 3, 3) with values 0..9
    /// weight: (1, 1, 2, 2) = [[[1, 0], [0, 0]]]
    /// stride=1, pad=0 → O = (3-2)+1 = 2 → 4 patches
    ///
    /// im2col yields 4 columns of length 4:
    ///   col[0] = [0, 1, 3, 4]   → y = 1*0 + 0*1 + 0*3 + 0*4 = 0
    ///   col[1] = [1, 2, 4, 5]   → y = 1*1 + 0*2 + 0*4 + 0*5 = 1
    ///   col[2] = [3, 4, 6, 7]   → y = 1*3 + 0*4 + 0*6 + 0*7 = 3
    ///   col[3] = [4, 5, 7, 8]   → y = 1*4 + 0*5 + 0*7 + 0*8 = 4
    ///
    /// With unit gradient, dW = dy @ x_col.T = [1,1,1,1] @ 4x4 = sum each row
    /// dW[0,0,0,0] = 0+1+3+4 = 8
    /// dW[0,0,0,1] = 1+2+4+5 = 12
    /// dW[0,0,1,0] = 3+4+6+7 = 20
    /// dW[0,0,1,1] = 4+5+7+8 = 24
    #[test]
    fn conv2d_backward_weight_manual() {
        let mut conv = Conv2d::new(1, 1, 2, 1, 0);
        conv.weight = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let input = Array3::from_shape_vec((1, 3, 3), (0..9).map(|x| x as f64).collect()).unwrap();
        let grad_output = Array3::<f64>::ones((1, 2, 2));

        let (_, grad_weight, _) = conv.backward(&input, &grad_output);

        assert_eq!(grad_weight.shape(), &[1, 1, 2, 2]);
        assert!((grad_weight[[0, 0, 0, 0]] - 8.0).abs() < 1e-12);
        assert!((grad_weight[[0, 0, 0, 1]] - 12.0).abs() < 1e-12);
        assert!((grad_weight[[0, 0, 1, 0]] - 20.0).abs() < 1e-12);
        assert!((grad_weight[[0, 0, 1, 1]] - 24.0).abs() < 1e-12);
    }
}
