// scirust-core/src/nn/conv_utils.rs
//
// im2col / col2im — convert image patches to column matrix and back.
//
// These utilities enable Conv2d to be implemented as a single matrix
// multiply (GEMM), which is highly efficient on modern hardware.

use ndarray::{s, Array2, Array3};

/// Helper: pad an input array and return the padded array plus output dimensions.
#[inline]
fn pad_input(
    input: &Array3<f64>,
    pad: usize,
) -> (Array3<f64>, usize, usize) {
    let (c, h, w) = input.dim();
    let h_padded = h + 2 * pad;
    let w_padded = w + 2 * pad;
    let mut padded = Array3::<f64>::zeros((c, h_padded, w_padded));
    {
        let mut slice = padded.slice_mut(s![.., pad..pad + h, pad..pad + w]);
        slice.assign(input);
    }
    (padded, h_padded, w_padded)
}

/// Compute the output spatial dimensions and number of patches.
#[inline]
fn output_dims(
    h_padded: usize,
    w_padded: usize,
    kernel_size: usize,
    stride: usize,
) -> (usize, usize, usize) {
    let out_h = (h_padded - kernel_size) / stride + 1;
    let out_w = (w_padded - kernel_size) / stride + 1;
    let num_patches = out_h * out_w;
    (out_h, out_w, num_patches)
}

/// Core im2col loop body shared by `im2col` and `im2col_with_buffer`.
///
/// Writes patch data from `padded` into `col` at the given column index.
/// Uses slice operations for the inner `kw` loop to enable auto-vectorization.
#[inline]
unsafe fn im2col_write_patch(
    padded: &Array3<f64>,
    col: &mut Array2<f64>,
    c: usize,
    kernel_size: usize,
    oh: usize,
    ow: usize,
    out_w: usize,
    h_start: usize,
    w_start: usize,
) {
    let col_idx = oh * out_w + ow;
    for kc in 0..c {
        for kh in 0..kernel_size {
            let row_start = (kc * kernel_size + kh) * kernel_size;
            let patch_slice = padded.slice(s![kc, h_start + kh, w_start..w_start + kernel_size]);
            // SAFETY: caller guarantees col is sized (kernel_size*kernel_size*c, num_patches)
            // and the slice ranges are within bounds.
            let mut dest = col.slice_mut(s![row_start..row_start + kernel_size, col_idx]);
            dest.assign(&patch_slice);
        }
    }
}

/// Core col2im loop body shared by `col2im` and `col2im_with_buffer`.
///
/// Accumulates column data back into the gradient image `grad`.
/// Uses slice operations for the inner `kw` loop to enable auto-vectorization.
#[inline]
unsafe fn col2im_add_patch(
    col: &Array2<f64>,
    grad: &mut Array3<f64>,
    c: usize,
    kernel_size: usize,
    oh: usize,
    ow: usize,
    out_w: usize,
    h_start: usize,
    w_start: usize,
) {
    let col_idx = oh * out_w + ow;
    for kc in 0..c {
        for kh in 0..kernel_size {
            let row_start = (kc * kernel_size + kh) * kernel_size;
            let patch_col = col.slice(s![row_start..row_start + kernel_size, col_idx]);
            // SAFETY: caller guarantees grad is sized (c, h_padded, w_padded)
            // and the slice ranges are within bounds.
            let mut dest = grad.slice_mut(s![kc, h_start + kh, w_start..w_start + kernel_size]);
            dest += &patch_col;
        }
    }
}

/// Convert image patches to column matrix for efficient GEMM-based convolution.
///
/// # Shapes
/// - `input`  : `(C, H, W)` — channels, height, width
/// - `output` : `(K*K*C, O*O)` where `O = (H + 2*pad - kernel_size) / stride + 1`
///   is the output spatial size.
///
/// Each column of the output corresponds to one image patch flattened
/// in row-major (channel, kernel_row, kernel_col) order.
#[inline]
pub fn im2col(
    input: &Array3<f64>,
    kernel_size: usize,
    stride: usize,
    pad: usize,
) -> Array2<f64> {
    let (c, _h, _w) = input.dim();
    let (padded, h_padded, w_padded) = pad_input(input, pad);
    let (out_h, out_w, num_patches) = output_dims(h_padded, w_padded, kernel_size, stride);
    let patch_len = kernel_size * kernel_size * c;

    let mut col = Array2::<f64>::zeros((patch_len, num_patches));

    for oh in 0..out_h {
        for ow in 0..out_w {
            let h_start = oh * stride;
            let w_start = ow * stride;
            // SAFETY: col was just allocated with correct dimensions above.
            unsafe {
                im2col_write_patch(&padded, &mut col, c, kernel_size, oh, ow, out_w, h_start, w_start);
            }
        }
    }

    col
}

/// Like `im2col` but reuses a pre-allocated buffer instead of allocating.
///
/// # Panics
/// Panics if `buffer` dimensions don't match the required output shape
/// `(K*K*C, O*O)` where `O = (H + 2*pad - kernel_size) / stride + 1`.
///
/// Returns the buffer (zero-copy) filled with the im2col result.
#[inline]
pub fn im2col_with_buffer<'a>(
    input: &Array3<f64>,
    kernel_size: usize,
    stride: usize,
    pad: usize,
    buffer: &'a mut Array2<f64>,
) -> &'a mut Array2<f64> {
    let (c, _h, _w) = input.dim();
    let (padded, h_padded, w_padded) = pad_input(input, pad);
    let (out_h, out_w, num_patches) = output_dims(h_padded, w_padded, kernel_size, stride);
    let patch_len = kernel_size * kernel_size * c;

    assert_eq!(
        buffer.dim(),
        (patch_len, num_patches),
        "im2col_with_buffer: buffer dimensions ({:?}) do not match required ({}, {})",
        buffer.dim(),
        patch_len,
        num_patches,
    );

    for oh in 0..out_h {
        for ow in 0..out_w {
            let h_start = oh * stride;
            let w_start = ow * stride;
            // SAFETY: buffer dimensions verified above.
            unsafe {
                im2col_write_patch(&padded, buffer, c, kernel_size, oh, ow, out_w, h_start, w_start);
            }
        }
    }

    buffer
}

/// Inverse of `im2col` — scatter gradient columns back to image positions.
///
/// # Shapes
/// - `col`          : `(K*K*C, O*O)` — gradient with respect to the column matrix
/// - `input_shape`  : `(C, H, W)` — original image shape (before padding)
/// - `output`       : `(C, H, W)` — gradient with respect to the input image
///
/// Each column is scattered (summed) into the corresponding patch location
/// in the output gradient. Overlapping patches accumulate naturally.
#[inline]
pub fn col2im(
    col: &Array2<f64>,
    input_shape: (usize, usize, usize),
    kernel_size: usize,
    stride: usize,
    pad: usize,
) -> Array3<f64> {
    let (c, h, w) = input_shape;
    let h_padded = h + 2 * pad;
    let w_padded = w + 2 * pad;
    let (out_h, out_w, _num_patches) = output_dims(h_padded, w_padded, kernel_size, stride);

    // Accumulate into padded gradient, then crop.
    let mut grad = Array3::<f64>::zeros((c, h_padded, w_padded));

    for oh in 0..out_h {
        for ow in 0..out_w {
            let h_start = oh * stride;
            let w_start = ow * stride;
            // SAFETY: grad was just allocated with correct dimensions above.
            unsafe {
                col2im_add_patch(col, &mut grad, c, kernel_size, oh, ow, out_w, h_start, w_start);
            }
        }
    }

    // Crop padding from the result.
    grad.slice(s![.., pad..pad + h, pad..pad + w]).to_owned()
}

/// Like `col2im` but reuses a pre-allocated buffer instead of allocating.
///
/// # Panics
/// Panics if `buffer` dimensions don't match `input_shape` (before padding).
///
/// Returns the buffer (zero-copy) filled with the col2im result.
/// The buffer is zeroed before accumulation so prior contents do not leak.
#[inline]
pub fn col2im_with_buffer<'a>(
    col: &Array2<f64>,
    input_shape: (usize, usize, usize),
    kernel_size: usize,
    stride: usize,
    pad: usize,
    buffer: &'a mut Array3<f64>,
) -> &'a mut Array3<f64> {
    let (c, h, w) = input_shape;
    let h_padded = h + 2 * pad;
    let w_padded = w + 2 * pad;
    let (out_h, out_w, _num_patches) = output_dims(h_padded, w_padded, kernel_size, stride);

    assert_eq!(
        buffer.dim(),
        (c, h, w),
        "col2im_with_buffer: buffer dimensions ({:?}) do not match input shape ({}, {}, {})",
        buffer.dim(),
        c,
        h,
        w,
    );

    // Zero the buffer before accumulating.
    buffer.fill(0.0);

    // Accumulate into a padded temporary, then crop into buffer.
    let mut grad_padded = Array3::<f64>::zeros((c, h_padded, w_padded));

    for oh in 0..out_h {
        for ow in 0..out_w {
            let h_start = oh * stride;
            let w_start = ow * stride;
            // SAFETY: grad_padded was just allocated with correct dimensions above.
            unsafe {
                col2im_add_patch(col, &mut grad_padded, c, kernel_size, oh, ow, out_w, h_start, w_start);
            }
        }
    }

    // Crop into the user-provided buffer.
    let cropped = grad_padded.slice(s![.., pad..pad + h, pad..pad + w]);
    buffer.assign(&cropped);

    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed: 1 channel, 2x2 image, 2x2 kernel, stride=1, pad=0.
    ///
    /// input:
    ///   [[1, 2],
    ///    [3, 4]]
    ///
    /// With K=2 and no padding, there is exactly 1 patch (the whole image).
    /// The flattened patch is [1, 2, 3, 4] (C=1, K=2 → 1*2*2 = 4 elements).
    #[test]
    fn im2col_single_patch() {
        let input = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let col = im2col(&input, 2, 1, 0);
        assert_eq!(col.shape(), &[4, 1]);
        assert!((col[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((col[[1, 0]] - 2.0).abs() < 1e-12);
        assert!((col[[2, 0]] - 3.0).abs() < 1e-12);
        assert!((col[[3, 0]] - 4.0).abs() < 1e-12);
    }

    /// 2 channels, 3x3 image, 2x2 kernel, stride=1, pad=0.
    /// Output shape: (2*2*2, 2*2) = (8, 4)
    #[test]
    fn im2col_multi_channel_shape() {
        let input = Array3::<f64>::ones((2, 3, 3));
        let col = im2col(&input, 2, 1, 0);
        assert_eq!(col.shape(), &[8, 4]);
        // All entries are 1 because the input is all ones.
        for val in col.iter() {
            assert!((*val - 1.0).abs() < 1e-12);
        }
    }

    /// Verify that col2im recovers the original image after im2col
    /// when there is no overlap (stride == kernel_size, no padding).
    #[test]
    fn col2im_no_overlap_roundtrip() {
        let input = Array3::from_shape_vec(
            (1, 4, 4),
            (0..16).map(|x| x as f64).collect(),
        )
        .unwrap();
        let col = im2col(&input, 2, 2, 0); // stride == kernel_size
        let recovered = col2im(&col, (1, 4, 4), 2, 2, 0);
        assert_eq!(recovered.shape(), input.shape());
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// Verify padding: 1x1 image with pad=1 and K=3 produces one patch.
    #[test]
    fn im2col_with_padding() {
        let input = Array3::from_shape_vec((1, 1, 1), vec![5.0]).unwrap();
        // With pad=1 and K=3, the padded image is 3x3, so O = (3-3)/1+1 = 1
        let col = im2col(&input, 3, 1, 1);
        assert_eq!(col.shape(), &[9, 1]);
        // Center element of the patch should be 5.0
        // In flattened (channel=0, k=3): row index = 0*9 + 1*3 + 1 = 4
        assert!((col[[4, 0]] - 5.0).abs() < 1e-12);
        // Corners are zero-padded
        assert!((col[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((col[[2, 0]] - 0.0).abs() < 1e-12);
        assert!((col[[6, 0]] - 0.0).abs() < 1e-12);
        assert!((col[[8, 0]] - 0.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // Tests for im2col_with_buffer / col2im_with_buffer
    // ------------------------------------------------------------------

    /// im2col_with_buffer produces the same result as im2col.
    #[test]
    fn im2col_with_buffer_matches_im2col() {
        let input = Array3::from_shape_vec((2, 3, 3), (0..18).map(|x| x as f64).collect()).unwrap();
        let col_ref = im2col(&input, 2, 1, 0);
        let (_, h, w) = input.dim();
        let h_padded = h + 0;
        let out_h = (h_padded - 2) / 1 + 1;
        let out_w = (w - 2) / 1 + 1;
        let num_patches = out_h * out_w;
        let patch_len = 2 * 2 * 2;
        let mut buffer = Array2::<f64>::zeros((patch_len, num_patches));
        let result = im2col_with_buffer(&input, 2, 1, 0, &mut buffer);
        assert_eq!(result.shape(), col_ref.shape());
        for (a, b) in col_ref.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// im2col_with_buffer panics on wrong buffer dimensions.
    #[test]
    #[should_panic(expected = "buffer dimensions")]
    fn im2col_with_buffer_wrong_dims() {
        let input = Array3::<f64>::ones((1, 4, 4));
        let mut bad_buffer = Array2::<f64>::zeros((1, 1)); // wrong size
        im2col_with_buffer(&input, 2, 2, 0, &mut bad_buffer);
    }

    /// col2im_with_buffer produces the same result as col2im.
    #[test]
    fn col2im_with_buffer_matches_col2im() {
        let input = Array3::from_shape_vec((1, 4, 4), (0..16).map(|x| x as f64).collect()).unwrap();
        let col = im2col(&input, 2, 2, 0);
        let ref_out = col2im(&col, (1, 4, 4), 2, 2, 0);
        let mut buffer = Array3::<f64>::zeros((1, 4, 4));
        let result = col2im_with_buffer(&col, (1, 4, 4), 2, 2, 0, &mut buffer);
        assert_eq!(result.shape(), ref_out.shape());
        for (a, b) in ref_out.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// col2im_with_buffer panics on wrong buffer dimensions.
    #[test]
    #[should_panic(expected = "buffer dimensions")]
    fn col2im_with_buffer_wrong_dims() {
        let col = Array2::<f64>::zeros((4, 4));
        let mut bad_buffer = Array3::<f64>::zeros((1, 1, 1)); // wrong size
        col2im_with_buffer(&col, (1, 4, 4), 2, 2, 0, &mut bad_buffer);
    }

    /// col2im_with_buffer zeroes the buffer before accumulating.
    #[test]
    fn col2im_with_buffer_zeroes_prior_contents() {
        let col = Array2::<f64>::zeros((4, 1));
        let mut buffer = Array3::<f64>::ones((1, 2, 2)); // pre-filled with 1.0
        let result = col2im_with_buffer(&col, (1, 2, 2), 2, 1, 0, &mut buffer);
        // Since col is all zeros, result should be all zeros, not ones.
        for val in result.iter() {
            assert!((*val - 0.0).abs() < 1e-12);
        }
    }

    /// Roundtrip via with_buffer variants matches original (non-overlapping).
    #[test]
    fn with_buffer_roundtrip() {
        let input = Array3::from_shape_vec(
            (2, 6, 6),
            (0..72).map(|x| x as f64).collect(),
        )
        .unwrap();
        let (c, h, w) = input.dim();

        // Non-overlapping: stride == kernel_size, no padding for exact roundtrip.
        let ks = 2;
        let st = 2;
        let pd = 0;
        let out_h = (h - ks) / st + 1;
        let out_w = (w - ks) / st + 1;
        let patch_len = ks * ks * c;
        let mut col_buf = Array2::<f64>::zeros((patch_len, out_h * out_w));
        let mut img_buf = Array3::<f64>::zeros((c, h, w));

        let col = im2col_with_buffer(&input, ks, st, pd, &mut col_buf);
        let recovered = col2im_with_buffer(col, (c, h, w), ks, st, pd, &mut img_buf);

        assert_eq!(recovered.shape(), input.shape());
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
