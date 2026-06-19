//! GPU-resident im2col / col2im for Conv2d on the autograd tape.

/// WGSL compute shader for im2col: unfold sliding windows into columns.
///
/// Input: (batch, channels, height, width) stored as (batch * C * H * W).
/// Output: (batch, C * K * K, out_h * out_w) columns for GEMM with weight.
pub const IM2COL_WGSL: &str = r#"
struct P {
    batch: u32, channels: u32, h: u32, w: u32,
    k: u32, stride: u32, pad: u32,
    out_h: u32, out_w: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> col: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col_idx = gid.x; // index into output column
    if (col_idx >= p.batch * p.channels * p.k * p.k * p.out_h * p.out_w) { return; }

    let total_spatial = p.out_h * p.out_w;
    let spatial_block = p.k * p.k;
    let chan_block = p.channels * spatial_block;
    
    let b = col_idx / (chan_block * total_spatial);
    let rem_b = col_idx % (chan_block * total_spatial);
    let c_inner = rem_b / (spatial_block * total_spatial);
    let rem_c = rem_b % (spatial_block * total_spatial);
    let k_inner = rem_c / total_spatial;
    let sp = rem_c % total_spatial;

    let out_y = sp / p.out_w;
    let out_x = sp % p.out_w;
    let ky = k_inner / p.k;
    let kx = k_inner % p.k;

    let in_y = out_y * p.stride + ky;
    let in_x = out_x * p.stride + kx;

    var val: f32 = 0.0;
    if (in_y >= p.pad && in_y < p.h + p.pad && in_x >= p.pad && in_x < p.w + p.pad) {
        let in_y_eff = in_y - p.pad;
        let in_x_eff = in_x - p.pad;
        let in_idx = ((b * p.channels + c_inner) * p.h + in_y_eff) * p.w + in_x_eff;
        val = input[in_idx];
    }

    col[col_idx] = val;
}
"#;

/// WGSL compute shader for col2im: accumulate columns back into the image.
///
/// Input: (batch * C * K * K, out_h * out_w) columns (gradient from GEMM).
/// Output: (batch, channels, height, width) accumulated image gradient.
pub const COL2IM_WGSL: &str = r#"
struct P {
    batch: u32, channels: u32, h: u32, w: u32,
    k: u32, stride: u32, pad: u32,
    out_h: u32, out_w: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       col: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col_idx = gid.x;
    if (col_idx >= p.batch * p.channels * p.k * p.k * p.out_h * p.out_w) { return; }
    let val = col[col_idx];

    let total_spatial = p.out_h * p.out_w;
    let spatial_block = p.k * p.k;
    let chan_block = p.channels * spatial_block;

    let b = col_idx / (chan_block * total_spatial);
    let rem_b = col_idx % (chan_block * total_spatial);
    let c_inner = rem_b / (spatial_block * total_spatial);
    let rem_c = rem_b % (spatial_block * total_spatial);
    let k_inner = rem_c / total_spatial;
    let sp = rem_c % total_spatial;

    let out_y = sp / p.out_w;
    let out_x = sp % p.out_w;
    let ky = k_inner / p.k;
    let kx = k_inner % p.k;

    let in_y = out_y * p.stride + ky;
    let in_x = out_x * p.stride + kx;

    if (in_y >= p.pad && in_y < p.h + p.pad && in_x >= p.pad && in_x < p.w + p.pad) {
        let in_y_eff = in_y - p.pad;
        let in_x_eff = in_x - p.pad;
        let in_idx = ((b * p.channels + c_inner) * p.h + in_y_eff) * p.w + in_x_eff;
        // Use atomicAdd for accumulation (multiple columns may write to same pixel)
        atomicAdd(&output[in_idx], val);
    }
}
"#;

/// CPU reference im2col (deterministic, for validation).
#[allow(clippy::too_many_arguments)]
pub fn cpu_im2col(
    input: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let out_h = (h + 2 * pad - k) / stride + 1;
    let out_w = (w + 2 * pad - k) / stride + 1;
    let col_rows = channels * k * k;
    let col_cols = out_h * out_w;
    let mut col = vec![0.0f32; batch * col_rows * col_cols];

    for b in 0..batch
    {
        for c in 0..channels
        {
            for ky in 0..k
            {
                for kx in 0..k
                {
                    for oy in 0..out_h
                    {
                        for ox in 0..out_w
                        {
                            let in_y = oy * stride + ky;
                            let in_x = ox * stride + kx;
                            let col_row = c * k * k + ky * k + kx;
                            let col_idx = ((b * col_rows + col_row) * out_h + oy) * out_w + ox;
                            if in_y >= pad && in_y < h + pad && in_x >= pad && in_x < w + pad
                            {
                                let in_y = in_y - pad;
                                let in_x = in_x - pad;
                                let in_idx = ((b * channels + c) * h + in_y) * w + in_x;
                                col[col_idx] = input[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    col
}

/// CPU reference col2im (deterministic, for validation).
#[allow(clippy::too_many_arguments)]
pub fn cpu_col2im(
    col: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let out_h = (h + 2 * pad - k) / stride + 1;
    let out_w = (w + 2 * pad - k) / stride + 1;
    let im_size = batch * channels * h * w;
    let mut im = vec![0.0f32; im_size];

    for b in 0..batch
    {
        for c in 0..channels
        {
            for ky in 0..k
            {
                for kx in 0..k
                {
                    for oy in 0..out_h
                    {
                        for ox in 0..out_w
                        {
                            let in_y = oy * stride + ky;
                            let in_x = ox * stride + kx;
                            if in_y >= pad && in_y < h + pad && in_x >= pad && in_x < w + pad
                            {
                                let in_y = in_y - pad;
                                let in_x = in_x - pad;
                                let in_idx = ((b * channels + c) * h + in_y) * w + in_x;
                                let col_row = c * k * k + ky * k + kx;
                                let col_idx =
                                    ((b * channels * k * k + col_row) * out_h + oy) * out_w + ox;
                                im[in_idx] += col[col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    im
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_im2col_values() {
        let (batch, ch, h, w, k, s, p) = (1, 1, 3, 3, 2, 1, 0);
        let input: Vec<f32> = (0..(batch * ch * h * w)).map(|i| i as f32).collect();
        let out_h = (h + 2 * p - k) / s + 1;
        let out_w = (w + 2 * p - k) / s + 1;
        let col = cpu_im2col(&input, batch, ch, h, w, k, s, p);
        // Column shape: (batch * ch * k * k, out_h * out_w)
        assert_eq!(col.len(), batch * ch * k * k * out_h * out_w);
        // First column element should be 0 (top-left pixel)
        assert!((col[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_im2col_shape() {
        let (batch, ch, h, w, k, s, p) = (2, 3, 5, 5, 3, 2, 1);
        let input = vec![1.0f32; batch * ch * h * w];
        let out_h = (h + 2 * p - k) / s + 1;
        let out_w = (w + 2 * p - k) / s + 1;
        let col = cpu_im2col(&input, batch, ch, h, w, k, s, p);
        let expected_rows = ch * k * k;
        let expected_cols = out_h * out_w;
        assert_eq!(col.len(), batch * expected_rows * expected_cols);
    }
}
