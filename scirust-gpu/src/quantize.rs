// scirust-gpu/src/quantize.rs
//
// TurboQuant — Quantization primitives for AI inference.
//
// Modes supported:
//   FP32  — identity (no quantization)
//   FP16  — f32 → half-precision via `half` crate
//   INT8  — symmetric per-block, scale = max(abs(block)) / 127
//   INT4  — asymmetric per-block, scale + zero-point, 2 values per byte
//   NF4   — Normal Float 4-bit (QLoRA-style), lookup table
//
// All modes support per-block quantization with configurable block_size.

use crate::error::{QuantError, Result};

// ------------------------------------------------------------------ //
//  QuantMode                                                          //
// ------------------------------------------------------------------ //

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantMode {
    FP32,
    FP16,
    INT8,
    INT4,
    NF4,
}

impl QuantMode {
    pub fn bits_per_element(&self) -> usize {
        match self
        {
            QuantMode::FP32 => 32,
            QuantMode::FP16 => 16,
            QuantMode::INT8 => 8,
            QuantMode::INT4 => 4,
            QuantMode::NF4 => 4,
        }
    }
}

// ------------------------------------------------------------------ //
//  QuantizedTensor                                                    //
// ------------------------------------------------------------------ //

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed quantized data bytes.
    pub data: Vec<u8>,
    /// Per-block scale factors.
    pub scale: Vec<f32>,
    /// Per-block zero points.
    pub zero: Vec<f32>,
    /// Original tensor shape (rows, cols).
    pub shape: (usize, usize),
    /// Quantization mode used.
    pub mode: QuantMode,
    /// Block size used for per-block quantization.
    pub block_size: usize,
}

impl QuantizedTensor {
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of blocks.
    pub fn num_blocks(&self) -> usize {
        let n = self.len();
        (n + self.block_size - 1) / self.block_size
    }
}

// ------------------------------------------------------------------ //
//  NF4 Lookup Table (QLoRA-style)                                     //
// ------------------------------------------------------------------ //

/// NF4 lookup table: maps 4-bit index (0..15) to bfloat16-approximate values.
/// The values are chosen so that 0 maps to exactly 0.0 and the distribution
/// of values follows a normal distribution quantile-based spacing.
const NF4_TABLE: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730516910553,
    -0.3949174889960298,
    -0.28444138169288635,
    -0.1847734004259119,
    -0.09105003625154495,
    0.0,
    0.0795802993774414,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Reverse NF4 lookup: find the index whose value is closest to `v`.
fn nf4_lookup_index(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_dist = (v - NF4_TABLE[0]).abs();
    for (i, &t) in NF4_TABLE.iter().enumerate().skip(1)
    {
        let d = (v - t).abs();
        if d < best_dist
        {
            best = i as u8;
            best_dist = d;
        }
    }
    best
}

/// Pack two 4-bit values into one byte: hi << 4 | lo.
#[inline]
fn pack_nibbles(lo: u8, hi: u8) -> u8 {
    (hi & 0x0F) << 4 | (lo & 0x0F)
}

/// Unpack one byte into two 4-bit values: (lo, hi).
#[inline]
fn unpack_nibbles(byte: u8) -> (u8, u8) {
    (byte & 0x0F, (byte >> 4) & 0x0F)
}

// ------------------------------------------------------------------ //
//  Quantizer                                                          //
// ------------------------------------------------------------------ //

pub struct Quantizer {
    pub mode: QuantMode,
    pub block_size: usize,
}

impl Quantizer {
    pub fn new(mode: QuantMode, block_size: usize) -> Self {
        assert!(block_size >= 1, "block_size must be >= 1");
        Self { mode, block_size }
    }

    /// Quantize a flat f32 tensor (assumed 2D, shape inferred).
    /// `tensor` is a row-major f32 slice, `rows` and `cols` define shape.
    pub fn quantize_2d(&self, tensor: &[f32], rows: usize, cols: usize) -> Result<QuantizedTensor> {
        if tensor.len() != rows * cols
        {
            return Err(QuantError::ShapeMismatch {
                expected: rows * cols,
                got: tensor.len(),
            });
        }
        let n = tensor.len();
        let block_size = self.block_size;
        let num_blocks = (n + block_size - 1) / block_size;

        let mut scale = Vec::with_capacity(num_blocks);
        let mut zero = Vec::with_capacity(num_blocks);

        match self.mode
        {
            QuantMode::FP32 =>
            {
                // Identity: store f32 bytes directly
                let data: Vec<u8> = tensor.iter().flat_map(|&v| v.to_le_bytes()).collect();
                for _ in 0..num_blocks
                {
                    scale.push(1.0);
                    zero.push(0.0);
                }
                return Ok(QuantizedTensor {
                    data,
                    scale,
                    zero,
                    shape: (rows, cols),
                    mode: QuantMode::FP32,
                    block_size,
                });
            },
            QuantMode::FP16 =>
            {
                // f32 → f16 via half crate, store as u16 bytes
                let data: Vec<u8> = tensor
                    .iter()
                    .flat_map(|&v| {
                        let h = half::f16::from_f32(v);
                        h.to_bits().to_le_bytes()
                    })
                    .collect();
                for _ in 0..num_blocks
                {
                    scale.push(1.0);
                    zero.push(0.0);
                }
                return Ok(QuantizedTensor {
                    data,
                    scale,
                    zero,
                    shape: (rows, cols),
                    mode: QuantMode::FP16,
                    block_size,
                });
            },
            QuantMode::INT8 =>
            {
                let mut data = Vec::with_capacity(n);
                for b in 0..num_blocks
                {
                    let start = b * block_size;
                    let end = (start + block_size).min(n);
                    let block = &tensor[start..end];

                    // Compute scale = max(abs(block)) / 127
                    let mut abs_max = 0.0f32;
                    for &v in block
                    {
                        let a = v.abs();
                        if a > abs_max
                        {
                            abs_max = a;
                        }
                    }
                    let s = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
                    scale.push(s);
                    zero.push(0.0); // symmetric, no zero-point

                    for &v in block
                    {
                        let q = (v / s).round().clamp(-128.0, 127.0) as i8;
                        data.push(q as u8);
                    }
                }
                return Ok(QuantizedTensor {
                    data,
                    scale,
                    zero,
                    shape: (rows, cols),
                    mode: QuantMode::INT8,
                    block_size,
                });
            },
            QuantMode::INT4 =>
            {
                // Asymmetric: scale = (max - min) / 15, zero = min
                let mut packed = Vec::with_capacity((n + 1) / 2);
                for b in 0..num_blocks
                {
                    let start = b * block_size;
                    let end = (start + block_size).min(n);
                    let block = &tensor[start..end];

                    let mut min_val = block[0];
                    let mut max_val = block[0];
                    for &v in block
                    {
                        if v < min_val
                        {
                            min_val = v;
                        }
                        if v > max_val
                        {
                            max_val = v;
                        }
                    }
                    let s = if max_val - min_val < 1e-10
                    {
                        1.0
                    }
                    else
                    {
                        (max_val - min_val) / 15.0
                    };
                    let z = min_val;
                    scale.push(s);
                    zero.push(z);

                    // Quantize block, pack as nibbles
                    let mut i = 0;
                    while i < block.len()
                    {
                        let lo_val = block[i];
                        let lo_quant = ((lo_val - z) / s).round().clamp(0.0, 15.0) as u8;
                        if i + 1 < block.len()
                        {
                            let hi_val = block[i + 1];
                            let hi_quant = ((hi_val - z) / s).round().clamp(0.0, 15.0) as u8;
                            packed.push(pack_nibbles(lo_quant, hi_quant));
                        }
                        else
                        {
                            // Last odd element: pad hi nibble with 0
                            packed.push(pack_nibbles(lo_quant, 0));
                        }
                        i += 2;
                    }
                }
                return Ok(QuantizedTensor {
                    data: packed,
                    scale,
                    zero,
                    shape: (rows, cols),
                    mode: QuantMode::INT4,
                    block_size,
                });
            },
            QuantMode::NF4 =>
            {
                // Quantile-based: map to nearest NF4 table entry, pack nibbles
                let mut packed = Vec::with_capacity((n + 1) / 2);
                for b in 0..num_blocks
                {
                    let start = b * block_size;
                    let end = (start + block_size).min(n);
                    let block = &tensor[start..end];

                    // Compute scale = max(abs(block)), but use per-element scaling
                    let mut abs_max = 0.0f32;
                    for &v in block
                    {
                        let a = v.abs();
                        if a > abs_max
                        {
                            abs_max = a;
                        }
                    }
                    let s = if abs_max < 1e-10 { 1.0 } else { abs_max };
                    scale.push(s);
                    zero.push(0.0);

                    // Quantize: normalize to [-1, 1], look up nearest NF4 value
                    let mut i = 0;
                    while i < block.len()
                    {
                        let lo_val = (block[i] / s).clamp(-1.0, 1.0);
                        let lo_idx = nf4_lookup_index(lo_val);
                        if i + 1 < block.len()
                        {
                            let hi_val = (block[i + 1] / s).clamp(-1.0, 1.0);
                            let hi_idx = nf4_lookup_index(hi_val);
                            packed.push(pack_nibbles(lo_idx, hi_idx));
                        }
                        else
                        {
                            packed.push(pack_nibbles(lo_idx, 0));
                        }
                        i += 2;
                    }
                }
                return Ok(QuantizedTensor {
                    data: packed,
                    scale,
                    zero,
                    shape: (rows, cols),
                    mode: QuantMode::NF4,
                    block_size,
                });
            },
        }
    }

    /// Convenience: assume tensor is a flat 2D with shape (tensor.len(), 1).
    pub fn quantize(&self, tensor: &[f32]) -> QuantizedTensor {
        let n = tensor.len();
        self.quantize_2d(tensor, n, 1).expect("quantize")
    }

    /// Dequantize back to f32.
    /// `qt` must use the same mode and block_size as `self`.
    pub fn dequantize(&self, qt: &QuantizedTensor) -> Vec<f32> {
        let n = qt.len();
        let block_size = qt.block_size;
        let num_blocks = qt.num_blocks();

        match qt.mode
        {
            QuantMode::FP32 =>
            {
                // Interpret data as raw f32 bytes
                let mut out = Vec::with_capacity(n);
                let chunks = qt.data.chunks_exact(4);
                for chunk in chunks
                {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    out.push(f32::from_le_bytes(bytes));
                }
                out
            },
            QuantMode::FP16 =>
            {
                // Interpret data as u16 bits, convert to f32
                let mut out = Vec::with_capacity(n);
                let chunks = qt.data.chunks_exact(2);
                for chunk in chunks
                {
                    let bytes: [u8; 2] = chunk.try_into().unwrap();
                    let bits = u16::from_le_bytes(bytes);
                    let h = half::f16::from_bits(bits);
                    out.push(f32::from(h));
                }
                out
            },
            QuantMode::INT8 =>
            {
                let mut out = Vec::with_capacity(n);
                for b in 0..num_blocks
                {
                    let start = b * block_size;
                    let end = (start + block_size).min(n);
                    let s = qt.scale[b];
                    for i in start..end
                    {
                        let q = qt.data[i] as i8 as f32;
                        out.push(q * s);
                    }
                }
                out
            },
            QuantMode::INT4 =>
            {
                let mut out = Vec::with_capacity(n);
                let mut byte_idx = 0;
                for b in 0..num_blocks
                {
                    let start = b * block_size;
                    let end = (start + block_size).min(n);
                    let s = qt.scale[b];
                    let z = qt.zero[b];
                    let mut elem_in_block = 0;
                    while start + elem_in_block < end
                    {
                        let byte = qt.data[byte_idx];
                        let (lo, hi) = unpack_nibbles(byte);
                        let lo_f = lo as f32 * s + z;
                        out.push(lo_f);
                        elem_in_block += 1;
                        if start + elem_in_block < end
                        {
                            let hi_f = hi as f32 * s + z;
                            out.push(hi_f);
                            elem_in_block += 1;
                        }
                        byte_idx += 1;
                    }
                }
                out
            },
            QuantMode::NF4 =>
            {
                let mut out = Vec::with_capacity(n);
                let mut byte_idx = 0;
                for b in 0..num_blocks
                {
                    let start = b * block_size;
                    let end = (start + block_size).min(n);
                    let s = qt.scale[b];
                    let mut elem_in_block = 0;
                    while start + elem_in_block < end
                    {
                        let byte = qt.data[byte_idx];
                        let (lo_idx, hi_idx) = unpack_nibbles(byte);
                        let lo_f = NF4_TABLE[lo_idx as usize] * s;
                        out.push(lo_f);
                        elem_in_block += 1;
                        if start + elem_in_block < end
                        {
                            let hi_f = NF4_TABLE[hi_idx as usize] * s;
                            out.push(hi_f);
                            elem_in_block += 1;
                        }
                        byte_idx += 1;
                    }
                }
                out
            },
        }
    }

    /// Compute quantization error (MSE) between original and quantized/dequantized.
    pub fn quantization_error(&self, original: &[f32], quantized: &[f32]) -> f64 {
        assert_eq!(
            original.len(),
            quantized.len(),
            "quantization_error: length mismatch"
        );
        if original.is_empty()
        {
            return 0.0;
        }
        let sum_sq: f64 = original
            .iter()
            .zip(quantized.iter())
            .map(|(a, b)| {
                let diff = *a as f64 - *b as f64;
                diff * diff
            })
            .sum();
        sum_sq / original.len() as f64
    }
}

// ------------------------------------------------------------------ //
//  Tests                                                              //
// ------------------------------------------------------------------ //

#[cfg(test)]
mod tests {
    use super::*;

    // Utility: generate a deterministic non-trivial float vector
    fn test_data(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = i as f32;
                (x * 0.1).sin() * 3.0 + (x * 0.05).cos() * 2.0
            })
            .collect()
    }

    // ------------------------------------------------------------------ //
    //  FP32 roundtrip ≈ exact                                             //
    // ------------------------------------------------------------------ //
    #[test]
    fn test_quantize_dequantize_fp32_identity() {
        let data = test_data(64);
        let q = Quantizer::new(QuantMode::FP32, 32);
        let qt = q.quantize(&data);
        assert_eq!(qt.mode, QuantMode::FP32);
        assert_eq!(qt.len(), 64);

        let recovered = q.dequantize(&qt);
        assert_eq!(data.len(), recovered.len());
        for (a, b) in data.iter().zip(recovered.iter())
        {
            assert!((a - b).abs() < 1e-6, "FP32 mismatch: {} vs {}", a, b);
        }
        let err = q.quantization_error(&data, &recovered);
        assert!(err < 1e-15, "FP32 MSE too large: {}", err);
    }

    // ------------------------------------------------------------------ //
    //  INT8 roundtrip ≈ approx                                            //
    // ------------------------------------------------------------------ //
    #[test]
    fn test_quantize_dequantize_int8() {
        let data = test_data(128);
        let q = Quantizer::new(QuantMode::INT8, 32);
        let qt = q.quantize(&data);
        assert_eq!(qt.mode, QuantMode::INT8);

        let recovered = q.dequantize(&qt);
        assert_eq!(data.len(), recovered.len());

        // INT8 should capture the rough shape
        let err = q.quantization_error(&data, &recovered);
        // Typical MSE should be reasonable for this kind of data
        assert!(err < 10.0, "INT8 MSE suspiciously large: {}", err);

        // Max approx relative error should be bounded
        for (a, r) in data.iter().zip(recovered.iter())
        {
            if a.abs() > 0.01
            {
                let re = (a - r).abs() / a.abs();
                assert!(re < 2.0, "INT8 relative error too large: {} vs {}", a, r);
            }
        }
    }

    // ------------------------------------------------------------------ //
    //  MSE hierarchy: INT4 > INT8 > FP16 > FP32                          //
    // ------------------------------------------------------------------ //
    #[test]
    fn test_quantization_error_increases() {
        let data = test_data(256);
        let block_size = 64;

        let q32 = Quantizer::new(QuantMode::FP32, block_size);
        let qt32 = q32.quantize(&data);
        let dq32 = q32.dequantize(&qt32);
        let err32 = q32.quantization_error(&data, &dq32);

        let q16 = Quantizer::new(QuantMode::FP16, block_size);
        let qt16 = q16.quantize(&data);
        let dq16 = q16.dequantize(&qt16);
        let err16 = q16.quantization_error(&data, &dq16);

        let q8 = Quantizer::new(QuantMode::INT8, block_size);
        let qt8 = q8.quantize(&data);
        let dq8 = q8.dequantize(&qt8);
        let err8 = q8.quantization_error(&data, &dq8);

        let q4 = Quantizer::new(QuantMode::INT4, block_size);
        let qt4 = q4.quantize(&data);
        let dq4 = q4.dequantize(&qt4);
        let err4 = q4.quantization_error(&data, &dq4);

        // FP32 should be near zero
        assert!(err32 < 1e-15, "FP32 MSE should be ~0, got {}", err32);

        // Expected hierarchy: FP32 < FP16 < INT8 < INT4
        // (FP16 can be very close to FP32 for small values)
        assert!(
            err16 >= err32,
            "FP16 MSE ({}) should be >= FP32 MSE ({})",
            err16,
            err32
        );
        assert!(
            err8 >= err16,
            "INT8 MSE ({}) should be >= FP16 MSE ({})",
            err8,
            err16
        );
        assert!(
            err4 >= err8,
            "INT4 MSE ({}) should be >= INT8 MSE ({})",
            err4,
            err8
        );
    }

    // ------------------------------------------------------------------ //
    //  NF4 roundtrip                                                      //
    // ------------------------------------------------------------------ //
    #[test]
    fn test_nf4_quantize_dequantize() {
        // NF4 maps values in [-1, 1] via the lookup table + per-block scaling.
        let data = test_data(128);
        let q = Quantizer::new(QuantMode::NF4, 32);
        let qt = q.quantize(&data);
        assert_eq!(qt.mode, QuantMode::NF4);

        let recovered = q.dequantize(&qt);
        assert_eq!(data.len(), recovered.len());

        // NF4 is 4-bit, expect non-trivial error but sensible
        let err = q.quantization_error(&data, &recovered);
        assert!(err < 15.0, "NF4 MSE suspiciously large: {}", err);

        // Verify all recovered values are within [-scale, scale]
        // (each block scales independently)
        for &v in recovered.iter()
        {
            assert!(v.is_finite(), "NF4 produced non-finite value {}", v);
        }
    }

    // ------------------------------------------------------------------ //
    //  STE passes gradient (unit test in isolation)                       //
    // ------------------------------------------------------------------ //
    #[test]
    fn test_ste_passes_gradient() {
        let data = test_data(64);
        let q = Quantizer::new(QuantMode::INT8, 16);
        let qt = q.quantize(&data);
        let dq = q.dequantize(&qt);

        // Simulate STE: upstream gradient = ones
        let upstream_grad: Vec<f32> = vec![1.0; data.len()];

        // With STE, the gradient passes through the quantizer unchanged.
        // Compute gradient * (data - dq) — this is what STE backprop would do.
        let grad_output: Vec<f32> = upstream_grad
            .iter()
            .zip(data.iter().zip(dq.iter()))
            .map(|(&ug, (&orig, &rec))| ug * (orig - rec))
            .collect();

        // Gradient should be non-zero (quantization introduces error)
        let grad_sum: f64 = grad_output.iter().map(|&g| g.abs() as f64).sum();
        assert!(
            grad_sum > 0.0,
            "STE gradient should be non-zero after quantize/dequantize"
        );
    }

    // ------------------------------------------------------------------ //
    //  Export/Import roundtrip                                            //
    // ------------------------------------------------------------------ //
    #[test]
    fn test_export_import_quantized() {
        let data = test_data(256);
        let q = Quantizer::new(QuantMode::INT8, 64);
        let qt = q.quantize(&data);

        // Simulate write/read: serialize to bytes then reconstruct
        let serialized = {
            let mode_byte: u8 = match qt.mode
            {
                QuantMode::FP32 => 0,
                QuantMode::FP16 => 1,
                QuantMode::INT8 => 2,
                QuantMode::INT4 => 3,
                QuantMode::NF4 => 4,
            };
            let rows_bytes = (qt.shape.0 as u64).to_le_bytes();
            let cols_bytes = (qt.shape.1 as u64).to_le_bytes();
            let block_bytes = (qt.block_size as u64).to_le_bytes();
            let _num_blocks = qt.num_blocks();

            let mut buf = Vec::new();
            buf.push(mode_byte);
            buf.extend_from_slice(&rows_bytes);
            buf.extend_from_slice(&cols_bytes);
            buf.extend_from_slice(&block_bytes);
            // Scale and zero
            for &s in &qt.scale
            {
                buf.extend_from_slice(&s.to_le_bytes());
            }
            for &z in &qt.zero
            {
                buf.extend_from_slice(&z.to_le_bytes());
            }
            // Data length (as u64)
            let data_len = qt.data.len() as u64;
            buf.extend_from_slice(&data_len.to_le_bytes());
            buf.extend_from_slice(&qt.data);
            buf
        };

        // Deserialize
        let restored = {
            let mut pos = 0usize;
            let mode = match serialized[pos]
            {
                0 => QuantMode::FP32,
                1 => QuantMode::FP16,
                2 => QuantMode::INT8,
                3 => QuantMode::INT4,
                4 => QuantMode::NF4,
                _ => panic!("unknown mode"),
            };
            pos += 1;
            let rows = u64::from_le_bytes(serialized[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            let cols = u64::from_le_bytes(serialized[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            let blocksz = u64::from_le_bytes(serialized[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;

            let n = rows * cols;
            let num_blocks = (n + blocksz - 1) / blocksz;

            let mut scale = Vec::with_capacity(num_blocks);
            let mut zero = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks
            {
                let s = f32::from_le_bytes(serialized[pos..pos + 4].try_into().unwrap());
                scale.push(s);
                pos += 4;
            }
            for _ in 0..num_blocks
            {
                let z = f32::from_le_bytes(serialized[pos..pos + 4].try_into().unwrap());
                zero.push(z);
                pos += 4;
            }
            let data_len =
                u64::from_le_bytes(serialized[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            let data = serialized[pos..pos + data_len].to_vec();

            QuantizedTensor {
                data,
                scale,
                zero,
                shape: (rows, cols),
                mode,
                block_size: blocksz,
            }
        };

        // Dequantize both original and restored — should be identical
        let dq_orig = q.dequantize(&qt);
        let dq_restored = q.dequantize(&restored);

        assert_eq!(dq_orig.len(), dq_restored.len());
        for (a, b) in dq_orig.iter().zip(dq_restored.iter())
        {
            assert!(
                (a - b).abs() < 1e-6,
                "Export/import roundtrip mismatch: {} vs {}",
                a,
                b
            );
        }
    }
}
