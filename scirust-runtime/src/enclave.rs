//! Hardware-Secured Enclave Runtime Target (TEE / TrustZone Compatibility)
//!
//! Provides a hardened, allocation-free entry point for model inference
//! inside hardware-isolated execution environments.
//! Communicates via raw, size-bounded pointers without OS allocator dependency.

use core::ptr;

/// Layout descriptor for enclave inference.
#[repr(C)]
pub struct EnclaveLayout {
    pub batch: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub has_bias: bool,
}

/// Enclave-compatible entry point for secure inference.
/// Runs in a strict #![no_std] environment.
#[no_mangle]
/// # Safety
/// Pointers must be valid.
pub unsafe extern "C" fn safe_enclave_infer(
    weight_ptr: *const f32,
    input_ptr: *const f32,
    output_ptr: *mut f32,
    bias_ptr: *const f32,
    dims: EnclaveLayout,
) -> i32 {
    // 1. Parameter Validation
    if weight_ptr.is_null() || input_ptr.is_null() || output_ptr.is_null()
    {
        return -1;
    }

    // 2. Allocation-free execution path (Static Matrix Multiplication)
    // Computes Output = Input * Weight.T (+ Bias)
    // Assuming Weight is (out_features, in_features) in row-major.

    for b in 0..dims.batch
    {
        for o in 0..dims.out_features
        {
            let mut acc = 0.0f32;

            // Dot product between input row and weight row
            for i in 0..dims.in_features
            {
                let input_val = *input_ptr.add(b * dims.in_features + i);
                let weight_val = *weight_ptr.add(o * dims.in_features + i);
                acc += input_val * weight_val;
            }

            // Add bias if provided
            if dims.has_bias && !bias_ptr.is_null()
            {
                acc += *bias_ptr.add(o);
            }

            // Write to secured output buffer
            *output_ptr.add(b * dims.out_features + o) = acc;
        }
    }

    0 // Success
}

/// Hardened fixed-memory wrapper for inference runtime.
pub struct EnclaveRuntime {
    // Internal state can be added here as fixed-size arrays if needed for #![no_std]
}

impl EnclaveRuntime {
    pub fn infer(
        &self,
        weights: &[f32],
        input: &[f32],
        output: &mut [f32],
        bias: Option<&[f32]>,
        dims: EnclaveLayout,
    ) -> Result<(), i32> {
        // Validate `dims` against the actual slice lengths BEFORE entering the
        // `unsafe` FFI path. The raw-pointer entry point cannot check this
        // itself, so a mismatched `dims` would otherwise read/write out of
        // bounds inside the TEE. All multiplications are checked to avoid
        // overflow on a malicious/corrupted `dims` (u64 on 64-bit → cast to
        // usize is fine, but the products can still overflow `usize`).
        let weight_elems = dims.out_features.checked_mul(dims.in_features).ok_or(-2)?;
        let input_elems = dims.batch.checked_mul(dims.in_features).ok_or(-2)?;
        let output_elems = dims.batch.checked_mul(dims.out_features).ok_or(-2)?;
        if weights.len() < weight_elems || input.len() < input_elems || output.len() < output_elems
        {
            return Err(-3);
        }
        // `has_bias` set requires a bias slice at least as long as `out_features`.
        // `map_or(true, …)` returns true (→ reject) when `bias` is `None`.
        if dims.has_bias && bias.is_none_or(|b| b.len() < dims.out_features)
        {
            return Err(-4);
        }

        let b_ptr = bias.map(|b| b.as_ptr()).unwrap_or(ptr::null());
        let res = unsafe {
            safe_enclave_infer(
                weights.as_ptr(),
                input.as_ptr(),
                output.as_mut_ptr(),
                b_ptr,
                dims,
            )
        };

        if res == 0 { Ok(()) } else { Err(res) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(batch: usize, inf: usize, outf: usize, has_bias: bool) -> EnclaveLayout {
        EnclaveLayout {
            batch,
            in_features: inf,
            out_features: outf,
            has_bias,
        }
    }

    #[test]
    fn infer_runs_for_consistent_dims() {
        // Weights (out=2, in=3) row-major, batch=1 input of 3, output of 2.
        let weights = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = [1.0, 0.0, 1.0];
        let mut output = [0.0f32; 2];
        let rt = EnclaveRuntime {};
        let r = rt.infer(&weights, &input, &mut output, None, layout(1, 3, 2, false));
        assert_eq!(r, Ok(()));
        // row0 = 1*1 + 2*0 + 3*1 = 4 ; row1 = 4*1 + 5*0 + 6*1 = 10
        assert!((output[0] - 4.0).abs() < 1e-6);
        assert!((output[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn infer_rejects_undersized_input() {
        let weights = [0.0f32; 6]; // (2,3)
        let input = [0.0f32; 2]; // needs 3
        let mut output = [0.0f32; 2];
        let rt = EnclaveRuntime {};
        assert_eq!(
            rt.infer(&weights, &input, &mut output, None, layout(1, 3, 2, false)),
            Err(-3)
        );
    }

    #[test]
    fn infer_rejects_undersized_weights() {
        let weights = [0.0f32; 5]; // needs 6
        let input = [0.0f32; 3];
        let mut output = [0.0f32; 2];
        let rt = EnclaveRuntime {};
        assert_eq!(
            rt.infer(&weights, &input, &mut output, None, layout(1, 3, 2, false)),
            Err(-3)
        );
    }

    #[test]
    fn infer_rejects_undersized_output() {
        let weights = [0.0f32; 6];
        let input = [0.0f32; 3];
        let mut output = [0.0f32; 1]; // needs 2
        let rt = EnclaveRuntime {};
        assert_eq!(
            rt.infer(&weights, &input, &mut output, None, layout(1, 3, 2, false)),
            Err(-3)
        );
    }

    #[test]
    fn infer_rejects_missing_bias_when_has_bias() {
        let weights = [0.0f32; 6];
        let input = [0.0f32; 3];
        let mut output = [0.0f32; 2];
        let rt = EnclaveRuntime {};
        // has_bias=true but bias=None -> -4
        assert_eq!(
            rt.infer(&weights, &input, &mut output, None, layout(1, 3, 2, true)),
            Err(-4)
        );
    }

    #[test]
    fn infer_rejects_undersized_bias() {
        let weights = [0.0f32; 6];
        let input = [0.0f32; 3];
        let mut output = [0.0f32; 2];
        let bias = [0.0f32; 1]; // needs 2
        let rt = EnclaveRuntime {};
        assert_eq!(
            rt.infer(
                &weights,
                &input,
                &mut output,
                Some(&bias),
                layout(1, 3, 2, true)
            ),
            Err(-4)
        );
    }
}
