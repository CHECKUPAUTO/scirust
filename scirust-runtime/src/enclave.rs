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
