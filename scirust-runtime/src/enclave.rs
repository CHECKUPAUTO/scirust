//! Hardware-secured enclave inference ABI.
//!
//! Version 1 carries an element count for every raw buffer. The legacy symbol
//! remains exported for binary compatibility, but fails closed because its
//! pointer-only contract cannot prove that a C/TEE-provided allocation is large
//! enough.

use core::mem::align_of;
use core::ptr;

pub const ENCLAVE_ABI_VERSION: u32 = 1;
pub const ENCLAVE_OK: i32 = 0;
pub const ENCLAVE_ERR_NULL_OR_ALIGN: i32 = -1;
pub const ENCLAVE_ERR_DIMENSIONS: i32 = -2;
pub const ENCLAVE_ERR_LENGTH: i32 = -3;
pub const ENCLAVE_ERR_BIAS: i32 = -4;
pub const ENCLAVE_ERR_ABI: i32 = -5;
pub const ENCLAVE_ERR_OVERLAP: i32 = -6;

/// Stable C layout for matrix dimensions. `has_bias` is a byte rather than a
/// Rust `bool`: every u8 bit-pattern is valid at an FFI boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnclaveLayout {
    pub batch: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub has_bias: u8,
}

impl EnclaveLayout {
    pub fn new(batch: usize, in_features: usize, out_features: usize, has_bias: bool) -> Self {
        Self {
            batch,
            in_features,
            out_features,
            has_bias: u8::from(has_bias),
        }
    }

    #[inline]
    pub fn uses_bias(self) -> bool {
        self.has_bias == 1
    }
}

/// Legacy pointer-only ABI. It is retained so existing dynamic linkers still
/// resolve the symbol, but it cannot safely inspect any buffer and therefore
/// requires callers to migrate to [`safe_enclave_infer_v1`].
///
/// # Safety
///
/// This compatibility entry point never dereferences its pointer arguments and
/// therefore imposes no pointer-validity requirements. Its unsafe signature is
/// retained to preserve the legacy C ABI.
#[deprecated(note = "use safe_enclave_infer_v1 with explicit buffer lengths")]
#[no_mangle]
pub unsafe extern "C" fn safe_enclave_infer(
    _weight_ptr: *const f32,
    _input_ptr: *const f32,
    _output_ptr: *mut f32,
    _bias_ptr: *const f32,
    _dims: EnclaveLayout,
) -> i32 {
    ENCLAVE_ERR_ABI
}

#[inline]
fn required_lengths(dims: EnclaveLayout) -> Option<(usize, usize, usize)> {
    Some((
        dims.out_features.checked_mul(dims.in_features)?,
        dims.batch.checked_mul(dims.in_features)?,
        dims.batch.checked_mul(dims.out_features)?,
    ))
}

#[inline]
fn valid_ptr<T>(ptr: *const T, required: usize) -> bool {
    required == 0 || (!ptr.is_null() && (ptr as usize).is_multiple_of(align_of::<T>()))
}

#[inline]
fn touched_range<T>(ptr: *const T, elements: usize) -> Option<(usize, usize)> {
    let start = ptr as usize;
    let bytes = elements.checked_mul(core::mem::size_of::<T>())?;
    Some((start, start.checked_add(bytes)?))
}

#[inline]
fn ranges_overlap(left: (usize, usize), right: (usize, usize)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

/// Versioned enclave inference entry point with explicit element counts.
///
/// # Safety
///
/// For every non-empty buffer, the pointer must reference at least the supplied
/// number of initialized `f32` elements for the duration of the call. Output
/// must be writable. The function validates the supplied counts, alignment,
/// arithmetic and touched-range overlap before dereferencing any pointer.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn safe_enclave_infer_v1(
    abi_version: u32,
    weight_ptr: *const f32,
    weight_len: usize,
    input_ptr: *const f32,
    input_len: usize,
    output_ptr: *mut f32,
    output_len: usize,
    bias_ptr: *const f32,
    bias_len: usize,
    dims: EnclaveLayout,
) -> i32 {
    if abi_version != ENCLAVE_ABI_VERSION
    {
        return ENCLAVE_ERR_ABI;
    }
    if dims.has_bias > 1
    {
        return ENCLAVE_ERR_DIMENSIONS;
    }
    let Some((weight_required, input_required, output_required)) = required_lengths(dims)
    else
    {
        return ENCLAVE_ERR_DIMENSIONS;
    };
    if weight_len < weight_required || input_len < input_required || output_len < output_required
    {
        return ENCLAVE_ERR_LENGTH;
    }
    if !valid_ptr(weight_ptr, weight_required)
        || !valid_ptr(input_ptr, input_required)
        || !valid_ptr(output_ptr.cast_const(), output_required)
    {
        return ENCLAVE_ERR_NULL_OR_ALIGN;
    }
    if dims.uses_bias() && (bias_len < dims.out_features || !valid_ptr(bias_ptr, dims.out_features))
    {
        return ENCLAVE_ERR_BIAS;
    }

    let Some(output_range) = touched_range(output_ptr.cast_const(), output_required)
    else
    {
        return ENCLAVE_ERR_DIMENSIONS;
    };
    let Some(weight_range) = touched_range(weight_ptr, weight_required)
    else
    {
        return ENCLAVE_ERR_DIMENSIONS;
    };
    let Some(input_range) = touched_range(input_ptr, input_required)
    else
    {
        return ENCLAVE_ERR_DIMENSIONS;
    };
    if ranges_overlap(output_range, weight_range) || ranges_overlap(output_range, input_range)
    {
        return ENCLAVE_ERR_OVERLAP;
    }
    if dims.uses_bias()
    {
        let Some(bias_range) = touched_range(bias_ptr, dims.out_features)
        else
        {
            return ENCLAVE_ERR_DIMENSIONS;
        };
        if ranges_overlap(output_range, bias_range)
        {
            return ENCLAVE_ERR_OVERLAP;
        }
    }

    for batch in 0..dims.batch
    {
        for output_feature in 0..dims.out_features
        {
            let mut acc = 0.0f32;
            for input_feature in 0..dims.in_features
            {
                let input_index = batch * dims.in_features + input_feature;
                let weight_index = output_feature * dims.in_features + input_feature;
                // SAFETY: all products, lengths, alignment and non-null
                // requirements were validated above.
                let input_value = unsafe { *input_ptr.add(input_index) };
                let weight_value = unsafe { *weight_ptr.add(weight_index) };
                acc += input_value * weight_value;
            }
            if dims.uses_bias()
            {
                acc += unsafe { *bias_ptr.add(output_feature) };
            }
            let output_index = batch * dims.out_features + output_feature;
            unsafe { *output_ptr.add(output_index) = acc };
        }
    }
    ENCLAVE_OK
}

pub struct EnclaveRuntime;

impl EnclaveRuntime {
    pub fn infer(
        &self,
        weights: &[f32],
        input: &[f32],
        output: &mut [f32],
        bias: Option<&[f32]>,
        dims: EnclaveLayout,
    ) -> Result<(), i32> {
        let bias_ptr = bias.map_or(ptr::null(), |values| values.as_ptr());
        let bias_len = bias.map_or(0, <[f32]>::len);
        let result = unsafe {
            safe_enclave_infer_v1(
                ENCLAVE_ABI_VERSION,
                weights.as_ptr(),
                weights.len(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                bias_ptr,
                bias_len,
                dims,
            )
        };
        if result == ENCLAVE_OK
        {
            Ok(())
        }
        else
        {
            Err(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(batch: usize, input: usize, output: usize, bias: bool) -> EnclaveLayout {
        EnclaveLayout::new(batch, input, output, bias)
    }

    #[test]
    fn infer_runs_for_consistent_buffers() {
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = [1.0, 0.0, 1.0];
        let bias = [0.5, -0.5];
        let mut output = [0.0; 2];
        EnclaveRuntime
            .infer(
                &weights,
                &input,
                &mut output,
                Some(&bias),
                layout(1, 3, 2, true),
            )
            .unwrap();
        assert_eq!(output, [4.5, 9.5]);
    }

    #[test]
    fn raw_abi_rejects_undersized_buffer_before_dereference() {
        let weights = [0.0; 5];
        let input = [0.0; 3];
        let mut output = [0.0; 2];
        let result = unsafe {
            safe_enclave_infer_v1(
                ENCLAVE_ABI_VERSION,
                weights.as_ptr(),
                weights.len(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                ptr::null(),
                0,
                layout(1, 3, 2, false),
            )
        };
        assert_eq!(result, ENCLAVE_ERR_LENGTH);
    }

    #[test]
    fn raw_abi_rejects_unknown_version_and_invalid_flag() {
        let mut output = [];
        let empty = [].as_ptr();
        let result = unsafe {
            safe_enclave_infer_v1(
                99,
                empty,
                0,
                empty,
                0,
                output.as_mut_ptr(),
                0,
                ptr::null(),
                0,
                layout(0, 0, 0, false),
            )
        };
        assert_eq!(result, ENCLAVE_ERR_ABI);

        let invalid = EnclaveLayout {
            has_bias: 2,
            ..layout(0, 0, 0, false)
        };
        let result = unsafe {
            safe_enclave_infer_v1(
                ENCLAVE_ABI_VERSION,
                empty,
                0,
                empty,
                0,
                output.as_mut_ptr(),
                0,
                ptr::null(),
                0,
                invalid,
            )
        };
        assert_eq!(result, ENCLAVE_ERR_DIMENSIONS);
    }

    #[test]
    fn legacy_abi_fails_closed() {
        #[allow(deprecated)]
        let result = unsafe {
            safe_enclave_infer(
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null(),
                layout(1, 1, 1, false),
            )
        };
        assert_eq!(result, ENCLAVE_ERR_ABI);
    }

    #[test]
    fn safe_wrapper_rejects_missing_bias() {
        let weights = [0.0; 6];
        let input = [0.0; 3];
        let mut output = [0.0; 2];
        assert_eq!(
            EnclaveRuntime.infer(&weights, &input, &mut output, None, layout(1, 3, 2, true)),
            Err(ENCLAVE_ERR_BIAS)
        );
    }
}
