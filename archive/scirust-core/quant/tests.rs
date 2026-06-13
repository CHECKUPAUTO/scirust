//! Tests de validation du pilier 5 (quantification native).

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_int8_quantize_dequantize() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();

        let (quantized, scale) = quantize_tensor_f32_to_i8(&data);
        let recovered = dequantize_i8_to_f32(&quantized, scale);

        assert_eq!(quantized.len(), 100);
        assert_eq!(recovered.len(), 100);

        let max_err: f32 = data
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(
            max_err < 0.02,
            "int8 quantization error too large: {}",
            max_err
        );
    }

    #[test]
    fn test_bf16_roundtrip() {
        let values = [0.0, 1.0, -1.0, 0.5, 2.0, -0.25, 100.0, -100.0];

        for &v in &values {
            let bf16 = f32_to_bf16(v);
            let back = bf16_to_f32(bf16);
            let error = (v - back).abs();
            assert!(
                error < 0.01 || v == 0.0,
                "bf16 roundtrip error too large for {}: {} vs {}",
                v,
                back,
                error
            );
        }
    }

    #[test]
    fn test_bf16_quantize_dequantize() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let bf16 = quantize_tensor_f32_to_bf16(&data);
        let recovered = dequantize_bf16_to_f32(&bf16);

        assert_eq!(bf16.len(), 100);
        assert_eq!(recovered.len(), 100);

        let max_err: f32 = data
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(
            max_err < 0.02,
            "bf16 max error: {}",
            max_err
        );
    }

    #[test]
    fn test_i4_quantize_dequantize() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 25.0).collect();
        let (packed, scale) = quantize_tensor_f32_to_i4(&data);
        let recovered = dequantize_i4_to_f32(&packed, scale);

        assert_eq!(recovered.len(), 100);
        let max_err: f32 = data
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(
            max_err < 0.1,
            "int4 max error: {}",
            max_err
        );
    }

    #[test]
    fn test_quant_tensor_int8() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let qt = QuantTensor::quantize_i8(&data, &[100]);

        assert_eq!(qt.format, QuantFormat::Int8);
        assert_eq!(qt.storage.len(), 100); // i8 = 1 byte
        assert_eq!(qt.numel(), 100);

        let recovered = qt.dequantize();
        assert_eq!(recovered.len(), 100);

        let max_err: f32 = data
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(max_err < 0.02, "int8 roundtrip error: {}", max_err);
    }

    #[test]
    fn test_quant_tensor_bf16() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let qt = QuantTensor::quantize_bf16(&data, &[100]);

        assert_eq!(qt.format, QuantFormat::Bf16);
        assert_eq!(qt.storage.len(), 200); // bf16 = 2 bytes
        assert_eq!(qt.numel(), 100);

        let recovered = qt.dequantize();
        assert_eq!(recovered.len(), 100);

        let max_err: f32 = data
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(max_err < 0.02, "bf16 roundtrip error: {}", max_err);
    }
}
