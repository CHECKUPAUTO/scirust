// scirust-core/src/tests/test_qat.rs
#[cfg(test)]
mod tests {
    use crate::autodiff::reverse::{Tape, Tensor};

    #[test]
    fn test_fake_quantize_ste() {
        let tape = Tape::new();
        // x = 0.7, scale = 0.1, zero_point = 0
        // Expected quantized: round(0.7/0.1) = 7.0
        // Expected dequantized: 7.0 * 0.1 = 0.7
        let x = tape.input(Tensor::from_vec(vec![0.72], 1, 1));
        let x_idx = x.idx();
        let y = x.fake_quantize_ste(0.1, 0);

        let val = tape.value(y.idx());
        assert!((val.data[0] - 0.7).abs() < 1e-6);

        // Backward: STE should pass 1.0 back
        let loss = y.sum();
        loss.backward();
        let grad = tape.grad(x_idx);
        assert_eq!(grad.data[0], 1.0);
    }
}
