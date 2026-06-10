#[cfg(test)]
mod tests {
    use crate::autodiff::reverse::{Tape, Tensor};
    use crate::nn::conv_utils::Padding;
    use crate::nn::conv2d::Conv2d;
    use crate::nn::init::{KaimingNormal, Zeros};
    use crate::nn::module::Module;
    use crate::nn::rng::PcgEngine;

    #[test]
    fn test_conv2d_gradient_correctness() {
        let mut rng = PcgEngine::new(42);
        let in_c = 1;
        let out_c = 1;
        let h = 4;
        let w = 4;
        let k = 3;

        let mut conv = Conv2d::new(
            in_c,
            out_c,
            k,
            1,
            Padding::Valid,
            &KaimingNormal,
            Some(&Zeros),
            &mut rng,
        )
        .input_dims(h, w);

        let tape = Tape::new();
        // Set fixed weights for predictable results
        conv.weight = Tensor::from_vec(vec![1.0; k * k], out_c, in_c * k * k);

        let x_data = (0..16).map(|i| i as f32).collect::<Vec<_>>();
        let x = tape.input(Tensor::from_vec(x_data, 1, 16));
        let x_idx = x.idx();

        let y = conv.forward(&tape, x);
        let loss = y.sum();
        loss.backward();

        let grad_x = tape.grad(x_idx);

        let expected_grad = vec![
            1.0, 2.0, 2.0, 1.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0, 2.0, 1.0, 2.0, 2.0, 1.0,
        ];

        for (i, &g) in grad_x.data.iter().enumerate()
        {
            assert!(
                (g - expected_grad[i]).abs() < 1e-5,
                "Grad mismatch at index {}: got {}, expected {}",
                i,
                g,
                expected_grad[i]
            );
        }
    }

    #[test]
    fn test_conv2d_forward_value_correctness() {
        let mut rng = PcgEngine::new(42);
        let in_c = 1;
        let out_c = 1;
        let h = 3;
        let w = 3;
        let k = 2;

        let mut conv = Conv2d::new(
            in_c,
            out_c,
            k,
            1,
            Padding::Valid,
            &KaimingNormal,
            Some(&Zeros),
            &mut rng,
        )
        .input_dims(h, w);

        let tape = Tape::new();
        // Set fixed weights: [1, 2, 3, 4]
        conv.weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], out_c, in_c * k * k);

        // Input: [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        let x = tape.input(Tensor::from_vec(vec![1.0; 9], 1, 9));

        let y = conv.forward(&tape, x);
        let val = tape.value(y.idx());

        // Each output element is sum of kernel weights = 1+2+3+4 = 10
        for &v in val.data.iter()
        {
            assert!(
                (v - 10.0).abs() < 1e-5,
                "Forward value mismatch: got {}, expected 10.0",
                v
            );
        }
        assert_eq!(val.data.len(), 4); // (3-2+1)^2 = 4
    }
}
