// scirust-core/src/tests/test_xai.rs
#[cfg(test)]
mod tests {
    use crate::autodiff::reverse::Tensor;
    use crate::xai::integrated_gradients;

    #[test]
    fn test_integrated_gradients_linear() {
        // f(x) = 2*x. Integrated Gradients should be 2*(x - baseline)
        let input = Tensor::from_vec(vec![1.0], 1, 1);
        let baseline = Tensor::from_vec(vec![0.0], 1, 1);
        let steps = 10;

        let attributions = integrated_gradients(&input, &baseline, steps, |x| x.scale(2.0));

        // Expected attribution: 2.0 * (1.0 - 0.0) = 2.0
        assert!((attributions.data[0] - 2.0).abs() < 1e-5);
    }
}
