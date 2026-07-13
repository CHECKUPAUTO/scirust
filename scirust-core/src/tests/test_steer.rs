// scirust-core/src/tests/test_steer.rs
#[cfg(test)]
mod tests {
    use crate::autodiff::reverse::{Tape, Tensor, Var};
    use crate::nn::module::{Module, SteerHook};

    struct MockLayer {
        name: String,
        weight_idx: usize,
    }

    impl Module for MockLayer {
        fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
            let w = Var::new(tape, self.weight_idx);
            input.matmul(w)
        }

        fn forward_steered<'t>(
            &mut self,
            tape: &'t Tape,
            input: Var<'t>,
            hook: Option<&SteerHook>,
        ) -> Var<'t> {
            let mut out = self.forward(tape, input);
            if let Some(h) = hook
                && h.target_layer == self.name
            {
                let shift_var = tape.input(h.shift.clone());
                out = out.add_broadcast(shift_var);
            }
            out
        }

        fn parameter_indices(&self) -> Vec<usize> {
            vec![self.weight_idx]
        }
        fn sync(&mut self, _tape: &Tape) {}
    }

    #[test]
    fn test_latent_steering() {
        let tape = Tape::new();
        let w_t = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let w_idx = tape.input(w_t).idx();

        let mut layer = MockLayer {
            name: "layer1".to_string(),
            weight_idx: w_idx,
        };

        let input = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));

        // No steering
        let out_normal = layer.forward_steered(&tape, input, None);
        assert_eq!(tape.value(out_normal.idx()).data, vec![1.0, 2.0]);

        // With steering
        let shift = Tensor::from_vec(vec![10.0, 20.0], 1, 2);
        let hook = SteerHook {
            target_layer: "layer1".to_string(),
            shift: shift.clone(),
        };
        let out_steered = layer.forward_steered(&tape, input, Some(&hook));
        assert_eq!(tape.value(out_steered.idx()).data, vec![11.0, 22.0]);
    }
}
