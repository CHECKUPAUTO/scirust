use scirust_core::autodiff::reverse::{Tape, Var};
use scirust_core::nn::init::Initializer;
use scirust_core::nn::linear::Linear;
use scirust_core::nn::module::Module;
use scirust_core::nn::rng::PcgEngine;

/// N-BEATS basic block.
pub struct NBeatsBlock {
    pub fc_stack: Vec<Linear>,
    pub backcast_head: Linear,
    pub forecast_head: Linear,
}

impl NBeatsBlock {
    pub fn new<W: Initializer, B: Initializer>(
        input_dim: usize,
        hidden_dim: usize,
        theta_dim: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let mut fc_stack = Vec::new();
        fc_stack.push(Linear::new(input_dim, hidden_dim, w_init, b_init, rng));
        for _ in 0..3
        {
            fc_stack.push(Linear::new(hidden_dim, hidden_dim, w_init, b_init, rng));
        }

        let backcast_head = Linear::new(hidden_dim, input_dim, w_init, b_init, rng);
        let forecast_head = Linear::new(hidden_dim, theta_dim, w_init, b_init, rng);

        Self {
            fc_stack,
            backcast_head,
            forecast_head,
        }
    }

    pub fn forward_both<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> (Var<'t>, Var<'t>) {
        let mut h = input;
        for fc in &mut self.fc_stack
        {
            h = fc.forward(tape, h).relu();
        }
        let backcast = self.backcast_head.forward(tape, h.clone());
        let forecast = self.forecast_head.forward(tape, h);
        (backcast, forecast)
    }
}

impl Module for NBeatsBlock {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let mut h = input;
        for fc in &mut self.fc_stack
        {
            h = fc.forward(tape, h).relu();
        }

        // Return forecast
        self.forecast_head.forward(tape, h)
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        for fc in &self.fc_stack
        {
            v.extend(fc.parameter_indices());
        }
        v.extend(self.backcast_head.parameter_indices());
        v.extend(self.forecast_head.parameter_indices());
        v
    }

    fn sync(&mut self, tape: &Tape) {
        for fc in &mut self.fc_stack
        {
            fc.sync(tape);
        }
        self.backcast_head.sync(tape);
        self.forecast_head.sync(tape);
    }
}
