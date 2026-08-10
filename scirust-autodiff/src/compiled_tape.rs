//! Prepared reverse-mode tape with fixed-arity operations.
//!
//! Graph construction may allocate once. After [`CompiledTapeBuilder::compile`],
//! forward and reverse evaluation reuse fixed `values`/`gradients` buffers and
//! create no per-node dependency vectors, reachability vectors, or cloned edge
//! lists on the hot path.

/// Stable index of one value in a compiled tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TapeSlot(usize);

impl TapeSlot {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Input { input: usize },
    Constant(f64),
    Add(TapeSlot, TapeSlot),
    Sub(TapeSlot, TapeSlot),
    Mul(TapeSlot, TapeSlot),
    Div(TapeSlot, TapeSlot),
    Neg(TapeSlot),
    Sin(TapeSlot),
    Cos(TapeSlot),
    Exp(TapeSlot),
    PowI(TapeSlot, i32),
}

/// Builder used only while tracing/preparing a graph.
#[derive(Debug, Default)]
pub struct CompiledTapeBuilder {
    ops: Vec<Op>,
    input_count: usize,
}

impl CompiledTapeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one external scalar input.
    pub fn input(&mut self) -> TapeSlot {
        let input = self.input_count;
        self.input_count += 1;
        self.push(Op::Input { input })
    }

    pub fn constant(&mut self, value: f64) -> TapeSlot {
        self.push(Op::Constant(value))
    }

    pub fn add(&mut self, left: TapeSlot, right: TapeSlot) -> TapeSlot {
        self.assert_existing(left);
        self.assert_existing(right);
        self.push(Op::Add(left, right))
    }

    pub fn sub(&mut self, left: TapeSlot, right: TapeSlot) -> TapeSlot {
        self.assert_existing(left);
        self.assert_existing(right);
        self.push(Op::Sub(left, right))
    }

    pub fn mul(&mut self, left: TapeSlot, right: TapeSlot) -> TapeSlot {
        self.assert_existing(left);
        self.assert_existing(right);
        self.push(Op::Mul(left, right))
    }

    pub fn div(&mut self, left: TapeSlot, right: TapeSlot) -> TapeSlot {
        self.assert_existing(left);
        self.assert_existing(right);
        self.push(Op::Div(left, right))
    }

    pub fn neg(&mut self, value: TapeSlot) -> TapeSlot {
        self.assert_existing(value);
        self.push(Op::Neg(value))
    }

    pub fn sin(&mut self, value: TapeSlot) -> TapeSlot {
        self.assert_existing(value);
        self.push(Op::Sin(value))
    }

    pub fn cos(&mut self, value: TapeSlot) -> TapeSlot {
        self.assert_existing(value);
        self.push(Op::Cos(value))
    }

    pub fn exp(&mut self, value: TapeSlot) -> TapeSlot {
        self.assert_existing(value);
        self.push(Op::Exp(value))
    }

    pub fn powi(&mut self, value: TapeSlot, exponent: i32) -> TapeSlot {
        self.assert_existing(value);
        self.push(Op::PowI(value, exponent))
    }

    /// Freeze the operation list and allocate reusable primal/adjoint buffers.
    pub fn compile(self, output: TapeSlot) -> CompiledTape {
        assert!(
            output.0 < self.ops.len(),
            "compiled tape output slot {} is outside {} operations",
            output.0,
            self.ops.len()
        );
        let len = self.ops.len();
        CompiledTape {
            ops: self.ops,
            input_count: self.input_count,
            output,
            values: vec![0.0; len],
            gradients: vec![0.0; len],
        }
    }

    fn push(&mut self, op: Op) -> TapeSlot {
        let slot = TapeSlot(self.ops.len());
        self.ops.push(op);
        slot
    }

    fn assert_existing(&self, slot: TapeSlot) {
        assert!(
            slot.0 < self.ops.len(),
            "tape slot {} must refer to an operation already emitted",
            slot.0
        );
    }
}

/// Prepared reverse-mode program reusable across input values.
#[derive(Debug)]
pub struct CompiledTape {
    ops: Vec<Op>,
    input_count: usize,
    output: TapeSlot,
    values: Vec<f64>,
    gradients: Vec<f64>,
}

impl CompiledTape {
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn operation_count(&self) -> usize {
        self.ops.len()
    }

    pub fn output_slot(&self) -> TapeSlot {
        self.output
    }

    /// Addresses of the two reusable execution buffers, useful for allocation
    /// regression tests and profilers.
    pub fn buffer_identities(&self) -> (usize, usize) {
        (self.values.as_ptr() as usize, self.gradients.as_ptr() as usize)
    }

    pub fn buffer_capacities(&self) -> (usize, usize) {
        (self.values.capacity(), self.gradients.capacity())
    }

    /// Evaluate the primal graph into the preallocated value buffer.
    pub fn forward(&mut self, inputs: &[f64]) -> f64 {
        assert_eq!(
            inputs.len(),
            self.input_count,
            "compiled tape input count mismatch"
        );

        for index in 0..self.ops.len()
        {
            self.values[index] = match self.ops[index] {
                Op::Input { input } => inputs[input],
                Op::Constant(value) => value,
                Op::Add(left, right) => self.values[left.0] + self.values[right.0],
                Op::Sub(left, right) => self.values[left.0] - self.values[right.0],
                Op::Mul(left, right) => self.values[left.0] * self.values[right.0],
                Op::Div(left, right) => self.values[left.0] / self.values[right.0],
                Op::Neg(value) => -self.values[value.0],
                Op::Sin(value) => self.values[value.0].sin(),
                Op::Cos(value) => self.values[value.0].cos(),
                Op::Exp(value) => self.values[value.0].exp(),
                Op::PowI(value, exponent) => self.values[value.0].powi(exponent),
            };
        }
        self.values[self.output.0]
    }

    /// Reverse the already-evaluated graph into caller-owned input-gradient storage.
    ///
    /// The method allocates nothing. A zero output adjoint skips an operation
    /// completely; this preserves the legacy tape invariant that dead singular
    /// subgraphs cannot turn `0 * NaN/Inf` into an unrelated NaN gradient.
    pub fn backward_into(&mut self, output_seed: f64, input_gradients: &mut [f64]) {
        assert_eq!(
            input_gradients.len(),
            self.input_count,
            "compiled tape gradient count mismatch"
        );
        input_gradients.fill(0.0);
        self.gradients.fill(0.0);
        self.gradients[self.output.0] = output_seed;

        for index in (0..self.ops.len()).rev()
        {
            let grad = self.gradients[index];
            if grad == 0.0
            {
                continue;
            }

            match self.ops[index] {
                Op::Input { input } => input_gradients[input] += grad,
                Op::Constant(_) => {},
                Op::Add(left, right) => {
                    self.gradients[left.0] += grad;
                    self.gradients[right.0] += grad;
                },
                Op::Sub(left, right) => {
                    self.gradients[left.0] += grad;
                    self.gradients[right.0] -= grad;
                },
                Op::Mul(left, right) => {
                    self.gradients[left.0] += grad * self.values[right.0];
                    self.gradients[right.0] += grad * self.values[left.0];
                },
                Op::Div(left, right) => {
                    let denominator = self.values[right.0];
                    self.gradients[left.0] += grad / denominator;
                    self.gradients[right.0] -=
                        grad * self.values[left.0] / (denominator * denominator);
                },
                Op::Neg(value) => self.gradients[value.0] -= grad,
                Op::Sin(value) => {
                    self.gradients[value.0] += grad * self.values[value.0].cos();
                },
                Op::Cos(value) => {
                    self.gradients[value.0] -= grad * self.values[value.0].sin();
                },
                Op::Exp(value) => {
                    self.gradients[value.0] += grad * self.values[index];
                },
                Op::PowI(value, exponent) => {
                    if exponent != 0
                    {
                        self.gradients[value.0] += grad
                            * exponent as f64
                            * self.values[value.0].powi(exponent - 1);
                    }
                },
            }
        }
    }

    /// Convenience for one full primal+reverse evaluation while still reusing
    /// all internal storage.
    pub fn value_and_gradient_into(
        &mut self,
        inputs: &[f64],
        input_gradients: &mut [f64],
    ) -> f64 {
        let value = self.forward(inputs);
        self.backward_into(1.0, input_gradients);
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_tape_matches_analytic_gradient() {
        // z = x² + x*y ; dz/dx = 2x+y, dz/dy=x.
        let mut builder = CompiledTapeBuilder::new();
        let x = builder.input();
        let y = builder.input();
        let x2 = builder.mul(x, x);
        let xy = builder.mul(x, y);
        let z = builder.add(x2, xy);
        let mut tape = builder.compile(z);

        let mut gradient = [0.0; 2];
        let value = tape.value_and_gradient_into(&[3.0, 2.0], &mut gradient);
        assert_eq!(value, 15.0);
        assert_eq!(gradient, [8.0, 3.0]);
    }

    #[test]
    fn repeated_execution_keeps_same_buffers() {
        let mut builder = CompiledTapeBuilder::new();
        let x = builder.input();
        let s = builder.sin(x);
        let e = builder.exp(x);
        let out = builder.mul(s, e);
        let mut tape = builder.compile(out);
        let identities = tape.buffer_identities();
        let capacities = tape.buffer_capacities();
        let mut gradient = [0.0; 1];

        for &input in &[0.25, 0.5, 1.0, 2.0]
        {
            let value = tape.value_and_gradient_into(&[input], &mut gradient);
            let expected_value = input.sin() * input.exp();
            let expected_grad = input.exp() * (input.sin() + input.cos());
            assert!((value - expected_value).abs() < 1e-12);
            assert!((gradient[0] - expected_grad).abs() < 1e-12);
            assert_eq!(tape.buffer_identities(), identities);
            assert_eq!(tape.buffer_capacities(), capacities);
        }
    }

    #[test]
    fn dead_singular_subgraph_does_not_poison_output_gradient() {
        let mut builder = CompiledTapeBuilder::new();
        let x = builder.input();
        let zero = builder.constant(0.0);
        let one = builder.constant(1.0);
        let _dead = builder.div(one, zero);
        let output = builder.add(x, x);
        let mut tape = builder.compile(output);
        let mut gradient = [0.0; 1];
        let value = tape.value_and_gradient_into(&[4.0], &mut gradient);
        assert_eq!(value, 8.0);
        assert_eq!(gradient[0], 2.0);
        assert!(gradient[0].is_finite());
    }

    #[test]
    fn powi_zero_at_zero_has_zero_gradient() {
        let mut builder = CompiledTapeBuilder::new();
        let x = builder.input();
        let out = builder.powi(x, 0);
        let mut tape = builder.compile(out);
        let mut gradient = [f64::NAN; 1];
        let value = tape.value_and_gradient_into(&[0.0], &mut gradient);
        assert_eq!(value, 1.0);
        assert_eq!(gradient[0], 0.0);
    }
}
