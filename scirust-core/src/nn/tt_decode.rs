//! Allocation-free batch-one direct contraction for two-core TT linear layers.
//!
//! [`super::TTLinear`] is the general differentiable Tensor-Train layer. Its current
//! autodiff path reconstructs the represented dense matrix before matmul, which is a
//! useful oracle but defeats the weight-bandwidth advantage during autoregressive
//! batch-one decode.
//!
//! This module evaluates the common `d = 2` TT case directly from its cores:
//!
//! ```text
//! W[(i0,i1),(o0,o1)] = sum_r G0[(i0,o0),r] * G1[r,(i1,o1)]
//! ```
//!
//! The contraction is split into one reusable caller-provided scratch tensor and a
//! final output contraction. No dense weight is materialized, no heap allocation
//! occurs in [`TwoCoreTtDecodePlan::forward_into`], and every reduction has a fixed
//! left-to-right order suitable for a CUDA differential oracle.

use core::fmt;

use crate::autodiff::reverse::Tensor;

use super::TTLinear;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtDecodeError {
    CoreCount {
        actual: usize,
    },
    DimensionCount {
        input: usize,
        output: usize,
        ranks: usize,
    },
    BoundaryRank {
        first: usize,
        last: usize,
    },
    ZeroDimension(&'static str),
    FeatureProduct {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    CoreShape {
        core: usize,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    BiasShape {
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    InputLength {
        expected: usize,
        actual: usize,
    },
    OutputLength {
        expected: usize,
        actual: usize,
    },
    ScratchLength {
        expected: usize,
        actual: usize,
    },
    Overflow,
}

impl fmt::Display for TtDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::CoreCount { actual } => write!(
                formatter,
                "direct TT decode requires exactly two cores, got {actual}"
            ),
            Self::DimensionCount {
                input,
                output,
                ranks,
            } => write!(
                formatter,
                "two-core TT decode requires 2 input dims, 2 output dims and 3 ranks; got {input}, {output}, {ranks}"
            ),
            Self::BoundaryRank { first, last } => write!(
                formatter,
                "TT boundary ranks must both equal 1, got r0={first}, r2={last}"
            ),
            Self::ZeroDimension(name) => write!(formatter, "TT decode dimension {name} is zero"),
            Self::FeatureProduct {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "TT {name} factor product mismatch: expected {expected}, got {actual}"
            ),
            Self::CoreShape {
                core,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                formatter,
                "TT core {core} shape mismatch: expected {expected_rows}x{expected_cols}, got {actual_rows}x{actual_cols}"
            ),
            Self::BiasShape {
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                formatter,
                "TT decode bias shape mismatch: expected 1x{expected_cols}, got {actual_rows}x{actual_cols}"
            ),
            Self::InputLength { expected, actual } => write!(
                formatter,
                "TT decode input length mismatch: expected {expected}, got {actual}"
            ),
            Self::OutputLength { expected, actual } => write!(
                formatter,
                "TT decode output length mismatch: expected {expected}, got {actual}"
            ),
            Self::ScratchLength { expected, actual } => write!(
                formatter,
                "TT decode scratch length mismatch: expected {expected}, got {actual}"
            ),
            Self::Overflow => write!(formatter, "TT decode size calculation overflow"),
        }
    }
}

impl std::error::Error for TtDecodeError {}

/// Validated view of a two-core [`TTLinear`] for batch-one direct contraction.
#[derive(Debug)]
pub struct TwoCoreTtDecodePlan<'a> {
    core0: &'a Tensor,
    core1: &'a Tensor,
    bias: &'a Tensor,
    in0: usize,
    in1: usize,
    out0: usize,
    out1: usize,
    rank: usize,
    input_features: usize,
    output_features: usize,
    scratch_len: usize,
}

impl<'a> TwoCoreTtDecodePlan<'a> {
    pub fn new(layer: &'a TTLinear) -> Result<Self, TtDecodeError> {
        if layer.cores.len() != 2
        {
            return Err(TtDecodeError::CoreCount {
                actual: layer.cores.len(),
            });
        }
        if layer.in_dims.len() != 2 || layer.out_dims.len() != 2 || layer.ranks.len() != 3
        {
            return Err(TtDecodeError::DimensionCount {
                input: layer.in_dims.len(),
                output: layer.out_dims.len(),
                ranks: layer.ranks.len(),
            });
        }
        if layer.ranks[0] != 1 || layer.ranks[2] != 1
        {
            return Err(TtDecodeError::BoundaryRank {
                first: layer.ranks[0],
                last: layer.ranks[2],
            });
        }

        let in0 = layer.in_dims[0];
        let in1 = layer.in_dims[1];
        let out0 = layer.out_dims[0];
        let out1 = layer.out_dims[1];
        let rank = layer.ranks[1];
        for (name, value) in [
            ("in0", in0),
            ("in1", in1),
            ("out0", out0),
            ("out1", out1),
            ("rank", rank),
        ]
        {
            if value == 0
            {
                return Err(TtDecodeError::ZeroDimension(name));
            }
        }

        let input_features = checked_mul(in0, in1)?;
        if input_features != layer.in_features
        {
            return Err(TtDecodeError::FeatureProduct {
                name: "input",
                expected: layer.in_features,
                actual: input_features,
            });
        }
        let output_features = checked_mul(out0, out1)?;
        if output_features != layer.out_features
        {
            return Err(TtDecodeError::FeatureProduct {
                name: "output",
                expected: layer.out_features,
                actual: output_features,
            });
        }

        let core0 = &layer.cores[0];
        let core1 = &layer.cores[1];
        let expected_core0_rows = checked_mul(in0, out0)?;
        require_core_shape(0, core0, expected_core0_rows, rank)?;
        let expected_core1_rows = checked_mul3(rank, in1, out1)?;
        require_core_shape(1, core1, expected_core1_rows, 1)?;
        if layer.bias.rows != 1 || layer.bias.cols != output_features
        {
            return Err(TtDecodeError::BiasShape {
                expected_cols: output_features,
                actual_rows: layer.bias.rows,
                actual_cols: layer.bias.cols,
            });
        }
        let scratch_len = checked_mul3(out0, rank, in1)?;

        Ok(Self {
            core0,
            core1,
            bias: &layer.bias,
            in0,
            in1,
            out0,
            out1,
            rank,
            input_features,
            output_features,
            scratch_len,
        })
    }

    #[must_use]
    pub const fn input_features(&self) -> usize {
        self.input_features
    }

    #[must_use]
    pub const fn output_features(&self) -> usize {
        self.output_features
    }

    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    #[must_use]
    pub const fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// TT core + bias scalar count read by a direct decode step.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.core0
            .data
            .len()
            .saturating_add(self.core1.data.len())
            .saturating_add(self.bias.data.len())
    }

    /// Dense matrix + bias scalar count represented by this layer.
    #[must_use]
    pub fn dense_parameter_count(&self) -> usize {
        self.input_features
            .saturating_mul(self.output_features)
            .saturating_add(self.output_features)
    }

    /// Number of multiply-accumulate terms in the fixed two-stage contraction.
    /// This is an arithmetic count, not a hardware-latency estimate.
    #[must_use]
    pub fn multiply_accumulate_count(&self) -> usize {
        let stage1 = self
            .in0
            .saturating_mul(self.in1)
            .saturating_mul(self.out0)
            .saturating_mul(self.rank);
        let stage2 = self
            .out0
            .saturating_mul(self.out1)
            .saturating_mul(self.in1)
            .saturating_mul(self.rank);
        stage1.saturating_add(stage2)
    }

    /// Evaluate one row directly from TT cores.
    ///
    /// Scratch layout is `[out0, rank, in1]`. Stage 1 visits input dimensions in
    /// lexicographic `(i1, i0, o0, r)` order; stage 2 reduces `(r, i1)` in fixed
    /// left-to-right order for every output. These orders are part of this decode
    /// oracle and should be preserved by device implementations that claim exact
    /// direct-contraction parity.
    pub fn forward_into(
        &self,
        input: &[f32],
        scratch: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), TtDecodeError> {
        require_length(
            input.len(),
            self.input_features,
            |expected, actual| TtDecodeError::InputLength { expected, actual },
        )?;
        require_length(
            scratch.len(),
            self.scratch_len,
            |expected, actual| TtDecodeError::ScratchLength { expected, actual },
        )?;
        require_length(
            output.len(),
            self.output_features,
            |expected, actual| TtDecodeError::OutputLength { expected, actual },
        )?;

        scratch.fill(0.0);
        for i1 in 0..self.in1
        {
            for i0 in 0..self.in0
            {
                let input_value = input[i0 * self.in1 + i1];
                for out0 in 0..self.out0
                {
                    let core0_base = (i0 * self.out0 + out0) * self.rank;
                    let scratch_base = out0 * self.rank * self.in1 + i1;
                    for rank in 0..self.rank
                    {
                        scratch[scratch_base + rank * self.in1] +=
                            input_value * self.core0.data[core0_base + rank];
                    }
                }
            }
        }

        for out0 in 0..self.out0
        {
            for out1 in 0..self.out1
            {
                let output_index = out0 * self.out1 + out1;
                let mut sum = self.bias.data[output_index];
                for rank in 0..self.rank
                {
                    let scratch_base = (out0 * self.rank + rank) * self.in1;
                    let core1_base = rank * self.in1 * self.out1;
                    for i1 in 0..self.in1
                    {
                        let core1_index = core1_base + i1 * self.out1 + out1;
                        sum += scratch[scratch_base + i1] * self.core1.data[core1_index];
                    }
                }
                output[output_index] = sum;
            }
        }
        Ok(())
    }
}

fn require_core_shape(
    core: usize,
    tensor: &Tensor,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), TtDecodeError> {
    if tensor.rows != expected_rows || tensor.cols != expected_cols
    {
        return Err(TtDecodeError::CoreShape {
            core,
            expected_rows,
            expected_cols,
            actual_rows: tensor.rows,
            actual_cols: tensor.cols,
        });
    }
    Ok(())
}

fn require_length(
    actual: usize,
    expected: usize,
    error: impl FnOnce(usize, usize) -> TtDecodeError,
) -> Result<(), TtDecodeError> {
    if actual != expected
    {
        return Err(error(expected, actual));
    }
    Ok(())
}

fn checked_mul(left: usize, right: usize) -> Result<usize, TtDecodeError> {
    left.checked_mul(right).ok_or(TtDecodeError::Overflow)
}

fn checked_mul3(a: usize, b: usize, c: usize) -> Result<usize, TtDecodeError> {
    a.checked_mul(b)
        .and_then(|value| value.checked_mul(c))
        .ok_or(TtDecodeError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::Zeros;
    use crate::nn::linear::Linear;
    use crate::nn::rng::PcgEngine;
    use crate::tn::reconstruct_matrix;

    fn small_tt() -> TTLinear {
        let zeros = Zeros;
        let mut rng = PcgEngine::new(7);
        let mut dense = Linear::new(4, 6, &zeros, &zeros, &mut rng);
        dense.weight = Tensor::from_vec(
            (0..24)
                .map(|index| (index as f32 - 11.0) * 0.03125)
                .collect(),
            4,
            6,
        );
        dense.bias = Tensor::from_vec(
            (0..6).map(|index| (index as f32 - 2.0) * 0.125).collect(),
            1,
            6,
        );
        TTLinear::from_linear(&dense, vec![2, 2], vec![2, 3], 4, 1.0e-7).unwrap()
    }

    fn dense_reference(layer: &TTLinear, input: &[f32]) -> Vec<f32> {
        let refs: Vec<&Tensor> = layer.cores.iter().collect();
        let weight = reconstruct_matrix(&refs, &layer.in_dims, &layer.out_dims, &layer.ranks);
        let mut output = layer.bias.data.clone();
        for output_col in 0..layer.out_features
        {
            for input_row in 0..layer.in_features
            {
                output[output_col] +=
                    input[input_row] * weight[input_row * layer.out_features + output_col];
            }
        }
        output
    }

    #[test]
    fn direct_two_core_contraction_matches_reconstructed_tt_matrix() {
        let layer = small_tt();
        let plan = TwoCoreTtDecodePlan::new(&layer).unwrap();
        let input = [0.5f32, -1.0, 0.25, 2.0];
        let expected = dense_reference(&layer, &input);
        let mut scratch = vec![0.0f32; plan.scratch_len()];
        let mut actual = vec![0.0f32; plan.output_features()];
        plan.forward_into(&input, &mut scratch, &mut actual).unwrap();
        for (index, (&expected, &actual)) in expected.iter().zip(&actual).enumerate()
        {
            assert!(
                (expected - actual).abs() <= 2.0e-5,
                "output {index}: expected={expected}, actual={actual}"
            );
        }
    }

    #[test]
    fn direct_plan_reports_compression_and_arithmetic_without_claiming_speed() {
        let layer = small_tt();
        let plan = TwoCoreTtDecodePlan::new(&layer).unwrap();
        assert!(plan.parameter_count() < plan.dense_parameter_count());
        assert!(plan.multiply_accumulate_count() > 0);
    }

    #[test]
    fn caller_owned_scratch_is_exactly_sized_and_reusable() {
        let layer = small_tt();
        let plan = TwoCoreTtDecodePlan::new(&layer).unwrap();
        let mut scratch = vec![0.0f32; plan.scratch_len()];
        let mut output = vec![0.0f32; plan.output_features()];
        let first_input = [1.0f32, 2.0, 3.0, 4.0];
        let second_input = [-1.0f32, 0.5, -0.25, 0.125];
        plan.forward_into(&first_input, &mut scratch, &mut output).unwrap();
        let first = output.clone();
        plan.forward_into(&second_input, &mut scratch, &mut output).unwrap();
        let second = output.clone();
        assert_ne!(first, second);
        let expected = dense_reference(&layer, &second_input);
        for (&expected, &actual) in expected.iter().zip(&second)
        {
            assert!((expected - actual).abs() <= 2.0e-5);
        }
    }

    #[test]
    fn malformed_scratch_fails_closed() {
        let layer = small_tt();
        let plan = TwoCoreTtDecodePlan::new(&layer).unwrap();
        let input = [0.0f32; 4];
        let mut scratch = vec![0.0f32; plan.scratch_len() - 1];
        let mut output = vec![0.0f32; plan.output_features()];
        assert!(matches!(
            plan.forward_into(&input, &mut scratch, &mut output),
            Err(TtDecodeError::ScratchLength { .. })
        ));
    }
}