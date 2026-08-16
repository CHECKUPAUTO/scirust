use super::plan::{Instr, Plan, PwOp};
use crate::autodiff::reverse::Tensor;
use crate::error::{Result, SciRustError};
use std::collections::HashMap;

impl Plan {
    /// Executes a compiled plan without dynamic feeds.
    ///
    /// This is the fallible counterpart of [`Plan::execute`]. It returns an
    /// error when the plan requires a named feed instead of panicking at the
    /// public API boundary.
    pub fn try_execute(&self) -> Result<Tensor> {
        self.try_execute_with(&[])
    }

    /// Executes a compiled plan with dynamic named feeds.
    ///
    /// Missing feeds are reported as [`SciRustError::InvalidConfig`]. A feed
    /// whose shape differs from the shape recorded at compilation is reported
    /// as [`SciRustError::ShapeMismatch`].
    ///
    /// The historical [`Plan::execute_with`] method remains available for
    /// compatibility; callers that process external or otherwise fallible input
    /// should prefer this method.
    pub fn try_execute_with(&self, feeds: &[(&str, Tensor)]) -> Result<Tensor> {
        let feed_map: HashMap<&str, &Tensor> = feeds.iter().map(|(k, v)| (*k, v)).collect();
        let mut buffers: Vec<Option<Tensor>> = vec![None; self.n_buffers];

        for instr in &self.instructions
        {
            match instr
            {
                Instr::LoadConst { output_buf, value } =>
                {
                    buffers[*output_buf] = Some(value.clone());
                },
                Instr::LoadFeed {
                    output_buf,
                    feed_name,
                    expected_shape,
                } =>
                {
                    let t = feed_map.get(feed_name.as_str()).ok_or_else(|| {
                        SciRustError::InvalidConfig(format!("missing lazy feed '{feed_name}'"))
                    })?;
                    let got = t.shape();
                    if got != *expected_shape
                    {
                        return Err(SciRustError::ShapeMismatch {
                            op: "Plan::try_execute_with",
                            expected: *expected_shape,
                            got,
                        });
                    }
                    buffers[*output_buf] = Some((*t).clone());
                },
                Instr::PointwiseChain {
                    output_buf,
                    input_bufs,
                    ops,
                    shape,
                } =>
                {
                    let result = run_pointwise_chain(&buffers, input_bufs, ops, *shape);
                    buffers[*output_buf] = Some(result);
                },
                Instr::MatMul {
                    output_buf,
                    a_buf,
                    b_buf,
                    m,
                    k,
                    n,
                } =>
                {
                    let a = buffers[*a_buf].as_ref().expect("buffer a non chargé");
                    let b = buffers[*b_buf].as_ref().expect("buffer b non chargé");
                    let mut out = Tensor::zeros(*m, *n);
                    for i in 0..*m
                    {
                        for j in 0..*n
                        {
                            let mut acc = 0.0f32;
                            for p in 0..*k
                            {
                                acc += a.data[i * k + p] * b.data[p * n + j];
                            }
                            out.data[i * n + j] = acc;
                        }
                    }
                    buffers[*output_buf] = Some(out);
                },
            }
        }

        Ok(buffers[self.output_buf]
            .take()
            .expect("buffer de sortie absent"))
    }
}

fn run_pointwise_chain(
    buffers: &[Option<Tensor>],
    input_bufs: &[usize],
    ops: &[PwOp],
    shape: (usize, usize),
) -> Tensor {
    let n = shape.0 * shape.1;
    let mut acc = vec![0.0f32; n];
    let inputs: Vec<&Tensor> = input_bufs
        .iter()
        .map(|b| buffers[*b].as_ref().expect("input non chargé"))
        .collect();

    for (i, slot) in acc.iter_mut().enumerate().take(n)
    {
        let mut a = 0.0f32;
        for op in ops
        {
            match op
            {
                PwOp::LoadInput(k) => a = inputs[*k].data[i],
                PwOp::Add(k) => a += inputs[*k].data[i],
                PwOp::Sub(k) => a -= inputs[*k].data[i],
                PwOp::Mul(k) => a *= inputs[*k].data[i],
                PwOp::Scale(s) => a *= s,
                PwOp::Relu => a = a.max(0.0),
                PwOp::Exp => a = a.exp(),
                PwOp::Log => a = a.max(1e-12).ln(),
            }
        }
        *slot = a;
    }

    Tensor::from_vec(acc, shape.0, shape.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{Compiler, LazyGraph, LazyTensor};

    #[test]
    fn try_execute_with_accepts_valid_feed() {
        let graph = LazyGraph::new();
        let x = LazyTensor::feed(graph.clone(), "x".into(), (1, 3));
        let y = x.scale(10.0).relu();
        let plan = Compiler::new(&graph).compile(y.id);

        let result = plan
            .try_execute_with(&[("x", Tensor::from_vec(vec![-1.0, 2.0, -3.0], 1, 3))])
            .unwrap();
        assert_eq!(result.data, vec![0.0, 20.0, 0.0]);
    }

    #[test]
    fn try_execute_reports_missing_feed() {
        let graph = LazyGraph::new();
        let x = LazyTensor::feed(graph.clone(), "x".into(), (1, 3));
        let plan = Compiler::new(&graph).compile(x.id);

        let err = plan.try_execute().unwrap_err();
        match err
        {
            SciRustError::InvalidConfig(message) =>
            {
                assert!(message.contains("missing lazy feed 'x'"));
            },
            other => panic!("expected InvalidConfig, got {other}"),
        }
    }

    #[test]
    fn try_execute_with_reports_feed_shape_mismatch() {
        let graph = LazyGraph::new();
        let x = LazyTensor::feed(graph.clone(), "x".into(), (1, 3));
        let plan = Compiler::new(&graph).compile(x.id);

        let err = plan
            .try_execute_with(&[("x", Tensor::from_vec(vec![1.0, 2.0], 1, 2))])
            .unwrap_err();
        match err
        {
            SciRustError::ShapeMismatch { op, expected, got } =>
            {
                assert_eq!(op, "Plan::try_execute_with");
                assert_eq!(expected, (1, 3));
                assert_eq!(got, (1, 2));
            },
            other => panic!("expected ShapeMismatch, got {other}"),
        }
    }
}
