use super::plan::{Instr, Plan};
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
    /// as [`SciRustError::ShapeMismatch`]. After validating caller-controlled
    /// feed input, execution is delegated to [`Plan::execute_with`] so there is
    /// only one numerical execution implementation to maintain.
    ///
    /// The historical [`Plan::execute_with`] method remains available for
    /// compatibility; callers that process external or otherwise fallible input
    /// should prefer this method.
    pub fn try_execute_with(&self, feeds: &[(&str, Tensor)]) -> Result<Tensor> {
        let feed_map: HashMap<&str, &Tensor> = feeds.iter().map(|(k, v)| (*k, v)).collect();

        for instr in &self.instructions
        {
            let Instr::LoadFeed {
                feed_name,
                expected_shape,
                ..
            } = instr
            else
            {
                continue;
            };

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
        }

        Ok(self.execute_with(feeds))
    }
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
