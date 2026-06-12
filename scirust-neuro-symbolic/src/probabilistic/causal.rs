use crate::core::{Reasoner, Result};
use std::collections::{HashMap, HashSet};

/// A linear **structural causal model** (SCM). Each endogenous node has a
/// structural equation `node = intercept + Σ coef·parent`. `intervene` performs
/// Pearl's `do(var = value)`: it severs the variable's incoming equation, fixes
/// its value, and propagates the consequences downstream.
pub struct CausalEngine {
    pub graph_desc: String,
    /// node -> (intercept, [(parent, coefficient)])
    equations: HashMap<String, (f64, Vec<(String, f64)>)>,
}

impl CausalEngine {
    pub fn new(desc: &str) -> Self {
        Self {
            graph_desc: desc.to_string(),
            equations: HashMap::new(),
        }
    }

    /// Define the structural equation for `node`.
    pub fn add_equation(&mut self, node: &str, intercept: f64, parents: Vec<(&str, f64)>) {
        let parents = parents
            .into_iter()
            .map(|(p, c)| (p.to_string(), c))
            .collect();
        self.equations
            .insert(node.to_string(), (intercept, parents));
    }

    fn all_nodes(&self) -> HashSet<String> {
        let mut nodes: HashSet<String> = self.equations.keys().cloned().collect();
        for (_, parents) in self.equations.values()
        {
            for (p, _) in parents
            {
                nodes.insert(p.clone());
            }
        }
        nodes
    }

    /// Evaluate the model, with `overrides` fixing the values of intervened nodes.
    fn solve(&self, overrides: &HashMap<String, f64>) -> HashMap<String, f64> {
        let nodes = self.all_nodes();
        let mut values: HashMap<String, f64> = nodes
            .iter()
            .map(|n| (n.clone(), overrides.get(n).copied().unwrap_or(0.0)))
            .collect();

        // For a DAG, |nodes| relaxation passes propagate all values.
        for _ in 0..=nodes.len()
        {
            for (node, (intercept, parents)) in &self.equations
            {
                if overrides.contains_key(node)
                {
                    continue; // do(node) cuts its structural equation
                }
                let v = intercept
                    + parents
                        .iter()
                        .map(|(p, c)| c * values.get(p).copied().unwrap_or(0.0))
                        .sum::<f64>();
                values.insert(node.clone(), v);
            }
        }
        values
    }

    /// Observational evaluation (no intervention).
    pub fn evaluate(&self) -> HashMap<String, f64> {
        self.solve(&HashMap::new())
    }

    /// `do(variable = value)`: returns the post-intervention node values.
    pub fn intervene(&self, variable: &str, value: f64) -> Result<HashMap<String, f64>> {
        let mut overrides = HashMap::new();
        overrides.insert(variable.to_string(), value);
        Ok(self.solve(&overrides))
    }
}

impl Reasoner for CausalEngine {
    fn name(&self) -> &str {
        "CausalEngine"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> CausalEngine {
        // X (exogenous) -> Z = 2X -> Y = 1 + 3Z
        let mut e = CausalEngine::new("X -> Z -> Y");
        e.add_equation("Z", 0.0, vec![("X", 2.0)]);
        e.add_equation("Y", 1.0, vec![("Z", 3.0)]);
        e
    }

    #[test]
    fn intervention_propagates_downstream() {
        let e = chain();
        let v = e.intervene("X", 5.0).unwrap();
        assert_eq!(v["Z"], 10.0);
        assert_eq!(v["Y"], 31.0);
    }

    #[test]
    fn intervention_cuts_incoming_edge() {
        let e = chain();
        // do(Z = 100) ignores Z's equation; Y = 1 + 3*100 regardless of X.
        let v = e.intervene("Z", 100.0).unwrap();
        assert_eq!(v["Z"], 100.0);
        assert_eq!(v["Y"], 301.0);
    }
}
