//! Multi-operand contraction planning.
//!
//! Given an einsum expression over several operands, [`ContractionPlan`] chooses
//! a **greedy pairwise contraction order** (each step contracts the pair whose
//! result has the fewest elements) and executes it with the `scirust-tensor-einsum`
//! engine. Pairwise contraction is far cheaper than the naive all-at-once
//! summation for chains of three or more tensors.

use scirust_tensor_core::TensorND;
use scirust_tensor_einsum::einsum;
use std::collections::BTreeMap;

pub struct ContractionPlan {
    pub operand_specs: Vec<Vec<char>>,
    pub output: Vec<char>,
}

impl ContractionPlan {
    /// Parse an einsum pattern such as `"ij,jk,kl->il"`.
    pub fn new(pattern: &str) -> Result<Self, String> {
        let pattern: String = pattern.chars().filter(|c| !c.is_whitespace()).collect();
        let (lhs, rhs) = pattern
            .split_once("->")
            .ok_or_else(|| "contraction pattern must contain '->'".to_string())?;
        let operand_specs = lhs.split(',').map(|s| s.chars().collect()).collect();
        Ok(Self {
            operand_specs,
            output: rhs.chars().collect(),
        })
    }

    fn sizes(&self, inputs: &[&TensorND]) -> BTreeMap<char, usize> {
        let mut sizes = BTreeMap::new();
        for (spec, t) in self.operand_specs.iter().zip(inputs)
        {
            for (p, &lab) in spec.iter().enumerate()
            {
                sizes.insert(lab, t.shape[p]);
            }
        }
        sizes
    }

    /// Execute the contraction, returning the result tensor.
    pub fn execute(&self, inputs: &[&TensorND]) -> Result<TensorND, String> {
        if inputs.len() != self.operand_specs.len()
        {
            return Err(format!(
                "plan expects {} operands, got {}",
                self.operand_specs.len(),
                inputs.len()
            ));
        }
        let sizes = self.sizes(inputs);
        let mut work: Vec<(Vec<char>, TensorND)> = self
            .operand_specs
            .iter()
            .zip(inputs)
            .map(|(s, t)| (s.clone(), (*t).clone()))
            .collect();

        while work.len() > 1
        {
            // Pick the pair whose contraction yields the smallest result.
            let (mut bi, mut bj, mut best_cost) = (0usize, 1usize, usize::MAX);
            for i in 0..work.len()
            {
                for j in (i + 1)..work.len()
                {
                    let res = result_labels(&work[i].0, &work[j].0, &work, i, j, &self.output);
                    let cost: usize = res.iter().map(|l| sizes[l]).product::<usize>().max(1);
                    if cost < best_cost
                    {
                        best_cost = cost;
                        bi = i;
                        bj = j;
                    }
                }
            }
            // Remove the higher index first to keep the lower one valid.
            let (sb, tb) = work.remove(bj);
            let (sa, ta) = work.remove(bi);
            let res = result_labels(&sa, &sb, &work, usize::MAX, usize::MAX, &self.output);
            let pat = format!(
                "{},{}->{}",
                sa.iter().collect::<String>(),
                sb.iter().collect::<String>(),
                res.iter().collect::<String>()
            );
            let r = einsum(&pat, &[&ta, &tb])?;
            work.push((res, r));
        }

        let (final_spec, final_t) = work.pop().ok_or("empty contraction")?;
        if final_spec == self.output
        {
            Ok(final_t)
        }
        else
        {
            // Reorder/reduce to the requested output layout.
            let pat = format!(
                "{}->{}",
                final_spec.iter().collect::<String>(),
                self.output.iter().collect::<String>()
            );
            einsum(&pat, &[&final_t])
        }
    }
}

/// Labels kept after contracting `sa` with `sb`: those appearing in the output
/// or still needed by another remaining operand.
fn result_labels(
    sa: &[char],
    sb: &[char],
    work: &[(Vec<char>, TensorND)],
    skip_i: usize,
    skip_j: usize,
    output: &[char],
) -> Vec<char> {
    let mut union: Vec<char> = Vec::new();
    for &c in sa.iter().chain(sb.iter())
    {
        if !union.contains(&c)
        {
            union.push(c);
        }
    }
    union
        .into_iter()
        .filter(|&c| {
            output.contains(&c)
                || work
                    .iter()
                    .enumerate()
                    .any(|(k, (spec, _))| k != skip_i && k != skip_j && spec.contains(&c))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_matrix_chain_matches_sequential() {
        // A(2x3) B(3x4) C(4x2) -> (2x2)
        let a = TensorND::new((1..=6).map(|v| v as f32).collect(), vec![2, 3]);
        let b = TensorND::new((1..=12).map(|v| v as f32).collect(), vec![3, 4]);
        let c = TensorND::new((1..=8).map(|v| v as f32).collect(), vec![4, 2]);
        let plan = ContractionPlan::new("ij,jk,kl->il").unwrap();
        let out = plan.execute(&[&a, &b, &c]).unwrap();
        assert_eq!(out.shape, vec![2, 2]);

        // Reference: ((A·B)·C)
        let ab = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        let abc = einsum("ik,kl->il", &[&ab, &c]).unwrap();
        assert_eq!(out.data, abc.data);
    }
}
