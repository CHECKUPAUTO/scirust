use scirust_core::autodiff::nd::{NdTape, NdVar};
use scirust_core::tensor::tensor_nd::TensorND;

use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct HamiltonianDynamicsConfig {
    pub ndim: usize,
    pub epsilon: f32,
}

impl Default for HamiltonianDynamicsConfig {
    fn default() -> Self {
        Self {
            ndim: 1,
            epsilon: 1e-4,
        }
    }
}

pub struct HamiltonianDynamics<F> {
    pub hamiltonian: F,
    pub config: HamiltonianDynamicsConfig,
}

impl<F> HamiltonianDynamics<F>
where
    F: for<'a> Fn(&'a NdTape, &'a [NdVar<'a>], &'a [NdVar<'a>], Option<NdVar<'a>>) -> NdVar<'a>,
{
    pub fn new(hamiltonian: F, config: HamiltonianDynamicsConfig) -> Self {
        Self {
            hamiltonian,
            config,
        }
    }

    pub fn vector_field(&self, q: &[f32], p: &[f32], t: f32) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = self.config.ndim;
        if q.len() != n || p.len() != n
        {
            return Err(VariationalError::DimensionMismatch {
                expected: n,
                got: if q.len() != n { q.len() } else { p.len() },
                context: "HamiltonianDynamics::vector_field".into(),
            });
        }

        let tape = NdTape::new();
        let q_vars: Vec<NdVar<'_>> = (0..n)
            .map(|i| tape.input(TensorND::new(vec![q[i]], vec![1, 1])))
            .collect();
        let p_vars: Vec<NdVar<'_>> = (0..n)
            .map(|i| tape.input(TensorND::new(vec![p[i]], vec![1, 1])))
            .collect();
        let tv = tape.input(TensorND::new(vec![t], vec![1, 1]));

        let H = (self.hamiltonian)(&tape, &q_vars, &p_vars, Some(tv));
        let grads = tape.backward(H);

        let dH_dq: Vec<f32> = grads[..n].iter().map(|g| g.data[0]).collect();
        let dH_dp: Vec<f32> = grads[n..2 * n].iter().map(|g| g.data[0]).collect();

        for &v in dH_dq.iter().chain(dH_dp.iter())
        {
            if !v.is_finite()
            {
                return Err(VariationalError::NonFiniteValue {
                    component: "Hamiltonian gradient",
                    value: v,
                });
            }
        }

        let dq = dH_dp;
        let dp: Vec<f32> = dH_dq.iter().map(|&x| -x).collect();

        Ok((dq, dp))
    }

    pub fn dynamics(&self, t: f32, state: &[f32], deriv: &mut [f32]) -> Result<()> {
        let n = self.config.ndim;
        if state.len() != 2 * n
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 2 * n,
                got: state.len(),
                context: "HamiltonianDynamics::dynamics".into(),
            });
        }

        let q = &state[..n];
        let p = &state[n..];
        let (dq, dp) = self.vector_field(q, p, t)?;

        deriv[..n].copy_from_slice(&dq);
        deriv[n..].copy_from_slice(&dp);

        Ok(())
    }
}

impl<F> std::fmt::Debug for HamiltonianDynamics<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HamiltonianDynamics")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic_hamiltonian<'a>(
        tape: &'a NdTape,
        _q: &'a [NdVar<'a>],
        p: &'a [NdVar<'a>],
        _t: Option<NdVar<'a>>,
    ) -> NdVar<'a> {
        let half = tape.input(TensorND::new(vec![0.5], vec![1, 1]));
        let ke = p[0].mul(p[0]).mul(half);
        ke.add(half)
    }

    #[test]
    fn test_harmonic_vector_field() {
        let config = HamiltonianDynamicsConfig {
            ndim: 1,
            epsilon: 1e-4,
        };
        let hd = HamiltonianDynamics::new(harmonic_hamiltonian, config);
        let result = hd.vector_field(&[1.0], &[0.5], 0.0).unwrap();
        assert_eq!(result.0.len(), 1);
        assert_eq!(result.1.len(), 1);
        assert!(
            (result.0[0] - 0.5).abs() < 0.1,
            "dq = {}, expected ~0.5",
            result.0[0]
        );
    }

    #[test]
    fn test_canonical_sign_convention() {
        fn wrong_sign_hamiltonian<'a>(
            tape: &'a NdTape,
            q: &'a [NdVar<'a>],
            _p: &'a [NdVar<'a>],
            _t: Option<NdVar<'a>>,
        ) -> NdVar<'a> {
            let half = tape.input(TensorND::new(vec![0.5], vec![1, 1]));
            q[0].mul(q[0]).mul(half)
        }

        let config = HamiltonianDynamicsConfig {
            ndim: 1,
            epsilon: 1e-4,
        };
        let hd = HamiltonianDynamics::new(wrong_sign_hamiltonian, config);
        let result = hd.vector_field(&[1.0], &[0.0], 0.0).unwrap();
        assert!(
            (result.0[0]).abs() < 0.1,
            "dq should be ~0 for this H (no p dependence)"
        );
        assert!(
            (result.1[0] + 1.0).abs() < 0.1,
            "dp should be -1 for H = 0.5*q^2, got {}",
            result.1[0]
        );
    }
}
