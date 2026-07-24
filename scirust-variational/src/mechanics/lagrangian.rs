use scirust_core::autodiff::nd::{NdTape, NdVar};

use crate::error::{Result, VariationalError};
use crate::euler_lagrange::autodiff::AutodiffEulerLagrange;

#[derive(Debug, Clone)]
pub struct LagrangianDynamicsConfig {
    pub ndim: usize,
    pub singularity_tol: f32,
    pub epsilon: f32,
}

impl Default for LagrangianDynamicsConfig {
    fn default() -> Self {
        Self {
            ndim: 1,
            singularity_tol: 1e-8,
            epsilon: 1e-4,
        }
    }
}

pub struct LagrangianDynamics<F> {
    pub lagrangian: F,
    pub config: LagrangianDynamicsConfig,
}

impl<F> LagrangianDynamics<F>
where
    F: for<'a> Fn(&'a NdTape, &'a [NdVar<'a>], &'a [NdVar<'a>], Option<NdVar<'a>>) -> NdVar<'a>,
{
    pub fn new(lagrangian: F, config: LagrangianDynamicsConfig) -> Self {
        Self { lagrangian, config }
    }

    pub fn acceleration(&self, q: &[f32], dq: &[f32], t: f32) -> Result<Vec<f32>> {
        if q.len() != self.config.ndim || dq.len() != self.config.ndim
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.config.ndim,
                got: q.len(),
                context: "LagrangianDynamics::acceleration".into(),
            });
        }

        let el = AutodiffEulerLagrange::new(self.config.ndim)
            .with_tolerances(self.config.epsilon, self.config.singularity_tol);

        let result = el.compute_acceleration(&self.lagrangian, q, dq, Some(t))?;
        Ok(result.acceleration)
    }

    pub fn dynamics(&self, t: f32, state: &[f32], deriv: &mut [f32]) -> Result<()> {
        let n = self.config.ndim;
        if state.len() != 2 * n
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 2 * n,
                got: state.len(),
                context: "LagrangianDynamics::dynamics".into(),
            });
        }

        let q = &state[..n];
        let dq = &state[n..];
        let acc = self.acceleration(q, dq, t)?;

        deriv[..n].copy_from_slice(dq);
        deriv[n..].copy_from_slice(&acc);
        Ok(())
    }

    pub fn compute_ode_rhs(&self, t: f32, state: &[f32], deriv: &mut [f32]) -> Result<()> {
        let n = self.config.ndim;
        if state.len() != 2 * n
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 2 * n,
                got: state.len(),
                context: "LagrangianDynamics::compute_ode_rhs".into(),
            });
        }
        let q = &state[..n];
        let dq = &state[n..];
        let acc = self.acceleration(q, dq, t)?;
        deriv[..n].copy_from_slice(dq);
        deriv[n..].copy_from_slice(&acc);
        Ok(())
    }
}

impl<F> std::fmt::Debug for LagrangianDynamics<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LagrangianDynamics")
            .field("config", &self.config)
            .finish()
    }
}
