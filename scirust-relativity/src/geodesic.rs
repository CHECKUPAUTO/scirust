//! Geodesic equations as SciRust dynamical systems.

use crate::Connection;
use scirust_sim::System;

/// First-order representation of the geodesic equation.
///
/// The state is laid out as:
///
/// `y = [x^0, ..., x^(D-1), u^0, ..., u^(D-1)]`,
///
/// where `u^mu = dx^mu / d lambda`.
#[derive(Debug, Clone)]
pub struct GeodesicSystem<C, const D: usize> {
    connection: C,
}

impl<C, const D: usize> GeodesicSystem<C, D> {
    /// Construct a geodesic system from a connection provider.
    #[must_use]
    pub const fn new(connection: C) -> Self {
        Self { connection }
    }

    /// Borrow the underlying connection provider.
    #[must_use]
    pub const fn connection(&self) -> &C {
        &self.connection
    }
}

impl<C, const D: usize> System for GeodesicSystem<C, D>
where
    C: Connection<D>,
{
    fn dim(&self) -> usize {
        2 * D
    }

    fn derivatives(&self, _parameter: f64, state: &[f64], output: &mut [f64]) {
        debug_assert_eq!(state.len(), 2 * D);
        debug_assert_eq!(output.len(), 2 * D);

        let mut coordinates = [0.0_f64; D];
        let mut velocity = [0.0_f64; D];

        coordinates.copy_from_slice(&state[..D]);
        velocity.copy_from_slice(&state[D..]);

        output[..D].copy_from_slice(&velocity);

        let symbols = self.connection.christoffel(&coordinates);

        for rho in 0..D
        {
            let mut acceleration = 0.0;

            for mu in 0..D
            {
                for nu in 0..D
                {
                    acceleration -= symbols[rho][mu][nu] * velocity[mu] * velocity[nu];
                }
            }

            output[D + rho] = acceleration;
        }
    }
}
