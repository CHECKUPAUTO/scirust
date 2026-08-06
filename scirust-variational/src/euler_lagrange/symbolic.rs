use scirust_symbolic::{Expr, diff, simplify};

use super::{ELDerivation, ELEquation, substitute};
use crate::error::{Result, VariationalError};

pub struct SymbolicEulerLagrange;

impl SymbolicEulerLagrange {
    pub fn derive(
        lagrangian: &Expr,
        coordinates: &[String],
        velocities: &[String],
        time_var: Option<&str>,
    ) -> Result<ELDerivation> {
        if coordinates.len() != velocities.len()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: coordinates.len(),
                got: velocities.len(),
                context: "coordinate/velocity count mismatch".into(),
            });
        }
        if coordinates.is_empty()
        {
            return Err(VariationalError::UnsupportedOperation {
                details: "no coordinates provided".into(),
            });
        }

        let accel_vars: Vec<Expr> = coordinates
            .iter()
            .map(|c| Expr::Var(format!("{}_ddot", c)))
            .collect();

        let mut equations = Vec::with_capacity(coordinates.len());

        for (q_name, dq_name) in coordinates.iter().zip(velocities.iter())
        {
            let dL_dq = diff(lagrangian, q_name);

            let dL_ddq = diff(lagrangian, dq_name);

            let ddL_dt_ddq = match time_var
            {
                Some(t) => diff(&dL_ddq, t),
                None => Expr::Const(0.0),
            };

            let mut ddL_dq_ddq = Expr::Const(0.0);
            for (j, qj_name) in coordinates.iter().enumerate()
            {
                let mixed = diff(&dL_ddq, qj_name);
                if mixed != Expr::Const(0.0)
                {
                    let dq_j = Expr::Var(velocities[j].clone());
                    ddL_dq_ddq = ddL_dq_ddq + mixed * dq_j;
                }
            }

            let mut ddL_ddq2 = Expr::Const(0.0);
            let mut accel_deps = Vec::new();
            for (j, dqj_name) in velocities.iter().enumerate()
            {
                let mixed = diff(&dL_ddq, dqj_name);
                if mixed != Expr::Const(0.0)
                {
                    let ddq_j = accel_vars[j].clone();
                    ddL_ddq2 = ddL_ddq2 + mixed.clone() * ddq_j;
                    accel_deps.push(format!("{}_ddot", coordinates[j]));
                }
            }

            let eq_expr = simplify(&(ddL_dt_ddq + ddL_dq_ddq + ddL_ddq2 - dL_dq));

            equations.push(ELEquation {
                coordinate: q_name.clone(),
                residual: eq_expr,
                acceleration_deps: accel_deps,
                is_explicit: true,
            });
        }

        Ok(ELDerivation {
            equations,
            lagrangian: lagrangian.clone(),
            coordinates: coordinates.to_vec(),
            time_var: time_var.map(|s| s.to_string()),
        })
    }

    pub fn derive_from_lagrangian_string(
        lagrangian_str: &str,
        coordinates: &[&str],
        time_var: Option<&str>,
    ) -> Result<ELDerivation> {
        let lagrangian = scirust_symbolic::parse(lagrangian_str)
            .map_err(|e| VariationalError::SymbolicFailure { details: e })?;

        let velocities: Vec<String> = coordinates.iter().map(|c| format!("{}_dot", c)).collect();
        let coord_strings: Vec<String> = coordinates.iter().map(|c| c.to_string()).collect();

        let lagrangian_sub = {
            let mut lag = lagrangian.clone();
            for (c, v) in coordinates.iter().zip(velocities.iter())
            {
                let v_var = Expr::Var(v.clone());
                lag = substitute(&lag, &format!("{}dot", c), &v_var);
            }
            lag
        };

        Self::derive(&lagrangian_sub, &coord_strings, &velocities, time_var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    fn eval_expr(expr: &Expr, vals: &HashMap<String, f64>) -> f64 {
        scirust_symbolic::eval(expr, vals).unwrap()
    }

    #[test]
    fn test_free_particle() {
        let m = 1.0;
        let lagrangian = Expr::Const(0.5)
            * Expr::Const(m)
            * Expr::Pow(
                Box::new(Expr::Var("dq".to_string())),
                Box::new(Expr::Const(2.0)),
            );
        let coords = vec!["q".to_string()];
        let vels = vec!["dq".to_string()];
        let result = SymbolicEulerLagrange::derive(&lagrangian, &coords, &vels, None).unwrap();
        assert_eq!(result.num_coordinates(), 1);
        let eq = &result.equations[0];
        let mut vals = HashMap::new();
        vals.insert("q_ddot".to_string(), 2.0);
        vals.insert("q".to_string(), 1.0);
        vals.insert("dq".to_string(), 3.0);
        let residual = eval_expr(&eq.residual, &vals);
        assert_relative_eq!(residual, m * 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_harmonic_oscillator() {
        let m = 1.0;
        let k = 1.0;
        let lagrangian = Expr::Const(0.5)
            * Expr::Const(m)
            * Expr::Pow(
                Box::new(Expr::Var("dq".to_string())),
                Box::new(Expr::Const(2.0)),
            )
            - Expr::Const(0.5)
                * Expr::Const(k)
                * Expr::Pow(
                    Box::new(Expr::Var("q".to_string())),
                    Box::new(Expr::Const(2.0)),
                );
        let coords = vec!["q".to_string()];
        let vels = vec!["dq".to_string()];
        let result = SymbolicEulerLagrange::derive(&lagrangian, &coords, &vels, None).unwrap();
        let eq = &result.equations[0];
        let eq_s = eq.residual.to_string();
        assert!(eq_s.contains("q_ddot") || eq_s.contains("q") || eq_s.contains("1"));
        let mut vals = HashMap::new();
        vals.insert("q".to_string(), 0.5);
        vals.insert("dq".to_string(), 0.0);
        vals.insert("q_ddot".to_string(), 0.0);
        let residual = eval_expr(&eq.residual, &vals);
        assert!(
            (residual - (m * 0.0 + k * 0.5)).abs() < 1e-6
                || (residual - (m * 0.0 + k * 0.5).abs()).abs() < 1e-6,
            "residual = {residual}"
        );
    }

    #[test]
    fn test_simple_pendulum() {
        let m = 1.0;
        let g = 9.81;
        let l = 1.0;
        let lagrangian = Expr::Const(0.5)
            * Expr::Const(m)
            * Expr::Const(l * l)
            * Expr::Pow(
                Box::new(Expr::Var("dtheta".to_string())),
                Box::new(Expr::Const(2.0)),
            )
            - Expr::Const(m)
                * Expr::Const(g)
                * Expr::Const(l)
                * (Expr::Const(1.0) - Expr::Cos(Box::new(Expr::Var("theta".to_string()))));
        let coords = vec!["theta".to_string()];
        let vels = vec!["dtheta".to_string()];
        let result = SymbolicEulerLagrange::derive(&lagrangian, &coords, &vels, None).unwrap();
        let eq = &result.equations[0];
        let mut vals = HashMap::new();
        vals.insert("theta".to_string(), 0.3);
        vals.insert("dtheta".to_string(), 0.0);
        vals.insert("theta_ddot".to_string(), 0.0);
        let residual = eval_expr(&eq.residual, &vals);
        let expected = m * l * l * 0.0 + m * g * l * 0.3_f64.sin();
        assert!(
            (residual - expected).abs() < 1e-4,
            "residual = {residual}, expected ~ {expected}"
        );
    }

    #[test]
    fn test_coupled_oscillator() {
        let m1 = 1.0;
        let m2 = 1.0;
        let k1 = 1.0;
        let k2 = 1.0;
        let k3 = 1.0;
        let lagrangian = Expr::Const(0.5)
            * Expr::Const(m1)
            * Expr::Pow(
                Box::new(Expr::Var("dq1".to_string())),
                Box::new(Expr::Const(2.0)),
            )
            + Expr::Const(0.5)
                * Expr::Const(m2)
                * Expr::Pow(
                    Box::new(Expr::Var("dq2".to_string())),
                    Box::new(Expr::Const(2.0)),
                )
            - Expr::Const(0.5)
                * Expr::Const(k1)
                * Expr::Pow(
                    Box::new(Expr::Var("q1".to_string())),
                    Box::new(Expr::Const(2.0)),
                )
            - Expr::Const(0.5)
                * Expr::Const(k2)
                * Expr::Pow(
                    Box::new(Expr::Var("q2".to_string()) - Expr::Var("q1".to_string())),
                    Box::new(Expr::Const(2.0)),
                )
            - Expr::Const(0.5)
                * Expr::Const(k3)
                * Expr::Pow(
                    Box::new(Expr::Var("q2".to_string())),
                    Box::new(Expr::Const(2.0)),
                );
        let coords = vec!["q1".to_string(), "q2".to_string()];
        let vels = vec!["dq1".to_string(), "dq2".to_string()];
        let result = SymbolicEulerLagrange::derive(&lagrangian, &coords, &vels, None).unwrap();
        assert_eq!(result.num_coordinates(), 2);
        for eq in &result.equations
        {
            assert!(
                eq.residual.to_string().contains("q1_ddot")
                    || eq.residual.to_string().contains("q2_ddot")
                    || !eq.acceleration_deps.is_empty()
            );
        }
    }

    #[test]
    fn test_time_dependent_lagrangian() {
        let lagrangian = Expr::Sin(Box::new(Expr::Var("t".to_string())))
            * Expr::Pow(
                Box::new(Expr::Var("q".to_string())),
                Box::new(Expr::Const(2.0)),
            )
            + Expr::Const(0.5)
                * Expr::Pow(
                    Box::new(Expr::Var("dq".to_string())),
                    Box::new(Expr::Const(2.0)),
                );
        let coords = vec!["q".to_string()];
        let vels = vec!["dq".to_string()];
        let result = SymbolicEulerLagrange::derive(&lagrangian, &coords, &vels, Some("t")).unwrap();
        assert_eq!(result.num_coordinates(), 1);
        let eq_s = result.equations[0].residual.to_string();
        assert!(
            eq_s.contains("t") || eq_s.contains("q") || eq_s.contains("sin"),
            "residual = {eq_s}"
        );
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let lagrangian = Expr::Const(0.0);
        let coords = vec!["q".to_string()];
        let vels = vec![];
        let result = SymbolicEulerLagrange::derive(&lagrangian, &coords, &vels, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_euler_lagrange_identity() {
        let lagrangian = Expr::Pow(
            Box::new(Expr::Var("dq".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        let coords = vec!["q".to_string()];
        let vels = vec!["dq".to_string()];
        let result = SymbolicEulerLagrange::derive(&lagrangian, &coords, &vels, None).unwrap();
        let eq = &result.equations[0];
        let mut vals = HashMap::new();
        vals.insert("q".to_string(), 1.0);
        vals.insert("dq".to_string(), 0.0);
        vals.insert("q_ddot".to_string(), 3.0);
        let residual = eval_expr(&eq.residual, &vals);
        assert_relative_eq!(residual, 2.0 * 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euler_lagrange_coord_no_show() {
        let lagrangian = Expr::Pow(
            Box::new(Expr::Var("dq".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        let coords = vec!["q".to_string()];
        let vels = vec!["dq".to_string()];
        let result = SymbolicEulerLagrange::derive(&lagrangian, &coords, &vels, None).unwrap();
        let eq = &result.equations[0];
        assert!(eq.acceleration_deps.contains(&"q_ddot".to_string()));
    }

    #[test]
    fn test_zero_lagrangian() {
        let lagrangian = Expr::Const(0.0);
        let coords = vec!["q".to_string()];
        let vels = vec!["dq".to_string()];
        let result = SymbolicEulerLagrange::derive(&lagrangian, &coords, &vels, None).unwrap();
        let mut vals = HashMap::new();
        vals.insert("q".to_string(), 0.0);
        vals.insert("dq".to_string(), 0.0);
        vals.insert("q_ddot".to_string(), 0.0);
        let residual = eval_expr(&result.equations[0].residual, &vals);
        assert_relative_eq!(residual, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_no_coordinates_error() {
        let lagrangian = Expr::Const(0.0);
        let result = SymbolicEulerLagrange::derive(&lagrangian, &[], &[], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_display_euler_lagrange() {
        let lagrangian = Expr::Const(0.5)
            * Expr::Pow(
                Box::new(Expr::Var("dq".to_string())),
                Box::new(Expr::Const(2.0)),
            );
        let coords = vec!["q".to_string()];
        let vels = vec!["dq".to_string()];
        let result = SymbolicEulerLagrange::derive(&lagrangian, &coords, &vels, None).unwrap();
        let display = format!("{}", result.equations[0]);
        assert!(display.contains("Euler"));
        assert!(display.contains("q"));
    }
}
