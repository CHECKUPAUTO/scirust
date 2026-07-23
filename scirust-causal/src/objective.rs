use crate::acyclicity::PolynomialAcyclicity;
use crate::cubic_score::CubicCausalScore;
use crate::error::CausalError;
use scirust_solvers::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub struct AugmentedLagrangianConfig {
    pub lambda_l1: f64,
    pub alpha: f64,
    pub rho: f64,
    pub smooth_l1_epsilon: f64,
}

impl AugmentedLagrangianConfig {
    pub fn new(
        lambda_l1: f64,
        alpha: f64,
        rho: f64,
        smooth_l1_epsilon: f64,
    ) -> Result<Self, CausalError> {
        if !lambda_l1.is_finite()
        {
            return Err(CausalError::InvalidConfiguration {
                name: "lambda_l1",
                value: lambda_l1,
            });
        }
        if !alpha.is_finite()
        {
            return Err(CausalError::InvalidConfiguration {
                name: "alpha",
                value: alpha,
            });
        }
        if !rho.is_finite() || rho <= 0.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "rho",
                value: rho,
            });
        }
        if !smooth_l1_epsilon.is_finite() || smooth_l1_epsilon <= 0.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "smooth_l1_epsilon",
                value: smooth_l1_epsilon,
            });
        }
        if lambda_l1 < 0.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "lambda_l1",
                value: lambda_l1,
            });
        }
        if alpha < 0.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "alpha",
                value: alpha,
            });
        }

        Ok(Self {
            lambda_l1,
            alpha,
            rho,
            smooth_l1_epsilon,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveEvaluation {
    pub total: f64,
    pub data_loss: f64,
    pub sparsity_penalty: f64,
    pub acyclicity: f64,
    pub augmented_penalty: f64,
    pub gradient: Matrix,
}

pub struct CausalObjective;

impl CausalObjective {
    fn smooth_l1_value(interactions: &Matrix, eps: f64) -> (f64, Matrix) {
        let (rows, cols) = interactions.shape();
        let mut val = 0.0;
        let mut grad = Matrix::zeros(rows, cols);

        for i in 0..rows
        {
            for j in 0..cols
            {
                let a = interactions[(i, j)];
                let denom = (a * a + eps * eps).sqrt();
                val += denom;
                grad[(i, j)] = a / denom;
            }
        }

        (val, grad)
    }

    pub fn evaluate(
        samples: &Matrix,
        interactions: &Matrix,
        config: &AugmentedLagrangianConfig,
    ) -> Result<ObjectiveEvaluation, CausalError> {
        let (data_loss, grad_data) = CubicCausalScore::loss_and_gradient(samples, interactions)?;

        let (sparsity, grad_sparsity_mat) =
            Self::smooth_l1_value(interactions, config.smooth_l1_epsilon);
        let sparsity_penalty = config.lambda_l1 * sparsity;

        let (acyclicity_val, grad_acyclicity) =
            PolynomialAcyclicity::value_and_gradient(interactions)?;

        let augmented_penalty = config.rho * 0.5 * acyclicity_val * acyclicity_val;
        let penalty_factor = config.alpha + config.rho * acyclicity_val;

        let total =
            data_loss + sparsity_penalty + config.alpha * acyclicity_val + augmented_penalty;

        let (rows, cols) = interactions.shape();
        let mut gradient = Matrix::zeros(rows, cols);
        for i in 0..rows
        {
            for j in 0..cols
            {
                let mut g = grad_data[(i, j)];
                g += config.lambda_l1 * grad_sparsity_mat[(i, j)];
                g += penalty_factor * grad_acyclicity[(i, j)];

                if !g.is_finite()
                {
                    return Err(CausalError::NonFiniteComputation {
                        operation: "objective_gradient",
                        index: i,
                        value: g,
                    });
                }
                gradient[(i, j)] = g;
            }
        }

        if !total.is_finite()
        {
            return Err(CausalError::NonFiniteComputation {
                operation: "objective_total",
                index: 0,
                value: total,
            });
        }

        Ok(ObjectiveEvaluation {
            total,
            data_loss,
            sparsity_penalty,
            acyclicity: acyclicity_val,
            augmented_penalty,
            gradient,
        })
    }
}
