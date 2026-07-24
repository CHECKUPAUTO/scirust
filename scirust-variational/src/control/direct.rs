use crate::control::problem::{ControlSolution, OptimalControlProblem};
use crate::error::{Result, VariationalError};

pub fn direct_shooting<D, RC, TC>(
    problem: &OptimalControlProblem<D, RC, TC>,
    u_initial_guess: &[f32],
) -> Result<ControlSolution>
where
    D: Fn(f32, &[f32], &[f32], &mut [f32]),
    RC: Fn(f32, &[f32], &[f32]) -> f32,
    TC: Fn(&[f32], f32) -> f32,
{
    let n = problem.num_time_steps;
    let m = problem.control_dim;
    let s = problem.state_dim;
    let (t0, tf) = problem.horizon;
    let dt = (tf - t0) / (n - 1) as f32;

    if u_initial_guess.len() != n * m
    {
        return Err(VariationalError::DimensionMismatch {
            expected: n * m,
            got: u_initial_guess.len(),
            context: "direct_shooting: control initial guess".into(),
        });
    }

    let times: Vec<f32> = (0..n).map(|i| t0 + i as f32 * dt).collect();

    let mut u = u_initial_guess.to_vec();
    let mut objective = f32::INFINITY;
    let mut best_u = u.clone();

    let lr = 1e-3;
    let eps = 1e-4;
    let max_iter = 1000;

    for _iter in 0..max_iter
    {
        let (_states, obj) = rollout(problem, &u, &times, s, m, n, dt);
        let mut grad = vec![0.0; n * m];

        for i in 0..n * m
        {
            let mut u_plus = u.clone();
            u_plus[i] += eps;
            let (_, obj_plus) = rollout(problem, &u_plus, &times, s, m, n, dt);

            let mut u_minus = u.clone();
            u_minus[i] -= eps;
            let (_, obj_minus) = rollout(problem, &u_minus, &times, s, m, n, dt);

            grad[i] = (obj_plus - obj_minus) / (2.0 * eps);
        }

        for i in 0..n * m
        {
            u[i] -= lr * grad[i];
            if let Some(ref bounds) = problem.control_bounds
            {
                let step = i % m;
                u[i] = u[i].clamp(bounds.lower[step], bounds.upper[step]);
            }
        }

        if obj < objective
        {
            objective = obj;
            best_u = u.clone();
        }

        let grad_norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        if grad_norm < 1e-6
        {
            break;
        }
    }

    let (final_states, final_obj) = rollout(problem, &best_u, &times, s, m, n, dt);

    let terminal_state = &final_states[final_states.len() - 1];
    let terminal_violation = (problem.terminal_cost)(terminal_state, tf);

    let u_matrix: Vec<Vec<f32>> = best_u.chunks(m).map(|c| c.to_vec()).collect();

    Ok(ControlSolution {
        times,
        states: final_states,
        controls: u_matrix,
        objective: final_obj,
        feasibility_residual: terminal_violation,
        converged: true,
        iterations: max_iter,
    })
}

fn rollout<D, RC, TC>(
    problem: &OptimalControlProblem<D, RC, TC>,
    u_flat: &[f32],
    times: &[f32],
    s: usize,
    m: usize,
    n: usize,
    dt: f32,
) -> (Vec<Vec<f32>>, f32)
where
    D: Fn(f32, &[f32], &[f32], &mut [f32]),
    RC: Fn(f32, &[f32], &[f32]) -> f32,
    TC: Fn(&[f32], f32) -> f32,
{
    let mut state = problem.initial_state.clone();
    let mut states = vec![state.clone()];
    let mut total_cost = 0.0;

    for i in 0..n - 1
    {
        let t = times[i];
        let u = &u_flat[i * m..(i + 1) * m];
        let mut deriv = vec![0.0; s];

        (problem.dynamics)(t, &state, u, &mut deriv);

        total_cost += (problem.running_cost)(t, &state, u) * dt;

        for j in 0..s
        {
            state[j] += dt * deriv[j];
        }

        if let Some(ref bounds) = problem.state_bounds
        {
            for j in 0..s
            {
                state[j] = state[j].clamp(bounds.lower[j], bounds.upper[j]);
            }
        }

        states.push(state.clone());
    }

    let tf = times[n - 1];
    total_cost += (problem.terminal_cost)(&states[states.len() - 1], tf);

    (states, total_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_integrator_direct_shooting() {
        let dynamics = |_t: f32, x: &[f32], u: &[f32], dx: &mut [f32]| {
            dx[0] = x[1];
            dx[1] = u[0];
        };
        let running_cost = |_t: f32, x: &[f32], u: &[f32]| -> f32 {
            0.5 * (x[0] * x[0] + x[1] * x[1] + u[0] * u[0])
        };
        let terminal_cost = |x: &[f32], _t: f32| -> f32 { 10.0 * (x[0] * x[0] + x[1] * x[1]) };

        let problem = OptimalControlProblem::new(
            2,
            1,
            dynamics,
            running_cost,
            terminal_cost,
            vec![1.0, 0.0],
            (0.0, 2.0),
        )
        .unwrap()
        .with_time_steps(50);

        let n = problem.num_time_steps;
        let u_init = vec![0.0; n];
        let solution = direct_shooting(&problem, &u_init).unwrap();
        assert_eq!(solution.states.len(), n);
        assert!(solution.objective.is_finite(), "objective should be finite");
    }
}
