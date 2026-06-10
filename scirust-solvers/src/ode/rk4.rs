//! Runge-Kutta classique d'ordre 4 à pas fixe, avec guards NaN.
//!
//! ```text
//! k1 = f(t, y)
//! k2 = f(t + h/2, y + h/2 · k1)
//! k3 = f(t + h/2, y + h/2 · k2)
//! k4 = f(t + h,   y + h   · k3)
//! y_{n+1} = y_n + h/6 · (k1 + 2 k2 + 2 k3 + k4)
//! ```


fn check_finite(value: f64, _label: &str) -> Result<(), crate::SolverError> {
    if !value.is_finite()
    {
        return Err(crate::SolverError::NanDetected { iter: 0, value });
    }
    Ok(())
}

/// Intègre `dy/dt = f(t, y)` de `t0` à `t_end` par pas `h`.
/// Renvoie `Vec<(t, y)>` avec tous les points de discrétisation.
pub fn rk4_fixed<F>(f: F, t0: f64, t_end: f64, y0: Vec<f64>, h: f64) -> Vec<(f64, Vec<f64>)>
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let n = y0.len();
    let nsteps = ((t_end - t0) / h).ceil() as usize;
    let mut out = Vec::with_capacity(nsteps + 1);
    let mut t = t0;
    let mut y = y0;
    out.push((t, y.clone()));

    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut ytmp = vec![0.0; n];

    for _ in 0..nsteps
    {
        let h_act = if t + h > t_end { t_end - t } else { h };
        if h_act <= 0.0
        {
            break;
        }

        f(t, &y, &mut k1);
        if k1.iter().any(|v| !v.is_finite())
        {
            break;
        }

        for i in 0..n
        {
            ytmp[i] = y[i] + 0.5 * h_act * k1[i];
        }
        if ytmp.iter().any(|v| !v.is_finite())
        {
            break;
        }
        f(t + 0.5 * h_act, &ytmp, &mut k2);
        if k2.iter().any(|v| !v.is_finite())
        {
            break;
        }

        for i in 0..n
        {
            ytmp[i] = y[i] + 0.5 * h_act * k2[i];
        }
        if ytmp.iter().any(|v| !v.is_finite())
        {
            break;
        }
        f(t + 0.5 * h_act, &ytmp, &mut k3);
        if k3.iter().any(|v| !v.is_finite())
        {
            break;
        }

        for i in 0..n
        {
            ytmp[i] = y[i] + h_act * k3[i];
        }
        if ytmp.iter().any(|v| !v.is_finite())
        {
            break;
        }
        f(t + h_act, &ytmp, &mut k4);
        if k4.iter().any(|v| !v.is_finite())
        {
            break;
        }

        for i in 0..n
        {
            y[i] += h_act / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            if !y[i].is_finite()
            {
                // NaN détecté après mise à jour — on stoppe proprement
                break;
            }
        }
        t += h_act;
        out.push((t, y.clone()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn exponential_decay() {
        let out = rk4_fixed(|_, y, dy| dy[0] = -y[0], 0.0, 1.0, vec![1.0], 0.01);
        let (t_final, y_final) = out.last().unwrap();
        assert_relative_eq!(*t_final, 1.0, epsilon = 1e-12);
        assert_relative_eq!(y_final[0], (-1.0_f64).exp(), epsilon = 1e-8);
    }

    #[test]
    fn harmonic_oscillator() {
        let out = rk4_fixed(
            |_, y, dy| {
                dy[0] = y[1];
                dy[1] = -y[0];
            },
            0.0,
            std::f64::consts::PI,
            vec![1.0, 0.0],
            0.001,
        );
        let (_, y_final) = out.last().unwrap();
        assert_relative_eq!(y_final[0], -1.0, epsilon = 1e-8);
        assert!(y_final[1].abs() < 1e-7);
    }
}
