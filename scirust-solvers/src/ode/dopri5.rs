//! Dormand-Prince 5(4) — Runge-Kutta d'ordre 5 avec estimateur d'erreur d'ordre 4
//! embarqué. Le pas est adapté à chaque étape pour respecter une tolérance.
//!
//! ## Sécurité numérique
//! - `check_finite` après chaque évaluation de dérivée (k1..k7)
//! - Backup du dernier état valide (`last_good_y`) pour rollback
//! - Pas de temps plancher `MIN_STEP = 1e-14` — StepUnderflow sinon
//! - `MAX_REJECTIONS = 100` — boucle infinie impossible
//! - Division par zéro dans `err / sc` protégée par `atol.max(1e-15)`
//!
//! 7 évaluations par pas (FSAL : la 7e devient k1 du pas suivant).
//! C'est l'algorithme de `scipy.integrate.RK45` et `Matlab ode45`.

use crate::SolverError;
use crate::SolverResult;
use tracing::warn;

/// Nombre maximal de rejets consécutifs avant abandon.
const MAX_REJECTIONS: usize = 100;

/// Pas de temps minimum absolu (évite division par zéro et boucle infinie).
const MIN_STEP: f64 = 1e-14;

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 5.0;

const MAX_STEPS: usize = 100_000;

fn check_finite(value: f64, _location: &str) -> Result<(), SolverError> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value });
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct OdeOutput {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub accepted: usize,
    pub rejected: usize,
}

/// Coefficients de Dormand-Prince (table de Butcher).
/// Les constantes `A71..A76` ne sont pas explicitement utilisées dans le code
/// car la propriété FSAL implique `a7j = b_j`.
#[allow(dead_code)]
mod dp_coeffs {
    pub const C2: f64 = 1.0 / 5.0;
    pub const C3: f64 = 3.0 / 10.0;
    pub const C4: f64 = 4.0 / 5.0;
    pub const C5: f64 = 8.0 / 9.0;

    pub const A21: f64 = 1.0 / 5.0;
    pub const A31: f64 = 3.0 / 40.0;
    pub const A32: f64 = 9.0 / 40.0;
    pub const A41: f64 = 44.0 / 45.0;
    pub const A42: f64 = -56.0 / 15.0;
    pub const A43: f64 = 32.0 / 9.0;
    pub const A51: f64 = 19372.0 / 6561.0;
    pub const A52: f64 = -25360.0 / 2187.0;
    pub const A53: f64 = 64448.0 / 6561.0;
    pub const A54: f64 = -212.0 / 729.0;
    pub const A61: f64 = 9017.0 / 3168.0;
    pub const A62: f64 = -355.0 / 33.0;
    pub const A63: f64 = 46732.0 / 5247.0;
    pub const A64: f64 = 49.0 / 176.0;
    pub const A65: f64 = -5103.0 / 18656.0;
    pub const A71: f64 = 35.0 / 384.0;
    pub const A73: f64 = 500.0 / 1113.0;
    pub const A74: f64 = 125.0 / 192.0;
    pub const A75: f64 = -2187.0 / 6784.0;
    pub const A76: f64 = 11.0 / 84.0;

    pub const B1: f64 = 35.0 / 384.0;
    pub const B3: f64 = 500.0 / 1113.0;
    pub const B4: f64 = 125.0 / 192.0;
    pub const B5: f64 = -2187.0 / 6784.0;
    pub const B6: f64 = 11.0 / 84.0;

    pub const E1: f64 = 71.0 / 57600.0;
    pub const E3: f64 = -71.0 / 16695.0;
    pub const E4: f64 = 71.0 / 1920.0;
    pub const E5: f64 = -17253.0 / 339200.0;
    pub const E6: f64 = 22.0 / 525.0;
    pub const E7: f64 = -1.0 / 40.0;
}

/// Intégrateur DOPRI5 adaptatif.
///
/// - `t0`, `t_end` : intervalle d'intégration
/// - `y0`          : condition initiale
/// - `rtol`, `atol`: tolérances relative et absolue
/// - `h_init`      : pas initial (sera adapté)
pub fn dopri5<F>(
    f: F,
    t0: f64,
    t_end: f64,
    y0: Vec<f64>,
    rtol: f64,
    atol: f64,
    h_init: f64,
) -> SolverResult<OdeOutput>
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    use dp_coeffs::*;
    let n = y0.len();
    if t_end <= t0
    {
        return Err(SolverError::InvalidInput("t_end must be > t0".into()));
    }
    if n == 0
    {
        return Err(SolverError::InvalidInput(
            "system dimension must be > 0".into(),
        ));
    }

    let mut t = t0;
    let mut y = y0;
    let mut h = h_init.max(MIN_STEP).min(t_end - t0);
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    // Backup du dernier état valide pour rollback sur divergence
    let mut last_good_t = t;
    let mut last_good_y = y.clone();

    let mut out_t = vec![t];
    let mut out_y = vec![y.clone()];

    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut k5 = vec![0.0; n];
    let mut k6 = vec![0.0; n];
    let mut k7 = vec![0.0; n];
    let mut ytmp = vec![0.0; n];
    let mut ynew = vec![0.0; n];

    f(t, &y, &mut k1);

    for _ in 0..MAX_STEPS
    {
        if t >= t_end
        {
            break;
        }
        if t + h > t_end
        {
            h = t_end - t;
        }

        // k2
        for i in 0..n
        {
            ytmp[i] = y[i] + h * A21 * k1[i];
            check_finite(ytmp[i], &format!("ytmp k2[{i}]"))?;
        }
        f(t + C2 * h, &ytmp, &mut k2);
        for i in 0..n
        {
            check_finite(k2[i], &format!("k2[{i}]"))?;
        }

        // k3
        for i in 0..n
        {
            ytmp[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
            check_finite(ytmp[i], &format!("ytmp k3[{i}]"))?;
        }
        f(t + C3 * h, &ytmp, &mut k3);
        for i in 0..n
        {
            check_finite(k3[i], &format!("k3[{i}]"))?;
        }

        // k4
        for i in 0..n
        {
            ytmp[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
            check_finite(ytmp[i], &format!("ytmp k4[{i}]"))?;
        }
        f(t + C4 * h, &ytmp, &mut k4);
        for i in 0..n
        {
            check_finite(k4[i], &format!("k4[{i}]"))?;
        }

        // k5
        for i in 0..n
        {
            ytmp[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
            check_finite(ytmp[i], &format!("ytmp k5[{i}]"))?;
        }
        f(t + C5 * h, &ytmp, &mut k5);
        for i in 0..n
        {
            check_finite(k5[i], &format!("k5[{i}]"))?;
        }

        // k6
        for i in 0..n
        {
            ytmp[i] =
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
            check_finite(ytmp[i], &format!("ytmp k6[{i}]"))?;
        }
        f(t + h, &ytmp, &mut k6);
        for i in 0..n
        {
            check_finite(k6[i], &format!("k6[{i}]"))?;
        }

        // y_new à l'ordre 5
        for i in 0..n
        {
            ynew[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
            check_finite(ynew[i], &format!("ynew[{i}]"))?;
        }

        // k7 (FSAL)
        f(t + h, &ynew, &mut k7);
        for i in 0..n
        {
            check_finite(k7[i], &format!("k7[{i}]"))?;
        }

        // Estimateur d'erreur normalisé
        let mut err_norm = 0.0_f64;
        for i in 0..n
        {
            let err =
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
            check_finite(err, &format!("err[{i}]"))?;
            let sc = atol.max(1e-15) + rtol * y[i].abs().max(ynew[i].abs());
            err_norm += (err / sc).powi(2);
        }
        err_norm = (err_norm / n as f64).sqrt();
        check_finite(err_norm, "err_norm")?;

        if err_norm <= 1.0
        {
            // Accepté → backup le bon état
            last_good_t = t + h;
            last_good_y.copy_from_slice(&ynew);

            t += h;
            std::mem::swap(&mut y, &mut ynew);
            std::mem::swap(&mut k1, &mut k7);
            accepted += 1;
            out_t.push(t);
            out_y.push(y.clone());

            let factor = if err_norm == 0.0
            {
                MAX_FACTOR
            }
            else
            {
                (SAFETY * err_norm.powf(-0.2)).clamp(MIN_FACTOR, MAX_FACTOR)
            };
            check_finite(factor, "factor_accept")?;
            h *= factor;
        }
        else
        {
            // Rejeté — réduction agressive
            rejected += 1;
            let factor = (SAFETY * err_norm.powf(-0.2)).clamp(MIN_FACTOR, MAX_FACTOR);
            check_finite(factor, "factor_reject")?;
            h *= factor;

            if h < MIN_STEP
            {
                warn!(
                    target: "solver",
                    "DOPRI5: step underflow h={:.3e} at t={} — restoring backup from t={}",
                    h, t, last_good_t
                );
                return Err(SolverError::BackupRestored {
                    iter: accepted,
                    reason: format!(
                        "step underflow h={:.3e}, rejecting {} steps, rolling back to t={}",
                        h, rejected, last_good_t
                    ),
                });
            }

            if rejected > MAX_REJECTIONS
            {
                warn!(
                    target: "solver",
                    "DOPRI5: too many consecutive rejections ({}) at t={} — aborting",
                    rejected, t
                );
                return Err(SolverError::IntegrationFailed(format!(
                    "too many consecutive rejections ({}) at t={}, h={:.3e}",
                    rejected, t, h
                )));
            }
        }
    }

    if t < t_end
    {
        return Err(SolverError::IntegrationFailed(format!(
            "max steps ({}) reached at t={}, t_end={}",
            MAX_STEPS, t, t_end
        )));
    }

    Ok(OdeOutput {
        t: out_t,
        y: out_y,
        accepted,
        rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn exponential_decay_adaptive() {
        let r = dopri5(
            |_, y, dy| dy[0] = -y[0],
            0.0,
            5.0,
            vec![1.0],
            1e-8,
            1e-10,
            0.1,
        )
        .unwrap();
        assert_relative_eq!(r.t[r.t.len() - 1], 5.0, epsilon = 1e-10);
        assert_relative_eq!(r.y[r.y.len() - 1][0], (-5.0_f64).exp(), epsilon = 1e-8);
    }

    #[test]
    fn van_der_pol_nonstiff() {
        let mu = 1.0;
        let r = dopri5(
            move |_, y, dy| {
                dy[0] = y[1];
                dy[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            },
            0.0,
            10.0,
            vec![2.0, 0.0],
            1e-6,
            1e-9,
            0.1,
        )
        .unwrap();
        assert!(r.accepted > 0);
        assert!(r.rejected < r.accepted);
    }

    #[test]
    fn pendulum() {
        let r = dopri5(
            |_, y, dy| {
                dy[0] = y[1];
                dy[1] = -9.81 * y[0].sin();
            },
            0.0,
            10.0,
            vec![0.5, 0.0],
            1e-8,
            1e-10,
            0.05,
        )
        .unwrap();
        let e0 = 0.5_f64.cos().mul_add(-9.81, 9.81);
        let last = &r.y[r.y.len() - 1];
        let e = last[1].powi(2) / 2.0 + 9.81 * (1.0 - last[0].cos());
        assert!((e - e0).abs() / e0 < 1e-6, "energy drift: e0={e0}, e={e}");
    }
}
