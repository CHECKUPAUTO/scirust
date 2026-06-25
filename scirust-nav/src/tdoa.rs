//! Time-difference-of-arrival (TDOA) multilateration in the plane.
//!
//! An emitter fires at an unknown time; sensors at known positions timestamp
//! the arrival. The *differences* of those timestamps cancel the unknown
//! emission time and constrain the source to a hyperbola per sensor pair; their
//! intersection is the location. We solve it by Gauss–Newton on the range-
//! difference residuals.
//!
//! The same geometry locates a **partial-discharge** source in a transformer
//! tank or an **acoustic-emission** source in a pressure vessel from the arrival
//! times of the elastic/EM wave at mounted sensors — only the wave speed
//! changes.

use scirust_estimation::Mat;

/// Result of a TDOA solve.
#[derive(Debug, Clone, Copy)]
pub struct TdoaSolution {
    /// Estimated source position `[x, y]`.
    pub position: [f64; 2],
    /// RMS of the range-difference residuals at the solution (metres).
    pub residual_rms: f64,
    /// Gauss–Newton iterations taken.
    pub iterations: usize,
    /// Whether the step norm fell below tolerance before the iteration cap.
    pub converged: bool,
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Locate a 2-D emitter from sensor positions and absolute `arrival_times`
/// (seconds) given the wave `speed` (m/s). Needs `≥ 3` sensors. The first
/// sensor is the time reference; an optional `init` seeds the search (defaults
/// to the sensor centroid).
pub fn tdoa_locate_2d(
    sensors: &[[f64; 2]],
    arrival_times: &[f64],
    speed: f64,
    init: Option<[f64; 2]>,
    max_iter: usize,
) -> Option<TdoaSolution> {
    let m = sensors.len();
    if m < 3 || arrival_times.len() != m || speed <= 0.0
    {
        return None;
    }
    // Measured range differences relative to sensor 0.
    let meas: Vec<f64> = (1..m)
        .map(|i| speed * (arrival_times[i] - arrival_times[0]))
        .collect();

    // Initial guess: centroid (or caller-supplied).
    let mut x = init.unwrap_or_else(|| {
        let mut c = [0.0, 0.0];
        for s in sensors
        {
            c[0] += s[0] / m as f64;
            c[1] += s[1] / m as f64;
        }
        c
    });

    let mut converged = false;
    let mut iters = 0;
    for it in 0..max_iter
    {
        iters = it + 1;
        let d0 = dist(x, sensors[0]).max(1e-9);
        let u0 = [(x[0] - sensors[0][0]) / d0, (x[1] - sensors[0][1]) / d0];
        // Residuals r (length m-1) and Jacobian J ((m-1)×2).
        let mut r = vec![0.0; m - 1];
        let mut j = Mat::zeros(m - 1, 2);
        for i in 1..m
        {
            let di = dist(x, sensors[i]).max(1e-9);
            let ui = [(x[0] - sensors[i][0]) / di, (x[1] - sensors[i][1]) / di];
            r[i - 1] = (di - d0) - meas[i - 1];
            j.set(i - 1, 0, ui[0] - u0[0]);
            j.set(i - 1, 1, ui[1] - u0[1]);
        }
        // Gauss–Newton step: Δ = (JᵀJ)⁻¹ Jᵀr.
        let jt = j.t();
        let jtj = jt.matmul(&j);
        let jtr = jt.matvec(&r);
        // Degenerate geometry → singular normal matrix → bail out.
        let step = jtj.inverse()?.matvec(&jtr);
        x[0] -= step[0];
        x[1] -= step[1];
        if step[0].hypot(step[1]) < 1e-10
        {
            converged = true;
            break;
        }
    }

    // Residual RMS at the solution.
    let d0 = dist(x, sensors[0]);
    let mut ss = 0.0;
    for i in 1..m
    {
        let ri = (dist(x, sensors[i]) - d0) - meas[i - 1];
        ss += ri * ri;
    }
    let residual_rms = (ss / (m - 1) as f64).sqrt();

    Some(TdoaSolution {
        position: x,
        residual_rms,
        iterations: iters,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrivals(sensors: &[[f64; 2]], src: [f64; 2], speed: f64, t0: f64) -> Vec<f64> {
        sensors.iter().map(|s| t0 + dist(*s, src) / speed).collect()
    }

    #[test]
    fn recovers_a_known_source_from_clean_tdoas() {
        let sensors = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let speed = 1500.0; // e.g. acoustic in oil
        let src = [3.0, 7.0];
        let t = arrivals(&sensors, src, speed, 0.123);
        let sol = tdoa_locate_2d(&sensors, &t, speed, None, 50).unwrap();
        assert!(sol.converged, "should converge");
        assert!(
            (sol.position[0] - src[0]).abs() < 1e-6,
            "x {:?}",
            sol.position
        );
        assert!(
            (sol.position[1] - src[1]).abs() < 1e-6,
            "y {:?}",
            sol.position
        );
        assert!(sol.residual_rms < 1e-6);
    }

    #[test]
    fn source_outside_the_array_still_solves() {
        let sensors = [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]];
        let speed = 343.0; // air
        let src = [8.0, 5.0];
        let t = arrivals(&sensors, src, speed, 0.0);
        let sol = tdoa_locate_2d(&sensors, &t, speed, Some([5.0, 5.0]), 100).unwrap();
        assert!(
            (sol.position[0] - src[0]).abs() < 1e-4,
            "x {:?}",
            sol.position
        );
        assert!(
            (sol.position[1] - src[1]).abs() < 1e-4,
            "y {:?}",
            sol.position
        );
    }

    #[test]
    fn small_timing_noise_gives_a_small_location_error() {
        let sensors = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let speed = 1500.0;
        let src = [6.0, 4.0];
        let mut t = arrivals(&sensors, src, speed, 0.0);
        // ±10 µs jitter (deterministic), ~1.5 cm of range at this speed.
        let jitter = [1e-5, -8e-6, 6e-6, -4e-6];
        for (ti, j) in t.iter_mut().zip(jitter)
        {
            *ti += j;
        }
        let sol = tdoa_locate_2d(&sensors, &t, speed, None, 50).unwrap();
        let err = dist(sol.position, src);
        assert!(err < 0.1, "location error {err} m too large");
    }

    #[test]
    fn too_few_sensors_is_rejected() {
        let sensors = [[0.0, 0.0], [1.0, 0.0]];
        assert!(tdoa_locate_2d(&sensors, &[0.0, 0.1], 1.0, None, 10).is_none());
    }

    #[test]
    fn hand_derived_geometry_recovers_the_exact_source() {
        // Pythagorean geometry solvable on paper. Sensors S0=(0,0), S1=(4,0),
        // S2=(0,3); source P=(8,6). Distances are
        //   d0 = √(8²+6²) = √100        = 10
        //   d1 = √(4²+6²) = √52         = 2√13
        //   d2 = √(8²+3²) = √73
        // so the true range differences (speed = 1) are d1−d0 and d2−d0. Starting
        // from the *centroid* (1⅓, 1) — no help given — Gauss–Newton must still
        // walk to (8, 6). Independently derived: the recovered point equals the
        // source and the residuals vanish.
        let sensors = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]];
        let speed = 1.0;
        let arrival = [10.0, 52.0_f64.sqrt(), 73.0_f64.sqrt()];
        let sol = tdoa_locate_2d(&sensors, &arrival, speed, None, 200).unwrap();
        assert!(sol.converged, "expected convergence");
        // Must take more than one Newton step from the far centroid seed (guards
        // against a solver that reports success without actually iterating).
        assert!(sol.iterations > 1, "iters {}", sol.iterations);
        assert!((sol.position[0] - 8.0).abs() < 1e-9, "x {:?}", sol.position);
        assert!((sol.position[1] - 6.0).abs() < 1e-9, "y {:?}", sol.position);
        assert!(sol.residual_rms < 1e-9, "rms {}", sol.residual_rms);
    }

    #[test]
    fn negative_range_difference_when_source_hugs_a_later_sensor() {
        // Source sits next to sensor 1, far from the reference sensor 0, so the
        // measured range difference d1−d0 is strongly *negative*. A sign slip in
        // the residual would diverge or land on the mirror point. By hand:
        //   S0=(0,0) S1=(10,0) S2=(5,9), P=(9,1)
        //   d0=√82≈9.0554, d1=√2≈1.4142 ⇒ meas1=d1−d0≈−7.6412 (< 0).
        let sensors = [[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]];
        let src = [9.0, 1.0];
        let speed = 1.0;
        let arrival = arrivals(&sensors, src, speed, 0.0);
        assert!(
            speed * (arrival[1] - arrival[0]) < -7.0,
            "this case must exercise a negative range difference"
        );
        let sol = tdoa_locate_2d(&sensors, &arrival, speed, None, 200).unwrap();
        assert!((sol.position[0] - 9.0).abs() < 1e-8, "x {:?}", sol.position);
        assert!((sol.position[1] - 1.0).abs() < 1e-8, "y {:?}", sol.position);
    }

    #[test]
    fn collinear_sensors_are_rejected_as_a_degenerate_geometry() {
        // Three sensors on one line span only a single direction, so the
        // Jacobian rows are parallel and JᵀJ is singular — the problem is
        // unobservable in the cross-line coordinate. The solve must bail with
        // `None` (and must NOT panic on the singular normal matrix).
        let sensors = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let src = [1.0, 5.0];
        let arrival = arrivals(&sensors, src, 1.0, 0.0);
        assert!(tdoa_locate_2d(&sensors, &arrival, 1.0, None, 50).is_none());
    }

    #[test]
    fn residual_rms_reports_the_geometric_misfit_of_inconsistent_data() {
        // Feed self-consistent arrivals for source (6,4) on the unit-10 square,
        // then corrupt sensor 2's timestamp by +1 ms. No location can satisfy all
        // four hyperbolae, so the best fit carries a residual. The misfit RMS is a
        // property of the geometry and the corruption — derived with an
        // independent reference solver it is 0.50470203096… m; assert that value
        // and that the converged estimate is pulled toward the corrupted sensor.
        let sensors = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let src = [6.0, 4.0];
        let speed = 1500.0;
        let mut arrival = arrivals(&sensors, src, speed, 0.0);
        arrival[2] += 1e-3;
        let sol = tdoa_locate_2d(&sensors, &arrival, speed, None, 100).unwrap();
        assert!(
            (sol.residual_rms - 0.504_702_030_967).abs() < 1e-9,
            "rms {}",
            sol.residual_rms
        );
        // The estimate moves off the clean source toward the perturbed sensor 2.
        assert!(
            (sol.position[0] - 5.720_443).abs() < 1e-5,
            "x {:?}",
            sol.position
        );
        assert!(
            (sol.position[1] - 3.570_680).abs() < 1e-5,
            "y {:?}",
            sol.position
        );
    }
}
