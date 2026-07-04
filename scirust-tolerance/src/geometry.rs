//! ISO 1101 geometric characteristics (form, orientation, location) and their
//! inertial form.
//!
//! [`crate::position`] covers positional location; this module covers the rest
//! of the computable geometric tolerances — the ones defined as the width of a
//! zone containing an extracted feature:
//!
//! - **Form** — [`straightness`] (a line), [`flatness`] (a surface),
//!   [`roundness`] (a circle), [`cylindricity`] (an axis) : the peak-to-valley
//!   range of the deviations from the least-squares reference feature.
//! - **Orientation** — [`parallelism`], [`perpendicularity`], [`angularity`] :
//!   the zone width `L·sin(Δθ)` a feature of length `L` sweeps when its axis
//!   departs from the datum by `Δθ`.
//! - **Location / composite** — [`profile`] (deviation from a nominal profile),
//!   [`total_runout`] (the full-indicator range about a datum axis).
//!
//! Each form characteristic also has an **inertial** reading (`*_inertia`): the
//! RMS of the same deviations, `√((1/n) Σ dⱼ²)` — the [`crate::form`] surface
//! inertia of that specific geometric feature, so a form defect can be
//! toleranced by inertia instead of by peak-to-valley range, which is far less
//! sensitive to a single outlier point.
//!
//! Reference fits are **least-squares** (the Gaussian / L2 associated feature of
//! ISO 5459 / ISO 12781). The strict minimum-zone (Chebyshev) value is never
//! larger; least-squares is the estimator most CMM software reports and the one
//! that ties cleanly to inertia.

/// Root-mean-square of a deviation set, `√((1/n) Σ dⱼ²)`; 0 for empty input.
fn rms(d: &[f64]) -> f64 {
    if d.is_empty()
    {
        return 0.0;
    }
    (d.iter().map(|x| x * x).sum::<f64>() / d.len() as f64).sqrt()
}

/// Peak-to-valley range `max − min` of a deviation set; 0 for empty input.
fn range(d: &[f64]) -> f64 {
    if d.is_empty()
    {
        return 0.0;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in d
    {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    hi - lo
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Solve a 3×3 linear system by Cramer's rule; `None` if (near-)singular.
fn solve3(m: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let d = det3(m);
    if d.abs() < 1e-14
    {
        return None;
    }
    let mut out = [0.0; 3];
    for (k, slot) in out.iter_mut().enumerate()
    {
        let mut mk = *m;
        for row in 0..3
        {
            mk[row][k] = b[row];
        }
        *slot = det3(&mk) / d;
    }
    Some(out)
}

/// Least-squares line `y = a + b·x` through 2D points, or `None` if fewer than
/// two distinct abscissae. Returns `(a, b)`.
pub fn least_squares_line(points: &[[f64; 2]]) -> Option<(f64, f64)> {
    let n = points.len();
    if n < 2
    {
        return None;
    }
    let nf = n as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for p in points
    {
        sx += p[0];
        sy += p[1];
        sxx += p[0] * p[0];
        sxy += p[0] * p[1];
    }
    let denom = nf * sxx - sx * sx;
    if denom.abs() < 1e-14
    {
        return None;
    }
    let b = (nf * sxy - sx * sy) / denom;
    let a = (sy - b * sx) / nf;
    Some((a, b))
}

fn line_residuals(points: &[[f64; 2]]) -> Vec<f64> {
    match least_squares_line(points)
    {
        Some((a, b)) => points.iter().map(|p| p[1] - (a + b * p[0])).collect(),
        None => Vec::new(),
    }
}

/// Straightness: the peak-to-valley range of the deviations of a nominally
/// straight profile from its least-squares line. 0 if unfittable.
pub fn straightness(points: &[[f64; 2]]) -> f64 {
    range(&line_residuals(points))
}

/// Inertial straightness: the RMS deviation from the least-squares line.
pub fn straightness_inertia(points: &[[f64; 2]]) -> f64 {
    rms(&line_residuals(points))
}

/// Least-squares plane `z = a + b·x + c·y` through 3D points, or `None` if the
/// points are collinear / too few. Returns `(a, b, c)`.
pub fn least_squares_plane(points: &[[f64; 3]]) -> Option<(f64, f64, f64)> {
    if points.len() < 3
    {
        return None;
    }
    let n = points.len() as f64;
    let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    let (mut sxz, mut syz) = (0.0, 0.0);
    for p in points
    {
        sx += p[0];
        sy += p[1];
        sz += p[2];
        sxx += p[0] * p[0];
        syy += p[1] * p[1];
        sxy += p[0] * p[1];
        sxz += p[0] * p[2];
        syz += p[1] * p[2];
    }
    let m = [[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]];
    let rhs = [sz, sxz, syz];
    let s = solve3(&m, &rhs)?;
    Some((s[0], s[1], s[2]))
}

fn plane_residuals(points: &[[f64; 3]]) -> Vec<f64> {
    match least_squares_plane(points)
    {
        Some((a, b, c)) => points
            .iter()
            .map(|p| p[2] - (a + b * p[0] + c * p[1]))
            .collect(),
        None => Vec::new(),
    }
}

/// Flatness: peak-to-valley range of deviations from the least-squares plane.
pub fn flatness(points: &[[f64; 3]]) -> f64 {
    range(&plane_residuals(points))
}

/// Inertial flatness: RMS deviation from the least-squares plane.
pub fn flatness_inertia(points: &[[f64; 3]]) -> f64 {
    rms(&plane_residuals(points))
}

/// Least-squares circle (Kåsa algebraic fit) through 2D points, or `None` if
/// unfittable. Returns `(center_x, center_y, radius)`.
pub fn least_squares_circle(points: &[[f64; 2]]) -> Option<(f64, f64, f64)> {
    let n = points.len();
    if n < 3
    {
        return None;
    }
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sxz, mut syz, mut sz) = (0.0, 0.0, 0.0);
    for p in points
    {
        let (x, y) = (p[0], p[1]);
        let z = x * x + y * y;
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
        sxz += x * z;
        syz += y * z;
        sz += z;
    }
    // Fit x²+y² + D x + E y + F = 0 (regress z = x²+y² on x, y, 1).
    let m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n as f64]];
    let rhs = [-sxz, -syz, -sz];
    let s = solve3(&m, &rhs)?;
    let (d, e, f) = (s[0], s[1], s[2]);
    let cx = -0.5 * d;
    let cy = -0.5 * e;
    let r2 = cx * cx + cy * cy - f;
    if r2 < 0.0
    {
        return None;
    }
    Some((cx, cy, r2.sqrt()))
}

fn circle_residuals(points: &[[f64; 2]]) -> Vec<f64> {
    match least_squares_circle(points)
    {
        Some((cx, cy, r)) => points
            .iter()
            .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt() - r)
            .collect(),
        None => Vec::new(),
    }
}

/// Roundness / circularity: peak-to-valley range of the radial deviations from
/// the least-squares circle.
pub fn roundness(points: &[[f64; 2]]) -> f64 {
    range(&circle_residuals(points))
}

/// Inertial roundness: RMS radial deviation from the least-squares circle.
pub fn roundness_inertia(points: &[[f64; 2]]) -> f64 {
    rms(&circle_residuals(points))
}

fn cylinder_residuals_axis_z(points: &[[f64; 3]]) -> Vec<f64> {
    let proj: Vec<[f64; 2]> = points.iter().map(|p| [p[0], p[1]]).collect();
    match least_squares_circle(&proj)
    {
        Some((cx, cy, r)) => points
            .iter()
            .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt() - r)
            .collect(),
        None => Vec::new(),
    }
}

/// Cylindricity for a cylinder whose axis is nominally along `z`: fits the
/// least-squares circle to the `(x, y)` projection of every point and returns
/// the peak-to-valley range of the radial deviations over all points (so it
/// captures roundness *and* taper/waviness along the axis).
pub fn cylindricity(points: &[[f64; 3]]) -> f64 {
    range(&cylinder_residuals_axis_z(points))
}

/// Inertial cylindricity (axis along `z`): RMS radial deviation over all points.
pub fn cylindricity_inertia(points: &[[f64; 3]]) -> f64 {
    rms(&cylinder_residuals_axis_z(points))
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Angle in radians between two 3D vectors, in `[0, π]`. Returns 0 if either
/// vector is null.
pub fn angle_between(u: [f64; 3], v: [f64; 3]) -> f64 {
    let (nu, nv) = (norm3(u), norm3(v));
    if nu == 0.0 || nv == 0.0
    {
        return 0.0;
    }
    let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv);
    dot.clamp(-1.0, 1.0).acos()
}

/// Angularity zone: the width `L·sin(Δθ)` swept by a feature of length `length`
/// whose axis `measured_dir` departs by `Δθ` from the direction that makes
/// `nominal_angle` (radians) with the datum `datum_dir`
/// (`Δθ = |∠(measured, datum) − nominal_angle|`).
pub fn angularity(
    measured_dir: [f64; 3],
    datum_dir: [f64; 3],
    nominal_angle: f64,
    length: f64,
) -> f64 {
    let dtheta = (angle_between(measured_dir, datum_dir) - nominal_angle).abs();
    length.abs() * dtheta.sin()
}

/// Parallelism zone `L·sin(θ)` of a feature of length `length` whose axis makes
/// angle `θ` with the datum (0 when perfectly parallel). Special case of
/// [`angularity`] with a nominal angle of 0.
pub fn parallelism(measured_dir: [f64; 3], datum_dir: [f64; 3], length: f64) -> f64 {
    angularity(measured_dir, datum_dir, 0.0, length)
}

/// Perpendicularity zone `L·|cos θ|` of a feature of length `length` whose axis
/// makes angle `θ` with the datum (0 when perfectly perpendicular). Special case
/// of [`angularity`] with a nominal angle of `π/2`.
pub fn perpendicularity(measured_dir: [f64; 3], datum_dir: [f64; 3], length: f64) -> f64 {
    angularity(measured_dir, datum_dir, std::f64::consts::FRAC_PI_2, length)
}

/// Profile tolerance (equal-bilateral zone) from signed normal deviations `dⱼ`
/// of the extracted profile/surface from its nominal: `2·max|dⱼ|`, the width of
/// the symmetric zone that contains every point. 0 for empty input.
pub fn profile(deviations: &[f64]) -> f64 {
    2.0 * deviations.iter().fold(0.0, |m, d| f64::max(m, d.abs()))
}

/// Inertial profile: RMS of the signed normal deviations from the nominal
/// profile — the form inertia of the profile defect.
pub fn profile_inertia(deviations: &[f64]) -> f64 {
    rms(deviations)
}

/// Total runout: the full-indicator range `max − min` of a set of radial (or
/// axial) indicator readings taken about the datum axis. Unlike [`roundness`]
/// it is measured against the **datum**, so it also captures eccentricity.
pub fn total_runout(readings: &[f64]) -> f64 {
    range(readings)
}

/// Runout from measured points about a datum axis passing through `datum_center`
/// (in the plane of the points): the range of the point radii `|Pⱼ − datum|`.
pub fn runout_from_points(points: &[[f64; 2]], datum_center: [f64; 2]) -> f64 {
    let radii: Vec<f64> = points
        .iter()
        .map(|p| ((p[0] - datum_center[0]).powi(2) + (p[1] - datum_center[1]).powi(2)).sqrt())
        .collect();
    range(&radii)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn perfect_features_have_zero_form_error() {
        // Collinear points ⇒ zero straightness.
        let line = [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]];
        assert_relative_eq!(straightness(&line), 0.0, epsilon = 1e-9);
        assert_relative_eq!(straightness_inertia(&line), 0.0, epsilon = 1e-9);
        // Coplanar points (z = 2 + x − y) ⇒ zero flatness.
        let plane = [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 3.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 1.0, 3.0],
        ];
        assert_relative_eq!(flatness(&plane), 0.0, epsilon = 1e-9);
        // Points exactly on a circle ⇒ zero roundness.
        let circ: Vec<[f64; 2]> = (0..12)
            .map(|k| {
                let t = k as f64 / 12.0 * std::f64::consts::TAU;
                [2.0 + 0.5 * t.cos(), -1.0 + 0.5 * t.sin()]
            })
            .collect();
        assert_relative_eq!(roundness(&circ), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn straightness_captures_a_single_bump() {
        // Flat line y=0 with one point lifted by 0.1 ⇒ range of residuals ≈ the
        // bump (the LS line barely tilts for a symmetric layout).
        let pts = [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.1], [1.0, 0.0], [2.0, 0.0]];
        assert!(straightness(&pts) > 0.05);
        assert!(straightness_inertia(&pts) <= straightness(&pts));
    }

    #[test]
    fn circle_fit_recovers_center_and_radius() {
        let circ: Vec<[f64; 2]> = (0..20)
            .map(|k| {
                let t = k as f64 / 20.0 * std::f64::consts::TAU;
                [3.0 + 1.5 * t.cos(), 4.0 + 1.5 * t.sin()]
            })
            .collect();
        let (cx, cy, r) = least_squares_circle(&circ).unwrap();
        assert_relative_eq!(cx, 3.0, epsilon = 1e-9);
        assert_relative_eq!(cy, 4.0, epsilon = 1e-9);
        assert_relative_eq!(r, 1.5, epsilon = 1e-9);
    }

    #[test]
    fn orientation_zones_are_geometric() {
        let z = [0.0, 0.0, 1.0];
        let x = [1.0, 0.0, 0.0];
        // Parallel to itself ⇒ 0; perpendicular pair ⇒ 0 perpendicularity.
        assert_relative_eq!(parallelism(z, z, 10.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(perpendicularity(x, z, 10.0), 0.0, epsilon = 1e-12);
        // 30° tilt from datum z, length 10 ⇒ parallelism zone 10·sin30° = 5.
        let tilt = [0.5, 0.0, 3.0f64.sqrt() / 2.0]; // 30° from z
        assert_relative_eq!(parallelism(tilt, z, 10.0), 5.0, epsilon = 1e-9);
        // Same tilt ⇒ perpendicularity zone 10·|cos30°| = 8.660…
        assert_relative_eq!(
            perpendicularity(tilt, z, 10.0),
            10.0 * (3.0f64.sqrt() / 2.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn profile_and_runout_are_ranges() {
        let dev = [0.02, -0.03, 0.01, -0.015];
        assert_relative_eq!(profile(&dev), 0.06, epsilon = 1e-12); // 2·max|d| = 2·0.03
        assert_relative_eq!(profile_inertia(&dev), rms(&dev), epsilon = 1e-12);
        let readings = [10.02, 9.98, 10.05, 9.99];
        assert_relative_eq!(total_runout(&readings), 0.07, epsilon = 1e-12);
    }

    #[test]
    fn cylindricity_of_a_perfect_cylinder_is_zero() {
        let mut pts = Vec::new();
        for level in 0..4
        {
            for k in 0..12
            {
                let t = k as f64 / 12.0 * std::f64::consts::TAU;
                pts.push([1.0 + 0.8 * t.cos(), -2.0 + 0.8 * t.sin(), level as f64]);
            }
        }
        assert_relative_eq!(cylindricity(&pts), 0.0, epsilon = 1e-9);
    }
}
