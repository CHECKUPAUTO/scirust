//! ISO 286 limits and fits — standard tolerance grades and hole/shaft fits.
//!
//! Where [`crate::interference`] treats a fit *statistically* (two random sizes
//! pitted against each other), this module is the deterministic **ISO 286**
//! standard that dimensions a drawing: the tolerance a grade `IT` allows at a
//! nominal size, the fundamental deviation a letter places the zone at, and the
//! clearance/interference a hole/shaft pairing guarantees at its extremes.
//!
//! ## Standard tolerance grades
//!
//! For a nominal diameter `D` (mm, the geometric mean of its ISO step) the
//! **standard tolerance factor** is
//!
//! ```text
//! i = 0.45·∛D + 0.001·D    (µm) ,
//! ```
//!
//! and grade `ITn` is a fixed multiple of `i` (`IT5 = 7i`, `IT6 = 10i`,
//! `IT7 = 16i`, … stepping ×10 every five grades). [`it_grade_tolerance`] returns
//! that formula value; the published tables round it to preferred numbers, so
//! expect agreement to within ~1 µm.
//!
//! ## Fundamental deviations and fits
//!
//! A tolerance zone is placed by its **fundamental deviation** (the limit nearest
//! the zero line). For the clearance-fit shaft letters `d, e, f, g, h` the upper
//! deviation `es ≤ 0` follows a closed formula ([`shaft_fundamental_deviation`]);
//! the hole-basis system fixes the hole at `H` (`EI = 0`). [`hole_basis_fit`]
//! combines an `H`-hole and such a shaft into the guaranteed clearances, and
//! [`fit_from_deviations`] classifies any pairing given explicit deviations —
//! covering the transition/interference letters that need a tabulated deviation.

use serde::{Deserialize, Serialize};

/// Standard tolerance factor `i = 0.45·∛D + 0.001·D` (µm), `D` in mm.
fn tolerance_factor(d_mm: f64) -> f64 {
    0.45 * d_mm.cbrt() + 0.001 * d_mm
}

/// Geometric-mean diameter (mm) of the ISO 286 size step containing `nominal`
/// (mm), for `0 < nominal ≤ 500`. The first step uses `√(1·3)` per the standard.
fn step_geomean(nominal_mm: f64) -> Option<f64> {
    const STEPS: [(f64, f64); 13] = [
        (1.0, 3.0),
        (3.0, 6.0),
        (6.0, 10.0),
        (10.0, 18.0),
        (18.0, 30.0),
        (30.0, 50.0),
        (50.0, 80.0),
        (80.0, 120.0),
        (120.0, 180.0),
        (180.0, 250.0),
        (250.0, 315.0),
        (315.0, 400.0),
        (400.0, 500.0),
    ];
    if nominal_mm <= 0.0 || nominal_mm > 500.0
    {
        return None;
    }
    // Each step is "over `prev_hi` up to and including `hi`" (ISO convention),
    // so a size exactly on a boundary belongs to the lower step.
    let mut prev_hi = 0.0;
    for &(lo, hi) in &STEPS
    {
        if nominal_mm > prev_hi && nominal_mm <= hi
        {
            return Some((lo * hi).sqrt());
        }
        prev_hi = hi;
    }
    None
}

/// Standard tolerance (µm) for grade `IT5..=IT18` at a nominal size (mm, ≤ 500),
/// `ITn = mₙ·i`. Returns `None` for a grade outside `5..=18` or a size outside
/// `(0, 500]`.
pub fn it_grade_tolerance(grade: u8, nominal_mm: f64) -> Option<f64> {
    // Multipliers for IT5..IT18 (×i).
    const MULT: [f64; 14] = [
        7.0, 10.0, 16.0, 25.0, 40.0, 63.0, 100.0, 160.0, 250.0, 400.0, 640.0, 1000.0, 1600.0,
        2500.0,
    ];
    if !(5..=18).contains(&grade)
    {
        return None;
    }
    let d = step_geomean(nominal_mm)?;
    Some(MULT[(grade - 5) as usize] * tolerance_factor(d))
}

/// Upper fundamental deviation `es` (µm, ≤ 0) of a clearance-fit **shaft** letter
/// among `d, e, f, g, h`, at a nominal size (mm). `h` is `0` (zero line); the
/// others follow the ISO 286 formulas. Returns `None` for another letter or an
/// out-of-range size (transition/interference letters need a tabulated value —
/// pass it to [`fit_from_deviations`]).
pub fn shaft_fundamental_deviation(letter: char, nominal_mm: f64) -> Option<f64> {
    let d = step_geomean(nominal_mm)?;
    let es = match letter
    {
        'h' => 0.0,
        'g' => -2.5 * d.powf(0.34),
        'f' => -5.5 * d.powf(0.41),
        'e' => -11.0 * d.powf(0.41),
        'd' => -16.0 * d.powf(0.44),
        _ => return None,
    };
    Some(es)
}

/// The three fit categories of ISO 286.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitType {
    /// Always assembles with play — minimum clearance `≥ 0`.
    Clearance,
    /// Either clearance or interference depending on the actual sizes.
    Transition,
    /// Always assembles with interference — maximum clearance `≤ 0`.
    Interference,
}

/// A hole/shaft fit: its extreme clearances (µm) and category. A negative
/// clearance is interference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Fit {
    /// Maximum clearance `ES_hole − ei_shaft` (µm).
    pub max_clearance: f64,
    /// Minimum clearance `EI_hole − es_shaft` (µm); negative ⇒ interference.
    pub min_clearance: f64,
    /// Fit category.
    pub fit_type: FitType,
}

/// Classify a fit from explicit deviations (µm): hole `[hole_lower, hole_upper]`
/// = `[EI, ES]`, shaft `[shaft_lower, shaft_upper]` = `[ei, es]`. Works for any
/// letters, including transition/interference zones.
pub fn fit_from_deviations(
    hole_lower: f64,
    hole_upper: f64,
    shaft_lower: f64,
    shaft_upper: f64,
) -> Fit {
    let max_clearance = hole_upper - shaft_lower;
    let min_clearance = hole_lower - shaft_upper;
    let fit_type = if min_clearance >= 0.0
    {
        FitType::Clearance
    }
    else if max_clearance <= 0.0
    {
        FitType::Interference
    }
    else
    {
        FitType::Transition
    };
    Fit {
        max_clearance,
        min_clearance,
        fit_type,
    }
}

/// Analyse a hole-basis fit `H<hole_grade>/<letter><shaft_grade>` at a nominal
/// size (mm), e.g. `hole_basis_fit(25.0, 7, 6, 'g')` for `H7/g6`. The hole is
/// `H` (`EI = 0`, `ES = +IT`); the shaft zone is `[es − IT, es]` with `es` the
/// [`shaft_fundamental_deviation`]. Returns `None` for an unsupported shaft
/// letter (`d..h` only) or out-of-range grade/size.
pub fn hole_basis_fit(
    nominal_mm: f64,
    hole_grade: u8,
    shaft_grade: u8,
    shaft_letter: char,
) -> Option<Fit> {
    let it_hole = it_grade_tolerance(hole_grade, nominal_mm)?;
    let it_shaft = it_grade_tolerance(shaft_grade, nominal_mm)?;
    let es = shaft_fundamental_deviation(shaft_letter, nominal_mm)?;
    Some(fit_from_deviations(0.0, it_hole, es - it_shaft, es))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn it_grades_match_published_tables() {
        // ISO 286-1 reference values (µm), within the formula's ~1 µm rounding.
        assert!((it_grade_tolerance(7, 25.0).unwrap() - 21.0).abs() < 1.0); // IT7 18–30
        assert!((it_grade_tolerance(6, 25.0).unwrap() - 13.0).abs() < 1.0); // IT6 18–30
        assert!((it_grade_tolerance(7, 40.0).unwrap() - 25.0).abs() < 1.0); // IT7 30–50
        assert!((it_grade_tolerance(7, 5.0).unwrap() - 12.0).abs() < 1.0); // IT7 3–6
        // Every five grades multiplies the tolerance by ~10.
        let a = it_grade_tolerance(6, 25.0).unwrap();
        let b = it_grade_tolerance(11, 25.0).unwrap();
        assert_relative_eq!(b / a, 10.0, epsilon = 1e-9);
        assert!(it_grade_tolerance(4, 25.0).is_none());
        assert!(it_grade_tolerance(7, 600.0).is_none());
    }

    #[test]
    fn shaft_deviations_match_iso() {
        // ISO 286 fundamental deviations at 18–30 mm (µm), ~1 µm rounding.
        assert!((shaft_fundamental_deviation('g', 25.0).unwrap() + 7.0).abs() < 1.0);
        assert!((shaft_fundamental_deviation('f', 25.0).unwrap() + 20.0).abs() < 1.0);
        assert!((shaft_fundamental_deviation('e', 25.0).unwrap() + 40.0).abs() < 1.5);
        assert_eq!(shaft_fundamental_deviation('h', 25.0).unwrap(), 0.0);
        assert!(shaft_fundamental_deviation('z', 25.0).is_none());
    }

    #[test]
    fn h7_g6_is_a_clearance_fit() {
        // Classic running fit H7/g6 at Ø20 mm: min ~7 µm, max ~41 µm clearance.
        let fit = hole_basis_fit(20.0, 7, 6, 'g').unwrap();
        assert_eq!(fit.fit_type, FitType::Clearance);
        assert!((fit.min_clearance - 7.0).abs() < 1.5);
        assert!((fit.max_clearance - 41.0).abs() < 2.0);
        assert!(fit.max_clearance > fit.min_clearance);
    }

    #[test]
    fn h7_h6_min_clearance_is_zero() {
        // H/h: es = 0 ⇒ minimum clearance exactly 0 (line-to-line).
        let fit = hole_basis_fit(20.0, 7, 6, 'h').unwrap();
        assert_relative_eq!(fit.min_clearance, 0.0, epsilon = 1e-12);
        assert_eq!(fit.fit_type, FitType::Clearance);
    }

    #[test]
    fn explicit_interference_zone_is_classified() {
        // Hole [0, +21], shaft [+35, +48] ⇒ always interference.
        let fit = fit_from_deviations(0.0, 21.0, 35.0, 48.0);
        assert_eq!(fit.fit_type, FitType::Interference);
        assert!(fit.max_clearance < 0.0);
        // A zone straddling the hole ⇒ transition.
        let t = fit_from_deviations(0.0, 21.0, -5.0, 10.0);
        assert_eq!(t.fit_type, FitType::Transition);
    }
}
