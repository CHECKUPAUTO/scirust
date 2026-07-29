//! Uniform periodic grid and centred finite-difference oracles.
//!
//! Tolerances here are all derived from the second-order truncation error
//! `E ~ C (k dx)^2`, never chosen to make a test pass.

use scirust_relativity::grid::{
    GridError, GridReduction, MINIMUM_STENCIL_POINTS, UniformGrid1d, periodic_first_derivative,
    periodic_first_derivative_all, periodic_second_derivative, periodic_second_derivative_all,
};

const TWO_PI: f64 = std::f64::consts::TAU;

fn sampled(grid: &UniformGrid1d, k: f64) -> Vec<f64> {
    (0..grid.points())
        .map(|i| (k * grid.coordinate(i)).sin())
        .collect()
}

#[test]
fn grid_is_half_open_with_spacing_from_the_point_count() {
    let grid = UniformGrid1d::new(8, 0.0, 1.0).expect("valid grid");
    assert_eq!(grid.points(), 8);
    // Half-open: spacing divides by N, not N-1, because x_max is the periodic
    // image of x_min and is not stored.
    assert!((grid.spacing() - 0.125).abs() < 1.0e-15);
    assert!((grid.coordinate(0) - 0.0).abs() < 1.0e-15);
    assert!((grid.coordinate(7) - 0.875).abs() < 1.0e-15);
    // The upper bound is reported but is not a stored point.
    assert!((grid.upper() - 1.0).abs() < 1.0e-15);
    assert!((grid.length() - 1.0).abs() < 1.0e-15);
    // Index N wraps onto index 0 rather than reaching x_max.
    assert!((grid.coordinate(8) - grid.coordinate(0)).abs() < 1.0e-15);
}

#[test]
fn grid_offsets_wrap_periodically_without_underflow() {
    let grid = UniformGrid1d::new(5, -1.0, 1.0).expect("valid grid");

    // Left neighbour of index zero is the last point, not an underflow.
    assert_eq!(grid.offset(0, -1), 4);
    // Right neighbour of the final point is index zero.
    assert_eq!(grid.offset(4, 1), 0);
    // Interior indexing is unaffected by wrapping.
    assert_eq!(grid.offset(2, -1), 1);
    assert_eq!(grid.offset(2, 1), 3);
    // Offsets larger than the grid, of either sign, stay in range.
    assert_eq!(grid.offset(0, -7), 3);
    assert_eq!(grid.offset(0, 12), 2);
    assert_eq!(grid.offset(3, -13), 0);
    // Wrapped and direct interior indexing agree.
    for index in 0..grid.points()
    {
        assert_eq!(grid.offset(index, 0), index);
        assert_eq!(grid.offset(index, 5), index);
        assert_eq!(grid.offset(index, -5), index);
    }
}

#[test]
fn nearest_index_snaps_samples_onto_grid_points() {
    let grid = UniformGrid1d::new(16, 0.0, 1.0).expect("valid grid");
    let dx = grid.spacing();

    for index in 0..grid.points()
    {
        let x = grid.coordinate(index);
        assert_eq!(grid.nearest_index(&[x, 0.0, 0.0]), index);
        // This is the property the BSSN grid provider depends on: a sample
        // taken exactly one spacing away must land on the neighbouring index
        // despite floating-point representation error.
        assert_eq!(
            grid.nearest_index(&[x + dx, 0.0, 0.0]),
            grid.offset(index, 1)
        );
        assert_eq!(
            grid.nearest_index(&[x - dx, 0.0, 0.0]),
            grid.offset(index, -1)
        );
        assert_eq!(
            grid.nearest_index(&[x + 2.0 * dx, 0.0, 0.0]),
            grid.offset(index, 2)
        );
        assert_eq!(
            grid.nearest_index(&[x - 2.0 * dx, 0.0, 0.0]),
            grid.offset(index, -2)
        );
    }

    // Coordinates far outside the domain wrap rather than saturate.
    assert_eq!(grid.nearest_index(&[1.0, 0.0, 0.0]), 0);
    assert_eq!(grid.nearest_index(&[-1.0, 0.0, 0.0]), 0);
    assert_eq!(
        grid.nearest_index(&[3.5, 0.0, 0.0]),
        grid.nearest_index(&[0.5, 0.0, 0.0])
    );
}

#[test]
fn grid_construction_rejects_invalid_domains() {
    assert_eq!(
        UniformGrid1d::new(2, 0.0, 1.0),
        Err(GridError::InsufficientPoints {
            axis: 0,
            requested: 2,
            minimum: MINIMUM_STENCIL_POINTS,
        })
    );
    assert_eq!(
        UniformGrid1d::new(8, f64::NAN, 1.0),
        Err(GridError::NonFiniteBound {
            axis: 0,
            bound: "lower",
        })
    );
    assert_eq!(
        UniformGrid1d::new(8, 0.0, f64::INFINITY),
        Err(GridError::NonFiniteBound {
            axis: 0,
            bound: "upper",
        })
    );
    // Zero-length and inverted domains are both non-positive length.
    assert_eq!(
        UniformGrid1d::new(8, 1.0, 1.0),
        Err(GridError::NonPositiveLength { axis: 0 })
    );
    assert_eq!(
        UniformGrid1d::new(8, 1.0, 0.0),
        Err(GridError::NonPositiveLength { axis: 0 })
    );
    // The smallest stencil-viable grid is accepted.
    assert!(UniformGrid1d::new(MINIMUM_STENCIL_POINTS, 0.0, 1.0).is_ok());
}

#[test]
fn derivative_helpers_reject_length_mismatch() {
    let grid = UniformGrid1d::new(8, 0.0, 1.0).expect("valid grid");
    let wrong = vec![0.0_f64; 7];
    assert_eq!(
        periodic_first_derivative(&wrong, &grid, 0),
        Err(GridError::LengthMismatch {
            expected: 8,
            actual: 7,
        })
    );
    assert_eq!(
        periodic_second_derivative(&wrong, &grid, 0),
        Err(GridError::LengthMismatch {
            expected: 8,
            actual: 7,
        })
    );
    let good = vec![0.0_f64; 8];
    let mut short_out = vec![0.0_f64; 7];
    assert!(periodic_first_derivative_all(&good, &grid, &mut short_out).is_err());
}

#[test]
fn first_derivative_matches_the_analytic_sine_oracle() {
    // f(x) = sin(kx), f'(x) = k cos(kx), exactly.
    let grid = UniformGrid1d::new(256, 0.0, 1.0).expect("valid grid");
    for harmonic in [1.0_f64, 2.0, 4.0]
    {
        let k = harmonic * TWO_PI;
        let samples = sampled(&grid, k);
        // Truncation bound for the centred first difference:
        // |E| <= |k^3 dx^2| / 6.
        let bound = k.powi(3) * grid.spacing().powi(2) / 6.0;
        for index in 0..grid.points()
        {
            let numerical = periodic_first_derivative(&samples, &grid, index).expect("derivative");
            let exact = k * (k * grid.coordinate(index)).cos();
            assert!(
                (numerical - exact).abs() <= bound,
                "k = {k}, index {index}: {numerical} vs {exact}, bound {bound}"
            );
        }
    }
}

#[test]
fn second_derivative_matches_the_analytic_sine_oracle() {
    // f(x) = sin(kx), f''(x) = -k^2 sin(kx), exactly.
    let grid = UniformGrid1d::new(256, 0.0, 1.0).expect("valid grid");
    for harmonic in [1.0_f64, 2.0, 4.0]
    {
        let k = harmonic * TWO_PI;
        let samples = sampled(&grid, k);
        // Truncation bound for the centred second difference:
        // |E| <= |k^4 dx^2| / 12.
        let bound = k.powi(4) * grid.spacing().powi(2) / 12.0;
        for index in 0..grid.points()
        {
            let numerical = periodic_second_derivative(&samples, &grid, index).expect("derivative");
            let exact = -k * k * (k * grid.coordinate(index)).sin();
            assert!(
                (numerical - exact).abs() <= bound,
                "k = {k}, index {index}: {numerical} vs {exact}, bound {bound}"
            );
        }
    }
}

#[test]
fn derivatives_converge_at_second_order() {
    // Observed order, measured over four resolutions -- not asserted from the
    // stencil formula. The wave number is fixed so only dx varies.
    let k = TWO_PI;
    let resolutions = [32_usize, 64, 128, 256];

    let mut first_errors = Vec::new();
    let mut second_errors = Vec::new();

    for &points in &resolutions
    {
        let grid = UniformGrid1d::new(points, 0.0, 1.0).expect("valid grid");
        let samples = sampled(&grid, k);
        let mut worst_first = 0.0_f64;
        let mut worst_second = 0.0_f64;
        for index in 0..grid.points()
        {
            let x = grid.coordinate(index);
            let d1 = periodic_first_derivative(&samples, &grid, index).expect("d1");
            let d2 = periodic_second_derivative(&samples, &grid, index).expect("d2");
            worst_first = worst_first.max((d1 - k * (k * x).cos()).abs());
            worst_second = worst_second.max((d2 + k * k * (k * x).sin()).abs());
        }
        first_errors.push(worst_first);
        second_errors.push(worst_second);
    }

    for errors in [&first_errors, &second_errors]
    {
        for window in errors.windows(2)
        {
            let order = (window[0] / window[1]).log2();
            // Second order means a ratio of 4 per halving, i.e. order 2. The
            // window is generous enough to absorb the mild resolution
            // dependence of the leading constant but far too tight to admit
            // first or third order.
            assert!(
                (order - 2.0).abs() < 0.05,
                "observed order {order} from errors {errors:?}"
            );
        }
    }
}

#[test]
fn whole_array_derivatives_agree_with_the_pointwise_form() {
    let grid = UniformGrid1d::new(64, -0.5, 0.5).expect("valid grid");
    let samples = sampled(&grid, 2.0 * TWO_PI);
    let mut first = vec![0.0_f64; grid.points()];
    let mut second = vec![0.0_f64; grid.points()];
    periodic_first_derivative_all(&samples, &grid, &mut first).expect("d1 all");
    periodic_second_derivative_all(&samples, &grid, &mut second).expect("d2 all");

    for index in 0..grid.points()
    {
        // Bit-for-bit: the two forms must be the same arithmetic, not merely
        // close, or the whole-array path is a second implementation.
        let point_first = periodic_first_derivative(&samples, &grid, index).expect("d1");
        let point_second = periodic_second_derivative(&samples, &grid, index).expect("d2");
        assert_eq!(first[index], point_first);
        assert_eq!(second[index], point_second);
    }
}

#[test]
fn constant_and_linear_fields_differentiate_exactly() {
    let grid = UniformGrid1d::new(16, 0.0, 1.0).expect("valid grid");

    // A constant field has exactly zero centred difference -- to the last bit,
    // because the stencil subtracts identical values.
    let constant = vec![2.5_f64; grid.points()];
    for index in 0..grid.points()
    {
        assert_eq!(
            periodic_first_derivative(&constant, &grid, index).expect("d1"),
            0.0
        );
        assert_eq!(
            periodic_second_derivative(&constant, &grid, index).expect("d2"),
            0.0
        );
    }

    // A linear field is differentiated exactly by a centred stencil at every
    // interior point. The wrap points are deliberately excluded: the field is
    // not periodic there, and a periodic stencil is right to say so.
    let slope = 3.0_f64;
    let linear: Vec<f64> = (0..grid.points())
        .map(|i| slope * grid.coordinate(i))
        .collect();
    for index in 1..grid.points() - 1
    {
        let d1 = periodic_first_derivative(&linear, &grid, index).expect("d1");
        assert!((d1 - slope).abs() < 1.0e-13);
        let d2 = periodic_second_derivative(&linear, &grid, index).expect("d2");
        assert!(d2.abs() < 1.0e-11);
    }
}

#[test]
fn reductions_are_deterministic_and_report_non_finite_samples() {
    let samples = [1.0_f64, -3.0, 2.0, -3.0];
    let reduction = GridReduction::of(&samples);
    assert!((reduction.signed_mean - (-0.75)).abs() < 1.0e-15);
    assert!((reduction.l1 - 2.25).abs() < 1.0e-15);
    assert!((reduction.l2 - (23.0_f64 / 4.0).sqrt()).abs() < 1.0e-15);
    assert!((reduction.max_abs - 3.0).abs() < 1.0e-15);
    // Ties resolve to the lowest index, deterministically.
    assert_eq!(reduction.max_index, 1);
    assert_eq!(reduction.non_finite, 0);

    // A non-finite sample is counted, not propagated: one NaN must not erase
    // every other diagnostic on the slice.
    let dirty = [1.0_f64, f64::NAN, 2.0, f64::INFINITY];
    let reduction = GridReduction::of(&dirty);
    assert_eq!(reduction.non_finite, 2);
    assert!(reduction.max_abs.is_finite());
    assert!((reduction.max_abs - 2.0).abs() < 1.0e-15);
    assert!((reduction.signed_mean - 1.5).abs() < 1.0e-15);

    // Repeated reduction of identical input is bit-identical.
    assert_eq!(GridReduction::of(&samples), GridReduction::of(&samples));
}

// ---------------------------------------------------------------------------
// Two dimensions
// ---------------------------------------------------------------------------

use scirust_relativity::grid::{
    UniformGrid2d, periodic_first_derivative_along, periodic_mixed_derivative,
    periodic_second_derivative_along,
};

fn unit_square(nx: usize, ny: usize) -> UniformGrid2d {
    UniformGrid2d::from_axes([nx, ny], [0.0, 0.0], [1.0, 1.0]).expect("valid grid")
}

#[test]
fn two_dimensional_indexing_round_trips_with_axis_zero_fastest() {
    let grid = unit_square(4, 3);
    assert_eq!(grid.dimensions(), 2);
    assert_eq!(grid.total_points(), 12);
    assert_eq!(grid.stride(0), 1);
    assert_eq!(grid.stride(1), 4);

    // Axis 0 varies fastest, so a 1D grid's linear index is its point index --
    // which is what makes the D = 1 behaviour unchanged by the generalisation.
    for linear in 0..grid.total_points()
    {
        let multi = grid.multi_index(linear);
        assert!(multi[0] < 4 && multi[1] < 3);
        assert_eq!(grid.linear_index(&multi), linear);
    }
    assert_eq!(grid.linear_index(&[0, 0]), 0);
    assert_eq!(grid.linear_index(&[1, 0]), 1);
    assert_eq!(grid.linear_index(&[0, 1]), 4);
    assert_eq!(grid.linear_index(&[3, 2]), 11);
}

#[test]
fn two_dimensional_neighbours_wrap_independently_on_each_axis() {
    let grid = unit_square(4, 3);

    // Wrapping on one axis must not disturb the other.
    let corner = grid.linear_index(&[0, 0]);
    assert_eq!(grid.multi_index(grid.neighbour(corner, 0, -1)), [3, 0]);
    assert_eq!(grid.multi_index(grid.neighbour(corner, 1, -1)), [0, 2]);

    let far = grid.linear_index(&[3, 2]);
    assert_eq!(grid.multi_index(grid.neighbour(far, 0, 1)), [0, 2]);
    assert_eq!(grid.multi_index(grid.neighbour(far, 1, 1)), [3, 0]);

    // Offsets larger than the grid, of either sign, stay in range.
    for linear in 0..grid.total_points()
    {
        assert_eq!(grid.neighbour(linear, 0, 4), linear);
        assert_eq!(grid.neighbour(linear, 1, -3), linear);
        assert_eq!(grid.neighbour(linear, 0, -9), grid.neighbour(linear, 0, -1));
    }
}

#[test]
fn two_dimensional_positions_and_nearest_index_agree() {
    let grid = unit_square(8, 8);
    let dx = grid.spacing_along(0);
    let dy = grid.spacing_along(1);

    for linear in 0..grid.total_points()
    {
        let position = grid.position(linear);
        assert_eq!(grid.nearest_index(&position), linear);
        // A sample one spacing away on either axis must land on that neighbour.
        let along_x = [position[0] + dx, position[1], 0.0];
        let along_y = [position[0], position[1] + dy, 0.0];
        assert_eq!(grid.nearest_index(&along_x), grid.neighbour(linear, 0, 1));
        assert_eq!(grid.nearest_index(&along_y), grid.neighbour(linear, 1, 1));
        // The unused third component cannot select a different point.
        let padded = [position[0], position[1], 12.5];
        assert_eq!(grid.nearest_index(&padded), linear);
    }
}

#[test]
fn mixed_derivative_matches_its_analytic_oracle() {
    // f = sin(kx) sin(ly), so d_x d_y f = k l cos(kx) cos(ly) exactly.
    // This stencil has no one-dimensional analogue: it is the derivative BSSN's
    // principal-part restructuring exists to control, and the whole reason a
    // second spatial dimension is worth adding.
    let k = TWO_PI;
    let l = 2.0 * TWO_PI;
    let grid = unit_square(128, 128);
    let samples: Vec<f64> = (0..grid.total_points())
        .map(|linear| {
            let p = grid.position(linear);
            (k * p[0]).sin() * (l * p[1]).sin()
        })
        .collect();

    // Truncation bound for the four-corner mixed stencil:
    // |E| <= |k l| ( k^2 dx^2 + l^2 dy^2 ) / 6.
    let dx = grid.spacing_along(0);
    let dy = grid.spacing_along(1);
    let bound = (k * l).abs() * (k * k * dx * dx + l * l * dy * dy) / 6.0;

    for linear in 0..grid.total_points()
    {
        let p = grid.position(linear);
        let numerical = periodic_mixed_derivative(&samples, &grid, linear, 0, 1).expect("mixed");
        let exact = k * l * (k * p[0]).cos() * (l * p[1]).cos();
        assert!(
            (numerical - exact).abs() <= bound,
            "at {p:?}: {numerical} vs {exact}, bound {bound}"
        );
    }

    // The mixed derivative is symmetric in its two axes to the last bit. That is
    // not automatic: the four corners are the same samples either way, but
    // floating-point addition is not associative, so the implementation
    // canonicalises the axis order to guarantee it. Without that, a Hessian
    // would be asymmetric in its last bit and would carry that asymmetry into
    // `Rtilde_ij`, which must be symmetric.
    for linear in 0..grid.total_points()
    {
        assert_eq!(
            periodic_mixed_derivative(&samples, &grid, linear, 0, 1).expect("xy"),
            periodic_mixed_derivative(&samples, &grid, linear, 1, 0).expect("yx")
        );
    }
}

#[test]
fn mixed_derivative_converges_at_second_order() {
    let k = TWO_PI;
    let l = TWO_PI;
    let mut errors = Vec::new();

    for &points in &[16_usize, 32, 64, 128]
    {
        let grid = unit_square(points, points);
        let samples: Vec<f64> = (0..grid.total_points())
            .map(|linear| {
                let p = grid.position(linear);
                (k * p[0]).sin() * (l * p[1]).sin()
            })
            .collect();
        let mut worst = 0.0_f64;
        for linear in 0..grid.total_points()
        {
            let p = grid.position(linear);
            let numerical =
                periodic_mixed_derivative(&samples, &grid, linear, 0, 1).expect("mixed");
            let exact = k * l * (k * p[0]).cos() * (l * p[1]).cos();
            worst = worst.max((numerical - exact).abs());
        }
        errors.push(worst);
    }

    for window in errors.windows(2)
    {
        let order = (window[0] / window[1]).log2();
        assert!(
            (order - 2.0).abs() < 0.05,
            "observed mixed-derivative order {order} from {errors:?}"
        );
    }
}

#[test]
fn mixed_derivative_on_the_diagonal_is_the_second_derivative() {
    // Asking for d_a d_a must give the compact three-point second difference,
    // not the four-corner stencil with a doubled spacing -- so a caller can
    // build a full Hessian without special-casing the diagonal.
    let grid = unit_square(32, 32);
    let k = TWO_PI;
    let samples: Vec<f64> = (0..grid.total_points())
        .map(|linear| (k * grid.position(linear)[0]).sin())
        .collect();
    for linear in 0..grid.total_points()
    {
        assert_eq!(
            periodic_mixed_derivative(&samples, &grid, linear, 0, 0).expect("aa"),
            periodic_second_derivative_along(&samples, &grid, linear, 0).expect("d2")
        );
    }
}

#[test]
fn per_axis_derivatives_see_only_their_own_axis() {
    // A field varying only along x must have exactly zero y-derivative, and
    // exactly zero mixed derivative -- to the last bit, because the stencil
    // subtracts identical samples.
    let grid = unit_square(16, 16);
    let k = TWO_PI;
    let samples: Vec<f64> = (0..grid.total_points())
        .map(|linear| (k * grid.position(linear)[0]).sin())
        .collect();

    for linear in 0..grid.total_points()
    {
        assert_eq!(
            periodic_first_derivative_along(&samples, &grid, linear, 1).expect("dy"),
            0.0
        );
        assert_eq!(
            periodic_second_derivative_along(&samples, &grid, linear, 1).expect("dyy"),
            0.0
        );
        assert_eq!(
            periodic_mixed_derivative(&samples, &grid, linear, 0, 1).expect("dxy"),
            0.0
        );
        // And the x-derivative is unaffected by the extra dimension. The bound
        // is the centred first difference's own truncation error at this
        // resolution, |k^3 dx^2| / 6, not a round number.
        let p = grid.position(linear);
        let dx = periodic_first_derivative_along(&samples, &grid, linear, 0).expect("dx");
        let bound = k.powi(3) * grid.spacing_along(0).powi(2) / 6.0;
        assert!((dx - k * (k * p[0]).cos()).abs() <= bound);
    }
}

#[test]
fn two_dimensional_construction_rejects_invalid_axes() {
    assert_eq!(
        UniformGrid2d::from_axes([2, 8], [0.0, 0.0], [1.0, 1.0]),
        Err(GridError::InsufficientPoints {
            axis: 0,
            requested: 2,
            minimum: MINIMUM_STENCIL_POINTS,
        })
    );
    assert_eq!(
        UniformGrid2d::from_axes([8, 2], [0.0, 0.0], [1.0, 1.0]),
        Err(GridError::InsufficientPoints {
            axis: 1,
            requested: 2,
            minimum: MINIMUM_STENCIL_POINTS,
        })
    );
    // The failing axis is named, which the pre-generalisation error could not do.
    assert_eq!(
        UniformGrid2d::from_axes([8, 8], [0.0, 1.0], [1.0, 1.0]),
        Err(GridError::NonPositiveLength { axis: 1 })
    );
    assert_eq!(
        UniformGrid2d::from_axes([8, 8], [0.0, f64::NAN], [1.0, 1.0]),
        Err(GridError::NonFiniteBound {
            axis: 1,
            bound: "lower",
        })
    );
    // An axis beyond the grid's dimensionality is rejected, not wrapped.
    let grid = unit_square(8, 8);
    let samples = vec![0.0_f64; grid.total_points()];
    assert_eq!(
        periodic_first_derivative_along(&samples, &grid, 0, 2),
        Err(GridError::AxisOutOfRange {
            axis: 2,
            dimensions: 2,
        })
    );
}
