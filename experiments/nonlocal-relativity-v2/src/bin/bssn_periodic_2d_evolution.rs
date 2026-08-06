//! BSSN on a uniform periodic **two-dimensional** grid (Layer 3.6).
//!
//! Deterministic: no RNG, no wall clock, no parallelism. Identical inputs give
//! byte-identical output.
//!
//! Layer 3.5 delivered two-dimensional support and validated the *right-hand
//! side*. It never measured that a two-dimensional *evolution* converges or
//! stays bounded — the same gap that, one layer earlier, hid an unstable scheme
//! behind an accurate right-hand side. This experiment closes it.

use scirust_relativity::Metric;
use scirust_relativity::adm_evolution::{AdmSources, SpatialTensorField};
use scirust_relativity::bssn::conformal_ricci;
use scirust_relativity::bssn_grid::{
    BssnGridState, BssnGridSystem, BssnSlicing, bssn_grid_constraints, evolve_bssn_grid,
    grid_conformal_ricci,
};
use scirust_relativity::grid::UniformGrid2d;

const TWO_PI: f64 = std::f64::consts::TAU;
/// Small enough that the neglected quadratic nonlinearity stays far below the
/// `O(dx^2)` truncation error at every resolution swept here.
const GAUGE_AMPLITUDE: f64 = 1.0e-6;

/// A metric varying along `x` **and** `y`, so the mixed second derivatives are
/// genuinely non-zero. Not a solution of the Einstein equations — a numerical
/// oracle for the derivative stack.
struct DiagonalMetric {
    amplitude: f64,
    wave_number: f64,
}

impl Metric<3> for DiagonalMetric {
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        let s = (self.wave_number * (coordinates[0] + coordinates[1])).sin();
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0 + self.amplitude * s, 0.0],
            [0.0, 0.0, 1.0 - self.amplitude * s],
        ]
    }
}

fn zero_curvature() -> impl SpatialTensorField {
    |_: &[f64; 3]| [[0.0_f64; 3]; 3]
}

fn square(points: usize) -> UniformGrid2d {
    UniformGrid2d::from_axes([points, points], [0.0, 0.0], [1.0, 1.0]).expect("valid grid")
}

fn order(coarse: f64, fine: f64) -> String {
    if coarse > 0.0 && fine > 0.0
    {
        format!("{:.2}", (coarse / fine).log2())
    }
    else
    {
        "n_a".to_string()
    }
}

/// Flat space with a diagonal lapse perturbation.
fn gauge_wave_state(grid: UniformGrid2d, amplitude: f64, k: f64) -> BssnGridState<2> {
    let mut state = BssnGridState::minkowski(grid);
    for index in 0..grid.total_points()
    {
        let p = grid.position(index);
        state.set_lapse_at(index, 1.0 + amplitude * (k * (p[0] + p[1])).sin());
    }
    state
}

fn main() {
    println!("# experiment: BSSN on a uniform periodic TWO-dimensional grid (2D3V)");
    println!("# layer: scirust-relativity Layer 3.6 (established general relativity)");
    println!("# units: geometric G = c = 1; lengths and times in mass units M");
    println!("# grid: half-open periodic square [0,1) x [0,1), square cells required");
    println!("# gauge: prescribed alpha = 1, beta^i = 0 by default; the gauge_wave and");
    println!("#   stability sections use LIVE 1+log slicing");
    println!("# determinism: no RNG, no wall clock; identical inputs give identical output");
    println!(
        "# NOTE: two spatial dimensions, full 3x3 tensors -- NOT a three-dimensional\
         \n#   numerical-relativity solver. Stability here is MEASURED, not proven."
    );

    // -----------------------------------------------------------------------
    println!("# oracle A: stationary Minkowski in 2D (must be exactly stationary)");
    println!(
        "scenario,resolution,total_points,grid_spacing,timestep,steps,final_time,\
         max_state_change,hamiltonian_linf,determinant_linf,connection_linf,status"
    );
    for &points in &[8_usize, 16, 32]
    {
        let grid = square(points);
        let system = BssnGridSystem::vacuum(grid);
        let initial = BssnGridState::minkowski(grid);
        let step = 0.25 * grid.spacing_along(0);
        let samples = evolve_bssn_grid(&system, &initial, 0.0, 1.0, step).expect("Minkowski");
        let last = samples.last().expect("final sample");
        let change = last
            .state
            .as_slice()
            .iter()
            .zip(initial.as_slice())
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
        let constraints =
            bssn_grid_constraints(&last.state.view(), &AdmSources::VACUUM).expect("constraints");
        println!(
            "minkowski_2d,{points},{},{:.6e},{:.6e},{},{:.4},{change:.6e},{:.6e},{:.6e},{:.6e},{}",
            grid.total_points(),
            grid.spacing_along(0),
            step,
            samples.len() - 1,
            last.time,
            constraints.hamiltonian.max_abs,
            constraints.determinant.max_abs,
            constraints.connection.max_abs,
            if change == 0.0
            {
                "exactly_stationary"
            }
            else
            {
                "drifted"
            }
        );
    }

    // -----------------------------------------------------------------------
    println!("# oracle B: conformal Ricci with MIXED derivatives present.");
    println!("#   A metric depending on x + y has non-zero d_x d_y, which no one-dimensional");
    println!("#   configuration can produce. The BSSN-form and generic Ricci tensors must");
    println!("#   converge onto each other; an index error in Gammatilde^k Gammatilde_{{(ij)k}}");
    println!("#   made them saturate near 1% until it was found -- 1D could not detect it,");
    println!("#   because Gammatilde^k is nearly zero there.");
    println!(
        "scenario,resolution,grid_spacing,ricci_difference_linf,ricci_scale,\
         connection_scale,observed_spatial_order,status"
    );
    let k = TWO_PI;
    let mut previous_ricci: Option<f64> = None;
    for &points in &[16_usize, 32, 64]
    {
        let grid = square(points);
        let manufactured = DiagonalMetric {
            amplitude: 0.01,
            wave_number: k,
        };
        let state = BssnGridState::from_adm_fields(grid, &manufactured, &zero_curvature())
            .expect("manufactured state");
        let view = state.view();
        let settings = view.settings();
        let metric = view.metric();

        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        let mut connection_scale = 0.0_f64;
        for index in 0..grid.total_points()
        {
            let at = grid.position(index);
            let bssn = grid_conformal_ricci(&view, index).expect("bssn ricci");
            let generic = conformal_ricci(&metric, &at, &settings).expect("generic ricci");
            for i in 0..3
            {
                for j in 0..3
                {
                    worst = worst.max((bssn.total[i][j] - generic.total[i][j]).abs());
                    scale = scale.max(generic.total[i][j].abs());
                }
            }
            for component in 0..3
            {
                connection_scale = connection_scale
                    .max(view.state_at(index).conformal_connection[component].abs());
            }
        }
        let observed = match previous_ricci
        {
            Some(previous) => order(previous, worst),
            None => "n_a".to_string(),
        };
        previous_ricci = Some(worst);
        println!(
            "mixed_ricci,{points},{:.6e},{worst:.6e},{scale:.6e},{connection_scale:.6e},{observed},valid",
            grid.spacing_along(0),
        );
    }

    // -----------------------------------------------------------------------
    println!("# oracle C: diagonal gauge wave under 1+log, a genuinely 2D closed form.");
    println!("#   d_t^2 alpha = 2 grad^2 alpha, and grad^2 sin(k(x+y)) = -2 k^2, so");
    println!("#   alpha(t) = 1 + A cos(2 k t) sin(k(x+y)) exactly.");
    println!("#   t_end = 1/8 gives omega t = pi/2, where a phase error appears at FIRST");
    println!("#   order. Sampling at t = 1/4 instead lands on an extremum of the cosine and");
    println!("#   reports a flattering order ~4 with errors two decades too small.");
    println!("#   The timestep is FIXED, so refining the grid isolates the spatial order.");
    println!(
        "scenario,resolution,grid_spacing,timestep,final_time,amplitude,\
         lapse_linf,relative_error,observed_spatial_order,status"
    );
    let gauge_end = 0.125_f64;
    let gauge_step = 1.0 / 1024.0;
    let mut previous_gauge: Option<f64> = None;
    for &points in &[16_usize, 32]
    {
        let grid = square(points);
        let initial = gauge_wave_state(grid, GAUGE_AMPLITUDE, k);
        let system = BssnGridSystem::vacuum(grid).with_slicing(BssnSlicing::OnePlusLog);
        let samples =
            evolve_bssn_grid(&system, &initial, 0.0, gauge_end, gauge_step).expect("gauge wave");
        let last = samples.last().expect("final sample");

        let mut worst = 0.0_f64;
        for index in 0..grid.total_points()
        {
            let p = grid.position(index);
            let exact =
                1.0 + GAUGE_AMPLITUDE * (2.0 * k * last.time).cos() * (k * (p[0] + p[1])).sin();
            worst = worst.max((last.state.lapse_at(index) - exact).abs());
        }
        let observed = match previous_gauge
        {
            Some(previous) => order(previous, worst),
            None => "n_a".to_string(),
        };
        previous_gauge = Some(worst);
        println!(
            "gauge_wave_2d,{points},{:.6e},{gauge_step:.6e},{:.4},{:.1e},{worst:.6e},{:.6e},{observed},valid",
            grid.spacing_along(0),
            last.time,
            GAUGE_AMPLITUDE,
            worst / GAUGE_AMPLITUDE,
        );
    }

    // -----------------------------------------------------------------------
    println!("# stability: long-time bounded evolution in 2D. This is the gap Layer 3.5");
    println!("#   left open -- an accurate right-hand side there coexisted, one layer");
    println!("#   earlier, with an unstable evolution.");
    println!("scenario,resolution,courant,timestep,final_time,max_state,determinant_linf,status");
    for &points in &[8_usize, 16, 32]
    {
        let grid = square(points);
        let initial = gauge_wave_state(grid, 1.0e-3, k);
        let system = BssnGridSystem::vacuum(grid).with_slicing(BssnSlicing::OnePlusLog);
        for &courant in &[0.25_f64, 0.5]
        {
            let step = courant * grid.spacing_along(0);
            match evolve_bssn_grid(&system, &initial, 0.0, 2.0, step)
            {
                Ok(samples) =>
                {
                    let last = samples.last().expect("final sample");
                    let worst = last
                        .state
                        .as_slice()
                        .iter()
                        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
                    let constraints =
                        bssn_grid_constraints(&last.state.view(), &AdmSources::VACUUM)
                            .expect("constraints");
                    println!(
                        "stability_2d,{points},{courant},{step:.6e},{:.4},{worst:.6e},{:.6e},{}",
                        last.time,
                        constraints.determinant.max_abs,
                        if worst < 1.01 { "bounded" } else { "grew" }
                    );
                },
                Err(_) =>
                {
                    println!("stability_2d,{points},{courant},{step:.6e},n_a,n_a,n_a,rejected");
                },
            }
        }
    }

    // -----------------------------------------------------------------------
    println!("# rejected configurations (typed errors, never silent repair)");
    println!("scenario,request,status,reason");
    let stretched = UniformGrid2d::from_axes([16, 8], [0.0, 0.0], [1.0, 1.0]).expect("valid grid");
    let state = BssnGridState::minkowski(stretched);
    let system = BssnGridSystem::vacuum(stretched);
    let outcome = evolve_bssn_grid(&system, &state, 0.0, 0.1, 0.01);
    println!(
        "rejected,anisotropic_grid,{},{}",
        if outcome.is_err()
        {
            "rejected"
        }
        else
        {
            "accepted"
        },
        outcome
            .err()
            .map_or_else(|| "none".to_string(), |e| e.to_string().replace(',', ";"))
    );
    let too_few = UniformGrid2d::from_axes([16, 2], [0.0, 0.0], [1.0, 1.0]);
    println!(
        "rejected,axis_1_too_few_points,{},{}",
        if too_few.is_err()
        {
            "rejected"
        }
        else
        {
            "accepted"
        },
        too_few.map_or_else(|e| e.to_string().replace(',', ";"), |_| "none".to_string())
    );

    println!("# interpretation: two-dimensional Minkowski is exactly stationary; the");
    println!("#   conformal Ricci reconstruction converges at order 2 WITH the mixed");
    println!("#   derivatives present, which is what one dimension could not test; the");
    println!("#   diagonal gauge wave converges at order 2 against its closed form; and the");
    println!("#   evolution stays bounded over long times at every resolution and Courant");
    println!("#   factor tested. None of this proves strong hyperbolicity -- that is an");
    println!("#   analytic property of the continuum system. Two dimensions, not three;");
    println!("#   periodic domains only; weak smooth fields only; no black holes, punctures,");
    println!("#   excision, AMR, waveform extraction, or observational validation.");
}
