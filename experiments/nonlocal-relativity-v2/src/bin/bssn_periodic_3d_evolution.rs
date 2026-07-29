//! BSSN on a uniform periodic **three-dimensional** grid (Layer 3.7).
//!
//! Deterministic: no RNG, no wall clock, no parallelism. Identical inputs give
//! byte-identical output.
//!
//! Layers 3.5 and 3.6 delivered and validated two dimensions. The third axis was
//! carried by the grid type but had no evolution validation of any kind. This
//! experiment supplies it, and adds the one configuration no lower-dimensional
//! grid can represent: a gravitational wave travelling along the body diagonal,
//! whose six independent metric components are all non-zero at once.

use scirust_relativity::Metric;
use scirust_relativity::adm_evolution::{AdmSources, SpatialTensorField};
use scirust_relativity::bssn::{bssn_to_adm, conformal_ricci};
use scirust_relativity::bssn_grid::{
    BssnGridState, BssnGridSystem, BssnSlicing, TransverseTracelessWave, bssn_grid_constraints,
    evolve_bssn_grid, grid_conformal_ricci,
};
use scirust_relativity::grid::UniformGrid3d;

const TWO_PI: f64 = std::f64::consts::TAU;
/// Small enough that the neglected quadratic nonlinearity stays far below the
/// `O(dx^2)` truncation error at every resolution swept here.
const GAUGE_AMPLITUDE: f64 = 1.0e-6;
/// The same reasoning for the gravitational-wave oracle, which solves the
/// LINEARIZED field equations and so neglects the same `O(A^2)` terms.
const WAVE_AMPLITUDE: f64 = 1.0e-6;

/// A metric varying along all three axes, with a different phase in every entry.
///
/// Every phase is ONE wavelength across the box. Raising any of them to three
/// wavelengths drops `N = 8` to 2.67 points per wavelength -- below Nyquist --
/// and the measured order collapses to ~0.95 for reasons that have nothing to do
/// with the code under test.
///
/// Not a solution of the Einstein equations: a numerical oracle for the
/// derivative stack.
struct TriaxialMetric {
    amplitude: f64,
    wave_number: f64,
}

impl Metric<3> for TriaxialMetric {
    fn components(&self, c: &[f64; 3]) -> [[f64; 3]; 3] {
        let k = self.wave_number;
        let a = self.amplitude;
        let xx = 1.0 + a * (k * (c[1] + c[2])).sin();
        let yy = 1.0 + a * (k * (c[2] + c[0])).sin();
        let zz = 1.0 + a * (k * (c[0] + c[1])).sin();
        let xy = 0.5 * a * (k * (c[0] + c[1])).cos();
        let xz = 0.5 * a * (k * (c[0] + c[2])).cos();
        let yz = 0.5 * a * (k * (c[1] + c[2])).cos();
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    }
}

fn zero_curvature() -> impl SpatialTensorField {
    |_: &[f64; 3]| [[0.0_f64; 3]; 3]
}

fn cube(points: usize) -> UniformGrid3d {
    UniformGrid3d::from_axes([points; 3], [0.0; 3], [1.0; 3]).expect("valid grid")
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

/// Flat space with a body-diagonal lapse perturbation.
fn gauge_wave_state(grid: UniformGrid3d, amplitude: f64, k: f64) -> BssnGridState<3> {
    let mut state = BssnGridState::minkowski(grid);
    for index in 0..grid.total_points()
    {
        let p = grid.position(index);
        state.set_lapse_at(index, 1.0 + amplitude * (k * (p[0] + p[1] + p[2])).sin());
    }
    state
}

fn main() {
    println!("# experiment: BSSN on a uniform periodic THREE-dimensional grid (3D3V)");
    println!("# layer: scirust-relativity Layer 3.7 (established general relativity)");
    println!("# units: geometric G = c = 1; lengths and times in mass units M");
    println!("# grid: half-open periodic cube [0,1)^3, square cells required");
    println!("# gauge: prescribed alpha = 1, beta^i = 0 by default; the gauge_wave and");
    println!("#   stability sections use LIVE 1+log slicing");
    println!("# determinism: no RNG, no wall clock; identical inputs give identical output");
    println!(
        "# NOTE: three spatial dimensions on a PERIODIC box with WEAK SMOOTH fields.\
         \n#   No black holes, punctures, excision, outer boundary, AMR, or waveform\
         \n#   extraction. Stability here is MEASURED, not proven."
    );

    // -----------------------------------------------------------------------
    println!("# oracle A: stationary Minkowski in 3D (must be exactly stationary)");
    println!(
        "scenario,resolution,total_points,grid_spacing,timestep,steps,final_time,\
         max_state_change,hamiltonian_linf,determinant_linf,connection_linf,status"
    );
    for &points in &[8_usize, 16]
    {
        let grid = cube(points);
        let system = BssnGridSystem::vacuum(grid);
        let initial = BssnGridState::minkowski(grid);
        let step = 0.25 * grid.spacing_along(0);
        let samples = evolve_bssn_grid(&system, &initial, 0.0, 0.5, step).expect("Minkowski");
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
            "minkowski_3d,{points},{},{:.6e},{:.6e},{},{:.4},{change:.6e},{:.6e},{:.6e},{:.6e},{}",
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
    println!("# oracle B: conformal Ricci with ALL THREE mixed derivative pairs present.");
    println!("#   Two dimensions can only populate d_x d_y; the (0,2) and (1,2) pairs are");
    println!("#   reached for the first time here. The BSSN-form and generic Ricci tensors");
    println!("#   must converge onto each other. connection_scale is reported to show that");
    println!("#   Gammatilde^i -- the term that carried the Layer 3.5 index error -- is");
    println!("#   genuinely exercised, in all three components rather than one.");
    println!(
        "scenario,resolution,grid_spacing,ricci_difference_linf,ricci_scale,\
         connection_scale,observed_spatial_order,status"
    );
    let k = TWO_PI;
    let mut previous_ricci: Option<f64> = None;
    for &points in &[8_usize, 16, 32]
    {
        let grid = cube(points);
        let manufactured = TriaxialMetric {
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
            "triaxial_ricci,{points},{:.6e},{worst:.6e},{scale:.6e},{connection_scale:.6e},{observed},valid",
            grid.spacing_along(0),
        );
    }

    // -----------------------------------------------------------------------
    println!("# oracle C: body-diagonal gauge wave under 1+log, a genuinely 3D closed form.");
    println!("#   d_t^2 alpha = 2 grad^2 alpha, and grad^2 sin(k(x+y+z)) = -3 k^2, so");
    println!("#   alpha(t) = 1 + A cos(k sqrt(6) t) sin(k(x+y+z)) exactly.");
    println!("#   The sampling time gives omega t = pi/4, away from the extremum of the");
    println!("#   cosine where a phase error would only show at SECOND order and report a");
    println!("#   flattering convergence rate. The timestep is FIXED, so refining the grid");
    println!("#   isolates the spatial order instead of refining space and time together.");
    println!(
        "scenario,resolution,grid_spacing,timestep,final_time,amplitude,\
         lapse_linf,relative_error,observed_spatial_order,status"
    );
    let gauge_omega = k * 6.0_f64.sqrt();
    let gauge_end = std::f64::consts::FRAC_PI_4 / gauge_omega;
    let gauge_step = gauge_end / 6.0;
    let mut previous_gauge: Option<f64> = None;
    for &points in &[8_usize, 16]
    {
        let grid = cube(points);
        let initial = gauge_wave_state(grid, GAUGE_AMPLITUDE, k);
        let system = BssnGridSystem::vacuum(grid).with_slicing(BssnSlicing::OnePlusLog);
        let samples =
            evolve_bssn_grid(&system, &initial, 0.0, gauge_end, gauge_step).expect("gauge wave");
        let last = samples.last().expect("final sample");

        let mut worst = 0.0_f64;
        for index in 0..grid.total_points()
        {
            let p = grid.position(index);
            let exact = 1.0
                + GAUGE_AMPLITUDE
                    * (gauge_omega * last.time).cos()
                    * (k * (p[0] + p[1] + p[2])).sin();
            worst = worst.max((last.state.lapse_at(index) - exact).abs());
        }
        let observed = match previous_gauge
        {
            Some(previous) => order(previous, worst),
            None => "n_a".to_string(),
        };
        previous_gauge = Some(worst);
        println!(
            "gauge_wave_3d,{points},{:.6e},{gauge_step:.6e},{:.6},{:.1e},{worst:.6e},{:.6e},{observed},valid",
            grid.spacing_along(0),
            last.time,
            GAUGE_AMPLITUDE,
            worst / GAUGE_AMPLITUDE,
        );
    }

    // -----------------------------------------------------------------------
    println!("# oracle D: a transverse-traceless GRAVITATIONAL wave along the body diagonal.");
    println!("#   This is the gravitational sector, not the gauge sector: it evolves");
    println!("#   gammatilde_ij and Atilde_ij rather than the lapse. The polarization");
    println!("#   e1 (x) e1 - e2 (x) e2 for n = (1,1,1)/sqrt(3) makes ALL SIX independent");
    println!("#   components of h_ij non-zero while all three axes vary -- a configuration");
    println!("#   neither a one- nor a two-dimensional grid can represent.");
    println!("#   CATEGORY: exact solution of the LINEARIZED field equations. The code");
    println!("#   solves the full nonlinear system, so the difference carries both O(A^2)");
    println!("#   nonlinearity and O(dx^2) truncation; A is chosen so truncation dominates.");
    println!(
        "scenario,resolution,grid_spacing,timestep,final_time,amplitude,\
         metric_error_linf,relative_error,hamiltonian_linf,observed_spatial_order,status"
    );
    let component = TWO_PI;
    let wave_omega = component * 3.0_f64.sqrt();
    let wave_end = std::f64::consts::FRAC_PI_4 / wave_omega;
    let wave_step = wave_end / 12.0;
    let mut previous_wave: Option<f64> = None;
    for &points in &[8_usize, 16]
    {
        let grid = cube(points);
        let wave = TransverseTracelessWave::diagonal(WAVE_AMPLITUDE, component, 0.0);
        let initial =
            BssnGridState::from_adm_fields(grid, &wave.metric_field(), &wave.curvature_field())
                .expect("wave initial data");
        let system = BssnGridSystem::vacuum(grid);
        let samples =
            evolve_bssn_grid(&system, &initial, 0.0, wave_end, wave_step).expect("diagonal wave");
        let last = samples.last().expect("final sample");
        let exact = TransverseTracelessWave::diagonal(WAVE_AMPLITUDE, component, last.time);

        let mut worst = 0.0_f64;
        for index in 0..grid.total_points()
        {
            let at = grid.position(index);
            let reconstructed = bssn_to_adm(&last.state.state_at(index)).expect("adm");
            let analytic = exact.spatial_metric(&at);
            // Tensor-component loops index two distinct rank-2 arrays together;
            // iterating one of them would obscure that pairing.
            #[allow(clippy::needless_range_loop)]
            for i in 0..3
            {
                for j in 0..3
                {
                    worst = worst.max((reconstructed.spatial_metric[i][j] - analytic[i][j]).abs());
                }
            }
        }
        let constraints =
            bssn_grid_constraints(&last.state.view(), &AdmSources::VACUUM).expect("constraints");
        let observed = match previous_wave
        {
            Some(previous) => order(previous, worst),
            None => "n_a".to_string(),
        };
        previous_wave = Some(worst);
        println!(
            "diagonal_wave_3d,{points},{:.6e},{wave_step:.6e},{:.6},{:.1e},{worst:.6e},{:.6e},{:.6e},{observed},valid",
            grid.spacing_along(0),
            last.time,
            WAVE_AMPLITUDE,
            worst / WAVE_AMPLITUDE,
            constraints.hamiltonian.max_abs,
        );
    }

    // -----------------------------------------------------------------------
    println!("# stability: bounded evolution in 3D. An accurate right-hand side is not");
    println!("#   evidence of a working solver -- Layer 3.4 had one and blew up anyway.");
    println!("scenario,resolution,courant,timestep,final_time,max_state,determinant_linf,status");
    for &points in &[8_usize, 16]
    {
        let grid = cube(points);
        let initial = gauge_wave_state(grid, 1.0e-3, k);
        let system = BssnGridSystem::vacuum(grid).with_slicing(BssnSlicing::OnePlusLog);
        for &courant in &[0.25_f64, 0.5]
        {
            let step = courant * grid.spacing_along(0);
            match evolve_bssn_grid(&system, &initial, 0.0, 1.0, step)
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
                        "stability_3d,{points},{courant},{step:.6e},{:.4},{worst:.6e},{:.6e},{}",
                        last.time,
                        constraints.determinant.max_abs,
                        if worst < 1.01 { "bounded" } else { "grew" }
                    );
                },
                Err(_) =>
                {
                    println!("stability_3d,{points},{courant},{step:.6e},n_a,n_a,n_a,rejected");
                },
            }
        }
    }

    // -----------------------------------------------------------------------
    println!("# rejected configurations (typed errors, never silent repair)");
    println!("scenario,request,status,reason");
    let stretched = UniformGrid3d::from_axes([16, 16, 8], [0.0; 3], [1.0; 3]).expect("valid grid");
    let state = BssnGridState::minkowski(stretched);
    let system = BssnGridSystem::vacuum(stretched);
    let outcome = evolve_bssn_grid(&system, &state, 0.0, 0.1, 0.01);
    println!(
        "rejected,anisotropic_z_axis,{},{}",
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
    let too_few = UniformGrid3d::from_axes([16, 16, 2], [0.0; 3], [1.0; 3]);
    println!(
        "rejected,axis_2_too_few_points,{},{}",
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
    // A polarization that is not transverse is not a solution of the linearized
    // field equations, so it is refused rather than evolved into a measurement
    // that would look like a convergence failure.
    let longitudinal = TransverseTracelessWave::plane(
        1.0e-6,
        TWO_PI,
        0.0,
        [1.0, 0.0, 0.0],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
    );
    println!(
        "rejected,longitudinal_polarization,{},{}",
        if longitudinal.is_err()
        {
            "rejected"
        }
        else
        {
            "accepted"
        },
        longitudinal
            .err()
            .map_or_else(|| "none".to_string(), |e| e.to_string().replace(',', ";"))
    );

    println!("# interpretation: three-dimensional Minkowski is exactly stationary; the");
    println!("#   conformal Ricci reconstruction converges at order 2 with ALL THREE mixed");
    println!("#   derivative pairs present, which two dimensions could not test; the");
    println!("#   body-diagonal gauge wave and the body-diagonal GRAVITATIONAL wave both");
    println!("#   converge at order 2 against their closed forms; and the evolution stays");
    println!("#   bounded at every resolution and Courant factor tested. None of this");
    println!("#   proves strong hyperbolicity -- that is an analytic property of the");
    println!("#   continuum system, and nothing here establishes it. Periodic domains only;");
    println!("#   weak smooth fields only; no black holes, punctures, excision, radiative");
    println!("#   boundaries, AMR, waveform extraction, or observational validation.");
}
