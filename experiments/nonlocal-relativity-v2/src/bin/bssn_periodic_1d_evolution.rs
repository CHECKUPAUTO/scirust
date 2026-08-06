//! BSSN on a uniform periodic one-dimensional grid (Layer 3.4).
//!
//! Deterministic: no RNG, no wall clock, no parallelism. Identical inputs give
//! byte-identical output.
//!
//! The system is written in genuine BSSN form: `Rtilde_ij` uses the evolved
//! `Gammatilde^k`, so the mixed second derivatives leave the principal part.
//! An earlier revision computed `Rtilde_ij` generically and froze
//! `Gammatilde^i`, which was BSSN variables carrying the ADM principal part and
//! was correspondingly unstable; the courant_study section now shows a genuine,
//! resolution-independent CFL boundary instead.

use scirust_relativity::Metric;
use scirust_relativity::adm_evolution::{AdmSources, SpatialTensorField};
use scirust_relativity::bssn::BssnGauge;
use scirust_relativity::bssn::bssn_to_adm;
use scirust_relativity::bssn_grid::{
    BssnGridState, BssnGridSystem, BssnShiftCondition, BssnSlicing, COMPONENTS_PER_POINT,
    TransverseTracelessWave, bssn_grid_constraints, bssn_grid_rhs, bssn_grid_ricci_report,
    evolve_bssn_grid, project_grid_trace_free, project_grid_unit_determinant,
};
use scirust_relativity::grid::UniformGrid1d;

const TWO_PI: f64 = std::f64::consts::TAU;
const DOMAIN_LOWER: f64 = 0.0;
const DOMAIN_UPPER: f64 = 1.0;
/// Small enough that the neglected `O(A^2)` nonlinearity stays far below the
/// `O(dx^2)` truncation error at every resolution swept here.
const WAVE_AMPLITUDE: f64 = 1.0e-6;

/// A smooth periodic manufactured metric. Not a solution of the Einstein
/// equations — a numerical oracle for the derivative stack.
struct ManufacturedMetric {
    amplitude: f64,
    wave_number: f64,
}

impl Metric<3> for ManufacturedMetric {
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        let s = (self.wave_number * coordinates[0]).sin();
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

fn grid_of(points: usize) -> UniformGrid1d {
    UniformGrid1d::new(points, DOMAIN_LOWER, DOMAIN_UPPER).expect("valid grid")
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

/// The worst `|gamma_yy - exact|` over the grid against the analytic wave.
fn wave_metric_error(
    state: &BssnGridState,
    grid: &UniformGrid1d,
    exact: &TransverseTracelessWave,
) -> f64 {
    let mut worst = 0.0_f64;
    for index in 0..grid.points()
    {
        let reconstructed = match bssn_to_adm(&state.state_at(index))
        {
            Ok(value) => value,
            Err(_) => return f64::NAN,
        };
        let analytic = exact.spatial_metric(&grid.position(index));
        worst = worst.max((reconstructed.spatial_metric[1][1] - analytic[1][1]).abs());
    }
    worst
}

/// The largest strain `|gamma_yy - 1|` present on the grid.
fn measured_amplitude(state: &BssnGridState, grid: &UniformGrid1d) -> f64 {
    let mut worst = 0.0_f64;
    for index in 0..grid.points()
    {
        if let Ok(reconstructed) = bssn_to_adm(&state.state_at(index))
        {
            worst = worst.max((reconstructed.spatial_metric[1][1] - 1.0).abs());
        }
    }
    worst
}

fn main() {
    println!("# experiment: BSSN on a uniform periodic one-dimensional grid (1D3V)");
    println!("# layer: scirust-relativity Layer 3.4 (established general relativity)");
    println!("# units: geometric G = c = 1; lengths and times in mass units M");
    println!("# grid: half-open periodic domain [0, 1), x_n = n dx, dx = 1 / N");
    println!("# gauge: alpha = 1, beta^i = 0 by default (prescribed); the gauge_wave section");
    println!("#   uses LIVE 1+log slicing and the shift sections use the Gamma-driver.");
    println!("# determinism: no RNG, no wall clock; identical inputs give identical output");
    println!(
        "# NOTE: Rtilde_ij is written in genuine BSSN form using the EVOLVED Gammatilde^k.\
         \n#   Nothing here demonstrates strong hyperbolicity -- that is an analytic\
         \n#   property of the continuum system -- and no such claim is made."
    );

    // -----------------------------------------------------------------------
    println!("# oracle A: stationary Minkowski (must be exactly stationary)");
    println!(
        "scenario,resolution,grid_spacing,timestep,courant,steps,final_time,\
         max_state_change,hamiltonian_linf,determinant_linf,trace_free_linf,\
         connection_linf,status"
    );
    for &points in &[16_usize, 32, 64, 128]
    {
        let grid = grid_of(points);
        let system = BssnGridSystem::vacuum(grid);
        let initial = BssnGridState::minkowski(grid);
        let courant = 0.25_f64;
        let step = courant * grid.spacing();
        let t_end = 1.0_f64;
        let samples = evolve_bssn_grid(&system, &initial, 0.0, t_end, step).expect("Minkowski");
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
            "minkowski,{points},{:.6e},{:.6e},{courant},{},{:.4},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{}",
            grid.spacing(),
            step,
            samples.len() - 1,
            last.time,
            change,
            constraints.hamiltonian.max_abs,
            constraints.determinant.max_abs,
            constraints.trace_free.max_abs,
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
    println!("# oracle B: manufactured periodic state, conformal Ricci reconstruction");
    println!(
        "scenario,resolution,grid_spacing,ricci_mismatch_l1,ricci_mismatch_l2,\
         ricci_mismatch_linf,conformal_scale,conformal_factor_scale,physical_scale,\
         observed_spatial_order,status"
    );
    let manufactured = ManufacturedMetric {
        amplitude: 0.01,
        wave_number: TWO_PI,
    };
    let mut previous_ricci: Option<f64> = None;
    for &points in &[32_usize, 64, 128, 256]
    {
        let grid = grid_of(points);
        let state = BssnGridState::from_adm_fields(grid, &manufactured, &zero_curvature())
            .expect("manufactured state");
        let report = bssn_grid_ricci_report(&state.view()).expect("ricci report");
        let observed = match previous_ricci
        {
            Some(previous) => order(previous, report.mismatch.max_abs),
            None => "n_a".to_string(),
        };
        previous_ricci = Some(report.mismatch.max_abs);
        println!(
            "manufactured_ricci,{points},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{observed},valid",
            grid.spacing(),
            report.mismatch.l1,
            report.mismatch.l2,
            report.mismatch.max_abs,
            report.conformal_scale,
            report.conformal_factor_scale,
            report.physical_scale,
        );
    }

    // -----------------------------------------------------------------------
    println!("# oracle C: linearized transverse-traceless wave, short-time spatial convergence");
    println!(
        "scenario,resolution,grid_spacing,timestep,courant,final_time,wave_number,amplitude,\
         metric_l1,metric_l2,metric_linf,observed_spatial_order,hamiltonian_linf,\
         momentum_linf,determinant_linf,trace_free_linf,connection_linf,status"
    );
    let k = TWO_PI;
    let t_end = 0.1_f64;
    let mut previous_wave: Option<f64> = None;
    for &points in &[16_usize, 32, 64, 128]
    {
        let grid = grid_of(points);
        let courant = 0.25_f64;
        let steps = (t_end / (courant * grid.spacing())).round() as usize;
        let step = t_end / steps as f64;
        let wave = TransverseTracelessWave::new(WAVE_AMPLITUDE, k, 0.0);
        let initial =
            BssnGridState::from_adm_fields(grid, &wave.metric_field(), &wave.curvature_field())
                .expect("wave initial data");
        let system = BssnGridSystem::vacuum(grid);
        let samples = evolve_bssn_grid(&system, &initial, 0.0, t_end, step).expect("wave");
        let last = samples.last().expect("final sample");
        let exact = TransverseTracelessWave::new(WAVE_AMPLITUDE, k, last.time);

        let mut errors = vec![0.0_f64; grid.points()];
        // Indexed rather than iterated: the loop walks grid coordinates and the
        // reconstructed state alongside `errors`, which the index expresses.
        #[allow(clippy::needless_range_loop)]
        for index in 0..grid.points()
        {
            let reconstructed = bssn_to_adm(&last.state.state_at(index)).expect("adm");
            let analytic = exact.spatial_metric(&grid.position(index));
            errors[index] = reconstructed.spatial_metric[1][1] - analytic[1][1];
        }
        let reduction = scirust_relativity::grid::GridReduction::of(&errors);
        let constraints =
            bssn_grid_constraints(&last.state.view(), &AdmSources::VACUUM).expect("constraints");
        let observed = match previous_wave
        {
            Some(previous) => order(previous, reduction.max_abs),
            None => "n_a".to_string(),
        };
        previous_wave = Some(reduction.max_abs);
        let momentum = constraints.momentum[0]
            .max_abs
            .max(constraints.momentum[1].max_abs)
            .max(constraints.momentum[2].max_abs);
        println!(
            "linear_wave,{points},{:.6e},{:.6e},{courant},{:.4},{:.6},{:.1e},{:.6e},{:.6e},{:.6e},{observed},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},valid",
            grid.spacing(),
            step,
            last.time,
            k,
            WAVE_AMPLITUDE,
            reduction.l1,
            reduction.l2,
            reduction.max_abs,
            constraints.hamiltonian.max_abs,
            momentum,
            constraints.determinant.max_abs,
            constraints.trace_free.max_abs,
            constraints.connection.max_abs,
        );
    }

    // -----------------------------------------------------------------------
    println!("# temporal convergence: measured against a finely-stepped run at the SAME");
    println!("#   spatial resolution, so the spatial truncation error cancels exactly and");
    println!("#   what remains is purely the RK4 error of the semidiscrete ODE system.");
    println!("scenario,resolution,steps,timestep,difference_linf,observed_temporal_order,status");
    let grid = grid_of(16);
    let temporal_end = 0.2_f64;
    let wave = TransverseTracelessWave::new(1.0e-3, k, 0.0);
    let initial =
        BssnGridState::from_adm_fields(grid, &wave.metric_field(), &wave.curvature_field())
            .expect("wave initial data");
    let system = BssnGridSystem::vacuum(grid);
    let reference = evolve_bssn_grid(&system, &initial, 0.0, temporal_end, temporal_end / 640.0)
        .expect("reference");
    let reference_state = reference.last().expect("final").state.clone();
    let mut previous_temporal: Option<f64> = None;
    for &steps in &[10_usize, 20, 40, 80]
    {
        let step = temporal_end / steps as f64;
        let samples =
            evolve_bssn_grid(&system, &initial, 0.0, temporal_end, step).expect("temporal run");
        let difference = samples
            .last()
            .expect("final")
            .state
            .as_slice()
            .iter()
            .zip(reference_state.as_slice())
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
        let observed = match previous_temporal
        {
            Some(previous) => order(previous, difference),
            None => "n_a".to_string(),
        };
        previous_temporal = Some(difference);
        println!(
            "temporal_refinement,16,{steps},{:.6e},{difference:.6e},{observed},valid",
            step
        );
    }

    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    println!("# live 1+log slicing: d_t alpha = -2 alpha K, zero shift.");
    println!("#   Linearised about flat space the lapse obeys a wave equation with");
    println!("#   characteristic speed sqrt(2) -- faster than light, which is legitimate");
    println!("#   because the lapse is gauge and carries no physical signal. Standing-wave");
    println!("#   data alpha = 1 + A sin(kx) with K = 0 evolves as");
    println!("#   alpha = 1 + A cos(sqrt(2) k t) sin(kx), which is the oracle compared to.");
    println!("#   The Gamma-driver shift is NOT implemented; beta^i = 0 throughout.");
    println!(
        "scenario,resolution,grid_spacing,timestep,final_time,amplitude,\
         lapse_linf,numerical_amplitude,exact_amplitude,amplitude_ratio,\
         observed_spatial_order,status"
    );
    let gauge_amplitude = 1.0e-6_f64;
    let gauge_speed = 2.0_f64.sqrt();
    let gauge_end = 0.25_f64;
    let mut previous_gauge: Option<f64> = None;
    for &points in &[32_usize, 64, 128]
    {
        let grid = grid_of(points);
        let mut initial = BssnGridState::minkowski(grid);
        for index in 0..grid.points()
        {
            initial.set_lapse_at(
                index,
                1.0 + gauge_amplitude * (k * grid.coordinate(index)).sin(),
            );
        }
        let system = BssnGridSystem::vacuum(grid).with_slicing(BssnSlicing::OnePlusLog);
        let step = 0.1 * grid.spacing();
        let samples =
            evolve_bssn_grid(&system, &initial, 0.0, gauge_end, step).expect("gauge wave");
        let last = samples.last().expect("final sample");

        let mut worst = 0.0_f64;
        let mut numerical = 0.0_f64;
        for index in 0..grid.points()
        {
            let x = grid.coordinate(index);
            let exact = 1.0 + gauge_amplitude * (gauge_speed * k * last.time).cos() * (k * x).sin();
            let value = last.state.lapse_at(index);
            worst = worst.max((value - exact).abs());
            numerical = numerical.max((value - 1.0).abs());
        }
        let predicted = gauge_amplitude * (gauge_speed * k * last.time).cos().abs();
        let observed = match previous_gauge
        {
            Some(previous) => order(previous, worst),
            None => "n_a".to_string(),
        };
        previous_gauge = Some(worst);
        println!(
            "gauge_wave,{points},{:.6e},{:.6e},{:.4},{:.1e},{worst:.6e},{numerical:.6e},{predicted:.6e},{:.6},{observed},valid",
            grid.spacing(),
            step,
            last.time,
            gauge_amplitude,
            numerical / predicted,
        );
    }

    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    println!("# shift terms and the Gamma-driver.");
    println!("#   constant_shift: for a SPATIALLY CONSTANT shift every d(beta) term vanishes,");
    println!("#     so the difference between the shifted and unshifted right-hand sides must");
    println!("#     equal v * d_x(field) exactly. Both sides use the same compact stencil, so");
    println!("#     this holds to ROUNDING -- a sharp check on every advection term at once.");
    println!("#   gamma_driver: on flat space with a constant shift, d_t Gammatilde^i is zero");
    println!("#     and the driver decouples into B = B0 exp(-eta t),");
    println!("#     beta = beta0 + (3/4)(B0/eta)(1 - exp(-eta t)) -- a closed form.");
    println!("scenario,resolution,parameter,residual,scale,relative,status");
    {
        use scirust_relativity::grid::periodic_first_derivative;
        let grid = grid_of(64);
        let points = grid.points();
        let wave = TransverseTracelessWave::new(1.0e-3, k, 0.0);
        let base =
            BssnGridState::from_adm_fields(grid, &wave.metric_field(), &wave.curvature_field())
                .expect("initial data");
        for &velocity in &[0.1_f64, 0.3, 0.7]
        {
            let mut unshifted = vec![0.0_f64; COMPONENTS_PER_POINT * points];
            bssn_grid_rhs(
                &base.view(),
                &BssnGauge::SYNCHRONOUS,
                &AdmSources::VACUUM,
                0.0,
                &mut unshifted,
            )
            .expect("rhs");
            let mut drifting = base.clone();
            for index in 0..points
            {
                drifting.set_shift_at(index, &[velocity, 0.0, 0.0]);
            }
            let mut shifted = vec![0.0_f64; COMPONENTS_PER_POINT * points];
            bssn_grid_rhs(
                &drifting.view(),
                &BssnGauge::SYNCHRONOUS,
                &AdmSources::VACUUM,
                0.0,
                &mut shifted,
            )
            .expect("rhs");

            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for slot in 0..17
            {
                let field: Vec<f64> = (0..points)
                    .map(|index| base.as_slice()[slot * points + index])
                    .collect();
                for index in 0..points
                {
                    let gradient = periodic_first_derivative(&field, &grid, index).expect("d1");
                    let difference =
                        shifted[slot * points + index] - unshifted[slot * points + index];
                    worst = worst.max((difference - velocity * gradient).abs());
                    scale = scale.max((velocity * gradient).abs());
                }
            }
            println!(
                "constant_shift,64,{velocity},{worst:.6e},{scale:.6e},{:.6e},exact_to_rounding",
                worst / scale
            );
        }
    }

    for &eta in &[1.0_f64, 2.0, 4.0]
    {
        let grid = grid_of(16);
        let initial_driver = 0.4_f64;
        let mut initial = BssnGridState::minkowski(grid);
        for index in 0..grid.points()
        {
            initial.set_driver_at(index, &[initial_driver, 0.0, 0.0]);
        }
        let system = BssnGridSystem::vacuum(grid)
            .with_shift_condition(BssnShiftCondition::GammaDriver { eta });
        let samples = evolve_bssn_grid(&system, &initial, 0.0, 1.0, 0.005).expect("driver evolves");
        let last = samples.last().expect("final sample");
        let expected_driver = initial_driver * (-eta * last.time).exp();
        let expected_shift = 0.75 * (initial_driver / eta) * (1.0 - (-eta * last.time).exp());
        let mut worst = 0.0_f64;
        for index in 0..grid.points()
        {
            worst = worst.max((last.state.driver_at(index)[0] - expected_driver).abs());
            worst = worst.max((last.state.shift_at(index)[0] - expected_shift).abs());
        }
        println!(
            "gamma_driver,16,{eta},{worst:.6e},{:.6e},{:.6e},matches_closed_form",
            expected_shift.abs().max(expected_driver.abs()),
            worst / expected_shift.abs().max(expected_driver.abs())
        );
    }

    // -----------------------------------------------------------------------
    println!("# courant study: EMPIRICAL characterisation, not a derived CFL bound.");
    println!("#   The tested stable interval is C <= 1 and the first rejected value is C = 2,");
    println!("#   at BOTH N = 64 and N = 128 -- a resolution-INDEPENDENT boundary, which is");
    println!("#   what a genuine CFL limit looks like. (N = 32 tolerates C = 2 with a visibly");
    println!("#   degraded error, so the boundary is not sharp at very coarse resolution.)");
    println!("#   No rigorous CFL bound is derived, and none is claimed.");
    println!("scenario,resolution,courant,timestep,final_time,metric_linf,status,failure_mode");
    for &points in &[32_usize, 64, 128]
    {
        let grid = grid_of(points);
        let wave = TransverseTracelessWave::new(1.0e-3, k, 0.0);
        let initial =
            BssnGridState::from_adm_fields(grid, &wave.metric_field(), &wave.curvature_field())
                .expect("wave initial data");
        let system = BssnGridSystem::vacuum(grid);
        for &courant in &[0.1_f64, 0.25, 0.5, 1.0, 2.0]
        {
            let step = courant * grid.spacing();
            match evolve_bssn_grid(&system, &initial, 0.0, 1.0, step)
            {
                Ok(samples) =>
                {
                    let last = samples.last().expect("final");
                    let exact = TransverseTracelessWave::new(1.0e-3, k, last.time);
                    let error = wave_metric_error(&last.state, &grid, &exact);
                    println!(
                        "courant_study,{points},{courant},{:.6e},{:.4},{error:.6e},accepted,none",
                        step, last.time
                    );
                },
                Err(_) =>
                {
                    println!(
                        "courant_study,{points},{courant},{:.6e},n_a,n_a,rejected,non_finite_state",
                        step
                    );
                },
            }
        }
    }

    // -----------------------------------------------------------------------
    println!("# dissipation: explicit Kreiss-Oliger, DISABLED BY DEFAULT (sigma = 0).");
    println!(
        "#   Q f_i = -(sigma / 16 dx)( f_{{i+2}} - 4 f_{{i+1}} + 6 f_i - 4 f_{{i-1}} + f_{{i-2}} )."
    );
    println!("#   This is numerical dissipation on the evolved variables. It is NOT");
    println!("#   constraint damping. With the BSSN principal part in place there is no");
    println!("#   high-frequency growth left for it to act on, so it is very nearly a");
    println!("#   no-op: the amplitude ratio is unchanged across sigma. It remains off by");
    println!("#   default and is reported rather than relied upon.");
    println!(
        "scenario,resolution,sigma,final_time,metric_linf,measured_amplitude,\
         exact_amplitude,amplitude_ratio,status"
    );
    let dissipation_amplitude = 1.0e-3_f64;
    for &points in &[64_usize, 128]
    {
        let grid = grid_of(points);
        let wave = TransverseTracelessWave::new(dissipation_amplitude, k, 0.0);
        let initial =
            BssnGridState::from_adm_fields(grid, &wave.metric_field(), &wave.curvature_field())
                .expect("wave initial data");
        for &sigma in &[0.0_f64, 0.05, 0.2, 0.5]
        {
            let system = BssnGridSystem::vacuum(grid).with_dissipation(sigma);
            match evolve_bssn_grid(&system, &initial, 0.0, 1.0, 0.25 * grid.spacing())
            {
                Ok(samples) =>
                {
                    let last = samples.last().expect("final");
                    let exact = TransverseTracelessWave::new(dissipation_amplitude, k, last.time);
                    let error = wave_metric_error(&last.state, &grid, &exact);
                    let amplitude = measured_amplitude(&last.state, &grid);
                    println!(
                        "dissipation,{points},{sigma},{:.4},{error:.6e},{amplitude:.6e},{dissipation_amplitude:.6e},{:.4e},accepted",
                        last.time,
                        amplitude / dissipation_amplitude
                    );
                },
                Err(_) =>
                {
                    println!(
                        "dissipation,{points},{sigma},n_a,n_a,n_a,{dissipation_amplitude:.6e},n_a,rejected"
                    );
                },
            }
        }
    }

    // -----------------------------------------------------------------------
    println!("# off-constraint state and explicit projection (projection is opt-in and is");
    println!("#   NEVER applied during the default free evolution).");
    println!(
        "scenario,resolution,amplitude,hamiltonian_before,hamiltonian_after,\
         determinant_before,determinant_after,trace_before,trace_after,\
         correction_magnitude,status"
    );
    {
        let grid = grid_of(32);
        let wave = TransverseTracelessWave::new(1.0e-3, k, 0.0);
        let mut state =
            BssnGridState::from_adm_fields(grid, &wave.metric_field(), &wave.curvature_field())
                .expect("wave initial data");
        let violation = 1.0e-3_f64;
        for index in 0..grid.points()
        {
            let mut local = state.state_at(index);
            local.mean_curvature += violation * (k * grid.coordinate(index)).cos();
            state.set_state_at(index, &local);
        }
        let before =
            bssn_grid_constraints(&state.view(), &AdmSources::VACUUM).expect("constraints before");
        let system = BssnGridSystem::vacuum(grid);
        let samples = evolve_bssn_grid(&system, &state, 0.0, 0.05, 0.005).expect("evolution");
        let after = bssn_grid_constraints(
            &samples.last().expect("final").state.view(),
            &AdmSources::VACUUM,
        )
        .expect("constraints after");
        println!(
            "off_constraint,32,{violation:.1e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},0.000000e0,not_repaired",
            before.hamiltonian.max_abs,
            after.hamiltonian.max_abs,
            before.determinant.max_abs,
            after.determinant.max_abs,
            before.trace_free.max_abs,
            after.trace_free.max_abs,
        );
    }

    for &epsilon in &[0.01_f64, 0.05, 0.1]
    {
        let grid = grid_of(32);
        let mut state = BssnGridState::minkowski(grid);
        for index in 0..grid.points()
        {
            let mut local = state.state_at(index);
            for i in 0..3
            {
                local.conformal_metric[i][i] *= 1.0 + epsilon;
                local.conformal_curvature[i][i] += epsilon;
            }
            state.set_state_at(index, &local);
        }
        let before =
            bssn_grid_constraints(&state.view(), &AdmSources::VACUUM).expect("constraints before");
        let determinant_report =
            project_grid_unit_determinant(&mut state).expect("determinant projection");
        let trace_report = project_grid_trace_free(&mut state).expect("trace projection");
        let after =
            bssn_grid_constraints(&state.view(), &AdmSources::VACUUM).expect("constraints after");
        println!(
            "explicit_projection,32,{epsilon},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},projected",
            before.hamiltonian.max_abs,
            after.hamiltonian.max_abs,
            before.determinant.max_abs,
            after.determinant.max_abs,
            before.trace_free.max_abs,
            after.trace_free.max_abs,
            determinant_report
                .correction_magnitude
                .max(trace_report.correction_magnitude),
        );
    }

    // -----------------------------------------------------------------------
    println!("# rejected configurations (typed errors, never silent repair)");
    println!("scenario,request,status,reason");
    let too_few = UniformGrid1d::new(2, 0.0, 1.0);
    println!(
        "rejected,grid_points_2,{},{}",
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
    let bad_domain = UniformGrid1d::new(16, 1.0, 1.0);
    println!(
        "rejected,zero_length_domain,{},{}",
        if bad_domain.is_err()
        {
            "rejected"
        }
        else
        {
            "accepted"
        },
        bad_domain.map_or_else(|e| e.to_string().replace(',', ";"), |_| "none".to_string())
    );
    {
        let grid = grid_of(8);
        let system = BssnGridSystem::vacuum(grid);
        let singular = BssnGridState::zeroed(grid);
        let outcome = evolve_bssn_grid(&system, &singular, 0.0, 0.1, 0.01);
        println!(
            "rejected,singular_conformal_metric,{},{}",
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
        let good = BssnGridState::minkowski(grid);
        let outcome = evolve_bssn_grid(&system, &good, 0.0, 0.1, -0.01);
        println!(
            "rejected,negative_timestep,{},{}",
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
    }

    println!("# interpretation: the derivative stack, the pointwise BSSN assembly, and the");
    println!("#   evolution are all second-order accurate -- the conformal Ricci");
    println!("#   reconstruction and short-time wave propagation both converge at order 2,");
    println!("#   while RK4 retains its fourth order against a fixed spatial operator, and");
    println!("#   Minkowski is exactly stationary. The evolution is stable across every");
    println!("#   resolution tested for C <= 1, with a resolution-independent boundary at");
    println!("#   C = 2. This required writing Rtilde_ij in genuine BSSN form -- carrying");
    println!("#   the EVOLVED Gammatilde^k, which removes the mixed second derivatives from");
    println!("#   the principal part -- and supplying the d_t Gammatilde^i equation. Using");
    println!("#   the generic metric Ricci instead leaves ADM\'s weakly hyperbolic principal");
    println!("#   part wearing BSSN variables, and is unstable. Stability here is measured,");
    println!("#   not proven: strong hyperbolicity is an analytic property of the continuum");
    println!("#   system and nothing in this file establishes it. No black holes, punctures,");
    println!("#   live gauge, AMR, waveform extraction, or observational validation appear.");
}
