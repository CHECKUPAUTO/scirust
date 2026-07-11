//! Outils MCP pour `scirust-sim` : exposent les simulateurs déterministes
//! (épidémiologie, plante batterie, stabilité réseau, zone thermique HVAC,
//! pharmacocinétique orale, cinétique raide de Robertson) comme outils
//! appelables par un agent — il décrit un scénario en JSON et récupère les
//! métriques clés (pic épidémique, tension/température de fin de décharge,
//! verdict de synchronisme, état stationnaire thermique, C_max/t_max/AUC,
//! concentrations finales et masse conservée) sans écrire de code
//! d'intégration. La cinétique raide utilise l'intégrateur implicite
//! Rosenbrock via la feature `stiff` de `scirust-sim`.

use crate::registry::McpTool;
use scirust_sim::battery::{BatteryParams, TheveninBattery};
use scirust_sim::chemistry::Robertson;
use scirust_sim::engine::FirstOrderForm;
use scirust_sim::epidemiology::Sir;
use scirust_sim::grid::SwingEquation;
use scirust_sim::hvac::ZoneThermal2R2C;
use scirust_sim::pharmacokinetics::OralOneCompartment;
use scirust_sim::simulate;
use scirust_sim::stiff_bridge::simulate_rosenbrock;
use serde_json::{Value, json};

/// Read a numeric field: `Ok(None)` if absent, `Err` if present but not a
/// finite number.
fn get_f64(args: &Value, field: &str) -> Result<Option<f64>, String> {
    match args.get(field)
    {
        None | Some(Value::Null) => Ok(None),
        Some(v) =>
        {
            let x = v
                .as_f64()
                .ok_or_else(|| format!("`{field}` must be a number"))?;
            if x.is_finite()
            {
                Ok(Some(x))
            }
            else
            {
                Err(format!("`{field}` must be finite"))
            }
        },
    }
}

fn req_f64(args: &Value, field: &str) -> Result<f64, String> {
    get_f64(args, field)?.ok_or_else(|| format!("missing `{field}`"))
}

fn opt_f64(args: &Value, field: &str, default: f64) -> Result<f64, String> {
    Ok(get_f64(args, field)?.unwrap_or(default))
}

pub fn sim_tools() -> Vec<McpTool> {
    vec![
        epidemic_tool(),
        battery_discharge_tool(),
        grid_stability_tool(),
        hvac_zone_tool(),
        pharmacokinetics_oral_tool(),
        stiff_robertson_tool(),
    ]
}

// ============================================================ //
//  SIR epidemic                                                //
// ============================================================ //

fn epidemic_tool() -> McpTool {
    McpTool {
        name: "sim_epidemic".to_string(),
        description: "Simulate a Kermack–McKendrick SIR epidemic (scirust-sim) from a \
            transmission rate `beta` and recovery rate `gamma`, returning the basic \
            reproduction number R0 = beta/gamma, the peak infected fraction and the day it \
            occurs, and the final attack rate (fraction of the population ever infected). \
            Deterministic."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "beta": { "type": "number", "description": "transmission rate (> 0)" },
                "gamma": { "type": "number", "description": "recovery rate (> 0); 1/gamma is the mean infectious period" },
                "initial_infected": { "type": "number", "description": "initial infected fraction in (0,1), default 0.001" },
                "days": { "type": "number", "description": "horizon in days, default 160" },
                "dt": { "type": "number", "description": "integration step in days, default 0.1" },
            },
            "required": ["beta", "gamma"],
        }),
        handler: Box::new(|args| {
            let beta = req_f64(&args, "beta")?;
            let gamma = req_f64(&args, "gamma")?;
            let i0 = opt_f64(&args, "initial_infected", 0.001)?;
            let days = opt_f64(&args, "days", 160.0)?;
            let dt = opt_f64(&args, "dt", 0.1)?;
            if !(i0 > 0.0 && i0 < 1.0)
            {
                return Err(format!("initial_infected = {i0} must lie in (0, 1)"));
            }

            let sir = Sir::new(beta, gamma).map_err(|e| e.to_string())?;
            let traj =
                simulate(&sir, &[1.0 - i0, i0, 0.0], 0.0, days, dt).map_err(|e| e.to_string())?;
            let infected = traj.column(1).ok_or("empty trajectory")?;

            let (peak_idx, &peak_infected) = infected
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .ok_or("empty trajectory")?;
            let last = traj.last_state().ok_or("empty trajectory")?;
            let attack_rate = 1.0 - last[0]; // everyone who left the susceptible pool

            Ok(json!({
                "r0": sir.r0(),
                "peak_infected_fraction": peak_infected,
                "peak_day": traj.t[peak_idx],
                "final_attack_rate": attack_rate,
                "final_susceptible_fraction": last[0],
                "epidemic": sir.r0() > 1.0,
            }))
        }),
    }
}

// ============================================================ //
//  Thévenin battery discharge                                  //
// ============================================================ //

fn battery_discharge_tool() -> McpTool {
    McpTool {
        name: "sim_battery_discharge".to_string(),
        description: "Simulate a Thévenin (1-RC) lithium-ion cell with self-heating \
            (scirust-sim, the scirust-bms plant) at constant current for `duration_s` seconds. \
            Returns the final state of charge, terminal voltage and cell temperature, plus the \
            steady-state temperature and polarization time constant. Positive `current` \
            discharges; negative charges."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "capacity_ah": { "type": "number", "description": "nominal capacity (A·h, > 0)" },
                "r0": { "type": "number", "description": "ohmic series resistance (ohm, > 0)" },
                "r1": { "type": "number", "description": "polarization resistance (ohm, > 0)" },
                "c1": { "type": "number", "description": "polarization capacitance (F, > 0)" },
                "current": { "type": "number", "description": "constant current (A); positive discharges" },
                "duration_s": { "type": "number", "description": "simulated duration in seconds (> 0)" },
                "ocv_min": { "type": "number", "description": "open-circuit voltage at soc=0, default 3.0" },
                "ocv_max": { "type": "number", "description": "open-circuit voltage at soc=1, default 4.2" },
                "r_th": { "type": "number", "description": "thermal resistance to ambient (K/W), default 5.0" },
                "c_th": { "type": "number", "description": "thermal capacitance (J/K), default 40.0" },
                "t_ambient": { "type": "number", "description": "ambient temperature, default 25.0" },
                "soc0": { "type": "number", "description": "initial state of charge in [0,1], default 1.0" },
                "temp0": { "type": "number", "description": "initial temperature, default = t_ambient" },
                "dt": { "type": "number", "description": "integration step (s), default duration_s/2000 clamped to [1e-3, 5]" },
            },
            "required": ["capacity_ah", "r0", "r1", "c1", "current", "duration_s"],
        }),
        handler: Box::new(|args| {
            let t_ambient = opt_f64(&args, "t_ambient", 25.0)?;
            let params = BatteryParams {
                capacity_ah: req_f64(&args, "capacity_ah")?,
                r0: req_f64(&args, "r0")?,
                r1: req_f64(&args, "r1")?,
                c1: req_f64(&args, "c1")?,
                current: req_f64(&args, "current")?,
                ocv_min: opt_f64(&args, "ocv_min", 3.0)?,
                ocv_max: opt_f64(&args, "ocv_max", 4.2)?,
                r_th: opt_f64(&args, "r_th", 5.0)?,
                c_th: opt_f64(&args, "c_th", 40.0)?,
                t_ambient,
            };
            let duration = req_f64(&args, "duration_s")?;
            let soc0 = opt_f64(&args, "soc0", 1.0)?;
            let temp0 = opt_f64(&args, "temp0", t_ambient)?;
            let dt = opt_f64(&args, "dt", (duration / 2000.0).clamp(1e-3, 5.0))?;

            let battery = TheveninBattery::new(params).map_err(|e| e.to_string())?;
            let traj = simulate(
                &battery,
                &battery.initial_state(soc0, temp0),
                0.0,
                duration,
                dt,
            )
            .map_err(|e| e.to_string())?;
            let last = traj.last_state().ok_or("empty trajectory")?;
            let terminal = battery.terminal_voltage(last).ok_or("bad state length")?;

            Ok(json!({
                "final_soc": last[0],
                "final_terminal_voltage": terminal,
                "final_temperature": last[2],
                "steady_state_temperature": battery.steady_state_temperature(),
                "polarization_time_constant_s": battery.polarization_time_constant(),
                "depleted": last[0] <= 0.0,
            }))
        }),
    }
}

// ============================================================ //
//  Grid swing-equation stability                               //
// ============================================================ //

fn grid_stability_tool() -> McpTool {
    McpTool {
        name: "sim_grid_stability".to_string(),
        description: "Analyze a synchronous machine on an infinite bus via the swing equation \
            (scirust-sim, the scirust-grid plant). Returns whether a synchronous operating \
            point exists (`P_m <= P_max`), the equilibrium rotor angle asin(P_m/P_max) and the \
            small-signal electromechanical frequency. If a `disturbance_angle_rad` and \
            `duration_s` are given, it simulates the transient and reports whether the rotor \
            settles back to equilibrium."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "inertia_h": { "type": "number", "description": "inertia constant H (s, > 0)" },
                "damping": { "type": "number", "description": "damping coefficient D (>= 0)" },
                "p_mech": { "type": "number", "description": "mechanical power (pu)" },
                "p_max": { "type": "number", "description": "peak electrical power P_max (pu, > 0)" },
                "frequency_hz": { "type": "number", "description": "system frequency, default 50" },
                "disturbance_angle_rad": { "type": "number", "description": "optional initial angle offset from equilibrium to test transient stability" },
                "duration_s": { "type": "number", "description": "transient horizon (s), default 10 (only used with disturbance_angle_rad)" },
            },
            "required": ["inertia_h", "damping", "p_mech", "p_max"],
        }),
        handler: Box::new(|args| {
            let frequency_hz = opt_f64(&args, "frequency_hz", 50.0)?;
            let inertia_h = req_f64(&args, "inertia_h")?;
            let damping = req_f64(&args, "damping")?;
            let p_mech = req_f64(&args, "p_mech")?;
            let p_max = req_f64(&args, "p_max")?;

            let sys = SwingEquation::new(frequency_hz, inertia_h, damping, p_mech, p_max)
                .map_err(|e| e.to_string())?;

            let equilibrium = sys.equilibrium_angle();
            let mut out = json!({
                "synchronized": equilibrium.is_some(),
                "equilibrium_angle_rad": equilibrium,
                "small_signal_frequency_hz": sys
                    .small_signal_frequency()
                    .map(|w| w / (2.0 * std::f64::consts::PI)),
            });

            if let (Some(delta_eq), Some(disturbance)) =
                (equilibrium, get_f64(&args, "disturbance_angle_rad")?)
            {
                let duration = opt_f64(&args, "duration_s", 10.0)?;
                let dt = (duration / 20_000.0).clamp(1e-4, 1e-2);
                let traj = simulate(
                    &FirstOrderForm(&sys),
                    &[delta_eq + disturbance, 0.0],
                    0.0,
                    duration,
                    dt,
                )
                .map_err(|e| e.to_string())?;
                let last = traj.last_state().ok_or("empty trajectory")?;
                let angle_error = (last[0] - delta_eq).abs();
                out["transient"] = json!({
                    "final_angle_rad": last[0],
                    "final_angle_error_rad": angle_error,
                    "final_speed_deviation": last[1],
                    "settled": angle_error < 1e-2 && last[1].abs() < 1e-2,
                });
            }

            Ok(out)
        }),
    }
}

// ============================================================ //
//  2R2C single-zone building thermal (HVAC plant)              //
// ============================================================ //

fn hvac_zone_tool() -> McpTool {
    McpTool {
        name: "sim_hvac_zone".to_string(),
        description: "Simulate a single-zone 2R2C building thermal model (scirust-sim, the \
            scirust-hvac plant): an air node coupled through the wall thermal mass to a fixed \
            outside temperature, driven by a constant HVAC heat input `q_hvac`. Returns the \
            exact linear steady-state air and wall temperatures, the zone heat-loss conductance \
            1/(R_aw+R_wo) in W/K, and the air/wall temperatures reached after `duration_s`. \
            Positive `q_hvac` heats, negative cools. Deterministic."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "c_air": { "type": "number", "description": "air thermal capacitance (J/K, > 0)" },
                "c_wall": { "type": "number", "description": "wall-mass thermal capacitance (J/K, > 0)" },
                "r_aw": { "type": "number", "description": "air-to-wall thermal resistance (K/W, > 0)" },
                "r_wo": { "type": "number", "description": "wall-to-outside thermal resistance (K/W, > 0)" },
                "t_outside": { "type": "number", "description": "outside temperature (held constant)" },
                "q_hvac": { "type": "number", "description": "HVAC heat delivered to the air (W); positive heats" },
                "t_air0": { "type": "number", "description": "initial air temperature, default = t_outside" },
                "t_wall0": { "type": "number", "description": "initial wall temperature, default = t_outside" },
                "duration_s": { "type": "number", "description": "simulated duration (s), default 12·c_wall·(r_aw+r_wo) (~a dozen slow time constants)" },
                "dt": { "type": "number", "description": "integration step (s), default duration_s/4000 clamped to [0.1, 30]" },
            },
            "required": ["c_air", "c_wall", "r_aw", "r_wo", "t_outside", "q_hvac"],
        }),
        handler: Box::new(|args| {
            let c_air = req_f64(&args, "c_air")?;
            let c_wall = req_f64(&args, "c_wall")?;
            let r_aw = req_f64(&args, "r_aw")?;
            let r_wo = req_f64(&args, "r_wo")?;
            let t_outside = req_f64(&args, "t_outside")?;
            let q_hvac = req_f64(&args, "q_hvac")?;
            let t_air0 = opt_f64(&args, "t_air0", t_outside)?;
            let t_wall0 = opt_f64(&args, "t_wall0", t_outside)?;
            let duration = opt_f64(&args, "duration_s", 12.0 * c_wall * (r_aw + r_wo))?;
            let dt = opt_f64(&args, "dt", (duration / 4000.0).clamp(0.1, 30.0))?;

            let zone = ZoneThermal2R2C::new(c_air, c_wall, r_aw, r_wo, t_outside, q_hvac)
                .map_err(|e| e.to_string())?;
            let (ss_air, ss_wall) = zone.steady_state();
            let traj = simulate(&zone, &[t_air0, t_wall0], 0.0, duration, dt)
                .map_err(|e| e.to_string())?;
            let last = traj.last_state().ok_or("empty trajectory")?;
            let reached = (last[0] - ss_air).abs() < 1e-2 && (last[1] - ss_wall).abs() < 1e-2;

            Ok(json!({
                "steady_state_t_air": ss_air,
                "steady_state_t_wall": ss_wall,
                "conductance_w_per_k": zone.conductance(),
                "final_t_air": last[0],
                "final_t_wall": last[1],
                "reached_steady_state": reached,
            }))
        }),
    }
}

// ============================================================ //
//  Oral one-compartment pharmacokinetics                       //
// ============================================================ //

fn pharmacokinetics_oral_tool() -> McpTool {
    McpTool {
        name: "sim_pharmacokinetics_oral".to_string(),
        description: "Simulate first-order oral absorption into a one-compartment body \
            (scirust-sim): a gut depot holding the bioavailable fraction F of `dose` empties at \
            rate `k_a` into a central compartment that eliminates at rate `k_e`, producing the \
            Bateman plasma-concentration curve. Returns the peak concentration C_max and the \
            time t_max it occurs, the terminal elimination half-life ln(2)/k_e, the total \
            exposure AUC(0..inf) = F·dose/(volume·k_e), and the plasma concentration at the end \
            of the horizon. Deterministic."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "k_a": { "type": "number", "description": "first-order absorption rate (1/time, > 0)" },
                "k_e": { "type": "number", "description": "first-order elimination rate (1/time, > 0)" },
                "volume": { "type": "number", "description": "volume of distribution (dose units per concentration, > 0)" },
                "dose": { "type": "number", "description": "administered dose (> 0)" },
                "bioavailability": { "type": "number", "description": "absorbed fraction F in (0, 1], default 1.0" },
                "duration": { "type": "number", "description": "horizon in time units, default 10 elimination half-lives" },
                "dt": { "type": "number", "description": "integration step, default duration/4000 clamped to [1e-4, 0.5]" },
            },
            "required": ["k_a", "k_e", "volume", "dose"],
        }),
        handler: Box::new(|args| {
            let k_a = req_f64(&args, "k_a")?;
            let k_e = req_f64(&args, "k_e")?;
            let volume = req_f64(&args, "volume")?;
            let dose = req_f64(&args, "dose")?;
            let f = opt_f64(&args, "bioavailability", 1.0)?;
            let half_life = std::f64::consts::LN_2 / k_e;
            let duration = opt_f64(&args, "duration", 10.0 * half_life)?;
            let dt = opt_f64(&args, "dt", (duration / 4000.0).clamp(1e-4, 0.5))?;

            let pk =
                OralOneCompartment::new(k_a, k_e, volume, f, dose).map_err(|e| e.to_string())?;
            let traj =
                simulate(&pk, &pk.initial_state(), 0.0, duration, dt).map_err(|e| e.to_string())?;

            // Peak central amount from the trajectory (robust even at k_a = k_e,
            // where the closed-form t_max is singular).
            let central = traj.column(1).ok_or("empty trajectory")?;
            let (peak_idx, &peak_amt) = central
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .ok_or("empty trajectory")?;
            let last = traj.last_state().ok_or("empty trajectory")?;

            let mut out = json!({
                "c_max": pk.concentration(peak_amt),
                "t_max": traj.t[peak_idx],
                "half_life": half_life,
                // Exact: AUC of concentration over [0, inf) = F·dose/(V·k_e).
                "auc_inf": f * dose / (volume * k_e),
                "final_concentration": pk.concentration(last[1]),
            });
            // Analytic t_max cross-check when the Bateman term is non-singular.
            if let Some(t) = pk.peak_time()
            {
                out["analytic_t_max"] = json!(t);
            }
            Ok(out)
        }),
    }
}

// ============================================================ //
//  Robertson stiff kinetics (implicit Rosenbrock solver)       //
// ============================================================ //

fn stiff_robertson_tool() -> McpTool {
    McpTool {
        name: "sim_stiff_robertson".to_string(),
        description: "Integrate the Robertson autocatalytic reaction — the canonical *stiff* ODE \
            benchmark, whose rate constants span nine orders of magnitude — with scirust-sim's \
            implicit **Rosenbrock-W(2,3)** solver (via scirust-stiff). An explicit method (RK4) \
            would need an impractically small step or blow up on the fast initial transient; the \
            implicit method's stability is decoupled from it. Returns the final species \
            concentrations [a, b, c], the conserved total mass a+b+c, and the fraction of mass \
            converted to C. Deterministic."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "k1": { "type": "number", "description": "rate A→B (> 0), default 0.04" },
                "k2": { "type": "number", "description": "rate B+B→B+C (> 0), default 3e7" },
                "k3": { "type": "number", "description": "rate B+C→A+C (> 0), default 1e4" },
                "duration": { "type": "number", "description": "integration horizon in time units, default 0.4 (Hairer & Wanner's reference point)" },
                "a0": { "type": "number", "description": "initial concentration of A, default 1.0" },
                "b0": { "type": "number", "description": "initial concentration of B, default 0.0" },
                "c0": { "type": "number", "description": "initial concentration of C, default 0.0" },
                "rtol": { "type": "number", "description": "relative tolerance (> 0), default 1e-7" },
                "atol": { "type": "number", "description": "absolute tolerance (> 0), default 1e-10" },
                "h0": { "type": "number", "description": "initial step (> 0), default 1e-6" },
            },
            "required": [],
        }),
        handler: Box::new(|args| {
            let k1 = opt_f64(&args, "k1", 0.04)?;
            let k2 = opt_f64(&args, "k2", 3.0e7)?;
            let k3 = opt_f64(&args, "k3", 1.0e4)?;
            let duration = opt_f64(&args, "duration", 0.4)?;
            let a0 = opt_f64(&args, "a0", 1.0)?;
            let b0 = opt_f64(&args, "b0", 0.0)?;
            let c0 = opt_f64(&args, "c0", 0.0)?;
            let rtol = opt_f64(&args, "rtol", 1e-7)?;
            let atol = opt_f64(&args, "atol", 1e-10)?;
            let h0 = opt_f64(&args, "h0", 1e-6)?;

            let rob = Robertson::new(k1, k2, k3).map_err(|e| e.to_string())?;
            let traj = simulate_rosenbrock(&rob, &[a0, b0, c0], 0.0, duration, rtol, atol, h0)
                .map_err(|e| e.to_string())?;
            let last = traj.last_state().ok_or("empty trajectory")?;
            let mass = last[0] + last[1] + last[2];
            let total0 = a0 + b0 + c0;

            Ok(json!({
                "final_a": last[0],
                "final_b": last[1],
                "final_c": last[2],
                "total_mass": mass,
                "mass_conserved": (mass - total0).abs() < 1e-6 * total0.abs().max(1.0),
                "fraction_converted_to_c": if total0 > 0.0 { last[2] / total0 } else { 0.0 },
                "steps": traj.t.len(),
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epidemic_tool_reports_r0_peak_and_attack_rate() {
        let tool = epidemic_tool();
        let out = (tool.handler)(json!({ "beta": 0.6, "gamma": 0.2 })).unwrap();
        assert!((out["r0"].as_f64().unwrap() - 3.0).abs() < 1e-12);
        assert_eq!(out["epidemic"], json!(true));
        assert!(out["peak_infected_fraction"].as_f64().unwrap() > 0.2);
        // R0 = 3 ⇒ ~94% ever infected.
        assert!(out["final_attack_rate"].as_f64().unwrap() > 0.9);
        assert!(out["peak_day"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn epidemic_tool_below_threshold_does_not_take_off() {
        let tool = epidemic_tool();
        let out = (tool.handler)(json!({ "beta": 0.1, "gamma": 0.2 })).unwrap();
        assert_eq!(out["epidemic"], json!(false));
        assert!(out["final_attack_rate"].as_f64().unwrap() < 0.05);
    }

    #[test]
    fn epidemic_tool_validates_inputs() {
        let tool = epidemic_tool();
        assert!((tool.handler)(json!({ "beta": 0.6 })).is_err()); // missing gamma
        assert!((tool.handler)(json!({ "beta": 0.6, "gamma": 0.0 })).is_err()); // bad rate
        assert!(
            (tool.handler)(json!({ "beta": 0.6, "gamma": 0.2, "initial_infected": 1.5 })).is_err()
        );
        assert!((tool.handler)(json!({ "beta": "x", "gamma": 0.2 })).is_err()); // non-numeric
    }

    #[test]
    fn battery_tool_tracks_coulomb_counting_and_voltage() {
        let tool = battery_discharge_tool();
        let out = (tool.handler)(json!({
            "capacity_ah": 2.0, "r0": 0.02, "r1": 0.01, "c1": 2000.0,
            "current": 4.0, "duration_s": 100.0,
        }))
        .unwrap();
        // SoC after 100 s at 4 A from a 2 A·h cell: 1 - 400/7200 ≈ 0.9444.
        assert!((out["final_soc"].as_f64().unwrap() - 0.944_444_444).abs() < 1e-6);
        assert!(out["final_terminal_voltage"].as_f64().unwrap() < 4.2);
        assert!(out["final_temperature"].as_f64().unwrap() > 25.0);
        assert_eq!(out["depleted"], json!(false));
    }

    #[test]
    fn battery_tool_validates_inputs() {
        let tool = battery_discharge_tool();
        assert!(
            (tool.handler)(
                json!({ "r0": 0.02, "r1": 0.01, "c1": 2000.0, "current": 4.0, "duration_s": 100.0 })
            )
            .is_err()
        ); // missing capacity
        assert!((tool.handler)(json!({
            "capacity_ah": 0.0, "r0": 0.02, "r1": 0.01, "c1": 2000.0, "current": 4.0, "duration_s": 100.0
        }))
        .is_err()); // bad capacity
    }

    #[test]
    fn grid_tool_reports_equilibrium_and_settling() {
        let tool = grid_stability_tool();
        let out = (tool.handler)(json!({
            "inertia_h": 5.0, "damping": 12.0, "p_mech": 1.0, "p_max": 2.0,
            "disturbance_angle_rad": 0.4, "duration_s": 20.0,
        }))
        .unwrap();
        assert_eq!(out["synchronized"], json!(true));
        assert!(
            (out["equilibrium_angle_rad"].as_f64().unwrap() - std::f64::consts::FRAC_PI_6).abs()
                < 1e-9
        );
        assert!((out["small_signal_frequency_hz"].as_f64().unwrap() - 1.174).abs() < 0.01);
        assert_eq!(out["transient"]["settled"], json!(true));
    }

    #[test]
    fn grid_tool_flags_loss_of_synchronism() {
        let tool = grid_stability_tool();
        let out = (tool.handler)(json!({
            "inertia_h": 5.0, "damping": 1.0, "p_mech": 3.0, "p_max": 2.0,
        }))
        .unwrap();
        assert_eq!(out["synchronized"], json!(false));
        assert_eq!(out["equilibrium_angle_rad"], Value::Null);
        assert!(out.get("transient").is_none());
    }

    #[test]
    fn hvac_tool_reports_linear_steady_state_and_settles() {
        let tool = hvac_zone_tool();
        // Explicit long horizon so the assertion does not ride the default heuristic.
        let out = (tool.handler)(json!({
            "c_air": 1200.0, "c_wall": 20000.0, "r_aw": 0.05, "r_wo": 0.2,
            "t_outside": 5.0, "q_hvac": 500.0, "duration_s": 100000.0,
        }))
        .unwrap();
        // t_air = 5 + 500·0.25 = 130; t_wall = 5 + 500·0.2 = 105.
        assert!((out["steady_state_t_air"].as_f64().unwrap() - 130.0).abs() < 1e-9);
        assert!((out["steady_state_t_wall"].as_f64().unwrap() - 105.0).abs() < 1e-9);
        // Conductance = 1/(0.05 + 0.2) = 4 W/K.
        assert!((out["conductance_w_per_k"].as_f64().unwrap() - 4.0).abs() < 1e-12);
        assert_eq!(out["reached_steady_state"], json!(true));
        assert!((out["final_t_air"].as_f64().unwrap() - 130.0).abs() < 1e-2);
    }

    #[test]
    fn hvac_tool_validates_inputs() {
        let tool = hvac_zone_tool();
        // Missing a required physical parameter.
        assert!(
            (tool.handler)(json!({
                "c_wall": 20000.0, "r_aw": 0.05, "r_wo": 0.2, "t_outside": 5.0, "q_hvac": 500.0
            }))
            .is_err()
        );
        // Non-positive capacitance.
        assert!(
            (tool.handler)(json!({
                "c_air": 0.0, "c_wall": 20000.0, "r_aw": 0.05, "r_wo": 0.2,
                "t_outside": 5.0, "q_hvac": 500.0
            }))
            .is_err()
        );
    }

    #[test]
    fn pk_oral_tool_reports_cmax_tmax_auc_and_half_life() {
        let tool = pharmacokinetics_oral_tool();
        let out = (tool.handler)(json!({
            "k_a": 1.2, "k_e": 0.25, "volume": 30.0, "dose": 100.0, "bioavailability": 0.8,
        }))
        .unwrap();
        // Analytic t_max = ln(k_a/k_e)/(k_a - k_e) = ln(4.8)/0.95 ≈ 1.6512.
        assert!((out["t_max"].as_f64().unwrap() - 1.6512).abs() < 0.05);
        assert!((out["analytic_t_max"].as_f64().unwrap() - 1.6512).abs() < 1e-3);
        // AUC(0..inf) = F·dose/(V·k_e) = 0.8·100/(30·0.25) ≈ 10.6667.
        assert!((out["auc_inf"].as_f64().unwrap() - 10.666_666_7).abs() < 1e-4);
        // Half-life = ln2/0.25 ≈ 2.7726.
        assert!((out["half_life"].as_f64().unwrap() - 2.772_589).abs() < 1e-4);
        assert!(out["c_max"].as_f64().unwrap() > 0.0);
        // The concentration has fallen well below its peak by the end of 10 half-lives.
        assert!(out["final_concentration"].as_f64().unwrap() < out["c_max"].as_f64().unwrap());
    }

    #[test]
    fn pk_oral_tool_validates_inputs() {
        let tool = pharmacokinetics_oral_tool();
        assert!((tool.handler)(json!({ "k_a": 1.2, "k_e": 0.25, "volume": 30.0 })).is_err()); // missing dose
        assert!(
            (tool.handler)(json!({ "k_a": 0.0, "k_e": 0.25, "volume": 30.0, "dose": 100.0 }))
                .is_err()
        ); // bad k_a
        assert!(
            (tool.handler)(json!({
                "k_a": 1.2, "k_e": 0.25, "volume": 30.0, "dose": 100.0, "bioavailability": 1.5
            }))
            .is_err()
        ); // bad F
    }

    #[test]
    fn stiff_robertson_tool_conserves_mass_and_matches_reference() {
        let tool = stiff_robertson_tool();
        // Defaults: classic constants, integrated to t = 0.4.
        let out = (tool.handler)(json!({})).unwrap();
        // Hairer & Wanner reference at t = 0.4: a ≈ 0.9851, b ≈ 3.4e-5, c ≈ 0.0149.
        assert!((out["final_a"].as_f64().unwrap() - 0.9851).abs() < 2e-3);
        let b = out["final_b"].as_f64().unwrap();
        assert!(b > 0.0 && b < 1e-4, "b = {b}");
        assert!((out["final_c"].as_f64().unwrap() - 0.0149).abs() < 2e-3);
        assert!((out["total_mass"].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(out["mass_conserved"], json!(true));
    }

    #[test]
    fn stiff_robertson_tool_validates_inputs() {
        let tool = stiff_robertson_tool();
        // Robertson::new rejects non-positive / non-finite rate constants.
        assert!((tool.handler)(json!({ "k1": -1.0 })).is_err());
        assert!((tool.handler)(json!({ "k2": 0.0 })).is_err());
    }
}
