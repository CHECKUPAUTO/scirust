//! Outils MCP pour `scirust-sim` : exposent les simulateurs déterministes
//! (épidémiologie, plante batterie, stabilité réseau) comme outils appelables
//! par un agent — il décrit un scénario en JSON et récupère les métriques
//! clés (pic épidémique, tension/température de fin de décharge, verdict de
//! synchronisme) sans écrire de code d'intégration.

use crate::registry::McpTool;
use scirust_sim::battery::{BatteryParams, TheveninBattery};
use scirust_sim::engine::FirstOrderForm;
use scirust_sim::epidemiology::Sir;
use scirust_sim::grid::SwingEquation;
use scirust_sim::simulate;
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
}
