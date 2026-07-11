//! Exemple de bout en bout — chiffrage d'une opération de chariotage.
//!
//! Enchaîne les modules de `scirust-machining` sur un cas concret :
//! choix du régime de coupe → vérification de la puissance broche →
//! durée de vie outil → optimum économique → temps et coût → état de
//! surface → tolérance générale de la cote → vérification d'un pignon.
//!
//! Lancer avec :
//!
//! ```text
//! cargo run -p scirust-machining --example atelier
//! ```

use scirust_machining::economics::MachiningEconomics;
use scirust_machining::forces::{KienzleModel, cutting_power_kw, motor_power_kw};
use scirust_machining::gears::{SpurGear, lewis_bending_stress, tangential_force_from_torque};
use scirust_machining::kinematics::{mrr_turning_cm3_min, spindle_speed_rpm};
use scirust_machining::roughness::theoretical_ra_turning;
use scirust_machining::time::turning_time_min;
use scirust_machining::tolerancing::{GeneralClass, general_linear_tolerance};
use scirust_machining::toollife::taylor_tool_life;

fn main() {
    // --- Données de l'opération : chariotage d'un arbre acier ---------------
    let diameter = 80.0_f64; // Ø pièce (mm)
    let length = 150.0_f64; // longueur à charioter (mm)
    let stock = 4.0_f64; // surépaisseur au rayon à retirer (mm)
    let ap = 2.0_f64; // profondeur de passe (mm)
    let feed = 0.25_f64; // avance (mm/tr)
    let vc = 200.0_f64; // vitesse de coupe (m/min)
    let nose_r = 0.8_f64; // rayon de bec (mm)

    println!("=== Chariotage Ø{diameter} × {length} mm, acier ===\n");

    // --- Cinématique --------------------------------------------------------
    let n = spindle_speed_rpm(vc, diameter);
    let q = mrr_turning_cm3_min(vc, ap, feed);
    println!("Régime      : Vc = {vc} m/min → N = {n:.0} tr/min");
    println!("Débit copeau: Q  = {q:.0} cm³/min");

    // --- Effort, puissance, vérification broche -----------------------------
    let steel = KienzleModel {
        kc11: 1700.0,
        mc: 0.25,
    };
    let fc = steel.cutting_force_turning(ap, feed, 90.0);
    let pc = cutting_power_kw(fc, vc);
    let pm = motor_power_kw(pc, 0.8); // rendement 80 %
    println!("\nEffort coupe: Fc = {fc:.0} N");
    println!("Puissance   : Pc = {pc:.2} kW à la coupe → Pmoteur = {pm:.2} kW (η=0,8)");
    let spindle_kw = 7.5;
    let verdict = if pm <= spindle_kw
    {
        "OK"
    }
    else
    {
        "INSUFFISANTE"
    };
    println!("Broche {spindle_kw} kW disponible → {verdict}");

    // --- Durée de vie outil (Taylor) et optimum économique (Gilbert) --------
    let (c, taylor_n) = (300.0, 0.25);
    let life = taylor_tool_life(vc, c, taylor_n);
    println!("\nTaylor      : Vc·T^n = {c} (n={taylor_n}) → durée de vie T = {life:.1} min");

    let eco = MachiningEconomics {
        n: taylor_n,
        c,
        tool_change_time_min: 2.0,
        tool_cost: 5.0,
        operating_rate: 1.0,
    };
    println!(
        "Gilbert     : Vc coût mini = {:.0} m/min | Vc production maxi = {:.0} m/min",
        eco.speed_min_cost(),
        eco.speed_max_production()
    );

    // --- Temps et coût de coupe ---------------------------------------------
    let time = turning_time_min(length, feed, n, stock, ap);
    let cost = time * eco.operating_rate;
    println!(
        "\nTemps coupe : {time:.2} min ({} passes) → coût pièce ≈ {cost:.2} €",
        (stock / ap).ceil() as u32
    );

    // --- État de surface et tolérance de la cote ----------------------------
    let ra = theoretical_ra_turning(feed, nose_r);
    println!("\nRugosité    : Ra théorique ≈ {ra:.2} µm (f={feed} mm/tr, r={nose_r} mm)");
    if let Some(tol) = general_linear_tolerance(diameter, GeneralClass::Medium)
    {
        println!("ISO 2768-m  : cote Ø{diameter} → tolérance générale ±{tol} mm");
    }

    // --- Vérification d'un pignon entraînant l'arbre ------------------------
    let pinion = SpurGear {
        module_mm: 2.0,
        teeth: 20,
        pressure_angle_deg: 20.0,
    };
    let torque = 9550.0 * pc / n; // couple de coupe ramené à la broche (N·m)
    let ft = tangential_force_from_torque(torque, pinion.pitch_diameter());
    let sigma = lewis_bending_stress(ft, 20.0, pinion.module_mm, 0.32);
    println!(
        "\nPignon m2 z20: d = {:.0} mm, Ft = {ft:.0} N → σ flexion (Lewis) ≈ {sigma:.0} MPa",
        pinion.pitch_diameter()
    );
}
