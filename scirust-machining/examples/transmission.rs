//! Exemple de bout en bout — chaîne cinématique moteur → réducteur → arbre.
//!
//! Enchaîne les modules d'éléments de machines de `scirust-machining` sur un
//! entraînement industriel : régime du moteur asynchrone → réduction par
//! engrenage → couple de sortie → flexion de denture (Lewis) → dimensionnement
//! de l'arbre en torsion → clavette → durée de vie du roulement (ISO 281).
//!
//! Lancer avec :
//!
//! ```text
//! cargo run -p scirust-machining --example transmission
//! ```

use scirust_machining::bearings::{BearingType, basic_rating_life_hours, basic_rating_life_revs};
use scirust_machining::gears::{
    SpurGear, gear_ratio, lewis_bending_stress, pitch_line_velocity_m_s,
    tangential_force_from_power,
};
use scirust_machining::induction_motor::{
    induction_rotor_speed_rpm, induction_synchronous_speed_rpm,
};
use scirust_machining::keyway_stress::keyway_shear_stress;
use scirust_machining::shaft_sizing::{shaft_diameter_from_torque, shaft_torque_from_power};

fn main() {
    // --- Moteur asynchrone triphasé -----------------------------------------
    let supply_hz = 50.0_f64; // fréquence réseau (Hz)
    let pole_pairs = 2.0_f64; // paires de pôles → 1500 tr/min de synchronisme
    let slip = 0.03_f64; // glissement nominal (3 %)
    let motor_power_kw = 7.5_f64; // puissance nominale (kW)

    let ns = induction_synchronous_speed_rpm(supply_hz, pole_pairs);
    let n_motor = induction_rotor_speed_rpm(ns, slip); // régime en charge
    println!(
        "Moteur      : Ns = {ns:.0} tr/min, N charge = {n_motor:.0} tr/min ({motor_power_kw} kW)"
    );

    // --- Réducteur à engrenage droit (pignon z20 → roue z80, i = 4) ---------
    let pinion = SpurGear {
        module_mm: 3.0,
        teeth: 20,
        pressure_angle_deg: 20.0,
    };
    let wheel = SpurGear {
        module_mm: 3.0,
        teeth: 80,
        pressure_angle_deg: 20.0,
    };
    let ratio = gear_ratio(pinion.teeth, wheel.teeth); // rapport de réduction
    let n_out = n_motor / ratio;
    println!(
        "Réducteur   : i = {ratio:.1}, Ø pignon = {:.0} mm, Ø roue = {:.0} mm → N sortie = {n_out:.0} tr/min",
        pinion.pitch_diameter(),
        wheel.pitch_diameter()
    );

    // --- Effort tangentiel et flexion de denture (Lewis) --------------------
    let v = pitch_line_velocity_m_s(pinion.pitch_diameter(), n_motor);
    let ft = tangential_force_from_power(motor_power_kw, v);
    let lewis_y = 0.32_f64; // facteur de forme pour z = 20 (fourni par table)
    let sigma = lewis_bending_stress(ft, 30.0, pinion.module_mm, lewis_y);
    println!(
        "Denture     : V = {v:.2} m/s, Ft = {ft:.0} N → σ flexion (Lewis) ≈ {sigma:.0} MPa (b = 30 mm)"
    );

    // --- Arbre de sortie : dimensionnement en torsion pure ------------------
    let omega_out = n_out * 2.0 * std::f64::consts::PI / 60.0; // rad/s
    let torque_out = shaft_torque_from_power(motor_power_kw * 1000.0, omega_out); // N·m
    let allow_shear = 40e6_f64; // cisaillement admissible (Pa) — fourni matériau
    let d_shaft = shaft_diameter_from_torque(torque_out, allow_shear) * 1000.0; // mm
    println!(
        "Arbre sortie: T = {torque_out:.1} N·m → Ø mini = {d_shaft:.1} mm (τ_adm = {:.0} MPa)",
        allow_shear / 1e6
    );

    // --- Clavette parallèle sur l'arbre de sortie ---------------------------
    let d_key = 40.0e-3_f64; // Ø arbre retenu (m)
    let key_w = 12.0e-3_f64; // largeur clavette (m)
    let key_l = 45.0e-3_f64; // longueur clavette (m)
    let tau_key = keyway_shear_stress(torque_out, d_key, key_w, key_l) / 1e6; // MPa
    println!("Clavette    : cisaillement ≈ {tau_key:.1} MPa (12×8, L = 45 mm)");

    // --- Roulement à billes en sortie (ISO 281, L10) ------------------------
    let c_dyn = 35_000.0_f64; // charge dynamique de base C (N) — catalogue
    let p_load = 4_800.0_f64; // charge dynamique équivalente P (N)
    let l10_mrev = basic_rating_life_revs(c_dyn, p_load, BearingType::Ball);
    let l10_h = basic_rating_life_hours(l10_mrev, n_out);
    println!(
        "Roulement   : L10 ≈ {l10_mrev:.0} Mtr → {l10_h:.0} h à {n_out:.0} tr/min (bille, C = 35 kN)"
    );
}
