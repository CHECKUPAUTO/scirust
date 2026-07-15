//! Exemple de bout en bout — circuit hydraulique et échange thermique.
//!
//! Enchaîne les modules de mécanique des fluides et de thermique de
//! `scirust-machining` : régime d'écoulement dans une conduite (Reynolds →
//! frottement de Colebrook → perte de charge de Darcy) → dimensionnement de la
//! pompe (puissance hydraulique et à l'arbre) → échangeur à contre-courant
//! (DTLM et flux thermique).
//!
//! Lancer avec :
//!
//! ```text
//! cargo run -p scirust-machining --example thermofluides
//! ```

use scirust_machining::bernoulli::reynolds_number;
use scirust_machining::lmtd::{lmtd_counterflow, lmtd_heat_duty};
use scirust_machining::pipe_flow::{colebrook_friction, darcy_head_loss};
use scirust_machining::pumps::{hydraulic_power, shaft_power};

fn main() {
    // --- Circuit : eau dans une conduite en acier ---------------------------
    let rho = 1000.0_f64; // masse volumique de l'eau (kg/m³)
    let mu = 1.0e-3_f64; // viscosité dynamique (Pa·s)
    let g = 9.81_f64;
    let diameter = 0.05_f64; // Ø intérieur (m)
    let velocity = 2.0_f64; // vitesse débitante (m/s)
    let length = 120.0_f64; // longueur de conduite (m)
    let roughness = 0.045e-3_f64; // rugosité acier (m) — fournie

    let re = reynolds_number(rho, velocity, diameter, mu);
    let f = colebrook_friction(re, roughness / diameter);
    let h_loss = darcy_head_loss(f, length, diameter, velocity, g);
    println!(
        "Écoulement  : Re = {re:.0} (turbulent), f = {f:.4} → perte de charge = {h_loss:.2} m"
    );

    // --- Pompe : débit à partir de la section, puissances -------------------
    let area = std::f64::consts::PI * diameter * diameter / 4.0;
    let flow = velocity * area; // débit volumique (m³/s)
    let static_head = 15.0_f64; // hauteur géométrique (m)
    let total_head = static_head + h_loss; // HMT (m)
    let p_hyd = hydraulic_power(rho, g, flow, total_head); // W
    let efficiency = 0.72_f64; // rendement de pompe — fourni
    let p_shaft = shaft_power(p_hyd, efficiency); // W
    println!(
        "Pompe       : Q = {:.1} L/s, HMT = {total_head:.1} m → P hydraulique = {:.2} kW, P arbre = {:.2} kW (η = {efficiency})",
        flow * 1000.0,
        p_hyd / 1000.0,
        p_shaft / 1000.0
    );

    // --- Échangeur à contre-courant : DTLM et flux --------------------------
    let (hot_in, hot_out) = (90.0_f64, 60.0_f64); // fluide chaud entrée/sortie (°C)
    let (cold_in, cold_out) = (20.0_f64, 45.0_f64); // fluide froid entrée/sortie (°C)
    let lmtd = lmtd_counterflow(hot_in, hot_out, cold_in, cold_out);
    let u = 1200.0_f64; // coefficient global (W/m²K) — fourni
    let area_hx = 8.0_f64; // surface d'échange (m²)
    let correction = 1.0_f64; // F = 1 pour un vrai contre-courant
    let q = lmtd_heat_duty(u, area_hx, lmtd, correction); // W
    println!(
        "Échangeur   : DTLM = {lmtd:.1} K → Q = {:.1} kW (U = {u} W/m²K, A = {area_hx} m²)",
        q / 1000.0
    );
}
