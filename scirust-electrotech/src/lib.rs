//! # scirust-electrotech — électrotechnique & génie électrique
//!
//! Primitives déterministes et pures-Rust de calcul en **électrotechnique** et
//! **génie électrique**, dans le prolongement de `scirust-machining` et selon la
//! même doctrine : documentation en français, identifiants anglais, une section
//! **« Limite honnête »** par module, et **aucune constante inventée** — les
//! grandeurs matériau / réseau / composant (résistivités, réactances,
//! rendements, facteurs de puissance…) sont **fournies par l'appelant** d'après
//! un catalogue, une norme ou une mesure.
//!
//! ## Modules
//!
//! ### Machines
//! - [`transformer`] — transformateur monophasé : rapport, impédance ramenée, régulation, rendement.
//! - [`synchronous_generator`] — alternateur synchrone : f.é.m. de Kapp, régime, puissance angle de charge.
//!
//! ### Électronique de puissance
//! - [`rectifier_single_phase`] — redresseur monophasé : moyennes/efficaces, ondulation, pont commandé.
//! - [`rectifier_three_phase`] — redresseur triphasé : P3 et pont de Graetz, commandé par angle d'amorçage.
//! - [`buck_converter`] — hacheur série (abaisseur) : tension de sortie, ondulations (CCM).
//! - [`boost_converter`] — hacheur parallèle (élévateur) : tension de sortie, courant d'entrée (CCM).
//! - [`buck_boost_converter`] — hacheur inverseur : magnitude de sortie, test d'élévation (CCM).
//! - [`inverter`] — onduleur de tension (VSI) : créneau, fondamental, MLI sinus-triangle.
//!
//! ### Circuits & régime sinusoïdal
//! - [`ac_power_single_phase`] — puissances monophasées (S, P, Q), facteur de puissance, impédance.
//! - [`power_factor_correction`] — compensation par condensateur : var, capacité, courant.
//! - [`rlc_series`] — circuit RLC série : impédance, résonance, facteur de qualité, bande.
//! - [`three_phase_systems`] — triphasé équilibré : relations étoile/triangle, puissance active.
//!
//! ### Installations & réseau
//! - [`cable_sizing`] — câble : résistance, chute de tension (1φ/3φ), densité de courant, pertes.
//! - [`short_circuit`] — courant de court-circuit : méthode per-unit, Icc transfo.
//! - [`earthing`] — mise à la terre : piquet (Dwight), plaque, groupement, montée de potentiel.
//! - [`transmission_line`] — ligne courte : chute/régulation, pertes, rendement, impédance caractéristique.
//!
//! ### Composants passifs
//! - [`capacitor`] — condensateur : énergie, charge, réactance, transitoire RC.
//! - [`inductor`] — bobine : énergie, réactance, transitoire RL, f.é.m. auto-induite.
//!
//! ## Positionnement
//!
//! Cette crate complète les autres briques électriques de SciRust sans les
//! dupliquer : `scirust-grid` (analytique réseau — fréquence/RoCoF,
//! synchrophaseurs, THD, détection d'îlotage, estimation d'état) et
//! `scirust-bms` (gestion de batterie — SoC/SoH, emballement thermique). Elle
//! apporte le **cœur de calcul déterministe** des machines, de l'électronique
//! de puissance, des circuits et du dimensionnement des installations.
//!
//! ## Conventions d'unités
//!
//! SI cohérent : tensions en V, courants en A, puissances en W (actives), var
//! (réactives), VA (apparentes), impédances en Ω, fréquences en Hz, angles en
//! radians pour les fonctions trigonométriques. Chaque fonction rappelle ses
//! unités.

pub mod ac_power_single_phase;
pub mod boost_converter;
pub mod buck_boost_converter;
pub mod buck_converter;
pub mod cable_sizing;
pub mod capacitor;
pub mod earthing;
pub mod inductor;
pub mod inverter;
pub mod power_factor_correction;
pub mod rectifier_single_phase;
pub mod rectifier_three_phase;
pub mod rlc_series;
pub mod short_circuit;
pub mod synchronous_generator;
pub mod three_phase_systems;
pub mod transformer;
pub mod transmission_line;

pub use ac_power_single_phase::{
    ac_active_power, ac_apparent_power, ac_impedance_magnitude, ac_power_factor_from_powers,
    ac_reactive_power,
};
pub use boost_converter::{
    boost_duty_for_output, boost_inductor_ripple_current, boost_input_current, boost_output_voltage,
};
pub use buck_boost_converter::{
    buckboost_duty_for_output, buckboost_inductor_ripple_current, buckboost_is_step_up,
    buckboost_output_voltage_magnitude,
};
pub use buck_converter::{
    buck_duty_for_output, buck_inductor_ripple_current, buck_output_ripple_voltage,
    buck_output_voltage,
};
pub use cable_sizing::{
    cable_current_density, cable_power_loss_per_phase, cable_resistance,
    cable_voltage_drop_single_phase, cable_voltage_drop_three_phase,
};
pub use capacitor::{
    cap_charge, cap_charging_voltage, cap_energy, cap_rc_time_constant, cap_reactance,
};
pub use earthing::{
    earth_fault_voltage, earth_parallel_electrodes, earth_plate_resistance, earth_rod_resistance,
};
pub use inductor::{
    ind_current_rise, ind_energy, ind_induced_emf, ind_reactance, ind_rl_time_constant,
};
pub use inverter::{
    inv_modulation_index, inv_sine_pwm_fundamental_peak, inv_square_wave_fundamental_rms,
    inv_square_wave_rms,
};
pub use power_factor_correction::{
    pfc_capacitance, pfc_capacitor_current, pfc_corrected_apparent_power,
    pfc_required_reactive_power,
};
pub use rectifier_single_phase::{
    rect1ph_controlled_fullwave_average, rect1ph_fullwave_average, rect1ph_fullwave_rms,
    rect1ph_halfwave_average, rect1ph_ripple_factor,
};
pub use rectifier_three_phase::{
    rect3ph_bridge_average, rect3ph_controlled_bridge_average, rect3ph_halfwave_average,
    rect3ph_ripple_frequency,
};
pub use rlc_series::{
    rlc_bandwidth, rlc_impedance_magnitude, rlc_phase_angle, rlc_quality_factor,
    rlc_resonant_frequency,
};
pub use short_circuit::{
    scc_base_current, scc_fault_power, scc_symmetrical_fault_current,
    scc_transformer_secondary_fault,
};
pub use synchronous_generator::{
    SYNCGEN_KAPP_COEFFICIENT, syncgen_generated_emf_rms, syncgen_power_angle_power,
    syncgen_synchronous_speed_rpm, syncgen_voltage_regulation,
};
pub use three_phase_systems::{
    tps_balanced_active_power, tps_line_current_delta, tps_line_voltage_star,
    tps_phase_current_delta, tps_phase_voltage_star,
};
pub use transformer::{
    xfmr_efficiency, xfmr_referred_impedance_to_primary, xfmr_secondary_voltage, xfmr_turns_ratio,
    xfmr_voltage_regulation,
};
pub use transmission_line::{
    line_efficiency, line_loss_three_phase, line_sending_voltage_short, line_surge_impedance,
    line_voltage_regulation,
};
