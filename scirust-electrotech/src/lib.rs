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
//! ### Machines & réseau (vol. 2)
//! - [`per_unit_system`] — système per-unit : impédance/courant de base, changement de base.
//! - [`transformer_three_phase`] — couplages Yy/Yd/Dy/Dd, indice horaire.
//! - [`autotransformer`] — rapport, puissance transitée/propre, économie de cuivre.
//! - [`induction_machine_circuit`] — schéma équivalent : couple-glissement, couple max.
//! - [`dc_machine`] — machine CC détaillée : f.c.é.m., couple, vitesse, rendement.
//!
//! ### Circuits (vol. 2)
//! - [`rlc_transient`] — réponse libre RLC : amortissement, régime, pulsation amortie.
//! - [`parallel_resonance`] — circuit bouchon : impédance dynamique, Q, bande.
//! - [`passive_filter`] — filtre RC/RL du 1er ordre : coupure, gain, déphasage.
//! - [`thevenin_norton`] — équivalents, courant de charge, transfert max de puissance.
//! - [`wye_delta_transform`] — transformation étoile-triangle (Kennelly).
//! - [`voltage_divider`] — diviseurs de tension (à vide/chargé) et de courant.
//!
//! ### Installations, protection & transport (vol. 2)
//! - [`skin_effect`] — profondeur de peau, R_ac/R_dc, aire effective.
//! - [`overcurrent_relay`] — relais à temps inverse IEC : PSM, courbes SI/VI/EI.
//! - [`earth_fault_loop`] — schéma TN : Zs, Icc présumé, Zs max, tension de contact.
//! - [`load_factor`] — facteurs de charge, demande, diversité, utilisation.
//! - [`battery_pack_config`] — pack série/parallèle : tension, capacité, énergie, courant.
//! - [`flyback_converter`] — convertisseur à isolement : sortie, rapport cyclique, contrainte.
//! - [`conductor_sag`] — flèche de ligne aérienne (parabolique, givre/vent).
//!
//! ### Machines, mesure & composants (vol. 3)
//! - [`transformer_tests`] — essais à vide/court-circuit : pertes fer, réactance magnétisante, Req/Xeq, rendement à charge partielle.
//! - [`instrument_transformer`] — transformateurs de mesure (TC/TP) : courant/tension secondaire, erreur de rapport, burden, facteur limite de précision.
//! - [`single_phase_motor`] — moteur asynchrone monophasé (double champ) : glissements direct/inverse, vitesse de synchronisme, capacité de démarrage.
//! - [`stepper_drive`] — commande pas à pas : angle de pas, micro-pas, vitesse, impulsions, couple utile.
//! - [`solar_pv`] — cellule/panneau photovoltaïque : facteur de forme, rendement, champ série/parallèle, correction en température.
//! - [`capacitor_bank`] — batterie de condensateurs : puissance réactive, capacité, gradins, résistance de décharge.
//! - [`rc_snubber`] — amortissement RC : capacité (dv/dt), résistance, puissance dissipée, amortissement critique.
//! - [`rectifier_smoothing`] — filtrage capacitif : ondulation crête-à-crête, facteur d'ondulation, capacité requise.
//! - [`heatsink_thermal`] — thermique de dissipateur : Tj, Rth série, puissance max, Rth,sa requise.
//!
//! ### Mesure, transport & protection (vol. 3)
//! - [`two_wattmeter`] — méthode des deux wattmètres : puissance active/réactive, angle et facteur de puissance.
//! - [`line_parameters`] — paramètres linéiques d'une ligne aérienne : inductance/capacité linéiques, réactance, DMG triphasée.
//! - [`corona_discharge`] — effet couronne (Peek) : densité de l'air, tension critique disruptive, perte.
//! - [`line_voltage_regulation`] — régulation d'un départ : chute de tension, régulation %, effet Ferranti.
//! - [`busbar`] — jeu de barres : résistance continue, densité de courant, force électrodynamique, pertes Joule.
//! - [`insulation_testing`] — essai d'isolement : résistance en continu, indice de polarisation, DAR, correction thermique.
//! - [`relay_coordination`] — sélectivité chronométrique : marge de discrimination, critère, rapport de réglage.
//! - [`dc_distribution`] — distribution CC : chute deux fils, section requise, pertes, rendement, charge répartie.
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

// Vol. 2
pub mod autotransformer;
pub mod battery_pack_config;
pub mod conductor_sag;
pub mod dc_machine;
pub mod earth_fault_loop;
pub mod flyback_converter;
pub mod induction_machine_circuit;
pub mod load_factor;
pub mod overcurrent_relay;
pub mod parallel_resonance;
pub mod passive_filter;
pub mod per_unit_system;
pub mod rlc_transient;
pub mod skin_effect;
pub mod thevenin_norton;
pub mod transformer_three_phase;
pub mod voltage_divider;
pub mod wye_delta_transform;

// Vol. 3
pub mod busbar;
pub mod capacitor_bank;
pub mod corona_discharge;
pub mod dc_distribution;
pub mod heatsink_thermal;
pub mod instrument_transformer;
pub mod insulation_testing;
pub mod line_parameters;
pub mod line_voltage_regulation;
pub mod rc_snubber;
pub mod rectifier_smoothing;
pub mod relay_coordination;
pub mod single_phase_motor;
pub mod solar_pv;
pub mod stepper_drive;
pub mod transformer_tests;
pub mod two_wattmeter;

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

// Vol. 2 — ré-exports à plat.
pub use autotransformer::{
    autoxfmr_copper_saving, autoxfmr_throughput_power, autoxfmr_voltage_ratio,
    autoxfmr_winding_power,
};
pub use battery_pack_config::{
    packcfg_cell_count, packcfg_continuous_current, packcfg_pack_capacity, packcfg_pack_energy,
    packcfg_pack_voltage,
};
pub use conductor_sag::{
    sag_conductor_length_parabolic, sag_parabolic, sag_tension_for_sag, sag_with_ice_wind,
};
pub use dc_machine::{dcm_back_emf, dcm_efficiency, dcm_speed, dcm_torque};
pub use earth_fault_loop::{
    earthloop_impedance, earthloop_max_impedance_for_disconnection,
    earthloop_prospective_fault_current, earthloop_touch_voltage,
};
pub use flyback_converter::{
    flyback_duty_for_output, flyback_output_voltage_ccm, flyback_reflected_voltage,
    flyback_switch_voltage_stress,
};
pub use induction_machine_circuit::{
    imc_airgap_power, imc_mechanical_power, imc_slip, imc_slip_at_max_torque, imc_torque,
};
pub use load_factor::{
    loadfactor_average_load_from_energy, loadfactor_demand_factor, loadfactor_diversity_factor,
    loadfactor_load_factor, loadfactor_utilization_factor,
};
pub use overcurrent_relay::{
    ocr_iec_extremely_inverse_time, ocr_iec_standard_inverse_time, ocr_iec_very_inverse_time,
    ocr_plug_setting_multiplier,
};
pub use parallel_resonance::{
    pres_bandwidth, pres_dynamic_impedance, pres_quality_factor, pres_resonant_frequency,
};
pub use passive_filter::{
    filt_first_order_gain, filt_gain_decibels, filt_phase_shift, filt_rc_cutoff_frequency,
    filt_rl_cutoff_frequency,
};
pub use per_unit_system::{
    pu_base_current, pu_base_impedance, pu_change_base_impedance, pu_per_unit_value,
};
pub use rlc_transient::{
    RlcRegime, rlctr_damped_frequency, rlctr_damping_coefficient, rlctr_damping_ratio,
    rlctr_regime, rlctr_undamped_natural_frequency,
};
pub use skin_effect::{
    skin_ac_resistance_ratio_thick, skin_depth, skin_depth_from_frequency, skin_effective_area,
};
pub use thevenin_norton::{
    thev_load_current, thev_maximum_power_transfer, thev_norton_current,
    thev_power_transfer_efficiency, thev_thevenin_voltage,
};
pub use transformer_three_phase::{
    xfmr3_line_voltage_ratio_dy, xfmr3_line_voltage_ratio_yd, xfmr3_line_voltage_ratio_yy,
    xfmr3_phase_shift_degrees,
};
pub use voltage_divider::{
    vdiv_current_divider, vdiv_current_divider_two_resistors, vdiv_loaded, vdiv_unloaded,
};
pub use wye_delta_transform::{
    ydelta_balanced_delta_to_wye, ydelta_balanced_wye_to_delta, ydelta_delta_to_wye,
    ydelta_wye_to_delta,
};

// Vol. 3 — ré-exports à plat.
pub use busbar::{
    BUSBAR_MU0, busbar_current_density, busbar_dc_resistance, busbar_power_loss,
    busbar_short_circuit_force,
};
pub use capacitor_bank::{
    capbank_capacitance_for_kvar, capbank_discharge_resistor, capbank_number_of_steps,
    capbank_reactive_power,
};
pub use corona_discharge::{
    corona_air_density_factor, corona_critical_disruptive_voltage, corona_is_active,
    corona_peek_loss,
};
pub use dc_distribution::{
    dcdist_conductor_size_for_drop, dcdist_distributed_load_drop, dcdist_efficiency,
    dcdist_power_loss, dcdist_two_wire_drop,
};
pub use heatsink_thermal::{
    heatsink_junction_temperature, heatsink_max_power_dissipation,
    heatsink_required_sink_resistance, heatsink_total_thermal_resistance,
};
pub use instrument_transformer::{
    insttr_accuracy_limit_factor, insttr_burden_impedance, insttr_ct_secondary_current,
    insttr_ratio_error, insttr_vt_secondary_voltage,
};
pub use insulation_testing::{
    instest_dielectric_absorption_ratio, instest_insulation_resistance, instest_polarization_index,
    instest_temperature_corrected_resistance,
};
pub use line_parameters::{
    VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY, linep_capacitance_per_length,
    linep_geometric_mean_distance_three_phase, linep_inductance_per_length,
    linep_reactance_per_length,
};
pub use line_voltage_regulation::{
    linreg_ferranti_rise, linreg_percent_regulation, linreg_receiving_voltage, linreg_voltage_drop,
};
pub use rc_snubber::{
    snub_capacitance, snub_critical_damping_resistance, snub_power_dissipation, snub_resistance,
};
pub use rectifier_smoothing::{
    smooth_average_dc_voltage, smooth_required_capacitance, smooth_ripple_factor,
    smooth_ripple_voltage_fullwave, smooth_ripple_voltage_halfwave,
};
pub use relay_coordination::{
    relaycoord_current_setting_ratio, relaycoord_discrimination_margin, relaycoord_is_selective,
    relaycoord_required_upstream_time,
};
pub use single_phase_motor::{
    spmot_backward_slip, spmot_forward_slip, spmot_starting_capacitance,
    spmot_synchronous_speed_rpm,
};
pub use solar_pv::{
    pv_array_power, pv_array_voltage, pv_efficiency, pv_fill_factor, pv_temperature_corrected_power,
};
pub use stepper_drive::{
    stepdrv_holding_to_working_torque, stepdrv_pulses_for_angle, stepdrv_resolution_microstepping,
    stepdrv_speed_rpm, stepdrv_step_angle,
};
pub use transformer_tests::{
    xfmrtest_efficiency_at_load, xfmrtest_equivalent_reactance, xfmrtest_equivalent_resistance,
    xfmrtest_iron_loss_resistance, xfmrtest_magnetizing_reactance,
};
pub use two_wattmeter::{
    wattm_power_factor, wattm_power_factor_angle, wattm_reactive_power, wattm_total_active_power,
};
