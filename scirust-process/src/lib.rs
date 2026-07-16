//! # scirust-process — génie des procédés & génie chimique
//!
//! Primitives déterministes et pures-Rust de calcul en **génie des procédés** et
//! **génie chimique**, dans le prolongement de `scirust-machining` et selon la
//! même doctrine : documentation en français, identifiants anglais, une section
//! **« Limite honnête »** par module, et **aucune constante inventée** — les
//! propriétés des corps purs et des mélanges (enthalpies, volatilités relatives,
//! coefficients de partage, constantes cinétiques, diffusivités…) sont
//! **fournies par l'appelant** d'après des tables, des corrélations ou des essais.
//!
//! ## Modules
//!
//! ### Bilans
//! - [`mass_balance`] — bilan matière : global, par constituant, recyclage, purge, bypass.
//! - [`energy_balance`] — bilan enthalpique : chaleur sensible/latente, mélange adiabatique.
//!
//! ### Réacteurs
//! - [`reaction_kinetics`] — cinétique : loi de vitesse, Arrhenius, conversion, demi-vie.
//! - [`cstr`] — réacteur parfaitement agité : dimensionnement, temps de passage, conversion.
//! - [`pfr`] — réacteur piston : conversion et temps de passage (ordre 1 et 2).
//! - [`batch_reactor`] — réacteur discontinu : temps de réaction (ordre 0/1/2), temps de cycle.
//! - [`residence_time_distribution`] — DTS : temps moyen, E(t)/F(t), réacteurs en série.
//!
//! ### Séparation
//! - [`distillation_mccabe`] — McCabe-Thiele/Fenske : volatilité, étages mini, reflux mini.
//! - [`flash_distillation`] — flash isotherme : Rachford-Rice, partage vapeur/liquide.
//! - [`absorption`] — colonne d'absorption : Kremser, débit minimal, NTU.
//! - [`liquid_extraction`] — extraction L-L : partage, courants croisés/contre-courant.
//! - [`leaching`] — lixiviation solide-liquide : étages, rétention, récupération.
//!
//! ### Transfert de matière
//! - [`mass_transfer`] — double film : flux, coefficient global, Sherwood/Schmidt.
//!
//! ### Opérations unitaires
//! - [`drying`] — séchage : humidité libre, allures constante/décroissante.
//! - [`evaporation`] — évaporation : eau évaporée, économie de vapeur, Dühring.
//! - [`crystallization`] — cristallisation : sursaturation, rendement (anhydre/hydrate).
//! - [`fluidization`] — fluidisation : vitesse minimale (Ergun), perte de charge, terminale.
//! - [`packed_bed`] — lit fixe : équation d'Ergun (Kozeny-Carman + Burke-Plummer).
//!
//! ### Réacteurs — vol. 2
//! - [`reaction_yield_selectivity`] — réactions multiples : rendement, sélectivité.
//! - [`adiabatic_reactor`] — élévation de température adiabatique, droite X-T.
//! - [`catalyst_effectiveness`] — module de Thiele, facteur d'efficacité (plaque/sphère).
//! - [`equilibrium_conversion`] — van't Hoff, conversion à l'équilibre, ΔG°.
//! - [`reactor_space_velocity`] — vitesse spatiale, GHSV/WHSV, temps de contact.
//! - [`reactor_recycle`] — recyclage : conversion globale, débit, purge.
//!
//! ### Équilibres de phases — vol. 2
//! - [`vapor_pressure`] — Antoine, Clausius-Clapeyron, chaleur latente.
//! - [`vle_raoult`] — équilibre L-V idéal : K-values, bulle, volatilité relative.
//! - [`henry_solubility`] — solubilité des gaz : loi de Henry, van't Hoff.
//!
//! ### Séparation — vol. 2
//! - [`batch_distillation`] — distillation discontinue (Rayleigh).
//! - [`packed_column_height`] — colonne à garnissage : HTU/NTU, Z = HTU·NTU.
//! - [`distillation_efficiency`] — rendement d'étage (Murphree/global), plateaux réels.
//! - [`membrane_separation`] — flux solution-diffusion, rejet, pression osmotique.
//! - [`adsorption_isotherm`] — Langmuir/Freundlich, masse d'adsorbant, facteur RL.
//!
//! ### Opérations & ingénierie — vol. 2
//! - [`sedimentation`] — chute entravée (Richardson-Zaki), aire d'épaississeur.
//! - [`pinch_analysis`] — intégration énergétique : CP·ΔT, températures décalées, utilité min.
//! - [`process_economics`] — six-dixièmes, facteur de Lang, retour, annuité.
//! - [`slurry_flow`] — transport de boues : densité, vitesse critique (Durand).
//!
//! ### Transfert thermique & machines — vol. 3
//! - [`heat_exchanger_lmtd`] — échangeur (DTLM) : Q = U·A·DTLM·F, aire requise, température de sortie.
//! - [`heat_exchanger_ntu`] — échangeur (ε-NTU) : NTU, rapport de capacités, efficacité co/contre-courant.
//! - [`pump_sizing`] — pompe : puissance hydraulique/arbre, NPSH disponible, vitesse spécifique, affinité.
//! - [`compressor_power`] — compression d'un gaz parfait : travail isentropique/polytropique, refoulement, multi-étagé.
//! - [`control_valve_cv`] — vanne de régulation : coefficient Cv/Kv, autorité, caractéristique égal pourcentage.
//! - [`orifice_meter`] — débitmètre à diaphragme : rapport β, débit massique/volumique, perte permanente.
//!
//! ## Positionnement
//!
//! Cette crate ouvre le domaine des **opérations unitaires** dans SciRust. Elle
//! complète `scirust-thermo` (thermodynamique des corps purs — processus gaz
//! parfait, cycles) et `scirust-fluids` (mécanique des fluides — nombres
//! adimensionnels, pertes de charge, Bernoulli) au niveau des **procédés** :
//! bilans, réacteurs, séparation, transfert de matière et équipements.
//!
//! ## Conventions d'unités
//!
//! SI cohérent : masses en kg, débits en kg/s (ou mol/s), énergies en J,
//! puissances en W, températures en K (sauf mention °C), pressions en Pa,
//! concentrations en mol/m³ ou fractions (précisé). Les fractions molaires et
//! massiques sont sans dimension. Chaque fonction rappelle ses unités.
//!
//! **Limite honnête générale** : ce sont des **modèles d'opérations unitaires
//! idéalisés** (étages théoriques, mélange parfait, régime permanent, équilibre
//! thermodynamique local). Les non-idéalités (rendement d'étage, pertes
//! thermiques, cinétique de transfert réelle) sont prises en compte via des
//! **coefficients fournis** par l'appelant, jamais inventés.

pub mod absorption;
pub mod batch_reactor;
pub mod crystallization;
pub mod cstr;
pub mod distillation_mccabe;
pub mod drying;
pub mod energy_balance;
pub mod evaporation;
pub mod flash_distillation;
pub mod fluidization;
pub mod leaching;
pub mod liquid_extraction;
pub mod mass_balance;
pub mod mass_transfer;
pub mod packed_bed;
pub mod pfr;
pub mod reaction_kinetics;
pub mod residence_time_distribution;

// Vol. 2
pub mod adiabatic_reactor;
pub mod adsorption_isotherm;
pub mod batch_distillation;
pub mod catalyst_effectiveness;
pub mod distillation_efficiency;
pub mod equilibrium_conversion;
pub mod henry_solubility;
pub mod membrane_separation;
pub mod packed_column_height;
pub mod pinch_analysis;
pub mod process_economics;
pub mod reaction_yield_selectivity;
pub mod reactor_recycle;
pub mod reactor_space_velocity;
pub mod sedimentation;
pub mod slurry_flow;
pub mod vapor_pressure;
pub mod vle_raoult;

// Vol. 3
pub mod compressor_power;
pub mod control_valve_cv;
pub mod heat_exchanger_lmtd;
pub mod heat_exchanger_ntu;
pub mod orifice_meter;
pub mod pump_sizing;

pub use absorption::{
    absorp_factor, absorp_kremser_fraction_absorbed, absorp_minimum_liquid_flow, absorp_ntu_dilute,
};
pub use batch_reactor::{
    batch_cycle_time, batch_time_first_order, batch_time_second_order, batch_time_zero_order,
};
pub use crystallization::{
    cryst_supersaturation_difference, cryst_supersaturation_ratio, cryst_yield_anhydrous,
    cryst_yield_hydrate,
};
pub use cstr::{
    cstr_first_order_conversion, cstr_required_space_time_first_order, cstr_space_time, cstr_volume,
};
pub use distillation_mccabe::{
    dist_equilibrium_vapor, dist_fenske_minimum_stages, dist_minimum_reflux_ratio,
    dist_rectifying_operating_slope, dist_relative_volatility,
};
pub use drying::{
    drying_constant_rate_time, drying_falling_rate_time, drying_free_moisture, drying_total_time,
};
pub use energy_balance::{
    enbal_adiabatic_mixing_temperature, enbal_enthalpy_balance, enbal_latent_heat,
    enbal_sensible_heat,
};
pub use evaporation::{
    evap_boiling_point_elevation_duhring, evap_multiple_effect_economy, evap_steam_economy,
    evap_water_evaporated,
};
pub use flash_distillation::{
    flash_equilibrium_vapor, flash_liquid_composition, flash_rachford_rice_term,
    flash_vapor_liquid_split,
};
pub use fluidization::{
    fluidize_minimum_velocity_fine, fluidize_pressure_drop, fluidize_ratio,
    fluidize_terminal_velocity_stokes,
};
pub use leaching::{
    leach_countercurrent_solute_recovery, leach_overflow_solute_fraction, leach_stage_efficiency,
    leach_underflow_solution,
};
pub use liquid_extraction::{
    extract_countercurrent_raffinate_fraction, extract_crosscurrent_raffinate_fraction,
    extract_distribution_coefficient, extract_factor, extract_single_stage_raffinate_fraction,
};
pub use mass_balance::{
    massbal_component_output, massbal_overall_steady, massbal_purge_flow, massbal_recycle_ratio,
    massbal_splitter_fraction,
};
pub use mass_transfer::{
    masstr_flux, masstr_log_mean_driving_force, masstr_overall_coefficient_gas, masstr_schmidt,
    masstr_sherwood,
};
pub use packed_bed::{
    packedbed_burke_plummer_gradient, packedbed_ergun_pressure_gradient,
    packedbed_kozeny_carman_gradient, packedbed_pressure_drop,
};
pub use pfr::{
    pfr_first_order_conversion, pfr_second_order_space_time, pfr_space_time_first_order,
    pfr_volume_first_order,
};
pub use reaction_kinetics::{
    kinetics_activation_energy_from_two_rates, kinetics_arrhenius, kinetics_first_order_conversion,
    kinetics_half_life_first_order, kinetics_rate,
};
pub use residence_time_distribution::{
    rtd_cumulative_cstr, rtd_exit_age_cstr, rtd_mean_residence_time, rtd_tanks_in_series_number,
};

// Vol. 2 — ré-exports à plat.
pub use adiabatic_reactor::{
    adiab_adiabatic_temperature, adiab_conversion_from_temperature, adiab_maximum_temperature_rise,
    adiab_temperature_rise,
};
pub use adsorption_isotherm::{
    adsorb_freundlich, adsorb_langmuir, adsorb_langmuir_linearized_loading,
    adsorb_mass_of_adsorbent, adsorb_separation_factor,
};
pub use batch_distillation::{
    bdist_average_distillate_fraction, bdist_distillate_moles, bdist_rayleigh_constant_volatility,
    bdist_recovery, bdist_remaining_moles,
};
pub use catalyst_effectiveness::{
    cateff_effectiveness_slab, cateff_effectiveness_sphere, cateff_observed_rate,
    cateff_thiele_modulus_first_order,
};
pub use distillation_efficiency::{
    deff_actual_stages, deff_column_height, deff_murphree_vapor, deff_overall_from_murphree,
};
pub use equilibrium_conversion::{
    equil_conversion_first_order_reversible, equil_gibbs_from_k, equil_k_from_gibbs,
    equil_vant_hoff,
};
pub use henry_solubility::{
    henry_concentration_from_pressure, henry_dissolved_fraction, henry_partial_pressure,
    henry_temperature_dependence,
};
pub use membrane_separation::{
    memb_concentration_factor, memb_osmotic_pressure, memb_recovery, memb_rejection,
    memb_solution_diffusion_flux,
};
pub use packed_column_height::{
    htu_height_of_transfer_unit, htu_number_of_transfer_units_absorption,
    htu_number_of_transfer_units_dilute, htu_packing_height,
};
pub use pinch_analysis::{
    pinch_heat_capacity_flowrate, pinch_minimum_hot_utility, pinch_shifted_temperature_cold,
    pinch_shifted_temperature_hot, pinch_stream_heat_duty,
};
pub use process_economics::{
    econ_annual_capital_charge, econ_lang_factor_capital, econ_payback_period, econ_scale_cost,
    econ_six_tenths_rule,
};
pub use reaction_yield_selectivity::{
    yieldsel_instantaneous_selectivity, yieldsel_overall_yield, yieldsel_selectivity,
    yieldsel_yield_from_conversion_selectivity,
};
pub use reactor_recycle::{
    recy_overall_conversion_with_separation, recy_purge_fraction, recy_reactor_feed,
    recy_recycle_ratio,
};
pub use reactor_space_velocity::{rsv_ghsv, rsv_space_time, rsv_space_velocity, rsv_whsv};
pub use sedimentation::{
    sed_hindered_settling_velocity, sed_solids_flux, sed_thickener_area,
    sed_underflow_concentration,
};
pub use slurry_flow::{
    slurry_durand_critical_velocity, slurry_mixture_density,
    slurry_relative_excess_pressure_gradient, slurry_volume_fraction_from_mass,
};
pub use vapor_pressure::{
    vp_antoine, vp_antoine_temperature, vp_clausius_clapeyron, vp_clausius_latent_heat,
};
pub use vle_raoult::{
    vle_bubble_pressure_binary, vle_equilibrium_ratio, vle_partial_pressure_raoult,
    vle_relative_volatility_raoult, vle_vapor_fraction_binary,
};

// Vol. 3 — ré-exports à plat.
pub use compressor_power::{
    cmp_discharge_temperature, cmp_isentropic_work, cmp_polytropic_work, cmp_power,
    cmp_stage_pressure_ratio,
};
pub use control_valve_cv::{
    cv_authority, cv_equal_percentage_opening, cv_flow_from_cv, cv_kv_from_cv, cv_liquid,
};
pub use heat_exchanger_lmtd::{
    lmtd_duty, lmtd_log_mean, lmtd_outlet_temp_from_duty, lmtd_required_area,
};
pub use heat_exchanger_ntu::{
    ntu_capacity_ratio, ntu_duty, ntu_effectiveness_counterflow, ntu_effectiveness_parallel,
    ntu_number,
};
pub use orifice_meter::{
    orif_beta_ratio, orif_differential_pressure, orif_mass_flow, orif_permanent_loss_fraction,
    orif_volumetric_flow,
};
pub use pump_sizing::{
    pump_affinity_flow, pump_hydraulic_power, pump_npsh_available, pump_shaft_power,
    pump_specific_speed,
};
