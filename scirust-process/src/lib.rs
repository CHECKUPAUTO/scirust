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
