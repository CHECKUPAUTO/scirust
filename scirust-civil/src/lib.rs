//! # scirust-civil — génie civil & structures BTP
//!
//! Primitives déterministes et pures-Rust de calcul en **génie civil** et
//! **structures du bâtiment**, dans le prolongement de `scirust-machining` et
//! selon la même doctrine : documentation en français, identifiants anglais, une
//! section **« Limite honnête »** par module, et **aucune constante inventée** —
//! les résistances caractéristiques des matériaux (`fck`, `fyk`, `fy`…) et les
//! **coefficients partiels de sécurité** (`γc`, `γs`, `γM`…) sont **fournis par
//! l'appelant** d'après l'Eurocode applicable et son Annexe Nationale.
//!
//! ## Modules
//!
//! ### Béton armé (Eurocode 2)
//! - [`rc_beam_flexure`] — flexion simple (ELU) : moment réduit, bras de levier, acier tendu.
//! - [`rc_shear`] — effort tranchant : bielles (VRd,c, VRd,s, VRd,max).
//! - [`rc_column`] — poteau : résistance axiale, élancement, élancement limite.
//! - [`rc_anchorage`] — ancrage/recouvrement : adhérence fbd, longueurs lbd/l0.
//! - [`rc_serviceability`] — ELS : limites de contrainte, ouverture de fissure, flèche L/d.
//! - [`rc_punching`] — poinçonnement de dalle : périmètre de contrôle, vRd,c.
//!
//! ### Charpente métallique (Eurocode 3)
//! - [`steel_section_class`] — classification des sections comprimées (classes 1..4).
//! - [`steel_bending`] — flexion/cisaillement : Mc,Rd (plastique/élastique), Vpl,Rd.
//! - [`steel_compression`] — flambement par flexion : λ̄, χ, Nb,Rd.
//! - [`steel_lateral_torsional`] — déversement : λ̄LT, χLT, Mb,Rd.
//! - [`steel_bolted_connection`] — assemblage boulonné : cisaillement, pression diamétrale, traction.
//!
//! ### Bois (Eurocode 5)
//! - [`timber_bending`] — flexion : fm,d (kmod, kh), contrainte, taux de travail.
//!
//! ### Géotechnique
//! - [`bearing_capacity`] — portance de fondation superficielle (Terzaghi).
//! - [`earth_pressure`] — poussée des terres (Rankine) : Ka/Kp, poussée résultante.
//! - [`settlement`] — tassement de consolidation œdométrique.
//! - [`slope_stability`] — stabilité de pente infinie (sec, cohérent, avec nappe).
//! - [`retaining_wall`] — mur-poids : renversement, glissement, contrainte de sol.
//!
//! ### Hydraulique à surface libre
//! - [`open_channel_manning`] — canal : Manning-Strickler, rayon hydraulique, profondeur critique.
//!
//! ### Béton armé — vol. 2 (Eurocode 2)
//! - [`rc_slab`] — dalle : moments sens porteur / 4 appuis, armatures minimales.
//! - [`rc_footing`] — semelle isolée : contrainte de sol, aire, moment en console.
//! - [`rc_torsion`] — torsion : treillis spatial, flux, armatures.
//! - [`prestressed_concrete`] — précontrainte : pertes, effort effectif, contraintes.
//!
//! ### Charpente métallique — vol. 2 (Eurocode 3)
//! - [`steel_tension_member`] — barre tendue : section brute/nette.
//! - [`steel_beam_column`] — flexion composée : interaction linéaire et de stabilité.
//! - [`steel_welded_connection`] — soudure d'angle : méthode directionnelle.
//! - [`steel_base_plate`] — platine de pied : résistance d'appui, épaisseur.
//!
//! ### Bois — vol. 2 (Eurocode 5)
//! - [`timber_compression`] — compression + flambement (kc).
//! - [`timber_connection`] — assemblage par organe (Johansen).
//!
//! ### Géotechnique — vol. 2
//! - [`pile_capacity`] — pieu : pointe + frottement, capacité admissible.
//! - [`consolidation_degree`] — degré de consolidation dans le temps (Terzaghi).
//! - [`seepage`] — écoulement (Darcy) : réseau, gradient critique, renard.
//! - [`soil_compaction`] — compactage : Proctor, compacité, saturation.
//!
//! ### Actions & hydrologie — vol. 2 (Eurocodes 0/1)
//! - [`wind_load`] — vent : vitesse, pression de pointe, force.
//! - [`snow_load`] — neige : charge sur toiture, forme, altitude, accumulation.
//! - [`load_combination`] — combinaisons ELU/ELS (EN 1990).
//! - [`rational_runoff`] — méthode rationnelle : débit de pointe, Kirpich.
//!
//! ## Positionnement
//!
//! Cette crate ouvre le domaine du génie civil dans SciRust (aucune crate ne le
//! couvrait ; `scirust-frame` est une structure de données tabulaire, sans
//! rapport). Elle apporte le **cœur de calcul déterministe** du béton armé, de
//! la charpente métallique et bois, de la géotechnique et de l'hydraulique à
//! surface libre.
//!
//! ## Conventions d'unités
//!
//! SI cohérent : forces en N, moments en N·m (sauf mention kN·m), contraintes et
//! résistances en Pa (sauf mention MPa — les fonctions le précisent), longueurs
//! en m (sauf mention mm), angles en radians pour les fonctions trigonométriques.
//! Les Eurocodes travaillant usuellement en N et mm (donc MPa), chaque fonction
//! rappelle explicitement ses unités.
//!
//! **Limite honnête générale** : ce sont des **modèles réglementaires
//! simplifiés** (Eurocodes 2/3/5, mécanique des sols classique). Les
//! vérifications complètes (combinaisons d'actions, imperfections, second ordre
//! global, dispositions constructives) restent à la charge de l'ingénieur ; la
//! crate calcule les résistances et sollicitations élémentaires à partir des
//! données **fournies**.

pub mod bearing_capacity;
pub mod earth_pressure;
pub mod open_channel_manning;
pub mod rc_anchorage;
pub mod rc_beam_flexure;
pub mod rc_column;
pub mod rc_punching;
pub mod rc_serviceability;
pub mod rc_shear;
pub mod retaining_wall;
pub mod settlement;
pub mod slope_stability;
pub mod steel_bending;
pub mod steel_bolted_connection;
pub mod steel_compression;
pub mod steel_lateral_torsional;
pub mod steel_section_class;
pub mod timber_bending;

// Vol. 2
pub mod consolidation_degree;
pub mod load_combination;
pub mod pile_capacity;
pub mod prestressed_concrete;
pub mod rational_runoff;
pub mod rc_footing;
pub mod rc_slab;
pub mod rc_torsion;
pub mod seepage;
pub mod snow_load;
pub mod soil_compaction;
pub mod steel_base_plate;
pub mod steel_beam_column;
pub mod steel_tension_member;
pub mod steel_welded_connection;
pub mod timber_compression;
pub mod timber_connection;
pub mod wind_load;

pub use bearing_capacity::{
    geobear_allowable_bearing, geobear_bearing_factor_nc, geobear_bearing_factor_nq,
    geobear_terzaghi_ultimate,
};
pub use earth_pressure::{
    earthp_active_pressure_at_depth, earthp_active_thrust, earthp_cohesion_reduction,
    earthp_rankine_active_coefficient, earthp_rankine_passive_coefficient,
};
pub use open_channel_manning::{
    channel_critical_depth_rectangular, channel_hydraulic_radius, channel_manning_discharge,
    channel_manning_velocity, channel_rectangular_area,
};
pub use rc_anchorage::{
    rcanchor_basic_anchorage_length, rcanchor_design_anchorage_length, rcanchor_design_bond_stress,
    rcanchor_lap_length,
};
pub use rc_beam_flexure::{
    rcbeam_design_concrete_strength, rcbeam_design_steel_strength, rcbeam_lever_arm,
    rcbeam_reduced_moment, rcbeam_required_steel_area,
};
pub use rc_column::{
    rccol_axial_resistance, rccol_limiting_slenderness, rccol_minimum_eccentricity,
    rccol_radius_of_gyration_rectangular, rccol_slenderness_ratio,
};
pub use rc_punching::{
    rcpunch_control_perimeter_rectangular, rcpunch_resistance_without_reinforcement,
    rcpunch_shear_stress, rcpunch_utilisation,
};
pub use rc_serviceability::{
    rcsls_concrete_stress_limit, rcsls_crack_spacing, rcsls_crack_width, rcsls_span_depth_ratio_ok,
    rcsls_steel_stress_limit,
};
pub use rc_shear::{
    rcshear_max_resistance, rcshear_resistance_with_stirrups,
    rcshear_resistance_without_reinforcement, rcshear_size_factor,
};
pub use retaining_wall::{
    retwall_base_pressure_max, retwall_overturning_safety, retwall_resultant_eccentricity,
    retwall_sliding_safety,
};
pub use settlement::{
    settle_final_from_strain, settle_normally_consolidated, settle_overconsolidated_recompression,
    settle_time_factor,
};
pub use slope_stability::{
    slope_infinite_dry_cohesionless, slope_infinite_with_cohesion, slope_infinite_with_seepage,
    slope_is_stable,
};
pub use steel_bending::{
    steelbend_elastic_moment_resistance, steelbend_plastic_moment_resistance,
    steelbend_plastic_shear_resistance, steelbend_shear_area_rolled_i, steelbend_utilisation,
};
pub use steel_bolted_connection::{
    steelbolt_bearing_resistance, steelbolt_group_shear_resistance, steelbolt_shear_resistance,
    steelbolt_tension_resistance,
};
pub use steel_compression::{
    steelcomp_buckling_resistance, steelcomp_euler_critical_load,
    steelcomp_non_dimensional_slenderness, steelcomp_reduction_factor,
};
pub use steel_lateral_torsional::{
    steellt_buckling_resistance_moment, steellt_non_dimensional_slenderness,
    steellt_reduction_factor,
};
pub use steel_section_class::{
    steelclass_epsilon, steelclass_flange_class, steelclass_flange_slenderness,
    steelclass_web_class_compression, steelclass_web_slenderness,
};
pub use timber_bending::{
    timber_bending_stress, timber_design_bending_strength, timber_size_factor_depth,
    timber_utilisation,
};

// Vol. 2 — ré-exports à plat.
pub use consolidation_degree::{
    consol_degree_high, consol_degree_low, consol_time_factor, consol_time_from_degree_low,
};
pub use load_combination::{
    loadcomb_accidental, loadcomb_sls_characteristic, loadcomb_sls_quasi_permanent,
    loadcomb_uls_fundamental,
};
pub use pile_capacity::{
    pile_allowable_capacity, pile_end_bearing, pile_shaft_friction, pile_ultimate_capacity,
};
pub use prestressed_concrete::{
    psc_concrete_stress_bottom, psc_effective_prestress, psc_elastic_shortening_loss,
    psc_prestress_force, psc_relaxation_loss,
};
pub use rational_runoff::{
    runoff_composite_coefficient, runoff_peak_flow, runoff_time_of_concentration_kirpich,
    runoff_volume,
};
pub use rc_footing::{
    rcfoot_cantilever_moment, rcfoot_required_area, rcfoot_soil_pressure_centric,
    rcfoot_soil_pressure_eccentric_max,
};
pub use rc_slab::{
    rcslab_effective_span, rcslab_minimum_reinforcement, rcslab_one_way_moment_udl,
    rcslab_two_way_moment,
};
pub use rc_torsion::{
    rctor_longitudinal_reinforcement, rctor_shear_flow, rctor_stirrup_area_per_spacing,
    rctor_thin_wall_thickness,
};
pub use seepage::{
    seep_critical_gradient, seep_darcy_velocity, seep_factor_of_safety_piping, seep_flow_rate,
};
pub use snow_load::{
    snow_altitude_adjustment, snow_drift_load, snow_load_on_roof, snow_shape_coefficient_monopitch,
};
pub use soil_compaction::{
    compact_degree_of_saturation, compact_dry_density, compact_relative_compaction,
    compact_zero_air_voids_density,
};
pub use steel_base_plate::{
    steelbase_additional_bearing_width, steelbase_bearing_strength,
    steelbase_plate_thickness_required, steelbase_required_area,
};
pub use steel_beam_column::{
    steelbc_axial_utilisation, steelbc_bending_utilisation, steelbc_linear_interaction,
    steelbc_stability_interaction,
};
pub use steel_tension_member::{
    steelten_design_resistance, steelten_gross_section_resistance, steelten_net_area,
    steelten_net_section_resistance,
};
pub use steel_welded_connection::{
    steelweld_limit_stress, steelweld_simplified_resistance_per_length, steelweld_throat,
    steelweld_von_mises_stress,
};
pub use timber_compression::{
    timbercomp_buckling_resistance, timbercomp_design_strength, timbercomp_instability_factor,
    timbercomp_relative_slenderness,
};
pub use timber_connection::{
    timberconn_capacity_thick_plate_yield, timberconn_capacity_thin_plate_single_shear,
    timberconn_embedment_strength, timberconn_yield_moment,
};
pub use wind_load::{
    wind_force, wind_mean_velocity, wind_peak_velocity_pressure, wind_pressure_on_surface,
};
